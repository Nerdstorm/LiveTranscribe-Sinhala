import Foundation
@preconcurrency import MLX
import MLXAudioSTT
import MLXNN
import Tokenizers

public enum ModelError: Error, CustomStringConvertible {
    case missingFile(URL)
    case unexpectedModel(String)
    case tokenizer(String)

    public var description: String {
        switch self {
        case .missingFile(let url): "\(url.path) is missing"
        case .unexpectedModel(let reason): "the model isn't the Qwen3-ASR this trainer knows: \(reason)"
        case .tokenizer(let reason): "the tokenizer isn't the one expected: \(reason)"
        }
    }
}

/// A Qwen3-ASR model loaded for training: every weight in one float type, none quantized, with
/// handles on the parts the training step calls directly.
public final class TrainableModel: @unchecked Sendable {
    public let model: Qwen3ASRModel
    public let encoder: AudioEncoder
    public let text: Qwen3ASRTextModel
    public let embedding: Embedding
    public let tokenizer: any Tokenizer
    public let format: PromptFormat
    public let directory: URL

    init(model: Qwen3ASRModel, encoder: AudioEncoder, text: Qwen3ASRTextModel, embedding: Embedding,
         tokenizer: any Tokenizer, format: PromptFormat, directory: URL) {
        self.model = model
        self.encoder = encoder
        self.text = text
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.format = format
        self.directory = directory
    }

    public static let audioPad = "<|audio_pad|>"
    public static let endOfTurn = "<|im_end|>"
    public static let endOfText = "<|endoftext|>"

    /// Loads a Qwen3-ASR model folder (config.json, safetensors, tokenizer files) as mlx-audio-swift
    /// loads it for the app, but with quantized layers dequantized and every weight cast to `dtype`.
    /// Training starts from the app's 8-bit model this way: its weights as the app computes with them.
    public static func load(directory: URL, dtype: DType = .float32) async throws -> TrainableModel {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else { throw ModelError.missingFile(configURL) }
        let config = try JSONDecoder().decode(Qwen3ASRConfig.self, from: Data(contentsOf: configURL))
        guard !config.isForcedAligner else { throw ModelError.unexpectedModel("it's the forced aligner") }

        // mlx-audio-swift writes tokenizer.json from vocab.json and merges.txt the first time it
        // loads a Qwen3-ASR folder; the app has usually done that already.
        if !FileManager.default.fileExists(atPath: directory.appending(component: "tokenizer.json").path) {
            _ = try await Qwen3ASRModel.fromModelDirectory(directory)
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)

        let weights = try loadWeights(directory: directory, config: config, dtype: dtype)
        let model = Qwen3ASRModel(config)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = tokenizer

        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        guard let text = modules["model"] as? Qwen3ASRTextModel else {
            throw ModelError.unexpectedModel("no text model at \"model\"")
        }
        guard let embedding = modules["model.embed_tokens"] as? Embedding else {
            throw ModelError.unexpectedModel("no embedding at \"model.embed_tokens\"")
        }
        guard config.textConfig.tieWordEmbeddings else {
            throw ModelError.unexpectedModel("its output layer isn't tied to the embeddings")
        }

        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token) else { throw ModelError.tokenizer("no \(token)") }
            return id
        }
        let audioPad = try id(audioPad)
        guard audioPad == config.audioTokenId else {
            throw ModelError.tokenizer("\(Self.audioPad) is \(audioPad), the config says \(config.audioTokenId)")
        }
        let encode: @Sendable (String) -> [Int] = { [tokenizer] in tokenizer.encode(text: $0, addSpecialTokens: false) }
        let format = PromptFormat(encode: encode, audioToken: audioPad, endOfTurn: try id(endOfTurn), padding: try id(endOfText))

        eval(model)
        return TrainableModel(model: model, encoder: try AudioEncoder(model: model), text: text, embedding: embedding,
                              tokenizer: tokenizer, format: format, directory: directory)
    }

    /// The folder's weights as the model's parameters: keys as Qwen3ASRModel.sanitize leaves them,
    /// quantized layers (a .weight, .scales and .biases triple) dequantized, all cast to `dtype`.
    static func loadWeights(directory: URL, config: Qwen3ASRConfig, dtype: DType) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw ModelError.missingFile(directory.appending(component: "model.safetensors")) }
        var raw: [String: MLXArray] = [:]
        for file in files {
            raw.merge(try loadArrays(url: file)) { _, new in new }
        }
        let sanitized = Qwen3ASRModel.sanitize(weights: raw, skipLmHead: config.textConfig.tieWordEmbeddings)
        let quantization = try Quantization.read(directory: directory)

        var weights: [String: MLXArray] = [:]
        for (key, value) in sanitized where !key.hasSuffix(".scales") && !key.hasSuffix(".biases") {
            let layer = String(key.dropLast(".weight".count))
            if key.hasSuffix(".weight"), let scales = sanitized[layer + ".scales"] {
                guard let quantization else {
                    throw ModelError.unexpectedModel("\(layer) is quantized but config.json has no quantization")
                }
                weights[key] = dequantized(
                    value, scales: scales, biases: sanitized[layer + ".biases"],
                    groupSize: quantization.groupSize, bits: quantization.bits, mode: .affine, dtype: .float32
                ).asType(dtype)
            } else {
                weights[key] = value.asType(dtype)
            }
        }
        return weights
    }
}

/// The quantization config.json declares: mlx-community's 8-bit model quantizes every layer of the
/// text model, affine, in groups of 64.
public struct Quantization: Codable, Equatable, Sendable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size", bits, mode
    }

    public init(groupSize: Int, bits: Int, mode: String = "affine") {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        bits = try container.decode(Int.self, forKey: .bits)
        mode = try container.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
    }

    public static func read(directory: URL) throws -> Quantization? {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        struct Config: Decodable { let quantization: Quantization? }
        let quantization = try JSONDecoder().decode(Config.self, from: data).quantization
        if let quantization, quantization.mode != "affine" {
            throw ModelError.unexpectedModel("quantization mode \(quantization.mode), not affine")
        }
        return quantization
    }
}
