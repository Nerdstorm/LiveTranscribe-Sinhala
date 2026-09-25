import Foundation
@preconcurrency import MLX

public enum ExportError: Error, CustomStringConvertible {
    case missingWeight(String)
    case unusedWeights([String])
    case mismatch(String)
    case config(String)

    public var description: String {
        switch self {
        case .missingWeight(let key): "the trained weights have no \(key)"
        case .unusedWeights(let keys): "the base model has no place for \(keys.count) trained weights, e.g. \(keys.prefix(3).joined(separator: ", "))"
        case .mismatch(let reason): "the export doesn't match the base model: \(reason)"
        case .config(let reason): "can't update config.json: \(reason)"
        }
    }
}

/// Writes trained weights as a model folder LiveTranscribe loads the way it loads the base model:
/// the base folder's files, with model.safetensors holding the trained weights in the base file's
/// layout (the same keys, dtypes and shapes; layers quantized where the base's are, affine, with
/// its group size and bits) and `language` added to config.json's support_languages.
public enum Export {
    public static let weightsFile = "model.safetensors"
    public static let indexFile = "model.safetensors.index.json"

    public struct Summary: Sendable {
        public let tensors: Int
        public let quantizedLayers: Int
        public let bytes: Int
        public let parameters: Int
    }

    public static func write(weights: Tensors, base: URL, to output: URL, language: String) throws -> Summary {
        let baseFiles = try FileManager.default.contentsOfDirectory(at: base, includingPropertiesForKeys: nil)
        let baseWeightFiles = baseFiles.filter { $0.pathExtension == "safetensors" }
        guard baseWeightFiles.count == 1, baseWeightFiles[0].lastPathComponent == weightsFile else {
            throw ExportError.mismatch("expected one \(weightsFile) in \(base.path)")
        }
        let (reference, metadata) = try loadArraysAndMetadata(url: baseWeightFiles[0])
        guard !reference.keys.contains(where: { $0.hasPrefix("thinker.") }) else {
            throw ExportError.mismatch("the base file isn't in MLX's layout (it has thinker. keys)")
        }
        let quantization = try Quantization.read(directory: base)

        var exported = Tensors(minimumCapacity: reference.count)
        var used = Set<String>()
        var quantizedLayers = 0
        for (key, expected) in reference where !key.hasSuffix(".scales") && !key.hasSuffix(".biases") {
            guard let trained = weights[key] else { throw ExportError.missingWeight(key) }
            used.insert(key)
            let layer = String(key.dropLast(".weight".count))
            if key.hasSuffix(".weight"), let scales = reference[layer + ".scales"] {
                guard let quantization else { throw ExportError.mismatch("\(layer) is quantized but config.json says nothing") }
                let (wq, s, b) = quantize(trained, groupSize: quantization.groupSize, bits: quantization.bits,
                                          dtype: scales.dtype)
                exported[key] = wq
                exported[layer + ".scales"] = s
                exported[layer + ".biases"] = b
                quantizedLayers += 1
            } else {
                exported[key] = trained.asType(expected.dtype)
            }
        }
        let unused = Set(weights.keys).subtracting(used)
        guard unused.isEmpty else { throw ExportError.unusedWeights(unused.sorted()) }
        for (key, expected) in reference {
            guard let array = exported[key] else { throw ExportError.mismatch("no \(key)") }
            guard array.shape == expected.shape, array.dtype == expected.dtype else {
                throw ExportError.mismatch("\(key) is \(array.dtype) \(array.shape), the base's is \(expected.dtype) \(expected.shape)")
            }
        }

        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        eval(Array(exported.values))
        try MLX.save(arrays: exported, metadata: metadata, url: output.appending(component: weightsFile))

        let bytes = exported.values.map(\.nbytes).reduce(0, +)
        let parameters = weights.values.map(\.size).reduce(0, +)
        for file in baseFiles where file.pathExtension != "safetensors" && file.lastPathComponent != indexFile {
            let destination = output.appending(component: file.lastPathComponent)
            if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
            if file.lastPathComponent == "config.json" {
                try addLanguage(language, config: file, to: destination)
            } else {
                try FileManager.default.copyItem(at: file, to: destination)
            }
        }
        try writeIndex(keys: exported.keys.sorted(), bytes: bytes, parameters: parameters,
                       to: output.appending(component: indexFile))
        return Summary(tensors: exported.count, quantizedLayers: quantizedLayers, bytes: bytes, parameters: parameters)
    }

    /// Affine quantization of a [..., columns] weight, laid out as MLX lays it out: per group of
    /// `groupSize` columns a scale and a bias in `dtype`, level 0 at the group's end with the larger
    /// magnitude, and `bits`-bit levels packed into uint32s from the low bits up, so MLX's
    /// dequantize reads it. The grid is chosen after rounding, not before: the bias is rounded to
    /// `dtype` outwards, past the group's end, and the scale's magnitude upwards, so every weight
    /// lies on the grid's span and dequantizes to within half a step of itself. MLX's own quantize
    /// picks levels against its float32 scale and bias and then rounds those to the stored dtype,
    /// which leaves some weights more than a step away (and the weights reach it in bfloat16).
    public static func quantize(_ weight: MLXArray, groupSize: Int, bits: Int, dtype: DType)
        -> (weight: MLXArray, scales: MLXArray, biases: MLXArray) {
        let shape = weight.shape
        let columns = shape.last!
        precondition(columns % groupSize == 0 && 32 % bits == 0, "\(columns) columns in groups of \(groupSize) at \(bits) bits")
        let groups = weight.asType(.float32).reshaped(-1, columns / groupSize, groupSize)
        let levels = Float((1 << bits) - 1)

        let low = groups.min(axis: -1), high = groups.max(axis: -1)
        let fromLow = abs(low) .> abs(high)
        let bias = `where`(fromLow, rounded(low, to: dtype, up: false), rounded(high, to: dtype, up: true))
        let span = `where`(fromLow, high - bias, bias - low)
        let magnitude = rounded(maximum(span / levels, Float(1e-7)), to: dtype, up: true)
        let scale = `where`(fromLow, magnitude, -magnitude)

        let quantized = clip(
            round((groups - bias.expandedDimensions(axis: -1)) / scale.expandedDimensions(axis: -1)),
            min: Float(0), max: levels
        ).asType(.uint32)
        let perWord = 32 / bits
        let shifts = MLXArray((0 ..< perWord).map { UInt32($0 * bits) })
        let packed = (quantized.reshaped(-1, columns / perWord, perWord) << shifts).sum(axis: -1).asType(.uint32)

        var packedShape = shape
        packedShape[packedShape.count - 1] = columns * bits / 32
        var groupShape = shape
        groupShape[groupShape.count - 1] = columns / groupSize
        return (packed.reshaped(packedShape), scale.asType(dtype).reshaped(groupShape), bias.asType(dtype).reshaped(groupShape))
    }

    /// Float32 values rounded to the nearest `dtype` value at or above them (`up`) or at or below
    /// them, returned as float32. For a 2-byte float, one step along its values is one step of its
    /// bit pattern: +1 away from zero, −1 towards it.
    public static func rounded(_ x: MLXArray, to dtype: DType, up: Bool) -> MLXArray {
        if dtype == .float32 { return x }
        precondition(dtype == .bfloat16 || dtype == .float16, "rounding to \(dtype)")
        let nearest = x.asType(dtype)
        let value = nearest.asType(.float32)
        let wrongWay = (up ? value .< x : value .> x) .&& (value .!= Float(0))
        let awayFromZero = up ? value .> Float(0) : value .< Float(0)
        let step = `where`(awayFromZero, Int32(1), Int32(-1))
        let bits = nearest.view(dtype: .uint16).asType(.int32) + `where`(wrongWay, step, Int32(0))
        return bits.asType(.uint16).view(dtype: dtype).asType(.float32)
    }

    /// config.json with `language` appended to support_languages, the rest of the file unchanged.
    static func addLanguage(_ language: String, config: URL, to destination: URL) throws {
        let text = try String(contentsOf: config, encoding: .utf8)
        let updated = try addingLanguage(language, toConfig: text)
        try updated.write(to: destination, atomically: true, encoding: .utf8)
    }

    public static func addingLanguage(_ language: String, toConfig text: String) throws -> String {
        guard var original = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
              var languages = original["support_languages"] as? [String] else {
            throw ExportError.config("no support_languages list")
        }
        if languages.contains(language) { return text }
        guard let list = text.range(of: #""support_languages"\s*:\s*\[[^\]]*"#, options: .regularExpression) else {
            throw ExportError.config("can't find the support_languages list")
        }
        let inside = text[list]
        let lastItem = inside.range(of: #""[^"]*"\s*$"#, options: .regularExpression)
        guard let lastItem else { throw ExportError.config("support_languages is empty") }
        // Copy the separator the file uses between items ("," and a newline and indent, or ", ").
        let items = inside.ranges(of: #/"[^"]*"/#)
        let separator = items.count >= 2
            ? String(inside[items[items.count - 2].upperBound ..< items[items.count - 1].lowerBound])
            : ", "
        let lastEnd = inside[lastItem].lastIndex(of: "\"")!
        var updated = text
        updated.insert(contentsOf: separator + "\"\(language)\"", at: text.index(after: lastEnd))

        languages.append(language)
        original["support_languages"] = languages
        guard let check = try JSONSerialization.jsonObject(with: Data(updated.utf8)) as? [String: Any],
              NSDictionary(dictionary: check).isEqual(to: original) else {
            throw ExportError.config("the edited file doesn't parse to the original plus \(language)")
        }
        return updated
    }

    static func writeIndex(keys: [String], bytes: Int, parameters: Int, to url: URL) throws {
        struct Index: Encodable {
            struct Metadata: Encodable {
                let total_size: Int
                let total_parameters: Int
            }
            let metadata: Metadata
            let weight_map: [String: String]
        }
        let index = Index(
            metadata: .init(total_size: bytes, total_parameters: parameters),
            weight_map: Dictionary(uniqueKeysWithValues: keys.map { ($0, weightsFile) })
        )
        try JSON.write(index, to: url)
    }
}
