import Foundation
@preconcurrency import MLX
import MLXAudioSTT
import MLXNN

/// How the audio encoder splits clips, in plain numbers, as mlx-audio-swift's
/// Qwen3ASRAudioEncoder splits a clip on its own:
///
/// - each clip into chunks of 100 mel frames (the last one shorter), every chunk zero-padded to
///   the longest before three stride-2 convolutions, which make 13 rows of a 100-frame chunk;
/// - of each chunk's rows it keeps the count its (float32) getFeatExtractOutputLengths gives,
///   which for a last, partial chunk can be more than the convolutions made: then it keeps them all;
/// - attention runs within windows of 8 chunks of one clip (n_window_infer 800 frames).
///
/// For one clip alone that's exactly the app's encoder. Given several clips at once, the app's
/// encoder lays its windows out from the uncapped counts, so a window can run into the next
/// clip; the trainer's encoder takes its windows from this layout instead, one clip at a time.
public struct EncoderLayout: Equatable, Sendable {
    public static let chunkFrames = 100
    public static let chunksPerWindow = 8

    public let clipFrames: [Int]
    /// Every clip's chunks, clip by clip: which clip, where in it and how many frames.
    public let chunkClips: [Int]
    public let chunkStarts: [Int]
    public let chunkLengths: [Int]
    /// What every chunk is padded to (the longest chunk), and the rows the convolutions make of it.
    public let paddedLength: Int
    public let convolvedLength: Int
    /// Rows the encoder keeps of each chunk, and of each clip: the clip's audio features.
    public let chunkRows: [Int]
    public let clipFeatures: [Int]
    /// The attention windows, as ranges of the kept rows (every clip's rows one after another).
    public let windows: [Range<Int>]

    public init(frames: [Int]) {
        precondition(!frames.isEmpty && frames.allSatisfy { $0 > 0 })
        clipFrames = frames
        var clips: [Int] = [], starts: [Int] = [], lengths: [Int] = []
        var chunksPerClip: [Int] = []
        for (clip, count) in frames.enumerated() {
            var start = 0
            var chunks = 0
            while start < count {
                let length = min(Self.chunkFrames, count - start)
                clips.append(clip)
                starts.append(start)
                lengths.append(length)
                start += length
                chunks += 1
            }
            chunksPerClip.append(chunks)
        }
        chunkClips = clips
        chunkStarts = starts
        chunkLengths = lengths
        paddedLength = lengths.max()!
        let convolved = Self.convolved(Self.convolved(Self.convolved(paddedLength)))
        convolvedLength = convolved
        chunkRows = lengths.map { min(convolved, AudioFeatures.audioTokens(frames: $0)) }

        var features = [Int](repeating: 0, count: frames.count)
        for (chunk, rows) in chunkRows.enumerated() { features[clips[chunk]] += rows }
        clipFeatures = features

        var windows: [Range<Int>] = []
        var chunk = 0, row = 0
        for count in chunksPerClip {
            for first in stride(from: 0, to: count, by: Self.chunksPerWindow) {
                let rows = chunkRows[(chunk + first) ..< (chunk + min(first + Self.chunksPerWindow, count))].reduce(0, +)
                windows.append(row ..< row + rows)
                row += rows
            }
            chunk += count
        }
        self.windows = windows
    }

    /// A stride-2, kernel-3, padding-1 convolution's output length.
    static func convolved(_ length: Int) -> Int { (length - 1) / 2 + 1 }

    public var longestWindow: Int { windows.map(\.count).max()! }
}

/// A batch of clips as the encoder takes them.
public struct EncoderBatch {
    public let layout: EncoderLayout
    /// [chunks, 128, paddedLength, 1]: every chunk's mel frames, zero-padded.
    public let chunks: MLXArray
    /// [windows, longest window]: which of the convolutions' rows (chunk × convolvedLength + row)
    /// fill each window, padded with row 0; and [windows, 1, 1, longest window], 0 for a real row
    /// and a large negative number for padding, added to the attention scores.
    public let windowRows: MLXArray
    public let windowMask: MLXArray
    /// [features]: where each kept row is in the windows' output (window × longest + row), in order.
    public let outputRows: MLXArray

    /// `clips` are mel frames, [frames, 128] each.
    public init(clips: [MLXArray]) {
        let layout = EncoderLayout(frames: clips.map { $0.dim(0) })
        self.layout = layout
        chunks = MLX.stacked((0 ..< layout.chunkLengths.count).map { chunk in
            let start = layout.chunkStarts[chunk], length = layout.chunkLengths[chunk]
            var frames = clips[layout.chunkClips[chunk]][start ..< start + length].transposed(1, 0)
            if length < layout.paddedLength {
                frames = MLX.padded(frames, widths: [IntOrPair((0, 0)), IntOrPair((0, layout.paddedLength - length))])
            }
            return frames
        }).expandedDimensions(axis: -1)

        // The kept rows, in order, as indices into the convolutions' output.
        var kept: [Int32] = []
        for (chunk, rows) in layout.chunkRows.enumerated() {
            for row in 0 ..< rows { kept.append(Int32(chunk * layout.convolvedLength + row)) }
        }
        let longest = layout.longestWindow
        var rows = [Int32](repeating: 0, count: layout.windows.count * longest)
        var mask = [Float](repeating: -1e9, count: layout.windows.count * longest)
        var output: [Int32] = []
        for (window, range) in layout.windows.enumerated() {
            for (offset, row) in range.enumerated() {
                rows[window * longest + offset] = kept[row]
                mask[window * longest + offset] = 0
                output.append(Int32(window * longest + offset))
            }
        }
        windowRows = MLXArray(rows, [layout.windows.count, longest])
        windowMask = MLXArray(mask, [layout.windows.count, 1, 1, longest])
        outputRows = MLXArray(output)
    }
}

/// Qwen3-ASR's audio encoder, run on the model's own layers (so it trains them) over a batch of
/// clips, computing for each clip what mlx-audio-swift's encoder computes for it alone.
public final class AudioEncoder: @unchecked Sendable {
    struct Layer {
        let attentionNorm: LayerNorm
        let query: Linear, key: Linear, value: Linear, output: Linear
        let feedForwardNorm: LayerNorm
        let up: Linear, down: Linear
    }

    let conv1: Conv2d, conv2: Conv2d, conv3: Conv2d
    let convOut: Linear
    let layers: [Layer]
    let finalNorm: LayerNorm
    let proj1: Linear, proj2: Linear
    /// The encoder's sinusoidal position table, [positions, width], float32.
    let positions: MLXArray
    let heads: Int

    public init(model: Qwen3ASRModel) throws {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        func module<T>(_ path: String, as _: T.Type = T.self) throws -> T {
            let key = path.isEmpty ? "audio_tower" : "audio_tower." + path
            guard let found = modules[key] as? T else { throw ModelError.unexpectedModel("no \(T.self) at \(key)") }
            return found
        }
        conv1 = try module("conv2d1")
        conv2 = try module("conv2d2")
        conv3 = try module("conv2d3")
        convOut = try module("conv_out")
        layers = try (0 ..< model.config.audioConfig.encoderLayers).map { index in
            let prefix = "layers.\(index)."
            return Layer(
                attentionNorm: try module(prefix + "self_attn_layer_norm"),
                query: try module(prefix + "self_attn.q_proj"), key: try module(prefix + "self_attn.k_proj"),
                value: try module(prefix + "self_attn.v_proj"), output: try module(prefix + "self_attn.out_proj"),
                feedForwardNorm: try module(prefix + "final_layer_norm"),
                up: try module(prefix + "fc1"), down: try module(prefix + "fc2")
            )
        }
        finalNorm = try module("ln_post")
        proj1 = try module("proj1")
        proj2 = try module("proj2")
        heads = model.config.audioConfig.encoderAttentionHeads
        let audio = model.config.audioConfig
        guard audio.nWindow * 2 == EncoderLayout.chunkFrames,
              audio.nWindowInfer / (audio.nWindow * 2) == EncoderLayout.chunksPerWindow else {
            throw ModelError.unexpectedModel("its encoder windows aren't 100-frame chunks, 8 to a window")
        }

        // The table the app's encoder adds (Qwen3ASRSinusoidalPE, not a parameter), read from it.
        let encoder: Module = try module("", as: Module.self)
        guard let table = Mirror(reflecting: encoder).children.first(where: { $0.label == "positionalEmbedding" })
            .flatMap({ child in Mirror(reflecting: child.value).children.first { $0.label == "_positionalEmbedding" }?.value })
            as? MLXArray
        else {
            throw ModelError.unexpectedModel("can't find the encoder's position table")
        }
        positions = table
    }

    /// The batch's audio features, [features, output width]: each clip's, one clip after another.
    public func callAsFunction(_ batch: EncoderBatch) -> MLXArray {
        var x = gelu(conv1(batch.chunks))
        x = gelu(conv2(x))
        x = gelu(conv3(x))
        let (chunks, frequencies, times, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        precondition(times == batch.layout.convolvedLength)
        x = x.transposed(0, 2, 3, 1).reshaped(chunks, times, channels * frequencies)
        x = convOut(x) + positions[0 ..< times].expandedDimensions(axis: 0)

        var hidden = x.reshaped(chunks * times, -1).take(batch.windowRows, axis: 0)
        let mask = batch.windowMask.asType(hidden.dtype)
        for layer in layers {
            hidden = Self.layer(layer, hidden, mask: mask, heads: heads)
        }
        hidden = hidden.reshaped(-1, hidden.dim(-1)).take(batch.outputRows, axis: 0)
        hidden = finalNorm(hidden)
        return proj2(gelu(proj1(hidden)))
    }

    /// Qwen3ASRAudioEncoderLayer: pre-norm attention within the window, then a GELU feed-forward.
    static func layer(_ layer: Layer, _ input: MLXArray, mask: MLXArray, heads: Int) -> MLXArray {
        let (batch, length, width) = (input.dim(0), input.dim(1), input.dim(2))
        let headWidth = width / heads
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped(batch, length, heads, headWidth).transposed(0, 2, 1, 3)
        }
        let normed = layer.attentionNorm(input)
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(layer.query(normed)), keys: split(layer.key(normed)), values: split(layer.value(normed)),
            scale: pow(Float(headWidth), -0.5), mask: .array(mask)
        )
        var hidden = input + layer.output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, width))
        hidden = hidden + layer.down(gelu(layer.up(layer.feedForwardNorm(hidden))))
        return hidden
    }
}
