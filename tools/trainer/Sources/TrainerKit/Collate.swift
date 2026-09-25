@preconcurrency import MLX

/// Where each example of a micro-batch sits once its rows are padded to one length: the model's
/// input tokens, which positions take audio features instead of token embeddings (and which
/// feature), and which positions the loss is taken at. Plain arrays, so it can be checked
/// without a GPU; `MicroBatch` turns it into MLX arrays.
public struct BatchLayout: Equatable, Sendable {
    public let rows: Int
    /// Every row's length: the longest example's, less its last token (nothing follows it).
    public let length: Int
    /// rows × length, row-major, padded after each example's last input token.
    public let inputs: [Int32]
    /// Per position: whether it takes an audio feature, and if so which of the micro-batch's
    /// (the encoder returns the clips' features one after another, in row order). A clip's first
    /// `audioFeatures` placeholders do; any after them keep their token embedding.
    public let isAudio: [Bool]
    public let audioIndex: [Int32]
    /// Flat positions (row × length + column) whose next token is a label token, and those tokens.
    public let targetPositions: [Int32]
    public let targetIDs: [Int32]
    /// The audio features the micro-batch's clips get, all together.
    public let audioFeatures: Int

    public init(examples: [TokenizedExample], padding: Int32) {
        precondition(!examples.isEmpty, "a micro-batch needs an example")
        rows = examples.count
        length = examples.map { $0.ids.count - 1 }.max()!
        var inputs = [Int32](repeating: padding, count: rows * length)
        var isAudio = [Bool](repeating: false, count: rows * length)
        var audioIndex = [Int32](repeating: 0, count: rows * length)
        var targetPositions: [Int32] = []
        var targetIDs: [Int32] = []
        var audioOffset = 0
        for (row, example) in examples.enumerated() {
            let base = row * length
            for column in 0 ..< example.ids.count - 1 {
                inputs[base + column] = example.ids[column]
            }
            for k in 0 ..< example.audioFeatures {
                isAudio[base + example.audioStart + k] = true
                audioIndex[base + example.audioStart + k] = Int32(audioOffset + k)
            }
            audioOffset += example.audioFeatures
            for column in example.targetPositions {
                targetPositions.append(Int32(base + column))
                targetIDs.append(example.ids[column + 1])
            }
        }
        self.inputs = inputs
        self.isAudio = isAudio
        self.audioIndex = audioIndex
        self.targetPositions = targetPositions
        self.targetIDs = targetIDs
        self.audioFeatures = audioOffset
    }

    public var targetCount: Int { targetIDs.count }
    /// Positions the decoder runs over, padding included.
    public var paddedTokens: Int { rows * length }
}

/// A micro-batch as the model takes it.
public struct MicroBatch {
    public let layout: BatchLayout
    /// [rows, length] int32.
    public let inputs: MLXArray
    /// [rows × length] bool and int32.
    public let isAudio: MLXArray
    public let audioIndex: MLXArray
    /// [targets] int32.
    public let targetPositions: MLXArray
    public let targetIDs: MLXArray
    /// The clips, one a row, as the encoder takes them.
    public let encoder: EncoderBatch

    /// `clips` are the examples' mel frames, [frames, 128] each, in the same order.
    public init(examples: [TokenizedExample], clips: [MLXArray], padding: Int32) {
        precondition(examples.count == clips.count)
        let layout = BatchLayout(examples: examples, padding: padding)
        self.layout = layout
        inputs = MLXArray(layout.inputs, [layout.rows, layout.length])
        isAudio = MLXArray(layout.isAudio)
        audioIndex = MLXArray(layout.audioIndex)
        targetPositions = MLXArray(layout.targetPositions)
        targetIDs = MLXArray(layout.targetIDs)

        encoder = EncoderBatch(clips: clips)
        precondition(encoder.layout.clipFeatures == examples.map(\.audioFeatures),
                     "the examples' audio features aren't what the encoder makes of their clips")
    }
}
