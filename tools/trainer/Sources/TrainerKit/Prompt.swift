/// The token sequence of one training example: the prompt LiveTranscribe gives Qwen3-ASR, then
/// the label, then <|im_end|>.
public struct TokenizedExample: Equatable, Sendable {
    /// Prompt, label and <|im_end|>.
    public let ids: [Int32]
    /// Tokens before the label: everything up to and including "<|im_start|>assistant\n".
    public let promptLength: Int
    /// Where the audio placeholders start, and how many there are.
    public let audioStart: Int
    public let audioTokens: Int
    /// How many of the placeholders, from the first, the encoder's features replace. The app's
    /// count of placeholders can be a few more than the encoder makes (AudioFeatures.audioTokens);
    /// the rest keep the <|audio_pad|> embedding, as mergeAudioFeatures leaves them.
    public let audioFeatures: Int

    public init(ids: [Int32], promptLength: Int, audioStart: Int, audioTokens: Int, audioFeatures: Int) {
        precondition(audioFeatures <= audioTokens)
        self.ids = ids
        self.promptLength = promptLength
        self.audioStart = audioStart
        self.audioTokens = audioTokens
        self.audioFeatures = audioFeatures
    }

    /// The label's tokens, <|im_end|> included: the ones the loss is taken over.
    public var labelLength: Int { ids.count - promptLength }

    /// Positions of the model's input (every token but the last) whose next token is a label
    /// token: where the logits are compared with the label.
    public var targetPositions: Range<Int> { (promptLength - 1) ..< (ids.count - 1) }
}

/// The prompt as LiveTranscribe runs the model: no language and no context, so the model writes
/// "language X<asr_text>" itself before the transcript, as buildPromptText does with no language:
///
///     <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
///     <|audio_pad|> × the clip's audio tokens
///     <|audio_end|><|im_end|>\n<|im_start|>assistant\n
///
/// The label is the record's text ("language Sinhala<asr_text>…") and <|im_end|>, the way Qwen's
/// fine-tuning script builds its targets.
public struct PromptFormat: Sendable {
    public static let prefixText = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
    public static let suffixText = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"

    public let prefix: [Int32]
    public let suffix: [Int32]
    public let audioToken: Int32
    public let endOfTurn: Int32
    /// Fills the rows of a batch after their last token; never a target.
    public let padding: Int32

    private let encode: @Sendable (String) -> [Int]

    /// `encode` tokenises text the way the model's tokenizer does, with special tokens such as
    /// <|im_start|> as single ids and nothing added at the ends.
    public init(encode: @escaping @Sendable (String) -> [Int], audioToken: Int, endOfTurn: Int, padding: Int) {
        self.encode = encode
        self.prefix = encode(Self.prefixText).map(Int32.init)
        self.suffix = encode(Self.suffixText).map(Int32.init)
        self.audioToken = Int32(audioToken)
        self.endOfTurn = Int32(endOfTurn)
        self.padding = Int32(padding)
    }

    /// The prompt for a clip with this many audio tokens.
    public func promptIDs(audioTokens: Int) -> [Int32] {
        prefix + Array(repeating: audioToken, count: audioTokens) + suffix
    }

    /// The label's tokens: the text, then <|im_end|>.
    public func labelIDs(_ text: String) -> [Int32] {
        encode(text).map(Int32.init) + [endOfTurn]
    }

    public func example(text: String, audioTokens: Int, audioFeatures: Int) -> TokenizedExample {
        let prompt = promptIDs(audioTokens: audioTokens)
        return TokenizedExample(
            ids: prompt + labelIDs(text), promptLength: prompt.count,
            audioStart: prefix.count, audioTokens: audioTokens, audioFeatures: audioFeatures
        )
    }
}
