import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT

/// A clip's model input: its log-mel frames, how many audio placeholders the app's prompt gives
/// it, and how many of those the encoder fills.
public struct ClipFeatures {
    /// [frames, 128], as Qwen3ASRModel.preprocessAudio computes them.
    public let mel: MLXArray
    public let audioTokens: Int
    public let audioFeatures: Int

    public var frames: Int { mel.dim(0) }
}

public enum AudioFeaturesError: Error, CustomStringConvertible {
    case placeholders(frames: Int, app: Int, trainer: Int)

    public var description: String {
        switch self {
        case .placeholders(let frames, let app, let trainer):
            "for \(frames) mel frames the app's prompt has \(app) audio placeholders and the trainer counts \(trainer): "
                + "mlx-audio-swift's getFeatExtractOutputLengths has changed, and AudioFeatures.audioTokens must follow it"
        }
    }
}

public enum AudioFeatures {
    public static let sampleRate = 16_000

    /// The <|audio_pad|> placeholders the app's prompt has for a clip of `frames` mel frames:
    /// getFeatExtractOutputLengths as mlx-audio-swift computes it, which is not quite Qwen's.
    /// Qwen's (and mlx-audio's Python) adds (frames // 100) × 13, in integers; the Swift port
    /// divides a whole-number array by 100, which MLX does in float32, so the fraction of the last
    /// partial second counts too (541 frames: 76 placeholders, where Qwen's count is 71). The
    /// training prompt copies the app's count, float32 steps and truncation included.
    public static func audioTokens(frames: Int) -> Int {
        let leave = frames % 100
        let afterFirst = floorDivide(leave - 1, 2) + 1
        let afterThird = floorDivide(floorDivide(afterFirst - 1, 2) + 1 - 1, 2) + 1
        let total = Float(afterThird) + Float(frames) / 100 * 13
        return Int(total)
    }

    static func floorDivide(_ a: Int, _ b: Int) -> Int {
        let quotient = a / b
        return (a % b != 0 && (a < 0) != (b < 0)) ? quotient - 1 : quotient
    }

    /// The recording's samples at 16 kHz, mono.
    public static func samples(of url: URL) throws -> MLXArray {
        let (_, audio) = try loadAudioArray(from: url, sampleRate: sampleRate)
        return audio.ndim > 1 ? audio.mean(axis: -1) : audio
    }

    /// A clip under a second padded with silence to one, as Qwen3ASRModel.generate pads it
    /// (splitAudioIntoChunks' minChunkDuration), so training sees what inference sees.
    public static func padded(_ samples: MLXArray) -> MLXArray {
        let short = sampleRate - samples.dim(0)
        return short > 0 ? MLX.padded(samples, widths: [IntOrPair((0, short))]) : samples
    }

    /// A clip's features as the app computes them: Qwen3ASRModel.preprocessAudio's log-mel frames
    /// and placeholder count, for the clip padded as generate pads it. Taking them from the model
    /// keeps the trainer on whatever front end the pinned mlx-audio-swift has (upstream changed its
    /// mel scale and window after the app's pin), rather than on a copy of it that could fall behind.
    public static func features(of samples: MLXArray, model: Qwen3ASRModel) throws -> ClipFeatures {
        let (input, _, placeholders) = model.preprocessAudio(padded(samples))
        let mel = input[0].transposed(1, 0).asType(.float32)
        let frames = mel.dim(0)
        // EncoderLayout counts each chunk's features with audioTokens(frames:), so it must still
        // be the app's count.
        guard placeholders == audioTokens(frames: frames) else {
            throw AudioFeaturesError.placeholders(frames: frames, app: placeholders, trainer: audioTokens(frames: frames))
        }
        return ClipFeatures(mel: mel, audioTokens: placeholders, audioFeatures: EncoderLayout(frames: [frames]).clipFeatures[0])
    }
}
