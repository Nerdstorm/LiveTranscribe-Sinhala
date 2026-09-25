@preconcurrency import MLX
import MLXAudioSTT
import MLXNN

/// The training forward pass: Qwen3ASRModel's, batched. The encoder turns every clip into
/// features (AudioEncoder: the app's encoder, a batch at a time), which replace the
/// <|audio_pad|> embeddings as mergeAudioFeatures replaces them; the decoder runs over the rows
/// with its causal mask; logits are computed only where the next token is a label token, from the
/// tied embeddings, as the model's own forward pass computes them.
public enum Forward {
    /// Puts audio features in the rows: position p takes `audio[audioIndex[p]]` where `isAudio[p]`,
    /// else its token embedding. mergeAudioFeatures does this for one row; this does it for many.
    public static func merge(tokens: MLXArray, audio: MLXArray, isAudio: MLXArray, audioIndex: MLXArray) -> MLXArray {
        let (rows, length, width) = (tokens.dim(0), tokens.dim(1), tokens.dim(2))
        let fromAudio = audio.asType(tokens.dtype).take(audioIndex, axis: 0)
        let flat = tokens.reshaped(rows * length, width)
        return MLX.where(isAudio.expandedDimensions(axis: -1), fromAudio, flat).reshaped(rows, length, width)
    }

    /// The decoder's final hidden states at the micro-batch's target positions, [targets, hidden].
    public static func targetStates(_ trainable: TrainableModel, _ batch: MicroBatch) -> MLXArray {
        let audio = trainable.encoder(batch.encoder)
        let embeddings = trainable.embedding(batch.inputs)
        let merged = merge(tokens: embeddings, audio: audio, isAudio: batch.isAudio, audioIndex: batch.audioIndex)
        let hidden = trainable.text(inputsEmbeds: merged)
        return hidden.reshaped(-1, hidden.dim(-1)).take(batch.targetPositions, axis: 0)
    }

    /// Cross-entropy at each target position, [targets], in float32.
    public static func tokenLosses(_ trainable: TrainableModel, _ batch: MicroBatch) -> MLXArray {
        let logits = trainable.embedding.asLinear(targetStates(trainable, batch)).asType(.float32)
        return crossEntropy(logits: logits, targets: batch.targetIDs, reduction: .none)
    }

    /// The summed cross-entropy of a micro-batch divided by `normalizer`, the target tokens of the
    /// whole optimizer step: summing the micro-batches' gradients then gives the gradient of the
    /// mean over every target token of the step. Returns that loss and each target token's own.
    public static func loss(_ trainable: TrainableModel, _ batch: MicroBatch, normalizer: MLXArray) -> [MLXArray] {
        let losses = tokenLosses(trainable, batch)
        return [losses.sum() / normalizer, stopGradient(losses)]
    }

    /// The target tokens the model would pick at each target position (teacher forcing), [targets].
    public static func greedyTargets(_ trainable: TrainableModel, _ batch: MicroBatch) -> MLXArray {
        trainable.embedding.asLinear(targetStates(trainable, batch)).argMax(axis: -1).asType(.int32)
    }
}
