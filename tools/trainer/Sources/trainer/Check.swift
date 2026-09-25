// trainer check: correctness checks on the model and the data before any training.
import ArgumentParser
import Foundation
@preconcurrency import MLX
import MLXAudioSTT
import MLXNN
import TrainerKit

struct Check: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Checks tokenisation, the batched loss, teacher forcing, gradients, full precision against 8-bit, and bfloat16 against float32."
    )

    @OptionGroup var model: ModelOption
    @Option(help: "Sinhala training records.") var train = "jsonl/train.jsonl"
    @Option(help: "Replay records: the base model's own transcripts.") var replay = "jsonl/replay.jsonl"
    @Option(help: "Records with references to transcribe with both precisions (FLEURS English test).")
    var english: String?
    @Option(help: "How many records each check uses.") var count = 16

    func run() async throws {
        let folder = try await modelFolder(model.model)
        let trainable = try await TrainableModel.load(directory: folder)
        let format = trainable.format
        let sinhala = Examples.sample(try readRecords([train]), count: count, seed: 1)
        let replayed = Examples.sample(try readRecords([replay]), count: count, seed: 1)
        var failures: [String] = []
        func verdict(_ passed: Bool, _ message: String) {
            say((passed ? "PASS  " : "FAIL  ") + message)
            if !passed { failures.append(message) }
        }

        // 1. Tokenisation: the prompt and label tokenised apart, joined, are the whole text tokenised.
        var tokenisationMismatches = 0
        for record in sinhala + replayed {
            let example = try Examples.prepare(record, for: trainable)
            let whole = PromptFormat.prefixText + String(repeating: TrainableModel.audioPad, count: example.tokens.audioTokens)
                + PromptFormat.suffixText + record.text + TrainableModel.endOfTurn
            let asApp = trainable.tokenizer.encode(text: whole).map(Int32.init)
            let plain = trainable.tokenizer.encode(text: whole, addSpecialTokens: false).map(Int32.init)
            if asApp != example.tokens.ids || plain != example.tokens.ids { tokenisationMismatches += 1 }
        }
        verdict(tokenisationMismatches == 0,
                "tokenisation: \(sinhala.count + replayed.count - tokenisationMismatches) of \(sinhala.count + replayed.count) examples tokenise the same whole and in parts")

        // 2. Each clip gets the app's count of placeholders and features, and the batched encoder
        //    computes what the app's encoder computes for the clip alone.
        let clips = try (sinhala + replayed).map { try Examples.prepare($0, for: trainable) }
        let batched = trainable.encoder(EncoderBatch(clips: clips.map(\.mel)))
        var countMismatches: [String] = []
        var largest: Float = 0, scale: Float = 0, aloneLargest: Float = 0
        var worst = ""
        var offset = 0
        for (clip, record) in zip(clips, sinhala + replayed) {
            let samples = AudioFeatures.padded(try AudioFeatures.samples(of: URL(fileURLWithPath: record.audio)))
            let (input, mask, placeholders) = trainable.model.preprocessAudio(samples)
            let alone = trainable.model.getAudioFeatures(input, featureAttentionMask: mask)
            if placeholders != clip.tokens.audioTokens || alone.dim(0) != clip.tokens.audioFeatures {
                countMismatches.append("\(clip.mel.dim(0)) frames: the app has \(placeholders) placeholders and \(alone.dim(0)) "
                                       + "features, the trainer \(clip.tokens.audioTokens) and \(clip.tokens.audioFeatures)")
            } else {
                let mine = batched[offset ..< offset + clip.tokens.audioFeatures]
                let mineAlone = trainable.encoder(EncoderBatch(clips: [clip.mel]))
                let difference = MLX.abs(mine - alone)
                let clipLargest = difference.max().item(Float.self)
                aloneLargest = max(aloneLargest, MLX.abs(mineAlone - alone).max().item(Float.self))
                if clipLargest > largest {
                    largest = clipLargest
                    let perRow = difference.max(axis: -1).asArray(Float.self)
                    let row = perRow.firstIndex(of: perRow.max()!)!
                    worst = "\(clip.mel.dim(0)) frames, \(clip.tokens.audioFeatures) rows, largest at row \(row); "
                        + String(format: "mean difference %.2e", difference.mean().item(Float.self))
                }
                scale = max(scale, MLX.abs(alone).max().item(Float.self))
            }
            offset += clip.tokens.audioFeatures
        }
        let extra = clips.filter { $0.tokens.audioTokens > $0.tokens.audioFeatures }.count
        verdict(countMismatches.isEmpty, "audio placeholders and features: \(clips.count - countMismatches.count) of \(clips.count) "
                + "clips counted as the app counts them (\(extra) with placeholders the encoder doesn't fill)"
                + countMismatches.prefix(5).map { "\n    " + $0 }.joined())
        // The same clips batched through the app's own encoder, where its batching is sound (no clip
        // with placeholders it doesn't fill): how far float32 sums done in another order drift.
        let sound = clips.filter { $0.tokens.audioTokens == $0.tokens.audioFeatures }
        let longestSound = sound.map { $0.mel.dim(0) }.max() ?? 0
        let appBatched = trainable.model.getAudioFeatures(
            MLX.stacked(sound.map { MLX.padded($0.mel, widths: [IntOrPair((0, longestSound - $0.mel.dim(0))), IntOrPair((0, 0))]).transposed(1, 0) }),
            featureAttentionMask: MLXArray(sound.flatMap { clip in
                (0 ..< longestSound).map { Int32($0 < clip.mel.dim(0) ? 1 : 0) }
            }, [sound.count, longestSound])
        )
        var appDrift: Float = 0
        var soundOffset = 0
        for clip in sound {
            let alone = trainable.model.getAudioFeatures(
                clip.mel.transposed(1, 0).expandedDimensions(axis: 0),
                featureAttentionMask: MLXArray.ones([1, clip.mel.dim(0)], type: Int32.self)
            )
            appDrift = max(appDrift, MLX.abs(appBatched[soundOffset ..< soundOffset + alone.dim(0)] - alone).max().item(Float.self))
            soundOffset += alone.dim(0)
        }
        say(String(format: "NOTE  the app's encoder, %d clips batched against each alone: largest difference %.2e", sound.count, appDrift))

        // One clip at a time the two encoders must agree to float32 rounding. Batched, sums run in
        // another order and the difference grows over 18 layers, but by no more than the app's own
        // encoder drifts when it batches; a real mistake would be of the features' size.
        verdict(countMismatches.isEmpty && aloneLargest < scale * 1e-4 && largest <= max(appDrift * 2, scale * 1e-4),
                String(format: "batched encoder: %d clips together match each alone through the app's encoder "
                       + "to %.2e (one at a time %.2e), with features up to %.2f; worst: ", clips.count, largest,
                       aloneLargest, scale) + worst)

        // For the record: whether gradients pass the eval() calls in the app's own encoder, which
        // training doesn't use (AudioEncoder replaces it).
        let probe = clips[0]
        let appEncoder = valueAndGrad(model: trainable.model) { model, arrays in
            [model.getAudioFeatures(arrays[0], featureAttentionMask: arrays[1]).sum()]
        }
        let (_, appGradients) = appEncoder(trainable.model, [
            probe.mel.transposed(1, 0).expandedDimensions(axis: 0),
            MLXArray.ones([1, probe.mel.dim(0)], type: Int32.self),
        ])
        let reached = appGradients.flattened().filter { $0.0.hasPrefix("audio_tower.") }
        let nonZero = reached.filter { MLX.abs($0.1).max().item(Float.self) > 0 }.count
        say("NOTE  the app's encoder: gradients reach \(nonZero) of \(reached.count) encoder tensors through its eval() calls")

        // 3. One batch's loss is the sum of each example's on its own.
        let mixed = try (Array(sinhala.prefix(count / 2)) + Array(replayed.prefix(count / 2))).map { try Examples.prepare($0, for: trainable) }
        let batch = MicroBatch(examples: mixed.map(\.tokens), clips: mixed.map(\.mel), padding: format.padding)
        let together = Forward.tokenLosses(trainable, batch).sum().item(Float.self)
        var apart: Float = 0
        for example in mixed {
            let single = MicroBatch(examples: [example.tokens], clips: [example.mel], padding: format.padding)
            apart += Forward.tokenLosses(trainable, single).sum().item(Float.self)
        }
        let difference = abs(together - apart) / max(apart, 1e-6)
        verdict(difference < 1e-3, String(format: "batched loss: %.4f for %d examples together, %.4f one at a time (%.2e apart)",
                                           together, mixed.count, apart, difference))

        // 4. Teacher forcing on the replay set reproduces the base model's own greedy transcripts.
        let replayExamples = try replayed.map { try Examples.prepare($0, for: trainable) }
        let replayBatch = MicroBatch(examples: replayExamples.map(\.tokens), clips: replayExamples.map(\.mel), padding: format.padding)
        let greedy = Forward.greedyTargets(trainable, replayBatch).asArray(Int32.self)
        let matches = zip(greedy, replayBatch.layout.targetIDs).filter { $0 == $1 }.count
        let replayLoss = Forward.tokenLosses(trainable, replayBatch).mean().item(Float.self)
        let agreement = Double(matches) / Double(greedy.count)
        verdict(agreement > 0.97 && replayLoss < 0.1,
                String(format: "teacher forcing: %.2f%% of %d replay tokens are the model's greedy choice, loss %.4f per token",
                       agreement * 100, greedy.count, replayLoss))

        // 5. Gradients reach the encoder as well as the decoder.
        let trainer = try Trainer(trainable: trainable, computeType: .float32, trainEncoder: true)
        let small = Array(mixed.prefix(4))
        let smallBatch = MicroBatch(examples: small.map(\.tokens), clips: small.map(\.mel), padding: format.padding)
        let (fullLoss, gradients) = trainer.gradients(smallBatch)
        func norm(_ prefix: String) -> (Float, Int, Int) {
            let chosen = gradients.filter { $0.key.hasPrefix(prefix) }
            let zero = chosen.filter { ($0.value * $0.value).sum().item(Float.self) == 0 }.count
            return (Gradients.globalNorm(chosen).item(Float.self), chosen.count, zero)
        }
        let (encoderNorm, encoderTensors, encoderZero) = norm("audio_tower.")
        let (decoderNorm, decoderTensors, decoderZero) = norm("model.")
        verdict(encoderNorm.isFinite && encoderNorm > 0 && encoderZero == 0,
                String(format: "encoder gradients: norm %.4g over %d tensors, %d all zero", encoderNorm, encoderTensors, encoderZero))
        verdict(decoderNorm.isFinite && decoderNorm > 0 && decoderZero == 0,
                String(format: "decoder gradients: norm %.4g over %d tensors, %d all zero", decoderNorm, decoderTensors, decoderZero))

        // 6. Full precision transcribes as the 8-bit model the app runs does.
        if let english {
            let records = Array(try readRecords([english]).prefix(count))
            let full = try Evaluation.transcribe(trainable.model, records: records)
            let quantized = try Evaluation.transcribe(try await Qwen3ASRModel.fromModelDirectory(folder), records: records)
            let same = zip(full, quantized).filter { $0.hypothesis == $1.hypothesis }.count
            let fullReport = Evaluation.report(full), quantizedReport = Evaluation.report(quantized)
            verdict(abs(fullReport.tally.wordErrorRate - quantizedReport.tally.wordErrorRate) < 0.02,
                    String(format: "full precision against 8-bit: %d of %d transcripts identical; WER %.2f%% and %.2f%%",
                           same, records.count, fullReport.tally.wordErrorRate * 100, quantizedReport.tally.wordErrorRate * 100))
        }

        // 7. Computing in bfloat16 (`--compute bfloat16`) gives float32's gradients, to bfloat16's
        //    precision: the same direction for the encoder, the decoder and the whole, and the same
        //    size. Last, because it leaves the model in bfloat16.
        let halfTrainer = try Trainer(trainable: trainable, computeType: .bfloat16, trainEncoder: true)
        let (halfLoss, halfGradients) = halfTrainer.gradients(smallBatch)
        func gradientAgreement(_ prefix: String) -> (cosine: Float, ratio: Float) {
            let keys = gradients.keys.filter { $0.hasPrefix(prefix) }.sorted()
            let full = Dictionary(uniqueKeysWithValues: keys.map { ($0, gradients[$0]!) })
            let half = Dictionary(uniqueKeysWithValues: keys.map { ($0, halfGradients[$0]!.asType(.float32)) })
            let dot = MLX.stacked(keys.map { (full[$0]! * half[$0]!).sum() }).sum().item(Float.self)
            let fullNorm = Gradients.globalNorm(full).item(Float.self), halfNorm = Gradients.globalNorm(half).item(Float.self)
            return (dot / (fullNorm * halfNorm), halfNorm / fullNorm)
        }
        let whole = gradientAgreement(""), encoderPart = gradientAgreement("audio_tower."),
            decoderPart = gradientAgreement("model.")
        let sameKeys = Set(halfGradients.keys) == Set(gradients.keys)
        let sameLoss = abs(halfLoss - fullLoss) / fullLoss < 0.01
        let sameGradients = [whole, encoderPart, decoderPart].allSatisfy { part in
            part.cosine > 0.98 && abs(part.ratio - 1) < 0.05
        }
        verdict(sameKeys && sameLoss && sameGradients,
                String(format: "bfloat16 against float32: loss %.4f and %.4f; gradient cosine %.4f (encoder %.4f, decoder %.4f), "
                       + "norms in bfloat16 %.3f× (encoder %.3f×, decoder %.3f×)", halfLoss, fullLoss, whole.cosine,
                       encoderPart.cosine, decoderPart.cosine, whole.ratio, encoderPart.ratio, decoderPart.ratio))

        if !failures.isEmpty { throw ExitCode(1) }
    }
}
