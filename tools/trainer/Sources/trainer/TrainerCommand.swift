// trainer: fine-tunes Qwen3-ASR 0.6B for Sinhala on this Mac with MLX (docs/training.md).
//
//   trainer check      correctness checks on the model and the data before any training
//   trainer overfit    trains on a few utterances until it can transcribe them from their audio
//   trainer bench      times optimizer steps on real data
//   trainer train      the fine-tune: resumable, with dev loss and CER along the way
//   trainer transcribe transcribes and scores a JSONL with a model folder, as the app runs it
//   trainer export     turns a snapshot into a model folder the app loads
import ArgumentParser
import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXNN
import TrainerKit

@main
struct TrainerCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "trainer",
        abstract: "Fine-tunes Qwen3-ASR for Sinhala with MLX.",
        subcommands: [Check.self, Overfit.self, Bench.self, Train.self, Transcribe.self, ExportCommand.self]
    )

    /// ArgumentParser's async entry point, with standard output line-buffered so a log file
    /// follows a long run as it goes.
    static func main() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        do {
            var command = try parseAsRoot()
            if var asynchronous = command as? AsyncParsableCommand {
                try await asynchronous.run()
            } else {
                try command.run()
            }
        } catch {
            exit(withError: error)
        }
    }
}

func say(_ message: String) {
    print("\(Date().formatted(date: .omitted, time: .standard))  \(message)")
}

/// A model folder, or a Hugging Face repo id resolved to its folder in mlx-audio's cache
/// (downloaded if it isn't there), as the app finds it.
func modelFolder(_ model: String) async throws -> URL {
    var isFolder: ObjCBool = false
    if FileManager.default.fileExists(atPath: model, isDirectory: &isFolder), isFolder.boolValue {
        return URL(fileURLWithPath: model, isDirectory: true)
    }
    guard let repo = Repo.ID(rawValue: model) else { throw ValidationError("\(model) is neither a folder nor a repo id") }
    return try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors")
}

func readRecords(_ paths: [String]) throws -> [Record] {
    try paths.flatMap { try Records.read(URL(fileURLWithPath: $0)) }
}

struct ModelOption: ParsableArguments {
    @Option(help: "The model to start from: a folder, or a Hugging Face repo id.")
    var model = "mlx-community/Qwen3-ASR-0.6B-8bit"
}

// MARK: - check

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

// MARK: - overfit

struct Overfit: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Trains on a few utterances until it can transcribe them from their audio alone."
    )

    @OptionGroup var model: ModelOption
    @Option(help: "Records to pick from.") var records = "jsonl/train.jsonl"
    @Option(help: "Utterances to learn.") var count = 32
    @Option(help: "Optimizer steps, each over all the utterances.") var steps = 100
    @Option(help: "Peak learning rate, reached after a linear warm-up over the first tenth of the steps, then decaying to 0.")
    var rate = 1e-4
    @Option(help: "Most padded tokens in one forward pass.") var tokenBudget = 4096
    @Option(help: "float32 or bfloat16.") var compute: ComputeType = .float32

    func run() async throws {
        let trainable = try await TrainableModel.load(directory: try await modelFolder(model.model))
        let chosen = Examples.sample(try readRecords([records]), count: count, seed: 2)
        let examples = try chosen.map { try Examples.prepare($0, for: trainable) }
        let trainer = try Trainer(trainable: trainable, computeType: compute, trainEncoder: true)
        let schedule = Schedule(peak: rate, totalSteps: steps, warmupFraction: 0.1)
        var loss = Double.infinity
        for step in 1 ... steps {
            let result = trainer.step(examples, rate: schedule.rate(step: step - 1), clip: 1, tokenBudget: tokenBudget)
            loss = result.loss
            if step == 1 || step % 10 == 0 {
                say(String(format: "step %d: loss %.4f, grad norm %.3f", step, result.loss, result.gradientNorm))
            }
        }

        // A low loss alone can come from the decoder learning the sentences by heart while it
        // still can't tell the clips apart: every token after a sentence's first few is easy. So
        // each clip is also transcribed from its audio alone, through the app's generate, and
        // teacher forcing through the training forward pass shows where each label first goes
        // wrong: at its first transcript token if the audio isn't being told apart, or nowhere if
        // training learned the clip and only the app's path disagrees.
        let shown = Array(examples.prefix(8))
        let transcripts = try Evaluation.transcribe(trainable.model, records: shown.map(\.record))
        for (example, transcript) in zip(shown, transcripts) {
            let batch = MicroBatch(examples: [example.tokens], clips: [example.mel], padding: trainable.format.padding)
            let greedy = Forward.greedyTargets(trainable, batch).asArray(Int32.self)
            let miss = zip(greedy, batch.layout.targetIDs).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            let forced = miss.map { "teacher forcing first misses label token \($0 + 1) of \(greedy.count)" }
                ?? "teacher forcing gets all \(greedy.count) label tokens"
            say("\(transcript.hypothesis == transcript.reference ? "same" : "DIFF")  \(forced)  \(transcript.hypothesis)")
        }
        let report = Evaluation.report(transcripts)
        say("after \(steps) steps: loss \(String(format: "%.4f", loss)); on 8 of them \(report.summary)")
        if loss > 0.05 || report.tally.characterErrorRate > 0.05 { throw ExitCode(1) }
    }
}

extension ComputeType: ExpressibleByArgument {}

// MARK: - bench

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Times optimizer steps on real training data.")

    @OptionGroup var model: ModelOption
    @Option(help: "Training records (repeat for several files).") var train: [String] = ["jsonl/train.jsonl", "jsonl/replay.jsonl"]
    @Option(help: "Steps to time, after one warm-up step.") var steps = 5
    @Option(help: "Utterances per optimizer step.") var utterances = 128
    @Option(help: "Most padded tokens in one forward pass. 4096 peaks at 35 GB in bfloat16 with the encoder trained; 8192 swaps on a 48 GB Mac.")
    var tokenBudget = 4096
    @Option(help: "float32 or bfloat16 (the weights the optimizer updates stay float32 either way).")
    var compute: ComputeType = .bfloat16
    @Flag(help: "Train the decoder only.") var freezeEncoder = false
    @Option(help: "MLX's buffer cache limit, MB.") var cacheLimitMB = 4096

    func run() async throws {
        Memory.cacheLimit = cacheLimitMB * 1024 * 1024
        let trainable = try await TrainableModel.load(directory: try await modelFolder(model.model))
        let all = try readRecords(train)
        let plan = EpochPlan(recordCount: all.count, utterancesPerStep: utterances, seed: 3)
        let order = plan.order(epoch: 0)
        let trainer = try Trainer(trainable: trainable, computeType: compute, trainEncoder: !freezeEncoder)
        say("memory limit \(Memory.memoryLimit / 1_000_000) MB; \(compute.rawValue), encoder \(freezeEncoder ? "frozen" : "trained"), budget \(tokenBudget) tokens")
        var timed: [Double] = []
        var utterancesDone = 0
        for step in 0 ... steps {
            let started = Date()
            let examples = try plan.records(step: step, in: order).map { try Examples.prepare(all[$0], for: trainable) }
            let loaded = Date()
            Memory.peakMemory = 0
            let result = trainer.step(examples, rate: 1e-5, clip: 1, tokenBudget: tokenBudget)
            let seconds = Date().timeIntervalSince(started)
            say(String(format: "step %d: %.1f s (%.1f s loading), %d micro-batches, %d padded tokens (%d audio features), %d targets, loss %.4f (%@), peak %.1f GB",
                       step, seconds, loaded.timeIntervalSince(started), result.microBatches, result.paddedTokens,
                       result.audioFeatures, result.targets, result.loss, result.parts, Double(Memory.peakMemory) / 1e9))
            if step > 0 {
                timed.append(seconds)
                utterancesDone += examples.count
            }
        }
        let total = timed.reduce(0, +)
        let perStep = total / Double(timed.count)
        say(String(format: "%.1f s a step, %.2f utterances a second; an epoch of %d steps would take %.1f h",
                   perStep, Double(utterancesDone) / total, EpochPlan(recordCount: all.count, utterancesPerStep: utterances, seed: 0).stepsPerEpoch,
                   perStep * Double(EpochPlan(recordCount: all.count, utterancesPerStep: utterances, seed: 0).stepsPerEpoch) / 3600))
    }
}

// MARK: - train

struct Train: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Fine-tunes the model. Run it again with the same options to carry on after a stop."
    )

    @OptionGroup var model: ModelOption
    @Option(help: "The run's folder: settings, saved states, snapshots and metrics.") var run: String
    @Option(help: "Training records (repeat for several files).") var train: [String] = ["jsonl/train.jsonl", "jsonl/replay.jsonl"]
    @Option(help: "Dev records.") var dev = "jsonl/dev.jsonl"
    @Option(help: "Peak learning rate.") var rate: Double
    @Option var epochs = 1
    @Option var seed: UInt64 = 52
    @Option(help: "Utterances per optimizer step.") var utterances = 128
    @Option(help: "Most padded tokens in one forward pass. 4096 peaks at 35 GB in bfloat16 with the encoder trained; 8192 swaps on a 48 GB Mac.")
    var tokenBudget = 4096
    @Option(help: "Share of the steps spent warming up.") var warmup = 0.02
    @Option(help: "Gradient norm clip.") var clip = 1.0
    @Option(help: "float32 or bfloat16 (the weights the optimizer updates stay float32 either way).")
    var compute: ComputeType = .bfloat16
    @Flag(help: "Train the decoder only.") var freezeEncoder = false
    @Option(help: "Minutes between saved states.") var saveEvery = 30.0
    @Option(help: "Saved states to keep.") var keep = 2
    @Option(help: "Steps between dev-loss checks.") var devLossEvery = 200
    @Option(help: "Dev utterances the loss is measured on.") var devLossUtterances = 1000
    @Option(help: "Snapshots with dev CER per epoch.") var evaluations = 4
    @Option(help: "Dev utterances transcribed for CER.") var devTranscribe = 500
    @Option(help: "MLX's buffer cache limit, MB.") var cacheLimitMB = 4096
    @Option(help: "Stop after this many steps in all (the schedule stays the whole run's).") var stopAfter: Int?

    func run() async throws {
        let folder = try await modelFolder(model.model)
        let records = try readRecords(train)
        let settings = RunSettings(
            model: folder.path, train: train, dev: dev, trainRecords: records.count,
            dataDigest: Examples.digest(records), seed: seed, epochs: epochs, utterancesPerStep: utterances,
            tokenBudget: tokenBudget, peakRate: rate, warmupFraction: warmup, clip: clip, computeType: compute,
            trainEncoder: !freezeEncoder, saveEveryMinutes: saveEvery, keepStates: keep, devLossEvery: devLossEvery,
            devLossUtterances: devLossUtterances, evaluationsPerEpoch: evaluations, devTranscribeUtterances: devTranscribe,
            cacheLimitMB: cacheLimitMB
        )
        let stop = StopSignal()
        let training = try await TrainingRun(
            settings: settings, folder: RunFolder(URL(fileURLWithPath: run, isDirectory: true)),
            records: records, dev: try readRecords([dev]), log: say
        )
        try training.run(stopAfter: stopAfter, interrupted: { stop.requested })
    }
}

/// Set by SIGINT or SIGTERM: the run saves its state after the step it's in, then exits.
final class StopSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    private var sources: [DispatchSourceSignal] = []

    init() {
        for signalNumber in [SIGINT, SIGTERM] {
            signal(signalNumber, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .global())
            source.setEventHandler { [weak self] in
                say("Stopping after this step (signal \(signalNumber)); send it again to stop at once")
                guard let self else { return }
                self.lock.lock()
                let second = self.flag
                self.flag = true
                self.lock.unlock()
                if second { exit(130) }
            }
            source.resume()
            sources.append(source)
        }
    }

    var requested: Bool {
        lock.lock()
        defer { lock.unlock() }
        return flag
    }
}

// MARK: - transcribe

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribes a JSONL's recordings with a model folder as the app loads it, and scores them."
    )

    @Option(help: "A model folder or repo id, loaded as the app loads it.") var model: String
    @Option(help: "Records to transcribe; their text is the reference.") var records: String
    @Option(help: "Where to write the transcripts (TSV); a .json report goes beside it.") var out: String
    @Option(help: "Transcribe only the first N.") var limit: Int?
    @Option(help: "Tell the model the language instead of letting it choose (the app doesn't).") var language: String?

    func run() async throws {
        let loaded = try await Qwen3ASRModel.fromModelDirectory(try await modelFolder(model))
        var chosen = try readRecords([records])
        if let limit { chosen = Array(chosen.prefix(limit)) }
        let transcripts = try Evaluation.transcribe(loaded, records: chosen, language: language) { index, _ in
            if index % 100 == 99 { say("\(index + 1) of \(chosen.count)") }
        }
        let report = Evaluation.report(transcripts)
        let url = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Transcripts.write(transcripts, to: url)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: url.deletingPathExtension().appendingPathExtension("json"))
        say(report.summary)
    }
}

// MARK: - export

struct ExportCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export",
        abstract: "Writes a snapshot's weights as a model folder the app loads, quantized like the base model."
    )

    @Option(help: "A snapshot or saved-state folder (its weights.safetensors).") var weights: String
    @Option(help: "The model folder or repo id training started from.") var base = "mlx-community/Qwen3-ASR-0.6B-8bit"
    @Option(help: "The folder to write.") var out: String
    @Option(help: "The language to add to support_languages.") var language = "Sinhala"
    @Option(help: "A JSONL whose first few recordings the exported model transcribes as a check.") var check: String?

    func run() async throws {
        let baseFolder = try await modelFolder(base)
        let trained = try loadArrays(url: Checkpoints.weights(in: URL(fileURLWithPath: weights, isDirectory: true)))
        let output = URL(fileURLWithPath: out, isDirectory: true)
        let summary = try Export.write(weights: trained, base: baseFolder, to: output, language: language)
        say("wrote \(output.path): \(summary.tensors) tensors, \(summary.quantizedLayers) quantized layers, "
            + "\(summary.bytes / 1_000_000) MB, \(summary.parameters) parameters")
        let reloaded = try await Qwen3ASRModel.fromModelDirectory(output)
        guard reloaded.config.supportLanguages.contains(language) else {
            throw ValidationError("the exported config.json doesn't list \(language)")
        }
        if let check {
            let records = Array(try readRecords([check]).prefix(5))
            for transcript in try Evaluation.transcribe(reloaded, records: records) {
                say("\(transcript.language ?? "?"): \(transcript.hypothesis)")
            }
        }
        say("reloaded as the app loads it")
    }
}
