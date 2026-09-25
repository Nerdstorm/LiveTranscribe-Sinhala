// trainer train: the fine-tune, resumable, with dev loss and CER along the way.
import ArgumentParser
import Foundation
import TrainerKit

struct Train: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Fine-tunes the model. Run it again with the same options to carry on after a stop."
    )

    @OptionGroup var model: ModelOption
    @Option(help: "The run's folder: settings, saved states, snapshots and metrics.") var run: String
    @Option(help: "Training records (repeat for several files).") var train: [String] = ["jsonl/train.jsonl", "jsonl/replay.jsonl"]
    @Option(help: "Dev records.") var dev = "jsonl/dev.jsonl"
    @Option(help: "Records in a language the model already knows (FLEURS English dev) that each evaluation also transcribes, to watch for forgetting.")
    var english: String?
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
            cacheLimitMB: cacheLimitMB, english: english
        )
        let stop = StopSignal()
        let training = try await TrainingRun(
            settings: settings, folder: RunFolder(URL(fileURLWithPath: run, isDirectory: true)),
            records: records, dev: try readRecords([dev]), english: try english.map { try readRecords([$0]) } ?? [],
            log: say
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
