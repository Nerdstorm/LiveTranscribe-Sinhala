// trainer bench: times optimizer steps on real data.
import ArgumentParser
import Foundation
@preconcurrency import MLX
import TrainerKit

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
