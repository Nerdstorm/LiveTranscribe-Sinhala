// trainer overfit: trains on a few utterances until it can transcribe them from their audio.
import ArgumentParser
import Foundation
@preconcurrency import MLX
import TrainerKit

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
