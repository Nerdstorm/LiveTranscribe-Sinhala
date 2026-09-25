// trainer transcribe: transcribes and scores a JSONL with a model folder, as the app runs it.
import ArgumentParser
import Foundation
import MLXAudioSTT
import TrainerKit

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
