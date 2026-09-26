// trainer export: turns a snapshot into a model folder the app loads.
import ArgumentParser
import Foundation
@preconcurrency import MLX
import MLXAudioSTT
import TrainerKit

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
    @Option(help: "How much of the fine-tuned weights to keep, from 0 to 1; the rest comes from the base model (weight-space ensembling). 1 exports them as they are.")
    var blend: Float = 1
    @Option(help: "The same for the audio encoder alone; --blend then covers the text model. Defaults to --blend.")
    var blendAudio: Float?

    func run() async throws {
        let baseFolder = try await modelFolder(base)
        var trained = try loadArrays(url: Checkpoints.weights(in: URL(fileURLWithPath: weights, isDirectory: true)))
        let audio = blendAudio ?? blend
        if blend < 1 || audio < 1 {
            trained = try Export.blend(trained, with: Export.baseWeights(in: baseFolder)) {
                $0.hasPrefix(Export.audioPrefix) ? audio : blend
            }
            say(String(format: "kept %.2f of the fine-tuned text model and %.2f of the audio encoder; the rest is the base model's",
                       blend, audio))
        }
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
