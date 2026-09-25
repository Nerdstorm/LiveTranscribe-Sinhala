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
import MLXAudioCore
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

extension ComputeType: ExpressibleByArgument {}
