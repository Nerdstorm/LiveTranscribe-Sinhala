// pseudo-label <model> <language> <audio folder> <labels.tsv>
//
// Labels every wav or flac file in the folder, in name order, with a Qwen3-ASR model told the
// language: the model's own transcripts, which the replay set trains back into it. The model is
// loaded as LiveTranscribe loads it and decodes greedily, with its defaults. The language must be
// one the model lists (support_languages in its config.json), such as English or Chinese.
//
// Each recording's line is appended to the labels file as soon as it's labelled: file, seconds of
// audio, milliseconds taken, text. To continue a run that stopped, run it again with the same
// labels file: recordings it already has are skipped. Progress goes to standard error, and the
// exit status is 1 if any recording failed (run again to retry those).
import Foundation
import LabelFile
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT

let sampleRate = 16_000

func note(_ message: String) {
    FileHandle.standardError.write(Data("pseudo-label: \(message)\n".utf8))
}

func fail(_ message: String) -> Never {
    note(message)
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count == 5 else {
    fail("usage: pseudo-label <model> <language> <audio folder> <labels.tsv>")
}
let repo = arguments[1]
let language = arguments[2]
let folder = URL(fileURLWithPath: arguments[3], isDirectory: true)
let output = URL(fileURLWithPath: arguments[4])

let recordings: [URL]
do {
    recordings = try LabelFile.recordings(in: folder)
} catch {
    fail("can't read \(folder.path): \(error.localizedDescription)")
}
guard !recordings.isEmpty else { fail("no wav or flac files in \(folder.path)") }

let loaded: any STTGenerationModel
do {
    // By the model type in its config.json, as LiveTranscribe's MLXTranscriber loads it.
    loaded = try await STT.loadModel(modelRepo: repo)
} catch {
    fail("can't load \(repo): \(error)")
}
guard let model = loaded as? Qwen3ASRModel else { fail("\(repo) isn't a Qwen3-ASR model") }
let languages = model.config.supportLanguages
guard languages.contains(language) else {
    fail("\(repo) doesn't list \(language); it lists \(languages.joined(separator: ", "))")
}
// The model's defaults, with the language given: the prompt ends "language English<asr_text>".
let defaults = model.defaultGenerationParameters
let parameters = STTGenerateParameters(
    maxTokens: defaults.maxTokens, temperature: defaults.temperature, topP: defaults.topP,
    topK: defaults.topK, verbose: defaults.verbose, language: language,
    chunkDuration: defaults.chunkDuration, minChunkDuration: defaults.minChunkDuration,
    repetitionPenalty: defaults.repetitionPenalty, repetitionContextSize: defaults.repetitionContextSize
)

var labelledBefore = Set<String>()
do {
    if FileManager.default.fileExists(atPath: output.path) {
        let resumed = try LabelFile.resume(String(contentsOf: output, encoding: .utf8))
        try resumed.contents.write(to: output, atomically: true, encoding: .utf8)
        labelledBefore = resumed.files
    } else {
        try (LabelFile.header + "\n").write(to: output, atomically: true, encoding: .utf8)
    }
} catch {
    fail("can't continue \(output.path): \(error)")
}
let remaining = recordings.filter { !labelledBefore.contains($0.lastPathComponent) }
note("\(recordings.count) recordings in \(folder.path), \(recordings.count - remaining.count) already labelled")
guard !remaining.isEmpty else { exit(0) }

let labels: FileHandle
do {
    labels = try FileHandle(forWritingTo: output)
    try labels.seekToEnd()
} catch {
    fail("can't write \(output.path): \(error.localizedDescription)")
}

let clock = ContinuousClock()
let started = clock.now
var labelled = 0
var failed = 0
var audioSeconds = 0.0
for (index, recording) in remaining.enumerated() {
    let name = recording.lastPathComponent
    do {
        let (_, audio) = try loadAudioArray(from: recording, sampleRate: sampleRate)
        let seconds = Double(audio.size) / Double(sampleRate)
        let before = clock.now
        let text = model.generate(audio: audio, generationParameters: parameters).text
        let milliseconds = before.duration(to: clock.now) / .milliseconds(1)
        try labels.write(contentsOf: Data(LabelFile.line(file: name, seconds: seconds, milliseconds: milliseconds, text: text).utf8))
        labelled += 1
        audioSeconds += seconds
    } catch {
        failed += 1
        note("\(name) failed: \(error)")
    }
    if (index + 1) % 100 == 0 || index + 1 == remaining.count {
        let elapsed = started.duration(to: clock.now) / .seconds(1)
        let speed = elapsed > 0 ? audioSeconds / elapsed : 0
        note(String(format: "%d of %d: %.0f s of audio in %.0f s (%.0f× real time)", index + 1, remaining.count, audioSeconds, elapsed, speed))
    }
}
try? labels.close()
if failed > 0 { fail("\(failed) of \(remaining.count) recordings failed; run again to retry them") }
