import Foundation
@preconcurrency import MLX
import MLXAudioSTT

/// One recording transcribed as the app transcribes it.
public struct Transcript: Codable, Equatable, Sendable {
    public let audio: String
    public let reference: String
    public let expectedLanguage: String?
    public let hypothesis: String
    public let language: String?
    public let seconds: Double
    public let milliseconds: Double
}

/// What a set of transcripts adds up to.
public struct TranscriptionReport: Codable, Equatable, Sendable {
    public var tally = ErrorTally()
    /// How often the model named each language.
    public var languages: [String: Int] = [:]
    /// Transcripts whose language is the one their record's label names.
    public var languageMatches = 0
    public var audioSeconds = 0.0
    public var milliseconds: [Double] = []

    public init() {}

    public mutating func add(_ transcript: Transcript) {
        tally.add(reference: transcript.reference, hypothesis: transcript.hypothesis)
        languages[transcript.language ?? "none", default: 0] += 1
        if let expected = transcript.expectedLanguage, expected == transcript.language { languageMatches += 1 }
        audioSeconds += transcript.seconds
        milliseconds.append(transcript.milliseconds)
    }

    public var languageAccuracy: Double {
        tally.utterances > 0 ? Double(languageMatches) / Double(tally.utterances) : 0
    }
    /// Seconds of audio per second of transcribing.
    public var realTimeFactor: Double {
        let total = milliseconds.reduce(0, +) / 1000
        return total > 0 ? audioSeconds / total : 0
    }
    public var medianMilliseconds: Double {
        let sorted = milliseconds.sorted()
        return sorted.isEmpty ? 0 : sorted[sorted.count / 2]
    }

    public var summary: String {
        var parts = [
            String(format: "%d utterances: CER %.2f%%, WER %.2f%%", tally.utterances,
                   tally.characterErrorRate * 100, tally.wordErrorRate * 100),
            String(format: "language right %.1f%%", languageAccuracy * 100),
            "\(tally.truncated) cut short",
        ]
        if let recall = tally.latinRecall {
            parts.append(String(format: "English words in English letters %.1f%% of %d", recall * 100, tally.latinInReferences))
        }
        if let precision = tally.latinPrecision {
            parts.append(String(format: "English-letter words right %.1f%% of %d", precision * 100, tally.latinInHypotheses))
        }
        parts.append(String(format: "%.0f× real time, median %.0f ms", realTimeFactor, medianMilliseconds))
        return parts.joined(separator: "; ")
    }
}

public enum Evaluation {
    /// Mean cross-entropy per target token over `records`, without gradients.
    public static func loss(_ trainable: TrainableModel, records: [Record], tokenBudget: Int, group: Int = 64) throws -> (loss: Double, targets: Int) {
        var sum = 0.0
        var targets = 0
        for start in stride(from: 0, to: records.count, by: group) {
            let prepared = try records[start ..< min(start + group, records.count)].map {
                try Examples.prepare($0, for: trainable)
            }
            for batch in Examples.microBatches(prepared, tokenBudget: tokenBudget, padding: trainable.format.padding) {
                let batchSum = Forward.tokenLosses(trainable, batch).sum()
                eval(batchSum)
                sum += Double(batchSum.item(Float.self))
                targets += batch.layout.targetCount
            }
            Memory.clearCache()
        }
        return (targets > 0 ? sum / Double(targets) : 0, targets)
    }

    /// The language a record's label names: "language Sinhala<asr_text>…" names Sinhala.
    public static func language(of text: String) -> String? {
        guard text.hasPrefix("language "), let marker = text.range(of: Records.transcriptMarker) else { return nil }
        let name = text[text.index(text.startIndex, offsetBy: "language ".count) ..< marker.lowerBound]
        return name.isEmpty ? nil : String(name)
    }

    /// Transcribes each record's recording with the model's own generate and its defaults, with no
    /// language given, as LiveTranscribe runs it.
    public static func transcribe(_ model: Qwen3ASRModel, records: [Record], language: String? = nil,
                                  each: (Int, Transcript) -> Void = { _, _ in }) throws -> [Transcript] {
        let defaults = model.defaultGenerationParameters
        let parameters = STTGenerateParameters(
            maxTokens: defaults.maxTokens, temperature: defaults.temperature, topP: defaults.topP,
            topK: defaults.topK, verbose: false, language: language,
            chunkDuration: defaults.chunkDuration, minChunkDuration: defaults.minChunkDuration,
            repetitionPenalty: defaults.repetitionPenalty, repetitionContextSize: defaults.repetitionContextSize
        )
        var transcripts: [Transcript] = []
        for (index, record) in records.enumerated() {
            let samples: MLXArray
            do {
                samples = try AudioFeatures.samples(of: URL(fileURLWithPath: record.audio))
            } catch {
                throw ExamplesError.audio(path: record.audio, underlying: error)
            }
            let started = Date()
            let output = model.generate(audio: samples, generationParameters: parameters)
            let transcript = Transcript(
                audio: record.audio,
                reference: Records.transcript(of: record.text),
                expectedLanguage: Self.language(of: record.text),
                hypothesis: output.text,
                language: output.language,
                seconds: Double(samples.dim(0)) / Double(AudioFeatures.sampleRate),
                milliseconds: Date().timeIntervalSince(started) * 1000
            )
            transcripts.append(transcript)
            each(index, transcript)
            if index % 50 == 49 { Memory.clearCache() }
        }
        return transcripts
    }

    public static func report(_ transcripts: [Transcript]) -> TranscriptionReport {
        var report = TranscriptionReport()
        for transcript in transcripts { report.add(transcript) }
        return report
    }
}
