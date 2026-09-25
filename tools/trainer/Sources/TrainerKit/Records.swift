import Foundation

/// One line of the fine-tuning JSONL that prepare_data.py and prepare_replay.py write, in the
/// format of Qwen's qwen3_asr_sft.py: a recording and the text the model should write for it,
/// "language Sinhala<asr_text>…".
public struct Record: Codable, Equatable, Hashable, Sendable {
    public let audio: String
    public let text: String

    public init(audio: String, text: String) {
        self.audio = audio
        self.text = text
    }
}

public enum RecordsError: Error, CustomStringConvertible, Equatable {
    case malformedLine(file: String, line: Int, reason: String)
    case noRecords(file: String)

    public var description: String {
        switch self {
        case .malformedLine(let file, let line, let reason): "\(file):\(line): \(reason)"
        case .noRecords(let file): "\(file) has no records"
        }
    }
}

public enum Records {
    /// The text a record's label must start with: the language as Qwen3-ASR writes it before the
    /// transcript.
    public static let transcriptMarker = "<asr_text>"

    /// Every record in a JSONL file, in order. Blank lines are skipped; a line that isn't a record
    /// with a non-empty audio path and a text containing the transcript marker stops the read.
    public static func read(_ url: URL) throws -> [Record] {
        try parse(String(contentsOf: url, encoding: .utf8), file: url.path)
    }

    public static func parse(_ contents: String, file: String) throws -> [Record] {
        let decoder = JSONDecoder()
        var records: [Record] = []
        for (index, line) in contents.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            guard !line.allSatisfy(\.isWhitespace) else { continue }
            let record: Record
            do {
                record = try decoder.decode(Record.self, from: Data(line.utf8))
            } catch {
                throw RecordsError.malformedLine(file: file, line: index + 1, reason: "not a JSON record with audio and text")
            }
            guard !record.audio.isEmpty else {
                throw RecordsError.malformedLine(file: file, line: index + 1, reason: "the audio path is empty")
            }
            guard record.text.contains(transcriptMarker) else {
                throw RecordsError.malformedLine(file: file, line: index + 1, reason: "the text has no \(transcriptMarker)")
            }
            records.append(record)
        }
        guard !records.isEmpty else { throw RecordsError.noRecords(file: file) }
        return records
    }

    /// The transcript part of a record's text: what follows the marker.
    public static func transcript(of text: String) -> String {
        guard let marker = text.range(of: transcriptMarker) else { return text }
        return String(text[marker.upperBound...])
    }
}
