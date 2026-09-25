import Foundation

/// The labels file pseudo-label writes: a header, then one line per recording (file, seconds of
/// audio, milliseconds taken, text). Each line is appended as its recording is labelled, so a run
/// that stops can be continued.
public enum LabelFile {
    public static let header = "file\tseconds\tms\ttext"

    /// The recordings pseudo-label reads.
    public static let audioExtensions: Set<String> = ["wav", "flac"]

    public enum Problem: Error, Equatable, CustomStringConvertible {
        /// The first line isn't ``LabelFile/header``: the file is something else.
        case unexpectedHeader(String)
        /// A line (numbered from 1) that doesn't have four columns and a file name.
        case malformedLine(Int)

        public var description: String {
            switch self {
            case .unexpectedHeader(let line):
                "it starts with \"\(line)\", not a labels header; give a new file or the one a run made"
            case .malformedLine(let number):
                "line \(number) isn't file, seconds, ms and text separated by tabs"
            }
        }
    }

    /// The audio files in a folder, in name order.
    public static func recordings(in folder: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            .filter { audioExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// What to keep of a labels file a run left, and the recordings it already has. A last line
    /// without a line break was cut off when the run stopped: it's dropped, and its recording is
    /// labelled again.
    public static func resume(_ contents: String) throws -> (contents: String, files: Set<String>) {
        var lines = contents.components(separatedBy: "\n")
        // What follows the last line break: nothing, or a line cut off.
        lines.removeLast()
        guard let first = lines.first else { return (header + "\n", []) }
        guard first == header else { throw Problem.unexpectedHeader(first) }
        var files = Set<String>()
        for (index, line) in lines.enumerated().dropFirst() {
            let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard fields.count == 4, !fields[0].isEmpty else { throw Problem.malformedLine(index + 1) }
            files.insert(String(fields[0]))
        }
        return (lines.map { $0 + "\n" }.joined(), files)
    }

    /// One line of the file, with its line break. Tabs and line breaks in the text become spaces,
    /// so each recording stays one line.
    public static func line(file: String, seconds: Double, milliseconds: Double, text: String) -> String {
        let flat = text.components(separatedBy: .newlines).joined(separator: " ")
            .replacingOccurrences(of: "\t", with: " ")
            .trimmingCharacters(in: .whitespaces)
        return "\(file)\t\(String(format: "%.2f", seconds))\t\(String(format: "%.0f", milliseconds))\t\(flat)\n"
    }
}
