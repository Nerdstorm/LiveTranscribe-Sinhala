import Foundation
import LabelFile
import Testing

struct LabelFileTests {
    private let header = LabelFile.header + "\n"

    @Test func aLineKeepsTheTextOnOneLine() {
        let line = LabelFile.line(file: "a.wav", seconds: 1.234, milliseconds: 56.7, text: " one\ttwo\nthree\u{2028}four ")
        #expect(line == "a.wav\t1.23\t57\tone two three four\n")
    }

    @Test func anEmptyTextIsStillALine() throws {
        let line = LabelFile.line(file: "a.wav", seconds: 1, milliseconds: 10, text: "")
        #expect(line == "a.wav\t1.00\t10\t\n")
        #expect(try LabelFile.resume(header + line).files == ["a.wav"])
    }

    @Test func resumingKeepsEveryCompleteLine() throws {
        let contents = header + "a.wav\t1.00\t10\tone\n" + "b.wav\t2.00\t20\ttwo\n"
        let resumed = try LabelFile.resume(contents)
        #expect(resumed.files == ["a.wav", "b.wav"])
        #expect(resumed.contents == contents)
    }

    @Test func resumingDropsALineCutOff() throws {
        let complete = header + "a.wav\t1.00\t10\tone\n"
        let resumed = try LabelFile.resume(complete + "b.wav\t2.0")
        #expect(resumed.files == ["a.wav"])
        #expect(resumed.contents == complete)
    }

    @Test(arguments: ["", "file\tsec"])
    func resumingANewOrCutOffFileStartsOver(contents: String) throws {
        let resumed = try LabelFile.resume(contents)
        #expect(resumed.files.isEmpty)
        #expect(resumed.contents == header)
    }

    @Test func resumingRefusesAnotherKindOfFile() {
        #expect(throws: LabelFile.Problem.unexpectedHeader("id\ttext")) {
            try LabelFile.resume("id\ttext\na\tb\n")
        }
    }

    @Test func resumingRefusesAMalformedLine() {
        #expect(throws: LabelFile.Problem.malformedLine(3)) {
            try LabelFile.resume(header + "a.wav\t1.00\t10\tone\n" + "b.wav two\n")
        }
    }

    @Test func recordingsAreTheAudioFilesInNameOrder() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for name in ["b.wav", "notes.txt", "c.FLAC", "a.wav"] {
            try Data().write(to: folder.appending(path: name))
        }
        #expect(try LabelFile.recordings(in: folder).map(\.lastPathComponent) == ["a.wav", "b.wav", "c.FLAC"])
    }
}
