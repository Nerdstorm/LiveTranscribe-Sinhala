import Foundation
import Testing
import TrainerKit

struct RecordsTests {
    @Test func readsEveryRecordAndSkipsBlankLines() throws {
        let contents = """
        {"audio": "/a.flac", "text": "language Sinhala<asr_text>මම"}

        {"audio": "/b.wav", "text": "language English<asr_text>hello"}
        """
        let records = try Records.parse(contents, file: "t.jsonl")
        #expect(records == [
            Record(audio: "/a.flac", text: "language Sinhala<asr_text>මම"),
            Record(audio: "/b.wav", text: "language English<asr_text>hello"),
        ])
    }

    @Test func aLineThatIsntARecordStopsTheRead() {
        #expect(throws: RecordsError.malformedLine(file: "t.jsonl", line: 2, reason: "not a JSON record with audio and text")) {
            try Records.parse("{\"audio\": \"/a\", \"text\": \"language X<asr_text>a\"}\n{\"audio\": \"/b\"}", file: "t.jsonl")
        }
    }

    @Test func aTextWithoutTheMarkerIsRejected() {
        #expect(throws: RecordsError.malformedLine(file: "t.jsonl", line: 1, reason: "the text has no <asr_text>")) {
            try Records.parse("{\"audio\": \"/a\", \"text\": \"hello\"}", file: "t.jsonl")
        }
    }

    @Test func anEmptyFileIsAnError() {
        #expect(throws: RecordsError.noRecords(file: "t.jsonl")) { try Records.parse("\n\n", file: "t.jsonl") }
    }

    @Test func theTranscriptIsWhatFollowsTheMarker() {
        #expect(Records.transcript(of: "language Sinhala<asr_text>මම meeting එකට") == "මම meeting එකට")
    }
}

/// A stand-in tokenizer: one id per character, with a few special tokens matched first, as the
/// real one matches them.
private let specials: [String: Int] = [
    "<|im_start|>": 1000, "<|im_end|>": 1001, "<|audio_start|>": 1002, "<|audio_end|>": 1003,
    "<|audio_pad|>": 1004, "<asr_text>": 1005, "<|endoftext|>": 1006,
]

private let toyEncode: @Sendable (String) -> [Int] = { text in
    var ids: [Int] = []
    var rest = Substring(text)
    outer: while !rest.isEmpty {
        for (token, id) in specials where rest.hasPrefix(token) {
            ids.append(id)
            rest = rest.dropFirst(token.count)
            continue outer
        }
        ids.append(Int(rest.unicodeScalars.first!.value))
        rest = rest.dropFirst()
    }
    return ids
}

struct PromptTests {
    let format = PromptFormat(encode: toyEncode, audioToken: 1004, endOfTurn: 1001, padding: 1006)

    @Test func thePromptIsTheAppsWithTheClipsAudioTokens() {
        let prompt = format.promptIDs(audioTokens: 3)
        let text = PromptFormat.prefixText + String(repeating: "<|audio_pad|>", count: 3) + PromptFormat.suffixText
        #expect(prompt.map(Int.init) == toyEncode(text))
    }

    @Test func theLabelEndsTheTurn() {
        #expect(format.labelIDs("ab").map(Int.init) == [97, 98, 1001])
    }

    @Test func anExampleIsThePromptThenTheLabel() {
        let example = format.example(text: "language X<asr_text>hi", audioTokens: 2, audioFeatures: 2)
        let whole = toyEncode(PromptFormat.prefixText + "<|audio_pad|><|audio_pad|>" + PromptFormat.suffixText
            + "language X<asr_text>hi<|im_end|>")
        #expect(example.ids.map(Int.init) == whole)
        #expect(example.audioStart == format.prefix.count)
        #expect(example.ids[example.audioStart ..< example.audioStart + 2].allSatisfy { $0 == 1004 })
        #expect(example.labelLength == toyEncode("language X<asr_text>hi").count + 1)
        // The positions whose next token is a label token: the prompt's last token through the
        // label's second-to-last.
        #expect(example.targetPositions == (example.promptLength - 1) ..< (example.ids.count - 1))
    }
}

struct AudioFeaturesTests {
    @Test func placeholdersAreCountedAsTheAppCountsThem() {
        // mlx-audio-swift's getFeatExtractOutputLengths in float32 (see audioTokens); Qwen's
        // integer version gives 71 for 541 frames, 77 for 591 and 87 for 671.
        let expected = [100: 13, 101: 14, 200: 26, 541: 76, 591: 88, 671: 96, 1000: 130, 1303: 170]
        for (frames, tokens) in expected {
            #expect(AudioFeatures.audioTokens(frames: frames) == tokens, "frames \(frames)")
        }
    }
}

struct EncoderLayoutTests {
    @Test func aClipIsChunkedAndItsLastChunkCappedAtWhatTheConvolutionsMake() {
        // 591 frames: five chunks of 100 (13 rows each) and one of 91, whose float count (23)
        // is more than the 13 rows the convolutions make of a chunk padded to 100.
        let layout = EncoderLayout(frames: [591])
        #expect(layout.chunkLengths == [100, 100, 100, 100, 100, 91])
        #expect(layout.paddedLength == 100)
        #expect(layout.convolvedLength == 13)
        #expect(layout.chunkRows == [13, 13, 13, 13, 13, 13])
        #expect(layout.clipFeatures == [78])
        #expect(layout.windows == [0 ..< 78])
    }

    @Test func windowsHoldEightChunksAndNeverSpanTwoClips() {
        // 1,303 frames: 13 full chunks and one of 3 (1 row); then a clip of 541 (5 × 13 + 11).
        let layout = EncoderLayout(frames: [1303, 541])
        #expect(layout.clipFeatures == [170, 76])
        #expect(layout.windows == [0 ..< 104, 104 ..< 170, 170 ..< 246])
        #expect(layout.chunkClips.filter { $0 == 0 }.count == 14)
        #expect(layout.chunkStarts.suffix(6) == [0, 100, 200, 300, 400, 500])
    }
}

struct EpochPlanTests {
    @Test func everyEpochVisitsEveryRecordOnce() {
        let plan = EpochPlan(recordCount: 1000, utterancesPerStep: 128, seed: 52)
        for epoch in 0 ..< 3 {
            #expect(plan.order(epoch: epoch).sorted() == Array(0 ..< 1000))
        }
        #expect(plan.order(epoch: 0) != plan.order(epoch: 1))
    }

    @Test func theOrderDependsOnlyOnTheSeedAndEpoch() {
        let a = EpochPlan(recordCount: 500, utterancesPerStep: 128, seed: 7)
        let b = EpochPlan(recordCount: 500, utterancesPerStep: 128, seed: 7)
        #expect(a.order(epoch: 2) == b.order(epoch: 2))
        #expect(a.order(epoch: 0) != EpochPlan(recordCount: 500, utterancesPerStep: 128, seed: 8).order(epoch: 0))
        // Pinned (computed with an independent Python copy of SplitMix64 and Fisher–Yates), so a
        // change to the shuffle, which would break resuming a run, shows up here.
        #expect(Array(a.order(epoch: 0).prefix(5)) == [238, 384, 250, 140, 481])
        #expect(Array(a.order(epoch: 1).prefix(5)) == [215, 262, 87, 138, 340])
    }

    @Test func stepsCoverTheEpochAndTheLastMayBeShort() {
        let plan = EpochPlan(recordCount: 300, utterancesPerStep: 128, seed: 1)
        #expect(plan.stepsPerEpoch == 3)
        let order = plan.order(epoch: 0)
        #expect(plan.records(step: 0, in: order).count == 128)
        #expect(plan.records(step: 2, in: order).count == 44)
        #expect(plan.position(ofStep: 7) == (2, 1))
    }
}

struct MicroBatchSplitTests {
    @Test func batchesStayWithinTheBudgetLongestFirst() {
        let lengths = [10, 50, 20, 40, 30, 5]
        let batches = MicroBatches.split(lengths: lengths, tokenBudget: 100)
        #expect(batches.flatMap { $0 }.sorted() == Array(0 ..< 6))
        for batch in batches {
            let longest = batch.map { lengths[$0] }.max()!
            #expect(batch.count * longest <= 100 || batch.count == 1)
        }
        #expect(batches.first == [1, 3])
    }

    @Test func anExampleOverTheBudgetGoesAlone() {
        #expect(MicroBatches.split(lengths: [300, 10, 10], tokenBudget: 100) == [[0], [1, 2]])
    }
}
