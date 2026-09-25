import Foundation
@preconcurrency import MLX
import Testing
import TrainerKit

/// Two examples of different lengths: [prefix 2][audio][suffix 1][label], the label ending in 9.
private func examples() -> [TokenizedExample] {
    [
        // prompt: 1 2 | 50 50 50 | 3 ; label: 4 5 9
        TokenizedExample(ids: [1, 2, 50, 50, 50, 3, 4, 5, 9], promptLength: 6, audioStart: 2, audioTokens: 3, audioFeatures: 3),
        // prompt: 1 2 | 50 50 50 | 3 ; label: 6 9 — three placeholders, the last left unfilled
        TokenizedExample(ids: [1, 2, 50, 50, 50, 3, 6, 9], promptLength: 6, audioStart: 2, audioTokens: 3, audioFeatures: 2),
    ]
}

struct BatchLayoutTests {
    @Test func rowsArePaddedAfterTheirLastInputToken() {
        let layout = BatchLayout(examples: examples(), padding: 0)
        #expect(layout.rows == 2)
        #expect(layout.length == 8)
        #expect(layout.inputs == [1, 2, 50, 50, 50, 3, 4, 5,
                                  1, 2, 50, 50, 50, 3, 6, 0])
    }

    @Test func audioPositionsPointAtTheirClipsFeaturesInRowOrder() {
        let layout = BatchLayout(examples: examples(), padding: 0)
        // Row 1's third placeholder keeps its embedding, as mergeAudioFeatures leaves it.
        #expect(layout.isAudio == [false, false, true, true, true, false, false, false,
                                   false, false, true, true, false, false, false, false])
        #expect(layout.audioIndex == [0, 0, 0, 1, 2, 0, 0, 0,
                                      0, 0, 3, 4, 0, 0, 0, 0])
        #expect(layout.audioFeatures == 5)
    }

    @Test func onlyLabelTokensAreTargets() {
        let layout = BatchLayout(examples: examples(), padding: 0)
        // Row 0: positions 5, 6, 7 predict 4, 5, 9. Row 1 (offset 8): positions 5, 6 predict 6, 9.
        #expect(layout.targetPositions == [5, 6, 7, 13, 14])
        #expect(layout.targetIDs == [4, 5, 9, 6, 9])
        #expect(layout.targetCount == 5)
        #expect(layout.paddedTokens == 16)
    }
}

struct MergeTests {
    @Test func audioFeaturesReplaceTheAudioPlaceholdersOnly() {
        let layout = BatchLayout(examples: examples(), padding: 0)
        // Token embeddings: each position's value is its token id; audio features are 100 + index.
        let tokens = MLXArray(layout.inputs.map(Float.init), [2, 8]).expandedDimensions(axis: -1)
        let audio = MLXArray((0 ..< 5).map { Float(100 + $0) }, [5, 1])
        let merged = Forward.merge(tokens: tokens, audio: audio, isAudio: MLXArray(layout.isAudio),
                                   audioIndex: MLXArray(layout.audioIndex))
        #expect(merged.shape == [2, 8, 1])
        #expect(merged.reshaped(-1).asArray(Float.self) == [1, 2, 100, 101, 102, 3, 4, 5,
                                                           1, 2, 103, 104, 50, 3, 6, 0])
    }

    @Test func gradientsFlowBackToTheAudioFeatures() {
        let layout = BatchLayout(examples: examples(), padding: 0)
        let tokens = MLXArray.zeros([2, 8, 1])
        let isAudio = MLXArray(layout.isAudio), audioIndex = MLXArray(layout.audioIndex)
        let gradient = grad { (audio: MLXArray) -> MLXArray in
            Forward.merge(tokens: tokens, audio: audio, isAudio: isAudio, audioIndex: audioIndex).sum()
        }(MLXArray.zeros([5, 1]))
        // Every feature is used exactly once.
        #expect(gradient.reshaped(-1).asArray(Float.self) == [1, 1, 1, 1, 1])
    }
}

struct ScheduleTests {
    // Values from Hugging Face's get_linear_schedule_with_warmup, computed in Python for 1,578
    // steps, 2% warm-up (ceil: 32 steps) and a peak of 2e-5.
    @Test func matchesHuggingFacesLinearScheduleWithWarmup() {
        let schedule = Schedule(peak: 2e-5, totalSteps: 1578, warmupFraction: 0.02)
        #expect(schedule.warmupSteps == 32)
        let expected: [(Int, Double)] = [(0, 0), (1, 6.25e-07), (31, 1.9375e-05), (32, 2e-05),
                                         (33, 1.9987063389391982e-05), (1577, 1.29366106080207e-08), (1578, 0)]
        for (step, rate) in expected {
            #expect(abs(schedule.rate(step: step) - rate) < 1e-15, "step \(step)")
        }
    }
}

struct OptimizerTests {
    @Test func adamWMatchesTorch() {
        // torch.optim.AdamW(lr=0.1) on one parameter at 1.0 with gradients 0.5 then -0.25,
        // computed in Python with torch's formulas.
        let optimizer = AdamW()
        var parameters: Tensors = ["p": MLXArray([Float(1)])]
        parameters = optimizer.update(parameters: parameters, gradients: ["p": MLXArray([Float(0.5)])], rate: 0.1)
        optimizer.evaluate(parameters)
        #expect(abs(parameters["p"]!.item(Float.self) - 0.900000002) < 1e-6)
        parameters = optimizer.update(parameters: parameters, gradients: ["p": MLXArray([Float(-0.25)])], rate: 0.1)
        optimizer.evaluate(parameters)
        #expect(abs(parameters["p"]!.item(Float.self) - 0.8733662987078463) < 1e-6)
        #expect(optimizer.steps == 2)
    }

    @Test func parametersWithoutAGradientStayPut() {
        let optimizer = AdamW()
        let parameters = optimizer.update(parameters: ["p": MLXArray([Float(1)]), "q": MLXArray([Float(2)])],
                                          gradients: ["p": MLXArray([Float(1)])], rate: 0.1)
        #expect(parameters["q"]!.item(Float.self) == 2)
    }

    @Test func clippingScalesDownOnlyNormsOverTheLimit() {
        #expect(Gradients.clipScale(norm: 0.5, max: 1) == 1)
        #expect(abs(Gradients.clipScale(norm: 4, max: 1) - 0.25) < 1e-6)
    }

    @Test func theGlobalNormCoversEveryTensor() {
        let norm = Gradients.globalNorm(["a": MLXArray([Float(3)]), "b": MLXArray([Float(4)])]).item(Float.self)
        #expect(abs(norm - 5) < 1e-6)
    }

    @Test func accumulatingAddsKeyByKey() {
        var sum = Tensors()
        Gradients.accumulate(["a": MLXArray([Float(1)])], into: &sum)
        Gradients.accumulate(["a": MLXArray([Float(2)]), "b": MLXArray([Float(5)])], into: &sum)
        #expect(sum["a"]!.item(Float.self) == 3)
        #expect(sum["b"]!.item(Float.self) == 5)
    }
}

struct LossTallyTests {
    @Test func splitsEachRowsLossesIntoTheNewLanguageOrTheReplay() {
        var tally = LossTally()
        // A micro-batch of a Sinhala row with 2 targets and a replay row with 3, then one of a Sinhala row.
        tally.add([1, 3, 0.5, 0.5, 2], rows: [(2, true), (3, false)])
        tally.add([4], rows: [(1, true)])
        #expect(tally.all.sum == 11 && tally.all.targets == 6)
        #expect(tally.newLanguage.mean == 8.0 / 3)
        #expect(tally.replay.mean == 1)
    }

    @Test func aPartWithNoTargetsHasNoMean() {
        var tally = LossTally()
        tally.add([2, 4], rows: [(2, false)])
        #expect(tally.newLanguage.mean == nil)
        #expect(tally.replay.mean == 3)
        #expect(LossTally().all.mean == nil)
    }
}

struct ScoringTests {
    // Expected values from scripts/make_replay.py's normalise, words and characters.
    @Test func normalisesAsTheReplayScriptDoes() {
        #expect(Scoring.words("Straße 1,000 people’s") == ["strasse", "1000", "people's"])
        #expect(String(String.UnicodeScalarView(Scoring.characters("Straße 1,000 people’s"))) == "strasse1000peoples")
        #expect(Scoring.words("He said: \"3,5\" — OK!") == ["he", "said", "35", "ok"])
        #expect(Scoring.words("L’homme (a) b") == ["l'homme", "a", "b"])
    }

    @Test func sinhalaCountsLettersAndSignsButNotJoiners() {
        #expect(Scoring.words("මම meeting එකට යනවා.") == ["මම", "meeting", "එකට", "යනවා"])
        #expect(Scoring.characters("මම meeting එකට යනවා.").count == 16)
        #expect(Scoring.characters("ශ්‍රී ලංකාව").count == 9)
        #expect(Scoring.words("ශ්‍රී ලංකාව") == ["ශ්", "රී", "ලංකාව"])
    }

    @Test func errorRatesMatchThePythonScorer() {
        var tally = ErrorTally()
        tally.add(reference: "මම meeting එකට යනවා", hypothesis: "මම මීටින් එකට යනවා")
        #expect(abs(tally.characterErrorRate - 0.4375) < 1e-12)
        #expect(abs(tally.wordErrorRate - 0.25) < 1e-12)
        #expect(tally.latinInReferences == 1 && tally.latinMatched == 0)
        #expect(Scoring.editDistance(Array("kitten"), Array("sitting")) == 3)
    }

    @Test func englishWordsInEnglishLettersAreCounted() {
        var tally = ErrorTally()
        tally.add(reference: "මම meeting එක cancel කරන්න", hypothesis: "මම meeting එක කැන්සල් කරන්න")
        #expect(tally.latinInReferences == 2)
        #expect(tally.latinMatched == 1)
        #expect(tally.latinRecall == 0.5)
        #expect(tally.latinPrecision == 1)
    }

    @Test func aTranscriptUnderHalfTheReferenceIsCutShort() {
        var tally = ErrorTally()
        tally.add(reference: "el perro corre por el parque", hypothesis: "El.")
        tally.add(reference: "hola", hypothesis: "hola")
        #expect(tally.truncated == 1)
    }
}

struct ExportTests {
    let config = """
    {
        "model_type": "qwen3_asr",
        "support_languages": [
            "Chinese",
            "English"
        ],
        "thinker_config": {"dtype": "bfloat16"}
    }
    """

    /// Weights spread like a trained layer's: a normal spread, a few large ones, some groups all
    /// of one sign.
    private func weights() -> MLXArray {
        MLXRandom.seed(3)
        let normal = MLXRandom.normal([16, 512]) * 0.02
        let spikes = MLXRandom.normal([16, 512]) * 0.2 * (MLXRandom.uniform(Float(0) ..< Float(1), [16, 512]) .> 0.99)
        let oneSign = MLX.abs(MLXRandom.normal([16, 128]) * 0.03)
        return MLX.concatenated([normal + spikes, MLX.concatenated([oneSign, -oneSign, oneSign * 3, -oneSign * 5], axis: 1)], axis: 0)
    }

    /// The weights a packed layer stands for, level × scale + bias in float32, unpacked by hand.
    private func unpacked(_ packed: MLXArray, _ scales: MLXArray, _ biases: MLXArray) -> MLXArray {
        let levels = ((packed.expandedDimensions(axis: -1) >> MLXArray([UInt32(0), 8, 16, 24])) & UInt32(255))
            .reshaped(packed.dim(0), scales.dim(1), 64).asType(.float32)
        return levels * scales.asType(.float32).expandedDimensions(axis: -1) + biases.asType(.float32).expandedDimensions(axis: -1)
    }

    @Test func quantizedWeightsComeBackWithinHalfAStep() {
        let w = weights()
        let (packed, scales, biases) = Export.quantize(w, groupSize: 64, bits: 8, dtype: .bfloat16)
        #expect(packed.dtype == .uint32 && packed.shape == [32, 128])
        #expect(scales.dtype == .bfloat16 && scales.shape == [32, 8] && biases.shape == [32, 8])
        let back = unpacked(packed, scales, biases)
        let step = MLX.abs(scales.asType(.float32)).expandedDimensions(axis: -1)
        let error = MLX.abs(back - w.reshaped(32, 8, 64)) / step
        #expect(error.max().item(Float.self) <= 0.5001)
        // MLX's own dequantize reads the same layout (it rounds to bfloat16 on the way, hence the looser bound).
        let mlx = dequantized(packed, scales: scales, biases: biases, groupSize: 64, bits: 8, dtype: .float32)
        #expect((MLX.abs(mlx.reshaped(32, 8, 64) - back) / step).max().item(Float.self) < 1)
    }

    @Test func itsErrorIsBelowMLXsOwnQuantize() {
        let w = weights()
        let (packed, scales, biases) = Export.quantize(w, groupSize: 64, bits: 8, dtype: .bfloat16)
        let ours = unpacked(packed, scales, biases).reshaped(32, 512)
        let (mlxPacked, mlxScales, mlxBiases) = quantized(w.asType(.bfloat16), groupSize: 64, bits: 8, mode: .affine)
        let theirs = unpacked(mlxPacked, mlxScales, mlxBiases!).reshaped(32, 512)
        #expect(MLX.abs(ours - w).mean().item(Float.self) < MLX.abs(theirs - w).mean().item(Float.self))
        #expect(MLX.abs(ours - w).max().item(Float.self) < MLX.abs(theirs - w).max().item(Float.self))
    }

    @Test func roundingToBFloat16GoesTheWayAsked() {
        let x = MLXArray([Float(1.001), -1.001, 0.3, -0.3, 1.0])
        let up = Export.rounded(x, to: .bfloat16, up: true).asArray(Float.self)
        let down = Export.rounded(x, to: .bfloat16, up: false).asArray(Float.self)
        // bfloat16 steps are 2⁻⁷ = 0.0078125 just above 1: 1.001 lies between 1 and 1.0078125.
        #expect(up == [1.0078125, -1, 0.30078125, -0.29882812, 1])
        #expect(down == [1, -1.0078125, 0.29882812, -0.30078125, 1])
    }

    @Test func addsTheLanguageAndLeavesTheRestAsItWas() throws {
        let updated = try Export.addingLanguage("Sinhala", toConfig: config)
        #expect(updated == config.replacingOccurrences(of: "\"English\"", with: "\"English\",\n        \"Sinhala\""))
    }

    @Test func aLanguageAlreadyListedChangesNothing() throws {
        #expect(try Export.addingLanguage("English", toConfig: config) == config)
    }

    @Test func aOneLineListKeepsItsSeparator() throws {
        let oneLine = #"{"support_languages": ["Chinese", "English"], "x": 1}"#
        #expect(try Export.addingLanguage("Sinhala", toConfig: oneLine)
            == #"{"support_languages": ["Chinese", "English", "Sinhala"], "x": 1}"#)
    }
}
