import Foundation
@preconcurrency import MLX
import MLXAudioSTT

/// A run's settings, written to run.json when it starts. Running `train` again on the same folder
/// with the same settings carries on from the newest saved state.
public struct RunSettings: Codable, Equatable, Sendable {
    /// The model folder training starts from.
    public var model: String
    public var train: [String]
    public var dev: String
    public var trainRecords: Int
    /// Examples.digest of the training records, in order.
    public var dataDigest: String
    public var seed: UInt64
    public var epochs: Int
    public var utterancesPerStep: Int
    /// Most padded tokens (rows × longest row) in one forward pass.
    public var tokenBudget: Int
    public var peakRate: Double
    public var warmupFraction: Double
    public var clip: Double
    public var computeType: ComputeType
    public var trainEncoder: Bool
    public var saveEveryMinutes: Double
    public var keepStates: Int
    /// Dev loss on `devLossUtterances` every `devLossEvery` steps.
    public var devLossEvery: Int
    public var devLossUtterances: Int
    /// Snapshots, dev loss and dev transcripts (CER) this many times an epoch, and at the end.
    public var evaluationsPerEpoch: Int
    public var devTranscribeUtterances: Int
    public var cacheLimitMB: Int
    /// Records in a language the model already knows (FLEURS English dev) that each transcribing
    /// evaluation also scores, to watch for forgetting; nil for none.
    public var english: String?

    public init(model: String, train: [String], dev: String, trainRecords: Int, dataDigest: String, seed: UInt64,
                epochs: Int, utterancesPerStep: Int, tokenBudget: Int, peakRate: Double, warmupFraction: Double,
                clip: Double, computeType: ComputeType, trainEncoder: Bool, saveEveryMinutes: Double, keepStates: Int,
                devLossEvery: Int, devLossUtterances: Int, evaluationsPerEpoch: Int, devTranscribeUtterances: Int,
                cacheLimitMB: Int, english: String? = nil) {
        self.model = model
        self.train = train
        self.dev = dev
        self.trainRecords = trainRecords
        self.dataDigest = dataDigest
        self.seed = seed
        self.epochs = epochs
        self.utterancesPerStep = utterancesPerStep
        self.tokenBudget = tokenBudget
        self.peakRate = peakRate
        self.warmupFraction = warmupFraction
        self.clip = clip
        self.computeType = computeType
        self.trainEncoder = trainEncoder
        self.saveEveryMinutes = saveEveryMinutes
        self.keepStates = keepStates
        self.devLossEvery = devLossEvery
        self.devLossUtterances = devLossUtterances
        self.evaluationsPerEpoch = evaluationsPerEpoch
        self.devTranscribeUtterances = devTranscribeUtterances
        self.cacheLimitMB = cacheLimitMB
        self.english = english
    }

    /// Whether a run with these settings trains as one with `other`'s: the same model, data and
    /// order, optimizer, schedule and precision, with dev numbers on the same utterances. The rest
    /// may change between sessions of a run: the token budget only regroups a step's utterances
    /// into forward passes (the step's gradient is the same, rounded differently), and the cache
    /// limit, saving and evaluation settings (the English check included) change speed, memory and
    /// bookkeeping.
    public func trainsLike(_ other: RunSettings) -> Bool {
        var mine = self
        mine.tokenBudget = other.tokenBudget
        mine.cacheLimitMB = other.cacheLimitMB
        mine.saveEveryMinutes = other.saveEveryMinutes
        mine.keepStates = other.keepStates
        mine.devLossEvery = other.devLossEvery
        mine.evaluationsPerEpoch = other.evaluationsPerEpoch
        mine.english = other.english
        return mine == other
    }

    /// Each setting that differs from `other`'s, as "name: other's value → this one's", by run.json's names.
    public func differences(from other: RunSettings) -> [String] {
        func object(_ settings: RunSettings) -> [String: NSObject] {
            guard let data = try? JSONEncoder().encode(settings),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: NSObject] else { return [:] }
            return object
        }
        let mine = object(self), theirs = object(other)
        return Set(mine.keys).union(theirs.keys).sorted().compactMap { key in
            mine[key] == theirs[key] ? nil
                : "\(key): \(theirs[key].map { "\($0)" } ?? "none") → \(mine[key].map { "\($0)" } ?? "none")"
        }
    }

    public func plan() -> EpochPlan {
        EpochPlan(recordCount: trainRecords, utterancesPerStep: utterancesPerStep, seed: seed)
    }

    public func schedule() -> Schedule {
        Schedule(peak: peakRate, totalSteps: plan().stepsPerEpoch * epochs, warmupFraction: warmupFraction)
    }

    /// The step counts (optimizer steps completed) at which a run evaluates.
    public func evaluationSteps() -> Set<Int> {
        let perEpoch = plan().stepsPerEpoch
        var steps = Set<Int>()
        for epoch in 0 ..< epochs {
            for part in 1 ... evaluationsPerEpoch {
                steps.insert(epoch * perEpoch + part * perEpoch / evaluationsPerEpoch)
            }
        }
        return steps
    }
}

struct StepLine: Codable {
    var kind = "step"
    let step: Int
    let epoch: Double
    let loss: Double
    let sinhalaLoss: Double?
    let replayLoss: Double?
    let rate: Double
    let gradientNorm: Float
    let targets: Int
    let utterances: Int
    let microBatches: Int
    let paddedTokens: Int
    let audioFeatures: Int
    let seconds: Double
    let loadSeconds: Double
    let peakGB: Double
    let time: Date
}

struct EvaluationLine: Codable {
    var kind = "evaluation"
    let step: Int
    let epoch: Double
    /// None for the English baseline a new run measures before its first step.
    let devLoss: Double?
    let devTargets: Int?
    let characterErrorRate: Double?
    let wordErrorRate: Double?
    let languageAccuracy: Double?
    let truncated: Int?
    var englishCharacterErrorRate: Double?
    var englishWordErrorRate: Double?
    let snapshot: String?
    let seconds: Double
    let time: Date
}

public final class TrainingRun {
    let settings: RunSettings
    let folder: RunFolder
    let records: [Record]
    let devLossRecords: [Record]
    let devTranscribeRecords: [Record]
    let englishRecords: [Record]
    let trainer: Trainer
    var state: SavedState
    let log: (String) -> Void

    /// Opens the run in `folder`: a new one, or the one already there if it trains the same way
    /// (`RunSettings.trainsLike`), recording any other setting that changed in run.json.
    public init(settings: RunSettings, folder: RunFolder, records: [Record], dev: [Record], english: [Record] = [],
                log: @escaping (String) -> Void) async throws {
        guard Examples.digest(records) == settings.dataDigest, records.count == settings.trainRecords else {
            throw RunError.dataChanged
        }
        if FileManager.default.fileExists(atPath: folder.settings.path) {
            let saved = try JSON.read(RunSettings.self, from: folder.settings)
            guard settings.trainsLike(saved) else { throw RunError.settingsDiffer(saved: saved, given: settings) }
            if settings != saved {
                log("Carrying on with changed settings: " + settings.differences(from: saved).joined(separator: "; "))
                try JSON.write(settings, to: folder.settings)
            }
        } else {
            try FileManager.default.createDirectory(at: folder.url, withIntermediateDirectories: true)
            try JSON.write(settings, to: folder.settings)
        }
        self.settings = settings
        self.folder = folder
        self.records = records
        let devSample = Examples.sample(dev, count: max(settings.devLossUtterances, settings.devTranscribeUtterances),
                                        seed: settings.seed)
        devLossRecords = Array(devSample.prefix(settings.devLossUtterances))
        devTranscribeRecords = Array(devSample.prefix(settings.devTranscribeUtterances))
        englishRecords = english
        self.log = log

        Memory.cacheLimit = settings.cacheLimitMB * 1024 * 1024
        let trainable = try await TrainableModel.load(directory: URL(fileURLWithPath: settings.model))
        trainer = try Trainer(trainable: trainable, computeType: settings.computeType, trainEncoder: settings.trainEncoder)
        if let saved = try Checkpoints.loadLatest(from: folder, optimizer: trainer.optimizer) {
            try trainer.restore(weights: saved.weights, optimizer: saved.optimizer)
            state = saved.state
            log("Carrying on from step \(state.step) (saved \(saved.state.savedAt.formatted()))")
        } else {
            trainer.load()
            state = SavedState(step: 0, adamSteps: 0, savedAt: Date(), trainingSeconds: 0)
            log("Starting at step 0")
        }
    }

    /// Trains until the schedule's last step, `stopAfter` steps in all, or `interrupted()` says to
    /// stop; saves the state before returning.
    public func run(stopAfter: Int? = nil, interrupted: () -> Bool = { false }) throws {
        let plan = settings.plan()
        let schedule = settings.schedule()
        let evaluationSteps = settings.evaluationSteps()
        let last = min(schedule.totalSteps, stopAfter ?? .max)
        log("\(records.count) training records, \(plan.stepsPerEpoch) steps an epoch, \(schedule.totalSteps) steps "
            + "(\(schedule.warmupSteps) warm-up), peak rate \(schedule.peak)")
        var epochOrder: (epoch: Int, order: [Int])?
        var lastSave = Date()
        var sinceSave = 0.0
        let metrics = try MetricsFile(folder.metrics)
        if state.step == 0 && !englishRecords.isEmpty {
            // The untrained model's English, for the evaluations to be compared with.
            let started = Date()
            let english = try transcribeEnglish(into: folder.url.appending(component: "english-step-0"))
            try metrics.append(EvaluationLine(
                step: 0, epoch: 0, devLoss: nil, devTargets: nil, characterErrorRate: nil, wordErrorRate: nil,
                languageAccuracy: nil, truncated: nil, englishCharacterErrorRate: english.tally.characterErrorRate,
                englishWordErrorRate: english.tally.wordErrorRate, snapshot: nil,
                seconds: Date().timeIntervalSince(started), time: Date()
            ))
            log("step 0: English " + english.summary + String(format: " (%.0f s)", Date().timeIntervalSince(started)))
        }

        while state.step < last && !interrupted() {
            let started = Date()
            let (epoch, stepInEpoch) = plan.position(ofStep: state.step)
            if epochOrder?.epoch != epoch { epochOrder = (epoch, plan.order(epoch: epoch)) }
            let chosen = plan.records(step: stepInEpoch, in: epochOrder!.order).map { records[$0] }
            let examples = try chosen.map { try Examples.prepare($0, for: trainer.trainable) }
            let loaded = Date()

            let rate = schedule.rate(step: state.step)
            Memory.peakMemory = 0
            let result = trainer.step(examples, rate: rate, clip: settings.clip, tokenBudget: settings.tokenBudget)
            let seconds = Date().timeIntervalSince(started)
            state.step += 1
            state.adamSteps = trainer.optimizer.steps
            state.trainingSeconds += seconds
            sinceSave += seconds

            let epochs = Double(state.step) / Double(plan.stepsPerEpoch)
            let peak = Double(Memory.peakMemory) / 1e9
            try metrics.append(StepLine(
                step: state.step, epoch: epochs, loss: result.loss, sinhalaLoss: result.sinhalaLoss,
                replayLoss: result.replayLoss, rate: rate, gradientNorm: result.gradientNorm,
                targets: result.targets, utterances: examples.count, microBatches: result.microBatches,
                paddedTokens: result.paddedTokens, audioFeatures: result.audioFeatures, seconds: seconds,
                loadSeconds: loaded.timeIntervalSince(started), peakGB: peak, time: Date()
            ))
            log(String(format: "step %d/%d (epoch %.3f): loss %.4f (%@), rate %.2e, grad norm %.3f; %.1f s (%.1f s loading), %.1f utt/s, peak %.1f GB",
                       state.step, schedule.totalSteps, epochs, result.loss, result.parts, rate, result.gradientNorm,
                       seconds, loaded.timeIntervalSince(started), Double(examples.count) / seconds, peak))

            if evaluationSteps.contains(state.step) || state.step == schedule.totalSteps {
                try evaluate(transcribing: true, metrics: metrics)
            } else if state.step % settings.devLossEvery == 0 {
                try evaluate(transcribing: false, metrics: metrics)
            }
            if Date().timeIntervalSince(lastSave) >= settings.saveEveryMinutes * 60 {
                try save()
                lastSave = Date()
                sinceSave = 0
            }
        }
        if sinceSave > 0 || interrupted() { try save() }
        log(state.step >= schedule.totalSteps ? "Finished all \(state.step) steps" : "Stopped after step \(state.step)")
    }

    func save() throws {
        let started = Date()
        state.savedAt = Date()
        try Checkpoints.save(weights: trainer.weights, optimizer: trainer.optimizer, state: state, in: folder,
                             keep: settings.keepStates)
        log(String(format: "Saved step %d in %.0f s", state.step, Date().timeIntervalSince(started)))
    }

    /// Dev loss, and with `transcribing` a bfloat16 snapshot and dev transcripts scored for CER.
    func evaluate(transcribing: Bool, metrics: MetricsFile) throws {
        let started = Date()
        let epochs = Double(state.step) / Double(settings.plan().stepsPerEpoch)
        let (loss, targets) = try Evaluation.loss(trainer.trainable, records: devLossRecords, tokenBudget: settings.tokenBudget)
        var report: TranscriptionReport?
        var english: TranscriptionReport?
        var snapshotPath: String?
        if transcribing {
            let snapshot = folder.snapshot(step: state.step)
            try Checkpoints.saveSnapshot(parameters: trainer.allParameters(), to: snapshot)
            snapshotPath = snapshot.path
            let transcripts = try Evaluation.transcribe(trainer.trainable.model, records: devTranscribeRecords)
            report = Evaluation.report(transcripts)
            try Transcripts.write(transcripts, to: snapshot.appending(component: "dev.tsv"))
            try JSON.write(report, to: snapshot.appending(component: "eval.json"))
            if !englishRecords.isEmpty { english = try transcribeEnglish(into: snapshot) }
        }
        let seconds = Date().timeIntervalSince(started)
        try metrics.append(EvaluationLine(
            step: state.step, epoch: epochs, devLoss: loss, devTargets: targets,
            characterErrorRate: report?.tally.characterErrorRate, wordErrorRate: report?.tally.wordErrorRate,
            languageAccuracy: report?.languageAccuracy, truncated: report?.tally.truncated,
            englishCharacterErrorRate: english?.tally.characterErrorRate, englishWordErrorRate: english?.tally.wordErrorRate,
            snapshot: snapshotPath, seconds: seconds, time: Date()
        ))
        log(String(format: "step %d: dev loss %.4f over %d tokens", state.step, loss, targets)
            + (report.map { "; dev " + $0.summary } ?? "") + (english.map { "; English " + $0.summary } ?? "")
            + String(format: " (%.0f s)", seconds))
        Memory.clearCache()
    }

    /// Transcribes the English records with the model as it is now, as the app would (no language
    /// given), and writes english.tsv and english.json into `folder`.
    func transcribeEnglish(into folder: URL) throws -> TranscriptionReport {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let transcripts = try Evaluation.transcribe(trainer.trainable.model, records: englishRecords)
        let report = Evaluation.report(transcripts)
        try Transcripts.write(transcripts, to: folder.appending(component: "english.tsv"))
        try JSON.write(report, to: folder.appending(component: "english.json"))
        return report
    }
}

public enum RunError: Error, CustomStringConvertible {
    case dataChanged
    case settingsDiffer(saved: RunSettings, given: RunSettings)

    public var description: String {
        switch self {
        case .dataChanged:
            "the training records aren't the ones in run.json; start a new run folder"
        case .settingsDiffer(let saved, let given):
            "the run folder's run.json trains another way; pass its options, or use a new folder: "
                + given.differences(from: saved).joined(separator: "; ")
        }
    }
}

/// metrics.jsonl, appended to a line at a time.
final class MetricsFile {
    let handle: FileHandle

    init(_ url: URL) throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    func append(_ line: some Encodable) throws {
        try handle.write(contentsOf: Data((try JSON.line(line) + "\n").utf8))
    }

    deinit { try? handle.close() }
}

public enum Transcripts {
    /// A tab-separated file: audio, seconds, milliseconds, expected and given language, reference, hypothesis.
    public static func write(_ transcripts: [Transcript], to url: URL) throws {
        func clean(_ text: String) -> String {
            text.replacingOccurrences(of: "\t", with: " ").replacingOccurrences(of: "\n", with: " ")
        }
        var lines = ["audio\tseconds\tms\texpected_language\tlanguage\treference\thypothesis"]
        for t in transcripts {
            lines.append([
                t.audio, String(format: "%.2f", t.seconds), String(format: "%.0f", t.milliseconds),
                t.expectedLanguage ?? "", t.language ?? "", clean(t.reference), clean(t.hypothesis),
            ].joined(separator: "\t"))
        }
        try (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    }
}
