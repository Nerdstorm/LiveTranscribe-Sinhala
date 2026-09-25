import Foundation
@preconcurrency import MLX

/// Where a run keeps its files. Everything lives in one folder (out/<run>, gitignored):
///
///     run.json                   the run's settings, fixed when it starts
///     state-<step>/              the latest resumable states (weights, optimizer, state.json)
///     snapshots/step-<step>/     bfloat16 weights at each evaluation, with eval.json
///     metrics.jsonl              one line per optimizer step and per evaluation
public struct RunFolder: Sendable {
    public let url: URL

    public init(_ url: URL) { self.url = url }

    public var settings: URL { url.appending(component: "run.json") }
    public var metrics: URL { url.appending(component: "metrics.jsonl") }
    public var snapshots: URL { url.appending(component: "snapshots", directoryHint: .isDirectory) }

    public func state(step: Int) -> URL { url.appending(component: "state-\(step)", directoryHint: .isDirectory) }
    public func snapshot(step: Int) -> URL { snapshots.appending(component: "step-\(step)", directoryHint: .isDirectory) }

    /// The complete saved states, oldest first. A state is complete once its state.json is written,
    /// which happens last, inside a folder renamed into place.
    public func savedStates() throws -> [(step: Int, url: URL)] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            .compactMap { folder -> (Int, URL)? in
                let name = folder.lastPathComponent
                guard name.hasPrefix("state-"), let step = Int(name.dropFirst("state-".count)),
                      FileManager.default.fileExists(atPath: folder.appending(component: "state.json").path)
                else { return nil }
                return (step, folder)
            }
            .sorted { $0.0 < $1.0 }
    }
}

/// What a saved state holds besides the tensors.
public struct SavedState: Codable, Equatable, Sendable {
    /// Optimizer steps completed; the run carries on with step `step`.
    public var step: Int
    public var adamSteps: Int
    public var savedAt: Date
    /// Seconds spent training so far, across restarts.
    public var trainingSeconds: Double
}

public enum Checkpoints {
    static let weightsFile = "weights.safetensors"
    static let optimizerFile = "optimizer.safetensors"
    static let stateFile = "state.json"

    /// Saves the trained weights and the optimizer's state as state-<step>, then removes all but
    /// the newest `keep` states. The folder is written under a temporary name and renamed, so a
    /// state that exists is complete.
    public static func save(weights: Tensors, optimizer: AdamW, state: SavedState, in run: RunFolder, keep: Int) throws {
        let final = run.state(step: state.step)
        let partial = run.url.appending(component: ".state-\(state.step).partial", directoryHint: .isDirectory)
        let manager = FileManager.default
        if manager.fileExists(atPath: partial.path) { try manager.removeItem(at: partial) }
        try manager.createDirectory(at: partial, withIntermediateDirectories: true)

        try MLX.save(arrays: weights, url: partial.appending(component: weightsFile))
        var moments = Tensors(minimumCapacity: optimizer.firstMoments.count * 2)
        for (key, value) in optimizer.firstMoments { moments["m." + key] = value }
        for (key, value) in optimizer.secondMoments { moments["v." + key] = value }
        try MLX.save(arrays: moments, url: partial.appending(component: optimizerFile))
        try JSON.write(state, to: partial.appending(component: stateFile))

        if manager.fileExists(atPath: final.path) { try manager.removeItem(at: final) }
        try manager.moveItem(at: partial, to: final)

        // The run's own older states: each is ~9 GB, and only the newest few are ever resumed from.
        for old in try run.savedStates().dropLast(keep) {
            try manager.removeItem(at: old.url)
        }
    }

    /// The newest saved state: its weights, a restored optimizer and its record.
    public static func loadLatest(from run: RunFolder, optimizer template: AdamW) throws -> (weights: Tensors, optimizer: AdamW, state: SavedState)? {
        guard let latest = try run.savedStates().last else { return nil }
        let state = try JSON.read(SavedState.self, from: latest.url.appending(component: stateFile))
        let weights = try loadArrays(url: latest.url.appending(component: weightsFile))
        let moments = try loadArrays(url: latest.url.appending(component: optimizerFile))
        var first = Tensors(), second = Tensors()
        for (key, value) in moments {
            if key.hasPrefix("m.") { first[String(key.dropFirst(2))] = value }
            if key.hasPrefix("v.") { second[String(key.dropFirst(2))] = value }
        }
        let optimizer = AdamW(
            beta1: template.beta1, beta2: template.beta2, epsilon: template.epsilon,
            weightDecay: template.weightDecay, steps: state.adamSteps, firstMoments: first, secondMoments: second
        )
        return (weights, optimizer, state)
    }

    /// Saves every parameter of the model in bfloat16: what an evaluation measured, and what
    /// `export` turns into a model folder.
    public static func saveSnapshot(parameters: Tensors, to folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try MLX.save(arrays: parameters.mapValues { $0.asType(.bfloat16) }, url: folder.appending(component: weightsFile))
    }

    public static func weights(in folder: URL) -> URL { folder.appending(component: weightsFile) }
}

enum JSON {
    static func write(_ value: some Encodable, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(value).write(to: url, options: .atomic)
    }

    static func read<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(type, from: Data(contentsOf: url))
    }

    /// One compact line, for appending to a .jsonl file.
    static func line(_ value: some Encodable) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}
