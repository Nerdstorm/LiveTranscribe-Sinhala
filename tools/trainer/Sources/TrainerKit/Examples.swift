import CryptoKit
import Foundation
@preconcurrency import MLX

/// A record ready for the model: its tokens and its clip's mel frames.
public struct PreparedExample {
    public let record: Record
    public let tokens: TokenizedExample
    /// [frames, 128].
    public let mel: MLXArray
    public let seconds: Double
}

public enum ExamplesError: Error, CustomStringConvertible {
    case audio(path: String, underlying: any Error)

    public var description: String {
        switch self {
        case .audio(let path, let underlying): "can't read \(path): \(underlying)"
        }
    }
}

public enum Examples {
    public static func prepare(_ record: Record, for trainable: TrainableModel) throws -> PreparedExample {
        let samples: MLXArray
        do {
            samples = try AudioFeatures.samples(of: URL(fileURLWithPath: record.audio))
        } catch {
            throw ExamplesError.audio(path: record.audio, underlying: error)
        }
        let features = try AudioFeatures.features(of: samples, model: trainable.model)
        return PreparedExample(
            record: record,
            tokens: trainable.format.example(text: record.text, audioTokens: features.audioTokens,
                                   audioFeatures: features.audioFeatures),
            mel: features.mel,
            seconds: Double(samples.dim(0)) / Double(AudioFeatures.sampleRate)
        )
    }

    /// The micro-batches of a set of prepared examples, each within `tokenBudget` padded tokens.
    public static func microBatches(_ examples: [PreparedExample], tokenBudget: Int, padding: Int32) -> [MicroBatch] {
        groupedMicroBatches(examples, tokenBudget: tokenBudget, padding: padding).map(\.batch)
    }

    /// The micro-batches with the indices of the examples in each, in row order.
    public static func groupedMicroBatches(_ examples: [PreparedExample], tokenBudget: Int,
                                           padding: Int32) -> [(examples: [Int], batch: MicroBatch)] {
        MicroBatches.split(lengths: examples.map { $0.tokens.ids.count - 1 }, tokenBudget: tokenBudget).map { group in
            (group, MicroBatch(examples: group.map { examples[$0].tokens }, clips: group.map { examples[$0].mel }, padding: padding))
        }
    }

    /// A fingerprint of a record list, order included: a run resumes only on the data it started with.
    public static func digest(_ records: [Record]) -> String {
        var hash = SHA256()
        for record in records {
            hash.update(data: Data("\(record.audio)\t\(record.text)\n".utf8))
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// `count` records picked from `records` by a shuffle with `seed`, the same ones every time.
    public static func sample(_ records: [Record], count: Int, seed: UInt64) -> [Record] {
        let order = EpochPlan(recordCount: records.count, utterancesPerStep: 1, seed: seed).order(epoch: 0)
        return order.prefix(count).map { records[$0] }
    }
}
