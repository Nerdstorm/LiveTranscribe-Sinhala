import Foundation
@preconcurrency import MLX
import MLXAudioSTT
import MLXNN

/// The float type the forward and backward passes run in. The weights the optimizer updates stay
/// float32 either way: in bfloat16, most Adam updates at these learning rates would round away.
public enum ComputeType: String, Codable, Sendable, CaseIterable {
    case float32, bfloat16

    public var dtype: DType { self == .float32 ? .float32 : .bfloat16 }
}

/// What one optimizer step did.
public struct StepResult: Sendable {
    /// Mean cross-entropy per target token over the step's examples.
    public let loss: Double
    public let targets: Int
    /// The same over the Sinhala examples, and over the rest (the replay set), when the step has any.
    public let sinhalaLoss: Double?
    public let replayLoss: Double?
    /// Before clipping.
    public let gradientNorm: Float
    public let microBatches: Int
    public let paddedTokens: Int
    public let audioFeatures: Int
}

/// A step's summed token losses: over all its examples, over those in the language being taught,
/// and over the rest (the replay set).
public struct LossTally: Sendable {
    public struct Part: Sendable {
        public private(set) var sum = 0.0
        public private(set) var targets = 0

        /// The mean loss per target token, or nil with no targets.
        public var mean: Double? { targets > 0 ? sum / Double(targets) : nil }

        mutating func add(_ loss: Double, targets count: Int) {
            sum += loss
            targets += count
        }
    }

    public private(set) var all = Part(), newLanguage = Part(), replay = Part()

    public init() {}

    /// Adds a micro-batch's per-token losses, which list each row's targets together, row by row.
    /// `rows` has each row's label length and whether it's in the language being taught.
    public mutating func add(_ tokenLosses: [Float], rows: [(targets: Int, newLanguage: Bool)]) {
        precondition(tokenLosses.count == rows.reduce(0) { $0 + $1.targets }, "one loss for each target token")
        var offset = 0
        for row in rows {
            let loss = tokenLosses[offset ..< offset + row.targets].reduce(0.0) { $0 + Double($1) }
            offset += row.targets
            all.add(loss, targets: row.targets)
            if row.newLanguage {
                newLanguage.add(loss, targets: row.targets)
            } else {
                replay.add(loss, targets: row.targets)
            }
        }
    }
}

/// Full fine-tuning of a Qwen3-ASR model: float32 weights the optimizer updates, a copy in the
/// compute type the model runs, gradients summed over micro-batches into one step.
public final class Trainer {
    /// The language being taught; every other language in the training data is replay.
    public static let newLanguage = "Sinhala"

    public let trainable: TrainableModel
    public let computeType: ComputeType
    /// The trained parameters, float32, keyed as trainableParameters().flattened() keys them.
    public private(set) var weights: Tensors
    public private(set) var optimizer: AdamW
    private let lossAndGradient: (Qwen3ASRModel, (MicroBatch, MLXArray)) -> ([MLXArray], ModuleParameters)

    /// With `trainEncoder` false the audio encoder is frozen: only the decoder (and the embeddings
    /// it shares with the output layer) learns.
    public init(trainable: TrainableModel, computeType: ComputeType, trainEncoder: Bool, optimizer: AdamW = AdamW()) throws {
        self.trainable = trainable
        self.computeType = computeType
        self.optimizer = optimizer
        if !trainEncoder {
            guard let encoder = trainable.model.namedModules().first(where: { $0.0 == "audio_tower" })?.1 else {
                throw ModelError.unexpectedModel("no audio encoder at \"audio_tower\"")
            }
            encoder.freeze()
        }
        weights = Dictionary(uniqueKeysWithValues: trainable.model.trainableParameters().flattened())
            .mapValues { $0.asType(.float32) }
        if computeType != .float32 {
            trainable.model.update(parameters: trainable.model.parameters().mapValues { $0.asType(computeType.dtype) })
        }
        lossAndGradient = valueAndGrad(model: trainable.model) { [trainable] _, arguments in
            Forward.loss(trainable, arguments.0, normalizer: arguments.1)
        }
        eval(Array(weights.values))
    }

    /// Carries on from a saved state: its weights and optimizer.
    public func restore(weights restored: Tensors, optimizer restoredOptimizer: AdamW) throws {
        let expected = Set(weights.keys), got = Set(restored.keys)
        guard expected == got else {
            throw ModelError.unexpectedModel(
                "the saved weights don't match the model: \(expected.subtracting(got).count) missing, \(got.subtracting(expected).count) extra"
            )
        }
        for (key, value) in restored where value.shape != weights[key]!.shape {
            throw ModelError.unexpectedModel("\(key) is \(value.shape) in the saved weights, \(weights[key]!.shape) in the model")
        }
        weights = restored.mapValues { $0.asType(.float32) }
        optimizer = restoredOptimizer
        load()
    }

    /// Puts the trained weights into the model, in the compute type.
    public func load() {
        trainable.model.update(parameters: ModuleParameters.unflattened(weights.mapValues { $0.asType(computeType.dtype) }))
        eval(trainable.model)
    }

    /// Every parameter of the model as it stands, trained or frozen: what a snapshot saves.
    public func allParameters() -> Tensors {
        var all = Dictionary(uniqueKeysWithValues: trainable.model.parameters().flattened())
        for (key, value) in weights { all[key] = value }
        return all
    }

    /// One optimizer step over `examples`: the loss is the mean over all their target tokens,
    /// however they're split into micro-batches of at most `tokenBudget` padded tokens.
    public func step(_ examples: [PreparedExample], rate: Double, clip: Double, tokenBudget: Int) -> StepResult {
        let targets = examples.map(\.tokens.labelLength).reduce(0, +)
        let normalizer = MLXArray(Float(targets))
        let batches = Examples.groupedMicroBatches(examples, tokenBudget: tokenBudget, padding: trainable.format.padding)
        var gradients = Tensors()
        var tally = LossTally()
        for (members, batch) in batches {
            let (values, batchGradients) = lossAndGradient(trainable.model, (batch, normalizer))
            Gradients.accumulate(Dictionary(uniqueKeysWithValues: batchGradients.flattened()), into: &gradients)
            eval([values[1]] + Array(gradients.values))
            tally.add(values[1].asArray(Float.self), rows: members.map {
                (examples[$0].tokens.labelLength, Evaluation.language(of: examples[$0].record.text) == Self.newLanguage)
            })
        }
        let norm = Gradients.globalNorm(gradients).item(Float.self)
        weights = optimizer.update(
            parameters: weights, gradients: gradients, rate: Float(rate),
            gradientScale: Gradients.clipScale(norm: norm, max: Float(clip))
        )
        gradients = [:]
        optimizer.evaluate(weights)
        load()
        return StepResult(
            loss: tally.all.mean ?? 0,
            targets: targets,
            sinhalaLoss: tally.newLanguage.mean,
            replayLoss: tally.replay.mean,
            gradientNorm: norm,
            microBatches: batches.count,
            paddedTokens: batches.map(\.batch.layout.paddedTokens).reduce(0, +),
            audioFeatures: batches.map(\.batch.layout.audioFeatures).reduce(0, +)
        )
    }

    /// The gradients of one micro-batch's summed loss, without updating anything: for checking
    /// that they reach every part of the model.
    public func gradients(_ batch: MicroBatch) -> (loss: Float, gradients: Tensors) {
        let (values, gradients) = lossAndGradient(trainable.model, (batch, MLXArray(Float(1))))
        let flat = Dictionary(uniqueKeysWithValues: gradients.flattened())
        eval([values[0]] + Array(flat.values))
        let loss = values[0].item(Float.self)
        load()
        return (loss, flat)
    }
}

extension StepResult {
    /// "Sinhala 0.4123, replay 0.0412", leaving out a part the step didn't have.
    public var parts: String {
        [sinhalaLoss.map { String(format: "\(Trainer.newLanguage) %.4f", $0) },
         replayLoss.map { String(format: "replay %.4f", $0) }].compactMap { $0 }.joined(separator: ", ")
    }
}
