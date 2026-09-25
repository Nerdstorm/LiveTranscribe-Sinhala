import Foundation
@preconcurrency import MLX

/// Flat parameter or gradient sets, keyed as Module.parameters().flattened() keys them
/// ("model.layers.0.mlp.up_proj.weight").
public typealias Tensors = [String: MLXArray]

/// The learning rate over a run: Hugging Face's get_linear_schedule_with_warmup, which Qwen's
/// fine-tuning script uses. It rises linearly from 0 over the warm-up steps, then falls linearly
/// to 0 at the last step. Steps count from 0, and step 0's rate is 0, as it is there.
public struct Schedule: Codable, Equatable, Sendable {
    public let peak: Double
    public let totalSteps: Int
    public let warmupSteps: Int

    /// The warm-up is `warmupFraction` of the run, rounded up, as Hugging Face's warmup_ratio is.
    public init(peak: Double, totalSteps: Int, warmupFraction: Double) {
        precondition(totalSteps > 0 && peak > 0 && (0 ..< 1).contains(warmupFraction))
        self.peak = peak
        self.totalSteps = totalSteps
        self.warmupSteps = Int((Double(totalSteps) * warmupFraction).rounded(.up))
    }

    public func rate(step: Int) -> Double {
        if step < warmupSteps {
            return peak * Double(step) / Double(max(1, warmupSteps))
        }
        return peak * max(0, Double(totalSteps - step) / Double(max(1, totalSteps - warmupSteps)))
    }
}

public enum Gradients {
    /// Adds `gradients` to `sum` (float32), key by key.
    public static func accumulate(_ gradients: Tensors, into sum: inout Tensors) {
        for (key, gradient) in gradients {
            let gradient = gradient.asType(.float32)
            sum[key] = sum[key].map { $0 + gradient } ?? gradient
        }
    }

    /// The L2 norm of all the gradients together.
    public static func globalNorm(_ gradients: Tensors) -> MLXArray {
        let squares = gradients.keys.sorted().map { key in
            let gradient = gradients[key]!.asType(.float32)
            return (gradient * gradient).sum()
        }
        return MLX.sqrt(MLX.stacked(squares).sum())
    }

    /// What torch.nn.utils.clip_grad_norm_ multiplies the gradients by for a norm of `norm`:
    /// max / (norm + 1e-6), but never more than 1.
    public static func clipScale(norm: Float, max maximum: Float) -> Float {
        min(1, maximum / (norm + 1e-6))
    }
}

/// AdamW as torch.optim.AdamW computes it, with its state kept here so a checkpoint can save and
/// restore it exactly (mlx-swift's optimizers keep theirs private).
public final class AdamW {
    public let beta1: Float
    public let beta2: Float
    public let epsilon: Float
    public let weightDecay: Float
    /// Updates applied so far: the t of the bias corrections.
    public private(set) var steps: Int
    public private(set) var firstMoments: Tensors
    public private(set) var secondMoments: Tensors

    public init(beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8, weightDecay: Float = 0,
                steps: Int = 0, firstMoments: Tensors = [:], secondMoments: Tensors = [:]) {
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weightDecay = weightDecay
        self.steps = steps
        self.firstMoments = firstMoments
        self.secondMoments = secondMoments
    }

    /// The parameters after one update with learning rate `rate`, the gradients multiplied by
    /// `gradientScale` first (the clip). The moments are updated here too; everything stays lazy,
    /// and evaluating the result a group of keys at a time lets each old buffer be reused.
    public func update(parameters: Tensors, gradients: Tensors, rate: Float, gradientScale: Float = 1) -> Tensors {
        steps += 1
        let t = Float(steps)
        let correction1 = 1 - pow(beta1, t)
        let correction2 = 1 - pow(beta2, t)
        let stepSize = rate / correction1
        let root2 = correction2.squareRoot()

        var updated = Tensors(minimumCapacity: parameters.count)
        for (key, parameter) in parameters {
            guard var gradient = gradients[key] else {
                updated[key] = parameter
                continue
            }
            gradient = gradient.asType(.float32)
            if gradientScale != 1 { gradient = gradient * gradientScale }
            let m = firstMoments[key].map { beta1 * $0 + (1 - beta1) * gradient } ?? (1 - beta1) * gradient
            let v = secondMoments[key].map { beta2 * $0 + (1 - beta2) * gradient.square() }
                ?? (1 - beta2) * gradient.square()
            firstMoments[key] = m
            secondMoments[key] = v
            var next = parameter
            if weightDecay != 0 { next = next * (1 - rate * weightDecay) }
            updated[key] = next - stepSize * m / (MLX.sqrt(v) / root2 + epsilon)
        }
        return updated
    }

    /// Evaluates the parameters and moments a few keys at a time, so the whole set is never held
    /// twice.
    public func evaluate(_ parameters: Tensors, groupSize: Int = 32) {
        let keys = parameters.keys.sorted()
        for start in stride(from: 0, to: keys.count, by: groupSize) {
            var arrays: [MLXArray] = []
            for key in keys[start ..< min(start + groupSize, keys.count)] {
                arrays.append(parameters[key]!)
                if let m = firstMoments[key] { arrays.append(m) }
                if let v = secondMoments[key] { arrays.append(v) }
            }
            eval(arrays)
        }
    }
}
