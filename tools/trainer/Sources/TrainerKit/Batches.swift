/// SplitMix64: a small generator whose sequence depends only on its seed, so a run's data order
/// can be rebuilt exactly when it resumes, on any machine or Swift version.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// A number in 0..<bound, without modulo bias.
    mutating func below(_ bound: Int) -> Int {
        let bound = UInt64(bound)
        let limit = UInt64.max - UInt64.max % bound
        while true {
            let value = next()
            if value < limit { return Int(value % bound) }
        }
    }
}

/// The order training visits the records in: every epoch a fresh shuffle from the run's seed,
/// cut into optimizer steps of `utterancesPerStep` records (the last step of an epoch may be
/// shorter). A run's position is its step count, so a resumed run carries on with the same data.
public struct EpochPlan: Equatable, Sendable {
    public let recordCount: Int
    public let utterancesPerStep: Int
    public let seed: UInt64

    public init(recordCount: Int, utterancesPerStep: Int, seed: UInt64) {
        precondition(recordCount > 0 && utterancesPerStep > 0)
        self.recordCount = recordCount
        self.utterancesPerStep = utterancesPerStep
        self.seed = seed
    }

    public var stepsPerEpoch: Int { (recordCount + utterancesPerStep - 1) / utterancesPerStep }

    /// The records of an epoch, shuffled (Fisher–Yates with SplitMix64 seeded by seed and epoch).
    public func order(epoch: Int) -> [Int] {
        var generator = SplitMix64(seed: seed &+ UInt64(epoch) &* 0x9E37_79B9_7F4A_7C15)
        var order = Array(0 ..< recordCount)
        for index in stride(from: recordCount - 1, to: 0, by: -1) {
            order.swapAt(index, generator.below(index + 1))
        }
        return order
    }

    /// The epoch a global step (counted from 0) falls in, and its step within that epoch.
    public func position(ofStep step: Int) -> (epoch: Int, step: Int) {
        (step / stepsPerEpoch, step % stepsPerEpoch)
    }

    /// The records of one step, from its epoch's order.
    public func records(step: Int, in order: [Int]) -> ArraySlice<Int> {
        let start = step * utterancesPerStep
        return order[start ..< min(start + utterancesPerStep, recordCount)]
    }
}

public enum MicroBatches {
    /// Splits a step's examples into micro-batches of similar length, longest first, each within
    /// `tokenBudget` counted with padding (rows × the longest row). An example longer than the
    /// budget gets a micro-batch of its own. Returns indices into `lengths`.
    public static func split(lengths: [Int], tokenBudget: Int) -> [[Int]] {
        // Longest first; equal lengths in their original order.
        let byLength = lengths.indices.sorted { lengths[$0] != lengths[$1] ? lengths[$0] > lengths[$1] : $0 < $1 }
        var batches: [[Int]] = []
        var current: [Int] = []
        for index in byLength {
            // Sorted longest first, so a batch's first row is its longest.
            let longest = current.first.map { lengths[$0] } ?? lengths[index]
            if !current.isEmpty && (current.count + 1) * longest > tokenBudget {
                batches.append(current)
                current = []
            }
            current.append(index)
        }
        if !current.isEmpty { batches.append(current) }
        return batches
    }
}
