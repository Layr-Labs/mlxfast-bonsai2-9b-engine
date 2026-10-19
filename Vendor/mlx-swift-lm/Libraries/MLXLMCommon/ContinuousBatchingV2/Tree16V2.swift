// Request-owned device layout. No process-global proposal or target data.
import Foundation
import MLX

public final class CBv2Tree16Layout {
    public let nodes: MLXArray
    public let leafPaths: MLXArray
    public let leafLengths: MLXArray
    public let owners: MLXArray
    public let nodePaths: MLXArray
    public let depths: MLXArray
    public let leafCount: MLXArray
    public let policyPacket: MLXArray?
    /// Published only at the existing acceptance fence, after stop/count truncation.
    public var acceptedRows: MLXArray?
    /// Set at the same fence only when every physical row belongs to one chain.
    public var acceptedChain = false

    public init(arrays: [MLXArray]) {
        precondition(arrays.count == 7 || arrays.count == 9)
        nodes = arrays[0]; leafPaths = arrays[1]; leafLengths = arrays[2]
        owners = arrays[3]; nodePaths = arrays[4]; depths = arrays[5]; leafCount = arrays[6]
        policyPacket = arrays.count == 9
            ? concatenated([arrays[7], arrays[8]], axis: 0).view(dtype: .int32) : nil
    }
    public var drafts: MLXArray { nodes[1..., 0].reshaped([1, 15]) }
    public var evaluationTargets: [MLXArray] {
        [nodes, leafPaths, leafLengths, owners, nodePaths, depths, leafCount]
            + (policyPacket.map { [$0] } ?? [])
    }
    public static func accept(nodes: [Int32], targets: [Int]) -> (tokens: [Int], rows: [Int]) {
        precondition(nodes.count == 80 && targets.count == 16)
        var row = 0
        var rows: [Int] = [], tokens: [Int] = []
        for _ in 0 ..< 16 {
            rows.append(row); tokens.append(targets[row])
            var next: Int?
            let target = targets[row]
            for candidate in 1 ..< 16 {
                let offset = candidate * 5
                let parent = Int(nodes[offset + 1])
                if parent != row { continue }
                let token = Int(nodes[offset])
                if token == target { next = candidate; break }
            }
            guard let next else { break }
            precondition(next > row)
            row = next
        }
        return (tokens, rows)
    }

    // Immutable diagnostic configuration: avoid rebuilding Foundation's entire
    // environment dictionary at every chain/lookup round.
    public static let fastDecision =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE16_FAST_DECISION"] != "0"
    public static let checkDecision =
        ProcessInfo.processInfo.environment["TREE16_TUNE_CHECK"] == "1"
    public static let captureFile = ProcessInfo.processInfo.environment["TREE16_CAPTURE"]
    public static let roundLogFile = ProcessInfo.processInfo.environment["TREE16_ROUND_LOG"]
    public static let adaptive =
        ["1", "adaptive"].contains(ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE16"] ?? "adaptive")
    public static let enabled = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE16"] ?? "adaptive"
        return ["1", "force", "adaptive"].contains(value)
    }()
}

/// Request-local reliability counts. Only already-verified target rows update these;
/// no prompt ids, cross-request state, offline labels or learned weight tensors.
public final class CBv2Tree16Policy {
    public init() {}
    private var counts = Array(repeating: Float(0), count: 10)
    private var hits = Array(repeating: Float(0), count: 10)
    private var easy = 0
    private var skip = 0
    public var table: MLXArray { MLXArray(counts + hits) }
    /// Costs in ms: measured chain period, fixed tree work, GDN work per replay row.
    public var costs: MLXArray { MLXArray([Float(11), Float(0.08), Float(0.03), Float(0.08)]) }
    public func shouldBuildTree() -> Bool {
        if skip > 0 { skip -= 1; return false }
        return true
    }
    public func invalidateFastGate() { easy = 0; skip = 0 }
    public func observe(parents: [Int], drafts: [Int], targets: [Int],
        probabilities: [Float], statistics: [Float]) {
        precondition(parents.count == 16)
        precondition(drafts.count == 15)
        precondition(targets.count == 16)
        precondition(probabilities.count == 16 && statistics.count == 4)
        for row in 1 ..< 16 {
            let p = probabilities[row]
            guard p.isFinite else { continue }
            let bin = max(0, min(9, Int(p * 10)))
            if counts[bin] >= 64 { counts[bin] *= 0.5; hits[bin] *= 0.5 }
            counts[bin] += 1
            if drafts[row - 1] == targets[parents[row]] { hits[bin] += 1 }
        }
        let gain = statistics[1] - statistics[0]
        let penalty = statistics[0] * statistics[2] / 11
        // Two consecutive high-confidence observations permit one plain-chain round,
        // without constructing tree scores or metadata. Probe again immediately after.
        if statistics[0] >= 14 && gain <= penalty * 0.5 { easy += 1 }
        else { easy = 0 }
        if easy >= 2 { skip = 1; easy = 0 }
    }
}
