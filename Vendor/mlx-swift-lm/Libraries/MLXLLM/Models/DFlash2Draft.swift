// Copyright © 2026 Apple Inc.

// DFlash 2 block drafter.
//
// This is a port of the reference MLX Python implementation, `dflash/model_mlx.py`
// of `z-lab/dflash` at `07ebd93`: `DFlashAttention`, `GroupedDynamicCausalConv`,
// `DFlash2DecoderLayer`, `CandidateSelector`, `DFlash2DraftModel`.
//
// The drafter is a BLOCK drafter, not a chain drafter. One `propose` call
// consumes a block of `depth + 1` input tokens — the last committed token
// followed by `depth` mask tokens — and returns `depth` draft tokens. The block
// attends to itself without a causal mask and to a rotating window of context
// keys and values projected from the target's own hidden states.
//
// The drafter owns NO embedding table and NO output projection. It binds the
// target's. On this track both are packed Hadamard modules, so this file calls
// the modules and never reads a raw `.weight`. See
// `docs/bonsai2-27b-port-notes.md` section 6.1 for the defect that rule exists
// to prevent.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public enum DFlash2LayerType: String, Codable, Sendable, Equatable {
    case fullAttention = "full_attention"
    case slidingAttention = "sliding_attention"
}

public enum DFlash2Error: LocalizedError, Sendable, Equatable {
    case missingSlidingWindow
    case invalidCacheCount(expected: Int, actual: Int)
    case invalidBlockSize(Int)
    case targetHiddenSizeMismatch(expected: Int, actual: Int)
    case notBound
    case vocabularyMismatch(drafter: Int, target: Int)
    case hiddenSizeMismatch(drafter: Int, target: Int)
    case targetLayerOutOfRange(layerId: Int, layerCount: Int)
    case emptyTapLayerIds
    case duplicateTapLayerIds([Int])
    case missingConfig(String)
    case unreadableDirectory(String)
    case emptyBlockContext

    public var errorDescription: String? {
        switch self {
        case .missingSlidingWindow:
            return "A DFlash 2 sliding_attention layer needs sliding_window."
        case .invalidCacheCount(let expected, let actual):
            return "The DFlash 2 drafter needs \(expected) caches; it got \(actual)."
        case .invalidBlockSize(let blockSize):
            return "The DFlash 2 block size \(blockSize) is less than 2."
        case .targetHiddenSizeMismatch(let expected, let actual):
            return
                "The fused target hidden state is \(actual) wide; the drafter needs \(expected)."
        case .notBound:
            return "The DFlash 2 drafter is not bound to a target model."
        case .vocabularyMismatch(let drafter, let target):
            return
                "The DFlash 2 drafter has vocabulary \(drafter); the target has \(target)."
        case .hiddenSizeMismatch(let drafter, let target):
            return
                "The DFlash 2 drafter has hidden size \(drafter); the target has \(target)."
        case .targetLayerOutOfRange(let layerId, let layerCount):
            return
                "The DFlash 2 target layer id \(layerId) is outside 0..<\(layerCount)."
        case .emptyTapLayerIds:
            return "The DFlash 2 tap needs at least one target layer id."
        case .duplicateTapLayerIds(let ids):
            return "The DFlash 2 target layer ids must be unique; they are \(ids)."
        case .missingConfig(let path):
            return "The DFlash 2 drafter directory has no config.json: \(path)."
        case .unreadableDirectory(let path):
            return "The DFlash 2 drafter directory cannot be read: \(path)."
        case .emptyBlockContext:
            return "The DFlash 2 drafter has no target context rows to propose from."
        }
    }
}

/// The DFlash 2 drafter configuration, decoded from the drafter's own
/// `config.json`.
///
/// The reference reads the block fields out of the `dflash_config` object and
/// the geometry out of the top level. `rope_theta` is NOT at the top level of a
/// DFlash 2 config: it is inside `rope_parameters`, and the reference falls back
/// to that object. This decoder does the same.
public struct DFlash2Configuration: Decodable, Sendable, Equatable {
    /// The `dflash_config` object.
    public struct Block: Codable, Sendable, Equatable {
        public var blockSize: Int
        public var maskTokenId: Int
        public var targetLayerIds: [Int]
        public var convKernelSize: Int
        public var convGroupSize: Int
        public var selectorRank: Int
        public var selectorTopK: Int
        public var inputEmbeddingScale: Float
        public var outputMultiplier: Float
        public var finalLogitSoftcapping: Float?

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case maskTokenId = "mask_token_id"
            case targetLayerIds = "target_layer_ids"
            case convKernelSize = "conv_kernel_size"
            case convGroupSize = "conv_group_size"
            case selectorRank = "selector_rank"
            case selectorTopK = "selector_top_k"
            case inputEmbeddingScale = "input_embedding_scale"
            case outputMultiplier = "output_multiplier"
            case finalLogitSoftcapping = "final_logit_softcapping"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 16
            self.maskTokenId = try container.decode(Int.self, forKey: .maskTokenId)
            self.targetLayerIds = try container.decode([Int].self, forKey: .targetLayerIds)
            self.convKernelSize =
                try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 0
            self.convGroupSize =
                try container.decodeIfPresent(Int.self, forKey: .convGroupSize) ?? 0
            self.selectorRank = try container.decodeIfPresent(Int.self, forKey: .selectorRank) ?? 0
            self.selectorTopK = try container.decodeIfPresent(Int.self, forKey: .selectorTopK) ?? 0
            self.inputEmbeddingScale =
                try container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale) ?? 1
            self.outputMultiplier =
                try container.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1
            self.finalLogitSoftcapping =
                try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        }
    }

    /// The `rope_parameters` object. A DFlash 2 config carries the rope base
    /// here rather than at the top level.
    public struct RopeParameters: Codable, Sendable, Equatable {
        public var ropeTheta: Float?
        public var ropeType: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }
    }

    public var architectures: [String]
    public var modelType: String
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var vocabularySize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    /// The rope scaling the drafter was trained with, handed to
    /// `initializeRope`: the `rope_parameters` object when its `rope_type`
    /// names a scaled rotation (for example `yarn`). Nil for the plain rotation.
    public var ropeScaling: [String: StringOrNumber]?
    public var maxPositionEmbeddings: Int
    public var numTargetLayers: Int
    public var layerTypes: [DFlash2LayerType]
    public var slidingWindow: Int?
    public var isCausal: Bool
    public var dflash: Block

    public var blockSize: Int { dflash.blockSize }
    public var maskTokenId: Int { dflash.maskTokenId }
    public var targetLayerIds: [Int] { dflash.targetLayerIds }
    /// The width of the fused target hidden state the drafter's `fc` consumes.
    public var targetHiddenSize: Int { dflash.targetLayerIds.count * hiddenSize }
    /// The number of draft tokens one block of `blockSize` inputs produces.
    public var draftDepth: Int { blockSize - 1 }

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numTargetLayers = "num_target_layers"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case isCausal = "is_causal"
        case dflash = "dflash_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.dflash = try container.decode(Block.self, forKey: .dflash)
        self.numTargetLayers =
            try container.decodeIfPresent(Int.self, forKey: .numTargetLayers) ?? hiddenLayers
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.isCausal = try container.decodeIfPresent(Bool.self, forKey: .isCausal) ?? false

        let ropeKey: CodingKeys? =
            container.contains(.ropeParameters)
            ? .ropeParameters : (container.contains(.ropeScaling) ? .ropeScaling : nil)
        let rope = try ropeKey.map { try container.decode(RopeParameters.self, forKey: $0) }
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? rope?.ropeTheta ?? 10000
        if let ropeKey, let ropeType = rope?.ropeType, ropeType != "default" {
            self.ropeScaling = try container.decode(
                [String: StringOrNumber].self, forKey: ropeKey)
        } else {
            self.ropeScaling = nil
        }

        let layerTypes =
            try container.decodeIfPresent([DFlash2LayerType].self, forKey: .layerTypes)
            ?? Array(repeating: .fullAttention, count: hiddenLayers)
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription:
                    "layer_types has \(layerTypes.count) entries; num_hidden_layers is \(hiddenLayers)."
            )
        }
        if layerTypes.contains(.slidingAttention), slidingWindow == nil {
            throw DecodingError.dataCorruptedError(
                forKey: .slidingWindow, in: container,
                debugDescription:
                    "A config with sliding_attention layers must define sliding_window.")
        }
        self.layerTypes = layerTypes

        guard dflash.blockSize >= 2 else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.block_size must be at least 2.")
        }
        guard dflash.maskTokenId >= 0, dflash.maskTokenId < vocabularySize else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription:
                    "dflash_config.mask_token_id \(dflash.maskTokenId) is outside 0..<\(vocabularySize)."
            )
        }
        guard !dflash.targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.target_layer_ids must not be empty.")
        }
        guard Set(dflash.targetLayerIds).count == dflash.targetLayerIds.count else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.target_layer_ids must be unique.")
        }
        for layerId in dflash.targetLayerIds
        where layerId < 0 || layerId >= numTargetLayers {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription:
                    "dflash_config target layer id \(layerId) is outside 0..<\(numTargetLayers).")
        }
        guard dflash.selectorTopK >= 1, dflash.selectorTopK <= vocabularySize else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription:
                    "dflash_config.selector_top_k \(dflash.selectorTopK) is outside 1...\(vocabularySize)."
            )
        }
        guard dflash.convKernelSize >= 1 else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.conv_kernel_size must be at least 1.")
        }
        guard dflash.convGroupSize >= 1, hiddenSize % dflash.convGroupSize == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription:
                    "dflash_config.conv_group_size \(dflash.convGroupSize) must divide hidden_size \(hiddenSize)."
            )
        }
    }
}

// MARK: - The target binding

/// The target-model surface the DFlash 2 drafter binds to.
///
/// The drafter owns no embedding table and no output projection. Both of these
/// MUST call the target's MODULE. On a packed Hadamard pack a path that reads
/// the raw `.weight` skips the transform and returns wrong numbers while it
/// still type-checks.
public protocol DFlash2Target: AnyObject {
    var dFlash2VocabularySize: Int { get }
    var dFlash2HiddenSize: Int { get }

    /// The target's embedding table, applied through the target's module.
    func embedTokensForDFlash2(_ tokens: MLXArray) -> MLXArray
    /// The target's output projection, applied through the target's module. The
    /// drafter applies any config-level logit transform itself.
    func logitsForDFlash2Hidden(_ hidden: MLXArray) -> MLXArray
    /// The same output projection over the vocabulary prefix `0 ..< rows` only:
    /// the target head's first `rows` output rows, through the same module type
    /// and transform. Proposals only; the target's verify head stays full.
    func draftLogitsForDFlash2Hidden(_ hidden: MLXArray, rows: Int) -> MLXArray
    /// The fast selectors' `[15, 128]` ids and values from the same prefix head
    /// without its full logits: every value is the head's own at that id, the
    /// ids may differ from the logits' top-k (proposals only). Nil keeps the
    /// logits.
    func draftShortlistForDFlash2Hidden(_ hidden: MLXArray, rows: Int) -> (ids: MLXArray, values: MLXArray)?
    /// Build, at load, the per-process state the target's `rows`-row verify head
    /// would otherwise build inside the first measured window. Never changes a token.
    func prepareDFlash2Verify(rows: Int)
}

extension DFlash2Target {
    public func prepareDFlash2Verify(rows: Int) {}
    public func draftLogitsForDFlash2Hidden(_ hidden: MLXArray, rows: Int) -> MLXArray {
        logitsForDFlash2Hidden(hidden)
    }
    public func draftShortlistForDFlash2Hidden(
        _ hidden: MLXArray, rows: Int
    ) -> (ids: MLXArray, values: MLXArray)? { nil }
}

/// Model-derived draft-head read budgets, independent of prompts or reference outputs.
/// A request may grow from one sixteenth to one eighth to half to full vocabulary as trusted anchors
/// leave its current prefix. An explicit DF2_VOCAB override supplies one tier;
/// zero or an unsupported size keeps the full head.
enum DF2Vocab {
    static let overrideRows = Int(ProcessInfo.processInfo.environment["DF2_VOCAB"] ?? "")

    static func widths(vocabularySize: Int, topK: Int) -> [Int] {
        let requested = overrideRows.map { [$0] } ?? [vocabularySize / 16, vocabularySize / 8, vocabularySize / 2]
        // The fast selector scans eight blocks of 256 threads. Keep all blocks
        // populated, in addition to satisfying the configured candidate count.
        let minimumRows = Swift.max(topK, 8 * 256)
        return Array(Set(requested.map { ($0 / 64) * 64 }.filter {
            $0 >= minimumRows && $0 < vocabularySize
        })).sorted()
    }
}

/// The target head's prefix rows as one module, kept OFF the module tree (see
/// `DFlash2TapSlot`). Its weight, scales and biases are row slices of the full
/// head's arrays: views of the same buffers, no copy. Built once per head
/// instance, at bind (load time), and kept only after a bitwise self-check.
public final class DFlash2DraftHeadSlot {
    private weak var source: HadamardQuantizedLinear?
    private var checkedRows: Set<Int> = []
    private var prefixes: [Int: HadamardQuantizedLinear] = [:]
    private var screenedRows: Set<Int> = []

    public init() {}

    /// The prefix head's screened candidates (`HadamardQuantizedLinear.draftScreen`);
    /// nil unless the screen qualified at bind.
    public func shortlist(
        _ hidden: MLXArray, of full: HadamardQuantizedLinear, rows: Int
    ) -> (ids: MLXArray, values: MLXArray)? {
        guard let prefix = head(of: full, rows: rows), screenedRows.contains(rows) else { return nil }
        return prefix.draftScreen(hidden, shortlist: DF2Fast.shortlist)
    }

    public func head(of full: HadamardQuantizedLinear, rows: Int) -> HadamardQuantizedLinear? {
        if source !== full {
            source = full
            checkedRows.removeAll()
            prefixes.removeAll()
            screenedRows.removeAll()
        }
        if checkedRows.contains(rows) { return prefixes[rows] }
        // At most three policy tiers; retain failures so requests do not repeat checks.
        guard checkedRows.count < 3 else { return nil }
        checkedRows.insert(rows)
        guard full.bias == nil, let biases = full.biases, rows % 64 == 0,
            rows > 0, rows < full.weight.dim(0)
        else {
            Self.report(rows, "off (head packing or row count does not qualify)")
            return nil
        }
        guard
            let candidate = try? HadamardQuantizedLinear(
                weight: full.weight[0 ..< rows], scales: full.scales[0 ..< rows],
                biases: biases[0 ..< rows], groupSize: full.groupSize, bits: full.bits,
                transform: full.transform, gdnLayout: full.gdnLayout)
        else {
            Self.report(rows, "off (prefix module rejected)")
            return nil
        }
        if let mismatch = Self.selfCheck(prefix: candidate, full: full, rows: rows) {
            Self.report(rows, "off (\(mismatch); full head kept)")
            return nil
        }
        Self.report(rows, "on (bitwise equal to the full head's first \(rows) columns, 15 rows, FP32 and BF16)")
        if DF2Fast.screenAllowed, rows >= 8 * 256 {
            let failure: String? = candidate.qualifyDraftScreen(shortlist: DF2Fast.shortlist)
            if failure == nil { screenedRows.insert(rows) }
            let state: String = failure.map { "off (\($0))" }
                ?? "on (bit-identical to its reference; rescore bit-identical to the head)"
            Self.report(rows, "K 2048 screen \(state)")
        }
        prefixes[rows] = candidate
        return candidate
    }

    /// The prefix head against the full head's first `rows` output columns on a
    /// fixed activation shaped like a proposal's (15 rows, a strided view of a
    /// 16-row block), in FP32 (the fast drafter's hidden) and BF16 (the plain
    /// drafter's). Nil when every bit matches, else the first difference found.
    static func selfCheck(
        prefix: HadamardQuantizedLinear, full: HadamardQuantizedLinear, rows: Int
    ) -> String? {
        let width = full.transform.width
        let values = (0 ..< 16 * width).map { i -> Float in
            let t = Double(i)
            return Float(Foundation.sin(t * 0.7548776) * (1 + 0.5 * Foundation.cos(t * 0.0137)))
        }
        let block = MLXArray(values, [1, 16, width])
        for dtype in [DType.float32, .bfloat16] {
            let x = block.asType(dtype)[0..., 1..., 0...]
            let p = prefix(x)
            let f = full(x)[0..., 0..., 0 ..< rows]
            guard p.shape == f.shape, p.dtype == f.dtype else {
                return "\(dtype) shape or dtype differs"
            }
            let pf = contiguous(p.asType(.float32)).view(dtype: .uint32)
            let ff = contiguous(f.asType(.float32)).view(dtype: .uint32)
            let perRow = (pf .!= ff).asType(.int32).reshaped([-1, rows]).sum(axis: -1)
            let counts = perRow.asArray(Int32.self)
            let bad = counts.indices.filter { counts[$0] != 0 }
            if !bad.isEmpty {
                return "\(dtype): \(counts.reduce(0) { $0 + Int($1) }) values differ in rows "
                    + bad.map(String.init).joined(separator: ",")
            }
        }
        return nil
    }

    private static func report(_ rows: Int, _ state: String) {
        FileHandle.standardError.write(Data("DF2 vocab prefix \(rows): \(state)\n".utf8))
    }
}

/// Where a tower keeps its tap state, deliberately OFF the module tree.
///
/// `Module` reflection classifies a stored property BY ITS DYNAMIC TYPE
/// (`Vendor/mlx-swift/Source/MLXNN/Module.swift`, `ModuleItem.build`):
///
/// ```swift
/// case let v as MLXArray:
///     return .value(.parameters(v))
/// ```
///
/// So a tapped hidden state stored directly on a `Module` becomes a PARAMETER
/// the moment it is non-nil. It would then join `parameters()`, and with it
/// strict checkpoint key verification, the quantization walk, and `eval(model)`
/// -- and on this track the loaded target is re-verified before every measured
/// window, so an unexpected parameter key is a refusal at measurement time. A
/// plain class is `.other`, which reflection ignores, so the tap lives here.
public final class DFlash2TapSlot {
    /// The layers whose OUTPUT hidden state the next forward keeps, in the
    /// order the drafter's `fc` expects them. Nil turns the tap off.
    public var layerIds: [Int]?

    /// The tapped layers of the last forward, fused along the feature axis.
    public var tappedHidden: MLXArray?

    public init() {}
}

/// A target that can hand out the OUTPUT hidden state of named layers.
///
/// The drafter never calls this. The ENGINE does: it asks the target for the
/// tapped layers of the positions the target has just consumed, and feeds them
/// to the drafter as the next block's context.
///
/// COST WHEN OFF. `dFlash2TapLayerIds` is nil unless a DFlash 2 drafter is
/// attached, and a nil id list means the tower does nothing beyond one
/// comparison per layer and retains nothing.
public protocol DFlash2TapTarget: DFlash2Target {
    /// The layers whose OUTPUT hidden state the next forward must keep, in the
    /// order the drafter's `fc` expects them. Nil turns the tap off.
    var dFlash2TapLayerIds: [Int]? { get set }

    /// The tapped layers of the LAST forward, fused along the feature axis:
    /// `[B, L, taps * hidden]`. Nil when the tap was off for that forward.
    var dFlash2TappedHidden: MLXArray? { get }

    /// How many layers the tower has, so the ids can be checked before a run.
    var dFlash2LayerCount: Int { get }
}

extension DFlash2TapTarget {
    /// Check the ids against this target and turn the tap on.
    ///
    /// A bad id is a refusal, not a clamp: a drafter reading the wrong layers
    /// would still produce tokens, and the only visible symptom would be a poor
    /// accept rate that looks like a property of the pairing.
    public func armDFlash2Tap(layerIds: [Int]) throws {
        guard !layerIds.isEmpty else {
            throw DFlash2Error.emptyTapLayerIds
        }
        guard Set(layerIds).count == layerIds.count else {
            throw DFlash2Error.duplicateTapLayerIds(layerIds)
        }
        let count = dFlash2LayerCount
        for layerId in layerIds where layerId < 0 || layerId >= count {
            throw DFlash2Error.targetLayerOutOfRange(layerId: layerId, layerCount: count)
        }
        dFlash2TapLayerIds = layerIds
    }
}

// MARK: - The sliding-window mask

/// The block attention mask the reference builds when a layer slides.
///
/// A DFlash 2 block is NOT causal inside itself: every block position sees
/// every other block position. Only the context half is windowed.
///
/// Reference (`DFlashAttention.__call__`):
/// ```python
/// query = ctx_len + mx.arange(L)[:, None]
/// key = mx.arange(ctx_len + L)[None]
/// context = (key < ctx_len) & (query - key < self.sliding_window)
/// block = key >= ctx_len
/// if self.is_causal:
///     block = block & (key <= query)
/// mask = context | block
/// ```
public enum DFlash2SlidingMask {
    /// `true` where attention is allowed. Shape `[blockLength, contextLength + blockLength]`.
    public static func make(
        contextLength: Int,
        blockLength: Int,
        slidingWindow: Int,
        isCausal: Bool
    ) -> MLXArray {
        let query = MLXArray(Int32(contextLength) ..< Int32(contextLength + blockLength))
            .reshaped(blockLength, 1)
        let key = MLXArray(Int32(0) ..< Int32(contextLength + blockLength))
            .reshaped(1, contextLength + blockLength)
        let context = (key .< Int32(contextLength)) .&& ((query - key) .< Int32(slidingWindow))
        var block = key .>= Int32(contextLength)
        if isCausal {
            block = block .&& (key .<= query)
        }
        return context .|| block
    }

    /// How many leading context rows the layer drops before it projects them,
    /// and the offset the cache advances by to stay aligned with the rows it
    /// keeps. `sliding_window - 1` rows survive.
    public static func contextSkip(contextLength: Int, slidingWindow: Int) -> Int {
        Swift.max(0, contextLength - (slidingWindow - 1))
    }
}

// MARK: - Attention

private final class DFlash2Attention: Module {
    let layerType: DFlash2LayerType
    let slidingWindow: Int?
    let isCausal: Bool
    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlash2Configuration, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self.slidingWindow = layerType == .slidingAttention ? config.slidingWindow : nil
        self.isCausal = config.isCausal
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// - Parameters:
    ///   - x: the block, `[B, blockLength, hidden]`.
    ///   - context: the projected target hidden state, `[B, contextLength, hidden]`.
    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache,
        conv0: (n: MLXArray, dyn: MLXArray, base: MLXArray)? = nil,
        ropeSpec: DF2Fast.RopeSpec? = nil,
        contextCodes: DF2Fast.W4Codes? = nil,
        precomputedContextKV: (MLXArray, MLXArray)? = nil,
        precomputedQKV: (MLXArray, MLXArray, MLXArray)? = nil,
        submitProjections: Bool = false
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        var context = context
        var contextLength = context.dim(1)
        var activeContextCodes = contextCodes
        var activePrecomputedKV = precomputedContextKV

        if let slidingWindow {
            let skip = DFlash2SlidingMask.contextSkip(
                contextLength: contextLength, slidingWindow: slidingWindow)
            if skip > 0 {
                context = context[0..., skip..., 0...]
                contextLength = context.dim(1)
                activeContextCodes = nil
                activePrecomputedKV = nil
                // The dropped rows still happened, so the cache's notion of
                // where the block sits has to move with them.
                if let base = cache as? BaseKVCache {
                    base.offset += skip
                }
            }
        }

        var queries: MLXArray
        var contextKeys: MLXArray? = nil, contextValues: MLXArray? = nil
        var blockKeys: MLXArray, blockValues: MLXArray
        if let precomputedQKV {
            (queries, blockKeys, blockValues) = precomputedQKV
        } else if let conv0,
            let qkv = DF2Fast.linearsConv0(
                conv0.n, dyn: conv0.dyn, base: conv0.base, [qProj, kProj, vProj])
        {
            (queries, blockKeys, blockValues) = (qkv[0], qkv[1], qkv[2])
            // The block's projections need nothing built after them: start
            // them on the GPU while the host builds the attention.
            if submitProjections { asyncEval(queries, blockKeys, blockValues) }
        } else if let qkv = DF2Fast.linears(x, [qProj, kProj, vProj]) {
            (queries, blockKeys, blockValues) = (qkv[0], qkv[1], qkv[2])
        } else {
            (queries, blockKeys, blockValues) = (qProj(x), kProj(x), vProj(x))
        }
        // A block with no new context rows (its context was absorbed ahead of
        // it) attends to the cache as it stands; nothing is projected for it.
        if contextLength > 0 {
            if let activePrecomputedKV {
                (contextKeys, contextValues) = (activePrecomputedKV.0, activePrecomputedKV.1)
            } else if let activeContextCodes,
                let kv = DF2Fast.w4LinearsFromCodes(
                    activeContextCodes, leadShape: [B, contextLength], [kProj, vProj])
            {
                (contextKeys, contextValues) = (kv[0], kv[1])
            } else if let kv = DF2Fast.linears(context, [kProj, vProj]) {
                (contextKeys, contextValues) = (kv[0], kv[1])
            } else if let kv = DF2Fast.fusedKVLinear(context, kProj: kProj, vProj: vProj) {
                (contextKeys, contextValues) = (kv.0, kv.1)
            } else {
                (contextKeys, contextValues) = (kProj(context), vProj(context))
            }
        }

        // The block sits immediately after the context, so both the queries and
        // the block's own keys rotate at the context's far end.
        let blockOffset = cache.offset + contextLength
        var combinedKV: (k: MLXArray, v: MLXArray)? = nil
        if let ropeSpec, B == 1, heads == 32, kvHeads == 8,
            contextLength > 0, contextLength <= 16,
            let fastCache = cache as? DFlash2LayerKVCache,
            let allOut = DF2Fast.qkvAllNormRope(
                q: queries, k: blockKeys, v: blockValues,
                ck: contextKeys!, cv: contextValues!,
                qNorm: qNorm, kNorm: kNorm, ropeSpec: ropeSpec,
                L: L, C: contextLength, blockOffset: blockOffset, ctxOffset: fastCache.offset)
        {
            queries = allOut.0
            combinedKV = (allOut.1, allOut.2)
        } else if let ropeSpec, B == 1, heads == 32, kvHeads == 8,
            let qkvOut = DF2Fast.qkvNormRope(
                q: queries, k: blockKeys, v: blockValues,
                qNorm: qNorm, kNorm: kNorm, ropeSpec: ropeSpec,
                L: L, offset: blockOffset)
        {
            (queries, blockKeys, blockValues) = (qkvOut.0, qkvOut.1, qkvOut.2)
            if contextLength > 0 {
                if contextLength <= 16,
                    let ctxOut = DF2Fast.ctxKVNormRope(
                        k: contextKeys!, v: contextValues!,
                        kNorm: kNorm, ropeSpec: ropeSpec,
                        C: contextLength, offset: cache.offset)
                {
                    (contextKeys, contextValues) = (ctxOut.0, ctxOut.1)
                } else {
                    contextKeys = DF2Fast.rms(
                        contextKeys!.reshaped(B, contextLength, kvHeads, -1), kNorm
                    )
                    .transposed(0, 2, 1, 3)
                    contextValues = contextValues!.reshaped(B, contextLength, kvHeads, -1)
                        .transposed(0, 2, 1, 3)
                    contextKeys = rope(contextKeys!, offset: cache.offset)
                }
            }
        } else {
            queries = DF2Fast.rms(queries.reshaped(B, L, heads, -1), qNorm).transposed(0, 2, 1, 3)
            if contextLength > 0 {
                contextKeys = DF2Fast.rms(
                    contextKeys!.reshaped(B, contextLength, kvHeads, -1), kNorm
                )
                .transposed(0, 2, 1, 3)
                contextValues = contextValues!.reshaped(B, contextLength, kvHeads, -1)
                    .transposed(0, 2, 1, 3)
            }
            blockKeys = DF2Fast.rms(blockKeys.reshaped(B, L, kvHeads, -1), kNorm)
                .transposed(0, 2, 1, 3)
            blockValues = blockValues.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            queries = rope(queries, offset: blockOffset)
            if contextLength > 0 { contextKeys = rope(contextKeys!, offset: cache.offset) }
            blockKeys = rope(blockKeys, offset: blockOffset)
        }

        // Only the CONTEXT keys and values enter the cache. The block's own keys
        // and values are concatenated for this forward and then dropped.
        let keys: MLXArray
        let values: MLXArray
        let cachedLength: Int
        if let fastCache = cache as? DFlash2LayerKVCache {
            if let combinedKV {
                let updated = fastCache.updateWithCombined(
                    combinedKeys: combinedKV.k, combinedValues: combinedKV.v,
                    C: contextLength, L: L)
                keys = updated.keys
                values = updated.values
                cachedLength = updated.cachedLength
            } else {
                let updated = fastCache.updateWithBlock(
                    contextKeys: contextKeys, contextValues: contextValues,
                    blockKeys: blockKeys, blockValues: blockValues)
                keys = updated.keys
                values = updated.values
                cachedLength = updated.cachedLength
            }
        } else {
            let (cachedKeys, cachedValues) =
                contextLength > 0
                ? cache.update(keys: contextKeys!, values: contextValues!)
                : Self.cachedKeysValues(cache)
            cachedLength = cachedKeys.dim(2)
            keys = concatenated([cachedKeys, blockKeys], axis: 2)
            values = concatenated([cachedValues, blockValues], axis: 2)
        }

        var mask: MLXArray?
        if let slidingWindow, DF2Fast.on, !isCausal, cachedLength + L <= slidingWindow {
            // Every block row sees every cached and block key: the window mask is all true.
            mask = nil
        } else if let slidingWindow {
            mask = DFlash2SlidingMask.make(
                contextLength: cachedLength,
                blockLength: L,
                slidingWindow: slidingWindow,
                isCausal: isCausal)
        } else if isCausal {
            mask = createCausalMask(n: L, offset: cachedLength)
        }

        let output: MLXArray
        if mask == nil,
            let specialized = DraftSDPA.attention(queries, keys, values, scale: scale)
        {
            output = specialized
        } else {
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        // MLX's full-attention path (L > 8) writes the output token-major, so the
        // [1, L, 4096] view is already row-contiguous: read it in place rather than
        // have the head-major quantizer's input check copy it. Same values, same
        // column order, same bits. The vector path (L <= 8) writes head-major.
        if B == 1, heads == 32, L > 8, DF2Fast.tokenMajorOProj {
            return DF2Fast.linear(output.transposed(0, 2, 1, 3).reshaped(B, L, -1), oProj)
        }
        if B == 1, heads == 32, let y = DF2Fast.linearHeadMajor(output, oProj, L: L) {
            return y
        }
        return DF2Fast.linear(output.transposed(0, 2, 1, 3).reshaped(B, L, -1), oProj)
    }

    /// The CONTEXT half of `callAsFunction`, with no block: the same sliding
    /// skip, the same k/v projections, norms, rotation at the cache's offset
    /// and `cache.update`, in the same order on the same operands. A later
    /// forward over the remaining rows continues the cache exactly where a
    /// single combined pass would have put it, which is also how every
    /// steady-state round already extends this cache.
    /// The cache's keys and values exactly as `update` returns them, with no
    /// rows added: both cache kinds' `state` is that slice.
    private static func cachedKeysValues(_ cache: KVCache) -> (MLXArray, MLXArray) {
        let state = cache.state
        precondition(state.count == 2, "DFlash2: a block without context rows needs an absorbed cache")
        return (state[0], state[1])
    }

    func absorbContext(
        _ context: MLXArray, rope: RoPELayer, cache: KVCache,
        contextCodes: DF2Fast.W4Codes? = nil, ropeSpec: DF2Fast.RopeSpec? = nil
    ) {
        var context = context
        var contextLength = context.dim(1)
        var activeContextCodes = contextCodes
        if let slidingWindow {
            let skip = DFlash2SlidingMask.contextSkip(
                contextLength: contextLength, slidingWindow: slidingWindow)
            if skip > 0 {
                context = context[0..., skip..., 0...]
                contextLength = context.dim(1)
                activeContextCodes = nil
                if let base = cache as? BaseKVCache {
                    base.offset += skip
                }
            }
        }
        let B = context.dim(0)
        var contextKeys: MLXArray, contextValues: MLXArray
        if let activeContextCodes,
            let kv = DF2Fast.w4LinearsFromCodes(
                activeContextCodes, leadShape: [B, contextLength], [kProj, vProj])
        {
            (contextKeys, contextValues) = (kv[0], kv[1])
        } else if let kv = DF2Fast.linears(context, [kProj, vProj]) {
            (contextKeys, contextValues) = (kv[0], kv[1])
        } else if let kv = DF2Fast.fusedKVLinear(context, kProj: kProj, vProj: vProj) {
            (contextKeys, contextValues) = (kv.0, kv.1)
        } else {
            (contextKeys, contextValues) = (kProj(context), vProj(context))
        }
        if let ropeSpec, B == 1, heads == 32, kvHeads == 8,
            (1 ... 16).contains(contextLength), contextKeys.dtype == .float32,
            contextValues.dtype == .float32,
            let ctxOut = DF2Fast.ctxKVNormRope(
                k: contextKeys, v: contextValues, kNorm: kNorm, ropeSpec: ropeSpec,
                C: contextLength, offset: cache.offset)
        {
            (contextKeys, contextValues) = (ctxOut.0, ctxOut.1)
        } else {
            contextKeys = DF2Fast.rms(contextKeys.reshaped(B, contextLength, kvHeads, -1), kNorm)
                .transposed(0, 2, 1, 3)
            contextValues = contextValues.reshaped(B, contextLength, kvHeads, -1)
                .transposed(0, 2, 1, 3)
            contextKeys = rope(contextKeys, offset: cache.offset)
        }
        if let fastCache = cache as? DFlash2LayerKVCache {
            fastCache.absorb(contextKeys: contextKeys, contextValues: contextValues)
        } else {
            _ = cache.update(keys: contextKeys, values: contextValues)
        }
    }
}

// MARK: - Fast in-place KV cache for DFlash 2 layers

private final class DFlash2LayerKVCache: KVCacheSimple {
    let maxWindowSize: Int?
    var kBuf: MLXArray?
    var vBuf: MLXArray?
    var start: Int = 0
    var end: Int = 0

    init(maxWindowSize: Int?) {
        self.maxWindowSize = maxWindowSize
        super.init()
    }

    override func innerState() -> [MLXArray] {
        [kBuf, vBuf].compactMap { $0 }
    }

    override var state: [MLXArray] {
        get {
            guard let kBuf, let vBuf, end > start else { return [] }
            return [kBuf[.ellipsis, start ..< end, 0...], vBuf[.ellipsis, start ..< end, 0...]]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("DFlash2LayerKVCache state must have 2 arrays")
            }
            kBuf = newValue[0]
            vBuf = newValue[1]
            start = 0
            end = newValue[0].dim(2)
            offset = end
        }
    }

    override var metaState: [String] {
        get { ["\(offset)", "\(start)", "\(end)"] }
        set {
            if newValue.count > 0, let o = Int(newValue[0]) { offset = o }
            if newValue.count > 1, let s = Int(newValue[1]) { start = s }
            if newValue.count > 2, let e = Int(newValue[2]) { end = e }
        }
    }

    @discardableResult
    override func trim(_ n: Int) -> Int {
        let count = end - start
        let t = Swift.min(count, Swift.min(offset, n))
        offset -= t
        end -= t
        return t
    }

    override func copy() -> any KVCache {
        let c = DFlash2LayerKVCache(maxWindowSize: maxWindowSize)
        c.offset = offset
        c.start = start
        c.end = end
        if let kBuf, let vBuf {
            c.kBuf = kBuf[.ellipsis]
            c.vBuf = vBuf[.ellipsis]
        }
        return c
    }

    private func ensureCapacity(
        extra: Int, B: Int, kvHeads: Int, kDim: Int, vDim: Int, kDType: DType, vDType: DType
    ) {
        if let currentKeys = kBuf, let currentValues = vBuf,
            end + extra <= currentKeys.dim(2),
            currentKeys.dtype == kDType, currentValues.dtype == vDType
        {
            return
        }
        let valid = end - start
        let needed = valid + extra
        let cap = ((needed + 512 + 255) / 256) * 256
        let newK = MLXArray.zeros([B, kvHeads, cap, kDim], dtype: kDType)
        let newV = MLXArray.zeros([B, kvHeads, cap, vDim], dtype: vDType)
        if let currentKeys = kBuf, let currentValues = vBuf, valid > 0 {
            newK[.ellipsis, ..<valid, 0...] =
                currentKeys[.ellipsis, start ..< end, 0...].asType(kDType)
            newV[.ellipsis, ..<valid, 0...] =
                currentValues[.ellipsis, start ..< end, 0...].asType(vDType)
        }
        kBuf = newK
        vBuf = newV
        start = 0
        end = valid
    }

    func absorb(contextKeys: MLXArray, contextValues: MLXArray) {
        let C = contextKeys.dim(2)
        guard C > 0 else { return }
        let B = contextKeys.dim(0)
        let kvHeads = contextKeys.dim(1)
        let kDim = contextKeys.dim(3)
        let vDim = contextValues.dim(3)

        if let maxW = maxWindowSize, end > start {
            let valid = end - start
            let trimSize = valid - maxW + 1
            if trimSize > 0 {
                start = Swift.min(end, start + trimSize)
            }
        }

        ensureCapacity(
            extra: C + 16, B: B, kvHeads: kvHeads, kDim: kDim, vDim: vDim,
            kDType: contextKeys.dtype, vDType: contextValues.dtype)
        kBuf![.ellipsis, end ..< (end + C), 0...] = contextKeys
        vBuf![.ellipsis, end ..< (end + C), 0...] = contextValues
        end += C
        offset += C
    }

    func updateWithBlock(
        contextKeys: MLXArray?, contextValues: MLXArray?,
        blockKeys: MLXArray, blockValues: MLXArray
    ) -> (keys: MLXArray, values: MLXArray, cachedLength: Int) {
        let C = contextKeys?.dim(2) ?? 0
        let L = blockKeys.dim(2)
        let B = blockKeys.dim(0)
        let kvHeads = blockKeys.dim(1)
        let kDim = blockKeys.dim(3)
        let vDim = blockValues.dim(3)

        if C > 0, let maxW = maxWindowSize, end > start {
            let valid = end - start
            let trimSize = valid - maxW + 1
            if trimSize > 0 {
                start = Swift.min(end, start + trimSize)
            }
        }

        ensureCapacity(
            extra: C + L, B: B, kvHeads: kvHeads, kDim: kDim, vDim: vDim,
            kDType: blockKeys.dtype, vDType: blockValues.dtype)
        if C > 0, let contextKeys, let contextValues {
            kBuf![.ellipsis, end ..< (end + C), 0...] = contextKeys
            vBuf![.ellipsis, end ..< (end + C), 0...] = contextValues
        }
        kBuf![.ellipsis, (end + C) ..< (end + C + L), 0...] = blockKeys
        vBuf![.ellipsis, (end + C) ..< (end + C + L), 0...] = blockValues
        end += C
        offset += C
        let cachedLength = end - start
        return (
            kBuf![.ellipsis, start ..< (end + L), 0...],
            vBuf![.ellipsis, start ..< (end + L), 0...],
            cachedLength
        )
    }

    func updateWithCombined(
        combinedKeys: MLXArray, combinedValues: MLXArray, C: Int, L: Int
    ) -> (keys: MLXArray, values: MLXArray, cachedLength: Int) {
        let B = combinedKeys.dim(0)
        let kvHeads = combinedKeys.dim(1)
        let kDim = combinedKeys.dim(3)
        let vDim = combinedValues.dim(3)

        if C > 0, let maxW = maxWindowSize, end > start {
            let valid = end - start
            let trimSize = valid - maxW + 1
            if trimSize > 0 {
                start = Swift.min(end, start + trimSize)
            }
        }

        ensureCapacity(
            extra: C + L, B: B, kvHeads: kvHeads, kDim: kDim, vDim: vDim,
            kDType: combinedKeys.dtype, vDType: combinedValues.dtype)
        kBuf![.ellipsis, end ..< (end + C + L), 0...] = combinedKeys
        vBuf![.ellipsis, end ..< (end + C + L), 0...] = combinedValues
        end += C
        offset += C
        let cachedLength = end - start
        return (
            kBuf![.ellipsis, start ..< (end + L), 0...],
            vBuf![.ellipsis, start ..< (end + L), 0...],
            cachedLength
        )
    }
}

// MARK: - The grouped dynamic causal convolution

/// A two-tap causal convolution over the block whose per-position, per-group
/// taps are predicted from the block itself.
///
/// Reference (`_grouped_dynamic_convolve`):
/// ```python
/// output[b, l, g, c] = sum_o (base[o][g * group + c] + dynamic[b, l, o, g])
///                            * hidden[b, l - o, g, c]
/// ```
/// with `hidden[b, l - o] = 0` when `l - o < 0`. The convolution carries no
/// state between blocks: a block is a fresh sequence, and context reaches the
/// block through attention only.
final class DFlash2GroupedDynamicCausalConv: Module {
    let kernelSize: Int
    let groupSize: Int
    let groups: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        self.groups = hiddenSize / groupSize
        // Tap 0 wraps the sub-layer input, tap 1 wraps its output.
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    static func convolve(
        hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let hiddenSize = hidden.dim(2)
        let groups = hiddenSize / groupSize
        let kernelSize = base.dim(0)
        let blocks = hidden.reshaped(batch, length, groups, groupSize)
        let dynamic = dynamic.reshaped(batch, length, kernelSize, groups, 1)

        var output = MLXArray.zeros(like: blocks)
        for offset in 0 ..< kernelSize {
            let values =
                offset == 0
                ? blocks
                : concatenated(
                    [
                        MLXArray.zeros(
                            [batch, offset, groups, groupSize], dtype: hidden.dtype),
                        blocks[0..., ..<(length - offset), 0..., 0...],
                    ], axis: 1)
            let kernel = base[offset].reshaped(1, 1, groups, groupSize).asType(hidden.dtype)
            output = output + kernel * values
            output = output + dynamic[0..., 0..., offset, 0..., 0...] * values
        }
        return output.reshaped(hidden.shape)
    }

    /// The first tap. Returns the convolved input and the dynamic taps the
    /// matching ``finish(_:dynamic:)`` needs.
    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let dynamic = kernelProjection(hidden)
            .reshaped(hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (
            Self.convolve(
                hidden: hidden,
                dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0],
                groupSize: groupSize),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    /// The second tap, over the sub-layer's output.
    var fastBase: MLXArray {
        if let b = DF2Fast.baseCache[ObjectIdentifier(self)] { return b }
        let b = contiguous(baseKernel.asType(.float32).reshaped([-1]))
        eval(b)
        DF2Fast.baseCache[ObjectIdentifier(self)] = b
        return b
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        Self.convolve(
            hidden: hidden, dynamic: dynamic, base: baseKernel[1], groupSize: groupSize)
    }
}

// MARK: - The decoder layer

private final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, conv0: (n: MLXArray, dyn: MLXArray, base: MLXArray)? = nil
    ) -> MLXArray {
        if let conv0,
            let gu = DF2Fast.linearsConv0(conv0.n, dyn: conv0.dyn, base: conv0.base, [gate, up]),
            let y = DF2Fast.linears(gu[0], [down], up: gu[1])
        {
            return y[0]
        }
        if let gu = DF2Fast.linears(x, [gate, up]), let y = DF2Fast.linears(gu[0], [down], up: gu[1]) {
            return y[0]
        }
        return down(silu(gate(x)) * up(x))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, conv0: nil)
    }
}

private final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlash2Attention
    @ModuleInfo var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2GroupedDynamicCausalConv

    init(_ config: DFlash2Configuration, layerIndex: Int) {
        _selfAttn.wrappedValue = DFlash2Attention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = DFlash2MLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionConv.wrappedValue = DFlash2GroupedDynamicCausalConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        _mlpConv.wrappedValue = DFlash2GroupedDynamicCausalConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        super.init()
    }

    func forwardFast(
        _ x: MLXArray, normedInput: MLXArray?, normedCodes: DF2Fast.W4Codes?,
        nextNorm: RMSNorm?, nextQStart: Int = 0,
        context: MLXArray, rope: RoPELayer, cache: KVCache,
        ropeSpec: DF2Fast.RopeSpec?, contextCodes: DF2Fast.W4Codes?,
        precomputedContextKV: (MLXArray, MLXArray)? = nil,
        submitAttention: Bool = false,
        front: DFlash2LayerZeroFront? = nil
    ) -> (h: MLXArray, nextNormed: MLXArray?, nextCodes: DF2Fast.W4Codes?) {
        let n1 = normedInput ?? DF2Fast.rms(x, inputLayerNorm)
        let leadShape = Array(n1.shape.dropLast())
        let p1: MLXArray
        if let front {
            p1 = front.p1
        } else {
            p1 =
                (normedCodes.flatMap {
                    DF2Fast.w4LinearsFromCodes($0, leadShape: leadShape, [attentionConv.kernelProjection])
                }?[0]) ?? DF2Fast.linear(n1, attentionConv.kernelProjection)
        }
        let base1 = attentionConv.fastBase
        let attnOut: MLXArray
        if DF2Fast.w4Group != nil {
            let frontQKV: (MLXArray, MLXArray, MLXArray)? = front.map { ($0.queries, $0.keys, $0.values) }
            attnOut = selfAttn(
                n1, context: context, rope: rope, cache: cache,
                conv0: (n: n1, dyn: p1, base: base1),
                ropeSpec: ropeSpec, contextCodes: contextCodes,
                precomputedContextKV: precomputedContextKV,
                precomputedQKV: frontQKV,
                submitProjections: submitAttention && DF2Fast.earlyFront)
        } else {
            let a = DF2Fast.dynConv(n1, dyn: p1, base: base1, part: 0, residual: nil)
            attnOut = selfAttn(
                a, context: context, rope: rope, cache: cache,
                ropeSpec: ropeSpec, contextCodes: contextCodes,
                precomputedContextKV: precomputedContextKV)
        }
        // Hand the attention half to the GPU while the rest is built.
        if submitAttention { asyncEval(attnOut) }
        let attended: MLXArray
        let n2: MLXArray
        let n2Codes: DF2Fast.W4Codes?
        if let fused = DF2Fast.dynConvRms(
            attnOut, dyn: p1, base: base1, residual: x, norm: postAttentionLayerNorm, qStart: 0)
        {
            (attended, n2, n2Codes) = fused
        } else {
            attended = DF2Fast.dynConv(attnOut, dyn: p1, base: base1, part: 1, residual: x)
            n2 = DF2Fast.rms(attended, postAttentionLayerNorm)
            n2Codes = nil
        }
        let p2 =
            (n2Codes.flatMap {
                DF2Fast.w4LinearsFromCodes($0, leadShape: leadShape, [mlpConv.kernelProjection])
            }?[0]) ?? DF2Fast.linear(n2, mlpConv.kernelProjection)
        let base2 = mlpConv.fastBase
        let mlpOut: MLXArray
        if DF2Fast.w4Group != nil {
            mlpOut = mlp(n2, conv0: (n: n2, dyn: p2, base: base2))
        } else {
            let m = DF2Fast.dynConv(n2, dyn: p2, base: base2, part: 0, residual: nil)
            mlpOut = mlp(m)
        }
        if let nextNorm,
            let fused = DF2Fast.dynConvRms(
                mlpOut, dyn: p2, base: base2, residual: attended, norm: nextNorm, qStart: nextQStart)
        {
            return (fused.0, fused.1, fused.2)
        }
        return (DF2Fast.dynConv(mlpOut, dyn: p2, base: base2, part: 1, residual: attended), nil, nil)
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache
    ) -> MLXArray {
        if DF2Fast.on, x.dim(0) == 1, x.size / x.dim(-1) <= 16 {
            return forwardFast(
                x, normedInput: nil, normedCodes: nil, nextNorm: nil, nextQStart: 0,
                context: context, rope: rope, cache: cache,
                ropeSpec: nil, contextCodes: nil
            ).h
        }
        let (attentionInput, attentionTaps) = attentionConv.prepare(inputLayerNorm(x))
        let attended = x
            + attentionConv.finish(
                selfAttn(attentionInput, context: context, rope: rope, cache: cache),
                dynamic: attentionTaps)
        let (mlpInput, mlpTaps) = mlpConv.prepare(postAttentionLayerNorm(attended))
        return attended + mlpConv.finish(mlp(mlpInput), dynamic: mlpTaps)
    }

    /// Context-only: nothing of the block runs (no norms, convolutions, MLP
    /// or attention output), only the attention sub-layer's context keys and
    /// values enter the cache.
    func absorbContext(
        _ context: MLXArray, rope: RoPELayer, cache: KVCache,
        contextCodes: DF2Fast.W4Codes? = nil, ropeSpec: DF2Fast.RopeSpec? = nil
    ) {
        selfAttn.absorbContext(
            context, rope: rope, cache: cache, contextCodes: contextCodes, ropeSpec: ropeSpec)
    }
}

private final class DFlash2PackedCodebook {
    let packed: MLXArray
    let lookup: MLXArray
    let rows: Int
    let width: Int

    init?(_ source: MLXArray) throws {
        guard source.dtype == .bfloat16, source.ndim == 2,
            source.dim(1) % 64 == 0, source.size > 0 else { return nil }
        rows = source.dim(0)
        width = source.dim(1)
        let bits = source.view(dtype: .uint16).asArray(UInt16.self)
        var rowBases = [UInt8](repeating: 0, count: rows)
        var globalMin = 256, globalMax = 0
        for row in 0 ..< rows {
            var low = 256, high = 0
            for i in row * width ..< (row + 1) * width {
                let exponent = Int((bits[i] >> 7) & 255)
                if exponent != 0 {
                    low = Swift.min(low, exponent)
                    high = Swift.max(high, exponent)
                }
            }
            guard low == 256 || high - low < 14 else { return nil }
            rowBases[row] = low == 256 ? 0 : UInt8(low)
            globalMin = Swift.min(globalMin, low)
            globalMax = Swift.max(globalMax, high)
        }
        // One base for a shared window; otherwise one exact window per row.
        let bases = globalMin == 256 || globalMax - globalMin < 14
            ? [globalMin == 256 ? UInt8(0) : UInt8(globalMin)] : rowBases
        var words = [UInt32](repeating: 0, count: bits.count / 8 * 3)
        for i in stride(from: 0, to: bits.count, by: 8) {
            var block: UInt64 = 0
            let out = i / 8 * 3
            let base = Int(bases[bases.count == 1 ? 0 : i / width])
            for j in 0 ..< 8 {
                let column = i % width + j
                let logical = column / 64 * 64 + (column % 64) / 2 + (column & 1) * 32
                let value = bits[i / width * width + logical]
                let exponent = Int((value >> 7) & 255)
                let field = exponent == 0 ? 0 : exponent - base + 1
                let code = ((value & 0x8000) >> 4) | UInt16(field << 7) | (value & 127)
                block |= UInt64(code) << ((j % 2) * 12)
                if j % 2 == 1 {
                    let shift = (j / 2 * 24) % 32
                    let word = j / 2 * 24 / 32
                    words[out + word] |= UInt32(truncatingIfNeeded: block << shift)
                    if shift > 8 {
                        words[out + word + 1] |= UInt32(block >> (32 - shift))
                    }
                    block = 0
                }
            }
        }
        packed = MLXArray(words)
        // UInt8 exponent bases replace the dependent value lookup.
        lookup = MLXArray(bases)
        let restored = try materialize().view(dtype: .uint16)
        let exact = MLX.all(restored .== source.view(dtype: .uint16))
        try withError { eval(exact) }
        guard exact.item(Bool.self) else { return nil }
    }

    func gather(_ ids: MLXArray, floatOutput: Bool = false) -> MLXArray {
        let output = gatherOutputs(ids, floatOutput: floatOutput)[0]
        return floatOutput ? output : output.view(dtype: .bfloat16)
    }

    private func gatherOutputs(_ ids: MLXArray, floatOutput: Bool) -> [MLXArray] {
        Self.gatherKernel(
            [packed, lookup, ids.asType(.uint32).reshaped(-1)],
            template: [("WIDTH", width), ("FLOAT_OUTPUT", floatOutput)],
            grid: (ids.size * width / 2, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [ids.shape + [width]], outputDTypes: [floatOutput ? .float32 : .uint16])
    }

    /// Every row as BF16. Throws, instead of trapping on an index, when MLX refused the gather.
    func materialize() throws -> MLXArray {
        guard let output = gatherOutputs(MLXArray(UInt32(0) ..< UInt32(rows)), floatOutput: false).first else {
            throw MLXError.caught("DFlash2 packed codebook: the gather returned no output")
        }
        return output.view(dtype: .bfloat16)
    }

    private static let gatherKernel = MLXFast.metalKernel(
        name: "dflash2_codebook_pack12_gather_widen",
        inputNames: ["packed", "lookup", "ids"], outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            if (i >= uint(ids_shape[0]) * (WIDTH / 2)) return;
            const uint row = ids[i / (WIDTH / 2)];
            const uint pair = i % (WIDTH / 2);
            const uint base = lookup[lookup_shape[0] == 1 ? 0u : row];
            const uint index = row * (WIDTH / 2) + pair;
            const uint outIndex = i / (WIDTH / 2) * WIDTH + (pair >> 5u) * 64u + (pair & 31u);
            const uint bit = (index & 3u) * 24u;
            const uint word = (index >> 2u) * 3u + (bit >> 5u);
            const uint shift = bit & 31u;
            uint codes = packed[word] >> shift;
            if (shift > 8u) codes |= packed[word + 1u] << (32u - shift);
            const uint lo = codes & 4095u, hi = (codes >> 12u) & 4095u;
            const uint le = (lo >> 7u) & 15u, he = (hi >> 7u) & 15u;
            const ushort2 values(
                ushort((lo & 127u) | ((lo & 2048u) << 4u) | ((le == 0u ? 0u : base + le - 1u) << 7u)),
                ushort((hi & 127u) | ((hi & 2048u) << 4u) | ((he == 0u ? 0u : base + he - 1u) << 7u)));
            if constexpr (FLOAT_OUTPUT) {
                const float2 widened(cast_to<float>(uint16_to_bfloat16(values.x)),
                                     cast_to<float>(uint16_to_bfloat16(values.y)));
                out[outIndex] = widened.x;
                out[outIndex + 32u] = widened.y;
            } else {
                out[outIndex] = values.x;
                out[outIndex + 32u] = values.y;
            }
            """)
}

/// This plain holder keeps the lookup arrays off Module reflection.
private final class DFlash2CodebookCopies {
    var predecessor: DFlash2PackedCodebook?
    var successor: DFlash2PackedCodebook?
}

// MARK: - The candidate selector

/// The low-rank edge-scored greedy path over the per-position candidate lists.
///
/// The track is greedy, so only the temperature-0 path of the reference
/// `CandidateSelector.select` is ported. The sampling path is not needed and is
/// not here.
final class DFlash2CandidateSelector: Module {
    let topK: Int
    private let codebookCopies = DFlash2CodebookCopies()

    @ParameterInfo(key: "predecessor_codebook") var predecessorCodebook: MLXArray
    @ParameterInfo(key: "successor_codebook") var successorCodebook: MLXArray
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(_ config: DFlash2Configuration) {
        self.topK = config.dflash.selectorTopK
        _predecessorCodebook.wrappedValue = MLXArray.zeros([
            config.vocabularySize, config.dflash.selectorRank,
        ])
        _successorCodebook.wrappedValue = MLXArray.zeros([
            config.vocabularySize, config.dflash.selectorRank,
        ])
        _hiddenProjection.wrappedValue = Linear(
            config.hiddenSize, config.dflash.selectorRank, bias: false)
        super.init()
    }

    // Strict loading finishes before these registered payloads change.
    func preparePackedCodebooks() {
        guard DF2Fast.on, topK == 16,
            predecessorCodebook.ndim == 2, predecessorCodebook.dim(1) == 256,
            successorCodebook.ndim == 2, successorCodebook.dim(1) == 256
        else { return }
        if let copy = try? DFlash2PackedCodebook(predecessorCodebook) {
            codebookCopies.predecessor = copy
            predecessorCodebook._updateInternal(copy.packed)
        }
        if let copy = try? DFlash2PackedCodebook(successorCodebook) {
            codebookCopies.successor = copy
            successorCodebook._updateInternal(copy.packed)
        }
    }

    var predecessorLookup: MLXArray? { codebookCopies.predecessor?.lookup }
    var successorLookup: MLXArray? { codebookCopies.successor?.lookup }

    /// The greedy path.
    ///
    /// - Parameters:
    ///   - hidden: the drafter's final hidden state, `[B, L, hidden]`.
    ///   - logits: the drafter's logits over the same positions, `[B, L, vocab]`.
    ///   - anchor: the token each path starts from, `[B]`.
    /// - Returns: the selected token at each position, `[B, L]`.
    func selectGreedy(hidden: MLXArray, logits: MLXArray, anchor: MLXArray) -> MLXArray {
        let vocabularySize = logits.dim(-1)
        let candidates = argPartition(logits, kth: vocabularySize - topK, axis: -1)[
            0..., 0..., (vocabularySize - topK)...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)

        var predecessor = anchor
        var path = [MLXArray]()
        path.reserveCapacity(hidden.dim(1))
        for position in 0 ..< hidden.dim(1) {
            let positionCandidates = candidates[0..., position, 0...]
            let predecessorValues = codebookCopies.predecessor?.gather(predecessor)
                ?? take(predecessorCodebook, predecessor, axis: 0)
            let successorValues = codebookCopies.successor?.gather(positionCandidates)
                ?? take(successorCodebook, positionCandidates, axis: 0)
            let edges =
                (predecessorValues.expandedDimensions(axis: 1)
                    * projected[0..., position, 0...].expandedDimensions(axis: 1)
                    * successorValues)
                .sum(axis: -1)
            let selected = (unary[0..., position, 0...] + edges).argMax(axis: -1)
            predecessor = takeAlong(
                positionCandidates, selected.expandedDimensions(axis: -1), axis: -1)[0..., 0]
            path.append(predecessor)
        }
        // `argPartition` indexes in UInt32, so the path inherits that dtype.
        // Draft tokens are token ids, and the engine reads them as Int32.
        return stacked(path, axis: 1).asType(.int32)
    }
}

// MARK: - The drafter

public final class DFlash2DraftModel: Module, @unchecked Sendable {
    public let config: DFlash2Configuration

    @ModuleInfo(key: "fc") public var fc: Linear
    @ModuleInfo(key: "hidden_norm") public var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") private var layers: [DFlash2DecoderLayer]
    @ModuleInfo public var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    private let rope: RoPELayer
    private var target: (any DFlash2Target)?

    /// The drafter's own parameter dtype. The Bonsai trunk runs its norms in
    /// FP32 and hands out FP32 activations, so the two tensors that cross from
    /// the target into the drafter — the embedded block and the fused target
    /// hidden state — are cast to this before they reach a drafter weight.
    public var dtype: DType { norm.weight.dtype }

    public init(config: DFlash2Configuration) {
        self.config = config
        _fc.wrappedValue = Linear(config.targetHiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DFlash2DecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(config)
        self.rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    // MARK: Binding

    public func bind(target: any DFlash2Target) throws {
        // The per-row kernels' wide-form bit check (load time, before any proposal).
        if DF2Fast.on, let group = DF2Fast.w4Group { DF2WideLanes.prepare(group: group) }
        if DF2Fast.on {
            DF2Fast.prepareTopkFast()
            HadamardQuantizedLinear.prepareDraftGather()
        }
        guard target.dFlash2VocabularySize == config.vocabularySize else {
            throw DFlash2Error.vocabularyMismatch(
                drafter: config.vocabularySize, target: target.dFlash2VocabularySize)
        }
        guard target.dFlash2HiddenSize == config.hiddenSize else {
            throw DFlash2Error.hiddenSizeMismatch(
                drafter: config.hiddenSize, target: target.dFlash2HiddenSize)
        }
        self.target = target
        // Build the prefix head (and run its bitwise self-check) at load, on
        // both drafter paths, so no proposal pays for it.
        for rows in DF2Vocab.widths(
            vocabularySize: config.vocabularySize, topK: config.dflash.selectorTopK)
        {
            let probe = MLXArray.zeros([1, 1, config.hiddenSize], dtype: .float32)
            eval(target.draftLogitsForDFlash2Hidden(probe, rows: rows))
        }
        if DF2Fast.on {
            DF2Fast.TreeWide.prepare(selector: candidateSelector, vocabulary: config.vocabularySize)
            try prepareFirstRound(target)
            checkLayerZeroFront()
        }
    }

    /// Prepare at load what the first round of the first request would otherwise
    /// build inside a measured decode window: the fast path's cached operands
    /// (FP32 scale and bias transposes, FP32 norm weights, conv bases), the
    /// drafter's kernels at both context shapes (more than 16 rows: the prompt;
    /// 16 or fewer: a round), and the target's exact verify head at the block
    /// width. The proposals run on a throwaway cache and are discarded; nothing
    /// here reaches a request's state.
    private func prepareFirstRound(_ target: any DFlash2Target) throws {
        let draftRows = DF2Vocab.widths(
            vocabularySize: config.vocabularySize, topK: config.dflash.selectorTopK).first
        let cache = try makeCache()
        DF2Fast.w4Recording = !DF2Fast.w4TilesOn
        for rows in [32, 16] {
            let context = MLXArray.zeros([1, rows, config.targetHiddenSize], dtype: dtype)
            eval(try propose(anchor: [0], targetHidden: context, cache: cache, blockSize: config.dflash.blockSize, draftRows: draftRows))
        }
        if DF2Fast.w4Recording {
            DF2Fast.w4Recording = false
            DF2Fast.w4AdoptTiles()
            if DF2Fast.w4TilesOn {
                // Compile the tiled kernels here, not in the first measured round.
                let warm = try makeCache()
                for rows in [32, 16] {
                    let context = MLXArray.zeros([1, rows, config.targetHiddenSize], dtype: dtype)
                    eval(try propose(anchor: [0], targetHidden: context, cache: warm, blockSize: config.dflash.blockSize, draftRows: draftRows))
                }
            }
        }
        target.prepareDFlash2Verify(rows: config.dflash.blockSize)
    }

    // MARK: The cache

    public func makeCache() throws -> [KVCache] {
        try config.layerTypes.map { layerType in
            switch layerType {
            case .fullAttention:
                return DF2Fast.on ? DFlash2LayerKVCache(maxWindowSize: nil) : StandardKVCache()
            case .slidingAttention:
                guard let slidingWindow = config.slidingWindow else {
                    throw DFlash2Error.missingSlidingWindow
                }
                return DF2Fast.on
                    ? DFlash2LayerKVCache(maxWindowSize: slidingWindow - 1)
                    : RotatingKVCache(maxSize: slidingWindow - 1, keep: 0)
            }
        }
    }

    /// The number of context rows the drafter can use. A drafter whose layers
    /// all slide keeps `sliding_window - 1` rows, so the engine never needs to
    /// hand it more than that at prefill.
    public var contextRowLimit: Int? {
        guard config.layerTypes.allSatisfy({ $0 == .slidingAttention }),
            let slidingWindow = config.slidingWindow
        else {
            return nil
        }
        return slidingWindow - 1
    }

    /// Align the drafter's cache with the number of tokens the target has
    /// committed.
    ///
    /// Under the block protocol the engine feeds exactly the accepted positions'
    /// hidden states each round, so the cache offset already tracks the
    /// committed length and this trims nothing. It is here because the
    /// reference keeps the same guard.
    public func trimCache(_ cache: [KVCache], toCommittedLength committed: Int) {
        guard let first = cache.first else { return }
        let excess = first.offset - committed
        guard excess > 0 else { return }
        for c in cache {
            c.trim(excess)
        }

    }

    // MARK: The forward pass

    /// The drafter trunk over one block.
    ///
    /// - Parameters:
    ///   - inputs: the block's token ids, `[B, blockLength]`.
    ///   - targetHidden: the fused target hidden state, `[B, contextLength, targetHiddenSize]`.
    ///   - logitsStart: how many leading block positions to drop before the head.
    func hiddenStates(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        logitsStart: Int
    ) throws -> MLXArray {
        try hiddenStatesWithCodes(
            inputs, targetHidden: targetHidden, cache: cache, logitsStart: logitsStart
        ).hidden
    }

    private func hiddenStatesWithCodes(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        logitsStart: Int,
        projected: DFlash2ContextProjection.Use? = nil
    ) throws -> (hidden: MLXArray, codes: DF2Fast.W4Codes?) {
        guard let target else { throw DFlash2Error.notBound }
        guard cache.count == layers.count else {
            throw DFlash2Error.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlash2Error.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize, actual: targetHidden.dim(-1))
        }

        // Decided once, before anything it replaces is built.
        let front: DFlash2LayerZeroFront? = projected.flatMap {
            usableFront($0, inputs: inputs, cache: cache, logitsStart: logitsStart)
        }
        // Both crossings from the target cast here. The target's embedding is a
        // MODULE call, never a raw weight read.
        var h: MLXArray
        if let front {
            h = front.h
        } else if DF2Fast.on, inputs.ndim == 2, inputs.dim(0) == 1, inputs.dim(1) == config.dflash.blockSize,
            logitsStart == 1, fc is QuantizedLinear
        {
            let anchorEmbed =
                projected?.anchorEmbedding
                ?? target.embedTokensForDFlash2(inputs[0..., 0 ..< 1]).asType(dtype)
            let key = ObjectIdentifier(self)
            let maskEmbed: MLXArray
            if let cached = DF2Fast.maskEmbedCache[key], cached.dim(1) == inputs.dim(1) - 1 {
                maskEmbed = cached
            } else {
                let m = contiguous(target.embedTokensForDFlash2(inputs[0..., 1...]).asType(dtype))
                eval(m)
                DF2Fast.maskEmbedCache[key] = m
                maskEmbed = m
            }
            h = concatenated([anchorEmbed, maskEmbed], axis: 1)
        } else {
            h = target.embedTokensForDFlash2(inputs).asType(dtype)
        }
        if config.dflash.inputEmbeddingScale != 1 {
            h = h * config.dflash.inputEmbeddingScale
        }
        var context: MLXArray
        let cCodes: DF2Fast.W4Codes?
        var deferredContextNorm: (input: MLXArray, group: Int)? = nil
        if let projected {
            context = projected.context
            cCodes = nil
        } else if targetHidden.dim(1) == 0 {
            context = MLXArray.zeros([targetHidden.dim(0), 0, config.hiddenSize], dtype: dtype)
            cCodes = nil
        } else {
            let fcOut = DF2Fast.linear(targetHidden.asType(dtype), fc)
            if let group = DF2Fast.w4Group, fcOut.dim(0) == 1,
                (1 ... 16).contains(fcOut.dim(1)),
                let rq = DF2Fast.rmsQuant(
                    fcOut, hiddenNorm, group: group,
                    writeNorm: !(DF2Fast.on && h.ndim == 3 && h.dim(0) == 1
                        && h.size / h.dim(-1) <= 16 && layers.count == 6
                        && (config.slidingWindow.map { fcOut.dim(1) < $0 } ?? true)))
            {
                if rq.0.size == 1 {
                    // Shape carrier only until every context K/V projection succeeds.
                    // A failed projection below materializes the original normalized output.
                    context = fcOut
                    cCodes = rq.1
                    deferredContextNorm = (fcOut, group)
                } else {
                    (context, cCodes) = rq
                }
            } else {
                context = DF2Fast.rms(fcOut, hiddenNorm)
                cCodes = nil
            }
        }

        if DF2Fast.on, h.ndim == 3, h.dim(0) == 1, h.size / h.dim(-1) <= 16, !layers.isEmpty {
            var rSpec = DF2Fast.ropeSpec(self)
            var normed: MLXArray?
            var normedCodes: DF2Fast.W4Codes?
            if let front {
                normed = front.normed
                rSpec = front.ropeSpec
            } else if let group = DF2Fast.w4Group, let baseSpec = rSpec, let first = cache.first,
                cache.allSatisfy({ $0.offset == first.offset }),
                let fused = DF2Fast.rmsQuantPhases(
                    h, layers[0].inputLayerNorm, group: group, ropeSpec: baseSpec,
                    C: context.dim(1), offset: first.offset)
            {
                normed = fused.0
                normedCodes = fused.1
                rSpec = fused.2
            } else if let group = DF2Fast.w4Group,
                let rq = DF2Fast.rmsQuant(h, layers[0].inputLayerNorm, group: group)
            {
                (normed, normedCodes) = rq
            } else {
                normed = DF2Fast.rms(h, layers[0].inputLayerNorm)
                normedCodes = nil
            }
            var precomputedKV: [(MLXArray, MLXArray)]? = projected?.contextKV
            if let cCodes, layers.count == 6 {
                let lead = [1, context.dim(1)]
                if let kv012 = DF2Fast.w4LinearsFromCodes(
                    cCodes, leadShape: lead,
                    [
                        layers[0].selfAttn.kProj, layers[0].selfAttn.vProj,
                        layers[1].selfAttn.kProj, layers[1].selfAttn.vProj,
                        layers[2].selfAttn.kProj, layers[2].selfAttn.vProj,
                    ]),
                    let kv345 = DF2Fast.w4LinearsFromCodes(
                        cCodes, leadShape: lead,
                        [
                            layers[3].selfAttn.kProj, layers[3].selfAttn.vProj,
                            layers[4].selfAttn.kProj, layers[4].selfAttn.vProj,
                            layers[5].selfAttn.kProj, layers[5].selfAttn.vProj,
                        ])
                {
                    precomputedKV = [
                        (kv012[0], kv012[1]), (kv012[2], kv012[3]), (kv012[4], kv012[5]),
                        (kv345[0], kv345[1]), (kv345[2], kv345[3]), (kv345[4], kv345[5]),
                    ]
                }
            }
            if precomputedKV == nil, let deferredContextNorm {
                // Preserve the original full-output path for any ineligible K/V group.
                context = DF2Fast.rmsQuant(
                    deferredContextNorm.input, hiddenNorm, group: deferredContextNorm.group)!.0
            }
            let blockLen = h.dim(1)
            for (index, layer) in layers.enumerated() {
                let isLast = index + 1 == layers.count
                let nextNorm = isLast ? norm : layers[index + 1].inputLayerNorm
                let nextQStart = isLast ? logitsStart : 0
                let step = layer.forwardFast(
                    h, normedInput: normed, normedCodes: normedCodes,
                    nextNorm: nextNorm, nextQStart: nextQStart,
                    context: context, rope: rope, cache: cache[index],
                    ropeSpec: rSpec, contextCodes: cCodes,
                    precomputedContextKV: precomputedKV?[index],
                    submitAttention: index == 0 && projected != nil,
                    front: index == 0 ? front : nil)
                h = step.h
                normed = step.nextNormed
                normedCodes = step.nextCodes
                // Launch the first layer promptly, then hand off pairs and the tail.
                if index == 0 || (index + 1) % 2 == 0 || isLast {
                    if let normed {
                        if isLast {
                            asyncEval(normed)
                        } else {
                            asyncEval(h, normed)
                        }
                    } else {
                        asyncEval(h)
                    }
                }
            }
            if var finalNormed = normed {
                if logitsStart > 0 && finalNormed.dim(1) != blockLen - logitsStart {
                    finalNormed = finalNormed[0..., logitsStart..., 0...]
                }
                return (finalNormed, normedCodes)
            }
            if logitsStart > 0 {
                h = h[0..., logitsStart..., 0...]
            }
            return (DF2Fast.rms(h, norm), nil)
        }

        for (index, layer) in layers.enumerated() {
            h = layer(h, context: context, rope: rope, cache: cache[index])
            // Hand each finished layer to the GPU while the next one is built.
            if DF2Fast.on { asyncEval(h) }
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return (DF2Fast.rms(h, norm), nil)
    }

    /// The projection's layer-0 front when this block is the one it was built
    /// for: the record fast path's block shape, every cache at the front's
    /// offset, and a phase table covering the context and the block.
    private func usableFront(
        _ projected: DFlash2ContextProjection.Use, inputs: MLXArray, cache: [KVCache], logitsStart: Int
    ) -> DFlash2LayerZeroFront? {
        guard let front = projected.front, DF2Fast.on, layers.count == 6, logitsStart == 1,
            config.dflash.inputEmbeddingScale == 1, fc is QuantizedLinear,
            inputs.ndim == 2, inputs.dim(0) == 1, inputs.dim(1) == front.h.dim(1),
            cache.count == layers.count,
            cache.allSatisfy({ $0.offset == front.cacheOffset })
        else { return nil }
        let phaseRowsNeeded: Int = projected.context.dim(1) + inputs.dim(1)
        guard phaseRowsNeeded <= front.ropeSpec.phaseRows else { return nil }
        return front
    }

    /// Absorb committed context rows into the drafter's caches without
    /// proposing: the context side of `hiddenStates` — the `fc` projection,
    /// `hiddenNorm`, and each layer's context keys and values — on the same
    /// operands in the same order, with no block forward. `blockContext` in
    /// a later `propose` continues these caches exactly as a steady-state
    /// round's incremental absorption does.
    public func absorbContext(targetHidden: MLXArray, cache: [KVCache]) throws {
        try absorbContext(targetHidden: targetHidden, cache: cache, lookupFC: nil)
    }

    func absorbContext(targetHidden: MLXArray, cache: [KVCache], lookupFC: MLXArray?) throws {
        guard cache.count == layers.count else {
            throw DFlash2Error.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlash2Error.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize, actual: targetHidden.dim(-1))
        }
        let fcOutput: MLXArray
        if let lookupFC, dtype == .bfloat16, targetHidden.dtype == .bfloat16,
            targetHidden.ndim == 3, targetHidden.dim(0) == 1,
            (1 ... 16).contains(targetHidden.dim(1)), targetHidden.dim(2) == 32768,
            config.hiddenSize == 4096, lookupFC.dtype == .float32,
            lookupFC.ndim == 3, lookupFC.dim(0) == 1,
            lookupFC.dim(1) == targetHidden.dim(1), lookupFC.dim(2) == config.hiddenSize
        {
            fcOutput = lookupFC
        } else {
            fcOutput = DF2Fast.linear(targetHidden.asType(dtype), fc)
        }
        let context = DF2Fast.rms(fcOutput, hiddenNorm)
        let contextCodes: DF2Fast.W4Codes?
        if DF2Fast.on, layers.count == 6, context.ndim == 3, context.dtype == .float32,
            context.dim(0) == 1, context.dim(2) == 4096, let group = DF2Fast.w4Group
        {
            contextCodes = DF2Fast.w4Quant(context, up: nil, group: group, headMajor: false)
        } else {
            contextCodes = nil
        }
        let ropeSpec = DF2Fast.on ? DF2Fast.ropeSpec(self) : nil
        let contextRows = context.dim(1)
        if let contextCodes, let ropeSpec, layers.count == 6,
            (1 ... 16).contains(contextRows),
            cache.allSatisfy({ $0 is DFlash2LayerKVCache && $0.offset == cache[0].offset }),
            layers.allSatisfy({
                let attention = $0.selfAttn
                return attention.heads == 32 && attention.kvHeads == 8
                    && (attention.slidingWindow.map { contextRows < $0 } ?? true)
            })
        {
            // Every layer keeps the same rows and offset; any context skip
            // keeps the original per-layer path below.
            // Two layers keep the 4096-column narrow dispatch eligible.
            var contextKV: [(MLXArray, MLXArray)] = []
            for index in stride(from: 0, to: layers.count, by: 2) {
                guard let kv = DF2Fast.w4LinearsFromCodes(
                    contextCodes, leadShape: [1, contextRows],
                    [
                        layers[index].selfAttn.kProj, layers[index].selfAttn.vProj,
                        layers[index + 1].selfAttn.kProj, layers[index + 1].selfAttn.vProj,
                    ])
                else { break }
                contextKV.append((kv[0], kv[1]))
                contextKV.append((kv[2], kv[3]))
            }
            if contextKV.count != 6 {
                contextKV = []
                for layer in layers {
                    guard let kv = DF2Fast.w4LinearsFromCodes(
                        contextCodes, leadShape: [1, contextRows],
                        [layer.selfAttn.kProj, layer.selfAttn.vProj])
                    else { break }
                    contextKV.append((kv[0], kv[1]))
                }
            }
            if contextKV.count == 6, contextKV.allSatisfy({ $0.1.dtype == .float32 }),
                let keys = DF2Fast.ctxKGroupNormRope(
                    keys: contextKV.map { $0.0 }, norms: layers.map { $0.selfAttn.kNorm },
                    ropeSpec: ropeSpec, C: contextRows, offset: cache[0].offset)
            {
                for index in layers.indices {
                    let values = contextKV[index].1.reshaped(1, contextRows, 8, 128)
                        .transposed(0, 2, 1, 3)
                    (cache[index] as! DFlash2LayerKVCache).absorb(
                        contextKeys: keys[index], contextValues: values)
                }
                return
            }
        }
        for (index, layer) in layers.enumerated() {
            layer.absorbContext(
                context, rope: rope, cache: cache[index], contextCodes: contextCodes, ropeSpec: ropeSpec)
        }
    }

    func logits(_ hidden: MLXArray, draftRows: Int? = nil) throws -> MLXArray {
        guard let target else { throw DFlash2Error.notBound }
        let rows = draftRows ?? config.vocabularySize
        var logits =
            rows >= config.dflash.selectorTopK && rows < config.vocabularySize
            ? target.draftLogitsForDFlash2Hidden(hidden, rows: rows)
            : target.logitsForDFlash2Hidden(hidden)
        if config.dflash.outputMultiplier != 1 {
            logits = logits * config.dflash.outputMultiplier
        }
        if let cap = config.dflash.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    /// The fast selectors' 128 (id, value) pairs per position: the target's
    /// screened prefix head when it qualified and this block takes it (15
    /// positions, a qualified model-derived prefix, no logit transform), else the top-k
    /// of the full draft logits.
    private func draftTop(_ hidden: MLXArray, blockSize: Int, draftRows: Int?) throws -> [MLXArray] {
        guard let target else { throw DFlash2Error.notBound }
        let rows = draftRows ?? config.vocabularySize
        let softcap: Float = config.dflash.finalLogitSoftcapping ?? 0
        if DF2Fast.screenAllowed, blockSize == 16, rows >= 8 * 256, rows < config.vocabularySize,
            rows >= config.dflash.selectorTopK, config.dflash.outputMultiplier == 1, softcap <= 0,
            let screened = target.draftShortlistForDFlash2Hidden(hidden, rows: rows)
        {
            return [screened.ids, screened.values]
        }
        return DF2Fast.top(try logits(hidden, draftRows: draftRows))
    }

    // MARK: Context projection ahead of acceptance

    /// The acceptance-independent head of the next block's forward, over every
    /// column a verify can confirm: the context `fc` projection, its norm and
    /// codes, every layer's context keys and values (before their norm and
    /// rotation), and the embedding of every candidate anchor. The same
    /// operations run on the same rows as in `hiddenStatesWithCodes`, so a
    /// round that confirms `c` columns reads the first `c` rows of each and the
    /// anchor row `c - 1` instead of recomputing them. nil when the fast path
    /// does not apply.
    public func projectContext(targetHidden: MLXArray, anchors: MLXArray) -> DFlash2ContextProjection? {
        guard let target, DF2Fast.on, fc is QuantizedLinear, layers.count == 6,
            let group = DF2Fast.w4Group, targetHidden.ndim == 3, targetHidden.dim(0) == 1,
            (1 ... 16).contains(targetHidden.dim(1)),
            targetHidden.dim(-1) == config.targetHiddenSize,
            anchors.size == targetHidden.dim(1)
        else { return nil }
        let fcInput = targetHidden.asType(dtype)
        let fcOut = DF2Fast.linear(fcInput, fc)
        guard let normed = DF2Fast.rmsQuant(fcOut, hiddenNorm, group: group) else { return nil }
        let context = normed.0
        guard let kv = contextKeysValues(normed.1, rows: context.dim(1)) else { return nil }
        let embeddings = target.embedTokensForDFlash2(anchors.reshaped([1, -1])).asType(dtype)
        let lookupFC = DF2Fast.fcLookupPrefixReuseEligible(fc, input: fcInput) ? fcOut : nil
        return DFlash2ContextProjection(
            anchorEmbeddings: embeddings, context: context, contextKV: kv, lookupFCOutput: lookupFC)
    }

    /// Every layer's context keys and values from the context codes, in the
    /// two six-output launches `hiddenStatesWithCodes` uses.
    private func contextKeysValues(_ codes: DF2Fast.W4Codes, rows: Int) -> [(MLXArray, MLXArray)]? {
        let lead = [1, rows]
        guard
            let kv012 = DF2Fast.w4LinearsFromCodes(
                codes, leadShape: lead,
                [
                    layers[0].selfAttn.kProj, layers[0].selfAttn.vProj,
                    layers[1].selfAttn.kProj, layers[1].selfAttn.vProj,
                    layers[2].selfAttn.kProj, layers[2].selfAttn.vProj,
                ]),
            let kv345 = DF2Fast.w4LinearsFromCodes(
                codes, leadShape: lead,
                [
                    layers[3].selfAttn.kProj, layers[3].selfAttn.vProj,
                    layers[4].selfAttn.kProj, layers[4].selfAttn.vProj,
                    layers[5].selfAttn.kProj, layers[5].selfAttn.vProj,
                ])
        else { return nil }
        return [
            (kv012[0], kv012[1]), (kv012[2], kv012[3]), (kv012[4], kv012[5]),
            (kv345[0], kv345[1]), (kv345[2], kv345[3]), (kv345[4], kv345[5]),
        ]
    }

    /// The projection rows a block over `contextRows` new rows reads, or nil
    /// when the block does not take the fast path the projection mirrors.
    private func projectionUse(
        _ projection: DFlash2ContextProjection, anchors: Int, blockSize: Int, contextRows: Int
    ) -> DFlash2ContextProjection.Use? {
        guard DF2Fast.on, anchors == 1, blockSize == config.dflash.blockSize, blockSize <= 16,
            fc is QuantizedLinear, layers.count == 6, DF2Fast.w4Group != nil,
            contextRows >= 1, contextRows <= projection.rows
        else { return nil }
        return projection.use(rows: contextRows)
    }

    /// Layer 0's front of the next proposal from this round's verify, queued
    /// with the context projection: the anchor row is `targetTokens[accepted]`'s
    /// embedding, gathered on the GPU by the leading-match count of
    /// `draftTokens` against `targetTokens`, so the host's acceptance is not
    /// waited for. The phase table covers every context width up to `1 + k`,
    /// each row a function of `cacheOffset + row` only. nil when the record
    /// fast path the front mirrors would not run.
    func projectLayerZeroFront(
        _ projection: DFlash2ContextProjection, draftTokens: MLXArray, targetTokens: MLXArray,
        cacheOffset: Int, retainCodes: Bool = false
    ) -> DFlash2LayerZeroFront? {
        let blockSize: Int = config.dflash.blockSize
        let k: Int = blockSize - 1
        let key = ObjectIdentifier(self)
        guard DF2Fast.on, blockSize == 16, layers.count == 6, fc is QuantizedLinear,
            config.dflash.inputEmbeddingScale == 1,
            let group = DF2Fast.w4Group, let baseSpec = DF2Fast.ropeSpec(self),
            draftTokens.size == k, targetTokens.size == blockSize, projection.rows == blockSize,
            let maskEmbed = DF2Fast.maskEmbedCache[key], maskEmbed.ndim == 3,
            maskEmbed.dim(0) == 1, maskEmbed.dim(1) == k, maskEmbed.dim(2) == config.hiddenSize,
            maskEmbed.dtype == projection.anchorEmbeddings.dtype
        else { return nil }
        let drafts: MLXArray = draftTokens.reshaped([k])
        let targets: MLXArray = targetTokens.reshaped([blockSize])[0 ..< k]
        let matches: MLXArray = (drafts .== targets).asType(.int32)
        let accepted: MLXArray = cumprod(matches, axis: 0).sum().asType(.int32)
        let anchor: MLXArray = take(projection.anchorEmbeddings, accepted.reshaped([1]), axis: 1)
        let h: MLXArray = concatenated([anchor, maskEmbed], axis: 1)
        let layer = layers[0]
        let attention = layer.selfAttn
        let projections: [Linear] = [attention.qProj, attention.kProj, attention.vProj]
        guard
            let fused = DF2Fast.rmsQuantPhases(
                h, layer.inputLayerNorm, group: group, ropeSpec: baseSpec,
                C: blockSize, offset: cacheOffset),
            let p1 = DF2Fast.w4LinearsFromCodes(
                fused.1, leadShape: [1, blockSize], [layer.attentionConv.kernelProjection])?.first,
            let qkv = DF2Fast.linearsConv0(
                fused.0, dyn: p1, base: layer.attentionConv.fastBase, projections)
        else { return nil }
        return DFlash2LayerZeroFront(
            cacheOffset: cacheOffset, h: h, normed: fused.0, codes: retainCodes ? fused.1 : nil, ropeSpec: fused.2,
            p1: p1, queries: qkv[0], keys: qkv[1], values: qkv[2])
    }

    // MARK: Proposing

    /// Draft `blockSize - 1` tokens in one forward.
    ///
    /// The block is the last committed token followed by `blockSize - 1` mask
    /// tokens. The head reads the mask positions only, and the greedy candidate
    /// path turns their logits into the draft.
    ///
    /// - Parameters:
    ///   - anchor: the last committed token, one per row.
    ///   - targetHidden: the fused target hidden state of the positions the
    ///     target has already consumed, `[B, contextLength, targetHiddenSize]`.
    /// - Returns: the draft tokens, `[B, blockSize - 1]`.
    public func propose(
        anchor: [Int],
        targetHidden: MLXArray,
        cache: [KVCache],
        blockSize: Int,
        projection: DFlash2ContextProjection? = nil,
        draftRows: Int? = nil
    ) throws -> MLXArray {
        try proposeWithHidden(
            anchor: anchor, targetHidden: targetHidden, cache: cache, blockSize: blockSize,
            projection: projection, draftRows: draftRows
        ).tokens
    }

    /// `propose`, with the final normed hidden state and its codes.
    private func proposeWithHidden(
        anchor: [Int],
        targetHidden: MLXArray,
        cache: [KVCache],
        blockSize: Int,
        projection: DFlash2ContextProjection?,
        draftRows: Int? = nil
    ) throws -> (tokens: MLXArray, hidden: MLXArray, codes: DF2Fast.W4Codes?) {
        guard blockSize >= 2 else { throw DFlash2Error.invalidBlockSize(blockSize) }
        let masks = Array(repeating: Int32(config.maskTokenId), count: blockSize - 1)
        let rows = anchor.flatMap { [Int32($0)] + masks }
        let block = MLXArray(rows, [anchor.count, blockSize])
        let contextUse = projection.flatMap {
            projectionUse($0, anchors: anchor.count, blockSize: blockSize, contextRows: targetHidden.dim(1))
        }

        let (hidden, finalCodes) = try hiddenStatesWithCodes(
            block, targetHidden: targetHidden, cache: cache, logitsStart: 1, projected: contextUse)
        if DF2Fast.on, anchor.count == 1 {
            let sel = candidateSelector
            let projected =
                (finalCodes.flatMap {
                    DF2Fast.w4LinearsFromCodes(
                        $0, leadShape: Array(hidden.shape.dropLast()), [sel.hiddenProjection])
                }?[0]) ?? DF2Fast.linear(hidden, sel.hiddenProjection)
            let top: [MLXArray] = try draftTop(hidden, blockSize: blockSize, draftRows: draftRows)
            if DF2Fast.treeOn, blockSize == 16 {
                let (tokens, desc) = DF2Fast.selectTree(
                    top: top, projected: projected,
                    anchor: MLXArray(anchor.map { Int32($0) }), pred: sel.predecessorCodebook,
                    succ: sel.successorCodebook,
                    predLookup: sel.predecessorLookup, succLookup: sel.successorLookup)
                CBv2BranchVerify.register(tokens: tokens, desc: desc)
                return (tokens, hidden, finalCodes)
            }
            let tokens = DF2Fast.select(
                top: top, projected: projected,
                anchor: MLXArray(anchor.map { Int32($0) }), pred: sel.predecessorCodebook,
                succ: sel.successorCodebook,
                predLookup: sel.predecessorLookup, succLookup: sel.successorLookup)
            return (tokens, hidden, finalCodes)
        }
        let tokens = candidateSelector.selectGreedy(
            hidden: hidden,
            logits: try logits(hidden, draftRows: draftRows),
            anchor: MLXArray(anchor.map { Int32($0) }))
        return (tokens, hidden, finalCodes)
    }

    // MARK: Loading

    public static func isDFlash2Directory(_ directory: URL) -> Bool {
        let configURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return false
        }
        let architectures = object["architectures"] as? [String] ?? []
        return architectures.contains("DFlash2DraftModel") && object["dflash_config"] != nil
    }

    public static func loadConfiguration(from directory: URL) throws -> DFlash2Configuration {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlash2Error.missingConfig(directory.path)
        }
        return try JSONDecoder().decode(
            DFlash2Configuration.self, from: Data(contentsOf: configURL))
    }

    /// Load the drafter from a staged directory.
    ///
    /// The weights load with `verify: [.all]`, so a checkpoint whose key set
    /// does not match this model exactly is a refusal rather than a silent
    /// partial load. The drafter carries no `embed_tokens` and no `lm_head`; it
    /// binds the target's.
    ///
    /// A quantized checkpoint declares `quantization` in its `config.json`. A
    /// module is quantized iff its `.scales` tensor is present, as in
    /// `loadWeights` in MLXLMCommon.
    public static func load(from directory: URL) throws -> DFlash2DraftModel {
        let config = try loadConfiguration(from: directory)
        let base = try JSONDecoder.json5().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: directory.appending(component: "config.json")))
        let drafter = DFlash2DraftModel(config: config)

        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
        else {
            throw DFlash2Error.unreadableDirectory(directory.path)
        }
        var weights = [String: MLXArray]()
        for url in entries where url.pathExtension == "safetensors" {
            weights.merge(try loadArrays(url: url)) { _, new in new }
        }

        if let quantization = base.perLayerQuantization {
            quantize(model: drafter) { path, _ in
                weights["\(path).scales"] != nil
                    ? quantization.quantization(layer: path)?.asTuple : nil
            }
        }
        try drafter.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(drafter)
        weights.removeAll()
        drafter.candidateSelector.preparePackedCodebooks()
        return drafter
    }
}

// MARK: - Fast drafter path (on by default; `DF2_FAST=0` restores the reference graph). Drafter-only
// numerics: it changes which tokens are proposed, never which tokens the target emits.

/// int8 activations (per row, per 32-group) x the pinned 8-bit g32 codes on the NAX
/// tensor units (16 x 32 x 32 per group; codes ^ 0x80 staged as int8, offsets + 128 * s);
/// fused grouped dynamic convolution; fused top-16 and greedy selector.
enum DF2Fast {
    static let on = ProcessInfo.processInfo.environment["DF2_FAST"] != "0"
    /// `DARKBLOOM_BONSAI_DF2_TOKEN_MAJOR_OPROJ=0` keeps the head-major o_proj input.
    static let tokenMajorOProj =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DF2_TOKEN_MAJOR_OPROJ"] != "0"
    /// `DARKBLOOM_BONSAI_DF2_SCREEN=0` keeps the full draft head (no K 2048 screen).
    static let screenAllowed =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DF2_SCREEN"] != "0"
    /// `DARKBLOOM_BONSAI_DF2_FRONT=0` keeps layer 0's projections in the attention-half submit.
    static let earlyFront =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DF2_FRONT"] != "0"
    /// `DARKBLOOM_BONSAI_DF2_FRONT2=0` builds layer 0's front after acceptance, in the proposal.
    static let frontEnabled =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DF2_FRONT2"] != "0"
    /// Written once by `checkLayerZeroFront` at bind, before any proposal; read-only afterwards.
    nonisolated(unsafe) static var frontMatched = false
    static let split = Int(ProcessInfo.processInfo.environment["DF2_SPLIT"] ?? "8") ?? 8
    static let header = """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """

    /// x [R, K] (or silu(x) * x2 when SWIGLU) -> xq int8 [16, K], ascale [16, G], xs [16, G].
    static let quant = MLXFast.metalKernel(
        name: "df2_act_q8", inputNames: ["x", "x2", "ksz"], outputNames: ["xq", "ascale", "xs"],
        source: """
            const int K = ksz[0]; const int R = ksz[1]; const int G = K / 32;
            const int m = int(threadgroup_position_in_grid.y);
            for (int g = int(thread_position_in_threadgroup.x); g < G; g += int(threads_per_threadgroup.x)) {
              float v[32]; float amax = 0.0f, sum = 0.0f;
              if (m < R) {
                for (int u = 0; u < 32; u++) {
                  float a = float(x[(size_t)m * K + g * 32 + u]);
                  if (SWIGLU) { const float b = float(x2[(size_t)m * K + g * 32 + u]); a = a / (1.0f + metal::exp(-a)) * b; }
                  v[u] = a; amax = max(amax, fabs(a)); sum += a;
                }
              } else {
                for (int u = 0; u < 32; u++) { v[u] = 0.0f; }
              }
              const float a = amax > 0.0f ? amax / 127.0f : 1.0f;
              const float ia = 1.0f / a;
              device char4* dst = (device char4*)(xq + (size_t)m * K + g * 32);
              for (int u = 0; u < 8; u++) {
                dst[u] = char4(int4(rint(float4(v[4*u], v[4*u+1], v[4*u+2], v[4*u+3]) * ia)));
              }
              ascale[m * G + g] = m < R ? a : 0.0f;
              xs[m * G + g] = sum;
            }
            """)

    static func matmulSource(layers: Int) -> String {
        var pick = """
            const int K = ksz[0];
            int bx = int(threadgroup_position_in_grid.x);
            const device uint32_t* w = w0; const device float* sT = s0; const device float* bT = b0;
            device float* out = out0; int N = ksz[2];

            """
        for l in 1 ..< layers {
            pick += "if (bx >= N / 32) { bx -= N / 32; w = w\(l); sT = s\(l); bT = b\(l); out = out\(l); N = ksz[\(2 + l)]; }\n"
        }
        return pick + """
            const int G = K / 32;
            const int n0 = bx * 32;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint tid = thread_position_in_threadgroup.x;
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
            tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
            threadgroup uint32_t bs[SPLIT][32 * 8];
            tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> Bt((threadgroup int8_t*)bs[sg], dextents<int, 2>(32, 32));
            auto tA0 = A.template slice<32, 16>(0, 0);
            auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(Bt)>, int32_t>();
            const int mrow = int((lane >> 1) & 3u) + 4 * int((lane >> 4) & 1u);
            const int nb = 4 * int(lane & 1u) + 8 * int((lane >> 3) & 1u);
            bool ok = cT.get_capacity() == 16;
            for (int i = 0; i < 16 && ok; i++) {
              auto ids = cT.get_multidimensional_index(i);
              ok = ids[0] == nb + (i & 3) + 16 * (i >> 3) && ids[1] == mrow + 8 * ((i >> 2) & 1);
            }
            float acc[16], accb[16];
            for (int i = 0; i < 16; i++) { acc[i] = 0.0f; accb[i] = 0.0f; }
            const int m0 = mrow, m1 = mrow + 8;
            const int sj = int(lane & 7u), sc = int(lane >> 3);
            const device uint32_t* wsrc = w + (size_t)(n0 + sc) * (K / 4) + sj;
            threadgroup uint32_t* sdst = bs[sg] + sc * 8 + sj;
            uint32_t wv[8];
            for (int t = 0; t < 8; t++) { wv[t] = wsrc[(size_t)(4 * t) * (K / 4) + int(sg) * 8]; }
            for (int g = int(sg); g < G; g += SPLIT) {
              for (int t = 0; t < 8; t++) { sdst[32 * t] = wv[t] ^ 0x80808080u; }
              if (g + SPLIT < G) { for (int t = 0; t < 8; t++) { wv[t] = wsrc[(size_t)(4 * t) * (K / 4) + (g + SPLIT) * 8]; } }
              simdgroup_barrier(mem_flags::mem_threadgroup);
              auto tA = A.template slice<32, 16>(g * 32, 0);
              op.run(tA, Bt, cT);
              const device float* sr = sT + (size_t)g * N + n0 + nb;
              const device float* br = bT + (size_t)g * N + n0 + nb;
              const float4 s0v = *(const device float4*)sr, s1v = *(const device float4*)(sr + 16);
              const float4 b0v = fma(float4(128.0f), s0v, *(const device float4*)br);
              const float4 b1v = fma(float4(128.0f), s1v, *(const device float4*)(br + 16));
              const float a0 = ascale[m0 * G + g], a1 = ascale[m1 * G + g];
              const float x0 = xs[m0 * G + g], x1 = xs[m1 * G + g];
              for (int i = 0; i < 16; i++) {
                const int j = (i & 3);
                const float s = (i >> 3) ? s1v[j] : s0v[j];
                const float b = (i >> 3) ? b1v[j] : b0v[j];
                const bool hi = (i >> 2) & 1;
                acc[i] = fma(s * (hi ? a1 : a0), float(cT[i]), acc[i]);
                accb[i] = fma(b, hi ? x1 : x0, accb[i]);
              }
              simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup float red[SPLIT][16 * 32];
            for (int i = 0; i < 16; i++) {
              const int m = mrow + 8 * ((i >> 2) & 1), n = nb + (i & 3) + 16 * (i >> 3);
              red[sg][m * 32 + n] = acc[i] + accb[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = int(tid); e < ksz[1] * 32; e += SPLIT * 32) {
              float v = 0.0f;
              for (int q = 0; q < SPLIT; q++) { v += red[q][e]; }
              out[(size_t)(e >> 5) * N + n0 + (e & 31)] = ok ? v : as_type<float>(0x7fc00000u);
            }
            """
    }

    static let matmuls: [MLXFast.MLXFastKernel] = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "df2_i8_matmul\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "xs", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: matmulSource(layers: layers), header: header, ensureRowContiguous: true)
    }

    final class Operands {
        let w: MLXArray, sT: MLXArray, bT: MLXArray, n: Int, k: Int
        init(w: MLXArray, sT: MLXArray, bT: MLXArray, n: Int, k: Int) {
            self.w = w; self.sT = sT; self.bT = bT; self.n = n; self.k = k
        }
    }
    final class FusedKVOperands {
        let w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int, nK: Int
        init(w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int, nK: Int) {
            self.w = w; self.scales = scales; self.biases = biases
            self.groupSize = groupSize; self.bits = bits; self.nK = nK
        }
    }
    final class RopeSpec {
        let freqs: MLXArray
        let mscale: Float
        private let floatLock = NSLock()
        private var cachedFloats: (epsBits: UInt32, mscaleBits: UInt32, array: MLXArray)?
        // Request-local immutable coefficients; never installed in ropeCache.
        let phases: MLXArray?
        let phaseOffset: Int
        let phaseRows: Int
        init(
            freqs: MLXArray, mscale: Float, phases: MLXArray? = nil,
            phaseOffset: Int = 0, phaseRows: Int = 0
        ) {
            self.freqs = freqs
            self.mscale = mscale
            self.phases = phases
            self.phaseOffset = phaseOffset
            self.phaseRows = phaseRows
        }

        func floatParameters(eps: Float) -> MLXArray {
            let epsBits = eps.bitPattern
            let mscaleBits = mscale.bitPattern
            floatLock.lock()
            defer { floatLock.unlock() }
            if let cachedFloats, cachedFloats.epsBits == epsBits,
                cachedFloats.mscaleBits == mscaleBits
            {
                return cachedFloats.array
            }
            let array = MLXArray([eps, mscale])
            cachedFloats = (epsBits, mscaleBits, array)
            return array
        }
    }
    struct W4Codes {
        let xq: MLXArray
        let ascale: MLXArray
        let xs: MLXArray
        let rows: Int
        let K: Int
        let group: Int
    }

    nonisolated(unsafe) static var operandCache: [ObjectIdentifier: Operands] = [:]
    nonisolated(unsafe) static var fusedKVCache: [ObjectIdentifier: FusedKVOperands] = [:]
    nonisolated(unsafe) static var baseCache: [ObjectIdentifier: MLXArray] = [:]
    nonisolated(unsafe) static var normCache: [ObjectIdentifier: MLXArray] = [:]
    nonisolated(unsafe) static var ropeCache: [ObjectIdentifier: RopeSpec] = [:]
    nonisolated(unsafe) static var maskEmbedCache: [ObjectIdentifier: MLXArray] = [:]

    static func normWeight(_ norm: RMSNorm) -> MLXArray {
        let key = ObjectIdentifier(norm)
        if let w = normCache[key] { return w }
        let weight = contiguous(norm.weight.asType(.float32))
        eval(weight)
        normCache[key] = weight
        return weight
    }

    /// `norm(x)` with the norm's weight widened to FP32 once (no per-call cast).
    static func rms(_ x: MLXArray, _ norm: RMSNorm) -> MLXArray {
        guard on else { return norm(x) }
        return MLXFast.rmsNorm(x.asType(.float32), weight: normWeight(norm), eps: norm.eps)
    }

    static func ropeSpec(_ model: DFlash2DraftModel) -> RopeSpec? {
        let key = ObjectIdentifier(model)
        if let r = ropeCache[key] { return r }
        let config = model.config
        guard config.headDim == 128 else { return nil }
        let dims = config.headDim
        let base = config.ropeTheta
        let ropeType: String = {
            if let sc = config.ropeScaling,
                let tv = sc["type"] ?? sc["rope_type"],
                case .string(let s) = tv
            {
                return s
            }
            return "default"
        }()
        let freqs: MLXArray
        let mscale: Float
        if ropeType == "yarn" || ropeType == "deepseek_yarn" || ropeType == "telechat3-yarn" {
            let scalingFactor = config.ropeScaling?["factor"]?.asFloat() ?? 32.0
            let origMax = config.ropeScaling?["original_max_position_embeddings"]?.asInt() ?? 4096
            let betaFast = config.ropeScaling?["beta_fast"]?.asFloat() ?? 32.0
            let betaSlow = config.ropeScaling?["beta_slow"]?.asFloat() ?? 1.0
            let ms = config.ropeScaling?["mscale"]?.asFloat() ?? 1.0
            let msAll = config.ropeScaling?["mscale_all_dim"]?.asFloat() ?? 0.0
            func findDim(_ numRot: Float) -> Float {
                Float(dims) * log(Float(origMax) / (numRot * 2 * Float.pi)) / (2 * log(base))
            }
            let low = Swift.max(Int(floor(findDim(betaFast))), 0)
            let high = Swift.min(Int(ceil(findDim(betaSlow))), dims - 1)
            var maxVal = Float(high)
            let minVal = Float(low)
            if minVal == maxVal { maxVal += 0.001 }
            let linearFunc =
                (MLXArray(0 ..< (dims / 2)).asType(.float32) - minVal) / (maxVal - minVal)
            let freqMask = 1.0 - clip(linearFunc, min: 0, max: 1)
            let exponents = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims)
            let freqExtra = pow(base, exponents)
            let freqInter = scalingFactor * pow(base, exponents)
            let f = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
            freqs = contiguous(f)
            func getMscale(_ scale: Float, _ m: Float) -> Float {
                scale <= 1 ? 1.0 : (0.1 * m * log(scale) + 1.0)
            }
            mscale = getMscale(scalingFactor, ms) / getMscale(scalingFactor, msAll)
        } else if ropeType == "default" {
            let exponents = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims)
            freqs = contiguous(pow(base, exponents))
            mscale = 1.0
        } else {
            return nil
        }
        eval(freqs)
        let spec = RopeSpec(freqs: freqs, mscale: mscale)
        ropeCache[key] = spec
        return spec
    }

    static func fusedKVLinear(_ x: MLXArray, kProj: Linear, vProj: Linear) -> (MLXArray, MLXArray)? {
        guard on else { return nil }
        let key = ObjectIdentifier(kProj)
        let op: FusedKVOperands
        if let cached = fusedKVCache[key] {
            op = cached
        } else {
            guard let qk = kProj as? QuantizedLinear, let qv = vProj as? QuantizedLinear,
                qk.mode == .affine, qv.mode == .affine, qk.bits == qv.bits,
                qk.groupSize == qv.groupSize, let bk = qk.biases, let bv = qv.biases,
                qk.bias == nil, qv.bias == nil
            else { return nil }
            let w = contiguous(concatenated([qk.weight, qv.weight], axis: 0))
            let s = contiguous(concatenated([qk.scales, qv.scales], axis: 0))
            let b = contiguous(concatenated([bk, bv], axis: 0))
            eval(w, s, b)
            op = FusedKVOperands(
                w: w, scales: s, biases: b, groupSize: qk.groupSize, bits: qk.bits,
                nK: qk.weight.dim(0))
            fusedKVCache[key] = op
        }
        let kv = quantizedMM(
            x, op.w, scales: op.scales, biases: op.biases, transpose: true,
            groupSize: op.groupSize, bits: op.bits, mode: .affine)
        return (kv[0..., 0..., ..<op.nK], kv[0..., 0..., op.nK...])
    }

    static func operands(_ layer: Linear) -> Operands? {
        if let o = operandCache[ObjectIdentifier(layer)] { return o }
        guard let q = layer as? QuantizedLinear, q.bits == 8, q.groupSize == 32, q.mode == .affine,
            let biases = q.biases, q.bias == nil, q.weight.dim(0) % 32 == 0
        else { return nil }
        let sT = contiguous(q.scales.asType(.float32).transposed())
        let bT = contiguous(biases.asType(.float32).transposed())
        eval(sT, bT)
        let o = Operands(w: q.weight, sT: sT, bT: bT, n: q.weight.dim(0), k: q.weight.dim(1) * 4)
        operandCache[ObjectIdentifier(layer)] = o
        return o
    }

    /// Activation codes of x [.., K] (rows <= 16); `up` makes them silu(x) * up.
    static func codes(_ x: MLXArray, up: MLXArray? = nil) -> (MLXArray, MLXArray, MLXArray, Int)? {
        let K = x.dim(-1)
        let rows = x.size / K
        guard (1 ... 16).contains(rows), K % 32 == 0 else { return nil }
        let x2 = up ?? x
        let o = quant(
            [x.reshaped([rows, K]), x2.reshaped([rows, K]), HostConstants.int32([Int32(K), Int32(rows)])],
            template: [("SWIGLU", up == nil ? 0 : 1)],
            grid: (128, 16, 1), threadGroup: (128, 1, 1),
            outputShapes: [[16 * K], [16 * (K / 32)], [16 * (K / 32)]],
            outputDTypes: [.int8, .float32, .float32])
        return (o[0], o[1], o[2], rows)
    }

    /// `layers(x)` (or of silu(x) * up) for sibling drafter linears sharing x; nil when not eligible.
    static func linears(_ x: MLXArray, _ layers: [Linear], up: MLXArray? = nil) -> [MLXArray]? {
        guard on, (1 ... 3).contains(layers.count) else { return nil }
        if let group = w4Group, let y = w4Linears(x, layers, up: up, group: group) {
            return y
        }
        guard (1 ... 16).contains(x.size / x.dim(-1)) else { return nil }
        let ops = layers.compactMap { operands($0) }
        guard ops.count == layers.count, ops.allSatisfy({ $0.k == x.dim(-1) }),
            let (xq, ascale, xs, rows) = codes(x, up: up)
        else { return nil }
        let ns = ops.map(\.n)
        let outs = matmuls[layers.count - 1](
            [xq] + ops.map(\.w) + ops.map(\.sT) + ops.map(\.bT)
                + [ascale, xs, HostConstants.int32([Int32(x.dim(-1)), Int32(rows)] + ns.map { Int32($0) })],
            template: [("SPLIT", split)],
            grid: (ns.reduce(0, +) / 32 * split * 32, 1, 1), threadGroup: (split * 32, 1, 1),
            outputShapes: ns.map { [rows, $0] }, outputDTypes: ns.map { _ in .float32 })
        let lead = Array(x.shape.dropLast())
        return zip(outs, ns).map { $0.reshaped(lead + [$1]) }
    }

    // MARK: 4-bit drafter linears, re-quantized at load (drafter numerics only)

    /// Input columns per 4-bit group (64 or 128); nil (`DF2_W4=0`) keeps the pinned 8-bit g32 codes.
    static let w4Group: Int? = {
        let v = ProcessInfo.processInfo.environment["DF2_W4"] ?? "64"
        return v == "128" ? 128 : (v == "0" ? nil : 64)
    }()

    final class W4Operands {
        var w: MLXArray, sT: MLXArray, bT: MLXArray
        let n: Int, k: Int
        var pair24: W4Pair24? = nil
        /// The same codes, scales and offsets in `w4Tile` order once `w4AdoptTiles` admitted them;
        /// `w`, `sT` and `bT` are then released.
        var tiles: MLXArray? = nil
        private let sizeLock = NSLock()
        private var cachedSizes: (words: [Int32], array: MLXArray)?
        init(w: MLXArray, sT: MLXArray, bT: MLXArray, n: Int, k: Int) {
            self.w = w; self.sT = sT; self.bT = bT; self.n = n; self.k = k
        }

        func dispatchSizes(K: Int, rows: Int, ns: [Int]) -> MLXArray {
            let words = [Int32(K), Int32(rows)] + ns.map { Int32($0) }
            sizeLock.lock()
            defer { sizeLock.unlock() }
            if let cachedSizes, cachedSizes.words == words {
                return cachedSizes.array
            }
            let array = MLXArray(words)
            cachedSizes = (words, array)
            return array
        }
    }
    nonisolated(unsafe) static var w4Cache: [ObjectIdentifier: W4Operands] = [:]

    /// An in-memory 4-bit affine copy of one pinned linear (the file on disk is untouched), built once
    /// at load: per group of `group` input columns, the range [lo, hi] shrunk toward zero by the factor
    /// in {1, 0.95, ..., 0.75} with the least squared error against the pinned weights, then FP16
    /// scale (hi - lo) / 15 and offset lo. Codes are row-major [N, K / 2]; a word holds k = 8j ... 8j + 3
    /// in its low nibbles and k = 8j + 4 ... 8j + 7 in its high nibbles. Scales and offsets are [G, N].
    static func w4Operands(_ layer: Linear, group: Int) -> W4Operands? {
        if let o = w4Cache[ObjectIdentifier(layer)] { return o }
        guard let q = layer as? QuantizedLinear, q.mode == .affine, let biases = q.biases, q.bias == nil
        else { return nil }
        let N = q.weight.dim(0)
        let K = q.weight.dim(1) * 32 / q.bits
        guard N % 32 == 0, K % group == 0 else { return nil }
        let G = K / group
        let w = dequantized(
            q.weight, scales: q.scales, biases: biases, groupSize: q.groupSize, bits: q.bits,
            dtype: .float32
        ).reshaped([N, G, group])
        let lo0 = w.min(axis: -1), hi0 = w.max(axis: -1)
        var best: MLXArray? = nil
        var lo = lo0, hi = hi0
        for c: Float in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75] {
            let l = lo0 * c, h = hi0 * c
            let sc = which(h .> l, (h - l) / 15, Float(1)).asType(.float16).asType(.float32)
            let bc = l.asType(.float16).asType(.float32)
            let u = clip(
                round((w - bc.expandedDimensions(axis: -1)) / sc.expandedDimensions(axis: -1)),
                min: 0, max: 15)
            let e = square(u * sc.expandedDimensions(axis: -1) + bc.expandedDimensions(axis: -1) - w)
                .sum(axis: -1)
            if let b0 = best {
                let better = e .< b0
                best = which(better, e, b0)
                lo = which(better, l, lo)
                hi = which(better, h, hi)
            } else {
                best = e
            }
            eval(best!, lo, hi)
        }
        let s = which(hi .> lo, (hi - lo) / 15, Float(1)).asType(.float16).asType(.float32)
        let b = lo.asType(.float16).asType(.float32)
        let u = clip(
            round((w - b.expandedDimensions(axis: -1)) / s.expandedDimensions(axis: -1)), min: 0, max: 15
        ).asType(.uint32).reshaped([N, K / 8, 2, 4])
        let bytes = u[0..., 0..., 0, 0...] | (u[0..., 0..., 1, 0...] << MLXArray(UInt32(4)))
        let packed = (bytes << MLXArray([0, 8, 16, 24] as [UInt32])).sum(axis: -1).asType(.uint32)
        let sT = contiguous(s.asType(.float16).transposed())
        let bT = contiguous(b.asType(.float16).transposed())
        eval(packed, sT, bT)
        let o = W4Operands(w: packed, sT: sT, bT: bT, n: N, k: K)
        if group == 64 && ((N == 4096 && K == 32768)
            || (N == 12288 && K == 4096) || (N == 4096 && K == 12288))
        {
            o.pair24 = W4Pair24(sT, bT)
        }
        if w4TilesOn {
            // A linear first used after adoption: same layout function as the checked ones.
            o.tiles = w4Tile(packed, sT: sT, bT: bT, n: N, k: K, group: group)
            eval(o.tiles!)
            w4Release(o)
        }
        w4Cache[ObjectIdentifier(layer)] = o
        return o
    }

    // MARK: 4-bit codes in load-time tiles

    static let signedNibbles =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DF2_SIGNED_NIBBLES"] != "0"

    /// uint4 words per tile: one 32-column block of one group, codes then FP16 scales then FP16 offsets.
    static func w4TileU4(group: Int) -> Int { 32 * (group / 32) + 8 }

    /// The record operands of one linear as tiles `[N / 32, G, w4TileU4]` (uint4 words; stored as
    /// uint32): per 32-column block and group, the block's 32 rows of `group / 2` code bytes in row
    /// order (with each nibble stored as `u ^ 8` when `signedNibbles` is on), then the block's 32 FP16
    /// scales, then its 32 FP16 offsets. A bijection on the stored bits: no value is rounded or dropped.
    static func w4Tile(_ w: MLXArray, sT: MLXArray, bT: MLXArray, n N: Int, k K: Int, group: Int) -> MLXArray {
        let G = K / group
        let words = group / 8
        let rawCodes = w.reshaped([N / 32, 32, G, words]).transposed(0, 2, 1, 3).reshaped([N / 32, G, 32 * words])
        let codes = signedNibbles ? (rawCodes ^ MLXArray(UInt32(0x8888_8888))) : rawCodes
        func cols(_ a: MLXArray) -> MLXArray {
            contiguous(a.reshaped([G, N / 32, 32]).transposed(1, 0, 2)).view(dtype: .uint32)
        }
        return contiguous(concatenated([codes, cols(sT), cols(bT)], axis: 2))
    }

    /// Drop the record-layout arrays of an operand that reads tiles (the tiles hold the same bits).
    static func w4Release(_ o: W4Operands) {
        let none = MLXArray.zeros([1], dtype: .uint32)
        o.w = none
        o.sT = none
        o.bT = none
    }

    /// True once every recorded launch passed `w4AdoptTiles`; later linears are built tiled at once.
    nonisolated(unsafe) static var w4TilesOn = false
    /// While true, `w4LinearsFromCodes` records each distinct launch (its linears, K, form, narrow).
    nonisolated(unsafe) static var w4Recording = false
    nonisolated(unsafe) static var w4Launches: [[ObjectIdentifier]] = []
    nonisolated(unsafe) static var w4LaunchSeen: Set<[ObjectIdentifier]> = []

    /// Load time, after the warm-up proposals built every 4-bit operand: build each linear's tiles,
    /// run every recorded launch through the record kernel (record layout) and the tiled kernel (tiles)
    /// on a fixed probe at rows 16 and 15, and compare the outputs bit for bit. All equal: every operand
    /// switches to its tiles and its record arrays are released. Any difference, compile or run error,
    /// or an operand that cannot be tiled: nothing changes and the record kernels keep running.
    /// `DF2_W4_TILES=0` skips this. One stderr line either way.
    static func w4AdoptTiles() {
        let value = ProcessInfo.processInfo.environment["DF2_W4_TILES"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let launches = w4Launches
        w4Launches = []
        w4LaunchSeen = []
        guard !w4TilesOn, let group = w4Group else { return }
        let note: String
        if ["0", "false", "no", "off"].contains(value ?? "") {
            note = "off (DF2_W4_TILES=0)"
        } else if launches.isEmpty {
            note = "off (no 4-bit launch recorded)"
        } else {
            var ops: [ObjectIdentifier: W4Operands] = [:]
            for launch in launches {
                for id in launch { ops[id] = w4Cache[id] }
            }
            var failure: String? = nil
            var tiles: [ObjectIdentifier: MLXArray] = [:]
            for (id, o) in ops where o.tiles == nil {
                guard o.n % 32 == 0, o.k % group == 0 else {
                    failure = "linear N \(o.n) K \(o.k) not tileable"
                    break
                }
                tiles[id] = w4Tile(o.w, sT: o.sT, bT: o.bT, n: o.n, k: o.k, group: group)
            }
            if failure == nil {
                eval(Array(tiles.values))
                failure = w4TilesCheck(launches, ops: ops, tiles: tiles, group: group)
            }
            if let failure {
                note = "off, record layout kept (\(failure))"
            } else {
                for (id, t) in tiles {
                    ops[id]!.tiles = t
                    w4Release(ops[id]!)
                }
                w4TilesOn = true
                note = "on (\(launches.count) launches, \(ops.count) linears; bit-identical to the record kernel at rows 16 and 15)"
            }
        }
        FileHandle.standardError.write(Data("DF2 w4 tiles: \(note)\n".utf8))
    }

    /// Nil when every launch's tiled output equals its record output bit for bit, else why not.
    static func w4TilesCheck(
        _ launches: [[ObjectIdentifier]], ops: [ObjectIdentifier: W4Operands],
        tiles: [ObjectIdentifier: MLXArray], group: Int
    ) -> String? {
        var failure: String? = nil
        w4CheckFailed = false
        w4CheckNote = ""
        withErrorHandler({ message in
            if !DF2Fast.w4CheckFailed { DF2Fast.w4CheckNote = String(message.prefix(160)) }
            DF2Fast.w4CheckFailed = true
        }) {
            for launch in launches {
                let lops = launch.map { ops[$0]! }
                let K = lops[0].k
                let G = K / group
                let ns = lops.map(\.n)
                let (form, narrow) = w4LaunchForm(ns)
                let xq = (arange(16 * K, dtype: .int32) * MLXArray(Int32(37)) + MLXArray(Int32(11 + K)))
                    % MLXArray(Int32(255)) - MLXArray(Int32(127))
                let j = arange(16 * G, dtype: .int32)
                let ascale = ((j % MLXArray(Int32(11))).asType(.float32) + 1) / 256
                let xs = ((j % MLXArray(Int32(17))).asType(.float32) - 8) / 64
                let xq8 = xq.asType(.int8)
                // The same operands reading tiles; the record objects keep their arrays until adoption.
                let none = MLXArray.zeros([1], dtype: .uint32)
                let tops = zip(launch, lops).map { (id, o) -> W4Operands in
                    let t = W4Operands(w: none, sT: none, bT: none, n: o.n, k: o.k)
                    t.tiles = tiles[id] ?? o.tiles
                    t.pair24 = o.pair24
                    return t
                }
                for rows in [16, 15] {
                    let outputs: (record: [MLXArray], candidate: [MLXArray])
                    var method = "plain"
                    if let metadata = w4LaunchMetadata(lops, K: K, rows: rows, group: group) {
                        // The kernel this launch runs at these rows: the record's cached choice.
                        let m = w4Pair24Selected(lops, ns: ns, metadata: metadata, form: form, narrow: narrow)
                            .method(rows: rows)
                        method = "\(m)"
                        func run(_ o: [W4Operands]) -> [MLXArray] {
                            w4Pair24Output(
                                m, form: form, narrow: narrow, xq: xq8, ascale: ascale, xs: xs, rows: rows,
                                ops: o, ns: ns, metadata: metadata)
                        }
                        outputs = (run(lops), run(tops))
                    } else {
                        outputs = (
                            w4Run(
                                form: form, narrow: narrow, xq: xq8, ascale: ascale, xs: xs, rows: rows, K: K,
                                ws: lops.map(\.w), sTs: lops.map(\.sT), bTs: lops.map(\.bT), ns: ns),
                            w4Run(
                                form: form, narrow: narrow, tiled: true, xq: xq8, ascale: ascale, xs: xs,
                                rows: rows, K: K, ws: w4CodeArrays(tops), sTs: [], bTs: [], ns: ns)
                        )
                    }
                    let record = concatenated(outputs.record, axis: 1).view(dtype: .uint32)
                    let candidate = concatenated(outputs.candidate, axis: 1).view(dtype: .uint32)
                    let differ = (record .!= candidate).asType(.int32).sum()
                    let nan = (record .== MLXArray(UInt32(0x7fc0_0000))).asType(.int32).sum()
                    eval(differ, nan)
                    if w4CheckFailed {
                        failure = "compile or run error: \(w4CheckNote)"
                        return
                    }
                    let d = differ.item(Int32.self)
                    if d != 0 {
                        failure = "\(d) words differ at K \(K), N \(ns), rows \(rows), form \(form), narrow \(narrow), \(method)"
                        return
                    }
                    if nan.item(Int32.self) != 0 {
                        failure = "record output NaN at K \(K), N \(ns), rows \(rows)"
                        return
                    }
                }
            }
        }
        if failure == nil && w4CheckFailed { failure = "compile or run error: \(w4CheckNote)" }
        return failure
    }

    /// x [R, K] (or silu(x) * x2, or head-major [1, 32, R, 128]) -> int8 codes per row and per KG-column group.
    /// 16 contiguous columns per thread (`c0 = tid * 16`), single-pass register buffering, and SIMD-shuffle group reduction.
    static let quantG = MLXFast.metalKernel(
        name: "df2_act_q8g", inputNames: ["x", "x2", "ksz"], outputNames: ["xq", "ascale", "xs"],
        source: """
            const int K = ksz[0]; const int R = ksz[1]; const int G = K / KG;
            const int tid = int(thread_position_in_grid.x);
            const uint lane = thread_index_in_simdgroup;
            const int c0 = tid * 16;
            if (c0 >= K) return;
            const int m = int(threadgroup_position_in_grid.y);
            const int g = tid / (KG / 16);
            const size_t dstBase = (size_t)m * K + c0;
            if (m >= R) {
              device char4* dst = (device char4*)(xq + dstBase);
              for (int u = 0; u < 4; u++) { dst[u] = char4(0); }
              if ((tid & ((KG / 16) - 1)) == 0) {
                ascale[m * G + g] = 0.0f;
                xs[m * G + g] = 0.0f;
              }
              return;
            }
            const size_t srcBase = HEAD_MAJOR
                ? (((size_t)(c0 >> 7) * R + m) * 128 + (c0 & 127))
                : dstBase;
            float nv[16];
            float local_amax = 0.0f;
            for (int u = 0; u < 16; u++) {
              float a = float(x[srcBase + u]);
              if (SWIGLU) { const float b = float(x2[srcBase + u]); a = a / (1.0f + metal::exp(-a)) * b; }
              nv[u] = a;
              local_amax = max(local_amax, fabs(a));
            }
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 1));
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 2));
            if (KG == 128) {
              local_amax = max(local_amax, simd_shuffle_xor(local_amax, 4));
            }
            float seq_sum = 0.0f;
            for (int step = 0; step < (KG / 16); step++) {
              if (step > 0) {
                seq_sum = simd_shuffle_up(seq_sum, 1);
              }
              for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
            }
            const float a = local_amax > 0.0f ? local_amax / 127.0f : 1.0f;
            const float ia = 1.0f / a;
            device char4* dst = (device char4*)(xq + dstBase);
            for (int u = 0; u < 4; u++) {
              const float4 fv = float4(nv[4 * u], nv[4 * u + 1], nv[4 * u + 2], nv[4 * u + 3]);
              dst[u] = char4(int4(rint(fv * ia)));
            }
            if ((int(lane) & ((KG / 16) - 1)) == (KG / 16) - 1) {
              ascale[m * G + g] = a;
              xs[m * G + g] = seq_sum;
            }
            """)

    /// Fused `dynConv(part: 0, residual: nil)` + `quantG` in one kernel: 16 contiguous columns (= 1 conv group) per thread.
    static let quantGConv0 = MLXFast.metalKernel(
        name: "df2_act_q8g_conv0", inputNames: ["h", "dyn", "base", "ksz"],
        outputNames: ["xq", "ascale", "xs"],
        source: """
            const int K = ksz[0]; const int R = ksz[1]; const int G = K / KG;
            const int tid = int(thread_position_in_grid.x);
            const uint lane = thread_index_in_simdgroup;
            const int c0 = tid * 16;
            if (c0 >= K) return;
            const int m = int(threadgroup_position_in_grid.y);
            const int g = tid / (KG / 16);
            const size_t rowBase = (size_t)m * K + c0;
            if (m >= R) {
              device char4* dst = (device char4*)(xq + rowBase);
              for (int u = 0; u < 4; u++) { dst[u] = char4(0); }
              if ((tid & ((KG / 16) - 1)) == 0) {
                ascale[m * G + g] = 0.0f;
                xs[m * G + g] = 0.0f;
              }
              return;
            }
            const int dw = 4 * (K / 16);
            const float d0 = float(dyn[m * dw + tid]);
            const float d1 = m >= 1 ? float(dyn[m * dw + (K >> 4) + tid]) : 0.0f;
            float nv[16];
            float local_amax = 0.0f;
            for (int u = 0; u < 16; u++) {
              const int c = c0 + u;
              float a = (float(base[c]) + d0) * float(h[rowBase + u]);
              if (m >= 1) {
                a += (float(base[K + c]) + d1) * float(h[rowBase - K + u]);
              }
              nv[u] = a;
              local_amax = max(local_amax, fabs(a));
            }
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 1));
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 2));
            if (KG == 128) {
              local_amax = max(local_amax, simd_shuffle_xor(local_amax, 4));
            }
            float seq_sum = 0.0f;
            for (int step = 0; step < (KG / 16); step++) {
              if (step > 0) {
                seq_sum = simd_shuffle_up(seq_sum, 1);
              }
              for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
            }
            const float a = local_amax > 0.0f ? local_amax / 127.0f : 1.0f;
            const float ia = 1.0f / a;
            device char4* dst = (device char4*)(xq + rowBase);
            for (int u = 0; u < 4; u++) {
              const float4 fv = float4(nv[4 * u], nv[4 * u + 1], nv[4 * u + 2], nv[4 * u + 3]);
              dst[u] = char4(int4(rint(fv * ia)));
            }
            if ((int(lane) & ((KG / 16) - 1)) == (KG / 16) - 1) {
              ascale[m * G + g] = a;
              xs[m * G + g] = seq_sum;
            }
            """)

    /// Optional FC metadata. Original half arrays remain available for the accepted fallback.
    final class W4Pair24 {
        let words: MLXArray, bases: MLXArray
        let shape: [Int]
        let count: Int
        let selection = W4Pair24SelectionCache()

        init?(_ scales: MLXArray, _ offsets: MLXArray) {
            guard scales.dtype == .float16, offsets.dtype == .float16,
                scales.ndim == 2, scales.shape == offsets.shape, scales.dim(1) % 4 == 0,
                scales.strides == [scales.dim(1), 1], offsets.strides == scales.strides,
                scales.size > 0, scales.size % 4 == 0, scales.size / 4 <= Int.max / 3
            else { return nil }
            let s = scales.view(dtype: .uint16).asArray(UInt16.self)
            let b = offsets.view(dtype: .uint16).asArray(UInt16.self)
            let sb = UInt32(s.min()!), bb = UInt32(b.min()!)
            guard UInt32(s.max()!) - sb <= 4095, UInt32(b.max()!) - bb <= 4095 else { return nil }
            func pair(_ i: Int) -> UInt32 {
                (UInt32(s[i]) - sb) | ((UInt32(b[i]) - bb) << 12)
            }
            var packed = [UInt32](repeating: 0, count: s.count / 4 * 3)
            for i in stride(from: 0, to: s.count, by: 4) {
                let a = pair(i), c = pair(i + 1), d = pair(i + 2), e = pair(i + 3)
                let j = i / 4 * 3
                packed[j] = a | (c << 24)
                packed[j + 1] = (c >> 8) | (d << 16)
                packed[j + 2] = (d >> 16) | (e << 8)
            }
            // Check every inverse before admitting this representation.
            for i in s.indices {
                let bit = (i & 3) * 24, j = i / 4 * 3 + bit / 32, shift = bit & 31
                var value = packed[j] >> shift
                if shift > 8 { value |= packed[j + 1] << (32 - shift) }
                guard sb + (value & 4095) == UInt32(s[i]),
                    bb + ((value >> 12) & 4095) == UInt32(b[i])
                else { return nil }
            }
            shape = scales.shape
            count = s.count
            words = MLXArray(packed)
            bases = MLXArray([sb | (bb << 16)])
        }
    }

    /// Three scalar loads; uint3 would add padding between these12-byte tiles.
    static let pair24Header = """
        inline void df2_pair24_load(const device uint* words, uint bases, size_t tile,
                                   thread half4& scales, thread half4& offsets) {
            const uint a = words[3 * tile], b = words[3 * tile + 1], c = words[3 * tile + 2];
            const uint4 pairs(a & 0x00ffffffu, ((a >> 24) | (b << 8)) & 0x00ffffffu,
                              ((b >> 16) | (c << 16)) & 0x00ffffffu, c >> 8);
            const ushort4 s = ushort4(pairs & uint4(4095u)) + ushort(bases & 65535u);
            const ushort4 o = ushort4((pairs >> 12) & uint4(4095u)) + ushort(bases >> 16);
            scales = as_type<half4>(s);
            offsets = as_type<half4>(o);
        }

        """

    /// Only metadata loads change. The original raw generator and FP32 arithmetic stay intact.
    /// `tiled`: the codes come from `w4Tile` tiles (`w4Source(tiled:recordMetadata:)`).
    static func w4Pair24Source(form: Int, narrow: Bool, tiled: Bool = false) -> String {
        var source = w4Source(layers: 1, group: 64, form: form, narrow: narrow, tiled: tiled, recordMetadata: true)
        func change(_ before: String, _ after: String) {
            precondition(source.contains(before))
            source = source.replacingOccurrences(of: before, with: after)
        }
        change("const device half* sT = s0; const device half* bT = b0;",
               "const device uint* metadata = s0; const uint metadataBases = b0[0];")
        if form == 0 {
            change("const device half* sr = sT + (size_t)g * N + n0 + nb;", """
                half4 sm0, sm1, om0, om1;
                const size_t tile = ((size_t)g * N + n0 + nb) / 4;
                df2_pair24_load(metadata, metadataBases, tile, sm0, om0);
                df2_pair24_load(metadata, metadataBases, tile + 4, sm1, om1);
                """)
            change("const device half* br = bT + (size_t)g * N + n0 + nb;", "")
            change("const float4 s0v = float4(*(const device half4*)sr), s1v = float4(*(const device half4*)(sr + 16));",
                   "const float4 s0v = float4(sm0), s1v = float4(sm1);")
            change("float4(*(const device half4*)br)", "float4(om0)")
            change("float4(*(const device half4*)(br + 16))", "float4(om1)")
        } else {
            change("const device half* sc = sT + n0 + nb;", "const size_t tile = (n0 + nb) / 4;")
            change("const device half* oc = bT + n0 + nb;", "")
            if narrow {
                change("half4 ps0 = *(const device half4*)(sc + (size_t)sg * N);", """
                    half4 ps0, po0;
                    df2_pair24_load(metadata, metadataBases, (size_t)sg * N / 4 + tile, ps0, po0);
                    """)
                change("half4 po0 = *(const device half4*)(oc + (size_t)sg * N);", "")
                change("ps0 = *(const device half4*)(sc + r);",
                       "df2_pair24_load(metadata, metadataBases, r / 4 + tile, ps0, po0);")
                change("po0 = *(const device half4*)(oc + r);", "")
            } else {
                change("half4 ps0 = *(const device half4*)(sc + (size_t)sg * N), ps1 = *(const device half4*)(sc + (size_t)sg * N + 16);", """
                    half4 ps0, ps1, po0, po1;
                    df2_pair24_load(metadata, metadataBases, (size_t)sg * N / 4 + tile, ps0, po0);
                    df2_pair24_load(metadata, metadataBases, (size_t)sg * N / 4 + tile + 4, ps1, po1);
                    """)
                change("half4 po0 = *(const device half4*)(oc + (size_t)sg * N), po1 = *(const device half4*)(oc + (size_t)sg * N + 16);", "")
                change("ps0 = *(const device half4*)(sc + r); ps1 = *(const device half4*)(sc + r + 16);", """
                    df2_pair24_load(metadata, metadataBases, r / 4 + tile, ps0, po0);
                    df2_pair24_load(metadata, metadataBases, r / 4 + tile + 4, ps1, po1);
                    """)
                change("po0 = *(const device half4*)(oc + r); po1 = *(const device half4*)(oc + r + 16);", "")
            }
        }
        precondition(!source.contains("sT") && !source.contains("bT") && !source.contains("const device half4*"))
        return source
    }

    static let w4Pair24Kernels: [Int: MLXFast.MLXFastKernel] = {
        var kernels: [Int: MLXFast.MLXFastKernel] = [:]
        for (form, narrow) in [(0, false), (1, false), (2, false), (1, true), (2, true)] {
            kernels[form * 2 + (narrow ? 1 : 0)] = MLXFast.metalKernel(
                name: "df2_w4g64_pair24f\(form)_n\(narrow ? 16 : 32)",
                inputNames: ["xq", "w0", "s0", "b0", "ascale", "xs", "ksz"], outputNames: ["out0"],
                source: w4Pair24Source(form: form, narrow: narrow), header: header + pair24Header,
                ensureRowContiguous: true)
        }
        return kernels
    }()

    static func w4Pair24Run(
        form: Int, narrow: Bool, xq: MLXArray, ascale: MLXArray, xs: MLXArray, rows: Int,
        op: W4Operands, metadata: W4Pair24, sizeDescriptor: MLXArray? = nil
    ) -> [MLXArray] {
        let columns = narrow ? 16 : 32
        let sizes = sizeDescriptor ?? HostConstants.int32([Int32(32768), Int32(rows), Int32(4096)])
        let kernel = op.tiles == nil
            ? w4Pair24Kernels[form * 2 + (narrow ? 1 : 0)]!
            : w4TiledVariant("df2_w4tg64_pair24f\(form)_n\(columns)") { name in
                MLXFast.metalKernel(
                    name: name, inputNames: ["xq", "w0", "s0", "b0", "ascale", "xs", "ksz"], outputNames: ["out0"],
                    source: w4Pair24Source(form: form, narrow: narrow, tiled: true), header: header + pair24Header,
                    ensureRowContiguous: true)
            }
        return kernel(
            [xq, op.tiles ?? op.w, metadata.words, metadata.bases, ascale, xs, sizes],
            template: [("SPLIT", split)], grid: (4096 / columns * split * 32, 1, 1),
            threadGroup: (split * 32, 1, 1), outputShapes: [[rows, 4096]], outputDTypes: [.float32])
    }

    /// Shared by the compact MLP consumer and its metadata-only proof.
    static func w4Pair24MLPSelection(layers: Int) -> String {
        precondition(layers == 1 || layers == 2)
        var source = """
            const device uint* metadata = s0;
            uint metadataBases = b0[0];

            """
        if layers == 2 {
            source += "if (li == 1) { metadata = s1; metadataBases = b1[0]; }\n"
        }
        return source
    }

    /// Keep the FC bodies exact; the MLP adds sibling selection and the accepted static-K form.
    static func w4Pair24MLPSource(layers: Int, form: Int, narrow: Bool, tiled: Bool = false) -> String {
        precondition(layers == 1 || (layers == 2 && !narrow))
        var source = w4Pair24Source(form: form, narrow: narrow, tiled: tiled)
        let pointer = "const device uint* metadata = s0; const uint metadataBases = b0[0];"
        precondition(source.contains(pointer))
        source = source.replacingOccurrences(of: pointer, with: "int li = 0;")
        if layers == 2 {
            let pick = "device float* out = out0; int N = ksz[2];"
            precondition(source.contains(pick))
            source = source.replacingOccurrences(of: pick, with: pick + """

                if (bx >= N / 32) { bx -= N / 32; w = w1; li = 1; out = out1; N = ksz[3]; }
                """)
        }
        precondition(source.contains("#define LO(v)"))
        source = source.replacingOccurrences(of: "#define LO(v)",
            with: w4Pair24MLPSelection(layers: layers) + "#define LO(v)")
        return w4Specialize(source)
    }

    static let w4Pair24MLPKernels: [Int: MLXFast.MLXFastKernel] = {
        var kernels: [Int: MLXFast.MLXFastKernel] = [:]
        for layers in 1 ... 2 {
            for (form, narrow) in [(0, false), (1, false), (2, false), (1, true), (2, true)]
                where layers == 1 || !narrow
            {
                let key = (layers - 1) * 6 + form * 2 + (narrow ? 1 : 0)
                kernels[key] = MLXFast.metalKernel(
                    name: "df2_w4g64_pair24_mlp_f\(form)_n\(narrow ? 16 : 32)_forms_\(layers)",
                    inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                        + (0 ..< layers).map { "b\($0)" } + ["ascale", "xs", "ksz"],
                    outputNames: (0 ..< layers).map { "out\($0)" },
                    source: w4Pair24MLPSource(layers: layers, form: form, narrow: narrow),
                    header: header + pair24Header, ensureRowContiguous: true)
            }
        }
        return kernels
    }()

    static func w4Pair24MLPMetadata(_ ops: [W4Operands]) -> [W4Pair24]? {
        let gateUp = ops.count == 2 && ops.allSatisfy { $0.k == 4096 && $0.n == 12288 }
        let down = ops.count == 1 && ops[0].k == 12288 && ops[0].n == 4096
        guard gateUp || down else { return nil }
        let metadata = ops.compactMap(\.pair24)
        return metadata.count == ops.count ? metadata : nil
    }

    /// These ordered inputs are also used by the proof; each payload keeps its own data header.
    static func w4Pair24MLPMetadataInputs(_ metadata: [W4Pair24]) -> [MLXArray] {
        metadata.map(\.words) + metadata.map(\.bases)
    }

    static func w4Pair24MLPRun(
        form: Int, narrow: Bool, xq: MLXArray, ascale: MLXArray, xs: MLXArray, rows: Int,
        ops: [W4Operands], metadata: [W4Pair24], sizeDescriptor: MLXArray? = nil
    ) -> [MLXArray] {
        let K = ops[0].k, ns = ops.map(\.n)
        precondition(w4FormsFit(K: K, rows: rows, ns: ns) && ops.count == metadata.count)
        let key = (ops.count - 1) * 6 + form * 2 + (narrow ? 1 : 0)
        let sizes = sizeDescriptor ?? HostConstants.int32([Int32(K), Int32(rows)] + ns.map { Int32($0) })
        let layers = ops.count
        let kernel = !w4Tiled(ops)
            ? w4Pair24MLPKernels[key]!
            : w4TiledVariant("df2_w4tg64_pair24_mlp_f\(form)_n\(narrow ? 16 : 32)_forms_\(layers)") { name in
                MLXFast.metalKernel(
                    name: name,
                    inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                        + (0 ..< layers).map { "b\($0)" } + ["ascale", "xs", "ksz"],
                    outputNames: (0 ..< layers).map { "out\($0)" },
                    source: w4Pair24MLPSource(layers: layers, form: form, narrow: narrow, tiled: true),
                    header: header + pair24Header, ensureRowContiguous: true)
            }
        return kernel(
            [xq] + w4CodeArrays(ops) + w4Pair24MLPMetadataInputs(metadata)
                + [ascale, xs, sizes],
            template: [("SPLIT", split), ("KC", K)],
            grid: (ns.reduce(0, +) / (narrow ? 16 : 32) * split * 32, 1, 1),
            threadGroup: (split * 32, 1, 1),
            outputShapes: ns.map { [rows, $0] }, outputDTypes: ns.map { _ in .float32 })
    }

    enum W4Pair24Method: Int {
        case raw, compact, quartet, rawWide, fcStatic
    }
    struct W4Pair24Choice: Equatable {
        let ordinary: W4Pair24Method
        let full16: W4Pair24Method
        func method(rows: Int) -> W4Pair24Method { rows == 16 ? full16 : ordinary }
    }
    struct W4Pair24SelectionKey: Equatable {
        let operands: [ObjectIdentifier]
        let K: Int
        let ns: [Int]
        let group: Int
        let form: Int
        let narrow: Bool
        let split: Int
    }
    final class W4Pair24SelectionCache {
        private let lock = NSLock()
        private var cached: (key: W4Pair24SelectionKey, choice: W4Pair24Choice)?
        func resolve(
            _ ops: [W4Operands], K: Int, ns: [Int], group: Int, form: Int, narrow: Bool, split: Int,
            build: () -> W4Pair24Choice
        ) -> W4Pair24Choice {
            lock.lock()
            defer { lock.unlock() }
            if let cached {
                let key = cached.key
                guard key.K == K, key.ns == ns, key.group == group, key.form == form,
                    key.narrow == narrow, key.split == split, key.operands.count == ops.count
                else { return W4Pair24Choice(ordinary: .raw, full16: .raw) }
                for index in ops.indices {
                    if key.operands[index] != ObjectIdentifier(ops[index]) {
                        return W4Pair24Choice(ordinary: .raw, full16: .raw)
                    }
                }
                return cached.choice
            }
            let key = W4Pair24SelectionKey(operands: ops.map { ObjectIdentifier($0) }, K: K, ns: ns,
                group: group, form: form, narrow: narrow, split: split)
            let choice = build()
            cached = (key, choice)
            return choice
        }

        func fcLookupPrefixCompatible(_ op: W4Operands) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard let cached else { return false }
            let key = cached.key
            guard key.operands.count == 1, key.operands[0] == ObjectIdentifier(op),
                key.K == 32768, key.ns.count == 1, key.ns[0] == 4096,
                key.group == 64, key.form == 2, key.narrow, key.split == 8,
                (cached.choice.ordinary == cached.choice.full16 || cached.choice.full16 == .fcStatic)
            else { return false }
            return cached.choice.ordinary == .raw || cached.choice.ordinary == .compact
        }
    }

    static func fcLookupPrefixReuseEligible(_ layer: Linear, input: MLXArray) -> Bool {
        guard on, w4Group == 64, split == 8, input.dtype == .bfloat16,
            input.ndim == 3, input.dim(0) == 1, input.dim(1) == 16, input.dim(2) == 32768,
            w4SelectedForm == 2, w4Narrow, w4Form2Narrow,
            let op = w4Cache[ObjectIdentifier(layer)], op.k == 32768, op.n == 4096,
            let metadata = op.pair24
        else { return false }
        let wide = DF2WideLanes.serves(
            .init(kind: .quantG, dtype: input.dtype, dtype2: input.dtype, k: 32768, group: 64, flag: 0))
        return !wide && metadata.selection.fcLookupPrefixCompatible(op)
    }

    /// Raw is the incumbent.3% in all3 rounds is a conservative initial policy.
    static func w4Pair24Choose(valid: [Bool], samples: [[UInt64]]) -> W4Pair24Choice {
        func wins(_ candidate: W4Pair24Method, _ control: W4Pair24Method) -> Bool {
            let a = candidate.rawValue, b = control.rawValue
            guard valid[a], valid[b], samples[a].count == 3, samples[b].count == 3 else { return false }
            return zip(samples[a], samples[b]).allSatisfy {
                $0.0 > 0 && $0.1 > 0 && Double($0.1) / Double($0.0) >= 1.03
            }
        }
        let ordinary: W4Pair24Method = wins(.compact, .raw) ? .compact : .raw
        let full16: W4Pair24Method =
            wins(.quartet, .raw) && (ordinary == .raw || wins(.quartet, .compact)) ? .quartet : ordinary
        return W4Pair24Choice(ordinary: ordinary, full16: full16)
    }

    static func w4Pair24DownWideEligible(
        _ ops: [W4Operands], rows: Int, ns: [Int], group: Int, form: Int, narrow: Bool, split: Int
    ) -> Bool {
        rows == 16 && group == 64 && split == 8 && form == 2 && narrow && ops.count == 1
            && ops[0].k == 12288 && ops[0].n == 4096 && ns == [4096]
    }

    static func w4Pair24ChooseWithWide(
        valid: [Bool], samples: [[UInt64]], eligible: Bool
    ) -> W4Pair24Choice {
        let current = w4Pair24Choose(valid: valid, samples: samples)
        guard eligible, valid.count == 4, samples.count == 4 else { return current }
        func beats(_ control: W4Pair24Method) -> Bool {
            let index = control.rawValue
            guard valid[3], valid[index], samples[3].count == 3, samples[index].count == 3 else { return false }
            return zip(samples[3], samples[index]).allSatisfy {
                $0.0 > 0 && $0.1 > 0 && Double($0.1) / Double($0.0) >= 1.03
            }
        }
        if beats(.raw) && (current.full16 == .raw || beats(current.full16)) {
            return W4Pair24Choice(ordinary: current.ordinary, full16: .rawWide)
        }
        return current
    }

    static func w4FCStaticEligible(
        _ ops: [W4Operands], rows: Int, ns: [Int], group: Int, form: Int, narrow: Bool, split: Int
    ) -> Bool {
        guard rows == 16, group == 64, form == 2, narrow, split == 8, ops.count == 1,
            ops[0].k == 32768, ops[0].n == 4096, ns == [4096] else { return false }
        let span = ops[0].n.multipliedReportingOverflow(by: ops[0].k)
        return !span.overflow && span.partialValue > 0 && span.partialValue < Int(UInt32.max)
    }

    static func w4Pair24ChooseWithFCStatic(
        valid: [Bool], samples: [[UInt64]], wideEligible: Bool, eligible: Bool
    ) -> W4Pair24Choice {
        let current = w4Pair24ChooseWithWide(valid: valid, samples: samples, eligible: wideEligible)
        guard eligible, valid.count == 5, samples.count == 5 else { return current }
        func beats(_ control: W4Pair24Method) -> Bool {
            let index = control.rawValue
            guard valid[4], valid[index], samples[4].count == 3, samples[index].count == 3 else { return false }
            return zip(samples[4], samples[index]).allSatisfy {
                $0.0 > 0 && $0.1 > 0 && Double($0.1) / Double($0.0) >= 1.03
            }
        }
        if beats(.raw) && (current.full16 == .raw || beats(current.full16)) {
            return W4Pair24Choice(ordinary: current.ordinary, full16: .fcStatic)
        }
        return current
    }

    static func w4FCStaticRun(
        xq: MLXArray, ascale: MLXArray, xs: MLXArray, op: W4Operands, sizeDescriptor: MLXArray? = nil
    ) -> [MLXArray] {
        let sizes = sizeDescriptor ?? HostConstants.int32([Int32(32768), Int32(16), Int32(4096)])
        if let tiles = op.tiles {
            return w4TiledKernel(form: 2, narrow: true, specialized: true, layers: 1)(
                [xq, tiles, ascale, xs, sizes],
                template: [("SPLIT", split), ("KC", 32768)],
                grid: (4096 / 16 * split * 32, 1, 1), threadGroup: (split * 32, 1, 1),
                outputShapes: [[16, 4096]], outputDTypes: [.float32])
        }
        return w4Form2NarrowFormsKernels[0](
            [xq, op.w, op.sT, op.bT, ascale, xs, sizes],
            template: [("SPLIT", split), ("KC", 32768)],
            grid: (4096 / 16 * split * 32, 1, 1), threadGroup: (split * 32, 1, 1),
            outputShapes: [[16, 4096]], outputDTypes: [.float32])
    }

    /// Build plus eval/encoding/default-stream completion; output roots live past the stop clock.
    static func w4Pair24Time(_ build: () -> [MLXArray]) -> UInt64? {
        do {
            return try withError {
                Stream.gpu.synchronize()
                let start = DispatchTime.now().uptimeNanoseconds
                let outputs = build()
                eval(outputs)
                Stream.gpu.synchronize()
                let end = DispatchTime.now().uptimeNanoseconds
                withExtendedLifetime(outputs) {}
                return end > start ? end - start : nil
            }
        } catch { return nil }
    }

    static func w4Pair24Exact(_ build: () -> [MLXArray], reference: MLXArray) -> Bool {
        (try? withError {
            let output = concatenated(build(), axis: 1)
            let finite = isFinite(output).all()
            eval(finite)
            guard finite.item(Bool.self) else { return false }
            let exact = (reference.view(dtype: .uint32) .== output.view(dtype: .uint32)).all()
            eval(exact)
            return exact.item(Bool.self)
        }) ?? false
    }

    static func w4Pair24Output(
        _ method: W4Pair24Method, form: Int, narrow: Bool,
        xq: MLXArray, ascale: MLXArray, xs: MLXArray, rows: Int,
        ops: [W4Operands], ns: [Int], metadata: [W4Pair24], sizeDescriptor: MLXArray? = nil
    ) -> [MLXArray] {
        switch method {
        case .raw:
            return w4Run(form: form, narrow: narrow, tiled: w4Tiled(ops), xq: xq, ascale: ascale, xs: xs, rows: rows,
                K: ops[0].k, ws: w4CodeArrays(ops), sTs: ops.map(\.sT), bTs: ops.map(\.bT),
                ns: ns, sizeDescriptor: sizeDescriptor)
        case .compact:
            if ops[0].k == 32768 {
                return w4Pair24Run(form: form, narrow: narrow, xq: xq, ascale: ascale, xs: xs,
                    rows: rows, op: ops[0], metadata: metadata[0], sizeDescriptor: sizeDescriptor)
            }
            return w4Pair24MLPRun(form: form, narrow: narrow, xq: xq, ascale: ascale, xs: xs,
                rows: rows, ops: ops, metadata: metadata, sizeDescriptor: sizeDescriptor)
        case .quartet:
            if rows == 16 && ops.count == 2 && ops.allSatisfy({ $0.k == 4096 && $0.n == 12288 })
                && form == 2 && !narrow
            {
                return w4Pair24QuartetWideRun(xq: xq, ascale: ascale, xs: xs,
                    ops: ops, metadata: metadata, sizeDescriptor: sizeDescriptor)
            }
            precondition(rows == 16 && ops.count == 1 && ops[0].k == 12288 && ops[0].n == 4096
                && form == 2 && narrow)
            return w4Pair24QuartetRun(xq: xq, ascale: ascale, xs: xs,
                op: ops[0], metadata: metadata[0], sizeDescriptor: sizeDescriptor)
        case .rawWide:
            precondition(w4Pair24DownWideEligible(ops, rows: rows, ns: ns, group: 64,
                form: form, narrow: narrow, split: split))
            return w4Run(form: form, narrow: false, tiled: w4Tiled(ops), xq: xq, ascale: ascale, xs: xs, rows: rows,
                K: ops[0].k, ws: w4CodeArrays(ops), sTs: ops.map(\.sT), bTs: ops.map(\.bT),
                ns: ns, sizeDescriptor: sizeDescriptor)
        case .fcStatic:
            precondition(w4FCStaticEligible(ops, rows: rows, ns: ns, group: 64,
                form: form, narrow: narrow, split: split))
            return w4FCStaticRun(xq: xq, ascale: ascale, xs: xs,
                op: ops[0], sizeDescriptor: sizeDescriptor)
        }
    }

    /// The existing bind preparation reaches this R16 key; runtime row changes do not retune it.
    static func w4Pair24Selected(
        _ ops: [W4Operands], ns: [Int], metadata: [W4Pair24], form: Int, narrow: Bool
    ) -> W4Pair24Choice {
        let K = ops[0].k, G = K / 64
        return metadata[0].selection.resolve(ops, K: K, ns: ns, group: 64,
            form: form, narrow: narrow, split: split) {
            let coldStart = DispatchTime.now().uptimeNanoseconds
            let xq = MLXArray((0 ..< 16 * K).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [16 * K])
            let a = MLXArray((0 ..< 16 * G).map { Float(1 + $0 % 11) / 256 }, [16 * G])
            let x = MLXArray((0 ..< 16 * G).map { Float($0 % 17 - 8) / 64 }, [16 * G])
            let sizes = MLXArray([Int32(K), Int32(16)] + ns.map { Int32($0) })
            func build(_ index: Int) -> [MLXArray] {
                w4Pair24Output(W4Pair24Method(rawValue: index)!, form: form, narrow: narrow,
                    xq: xq, ascale: a, xs: x, rows: 16, ops: ops, ns: ns, metadata: metadata, sizeDescriptor: sizes)
            }
            let quartet = ops.count == 1 && K == 12288 && ns == [4096] && form == 2 && narrow
            let wideQuartet = ops.count == 2 && K == 4096 && ns == [12288, 12288] && form == 2 && !narrow
            let rawWide = w4Pair24DownWideEligible(ops, rows: 16, ns: ns, group: 64,
                form: form, narrow: narrow, split: split)
            let fcStatic = w4FCStaticEligible(ops, rows: 16, ns: ns, group: 64,
                form: form, narrow: narrow, split: split)
            let methods = fcStatic ? [0, 1, 4] : (rawWide ? [0, 1, 2, 3] : (quartet || wideQuartet ? [0, 1, 2] : [0, 1]))
            let candidateCount = fcStatic ? 5 : (rawWide ? 4 : 3)
            var valid = [Bool](repeating: false, count: candidateCount)
            var reference: MLXArray?
            valid[0] = (try? withError {
                let output = concatenated(build(0), axis: 1)
                let finite = isFinite(output).all()
                eval(finite)
                reference = output
                return finite.item(Bool.self)
            }) ?? false
            if valid[0], let reference {
                for index in methods.dropFirst() {
                    valid[index] = w4Pair24Exact({ build(index) }, reference: reference)
                }
            }
            reference = nil
            var warm = [UInt64](repeating: 0, count: candidateCount)
            var samples = [[UInt64]](repeating: [], count: candidateCount)
            var legs = [[[UInt64]]](repeating: [], count: candidateCount)
            if valid[0] {
                for index in methods where valid[index] {
                    if let elapsed = w4Pair24Time({ build(index) }) { warm[index] = elapsed }
                    else { valid[index] = false }
                }
                if valid[0] {
                    let orders = fcStatic
                        ? [[0, 1, 4], [1, 4, 0], [4, 0, 1]]
                        : (rawWide
                            ? [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]]
                            : [[0, 1, 2], [1, 2, 0], [2, 0, 1]])
                    for order in orders {
                        if !valid[0] { break }
                        var round = [[UInt64]](repeating: [], count: candidateCount)
                        for mirrored in [order, Array(order.reversed())] {
                            if !valid[0] { break }
                            for index in mirrored where methods.contains(index) && valid[index] {
                                if let elapsed = w4Pair24Time({ build(index) }) { round[index].append(elapsed) }
                                else { valid[index] = false; samples[index] = [] }
                            }
                        }
                        for index in methods {
                            legs[index].append(round[index])
                            if valid[index] && round[index].count == 2 {
                                samples[index].append(round[index][0] + round[index][1])
                            }
                        }
                    }
                }
            }
            let choice = w4Pair24ChooseWithFCStatic(valid: valid, samples: samples, wideEligible: rawWide, eligible: fcStatic)
            withExtendedLifetime((xq, a, x, sizes)) {}
            let cold = DispatchTime.now().uptimeNanoseconds - coldStart
            FileHandle.standardError.write(Data(
                "DF2 pair24 select K\(K) N\(ns) form\(form) narrow=\(narrow) R16 policy=all3>=1.03 valid=\(valid) warm_ns=\(warm) legs_ns=\(legs) round_sums_ns=\(samples) ordinary=\(choice.ordinary) full16=\(choice.full16) calibration_ns=\(cold)\n".utf8))
            return choice
        }
    }

    /// All32 lanes reach each gather; bits4,3,0 identify one four-lane quartet.
    static let pair24QuartetHeader = """
        inline void df2_pair24_quartet_load(const device uint* words, uint bases, uint lane, size_t tile,
                                           thread half4& scales, thread half4& offsets) {
            const uint component = (lane >> 1) & 3u;
            const uint bit = 24u * component, shift = bit & 31u;
            const size_t word = 3 * tile + bit / 32u;
            uint value = words[word] >> shift;
            if (shift > 8u) value |= words[word + 1] << (32u - shift);
            const ushort s = ushort(bases & 65535u) + ushort(value & 4095u);
            const ushort o = ushort(bases >> 16) + ushort((value >> 12) & 4095u);
            const uint pair = uint(s) | (uint(o) << 16);
            uint4 pairs(pair, simd_shuffle_xor(pair, 2), simd_shuffle_xor(pair, 4),
                        simd_shuffle_xor(pair, 6));
            pairs = (component & 1u) != 0u ? pairs.yxwz : pairs;
            pairs = (component & 2u) != 0u ? pairs.zwxy : pairs;
            scales = as_type<half4>(ushort4(pairs & uint4(65535u)));
            offsets = as_type<half4>(ushort4(pairs >> 16));
        }
        
        """

    static func w4Pair24QuartetSource(tiled: Bool = false) -> String {
        let source = w4Pair24MLPSource(layers: 1, form: 2, narrow: true, tiled: tiled)
        let call = "df2_pair24_load(metadata, metadataBases, "
        precondition(source.components(separatedBy: call).count == 3)
        return source.replacingOccurrences(of: call,
            with: "df2_pair24_quartet_load(metadata, metadataBases, lane, ")
    }

    static let w4Pair24QuartetKernel = MLXFast.metalKernel(
        name: "df2_w4g64_pair24_quartet_xor_mlp_f2_n16_forms_1",
        inputNames: ["xq", "w0", "s0", "b0", "ascale", "xs", "ksz"], outputNames: ["out0"],
        source: w4Pair24QuartetSource(), header: header + pair24Header + pair24QuartetHeader,
        ensureRowContiguous: true)

    static func w4Pair24QuartetRun(
        xq: MLXArray, ascale: MLXArray, xs: MLXArray, op: W4Operands, metadata: W4Pair24,
        sizeDescriptor: MLXArray? = nil
    ) -> [MLXArray] {
        precondition(op.k == 12288 && op.n == 4096)
        let sizes = sizeDescriptor ?? HostConstants.int32([Int32(12288), Int32(16), Int32(4096)])
        let kernel = op.tiles == nil
            ? w4Pair24QuartetKernel
            : w4TiledVariant("df2_w4tg64_pair24_quartet_xor_mlp_f2_n16_forms_1") { name in
                MLXFast.metalKernel(
                    name: name, inputNames: ["xq", "w0", "s0", "b0", "ascale", "xs", "ksz"], outputNames: ["out0"],
                    source: w4Pair24QuartetSource(tiled: true), header: header + pair24Header + pair24QuartetHeader,
                    ensureRowContiguous: true)
            }
        return kernel(
            [xq, op.tiles ?? op.w, metadata.words, metadata.bases, ascale, xs, sizes],
            template: [("SPLIT", split), ("KC", 12288)],
            grid: (4096 / 16 * split * 32, 1, 1), threadGroup: (split * 32, 1, 1),
            outputShapes: [[16, 4096]], outputDTypes: [.float32])
    }

    /// Form2 wide gate/up only. The existing selector checks and measures this alternative.
    static func w4Pair24QuartetWideSource(tiled: Bool = false) -> String {
        let source = w4Pair24MLPSource(layers: 2, form: 2, narrow: false, tiled: tiled)
        let call = "df2_pair24_load(metadata, metadataBases, "
        precondition(source.components(separatedBy: call).count == 5)
        return source.replacingOccurrences(of: call,
            with: "df2_pair24_quartet_load(metadata, metadataBases, lane, ")
    }

    static let w4Pair24QuartetWideKernel = MLXFast.metalKernel(
        name: "df2_w4g64_pair24_quartet_xor_mlp_f2_n32_forms_2",
        inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "xs", "ksz"],
        outputNames: ["out0", "out1"],
        source: w4Pair24QuartetWideSource(), header: header + pair24Header + pair24QuartetHeader,
        ensureRowContiguous: true)

    static func w4Pair24QuartetWideRun(
        xq: MLXArray, ascale: MLXArray, xs: MLXArray, ops: [W4Operands], metadata: [W4Pair24],
        sizeDescriptor: MLXArray? = nil
    ) -> [MLXArray] {
        precondition(ops.count == 2 && metadata.count == 2
            && ops.allSatisfy { $0.k == 4096 && $0.n == 12288 })
        let sizes = sizeDescriptor ?? HostConstants.int32([Int32(4096), Int32(16), Int32(12288), Int32(12288)])
        let kernel = !w4Tiled(ops)
            ? w4Pair24QuartetWideKernel
            : w4TiledVariant("df2_w4tg64_pair24_quartet_xor_mlp_f2_n32_forms_2") { name in
                MLXFast.metalKernel(
                    name: name, inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "xs", "ksz"],
                    outputNames: ["out0", "out1"],
                    source: w4Pair24QuartetWideSource(tiled: true),
                    header: header + pair24Header + pair24QuartetHeader, ensureRowContiguous: true)
            }
        return kernel(
            [xq] + w4CodeArrays(ops) + w4Pair24MLPMetadataInputs(metadata) + [ascale, xs, sizes],
            template: [("SPLIT", split), ("KC", 4096)],
            grid: (24576 / 32 * split * 32, 1, 1), threadGroup: (split * 32, 1, 1),
            outputShapes: [[16, 12288], [16, 12288]], outputDTypes: [.float32, .float32])
    }

    /// int8 activations x the 4-bit codes on the NAX tensor units, one 16 x 32 x `group` op per group,
    /// SPLIT simdgroups over K, 32 output columns per threadgroup, the codes of the next group
    /// prefetched in registers; the epilogue adds s * a * C + (lo + 8 s) * sum(x) per group.
    /// `form` 0 (the record kernel): each simdgroup stages its 32 x `group` tile as int8 (nibble - 8)
    /// in threadgroup memory and reads the group's scales, offsets and activation sums after the op.
    /// Form 1: the same, with those epilogue operands loaded one group ahead, beside the code
    /// prefetch, so no device load sits between the op and the epilogue. The tile, the op and the
    /// epilogue arithmetic are the same in both forms, so the output bits are the same; `w4Form`
    /// checks that on the device before form 1 is used.
    /// `narrow` (form 1 only): 16 output columns per threadgroup instead of 32, so a call launches
    /// twice the threadgroups. The op, its layout guard, the group loop and the FP32 steps per output
    /// are form 1's; columns 16-31 of the staged tile and their destination elements (8-15) are dead.
    /// `tiled`: the operands are `w4Tile` tiles (input `w<l>` only, no `s<l>` / `b<l>`). Lane `lane`'s
    /// load p of group g is uint4 `p * 32 + lane` of the tile (the record reads the same 16 bytes from
    /// row `lane / ru + p * 32 / ru`, uint4 `lane % ru`), so each load is 512 contiguous bytes; scales
    /// and offsets are read from the tile's tail at the record's columns. Staging, op, epilogue and
    /// reduction text are the record's. `recordMetadata` (with `tiled`): only the codes come from the
    /// tiles; inputs `s<l>` / `b<l>` and every scale and offset read stay the record's text.
    static func w4Source(
        layers: Int, group: Int, form: Int = 0, narrow: Bool = false, tiled: Bool = false,
        recordMetadata: Bool = false
    ) -> String {
        precondition(!narrow || form == 1 || form == 2)
        let ru = group / 32  // packed uint4 per row and group
        let tw = group / 16  // unpacked uint4 per row and group
        let cols = narrow ? 16 : 32  // output columns per threadgroup
        let pr = narrow ? ru / 2 : ru  // packed uint4 rows each lane stages per group
        let ne = narrow ? 8 : 16  // live destination elements per lane
        let tileMetadata = tiled && !recordMetadata
        var pick = tileMetadata
            ? """
            const int K = ksz[0];
            int bx = int(threadgroup_position_in_grid.x);
            const device uint32_t* w = w0;
            device float* out = out0; int N = ksz[2];

            """
            : """
            const int K = ksz[0];
            int bx = int(threadgroup_position_in_grid.x);
            const device uint32_t* w = w0; const device half* sT = s0; const device half* bT = b0;
            device float* out = out0; int N = ksz[2];

            """
        for l in 1 ..< layers {
            pick += tileMetadata
                ? "if (bx >= N / \(cols)) { bx -= N / \(cols); w = w\(l); out = out\(l); N = ksz[\(2 + l)]; }\n"
                : "if (bx >= N / \(cols)) { bx -= N / \(cols); w = w\(l); sT = s\(l); bT = b\(l); out = out\(l); N = ksz[\(2 + l)]; }\n"
        }
        let operandB =
            """
            threadgroup uint4 tiles[SPLIT * 32 * \(tw)];
            threadgroup uint4* bs = tiles + sg * (32 * \(tw));
            tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> Bt((threadgroup int8_t*)bs, dextents<int, 2>(\(group), 32));
            auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(Bt)>, int32_t>();

            """
        let tileU4 = w4TileU4(group: group)
        let codeSetup = tiled
            ? """
            const int lr = int(lane) / \(ru), lc = int(lane) % \(ru);
            const device uint4* tb = (const device uint4*)w + (size_t)(n0 >> 5) * (size_t)G * \(tileU4);
            const device uint4* wsrc = tb + \(narrow ? "(n0 & 16) * \(ru) + " : "")int(lane);
            \(tileMetadata ? "const device half* tsc = (const device half*)(tb + \(32 * ru)) + (n0 & 31) + nb;" : "")
            uint4 wv[\(pr)];
            for (int p = 0; p < \(pr); p++) { wv[p] = wsrc[(size_t)sg * \(tileU4) + p * 32]; }

            """
            : """
            const int lr = int(lane) / \(ru), lc = int(lane) % \(ru);
            const size_t rowU4 = (size_t)K / 32;
            const device uint4* wsrc = (const device uint4*)w + (size_t)(n0 + lr) * rowU4 + lc;
            uint4 wv[\(pr)];
            for (int p = 0; p < \(pr); p++) { wv[p] = wsrc[(size_t)(p * \(32 / ru)) * rowU4 + int(sg) * \(ru)]; }

            """
        let stage =
            """
              for (int p = 0; p < \(pr); p++) {
                const int r = lr + p * \(32 / ru);
                const uint4 x = wv[p];
                bs[r * \(tw) + 2 * lc] = uint4(LO(x.x), HI(x.x), LO(x.y), HI(x.y));
                bs[r * \(tw) + 2 * lc + 1] = uint4(LO(x.z), HI(x.z), LO(x.w), HI(x.w));
              }

            """
        let prefetchCodes = tiled
            ? "for (int p = 0; p < \(pr); p++) { wv[p] = wsrc[(size_t)(g + SPLIT) * \(tileU4) + p * 32]; }"
            : "for (int p = 0; p < \(pr); p++) { wv[p] = wsrc[(size_t)(p * \(32 / ru)) * rowU4 + (g + SPLIT) * \(ru)]; }"
        let pickS = narrow ? "s0v[j]" : "(i >> 3) ? s1v[j] : s0v[j]"
        let pickB = narrow ? "b0v[j]" : "(i >> 3) ? b1v[j] : b0v[j]"
        let epilogue = """
                for (int i = 0; i < \(ne); i++) {
                  const int j = (i & 3);
                  const float s = \(pickS);
                  const float b = \(pickB);
                  const bool hi = (i >> 2) & 1;
                  acc[i] = fma(s * (hi ? a1 : a0), float(cT[i]), acc[i]);
                  accb[i] = fma(b, hi ? x1 : x0, accb[i]);
                }

            """
        let biasEpilogue = """
                for (int i = 0; i < \(ne); i++) {
                  const int j = (i & 3);
                  const float b = \(pickB);
                  const bool hi = (i >> 2) & 1;
                  accb[i] = fma(b, hi ? x1 : x0, accb[i]);
                }

            """
        let dotEpilogue = """
                for (int i = 0; i < \(ne); i++) {
                  const int j = (i & 3);
                  const float s = \(pickS);
                  const bool hi = (i >> 2) & 1;
                  acc[i] = fma(s * (hi ? a1 : a0), float(cT[i]), acc[i]);
                }

            """
        var loop: String
        if form == 0 {
            loop = """
                for (int g = int(sg); g < G; g += SPLIT) {
                \(stage)  if (g + SPLIT < G) { \(prefetchCodes) }
                  simdgroup_barrier(mem_flags::mem_threadgroup);
                  auto tA = A.template slice<\(group), 16>(g * \(group), 0);
                  op.run(tA, Bt, cT);
                  const device half* sr = sT + (size_t)g * N + n0 + nb;
                  const device half* br = bT + (size_t)g * N + n0 + nb;
                  const float4 s0v = float4(*(const device half4*)sr), s1v = float4(*(const device half4*)(sr + 16));
                  const float4 b0v = fma(float4(8.0f), s0v, float4(*(const device half4*)br));
                  const float4 b1v = fma(float4(8.0f), s1v, float4(*(const device half4*)(br + 16)));
                  const float a0 = ascale[m0 * G + g], a1 = ascale[m1 * G + g];
                  const float x0 = xs[m0 * G + g], x1 = xs[m1 * G + g];
                \(epilogue)  simdgroup_barrier(mem_flags::mem_threadgroup);
                }

                """
        } else if form == 2 && narrow {
            let barrier = "  simdgroup_barrier(mem_flags::mem_threadgroup);\n"
            loop = """
                const device half* sc = sT + n0 + nb;
                const device half* oc = bT + n0 + nb;
                half4 ps0 = *(const device half4*)(sc + (size_t)sg * N);
                half4 po0 = *(const device half4*)(oc + (size_t)sg * N);
                float pa0 = ascale[m0 * G + int(sg)], pa1 = ascale[m1 * G + int(sg)];
                float px0 = xs[m0 * G + int(sg)], px1 = xs[m1 * G + int(sg)];
                for (int g = int(sg); g < G; g += SPLIT) {
                \(stage)  const half4 cs0 = ps0;
                  const float a0 = pa0, a1 = pa1;
                  {
                    const half4 co0 = po0;
                    const float x0 = px0, x1 = px1;
                    const float4 s0v = float4(cs0);
                    const float4 b0v = fma(float4(8.0f), s0v, float4(co0));
                \(biasEpilogue)  }
                  if (g + SPLIT < G) {
                    \(prefetchCodes)
                    const size_t r = (size_t)(g + SPLIT) * N;
                    ps0 = *(const device half4*)(sc + r);
                    po0 = *(const device half4*)(oc + r);
                    pa0 = ascale[m0 * G + g + SPLIT]; pa1 = ascale[m1 * G + g + SPLIT];
                    px0 = xs[m0 * G + g + SPLIT]; px1 = xs[m1 * G + g + SPLIT];
                  }
                \(barrier)  auto tA = A.template slice<\(group), 16>(g * \(group), 0);
                  op.run(tA, Bt, cT);
                  const float4 s0v = float4(cs0);
                \(dotEpilogue)\(barrier)}

                """
        } else if form == 2 {
            let barrier = "  simdgroup_barrier(mem_flags::mem_threadgroup);\n"
            loop = """
                const device half* sc = sT + n0 + nb;
                const device half* oc = bT + n0 + nb;
                half4 ps0 = *(const device half4*)(sc + (size_t)sg * N), ps1 = *(const device half4*)(sc + (size_t)sg * N + 16);
                half4 po0 = *(const device half4*)(oc + (size_t)sg * N), po1 = *(const device half4*)(oc + (size_t)sg * N + 16);
                float pa0 = ascale[m0 * G + int(sg)], pa1 = ascale[m1 * G + int(sg)];
                float px0 = xs[m0 * G + int(sg)], px1 = xs[m1 * G + int(sg)];
                for (int g = int(sg); g < G; g += SPLIT) {
                \(stage)  const half4 cs0 = ps0, cs1 = ps1;
                  const float a0 = pa0, a1 = pa1;
                  {
                    const half4 co0 = po0, co1 = po1;
                    const float x0 = px0, x1 = px1;
                    const float4 s0v = float4(cs0), s1v = float4(cs1);
                    const float4 b0v = fma(float4(8.0f), s0v, float4(co0));
                    const float4 b1v = fma(float4(8.0f), s1v, float4(co1));
                \(biasEpilogue)  }
                  if (g + SPLIT < G) {
                    \(prefetchCodes)
                    const size_t r = (size_t)(g + SPLIT) * N;
                    ps0 = *(const device half4*)(sc + r); ps1 = *(const device half4*)(sc + r + 16);
                    po0 = *(const device half4*)(oc + r); po1 = *(const device half4*)(oc + r + 16);
                    pa0 = ascale[m0 * G + g + SPLIT]; pa1 = ascale[m1 * G + g + SPLIT];
                    px0 = xs[m0 * G + g + SPLIT]; px1 = xs[m1 * G + g + SPLIT];
                  }
                \(barrier)  auto tA = A.template slice<\(group), 16>(g * \(group), 0);
                  op.run(tA, Bt, cT);
                  const float4 s0v = float4(cs0), s1v = float4(cs1);
                \(dotEpilogue)\(barrier)}

                """
        } else if narrow {
            let barrier = "  simdgroup_barrier(mem_flags::mem_threadgroup);\n"
            loop = """
                const device half* sc = sT + n0 + nb;
                const device half* oc = bT + n0 + nb;
                half4 ps0 = *(const device half4*)(sc + (size_t)sg * N);
                half4 po0 = *(const device half4*)(oc + (size_t)sg * N);
                float pa0 = ascale[m0 * G + int(sg)], pa1 = ascale[m1 * G + int(sg)];
                float px0 = xs[m0 * G + int(sg)], px1 = xs[m1 * G + int(sg)];
                for (int g = int(sg); g < G; g += SPLIT) {
                \(stage)  const half4 cs0 = ps0, co0 = po0;
                  const float a0 = pa0, a1 = pa1, x0 = px0, x1 = px1;
                  if (g + SPLIT < G) {
                    \(prefetchCodes)
                    const size_t r = (size_t)(g + SPLIT) * N;
                    ps0 = *(const device half4*)(sc + r);
                    po0 = *(const device half4*)(oc + r);
                    pa0 = ascale[m0 * G + g + SPLIT]; pa1 = ascale[m1 * G + g + SPLIT];
                    px0 = xs[m0 * G + g + SPLIT]; px1 = xs[m1 * G + g + SPLIT];
                  }
                \(barrier)  auto tA = A.template slice<\(group), 16>(g * \(group), 0);
                  op.run(tA, Bt, cT);
                  const float4 s0v = float4(cs0);
                  const float4 b0v = fma(float4(8.0f), s0v, float4(co0));
                \(epilogue)\(barrier)}

                """
        } else {
            let barrier = "  simdgroup_barrier(mem_flags::mem_threadgroup);\n"
            loop = """
                const device half* sc = sT + n0 + nb;
                const device half* oc = bT + n0 + nb;
                half4 ps0 = *(const device half4*)(sc + (size_t)sg * N), ps1 = *(const device half4*)(sc + (size_t)sg * N + 16);
                half4 po0 = *(const device half4*)(oc + (size_t)sg * N), po1 = *(const device half4*)(oc + (size_t)sg * N + 16);
                float pa0 = ascale[m0 * G + int(sg)], pa1 = ascale[m1 * G + int(sg)];
                float px0 = xs[m0 * G + int(sg)], px1 = xs[m1 * G + int(sg)];
                for (int g = int(sg); g < G; g += SPLIT) {
                \(stage)  const half4 cs0 = ps0, cs1 = ps1, co0 = po0, co1 = po1;
                  const float a0 = pa0, a1 = pa1, x0 = px0, x1 = px1;
                  if (g + SPLIT < G) {
                    \(prefetchCodes)
                    const size_t r = (size_t)(g + SPLIT) * N;
                    ps0 = *(const device half4*)(sc + r); ps1 = *(const device half4*)(sc + r + 16);
                    po0 = *(const device half4*)(oc + r); po1 = *(const device half4*)(oc + r + 16);
                    pa0 = ascale[m0 * G + g + SPLIT]; pa1 = ascale[m1 * G + g + SPLIT];
                    px0 = xs[m0 * G + g + SPLIT]; px1 = xs[m1 * G + g + SPLIT];
                  }
                \(barrier)  auto tA = A.template slice<\(group), 16>(g * \(group), 0);
                  op.run(tA, Bt, cT);
                  const float4 s0v = float4(cs0), s1v = float4(cs1);
                  const float4 b0v = fma(float4(8.0f), s0v, float4(co0));
                  const float4 b1v = fma(float4(8.0f), s1v, float4(co1));
                \(epilogue)\(barrier)}

                """
        }
        if tileMetadata {
            // Scale and offset rows of group g: the tile's tail, one tile (8 * tileU4 halves) per group.
            let tileH = 8 * tileU4
            for (from, to) in [
                ("sT + (size_t)g * N + n0 + nb", "tsc + (size_t)g * \(tileH)"),
                ("bT + (size_t)g * N + n0 + nb", "tsc + 32 + (size_t)g * \(tileH)"),
                ("sT + n0 + nb", "tsc"),
                ("bT + n0 + nb", "tsc + 32"),
                ("(size_t)sg * N", "(size_t)sg * \(tileH)"),
                ("(size_t)(g + SPLIT) * N", "(size_t)(g + SPLIT) * \(tileH)"),
            ] {
                loop = loop.replacingOccurrences(of: from, with: to)
            }
            precondition(!loop.contains("sT") && !loop.contains("bT") && !loop.contains("* N"))
        }
        if tiled && signedNibbles {
            loop = loop.replacingOccurrences(of: "ascale[", with: "0.0625f * ascale[")
            precondition(loop.components(separatedBy: "0.0625f * ascale[").count == (form == 0 ? 3 : 5))
        }
        let unpack = (tiled && signedNibbles)
            ? """
            #define LO(v) (((v) << 4) & 0xF0F0F0F0u)
            #define HI(v) ((v) & 0xF0F0F0F0u)
            """
            : """
            #define LO(v) ((((v) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
            #define HI(v) (((((v) >> 4) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
            """
        let reduce =
            """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float* red = (threadgroup float*)tiles;

            """
        return pick + unpack + """

            const int G = K / \(group);
            const int n0 = bx * \(cols);
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint tid = thread_position_in_threadgroup.x;
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, \(group), false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
            tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
            auto tA0 = A.template slice<\(group), 16>(0, 0);

            """ + operandB + """
            const int mrow = int((lane >> 1) & 3u) + 4 * int((lane >> 4) & 1u);
            const int nb = 4 * int(lane & 1u) + 8 * int((lane >> 3) & 1u);
            bool ok = cT.get_capacity() == 16;
            for (int i = 0; i < 16 && ok; i++) {
              auto ids = cT.get_multidimensional_index(i);
              ok = ids[0] == nb + (i & 3) + 16 * (i >> 3) && ids[1] == mrow + 8 * ((i >> 2) & 1);
            }
            float acc[\(ne)], accb[\(ne)];
            for (int i = 0; i < \(ne); i++) { acc[i] = 0.0f; accb[i] = 0.0f; }
            const int m0 = mrow, m1 = mrow + 8;

            """ + codeSetup + loop + reduce + """
            for (int i = 0; i < \(ne); i++) {
              const int m = mrow + 8 * ((i >> 2) & 1), n = nb + (i & 3) + 16 * (i >> 3);
              red[sg * \(16 * cols) + m * \(cols) + n] = acc[i] + accb[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = int(tid); e < ksz[1] * \(cols); e += SPLIT * 32) {
              float v = 0.0f;
              for (int q = 0; q < SPLIT; q++) { v += red[q * \(16 * cols) + e]; }
              out[(size_t)(e >> \(narrow ? 4 : 5)) * N + n0 + (e & \(cols - 1))] = ok ? v : as_type<float>(0x7fc00000u);
            }
            #undef LO
            #undef HI
            """
    }

    /// One kernel set per form (index = form), for 1 ... 6 sibling layers; form 0 keeps the record names.
    static let w4KernelSets: [[MLXFast.MLXFastKernel]] = (0 ... 2).map { form in
        (1 ... 6).map { layers in
            let group = w4Group ?? 64
            return MLXFast.metalKernel(
                name: form == 0 ? "df2_w4g\(group)_\(layers)" : "df2_w4g\(group)f\(form)_\(layers)",
                inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                    + (0 ..< layers).map { "b\($0)" } + ["ascale", "xs", "ksz"],
                outputNames: (0 ..< layers).map { "out\($0)" },
                source: w4Source(layers: layers, group: group, form: form), header: header,
                ensureRowContiguous: true)
        }
    }

    static func w4Run(
        form: Int, narrow: Bool = false, tiled: Bool = false, xq: MLXArray, ascale: MLXArray, xs: MLXArray,
        rows: Int, K: Int, ws: [MLXArray], sTs: [MLXArray], bTs: [MLXArray], ns: [Int],
        sizeDescriptor: MLXArray? = nil
    ) -> [MLXArray] {
        let sizes = sizeDescriptor ?? HostConstants.int32([Int32(K), Int32(rows)] + ns.map { Int32($0) })
        let formsFit = w4FormsFit(K: K, rows: rows, ns: ns)
        if tiled {
            return w4TiledKernel(form: form, narrow: narrow, specialized: formsFit, layers: ws.count)(
                [xq] + ws + [ascale, xs, sizes],
                template: formsFit ? [("SPLIT", split), ("KC", K)] : [("SPLIT", split)],
                grid: (ns.reduce(0, +) / (narrow ? 16 : 32) * split * 32, 1, 1),
                threadGroup: (split * 32, 1, 1),
                outputShapes: ns.map { [rows, $0] }, outputDTypes: ns.map { _ in .float32 })
        }
        let kernels =
            narrow
            ? (form == 2
                ? (formsFit ? w4Form2NarrowFormsKernels : w4Form2NarrowKernels)
                : (formsFit ? w4NarrowFormsKernels : w4NarrowKernels))
            : (formsFit ? w4FormsKernelSets : w4KernelSets)[form]
        return kernels[ws.count - 1](
            [xq] + ws + sTs + bTs + [ascale, xs, sizes],
            template: formsFit ? [("SPLIT", split), ("KC", K)] : [("SPLIT", split)],
            grid: (ns.reduce(0, +) / (narrow ? 16 : 32) * split * 32, 1, 1),
            threadGroup: (split * 32, 1, 1),
            outputShapes: ns.map { [rows, $0] }, outputDTypes: ns.map { _ in .float32 })
    }

    nonisolated(unsafe) private static var w4TiledKernels: [String: MLXFast.MLXFastKernel] = [:]

    /// The tiled twin of the w4 kernel `w4Run` picks for (form, narrow, specialized, layers): the same
    /// source with the code, scale and offset reads taken from `w4Tile` tiles (`w4Source(tiled: true)`).
    static func w4TiledKernel(form: Int, narrow: Bool, specialized: Bool, layers: Int) -> MLXFast.MLXFastKernel {
        let group = w4Group ?? 64
        let name = "df2_w4tg\(group)f\(form)\(narrow ? "n" : "")\(specialized ? "_forms" : "")_\(layers)"
        if let k = w4TiledKernels[name] { return k }
        let raw = w4Source(layers: layers, group: group, form: form, narrow: narrow, tiled: true)
        let k = MLXFast.metalKernel(
            name: name,
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + ["ascale", "xs", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: specialized ? w4Specialize(raw) : raw, header: header,
            ensureRowContiguous: true)
        w4TiledKernels[name] = k
        return k
    }

    /// A tiled twin built on first use and kept by name (the pair24 kernel families).
    static func w4TiledVariant(_ name: String, _ make: (String) -> MLXFast.MLXFastKernel) -> MLXFast.MLXFastKernel {
        if let k = w4TiledKernels[name] { return k }
        let k = make(name)
        w4TiledKernels[name] = k
        return k
    }

    /// True when these operands read tiles (adoption is all or nothing, so never a mix).
    static func w4Tiled(_ ops: [W4Operands]) -> Bool {
        let tiled = !ops.isEmpty && ops.allSatisfy { $0.tiles != nil }
        precondition(tiled || ops.allSatisfy { $0.tiles == nil })
        return tiled
    }

    /// The code input of each operand: its tiles once adopted, else the record codes.
    static func w4CodeArrays(_ ops: [W4Operands]) -> [MLXArray] {
        w4Tiled(ops) ? ops.map { $0.tiles! } : ops.map(\.w)
    }

    nonisolated(unsafe) private static var w4CheckFailed = false
    nonisolated(unsafe) private static var w4CheckNote = ""

    /// The w4 kernel form, decided once, at the first 4-bit matmul. `DF2_W4_FORM=0` keeps the
    /// record kernel. Otherwise form 1 is used only when it compiles and its output bits equal the
    /// record kernel's on a fixed, input-independent problem at each drafter K (4096, 12288, 32768),
    /// 15 and 16 rows, and 1, 2, 3 and 6 sibling layers of 64 columns each; a compile error or any
    /// differing bit keeps the record kernel. Each problem runs the kernel variant the engine runs for
    /// its shape (the specialized set at K 4096 and 12288, the general set at 32768).
    static let w4Form: Int = {
        let value = ProcessInfo.processInfo.environment["DF2_W4_FORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let group = w4Group ?? 64
        let chosen: Int
        let note: String
        if ["0", "false", "no", "off"].contains(value ?? "") {
            chosen = 0
            note = "off (DF2_W4_FORM=0)"
        } else {
            var finite = true
            let failure = w4Check(form: 1, group: group, finite: &finite)
            chosen = failure == nil ? 1 : 0
            note = failure.map { "record kernel kept (\($0))" }
                ?? "bit-identical to the record kernel at K 4096, 12288, 32768; rows 15, 16; layers 1, 2, 3, 6"
                    + (finite ? "" : " (record output not finite on this device)")
        }
        FileHandle.standardError.write(
            Data("DF2 w4 matmul form \(chosen): \(note)\n".utf8))
        return chosen
    }()

    /// Form 2 keeps the admitted form 1 as its first fallback.
    static let w4SelectedForm: Int = {
        let accepted = w4Form
        guard accepted == 1 else { return accepted }
        let value = ProcessInfo.processInfo.environment["DF2_W4_FORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if value == "1" { return accepted }
        var finite = true
        let failure = w4Check(form: 2, group: w4Group ?? 64, finite: &finite)
        let chosen = failure == nil ? 2 : accepted
        let note = failure.map { "accepted form 1 kept (\($0))" }
            ?? "finite and bit-identical to accepted form 1 on fixed checks"
        FileHandle.standardError.write(Data("DF2 w4 current-bias form \(chosen): \(note)\n".utf8))
        return chosen
    }()

    /// Nil when `form` matches the record kernel bit for bit on every check problem, else why not.
    static func w4Check(form: Int, group: Int, finite: inout Bool) -> String? {
        let n = 64
        var failure: String? = nil
        w4CheckFailed = false
        w4CheckNote = ""
        withErrorHandler({ message in
            if !DF2Fast.w4CheckFailed { DF2Fast.w4CheckNote = String(message.prefix(160)) }
            DF2Fast.w4CheckFailed = true
        }) {
            for k in [4096, 12288, 32768] where k % group == 0 && k / group >= split {
                let g = k / group
                let w = MLXArray(
                    (0 ..< n * k / 8).map { UInt32(truncatingIfNeeded: UInt64($0 + 1) &* 2_654_435_761) },
                    [n, k / 8])
                let scaleValues: [Float16] = (0 ..< g * n).map { (i: Int) -> Float16 in
                    let numerator: Float = Float(1 + i % 13)
                    let value: Float = numerator / 1024
                    return Float16(value)
                }
                let sT = MLXArray(scaleValues, [g, n])
                let bT = MLXArray((0 ..< g * n).map { Float16(-Float($0 % 7) / 512) }, [g, n])
                let xq = MLXArray(
                    (0 ..< 16 * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [16 * k])
                let activationScales: [Float] = (0 ..< 16 * g).map { (i: Int) -> Float in
                    let numerator: Float = Float(1 + i % 11)
                    return numerator / 256
                }
                let ascale = MLXArray(activationScales, [16 * g])
                let activationSums: [Float] = (0 ..< 16 * g).map { (i: Int) -> Float in
                    let numerator: Float = Float(i % 17 - 8)
                    return numerator / 64
                }
                let xs = MLXArray(activationSums, [16 * g])
                for rows in [16, 15] {
                    for layers in [1, 2, 3, 6] {
                        func bits(_ f: Int) -> MLXArray {
                            let outs = w4Run(
                                form: f, xq: xq, ascale: ascale, xs: xs, rows: rows, K: k,
                                ws: Array(repeating: w, count: layers),
                                sTs: Array(repeating: sT, count: layers),
                                bTs: Array(repeating: bT, count: layers),
                                ns: Array(repeating: n, count: layers))
                            return concatenated(outs, axis: 1).view(dtype: .uint32)
                        }
                        let record = bits(form == 2 ? 1 : 0)
                        let candidate = bits(form)
                        if form == 2 {
                            let recordFinite = isFinite(record.view(dtype: .float32)).all()
                            let candidateFinite = isFinite(candidate.view(dtype: .float32)).all()
                            eval(recordFinite, candidateFinite)
                            if w4CheckFailed {
                                failure = "compile or run error: \(w4CheckNote)"
                                return
                            }
                            if !recordFinite.item(Bool.self) || !candidateFinite.item(Bool.self) {
                                finite = false
                                failure = "non-finite control or candidate at K \(k), rows \(rows), layers \(layers)"
                                return
                            }
                        }
                        let differ = (record .!= candidate).asType(.int32).sum()
                        let nonFinite = (record .== MLXArray(UInt32(0x7fc0_0000))).asType(.int32).sum()
                        eval(differ, nonFinite)
                        if w4CheckFailed {
                            failure = "compile or run error: \(w4CheckNote)"
                            return
                        }
                        let d = differ.item(Int32.self)
                        if d != 0 {
                            failure = "\(d) words differ at K \(k), rows \(rows), layers \(layers)"
                            return
                        }
                        if nonFinite.item(Int32.self) != 0 { finite = false }
                    }
                }
            }
        }
        if failure == nil && w4CheckFailed { failure = "compile or run error: \(w4CheckNote)" }
        return failure
    }

    /// The eligible W4 matrices fit in 32-bit element offsets. Specialize their
    /// input stride while leaving tensor products and all FP32 reductions unchanged.
    /// One set per w4 kernel form (index = form); form 0 keeps the names above.
    static let w4FormsKernelSets: [[MLXFast.MLXFastKernel]] = (0 ... 2).map { form in
        (1 ... 6).map { layers in
            let group = w4Group ?? 64
            let source = w4Specialize(w4Source(layers: layers, group: group, form: form))
            return MLXFast.metalKernel(
                name: form == 0 ? "df2_w4g\(group)_forms_\(layers)" : "df2_w4g\(group)f\(form)_forms_\(layers)",
                inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                    + (0 ..< layers).map { "b\($0)" } + ["ascale", "xs", "ksz"],
                outputNames: (0 ..< layers).map { "out\($0)" },
                source: source, header: header, ensureRowContiguous: true)
        }
    }

    /// The compile-time K and 32-bit offset text replacements of the specialized kernels.
    static func w4Specialize(_ source: String) -> String {
        source
            .replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
            .replacingOccurrences(of: "const size_t rowU4", with: "const uint rowU4")
            .replacingOccurrences(of: "(size_t)", with: "(uint)")
    }

    /// Form 1 at 16 output columns per threadgroup (`w4Source(narrow: true)`), for 1 ... 6 sibling
    /// layers: the general set, and the specialized set `w4FormsFit` picks, as for the wide kernels.
    static let w4NarrowKernels: [MLXFast.MLXFastKernel] = (1 ... 6).map { layers in
        w4NarrowKernel(layers: layers, specialized: false)
    }
    static let w4NarrowFormsKernels: [MLXFast.MLXFastKernel] = (1 ... 6).map { layers in
        w4NarrowKernel(layers: layers, specialized: true)
    }
    static let w4Form2NarrowKernels: [MLXFast.MLXFastKernel] = (1 ... 6).map { layers in
        w4NarrowKernel(layers: layers, specialized: false, form: 2)
    }
    static let w4Form2NarrowFormsKernels: [MLXFast.MLXFastKernel] = (1 ... 6).map { layers in
        w4NarrowKernel(layers: layers, specialized: true, form: 2)
    }
    static func w4NarrowKernel(layers: Int, specialized: Bool, form: Int = 1) -> MLXFast.MLXFastKernel {
        let group = w4Group ?? 64
        let source = w4Source(layers: layers, group: group, form: form, narrow: true)
        return MLXFast.metalKernel(
            name: "df2_w4g\(group)f\(form)n\(specialized ? "_forms" : "")_\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "xs", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: specialized ? w4Specialize(source) : source, header: header,
            ensureRowContiguous: true)
    }

    /// The widest call (output columns summed over its sibling layers) that takes the narrow
    /// kernels: 4096 columns launch 256 threadgroups instead of 128.
    static let w4NarrowColumns = 4096

    /// The drafter's own narrow-eligible shapes: (K, output columns per sibling layer).
    static let w4NarrowShapes: [(k: Int, ns: [Int])] = [
        (4096, [4096]), (12288, [4096]), (32768, [4096]), (4096, [1024]), (4096, [256]),
        (4096, [1024, 1024]), (4096, [64, 64, 64]), (12288, [64, 64, 64, 64, 64, 64]),
    ]

    /// Whether narrow calls take the narrow kernels, decided once, at the first narrow-eligible
    /// 4-bit matmul. `DF2_W4_NARROW=0` keeps the wide kernels. Otherwise the narrow kernels are used
    /// only when form 1 is in use and their output bits equal the record kernel's (form 0, 32
    /// columns) on a fixed, input-independent problem at each of `w4NarrowShapes`, 15 and 16 rows;
    /// a compile error or any differing bit keeps the wide kernels.
    static let w4Narrow: Bool = {
        let value = ProcessInfo.processInfo.environment["DF2_W4_NARROW"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let chosen: Bool
        let note: String
        if ["0", "false", "no", "off"].contains(value ?? "") {
            chosen = false
            note = "off (DF2_W4_NARROW=0)"
        } else if w4Form != 1 {
            chosen = false
            note = "off (w4 matmul form \(w4Form))"
        } else {
            var finite = true
            let failure = w4NarrowCheck(group: w4Group ?? 64, finite: &finite)
            chosen = failure == nil
            note = failure.map { "wide kernels kept (\($0))" }
                ?? "16 columns per threadgroup for calls up to \(w4NarrowColumns) columns; bit-identical to the record kernel at "
                    + w4NarrowShapes.map { "K \($0.k) N \($0.ns.map(String.init).joined(separator: "+"))" }
                    .joined(separator: ", ") + "; rows 15, 16"
                    + (finite ? "" : " (record output not finite on this device)")
        }
        FileHandle.standardError.write(Data("DF2 w4 matmul narrow: \(note)\n".utf8))
        return chosen
    }()

    /// Narrow form 2 must pass after accepted narrow form 1 has passed.
    static let w4Form2Narrow: Bool = {
        guard w4SelectedForm == 2, w4Narrow else { return false }
        var finite = true
        let failure = w4NarrowCheck(group: w4Group ?? 64, form: 2, finite: &finite)
        let chosen = failure == nil
        let note = failure.map { "accepted narrow form 1 kept (\($0))" }
            ?? "finite and bit-identical to accepted narrow form 1 on fixed checks"
        FileHandle.standardError.write(Data("DF2 w4 current-bias narrow: \(note)\n".utf8))
        return chosen
    }()

    /// Nil when the narrow kernels match the record kernel bit for bit at every narrow shape.
    static func w4NarrowCheck(group: Int, form: Int = 1, finite: inout Bool) -> String? {
        var failure: String? = nil
        w4CheckFailed = false
        w4CheckNote = ""
        withErrorHandler({ message in
            if !DF2Fast.w4CheckFailed { DF2Fast.w4CheckNote = String(message.prefix(160)) }
            DF2Fast.w4CheckFailed = true
        }) {
            for (k, ns) in w4NarrowShapes where k % group == 0 && k / group >= split {
                let g = k / group
                let xq = MLXArray(
                    (0 ..< 16 * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [16 * k])
                let activationScales: [Float] = (0 ..< 16 * g).map { (i: Int) -> Float in
                    let numerator: Float = Float(1 + i % 11)
                    return numerator / 256
                }
                let ascale = MLXArray(activationScales, [16 * g])
                let activationSums: [Float] = (0 ..< 16 * g).map { (i: Int) -> Float in
                    let numerator: Float = Float(i % 17 - 8)
                    return numerator / 64
                }
                let xs = MLXArray(activationSums, [16 * g])
                var ws: [MLXArray] = [], sTs: [MLXArray] = [], bTs: [MLXArray] = []
                for (l, n) in ns.enumerated() {
                    let i = arange(n * k / 8, dtype: .uint32) + MLXArray(UInt32(l * 7919 + 1))
                    let h = i * MLXArray(UInt32(2_654_435_761))
                    ws.append((h ^ (h >> MLXArray(UInt32(15)))).reshaped([n, k / 8]))
                    let j = arange(g * n, dtype: .int32) + MLXArray(Int32(l))
                    sTs.append(
                        ((j % MLXArray(Int32(13))).asType(.float32) + 1) / 1024)
                    bTs.append(-(j % MLXArray(Int32(7))).asType(.float32) / 512)
                    sTs[l] = sTs[l].asType(.float16).reshaped([g, n])
                    bTs[l] = bTs[l].asType(.float16).reshaped([g, n])
                }
                eval(ws + sTs + bTs)
                for rows in [16, 15] {
                    let record = concatenated(
                        w4Run(
                            form: form == 2 ? 1 : 0, narrow: form == 2,
                            xq: xq, ascale: ascale, xs: xs, rows: rows, K: k, ws: ws, sTs: sTs,
                            bTs: bTs, ns: ns), axis: 1
                    ).view(dtype: .uint32)
                    let candidate = concatenated(
                        w4Run(
                            form: form, narrow: true, xq: xq, ascale: ascale, xs: xs, rows: rows, K: k,
                            ws: ws, sTs: sTs, bTs: bTs, ns: ns), axis: 1
                    ).view(dtype: .uint32)
                    if form == 2 {
                        let recordFinite = isFinite(record.view(dtype: .float32)).all()
                        let candidateFinite = isFinite(candidate.view(dtype: .float32)).all()
                        eval(recordFinite, candidateFinite)
                        if w4CheckFailed {
                            failure = "compile or run error: \(w4CheckNote)"
                            return
                        }
                        if !recordFinite.item(Bool.self) || !candidateFinite.item(Bool.self) {
                            finite = false
                            failure = "non-finite control or candidate at K \(k), rows \(rows), N \(ns)"
                            return
                        }
                    }
                    let differ = (record .!= candidate).asType(.int32).sum()
                    let nonFinite = (record .== MLXArray(UInt32(0x7fc0_0000))).asType(.int32).sum()
                    eval(differ, nonFinite)
                    if w4CheckFailed {
                        failure = "compile or run error: \(w4CheckNote)"
                        return
                    }
                    let d = differ.item(Int32.self)
                    if d != 0 {
                        failure = "\(d) words differ at K \(k), N \(ns), rows \(rows)"
                        return
                    }
                    if nonFinite.item(Int32.self) != 0 { finite = false }
                }
            }
        }
        if failure == nil && w4CheckFailed { failure = "compile or run error: \(w4CheckNote)" }
        return failure
    }

    /// The matrices the specialized kernels above take: K 4096 or 12288, 1 ... 16 rows, and every
    /// N * K below 2^32.
    static func w4FormsFit(K: Int, rows: Int, ns: [Int]) -> Bool {
        (K == 4096 || K == 12288) && (1 ... 16).contains(rows)
            && ns.allSatisfy { $0 > 0 && UInt64($0) * UInt64(K) < UInt64(UInt32.max) }
    }

    static func w4Quant(
        _ x: MLXArray, up: MLXArray?, group: Int, headMajor: Bool, rowsOverride: Int? = nil,
        kOverride: Int? = nil
    ) -> W4Codes? {
        let K = kOverride ?? x.dim(-1)
        let rows = rowsOverride ?? (x.size / K)
        guard (1 ... 16).contains(rows), (group == 64 || group == 128), K % group == 0,
            K % 16 == 0, split > 0, K / group >= split
        else {
            return nil
        }
        let x2 = up ?? x
        let wide = DF2WideLanes.serves(
            .init(kind: .quantG, dtype: x.dtype, dtype2: x2.dtype, k: K, group: group,
                flag: (up == nil ? 0 : 1) + (headMajor ? 2 : 0)))
        let q = quantGRun(x, x2, K: K, rows: rows, group: group, headMajor: headMajor, swiglu: up != nil, wide: wide)
        return W4Codes(xq: q[0], ascale: q[1], xs: q[2], rows: rows, K: K, group: group)
    }

    /// `df2_act_q8g` at 16 columns per thread, or its `DF2WideLaneSource` twin (`wide`).
    static func quantGRun(
        _ x: MLXArray, _ x2: MLXArray, K: Int, rows: Int, group: Int, headMajor: Bool, swiglu: Bool, wide: Bool
    ) -> [MLXArray] {
        let G = K / group
        let totalThreads = K / (wide ? DF2WideLanes.e : 16)
        let tg = Swift.min(256, totalThreads)
        return (wide ? DF2WideLanes.quantG : quantG)(
            [x, x2, HostConstants.int32([Int32(K), Int32(rows)])],
            template: [("SWIGLU", swiglu ? 1 : 0), ("KG", group), ("HEAD_MAJOR", headMajor ? 1 : 0)],
            grid: (totalThreads, 16, 1), threadGroup: (tg, 1, 1),
            outputShapes: [[16 * K], [16 * G], [16 * G]],
            outputDTypes: [.int8, .float32, .float32])
    }

    /// `df2_act_q8g_conv0` at 16 columns per thread, or its `DF2WideLaneSource` twin (`wide`).
    static func conv0Run(
        _ h: MLXArray, dyn: MLXArray, base: MLXArray, K: Int, rows: Int, group: Int, wide: Bool
    ) -> [MLXArray] {
        let G = K / group
        let totalThreads = K / (wide ? DF2WideLanes.e : 16)
        let tg = Swift.min(256, totalThreads)
        return (wide ? DF2WideLanes.quantGConv0 : quantGConv0)(
            [h, dyn, base, HostConstants.int32([Int32(K), Int32(rows)])],
            template: [("KG", group)],
            grid: (totalThreads, 16, 1), threadGroup: (tg, 1, 1),
            outputShapes: [[16 * K], [16 * G], [16 * G]],
            outputDTypes: [.int8, .float32, .float32])
    }

    static func w4LinearsFromCodes(
        _ codes: W4Codes, leadShape: [Int], _ layers: [Linear]
    ) -> [MLXArray]? {
        guard on, (1 ... 6).contains(layers.count) else { return nil }
        let ops = layers.compactMap { w4Operands($0, group: codes.group) }
        guard ops.count == layers.count, ops.allSatisfy({ $0.k == codes.K }) else { return nil }
        let ns = ops.map(\.n)
        let (form, narrow) = w4LaunchForm(ns)
        if w4Recording {
            let ids = layers.map { ObjectIdentifier($0) }
            if w4LaunchSeen.insert(ids).inserted { w4Launches.append(ids) }
        }
        let sizeDescriptor = ops[0].dispatchSizes(K: codes.K, rows: codes.rows, ns: ns)
        let metadata = w4LaunchMetadata(ops, K: codes.K, rows: codes.rows, group: codes.group)
        let outs: [MLXArray]
        if let metadata {
            let method = w4Pair24Selected(ops, ns: ns, metadata: metadata, form: form, narrow: narrow).method(rows: codes.rows)
            outs = w4Pair24Output(method, form: form, narrow: narrow, xq: codes.xq, ascale: codes.ascale,
                xs: codes.xs, rows: codes.rows, ops: ops, ns: ns, metadata: metadata, sizeDescriptor: sizeDescriptor)
        } else {
            outs = w4Run(
                form: form, narrow: narrow, tiled: w4Tiled(ops), xq: codes.xq, ascale: codes.ascale, xs: codes.xs,
                rows: codes.rows, K: codes.K, ws: w4CodeArrays(ops), sTs: ops.map(\.sT), bTs: ops.map(\.bT),
                ns: ns, sizeDescriptor: sizeDescriptor)
        }
        return zip(outs, ns).map { $0.reshaped(leadShape + [$1]) }
    }

    /// The compact metadata a launch reads (the record's choice), or nil for the plain kernels.
    static func w4LaunchMetadata(_ ops: [W4Operands], K: Int, rows: Int, group: Int) -> [W4Pair24]? {
        if group == 64, ops.count == 1, ops[0].k == 32768, ops[0].n == 4096 {
            return ops[0].pair24.map { [$0] }
        } else if group == 64, w4FormsFit(K: K, rows: rows, ns: ops.map(\.n)) {
            return w4Pair24MLPMetadata(ops)
        }
        return nil
    }

    /// The kernel form and width a launch of these output widths takes (the record's choice).
    static func w4LaunchForm(_ ns: [Int]) -> (form: Int, narrow: Bool) {
        let preferredForm = w4SelectedForm
        let narrow = preferredForm != 0 && ns.reduce(0, +) <= w4NarrowColumns && w4Narrow
        let form = preferredForm == 2 && narrow && !w4Form2Narrow ? 1 : preferredForm
        return (form, narrow)
    }

    /// `layers(x)` (or of silu(x) * up) through the 4-bit copies, rows <= 16; nil when not eligible.
    static func w4Linears(_ x: MLXArray, _ layers: [Linear], up: MLXArray?, group: Int) -> [MLXArray]? {
        guard let codes = w4Quant(x, up: up, group: group, headMajor: false) else { return nil }
        return w4LinearsFromCodes(codes, leadShape: Array(x.shape.dropLast()), layers)
    }

    /// `layers(dynConv(h, dyn, base, part: 0, residual: nil))` with `dynConv(part: 0)` fused into `quantGConv0`.
    static func linearsConv0(
        _ h: MLXArray, dyn: MLXArray, base: MLXArray, _ layers: [Linear]
    ) -> [MLXArray]? {
        guard on, (1 ... 6).contains(layers.count), let group = w4Group, (group == 64 || group == 128)
        else { return nil }
        let K = h.dim(-1)
        let rows = h.size / K
        guard (1 ... 16).contains(rows), K % group == 0, K % 16 == 0, split > 0, K / group >= split
        else { return nil }
        let ops = layers.compactMap { w4Operands($0, group: group) }
        guard ops.count == layers.count, ops.allSatisfy({ $0.k == K }) else { return nil }
        let wide = DF2WideLanes.serves(
            .init(kind: .quantGConv0, dtype: h.dtype, dtype2: dyn.dtype, k: K, group: group, flag: base.dtype == .float32 ? 0 : 1))
        let q = conv0Run(h.reshaped([rows, K]), dyn: dyn, base: base, K: K, rows: rows, group: group, wide: wide)
        let codes = W4Codes(xq: q[0], ascale: q[1], xs: q[2], rows: rows, K: K, group: group)
        return w4LinearsFromCodes(codes, leadShape: Array(h.shape.dropLast()), layers)
    }

    /// `layer(output.transposed(0, 2, 1, 3).reshaped(1, L, 4096))` reading head-major `[1, 32, L, 128]` directly.
    static func linearHeadMajor(_ xHeadMajor: MLXArray, _ layer: Linear, L: Int) -> MLXArray? {
        guard on, let group = w4Group,
            let codes = w4Quant(
                xHeadMajor, up: nil, group: group, headMajor: true, rowsOverride: L,
                kOverride: 4096),
            let outs = w4LinearsFromCodes(codes, leadShape: [1, L], [layer])
        else { return nil }
        return outs[0]
    }

    static func linear(_ x: MLXArray, _ layer: Linear) -> MLXArray {
        linears(x, [layer])?[0] ?? layer(x)
    }

    /// Fused `qNorm` + `kNorm` + `YarnRoPE` (`mscale` + `RoPE`) + `[1, L, H, 128] -> [1, H, L, 128]` transpose
    /// for block `(queries, blockKeys, blockValues)` (`Hq = 32, Hk = 8`).
    static let qkvNormRopeKernel = MLXFast.metalKernel(
        name: "df2_qkv_norm_rope",
        inputNames: ["q", "k", "v", "qw", "kw", "freqs", "params", "fparams"],
        outputNames: ["q_out", "k_out", "v_out"],
        source: """
            const int L = params[0];
            const int offset = params[1];
            const float eps = fparams[0];
            const float mscale = fparams[1];
            const int vecIdx = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
            if (vecIdx >= 48 * L) return;
            const uint lane = thread_index_in_simdgroup;
            const int hTotal = vecIdx / L;
            const int s = vecIdx % L;
            const int d0 = int(lane);
            const int d1 = d0 + 32;
            if (hTotal >= 40) {
              const int h = hTotal - 40;
              const size_t srcBase = ((size_t)s * 8 + h) * 128;
              const size_t dstBase = ((size_t)h * L + s) * 128;
              v_out[dstBase + d0] = float(v[srcBase + d0]);
              v_out[dstBase + d0 + 64] = float(v[srcBase + d0 + 64]);
              v_out[dstBase + d1] = float(v[srcBase + d1]);
              v_out[dstBase + d1 + 64] = float(v[srcBase + d1 + 64]);
              return;
            }
            const bool isK = hTotal >= 32;
            const int h = isK ? (hTotal - 32) : hTotal;
            const int H = isK ? 8 : 32;
            const size_t srcBase = ((size_t)s * H + h) * 128;
            const size_t dstBase = ((size_t)h * L + s) * 128;
            const device float* src = isK ? k : q;
            float acc = 0.0f;
            for (int i = 0; i < 4; i++) {
              const float xi = float(src[srcBase + 4 * int(lane) + i]);
              acc += xi * xi;
            }
            const float inv_mean = metal::precise::rsqrt(simd_sum(acc) / 128u + eps);
            const float x0 = float(src[srcBase + d0]);
            const float y0 = float(src[srcBase + d0 + 64]);
            const float x1 = float(src[srcBase + d1]);
            const float y1 = float(src[srcBase + d1 + 64]);
            const device float* w = isK ? kw : qw;
            const float nx0 = (w[d0] * (x0 * inv_mean)) * mscale;
            const float ny0 = (w[d0 + 64] * (y0 * inv_mean)) * mscale;
            const float nx1 = (w[d1] * (x1 * inv_mean)) * mscale;
            const float ny1 = (w[d1 + 64] * (y1 * inv_mean)) * mscale;
            const float pos = float(offset + s);
            const float th0 = pos * (1.0f / freqs[d0]);
            const float c0 = metal::fast::cos(th0), s0 = metal::fast::sin(th0);
            const float th1 = pos * (1.0f / freqs[d1]);
            const float c1 = metal::fast::cos(th1), s1 = metal::fast::sin(th1);
            device float* dst = isK ? k_out : q_out;
            dst[dstBase + d0] = nx0 * c0 - ny0 * s0;
            dst[dstBase + d0 + 64] = nx0 * s0 + ny0 * c0;
            dst[dstBase + d1] = nx1 * c1 - ny1 * s1;
            dst[dstBase + d1 + 64] = nx1 * s1 + ny1 * c1;
            """)

    /// Fused `kNorm` + `YarnRoPE` (`mscale` + `RoPE`) + `[1, C, 8, 128] -> [1, 8, C, 128]` transpose
    /// for context `(contextKeys, contextValues)`.
    static let ctxKVNormRopeKernel = MLXFast.metalKernel(
        name: "df2_ctx_kv_norm_rope",
        inputNames: ["k", "v", "kw", "freqs", "params", "fparams"],
        outputNames: ["k_out", "v_out"],
        source: """
            const int C = params[0];
            const int offset = params[1];
            const float eps = fparams[0];
            const float mscale = fparams[1];
            const int vecIdx = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
            if (vecIdx >= 16 * C) return;
            const uint lane = thread_index_in_simdgroup;
            const int hTotal = vecIdx / C;
            const int s = vecIdx % C;
            const int d0 = int(lane);
            const int d1 = d0 + 32;
            if (hTotal >= 8) {
              const int h = hTotal - 8;
              const size_t srcBase = ((size_t)s * 8 + h) * 128;
              const size_t dstBase = ((size_t)h * C + s) * 128;
              v_out[dstBase + d0] = float(v[srcBase + d0]);
              v_out[dstBase + d0 + 64] = float(v[srcBase + d0 + 64]);
              v_out[dstBase + d1] = float(v[srcBase + d1]);
              v_out[dstBase + d1 + 64] = float(v[srcBase + d1 + 64]);
              return;
            }
            const int h = hTotal;
            const size_t srcBase = ((size_t)s * 8 + h) * 128;
            const size_t dstBase = ((size_t)h * C + s) * 128;
            float acc = 0.0f;
            for (int i = 0; i < 4; i++) {
              const float xi = float(k[srcBase + 4 * int(lane) + i]);
              acc += xi * xi;
            }
            const float inv_mean = metal::precise::rsqrt(simd_sum(acc) / 128u + eps);
            const float x0 = float(k[srcBase + d0]);
            const float y0 = float(k[srcBase + d0 + 64]);
            const float x1 = float(k[srcBase + d1]);
            const float y1 = float(k[srcBase + d1 + 64]);
            const float nx0 = (kw[d0] * (x0 * inv_mean)) * mscale;
            const float ny0 = (kw[d0 + 64] * (y0 * inv_mean)) * mscale;
            const float nx1 = (kw[d1] * (x1 * inv_mean)) * mscale;
            const float ny1 = (kw[d1 + 64] * (y1 * inv_mean)) * mscale;
            const float pos = float(offset + s);
            const float th0 = pos * (1.0f / freqs[d0]);
            const float c0 = metal::fast::cos(th0), s0 = metal::fast::sin(th0);
            const float th1 = pos * (1.0f / freqs[d1]);
            const float c1 = metal::fast::cos(th1), s1 = metal::fast::sin(th1);
            k_out[dstBase + d0] = nx0 * c0 - ny0 * s0;
            k_out[dstBase + d0 + 64] = nx0 * s0 + ny0 * c0;
            k_out[dstBase + d1] = nx1 * c1 - ny1 * s1;
            k_out[dstBase + d1 + 64] = nx1 * s1 + ny1 * c1;
            """)

    /// Six independent context-key projections share factors within each
    /// two-layer, four-head group. Values remain transpose views for the cache.
    static let ctxKGroupNormRopeKernel = MLXFast.metalKernel(
        name: "df2_ctx_k_group_norm_rope",
        inputNames: (0 ..< 6).map { "k\($0)" } + (0 ..< 6).map { "w\($0)" }
            + ["freqs", "params", "fparams"],
        outputNames: (0 ..< 6).map { "k_out\($0)" },
        source: """
            const int C = params[0];
            const int offset = params[1];
            const float eps = fparams[0];
            const float mscale = fparams[1];
            const int s = int(threadgroup_position_in_grid.x);
            const int group = int(threadgroup_position_in_grid.y);
            const int pair = group / 2;
            const uint lane = thread_index_in_simdgroup;
            const int h = 4 * (group % 2) + int(simdgroup_index_in_threadgroup);
            const int d0 = int(lane), d1 = d0 + 32;
            threadgroup float4 factors[32];
            if (simdgroup_index_in_threadgroup == 0) {
              const float pos = float(offset + s);
              const float th0 = pos * (1.0f / freqs[d0]);
              const float c0 = metal::fast::cos(th0), s0 = metal::fast::sin(th0);
              const float th1 = pos * (1.0f / freqs[d1]);
              const float c1 = metal::fast::cos(th1), s1 = metal::fast::sin(th1);
              factors[lane] = float4(c0, s0, c1, s1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float4 cs = factors[lane];
            const float c0 = cs.x, s0 = cs.y, c1 = cs.z, s1 = cs.w;
            const size_t srcBase = ((size_t)s * 8 + h) * 128;
            const size_t dstBase = ((size_t)h * C + s) * 128;
            for (int local = 0; local < 2; local++) {
              const int layer = pair * 2 + local;
              const device float* k = layer == 0 ? k0 : layer == 1 ? k1 : layer == 2 ? k2 : layer == 3 ? k3 : layer == 4 ? k4 : k5;
              const device float* kw = layer == 0 ? w0 : layer == 1 ? w1 : layer == 2 ? w2 : layer == 3 ? w3 : layer == 4 ? w4 : w5;
              device float* dst = layer == 0 ? k_out0 : layer == 1 ? k_out1 : layer == 2 ? k_out2 : layer == 3 ? k_out3 : layer == 4 ? k_out4 : k_out5;
              float acc = 0.0f;
              for (int i = 0; i < 4; i++) {
                const float xi = float(k[srcBase + 4 * int(lane) + i]);
                acc += xi * xi;
              }
              const float inv_mean = metal::precise::rsqrt(simd_sum(acc) / 128u + eps);
              const float x0 = float(k[srcBase + d0]);
              const float y0 = float(k[srcBase + d0 + 64]);
              const float x1 = float(k[srcBase + d1]);
              const float y1 = float(k[srcBase + d1 + 64]);
              const float nx0 = (kw[d0] * (x0 * inv_mean)) * mscale;
              const float ny0 = (kw[d0 + 64] * (y0 * inv_mean)) * mscale;
              const float nx1 = (kw[d1] * (x1 * inv_mean)) * mscale;
              const float ny1 = (kw[d1 + 64] * (y1 * inv_mean)) * mscale;
              dst[dstBase + d0] = nx0 * c0 - ny0 * s0;
              dst[dstBase + d0 + 64] = nx0 * s0 + ny0 * c0;
              dst[dstBase + d1] = nx1 * c1 - ny1 * s1;
              dst[dstBase + d1 + 64] = nx1 * s1 + ny1 * c1;
            }
            """)

    static func ctxKGroupNormRope(
        keys: [MLXArray], norms: [RMSNorm], ropeSpec: RopeSpec, C: Int, offset: Int
    ) -> [MLXArray]? {
        guard on, (1 ... 16).contains(C), keys.count == 6, norms.count == 6,
            keys.allSatisfy({ $0.dtype == .float32 && $0.shape == [1, C, 1024] }),
            norms.allSatisfy({ $0.eps == norms[0].eps })
        else { return nil }
        let inputs = keys + norms.map { normWeight($0) } + [
            ropeSpec.freqs, HostConstants.recentInt32([Int32(C), Int32(offset)]),
            ropeSpec.floatParameters(eps: norms[0].eps),
        ]
        return ctxKGroupNormRopeKernel(
            inputs, grid: (C * 128, 6, 1), threadGroup: (128, 1, 1),
            outputShapes: Array(repeating: [1, 8, C, 128], count: 6),
            outputDTypes: Array(repeating: .float32, count: 6))
    }

    static func qkvNormRope(
        q: MLXArray, k: MLXArray, v: MLXArray,
        qNorm: RMSNorm, kNorm: RMSNorm, ropeSpec: RopeSpec, L: Int, offset: Int
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard on, L > 0 else { return nil }
        let qw = normWeight(qNorm)
        let kw = normWeight(kNorm)
        let totalVecs = 48 * L
        let tgs = (totalVecs + 3) / 4
        let outs = qkvNormRopeKernel(
            [
                q, k, v, qw, kw, ropeSpec.freqs,
                HostConstants.recentInt32([Int32(L), Int32(offset)]),
                ropeSpec.floatParameters(eps: qNorm.eps),
            ],
            grid: (tgs * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[1, 32, L, 128], [1, 8, L, 128], [1, 8, L, 128]],
            outputDTypes: [.float32, .float32, .float32])
        return (outs[0], outs[1], outs[2])
    }

    static func ctxKVNormRope(
        k: MLXArray, v: MLXArray,
        kNorm: RMSNorm, ropeSpec: RopeSpec, C: Int, offset: Int
    ) -> (MLXArray, MLXArray)? {
        guard on, C > 0 else { return nil }
        let kw = normWeight(kNorm)
        let totalVecs = 16 * C
        let tgs = (totalVecs + 3) / 4
        let outs = ctxKVNormRopeKernel(
            [
                k, v, kw, ropeSpec.freqs,
                HostConstants.recentInt32([Int32(C), Int32(offset)]),
                ropeSpec.floatParameters(eps: kNorm.eps),
            ],
            grid: (tgs * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[1, 8, C, 128], [1, 8, C, 128]],
            outputDTypes: [.float32, .float32])
        return (outs[0], outs[1])
    }

    /// Fused `qNorm` + `kNorm` + `YarnRoPE` + `[1, S, H, 128] -> [1, H, S, 128]` transpose for BOTH
    /// block `(queries, blockKeys, blockValues)` and context `(contextKeys, contextValues)`, producing
    /// `queries` `[1, 32, L, 128]`, `combinedKeys` `[1, 8, C + L, 128]`, `combinedValues` `[1, 8, C + L, 128]`.
    static let qkvAllNormRopeKernel = MLXFast.metalKernel(
        name: "df2_qkv_all_norm_rope",
        inputNames: ["q", "bk", "bv", "ck", "cv", "qw", "kw", "freqs", "params", "fparams"],
        outputNames: ["q_out", "k_out", "v_out"],
        source: """
            const int L = params[0];
            const int C = params[1];
            const int offset = params[2];
            const int CL = C + L;
            const float eps = fparams[0];
            const float mscale = fparams[1];
            const int vecIdx = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
            if (vecIdx >= 32 * L + 16 * CL) return;
            const uint lane = thread_index_in_simdgroup;
            const int d0 = int(lane);
            const int d1 = d0 + 32;
            if (vecIdx >= 32 * L + 8 * CL) {
              const int vIdx = vecIdx - (32 * L + 8 * CL);
              const int h = vIdx / CL;
              const int t = vIdx % CL;
              const device float* vsrc = (t < C) ? cv : bv;
              const int s = (t < C) ? t : (t - C);
              const size_t srcBase = ((size_t)s * 8 + h) * 128;
              const size_t dstBase = ((size_t)h * CL + t) * 128;
              v_out[dstBase + d0] = float(vsrc[srcBase + d0]);
              v_out[dstBase + d0 + 64] = float(vsrc[srcBase + d0 + 64]);
              v_out[dstBase + d1] = float(vsrc[srcBase + d1]);
              v_out[dstBase + d1 + 64] = float(vsrc[srcBase + d1 + 64]);
              return;
            }
            const bool isK = vecIdx >= 32 * L;
            int h, t, s, H, S_out;
            const device float* src;
            const device float* w;
            device float* dst;
            float pos;
            int posRow;
            if (!isK) {
              h = vecIdx / L;
              s = vecIdx % L;
              t = s;
              H = 32;
              S_out = L;
              src = q;
              w = qw;
              dst = q_out;
              pos = float(offset + C + s);
              posRow = C + s;
            } else {
              const int kIdx = vecIdx - 32 * L;
              h = kIdx / CL;
              t = kIdx % CL;
              s = (t < C) ? t : (t - C);
              H = 8;
              S_out = CL;
              src = (t < C) ? ck : bk;
              w = kw;
              dst = k_out;
              pos = float(offset + t);
              posRow = t;
            }
            const size_t srcBase = ((size_t)s * H + h) * 128;
            const size_t dstBase = ((size_t)h * S_out + t) * 128;
            float acc = 0.0f;
            for (int i = 0; i < 4; i++) {
              const float xi = float(src[srcBase + 4 * int(lane) + i]);
              acc += xi * xi;
            }
            const float inv_mean = metal::precise::rsqrt(simd_sum(acc) / 128u + eps);
            const float x0 = float(src[srcBase + d0]);
            const float y0 = float(src[srcBase + d0 + 64]);
            const float x1 = float(src[srcBase + d1]);
            const float y1 = float(src[srcBase + d1 + 64]);
            const float nx0 = (w[d0] * (x0 * inv_mean)) * mscale;
            const float ny0 = (w[d0 + 64] * (y0 * inv_mean)) * mscale;
            const float nx1 = (w[d1] * (x1 * inv_mean)) * mscale;
            const float ny1 = (w[d1 + 64] * (y1 * inv_mean)) * mscale;
            float c0, s0, c1, s1;
            if (PHASES) {
              const size_t phaseBase = ((size_t)posRow * 32 + lane) * 4;
              c0 = freqs[phaseBase]; s0 = freqs[phaseBase + 1];
              c1 = freqs[phaseBase + 2]; s1 = freqs[phaseBase + 3];
            } else {
              const float th0 = pos * (1.0f / freqs[d0]);
              c0 = metal::fast::cos(th0); s0 = metal::fast::sin(th0);
              const float th1 = pos * (1.0f / freqs[d1]);
              c1 = metal::fast::cos(th1); s1 = metal::fast::sin(th1);
            }
            dst[dstBase + d0] = nx0 * c0 - ny0 * s0;
            dst[dstBase + d0 + 64] = nx0 * s0 + ny0 * c0;
            dst[dstBase + d1] = nx1 * c1 - ny1 * s1;
            dst[dstBase + d1 + 64] = nx1 * s1 + ny1 * c1;
            """)

    static func qkvAllNormRope(
        q: MLXArray, k: MLXArray, v: MLXArray,
        ck: MLXArray, cv: MLXArray,
        qNorm: RMSNorm, kNorm: RMSNorm, ropeSpec: RopeSpec,
        L: Int, C: Int, blockOffset: Int, ctxOffset: Int
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard on, L > 0, C > 0, blockOffset == ctxOffset + C else { return nil }
        let qw = normWeight(qNorm)
        let kw = normWeight(kNorm)
        let CL = C + L
        let totalVecs = 32 * L + 16 * CL
        let tgs = (totalVecs + 3) / 4
        let phases = ropeSpec.phaseOffset == ctxOffset && ropeSpec.phaseRows >= C + L
            ? ropeSpec.phases : nil
        let outs = qkvAllNormRopeKernel(
            [
                q, k, v, ck, cv, qw, kw, phases ?? ropeSpec.freqs,
                HostConstants.recentInt32([Int32(L), Int32(C), Int32(ctxOffset)]),
                ropeSpec.floatParameters(eps: qNorm.eps),
            ],
            template: [("PHASES", phases != nil)],
            grid: (tgs * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[1, 32, L, 128], [1, 8, CL, 128], [1, 8, CL, 128]],
            outputDTypes: [.float32, .float32, .float32])
        return (outs[0], outs[1], outs[2])
    }

    /// Fused `rmsNorm(x, weight, eps)` + `W4Codes` int8 group quantization for `D = 4096, L <= 16`.
    static let rmsQuantKernel = MLXFast.metalKernel(
        name: "df2_rms_q8g",
        inputNames: ["x", "weight", "dsz", "eps_arr"],
        outputNames: ["out_norm", "xq", "ascale", "xs"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int l = int(threadgroup_position_in_grid.y);
            const int L = dsz[1];
            const int c0 = int(tid) * 16;
            const int G = D / KG;
            const int g = int(tid) / (KG / 16);
            if (l >= L) {
              if (l < 16) {
                const size_t qRowBase = (size_t)l * D + c0;
                device char4* dst = (device char4*)(xq + qRowBase);
                for (int u = 0; u < 4; u++) { dst[u] = char4(0); }
                if ((int(tid) & ((KG / 16) - 1)) == 0) {
                  ascale[l * G + g] = 0.0f;
                  xs[l * G + g] = 0.0f;
                }
              }
              return;
            }
            const size_t rowBase = (size_t)l * D + c0;
            float v[16];
            float s4[4];
            for (int k = 0; k < 4; k++) {
              float ak = 0.0f;
              for (int i = 0; i < 4; i++) {
                const int u = 4 * k + i;
                const float a = float(x[rowBase + u]);
                v[u] = a;
                ak += a * a;
              }
              s4[k] = ak;
            }
            float sq = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            sq += simd_shuffle_xor(sq, 4);
            threadgroup float shared_sq[32];
            if ((lane & 7u) == 0) { shared_sq[(sg << 2) | (lane >> 3)] = sq; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float tot = simd_sum(shared_sq[lane]);
              if (lane == 0) {
                shared_sq[0] = metal::precise::rsqrt(tot / uint(D) + eps_arr[0]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float scale = shared_sq[0];
            float nv[16];
            float local_amax = 0.0f;
            for (int u = 0; u < 16; u++) {
              const float y = float(weight[c0 + u]) * (v[u] * scale);
              if (WRITE_NORM) { out_norm[rowBase + u] = y; }
              nv[u] = y;
              local_amax = max(local_amax, fabs(y));
            }
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 1));
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 2));
            if (KG == 128) {
              local_amax = max(local_amax, simd_shuffle_xor(local_amax, 4));
            }
            float seq_sum = 0.0f;
            for (int step = 0; step < (KG / 16); step++) {
              if (step > 0) {
                seq_sum = simd_shuffle_up(seq_sum, 1);
              }
              for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
            }
            const float a = local_amax > 0.0f ? local_amax / 127.0f : 1.0f;
            const float ia = 1.0f / a;
            device char4* dst = (device char4*)(xq + rowBase);
            for (int u = 0; u < 4; u++) {
              const float4 fv = float4(nv[4 * u], nv[4 * u + 1], nv[4 * u + 2], nv[4 * u + 3]);
              dst[u] = char4(int4(rint(fv * ia)));
            }
            if ((int(lane) & ((KG / 16) - 1)) == (KG / 16) - 1) {
              ascale[l * G + g] = a;
              xs[l * G + g] = seq_sum;
            }
            """)

    /// First request RMS/quant plus exact rotary coefficients, without an extra dispatch.
    static let rmsQuantPhasesKernel = MLXFast.metalKernel(
        name: "df2_rms_q8g_phases",
        inputNames: ["x", "weight", "dsz", "eps_arr", "freqs", "phase_params"],
        outputNames: ["out_norm", "xq", "ascale", "xs", "phases"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int l = int(threadgroup_position_in_grid.y);
            const int L = dsz[1];
            if (tid < 32u) {
              for (int phaseRow = l; phaseRow < phase_params[0]; phaseRow += 16) {
                const float pos = float(phase_params[1] + phaseRow);
                const float th0 = pos * (1.0f / freqs[lane]);
                const float th1 = pos * (1.0f / freqs[lane + 32u]);
                const size_t phaseBase = ((size_t)phaseRow * 32 + lane) * 4;
                phases[phaseBase] = metal::fast::cos(th0);
                phases[phaseBase + 1] = metal::fast::sin(th0);
                phases[phaseBase + 2] = metal::fast::cos(th1);
                phases[phaseBase + 3] = metal::fast::sin(th1);
              }
            }
            const int c0 = int(tid) * 16;
            const int G = D / KG;
            const int g = int(tid) / (KG / 16);
            if (l >= L) {
              if (l < 16) {
                const size_t qRowBase = (size_t)l * D + c0;
                device char4* dst = (device char4*)(xq + qRowBase);
                for (int u = 0; u < 4; u++) { dst[u] = char4(0); }
                if ((int(tid) & ((KG / 16) - 1)) == 0) {
                  ascale[l * G + g] = 0.0f;
                  xs[l * G + g] = 0.0f;
                }
              }
              return;
            }
            const size_t rowBase = (size_t)l * D + c0;
            float v[16];
            float s4[4];
            for (int k = 0; k < 4; k++) {
              float ak = 0.0f;
              for (int i = 0; i < 4; i++) {
                const int u = 4 * k + i;
                const float a = float(x[rowBase + u]);
                v[u] = a;
                ak += a * a;
              }
              s4[k] = ak;
            }
            float sq = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            sq += simd_shuffle_xor(sq, 4);
            threadgroup float shared_sq[32];
            if ((lane & 7u) == 0) { shared_sq[(sg << 2) | (lane >> 3)] = sq; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float tot = simd_sum(shared_sq[lane]);
              if (lane == 0) {
                shared_sq[0] = metal::precise::rsqrt(tot / uint(D) + eps_arr[0]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float scale = shared_sq[0];
            float nv[16];
            float local_amax = 0.0f;
            for (int u = 0; u < 16; u++) {
              const float y = float(weight[c0 + u]) * (v[u] * scale);
              if (WRITE_NORM) { out_norm[rowBase + u] = y; }
              nv[u] = y;
              local_amax = max(local_amax, fabs(y));
            }
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 1));
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 2));
            if (KG == 128) {
              local_amax = max(local_amax, simd_shuffle_xor(local_amax, 4));
            }
            float seq_sum = 0.0f;
            for (int step = 0; step < (KG / 16); step++) {
              if (step > 0) {
                seq_sum = simd_shuffle_up(seq_sum, 1);
              }
              for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
            }
            const float a = local_amax > 0.0f ? local_amax / 127.0f : 1.0f;
            const float ia = 1.0f / a;
            device char4* dst = (device char4*)(xq + rowBase);
            for (int u = 0; u < 4; u++) {
              const float4 fv = float4(nv[4 * u], nv[4 * u + 1], nv[4 * u + 2], nv[4 * u + 3]);
              dst[u] = char4(int4(rint(fv * ia)));
            }
            if ((int(lane) & ((KG / 16) - 1)) == (KG / 16) - 1) {
              ascale[l * G + g] = a;
              xs[l * G + g] = seq_sum;
            }
            """
    )

    static let rmsQuantPhasesWideKernel = MLXFast.metalKernel(
        name: "df2_rms_q8g_phases_e\(DF2WideLanes.e)",
        inputNames: ["x", "weight", "dsz", "eps_arr", "freqs", "phase_params"],
        outputNames: ["out_norm", "xq", "ascale", "xs", "phases"],
        source: DF2WideLaneSource.source(.rmsQuant, e: DF2WideLanes.e).replacingOccurrences(
            of: "const int L = dsz[1];",
            with: """
                const int L = dsz[1];
                if (tid < 32u) {
                  for (int phaseRow = l; phaseRow < phase_params[0]; phaseRow += 16) {
                    const float pos = float(phase_params[1] + phaseRow);
                    const float th0 = pos * (1.0f / freqs[lane]);
                    const float th1 = pos * (1.0f / freqs[lane + 32u]);
                    const size_t phaseBase = ((size_t)phaseRow * 32 + lane) * 4;
                    phases[phaseBase] = metal::fast::cos(th0);
                    phases[phaseBase + 1] = metal::fast::sin(th0);
                    phases[phaseBase + 2] = metal::fast::cos(th1);
                    phases[phaseBase + 3] = metal::fast::sin(th1);
                  }
                }
                """))

    static func rmsQuantPhases(
        _ x: MLXArray, _ norm: RMSNorm, group: Int, ropeSpec: RopeSpec,
        C: Int, offset: Int
    ) -> (MLXArray, W4Codes, RopeSpec)? {
        let D = x.dim(-1), L = x.size / D
        guard on, D == 4096, (1 ... 16).contains(L), (1 ... 16).contains(C),
            (group == 64 || group == 128), split > 0, D / group >= split,
            offset >= 0, offset <= Int(Int32.max) - C - L
        else { return nil }
        let w = normWeight(norm)
        // Keep the existing alternate-wide selection and self-check path intact.
        let wide = DF2WideLanes.serves(
            .init(kind: .rmsQuant, dtype: x.dtype, dtype2: w.dtype, k: D, group: group, flag: 0))
        let rows = C + L, G = D / group
        let lanes = D / (wide ? DF2WideLanes.e : 16)
        let outs = (wide ? rmsQuantPhasesWideKernel : rmsQuantPhasesKernel)(
            [x, w, HostConstants.int32([Int32(D), Int32(L)]), HostConstants.float32([norm.eps]), ropeSpec.freqs,
                HostConstants.recentInt32([Int32(rows), Int32(offset)])],
            template: [("D", D), ("KG", group), ("WRITE_NORM", 1)],
            grid: (lanes, 16, 1), threadGroup: (lanes, 1, 1),
            outputShapes: [x.shape, [16 * D], [16 * G], [16 * G], [rows, 32, 4]],
            outputDTypes: [.float32, .int8, .float32, .float32, .float32])
        let codes = W4Codes(
            xq: outs[1], ascale: outs[2], xs: outs[3], rows: L, K: D, group: group)
        let phases = RopeSpec(
            freqs: ropeSpec.freqs, mscale: ropeSpec.mscale, phases: outs[4],
            phaseOffset: offset, phaseRows: rows)
        return (outs[0], codes, phases)
    }

    static func rmsQuant(
        _ x: MLXArray, _ norm: RMSNorm, group: Int, writeNorm: Bool = true
    ) -> (MLXArray, W4Codes)? {
        let D = x.dim(-1), L = x.size / D
        guard on, D == 4096, (1 ... 16).contains(L),
            (group == 64 || group == 128), split > 0, D / group >= split
        else {
            return nil
        }
        let w = normWeight(norm)
        let wide = DF2WideLanes.serves(
            .init(kind: .rmsQuant, dtype: x.dtype, dtype2: w.dtype, k: D, group: group, flag: 0))
        let outs = rmsQuantRun(
            x, w, D: D, L: L, eps: norm.eps, group: group, wide: wide, writeNorm: writeNorm)
        let codes = W4Codes(xq: outs[1], ascale: outs[2], xs: outs[3], rows: L, K: D, group: group)
        return (outs[0], codes)
    }

    /// `df2_rms_q8g` at 256 lanes per row, or its `DF2WideLaneSource` twin (`wide`).
    static func rmsQuantRun(
        _ x: MLXArray, _ w: MLXArray, D: Int, L: Int, eps: Float, group: Int, wide: Bool,
        writeNorm: Bool = true
    ) -> [MLXArray] {
        let G = D / group
        let lanes = D / (wide ? DF2WideLanes.e : 16)
        return (wide ? DF2WideLanes.rmsQuant : rmsQuantKernel)(
            [x, w, HostConstants.int32([Int32(D), Int32(L)]), HostConstants.float32([eps])],
            template: [("D", D), ("KG", group), ("WRITE_NORM", writeNorm ? 1 : 0)],
            grid: (lanes, Swift.max(L, 16), 1), threadGroup: (lanes, 1, 1),
            outputShapes: [writeNorm ? x.shape : [1], [16 * D], [16 * G], [16 * G]],
            outputDTypes: [.float32, .int8, .float32, .float32])
    }

    /// `df2_dyn_conv_rms_q8g` at 256 lanes per row, or its `DF2WideLaneSource` twin (`wide`).
    static func convRmsQuantRun(
        _ h: MLXArray, dyn: MLXArray, base: MLXArray, residual: MLXArray, weight: MLXArray,
        D: Int, L: Int, qStart: Int, eps: Float, group: Int, wide: Bool
    ) -> [MLXArray] {
        let G = D / group
        let qRows = L - qStart
        let writeRes = qStart == 0
        let resShape = writeRes ? h.shape : [1]
        let normShape = qStart > 0 ? (Array(h.shape.dropLast(2)) + [qRows, D]) : h.shape
        let lanes = D / (wide ? DF2WideLanes.e : 16)
        return (wide ? DF2WideLanes.convRmsQuant : convRmsQuantKernel)(
            [
                h, dyn, base, residual, weight,
                HostConstants.int32([Int32(D), Int32(L), Int32(qStart)]),
                HostConstants.float32([eps]),
            ],
            template: [("D", D), ("KG", group), ("WRITE_RES", writeRes ? 1 : 0)],
            grid: (lanes, Swift.max(L, qStart + 16), 1), threadGroup: (lanes, 1, 1),
            outputShapes: [resShape, normShape, [16 * D], [16 * G], [16 * G]],
            outputDTypes: [.float32, .float32, .int8, .float32, .float32])
    }

    /// One part of `DFlash2GroupedDynamicCausalConv` (B = 1) in one kernel, plus an optional residual.
    static let conv = MLXFast.metalKernel(
        name: "df2_dyn_conv", inputNames: ["h", "dyn", "base", "res", "dsz"], outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            const int L = dsz[1];
            if (int(i) >= L * D) return;
            const int l = int(i) / D, c = int(i) % D, g = c / 16;
            const int dw = 4 * (D / 16);
            float acc = RES ? float(res[i]) : 0.0f;
            acc += (float(base[(PART * 2 + 0) * D + c]) + float(dyn[l * dw + PART * (dw / 2) + g])) * float(h[i]);
            if (l >= 1) {
              acc += (float(base[(PART * 2 + 1) * D + c]) + float(dyn[l * dw + PART * (dw / 2) + D / 16 + g])) * float(h[i - D]);
            }
            out[i] = acc;
            """)

    /// Fused `dynConv(part: 1, residual: res)` + `rmsNorm(out_res, weight, eps)` for `D = 4096`.
    static let convRms = MLXFast.metalKernel(
        name: "df2_dyn_conv_rms",
        inputNames: ["h", "dyn", "base", "res", "weight", "dsz", "eps_arr"],
        outputNames: ["out_res", "out_norm"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int l = int(threadgroup_position_in_grid.y);
            const int L = dsz[1];
            if (l >= L) return;
            const int dw = 4 * (D / 16);
            const int g = int(tid);
            const int c0 = g * 16;
            const size_t rowBase = (size_t)l * D + c0;
            const float d0 = float(dyn[l * dw + (dw / 2) + g]);
            const float d1 = l >= 1 ? float(dyn[l * dw + (dw / 2) + (D / 16) + g]) : 0.0f;
            float v[16];
            float s4[4];
            for (int k = 0; k < 4; k++) {
              float ak = 0.0f;
              for (int i = 0; i < 4; i++) {
                const int u = 4 * k + i;
                const int c = c0 + u;
                float acc = float(res[rowBase + u]) + (float(base[2 * D + c]) + d0) * float(h[rowBase + u]);
                if (l >= 1) {
                  acc += (float(base[3 * D + c]) + d1) * float(h[rowBase - D + u]);
                }
                v[u] = acc;
                out_res[rowBase + u] = acc;
                ak += acc * acc;
              }
              s4[k] = ak;
            }
            float sq = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            sq += simd_shuffle_xor(sq, 4);
            threadgroup float shared_sq[32];
            if ((lane & 7u) == 0) { shared_sq[(sg << 2) | (lane >> 3)] = sq; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float tot = simd_sum(shared_sq[lane]);
              if (lane == 0) {
                shared_sq[0] = metal::precise::rsqrt(tot / uint(D) + eps_arr[0]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float scale = shared_sq[0];
            for (int u = 0; u < 16; u++) {
              out_norm[rowBase + u] = float(weight[c0 + u]) * (v[u] * scale);
            }
            """)

    /// Fused `dynConv(part: 1, residual: res)` + `rmsNorm(out_res, weight, eps)` + `W4Codes` int8 quantization
    /// on rows `qStart ..< L` for `D = 4096`.
    static let convRmsQuantKernel = MLXFast.metalKernel(
        name: "df2_dyn_conv_rms_q8g",
        inputNames: ["h", "dyn", "base", "res", "weight", "dsz", "eps_arr"],
        outputNames: ["out_res", "out_norm", "xq", "ascale", "xs"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int l = int(threadgroup_position_in_grid.y);
            const int L = dsz[1];
            const int qStart = dsz[2];
            const int qm = l - qStart;
            const int c0 = int(tid) * 16;
            const int G = D / KG;
            const int qg = int(tid) / (KG / 16);
            if (!WRITE_RES && l < qStart) return;
            if (l >= L) {
              if (qm >= 0 && qm < 16) {
                const size_t qRowBase = (size_t)qm * D + c0;
                device char4* dst = (device char4*)(xq + qRowBase);
                for (int u = 0; u < 4; u++) { dst[u] = char4(0); }
                if ((int(tid) & ((KG / 16) - 1)) == 0) {
                  ascale[qm * G + qg] = 0.0f;
                  xs[qm * G + qg] = 0.0f;
                }
              }
              return;
            }
            const int dw = 4 * (D / 16);
            const int g = int(tid);
            const size_t rowBase = (size_t)l * D + c0;
            const float d0 = float(dyn[l * dw + (dw / 2) + g]);
            const float d1 = l >= 1 ? float(dyn[l * dw + (dw / 2) + (D / 16) + g]) : 0.0f;
            float v[16];
            float s4[4];
            for (int k = 0; k < 4; k++) {
              float ak = 0.0f;
              for (int i = 0; i < 4; i++) {
                const int u = 4 * k + i;
                const int c = c0 + u;
                float acc = float(res[rowBase + u]) + (float(base[2 * D + c]) + d0) * float(h[rowBase + u]);
                if (l >= 1) {
                  acc += (float(base[3 * D + c]) + d1) * float(h[rowBase - D + u]);
                }
                v[u] = acc;
                if (WRITE_RES) { out_res[rowBase + u] = acc; }
                ak += acc * acc;
              }
              s4[k] = ak;
            }
            float sq = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            sq += simd_shuffle_xor(sq, 4);
            threadgroup float shared_sq[32];
            if ((lane & 7u) == 0) { shared_sq[(sg << 2) | (lane >> 3)] = sq; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float tot = simd_sum(shared_sq[lane]);
              if (lane == 0) {
                shared_sq[0] = metal::precise::rsqrt(tot / uint(D) + eps_arr[0]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float scale = shared_sq[0];
            float nv[16];
            float local_amax = 0.0f;
            for (int u = 0; u < 16; u++) {
              const float y = float(weight[c0 + u]) * (v[u] * scale);
              nv[u] = y;
              local_amax = max(local_amax, fabs(y));
            }
            if (qm >= 0 && qm < 16) {
              const size_t qRowBase = (size_t)qm * D + c0;
              for (int u = 0; u < 16; u++) {
                out_norm[qRowBase + u] = nv[u];
              }
              local_amax = max(local_amax, simd_shuffle_xor(local_amax, 1));
              local_amax = max(local_amax, simd_shuffle_xor(local_amax, 2));
              if (KG == 128) {
                local_amax = max(local_amax, simd_shuffle_xor(local_amax, 4));
              }
              float seq_sum = 0.0f;
              for (int step = 0; step < (KG / 16); step++) {
                if (step > 0) {
                  seq_sum = simd_shuffle_up(seq_sum, 1);
                }
                for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
              }
              const float a = local_amax > 0.0f ? local_amax / 127.0f : 1.0f;
              const float ia = 1.0f / a;
              device char4* dst = (device char4*)(xq + qRowBase);
              for (int u = 0; u < 4; u++) {
                const float4 fv = float4(nv[4 * u], nv[4 * u + 1], nv[4 * u + 2], nv[4 * u + 3]);
                dst[u] = char4(int4(rint(fv * ia)));
              }
              if ((int(lane) & ((KG / 16) - 1)) == (KG / 16) - 1) {
                ascale[qm * G + qg] = a;
                xs[qm * G + qg] = seq_sum;
              }
            }
            """)

    static func dynConvRms(
        _ h: MLXArray, dyn: MLXArray, base: MLXArray, residual: MLXArray, norm: RMSNorm,
        qStart: Int = 0
    ) -> (MLXArray, MLXArray, W4Codes?)? {
        let D = h.dim(-1), L = h.size / D
        guard on, D == 4096 else { return nil }
        let w = normWeight(norm)
        let qRows = L - qStart
        if (1 ... 16).contains(qRows), let group = w4Group, (group == 64 || group == 128),
            split > 0, D / group >= split
        {
            let wide = DF2WideLanes.serves(
                .init(kind: .convRmsQuant, dtype: h.dtype, dtype2: residual.dtype, k: D, group: group,
                    flag: (qStart == 0 ? 1 : 0) + (dyn.dtype == .float32 ? 0 : 2) + (base.dtype == .float32 ? 0 : 4)))
            let outs = convRmsQuantRun(
                h, dyn: dyn, base: base, residual: residual, weight: w,
                D: D, L: L, qStart: qStart, eps: norm.eps, group: group, wide: wide)
            let codes = W4Codes(
                xq: outs[2], ascale: outs[3], xs: outs[4], rows: qRows, K: D, group: group)
            return (outs[0], outs[1], codes)
        }
        let outs = convRms(
            [h, dyn, base, residual, w, HostConstants.int32([Int32(D), Int32(L)]), HostConstants.float32([norm.eps])],
            template: [("D", D)],
            grid: (256, L, 1), threadGroup: (256, 1, 1),
            outputShapes: [h.shape, h.shape], outputDTypes: [.float32, .float32])
        return (outs[0], outs[1], nil)
    }

    static func dynConv(_ h: MLXArray, dyn: MLXArray, base: MLXArray, part: Int, residual: MLXArray?) -> MLXArray {
        let D = h.dim(-1), L = h.size / D
        return conv(
            [h, dyn, base, residual ?? h, HostConstants.int32([Int32(D), Int32(L)])],
            template: [("D", D), ("PART", part), ("RES", residual == nil ? 0 : 1)],
            grid: ((L * D + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [h.shape], outputDTypes: [.float32])[0]
    }

    /// Multi-block parallel top-16 per (row, block): 8 blocks of 256 threads per row (120 blocks for P=15).
    static let topk = MLXFast.metalKernel(
        name: "df2_topk16_mb8", inputNames: ["logits"], outputNames: ["ids", "vals"],
        source: topkSource)

    static let topkSource = """
            const uint row = threadgroup_position_in_grid.x;
            const uint blk = threadgroup_position_in_grid.y;
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
            const int Vn = logits_shape[1];
            const device float* lr = logits + (size_t)row * Vn;
            const float NEG = -INFINITY;
            float tv[4] = {NEG, NEG, NEG, NEG}; int ti[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
            for (int n = int(blk * 256u + tid); n < Vn; n += 2048) {
              const float v = lr[n];
              if (v > tv[3]) {
                int k = 3;
                while (k > 0 && v > tv[k - 1]) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; k--; }
                tv[k] = v; ti[k] = n;
              }
            }
            threadgroup float wv[128]; threadgroup int wi[128];
            for (int r = 0; r < 16; r++) {
              const float bv = tv[0];
              const int bi = ti[0];
              const float m = simd_max(bv);
              const int best_id = simd_min(bv == m ? bi : 0x7fffffff);
              if (bv == m && bi == best_id) {
                wv[sg * 16 + r] = m;
                wi[sg * 16 + r] = best_id;
                tv[0] = tv[1]; tv[1] = tv[2]; tv[2] = tv[3]; tv[3] = NEG;
                ti[0] = ti[1]; ti[1] = ti[2]; ti[2] = ti[3]; ti[3] = 0x7fffffff;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              float sv[4]; int si[4];
              for (int k = 0; k < 4; k++) {
                sv[k] = wv[lane * 4 + k];
                si[k] = wi[lane * 4 + k];
              }
              const size_t outBase = ((size_t)row * 8u + blk) * 16u;
              for (int r = 0; r < 16; r++) {
                float bv = NEG; int bi = 0x7fffffff; int bk = 0;
                for (int k = 0; k < 4; k++) {
                  const float v = sv[k]; const int id = si[k];
                  if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; bk = k; }
                }
                const float m = simd_max(bv);
                const int best_id = simd_min(bv == m ? bi : 0x7fffffff);
                if (bv == m && bi == best_id) {
                  ids[outBase + r] = best_id;
                  vals[outBase + r] = m;
                  sv[bk] = NEG;
                }
              }
            }
            """

    /// `DARKBLOOM_BONSAI_DF2_TOPK_FAST=0` keeps the record top-k.
    static let topkFastAllowed = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DF2_TOPK_FAST"] != "0"
    /// Written once by `prepareTopkFast` at bind, before any proposal; read-only afterwards.
    nonisolated(unsafe) static var topkFastPassed = false
    nonisolated(unsafe) private static var topkFastPrepared = false

    /// `topkSource` with stage 1 issuing eight strided loads before it inserts them, in the
    /// record's order. Every comparison and its order are the record's, so the ids and values
    /// are too; the tail past the last full chunk runs the record loop. Nil unless the cut
    /// matches once.
    static let topkFastSource: String? = {
        let loop = """
            for (int n = int(blk * 256u + tid); n < Vn; n += 2048) {
              const float v = lr[n];
              if (v > tv[3]) {
                int k = 3;
                while (k > 0 && v > tv[k - 1]) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; k--; }
                tv[k] = v; ti[k] = n;
              }
            }
            """
        let chunked = """
            int n = int(blk * 256u + tid);
            for (; n + 7 * 2048 < Vn; n += 8 * 2048) {
              float xs[8];
              #pragma clang loop unroll(full)
              for (int c = 0; c < 8; c++) { xs[c] = lr[n + c * 2048]; }
              #pragma clang loop unroll(full)
              for (int c = 0; c < 8; c++) {
                const float v = xs[c];
                if (v > tv[3]) {
                  int k = 3;
                  while (k > 0 && v > tv[k - 1]) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; k--; }
                  tv[k] = v; ti[k] = n + c * 2048;
                }
              }
            }
            for (; n < Vn; n += 2048) {
              const float v = lr[n];
              if (v > tv[3]) {
                int k = 3;
                while (k > 0 && v > tv[k - 1]) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; k--; }
                tv[k] = v; ti[k] = n;
              }
            }
            """
        guard topkSource.components(separatedBy: loop).count == 2 else { return nil }
        return topkSource.replacingOccurrences(of: loop, with: chunked)
    }()

    static let topkFast: MLXFast.MLXFastKernel? = topkFastSource.map { source in
        MLXFast.metalKernel(
            name: "df2_topk16_mb8_c8", inputNames: ["logits"], outputNames: ["ids", "vals"], source: source)
    }

    /// Runs once at drafter bind, before any proposal or measured window: `topkFast` against
    /// `topk` on `topkCheckRows`. `topkFastPassed` is set only when every id and value bit
    /// matched; a mismatch or an MLX error keeps the record kernel.
    static func prepareTopkFast() {
        guard !topkFastPrepared else { return }
        topkFastPrepared = true
        var note = "off (DARKBLOOM_BONSAI_DF2_TOPK_FAST=0)"
        if topkFastAllowed, let fast = topkFast {
            do {
                let detail: String? = try withError { try topkFastMismatch(fast) }
                topkFastPassed = detail == nil
                note = detail.map { "off (\($0))" }
                    ?? "on (\(topkCheckShapes.reduce(0) { $0 + $1.rows }) rows of \(topkCheckKinds) kinds at V "
                    + topkCheckShapes.map { String($0.vocab) }.joined(separator: ", ")
                    + " bitwise equal to the record kernel, ids and values)"
            } catch {
                note = "off (error: \(error))"
            }
        } else if topkFastAllowed {
            note = "off (a cut of the record source did not match once)"
        }
        FileHandle.standardError.write(Data("DF2 topk fast: \(note)\n".utf8))
    }

    static let topkCheckShapes: [(rows: Int, vocab: Int)] = [(15, 98304), (15, 98304), (15, 10540), (2, 248320)]
    static let topkCheckKinds = 15

    private static func topkFastMismatch(_ fast: MLXFast.MLXFastKernel) throws -> String? {
        var first = 0
        for (c, shape) in topkCheckShapes.enumerated() {
            let logits = MLXArray(topkCheckRows(shape.rows, shape.vocab, first: first), [shape.rows, shape.vocab])
            first += shape.rows
            let record: [MLXArray] = try checkedOutputs(top(logits, kernel: topk), 2, "topk record")
            let mine: [MLXArray] = try checkedOutputs(top(logits, kernel: fast), 2, "topk fast")
            let same: MLXArray = MLX.all(record[0] .== mine[0])
                .&& MLX.all(record[1].view(dtype: .uint32) .== mine[1].view(dtype: .uint32))
            eval(same)
            if !same.item(Bool.self) { return "mismatch at case \(c), \(shape.rows) x \(shape.vocab)" }
        }
        return nil
    }

    /// Row r of a check case has kind (first + r) mod 15: pseudo-random, ties on a 0.5 grid, all
    /// zero, signed zeros, an ascending and a descending ramp, half NaN, all NaN, -inf with three
    /// finite values in one block (the stale-id rounds of the merge), NaN with sparse finite
    /// values, +inf sprinkled, subnormals, seven distinct values, one thread's elements on top,
    /// and one lane per simdgroup tied on top over a 0.5 grid.
    private static func topkCheckRows(_ rows: Int, _ vocab: Int, first: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * vocab)
        for r in 0 ..< rows {
            let kind = (first + r) % topkCheckKinds
            for n in 0 ..< vocab {
                let h = UInt32(truncatingIfNeeded: (UInt64(n) &* 2_654_435_761 &+ UInt64(first + r) &* 40_503) >> 7)
                let x = Float(Int(h % 200_001) - 100_000) / 1000
                let u = h % 1000
                let v: Float
                switch kind {
                case 0: v = x
                case 1: v = (x * 2).rounded() / 2
                case 2: v = 0
                case 3: v = u < 500 ? 0 : -0.0
                case 4: v = Float(n) * 0.001 + Float(r)
                case 5: v = -Float(n) * 0.001
                case 6: v = u < 500 ? .nan : x
                case 7: v = .nan
                case 8: v = (n % 2048 == 300 && n < 6000) ? x : (u < 500 ? -.infinity : .nan)
                case 9: v = n % 5003 == 11 ? x : .nan
                case 10: v = n % 777 == 5 ? .infinity : x
                case 11: v = x * 0x1p-130
                case 12: v = Float(h % 7) * 0.25
                case 13: v = n % 256 == 7 ? 1000 + x : x
                default: v = n % 32 == 3 ? 50 : (x * 2).rounded() / 2
                }
                out[r * vocab + n] = v
            }
        }
        return out
    }

    static let selector = MLXFast.metalKernel(
        name: "df2_selector_mb8", inputNames: ["cand", "unary", "proj", "pred", "succ", "anchor", "predLookup", "succLookup"],
        outputNames: ["path"],
        source: selectorSource)

    static let selectorSource = """
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
            const int P = int(cand_shape[0]);
            const float NEG = -INFINITY;
            threadgroup int tc[16][16];
            threadgroup float tu[16][16];
            for (int p = int(sg); p < P; p += 16) {
              float cv[4]; int ci[4];
              const size_t base = (size_t)p * 128u + lane * 4u;
              for (int k = 0; k < 4; k++) {
                cv[k] = float(unary[base + k]);
                ci[k] = cand[base + k];
              }
              for (int r = 0; r < 16; r++) {
                float bv = NEG; int bi = 0x7fffffff; int bk = 0;
                for (int k = 0; k < 4; k++) {
                  const float v = cv[k];
                  const int id = ci[k];
                  if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; bk = k; }
                }
                const float m = simd_max(bv);
                const int best_id = simd_min(bv == m ? bi : 0x7fffffff);
                if (bv == m && bi == best_id) {
                  tc[p][r] = best_id;
                  tu[p][r] = m;
                  cv[bk] = NEG;
                }
              }
            }
            threadgroup float ev[2][16];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int pr = anchor[0];
            for (int p = 0; p < P; p++) {
              const int c = tc[p][sg];
              uint predBase = 0u, succBase = 0u;
              if constexpr (PRED_PACKED) { predBase = predLookup[predLookup_shape[0] == 1 ? 0 : pr]; }
              if constexpr (SUCC_PACKED) { succBase = succLookup[succLookup_shape[0] == 1 ? 0 : c]; }
              float acc = 0.0f;
              for (int r = int(lane); r < 256; r += 64) {
                const uint pair = uint(r >> 6) * 32u + lane;
                const uint bit = (pair & 3u) * 24u;
                const size_t offset = (pair >> 2u) * 3u + (bit >> 5u);
                const uint shift = bit & 31u;
                uint predCodes = 0u, succCodes = 0u;
                if constexpr (PRED_PACKED) {
                  const size_t word = (size_t)pr * 96u + offset;
                  predCodes = uint(pred[word]) >> shift;
                  if (shift > 8u) predCodes |= uint(pred[word + 1u]) << (32u - shift);
                }
                if constexpr (SUCC_PACKED) {
                  const size_t word = (size_t)c * 96u + offset;
                  succCodes = uint(succ[word]) >> shift;
                  if (shift > 8u) succCodes |= uint(succ[word + 1u]) << (32u - shift);
                }
                for (int part = 0; part < 2; part++) {
                  const int component = r + part * 32;
                  float pv, sv;
                  if constexpr (PRED_PACKED) {
                    const uint code = (predCodes >> (part * 12)) & 4095u;
                    const uint exponent = (code >> 7u) & 15u;
                    const ushort value = ushort((code & 127u) | ((code & 2048u) << 4u)
                        | ((exponent == 0u ? 0u : predBase + exponent - 1u) << 7u));
                    pv = cast_to<float>(uint16_to_bfloat16(value));
                  } else { pv = float(pred[(size_t)pr * 256 + component]); }
                  if constexpr (SUCC_PACKED) {
                    const uint code = (succCodes >> (part * 12)) & 4095u;
                    const uint exponent = (code >> 7u) & 15u;
                    const ushort value = ushort((code & 127u) | ((code & 2048u) << 4u)
                        | ((exponent == 0u ? 0u : succBase + exponent - 1u) << 7u));
                    sv = cast_to<float>(uint16_to_bfloat16(value));
                  } else { sv = float(succ[(size_t)c * 256 + component]); }
                  acc += pv * float(proj[p * 256 + component]) * sv;
                }
              }
              acc = simd_sum(acc);
              const int buf = p & 1;
              if (lane == 0) { ev[buf][sg] = tu[p][sg] + acc; }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              const float v = lane < 16u ? ev[buf][lane] : NEG;
              const float m = simd_max(v);
              const uint best = simd_min(v == m ? lane : 16u);
              pr = tc[p][best];
              if (tid == 0) { path[p] = pr; }
            }
            """

    nonisolated(unsafe) private static let cachedSelectorLookupSentinel = MLXArray([UInt16(0)])
    /// One shared `[0]` uint16 stand-in (a fresh one per call with `DARKBLOOM_BONSAI_HOSTCUT=0`).
    private static var selectorLookupSentinel: MLXArray {
        HostConstants.enabled ? cachedSelectorLookupSentinel : MLXArray([UInt16(0)])
    }

    /// `topk` over `[..., V]` logits: `[P, 128]` ids and values. The selectors
    /// read only these pairs, in any order.
    static func top(_ logits: MLXArray) -> [MLXArray] {
        top(logits, kernel: topkFastPassed ? topkFast ?? topk : topk)
    }

    static func top(_ logits: MLXArray, kernel: MLXFast.MLXFastKernel) -> [MLXArray] {
        let V = logits.dim(-1), P = logits.size / V
        return kernel(
            [logits.reshaped([P, V]).asType(.float32)],
            grid: (P * 256, 8, 1), threadGroup: (256, 1, 1),
            outputShapes: [[P, 128], [P, 128]], outputDTypes: [.int32, .float32])
    }

    /// The ids of `top`, for the draft-head screen's partial scores.
    static func shortlist(_ partial: MLXArray) -> MLXArray { top(partial)[0] }

    /// Fused top-16 + greedy selector for `selectGreedy` (B = 1).
    static func select(
        logits: MLXArray, projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
        predLookup: MLXArray? = nil, succLookup: MLXArray? = nil
    ) -> MLXArray {
        select(
            top: top(logits), projected: projected, anchor: anchor, pred: pred, succ: succ,
            predLookup: predLookup, succLookup: succLookup)
    }

    /// `select` on precomputed `top` pairs.
    static func select(
        top: [MLXArray], projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
        predLookup: MLXArray? = nil, succLookup: MLXArray? = nil
    ) -> MLXArray {
        let P = top[0].dim(0)
        let path = selector(
            [top[0], top[1], projected.reshaped([P, -1]).asType(.float32), pred, succ, anchor.asType(.int32),
             predLookup ?? selectorLookupSentinel, succLookup ?? selectorLookupSentinel],
            template: [("PRED_PACKED", predLookup != nil), ("SUCC_PACKED", succLookup != nil)],
            grid: (512, 1, 1), threadGroup: (512, 1, 1),
            outputShapes: [[P]], outputDTypes: [.int32])[0]
        return path.reshaped([1, P])
    }

    /// The selector's edge sum for predecessor token `pr` and candidate `c` at position `p`,
    /// cut verbatim from `selectorSource` (nil when the cut points are not found once each).
    static let selectorEdgeText: String? = {
        let s = selectorSource
        let from = "\n  uint predBase = 0u, succBase = 0u;\n"
        let to = "\n  acc = simd_sum(acc);\n"
        guard s.components(separatedBy: from).count == 2, s.components(separatedBy: to).count == 2,
            let a = s.range(of: from), let b = s.range(of: to), a.lowerBound < b.lowerBound
        else { return nil }
        return String(s[s.index(after: a.lowerBound) ..< b.upperBound])
    }()

    /// Dynamic draft tree (EAGLE-2 style) in place of the greedy walk, `TREE_NODES` nodes for
    /// one 16-row verify. Each expanded node (the anchor first) scores its 16 candidates at the
    /// next position with the selector's own ev (unary + the selector's edge sum, same lane
    /// code) and gives each child its parent's path probability times the softmax of those 16
    /// scores (temperature: `treeSelector`); the frontier node with the highest path probability (lowest insertion slot on a
    /// tie) is expanded next. Rows: 0 = anchor, then the nodes in DFS preorder, children in
    /// expansion order (a chain is the identity). Outputs `tokens` [15] (row order) and `desc`
    /// int32 [333]: parent row [16], depth [16] (anchor 0), ancestor row by depth [16][16]
    /// (-1 past the row's depth), then (depth, candidate, parent) per node in expansion order.
    static let treeSource: String? = selectorEdgeText.flatMap { edge -> String? in
        let s = selectorSource
        guard let cut = s.range(of: "\nthreadgroup float ev[2][16];\n") else { return nil }
        let merge = String(s[...cut.lowerBound])
        let loop = "for (int p = int(sg); p < P; p += 16) {"
        guard merge.components(separatedBy: loop).count == 2 else { return nil }
        return merge.replacingOccurrences(of: loop, with: "for (int p = int(sg); p < P; p += int(NSG)) {")
            .replacingOccurrences(
                of: "const int P = int(cand_shape[0]);",
                with: "const int P = int(cand_shape[0]);\nconst uint NSG = simdgroups_per_threadgroup;\nconst uint NT = threads_per_threadgroup.x;")
            + """
            threadgroup float fprob[256];
            threadgroup float tev[16];
            threadgroup float rv[32];
            threadgroup int ri[32];
            threadgroup int nd[16], nj[16], npar[16];
            for (uint s = tid; s < 256u; s += NT) { fprob[s] = NEG; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int pr = anchor[0];
            int pj = -1;
            int pd = -1;
            float pp = 1.0f;
            float tsc = TREE_SCALE;
            for (int e = 0; e < TREE_NODES; e++) {
              const int p = pd + 1;
              if (p < P) {
                for (uint j = sg; j < 16u; j += NSG) {
                  const int c = tc[p][j];
            \(edge)\
                  if (lane == 0) { tev[j] = tu[p][j] + acc; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < 16u) {
                  float mx = tev[0];
                  for (int c = 1; c < 16; c++) { mx = max(mx, tev[c]); }
                  if (TREE_SPLIT && e == 0) {
                    float z1 = 0.0f;
                    for (int c = 0; c < 16; c++) { z1 += metal::precise::exp(tev[c] - mx); }
                    if (z1 * 0.8f > 1.0f) { tsc = 1.0f; }
                  }
                  float z = 0.0f, mine = 0.0f;
                  for (int c = 0; c < 16; c++) {
                    const float x = metal::precise::exp((tev[c] - mx) * tsc);
                    z += x;
                    if (c == int(tid)) { mine = x; }
                  }
                  fprob[16 * e + int(tid)] = pp * (mine / z);
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              float bv = NEG;
              int bs = 0x7fffffff;
              for (uint s = tid; s < uint(16 * (e + 1)); s += NT) {
                const float v = fprob[s];
                if (v > bv) { bv = v; bs = int(s); }
              }
              const float m = simd_max(bv);
              const int ms = simd_min(bv == m ? bs : 0x7fffffff);
              if (lane == 0) { rv[sg] = m; ri[sg] = ms; }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              float wv = rv[0];
              int ws = ri[0];
              for (uint g = 1; g < NSG; g++) {
                if (rv[g] > wv || (rv[g] == wv && ri[g] < ws)) { wv = rv[g]; ws = ri[g]; }
              }
              const int pe = ws / 16 - 1;
              pj = ws % 16;
              pd = pe < 0 ? 0 : nd[pe] + 1;
              pr = tc[pd][pj];
              pp = wv;
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (tid == 0) { nd[e] = pd; nj[e] = pj; npar[e] = pe; fprob[ws] = NEG; }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
              int order[16];
              int row[16];
              int stk[16];
              int sp = 0, cnt = 0;
              for (int n = TREE_NODES - 1; n >= 0; n--) { if (npar[n] < 0) { stk[sp++] = n; } }
              while (sp > 0) {
                const int u = stk[--sp];
                order[cnt++] = u;
                for (int n = TREE_NODES - 1; n > u; n--) { if (npar[n] == u) { stk[sp++] = n; } }
              }
              for (int i = 0; i < TREE_NODES; i++) { row[order[i]] = i + 1; }
              int par[16];
              int dep[16];
              par[0] = -1;
              dep[0] = 0;
              for (int i = 0; i < TREE_NODES; i++) {
                const int u = order[i];
                tokens[i] = tc[nd[u]][nj[u]];
                par[i + 1] = npar[u] < 0 ? 0 : row[npar[u]];
                dep[i + 1] = nd[u] + 1;
              }
              for (int t = 0; t < 16; t++) {
                desc[t] = t <= TREE_NODES ? par[t] : -1;
                desc[16 + t] = t <= TREE_NODES ? dep[t] : 0;
                for (int d = 0; d < 16; d++) { desc[32 + 16 * t + d] = -1; }
                if (t <= TREE_NODES) {
                  int x = t;
                  for (int d = dep[t]; d >= 0; d--) { desc[32 + 16 * t + d] = x; x = par[x]; }
                }
              }
              for (int n = 0; n < 15; n++) {
                desc[288 + 3 * n] = n < TREE_NODES ? nd[n] : -1;
                desc[289 + 3 * n] = n < TREE_NODES ? nj[n] : -1;
                desc[290 + 3 * n] = n < TREE_NODES ? npar[n] : -1;
              }
            }
            """
    }

    /// The tree's child softmax temperature, chosen per block from the anchor's own softmax at
    /// temperature 1. When the top first-position candidate holds less than 0.8 of it, the block is
    /// likely to stop early, so the tree keeps temperature 1 and spends nodes on side branches. Otherwise
    /// it runs at temperature 0.8 (scores times 1.25, exact in float) and keeps a confident chain's full
    /// depth. `DARKBLOOM_BONSAI_TREE_SPLIT=0` keeps 0.8 on every block;
    /// `DARKBLOOM_BONSAI_TREE_SHARP=0` keeps 1 on every block.
    static let treeSharp: Bool = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE_SHARP"] != "0"
    static let treeSplit: Bool = treeSharp
        && ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE_SPLIT"] != "0"

    static let treeSelector: MLXFast.MLXFastKernel? = treeSource.map { source in
        MLXFast.metalKernel(
            name: treeSplit ? "df2_tree_split_mb8" : (treeSharp ? "df2_tree_t08_mb8" : "df2_tree_mb8"),
            inputNames: ["cand", "unary", "proj", "pred", "succ", "anchor", "predLookup", "succLookup"],
            outputNames: ["tokens", "desc"],
            source: treeConfigured(source))
    }

    /// A tree source with this process's `TREE_SCALE` and `TREE_SPLIT`.
    static func treeConfigured(_ source: String) -> String {
        source.replacingOccurrences(of: "TREE_SCALE", with: treeSharp ? "1.25f" : "1.0f")
            .replacingOccurrences(of: "TREE_SPLIT", with: treeSplit ? "true" : "false")
    }

    /// Verify a draft tree instead of the greedy chain; `DARKBLOOM_BONSAI_TREE=0` keeps the chain.
    static let treeOn: Bool = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE"] != "0"
        && treeSelector != nil

    /// Top-16 + the dynamic tree for one block: `(tokens [1, 15], desc [333])`.
    static func selectTree(
        logits: MLXArray, projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
        predLookup: MLXArray? = nil, succLookup: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        selectTree(
            top: top(logits), projected: projected, anchor: anchor, pred: pred, succ: succ,
            predLookup: predLookup, succLookup: succLookup)
    }

    /// `selectTree` on precomputed `top` pairs: the split kernels once `TreeWide.prepare` qualified them.
    static func selectTree(
        top: [MLXArray], projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
        predLookup: MLXArray? = nil, succLookup: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        if TreeWide.passed, let wide = TreeWide.kernels {
            return wide.select(
                top: top, projected: projected, anchor: anchor, pred: pred, succ: succ,
                predLookup: predLookup, succLookup: succLookup)
        }
        return recordTree(
            top: top, projected: projected, anchor: anchor, pred: pred, succ: succ,
            predLookup: predLookup, succLookup: succLookup)
    }

    /// The single-kernel tree selector (`treeSelector`).
    static func recordTree(
        top: [MLXArray], projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
        predLookup: MLXArray? = nil, succLookup: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let P = top[0].dim(0)
        let outs = recordTreeOutputs(
            top: top, projected: projected, anchor: anchor, pred: pred, succ: succ,
            predLookup: predLookup, succLookup: succLookup)
        return (outs[0].reshaped([1, P]), outs[1])
    }

    /// `treeSelector`'s outputs `[tokens [P], desc [333]]`, unchecked.
    static func recordTreeOutputs(
        top: [MLXArray], projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
        predLookup: MLXArray?, succLookup: MLXArray?
    ) -> [MLXArray] {
        let P = top[0].dim(0)
        precondition(P == 15, "DF2 tree: block of \(P) positions")
        let threads = 512
        return treeSelector!(
            [top[0], top[1], projected.reshaped([P, -1]).asType(.float32), pred, succ, anchor.asType(.int32),
             predLookup ?? selectorLookupSentinel, succLookup ?? selectorLookupSentinel],
            template: [("PRED_PACKED", predLookup != nil), ("SUCC_PACKED", succLookup != nil),
                ("TREE_NODES", 15)],
            grid: (threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[P], [333]], outputDTypes: [.int32, .int32])
    }

    /// `outputs` when it holds `count` arrays. A kernel application that MLX refused returns none,
    /// so the load-time checks throw here instead of trapping on an index.
    static func checkedOutputs(_ outputs: [MLXArray], _ count: Int, _ what: String) throws -> [MLXArray] {
        guard outputs.count == count else {
            throw MLXError.caught("\(what): \(outputs.count) of \(count) outputs")
        }
        return outputs
    }
}

// MARK: - The tree selector as merge, edge and pick kernels

extension DF2Fast {
    /// The tree selector in three kernels. M merges each position's top 16 into device memory (one
    /// simdgroup per position). W scores every (position, predecessor slot, candidate) edge into
    /// `edges` [P][17][16], eight predecessor slots per simdgroup: slot 16 is the anchor (position 0),
    /// slot i is the candidate `tc[p - 1][i]`. W scores rows the record may never expand, so it reads a
    /// codebook row only for an id inside the codebook (row 0 otherwise); the record reads only expanded
    /// rows, and an expanded out-of-range id is out of bounds there too. S' runs the 15 expansions on
    /// one simdgroup, reading the expanded node's edge row by its slot, never by token. The merge, edge, softmax and descriptor
    /// text is cut from `treeSource` and `selectorEdgeText`; `prepare` qualifies the three kernels
    /// against the record kernel once, at bind. `DARKBLOOM_BONSAI_TREE_WIDE=0` keeps the record kernel.
    enum TreeWide {
        static let enabled: Bool = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE_WIDE"] != "0"
        /// Written once by `prepare` at bind, before any proposal; read-only afterwards.
        nonisolated(unsafe) static var passed = false
        nonisolated(unsafe) static var prepared = false
        static let slotsPerGroup: Int = 8

        struct Sources {
            let merge: String
            let edges: String
            let pick: String
            let reference: String
            let pickReference: String
        }

        /// nil when any cut point of the record's sources is not found exactly once.
        static let sources: Sources? = {
            guard let tree = treeSource, let edge = selectorEdgeText, let wide = wideEdge(edge),
                let fprob = tree.range(of: "threadgroup float fprob[256];"),
                tree.components(separatedBy: "threadgroup float fprob[256];").count == 2,
                let softmax = span(
                    tree, from: "    if (tid < 16u) {\n      float mx = tev[0];",
                    through: "      fprob[16 * e + int(tid)] = pp * (mine / z);\n    }\n"),
                let desc = span(tree, from: "if (tid == 0) {\n  int order[16];", through: nil)
            else { return nil }
            let merge: String = String(tree[..<fprob.lowerBound]) + """
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = tid; s < uint(P * 16); s += NT) { mtc[s] = tc[s / 16][s % 16]; mtu[s] = tu[s / 16][s % 16]; }

                """
            let edges: String = "constexpr int KS = \(slotsPerGroup);\n" + """
                const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
                const uint tg = threadgroup_position_in_grid.x;
                const int p = tg == 0u ? 0 : 1 + int(tg - 1u) / (16 / KS);
                const int s0 = tg == 0u ? 16 : KS * int((tg - 1u) % uint(16 / KS));
                const uint j = sg;
                const int predRows = PRED_PACKED ? int(pred_shape[0]) / 96 : int(pred_shape[0]);
                const int succRows = SUCC_PACKED ? int(succ_shape[0]) / 96 : int(succ_shape[0]);
                const int cRaw = mtc[p * 16 + int(j)];
                const int c = cRaw >= 0 && cRaw < succRows
                    && (!SUCC_PACKED || succLookup_shape[0] == 1 || cRaw < int(succLookup_shape[0])) ? cRaw : 0;
                int pr[KS];
                for (int k = 0; k < KS; k++) {
                  const int t = p == 0 ? anchor[0] : mtc[(p - 1) * 16 + s0 + k];
                  pr[k] = t >= 0 && t < predRows
                      && (!PRED_PACKED || predLookup_shape[0] == 1 || t < int(predLookup_shape[0])) ? t : 0;
                }

                """ + wide + """
                for (int k = 0; k < KS; k++) {
                  const float sum = simd_sum(acc[k]);
                  if (lane == 0 && (p > 0 || k == 0)) { edges[(p * 17 + s0 + k) * 16 + int(j)] = mtu[p * 16 + int(j)] + sum; }
                }

                """
            let pick: String = """
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const int P = int(mtc_shape[0]);
                const float NEG = -INFINITY;
                threadgroup int tc[16][16];
                threadgroup float fprob[256];
                threadgroup float tev[16];
                threadgroup int nd[16], nj[16], npar[16];
                for (uint s = lane; s < uint(P * 16); s += 32u) { tc[s / 16][s % 16] = mtc[s]; }
                for (uint s = lane; s < 256u; s += 32u) { fprob[s] = NEG; }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                int pj = -1;
                int pd = -1;
                float pp = 1.0f;
                float tsc = TREE_SCALE;
                for (int e = 0; e < TREE_NODES; e++) {
                  const int p = pd + 1;
                  if (p < P) {
                    if (tid < 16u) { tev[tid] = edges[(p * 17 + (p == 0 ? 16 : pj)) * 16 + int(tid)]; }
                    simdgroup_barrier(mem_flags::mem_threadgroup);

                """ + softmax + """
                  }
                  simdgroup_barrier(mem_flags::mem_threadgroup);
                  float bv = NEG;
                  int bs = 0x7fffffff;
                  for (uint s = lane; s < uint(16 * (e + 1)); s += 32u) {
                    const float v = fprob[s];
                    if (v > bv) { bv = v; bs = int(s); }
                  }
                  const float m = simd_max(bv);
                  const int ws = simd_min(bv == m ? bs : 0x7fffffff);
                  const int pe = ws / 16 - 1;
                  pj = ws % 16;
                  pd = pe < 0 ? 0 : nd[pe] + 1;
                  pp = m;
                  simdgroup_barrier(mem_flags::mem_threadgroup);
                  if (lane == 0) { nd[e] = pd; nj[e] = pj; npar[e] = pe; fprob[ws] = NEG; }
                  simdgroup_barrier(mem_flags::mem_threadgroup);
                }

                """ + desc
            guard let edgeStore = replacingOnce(
                    tree, "if (lane == 0) { tev[j] = tu[p][j] + acc; }",
                    "if (lane == 0) { tev[j] = tu[p][j] + acc; dtev[16 * e + int(j)] = tev[j]; }"),
                let slotStore = replacingOnce(
                    edgeStore, "  const int p = pd + 1;\n",
                    "  const int p = pd + 1;\n  if (tid == 0u) { dslot[e] = p < P ? p * 17 + (p == 0 ? 16 : pj) : -1; }\n"),
                let reference = instrumented(slotStore), let pickReference = instrumented(pick)
            else { return nil }
            return Sources(merge: merge, edges: edges, pick: pick, reference: reference, pickReference: pickReference)
        }()

        static let kernels: Kernels? = sources.map(Kernels.init)

        /// `text` from the start of `from` through the end of `through` (or to the end), each found once.
        static func span(_ text: String, from: String, through: String?) -> String? {
            guard text.components(separatedBy: from).count == 2, let a = text.range(of: from) else { return nil }
            guard let through else { return String(text[a.lowerBound...]) }
            guard text.components(separatedBy: through).count == 2, let b = text.range(of: through),
                a.lowerBound < b.lowerBound
            else { return nil }
            return String(text[a.lowerBound ..< b.upperBound])
        }

        /// `text` with `old` replaced by `new`; nil unless `old` occurs exactly once.
        static func replacingOnce(_ text: String, _ old: String, _ new: String) -> String? {
            guard text.components(separatedBy: old).count == 2 else { return nil }
            return text.replacingOccurrences(of: old, with: new)
        }

        /// `text` split before each anchor; nil unless each anchor occurs once, after the previous one.
        static func pieces(_ text: String, at anchors: [String]) -> [String]? {
            var bounds: [String.Index] = [text.startIndex]
            for anchor in anchors {
                guard text.components(separatedBy: anchor).count == 2, let r = text.range(of: anchor),
                    r.lowerBound > bounds[bounds.count - 1]
                else { return nil }
                bounds.append(r.lowerBound)
            }
            bounds.append(text.endIndex)
            return (1 ..< bounds.count).map { String(text[bounds[$0 - 1] ..< bounds[$0]]) }
        }

        /// `selectorEdgeText` for `KS` predecessors `pr[k]` of one candidate `c`. The predecessor's
        /// lookup, codes and value and the accumulation run once per slot inside the record's component
        /// loop, so the slots' loads overlap; every statement is the record's text, with `pr`,
        /// `predBase`, `predCodes` and `acc` indexed by the slot, so each slot keeps the record's
        /// arithmetic and component order. The caller takes `simd_sum` of each `acc[k]`.
        static func wideEdge(_ edge: String) -> String? {
            guard let e = pieces(edge, at: [
                "  if constexpr (PRED_PACKED) { predBase",
                "  if constexpr (SUCC_PACKED) { succBase",
                "  float acc = 0.0f;\n",
                "  for (int r = int(lane); r < 256; r += 64) {\n",
                "    uint predCodes = 0u, succCodes = 0u;\n",
                "    if constexpr (PRED_PACKED) {\n      const size_t word",
                "    if constexpr (SUCC_PACKED) {\n      const size_t word",
                "      float pv, sv;\n",
                "      if constexpr (PRED_PACKED) {\n        const uint code",
                "      if constexpr (SUCC_PACKED) {\n        const uint code",
                "      acc += pv",
                "    }\n  }\n  acc = simd_sum(acc);\n",
            ]) else { return nil }
            let replaced: [Int: String] = [
                0: "  uint predBase = 0u, succBase = 0u;\n", 3: "  float acc = 0.0f;\n",
                5: "    uint predCodes = 0u, succCodes = 0u;\n", 8: "      float pv, sv;\n",
                12: "    }\n  }\n  acc = simd_sum(acc);\n",
            ]
            guard replaced.allSatisfy({ e[$0.key] == $0.value }) else { return nil }
            func perSlot(_ text: String) -> String {
                ["pr", "predBase", "predCodes", "acc"].reduce(text) { t, name in
                    t.replacingOccurrences(of: "\\b\(name)\\b", with: "\(name)[k]", options: .regularExpression)
                }
            }
            return "uint predBase[KS];\nuint succBase = 0u;\n"
                + "for (int k = 0; k < KS; k++) {\npredBase[k] = 0u;\n" + perSlot(e[1]) + "}\n"
                + e[2]
                + "float acc[KS];\nfor (int k = 0; k < KS; k++) { acc[k] = 0.0f; }\n"
                + e[4]
                + "uint predCodes[KS], succCodes = 0u;\n"
                + "for (int k = 0; k < KS; k++) {\npredCodes[k] = 0u;\n" + perSlot(e[6]) + "}\n"
                + e[7]
                + "float sv;\n" + e[10]
                + "for (int k = 0; k < KS; k++) {\nfloat pv;\n" + perSlot(e[9]) + perSlot(e[11]) + "}\n"
                + "    }\n  }\n"
        }

        /// Stores of the anchor softmax's `z1` and the block's scale, every normalized probability
        /// and every inserted `fprob`, added after the record's statements (stores only).
        static func instrumented(_ text: String) -> String? {
            let split: String = "if (z1 * 0.8f > 1.0f) { tsc = 1.0f; }"
            let insert: String = "fprob[16 * e + int(tid)] = pp * (mine / z);"
            return replacingOnce(text, split, split + "\nif (tid == 0u) { dz1[0] = z1; dsc[0] = tsc; }")
                .flatMap {
                    replacingOnce(
                        $0, insert,
                        insert + " dq[16 * e + int(tid)] = mine / z; dfp[16 * e + int(tid)] = fprob[16 * e + int(tid)];")
                }
        }

        struct Kernels {
            let merger: MLXFast.MLXFastKernel
            let scorer: MLXFast.MLXFastKernel
            let picker: MLXFast.MLXFastKernel
            let reference: MLXFast.MLXFastKernel
            let pickerReference: MLXFast.MLXFastKernel

            init(_ s: Sources) {
                let tag: String = treeSplit ? "split" : (treeSharp ? "t08" : "t1")
                let recordInputs: [String] = ["cand", "unary", "proj", "pred", "succ", "anchor", "predLookup", "succLookup"]
                let probes: [String] = ["dz1", "dsc", "dq", "dfp"]
                merger = MLXFast.metalKernel(
                    name: "df2_tree_wide_merge", inputNames: ["cand", "unary"], outputNames: ["mtc", "mtu"],
                    source: s.merge)
                scorer = MLXFast.metalKernel(
                    name: "df2_tree_wide_edges_ks\(slotsPerGroup)",
                    inputNames: ["mtc", "mtu", "proj", "pred", "succ", "anchor", "predLookup", "succLookup"],
                    outputNames: ["edges"], source: s.edges)
                picker = MLXFast.metalKernel(
                    name: "df2_tree_wide_pick_\(tag)", inputNames: ["mtc", "edges"], outputNames: ["tokens", "desc"],
                    source: treeConfigured(s.pick))
                reference = MLXFast.metalKernel(
                    name: "df2_tree_wide_reference_\(tag)", inputNames: recordInputs,
                    outputNames: ["tokens", "desc", "dtev", "dslot"] + probes, source: treeConfigured(s.reference))
                pickerReference = MLXFast.metalKernel(
                    name: "df2_tree_wide_pick_reference_\(tag)", inputNames: ["mtc", "edges"],
                    outputNames: ["tokens", "desc"] + probes, source: treeConfigured(s.pickReference))
            }

            /// M, then W on M's arrays: `mtc`, `mtu` [P, 16] and `edges` [P * 17 * 16]. Throws when MLX
            /// refused an application.
            func tables(
                top: [MLXArray], projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
                predLookup: MLXArray?, succLookup: MLXArray?
            ) throws -> (mtc: MLXArray, mtu: MLXArray, edges: MLXArray) {
                let P: Int = top[0].dim(0)
                let merged: [MLXArray] = try checkedOutputs(merger(
                    [top[0], top[1]], grid: (512, 1, 1), threadGroup: (512, 1, 1),
                    outputShapes: [[P, 16], [P, 16]], outputDTypes: [.int32, .float32]), 2, "tree wide merge")
                let groups: Int = 1 + (P - 1) * (16 / slotsPerGroup)
                let edges: [MLXArray] = try checkedOutputs(scorer(
                    [merged[0], merged[1], projected.reshaped([P, -1]).asType(.float32), pred, succ,
                     anchor.asType(.int32), predLookup ?? selectorLookupSentinel, succLookup ?? selectorLookupSentinel],
                    template: [("PRED_PACKED", predLookup != nil), ("SUCC_PACKED", succLookup != nil)],
                    grid: (groups * 512, 1, 1), threadGroup: (512, 1, 1),
                    outputShapes: [[P * 17 * 16]], outputDTypes: [.float32]), 1, "tree wide edges")
                return (merged[0], merged[1], edges[0])
            }

            /// M -> W -> S': `(tokens [1, 15], desc [333])`, as `recordTree`, or `recordTree` itself
            /// when MLX refused an application.
            func select(
                top: [MLXArray], projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
                predLookup: MLXArray?, succLookup: MLXArray?
            ) -> (MLXArray, MLXArray) {
                let P: Int = top[0].dim(0)
                precondition(P == 15, "DF2 tree: block of \(P) positions")
                if let t = try? tables(
                    top: top, projected: projected, anchor: anchor, pred: pred, succ: succ,
                    predLookup: predLookup, succLookup: succLookup)
                {
                    let outs: [MLXArray] = picker(
                        [t.mtc, t.edges], template: [("TREE_NODES", 15)], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                        outputShapes: [[P], [333]], outputDTypes: [.int32, .int32])
                    if outs.count == 2 { return (outs[0].reshaped([1, P]), outs[1]) }
                }
                return recordTree(
                    top: top, projected: projected, anchor: anchor, pred: pred, succ: succ,
                    predLookup: predLookup, succLookup: succLookup)
            }
        }

        // MARK: Qualification

        struct Codebooks {
            let name: String
            let pred: MLXArray
            let succ: MLXArray
            let predLookup: MLXArray?
            let succLookup: MLXArray?
            let rows: Int
        }

        enum Kind: String, CaseIterable {
            case random, wide, ties, repeatedIds, flat, underflow, cancellation
            case splitBelow, splitAt, splitAbove, chain, branching
        }

        struct Case {
            let kind: Kind
            let ids: [Int32]
            let values: [Float]
            let proj: [Float]
            let anchor: Int32
        }

        struct Rng {
            var state: UInt64
            mutating func next() -> UInt64 {
                state &+= 0x9E37_79B9_7F4A_7C15
                var z: UInt64 = state
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                return z ^ (z >> 31)
            }
            mutating func unit() -> Float { Float(next() >> 40) / 16_777_216 }
            /// Roughly standard normal (four uniforms).
            mutating func normal() -> Float { (unit() + unit() + unit() + unit() - 2) * 1.7320508 }
        }

        /// One block of inputs. Ids in a row are distinct (`repeatedIds`: the last 8 repeat the first 8,
        /// all 16 ranked on top), so no two lanes hold the same (value, id) pair. The split cases put the
        /// anchor softmax at z1 = 1 + 15 exp(-d) around the 1.25 threshold (zero edges at position 0).
        /// `gain` scales `proj` so an edge sum at scale 1 has unit RMS for the codebooks in use.
        static func makeCase(_ kind: Kind, rows: Int, seed: UInt64, gain: Float) -> Case {
            var rng = Rng(state: seed)
            var ids = [Int32](repeating: 0, count: 15 * 128)
            var values = [Float](repeating: 0, count: 15 * 128)
            var proj = [Float](repeating: 0, count: 15 * 256)
            let ln60: Float = 4.0943446
            let gap: Float = kind == .splitBelow ? ln60 - 0.001 : (kind == .splitAbove ? ln60 + 0.001 : ln60)
            for p in 0 ..< 15 {
                let start: Int = Int(rng.next() % UInt64(rows))
                for i in 0 ..< 128 {
                    let source: Int = kind == .repeatedIds && i >= 120 ? i - 120 : i
                    ids[p * 128 + i] = Int32((start + source * 7919) % rows)
                    let n: Float = rng.normal()
                    let v: Float
                    switch kind {
                    case .random, .wide, .underflow: v = 3 * n
                    case .ties: v = (6 * n).rounded() / 2
                    case .repeatedIds: v = i < 8 || i >= 120 ? 5 + n : n - 2
                    case .flat: v = 1
                    case .cancellation: v = 1000 + n
                    case .splitBelow, .splitAt, .splitAbove:
                        v = p > 0 ? 3 * n : (i == 0 ? 10 : (i < 16 ? 10 - gap : -100))
                    case .chain: v = i == 0 ? 20 : n - 10
                    case .branching: v = i == 0 ? 5 : (i == 1 ? 4.95 : n - 5)
                    }
                    values[p * 128 + i] = v
                }
                let scale: Float
                switch kind {
                case .random, .ties, .repeatedIds: scale = 1
                case .wide: scale = 4
                case .flat: scale = 0
                case .underflow: scale = 40
                case .cancellation: scale = 1000
                case .splitBelow, .splitAt, .splitAbove: scale = p == 0 ? 0 : 1
                case .chain: scale = 0.1
                case .branching: scale = 0.05
                }
                for k in 0 ..< 256 { proj[p * 256 + k] = scale * gain * rng.normal() }
            }
            return Case(kind: kind, ids: ids, values: values, proj: proj, anchor: Int32(rng.next() % UInt64(rows)))
        }

        /// BF16 rows of magnitude 2^-4 to 2^2 (some zero). `spread` scales row r by 2^(r % 20 - 10), so
        /// `DFlash2PackedCodebook` keeps one exponent base per row instead of one shared base.
        static func syntheticCodebook(rows: Int, seed: UInt64, spread: Bool) -> MLXArray {
            var rng = Rng(state: seed)
            var v = [Float](repeating: 0, count: rows * 256)
            for i in 0 ..< v.count {
                let shift: Int = spread ? i / 256 % 20 - 10 : 0
                let sign: Float = rng.unit() < 0.5 ? -1 : 1
                let exponent: Int = Int(rng.unit() * 6) - 4 + shift
                v[i] = i % 37 == 0 ? 0 : sign * (1 + rng.unit()) * Float(sign: .plus, exponent: exponent, significand: 1)
            }
            return MLXArray(v, [rows, 256]).asType(.bfloat16)
        }

        /// The drafter's own codebooks as bound, and synthetic ones covering both packings, both
        /// lookup forms (one shared exponent base, one per row) and mixed packings.
        static func codebookSets(_ selector: DFlash2CandidateSelector, vocabulary: Int) throws -> [Codebooks]? {
            let rows: Int = 4096
            let narrow: MLXArray = syntheticCodebook(rows: rows, seed: 84, spread: false)
            let spread: MLXArray = syntheticCodebook(rows: rows, seed: 85, spread: true)
            guard let n = try DFlash2PackedCodebook(narrow), let s = try DFlash2PackedCodebook(spread),
                n.lookup.size == 1, s.lookup.size == rows
            else { return nil }
            return [
                Codebooks(
                    name: "drafter", pred: selector.predecessorCodebook, succ: selector.successorCodebook,
                    predLookup: selector.predecessorLookup, succLookup: selector.successorLookup, rows: vocabulary),
                Codebooks(
                    name: "packed, shared/per-row bases", pred: n.packed, succ: s.packed,
                    predLookup: n.lookup, succLookup: s.lookup, rows: rows),
                Codebooks(
                    name: "packed, per-row/shared bases", pred: s.packed, succ: n.packed,
                    predLookup: s.lookup, succLookup: n.lookup, rows: rows),
                Codebooks(name: "unpacked", pred: narrow, succ: spread, predLookup: nil, succLookup: nil, rows: rows),
                Codebooks(
                    name: "packed pred, unpacked succ", pred: n.packed, succ: spread,
                    predLookup: n.lookup, succLookup: nil, rows: rows),
                Codebooks(
                    name: "unpacked pred, packed succ", pred: narrow, succ: s.packed,
                    predLookup: nil, succLookup: s.lookup, rows: rows),
            ]
        }

        static func same(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            MLX.all(a.reshaped([-1]).view(dtype: .uint32) .== b.reshaped([-1]).view(dtype: .uint32))
        }

        /// One case: the record kernel, its instrumented copy, M -> W -> S' and S' instrumented on the
        /// same W table. Returns (every check equal, the block took scale 1).
        static func check(_ k: Kernels, _ books: Codebooks, _ c: Case) throws -> (MLXArray, MLXArray) {
            let top: [MLXArray] = [MLXArray(c.ids, [15, 128]), MLXArray(c.values, [15, 128])]
            let projected: MLXArray = MLXArray(c.proj, [15, 256])
            let anchor: MLXArray = MLXArray([c.anchor])
            let record: [MLXArray] = try checkedOutputs(recordTreeOutputs(
                top: top, projected: projected, anchor: anchor, pred: books.pred, succ: books.succ,
                predLookup: books.predLookup, succLookup: books.succLookup), 2, "tree record")
            let tokens: MLXArray = record[0].reshaped([1, 15]), desc: MLXArray = record[1]
            let ref: [MLXArray] = try checkedOutputs(k.reference(
                [top[0], top[1], projected, books.pred, books.succ, anchor,
                 books.predLookup ?? selectorLookupSentinel, books.succLookup ?? selectorLookupSentinel],
                template: [("PRED_PACKED", books.predLookup != nil), ("SUCC_PACKED", books.succLookup != nil),
                    ("TREE_NODES", 15)],
                grid: (512, 1, 1), threadGroup: (512, 1, 1),
                outputShapes: [[15], [333], [240], [15], [1], [1], [240], [240]],
                outputDTypes: [.int32, .int32, .float32, .int32, .float32, .float32, .float32, .float32],
                initValue: 0), 8, "tree reference")
            let t = try k.tables(
                top: top, projected: projected, anchor: anchor, pred: books.pred, succ: books.succ,
                predLookup: books.predLookup, succLookup: books.succLookup)
            let shipped: [MLXArray] = try checkedOutputs(k.picker(
                [t.mtc, t.edges], template: [("TREE_NODES", 15)], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[15], [333]], outputDTypes: [.int32, .int32]), 2, "tree wide pick")
            let probe: [MLXArray] = try checkedOutputs(k.pickerReference(
                [t.mtc, t.edges], template: [("TREE_NODES", 15)], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[15], [333], [1], [1], [240], [240]],
                outputDTypes: [.int32, .int32, .float32, .float32, .float32, .float32], initValue: 0),
                6, "tree wide pick reference")
            let expanded: MLXArray = take(t.edges.reshaped([-1, 16]), MLX.maximum(ref[3], 0), axis: 0)
            let equal: [MLXArray] = [
                same(ref[0], tokens), same(ref[1], desc),
                same(shipped[0], tokens), same(shipped[1], desc),
                same(probe[0], tokens), same(probe[1], desc),
                MLX.all(ref[3] .>= 0), same(expanded, ref[2]),
                same(probe[2], ref[4]), same(probe[3], ref[5]), same(probe[4], ref[6]), same(probe[5], ref[7]),
            ]
            return (MLX.all(MLX.stacked(equal)), MLX.all(ref[5] .== 1))
        }

        /// 1 / RMS of W's edge sums for a `.random` case at gain 1 (the codebooks' edge scale).
        static func edgeGain(_ k: Kernels, _ books: Codebooks) throws -> Float {
            let c: Case = makeCase(.random, rows: books.rows, seed: 7, gain: 1)
            let t = try k.tables(
                top: [MLXArray(c.ids, [15, 128]), MLXArray(c.values, [15, 128])],
                projected: MLXArray(c.proj, [15, 256]), anchor: MLXArray([c.anchor]), pred: books.pred,
                succ: books.succ, predLookup: books.predLookup, succLookup: books.succLookup)
            let edges: MLXArray = t.edges.reshaped([15, 17, 16])[1..., 0 ..< 16, 0...]
            let sums: MLXArray = edges - t.mtu[1...].reshaped([14, 1, 16])
            let rms: Float = MLX.sqrt(MLX.mean(sums * sums)).item(Float.self)
            return rms.isFinite && rms > 0 ? 1 / rms : 1
        }

        static func qualify(_ k: Kernels, selector: DFlash2CandidateSelector, vocabulary: Int) throws -> (Bool, String) {
            guard let sets = try codebookSets(selector, vocabulary: vocabulary) else {
                return (false, "synthetic codebooks did not pack as intended")
            }
            var names: [String] = []
            var equal: [MLXArray] = []
            var scaleOne: [MLXArray] = []
            for (n, books) in sets.enumerated() {
                let gain: Float = try edgeGain(k, books)
                for (m, kind) in Kind.allCases.enumerated() {
                    for seed in kind == .random ? [0, 1, 2] : [0] {
                        let c: Case = makeCase(
                            kind, rows: books.rows, seed: UInt64(1000 * n + 10 * m + seed), gain: gain)
                        let (ok, one) = try check(k, books, c)
                        names.append("\(books.name)/\(kind.rawValue)/\(seed)")
                        equal.append(ok)
                        scaleOne.append(one)
                    }
                }
            }
            let oks: [Bool] = MLX.stacked(equal).asArray(Bool.self)
            let ones: Int = scaleOne.map { $0.item(Bool.self) ? 1 : 0 }.reduce(0, +)
            if let bad = oks.firstIndex(of: false) {
                return (false, "mismatch at \(names[bad]) (\(oks.filter { !$0 }.count) of \(oks.count) cases)")
            }
            let split: String = treeSplit ? "; block scale 1 in \(ones) of \(oks.count)" : ""
            return (true, "\(oks.count) cases over \(sets.count) codebook sets bitwise equal to the record kernel"
                + " (tokens, descriptor, every expanded edge, z1, scale, probabilities, fprob)\(split)")
        }

        /// Runs once at drafter bind, before any proposal or measured window. Calls the kernels
        /// directly, never `selectTree`, and sets `passed` only after every check evaluated equal;
        /// any mismatch or MLX error keeps the record kernel.
        static func prepare(selector: DFlash2CandidateSelector, vocabulary: Int) {
            guard !prepared else { return }
            prepared = true
            var note: String = "off (DARKBLOOM_BONSAI_TREE_WIDE=0)"
            if enabled && !treeOn {
                note = "off (tree selector off)"
            } else if enabled, let kernels {
                let start: Date = Date()
                do {
                    let (ok, detail) = try withError { try qualify(kernels, selector: selector, vocabulary: vocabulary) }
                    let seconds: String = String(format: "%.1f", Date().timeIntervalSince(start))
                    passed = ok
                    note = (ok ? "on (" : "off (") + detail + ", \(seconds) s)"
                } catch {
                    note = "off (error: \(error))"
                }
            } else if enabled {
                note = "off (a cut of the record source did not match once)"
            }
            FileHandle.standardError.write(Data("DF2 tree wide: \(note)\n".utf8))
        }
    }
}

/// See `DFlash2DraftModel.projectContext`.
public final class DFlash2ContextProjection {
    let anchorEmbeddings: MLXArray
    let context: MLXArray
    let contextKV: [(MLXArray, MLXArray)]
    let lookupFCOutput: MLXArray?
    var front: DFlash2LayerZeroFront?

    init(
        anchorEmbeddings: MLXArray, context: MLXArray, contextKV: [(MLXArray, MLXArray)],
        lookupFCOutput: MLXArray?
    ) {
        self.anchorEmbeddings = anchorEmbeddings
        self.context = context
        self.contextKV = contextKV
        self.lookupFCOutput = lookupFCOutput
    }

    /// The columns the projection covers.
    public var rows: Int { context.dim(1) }

    public var evaluationTargets: [MLXArray] {
        var arrays = [anchorEmbeddings, context]
        if let lookupFCOutput { arrays.append(lookupFCOutput) }
        for (keys, values) in contextKV {
            arrays.append(keys)
            arrays.append(values)
        }
        return arrays
    }

    /// The first `rows` context rows, and the anchor at column `rows - 1`.
    struct Use {
        let anchorEmbedding: MLXArray
        let context: MLXArray
        let contextKV: [(MLXArray, MLXArray)]
        let front: DFlash2LayerZeroFront?
    }

    /// The projection of the verify columns `rows` (a draft tree's accepted path), in order.
    func gathered(_ rows: [Int32]) -> DFlash2ContextProjection {
        let index = MLXArray(rows)
        return DFlash2ContextProjection(
            anchorEmbeddings: anchorEmbeddings.take(index, axis: 1),
            context: context.take(index, axis: 1),
            contextKV: contextKV.map { ($0.0.take(index, axis: 1), $0.1.take(index, axis: 1)) },
            lookupFCOutput: lookupFCOutput?.take(index, axis: 1))
    }

    func use(rows: Int) -> Use {
        let kv = contextKV.map { pair -> (MLXArray, MLXArray) in
            (pair.0[0..., 0 ..< rows, 0...], pair.1[0..., 0 ..< rows, 0...])
        }
        return Use(
            anchorEmbedding: anchorEmbeddings[0..., (rows - 1) ..< rows, 0...],
            context: context[0..., 0 ..< rows, 0...],
            contextKV: kv,
            front: front)
    }
}

/// Layer 0's front of the next proposal, built on the GPU behind the verify
/// (`DFlash2DraftModel.projectLayerZeroFront`): the block input with its anchor
/// gathered by the device acceptance count, layer 0's input norm and codes, a
/// rotation table for every context width up to 16, the attention conv kernel
/// projection and q, k, v before their norm and rotation. Valid for a proposal
/// whose caches all sit at `cacheOffset`.
final class DFlash2LayerZeroFront {
    let cacheOffset: Int
    let h: MLXArray
    let normed: MLXArray
    /// Layer 0's input codes, kept only for the load check (p1 replaces them at runtime).
    let codes: DF2Fast.W4Codes?
    let ropeSpec: DF2Fast.RopeSpec
    let p1: MLXArray
    let queries: MLXArray
    let keys: MLXArray
    let values: MLXArray

    init(
        cacheOffset: Int, h: MLXArray, normed: MLXArray, codes: DF2Fast.W4Codes?,
        ropeSpec: DF2Fast.RopeSpec, p1: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        self.cacheOffset = cacheOffset
        self.h = h
        self.normed = normed
        self.codes = codes
        self.ropeSpec = ropeSpec
        self.p1 = p1
        self.queries = queries
        self.keys = keys
        self.values = values
    }

    var evaluationTargets: [MLXArray] {
        var arrays: [MLXArray] = [h, normed, p1, queries, keys, values]
        if let phases = ropeSpec.phases { arrays.append(phases) }
        return arrays
    }
}

extension DFlash2DraftModel {
    /// Resolve `DF2Fast.frontMatched` once at bind, after the warmup filled the
    /// mask embedding cache. On synthetic rounds (accepted 0, 7 and 15, a
    /// mismatching draft followed by matching ones where one exists; cache
    /// offsets 0 and 37) the front must equal the record path bit for bit: its
    /// pieces, layer 0's normed and rotated q|k|v under both phase tables, and
    /// a whole proposal's draft ids, final hidden state and codes.
    func checkLayerZeroFront() {
        var note = "off"
        if DF2Fast.frontEnabled {
            do {
                note = try layerZeroFrontMismatch().map { "failed (\($0))" } ?? "matched"
            } catch {
                note = "failed (\(error))"
            }
        }
        DF2Fast.frontMatched = note == "matched"
        FileHandle.standardError.write(Data("DF2 front2 check: \(note)\n".utf8))
    }

    private func layerZeroFrontMismatch() throws -> String? {
        let blockSize: Int = config.dflash.blockSize
        guard blockSize == 16 else { return "block size \(blockSize)" }
        let k: Int = blockSize - 1
        let width: Int = config.targetHiddenSize
        let tokens: [Int32] = (0 ..< blockSize).map { Int32(1000 + 37 * $0) }
        let ramp: MLXArray = MLXArray(0 ..< (blockSize * width)).asType(.float32)
        let wave: MLXArray = MLX.sin(ramp * 0.7548776) * 2
        let hidden: MLXArray = wave.reshaped([1, blockSize, width]).asType(dtype)
        for accepted in [0, 7, 15] {
            var drafts: [Int32] = Array(tokens[0 ..< k])
            if accepted < k { drafts[accepted] += 1 }
            for offset in [0, 37] {
                if let note = try frontMismatch(
                    hidden: hidden, tokens: tokens, drafts: drafts, accepted: accepted, offset: offset)
                {
                    return "accepted \(accepted), offset \(offset): \(note)"
                }
            }
        }
        return nil
    }

    /// The first output that differs, or nil.
    private func frontMismatch(
        hidden: MLXArray, tokens: [Int32], drafts: [Int32], accepted: Int, offset: Int
    ) throws -> String? {
        let targets = MLXArray(tokens)
        guard let projection = projectContext(targetHidden: hidden, anchors: targets) else {
            return "no projection"
        }
        guard
            let front = projectLayerZeroFront(
                projection, draftTokens: MLXArray(drafts), targetTokens: targets, cacheOffset: offset,
                retainCodes: true),
            let frontCodes = front.codes
        else { return "no front" }
        let rows: Int = accepted + 1
        let L: Int = config.dflash.blockSize
        let use = projection.use(rows: rows)
        let layer = layers[0]
        let attention = layer.selfAttn
        let projections: [Linear] = [attention.qProj, attention.kProj, attention.vProj]
        guard let group = DF2Fast.w4Group, let baseSpec = DF2Fast.ropeSpec(self),
            let mask = DF2Fast.maskEmbedCache[ObjectIdentifier(self)]
        else { return "no record operands" }
        let h: MLXArray = concatenated([use.anchorEmbedding, mask], axis: 1)
        guard
            let record = DF2Fast.rmsQuantPhases(
                h, layer.inputLayerNorm, group: group, ropeSpec: baseSpec, C: rows, offset: offset),
            let p1 = DF2Fast.w4LinearsFromCodes(
                record.1, leadShape: [1, L], [layer.attentionConv.kernelProjection])?.first,
            let qkv = DF2Fast.linearsConv0(
                record.0, dyn: p1, base: layer.attentionConv.fastBase, projections),
            let recordPhases = record.2.phases, let frontPhases = front.ropeSpec.phases
        else { return "no record pieces" }
        let contextKV = use.contextKV[0]
        guard
            let recordRotated = DF2Fast.qkvAllNormRope(
                q: qkv[0], k: qkv[1], v: qkv[2], ck: contextKV.0, cv: contextKV.1,
                qNorm: attention.qNorm, kNorm: attention.kNorm, ropeSpec: record.2,
                L: L, C: rows, blockOffset: offset + rows, ctxOffset: offset),
            let frontRotated = DF2Fast.qkvAllNormRope(
                q: front.queries, k: front.keys, v: front.values, ck: contextKV.0, cv: contextKV.1,
                qNorm: attention.qNorm, kNorm: attention.kNorm, ropeSpec: front.ropeSpec,
                L: L, C: rows, blockOffset: offset + rows, ctxOffset: offset)
        else { return "no rotation" }

        let anchor = Int(tokens[accepted])
        let frontRun = try proposalOutputs(
            projection, front: front, rows: rows, anchor: anchor, hidden: hidden, offset: offset)
        guard frontRun.used else { return "front not consumed" }
        let recordRun = try proposalOutputs(
            projection, front: nil, rows: rows, anchor: anchor, hidden: hidden, offset: offset)

        var pairs: [(String, MLXArray, MLXArray)] = [
            ("h", front.h, h), ("normed", front.normed, record.0),
            ("xq", frontCodes.xq, record.1.xq), ("ascale", frontCodes.ascale, record.1.ascale),
            ("xs", frontCodes.xs, record.1.xs),
            ("phases", frontPhases[0 ..< (rows + L)], recordPhases),
            ("p1", front.p1, p1), ("q", front.queries, qkv[0]), ("k", front.keys, qkv[1]),
            ("v", front.values, qkv[2]),
        ]
        pairs.append(("rotated q", frontRotated.0, recordRotated.0))
        pairs.append(("rotated k", frontRotated.1, recordRotated.1))
        pairs.append(("rotated v", frontRotated.2, recordRotated.2))
        pairs.append(("draft ids", frontRun.tokens, recordRun.tokens))
        pairs.append(("final hidden", frontRun.hidden, recordRun.hidden))
        switch (frontRun.codes, recordRun.codes) {
        case (.some(let a), .some(let b)):
            pairs.append(("final xq", a.xq, b.xq))
            pairs.append(("final ascale", a.ascale, b.ascale))
            pairs.append(("final xs", a.xs, b.xs))
        case (.none, .none):
            break
        default:
            return "final codes"
        }
        let checks: [MLXArray] = pairs.map { Self.bitsEqual($0.1, $0.2) }
        eval(checks)
        for (pair, check) in zip(pairs, checks) where !check.item(Bool.self) {
            return pair.0
        }
        return nil
    }

    /// One proposal over the projection's first `rows` columns on fresh caches
    /// at `offset`, with or without the front, and whether the front was taken.
    private func proposalOutputs(
        _ projection: DFlash2ContextProjection, front: DFlash2LayerZeroFront?, rows: Int,
        anchor: Int, hidden: MLXArray, offset: Int
    ) throws -> (tokens: MLXArray, hidden: MLXArray, codes: DF2Fast.W4Codes?, used: Bool) {
        let blockSize: Int = config.dflash.blockSize
        let cache = try makeCache()
        for layerCache in cache {
            (layerCache as? BaseKVCache)?.offset = offset
        }
        projection.front = front
        defer { projection.front = nil }
        let block = MLXArray.zeros([1, blockSize], dtype: .int32)
        let used = usableFront(projection.use(rows: rows), inputs: block, cache: cache, logitsStart: 1) != nil
        let context: MLXArray = hidden[0..., 0 ..< rows, 0...]
        let run = try proposeWithHidden(
            anchor: [anchor], targetHidden: context, cache: cache, blockSize: blockSize,
            projection: projection)
        return (run.tokens, run.hidden, run.codes, used)
    }

    /// Every bit equal. BF16, FP32, int8 and token ids all widen to FP32 exactly.
    private static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        guard a.shape == b.shape, a.dtype == b.dtype else { return MLXArray(false) }
        let x: MLXArray = contiguous(a.asType(.float32)).view(dtype: .uint32)
        let y: MLXArray = contiguous(b.asType(.float32)).view(dtype: .uint32)
        return MLX.all(x .== y)
    }
}

/// Serves `DF2WideLaneSource` for the drafter's per-row kernels. Each input combination (kernel,
/// dtypes, width, group, form) runs wide only after a load-time check found every output bit equal to
/// the record kernel's; anything unchecked, failed or erroring keeps the record kernel.
/// `DF2_WIDE=0` keeps every record kernel. The default serves the conv and row-norm kernels
/// (`df2_act_q8g_conv0`, `df2_rms_q8g`, `df2_dyn_conv_rms_q8g`) at 512 lanes per 4096-column row;
/// a value containing `all` adds `df2_act_q8g`, one containing `1024` uses 1024 lanes.
enum DF2WideLanes {
    typealias Kind = DF2WideLaneSource.Kind

    struct Combo: Hashable {
        let kind: Kind
        let dtype: DType
        let dtype2: DType
        let k: Int
        let group: Int
        let flag: Int
    }

    static let setting: String =
        (ProcessInfo.processInfo.environment["DF2_WIDE"] ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    static let enabled = !["0", "false", "no", "off"].contains(setting)
    static let e = setting.contains("1024") ? 4 : 8
    /// `df2_act_q8g` runs wide by default too (still only after its load check);
    /// `DARKBLOOM_BONSAI_DF2_QUANTG_WIDE=0` restores the `all`-only opt-in.
    static let quantGDefault =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DF2_QUANTG_WIDE"] != "0"
    static func offered(_ kind: Kind) -> Bool {
        enabled && (kind != .quantG || quantGDefault || setting.contains("all"))
    }

    static let quantG = MLXFast.metalKernel(
        name: "df2_act_q8g_e\(e)", inputNames: ["x", "x2", "ksz"], outputNames: ["xq", "ascale", "xs"],
        source: DF2WideLaneSource.source(.quantG, e: e))
    static let quantGConv0 = MLXFast.metalKernel(
        name: "df2_act_q8g_conv0_e\(e)", inputNames: ["h", "dyn", "base", "ksz"],
        outputNames: ["xq", "ascale", "xs"], source: DF2WideLaneSource.source(.quantGConv0, e: e))
    static let rmsQuant = MLXFast.metalKernel(
        name: "df2_rms_q8g_e\(e)", inputNames: ["x", "weight", "dsz", "eps_arr"],
        outputNames: ["out_norm", "xq", "ascale", "xs"], source: DF2WideLaneSource.source(.rmsQuant, e: e))
    static let convRmsQuant = MLXFast.metalKernel(
        name: "df2_dyn_conv_rms_q8g_e\(e)",
        inputNames: ["h", "dyn", "base", "res", "weight", "dsz", "eps_arr"],
        outputNames: ["out_res", "out_norm", "xq", "ascale", "xs"],
        source: DF2WideLaneSource.source(.convRmsQuant, e: e))

    /// Written once by `prepare` at bind, before any proposal; read-only afterwards.
    nonisolated(unsafe) static var passed: Set<Combo> = []
    nonisolated(unsafe) static var prepared = false

    static func serves(_ combo: Combo) -> Bool {
        enabled && passed.contains(combo)
    }

    private static func log(_ s: String) {
        FileHandle.standardError.write(Data("DF2 wide lanes: \(s)\n".utf8))
    }

    /// Deterministic synthetic rows: row 0 all zero, row 1 at 30x, the rest O(1).
    private static func rows(_ r: Int, _ width: Int, seed: Float, dtype: DType = .float32) -> MLXArray {
        let i = MLXArray(0 ..< (r * width)).asType(.float32)
        let v = MLX.sin(i * (0.7548776 + seed)) * (1 + 0.5 * MLX.cos(i * 0.0137 + seed)) * 2
        let scale = MLXArray((0 ..< r).map { Float($0 == 0 ? 0 : ($0 == 1 ? 30 : 1)) }).reshaped([r, 1])
        return (v.reshaped([r, width]) * scale).asType(dtype)
    }

    private static func same(_ a: [MLXArray], _ b: [MLXArray], skip: Set<Int> = []) -> MLXArray {
        var eq: [MLXArray] = []
        for (j, (x, y)) in zip(a, b).enumerated() where !skip.contains(j) {
            eq.append(MLX.all(x.view(dtype: .uint32) .== y.view(dtype: .uint32)))
        }
        return MLX.all(MLX.stacked(eq))
    }

    /// One combination at 3, 11 and 16 rows: record kernel vs wide kernel, every output bit.
    private static func check(_ c: Combo) -> Bool {
        var eqs: [MLXArray] = []
        for r in [3, 11, 16] {
            switch c.kind {
            case .quantG:
                let swiglu = c.flag & 1 != 0, headMajor = c.flag & 2 != 0
                let x = rows(r, c.k, seed: 0.01, dtype: c.dtype)
                let x2 = swiglu ? rows(r, c.k, seed: 0.02, dtype: c.dtype2) : x
                let xin = headMajor ? x.reshaped([r, 32, 128]).transposed(1, 0, 2).reshaped([1, 32, r, 128]) : x
                let x2in = headMajor ? xin : x2
                let a = DF2Fast.quantGRun(xin, x2in, K: c.k, rows: r, group: c.group, headMajor: headMajor, swiglu: swiglu, wide: false)
                let b = DF2Fast.quantGRun(xin, x2in, K: c.k, rows: r, group: c.group, headMajor: headMajor, swiglu: swiglu, wide: true)
                eqs.append(same(a, b))
            case .quantGConv0:
                let h = rows(r, c.k, seed: 0.03, dtype: c.dtype)
                let dyn = (rows(r, c.k / 4, seed: 0.04) * 0.2).asType(c.dtype2)
                let base = (rows(4, c.k, seed: 0.05)[2...] * 0.2).reshaped([-1]).asType(c.flag == 0 ? .float32 : .bfloat16)
                let a = DF2Fast.conv0Run(h, dyn: dyn, base: base, K: c.k, rows: r, group: c.group, wide: false)
                let b = DF2Fast.conv0Run(h, dyn: dyn, base: base, K: c.k, rows: r, group: c.group, wide: true)
                eqs.append(same(a, b))
            case .rmsQuant:
                let x = rows(r, c.k, seed: 0.06, dtype: c.dtype).reshaped([1, r, c.k])
                let w = (MLX.abs(rows(2, c.k, seed: 0.07)[1]) * 0.25 + 0.5).asType(c.dtype2)
                let a = DF2Fast.rmsQuantRun(x, w, D: c.k, L: r, eps: 1e-6, group: c.group, wide: false)
                let b = DF2Fast.rmsQuantRun(x, w, D: c.k, L: r, eps: 1e-6, group: c.group, wide: true)
                eqs.append(same(a, b))
            case .convRmsQuant:
                let qStart = c.flag & 1 != 0 ? 0 : 1
                let h = rows(r, c.k, seed: 0.08, dtype: c.dtype).reshaped([1, r, c.k])
                let res = rows(r, c.k, seed: 0.09, dtype: c.dtype2).reshaped([1, r, c.k])
                let dyn = (rows(r, c.k / 4, seed: 0.10) * 0.2).asType(c.flag & 2 != 0 ? .bfloat16 : .float32)
                    .reshaped([1, r, c.k / 4])
                let base = (rows(4, c.k, seed: 0.11)[2...] * 0.2).reshaped([-1])
                    .asType(c.flag & 4 != 0 ? .bfloat16 : .float32)
                let base4 = concatenated([base, base])
                let w = MLX.abs(rows(2, c.k, seed: 0.12)[1]) * 0.25 + 0.5
                let a = DF2Fast.convRmsQuantRun(
                    h, dyn: dyn, base: base4, residual: res, weight: w,
                    D: c.k, L: r, qStart: qStart, eps: 1e-6, group: c.group, wide: false)
                let b = DF2Fast.convRmsQuantRun(
                    h, dyn: dyn, base: base4, residual: res, weight: w,
                    D: c.k, L: r, qStart: qStart, eps: 1e-6, group: c.group, wide: true)
                eqs.append(same(a, b, skip: qStart == 0 ? [] : [0]))
            }
        }
        return MLX.all(MLX.stacked(eqs)).item(Bool.self)
    }

    /// The combinations the drafter's fast path dispatches (FP32 hidden; BF16 token embeddings and
    /// target hidden at the first norm, the first residual and the context projection input).
    static func combos(group: Int) -> [Combo] {
        var c: [Combo] = []
        for (k, d, flag) in [
            (4096, DType.float32, 0), (4096, .bfloat16, 0), (12288, .float32, 1), (4096, .float32, 2),
            (32768, .bfloat16, 0), (32768, .float32, 0),
        ] {
            c.append(Combo(kind: .quantG, dtype: d, dtype2: d, k: k, group: group, flag: flag))
        }
        c.append(Combo(kind: .quantGConv0, dtype: .float32, dtype2: .float32, k: 4096, group: group, flag: 0))
        for d in [DType.float32, .bfloat16] {
            c.append(Combo(kind: .rmsQuant, dtype: d, dtype2: .float32, k: 4096, group: group, flag: 0))
            for writeRes in [1, 0] {
                c.append(Combo(kind: .convRmsQuant, dtype: .float32, dtype2: d, k: 4096, group: group, flag: writeRes))
            }
        }
        return c
    }

    /// Runs the check once, at drafter bind (load time, before any proposal or timed window).
    static func prepare(group: Int) {
        guard !prepared else { return }
        prepared = true
        guard enabled else {
            log("off (DF2_WIDE)")
            return
        }
        for kind in Kind.allCases where offered(kind) {
            var ok = 0, failed: [String] = []
            for combo in combos(group: group) where combo.kind == kind {
                // An MLX error in the check (a pipeline that cannot run this many threads per
                // threadgroup on this GPU, a compile error) keeps the record kernel.
                let name = "\(combo.dtype)/\(combo.dtype2)/k\(combo.k)/form\(combo.flag)"
                do {
                    if try withError({ check(combo) }) {
                        passed.insert(combo)
                        ok += 1
                    } else {
                        failed.append(name + " mismatch")
                    }
                } catch {
                    failed.append(name + " error \(error)")
                }
            }
            log("\(kind.rawValue) e\(e) load check: \(ok) combinations bitwise equal at rows 3, 11, 16"
                + (failed.isEmpty ? "" : "; record kernel kept for " + failed.joined(separator: ", ")))
        }
    }
}

// WIDELANES-GEN BEGIN
/// The drafter's four per-row activation kernels (`df2_act_q8g`, `df2_act_q8g_conv0`, `df2_rms_q8g`,
/// `df2_dyn_conv_rms_q8g`) with `e` contiguous columns per thread instead of 16: e = 8 puts 512 lanes on a
/// 4096-column row. Every element formula is the 16-column kernel's. The row RMS keeps its reduction tree:
/// each thread still sums squares in quads, the record's 16-column partial ((q0 + q1) + (q2 + q3)) is rebuilt
/// with xor shuffles (FP32 addition commutes bit for bit), then the record's xor 1, 2, 4 butterfly over 128
/// columns runs at strides 16/e, 32/e, 64/e and the same 32 block partials meet in one simd_sum. The group
/// sum stays strictly sequential in column order, chained over KG/e lanes, and the group max is order-free.
/// So every output bit equals the 16-column kernel's.
enum DF2WideLaneSource {
    enum Kind: String, CaseIterable {
        case quantG, quantGConv0, rmsQuant, convRmsQuant
        var hasRowNorm: Bool { self == .rmsQuant || self == .convRmsQuant }
    }

    /// Group max, sequential group sum and int8 codes of `nv[e]`; scale and sum land at `row * G + group`.
    static func tail(e: Int, row: String, group: String, base: String) -> String {
        """
        for (int s = 1; s < (KG / \(e)); s <<= 1) {
          local_amax = max(local_amax, simd_shuffle_xor(local_amax, ushort(s)));
        }
        float seq_sum = 0.0f;
        for (int step = 0; step < (KG / \(e)); step++) {
          if (step > 0) {
            seq_sum = simd_shuffle_up(seq_sum, 1);
          }
          for (int u = 0; u < \(e); u++) { seq_sum += nv[u]; }
        }
        const float a = local_amax > 0.0f ? local_amax / 127.0f : 1.0f;
        const float ia = 1.0f / a;
        device char4* dst = (device char4*)(xq + \(base));
        for (int u = 0; u < \(e / 4); u++) {
          const float4 fv = float4(nv[4 * u], nv[4 * u + 1], nv[4 * u + 2], nv[4 * u + 3]);
          dst[u] = char4(int4(rint(fv * ia)));
        }
        if ((int(lane) & ((KG / \(e)) - 1)) == (KG / \(e)) - 1) {
          ascale[\(row) * G + \(group)] = a;
          xs[\(row) * G + \(group)] = seq_sum;
        }

        """
    }

    /// Zero codes, scale and sum for a padding row `row` (rows past the live count, up to 16).
    static func zeroRow(e: Int, row: String, group: String, base: String, lanes: String) -> String {
        """
        device char4* zdst = (device char4*)(xq + \(base));
        for (int u = 0; u < \(e / 4); u++) { zdst[u] = char4(0); }
        if ((\(lanes) & ((KG / \(e)) - 1)) == 0) {
          ascale[\(row) * G + \(group)] = 0.0f;
          xs[\(row) * G + \(group)] = 0.0f;
        }

        """
    }

    /// From the quad sums `s4[e / 4]` to `scale` = rsqrt(mean square + eps) with the record's tree.
    static func rowScale(e: Int) -> String {
        let lps = 16 / e  // lanes per record thread
        let first: String
        switch e {
        case 16: first = "float sq = (s4[0] + s4[1]) + (s4[2] + s4[3]);\n"
        case 8: first = "float sq = s4[0] + s4[1];\nsq += simd_shuffle_xor(sq, 1);\n"
        default: first = "float sq = s4[0];\nsq += simd_shuffle_xor(sq, 1);\nsq += simd_shuffle_xor(sq, 2);\n"
        }
        return first + """
            sq += simd_shuffle_xor(sq, \(lps));
            sq += simd_shuffle_xor(sq, \(2 * lps));
            sq += simd_shuffle_xor(sq, \(4 * lps));
            threadgroup float shared_sq[32];
            if ((tid & \(8 * lps - 1)u) == 0) { shared_sq[tid / \(8 * lps)u] = sq; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float tot = simd_sum(shared_sq[lane]);
              if (lane == 0) {
                shared_sq[0] = metal::precise::rsqrt(tot / uint(D) + eps_arr[0]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float scale = shared_sq[0];

            """
    }

    static func source(_ kind: Kind, e: Int) -> String {
        switch kind {
        case .quantG:
            return """
                const int K = ksz[0]; const int R = ksz[1]; const int G = K / KG;
                const int tid = int(thread_position_in_grid.x);
                const uint lane = thread_index_in_simdgroup;
                const int c0 = tid * \(e);
                if (c0 >= K) return;
                const int m = int(threadgroup_position_in_grid.y);
                const int g = tid / (KG / \(e));
                const size_t dstBase = (size_t)m * K + c0;
                if (m >= R) {

                """ + zeroRow(e: e, row: "m", group: "g", base: "dstBase", lanes: "tid") + """
                  return;
                }
                const size_t srcBase = HEAD_MAJOR
                    ? (((size_t)(c0 >> 7) * R + m) * 128 + (c0 & 127))
                    : dstBase;
                float nv[\(e)];
                float local_amax = 0.0f;
                for (int u = 0; u < \(e); u++) {
                  float a = float(x[srcBase + u]);
                  if (SWIGLU) { const float b = float(x2[srcBase + u]); a = a / (1.0f + metal::exp(-a)) * b; }
                  nv[u] = a;
                  local_amax = max(local_amax, fabs(a));
                }

                """ + tail(e: e, row: "m", group: "g", base: "dstBase")
        case .quantGConv0:
            return """
                const int K = ksz[0]; const int R = ksz[1]; const int G = K / KG;
                const int tid = int(thread_position_in_grid.x);
                const uint lane = thread_index_in_simdgroup;
                const int c0 = tid * \(e);
                if (c0 >= K) return;
                const int m = int(threadgroup_position_in_grid.y);
                const int g = tid / (KG / \(e));
                const size_t rowBase = (size_t)m * K + c0;
                if (m >= R) {

                """ + zeroRow(e: e, row: "m", group: "g", base: "rowBase", lanes: "tid") + """
                  return;
                }
                const int dw = 4 * (K / 16);
                const int cg = c0 >> 4;
                const float d0 = float(dyn[m * dw + cg]);
                const float d1 = m >= 1 ? float(dyn[m * dw + (K >> 4) + cg]) : 0.0f;
                float nv[\(e)];
                float local_amax = 0.0f;
                for (int u = 0; u < \(e); u++) {
                  const int c = c0 + u;
                  float a = (float(base[c]) + d0) * float(h[rowBase + u]);
                  if (m >= 1) {
                    a += (float(base[K + c]) + d1) * float(h[rowBase - K + u]);
                  }
                  nv[u] = a;
                  local_amax = max(local_amax, fabs(a));
                }

                """ + tail(e: e, row: "m", group: "g", base: "rowBase")
        case .rmsQuant:
            return """
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const int l = int(threadgroup_position_in_grid.y);
                const int L = dsz[1];
                const int c0 = int(tid) * \(e);
                const int G = D / KG;
                const int g = int(tid) / (KG / \(e));
                if (l >= L) {
                  if (l < 16) {
                    const size_t qRowBase = (size_t)l * D + c0;

                """ + zeroRow(e: e, row: "l", group: "g", base: "qRowBase", lanes: "int(tid)") + """
                  }
                  return;
                }
                const size_t rowBase = (size_t)l * D + c0;
                float v[\(e)];
                float s4[\(e / 4)];
                for (int k = 0; k < \(e / 4); k++) {
                  float ak = 0.0f;
                  for (int i = 0; i < 4; i++) {
                    const int u = 4 * k + i;
                    const float a = float(x[rowBase + u]);
                    v[u] = a;
                    ak += a * a;
                  }
                  s4[k] = ak;
                }

                """ + rowScale(e: e) + """
                float nv[\(e)];
                float local_amax = 0.0f;
                for (int u = 0; u < \(e); u++) {
                  const float y = float(weight[c0 + u]) * (v[u] * scale);
                  if (WRITE_NORM) { out_norm[rowBase + u] = y; }
                  nv[u] = y;
                  local_amax = max(local_amax, fabs(y));
                }

                """ + tail(e: e, row: "l", group: "g", base: "rowBase")
        case .convRmsQuant:
            return """
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const int l = int(threadgroup_position_in_grid.y);
                const int L = dsz[1];
                const int qStart = dsz[2];
                const int qm = l - qStart;
                const int c0 = int(tid) * \(e);
                const int G = D / KG;
                const int qg = int(tid) / (KG / \(e));
                if (!WRITE_RES && l < qStart) return;
                if (l >= L) {
                  if (qm >= 0 && qm < 16) {
                    const size_t qRowBase = (size_t)qm * D + c0;

                """ + zeroRow(e: e, row: "qm", group: "qg", base: "qRowBase", lanes: "int(tid)") + """
                  }
                  return;
                }
                const int dw = 4 * (D / 16);
                const int g = c0 >> 4;
                const size_t rowBase = (size_t)l * D + c0;
                const float d0 = float(dyn[l * dw + (dw / 2) + g]);
                const float d1 = l >= 1 ? float(dyn[l * dw + (dw / 2) + (D / 16) + g]) : 0.0f;
                float v[\(e)];
                float s4[\(e / 4)];
                for (int k = 0; k < \(e / 4); k++) {
                  float ak = 0.0f;
                  for (int i = 0; i < 4; i++) {
                    const int u = 4 * k + i;
                    const int c = c0 + u;
                    float acc = float(res[rowBase + u]) + (float(base[2 * D + c]) + d0) * float(h[rowBase + u]);
                    if (l >= 1) {
                      acc += (float(base[3 * D + c]) + d1) * float(h[rowBase - D + u]);
                    }
                    v[u] = acc;
                    if (WRITE_RES) { out_res[rowBase + u] = acc; }
                    ak += acc * acc;
                  }
                  s4[k] = ak;
                }

                """ + rowScale(e: e) + """
                float nv[\(e)];
                float local_amax = 0.0f;
                for (int u = 0; u < \(e); u++) {
                  const float y = float(weight[c0 + u]) * (v[u] * scale);
                  nv[u] = y;
                  local_amax = max(local_amax, fabs(y));
                }
                if (qm >= 0 && qm < 16) {
                  const size_t qRowBase = (size_t)qm * D + c0;
                  for (int u = 0; u < \(e); u++) {
                    out_norm[qRowBase + u] = nv[u];
                  }

                """ + tail(e: e, row: "qm", group: "qg", base: "qRowBase") + """
                }

                """
        }
    }
}
// WIDELANES-GEN END


// The FP32 NAX 16x32x16 operations and the 32-key online softmax below retain
// steel_attention_nax's per-row reduction order. Only live query rows run.
private enum DraftSDPA {

    static let enabled: Bool = {
        guard #available(macOS 26.2, *) else { return false }
        guard ProcessInfo.processInfo.environment["DF2_SDPA16"] != "0" else { return false }
        guard ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] != "0" else { return false }
        return GPU.deviceInfo().architecture.hasPrefix("applegpu_g17")
    }()

    static let columns: Int = {
        let requested = Int(ProcessInfo.processInfo.environment["DF2_SDPA16_COLUMNS"] ?? "32") ?? 32
        return [32, 64, 128].contains(requested) ? requested : 32
    }()

    static let queryRows: Int = {
        let requested = Int(ProcessInfo.processInfo.environment["DF2_SDPA16_QROWS"] ?? "16") ?? 16
        return [4, 8, 16].contains(requested) ? requested : 16
    }()
    static let groupedHeads: Int = {
        let requested = Int(ProcessInfo.processInfo.environment["DF2_SDPA16_HEADS"] ?? "1") ?? 1
        return queryRows == 16 && [1, 2, 4].contains(requested) ? requested : 1
    }()
    static let cachedQueries = ProcessInfo.processInfo.environment["DF2_SDPA16_CACHEQ"] != "0"
    static let limitedThreads = ProcessInfo.processInfo.environment["DF2_SDPA16_LIMIT"] != "0"

    static func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                          scale: Float) -> MLXArray? {
        guard enabled, q.dtype == .float32, k.dtype == .float32, v.dtype == .float32,
            q.ndim == 4, k.ndim == 4, v.ndim == 4,
            q.shape4 == (1, 32, 16, 128)
        else { return nil }
        let kShape = k.shape4
        guard kShape.0 == 1, kShape.1 == 8, kShape.2 >= 16, kShape.3 == 128,
            v.shape4 == kShape,
            q.hasNonNegativeUnitInnerStride4, k.hasNonNegativeUnitInnerStride4,
            v.hasNonNegativeUnitInnerStride4
        else { return nil }
        return run(q, k, v, scale: scale, columns: columns, cacheQueries: cachedQueries,
                   queryRows: queryRows, headsPerGroup: groupedHeads, limitThreads: limitedThreads)
    }

    static let header = """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        // Fragment layout and reduction from Apple's steel/attn/nax.h.
        namespace df2sdpa {
        using F = metal::vec<float, 8>;
        METAL_FUNC short2 fragment_coord() {
          const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
          const short qid = lane >> 2;
          return short2{short(((qid & 2) | (lane & 1)) * 4),
                        short((qid & 4) | ((lane >> 1) & 3))};
        }
        METAL_FUNC F load(const device float* p, int stride, int rows = 16) {
          const short2 sc = fragment_coord();
          F f;
          #pragma clang loop unroll(full)
          for (short i = 0; i < 2; ++i) {
            const int r = sc.y + i * 8;
            const float4 v4 = r < rows ? *((const device float4*)(p + r * stride + sc.x)) : float4(0.0f);
            f[i * 4 + 0] = v4.x;
            f[i * 4 + 1] = v4.y;
            f[i * 4 + 2] = v4.z;
            f[i * 4 + 3] = v4.w;
          }
          return f;
        }
        template<bool TRANSPOSE_B>
        METAL_FUNC void mma(thread F& c0, thread F& c1,
                            const thread F& a, const thread F& b0, const thread F& b1) {
          constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
              16, 32, 16, false, TRANSPOSE_B, true,
              mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
          mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
          auto at = op.template get_left_input_cooperative_tensor<float, float, float>();
          auto bt = op.template get_right_input_cooperative_tensor<float, float, float>();
          auto ct = op.template get_destination_cooperative_tensor<
              metal::remove_addrspace_t<decltype(at)>,
              metal::remove_addrspace_t<decltype(bt)>, float>();
          #pragma clang loop unroll(full)
          for (short i = 0; i < 8; ++i) {
            at[i] = a[i]; bt[i] = b0[i]; bt[8 + i] = b1[i];
            ct[i] = c0[i]; ct[8 + i] = c1[i];
          }
          op.run(at, bt, ct);
          #pragma clang loop unroll(full)
          for (short i = 0; i < 8; ++i) { c0[i] = ct[i]; c1[i] = ct[8 + i]; }
        }
        template<bool MAXIMUM>
        METAL_FUNC float reduce_op(float a, float b) {
          if constexpr (MAXIMUM) { return metal::max(a, b); }
          else { return a + b; }
        }
        template<bool MAXIMUM>
        METAL_FUNC void row_reduce(const thread F& f, thread float2& acc) {
          #pragma clang loop unroll(full)
          for (short i = 0; i < 2; ++i) {
            float t = reduce_op<MAXIMUM>(
                reduce_op<MAXIMUM>(f[i * 4], f[i * 4 + 1]),
                reduce_op<MAXIMUM>(f[i * 4 + 2], f[i * 4 + 3]));
            float x = simd_shuffle_xor(t, ushort(1));
            t = reduce_op<MAXIMUM>(t, x);
            x = simd_shuffle_xor(t, ushort(8));
            t = reduce_op<MAXIMUM>(t, x);
            acc[i] = reduce_op<MAXIMUM>(acc[i], t);
          }
        }
        }
        """

    static let source = """
        using namespace df2sdpa;
        const int head = int(threadgroup_position_in_grid.y) * HCOUNT + int(simdgroup_index_in_threadgroup);
        const int ds = int(threadgroup_position_in_grid.x) * DCS;
        const int qs = int(threadgroup_position_in_grid.z) * Q_ROWS;
        const int n = k_shape[2];
        const device float* Q = q + head * q_strides[1] + qs * q_strides[2];
        const device float* K = k + (head / 4) * k_strides[1];
        const device float* V = v + (head / 4) * v_strides[1];
        const short2 sc = fragment_coord();
        const metal::uniform<float> scale2 = make_uniform(scale_in) * make_uniform(1.44269504089f);
        F o[DCS / 16];
        F cached_q[8];
        if constexpr (CACHE_Q) {
          #pragma clang loop unroll(full)
          for (short d = 0; d < 8; ++d) { cached_q[d] = load(Q + d * 16, int(q_strides[2]), Q_ROWS); }
        }
        #pragma clang loop unroll(full)
        for (short d = 0; d < DCS / 16; ++d) { o[d] = F(0); }
        float2 mx(-FLT_MAX), sum(0);
        for (int kb = 0; kb < n; kb += 32) {
          F s0(0), s1(0);
          #pragma clang loop unroll_count(4)
          for (short d = 0; d < 8; ++d) {
            F a;
            if constexpr (CACHE_Q) { a = cached_q[d]; }
            else { a = load(Q + d * 16, int(q_strides[2]), Q_ROWS); }
            const F b0 = load(K + kb * k_strides[2] + d * 16, int(k_strides[2]), n - kb);
            const F b1 = load(K + (kb + 16) * k_strides[2] + d * 16, int(k_strides[2]), n - kb - 16);
            mma<true>(s0, s1, a, b0, b1);
          }
          #pragma clang loop unroll(full)
          for (short i = 0; i < 8; ++i) {
            s0[i] *= float(scale2); s1[i] *= float(scale2);
            const int c = kb + sc.x + i % 4;
            s0[i] = c < n ? s0[i] : -FLT_MAX;
            s1[i] = c + 16 < n ? s1[i] : -FLT_MAX;
          }
          float2 nm = mx;
          row_reduce<true>(s0, nm); row_reduce<true>(s1, nm);
          #pragma clang loop unroll(full)
          for (short i = 0; i < 8; ++i) {
            s0[i] = fast::exp2(s0[i] - nm[i / 4]);
            s1[i] = fast::exp2(s1[i] - nm[i / 4]);
          }
          const float2 factor = fast::exp2(mx - nm);
          mx = nm;
          sum *= factor;
          row_reduce<false>(s0, sum); row_reduce<false>(s1, sum);
          #pragma clang loop unroll(full)
          for (short d = 0; d < DCS / 16; ++d) {
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; ++i) { o[d][i] *= factor[i / 4]; }
          }
          simdgroup_barrier(mem_flags::mem_none);
          #pragma clang loop unroll(full)
          for (short d = 0; d < DCS / 16; d += 2) {
            if constexpr (DCS == 128) {
              if (d == 4) { threadgroup_barrier(mem_flags::mem_none); }
            }
            const F b00 = load(V + kb * v_strides[2] + ds + d * 16, int(v_strides[2]), n - kb);
            const F b01 = load(V + kb * v_strides[2] + ds + (d + 1) * 16, int(v_strides[2]), n - kb);
            mma<false>(o[d], o[d + 1], s0, b00, b01);
            const F b10 = load(V + (kb + 16) * v_strides[2] + ds + d * 16, int(v_strides[2]), n - kb - 16);
            const F b11 = load(V + (kb + 16) * v_strides[2] + ds + (d + 1) * 16, int(v_strides[2]), n - kb - 16);
            mma<false>(o[d], o[d + 1], s1, b10, b11);
          }
        }
        threadgroup_barrier(mem_flags::mem_none);
        float2 rcp;
        #pragma clang loop unroll(full)
        for (short i = 0; i < 2; ++i) { rcp[i] = 1.0f / sum[i]; }
        #pragma clang loop unroll(full)
        for (short d = 0; d < DCS / 16; ++d) {
          #pragma clang loop unroll(full)
          for (short h = 0; h < 2; ++h) {
            const int r = sc.y + h * 8;
            // Match MLX full attention's token-major physical output layout.
            if (r < Q_ROWS) {
              const int c = ds + d * 16 + sc.x;
              *((device float4*)(out + ((qs + r) * 32 + head) * 128 + c)) =
                  float4(o[d][h * 4 + 0] * rcp[h], o[d][h * 4 + 1] * rcp[h],
                         o[d][h * 4 + 2] * rcp[h], o[d][h * 4 + 3] * rcp[h]);
            }
          }
        }
        """

    static func makeKernel(heads: Int, limitThreads: Bool) -> MLXFast.MLXFastKernel {
        var h = header
        var s = source
        if limitThreads {
            // Scope the attribute to the definition; Metal rejects it when
            // repeated on MLX's following explicit instantiation declaration.
            h += "\n#pragma clang attribute push([[max_total_threads_per_threadgroup(\(32 * heads))]], apply_to=function)\n"
            s += "\n#pragma clang attribute pop\n"
        }
        return MLXFast.metalKernel(
            name: "df2_sdpa16_h\(heads)_t\(limitThreads ? 32 * heads : 0)",
            inputNames: ["q", "k", "v", "scale_in"], outputNames: ["out"],
            source: s, header: h, ensureRowContiguous: false)
    }
    private static let plain1 = makeKernel(heads: 1, limitThreads: false)
    private static let plain2 = makeKernel(heads: 2, limitThreads: false)
    private static let plain4 = makeKernel(heads: 4, limitThreads: false)
    private static let limited1 = makeKernel(heads: 1, limitThreads: true)
    private static let limited2 = makeKernel(heads: 2, limitThreads: true)
    private static let limited4 = makeKernel(heads: 4, limitThreads: true)
    private static let defaultScale: Float = Float(1.0 / Double(128).squareRoot())
    nonisolated(unsafe) private static let defaultScaleArray: MLXArray = {
        let a = MLXArray(defaultScale)
        eval(a)
        return a
    }()

    static func run(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float,
                    columns: Int = 128, cacheQueries: Bool = false,
                    queryRows: Int = 16, headsPerGroup: Int = 1,
                    limitThreads: Bool = false) -> MLXArray {
        let kernel: MLXFast.MLXFastKernel
        switch (limitThreads, headsPerGroup) {
        case (false, 1): kernel = plain1
        case (false, 2): kernel = plain2
        case (false, _): kernel = plain4
        case (true, 1): kernel = limited1
        case (true, 2): kernel = limited2
        case (true, _): kernel = limited4
        }
        let scaleArray = scale.bitPattern == defaultScale.bitPattern ? defaultScaleArray : MLXArray(scale)
        return kernel([q, k, v, scaleArray],
               template: [("DCS", columns), ("CACHE_Q", cacheQueries), ("Q_ROWS", queryRows), ("HCOUNT", headsPerGroup)],
               grid: (32 * headsPerGroup * (128 / columns), 32 / headsPerGroup, 16 / queryRows),
               threadGroup: (32 * headsPerGroup, 1, 1),
               outputShapes: [[1, 16, 32, 128]], outputDTypes: [.float32])[0].transposed(0, 2, 1, 3)
    }
}
