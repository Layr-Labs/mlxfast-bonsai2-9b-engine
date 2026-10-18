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
    /// Build, at load, the per-process state the target's `rows`-row verify head
    /// would otherwise build inside the first measured window. Never changes a token.
    func prepareDFlash2Verify(rows: Int)
}

extension DFlash2Target {
    public func prepareDFlash2Verify(rows: Int) {}
    public func draftLogitsForDFlash2Hidden(_ hidden: MLXArray, rows: Int) -> MLXArray {
        logitsForDFlash2Hidden(hidden)
    }
}

/// The drafter's vocabulary: the first `DF2Vocab.rows` token ids. Ids 0 ..< 96k
/// are the tokenizer's Latin, code and byte-level merges (rounded to 98304 = 96 * 1024);
/// the rows above 96k are the CJK and other scripts block. A draft never
/// reaches the output unverified: the target's full head checks every drafted
/// token, so a narrower draft vocabulary changes only which tokens are proposed.
/// `DF2_VOCAB=0` restores the full head.
enum DF2Vocab {
    static let rows: Int = Int(ProcessInfo.processInfo.environment["DF2_VOCAB"] ?? "") ?? 98304
}

/// The target head's prefix rows as one module, kept OFF the module tree (see
/// `DFlash2TapSlot`). Its weight, scales and biases are row slices of the full
/// head's arrays: views of the same buffers, no copy. Built once per head
/// instance, at bind (load time), and kept only after a bitwise self-check.
public final class DFlash2DraftHeadSlot {
    private var source: ObjectIdentifier?
    private var rows = 0
    private var prefix: HadamardQuantizedLinear?

    public init() {}

    public func head(of full: HadamardQuantizedLinear, rows: Int) -> HadamardQuantizedLinear? {
        if source == ObjectIdentifier(full), self.rows == rows { return prefix }
        source = ObjectIdentifier(full)
        self.rows = rows
        prefix = nil
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
        prefix = candidate
        return prefix
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
        precomputedContextKV: (MLXArray, MLXArray)? = nil
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
        if let conv0,
            let qkv = DF2Fast.linearsConv0(
                conv0.n, dyn: conv0.dyn, base: conv0.base, [qProj, kProj, vProj])
        {
            (queries, blockKeys, blockValues) = (qkv[0], qkv[1], qkv[2])
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

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
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

    func absorbContext(_ context: MLXArray, rope: RoPELayer, cache: KVCache) {
        var context = context
        var contextLength = context.dim(1)
        if let slidingWindow {
            let skip = DFlash2SlidingMask.contextSkip(
                contextLength: contextLength, slidingWindow: slidingWindow)
            if skip > 0 {
                context = context[0..., skip..., 0...]
                contextLength = context.dim(1)
                if let base = cache as? BaseKVCache {
                    base.offset += skip
                }
            }
        }
        let B = context.dim(0)
        var contextKeys: MLXArray, contextValues: MLXArray
        if let kv = DF2Fast.linears(context, [kProj, vProj]) {
            (contextKeys, contextValues) = (kv[0], kv[1])
        } else if let kv = DF2Fast.fusedKVLinear(context, kProj: kProj, vProj: vProj) {
            (contextKeys, contextValues) = (kv.0, kv.1)
        } else {
            (contextKeys, contextValues) = (kProj(context), vProj(context))
        }
        contextKeys = DF2Fast.rms(contextKeys.reshaped(B, contextLength, kvHeads, -1), kNorm)
            .transposed(0, 2, 1, 3)
        contextValues = contextValues.reshaped(B, contextLength, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        contextKeys = rope(contextKeys, offset: cache.offset)
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
        precomputedContextKV: (MLXArray, MLXArray)? = nil
    ) -> (h: MLXArray, nextNormed: MLXArray?, nextCodes: DF2Fast.W4Codes?) {
        let n1 = normedInput ?? DF2Fast.rms(x, inputLayerNorm)
        let leadShape = Array(n1.shape.dropLast())
        let p1 =
            (normedCodes.flatMap {
                DF2Fast.w4LinearsFromCodes($0, leadShape: leadShape, [attentionConv.kernelProjection])
            }?[0]) ?? DF2Fast.linear(n1, attentionConv.kernelProjection)
        let base1 = attentionConv.fastBase
        let attnOut: MLXArray
        if DF2Fast.w4Group != nil {
            attnOut = selfAttn(
                n1, context: context, rope: rope, cache: cache,
                conv0: (n: n1, dyn: p1, base: base1),
                ropeSpec: ropeSpec, contextCodes: contextCodes,
                precomputedContextKV: precomputedContextKV)
        } else {
            let a = DF2Fast.dynConv(n1, dyn: p1, base: base1, part: 0, residual: nil)
            attnOut = selfAttn(
                a, context: context, rope: rope, cache: cache,
                ropeSpec: ropeSpec, contextCodes: contextCodes,
                precomputedContextKV: precomputedContextKV)
        }
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
    func absorbContext(_ context: MLXArray, rope: RoPELayer, cache: KVCache) {
        selfAttn.absorbContext(context, rope: rope, cache: cache)
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
        let restored = materialize().view(dtype: .uint16)
        let exact = MLX.all(restored .== source.view(dtype: .uint16))
        try withError { eval(exact) }
        guard exact.item(Bool.self) else { return nil }
    }

    func gather(_ ids: MLXArray, floatOutput: Bool = false) -> MLXArray {
        let shape = ids.shape + [width]
        let output = Self.gatherKernel(
            [packed, lookup, ids.asType(.uint32).reshaped(-1)],
            template: [("WIDTH", width), ("FLOAT_OUTPUT", floatOutput)],
            grid: (ids.size * width / 2, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [shape], outputDTypes: [floatOutput ? .float32 : .uint16])[0]
        return floatOutput ? output : output.view(dtype: .bfloat16)
    }

    func materialize() -> MLXArray {
        gather(MLXArray(UInt32(0) ..< UInt32(rows)))
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
        let rows = DF2Vocab.rows
        if rows >= config.dflash.selectorTopK && rows < config.vocabularySize {
            let probe = MLXArray.zeros([1, 1, config.hiddenSize], dtype: .float32)
            eval(target.draftLogitsForDFlash2Hidden(probe, rows: rows))
        }
        if DF2Fast.on { try prepareFirstRound(target) }
    }

    /// Prepare at load what the first round of the first request would otherwise
    /// build inside a measured decode window: the fast path's cached operands
    /// (FP32 scale and bias transposes, FP32 norm weights, conv bases), the
    /// drafter's kernels at both context shapes (more than 16 rows: the prompt;
    /// 16 or fewer: a round), and the target's exact verify head at the block
    /// width. The proposals run on a throwaway cache and are discarded; nothing
    /// here reaches a request's state.
    private func prepareFirstRound(_ target: any DFlash2Target) throws {
        let cache = try makeCache()
        for rows in [32, 16] {
            let context = MLXArray.zeros([1, rows, config.targetHiddenSize], dtype: dtype)
            eval(try propose(anchor: [0], targetHidden: context, cache: cache, blockSize: config.dflash.blockSize))
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
        logitsStart: Int
    ) throws -> (hidden: MLXArray, codes: DF2Fast.W4Codes?) {
        guard let target else { throw DFlash2Error.notBound }
        guard cache.count == layers.count else {
            throw DFlash2Error.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlash2Error.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize, actual: targetHidden.dim(-1))
        }

        // Both crossings from the target cast here. The target's embedding is a
        // MODULE call, never a raw weight read.
        var h: MLXArray
        if DF2Fast.on, inputs.ndim == 2, inputs.dim(0) == 1, inputs.dim(1) == config.dflash.blockSize,
            logitsStart == 1, fc is QuantizedLinear
        {
            let anchorEmbed = target.embedTokensForDFlash2(inputs[0..., 0 ..< 1]).asType(dtype)
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
        let context: MLXArray
        let cCodes: DF2Fast.W4Codes?
        if targetHidden.dim(1) == 0 {
            context = MLXArray.zeros([targetHidden.dim(0), 0, config.hiddenSize], dtype: dtype)
            cCodes = nil
        } else {
            let fcOut = DF2Fast.linear(targetHidden.asType(dtype), fc)
            if let group = DF2Fast.w4Group, fcOut.dim(0) == 1,
                (1 ... 16).contains(fcOut.dim(1)),
                let rq = DF2Fast.rmsQuant(fcOut, hiddenNorm, group: group)
            {
                (context, cCodes) = rq
            } else {
                context = DF2Fast.rms(fcOut, hiddenNorm)
                cCodes = nil
            }
        }

        if DF2Fast.on, h.ndim == 3, h.dim(0) == 1, h.size / h.dim(-1) <= 16, !layers.isEmpty {
            let rSpec = DF2Fast.ropeSpec(self)
            var normed: MLXArray?
            var normedCodes: DF2Fast.W4Codes?
            if let group = DF2Fast.w4Group,
                let rq = DF2Fast.rmsQuant(h, layers[0].inputLayerNorm, group: group)
            {
                (normed, normedCodes) = rq
            } else {
                normed = DF2Fast.rms(h, layers[0].inputLayerNorm)
                normedCodes = nil
            }
            var precomputedKV: [(MLXArray, MLXArray)]? = nil
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
                    precomputedContextKV: precomputedKV?[index])
                h = step.h
                normed = step.nextNormed
                normedCodes = step.nextCodes
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

    /// Absorb committed context rows into the drafter's caches without
    /// proposing: the context side of `hiddenStates` — the `fc` projection,
    /// `hiddenNorm`, and each layer's context keys and values — on the same
    /// operands in the same order, with no block forward. `blockContext` in
    /// a later `propose` continues these caches exactly as a steady-state
    /// round's incremental absorption does.
    public func absorbContext(targetHidden: MLXArray, cache: [KVCache]) throws {
        guard cache.count == layers.count else {
            throw DFlash2Error.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlash2Error.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize, actual: targetHidden.dim(-1))
        }
        let context = DF2Fast.rms(DF2Fast.linear(targetHidden.asType(dtype), fc), hiddenNorm)
        for (index, layer) in layers.enumerated() {
            layer.absorbContext(context, rope: rope, cache: cache[index])
        }
    }

    func logits(_ hidden: MLXArray) throws -> MLXArray {
        guard let target else { throw DFlash2Error.notBound }
        let rows = DF2Vocab.rows
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
        blockSize: Int
    ) throws -> MLXArray {
        guard blockSize >= 2 else { throw DFlash2Error.invalidBlockSize(blockSize) }
        let masks = Array(repeating: Int32(config.maskTokenId), count: blockSize - 1)
        let rows = anchor.flatMap { [Int32($0)] + masks }
        let block = MLXArray(rows, [anchor.count, blockSize])

        let (hidden, finalCodes) = try hiddenStatesWithCodes(
            block, targetHidden: targetHidden, cache: cache, logitsStart: 1)
        if DF2Fast.on, anchor.count == 1 {
            let sel = candidateSelector
            let projected =
                (finalCodes.flatMap {
                    DF2Fast.w4LinearsFromCodes(
                        $0, leadShape: Array(hidden.shape.dropLast()), [sel.hiddenProjection])
                }?[0]) ?? DF2Fast.linear(hidden, sel.hiddenProjection)
            return DF2Fast.select(
                logits: try logits(hidden), projected: projected,
                anchor: MLXArray(anchor.map { Int32($0) }), pred: sel.predecessorCodebook,
                succ: sel.successorCodebook,
                predLookup: sel.predecessorLookup, succLookup: sel.successorLookup)
        }
        return candidateSelector.selectGreedy(
            hidden: hidden,
            logits: try logits(hidden),
            anchor: MLXArray(anchor.map { Int32($0) }))
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
        init(freqs: MLXArray, mscale: Float) {
            self.freqs = freqs
            self.mscale = mscale
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
            [x.reshaped([rows, K]), x2.reshaped([rows, K]), MLXArray([Int32(K), Int32(rows)])],
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
                + [ascale, xs, MLXArray([Int32(x.dim(-1)), Int32(rows)] + ns.map { Int32($0) })],
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
        let w: MLXArray, sT: MLXArray, bT: MLXArray, n: Int, k: Int
        var pair24: W4Pair24? = nil
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
        w4Cache[ObjectIdentifier(layer)] = o
        return o
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
            const uint grpBase = lane & ~uint((KG / 16) - 1);
            for (int step = 0; step < (KG / 16); step++) {
              seq_sum = simd_shuffle(seq_sum, ushort(grpBase | uint(step > 0 ? step - 1 : 0)));
              if ((int(lane) & ((KG / 16) - 1)) == step) {
                for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
              }
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
            const uint grpBase = lane & ~uint((KG / 16) - 1);
            for (int step = 0; step < (KG / 16); step++) {
              seq_sum = simd_shuffle(seq_sum, ushort(grpBase | uint(step > 0 ? step - 1 : 0)));
              if ((int(lane) & ((KG / 16) - 1)) == step) {
                for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
              }
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
        var admitted: [Int: Bool] = [:]

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
    static func w4Pair24Source(form: Int, narrow: Bool) -> String {
        var source = w4Source(layers: 1, group: 64, form: form, narrow: narrow)
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
        let sizes = sizeDescriptor ?? MLXArray([Int32(32768), Int32(rows), Int32(4096)])
        return w4Pair24Kernels[form * 2 + (narrow ? 1 : 0)]!(
            [xq, op.w, metadata.words, metadata.bases, ascale, xs, sizes],
            template: [("SPLIT", split)], grid: (4096 / columns * split * 32, 1, 1),
            threadGroup: (split * 32, 1, 1), outputShapes: [[rows, 4096]], outputDTypes: [.float32])
    }

    /// One fixed, input-independent FC comparison; a decline keeps the original raw path.
    static func w4Pair24Admitted(_ op: W4Operands, metadata: W4Pair24, form: Int, narrow: Bool) -> Bool {
        let key = form * 2 + (narrow ? 1 : 0)
        if let result = metadata.admitted[key] { return result }
        let K = 32768, G = K / 64
        let xq = MLXArray((0 ..< 16 * K).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [16 * K])
        let a = MLXArray((0 ..< 16 * G).map { Float(1 + $0 % 11) / 256 }, [16 * G])
        let x = MLXArray((0 ..< 16 * G).map { Float($0 % 17 - 8) / 64 }, [16 * G])
        var accepted = false
        w4CheckFailed = false
        w4CheckNote = ""
        withErrorHandler({ message in
            if !DF2Fast.w4CheckFailed { DF2Fast.w4CheckNote = String(message.prefix(160)) }
            DF2Fast.w4CheckFailed = true
        }) {
            let control = w4Run(form: form, narrow: narrow, xq: xq, ascale: a, xs: x, rows: 16,
                K: K, ws: [op.w], sTs: [op.sT], bTs: [op.bT], ns: [4096])[0]
            let packed = w4Pair24Run(form: form, narrow: narrow, xq: xq, ascale: a, xs: x, rows: 16,
                op: op, metadata: metadata)[0]
            let finite = isFinite(control).all() .&& isFinite(packed).all()
            eval(finite)
            if w4CheckFailed || !finite.item(Bool.self) { return }
            let exact = (control.view(dtype: .uint32) .== packed.view(dtype: .uint32)).all()
            eval(exact)
            if !w4CheckFailed { accepted = exact.item(Bool.self) }
        }
        metadata.admitted[key] = accepted
        FileHandle.standardError.write(Data("DF2 FC pair24 form\(form) narrow=\(narrow): \(accepted ? "finite/exact" : "raw fallback kept")\n".utf8))
        return accepted
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
    static func w4Pair24MLPSource(layers: Int, form: Int, narrow: Bool) -> String {
        precondition(layers == 1 || (layers == 2 && !narrow))
        var source = w4Pair24Source(form: form, narrow: narrow)
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
        let sizes = sizeDescriptor ?? MLXArray([Int32(K), Int32(rows)] + ns.map { Int32($0) })
        return w4Pair24MLPKernels[key]!(
            [xq] + ops.map(\.w) + w4Pair24MLPMetadataInputs(metadata)
                + [ascale, xs, sizes],
            template: [("SPLIT", split), ("KC", K)],
            grid: (ns.reduce(0, +) / (narrow ? 16 : 32) * split * 32, 1, 1),
            threadGroup: (split * 32, 1, 1),
            outputShapes: ns.map { [rows, $0] }, outputDTypes: ns.map { _ in .float32 })
    }

    struct W4Pair24MLPAdmissionKey: Hashable {
        let operands: [ObjectIdentifier]
        let form: Int
        let narrow: Bool
    }
    nonisolated(unsafe) static var w4Pair24MLPAdmissions: [W4Pair24MLPAdmissionKey: Bool] = [:]

    /// One fixed comparison per ordered immutable cohort and actual consumer form.
    static func w4Pair24MLPAdmitted(
        _ ops: [W4Operands], metadata: [W4Pair24], form: Int, narrow: Bool
    ) -> Bool {
        let key = W4Pair24MLPAdmissionKey(
            operands: ops.map { ObjectIdentifier($0) }, form: form, narrow: narrow)
        if let result = w4Pair24MLPAdmissions[key] { return result }
        let K = ops[0].k, G = K / 64, ns = ops.map(\.n)
        let xq = MLXArray((0 ..< 16 * K).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [16 * K])
        let a = MLXArray((0 ..< 16 * G).map { Float(1 + $0 % 11) / 256 }, [16 * G])
        let x = MLXArray((0 ..< 16 * G).map { Float($0 % 17 - 8) / 64 }, [16 * G])
        var accepted = false
        w4CheckFailed = false
        w4CheckNote = ""
        withErrorHandler({ message in
            if !DF2Fast.w4CheckFailed { DF2Fast.w4CheckNote = String(message.prefix(160)) }
            DF2Fast.w4CheckFailed = true
        }) {
            let control = concatenated(w4Run(
                form: form, narrow: narrow, xq: xq, ascale: a, xs: x, rows: 16,
                K: K, ws: ops.map(\.w), sTs: ops.map(\.sT), bTs: ops.map(\.bT), ns: ns), axis: 1)
            let packed = concatenated(w4Pair24MLPRun(
                form: form, narrow: narrow, xq: xq, ascale: a, xs: x, rows: 16,
                ops: ops, metadata: metadata), axis: 1)
            let finite = isFinite(control).all() .&& isFinite(packed).all()
            eval(finite)
            if w4CheckFailed || !finite.item(Bool.self) { return }
            let exact = (control.view(dtype: .uint32) .== packed.view(dtype: .uint32)).all()
            eval(exact)
            if !w4CheckFailed { accepted = exact.item(Bool.self) }
        }
        w4Pair24MLPAdmissions[key] = accepted
        FileHandle.standardError.write(Data(
            "DF2 MLP pair24 K\(K) N\(ns) form\(form) narrow=\(narrow): \(accepted ? "finite/exact" : "raw fused fallback kept")\n".utf8))
        return accepted
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
    static func w4Source(layers: Int, group: Int, form: Int = 0, narrow: Bool = false) -> String {
        precondition(!narrow || form == 1 || form == 2)
        let ru = group / 32  // packed uint4 per row and group
        let tw = group / 16  // unpacked uint4 per row and group
        let cols = narrow ? 16 : 32  // output columns per threadgroup
        let pr = narrow ? ru / 2 : ru  // packed uint4 rows each lane stages per group
        let ne = narrow ? 8 : 16  // live destination elements per lane
        var pick = """
            const int K = ksz[0];
            int bx = int(threadgroup_position_in_grid.x);
            const device uint32_t* w = w0; const device half* sT = s0; const device half* bT = b0;
            device float* out = out0; int N = ksz[2];

            """
        for l in 1 ..< layers {
            pick += "if (bx >= N / \(cols)) { bx -= N / \(cols); w = w\(l); sT = s\(l); bT = b\(l); out = out\(l); N = ksz[\(2 + l)]; }\n"
        }
        let operandB =
            """
            threadgroup uint4 tiles[SPLIT * 32 * \(tw)];
            threadgroup uint4* bs = tiles + sg * (32 * \(tw));
            tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> Bt((threadgroup int8_t*)bs, dextents<int, 2>(\(group), 32));
            auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(Bt)>, int32_t>();

            """
        let codeSetup =
            """
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
        let prefetchCodes =
            "for (int p = 0; p < \(pr); p++) { wv[p] = wsrc[(size_t)(p * \(32 / ru)) * rowU4 + (g + SPLIT) * \(ru)]; }"
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
        let loop: String
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
        let reduce =
            """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float* red = (threadgroup float*)tiles;

            """
        return pick + """
            #define LO(v) ((((v) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
            #define HI(v) (((((v) >> 4) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
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
        form: Int, narrow: Bool = false, xq: MLXArray, ascale: MLXArray, xs: MLXArray, rows: Int,
        K: Int, ws: [MLXArray], sTs: [MLXArray], bTs: [MLXArray], ns: [Int],
        sizeDescriptor: MLXArray? = nil
    ) -> [MLXArray] {
        let sizes = sizeDescriptor ?? MLXArray([Int32(K), Int32(rows)] + ns.map { Int32($0) })
        let formsFit = w4FormsFit(K: K, rows: rows, ns: ns)
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
                let sT = MLXArray((0 ..< g * n).map { Float16(Float(1 + $0 % 13) / 1024) }, [g, n])
                let bT = MLXArray((0 ..< g * n).map { Float16(-Float($0 % 7) / 512) }, [g, n])
                let xq = MLXArray(
                    (0 ..< 16 * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [16 * k])
                let ascale = MLXArray((0 ..< 16 * g).map { Float(1 + $0 % 11) / 256 }, [16 * g])
                let xs = MLXArray((0 ..< 16 * g).map { Float($0 % 17 - 8) / 64 }, [16 * g])
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
                let ascale = MLXArray((0 ..< 16 * g).map { Float(1 + $0 % 11) / 256 }, [16 * g])
                let xs = MLXArray((0 ..< 16 * g).map { Float($0 % 17 - 8) / 64 }, [16 * g])
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
            [x, x2, MLXArray([Int32(K), Int32(rows)])],
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
            [h, dyn, base, MLXArray([Int32(K), Int32(rows)])],
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
        let preferredForm = w4SelectedForm
        let narrow = preferredForm != 0 && ns.reduce(0, +) <= w4NarrowColumns && w4Narrow
        let form = preferredForm == 2 && narrow && !w4Form2Narrow ? 1 : preferredForm
        let sizeDescriptor = ops[0].dispatchSizes(K: codes.K, rows: codes.rows, ns: ns)
        let outs: [MLXArray]
        if ops.count == 1, ops[0].k == 32768, ops[0].n == 4096, let metadata = ops[0].pair24,
            w4Pair24Admitted(ops[0], metadata: metadata, form: form, narrow: narrow)
        {
            outs = w4Pair24Run(form: form, narrow: narrow, xq: codes.xq, ascale: codes.ascale,
                xs: codes.xs, rows: codes.rows, op: ops[0], metadata: metadata, sizeDescriptor: sizeDescriptor)
        } else if codes.group == 64, w4FormsFit(K: codes.K, rows: codes.rows, ns: ns),
            let metadata = w4Pair24MLPMetadata(ops),
            w4Pair24MLPAdmitted(ops, metadata: metadata, form: form, narrow: narrow)
        {
            outs = w4Pair24MLPRun(form: form, narrow: narrow, xq: codes.xq, ascale: codes.ascale,
                xs: codes.xs, rows: codes.rows, ops: ops, metadata: metadata, sizeDescriptor: sizeDescriptor)
        } else {
            outs = w4Run(
                form: form, narrow: narrow, xq: codes.xq, ascale: codes.ascale, xs: codes.xs,
                rows: codes.rows, K: codes.K, ws: ops.map(\.w), sTs: ops.map(\.sT), bTs: ops.map(\.bT),
                ns: ns, sizeDescriptor: sizeDescriptor)
        }
        return zip(outs, ns).map { $0.reshaped(leadShape + [$1]) }
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
                MLXArray([Int32(L), Int32(offset)]),
                MLXArray([qNorm.eps, ropeSpec.mscale]),
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
                MLXArray([Int32(C), Int32(offset)]),
                MLXArray([kNorm.eps, ropeSpec.mscale]),
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
            const float th0 = pos * (1.0f / freqs[d0]);
            const float c0 = metal::fast::cos(th0), s0 = metal::fast::sin(th0);
            const float th1 = pos * (1.0f / freqs[d1]);
            const float c1 = metal::fast::cos(th1), s1 = metal::fast::sin(th1);
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
        let outs = qkvAllNormRopeKernel(
            [
                q, k, v, ck, cv, qw, kw, ropeSpec.freqs,
                MLXArray([Int32(L), Int32(C), Int32(ctxOffset)]),
                MLXArray([qNorm.eps, ropeSpec.mscale]),
            ],
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
              out_norm[rowBase + u] = y;
              nv[u] = y;
              local_amax = max(local_amax, fabs(y));
            }
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 1));
            local_amax = max(local_amax, simd_shuffle_xor(local_amax, 2));
            if (KG == 128) {
              local_amax = max(local_amax, simd_shuffle_xor(local_amax, 4));
            }
            float seq_sum = 0.0f;
            const uint grpBase = lane & ~uint((KG / 16) - 1);
            for (int step = 0; step < (KG / 16); step++) {
              seq_sum = simd_shuffle(seq_sum, ushort(grpBase | uint(step > 0 ? step - 1 : 0)));
              if ((int(lane) & ((KG / 16) - 1)) == step) {
                for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
              }
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

    static func rmsQuant(_ x: MLXArray, _ norm: RMSNorm, group: Int) -> (MLXArray, W4Codes)? {
        let D = x.dim(-1), L = x.size / D
        guard on, D == 4096, (1 ... 16).contains(L),
            (group == 64 || group == 128), split > 0, D / group >= split
        else {
            return nil
        }
        let w = normWeight(norm)
        let wide = DF2WideLanes.serves(
            .init(kind: .rmsQuant, dtype: x.dtype, dtype2: w.dtype, k: D, group: group, flag: 0))
        let outs = rmsQuantRun(x, w, D: D, L: L, eps: norm.eps, group: group, wide: wide)
        let codes = W4Codes(xq: outs[1], ascale: outs[2], xs: outs[3], rows: L, K: D, group: group)
        return (outs[0], codes)
    }

    /// `df2_rms_q8g` at 256 lanes per row, or its `DF2WideLaneSource` twin (`wide`).
    static func rmsQuantRun(
        _ x: MLXArray, _ w: MLXArray, D: Int, L: Int, eps: Float, group: Int, wide: Bool
    ) -> [MLXArray] {
        let G = D / group
        let lanes = D / (wide ? DF2WideLanes.e : 16)
        return (wide ? DF2WideLanes.rmsQuant : rmsQuantKernel)(
            [x, w, MLXArray([Int32(D), Int32(L)]), MLXArray([eps])],
            template: [("D", D), ("KG", group)],
            grid: (lanes, Swift.max(L, 16), 1), threadGroup: (lanes, 1, 1),
            outputShapes: [x.shape, [16 * D], [16 * G], [16 * G]],
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
                MLXArray([Int32(D), Int32(L), Int32(qStart)]),
                MLXArray([eps]),
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
              const uint grpBase = lane & ~uint((KG / 16) - 1);
              for (int step = 0; step < (KG / 16); step++) {
                seq_sum = simd_shuffle(seq_sum, ushort(grpBase | uint(step > 0 ? step - 1 : 0)));
                if ((int(lane) & ((KG / 16) - 1)) == step) {
                  for (int u = 0; u < 16; u++) { seq_sum += nv[u]; }
                }
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
            [h, dyn, base, residual, w, MLXArray([Int32(D), Int32(L)]), MLXArray([norm.eps])],
            template: [("D", D)],
            grid: (256, L, 1), threadGroup: (256, 1, 1),
            outputShapes: [h.shape, h.shape], outputDTypes: [.float32, .float32])
        return (outs[0], outs[1], nil)
    }

    static func dynConv(_ h: MLXArray, dyn: MLXArray, base: MLXArray, part: Int, residual: MLXArray?) -> MLXArray {
        let D = h.dim(-1), L = h.size / D
        return conv(
            [h, dyn, base, residual ?? h, MLXArray([Int32(D), Int32(L)])],
            template: [("D", D), ("PART", part), ("RES", residual == nil ? 0 : 1)],
            grid: ((L * D + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [h.shape], outputDTypes: [.float32])[0]
    }

    /// Multi-block parallel top-16 per (row, block): 8 blocks of 256 threads per row (120 blocks for P=15).
    static let topk = MLXFast.metalKernel(
        name: "df2_topk16_mb8", inputNames: ["logits"], outputNames: ["ids", "vals"],
        source: """
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
            """)

    static let selector = MLXFast.metalKernel(
        name: "df2_selector_mb8", inputNames: ["cand", "unary", "proj", "pred", "succ", "anchor", "predLookup", "succLookup"],
        outputNames: ["path"],
        source: """
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
            """)

    private static var selectorLookupSentinel: MLXArray { MLXArray([UInt16(0)]) }

    /// Fused top-16 + greedy selector for `selectGreedy` (B = 1).
    static func select(
        logits: MLXArray, projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray,
        predLookup: MLXArray? = nil, succLookup: MLXArray? = nil
    ) -> MLXArray {
        let V = logits.dim(-1), P = logits.size / V
        let top = topk(
            [logits.reshaped([P, V]).asType(.float32)],
            grid: (P * 256, 8, 1), threadGroup: (256, 1, 1),
            outputShapes: [[P, 128], [P, 128]], outputDTypes: [.int32, .float32])
        let path = selector(
            [top[0], top[1], projected.reshaped([P, -1]).asType(.float32), pred, succ, anchor.asType(.int32),
             predLookup ?? selectorLookupSentinel, succLookup ?? selectorLookupSentinel],
            template: [("PRED_PACKED", predLookup != nil), ("SUCC_PACKED", succLookup != nil)],
            grid: (512, 1, 1), threadGroup: (512, 1, 1),
            outputShapes: [[P]], outputDTypes: [.int32])[0]
        return path.reshaped([1, P])
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
    static func offered(_ kind: Kind) -> Bool {
        enabled && (kind != .quantG || setting.contains("all"))
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
        const uint grpBase = lane & ~uint((KG / \(e)) - 1);
        for (int step = 0; step < (KG / \(e)); step++) {
          seq_sum = simd_shuffle(seq_sum, ushort(grpBase | uint(step > 0 ? step - 1 : 0)));
          if ((int(lane) & ((KG / \(e)) - 1)) == step) {
            for (int u = 0; u < \(e); u++) { seq_sum += nv[u]; }
          }
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
                  out_norm[rowBase + u] = y;
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
