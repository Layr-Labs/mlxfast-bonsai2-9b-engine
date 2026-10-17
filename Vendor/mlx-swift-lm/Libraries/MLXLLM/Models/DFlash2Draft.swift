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
    /// Build, at load, the per-process state the target's `rows`-row verify head
    /// would otherwise build inside the first measured window. Never changes a token.
    func prepareDFlash2Verify(rows: Int)
}

extension DFlash2Target {
    public func prepareDFlash2Verify(rows: Int) {}
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
        _ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        var context = context
        var contextLength = context.dim(1)

        if let slidingWindow {
            let skip = DFlash2SlidingMask.contextSkip(
                contextLength: contextLength, slidingWindow: slidingWindow)
            if skip > 0 {
                context = context[0..., skip..., 0...]
                contextLength = context.dim(1)
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
        if let qkv = DF2Fast.linears(x, [qProj, kProj, vProj]) {
            (queries, blockKeys, blockValues) = (qkv[0], qkv[1], qkv[2])
        } else {
            (queries, blockKeys, blockValues) = (qProj(x), kProj(x), vProj(x))
        }
        // A block with no new context rows (its context was absorbed ahead of
        // it) attends to the cache as it stands; nothing is projected for it.
        if contextLength > 0 {
            if let kv = DF2Fast.linears(context, [kProj, vProj]) {
                (contextKeys, contextValues) = (kv[0], kv[1])
            } else {
                (contextKeys, contextValues) = (kProj(context), vProj(context))
            }
        }

        queries = DF2Fast.rms(queries.reshaped(B, L, heads, -1), qNorm).transposed(0, 2, 1, 3)
        if contextLength > 0 {
            contextKeys = DF2Fast.rms(contextKeys!.reshaped(B, contextLength, kvHeads, -1), kNorm)
                .transposed(0, 2, 1, 3)
            contextValues = contextValues!.reshaped(B, contextLength, kvHeads, -1)
                .transposed(0, 2, 1, 3)
        }
        blockKeys = DF2Fast.rms(blockKeys.reshaped(B, L, kvHeads, -1), kNorm).transposed(0, 2, 1, 3)
        blockValues = blockValues.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        // The block sits immediately after the context, so both the queries and
        // the block's own keys rotate at the context's far end.
        let blockOffset = cache.offset + contextLength
        queries = rope(queries, offset: blockOffset)
        if contextLength > 0 { contextKeys = rope(contextKeys!, offset: cache.offset) }
        blockKeys = rope(blockKeys, offset: blockOffset)

        // Only the CONTEXT keys and values enter the cache. The block's own keys
        // and values are concatenated for this forward and then dropped.
        let (cachedKeys, cachedValues) =
            contextLength > 0
            ? cache.update(keys: contextKeys!, values: contextValues!) : Self.cachedKeysValues(cache)
        let cachedLength = cachedKeys.dim(2)
        let keys = concatenated([cachedKeys, blockKeys], axis: 2)
        let values = concatenated([cachedValues, blockValues], axis: 2)

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
        } else {
            (contextKeys, contextValues) = (kProj(context), vProj(context))
        }
        contextKeys = DF2Fast.rms(contextKeys.reshaped(B, contextLength, kvHeads, -1), kNorm)
            .transposed(0, 2, 1, 3)
        contextValues = contextValues.reshaped(B, contextLength, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        contextKeys = rope(contextKeys, offset: cache.offset)
        _ = cache.update(keys: contextKeys, values: contextValues)
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let gu = DF2Fast.linears(x, [gate, up]), let y = DF2Fast.linears(gu[0], [down], up: gu[1]) {
            return y[0]
        }
        return down(silu(gate(x)) * up(x))
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

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache
    ) -> MLXArray {
        if DF2Fast.on, x.dim(0) == 1, x.size / x.dim(-1) <= 16 {
            let n1 = DF2Fast.rms(x, inputLayerNorm)
            let p1 = DF2Fast.linear(n1, attentionConv.kernelProjection)
            let base1 = attentionConv.fastBase
            let a = DF2Fast.dynConv(n1, dyn: p1, base: base1, part: 0, residual: nil)
            let attended = DF2Fast.dynConv(
                selfAttn(a, context: context, rope: rope, cache: cache), dyn: p1, base: base1, part: 1,
                residual: x)
            let n2 = DF2Fast.rms(attended, postAttentionLayerNorm)
            let p2 = DF2Fast.linear(n2, mlpConv.kernelProjection)
            let base2 = mlpConv.fastBase
            let m = DF2Fast.dynConv(n2, dyn: p2, base: base2, part: 0, residual: nil)
            return DF2Fast.dynConv(mlp(m), dyn: p2, base: base2, part: 1, residual: attended)
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

// MARK: - The candidate selector

/// The low-rank edge-scored greedy path over the per-position candidate lists.
///
/// The track is greedy, so only the temperature-0 path of the reference
/// `CandidateSelector.select` is ported. The sampling path is not needed and is
/// not here.
final class DFlash2CandidateSelector: Module {
    let topK: Int

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
            let edges =
                (take(predecessorCodebook, predecessor, axis: 0)
                    .expandedDimensions(axis: 1)
                    * projected[0..., position, 0...].expandedDimensions(axis: 1)
                    * take(successorCodebook, positionCandidates, axis: 0))
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
        guard target.dFlash2VocabularySize == config.vocabularySize else {
            throw DFlash2Error.vocabularyMismatch(
                drafter: config.vocabularySize, target: target.dFlash2VocabularySize)
        }
        guard target.dFlash2HiddenSize == config.hiddenSize else {
            throw DFlash2Error.hiddenSizeMismatch(
                drafter: config.hiddenSize, target: target.dFlash2HiddenSize)
        }
        self.target = target
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
                return StandardKVCache()
            case .slidingAttention:
                guard let slidingWindow = config.slidingWindow else {
                    throw DFlash2Error.missingSlidingWindow
                }
                return RotatingKVCache(maxSize: slidingWindow - 1, keep: 0)
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
        var h = target.embedTokensForDFlash2(inputs).asType(dtype)
        if config.dflash.inputEmbeddingScale != 1 {
            h = h * config.dflash.inputEmbeddingScale
        }
        let context =
            targetHidden.dim(1) == 0
            ? MLXArray.zeros([targetHidden.dim(0), 0, config.hiddenSize], dtype: dtype)
            : DF2Fast.rms(DF2Fast.linear(targetHidden.asType(dtype), fc), hiddenNorm)

        for (index, layer) in layers.enumerated() {
            h = layer(h, context: context, rope: rope, cache: cache[index])
            // Hand each finished layer to the GPU while the next one is built.
            if DF2Fast.on { asyncEval(h) }
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return DF2Fast.rms(h, norm)
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
        var logits = target.logitsForDFlash2Hidden(hidden)
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

        let hidden = try hiddenStates(
            block, targetHidden: targetHidden, cache: cache, logitsStart: 1)
        if DF2Fast.on, anchor.count == 1 {
            let sel = candidateSelector
            return DF2Fast.select(
                logits: try logits(hidden), projected: DF2Fast.linear(hidden, sel.hiddenProjection),
                anchor: MLXArray(anchor.map { Int32($0) }), pred: sel.predecessorCodebook,
                succ: sel.successorCodebook)
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
    nonisolated(unsafe) static var operandCache: [ObjectIdentifier: Operands] = [:]
    nonisolated(unsafe) static var baseCache: [ObjectIdentifier: MLXArray] = [:]
    nonisolated(unsafe) static var normCache: [ObjectIdentifier: MLXArray] = [:]

    /// `norm(x)` with the norm's weight widened to FP32 once (no per-call cast).
    static func rms(_ x: MLXArray, _ norm: RMSNorm) -> MLXArray {
        guard on else { return norm(x) }
        let key = ObjectIdentifier(norm)
        let weight: MLXArray
        if let w = normCache[key] {
            weight = w
        } else {
            weight = norm.weight.asType(.float32)
            eval(weight)
            normCache[key] = weight
        }
        return MLXFast.rmsNorm(x.asType(.float32), weight: weight, eps: norm.eps)
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
        init(w: MLXArray, sT: MLXArray, bT: MLXArray, n: Int, k: Int) {
            self.w = w; self.sT = sT; self.bT = bT; self.n = n; self.k = k
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
        w4Cache[ObjectIdentifier(layer)] = o
        return o
    }

    /// x [R, K] (or silu(x) * x2) -> int8 codes per row and per KG-column group, with the group's
    /// scale and its unquantized sum (for the offset term).
    static let quantG = MLXFast.metalKernel(
        name: "df2_act_q8g_simd", inputNames: ["x", "x2", "ksz"], outputNames: ["xq", "ascale", "xs"],
        source: """
            const int K = ksz[0]; const int R = ksz[1]; const int G = K / KG;
            const int m = int(threadgroup_position_in_grid.y);
            const int lane = int(thread_index_in_simdgroup);
            const int g = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
            if (g >= G) return;
            const size_t base = (size_t)m * K + g * KG;
            float values[KG / 32];
            float amax = 0.0f, sum = 0.0f;
            for (int u = 0; u < KG / 32; u++) {
              float v = 0.0f;
              if (m < R) {
                const size_t i = base + lane + u * 32;
                v = float(x[i]);
                if (SWIGLU) { const float b = float(x2[i]); v = v / (1.0f + metal::exp(-v)) * b; }
              }
              values[u] = v;
              amax = max(amax, fabs(v));
              sum += v;
            }
            amax = simd_max(amax);
            sum = simd_sum(sum);
            const float a = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float ia = 1.0f / a;
            for (int u = 0; u < KG / 32; u++) {
              xq[base + lane + u * 32] = char(int(rint(values[u] * ia)));
            }
            if (lane == 0) {
              ascale[m * G + g] = m < R ? a : 0.0f;
              xs[m * G + g] = sum;
            }
            """)

    /// int8 activations x the 4-bit codes on the NAX tensor units, one 16 x 32 x `group` op per group:
    /// each simdgroup stages its 32 x `group` tile as int8 (nibble - 8) in threadgroup memory, and the
    /// epilogue adds s * a * C + (lo + 8 s) * sum(x) per group. 32 output columns per threadgroup,
    /// SPLIT simdgroups over K, the codes of the next group prefetched in registers.
    static func w4Source(layers: Int, group: Int) -> String {
        let ru = group / 32  // packed uint4 per row and group
        let tw = group / 16  // unpacked uint4 per row and group
        var pick = """
            const int K = ksz[0];
            int bx = int(threadgroup_position_in_grid.x);
            const device uint32_t* w = w0; const device half* sT = s0; const device half* bT = b0;
            device float* out = out0; int N = ksz[2];

            """
        for l in 1 ..< layers {
            pick += "if (bx >= N / 32) { bx -= N / 32; w = w\(l); sT = s\(l); bT = b\(l); out = out\(l); N = ksz[\(2 + l)]; }\n"
        }
        return pick + """
            #define LO(v) ((((v) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
            #define HI(v) (((((v) >> 4) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
            const int G = K / \(group);
            const int n0 = bx * 32;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint tid = thread_position_in_threadgroup.x;
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, \(group), false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
            tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
            threadgroup uint4 tiles[SPLIT * 32 * \(tw)];
            threadgroup uint4* bs = tiles + sg * (32 * \(tw));
            tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> Bt((threadgroup int8_t*)bs, dextents<int, 2>(\(group), 32));
            auto tA0 = A.template slice<\(group), 16>(0, 0);
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
            const int lr = int(lane) / \(ru), lc = int(lane) % \(ru);
            const size_t rowU4 = (size_t)K / 32;
            const device uint4* wsrc = (const device uint4*)w + (size_t)(n0 + lr) * rowU4 + lc;
            uint4 wv[\(ru)];
            for (int p = 0; p < \(ru); p++) { wv[p] = wsrc[(size_t)(p * \(32 / ru)) * rowU4 + int(sg) * \(ru)]; }
            for (int g = int(sg); g < G; g += SPLIT) {
              for (int p = 0; p < \(ru); p++) {
                const int r = lr + p * \(32 / ru);
                const uint4 x = wv[p];
                bs[r * \(tw) + 2 * lc] = uint4(LO(x.x), HI(x.x), LO(x.y), HI(x.y));
                bs[r * \(tw) + 2 * lc + 1] = uint4(LO(x.z), HI(x.z), LO(x.w), HI(x.w));
              }
              if (g + SPLIT < G) { for (int p = 0; p < \(ru); p++) { wv[p] = wsrc[(size_t)(p * \(32 / ru)) * rowU4 + (g + SPLIT) * \(ru)]; } }
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
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float* red = (threadgroup float*)tiles;
            for (int i = 0; i < 16; i++) {
              const int m = mrow + 8 * ((i >> 2) & 1), n = nb + (i & 3) + 16 * (i >> 3);
              red[sg * 512 + m * 32 + n] = acc[i] + accb[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = int(tid); e < ksz[1] * 32; e += SPLIT * 32) {
              float v = 0.0f;
              for (int q = 0; q < SPLIT; q++) { v += red[q * 512 + e]; }
              out[(size_t)(e >> 5) * N + n0 + (e & 31)] = ok ? v : as_type<float>(0x7fc00000u);
            }
            #undef LO
            #undef HI
            """
    }

    static let w4Kernels: [MLXFast.MLXFastKernel] = (1 ... 3).map { layers in
        let group = w4Group ?? 64
        return MLXFast.metalKernel(
            name: "df2_w4g\(group)_\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "xs", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: w4Source(layers: layers, group: group), header: header, ensureRowContiguous: true)
    }

    /// `layers(x)` (or of silu(x) * up) through the 4-bit copies, rows <= 16; nil when not eligible.
    static func w4Linears(_ x: MLXArray, _ layers: [Linear], up: MLXArray?, group: Int) -> [MLXArray]? {
        let K = x.dim(-1)
        let rows = x.size / K
        // Each SIMD group prefetches its first K tile before entering the loop.
        guard (1 ... 16).contains(rows), K % group == 0,
            split > 0, K / group >= split
        else { return nil }
        let ops = layers.compactMap { w4Operands($0, group: group) }
        guard ops.count == layers.count, ops.allSatisfy({ $0.k == K }) else { return nil }
        let G = K / group
        let q = quantG(
            [x.reshaped([rows, K]), (up ?? x).reshaped([rows, K]), MLXArray([Int32(K), Int32(rows)])],
            template: [("SWIGLU", up == nil ? 0 : 1), ("KG", group)],
            grid: ((G + 3) / 4 * 128, 16, 1), threadGroup: (128, 1, 1),
            outputShapes: [[16 * K], [16 * G], [16 * G]],
            outputDTypes: [.int8, .float32, .float32])
        let ns = ops.map(\.n)
        let sizes = MLXArray([Int32(K), Int32(rows)] + ns.map { Int32($0) })
        var inputs: [MLXArray] = [q[0]]
        inputs += ops.map(\.w)
        inputs += ops.map(\.sT)
        inputs += ops.map(\.bT)
        inputs += [q[1], q[2], sizes]
        let kernel = w4Kernels[layers.count - 1]
        let outs = kernel(
            inputs,
            template: [("SPLIT", split)],
            grid: (ns.reduce(0, +) / 32 * split * 32, 1, 1), threadGroup: (split * 32, 1, 1),
            outputShapes: ns.map { [rows, $0] }, outputDTypes: ns.map { _ in .float32 })
        let lead = Array(x.shape.dropLast())
        return zip(outs, ns).map { $0.reshaped(lead + [$1]) }
    }

    static func linear(_ x: MLXArray, _ layer: Linear) -> MLXArray {
        linears(x, [layer])?[0] ?? layer(x)
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

    static func dynConv(_ h: MLXArray, dyn: MLXArray, base: MLXArray, part: Int, residual: MLXArray?) -> MLXArray {
        let D = h.dim(-1), L = h.size / D
        return conv(
            [h, dyn, base, residual ?? h, MLXArray([Int32(D), Int32(L)])],
            template: [("D", D), ("PART", part), ("RES", residual == nil ? 0 : 1)],
            grid: ((L * D + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [h.shape], outputDTypes: [.float32])[0]
    }

    static let topk = MLXFast.metalKernel(
        name: "df2_topk16", inputNames: ["logits", "vsz"], outputNames: ["ids", "vals"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
            const int Vn = vsz[0];
            const device float* lr = logits + (size_t)row * Vn;
            const float NEG = -INFINITY;
            float tv[4] = {NEG, NEG, NEG, NEG}; int ti[4] = {-1, -1, -1, -1};
            for (int n = int(tid); n < Vn; n += 512) {
              const float v = lr[n];
              if (v > tv[3]) {
                int k = 3;
                while (k > 0 && v > tv[k - 1]) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; k--; }
                tv[k] = v; ti[k] = n;
              }
            }
            threadgroup float cv[2048]; threadgroup int ci[2048];
            threadgroup float wv[32]; threadgroup int ws[32];
            if (tid < 32) { wv[tid] = NEG; ws[tid] = 0x7fffffff; }
            for (int k = 0; k < 4; k++) { cv[tid * 4 + k] = tv[k]; ci[tid * 4 + k] = ti[k]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int r = 0; r < 16; r++) {
              float bv = NEG; int bs = 0x7fffffff;
              for (int k = 0; k < 4; k++) { const float v = cv[tid * 4 + k]; if (v > bv) { bv = v; bs = int(tid * 4 + k); } }
              const float m = simd_max(bv);
              int sl = bv == m ? bs : 0x7fffffff; sl = simd_min(sl);
              if (lane == 0) { wv[sg] = m; ws[sg] = sl; }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (sg == 0) {
                const float v2 = wv[lane]; const float m2 = simd_max(v2);
                int s2 = v2 == m2 ? ws[lane] : 0x7fffffff; s2 = simd_min(s2);
                if (lane == 0) { ids[row * 16 + r] = ci[s2]; vals[row * 16 + r] = cv[s2]; cv[s2] = NEG; }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """)

    static let selector = MLXFast.metalKernel(
        name: "df2_selector", inputNames: ["cand", "unary", "proj", "pred", "succ", "anchor"], outputNames: ["path"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
            const int P = int(cand_shape[0]);
            threadgroup float ev[16]; threadgroup int cur;
            if (tid == 0) { cur = anchor[0]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int p = 0; p < P; p++) {
              const int pr = cur;
              const int c = cand[p * 16 + sg];
              float acc = 0.0f;
              for (int r = int(lane); r < 256; r += 32) {
                acc += float(pred[(size_t)pr * 256 + r]) * float(proj[p * 256 + r]) * float(succ[(size_t)c * 256 + r]);
              }
              acc = simd_sum(acc);
              if (lane == 0) { ev[sg] = float(unary[p * 16 + sg]) + acc; }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (tid == 0) {
                int best = 0; float bv = ev[0];
                for (int j = 1; j < 16; j++) { if (ev[j] > bv) { bv = ev[j]; best = j; } }
                cur = cand[p * 16 + best]; path[p] = cur;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """)

    /// Fused top-16 + greedy selector for `selectGreedy` (B = 1).
    static func select(
        logits: MLXArray, projected: MLXArray, anchor: MLXArray, pred: MLXArray, succ: MLXArray
    ) -> MLXArray {
        let V = logits.dim(-1), P = logits.size / V
        let top = topk(
            [logits.reshaped([P, V]).asType(.float32), MLXArray([Int32(V)])],
            grid: (P * 512, 1, 1), threadGroup: (512, 1, 1),
            outputShapes: [[P, 16], [P, 16]], outputDTypes: [.int32, .float32])
        let path = selector(
            [top[0], top[1], projected.reshaped([P, -1]).asType(.float32), pred, succ, anchor.asType(.int32)],
            grid: (512, 1, 1), threadGroup: (512, 1, 1),
            outputShapes: [[P]], outputDTypes: [.int32])[0]
        return path.reshaped([1, P])
    }
}
