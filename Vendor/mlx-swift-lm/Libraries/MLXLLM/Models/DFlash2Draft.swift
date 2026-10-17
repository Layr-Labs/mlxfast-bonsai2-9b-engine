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

/// The same sliding mask is reused by layers with equal geometry.
final class DFlash2SlidingMaskMemo {
    private var key: [Int]?
    private var cached: MLXArray?

    var retainedArrays: [MLXArray] { cached.map { [$0] } ?? [] }

    func mask(
        contextLength: Int, blockLength: Int, slidingWindow: Int, isCausal: Bool
    ) -> MLXArray {
        let requested = [contextLength, blockLength, slidingWindow, isCausal ? 1 : 0]
        if let cached, key == requested { return cached }
        let made = DFlash2SlidingMask.make(
            contextLength: contextLength, blockLength: blockLength,
            slidingWindow: slidingWindow, isCausal: isCausal)
        key = requested
        cached = made
        return made
    }
}

/// Stack Q8 rows without changing packed words, scales, or biases.
private final class DFlash2QuantizedStack {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let ends: [Int]

    init?(_ layers: [Linear]) {
        let pieces = layers.compactMap { $0 as? QuantizedLinear }
        guard pieces.count == layers.count, let first = pieces.first,
            pieces.allSatisfy({
                $0.bits == 8 && $0.groupSize == 32 && $0.mode == .affine && $0.bias == nil
                    && $0.weight.dtype == .uint32 && $0.weight.ndim == 2
                    && $0.weight.dim(1) == first.weight.dim(1)
                    && $0.scales.dtype == .float32 && $0.scales.ndim == 2
                    && $0.scales.dim(1) == first.scales.dim(1)
                    && $0.biases?.dtype == .float32 && $0.biases?.shape == $0.scales.shape
            })
        else { return nil }
        var ends = [0]
        for piece in pieces { ends.append(ends.last! + piece.weight.dim(0)) }
        self.ends = ends
        weight = concatenated(pieces.map { $0.weight }, axis: 0)
        scales = concatenated(pieces.map { $0.scales }, axis: 0)
        biases = concatenated(pieces.map { $0.biases! }, axis: 0)
        eval(weight, scales, biases)
        for (index, piece) in pieces.enumerated() {
            let rows = ends[index] ..< ends[index + 1]
            piece.update(parameters: ModuleParameters.unflattened([
                "weight": weight[rows, 0...], "scales": scales[rows, 0...],
                "biases": biases[rows, 0...],
            ]))
        }
    }

    private init(weight: MLXArray, scales: MLXArray, biases: MLXArray, ends: [Int]) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.ends = ends
        eval(weight, scales, biases)
    }

    func droppingFirst() -> DFlash2QuantizedStack {
        let start = ends[1]
        return DFlash2QuantizedStack(
            weight: weight[start..., 0...], scales: scales[start..., 0...],
            biases: biases[start..., 0...], ends: ends.dropFirst().map { $0 - start })
    }

    func product(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x, weight, scales: scales, biases: biases,
            transpose: true, groupSize: 32, bits: 8)
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let wide = product(x)
        return (0 ..< ends.count - 1).map { wide[.ellipsis, ends[$0] ..< ends[$0 + 1]] }
    }
}

private enum DFlash2StackedSwiGLU {
    static func apply(_ wide: MLXArray) -> MLXArray {
        let width = wide.dim(-1) / 2
        return kernel(
            [wide], template: [("WIDTH", width)],
            grid: (wide.size / 2, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [Array(wide.shape.dropLast()) + [width]],
            outputDTypes: [.float32])[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_stacked_q8_swiglu", inputNames: ["wide"], outputNames: ["out"],
        source: """
            #pragma clang fp contract(off)
            const uint i = thread_position_in_grid.x;
            const uint row = i / WIDTH;
            const uint col = i % WIDTH;
            const float gate = wide[row * (2 * WIDTH) + col];
            const float up = wide[row * (2 * WIDTH) + WIDTH + col];
            const float y = 1 / (1 + metal::exp(metal::abs(gate)));
            const float sigmoid = gate < 0 ? y : 1 - y;
            const float activated = gate * sigmoid;
            out[i] = activated * up;
            """)
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
    private var blockStack: DFlash2QuantizedStack?
    private var contextStack: DFlash2QuantizedStack?

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

    func prepareStacks() {
        blockStack = DFlash2QuantizedStack([qProj, kProj, vProj])
        contextStack = blockStack?.droppingFirst()
    }

    /// - Parameters:
    ///   - x: the block, `[B, blockLength, hidden]`.
    ///   - context: the projected target hidden state, `[B, contextLength, hidden]`.
    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache,
        masks: DFlash2SlidingMaskMemo
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

        let block = blockStack?(x)
        let contextProjection = contextStack?(context)
        var queries = block?[0] ?? qProj(x)
        var contextKeys = contextProjection?[0] ?? kProj(context)
        var contextValues = contextProjection?[1] ?? vProj(context)
        var blockKeys = block?[1] ?? kProj(x)
        var blockValues = block?[2] ?? vProj(x)

        queries = qNorm(queries.reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
        contextKeys = kNorm(contextKeys.reshaped(B, contextLength, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        contextValues = contextValues.reshaped(B, contextLength, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        blockKeys = kNorm(blockKeys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        blockValues = blockValues.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        // The block sits immediately after the context, so both the queries and
        // the block's own keys rotate at the context's far end.
        let blockOffset = cache.offset + contextLength
        queries = rope(queries, offset: blockOffset)
        contextKeys = rope(contextKeys, offset: cache.offset)
        blockKeys = rope(blockKeys, offset: blockOffset)

        // Only the CONTEXT keys and values enter the cache. The block's own keys
        // and values are concatenated for this forward and then dropped.
        let (cachedKeys, cachedValues) = cache.update(keys: contextKeys, values: contextValues)
        let cachedLength = cachedKeys.dim(2)
        let keys = concatenated([cachedKeys, blockKeys], axis: 2)
        let values = concatenated([cachedValues, blockValues], axis: 2)

        var mask: MLXArray?
        if let slidingWindow {
            mask = masks.mask(
                contextLength: cachedLength,
                blockLength: L,
                slidingWindow: slidingWindow,
                isCausal: isCausal)
        } else if isCausal {
            mask = createCausalMask(n: L, offset: cachedLength)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// Preserve each operation's dtype, including BF16 input with FP32 taps.
private enum DFlash2GroupedConvKernel {
    static func apply(
        hidden: MLXArray, projection: MLXArray, base: MLXArray,
        groupSize: Int, tap: Int, residual: MLXArray? = nil
    ) -> MLXArray? {
        guard hidden.ndim == 3, base.ndim == 3 else { return nil }
        let batch = hidden.dim(0), length = hidden.dim(1), width = hidden.dim(2)
        let kernelSize = base.dim(1), groups = width / groupSize
        let types: [DType] = [.bfloat16, .float16, .float32]
        guard batch > 0, length > 0, width % 256 == 0,
            types.contains(hidden.dtype), types.contains(projection.dtype),
            types.contains(base.dtype), base.shape == [2, kernelSize, width],
            projection.shape == [batch, length, 2 * kernelSize * groups],
            residual.map({ $0.shape == hidden.shape && types.contains($0.dtype) }) ?? true
        else { return nil }
        let convType = hidden.dtype == projection.dtype ? hidden.dtype : .float32
        let outType = residual.map { $0.dtype == convType ? convType : .float32 } ?? convType
        let template: [(String, any KernelTemplateArg)] = [
            ("H", hidden.dtype), ("D", projection.dtype), ("B", base.dtype),
            ("P", convType), ("OutT", outType), ("KS", kernelSize),
            ("GS", groupSize), ("TAP", tap),
        ]
        let inputs = [hidden, projection, base] + (residual.map { [$0] } ?? [])
        return (residual == nil ? plain : added)(
            inputs, template: template, grid: (width, length, batch),
            threadGroup: (256, 1, 1), outputShapes: [hidden.shape],
            outputDTypes: [outType])[0]
    }

    private static let header = """
        template <typename H, typename D, typename B, typename P, int KS, int GS, int TAP>
        inline P dflash2_grouped_conv_exact(
            const device H* h, const device D* dyn, const device B* base,
            uint b, uint l, uint c, uint L, uint W) {
        #pragma clang fp contract(off)
          const uint G = W / GS;
          const uint g = c / GS;
          const size_t row = size_t(b) * L + l;
          P out = P(0.0f);
          for (int o = 0; o < KS; ++o) {
            const H v = l >= uint(o) ? h[(row - o) * W + c] : H(0.0f);
            const H kb = H(base[(size_t(TAP) * KS + o) * W + c]);
            const H kv = H(kb * v);
            if (o == 0) out = P(H(H(0.0f) + kv));
            else out = P(out + P(kv));
            const P d = P(dyn[row * (2 * KS * G) + (size_t(TAP) * KS + o) * G + g]);
            const P dv = P(d * P(v));
            out = P(out + dv);
          }
          return out;
        }
        """

    private static let source = """
        #pragma clang fp contract(off)
        const uint c = thread_position_in_grid.x;
        const uint l = thread_position_in_grid.y;
        const uint b = thread_position_in_grid.z;
        const uint W = threads_per_grid.x;
        const uint L = threads_per_grid.y;
        const size_t i = (size_t(b) * L + l) * W + c;
        const P conv = dflash2_grouped_conv_exact<H, D, B, P, KS, GS, TAP>(
            h, dyn, base, b, l, c, L, W);
        """

    private static let plain = MLXFast.metalKernel(
        name: "dflash2_grouped_conv_exact", inputNames: ["h", "dyn", "base"],
        outputNames: ["out"], source: source + "\nout[i] = OutT(conv);\n",
        header: header)
    private static let added = MLXFast.metalKernel(
        name: "dflash2_grouped_conv_exact_residual", inputNames: ["h", "dyn", "base", "res"],
        outputNames: ["out"], source: source + "\nout[i] = OutT(res[i]) + OutT(conv);\n",
        header: header)
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
        let projection = kernelProjection(hidden)
        if let fused = DFlash2GroupedConvKernel.apply(
            hidden: hidden, projection: projection, base: baseKernel,
            groupSize: groupSize, tap: 0)
        {
            return (fused, projection)
        }
        let dynamic = projection
            .reshaped(hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (
            Self.convolve(
                hidden: hidden,
                dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0],
                groupSize: groupSize),
            projection
        )
    }

    /// The second tap, over the sub-layer's output.
    func finish(_ hidden: MLXArray, projection: MLXArray, residual: MLXArray) -> MLXArray {
        if let fused = DFlash2GroupedConvKernel.apply(
            hidden: hidden, projection: projection, base: baseKernel,
            groupSize: groupSize, tap: 1, residual: residual)
        {
            return fused
        }
        let dynamic = projection
            .reshaped(hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return residual + Self.convolve(
            hidden: hidden, dynamic: dynamic[0..., 0..., 1, 0..., 0...],
            base: baseKernel[1], groupSize: groupSize)
    }
}

// MARK: - The decoder layer

private let dflash2SwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { gate, up in
        (gate * MLX.sigmoid(gate)) * up
    }
}()

private final class DFlash2MLP: Module, UnaryLayer {
    private var gateUpStack: DFlash2QuantizedStack?
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func prepareStack() {
        gateUpStack = DFlash2QuantizedStack([gate, up])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let gateUpStack {
            return down(DFlash2StackedSwiGLU.apply(gateUpStack.product(x)))
        }
        let g = gate(x), u = up(x)
        let activated = g.dtype == .float32 && u.dtype == .float32
            ? dflash2SwiGLU(g, u) : silu(g) * u
        return down(activated)
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

    func prepareStacks() {
        selfAttn.prepareStacks()
        mlp.prepareStack()
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache,
        masks: DFlash2SlidingMaskMemo
    ) -> MLXArray {
        let (attentionInput, attentionTaps) = attentionConv.prepare(inputLayerNorm(x))
        let attended = attentionConv.finish(
            selfAttn(attentionInput, context: context, rope: rope, cache: cache, masks: masks),
            projection: attentionTaps, residual: x)
        let (mlpInput, mlpTaps) = mlpConv.prepare(postAttentionLayerNorm(attended))
        return mlpConv.finish(mlp(mlpInput), projection: mlpTaps, residual: attended)
    }
}

/// A lossless, inference-only resident copy of one BF16 codebook. The
/// checkpoint is verified before its registered payload is replaced.
private final class DFlash2PackedCodebook {
    let packed: MLXArray
    let lookup: MLXArray
    let rows: Int
    let width: Int

    init?(_ source: MLXArray) throws {
        guard source.dtype == .bfloat16, source.ndim == 2,
            source.dim(1) % 8 == 0, source.size > 0 else { return nil }
        rows = source.dim(0)
        width = source.dim(1)
        let bits = source.view(dtype: .uint16).asArray(UInt16.self)
        var codes = [UInt16](repeating: .max, count: 65536)
        var values = [UInt16]()
        var words = [UInt32](repeating: 0, count: bits.count / 8 * 3)
        for i in stride(from: 0, to: bits.count, by: 8) {
            var block: UInt64 = 0
            let out = i / 8 * 3
            for j in 0 ..< 8 {
                let value = Int(bits[i + j])
                if codes[value] == .max {
                    guard values.count < 4096 else { return nil }
                    codes[value] = UInt16(values.count)
                    values.append(bits[i + j])
                }
                block |= UInt64(codes[value]) << ((j % 2) * 12)
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
        lookup = MLXArray(values)
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
            const uint index = ids[i / (WIDTH / 2)] * (WIDTH / 2) + i % (WIDTH / 2);
            const uint bit = (index & 3u) * 24u;
            const uint word = (index >> 2u) * 3u + (bit >> 5u);
            const uint shift = bit & 31u;
            uint codes = packed[word] >> shift;
            if (shift > 8u) codes |= packed[word + 1u] << (32u - shift);
            const ushort2 values(lookup[codes & 4095u], lookup[(codes >> 12u) & 4095u]);
            if constexpr (FLOAT_OUTPUT) {
                const float2 widened(cast_to<float>(uint16_to_bfloat16(values.x)),
                                     cast_to<float>(uint16_to_bfloat16(values.y)));
                reinterpret_cast<device float2*>(out)[i] = widened;
            } else {
                reinterpret_cast<device ushort2*>(out)[i] = values;
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

    /// Called only after strict checkpoint loading. The two registered keys
    /// remain present; their inference payloads become packed UInt32 words.
    func preparePackedCodebooks() {
        codebookCopies.predecessor = try? DFlash2PackedCodebook(predecessorCodebook)
        codebookCopies.successor = try? DFlash2PackedCodebook(successorCodebook)
        if let copy = codebookCopies.predecessor {
            predecessorCodebook._updateInternal(copy.packed)
        }
        if let copy = codebookCopies.successor {
            successorCodebook._updateInternal(copy.packed)
        }
    }

    // Packed words are parameters; these lookup tables have separate buffers.
    var residencyLookupArrays: [MLXArray] {
        [codebookCopies.predecessor?.lookup, codebookCopies.successor?.lookup].compactMap { $0 }
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
        let candidates: MLXArray
        if topK == 16, vocabularySize % 512 == 0,
            StreamOrDevice.default == StreamOrDevice.gpu
        {
            let blocks = vocabularySize / 512
            let tiled = logits.reshaped([-1, blocks, 512])
            let local = argPartition(tiled, kth: 512 - topK, axis: -1)[
                0..., 0..., (512 - topK)...]
            let offsets = MLXArray((0 ..< blocks).map { UInt32($0 * 512) })
                .reshaped([1, blocks, 1])
            let ids = (local + offsets).reshaped([-1, blocks * topK])
            let values = takeAlong(tiled, local, axis: -1).reshaped([-1, blocks * topK])
            let selected = argPartition(values, kth: blocks * topK - topK, axis: -1)[
                0..., (blocks * topK - topK)...]
            candidates = takeAlong(ids, selected, axis: -1)
                .reshaped(Array(logits.shape.dropLast()) + [topK])
        } else {
            candidates = argPartition(logits, kth: vocabularySize - topK, axis: -1)[
                0..., 0..., (vocabularySize - topK)...]
        }
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)
        let floatCodebookOutput = projected.dtype == .float32

        var predecessor = anchor
        var path = [MLXArray]()
        path.reserveCapacity(hidden.dim(1))
        for position in 0 ..< hidden.dim(1) {
            let positionCandidates = candidates[0..., position, 0...]
            let edges =
                ((codebookCopies.predecessor?.gather(predecessor, floatOutput: floatCodebookOutput)
                    ?? take(predecessorCodebook, predecessor, axis: 0))
                    .expandedDimensions(axis: 1)
                    * projected[0..., position, 0...].expandedDimensions(axis: 1)
                    * (codebookCopies.successor?.gather(positionCandidates, floatOutput: floatCodebookOutput)
                        ?? take(successorCodebook, positionCandidates, axis: 0)))
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
    private static let submitFirstLayer: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH_FIRST_SUBMIT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

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
        let context = hiddenNorm(fc(targetHidden.asType(dtype)))

        let masks = DFlash2SlidingMaskMemo()
        for (index, layer) in layers.enumerated() {
            h = layer(h, context: context, rope: rope, cache: cache[index], masks: masks)
            if index == 0, Self.submitFirstLayer { asyncEval(h) }
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return norm(h)
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
        return candidateSelector.selectGreedy(
            hidden: hidden,
            logits: try logits(hidden),
            anchor: MLXArray(anchor.map { Int32($0) }))
    }

    /// Build only the first layer before readback; finish after exact adoption.
    public func prepareNextBlock(
        anchor: MLXArray, targetHidden: MLXArray, cache: [KVCache], blockSize: Int
    ) throws -> (leading: MLXArray, retained: [MLXArray], finish: () throws -> MLXArray) {
        guard blockSize >= 2 else { throw DFlash2Error.invalidBlockSize(blockSize) }
        guard let target else { throw DFlash2Error.notBound }
        guard cache.count == layers.count else {
            throw DFlash2Error.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlash2Error.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize, actual: targetHidden.dim(-1))
        }
        let batch = anchor.size
        let masks = MLXArray(Array(
            repeating: Int32(config.maskTokenId), count: batch * (blockSize - 1)))
            .reshaped([batch, blockSize - 1])
        let block = concatenated(
            [anchor.asType(.int32).reshaped([batch, 1]), masks], axis: 1)
        var h = target.embedTokensForDFlash2(block).asType(dtype)
        if config.dflash.inputEmbeddingScale != 1 {
            h = h * config.dflash.inputEmbeddingScale
        }
        let context = hiddenNorm(fc(targetHidden.asType(dtype)))
        let memo = DFlash2SlidingMaskMemo()
        let leading = layers[0](h, context: context, rope: rope, cache: cache[0], masks: memo)
        if Self.submitFirstLayer { asyncEval(leading) }
        return (leading, [leading, context, anchor] + memo.retainedArrays, { [self] in
            var h = leading
            for (index, layer) in layers.enumerated().dropFirst() {
                h = layer(h, context: context, rope: rope, cache: cache[index], masks: memo)
            }
            let hidden = norm(h[0..., 1..., 0...])
            return candidateSelector.selectGreedy(
                hidden: hidden, logits: try logits(hidden), anchor: anchor.reshaped([batch]))
        })
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
        weights.removeAll()
        drafter.candidateSelector.preparePackedCodebooks()
        for layer in drafter.layers { layer.prepareStacks() }
        eval(drafter)
        return drafter
    }
}
