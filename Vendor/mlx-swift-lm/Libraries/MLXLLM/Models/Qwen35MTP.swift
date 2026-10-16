// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import CoreFoundation
import Foundation
@_spi(QuantizedConstantCache) import MLX
import MLXLMCommon
import MLXNN

// The pinned MTP head carries FP16 affine metadata while the target feeds it
// FP32 activations. Reuse only the exact constant widening that quantizedMM
// would otherwise repeat at every draft step. This cache is input-independent;
// the packed weights and the module's original parameter tree remain unchanged.
final class Qwen35MTPQuantizedLinear: QuantizedLinear {
    private let mtpScaleCastCache = ConstantArrayCastCache()
    private let mtpOffsetCastCache = ConstantArrayCastCache()

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer {
            mtpScaleCastCache.clear()
            mtpOffsetCastCache.clear()
        }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard mode == .affine, x.dtype == .float32, scales.dtype == .float16 else {
            return super.callAsFunction(x)
        }
        let widenedScales = mtpScaleCastCache.cachedCast(
            scales, to: x.dtype, allowFloat16: true) ?? scales
        let widenedOffsets = biases.map {
            mtpOffsetCastCache.cachedCast($0, to: x.dtype, allowFloat16: true) ?? $0
        }
        var output = quantizedMM(
            x, weight, scales: widenedScales, biases: widenedOffsets, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
        if let bias { output = output + bias }
        return output
    }
}

// MARK: - Module-level MTP flag

/// Controls whether Qwen3.5/3.6 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _qwen35MTPEnabled: Bool = false

/// Fail-loud errors for the production-safe inline Qwen MTP loader.
///
/// Unlike the legacy process-global attachment flag above, this loader builds
/// an assistant explicitly for one already-loaded target and never changes
/// model construction behavior process-wide.
public enum Qwen35InlineMTPError: Error, LocalizedError, Sendable, Equatable {
    case invalidConfiguration(String)
    case incompatibleTarget(field: String, artifact: Int, target: Int)
    case invalidWeightIndex(String)
    case missingWeights
    case duplicateWeight(String)
    case missingQuantization(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid inline Qwen MTP configuration: \(detail)."
        case .incompatibleTarget(let field, let artifact, let target):
            return "Inline Qwen MTP target mismatch at \(field): artifact=\(artifact), target=\(target)."
        case .invalidWeightIndex(let detail):
            return "Invalid inline Qwen MTP weight index: \(detail)."
        case .missingWeights:
            return "The checkpoint declares inline Qwen MTP but contains no matching tensors."
        case .duplicateWeight(let key):
            return "The inline Qwen MTP tensor \(key) appears more than once."
        case .missingQuantization(let path):
            return "The quantized inline Qwen MTP module \(path) has no matching quantization entry."
        }
    }
}

/// Parsed, bounded metadata for an inline Qwen MTP assistant.
/// Internal so focused tests can validate checkpoint interpretation without
/// constructing hundreds of MiB of weights.
struct Qwen35InlineMTPMetadata: Sendable {
    let textConfiguration: Qwen35TextConfiguration
    let prefix: String
    let blockSize: Int
    let quantization: BaseConfiguration.PerLayerQuantization?

    func resolvedQuantization(
        for path: String
    ) -> BaseConfiguration.Quantization? {
        quantization?.quantization(layer: path)
    }
}

// MARK: - MTP history KV

extension Qwen35Attention {
    /// Append committed proposal-head history without computing unused query,
    /// gate, attention, or output-projection rows.
    func appendMTPHistoryKV(_ x: MLXArray, cache: any KVCache) {
        let batch = x.dim(0)
        let length = x.dim(1)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        _ = cache.update(keys: keys, values: values)
    }
}

// MARK: - MTPDecoderLayer

/// Full-attention transformer layer used inside the Qwen3.5/3.6 MTP head.
/// Unlike `Qwen35DecoderLayer`, this always uses full attention (never SSM/linear).
/// MoE config is honoured when `num_experts > 0`.
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPDecoderLayer
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            // Split gate/up: the assistant's quantization table
            // (`mtplx_mtp_quantization`) and the checkpoint's mtp.* tensors
            // are keyed on the split module paths, and the MTP head loads
            // outside the target sanitizers that perform gate/up fusion.
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args, fuseGateUp: false)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    /// Populate this layer's K/V history without computing a dead decoder
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendMTPHistoryKV(inputLayerNorm(x), cache: cache)
    }
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's final-normalized hidden state at position t with the
/// embedding of the sampled main token (t+1) to predict the draft token at (t+2).
///
/// Architecture (port of PR #990):
/// ```
/// pre_fc_norm_hidden:    RMSNorm(hidden_size)
/// pre_fc_norm_embedding: RMSNorm(hidden_size)
/// fc:                    Linear(hidden_size * 2 → hidden_size, bias: false)
/// layers:                [MTPDecoderLayer]  × mtp_num_hidden_layers
/// norm:                  RMSNorm(hidden_size)
/// ```
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPModule
final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // `layers` uses the default ModuleInfo key derived from the property name.
    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    /// An UNQUANTIZED head computes in its own weight dtype. The Bonsai 2 9B
    /// `mtp/` drafter is FP16 and the Prism trunk hands it FP32 activations;
    /// a mixed-dtype matmul promotes, and so copies, every head weight on
    /// every draft step. The activations entering the head are cast instead.
    /// The weights are never cast, and a quantized head keeps the incoming
    /// dtype. Only the draft numerics move; the target verifies every token.
    private func entering(_ x: MLXArray) -> MLXArray {
        guard !(fc is QuantizedLinear), x.dtype != fc.weight.dtype else { return x }
        return x.asType(fc.weight.dtype)
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(entering(embeds))
        let h = preFcNormHidden(entering(hidden))
        var fused = fc(concatenated([e, h], axis: -1))

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }

        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        return norm(fused)
    }

    /// Append every leading committed row through a K/V-only path and compute
    /// a full decoder output only for the final proposal row. Multi-layer heads
    /// fail closed before any cache mutation because later layers require the
    /// omitted leading outputs.
    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
            hidden.dim(1) > 1,
            nextTokenIds.dim(1) == hidden.dim(1)
        else { return nil }

        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(entering(embeds))
        let h = preFcNormHidden(entering(hidden))
        let fused = fc(concatenated([e, h], axis: -1))
        let historyCount = fused.dim(1) - 1
        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }
}

// MARK: - Artifact-scoped inline assistant

/// An explicitly loaded Qwen3.5/3.6 MTP assistant bound to one target.
///
/// The assistant owns only the tensors below `mtp.*`. Target embeddings and
/// the LM head are called through the bound target, so loading this object does
/// not duplicate the target checkpoint and does not require
/// `_qwen35MTPEnabled`.
public final class Qwen35InlineMTPAssistant: Module, @unchecked Sendable {
    static let cacheAllocationStep = 256
    /// The longest draft chain: the head is applied to its own output at most
    /// this many times in one round. David ruled 6 for the Bonsai 2 9B MLX
    /// track (2026-10-14), the depth the CUDA track serves. A fixed declared
    /// depth above it is refused by the round driver, never clamped.
    public static let maximumDepth = 6

    private let mtp: Qwen35MTPModule
    private let target: Qwen35TextModel
    private let installedVerificationMode: CBv2MTPVerificationMode
    /// Draft-only output head over a fixed slice of the target vocabulary.
    /// nil (small fixtures, tied or unpacked heads) keeps the full head.
    private let draftVocabulary: Qwen35MTPDraftVocabulary?

    public let blockSize: Int
    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }

    var prefixCheckpointGeometry: (width: Int, dtype: DType, vocabulary: Int, verification: String) {
        (target.configuration.hiddenSize, target.model.norm.weight.dtype,
         target.vocabularySize, installedVerificationMode.rawValue)
    }

    private init(
        configuration: Qwen35TextConfiguration,
        blockSize: Int,
        target: Qwen35TextModel,
        verificationMode: CBv2MTPVerificationMode?
    ) {
        self.mtp = Qwen35MTPModule(configuration)
        self.blockSize = blockSize
        self.target = target
        self.draftVocabulary = Qwen35MTPDraftVocabulary(target: target)
        self.installedVerificationMode = Self.resolvedVerificationMode(
            requested: verificationMode,
            forceSerialEnvironment: Self.forceSerialVerification)
        super.init()
    }

    static func resolvedVerificationMode(
        requested: CBv2MTPVerificationMode?, forceSerialEnvironment: Bool
    ) -> CBv2MTPVerificationMode {
        if forceSerialEnvironment { return .serialTarget }
        return requested ?? .rectangular
    }

    /// Allocate the assistant's own autoregressive full-attention KV.
    /// Target recurrent and attention state is never reused as assistant KV.
    public func makeCache() -> [any KVCache] {
        mtp.layers.map { _ in
            let cache = KVCacheSimple()
            cache.step = Self.cacheAllocationStep
            return cache as any KVCache
        }
    }

    /// Advance the assistant by one or more already-chosen target/draft tokens.
    /// Returns logits from the target's shared output projection and the
    /// assistant hidden used to continue a draft chain.
    public func forward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let output = mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
        return (headLogits(output), output)
    }

    /// The target's shared output projection over assistant hidden states.
    func headLogits(_ output: MLXArray) -> MLXArray {
        if target.configuration.tieWordEmbeddings {
            return target.model.embedTokens.asLinear(output)
        }
        return target.lmHead!(output)
    }

    /// Apply the exact final norm of the bound target once at the boundary
    /// where target-derived hidden enters assistant history. Target hidden
    /// capture is deliberately pre-norm; recursive assistant hidden is already
    /// post-`mtp.norm` and must never pass through this helper.
    func targetFinalNorm(_ hidden: MLXArray) -> MLXArray {
        target.model.norm(hidden)
    }

    /// Assistant hidden states WITHOUT the output projection — the draft
    /// path applies the head to the last position only (or to a shortlist).
    func moduleForward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray {
        mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
    }

    /// History-flush specialization. Returns one final hidden row while every
    /// leading trusted row contributes only K/V state. nil means the head
    /// geometry cannot safely omit intermediate layer outputs.
    func moduleLastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray? {
        mtp.lastHiddenWithKVOnlyHistory(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
    }

    /// Draft logits over an engine-provided shortlist of token ids. The
    /// draft needs only an argmax, so score the gathered head rows instead
    /// of streaming the full `[V, H]` output projection (~K/V of the bytes;
    /// the 4-bit Qwen3.6 lm_head alone is ~286 MB per read). Quantized
    /// heads gather packed rows plus their scales/biases and stay on the
    /// quantized-matmul path; float heads (small test fixtures) gather
    /// plain rows.
    ///
    /// A PACKED HADAMARD head folds the transform into its rows, so the
    /// rotation the module applies in `callAsFunction` has to be applied to
    /// the hidden state here as well. `HadamardQuantizedLinear` IS a
    /// `QuantizedLinear`, so without the rotation the cast below succeeds and
    /// scores folded rows against an unrotated hidden state.
    /// `HadamardQuantizedEmbedding` is NOT a `QuantizedEmbedding`, so without
    /// its own branch the tied path reads packed integers as floats.
    func shortlistLogits(hidden: MLXArray, ids: MLXArray) -> MLXArray {
        if target.configuration.tieWordEmbeddings {
            let embed = target.model.embedTokens
            if let packed = embed as? HadamardQuantizedEmbedding {
                return quantizedMM(
                    packed.transform(hidden), packed.weight[ids],
                    scales: packed.scales[ids],
                    biases: packed.biases.map { $0[ids] },
                    transpose: true,
                    groupSize: packed.groupSize, bits: packed.bits,
                    mode: packed.mode)
            }
            if let quantized = embed as? QuantizedEmbedding {
                return quantizedMM(
                    hidden, quantized.weight[ids],
                    scales: quantized.scales[ids],
                    biases: quantized.biases.map { $0[ids] },
                    transpose: true,
                    groupSize: quantized.groupSize, bits: quantized.bits,
                    mode: quantized.mode)
            }
            return matmul(hidden, embed.weight[ids].transposed(1, 0))
        }
        let head = target.lmHead!
        let projected = (head as? HadamardQuantizedLinear).map { $0.transform(hidden) } ?? hidden
        if let quantized = head as? QuantizedLinear {
            var logits = quantizedMM(
                projected, quantized.weight[ids],
                scales: quantized.scales[ids],
                biases: quantized.biases.map { $0[ids] },
                transpose: true,
                groupSize: quantized.groupSize, bits: quantized.bits,
                mode: quantized.mode)
            if let bias = quantized.bias { logits = logits + bias[ids] }
            return logits
        }
        var logits = matmul(projected, head.weight[ids].transposed(1, 0))
        if let bias = head.bias { logits = logits + bias[ids] }
        return logits
    }

    /// Strictly load either an inline assistant declared by a combined
    /// checkpoint or a standalone `qwen3_5_mtp` artifact. Inline artifacts
    /// read only indexed keys below their declared prefix; standalone
    /// artifacts read every tensor from their bounded safetensors directory.
    public static func load(
        from modelDirectory: URL,
        target: any LanguageModel,
        verificationMode: CBv2MTPVerificationMode? = nil
    ) throws -> Qwen35InlineMTPAssistant {
        let serving = target
        let target = try qwen35TextTarget(target)
        let resolvedVerificationMode = Self.resolvedVerificationMode(
            requested: verificationMode,
            forceSerialEnvironment: Self.forceSerialVerification)
        if resolvedVerificationMode == .rectangularExact && !target.model.exactTargetVerify {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "rectangular_exact verification requires exact target arithmetic")
        }
        let metadata = try loadMetadata(from: modelDirectory)
        try validate(metadata.textConfiguration, against: target.configuration)

        let indexed: [String: MLXArray]
        if metadata.prefix.isEmpty {
            indexed = try loadStandaloneWeights(from: modelDirectory)
        } else {
            indexed = try loadIndexedWeights(
                from: modelDirectory, prefix: metadata.prefix)
        }
        let scaledPaths = Set(indexed.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        })
        for path in scaledPaths
        where metadata.resolvedQuantization(for: path) == nil {
            throw Qwen35InlineMTPError.missingQuantization(path)
        }

        var configuration = metadata.textConfiguration
        configuration.mtpNumHiddenLayers = try decoderLayerCount(
            in: indexed.keys, declared: metadata.textConfiguration.mtpNumHiddenLayers)
        let assistant = Qwen35InlineMTPAssistant(
            configuration: configuration,
            blockSize: metadata.blockSize,
            target: target,
            verificationMode: verificationMode)
        if !scaledPaths.isEmpty {
            quantize(
                model: assistant.mtp,
                filter: { path, _ in
                    guard scaledPaths.contains(path) else { return nil }
                    return metadata.resolvedQuantization(for: path)?.asTuple
                },
                apply: { module, groupSize, bits, mode in
                    guard let linear = module as? Linear, !(linear is QuantizedLinear) else {
                        return quantizeSingle(
                            layer: module, groupSize: groupSize, bits: bits, mode: mode)
                    }
                    return Qwen35MTPQuantizedLinear(
                        weight: linear.weight, bias: linear.bias,
                        groupSize: groupSize, bits: bits, mode: mode)
                })
        }

        try assistant.mtp.update(
            parameters: ModuleParameters.unflattened(indexed), verify: [.all])
        eval(assistant.mtp)
        requantizeHeadLinears(in: assistant.mtp)
        // Warm after re-quantization, so the warm round compiles the
        // kernels of the head geometry the scored rounds run.
        assistant.warmEngineRound(serving: serving)
        // Once more at the resident's boot warm, after the runner has adopted
        // the model (27b 76841fed: a load-time round alone left part of the
        // first timed round's cost in place in some processes).
        CBv2DeferredLoadWarm.register { [weak assistant] in
            assistant?.warmEngineRound(serving: serving)
        }
        return assistant
    }

    /// On-load head re-encoding, permitted by the head re-quantization
    /// exception (participant contract section 4.4): in memory only, within
    /// the declaration cap, after the export's tensors were bound at their
    /// own geometry. The export on disk and every target tensor — the shared
    /// embedding, the lm_head, and the draft-vocabulary copy of its rows —
    /// stay untouched. The head only proposes; re-encoding can move draft
    /// choices and therefore acceptance, never bypass target verification.
    static let headRequantGeometry: (groupSize: Int, bits: Int)? = (128, 4)

    private static func requantizeHeadLinears(in head: Qwen35MTPModule) {
        guard let geometry = headRequantGeometry else { return }
        quantize(
            model: head,
            filter: { _, module in
                guard let linear = module as? Qwen35MTPQuantizedLinear,
                    linear.mode == .affine,
                    linear.groupSize != geometry.groupSize || linear.bits != geometry.bits,
                    (linear.weight.dim(1) * 32 / linear.bits) % geometry.groupSize == 0
                else { return nil }
                return (geometry.groupSize, geometry.bits, QuantizationMode.affine)
            },
            apply: { module, groupSize, bits, mode in
                guard let linear = module as? Qwen35MTPQuantizedLinear else { return nil }
                let dense = dequantized(
                    linear.weight, scales: linear.scales, biases: linear.biases,
                    groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode,
                    dtype: .float16)
                let (weight, scales, biases) = quantized(
                    dense, groupSize: groupSize, bits: bits, mode: mode)
                let replacement = Qwen35MTPQuantizedLinear(
                    weight: weight, bias: linear.bias, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: mode)
                // Materialize the packed replacement now so the dense FP16
                // intermediate frees before the next module dequantizes.
                eval(replacement)
                return replacement
            })
    }

    private static func qwen35TextTarget(
        _ target: any LanguageModel
    ) throws -> Qwen35TextModel {
        if let target = target as? Qwen35TextModel { return target }
        if let target = target as? Qwen35Model { return target.languageModel }
        throw Qwen35InlineMTPError.invalidConfiguration(
            "target type \(String(describing: type(of: target))) is not Qwen3.5/3.6")
    }

    private static let globalQuantizationKeys: Set<String> = [
        "group_size", "bits", "mode"
    ]

    /// Scalar keys understood or deliberately ignored by
    /// `BaseConfiguration.QuantizationContainer`; all other keys name modules.
    private static let quantizationMetadataKeys =
        globalQuantizationKeys.union([
            "quant_method", "linear_class", "quantization_mode"
        ])

    private static func selectedJSONObject(
        in root: [String: Any],
        primaryKey: String,
        fallbackKey: String
    ) throws -> [String: Any]? {
        for key in [primaryKey, fallbackKey] {
            guard let raw = root[key], !(raw is NSNull) else { continue }
            guard let object = raw as? [String: Any] else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "standalone MTP quantization must be an object")
            }
            return object
        }
        return nil
    }

    private static func decodeQuantization(
        _ rawQuantization: [String: Any]
    ) throws -> BaseConfiguration.PerLayerQuantization {
        var perLayer = [String: BaseConfiguration.QuantizationOption]()
        for (path, raw) in rawQuantization
        where !quantizationMetadataKeys.contains(path) {
            if CFGetTypeID(raw as CFTypeRef) == CFBooleanGetTypeID() {
                guard (raw as? Bool) == false else {
                    throw Qwen35InlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                perLayer[path] = .skip
            } else {
                guard let object = raw as? [String: Any] else {
                    throw Qwen35InlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                let data = try JSONSerialization.data(withJSONObject: object)
                perLayer[path] = .quantize(
                    try JSONDecoder().decode(
                        BaseConfiguration.Quantization.self, from: data))
            }
        }

        let hasGlobalQuantization = rawQuantization.keys.contains {
            globalQuantizationKeys.contains($0)
        }
        let global: BaseConfiguration.Quantization?
        if hasGlobalQuantization {
            let data = try JSONSerialization.data(withJSONObject: rawQuantization)
            global = try JSONDecoder().decode(
                BaseConfiguration.Quantization.self, from: data)
        } else {
            global = nil
        }
        return BaseConfiguration.PerLayerQuantization(
            quantization: global, perLayerQuantization: perLayer)
    }

    static func loadMetadata(from directory: URL) throws -> Qwen35InlineMTPMetadata {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = root["text_config"] as? [String: Any]
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "text_config is required")
        }

        let prefix: String
        let blockSize: Int
        let rawQuantization: [String: Any]?
        if let inline = root["mtplx_mtp"] as? [String: Any],
            inline["included"] as? Bool == true
        {
            prefix = (inline["prefix"] as? String) ?? "mtp."
            guard prefix == "mtp." else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "only the mtp. prefix is supported")
            }
            blockSize = (inline["block_size"] as? NSNumber)?.intValue ?? 3
            guard let quantization = root["mtplx_mtp_quantization"] as? [String: Any]
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "mtplx_mtp_quantization is required")
            }
            rawQuantization = quantization
        } else {
            guard (root["model_type"] as? String)?.lowercased() == "qwen3_5_mtp"
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "mtplx_mtp.included=true or model_type=qwen3_5_mtp is required")
            }
            prefix = ""
            blockSize = (root["block_size"] as? NSNumber)?.intValue ?? 3
            rawQuantization = try selectedJSONObject(
                in: root,
                primaryKey: "quantization",
                fallbackKey: "quantization_config")
        }
        guard (2...8).contains(blockSize) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "block_size \(blockSize) is outside 2...8")
        }
        let textData = try JSONSerialization.data(withJSONObject: text)
        let configuration = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: textData)
        // Zero is legal here: a standalone export can restate the TRUNK's
        // text_config, whose mtp_num_hidden_layers is 0 (the prism-ml Bonsai
        // 2 9B `mtp/` drafter does). `load` reads the depth off the tensors.
        guard (0...4).contains(configuration.mtpNumHiddenLayers) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "mtp_num_hidden_layers must be within 0...4")
        }

        let quantization: BaseConfiguration.PerLayerQuantization?
        if let rawQuantization, !rawQuantization.isEmpty {
            quantization = try decodeQuantization(rawQuantization)
        } else {
            quantization = nil
        }
        return Qwen35InlineMTPMetadata(
            textConfiguration: configuration,
            prefix: prefix,
            blockSize: blockSize,
            quantization: quantization)
    }

    /// The head's decoder depth, read off its tensors: the `layers.<i>.`
    /// indices must be exactly `0..<n`, with n within 1...4. A non-zero
    /// `mtp_num_hidden_layers` must agree with them. Zero means the export
    /// restated the trunk's text_config and declares no head depth.
    static func decoderLayerCount(
        in keys: some Sequence<String>, declared: Int
    ) throws -> Int {
        var indices = Set<Int>()
        for key in keys where key.hasPrefix("layers.") {
            let rest = key.dropFirst("layers.".count)
            guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "tensor \(key) names no decoder layer index")
            }
            indices.insert(index)
        }
        guard (1...4).contains(indices.count), indices == Set(0 ..< indices.count) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "the head's decoder layers must be 0..<n with n within 1...4")
        }
        guard declared == 0 || declared == indices.count else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "mtp_num_hidden_layers \(declared) disagrees with the "
                    + "\(indices.count) decoder layer(s) in the tensors")
        }
        return indices.count
    }

    private struct WeightIndex: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    private static func loadIndexedWeights(
        from directory: URL,
        prefix: String
    ) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let index: WeightIndex
        do {
            index = try JSONDecoder().decode(WeightIndex.self, from: Data(contentsOf: indexURL))
        } catch {
            throw Qwen35InlineMTPError.invalidWeightIndex(String(describing: error))
        }

        var byFile: [String: [(source: String, destination: String)]] = [:]
        for (key, file) in index.weightMap where key.hasPrefix(prefix) {
            guard file == URL(fileURLWithPath: file).lastPathComponent,
                file.hasSuffix(".safetensors")
            else {
                throw Qwen35InlineMTPError.invalidWeightIndex(
                    "unsafe shard path for \(key)")
            }
            let destination = String(key.dropFirst(prefix.count))
            guard !destination.isEmpty else {
                throw Qwen35InlineMTPError.invalidWeightIndex("empty stripped key")
            }
            byFile[file, default: []].append((key, destination))
        }
        guard !byFile.isEmpty else { throw Qwen35InlineMTPError.missingWeights }

        var weights: [String: MLXArray] = [:]
        for file in byFile.keys.sorted() {
            let url = directory.appendingPathComponent(file)
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for entry in byFile[file]! {
                guard let value = shard[entry.source] else {
                    throw Qwen35InlineMTPError.invalidWeightIndex(
                        "indexed tensor \(entry.source) is absent from \(file)")
                }
                guard weights.updateValue(value, forKey: entry.destination) == nil else {
                    throw Qwen35InlineMTPError.duplicateWeight(entry.destination)
                }
            }
        }
        return weights
    }

    static func loadStandaloneWeights(
        from directory: URL
    ) throws -> [String: MLXArray] {
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            throw Qwen35InlineMTPError.invalidWeightIndex(String(describing: error))
        }
        guard !urls.isEmpty, urls.count <= 64 else {
            throw Qwen35InlineMTPError.missingWeights
        }

        var weights: [String: MLXArray] = [:]
        for url in urls {
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in shard {
                guard !key.isEmpty, key.utf8.count <= 1024 else {
                    throw Qwen35InlineMTPError.invalidWeightIndex(
                        "standalone tensor key is empty or oversized")
                }
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw Qwen35InlineMTPError.duplicateWeight(key)
                }
            }
        }
        guard !weights.isEmpty else { throw Qwen35InlineMTPError.missingWeights }
        return weights
    }

    private static func validate(
        _ artifact: Qwen35TextConfiguration,
        against target: Qwen35TextConfiguration
    ) throws {
        let fields: [(String, Int, Int)] = [
            ("hidden_size", artifact.hiddenSize, target.hiddenSize),
            ("vocab_size", artifact.vocabularySize, target.vocabularySize),
            ("num_attention_heads", artifact.attentionHeads, target.attentionHeads),
            ("num_key_value_heads", artifact.kvHeads, target.kvHeads),
            ("head_dim", artifact.headDim ?? 0, target.headDim ?? 0),
            ("num_experts", artifact.numExperts, target.numExperts),
            ("num_experts_per_tok", artifact.numExpertsPerTok, target.numExpertsPerTok),
        ]
        for (field, artifactValue, targetValue) in fields
        where artifactValue != targetValue {
            throw Qwen35InlineMTPError.incompatibleTarget(
                field: field, artifact: artifactValue, target: targetValue)
        }
    }
}

extension Qwen35InlineMTPAssistant: CBv2MTPRequestStatefulDrafter {
    final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]

        /// Trusted target transitions not yet appended to the persistent head KV.
        /// Each hidden row at position t is paired with the token at t+1.
        var backlogHidden: [MLXArray] = []
        var backlogTokens: [MLXArray] = []
        /// Last trusted target hidden in an observed chunk. It becomes the
        /// preceding row when the next observed target chunk crosses a boundary.
        var targetHiddenFrontier: MLXArray?

        /// Cache geometry captured before and after the round's trusted flush.
        /// Every later head-chain input is speculative and is trimmed to
        /// `roundValidHistoryOffset` at finalize.
        var roundBaseOffset = 0
        var roundValidHistoryOffset = 0
        var roundDraftSteps = 0
        var roundInFlight = false
        var isReleased = false

        /// Trusted inputs moved out of the backlog for this round. Retaining
        /// their original roots both fences lazy concatenation and lets discard
        /// restore them without a host read.
        var roundTrustedHidden: [MLXArray] = []
        var roundTrustedTokens: [MLXArray] = []
        /// Final proposal hidden rows and draft ids that retain each lazy
        /// head-step graph until the engine's existing finalize synchronization.
        var roundRoots: [MLXArray] = []

        var cacheOffset: Int {
            guard let first = caches.first else { return 0 }
            precondition(
                caches.dropFirst().allSatisfy { $0.offset == first.offset },
                "inline Qwen MTP cache offsets diverged")
            return first.offset
        }

        private var backlogInputCount: Int {
            backlogTokens.reduce(0) { total, tokens in
                let (next, overflow) = total.addingReportingOverflow(tokens.dim(1))
                return overflow ? Int.max : next
            }
        }

        var committedInputCount: Int {
            let committedCache =
                roundInFlight ? roundValidHistoryOffset : cacheOffset
            let (total, overflow) = committedCache.addingReportingOverflow(
                backlogInputCount)
            return overflow ? Int.max : total
        }

        var stagedInputCount: Int {
            guard roundInFlight else { return 0 }
            return max(0, cacheOffset - roundValidHistoryOffset)
        }

        var materializedBytes: Int {
            let arrays =
                caches.flatMap { $0.innerState() }
                + backlogHidden + backlogTokens
                + [targetHiddenFrontier].compactMap { $0 }
                + roundTrustedHidden + roundTrustedTokens + roundRoots
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        init(caches: [any KVCache]) { self.caches = caches }

        func clearRound() {
            roundBaseOffset = cacheOffset
            roundValidHistoryOffset = cacheOffset
            roundDraftSteps = 0
            roundInFlight = false
            roundTrustedHidden.removeAll(keepingCapacity: true)
            roundTrustedTokens.removeAll(keepingCapacity: true)
            roundRoots.removeAll(keepingCapacity: true)
        }

        func clearAll() {
            caches.removeAll(keepingCapacity: false)
            backlogHidden.removeAll(keepingCapacity: false)
            backlogTokens.removeAll(keepingCapacity: false)
            targetHiddenFrontier = nil
            roundTrustedHidden.removeAll(keepingCapacity: false)
            roundTrustedTokens.removeAll(keepingCapacity: false)
            roundRoots.removeAll(keepingCapacity: false)
            roundBaseOffset = 0
            roundValidHistoryOffset = 0
            roundDraftSteps = 0
            roundInFlight = false
            isReleased = true
        }
    }

    private final class UnusedPreparedCapture: CBv2MTPPreparedCapture {}

    public var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    /// Verification policy. Rectangular scores all 1+k columns in one
    /// `[B, 1+k]` target forward. Widths one and two stage ordinary captured
    /// recurrent states. At S>=3 each GatedDeltaNet layer runs one full-window
    /// recurrence and retains compact transformed inputs: full acceptance
    /// installs the final state directly, while a strict accepted prefix lazily
    /// replays only that prefix. This replaces the serial per-column loop whose
    /// k+1 full-model re-reads made the round ≤1.0x by construction.
    ///
    /// NUMERICS POLICY: bitwise greedy parity with serial decode is NOT the
    /// bar here — batched verify changes accumulation geometry exactly like
    /// every other CBv2 batch-shape change (chunked prefill, B>1 decode).
    /// Distribution-exactness is the invariant: committed tokens are always
    /// target-authoritative (argmax when greedy, genuine target samples
    /// under target-prefix acceptance). The serial oracle remains available
    /// via `DARKBLOOM_QWEN_MTP_SERIAL=1` for A/B and certification runs.
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        installedVerificationMode
    }
    /// Production Qwen drafting self-applies the recurrent MTP head up to
    /// `maximumDepth` times. The legacy double-forward oracle stays a k=1 A/B
    /// control.
    public var maximumDraftTokens: Int? { Self.forceDoubleForward ? 1 : Self.maximumDepth }
    public var maximumSpeculativeBatch: Int? { 1 }
    /// Target-prefix acceptance is exact at any temperature (every committed
    /// token IS a target sample), so this drafter lifts the greedy gate when
    /// the engine sampler supports verify pre-sampling.
    public var supportsTargetPrefixAcceptance: Bool { true }

    /// `DARKBLOOM_QWEN_MTP_SERIAL=1/true/yes/on` pins the serial per-column
    /// verification oracle (the pre-capture-verify behavior).
    static let forceSerialVerification: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SERIAL"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()
    /// `DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD=1/true/yes/on` pins the pre-trim
    /// draft step (proposal forward + staging re-forward, both through the
    /// full lm_head) as the paired A/B control arm.
    static let forceDoubleForward: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// `DARKBLOOM_QWEN_MTP_SHORTLIST=<K>` opts the draft head into an
    /// engine-provided top-K shortlist (score K gathered lm_head rows
    /// instead of streaming all 248,320). DEFAULT OFF: measured on M4 Max
    /// (release, B=1 greedy, 256 tok, paired session 2026-08-13) the
    /// shortlist is a net LOSS at every practical K because the ids come
    /// from the target's distribution one position BEHIND the draft:
    ///   K=256   acceptance 0.79→0.58,  95.5 tok/s (full head: 116.5)
    ///   K=2048  acceptance      0.68, 110.3 tok/s
    ///   K=16384 acceptance      0.77, 115.2 tok/s — coverage recovered,
    ///           but per-round verify-side top-K + row gather costs more
    ///           than the ~1.3 ms full 286 MB 4-bit head read it replaces.
    /// Kept env-gated for deeper draft chains (k≥2 amortizes the top-K
    /// cost across several head evaluations per round).
    static let shortlistSize: Int? = {
        guard
            let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SHORTLIST"],
            let value = Int(raw), value > 0
        else { return nil }
        return value
    }()

    /// Screened draft argmax (`HadamardQuantizedLinear.screenedArgMax`):
    /// the leading `width` rotated inputs of every draft-vocabulary row, then
    /// the full width for the best row of each `chunk`-row block. Local
    /// experiments override it with `QWEN35_DRAFT_SCREEN=<width>,<chunk>` or
    /// `off`.
    static let draftScreen: (Int, Int)? = {
        guard let raw = ProcessInfo.processInfo.environment["QWEN35_DRAFT_SCREEN"] else {
            return (1536, 32)
        }
        let parts = raw.split(separator: ",").compactMap { Int($0) }
        return parts.count == 2 ? (parts[0], parts[1]) : nil
    }()

    /// Prompt-lookup drafting: the n-gram the engine matches against the
    /// request's own prompt and committed tokens to propose a round's chain
    /// without running this head. Default 6. Local experiments override it
    /// with `QWEN35_PLD_NGRAM=<n>`; `0` or `off` disables it.
    static let promptLookupNgram: Int? = {
        guard let raw = ProcessInfo.processInfo.environment["QWEN35_PLD_NGRAM"] else {
            return 6
        }
        guard let value = Int(raw), value > 0 else { return nil }
        return value
    }()

    public var lookupDraftNgram: Int? { Self.promptLookupNgram }

    /// The head's chain depth on rounds whose prompt lookup misses, when the
    /// declared depth is deeper: lookup rounds propose the declared depth.
    /// Default 3. Local experiments override it with
    /// `QWEN35_PLD_MISS_DEPTH=<k>`; `0` or `off` keeps the declared depth.
    static let promptLookupMissDepth: Int? = {
        guard let raw = ProcessInfo.processInfo.environment["QWEN35_PLD_MISS_DEPTH"] else {
            return 3
        }
        guard let value = Int(raw), value > 0 else { return nil }
        return value
    }()

    public var lookupMissDraftDepth: Int? { Self.promptLookupMissDepth }

    /// Shortlisting and the double-forward oracle are intentionally separate
    /// controls: the oracle scores the complete shared output head.
    public var draftShortlistSize: Int? {
        Self.forceDoubleForward ? nil : Self.shortlistSize
    }
    public var requestStateBytesPerToken: Int {
        Self.stateBytesPerToken(
            configuration: target.configuration,
            layerCount: mtp.layers.count,
            cacheElementBytes: mtp.norm.weight.dtype.size,
            hiddenElementBytes: target.model.norm.weight.dtype.size)
    }
    public var requestStateTokenGranularity: Int { Self.cacheAllocationStep }
    /// A round stages at most `maximumDepth` head inputs past the committed
    /// history before finalize trims them.
    public var requestStateTokenAllocationPadding: Int { Self.maximumDepth }

    public var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        let config = target.configuration
        let (elements, geometryOverflow) = config.kvHeads.multipliedReportingOverflow(by: config.headDim ?? 0)
        let (cacheRow, cacheOverflow) = elements.multipliedReportingOverflow(by: mtp.norm.weight.dtype.size)
        let (cacheCount, countOverflow) = mtp.layers.count.multipliedReportingOverflow(by: 2)
        let (hiddenRow, hiddenOverflow) = config.hiddenSize.multipliedReportingOverflow(
            by: target.model.norm.weight.dtype.size)
        guard !geometryOverflow, !cacheOverflow, !countOverflow, !hiddenOverflow else { return nil }
        return [
            .init(bytesPerToken: cacheRow, allocationCount: cacheCount,
                  tokenGranularity: Self.cacheAllocationStep, tokenPadding: Self.maximumDepth),
            // Backlog and frontier are views of normalized committed chunks;
            // the bound covers every possible partition of the retained rows.
            .init(bytesPerToken: hiddenRow, tokenPadding: Self.maximumDepth, partitioned: true),
            .init(bytesPerToken: MemoryLayout<Int32>.stride, tokenPadding: Self.maximumDepth, partitioned: true),
        ]
    }

    static func cacheBytesPerToken(
        configuration: Qwen35TextConfiguration,
        layerCount: Int,
        elementBytes: Int
    ) -> Int {
        let (headsByDimension, geometryOverflow) = configuration.kvHeads
            .multipliedReportingOverflow(by: configuration.headDim ?? 0)
        let (kvElements, kvOverflow) = headsByDimension.multipliedReportingOverflow(by: 2)
        let (layerElements, layerOverflow) = kvElements.multipliedReportingOverflow(
            by: layerCount)
        let (bytes, byteOverflow) = layerElements.multipliedReportingOverflow(
            by: elementBytes)
        return geometryOverflow || kvOverflow || layerOverflow || byteOverflow ? Int.max : bytes
    }

    static func stateBytesPerToken(
        configuration: Qwen35TextConfiguration,
        layerCount: Int,
        cacheElementBytes: Int,
        hiddenElementBytes: Int
    ) -> Int {
        let cacheBytes = cacheBytesPerToken(
            configuration: configuration, layerCount: layerCount,
            elementBytes: cacheElementBytes)
        let (hiddenBytes, hiddenOverflow) = configuration.hiddenSize
            .multipliedReportingOverflow(by: hiddenElementBytes)
        guard cacheBytes != Int.max, !hiddenOverflow else { return Int.max }
        let (withHidden, hiddenAdditionOverflow) = cacheBytes.addingReportingOverflow(
            hiddenBytes)
        let (withToken, tokenAdditionOverflow) = withHidden.addingReportingOverflow(
            MemoryLayout<Int32>.stride)
        return hiddenAdditionOverflow || tokenAdditionOverflow ? Int.max : withToken
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        RequestState(caches: makeCache())
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP observed released request state")
        precondition(!state.roundInFlight, "inline Qwen MTP observed target during a round")
        precondition(
            observation.tokens.ndim == 2 && observation.hidden.ndim == 3
                && observation.tokens.dim(0) == 1 && observation.hidden.dim(0) == 1
                && observation.tokens.dim(1) == observation.hidden.dim(1),
            "inline Qwen MTP target observation shape mismatch")

        let count = observation.tokens.dim(1)
        guard count > 0 else { return }
        let normalizedHidden = targetFinalNorm(observation.hidden)

        // Cross-chunk transition: the preceding chunk's final normalized
        // target hidden pairs with this chunk's first target input.
        if let frontier = state.targetHiddenFrontier {
            state.backlogHidden.append(frontier)
            state.backlogTokens.append(observation.tokens[0..., 0 ..< 1])
        }
        // Intra-chunk transitions: finalNorm(hidden[t]) conditions token[t+1].
        if count > 1 {
            state.backlogHidden.append(normalizedHidden[0..., 0 ..< count - 1, 0...])
            state.backlogTokens.append(observation.tokens[0..., 1 ..< count])
        }
        state.targetHiddenFrontier =
            normalizedHidden[0..., (count - 1) ..< count, 0...]
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        UnusedPreparedCapture()
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("inline Qwen MTP requires request-owned assistant state")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP drafted with released request state")
        precondition(
            tokens.ndim == 2 && hidden.ndim == 3
                && tokens.dim(0) == 1 && tokens.dim(1) == 1
                && hidden.dim(0) == 1 && hidden.dim(1) == 1,
            "inline Qwen MTP draft input shape mismatch")

        if Self.forceDoubleForward {
            return legacyDoubleForwardDraftStep(tokens: tokens, hidden: hidden, state: state)
        }

        let isFirstStep = !state.roundInFlight
        let feed: (tokens: MLXArray, hidden: MLXArray)
        if isFirstStep {
            feed = beginRound(tokens: tokens, hidden: hidden, state: state)
        } else {
            precondition(
                state.roundDraftSteps < Self.maximumDepth,
                "inline Qwen MTP exceeded its \(Self.maximumDepth)-step draft chain")
            feed = (tokens, hidden)
        }

        let output =
            isFirstStep
            ? (moduleLastHiddenWithKVOnlyHistory(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
                ?? moduleForward(
                    hidden: feed.hidden, tokens: feed.tokens, cache: state.caches))
            : moduleForward(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        let lastHidden = output[0..., (output.dim(1) - 1)..., 0...]
        let draft = draftToken(hidden: lastHidden, shortlist: shortlist)
        state.roundRoots.append(contentsOf: [lastHidden, draft])
        state.roundDraftSteps += 1

        if isFirstStep {
            // The engine immediately submits evaluationTargets for this first
            // cache generation before constructing a deeper draft step.
            state.roundValidHistoryOffset = state.cacheOffset
        }
        return (draft, lastHidden)
    }

    private func beginRound(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "inline Qwen MTP round already in flight")
        precondition(
            state.backlogHidden.count == state.backlogTokens.count,
            "inline Qwen MTP trusted backlog diverged")

        state.roundBaseOffset = state.cacheOffset
        state.roundValidHistoryOffset = state.cacheOffset
        state.roundDraftSteps = 0
        state.roundInFlight = true
        state.roundTrustedHidden = state.backlogHidden
        state.roundTrustedTokens = state.backlogTokens
        state.backlogHidden.removeAll(keepingCapacity: true)
        state.backlogTokens.removeAll(keepingCapacity: true)

        // The current target carry is trusted and completes the frontier
        // transition. Normalize it exactly once before it enters head history.
        state.roundTrustedHidden.append(targetFinalNorm(hidden))
        state.roundTrustedTokens.append(tokens)
        state.targetHiddenFrontier = nil

        if state.roundTrustedTokens.count == 1 {
            return (state.roundTrustedTokens[0], state.roundTrustedHidden[0])
        }
        let feedTokens = concatenated(state.roundTrustedTokens, axis: 1)
        let feedHidden = concatenated(state.roundTrustedHidden, axis: 1)
        return (feedTokens, feedHidden)
    }

    private func draftToken(hidden: MLXArray, shortlist: MLXArray?) -> MLXArray {
        if let shortlist {
            let logits = shortlistLogits(hidden: hidden, ids: shortlist)
            return shortlist[argMax(logits[0..., -1, 0...], axis: -1)]
                .asType(.int32)
        }
        if let draftVocabulary {
            if let (width, chunk) = Self.draftScreen,
                let local = draftVocabulary.head.screenedArgMax(hidden, width: width, chunk: chunk)
            {
                return draftVocabulary.ids[local]
            }
            let local = argMax(draftVocabulary.head(hidden)[0..., -1, 0...], axis: -1)
            return draftVocabulary.ids[local]
        }
        return argMax(headLogits(hidden)[0..., -1, 0...], axis: -1)
            .asType(.int32)
    }

    /// Explicit pre-cutover A/B oracle. It shares production history upkeep,
    /// but stages the proposed draft through a second complete MTP/lm-head
    /// forward and performs the historical blocking mid-round evaluation.
    private func legacyDoubleForwardDraftStep(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "inline Qwen MTP round already in flight")
        let feed = beginRound(tokens: tokens, hidden: hidden, state: state)
        let output =
            moduleLastHiddenWithKVOnlyHistory(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
            ?? moduleForward(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        let lastHidden = output[0..., (output.dim(1) - 1)..., 0...]
        let logits = headLogits(lastHidden)
        let draft = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
        state.roundRoots.append(contentsOf: [lastHidden, draft])
        state.roundDraftSteps = 1
        state.roundValidHistoryOffset = state.cacheOffset

        eval([draft, lastHidden] + state.caches.flatMap { $0.innerState() })
        _ = forward(
            hidden: lastHidden, tokens: draft.reshaped([1, 1]), cache: state.caches)
        return (draft, lastHidden)
    }

    public func evaluationTargets(
        for requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return []
        }
        return state.caches.flatMap { $0.innerState() }
            + state.backlogHidden + state.backlogTokens
            + [state.targetHiddenFrontier].compactMap { $0 }
            + state.roundTrustedHidden + state.roundTrustedTokens + state.roundRoots
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP finalized released request state")
        precondition(state.roundInFlight, "inline Qwen MTP finalized without a round")
        precondition(
            (0 ... state.roundDraftSteps + 1).contains(confirmedInputTokens),
            "inline Qwen MTP confirmed prefix exceeds the draft round")
        precondition(
            committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1
                && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "inline Qwen MTP committed target rows mismatch")
        let committedDraftCount = committedDraftTokens.dim(1)
        precondition(
            committedDraftCount <= state.roundDraftSteps
                && committedDraftCount <= max(0, confirmedInputTokens - 1),
            "inline Qwen MTP committed drafts exceed confirmed target inputs")

        trim(state: state, to: state.roundValidHistoryOffset)
        if committedDraftCount > 0 {
            // These are pre-final-norm target verify hiddens, never speculative
            // assistant hiddens. Normalize exactly once, retain lazily, and
            // flush them with the next carry.
            state.backlogTokens.append(committedDraftTokens)
            state.backlogHidden.append(targetFinalNorm(committedTargetHidden))
        }
        state.clearRound()
    }

    public func commitLookupRound(
        requestState: any CBv2MTPRequestState,
        seedToken: MLXArray,
        seedHidden: MLXArray,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP committed released request state")
        precondition(!state.roundInFlight, "inline Qwen MTP lookup round met a staged round")
        precondition(
            seedToken.ndim == 2 && seedHidden.ndim == 3
                && seedToken.dim(0) == 1 && seedToken.dim(1) == 1
                && seedHidden.dim(0) == 1 && seedHidden.dim(1) == 1
                && committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1
                && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "inline Qwen MTP lookup round rows mismatch")

        // The history `beginRound` + `finalizeRound` would leave, minus the
        // head forward: the carry transition completes the frontier (which
        // therefore never enters history itself), then the accepted drafts
        // with their target verify hiddens. Everything stays queued for the
        // next drafted round's trusted flush, each row exactly once.
        state.backlogHidden.append(targetFinalNorm(seedHidden))
        state.backlogTokens.append(seedToken)
        state.targetHiddenFrontier = nil
        if committedDraftTokens.dim(1) > 0 {
            state.backlogTokens.append(committedDraftTokens)
            state.backlogHidden.append(targetFinalNorm(committedTargetHidden))
        }
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState,
            !state.isReleased, state.roundInFlight
        else { return }

        trim(state: state, to: state.roundBaseOffset)
        // Restore all trusted transitions consumed by the abandoned graph in
        // original order. No speculative assistant hidden enters the backlog.
        state.backlogHidden =
            state.roundTrustedHidden + state.backlogHidden
        state.backlogTokens =
            state.roundTrustedTokens + state.backlogTokens
        state.clearRound()
    }

    private func trim(state: RequestState, to offset: Int) {
        let rollback = state.cacheOffset - offset
        precondition(rollback >= 0, "inline Qwen MTP cache checkpoint moved forward")
        guard rollback > 0 else { return }
        for cache in state.caches {
            precondition(cache.trim(rollback) == rollback)
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return
        }
        state.clearAll()
    }
}

/// The draft chain proposes tokens only; the pinned target verifies and
/// decides every emitted token. So the draft head may score a fixed subset of
/// the vocabulary: a draft outside the subset is a rejected proposal, never a
/// wrong output. Byte-level BPE ids follow merge order, so the leading rows
/// hold the most frequent tokens, and added/special tokens sit at the tail.
/// The subset is fixed at load from the head's geometry alone (input- and
/// prompt-independent): the leading `leadingRows` rows plus the trailing
/// `trailingRows` rows of the target's own packed `lm_head`, copied once,
/// byte-identical, into one contiguous draft-only head. The target's head and
/// quantization are untouched; each draft step reads only the subset rows.
final class Qwen35MTPDraftVocabulary: @unchecked Sendable {
    static let leadingRows = 98_304
    static let trailingRows = 512

    let head: HadamardQuantizedLinear
    /// Target token id of each draft-head row (int32).
    let ids: MLXArray

    init?(target: Qwen35TextModel) {
        guard !target.configuration.tieWordEmbeddings,
            let full = target.lmHead as? HadamardQuantizedLinear,
            full.bias == nil, let biases = full.biases
        else { return nil }
        let vocabulary = full.weight.dim(0)
        let leading = Self.leadingRows
        let tailStart = vocabulary - Self.trailingRows
        guard leading < tailStart, (leading + Self.trailingRows) % 8 == 0 else { return nil }
        func rows(_ array: MLXArray) -> MLXArray {
            concatenated([array[0 ..< leading], array[tailStart ..< vocabulary]], axis: 0)
        }
        let (weight, scales, offsets) = (rows(full.weight), rows(full.scales), rows(biases))
        var tokenIds = [Int32](0 ..< Int32(leading))
        tokenIds.append(contentsOf: Int32(tailStart) ..< Int32(vocabulary))
        let ids = MLXArray(tokenIds)
        eval(weight, scales, offsets, ids)
        guard
            let head = try? HadamardQuantizedLinear(
                weight: weight, scales: scales, biases: offsets,
                groupSize: full.groupSize, bits: full.bits, transform: full.transform)
        else { return nil }
        self.head = head
        self.ids = ids
    }
}

// MARK: - Load-time engine round warm

extension Qwen35InlineMTPAssistant {
    /// `MLXFAST_MTP_ROUND_WARM=0` skips `warmEngineRound`.
    static let engineRoundWarmEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_MTP_ROUND_WARM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The declared depth of this track's chain (mtp-head.manifest.json) and
    /// the scored seed width. At the declared depth the engine runs two round
    /// shapes: a prompt-lookup hit verifies 1 + 6 rows, a miss drafts a
    /// depth-3 chain with the head and verifies 1 + 3 rows
    /// (`promptLookupMissDepth`).
    static let warmDepth = 6
    static let warmPromptRows = 512
    /// The period of the second warm prompt: its newest n-gram always recurs,
    /// so the lookup hits and the rounds verify at the declared width.
    static let warmLookupPeriod = 64

    /// Runs two short requests through a real `EngineV2` on throwaway state:
    /// the seed prefill seam, the seed step and a few rounds of each verify
    /// width the declared depth runs (draft chain, draft head, rectangular
    /// verify, top-two, finalize, rollback; and lookup-proposed rounds).
    ///
    /// Nothing before the timed decode window runs the engine's MTP round:
    /// the warmup and timed prefills go through the teacher-forced stepper.
    /// So a process paid each first-use cost (Metal pipeline builds, lazily
    /// widened head constants, allocator growth) inside the window's first
    /// round, and the window is only 128 tokens. The prompt is the resident
    /// warm's fixed token pattern, greedy, so nothing depends on any
    /// request's input; the engine, its caches and the request state are
    /// released and the buffer cache is drained before anything is served.
    func warmEngineRound(serving: any LanguageModel) {
        guard Self.engineRoundWarmEnabled else { return }
        let layerKinds: [CBv2LayerKind]
        let caches: [any CBv2AttendingLayerCache]
        let make: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache = {
            index, kind in CBv2LayerCache(layerIndex: index, kind: kind)
        }
        if let model = serving as? Qwen35Model, let built = try? model.newCacheV2(makeLayerCache: make) {
            (layerKinds, caches) = (model.cbv2LayerKinds, built)
        } else if let model = serving as? Qwen35TextModel,
            let built = try? model.newCacheV2(makeLayerCache: make)
        {
            (layerKinds, caches) = (model.cbv2LayerKinds, built)
        } else {
            return
        }
        let depth = Self.warmDepth
        let rows = Self.warmPromptRows
        weak var released: EngineV2?
        do {
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(serving),
                layerKinds: layerKinds,
                backend: CBv2ContiguousKVBackend(
                    config: CBv2ContiguousBackendConfig(bytesCapacity: 1 << 30)),
                cacheProvider: CBv2LayerCacheBank(caches: caches),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 1,
                    prefillChunkSize: max(CBv2SchedulerConfig().prefillChunkSize, rows),
                    maxWaiting: 1,
                    enablePrefixCache: false),
                mtpDrafter: self,
                mtpConfig: CBv2MTPConfig(
                    enabled: true,
                    maxDraftTokens: depth,
                    maxSpeculativeBatch: 1,
                    fixedDraftTokens: depth,
                    verificationMode: .automatic,
                    maxAutomaticRectangularTokens: 1 + depth,
                    draftTokenCeiling: CBv2MTPConfig.testedMaxDraftTokens))
            released = engine
            // Two greedy requests, one after the other: a non-repeating
            // pattern, whose rounds miss the lookup and run the head's chain
            // (4-row verify), then a periodic one, whose rounds hit it
            // (7-row verify). Each request has its own lookup index.
            let period = Self.warmLookupPeriod
            let prompts: [[Int]] = [
                (0 ..< rows).map { 100 + ($0 &* 7919) % 20_000 },
                (0 ..< rows).map { 100 + (($0 % period) &* 7919) % 20_000 },
            ]
            let warmRequests: [CBv2Request] = prompts.enumerated().map { index, prompt in
                var request = CBv2Request(
                    id: CBv2RequestID(UInt64(index + 1)),
                    promptTokens: prompt,
                    maxTokens: 2 + 4 * (1 + depth))
                request.sampling = CBv2SamplingParams(temperature: 0, topP: 1, topK: 0)
                request.stopTokens = []
                return request
            }
            let done = DispatchSemaphore(value: 0)
            Task.detached {
                for warmRequest in warmRequests {
                    if let events = try? engine.submit(warmRequest) {
                        for await _ in events {}
                    }
                }
                await engine.shutdown()
                done.signal()
            }
            done.wait()
        }
        var waited = 0
        while released != nil, waited < 500 {
            usleep(1_000)
            waited += 1
        }
        Stream().synchronize()
        Memory.clearCache()
    }
}
