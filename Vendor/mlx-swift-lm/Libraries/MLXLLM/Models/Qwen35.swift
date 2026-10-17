//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Cmlx
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4
    var mropeSection: [Int] = [11, 11, 10]

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    // MTP — number of Multi-Token Prediction head layers.
    // Port of omlx commit 696d90a: patches/mlx_lm_mtp/qwen35_model.py
    // `_patch_text_model_args` attaches this from config.json at runtime.
    var mtpNumHiddenLayers: Int = 0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
            self.mropeSection = ropeParameters["mrope_section"]?.asInts() ?? [11, 11, 10]
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
            self.mropeSection = self.ropeScaling?["mrope_section"]?.asInts() ?? [11, 11, 10]
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }

    /// Compact CBv2 attention-storage layout. Recurrent layers are absent by
    /// design; `modelLayerIndex` maps each of the 10 full-attention KV rows
    /// back to its original transformer layer.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        precondition(fullAttentionInterval > 0, "full_attention_interval must be positive")
        let dimension = headDim ?? (hiddenSize / attentionHeads)
        return (0 ..< hiddenLayers).compactMap { modelLayerIndex in
            guard (modelLayerIndex + 1) % fullAttentionInterval == 0 else { return nil }
            return CBv2LayerKind(
                attention: .full,
                headDim: dimension,
                kvHeads: kvHeads,
                queryHeads: attentionHeads,
                modelLayerIndex: modelLayerIndex)
        }
    }

    public func cbv2RecurrentStateSpec(
        activationDType: DType = .bfloat16
    ) -> CBv2RecurrentStateSpec {
        precondition(fullAttentionInterval > 0, "full_attention_interval must be positive")
        let keyDim = linearNumKeyHeads * linearKeyHeadDim
        let valueDim = linearNumValueHeads * linearValueHeadDim
        let convDim = 2 * keyDim + valueDim
        let layers: [CBv2RecurrentLayerStateSpec] = (0 ..< hiddenLayers).compactMap {
            modelLayerIndex -> CBv2RecurrentLayerStateSpec? in
            guard (modelLayerIndex + 1) % fullAttentionInterval != 0 else { return nil }
            return CBv2RecurrentLayerStateSpec(
                modelLayerIndex: modelLayerIndex,
                convShape: [1, max(0, linearConvKernelDim - 1), convDim],
                convDType: activationDType,
                ssmShape: [1, linearNumValueHeads, linearValueHeadDim, linearKeyHeadDim],
                ssmDType: .float32)
        }
        return CBv2RecurrentStateSpec(layers: layers)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities = CBv2ModelCapabilities.initialRecurrentTarget
        capabilities.supportsMTP = true
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        capabilities.supportsCompactRecurrentMTPReplay = true
        // Rectangular [B, L] prompt cohorts: one recurrent state row per
        // batch row; each packed row attends its own KV (see
        // `cbv2SupportsPackedPrefill` on the prefill conformance).
        capabilities.supportsPackedPrefill = true
        // Dense and MoE layers share the same attention/recurrent state.
        // Routed and shared experts are token-local and add no checkpoint state.
        capabilities.supportsRecurrentCheckpointReuse = true
        return capabilities
    }
}

// MARK: - GatedDeltaNet

private enum Qwen35DenseNativeGateProjection {
    final class Cache {
        let identities: [UInt]
        let sources: [MLXArray]
        let weight: MLXArray

        init(_ sources: [MLXArray], identities: [UInt]) {
            self.identities = identities
            self.sources = sources.map { source in
                var snapshot = mlx_array_new()
                mlx_array_set(&snapshot, source.ctx)
                return MLXArray(snapshot)
            }
            weight = concatenated(sources, axis: 0)
        }
    }

    private static let useNAX: Bool = {
        guard GPU.gemma4ExpertQMMDiagnostics().naxAvailable else { return false }
        guard let value = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] else {
            return true
        }
        guard let enabled = Int32(value) else { return false }
        return enabled != 0
    }()

    static func apply(_ x: MLXArray, b: Linear, a: Linear, cache: inout Cache?)
        -> (MLXArray, MLXArray)?
    {
        guard useNAX, ObjectIdentifier(type(of: b)) == ObjectIdentifier(Linear.self),
            ObjectIdentifier(type(of: a)) == ObjectIdentifier(Linear.self),
            b.bias == nil, a.bias == nil, x.ndim == 3, x.dtype == .float32,
            b.weight.dtype == .float32, a.weight.dtype == .float32,
            b.weight.shape == [32, 4096], a.weight.shape == b.weight.shape,
            x.dim(2) == 4096, (2 ... 16).contains(x.dim(0) * x.dim(1)),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let sources = [b.weight, a.weight]
        var identities = [UInt]()
        for source in sources {
            var identity: UInt = 0
            var canCache = false
            guard _mlx_array_constant_cache_identity(&identity, &canCache, source.ctx) == 0, canCache
            else { return nil }
            identities.append(identity)
        }
        if cache?.identities != identities { cache = Cache(sources, identities: identities) }
        let output = matmul(x, cache!.weight.transposed(1, 0))
        return (output[0..., 0..., 0..<32], output[0..., 0..., 32..<64])
    }
}

/// The dense FP32 `in_proj_b` and `in_proj_a` of a few-row window (MTP
/// verify) in one launch. Each simdgroup owns one output column and reads
/// that weight row once for all rows; FP32 FMAs, then `simd_sum`. Single-row
/// decode and wide prefill keep the stock matmul.
private enum Qwen35DenseGateProjection {
    static let maximumRows = 16
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_dense_ba_rows",
        inputNames: ["x", "wb", "wa"], outputNames: ["b", "a"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint n = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
            const int K = x_shape[1];
            const uint N = wb_shape[0];
            const bool is_a = n >= N;
            const uint col = is_a ? n - N : n;
            const device float4* wr = (const device float4*)((is_a ? wa : wb) + col * K);
            float acc[ROWS];
            for (int v = 0; v < ROWS; v++) acc[v] = 0.0f;
            for (int k4 = lane; k4 < K / 4; k4 += 32) {
              const float4 wv = wr[k4];
              for (int v = 0; v < ROWS; v++) {
                const float4 xv = ((const device float4*)(x + v * K))[k4];
                acc[v] = fma(xv.x, wv.x, acc[v]);
                acc[v] = fma(xv.y, wv.y, acc[v]);
                acc[v] = fma(xv.z, wv.z, acc[v]);
                acc[v] = fma(xv.w, wv.w, acc[v]);
              }
            }
            for (int v = 0; v < ROWS; v++) {
              const float total = simd_sum(acc[v]);
              if (lane == 0) (is_a ? a : b)[v * N + col] = total;
            }
            """)

    private static let tiledKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_dense_ba_rows16_tile4",
        inputNames: ["x", "wb", "wa"], outputNames: ["b", "a"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint task = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
            const int K = x_shape[1];
            const uint N = wb_shape[0];
            const uint n = task % (2 * N);
            const uint row0 = task / (2 * N) * TILE;
            const bool is_a = n >= N;
            const uint col = is_a ? n - N : n;
            const device float4* wr = (const device float4*)((is_a ? wa : wb) + col * K);
            float acc[TILE];
            for (int v = 0; v < TILE; v++) acc[v] = 0.0f;
            for (int k4 = lane; k4 < K / 4; k4 += 32) {
              const float4 wv = wr[k4];
              for (int v = 0; v < TILE; v++) {
                if (row0 + v >= ROWS) continue;
                const float4 xv = ((const device float4*)(x + (row0 + v) * K))[k4];
                acc[v] = fma(xv.x, wv.x, acc[v]);
                acc[v] = fma(xv.y, wv.y, acc[v]);
                acc[v] = fma(xv.z, wv.z, acc[v]);
                acc[v] = fma(xv.w, wv.w, acc[v]);
              }
            }
            for (int v = 0; v < TILE; v++) {
              if (row0 + v >= ROWS) continue;
              const float total = simd_sum(acc[v]);
              if (lane == 0) (is_a ? a : b)[(row0 + v) * N + col] = total;
            }
            """)

    static func apply(_ x: MLXArray, b: Linear, a: Linear) -> (MLXArray, MLXArray)? {
        guard !(b is QuantizedLinear), !(a is QuantizedLinear), b.bias == nil, a.bias == nil,
            x.dtype == .float32, b.weight.dtype == .float32, a.weight.dtype == .float32,
            b.weight.ndim == 2, b.weight.shape == a.weight.shape, x.ndim == 3,
            b.weight.dim(1) == x.dim(2), x.dim(2) % 4 == 0,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.dim(0) * x.dim(1)
        guard (2 ... maximumRows).contains(rows) else { return nil }
        let N = b.weight.dim(0)
        let outputs: [MLXArray]
        if rows == 16 {
            outputs = tiledKernel(
                [x.reshaped([rows, x.dim(2)]), b.weight, a.weight],
                template: [("ROWS", rows), ("TILE", 4)],
                grid: (32, 8 * N, 1), threadGroup: (32, 2, 1),
                outputShapes: [[rows, N], [rows, N]], outputDTypes: [.float32, .float32])
        } else {
            outputs = kernel(
                [x.reshaped([rows, x.dim(2)]), b.weight, a.weight], template: [("ROWS", rows)],
                grid: (32, 2 * N, 1), threadGroup: (32, 2, 1),
                outputShapes: [[rows, N], [rows, N]], outputDTypes: [.float32, .float32])
        }
        let shape = [x.dim(0), x.dim(1), N]
        return (outputs[0].reshaped(shape), outputs[1].reshaped(shape))
    }
}

private enum Qwen35FusedGDNPrework {
    static let kernel = MLXFast.metalKernel(name: "qwen35_gdn_prework",
    inputNames: ["conv_input", "conv_weight", "a", "b", "a_log", "dt_bias"],
    outputNames: ["q_out", "k_out", "v_out", "g_out", "beta_out"],
    source: """
        #pragma clang fp reassociate(off)
        const uint lane = thread_index_in_simdgroup;
        const uint head = threadgroup_position_in_grid.y * 4u + simdgroup_index_in_threadgroup;
        const uint row = threadgroup_position_in_grid.z;
        const uint T = conv_input_shape[1] - 3u;
        const uint batch = row / T;
        const uint position = row % T;
        const uint channel = head * 128u + lane * 4u;
        const uint input_at = (batch * (T + 3u) + position) * CD + channel;
        const float4 w0 = *((const device float4*)(conv_weight + (channel + 0u) * 4u));
        const float4 w1 = *((const device float4*)(conv_weight + (channel + 1u) * 4u));
        const float4 w2 = *((const device float4*)(conv_weight + (channel + 2u) * 4u));
        const float4 w3 = *((const device float4*)(conv_weight + (channel + 3u) * 4u));
        float4 value = fma(*((const device float4*)(conv_input + input_at)),
            float4(w0.x, w1.x, w2.x, w3.x), float4(0.0f));
        value = fma(*((const device float4*)(conv_input + input_at + CD)),
            float4(w0.y, w1.y, w2.y, w3.y), value);
        value = fma(*((const device float4*)(conv_input + input_at + 2u * CD)),
            float4(w0.z, w1.z, w2.z, w3.z), value);
        value = fma(*((const device float4*)(conv_input + input_at + 3u * CD)),
            float4(w0.w, w1.w, w2.w, w3.w), value);
        const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(value)));
        value = value * select(1.0f - e, e, value < 0.0f);
        if (head < 2u * HK) {
          float square_sum = 0.0f;
          #pragma clang loop unroll(full)
          for (int i = 0; i < 4; i++) square_sum += value[i] * value[i];
          const float inv = metal::precise::rsqrt(simd_sum(square_sum) / 128.0f + as_type<float>(uint(EPS_BITS)));
          const float scale = as_type<float>(uint(head < HK ? Q_SCALE_BITS : K_SCALE_BITS));
          const uint at = (row * HK + head % HK) * 128u + lane * 4u;
          const float4 out = float4(scale * (value.x * inv), scale * (value.y * inv),
              scale * (value.z * inv), scale * (value.w * inv));
          if (head < HK) *((device float4*)(q_out + at)) = out;
          else *((device float4*)(k_out + at)) = out;
        } else {
          const uint at = (row * HV + head - 2u * HK) * 128u + lane * 4u;
          *((device float4*)(v_out + at)) = value;
        }
        if (PREPARE_GATES && head < HV && lane == 0) {
          const uint at = row * HV + head;
          const float negative_a = -metal::precise::exp(a_log[head]);
          g_out[at] = qwen35_simd_gate_decay(a[at], dt_bias[head], negative_a);
          beta_out[at] = qwen35_simd_gate_beta(b[at]);
        }
        """,
    header: """
        inline float qwen35_simd_gate_decay(float a, float bias, float negative_a) {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            const float x = a + bias;
            float sp;
            if (metal::isnan(x)) {
                sp = metal::numeric_limits<float>::quiet_NaN();
            } else {
                const float maximum = metal::max(x, 0.0f);
                const float minimum = metal::min(x, 0.0f);
                constexpr float infinity = metal::numeric_limits<float>::infinity();
                if (minimum == -infinity || maximum == infinity) {
                    sp = maximum;
                } else {
                    const float z = metal::exp(minimum - maximum);
                    const float xp1 = 1.0f + z;
                    const float logarithm = xp1 == metal::numeric_limits<float>::max()
                        ? metal::numeric_limits<float>::max()
                        : (xp1 == 1.0f ? z : z * (metal::log(xp1) / (xp1 - 1.0f)));
                    sp = maximum + logarithm;
                }
            }
            return metal::precise::exp(negative_a * sp);
        }

        inline float qwen35_simd_gate_beta(float b) {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            const float y = 1.0f / (1.0f + metal::exp(metal::abs(b)));
            return b < 0.0f ? y : 1.0f - y;
        }
        """, ensureRowContiguous: true)
    static let invScale = pow(Float(128), -0.5)

    static func apply(
        _ input: MLXArray, weight: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray, keyHeads: Int, valueHeads: Int,
        keyDimensions: Int, valueDimensions: Int, foldValue: Bool
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray?, g: MLXArray?, beta: MLXArray?)? {
        guard input.ndim == 3, input.dim(0) == 1, input.dim(1) >= 4,
            keyDimensions == 128, valueDimensions == 128,
            valueHeads == 2 * keyHeads, keyHeads % 2 == 0,
            input.dim(2) == (2 * keyHeads + valueHeads) * 128,
            weight.shape == [input.dim(2), 4, 1],
            [input, weight, a, b, aLog, dtBias].allSatisfy({ $0.dtype == .float32 }),
            a.shape == [1, input.dim(1) - 3, valueHeads], b.shape == a.shape,
            aLog.shape == [valueHeads], dtBias.shape == [valueHeads],
            input.size <= Int(UInt32.max), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = input.dim(1) - 3
        guard rows <= 16 || foldValue, !foldValue || rows >= 64 else { return nil }
        let out = kernel(
            [input, weight, a, b, aLog, dtBias],
            template: [("CD", input.dim(2)), ("HK", keyHeads), ("HV", valueHeads),
                ("PREPARE_GATES", foldValue), ("EPS_BITS", Int(Float(1e-6).bitPattern)),
                ("Q_SCALE_BITS", Int(pow(invScale, 2).bitPattern)),
                ("K_SCALE_BITS", Int(invScale.bitPattern))],
            grid: (32, foldValue ? 2 * keyHeads : 2 * keyHeads + valueHeads, rows),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, rows, keyHeads, 128], [1, rows, keyHeads, 128],
                foldValue ? [1] : [1, rows, valueHeads, 128],
                foldValue ? [1, rows, valueHeads] : [1],
                foldValue ? [1, rows, valueHeads] : [1]],
            outputDTypes: [.float32, .float32, .float32, .float32, .float32])
        return (out[0], out[1], foldValue ? nil : out[2],
            foldValue ? out[3] : nil, foldValue ? out[4] : nil)
    }
}

private enum Qwen35FusedGDNCompleteShortPrework {
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_complete_short_prework",
        inputNames: ["state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias"],
        outputNames: ["conv_input", "q_out", "k_out", "v_out", "g_out", "beta_out"],
        source: """
            #pragma clang fp reassociate(off)
            const uint lane = thread_index_in_simdgroup;
            const uint head = threadgroup_position_in_grid.y * 4u + simdgroup_index_in_threadgroup;
            const uint row = threadgroup_position_in_grid.z;
            const uint channel = head * 128u + lane * 4u;
            auto short_span32 = [](size_t rows, size_t cols, size_t rs, size_t cs) -> bool {
              constexpr size_t limit = 0xffffffffu;
              const size_t r = rows - 1u, c = cols - 1u;
              if (r != 0u && rs != 0u && (r > limit || rs > limit)) return false;
              if (c != 0u && cs != 0u && (c > limit || cs > limit)) return false;
              const size_t row_max = r * rs, col_max = c * cs;
              return row_max <= limit && col_max <= limit - row_max;
            };
            const bool io32 = short_span32(3u, 8192u, state_strides[1], state_strides[2])
                && short_span32(qkv_shape[1], 8192u, qkv_strides[1], qkv_strides[2])
                && short_span32(8192u, 4u, conv_weight_strides[0], conv_weight_strides[1])
                && short_span32(qkv_shape[1], 32u, a_strides[1], a_strides[2])
                && short_span32(qkv_shape[1], 32u, b_strides[1], b_strides[2])
                && short_span32(1u, 32u, 0u, a_log_strides[0])
                && short_span32(1u, 32u, 0u, dt_bias_strides[0]);
            auto read_input = [&](uint r) -> float4 {
              if (io32) {
                const device float* source = r < 3u ? state : qkv;
                const uint rs = uint(r < 3u ? state_strides[1] : qkv_strides[1]);
                const uint cs = uint(r < 3u ? state_strides[2] : qkv_strides[2]);
                const uint at = (r < 3u ? r : r - 3u) * rs + channel * cs;
                return float4(source[at], source[at + cs], source[at + 2u * cs], source[at + 3u * cs]);
              }
              const device float* source = r < 3u
                  ? state + size_t(r) * state_strides[1]
                  : qkv + size_t(r - 3u) * qkv_strides[1];
              const size_t stride = r < 3u ? state_strides[2] : qkv_strides[2];
              source += size_t(channel) * stride;
              return float4(source[0], source[stride], source[2u * stride], source[3u * stride]);
            };
            auto read_weight = [&](uint c) -> float4 {
              if (io32) {
                const uint at = c * uint(conv_weight_strides[0]);
                const uint cs = uint(conv_weight_strides[1]);
                return float4(conv_weight[at], conv_weight[at + cs],
                    conv_weight[at + 2u * cs], conv_weight[at + 3u * cs]);
              }
              const device float* source = conv_weight + size_t(c) * conv_weight_strides[0];
              const size_t stride = conv_weight_strides[1];
              return float4(source[0], source[stride], source[2u * stride], source[3u * stride]);
            };
            auto read_scalar = [&](const device float* source, uint r, uint c, size_t rs, size_t cs) -> float {
              if (io32) return source[r * uint(rs) + c * uint(cs)];
              return source[size_t(r) * rs + size_t(c) * cs];
            };
            const float4 x0 = read_input(row);
            const float4 x1 = read_input(row + 1u);
            const float4 x2 = read_input(row + 2u);
            const float4 x3 = read_input(row + 3u);
            *((device float4*)(conv_input + (row + 3u) * 8192u + channel)) = x3;
            if (row == 0u) {
              *((device float4*)(conv_input + channel)) = x0;
              *((device float4*)(conv_input + 8192u + channel)) = x1;
              *((device float4*)(conv_input + 2u * 8192u + channel)) = x2;
            }
            const float4 w0 = read_weight(channel);
            const float4 w1 = read_weight(channel + 1u);
            const float4 w2 = read_weight(channel + 2u);
            const float4 w3 = read_weight(channel + 3u);
            float4 value = fma(x0, float4(w0.x, w1.x, w2.x, w3.x), float4(0.0f));
            value = fma(x1, float4(w0.y, w1.y, w2.y, w3.y), value);
            value = fma(x2, float4(w0.z, w1.z, w2.z, w3.z), value);
            value = fma(x3, float4(w0.w, w1.w, w2.w, w3.w), value);
            const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(value)));
            value = value * select(1.0f - e, e, value < 0.0f);
            if (head < 32u) {
              float square_sum = 0.0f;
              #pragma clang loop unroll(full)
              for (int i = 0; i < 4; i++) square_sum += value[i] * value[i];
              const float inv = metal::precise::rsqrt(simd_sum(square_sum) / 128.0f + as_type<float>(uint(EPS_BITS)));
              const float scale = as_type<float>(uint(head < 16u ? Q_SCALE_BITS : K_SCALE_BITS));
              const uint at = (row * 16u + head % 16u) * 128u + lane * 4u;
              const float4 result = float4(scale * (value.x * inv), scale * (value.y * inv),
                  scale * (value.z * inv), scale * (value.w * inv));
              if (head < 16u) *((device float4*)(q_out + at)) = result;
              else *((device float4*)(k_out + at)) = result;
            } else {
              const uint at = (row * 32u + head - 32u) * 128u + lane * 4u;
              *((device float4*)(v_out + at)) = value;
            }
            if (head < 32u && lane == 0u) {
              const float av = read_scalar(a, row, head, a_strides[1], a_strides[2]);
              const float bv = read_scalar(b, row, head, b_strides[1], b_strides[2]);
              const float logv = read_scalar(a_log, 0u, head, 0u, a_log_strides[0]);
              const float bias = read_scalar(dt_bias, 0u, head, 0u, dt_bias_strides[0]);
              g_out[row * 32u + head] = qwen35_simd_gate_decay(av, bias, -metal::precise::exp(logv));
              beta_out[row * 32u + head] = qwen35_simd_gate_beta(bv);
            }
            """,
        header: """
        inline float qwen35_simd_gate_decay(float a, float bias, float negative_a) {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            const float x = a + bias;
            float sp;
            if (metal::isnan(x)) {
                sp = metal::numeric_limits<float>::quiet_NaN();
            } else {
                const float maximum = metal::max(x, 0.0f);
                const float minimum = metal::min(x, 0.0f);
                constexpr float infinity = metal::numeric_limits<float>::infinity();
                if (minimum == -infinity || maximum == infinity) {
                    sp = maximum;
                } else {
                    const float z = metal::exp(minimum - maximum);
                    const float xp1 = 1.0f + z;
                    const float logarithm = xp1 == metal::numeric_limits<float>::max()
                        ? metal::numeric_limits<float>::max()
                        : (xp1 == 1.0f ? z : z * (metal::log(xp1) / (xp1 - 1.0f)));
                    sp = maximum + logarithm;
                }
            }
            return metal::precise::exp(negative_a * sp);
        }

        inline float qwen35_simd_gate_beta(float b) {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            const float y = 1.0f / (1.0f + metal::exp(metal::abs(b)));
            return b < 0.0f ? y : 1.0f - y;
        }
        """, ensureRowContiguous: false)

    static func apply(
        state: MLXArray, qkv: MLXArray, weight: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray,
        keyHeads: Int, valueHeads: Int, keyDimensions: Int, valueDimensions: Int
    ) -> (convInput: MLXArray, q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray)? {
        guard state.shape == [1, 3, 8192], qkv.ndim == 3,
            qkv.dim(0) == 1, (1 ... 16).contains(qkv.dim(1)), qkv.dim(2) == 8192,
            weight.shape == [8192, 4, 1], keyHeads == 16, valueHeads == 32,
            keyDimensions == 128, valueDimensions == 128,
            a.shape == [1, qkv.dim(1), 32], b.shape == a.shape,
            aLog.shape == [32], dtBias.shape == [32],
            [state, qkv, weight, a, b, aLog, dtBias].allSatisfy({ $0.dtype == .float32 }),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = qkv.dim(1)
        let invScale = pow(Float(128), -0.5)
        let result = kernel(
            [state, qkv, weight, a, b, aLog, dtBias],
            template: [("EPS_BITS", Int(Float(1e-6).bitPattern)),
                ("Q_SCALE_BITS", Int(pow(invScale, 2).bitPattern)),
                ("K_SCALE_BITS", Int(invScale.bitPattern))],
            grid: (32, 64, rows), threadGroup: (32, 4, 1),
            outputShapes: [[1, rows + 3, 8192], [1, rows, 16, 128],
                [1, rows, 16, 128], [1, rows, 32, 128], [1, rows, 32], [1, rows, 32]],
            outputDTypes: [.float32, .float32, .float32, .float32, .float32, .float32])
        return (result[0], result[1], result[2], result[3], result[4], result[5])
    }
}

// Register layout from the accepted 27B engine PR #649 (MIT lineage).
private enum Qwen35FusedQKNorm {
    private static let invScale = pow(Float(128), -0.5)
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_qk_norm128",
        inputNames: ["x"], outputNames: ["q_out", "k_out"],
        source: """
            #pragma clang fp reassociate(off)
            const uint row = threadgroup_position_in_grid.y;
            const uint lane = thread_index_in_simdgroup;
            const uint offset = (row / HEADS) * CONV_DIM + (row % HEADS) * 128 + lane * 4;
            const float4 q4 = ((const device float4*)(x + offset))[0];
            const float4 k4 = ((const device float4*)(x + offset + HEADS * 128))[0];
            const float q[4] = {q4.x, q4.y, q4.z, q4.w};
            const float k[4] = {k4.x, k4.y, k4.z, k4.w};
            float q_sum = 0.0f, k_sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
              q_sum += q[i] * q[i];
              k_sum += k[i] * k[i];
            }
            const float eps = as_type<float>(uint(EPS_BITS));
            const float q_inv = metal::precise::rsqrt(simd_sum(q_sum) / 128.0f + eps);
            const float k_inv = metal::precise::rsqrt(simd_sum(k_sum) / 128.0f + eps);
            const uint output_offset = row * 128 + lane * 4;
            const float q_scale = as_type<float>(uint(Q_SCALE_BITS));
            const float k_scale = as_type<float>(uint(K_SCALE_BITS));
            ((device float4*)(q_out + output_offset))[0] = float4(
              q_scale * (q[0] * q_inv),
              q_scale * (q[1] * q_inv),
              q_scale * (q[2] * q_inv),
              q_scale * (q[3] * q_inv)
            );
            ((device float4*)(k_out + output_offset))[0] = float4(
              k_scale * (k[0] * k_inv),
              k_scale * (k[1] * k_inv),
              k_scale * (k[2] * k_inv),
              k_scale * (k[3] * k_inv)
            );
            """)

    static func apply(_ x: MLXArray, heads: Int, dimensions: Int) -> (MLXArray, MLXArray)? {
        guard x.dtype == .float32, x.ndim == 3, dimensions == 128,
            x.dim(2) >= 2 * heads * dimensions, x.size > 0, x.size <= Int(UInt32.max)
        else { return nil }
        let shape = [x.dim(0), x.dim(1), heads, dimensions]
        let outputs = kernel(
            [x], template: [("HEADS", heads), ("CONV_DIM", x.dim(2)),
                ("EPS_BITS", Int(Float(1e-6).bitPattern)),
                ("Q_SCALE_BITS", Int(pow(invScale, 2).bitPattern)),
                ("K_SCALE_BITS", Int(invScale.bitPattern))],
            grid: (32, x.dim(0) * x.dim(1) * heads, 1), threadGroup: (32, 1, 1),
            outputShapes: [shape, shape], outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }
}

/// Verify-width GDN convolution in one kernel: the concatenation of the
/// retained three-row convolution state with the new QKV rows (written out,
/// since the replay tape keeps it), MLX's depthwise `conv1d` over it
/// (`depthwise_conv_1d`: taps in order, one FP32 multiply-add each from
/// zero) and MLX's `silu` (`x * sigmoid(x)`, stable sigmoid). Outputs are the
/// bits of `concatenated`, `conv1d` and `silu`.
private enum Qwen35FusedConvSilu {
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_verify_conv_silu",
        inputNames: ["state", "qkv", "conv_weight"], outputNames: ["conv_in", "conv_out"],
        source: """
            #pragma clang fp reassociate(off)
            const uint c = thread_position_in_grid.x;
            const uint t = thread_position_in_grid.y;
            const uint C = uint(CONV_DIM);
            auto conv_span32 = [](size_t rows, size_t cols, size_t rs, size_t cs) -> bool {
              constexpr size_t limit = 0xffffffffu;
              const size_t r = rows - 1u, c = cols - 1u;
              if (r != 0u && rs != 0u && (r > limit || rs > limit)) return false;
              if (c != 0u && cs != 0u && (c > limit || cs > limit)) return false;
              const size_t row_max = r * rs, col_max = c * cs;
              return row_max <= limit && col_max <= limit - row_max;
            };
            const bool io32 = conv_span32(3u, C, state_strides[1], state_strides[2])
                && conv_span32(qkv_shape[1], C, qkv_strides[1], qkv_strides[2])
                && conv_span32(C, 4u, conv_weight_strides[0], conv_weight_strides[1]);
            auto read_input = [&](uint r) -> float {
              if (io32) {
                const device float* source = r < 3u ? state : qkv;
                const uint rs = uint(r < 3u ? state_strides[1] : qkv_strides[1]);
                const uint cs = uint(r < 3u ? state_strides[2] : qkv_strides[2]);
                return source[(r < 3u ? r : r - 3u) * rs + c * cs];
              }
              return r < 3u ? state[size_t(r) * state_strides[1] + size_t(c) * state_strides[2]]
                  : qkv[size_t(r - 3u) * qkv_strides[1] + size_t(c) * qkv_strides[2]];
            };
            auto read_weight = [&](uint tap) -> float {
              if (io32) return conv_weight[c * uint(conv_weight_strides[0]) + tap * uint(conv_weight_strides[1])];
              return conv_weight[size_t(c) * conv_weight_strides[0] + size_t(tap) * conv_weight_strides[1]];
            };
            conv_in[t * C + c] = read_input(t);
            if (t >= 3u) {
              const uint p = t - 3u;
              float acc = 0.0f;
              for (uint tap = 0; tap < 4u; ++tap) {
                const uint r = p + tap;
                const float v = read_input(r);
                acc = fma(v, read_weight(tap), acc);
              }
              const float e = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
              const float sig = acc < 0.0f ? e : 1.0f - e;
              conv_out[p * C + c] = acc * sig;
            }
            """,
        ensureRowContiguous: false)

    static func apply(state: MLXArray, qkv: MLXArray, weight: MLXArray) -> (MLXArray, MLXArray)? {
        guard state.ndim == 3, state.dim(0) == 1, state.dim(1) == 3, qkv.ndim == 3,
            qkv.dim(0) == 1, (1 ... 16).contains(qkv.dim(1)), qkv.dim(2) == state.dim(2),
            state.dtype == .float32, qkv.dtype == .float32, weight.dtype == .float32,
            weight.shape == [state.dim(2), 4, 1], StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let C = state.dim(2)
        let S = qkv.dim(1)
        let outputs = kernel(
            [state, qkv, weight], template: [("CONV_DIM", C)],
            grid: (C, S + 3, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, S + 3, C], [1, S, C]], outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }
}

/// Fresh prompt scan from the 27B 8-token matrix path (final tree 0c7c392).
/// The matrix products change FP32 accumulation order relative to the register
/// recurrence. Only new B1 requests use it; retained and verify states do not.
private enum Qwen35FreshChunkGDN {
    static func runConvolvedValue(
        q: MLXArray, k: MLXArray, input: MLXArray, weight: MLXArray,
        g: MLXArray, beta: MLXArray
    ) -> (MLXArray, MLXArray)? {
        guard q.ndim == 4, q.shape == k.shape, q.dim(0) == 1,
            q.dim(1) >= 64, q.dim(1) % 8 == 0,
            q.dim(2) == 16, q.dim(3) == 128,
            g.shape == [1, q.dim(1), 32], beta.shape == g.shape,
            input.shape == [1, q.dim(1) + 3, 8192], weight.shape == [8192, 4, 1],
            q.dtype == .float32, k.dtype == .float32,
            g.dtype == .float32, beta.dtype == .float32,
            input.dtype == .float32, weight.dtype == .float32,
            (q.dim(1) + 3) * 8192 < Int(Int32.max)
        else { return nil }
        let T = q.dim(1)
        let q = q.contiguous()
        let k = k.contiguous()
        let rows = MLXArray(Int32(T))
        let prepared = prepKernel(
            [q, k, g, beta, rows],
            template: [("C", 8), ("Dk", 128), ("Hk", 16), ("Hv", 32)],
            grid: (32, T / 8, 16), threadGroup: (32, 1, 1),
            outputShapes: [[1, 32, T / 8, 8, 8], [1, 32, T / 8, 8, 8], [1, 32, T / 8, 2, 8]],
            outputDTypes: [.float32, .float32, .float32])
        let result = scanKernel(
            [q, k, input, weight, prepared[0], prepared[1], prepared[2], rows],
            template: [
                ("C", 8), ("Dk", 128), ("Dv", 128), ("Hk", 16), ("Hv", 32), ("NS", 4), ("KS", 4),
            ],
            grid: (32, 16, 32), threadGroup: (32, 4, 1),
            outputShapes: [[1, T, 32, 128], [1, 32, 128, 128]], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }

    private static let prepKernel = MLXFast.metalKernel(
        name: "qwen35_fresh_chunk_prep",
        inputNames: ["q", "k", "g", "beta", "T"], outputNames: ["tp", "pm", "gf"],
        source: """
        // grid: (32, NC, B * Hk). One simdgroup per (b, key head, chunk): K K^T and
        // Q K^T are formed once and serve the Hv / Hk value heads of this key
        // head, which run side by side in lane groups of C lanes (lane = head
        // group * C + row), GP heads per pass.
        constexpr int HR = Hv / Hk;
        constexpr int GP = (32 / C) < HR ? (32 / C) : HR;
        constexpr int CT = C / 8;
        constexpr int LD = C + 1;
        threadgroup float KK[C * LD];
        threadgroup float QK[C * LD];
        threadgroup float Ah[GP * C * LD];
        threadgroup float Th[GP * C * LD];
        const uint lane = thread_index_in_simdgroup;
        const int T_ = T;
        const int NC = T_ / C;
        const int n = int(thread_position_in_grid.y);
        const int bk = int(thread_position_in_grid.z);
        const int b_idx = bk / Hk;
        const int hk = bk % Hk;
        const int t0 = n * C;
        const int ks = Hk * Dk;
        const device float* k_ = k + ((uint)b_idx * T_ + t0) * ks + hk * Dk;
        const device float* q_ = q + ((uint)b_idx * T_ + t0) * ks + hk * Dk;
        
        // lower tiles of K K^T and Q K^T
        _Pragma("clang loop unroll(full)")
        for (int ti = 0; ti < CT; ++ti) {
          _Pragma("clang loop unroll(full)")
          for (int tj = 0; tj <= ti; ++tj) {
            simdgroup_float8x8 akk = simdgroup_float8x8(0);
            simdgroup_float8x8 aqk = simdgroup_float8x8(0);
            _Pragma("clang loop unroll(full)")
            for (int d = 0; d < Dk / 8; ++d) {
              simdgroup_float8x8 ka, qa, kb;
              simdgroup_load(ka, k_ + (ti * 8) * ks + d * 8, ks);
              simdgroup_load(qa, q_ + (ti * 8) * ks + d * 8, ks);
              simdgroup_load(kb, k_ + (tj * 8) * ks + d * 8, ks, ulong2(0, 0), true);
              simdgroup_multiply_accumulate(akk, ka, kb, akk);
              simdgroup_multiply_accumulate(aqk, qa, kb, aqk);
            }
            simdgroup_store(akk, KK + (ti * 8) * LD + tj * 8, LD);
            simdgroup_store(aqk, QK + (ti * 8) * LD + tj * 8, LD);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        const int grp = int(lane) / C;
        const int row = int(lane) % C;
        const int gbase = grp * C;
        for (int h0 = 0; h0 < HR; h0 += GP) {
          const int h = h0 + grp;
          const bool live = grp < GP && h < HR;
          const int hv = hk * HR + (live ? h : 0);
          const uint slot = ((uint)b_idx * Hv + hv) * NC + n;
          // in-chunk inclusive prefix of log g within each lane group. A row
          // whose g underflowed (g == 0, or subnormal) has no finite log: it
          // zeroes every decay product spanning it, so it adds 0 to the prefix
          // and `zr` (the group's underflowed rows, one bit per row) masks
          // the products that span it.
          float gam = 0.0f;
          float bet = 0.0f;
          bool zero = false;
          if (live) {
            const float gv = g[((uint)b_idx * T_ + t0 + row) * Hv + hv];
            zero = !(gv >= FLT_MIN);
            gam = zero ? 0.0f : metal::precise::log(zero ? 1.0f : gv);
            bet = beta[((uint)b_idx * T_ + t0 + row) * Hv + hv];
          }
          const uint zr = uint(static_cast<simd_vote::vote_t>(simd_ballot(zero)) >> gbase) & ((1u << C) - 1u);
          _Pragma("clang loop unroll(full)")
          for (int off = 1; off < C; off <<= 1) {
            const float up = simd_shuffle_up(gam, ushort(off));
            gam += row >= off ? up : 0.0f;
          }
          const float gam_last = simd_shuffle(gam, ushort(gbase + C - 1));
          // rows (j, i] as a bit mask: ((2 << i) - 1) & ~((2 << j) - 1)
          const uint upto = (2u << row) - 1u;
          // decay factors for the scan: exp(gam_i), exp(gam_last - gam_i)
          if (live) {
            device float* gf_ = gf + slot * 2 * C;
            gf_[row] = (zr & upto) == 0u ? metal::precise::exp(gam) : 0.0f;
            gf_[C + row] = (zr & ~upto) == 0u ? metal::precise::exp(gam_last - gam) : 0.0f;
          }
          // P row (lower incl. diagonal) to device; A row (strict lower) to Ah
          threadgroup float* A_ = Ah + grp * C * LD;
          threadgroup float* M_ = Th + grp * C * LD;
          device float* pm_ = pm + slot * C * C;
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < C; ++j) {
            const float gj = simd_shuffle(gam, ushort(gbase + j));
            if (live) {
              float pv = 0.0f;
              float av = 0.0f;
              if (j <= row) {
                const uint span = upto & ~((2u << j) - 1u);
                const float e = (zr & span) == 0u ? metal::precise::exp(gam - gj) : 0.0f;
                pv = QK[row * LD + j] * e;
                if (j < row) av = bet * KK[row * LD + j] * e;
              }
              pm_[row * C + j] = pv;
              A_[row * LD + j] = av;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          // T = (I + A)^-1, column m = row per lane, stored transposed in M_[m][i]
          if (live) {
            const int m = row;
            for (int i = 0; i < C; ++i) M_[m * LD + i] = (i == m) ? 1.0f : 0.0f;
            for (int i = m + 1; i < C; ++i) {
              float s0 = 0.0f, s1 = 0.0f;
              int j = m;
              for (; j + 1 < i; j += 2) {
                s0 = metal::fma(A_[i * LD + j], M_[m * LD + j], s0);
                s1 = metal::fma(A_[i * LD + j + 1], M_[m * LD + j + 1], s1);
              }
              if (j < i) s0 = metal::fma(A_[i * LD + j], M_[m * LD + j], s0);
              M_[m * LD + i] = -(s0 + s1);
            }
            // T' = T diag(beta): T'[i][m] = T[i][m] * beta_m
            device float* tp_ = tp + slot * C * C;
            for (int i = 0; i < C; ++i) tp_[i * C + m] = M_[m * LD + i] * bet;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """)

    private static let scanKernel = MLXFast.metalKernel(
        name: "qwen35_fresh_chunk_vfold_pf_io32",
        inputNames: ["q", "k", "qkv", "w", "tp", "pm", "gf", "T"], outputNames: ["y", "state_out"],
        source: """
        // grid: (32, Dv / 8, B * Hv), threadgroup (32, NS, 1). One simdgroup per
        // 8 state rows (its S^T columns held in registers across the chunks); the
        // threadgroup shares each chunk's K, Q, T' and P through threadgroup
        // memory.
        constexpr int CT = C / 8;
        constexpr int DT = Dk / 8;
        constexpr int LK = Dk + 8;
        constexpr int LC = C + 8;
        constexpr int NT = 32 * NS;
        constexpr int KQ4 = C * (Dk / 4);        // float4s of one chunk of K (or Q)
        constexpr int TP4 = C * (C / 4);         // float4s of one chunk of T' (or P)
        threadgroup float Ksh[C * LK];
        threadgroup float Qsh[C * LK];
        threadgroup float TPsh[2 * C * LC];
        const uint lane = thread_index_in_simdgroup;
        const int tid = int(thread_index_in_threadgroup);
        const int T_ = T;
        const int NC = T_ / C;
        const int r0 = int(thread_position_in_grid.y) * 8;
        const int bh = int(thread_position_in_grid.z);
        const int b_idx = bh / Hv;
        const int hv = bh % Hv;
        const int hk = hv / (Hv / Hk);
        const int ks = Hk * Dk;
        const int vs = Hv * Dv;
        const short qid = lane / 4;
        const short fm = (qid & 4) + ((lane / 2) % 4);
        const short fn = (qid & 2) * 2 + (lane % 2) * 2;
        const device float* kbase = k + (uint)b_idx * T_ * ks + hk * Dk;
        const device float* qbase = q + (uint)b_idx * T_ * ks + hk * Dk;
        const device float* tbase = tp + (uint)bh * NC * C * C;
        const device float* pbase = pm + (uint)bh * NC * C * C;
        
        // Four padded convolution taps, then the existing stable SiLU.
        constexpr int VOFF = 2 * Hk * Dk;
        auto fresh_span32 = [](uint rows, uint cols, size_t rs, size_t cs) -> bool {
          constexpr size_t limit = 0xffffffffu;
          const size_t r = rows - 1u, c = cols - 1u;
          if (r != 0u && rs != 0u && (r > limit || rs > limit)) return false;
          if (c != 0u && cs != 0u && (c > limit || cs > limit)) return false;
          const size_t row_max = r * rs, col_max = c * cs;
          return row_max <= limit && col_max <= limit - row_max;
        };
        const bool value_io32 = fresh_span32(uint(T_ + 3), 8192u, qkv_strides[1], qkv_strides[2])
            && fresh_span32(8192u, 4u, w_strides[0], w_strides[1]);
        const size_t vqb = size_t(b_idx) * qkv_strides[0];
        const size_t vqs1 = qkv_strides[1];
        const size_t vqs2 = qkv_strides[2];
        const uint vqs1_32 = uint(qkv_strides[1]);
        const uint vqs2_32 = uint(qkv_strides[2]);
        const uint vws0_32 = uint(w_strides[0]);
        const uint vws1_32 = uint(w_strides[1]);
        const int vcol0 = VOFF + hv * Dv + r0 + fn;
        float vw0[KS], vw1[KS];
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < KS; j++) {
          if (value_io32) {
            vw0[j] = w[uint(vcol0) * vws0_32 + uint(j) * vws1_32];
            vw1[j] = w[uint(vcol0 + 1) * vws0_32 + uint(j) * vws1_32];
          } else {
            vw0[j] = w[vcol0 * w_strides[0] + j * w_strides[1]];
            vw1[j] = w[(vcol0 + 1) * w_strides[0] + j * w_strides[1]];
          }
        }
        auto vsilu = [&](float acc) -> float {
          const float sy = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
          const float sig = (acc < 0.0f) ? sy : 1.0f - sy;
          return acc * sig;
        };
        auto vconv = [&](int t, int col, thread const float* wt) -> float {
          float acc = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < KS; j++) {
            const int r = t + j;
            const float xv = value_io32
                ? float(qkv[uint(r) * vqs1_32 + uint(col) * vqs2_32])
                : float(qkv[vqb + r * vqs1 + col * vqs2]);
            acc = fma(xv, wt[j], acc);
          }
          return vsilu(acc);
        };
        simdgroup_float8x8 St[DT];
        _Pragma("clang loop unroll(full)")
        for (int d = 0; d < DT; ++d)
          St[d] = simdgroup_float8x8(0);
        
        constexpr int RK = (KQ4 + NT - 1) / NT;
        constexpr int RT = (2 * TP4 + NT - 1) / NT;
        float4 pk_[RK], pq_[RK], pt_[RT];
        auto pf_load = [&](int nn) {
          const int tt0 = nn * C;
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < RK; ++i) {
            const int e = tid + i * NT;
            if (e < KQ4) {
              const int row = e / (Dk / 4);
              const int c4 = (e % (Dk / 4)) * 4;
              const uint src = (uint)(tt0 + row) * ks + c4;
              pk_[i] = *(const device float4*)(kbase + src);
              pq_[i] = *(const device float4*)(qbase + src);
            }
          }
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < RT; ++i) {
            const int e = tid + i * NT;
            if (e < 2 * TP4) {
              const int which = e / TP4;
              const int f = e % TP4;
              const device float* src = (which == 0 ? tbase : pbase) + (uint)nn * C * C;
              pt_[i] = *(const device float4*)(src + f * 4);
            }
          }
        };
        auto pf_store = [&]() {
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < RK; ++i) {
            const int e = tid + i * NT;
            if (e < KQ4) {
              const int row = e / (Dk / 4);
              const int c4 = (e % (Dk / 4)) * 4;
              *(threadgroup float4*)(Ksh + row * LK + c4) = pk_[i];
              *(threadgroup float4*)(Qsh + row * LK + c4) = pq_[i];
            }
          }
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < RT; ++i) {
            const int e = tid + i * NT;
            if (e < 2 * TP4) {
              const int which = e / TP4;
              const int f = e % TP4;
              const int row = f / (C / 4);
              const int c4 = (f % (C / 4)) * 4;
              *(threadgroup float4*)(TPsh + which * C * LC + row * LC + c4) = pt_[i];
            }
          }
        };
        pf_load(0);
        
        for (int n = 0; n < NC; ++n) {
          const int t0 = n * C;
          
          device float* y_ = y + ((uint)b_idx * T_ + t0) * vs + hv * Dv + r0;
          const device float* gf_ = gf + ((uint)bh * NC + n) * 2 * C;
          // The next chunk is prefetched while this chunk computes.
          threadgroup_barrier(mem_flags::mem_threadgroup);
          pf_store();
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (n + 1 < NC) { pf_load(n + 1); }
        
          // X = K S^T, Xq = Q S^T (C x 8 each)
          simdgroup_float8x8 Xk[CT];
          simdgroup_float8x8 Xq[CT];
          _Pragma("clang loop unroll(full)")
          for (int ti = 0; ti < CT; ++ti) {
            Xk[ti] = simdgroup_float8x8(0);
            Xq[ti] = simdgroup_float8x8(0);
          }
          _Pragma("clang loop unroll(full)")
          for (int d = 0; d < DT; ++d) {
            _Pragma("clang loop unroll(full)")
            for (int ti = 0; ti < CT; ++ti) {
              simdgroup_float8x8 ka, qa;
              simdgroup_load(ka, Ksh + (ti * 8) * LK + d * 8, LK);
              simdgroup_load(qa, Qsh + (ti * 8) * LK + d * 8, LK);
              simdgroup_multiply_accumulate(Xk[ti], ka, St[d], Xk[ti]);
              simdgroup_multiply_accumulate(Xq[ti], qa, St[d], Xq[ti]);
            }
          }
          // Z = V - diag(exp gam) Xk (in Xk); Xq <- diag(exp gam) Xq
          _Pragma("clang loop unroll(full)")
          for (int ti = 0; ti < CT; ++ti) {
            const int row = ti * 8 + fm;
            const float eg = gf_[row];
            thread auto& zk = Xk[ti].thread_elements();
            thread auto& zq = Xq[ti].thread_elements();
            const float2 vv = float2(vconv(t0 + row, vcol0, vw0), vconv(t0 + row, vcol0 + 1, vw1));
            zk[0] = vv.x - eg * zk[0];
            zk[1] = vv.y - eg * zk[1];
            zq[0] = eg * zq[0];
            zq[1] = eg * zq[1];
          }
          // Delta = T' Z (T' lower triangular)
          simdgroup_float8x8 Dl[CT];
          _Pragma("clang loop unroll(full)")
          for (int ti = 0; ti < CT; ++ti) {
            Dl[ti] = simdgroup_float8x8(0);
            _Pragma("clang loop unroll(full)")
            for (int tj = 0; tj <= ti; ++tj) {
              simdgroup_float8x8 ta;
              simdgroup_load(ta, TPsh + (ti * 8) * LC + tj * 8, LC);
              simdgroup_multiply_accumulate(Dl[ti], ta, Xk[tj], Dl[ti]);
            }
          }
          // Y = diag(exp gam) Q S^T + P Delta
          _Pragma("clang loop unroll(full)")
          for (int ti = 0; ti < CT; ++ti) {
            _Pragma("clang loop unroll(full)")
            for (int tj = 0; tj <= ti; ++tj) {
              simdgroup_float8x8 pa;
              simdgroup_load(pa, TPsh + C * LC + (ti * 8) * LC + tj * 8, LC);
              simdgroup_multiply_accumulate(Xq[ti], pa, Dl[tj], Xq[ti]);
            }
            const int row = ti * 8 + fm;
            thread auto& ye = Xq[ti].thread_elements();
            *(device float2*)(y_ + row * vs + fn) = float2(ye[0], ye[1]);
          }
          // Delta~ = diag(exp(gam_C - gam)) Delta
          _Pragma("clang loop unroll(full)")
          for (int ti = 0; ti < CT; ++ti) {
            const int row = ti * 8 + fm;
            const float r = gf_[C + row];
            thread auto& de = Dl[ti].thread_elements();
            de[0] = de[0] * r;
            de[1] = de[1] * r;
          }
          // S^T <- exp(gam_C) S^T + K^T Delta~
          const float eC = gf_[C - 1];
          _Pragma("clang loop unroll(full)")
          for (int d = 0; d < DT; ++d) {
            thread auto& se = St[d].thread_elements();
            se[0] = se[0] * eC;
            se[1] = se[1] * eC;
            _Pragma("clang loop unroll(full)")
            for (int ti = 0; ti < CT; ++ti) {
              simdgroup_float8x8 kt;
              simdgroup_load(kt, Ksh + (ti * 8) * LK + d * 8, LK, ulong2(0, 0), true);
              simdgroup_multiply_accumulate(St[d], kt, Dl[ti], St[d]);
            }
          }
        }
        _Pragma("clang loop unroll(full)")
        for (int d = 0; d < DT; ++d)
          simdgroup_store(St[d], state_out + ((uint)bh * Dv + r0) * Dk + d * 8, Dk, ulong2(0, 0), true);
        """, ensureRowContiguous: false)
}

private enum Qwen35RegisterGDN {
    /// Time steps of k, q, g and beta staged in threadgroup memory per
    /// refill (4 KB for k and q at 128-wide heads) for prefill.
    static let stagedSteps = 4

    private static func source(valueRead: String, stageThreadgroup: Bool = false) -> String {
        """
        const uint T = q_shape[1];
        constexpr int R = 16;
        constexpr int DVPL = 2;
        constexpr int LPD = Dk / R;
        constexpr int DVPS = (32 / LPD) * DVPL;
        constexpr int DVPT = DVPS * 4;
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint dk0 = (lane % LPD) * R;
        const uint dvbase = threadgroup_position_in_grid.y * DVPT + sg * DVPS + (lane / LPD) * DVPL;
        const device float* q_ = q + (b_idx * T * Hk + hk_idx) * Dk + dk0;
        const device float* k_ = k + (b_idx * T * Hk + hk_idx) * Dk + dk0;
        const device float* v_ = v + (b_idx * T * Hv + hv_idx) * Dv + dvbase;
        auto g_ = g + b_idx * T * Hv + hv_idx;
        auto beta_ = beta + b_idx * T * Hv + hv_idx;
        device float* y_ = y + (b_idx * T * Hv + hv_idx) * Dv + dvbase;
        float state[DVPL][R];
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          const device float4* s_in4 = (const device float4*)(state_in + (n * Dv + dvbase + d) * Dk + dk0);
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 s4 = s_in4[j];
            state[d][4 * j + 0] = s4.x;
            state[d][4 * j + 1] = s4.y;
            state[d][4 * j + 2] = s4.z;
            state[d][4 * j + 3] = s4.w;
          }
        }
        float kr[R];
        float qr[R];
        \(stageThreadgroup ? """
        // Every lane of the threadgroup works on the same head, so each
        // step's k, q, g and beta are shared: they are staged in threadgroup
        // memory once per TS steps with coalesced loads instead of being
        // re-read from device memory by every lane at every step.
        threadgroup float4 ks[TS][Dk / 4];
        threadgroup float4 qs[TS][Dk / 4];
        threadgroup float gs[TS];
        threadgroup float bs[TS];
        const uint ltid = thread_position_in_threadgroup.x;
        const device float* kbase = k + (b_idx * T * Hk + hk_idx) * Dk;
        const device float* qbase = q + (b_idx * T * Hk + hk_idx) * Dk;
        const auto gbase = g + b_idx * T * Hv + hv_idx;
        const auto bbase = beta + b_idx * T * Hv + hv_idx;
        """ : "")
        for (int t = 0; t < T; ++t) {
          \(stageThreadgroup ? """
          const int s = t % TS;
          if (s == 0) {
            const int tn = min(TS, int(T) - t);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = ltid; idx < uint(tn) * (Dk / 4); idx += 128u) {
              const uint ss = idx / (Dk / 4), jj = idx % (Dk / 4);
              ks[ss][jj] = ((const device float4*)(kbase + (t + ss) * Hk * Dk))[jj];
              qs[ss][jj] = ((const device float4*)(qbase + (t + ss) * Hk * Dk))[jj];
            }
            if (ltid < uint(tn)) {
              gs[ltid] = gbase[(t + ltid) * Hv];
              bs[ltid] = bbase[(t + ltid) * Hv];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          const float gt = gs[s];
          const float bt = bs[s];
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ks[s][dk0 / 4 + j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            const float4 q4 = qs[s][dk0 / 4 + j];
            qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
          }
          """ : """
          const float gt = g_[0];
          const float bt = beta_[0];
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ((const device float4*)k_)[j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            const float4 q4 = ((const device float4*)q_)[j];
            qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
          }
          """)
          float kv[DVPL];
          float vt[DVPL];
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            \(valueRead)
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < R / 4; ++j) {
              state[d][4 * j] *= gt; state[d][4 * j + 1] *= gt;
              state[d][4 * j + 2] *= gt; state[d][4 * j + 3] *= gt;
              a0 = fma(state[d][4 * j], kr[4 * j], a0);
              a1 = fma(state[d][4 * j + 1], kr[4 * j + 1], a1);
              a2 = fma(state[d][4 * j + 2], kr[4 * j + 2], a2);
              a3 = fma(state[d][4 * j + 3], kr[4 * j + 3], a3);
            }
            kv[d] = (a0 + a1) + (a2 + a3);
          }
          #pragma clang loop unroll(full)
          for (int o = LPD / 2; o > 0; o >>= 1) {
            #pragma clang loop unroll(full)
            for (int d = 0; d < DVPL; ++d) {
              kv[d] += simd_shuffle_xor(kv[d], o);
            }
          }
          float out[DVPL];
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            const float delta = (vt[d] - kv[d]) * bt;
            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < R / 4; ++j) {
              state[d][4 * j] = fma(kr[4 * j], delta, state[d][4 * j]);
              state[d][4 * j + 1] = fma(kr[4 * j + 1], delta, state[d][4 * j + 1]);
              state[d][4 * j + 2] = fma(kr[4 * j + 2], delta, state[d][4 * j + 2]);
              state[d][4 * j + 3] = fma(kr[4 * j + 3], delta, state[d][4 * j + 3]);
              o0 = fma(state[d][4 * j], qr[4 * j], o0);
              o1 = fma(state[d][4 * j + 1], qr[4 * j + 1], o1);
              o2 = fma(state[d][4 * j + 2], qr[4 * j + 2], o2);
              o3 = fma(state[d][4 * j + 3], qr[4 * j + 3], o3);
            }
            out[d] = (o0 + o1) + (o2 + o3);
          }
          #pragma clang loop unroll(full)
          for (int o = LPD / 2; o > 0; o >>= 1) {
            #pragma clang loop unroll(full)
            for (int d = 0; d < DVPL; ++d) {
              out[d] += simd_shuffle_xor(out[d], o);
            }
          }
          if (lane % LPD == 0) {
            #pragma clang loop unroll(full)
            for (int d = 0; d < DVPL; ++d) {
              y_[d] = out[d];
            }
          }
          \(stageThreadgroup ? "v_ += Hv * Dv; y_ += Hv * Dv;" : "q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;")
        }
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          device float4* s_out4 = (device float4*)(state_out + (n * Dv + dvbase + d) * Dk + dk0);
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            s_out4[j] = float4(state[d][4 * j + 0], state[d][4 * j + 1], state[d][4 * j + 2], state[d][4 * j + 3]);
          }
        }
        """
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_register_v3",
        inputNames: ["q", "k", "v", "g", "beta", "state_in"],
        outputNames: ["y", "state_out"],
        source: source(valueRead: "vt[d] = v_[d];", stageThreadgroup: false))

    private static let valueKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_value_conv",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "conv_weight"],
        outputNames: ["y", "state_out"],
        source: source(valueRead: """
            vt[d] = 0.0f;
            if (lane % LPD == 0) {
              const uint channel = 2u * Hk * Dk + hv_idx * Dv + dvbase + d;
              const float4 cw4 = *((const device float4*)(conv_weight + channel * 4u));
              const uint v_base = (b_idx * (T + 3u) + t) * v_shape[2] + channel;
              const uint v_stride = v_shape[2];
              float cv = fma(v[v_base], cw4.x, 0.0f);
              cv = fma(v[v_base + v_stride], cw4.y, cv);
              cv = fma(v[v_base + 2u * v_stride], cw4.z, cv);
              cv = fma(v[v_base + 3u * v_stride], cw4.w, cv);
              const float sig = 1.0f / (1.0f + metal::exp(metal::abs(cv)));
              vt[d] = cv * (cv < 0.0f ? sig : 1.0f - sig);
            }
            vt[d] = simd_shuffle(vt[d], ushort((lane / LPD) * LPD));
            """, stageThreadgroup: true))


    static func run(q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray,
                    beta: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
        let B = k.dim(0), T = k.dim(1), Hk = k.dim(2), Dk = k.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)
        let result = kernel(
            [q, k, v, g, beta, state],
            template: [("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
            grid: (128, Dv / 32, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }
    static func runConvolvedValue(
        q: MLXArray, k: MLXArray, input: MLXArray, weight: MLXArray,
        g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let B = q.dim(0), T = q.dim(1), Hk = q.dim(2), Dk = q.dim(3), Hv = g.dim(2)
        let result = valueKernel(
            [q, k, input, g, beta, state, weight],
            template: [("Dk", Dk), ("Dv", 128), ("Hk", Hk), ("Hv", Hv), ("TS", stagedSteps)],
            grid: (128, 4, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [[B, T, Hv, 128], state.shape], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }

}

private enum Qwen35RegisterGatedSIMD {
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_register_gate_simd",
        inputNames: ["q", "k", "v", "a", "b", "a_log", "dt_bias", "state_in"],
        outputNames: ["y", "state_out"],
        source: """
        const uint T = q_shape[1];
        constexpr int R = 16;
        constexpr int DVPL = 2;
        constexpr int LPD = Dk / R;
        constexpr int DVPS = (32 / LPD) * DVPL;
        constexpr int DVPT = DVPS * 4;
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint dk0 = (lane % LPD) * R;
        const uint dvbase = threadgroup_position_in_grid.y * DVPT + sg * DVPS + (lane / LPD) * DVPL;
        const device float* q_ = q + (b_idx * T * Hk + hk_idx) * Dk + dk0;
        const device float* k_ = k + (b_idx * T * Hk + hk_idx) * Dk + dk0;
        const device float* v_ = v + (b_idx * T * Hv + hv_idx) * Dv + dvbase;
        const device float* a_ = a + b_idx * T * Hv + hv_idx;
        const device float* bb_ = b + b_idx * T * Hv + hv_idx;
        float negative_a = 0.0f;
        float bias = 0.0f;
        if (lane == 0 && !PREPARED_GATES) {
          negative_a = -metal::precise::exp(a_log[hv_idx]);
          bias = dt_bias[hv_idx];
        }
        device float* y_ = y + (b_idx * T * Hv + hv_idx) * Dv + dvbase;
        float state[DVPL][R];
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          const device float4* s_in4 = (const device float4*)(state_in + (n * Dv + dvbase + d) * Dk + dk0);
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 s4 = s_in4[j];
            state[d][4 * j + 0] = s4.x;
            state[d][4 * j + 1] = s4.y;
            state[d][4 * j + 2] = s4.z;
            state[d][4 * j + 3] = s4.w;
          }
        }
        float kr[R];
        float qr[R];
        for (int t = 0; t < T; ++t) {
          float local_g = 0.0f, local_beta = 0.0f;
          if (lane == 0) {
            if (PREPARED_GATES) {
              local_g = a_[0];
              local_beta = bb_[0];
            } else {
              local_g = qwen35_simd_gate_decay(a_[0], bias, negative_a);
              local_beta = qwen35_simd_gate_beta(bb_[0]);
            }
          }
          const float gt = simd_broadcast(local_g, 0);
          const float bt = simd_broadcast(local_beta, 0);
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ((const device float4*)k_)[j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            const float4 q4 = ((const device float4*)q_)[j];
            qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
          }
          float kv[DVPL];
          float vt[DVPL];
          const float2 v2 = *((const device float2*)v_);
          vt[0] = v2.x; vt[1] = v2.y;
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < R / 4; ++j) {
              state[d][4 * j] *= gt; state[d][4 * j + 1] *= gt;
              state[d][4 * j + 2] *= gt; state[d][4 * j + 3] *= gt;
              a0 = fma(state[d][4 * j], kr[4 * j], a0);
              a1 = fma(state[d][4 * j + 1], kr[4 * j + 1], a1);
              a2 = fma(state[d][4 * j + 2], kr[4 * j + 2], a2);
              a3 = fma(state[d][4 * j + 3], kr[4 * j + 3], a3);
            }
            kv[d] = (a0 + a1) + (a2 + a3);
          }
          kv[0] += simd_shuffle_xor(kv[0], 4);
          kv[1] += simd_shuffle_xor(kv[1], 4);
          kv[0] += simd_shuffle_xor(kv[0], 2);
          kv[1] += simd_shuffle_xor(kv[1], 2);
          kv[0] += simd_shuffle_xor(kv[0], 1);
          kv[1] += simd_shuffle_xor(kv[1], 1);
          float out[DVPL];
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            const float delta = (vt[d] - kv[d]) * bt;
            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < R / 4; ++j) {
              state[d][4 * j] = fma(kr[4 * j], delta, state[d][4 * j]);
              state[d][4 * j + 1] = fma(kr[4 * j + 1], delta, state[d][4 * j + 1]);
              state[d][4 * j + 2] = fma(kr[4 * j + 2], delta, state[d][4 * j + 2]);
              state[d][4 * j + 3] = fma(kr[4 * j + 3], delta, state[d][4 * j + 3]);
              o0 = fma(state[d][4 * j], qr[4 * j], o0);
              o1 = fma(state[d][4 * j + 1], qr[4 * j + 1], o1);
              o2 = fma(state[d][4 * j + 2], qr[4 * j + 2], o2);
              o3 = fma(state[d][4 * j + 3], qr[4 * j + 3], o3);
            }
            out[d] = (o0 + o1) + (o2 + o3);
          }
          out[0] += simd_shuffle_xor(out[0], 4);
          out[1] += simd_shuffle_xor(out[1], 4);
          out[0] += simd_shuffle_xor(out[0], 2);
          out[1] += simd_shuffle_xor(out[1], 2);
          out[0] += simd_shuffle_xor(out[0], 1);
          out[1] += simd_shuffle_xor(out[1], 1);
          if (lane % LPD == 0) {
            *((device float2*)y_) = float2(out[0], out[1]);
          }
          q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; a_ += Hv; bb_ += Hv;
        }
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          device float4* s_out4 = (device float4*)(state_out + (n * Dv + dvbase + d) * Dk + dk0);
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            s_out4[j] = float4(state[d][4 * j + 0], state[d][4 * j + 1], state[d][4 * j + 2], state[d][4 * j + 3]);
          }
        }
        """,
        header: """
        inline float qwen35_simd_gate_decay(float a, float bias, float negative_a) {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            const float x = a + bias;
            float sp;
            if (metal::isnan(x)) {
                sp = metal::numeric_limits<float>::quiet_NaN();
            } else {
                const float maximum = metal::max(x, 0.0f);
                const float minimum = metal::min(x, 0.0f);
                constexpr float infinity = metal::numeric_limits<float>::infinity();
                if (minimum == -infinity || maximum == infinity) {
                    sp = maximum;
                } else {
                    const float z = metal::exp(minimum - maximum);
                    const float xp1 = 1.0f + z;
                    const float logarithm = xp1 == metal::numeric_limits<float>::max()
                        ? metal::numeric_limits<float>::max()
                        : (xp1 == 1.0f ? z : z * (metal::log(xp1) / (xp1 - 1.0f)));
                    sp = maximum + logarithm;
                }
            }
            return metal::precise::exp(negative_a * sp);
        }

        inline float qwen35_simd_gate_beta(float b) {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            const float y = 1.0f / (1.0f + metal::exp(metal::abs(b)));
            return b < 0.0f ? y : 1.0f - y;
        }
        """)

    static func run(q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray,
                    b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
                    state: MLXArray, preparedGates: Bool = false) -> (MLXArray, MLXArray) {
        let B = k.dim(0), T = k.dim(1), Hk = k.dim(2), Dk = k.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)
        let result = kernel(
            [q, k, v, a, b, aLog, dtBias, state],
            template: [("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("PREPARED_GATES", preparedGates)],
            grid: (128, Dv / 32, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }
}

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    // Inference-only cache. It is intentionally not registered in the module
    // topology, so checkpoint and adapter paths remain stable.
    private var fusedInProj: Linear?
    private var nativeGateProjection: Qwen35DenseNativeGateProjection.Cache?
    private var fusedInputSourceSignature: [MLXArray]?
    private var fusedInputPermanentlyIneligible = false

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        self.fusedInProj = nil
        self.fusedInputSourceSignature = nil

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    private func exactQuantizedInputProjections() -> (
        qkv: QuantizedLinear, z: QuantizedLinear,
        b: QuantizedLinear, a: QuantizedLinear
    )? {
        guard let qkv = inProjQKV as? QuantizedLinear,
            let z = inProjZ as? QuantizedLinear,
            let b = inProjB as? QuantizedLinear,
            let a = inProjA as? QuantizedLinear,
            ObjectIdentifier(type(of: qkv)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: z)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: b)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: a)) == ObjectIdentifier(QuantizedLinear.self),
            qkv.bias == nil, z.bias == nil, b.bias == nil, a.bias == nil,
            qkv.bits == z.bits, qkv.bits == b.bits, qkv.bits == a.bits,
            qkv.groupSize == z.groupSize,
            qkv.groupSize == b.groupSize,
            qkv.groupSize == a.groupSize,
            qkv.mode == z.mode, qkv.mode == b.mode, qkv.mode == a.mode
        else { return nil }
        return (qkv, z, b, a)
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        let prefixes = ["in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a."]
        let replacesInputProjection = parameters.flattened().contains { key, _ in
            prefixes.contains(where: key.hasPrefix)
        }
        let result = try super.update(
            parameters: parameters, verify: verify,
            path: path, modulePath: modulePath)
        if replacesInputProjection {
            fusedInProj = nil
            fusedInputSourceSignature = nil
            fusedInputPermanentlyIneligible = false
        }
        return result
    }

    override func updateModule(key: String, _ value: Any) throws {
        try super.updateModule(key: key, value)
        if key == "in_proj_qkv" || key == "in_proj_z"
            || key == "in_proj_b" || key == "in_proj_a"
        {
            fusedInProj = nil
            fusedInputSourceSignature = nil
            fusedInputPermanentlyIneligible = false
        }
    }

    var hasFusedInputProjection: Bool { fusedInProj != nil }

    private func inputProjectionSourceSignature(
        _ projections: (
            qkv: QuantizedLinear, z: QuantizedLinear,
            b: QuantizedLinear, a: QuantizedLinear
        )
    ) -> [MLXArray] {
        [
            projections.qkv.weight, projections.qkv.scales,
            projections.z.weight, projections.z.scales,
            projections.b.weight, projections.b.scales,
            projections.a.weight, projections.a.scales,
        ] + [
            projections.qkv.biases, projections.z.biases,
            projections.b.biases, projections.a.biases,
        ].compactMap { $0 }
    }

    private func sourceSignatureMatches(_ current: [MLXArray], _ cached: [MLXArray]) -> Bool {
        current.count == cached.count
            && zip(current, cached).allSatisfy { $0 === $1 }
    }

    private func exactFrozenQuantizedInputProjections() -> (
        qkv: QuantizedLinear, z: QuantizedLinear,
        b: QuantizedLinear, a: QuantizedLinear
    )? {
        // Reject ineligible module types and policies before traversing the
        // trainable-parameter tree on hot decode forwards.
        guard let qkv = inProjQKV as? QuantizedLinear,
            let z = inProjZ as? QuantizedLinear,
            let b = inProjB as? QuantizedLinear,
            let a = inProjA as? QuantizedLinear,
            ObjectIdentifier(type(of: qkv)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: z)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: b)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: a)) == ObjectIdentifier(QuantizedLinear.self),
            qkv.bias == nil, z.bias == nil, b.bias == nil, a.bias == nil,
            qkv.bits == z.bits, qkv.bits == b.bits, qkv.bits == a.bits,
            qkv.groupSize == z.groupSize,
            qkv.groupSize == b.groupSize,
            qkv.groupSize == a.groupSize,
            qkv.mode == z.mode, qkv.mode == b.mode, qkv.mode == a.mode
        else {
            fusedInputPermanentlyIneligible = true
            return nil
        }
        let prefixes = ["in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a."]
        guard !trainableParameters().flattened().contains(where: { key, _ in
            prefixes.contains(where: key.hasPrefix)
        }) else { return nil }
        return (qkv, z, b, a)
    }

    @discardableResult
    func prepareFusedInputProjection() -> Bool {
        if fusedInputPermanentlyIneligible { return false }
        if let fusedInputSourceSignature {
            guard let projections = exactFrozenQuantizedInputProjections(),
                sourceSignatureMatches(inputProjectionSourceSignature(projections), fusedInputSourceSignature)
            else {
                fusedInProj = nil
                self.fusedInputSourceSignature = nil
                return false
            }
            return fusedInProj != nil
        }
        guard let projections = exactFrozenQuantizedInputProjections() else { return false }
        let fusedBiases: MLXArray?
        switch (
            projections.qkv.biases, projections.z.biases,
            projections.b.biases, projections.a.biases
        ) {
        case let (.some(qkv), .some(z), .some(b), .some(a)):
            fusedBiases = concatenated([qkv, z, b, a], axis: 0)
        case (nil, nil, nil, nil):
            fusedBiases = nil
        default:
            return false
        }
        let fusedWeight = concatenated([
            projections.qkv.weight, projections.z.weight,
            projections.b.weight, projections.a.weight,
        ], axis: 0)
        let fusedScales = concatenated([
            projections.qkv.scales, projections.z.scales,
            projections.b.scales, projections.a.scales,
        ], axis: 0)
        eval(fusedWeight, fusedScales)
        if let fusedBiases { eval(fusedBiases) }

        let fused = QuantizedLinear(
            weight: fusedWeight, bias: nil,
            scales: fusedScales, biases: fusedBiases,
            groupSize: projections.qkv.groupSize,
            bits: projections.qkv.bits,
            mode: projections.qkv.mode)
        fused.freeze()

        let qkvRows = keyDim * 2 + valueDim
        let zRows = valueDim
        let bRows = numVHeads
        let ranges = [
            0 ..< qkvRows,
            qkvRows ..< (qkvRows + zRows),
            (qkvRows + zRows) ..< (qkvRows + zRows + bRows),
            (qkvRows + zRows + bRows) ..< (qkvRows + zRows + 2 * bRows),
        ]
        func sourceView(_ rows: Range<Int>) -> QuantizedLinear {
            let view = QuantizedLinear(
                weight: fusedWeight[rows], bias: nil,
                scales: fusedScales[rows],
                biases: fusedBiases.map { $0[rows] },
                groupSize: projections.qkv.groupSize,
                bits: projections.qkv.bits,
                mode: projections.qkv.mode)
            view.freeze()
            return view
        }
        // Preserve checkpoint/adaptor-facing module names as views into the
        // one fused physical allocation. A later module replacement invalidates
        // `fusedInProj` through updateModule before the next forward.
        try! update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(sourceView(ranges[0])),
                "in_proj_z": .value(sourceView(ranges[1])),
                "in_proj_b": .value(sourceView(ranges[2])),
                "in_proj_a": .value(sourceView(ranges[3])),
            ]), verify: [])
        // updateModule invalidates on source replacement; assign only after
        // the stable named views have been installed.
        fusedInProj = fused
        guard let current = exactFrozenQuantizedInputProjections() else {
            fusedInProj = nil
            return false
        }
        fusedInputSourceSignature = inputProjectionSourceSignature(current)
        return true
    }

    private func projectInputs(
        _ inputs: MLXArray, B: Int, S: Int,
        boundary: HadamardQuantizedLinear.VerifyBoundary? = nil,
        projectedInputs: [MLXArray]? = nil
    ) -> (
        qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
    ) {
        guard projectedInputs == nil, prepareFusedInputProjection(), let fusedInProj else {
            let shared = projectedInputs ?? boundary.flatMap {
                HadamardQuantizedLinear.boundaryProjections($0, [inProjQKV, inProjZ])
            } ?? (S >= 128
                ? HadamardQuantizedLinear.sharedPromptProjections(inputs, [inProjQKV, inProjZ])
                : HadamardQuantizedLinear.sharedShortProjections(inputs, [inProjQKV, inProjZ]))
            let qkv = shared?[0] ?? inProjQKV(inputs)
            let z = (shared?[1] ?? inProjZ(inputs)).reshaped(B, S, numVHeads, headVDim)
            if let ba = Qwen35DenseNativeGateProjection.apply(
                inputs, b: inProjB, a: inProjA, cache: &nativeGateProjection)
            {
                return (qkv, z, ba.0, ba.1)
            }
            if let ba = Qwen35DenseGateProjection.apply(inputs, b: inProjB, a: inProjA) {
                return (qkv, z, ba.0, ba.1)
            }
            return (qkv, z, inProjB(inputs), inProjA(inputs))
        }
        let outFused = fusedInProj(inputs)
        let qkvDim = keyDim * 2 + valueDim
        let zDim = valueDim
        let bDim = numVHeads
        let total = qkvDim + zDim + 2 * bDim
        return (
            outFused[0..., 0..., 0 ..< qkvDim],
            outFused[0..., 0..., qkvDim ..< (qkvDim + zDim)].reshaped(
                B, S, numVHeads, headVDim),
            outFused[0..., 0..., (qkvDim + zDim) ..< (qkvDim + zDim + bDim)],
            outFused[0..., 0..., (qkvDim + zDim + bDim) ..< total]
        )
    }

    /// The retained convolution tail is three rows, but a slice aliases the
    /// whole chunk. A packed checkpoint carries that parent into every
    /// recurrent transaction, which costs gigabytes over a prefill. Copying
    /// the tail drops the parent. It moves bytes; it changes no bit, no
    /// precision and no recurrence, and it is skipped for single-token decode
    /// and for every other checkpoint format.
    private func retainedConvTail(
        of convInput: MLXArray, keeping nKeep: Int, chunkWidth: Int
    ) -> MLXArray {
        let tail = convInput[0..., (convInput.dim(1) - nKeep)...]
        guard chunkWidth > 1, inProjQKV is HadamardQuantizedLinear else { return tail }
        return contiguous(tail)
    }

    private func normalizedQK(_ convOut: MLXArray, B: Int, S: Int) -> (MLXArray, MLXArray) {
        if let fused = Qwen35FusedQKNorm.apply(
            convOut, heads: numKHeads, dimensions: headKDim)
        {
            return fused
        }
        let q = convOut[0..., 0..., 0 ..< keyDim].reshaped(B, S, numKHeads, headKDim)
        let k = convOut[0..., 0..., keyDim ..< (2 * keyDim)].reshaped(B, S, numKHeads, headKDim)
        let invScale = pow(Float(headKDim), -0.5)
        return (
            MLXArray(pow(invScale, 2)).asType(q.dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6),
            MLXArray(invScale).asType(k.dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6))
    }


    private func preparedConvolutionInputs(
        _ input: MLXArray, a: MLXArray, b: MLXArray,
        foldValue: Bool, timewise: Bool = false
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray?, g: MLXArray?, beta: MLXArray?, backing: MLXArray?) {
        if !timewise, convKernelSize == 4, conv1d.bias == nil,
            let prepared = Qwen35FusedGDNPrework.apply(
                input, weight: conv1d.weight, a: a, b: b, aLog: aLog, dtBias: dtBias,
                keyHeads: numKHeads, valueHeads: numVHeads,
                keyDimensions: headKDim, valueDimensions: headVDim, foldValue: foldValue)
        {
            return (prepared.q, prepared.k, prepared.v, prepared.g, prepared.beta, prepared.v)
        }
        let B = input.dim(0), S = input.dim(1) - convKernelSize + 1
        let convolved: MLXArray
        if timewise {
            convolved = silu(concatenated((0 ..< S).map { position in
                conv1d(input[0..., position ..< (position + convKernelSize), 0...])
            }, axis: 1))
        } else if foldValue {
            convolved = silu(MLX.conv1d(
                input[0..., 0..., 0 ..< (2 * keyDim)],
                conv1d.weight[0 ..< (2 * keyDim), 0..., 0...], groups: 2 * keyDim))
        } else {
            convolved = silu(conv1d(input))
        }
        let (q, k) = normalizedQK(convolved, B: B, S: S)
        let v = foldValue ? nil
            : convolved[0..., 0..., (2 * keyDim)...].reshaped(B, S, numVHeads, headVDim)
        return (q, k, v, nil, nil, convolved)
    }

    private func updateRecurrence(
        q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray, state: MLXArray?, mask: MLXArray?,
        preparedGates: (g: MLXArray, beta: MLXArray)? = nil
    ) -> (MLXArray, MLXArray) {
        guard mask == nil, headKDim == 128, headVDim == 128,
            q.dtype == .float32, k.dtype == .float32, v.dtype == .float32
        else {
            return gatedDeltaUpdate(
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                state: state, mask: mask)
        }
        if q.dim(1) <= 16, a.dtype == .float32, b.dtype == .float32,
            aLog.dtype == .float32, dtBias.dtype == .float32
        {
            let initial = state ?? MLXArray.zeros(
                [q.dim(0), numVHeads, headVDim, headKDim], dtype: .float32)
            let floatState = initial.dtype == .float32 ? initial : initial.asType(.float32)
            return Qwen35RegisterGatedSIMD.run(
                q: q, k: k, v: v, a: preparedGates?.g ?? a, b: preparedGates?.beta ?? b,
                aLog: aLog, dtBias: dtBias, state: floatState, preparedGates: preparedGates != nil)
        }
        let beta = sigmoid(b).asType(.float32)
        let g = computeGatedDeltaG(aLog, a, dtBias)
        let initial = state ?? MLXArray.zeros(
            [q.dim(0), numVHeads, headVDim, headKDim], dtype: .float32)
        let floatState = initial.dtype == .float32 ? initial : initial.asType(.float32)
        return Qwen35RegisterGDN.run(
            q: q, k: k, v: v, g: g, beta: beta, state: floatState)
    }

    // MARK: - _processChunk (MTP helper)

    /// Process one time-chunk of the linear-attention layer.
    ///
    /// Extracted from `callAsFunction` so the MTP verify cycle can run the prefix
    /// (n_confirmed tokens) and draft suffix separately, snapshotting the SSM/conv
    /// state in between for rollback on draft rejection.
    ///
    /// Port of omlx commit 696d90a:
    ///   patches/mlx_lm_mtp/qwen35_model.py `GatedDeltaNet._process_chunk`
    ///
    /// - Parameters:
    ///   - qkv: Already-masked QKV for this chunk [B, S_chunk, conv_dim]
    ///   - a, b: Input projections for this chunk [B, S_chunk, ...]
    ///   - convState: Initial conv state [B, conv_kernel_size-1, conv_dim]
    ///   - ssmState: Initial SSM state (nil on first token)
    ///   - mask: SSM mask for `gatedDeltaUpdate` (optional)
    /// - Returns: `(out, newConvState, newSsmState)`
    private func processChunk(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        ssmState: MLXArray?,
        mask: MLXArray?,
        freshState: Bool = false
    ) -> (out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray) {
        let B = qkv.dim(0)
        let S = qkv.dim(1)

        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = retainedConvTail(of: convInput, keeping: nKeep, chunkWidth: S)
        let foldValue = mask == nil && S >= 64 && headKDim == 128 && headVDim == 128
            && convKernelSize == 4 && convInput.dtype == .float32
            && conv1d.weight.dtype == .float32 && conv1d.weight.shape == [convDim, 4, 1]
            && conv1d.bias == nil && aLog.dtype == .float32 && dtBias.dtype == .float32
        let prepared = preparedConvolutionInputs(convInput, a: a, b: b, foldValue: foldValue)
        let (qNormed, kNormed) = (prepared.q, prepared.k)
        let out: MLXArray
        let newSsmState: MLXArray
        if foldValue {
            let beta = prepared.beta ?? sigmoid(b).asType(.float32)
            let g = prepared.g ?? computeGatedDeltaG(aLog, a, dtBias)
            let initial = ssmState ?? MLXArray.zeros(
                [B, numVHeads, headVDim, headKDim], dtype: .float32)
            if freshState, let fresh = Qwen35FreshChunkGDN.runConvolvedValue(
                q: qNormed, k: kNormed, input: convInput, weight: conv1d.weight, g: g, beta: beta)
            {
                (out, newSsmState) = fresh
            } else {
                (out, newSsmState) = Qwen35RegisterGDN.runConvolvedValue(
                    q: qNormed, k: kNormed, input: convInput, weight: conv1d.weight,
                    g: g, beta: beta, state: initial.asType(.float32))
            }
        } else {
            let v = prepared.v!
            (out, newSsmState) = updateRecurrence(
                q: qNormed, k: kNormed, v: v, a: a, b: b,
                aLog: aLog, dtBias: dtBias, state: ssmState, mask: mask)
        }
        return (out, newConvState, newSsmState)
    }

    /// Run one legacy-cache verify chunk while retaining only the transformed
    /// recurrence inputs needed to rebuild a shorter committed prefix. The
    /// CBv2 rectangular path below uses the same tape and replay math.
    private func processChunkStashingPrefix(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        ssmState: MLXArray?,
        mask: MLXArray?
    ) -> (
        out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray,
        tape: ArraysCache.PrefixReplayTape
    ) {
        let S = qkv.dim(1)
        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = retainedConvTail(of: convInput, keeping: nKeep, chunkWidth: S)
        let prepared = preparedConvolutionInputs(convInput, a: a, b: b, foldValue: false)
        let (qNormed, kNormed) = (prepared.q, prepared.k)
        let v = prepared.v!

        let recurrence = updateRecurrence(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: ssmState,
            mask: mask
        )
        let tape = ArraysCache.PrefixReplayTape(
            convInput: convInput,
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            ssmPre: ssmState.map { $0[.ellipsis] },
            mask: mask.map { $0[.ellipsis] },
            rowCount: S,
            convStateRows: nKeep
        )
        return (recurrence.0, newConvState, recurrence.1, tape)
    }

    private func canReplayPrefix(
        tape: ArraysCache.PrefixReplayTape, committedRows: Int
    ) -> Bool {
        guard committedRows > 0,
              committedRows < tape.rowCount,
              tape.convStateRows == convKernelSize - 1,
              tape.convInput.ndim == 3,
              tape.q.ndim == 4,
              tape.k.ndim == 4,
              tape.v.ndim == 4,
              tape.a.ndim == 3,
              tape.b.ndim == 3
        else { return false }

        let batch = tape.q.dim(0)
        guard tape.convInput.shape
                  == [batch, tape.convStateRows + tape.rowCount, convDim],
              tape.q.shape == [batch, tape.rowCount, numKHeads, headKDim],
              tape.k.shape == [batch, tape.rowCount, numKHeads, headKDim],
              tape.v.shape == [batch, tape.rowCount, numVHeads, headVDim],
              tape.a.shape == [batch, tape.rowCount, numVHeads],
              tape.b.shape == [batch, tape.rowCount, numVHeads]
        else { return false }

        if let ssmPre = tape.ssmPre,
           ssmPre.shape != [batch, numVHeads, headVDim, headKDim]
        {
            return false
        }
        if let mask = tape.mask,
           mask.shape != [batch, tape.rowCount]
        {
            return false
        }
        return true
    }

    fileprivate func canReplayPrefix(
        cache: MambaCache, committedRows: Int
    ) -> Bool {
        guard let tape = cache.prefixReplayTape else { return false }
        return canReplayPrefix(tape: tape, committedRows: committedRows)
    }

    private func replayedPrefixState(
        tape: ArraysCache.PrefixReplayTape, committedRows: Int
    ) -> CBv2RecurrentLayerState {
        precondition(
            canReplayPrefix(tape: tape, committedRows: committedRows),
            "Qwen35 invalid compact recurrent prefix replay")
        let rows = 0 ..< committedRows
        let boundarySsm = updateRecurrence(
            q: tape.q[0..., rows, 0...],
            k: tape.k[0..., rows, 0...],
            v: tape.v[0..., rows, 0...],
            a: tape.a[0..., rows, 0...],
            b: tape.b[0..., rows, 0...],
            aLog: aLog,
            dtBias: dtBias,
            state: tape.ssmPre,
            mask: tape.mask.map { $0[0..., rows] }
        ).1
        let boundaryConvView = tape.convInput[
            0...,
            committedRows ..< (committedRows + tape.convStateRows),
            0...]
        let boundaryConv = boundaryConvView + MLXArray.zeros(
            boundaryConvView.shape, dtype: boundaryConvView.dtype)
        return CBv2RecurrentLayerState(conv: boundaryConv, ssm: boundarySsm)
    }

    /// Reconstruct the fp32 recurrent state after `committedRows` verify rows
    /// from the exact pre-verify state and transformed recurrence inputs.
    /// Call only after every GDN layer has passed `canReplayPrefix`.
    fileprivate func replayPrefix(
        cache: MambaCache, committedRows: Int
    ) -> Bool {
        guard canReplayPrefix(cache: cache, committedRows: committedRows),
              let tape = cache.prefixReplayTape
        else { return false }

        let state = replayedPrefixState(tape: tape, committedRows: committedRows)
        cache[0] = state.conv
        cache[1] = state.ssm
        cache.clearMTPTransientState()
        return true
    }

    // MARK: - callAsFunction

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        nConfirmed: Int = 0
    ) -> MLXArray {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py GatedDeltaNet.__call__
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        // A verify tape belongs to exactly one forward. Starting another
        // forward consumes neither its values nor its state, so release it
        // before constructing this forward's transient rollback data.
        cache?.clearMTPTransientState()

        var (qkv, z, b, a) = projectInputs(inputs, B: B, S: S)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        // Apply mask to full qkv before any chunking.
        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let ssmState = cache?[1]
        let out: MLXArray
        let finalConvState: MLXArray
        let finalSsmState: MLXArray
        var pendingPrefixTape: ArraysCache.PrefixReplayTape?

        if nConfirmed == 1 && S >= 3 && mask == nil {
            // A wider MTP verify stays as one recurrence. Retain transformed
            // inputs instead of one full fp32 SSM checkpoint per boundary.
            let (o, c, s, tape) = processChunkStashingPrefix(
                qkv: qkv, a: a, b: b,
                convState: convState, ssmState: ssmState, mask: mask)
            out = o
            finalConvState = c
            finalSsmState = s
            pendingPrefixTape = tape
        } else if nConfirmed > 0 && nConfirmed < S {
            // Width-2 and masked verification retain the established rollback
            // snapshot path. CBv2 capture uses its separate request-state API.
            let maskC = mask.map { $0[0..., 0..<nConfirmed] }
            let maskD = mask.map { $0[0..., nConfirmed...] }

            let (outC, convC, ssmC) = processChunk(
                qkv: qkv[0..., 0..<nConfirmed, 0...],
                a: a[0..., 0..<nConfirmed, 0...],
                b: b[0..., 0..<nConfirmed, 0...],
                convState: convState,
                ssmState: ssmState,
                mask: maskC
            )
            cache?.rollbackState = (convC, ssmC)

            let (outD, convF, ssmF) = processChunk(
                qkv: qkv[0..., nConfirmed..., 0...],
                a: a[0..., nConfirmed..., 0...],
                b: b[0..., nConfirmed..., 0...],
                convState: convC,
                ssmState: ssmC,
                mask: maskD
            )
            out = concatenated([outC, outD], axis: 1)
            finalConvState = convF
            finalSsmState = ssmF
        } else {
            let (o, c, s) = processChunk(
                qkv: qkv, a: a, b: b,
                convState: convState,
                ssmState: ssmState,
                mask: mask,
                freshState: nConfirmed == 0 && ssmState == nil
            )
            out = o
            finalConvState = c
            finalSsmState = s
        }

        if let cache {
            cache[0] = finalConvState
            cache[1] = finalSsmState
            cache.prefixReplayTape = pendingPrefixTape
        }

        let normedOut = norm(out, gate: z)
        return outProj(normedOut.reshaped(B, S, -1))
    }

    /// CBv2 target path. Request-owned conv/SSM rows are gathered into the
    /// active rectangle, evaluated once, then split back into their owning
    /// transactions. No recurrent tensor is represented as attention KV.
    func cbv2Forward(
        _ inputs: MLXArray,
        modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation],
        projectedInputs: [MLXArray]? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        precondition(recurrentState.count == B, "Qwen35 CBv2 recurrent row count mismatch")

        let (qkv, z, b, a) = projectInputs(inputs, B: B, S: S, projectedInputs: projectedInputs)

        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        convRows.reserveCapacity(B)
        ssmRows.reserveCapacity(B)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: modelLayerIndex)
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros(
                        [1, convKernelSize - 1, convDim], dtype: inputs.dtype))
            ssmRows.append(
                state?.ssm
                    ?? MLXArray.zeros(
                        [1, numVHeads, headVDim, headKDim], dtype: .float32))
        }

        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssmState = ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)
        let (out, newConvState, newSsmState) = processChunk(
            qkv: qkv, a: a, b: b,
            convState: convState, ssmState: ssmState, mask: nil,
            freshState: B == 1 && recurrentState[0].inputState(modelLayerIndex: modelLayerIndex) == nil)

        for (row, evaluation) in recurrentState.enumerated() {
            do {
                try evaluation.stage(
                    modelLayerIndex: modelLayerIndex,
                    conv: newConvState[row ..< row + 1],
                    ssm: newSsmState[row ..< row + 1])
            } catch {
                preconditionFailure(
                    "Qwen35 CBv2 recurrent stage failed at layer \(modelLayerIndex): \(error)")
            }
        }

        let normedOut = norm(out, gate: z)
        return outProj(normedOut.reshaped(B, S, -1))
    }

    /// CBv2 MTP rectangular verify path. Widths one and two retain the
    /// established captured-boundary implementation. Wider windows run the
    /// recurrence once over the whole window and retain compact transformed
    /// inputs, allowing a strict accepted prefix to be replayed lazily without
    /// another target-model forward or a per-position fp32 SSM stack.
    func cbv2ForwardCaptured(
        _ inputs: MLXArray,
        modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation],
        exactTargetVerify: Bool = false,
        boundary: HadamardQuantizedLinear.VerifyBoundary? = nil,
        projectedInputs: [MLXArray]? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        precondition(recurrentState.count == B, "Qwen35 CBv2 recurrent row count mismatch")
        precondition(S >= 1, "Qwen35 capture-verify window must be non-empty")

        let qkv: MLXArray
        let z: MLXArray
        let b: MLXArray
        let a: MLXArray
        if exactTargetVerify {
            let exact = qwen35A3BExactW4G64ProjectionQuad(
                inProjQKV, inProjZ, inProjB, inProjA, inputs)
            qkv = exact.0
            z = exact.1.reshaped(B, S, numVHeads, headVDim)
            b = exact.2
            a = exact.3
        } else {
            // Preserve main's fused GDN projection construction and graph.
            (qkv, z, b, a) = projectInputs(inputs, B: B, S: S, boundary: boundary, projectedInputs: projectedInputs)
        }

        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        convRows.reserveCapacity(B)
        ssmRows.reserveCapacity(B)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: modelLayerIndex)
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros(
                        [1, convKernelSize - 1, convDim], dtype: inputs.dtype))
            ssmRows.append(
                state?.ssm
                    ?? MLXArray.zeros(
                        [1, numVHeads, headVDim, headKDim], dtype: .float32))
        }
        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssmState = ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)

        // Conv over the whole window in one call (same as processChunk); the
        // per-position conv tail is a free slice of the padded input: after
        // consuming position s, the retained tail is convInput[:, s+1 ..<
        // s+1+nKeep].
        let nKeep = convKernelSize - 1
        let fusedPrework = exactTargetVerify || conv1d.bias != nil || convKernelSize != 4
            ? nil : Qwen35FusedGDNCompleteShortPrework.apply(
                state: convState, qkv: qkv, weight: conv1d.weight, a: a, b: b,
                aLog: aLog, dtBias: dtBias, keyHeads: numKHeads, valueHeads: numVHeads,
                keyDimensions: headKDim, valueDimensions: headVDim)
        let convInput: MLXArray
        let prepared: (q: MLXArray, k: MLXArray, v: MLXArray, backing: MLXArray,
                       gates: (g: MLXArray, beta: MLXArray)?)
        if let fusedPrework {
            convInput = fusedPrework.convInput
            prepared = (fusedPrework.q, fusedPrework.k, fusedPrework.v, fusedPrework.v,
                (fusedPrework.g, fusedPrework.beta))
        } else {
            let candidateInput = concatenated([convState, qkv], axis: 1)
            let fusedConv = exactTargetVerify || conv1d.bias != nil || convKernelSize != 4
                ? nil : Qwen35FusedConvSilu.apply(state: convState, qkv: qkv, weight: conv1d.weight)
            convInput = fusedConv?.0 ?? candidateInput
            let convOut: MLXArray
            if let fusedConv {
                convOut = fusedConv.1
            } else if exactTargetVerify, S > 1 {
                convOut = silu(concatenated(
                    (0 ..< S).map { position in
                        conv1d(convInput[
                            0..., position ..< (position + convKernelSize), 0...])
                    }, axis: 1))
            } else {
                convOut = silu(conv1d(convInput))
            }
            let qk = normalizedQK(convOut, B: B, S: S)
            let value = convOut[0..., 0..., (2 * keyDim)...].reshaped(B, S, numVHeads, headVDim)
            prepared = (qk.0, qk.1, value, convOut, nil)
        }
        let (qNormed, kNormed) = (prepared.q, prepared.k)
        let v = prepared.v

        let out: MLXArray
        if S >= 3 {
            let recurrence = updateRecurrence(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: ssmState,
                mask: nil, preparedGates: prepared.gates)
            out = recurrence.0
            let finalSsmState = recurrence.1

            for (row, evaluation) in recurrentState.enumerated() {
                let rowRange = row ..< (row + 1)
                let finalConv = convInput[rowRange, S ..< (S + nKeep), 0...]
                let finalSSM = finalSsmState[rowRange]
                let tape = ArraysCache.PrefixReplayTape(
                    convInput: convInput[rowRange],
                    q: qNormed[rowRange],
                    k: kNormed[rowRange],
                    v: v[rowRange],
                    a: a[rowRange],
                    b: b[rowRange],
                    ssmPre: ssmState[rowRange],
                    mask: nil,
                    rowCount: S,
                    convStateRows: nKeep)
                // Count unique additional buffers retained by this stage.
                // `finalConv` aliases `convInput` until full acceptance detaches
                // its exact tail at commit. A one-row `ssmPre` aliases the
                // already-accounted committed generation when it existed before
                // this verify. The value tensor keeps its source allocation.
                let convOutBacking = prepared.backing[rowRange]
                var roots = [
                    tape.convInput, tape.q, tape.k, convOutBacking, tape.a, tape.b,
                ]
                let inputSSM =
                    evaluation.inputState(modelLayerIndex: modelLayerIndex)?.ssm
                if B > 1 || inputSSM == nil {
                    if let ssmPre = tape.ssmPre { roots.append(ssmPre) }
                }
                var strictReplayRoots = roots
                if B == 1, let inputSSM {
                    // The pending baseline already charges this state. After
                    // strict acceptance it becomes an additional replay input.
                    strictReplayRoots.append(inputSSM)
                }

                func checkedByteCount(_ arrays: [MLXArray]) -> Int {
                    var total = 0
                    for array in arrays {
                        let (bytes, multiplyOverflow) =
                            array.size.multipliedReportingOverflow(by: array.dtype.size)
                        let (sum, addOverflow) = total.addingReportingOverflow(bytes)
                        guard !multiplyOverflow, !addOverflow else {
                            preconditionFailure(
                                "Qwen35 compact recurrent replay byte accounting overflow")
                        }
                        total = sum
                    }
                    return total
                }
                let materializedBytes = checkedByteCount(roots + [finalSSM])
                let strictReplayRetainedBytes = checkedByteCount(strictReplayRoots)
                let fullAcceptanceRetainedBytes = checkedByteCount([tape.convInput])
                do {
                    try evaluation.stagePrefixReplay(
                        modelLayerIndex: modelLayerIndex,
                        positions: S,
                        finalConv: finalConv,
                        finalSSM: finalSSM,
                        materializedByteCount: materializedBytes,
                        evaluationRoots: strictReplayRoots,
                        strictReplayRetainedByteCount: strictReplayRetainedBytes,
                        strictReplayRetainedRoots: strictReplayRoots,
                        fullAcceptanceRetainedByteCount: fullAcceptanceRetainedBytes,
                        fullAcceptanceRetainedRoots: [tape.convInput],
                        fullAcceptance: {
                            let detachedConv = finalConv + MLXArray.zeros(
                                finalConv.shape, dtype: finalConv.dtype)
                            return CBv2RecurrentLayerState(
                                conv: detachedConv, ssm: finalSSM)
                        },
                        replay: { [unowned self] keepPositions in
                            self.replayedPrefixState(
                                tape: tape, committedRows: keepPositions)
                        })
                } catch {
                    preconditionFailure(
                        "Qwen35 CBv2 compact replay stage failed at layer "
                            + "\(modelLayerIndex): \(error)")
                }
            }
        } else {
            var outs: [MLXArray] = []
            var ssmStates: [MLXArray] = []
            outs.reserveCapacity(S)
            ssmStates.reserveCapacity(S)
            var state = ssmState
            for s in 0 ..< S {
                let (stepOut, next) = updateRecurrence(
                    q: qNormed[0..., s ..< (s + 1)],
                    k: kNormed[0..., s ..< (s + 1)],
                    v: v[0..., s ..< (s + 1)],
                    a: a[0..., s ..< (s + 1)],
                    b: b[0..., s ..< (s + 1)],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: state,
                    mask: nil, preparedGates: prepared.gates.map {
                        (g: $0.g[0..., s ..< (s + 1)], beta: $0.beta[0..., s ..< (s + 1)])
                    })
                outs.append(stepOut)
                ssmStates.append(next)
                state = next
            }

            for (row, evaluation) in recurrentState.enumerated() {
                let convStack = concatenated(
                    (0 ..< S).map { s in
                        convInput[row ..< (row + 1), (s + 1) ..< (s + 1 + nKeep)]
                    }, axis: 0)
                let ssmStack = concatenated(
                    ssmStates.map { $0[row ..< (row + 1)] }, axis: 0)
                do {
                    try evaluation.stageCaptured(
                        modelLayerIndex: modelLayerIndex,
                        conv: convStack, ssm: ssmStack, positions: S)
                } catch {
                    preconditionFailure(
                        "Qwen35 CBv2 captured stage failed at layer "
                            + "\(modelLayerIndex): \(error)")
                }
            }
            out = outs.count == 1 ? outs[0] : concatenated(outs, axis: 1)
        }
        if !exactTargetVerify, out.ndim == 4, out.dim(-1) == 128,
            let packed = outProj as? HadamardQuantizedLinear,
            let gated = packed.callAfterGatedNorm(
                out, gate: z, normWeight: norm.weight, eps: norm.eps)
        {
            return gated
        }
        let normedOut = norm(out, gate: z)
        let projectionInput = normedOut.reshaped(B, S, -1)
        if exactTargetVerify {
            return qwen35A3BExactW4G64Projection(outProj, projectionInput)
        }
        return outProj(projectionInput)
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    private static let lastQueryAttentionEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_LAST_QUERY_ATTENTION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer
    let mrope: Qwen35MRoPE

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )
        self.mrope = Qwen35MRoPE(
            rope: self.rope, dim: max(1, ropeDims), base: args.ropeTheta,
            scalingConfig: args.ropeScaling,
            sections: args.mropeSection)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let shared = L >= 128
            ? HadamardQuantizedLinear.sharedPromptProjections(x, [qProj, kProj, vProj])
            : HadamardQuantizedLinear.sharedShortProjections(x, [qProj, kProj, vProj])
        let qProjOutput = shared?[0] ?? qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = shared?[1] ?? kProj(x)
        var values = shared?[2] ?? vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }

    func canUseLastQueryPrefill(
        _ x: MLXArray, cache: any CBv2AttendingLayerCache,
        positionIds: MLXArray? = nil, exactTargetVerify: Bool = false
    ) -> Bool {
        guard Self.lastQueryAttentionEnabled, x.ndim == 3, x.dim(0) == 1,
            x.dim(1) >= 128, x.dim(2) == 4096,
            attentionHeads == 16, kvHeads == 4, qNorm.weight.dim(0) == 256,
            positionIds == nil, !exactTargetVerify,
            cache.kind.attention == .full, cache.kind.sharesKVWithLayer == nil,
            (cache as? CBv2LayerCache)?.mtpSerializesRectangularAttention != true,
            cache is any CBv2LastQueryPrefillLayerCache,
            let packedQuery = qProj as? HadamardQuantizedLinear,
            let packedOutput = oProj as? HadamardQuantizedLinear
        else { return false }
        return packedQuery.canCallPromptLastRow(x) && packedOutput.canCallPromptLastRow(x)
    }

    func cbv2Forward(
        _ x: MLXArray, cache: any CBv2AttendingLayerCache,
        positionIds: MLXArray? = nil,
        exactTargetVerify: Bool = false,
        boundary: HadamardQuantizedLinear.VerifyBoundary? = nil,
        projectedInputs: [MLXArray]? = nil,
        lastQueryOnly: Bool = false
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let narrowsToLastQuery = lastQueryOnly && canUseLastQueryPrefill(
            x, cache: cache, positionIds: positionIds, exactTargetVerify: exactTargetVerify)
        let queryRows = narrowsToLastQuery ? 1 : L

        let projectionLayers = narrowsToLastQuery ? [kProj, vProj] : [qProj, kProj, vProj]
        let shared = projectedInputs ?? (!exactTargetVerify
            ? boundary.flatMap {
                HadamardQuantizedLinear.boundaryProjections($0, projectionLayers)
            } ?? (L >= 128
                ? HadamardQuantizedLinear.sharedPromptProjections(x, projectionLayers)
                : HadamardQuantizedLinear.sharedShortProjections(x, projectionLayers))
            : nil)
        let qProjOutput: MLXArray
        if narrowsToLastQuery {
            if let supplied = projectedInputs?[0], supplied.dim(1) == 1 {
                qProjOutput = supplied
            } else {
                qProjOutput = (qProj as! HadamardQuantizedLinear).callPromptLastRow(x)!
            }
        } else {
            qProjOutput = exactTargetVerify
                ? qwen35A3BExactW4G64Projection(qProj, x) : (shared?[0] ?? qProj(x))
        }
        let projectedQueryRows = qProjOutput.dim(1)
        let qGate = qProjOutput.reshaped(B, projectedQueryRows, attentionHeads, -1)
        let queryGate = narrowsToLastQuery
            ? qGate[0..., (projectedQueryRows - 1)..., 0..., 0...] : qGate
        let qSplit = queryGate.split(parts: 2, axis: -1)
        var queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        let gate = qSplit[1].reshaped(B, queryRows, -1)
        let keyIndex = narrowsToLastQuery && projectedInputs == nil ? 0 : 1
        let valueIndex = narrowsToLastQuery && projectedInputs == nil ? 1 : 2
        let kProjection = exactTargetVerify
            ? qwen35A3BExactW4G64Projection(kProj, x) : (shared?[keyIndex] ?? kProj(x))
        var keys = kNorm(kProjection.reshaped(B, L, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let vProjection = exactTargetVerify
            ? qwen35A3BExactW4G64Projection(vProj, x) : (shared?[valueIndex] ?? vProj(x))
        let values = vProjection.reshaped(B, L, kvHeads, -1)
            .transposed(0, 2, 1, 3)

        // Text-only Qwen positions are ordinary scalar-equivalent positions,
        // but histories differ across rows. Capture the per-row device offsets
        // before the cache advances and use the array RoPE overload.
        if let positionIds {
            (queries, keys) = mrope.apply(
                queries: queries, keys: keys, positionIds: positionIds)
        } else {
            let offsets = cache.positionOffsets + 0
            queries = rope(queries, offset: narrowsToLastQuery ? offsets + Int32(L - 1) : offsets)
            keys = rope(keys, offset: offsets)
        }

        let attended: MLXArray
        if narrowsToLastQuery, let lastQuery = cache as? any CBv2LastQueryPrefillLayerCache {
            attended = lastQuery.updateAndAttendLastQuery(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        } else {
            attended = cache.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        }
        if !narrowsToLastQuery, !exactTargetVerify, B == 1, let packed = oProj as? HadamardQuantizedLinear,
            let gated = packed.callAfterAttentionGate(attended, queryProjection: qProjOutput)
        {
            return gated
        }
        let output = attended
            .transposed(0, 2, 1, 3)
            .reshaped(B, queryRows, -1)
        let projectionInput = sigmoidMultiply(output, gate)
        if narrowsToLastQuery {
            return (oProj as! HadamardQuantizedLinear).callPromptLastRow(projectionInput)!
        }
        return exactTargetVerify
            ? qwen35A3BExactW4G64Projection(oProj, projectionInput)
            : oProj(projectionInput)
    }
}

/// Qwen3.5 interleaved 3-axis M-RoPE. Request positions arrive as function
/// inputs; the module retains configuration only.
final class Qwen35MRoPE {
    private let rope: RoPELayer
    private let rotaryDim: Int
    private let defaultInvFreq: MLXArray?
    private let sections: [Int]
    // Which of the three position planes (t/h/w) owns each frequency,
    // precomputed as [1, 1, 1, rotaryDim/2] so the default path interleaves
    // with one takeAlong instead of a per-frequency slice loop.
    private let mropeIndices: MLXArray

    init(
        rope: RoPELayer, dim: Int, base: Float,
        scalingConfig: [String: StringOrNumber]?, sections: [Int]
    ) {
        self.rope = rope
        self.rotaryDim = max(1, dim)
        let ropeType: String = {
            if let value = scalingConfig?["type"] ?? scalingConfig?["rope_type"],
                case .string(let type) = value
            {
                return type
            }
            return "default"
        }()
        if ropeType == "default" || ropeType == "mrope" {
            let exponents = MLXArray(stride(from: 0, to: self.rotaryDim, by: 2))
                .asType(.float32) / Float(self.rotaryDim)
            self.defaultInvFreq = 1 / pow(MLXArray(base), exponents)
        } else {
            self.defaultInvFreq = nil
        }
        let resolvedSections = sections.count >= 3 ? sections : [11, 11, 10]
        self.sections = resolvedSections
        let frequencyCount = max(1, self.rotaryDim / 2)
        var indices = [Int32](repeating: 0, count: frequencyCount)
        for (dimension, offset) in [(1, 1), (2, 2)] {
            let end = min(resolvedSections[dimension] * 3, frequencyCount)
            for index in stride(from: offset, to: end, by: 3) {
                indices[index] = Int32(dimension)
            }
        }
        self.mropeIndices = MLXArray(indices).reshaped(1, 1, 1, -1)
    }

    private func axis(forFrequency index: Int, frequencyCount: Int) -> Int {
        for (axis, offset) in [(1, 1), (2, 2)] {
            let length = min(sections[axis] * 3, frequencyCount)
            if index >= offset && index < length && (index - offset) % 3 == 0 {
                return axis
            }
        }
        return 0
    }

    func apply(
        queries: MLXArray, keys: MLXArray, positionIds: MLXArray
    ) -> (MLXArray, MLXArray) {
        var positions = positionIds
        if positions.ndim == 2 {
            positions = broadcast(
                positions[.newAxis, 0..., 0...],
                to: [3, positions.dim(0), positions.dim(1)])
        }
        precondition(positions.ndim == 3 && positions.dim(0) == 3)
        precondition(rotaryDim % 2 == 0 && rotaryDim <= queries.dim(-1))

        if let defaultInvFreq {
            let all = positions.asType(.float32)[0..., 0..., 0..., .newAxis]
                * defaultInvFreq[.newAxis, .newAxis, .newAxis, 0...]
            let frequency = takeAlong(all, mropeIndices, axis: 0).squeezed(axis: 0)
            let angles = concatenated([frequency, frequency], axis: -1)
            let cosine = cos(angles).asType(queries.dtype).expandedDimensions(axis: 1)
            let sine = sin(angles).asType(queries.dtype).expandedDimensions(axis: 1)
            func applyDefault(_ value: MLXArray) -> MLXArray {
                let rotating = value[.ellipsis, ..<rotaryDim]
                let half = rotating.dim(-1) / 2
                let rotatedHalf = concatenated(
                    [-rotating[.ellipsis, half...], rotating[.ellipsis, ..<half]], axis: -1)
                let rotated = rotating * cosine + rotatedHalf * sine
                return rotaryDim < value.dim(-1)
                    ? concatenated([rotated, value[.ellipsis, rotaryDim...]], axis: -1)
                    : rotated
            }
            return (applyDefault(queries), applyDefault(keys))
        }

        let queryHeads = queries.dim(1)
        let combined = concatenated([queries, keys], axis: 1)
        let batch = combined.dim(0)
        let heads = combined.dim(1)
        let length = combined.dim(2)
        let headDim = combined.dim(3)
        let flattened = combined.transposed(0, 2, 1, 3)
            .reshaped([batch * length, heads, 1, headDim])
        let byAxis = (0 ..< 3).map { axis in
            rope(flattened, offset: positions[axis, 0..., 0...].flattened())
                .reshaped([batch, length, heads, headDim])
                .transposed(0, 2, 1, 3)
        }
        let frequencyCount = rotaryDim / 2
        var firstHalf: [MLXArray] = []
        var secondHalf: [MLXArray] = []
        firstHalf.reserveCapacity(frequencyCount)
        secondHalf.reserveCapacity(frequencyCount)
        for index in 0 ..< frequencyCount {
            let rotated = byAxis[axis(forFrequency: index, frequencyCount: frequencyCount)]
            firstHalf.append(rotated[.ellipsis, index ..< index + 1])
            secondHalf.append(
                rotated[.ellipsis, frequencyCount + index ..< frequencyCount + index + 1])
        }
        var result = concatenated(firstHalf + secondHalf, axis: -1)
        if rotaryDim < combined.dim(-1) {
            result = concatenated([result, combined[.ellipsis, rotaryDim...]], axis: -1)
        }
        return (
            result[0..., ..<queryHeads, 0..., 0...],
            result[0..., queryHeads..., 0..., 0...])
    }
}

func qwen35FlattenMoEInputs(
    x: MLXArray, indices: MLXArray, scores: MLXArray
) -> (x: MLXArray, indices: MLXArray, scores: MLXArray) {
    precondition(x.ndim >= 2 && indices.ndim >= 2 && scores.shape == indices.shape)
    return (
        x.reshaped([-1, x.dim(-1)]),
        indices.reshaped([-1, indices.dim(-1)]),
        scores.reshaped([-1, scores.dim(-1)])
    )
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int
    private let routerFinalizer: Qwen35A3BRouterFinalizer

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    /// - Parameter fuseGateUp: when true the routed experts use one fused
    ///   `gate_up_proj` SwitchLinear (one gather_qmm serves gate+up; the
    ///   sanitizers concatenate split checkpoints into the fused layout).
    ///   The inline MTP head passes false: its weights load through
    ///   `Qwen35InlineMTPAssistant` whose per-path quantization table is
    ///   keyed on the split `gate_proj`/`up_proj` module paths.
    init(_ args: Qwen35TextConfiguration, fuseGateUp: Bool = true) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok
        self.routerFinalizer = qwen35A3BRouterFinalizer(
            hidden: args.hiddenSize, experts: args.numExperts,
            topK: args.numExpertsPerTok, normalize: args.normTopkProb)

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts,
            fuseGateUp: fuseGateUp,
            weightedReductionProfile: .qwen35ProductionSwiGLU
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, exactTargetVerify: false)
    }

    func callAsFunction(
        _ x: MLXArray, exactTargetVerify: Bool
    ) -> MLXArray {
        var gates = exactTargetVerify
            ? qwen35A3BExactTimewiseProjection(gate, x) : gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let (inds, scores) = routerFinalizer(gates)

        let tokenShape = x.shape
        let flattened = qwen35FlattenMoEInputs(x: x, indices: inds, scores: scores)
        let flatX = flattened.x
        let flatIndices = flattened.indices
        let flatScores = flattened.scores
        let combined = switchMLP.callAndWeightedReduce(
            flatX, flatIndices, weights: flatScores.asType(x.dtype),
            fuseSortedReduction: true, isProductionPrefill: true
        ).reshaped(tokenShape)

        var sharedY = sharedExpert.qwen35TargetVerify(
            x, exact: exactTargetVerify)
        let sharedGate = exactTargetVerify
            ? qwen35A3BExactTimewiseProjection(sharedExpertGate, x)
            : sharedExpertGate(x)
        sharedY = sigmoid(sharedGate) * sharedY

        return combined + sharedY
    }
}

extension Qwen3NextMLP {
    func qwen35ProjectedGateUp(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
        if let down = downProj as? HadamardQuantizedLinear,
            let output = down.callAfterSwiGLU(gate: gate, up: up)
        {
            return output
        }
        return downProj(silu(gate) * up)
    }

    func qwen35TargetVerify(_ x: MLXArray, exact: Bool) -> MLXArray {
        if !exact, x.ndim >= 2,
            let shared = x.dim(-2) >= 128
                ? HadamardQuantizedLinear.sharedPromptProjections(x, [gateProj, upProj])
                : HadamardQuantizedLinear.sharedShortProjections(x, [gateProj, upProj])
        {
            return qwen35ProjectedGateUp(shared[0], shared[1])
        }
        if !exact,
            let shared = HadamardQuantizedLinear.sharedShortProjections(x, [gateProj, upProj])
        {
            return qwen35ProjectedGateUp(shared[0], shared[1])
        }
        guard exact else { return self(x) }
        let (gate, up) = qwen35A3BExactW4G64ProjectionPair(
            gateProj, upProj, x)
        return qwen35A3BExactW4G64Projection(downProj, silu(gate) * up)
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        nConfirmed: Int = 0
    ) -> MLXArray {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py DecoderLayer.__call__
        // Passes nConfirmed through to the linear-attention sublayer.
        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                nConfirmed: nConfirmed)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    func cbv2Forward(
        _ x: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false,
        exactTargetVerify: Bool = false,
        lastPositionOnly: Bool = false,
        lastQueryOnly: Bool = false
    ) -> MLXArray {
        let out = cbv2ForwardParts(
            x, modelLayerIndex: modelLayerIndex, attentionCache: attentionCache,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: captureRecurrentWindow,
            exactTargetVerify: exactTargetVerify, lastPositionOnly: lastPositionOnly,
            lastQueryOnly: lastQueryOnly)
        return out.h + out.f
    }

    static let pendingResidualEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_FUSED_RESNORM_INPUT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private func fusedInputBoundary(
        _ x: MLXArray, pending: MLXArray, firstProjectionLastRow: Bool = false
    )
        -> (residual: MLXArray, projections: [MLXArray], normed: MLXArray?)?
    {
        guard ObjectIdentifier(type(of: inputLayerNorm)) == ObjectIdentifier(RMSNorm.self)
        else { return nil }
        let projections: [Linear]
        if let linearAttn {
            projections = [linearAttn.inProjQKV, linearAttn.inProjZ]
        } else if let selfAttn {
            projections = [selfAttn.qProj, selfAttn.kProj, selfAttn.vProj]
        } else {
            return nil
        }
        return HadamardQuantizedLinear.sharedResidualNormProjections(
            x, residual: pending, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
            projections: projections, keepNormed: isLinear,
            firstProjectionLastRow: firstProjectionLastRow)
    }

    func cbv2ForwardParts(
        _ x: MLXArray, pending: MLXArray? = nil,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false,
        exactTargetVerify: Bool = false,
        lastPositionOnly: Bool = false,
        lastQueryOnly: Bool = false
    ) -> (input: MLXArray, h: MLXArray, f: MLXArray) {
        var input = x
        var normed: MLXArray?
        var projected: [MLXArray]?
        let selectedLastQuery: Bool
        if lastQueryOnly, let selfAttn, let attentionCache {
            selectedLastQuery = selfAttn.canUseLastQueryPrefill(
                x, cache: attentionCache, positionIds: positionIds,
                exactTargetVerify: exactTargetVerify)
        } else {
            selectedLastQuery = false
        }
        if let pending {
            if let boundary = fusedInputBoundary(
                x, pending: pending, firstProjectionLastRow: selectedLastQuery)
            {
                input = boundary.residual
                normed = boundary.normed
                projected = boundary.projections
            } else {
                input = x + pending
            }
        }
        let layerInput = normed ?? inputLayerNorm(input)
        let r: MLXArray
        if isLinear {
            precondition(attentionCache == nil, "Qwen35 recurrent layer received attention KV")
            if captureRecurrentWindow {
                r = linearAttn!.cbv2ForwardCaptured(
                    layerInput, modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState,
                    exactTargetVerify: exactTargetVerify, projectedInputs: projected)
            } else {
                r = linearAttn!.cbv2Forward(
                    layerInput, modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState, projectedInputs: projected)
            }
        } else {
            guard let attentionCache else {
                preconditionFailure("Qwen35 full-attention layer is missing its CBv2 cache")
            }
            r = selfAttn!.cbv2Forward(
                layerInput, cache: attentionCache, positionIds: positionIds,
                exactTargetVerify: exactTargetVerify, projectedInputs: projected,
                lastQueryOnly: selectedLastQuery)
        }
        if (!lastPositionOnly || input.dim(1) == 1), let dense = mlp as? Qwen3NextMLP,
            let boundary = HadamardQuantizedLinear.sharedResidualNormProjections(
                input, residual: r, weight: postAttentionLayerNorm.weight, eps: postAttentionLayerNorm.eps,
                projections: [dense.gateProj, dense.upProj])
        {
            return (input, boundary.residual, dense.qwen35ProjectedGateUp(
                boundary.projections[0], boundary.projections[1]))
        }
        let h: MLXArray
        if lastPositionOnly && input.dim(1) > 1 {
            let lastInput = input[0..., (input.dim(1) - 1)..., 0...]
            let lastResidual = r.dim(1) == 1 ? r : r[0..., (r.dim(1) - 1)..., 0...]
            h = lastInput + lastResidual
        } else {
            h = input + r
        }
        let normalized = postAttentionLayerNorm(h)
        let feedForward: MLXArray
        if let sparse = mlp as? Qwen35SparseMoeBlock {
            feedForward = sparse(
                normalized, exactTargetVerify: exactTargetVerify)
        } else if let dense = mlp as? Qwen3NextMLP {
            feedForward = dense.qwen35TargetVerify(
                normalized, exact: exactTargetVerify)
        } else {
            preconditionFailure("Qwen35 decoder has an unsupported MLP module")
        }
        return (input, h, feedForward)
    }
}

extension Qwen35DecoderLayer {
    /// `cbv2Forward(x)` on the capture-verify path, returning its residual
    /// and MLP output unadded so the next layer can fuse the add into its
    /// input kernel. Nil when the MLP is not dense.
    func cbv2ForwardSplitOutput(
        _ x: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> (residual: MLXArray, delta: MLXArray)? {
        guard let dense = mlp as? Qwen3NextMLP else { return nil }
        if !isLinear && attentionCache == nil { return nil }
        let r: MLXArray
        if isLinear {
            r = linearAttn!.cbv2ForwardCaptured(
                inputLayerNorm(x), modelLayerIndex: modelLayerIndex,
                recurrentState: recurrentState, exactTargetVerify: false)
        } else {
            r = selfAttn!.cbv2Forward(
                inputLayerNorm(x), cache: attentionCache!, positionIds: positionIds,
                exactTargetVerify: false)
        }
        if let post = HadamardQuantizedLinear.sharedResidualNormProjections(
            x, residual: r, weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps, projections: [dense.gateProj, dense.upProj])
        {
            return (post.residual, dense.qwen35ProjectedGateUp(post.projections[0], post.projections[1]))
        }
        let h = x + r
        return (h, dense.qwen35TargetVerify(postAttentionLayerNorm(h), exact: false))
    }

    /// Verify-width layer whose input arrives as the previous layer's
    /// residual `x` plus its last output `delta`, not yet added. The add, the
    /// input RMSNorm and the int8 codes of the rotated norm come from one
    /// kernel (`HadamardQuantizedLinear.verifyBoundary`) and feed every input
    /// projection; the layer then returns its own residual and MLP output,
    /// again unadded, for the next layer. Every value is the bits of the
    /// unfused `cbv2Forward(x + delta)`. Nil (nothing built) when the layer
    /// or the shapes do not qualify.
    func cbv2ForwardVerifyFused(
        _ x: MLXArray, delta: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> (residual: MLXArray, delta: MLXArray)? {
        guard let dense = mlp as? Qwen3NextMLP else { return nil }
        let firstProjections: [Linear]
        if isLinear {
            guard let linearAttn else { return nil }
            firstProjections = [linearAttn.inProjQKV, linearAttn.inProjZ]
        } else {
            guard let selfAttn, attentionCache != nil else { return nil }
            firstProjections = [selfAttn.qProj, selfAttn.kProj, selfAttn.vProj]
        }
        guard
            let boundary = HadamardQuantizedLinear.verifyBoundary(
                x, delta: delta, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                projections: firstProjections)
        else { return nil }
        let r: MLXArray
        if isLinear {
            r = linearAttn!.cbv2ForwardCaptured(
                boundary.normalized, modelLayerIndex: modelLayerIndex,
                recurrentState: recurrentState, exactTargetVerify: false,
                boundary: boundary)
        } else {
            r = selfAttn!.cbv2Forward(
                boundary.normalized, cache: attentionCache!, positionIds: positionIds,
                exactTargetVerify: false, boundary: boundary)
        }
        if let post = HadamardQuantizedLinear.sharedResidualNormProjections(
            boundary.residual, residual: r, weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps, projections: [dense.gateProj, dense.upProj])
        {
            return (post.residual, dense.qwen35ProjectedGateUp(post.projections[0], post.projections[1]))
        }
        let h = boundary.residual + r
        return (h, dense.qwen35TargetVerify(postAttentionLayerNorm(h), exact: false))
    }
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    /// Metadata-only inspection for packed variants whose normalizers promote
    /// activations. No tensor evaluation or serving arithmetic changes.
    var cbv2UniformLayerNormDType: DType? {
        guard let dtype = layers.first?.inputLayerNorm.weight.dtype,
            layers.allSatisfy({
                $0.inputLayerNorm.weight.dtype == dtype
                    && $0.postAttentionLayerNorm.weight.dtype == dtype
            })
        else { return nil }
        return dtype
    }

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int
    let exactTargetVerify: Bool

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1
        self.exactTargetVerify =
            Qwen35A3BConstructionContext.targetVerifyArithmetic == .exactM1

        super.init()
    }

    /// Returns the pre-norm hidden state from the final layer.
    ///
    /// The caller (`Qwen35TextModel`) applies `norm` and the LM head on top.
    /// This split returns logits and raw capture rows in one pass; the bound
    /// Qwen MTP adapter applies this same final norm at trusted-history ingress.
    ///
    /// Port of omlx commit 696d90a:
    ///   patches/mlx_lm_mtp/qwen35_model.py `_patch_qwen3_5_text_model`
    ///   (returns hidden_states before self.model.norm so TextModel can apply it)
    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        nConfirmed: Int = 0
    ) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask,
                cache: cacheArray?[i], nConfirmed: nConfirmed)
        }

        // Return pre-norm hidden states. Norm is applied by Qwen35TextModel.
        return hiddenStates
    }

    /// Rebuild every recurrent layer at the same committed verify boundary.
    /// Eligibility is checked for every layer before persistent state changes.
    /// Attention caches are intentionally untouched; the caller trims their
    /// rejected suffix after recurrent replay succeeds.
    func replayRecurrentPrefix(
        cache: [KVCache?], committedRows: Int
    ) -> Bool {
        guard cache.count == layers.count, committedRows > 0 else {
            clearRecurrentPrefixReplay(cache: cache)
            return false
        }

        var foundRecurrentLayer = false
        for (i, layer) in layers.enumerated() where layer.isLinear {
            foundRecurrentLayer = true
            guard let mamba = cache[i] as? MambaCache,
                  let linear = layer.linearAttn,
                  linear.canReplayPrefix(
                    cache: mamba, committedRows: committedRows)
            else {
                clearRecurrentPrefixReplay(cache: cache)
                return false
            }
        }
        guard foundRecurrentLayer else {
            clearRecurrentPrefixReplay(cache: cache)
            return false
        }

        for (i, layer) in layers.enumerated() where layer.isLinear {
            guard let mamba = cache[i] as? MambaCache,
                  let linear = layer.linearAttn,
                  linear.replayPrefix(
                    cache: mamba, committedRows: committedRows)
            else {
                clearRecurrentPrefixReplay(cache: cache)
                return false
            }
        }
        return true
    }

    /// Discard verify-only recurrent data without changing committed cache
    /// state. Used after full acceptance and by fallback/reset paths.
    func clearRecurrentPrefixReplay(cache: [KVCache?]) {
        for case let arrays as ArraysCache in cache.compactMap({ $0 }) {
            arrays.clearMTPTransientState()
        }
    }

    /// The DFlash 2 layer tap, held in a plain box so `Module` reflection
    /// classifies it as `.other` and never walks into it. A bare `MLXArray?`
    /// stored here would be discovered as a PARAMETER; `DFlash2TapSlot`
    /// documents why that matters.
    let dFlash2Tap = DFlash2TapSlot()

    /// Layers per early verify submission.
    static let verifySubmitEvery = 4

    /// Verify: submit after the first layer, so the GPU starts on the round
    /// while the host builds the rest, then after every `verifySubmitEvery`
    /// layers. Submission points change only when encoded work is
    /// committed; the graph and its kernels are the same.
    static func submitsVerify(after layerIndex: Int, of count: Int) -> Bool {
        layerIndex + 1 < count && (layerIndex == 0 || (layerIndex + 1) % verifySubmitEvery == 0)
    }

    func cbv2Forward(
        _ inputs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false,
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        precondition(
            caches.count == layers.filter({ !$0.isLinear }).count,
            "Qwen35 CBv2 requires only full-attention caches")
        let shapeCall = CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(liveBatchRows: inputs.dim(0), sequenceWidth: inputs.dim(1)) : nil
        defer { shapeCall?.end() }
        var hiddenStates = inputEmbeddings ?? embedTokens(inputs)
        // Read the tap ONCE. A nil list costs one comparison per layer and
        // allocates nothing; the drafter is not attached on a serial leg.
        let tapLayerIds = dFlash2Tap.layerIds
        let prefetchResidency = !captureRecurrentWindow && inputs.dim(1) > 1
            && CBv2ResidencyPrefetch.isArmed
        var tapped = [MLXArray?](
            repeating: nil, count: tapLayerIds?.count ?? 0)
        var attentionIndex = 0
        // Verify: carry each layer's output as (residual, delta) so the next
        // layer forms the add, its input norm and its codes in one kernel.
        let fuseBoundaries = captureRecurrentWindow && !exactTargetVerify
            && tapLayerIds == nil && !lastPositionOnly
            && hiddenStates.ndim == 3 && hiddenStates.dim(0) == 1
            && (2 ... 8).contains(hiddenStates.dim(1))
        var pendingDelta: MLXArray? = nil
        let pendingPath = Qwen35DecoderLayer.pendingResidualEnabled
            && !(captureRecurrentWindow && exactTargetVerify) && !fuseBoundaries
        var pending: MLXArray?
        var pendingTapSlot: Int?
        for (modelLayerIndex, layer) in layers.enumerated() {
            let attentionCache: (any CBv2AttendingLayerCache)?
            if layer.isLinear {
                attentionCache = nil
            } else {
                attentionCache = caches[attentionIndex]
                precondition(
                    attentionCache!.kind.modelLayerIndex == nil
                        || attentionCache!.kind.modelLayerIndex == modelLayerIndex,
                    "Qwen35 CBv2 attention cache mapped to the wrong model layer")
                attentionIndex += 1
            }
            let submitAfterLayer =
                (captureRecurrentWindow || !lastPositionOnly || inputs.dim(1) >= 64)
                && Self.submitsVerify(after: modelLayerIndex, of: layers.count)
            let prefetchAfterLayer = prefetchResidency
                && [4, 8, 16, 24].contains(modelLayerIndex + 1)
            let lastQueryOnly = lastPositionOnly && !captureRecurrentWindow
                && modelLayerIndex == layers.count - 1 && !layer.isLinear
                && positionIds == nil && hiddenStates.dim(1) > 1
                && !(tapLayerIds?.contains(modelLayerIndex) ?? false)
            if pendingPath {
                let out = layer.cbv2ForwardParts(
                    hiddenStates, pending: pending, modelLayerIndex: modelLayerIndex,
                    attentionCache: attentionCache, recurrentState: recurrentState,
                    positionIds: positionIds, captureRecurrentWindow: captureRecurrentWindow,
                    lastPositionOnly: lastPositionOnly && !captureRecurrentWindow
                        && modelLayerIndex == layers.count - 1
                        && !(tapLayerIds?.contains(modelLayerIndex) ?? false),
                    lastQueryOnly: lastQueryOnly)
                if let slot = pendingTapSlot { tapped[slot] = out.input }
                hiddenStates = out.h
                pending = out.f
                pendingTapSlot = tapLayerIds?.firstIndex(of: modelLayerIndex)
                if submitAfterLayer || prefetchAfterLayer {
                    asyncEval([out.h, out.f])
                }
                if prefetchAfterLayer {
                    CBv2ResidencyPrefetch.submitDue(completedLayers: modelLayerIndex + 1)
                }
                continue
            }
            if fuseBoundaries, let delta = pendingDelta,
                let fused = layer.cbv2ForwardVerifyFused(
                    hiddenStates, delta: delta, modelLayerIndex: modelLayerIndex,
                    attentionCache: attentionCache, recurrentState: recurrentState,
                    positionIds: positionIds)
            {
                (hiddenStates, pendingDelta) = (fused.residual, fused.delta)
                if Self.submitsVerify(after: modelLayerIndex, of: layers.count) {
                    asyncEval(hiddenStates, fused.delta)
                }
                continue
            }
            if let delta = pendingDelta {
                hiddenStates = hiddenStates + delta
                pendingDelta = nil
            }
            if fuseBoundaries, modelLayerIndex + 1 < layers.count,
                let split = layer.cbv2ForwardSplitOutput(
                    hiddenStates, modelLayerIndex: modelLayerIndex,
                    attentionCache: attentionCache, recurrentState: recurrentState,
                    positionIds: positionIds)
            {
                (hiddenStates, pendingDelta) = (split.residual, split.delta)
                if Self.submitsVerify(after: modelLayerIndex, of: layers.count) {
                    asyncEval(hiddenStates, split.delta)
                }
                continue
            }
            hiddenStates = layer.cbv2Forward(
                hiddenStates,
                modelLayerIndex: modelLayerIndex,
                attentionCache: attentionCache,
                recurrentState: recurrentState,
                positionIds: positionIds,
                captureRecurrentWindow: captureRecurrentWindow,
                exactTargetVerify: captureRecurrentWindow && exactTargetVerify,
                lastPositionOnly: lastPositionOnly && !captureRecurrentWindow
                    && modelLayerIndex == layers.count - 1
                    && !(tapLayerIds?.contains(modelLayerIndex) ?? false),
                lastQueryOnly: lastQueryOnly)
            // `hiddenStates` here IS the OUTPUT hidden state of this layer,
            // which is what the reference taps (`_LayerHook` wraps the layer and
            // keeps what it returned).
            if let tapLayerIds, let slot = tapLayerIds.firstIndex(of: modelLayerIndex) {
                tapped[slot] = hiddenStates
            }
            // Verify & Prefill: hand finished layers to the GPU while the rest
            // of the graph is still being built. Same graph and same kernels.
            // A prompt forward that keeps only its last position (the
            // stepper's timed prefill) is included: without it the GPU waits
            // until the host has built all layers.
            if submitAfterLayer || prefetchAfterLayer
            {
                asyncEval(hiddenStates)
            }
            if prefetchAfterLayer {
                CBv2ResidencyPrefetch.submitDue(completedLayers: modelLayerIndex + 1)
            }
        }
        if let delta = pendingDelta {
            hiddenStates = hiddenStates + delta
        }
        if let pending { hiddenStates = hiddenStates + pending }
        if let slot = pendingTapSlot { tapped[slot] = hiddenStates }
        if tapLayerIds == nil {
            dFlash2Tap.tappedHidden = nil
        } else {
            dFlash2Tap.tappedHidden = concatenated(tapped.map { $0! }, axis: -1)
        }
        return hiddenStates
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// The fused output-head top-two of the last captured verify forward.
    let verifyHeadTopTwo = Qwen35VerifyHeadTopTwoSlot()

    /// MTP head. Non-nil only when `_qwen35MTPEnabled == true` at init time
    /// AND `args.mtpNumHiddenLayers > 0`.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.__init__ (MTPModule attachment)
    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPModule?

    /// Checkpoint quantization policy staged by `loadWeights` (via
    /// `QuantizationPolicyReceiving`) before `sanitize` runs; drives the
    /// per-layer decision whether routed-expert gate/up halves may fuse.
    /// `nil` for unquantized checkpoints.
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        // Attach MTP head only when enabled and config declares MTP layers.
        // omlx: `if n_mtp > 0 and is_mtp_active(): self.mtp = q35.MTPModule(args)`
        if args.mtpNumHiddenLayers > 0 && _qwen35MTPEnabled {
            _mtp.wrappedValue = Qwen35MTPModule(args)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Inner model now returns pre-norm hidden; apply norm + lm_head here.
        // omlx: TextModel.__call__ (normed = self.model.norm(hidden); out = lm_head(normed))
        let hidden = model(inputs, cache: cache)
        var out = model.norm(hidden)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public var cbv2LayerKinds: [CBv2LayerKind] { configuration.cbv2LayerKinds }

    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        configuration.cbv2RecurrentStateSpec(
            activationDType: cbv2CheckpointActivationDType ?? model.embedTokens.weight.dtype)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { configuration.cbv2Capabilities }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { storageIndex, kind in
            try makeLayerCache(kind.modelLayerIndex ?? storageIndex, kind)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py TextModel.sanitize
        //
        // Key differences from stock mlx-lm:
        //  1. Gate norm shift on unsanitized conv1d shape ONLY (not on MTP key presence).
        //     Stock code uses `hasMTPWeights || hasUnsanitizedConv1d`, which double-shifts
        //     already-converted MLX checkpoints that have mtp.* keys.
        //  2. Keep mtp.* keys when the MTP head is attached; strip them otherwise.
        //  3. Extend norm-shift key set with MTP-specific norm names.

        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasUnsanitizedConv1d  // NOT hasMTPWeights

        var weights = weights

        // Routed experts are built fused (`SwitchGLU(fuseGateUp: true)`), and
        // the `qwen3_5_text` registry entry reaches this sanitizer directly —
        // without the MoE/VLM wrappers that call the fusion helper. Apply it
        // here so raw stacked `experts.gate_up_proj` exports and converted
        // split `switch_mlp.{gate,up}_proj.*` checkpoints both match the
        // module tree. Idempotent, so the wrapper paths that already fused
        // are unaffected; `mtp.*` keys stay split.
        if configuration.numExperts > 0 {
            weights = qwen35FuseSwitchMLPGateUp(
                weights: weights,
                perLayerQuantization: checkpointPerLayerQuantization,
                setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: self) })
        }

        // Keep mtp.* keys if the head is attached; strip them otherwise.
        // omlx: `if not hasattr(self, "mtp"): weights = {k:v if "mtp." not in k}`
        if mtp == nil {
            weights = weights.filter { !$0.key.contains("mtp.") }
        } else if !weights.keys.contains(where: { $0.contains("mtp.") }) {
            // MTP enabled but no mtp.* keys in checkpoint → needs re-conversion.
            // omlx: raises ValueError with "weights are missing the mtp.* tensors"
            print(
                "[WARNING] Qwen35TextModel.sanitize: MTP head is enabled but no mtp.* "
                + "weights found. Load will likely fail or produce garbage. "
                + "Re-convert the checkpoint with a converter that preserves MTP weights.")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Extended norm key set includes MTP-specific names.
        // omlx: norm_keys tuple with ".pre_fc_norm_hidden.weight" etc.
        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

extension Qwen35TextModel: CBv2PositionAxisProviding {
    public var cbv2PositionAxisCount: Int? { 3 }
}

extension Qwen35TextModel: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        // The unpositioned seam serves the stepper and plain decode. Both read
        // the final position only. Keep every cache write whole.
        positionedForward(
            tokens, inputEmbedding: nil, cache: caches,
            recurrentState: recurrentState, positionIds: nil, lastPositionOnly: true)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        positionedForward(
            tokens, inputEmbedding: nil, cache: caches,
            recurrentState: recurrentState, positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        model.embedTokens(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        preconditionFailure("Qwen35 embedding prefill requires request-owned recurrent state")
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        positionedForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }

    private func positionedForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        let caches = cache ?? []
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            lastPositionOnly: lastPositionOnly)
        let headInput = lastPositionOnly && hidden.dim(1) > 1
            ? hidden[0..., (hidden.dim(1) - 1)..., 0...] : hidden
        let normalized = model.norm(headInput)
        return lmHead.map { $0(normalized) } ?? model.embedTokens.asLinear(normalized)
    }
}

// MARK: - ContinuousBatchingV2 prompt-only output narrowing

/// CBv2 consumes only the final prompt position, so the prompt path skips
/// the [B, L, 248320] vocabulary projection for discarded rows: intermediate
/// chunks return a one-element hidden handle, the frontier chunk projects
/// exactly one row. Every K/V write and recurrent-state stage remains full.
/// Without explicit positions, the untapped final attention/MLP can consume
/// only the last row. MTP history and capture paths keep their full rows.
extension Qwen35TextModel: CBv2RecurrentLanguageModelPrefillForwardable {
    /// The Qwen trunk is shape-generic over `[B, L]` with one recurrent
    /// state row per batch row (`recurrentState.count == B` precondition),
    /// and CBv2 attention attends each packed row against its OWN KV — a
    /// packed row is semantically identical to running alone.
    public var cbv2SupportsPackedPrefill: Bool { true }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        let caches = cache ?? []
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            lastPositionOnly: positionIds == nil)
        switch requirement {
        case .evaluationOnly:
            // Small handle whose graph depends on the whole trunk — forcing
            // it commits every layer's K/V write and recurrent stage.
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            let last = model.norm(hidden[0..., -1, 0...])
            return lmHead.map { $0(last) } ?? model.embedTokens.asLinear(last)
        }
    }
}

extension Qwen35TextModel: CBv2RecurrentMTPForwardable {
    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 MTP target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds)
        let normalized = model.norm(hidden)
        let logits = lmHead.map { $0(normalized) } ?? model.embedTokens.asLinear(normalized)
        return (logits, hidden)
    }
}

extension Qwen35TextModel: CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let attending = caches.map { $0 as! any CBv2AttendingLayerCache }
        // The block drafter consumes taps; its prompt head uses one row.
        let hidden = model.cbv2Forward(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            lastPositionOnly: model.dFlash2Tap.layerIds != nil)
        switch requirement {
        case .evaluationOnly:
            return (hidden[0..., (hidden.dim(1) - 1)..., 0 ..< 1], hidden)
        case .lastPositionLogits:
            let last = model.norm(hidden[0..., (hidden.dim(1) - 1)..., 0...])
            let logits = lmHead.map { $0(last) } ?? model.embedTokens.asLinear(last)
            return (logits, hidden)
        }
    }
}

extension Qwen35TextModel: CBv2RecurrentCaptureMTPForwardable {
    /// MTP capture-verify: identical to `cbv2ForwardWithHidden` except each
    /// GatedDeltaNet layer stages per-position captured conv/SSM stacks so
    /// finalize can commit the accepted position on device. Attention and
    /// MoE/projection layers are stateless over the window and run batched.
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 MTP target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: true)
        let normalized = model.norm(hidden)
        let logits: MLXArray
        verifyHeadTopTwo.value = nil
        if let lmHead {
            logits = model.exactTargetVerify
                ? qwen35A3BExactW4G64Projection(lmHead, normalized)
                : lmHead(normalized)
            if !model.exactTargetVerify, let head = lmHead as? HadamardQuantizedLinear {
                verifyHeadTopTwo.value = head.topTwoRows(normalized)
            }
        } else if model.exactTargetVerify, normalized.dim(1) > 1 {
            logits = qwen35A3BTimewiseProjection(normalized) {
                model.embedTokens.asLinear($0)
            }
        } else {
            logits = model.embedTokens.asLinear(normalized)
        }
        return (logits, hidden)
    }
}

// MARK: - Qwen35TextModel + DFlash 2

/// The target surface the DFlash 2 block drafter binds to, and the layer tap
/// the engine reads between rounds.
///
/// BOTH SHARED TENSORS GO THROUGH THE MODULE. On this pack `embed_tokens` and
/// `lm_head` are packed Hadamard modules: `HadamardQuantizedLinear` IS a
/// `QuantizedLinear` and `HadamardQuantizedEmbedding` is NOT a
/// `QuantizedEmbedding`, so a path that casts and reads `.weight` type-checks,
/// runs, and returns numbers with the folded transform left unapplied. The two
/// calls below are plain module calls for exactly that reason; see
/// `Qwen35InlineMTPAssistant.headLogits` and `docs/bonsai2-27b-port-notes.md`
/// section 6.1.
extension Qwen35TextModel: DFlash2TapTarget {
    public var dFlash2VocabularySize: Int { vocabularySize }
    public var dFlash2HiddenSize: Int { configuration.hiddenSize }
    public var dFlash2LayerCount: Int { configuration.hiddenLayers }

    public var dFlash2TapLayerIds: [Int]? {
        get { model.dFlash2Tap.layerIds }
        set { model.dFlash2Tap.layerIds = newValue }
    }

    public var dFlash2TappedHidden: MLXArray? { model.dFlash2Tap.tappedHidden }

    public func embedTokensForDFlash2(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func logitsForDFlash2Hidden(_ hidden: MLXArray) -> MLXArray {
        lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }
}

extension Qwen35TextModel: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        precondition(logits.ndim == 3, "Qwen35 MTP policy logits must be [B,L,V]")
        let batch = logits.dim(0)
        let length = logits.dim(1)
        let vocabularySize = logits.dim(2)
        precondition(batch > 0 && length > 0 && vocabularySize >= 2)
        let rows = batch * length
        let topTwo = qwen35MTPTopTwoRows(
            logits.reshaped([1, rows, vocabularySize]))
        return (
            topTwo.ids.reshaped([batch, length, 2]),
            topTwo.values.reshaped([batch, length, 2]))
    }

    public func cbv2MTPTakeVerifyHeadTopTwo() -> (ids: MLXArray, values: MLXArray)? {
        defer { verifyHeadTopTwo.value = nil }
        return verifyHeadTopTwo.value
    }
}

// MARK: - Qwen35TextModel + MTPCapable

extension Qwen35TextModel: MTPCapable {
    public var hasMTPHead: Bool { mtp != nil }

    /// Run a backbone forward that also returns pre-final-norm hidden states.
    ///
    /// Returns `(logits, preNormHidden)` where `preNormHidden` is the raw
    /// backbone output before `model.norm`. Qwen MTP applies this exact final
    /// norm once at the target-to-assistant boundary, followed by the head's
    /// separately trained `pre_fc_norm_hidden`; recursive assistant hidden
    /// bypasses the target final norm.
    ///
    /// PR #990: `return out, hidden  # pre-norm hidden for MTP head`
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.__call__ with return_hidden=True
    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        let hidden = model(input.tokens, cache: cacheOpt, nConfirmed: nConfirmed)
        let normed = model.norm(hidden)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(normed)
        } else {
            logits = model.embedTokens.asLinear(normed)
        }
        // Return raw capture hidden. The bound Qwen MTP adapter owns the exact
        // target final-norm application when these rows enter trusted history.
        return (logits, hidden)
    }

    /// Commit a shorter accepted recurrent prefix from the most recent wide
    /// verify. On false, transient rollback data is cleared so a fallback
    /// repair cannot accidentally reuse an incompatible tape.
    public func replayRecurrentPrefix(
        cache: [any KVCache], committedRows: Int
    ) -> Bool {
        model.replayRecurrentPrefix(
            cache: cache.map { Optional($0) },
            committedRows: committedRows)
    }

    /// Release a verify tape after full acceptance or an external fallback.
    /// Persistent recurrent state and attention offsets are unchanged.
    public func clearRecurrentPrefixReplay(cache: [any KVCache]) {
        model.clearRecurrentPrefixReplay(cache: cache.map { Optional($0) })
    }

    /// Run the MTP head forward.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.mtp_forward
    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        guard let mtp else {
            fatalError("mtpForward called but MTP head is not attached. "
                + "Set _qwen35MTPEnabled = true before loading the model.")
        }
        let mtpOut = mtp(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: model.embedTokens,
            cache: cache)
        if configuration.tieWordEmbeddings {
            return model.embedTokens.asLinear(mtpOut)
        }
        return lmHead!(mtpOut)
    }

    /// Allocate a fresh KV cache for the MTP head layers.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.make_mtp_cache
    public func makeMTPCache() -> [any KVCache] {
        guard let mtp else { return [] }
        return mtp.layers.map { _ in KVCacheSimple() as any KVCache }
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public var cbv2LayerKinds: [CBv2LayerKind] { languageModel.cbv2LayerKinds }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        languageModel.cbv2RecurrentStateSpec
    }
    public var cbv2Capabilities: CBv2ModelCapabilities { languageModel.cbv2Capabilities }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try languageModel.newCacheV2(makeLayerCache: makeLayerCache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        languageModel.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { languageModel.supportsVisionSpanPrefill }
    public var supportsCausalVisionPrefill: Bool { languageModel.supportsCausalVisionPrefill }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        languageModel.scaledInputEmbeddings(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        languageModel.embeddingForward(inputs, inputEmbedding: inputEmbedding, cache: cache)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.embeddingForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }
}

extension Qwen35Model: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool {
        languageModel.cbv2SupportsPackedPrefill
    }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        languageModel.cbv2RecurrentPrefill(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds,
            requirement: requirement)
    }
}

extension Qwen35Model: CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenForPrefill(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, requirement: requirement)
    }
}

extension Qwen35Model: CBv2RecurrentMTPForwardable {
    public var cbv2MTPTargetIdentity: ObjectIdentifier {
        ObjectIdentifier(languageModel)
    }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHidden(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }
}

extension Qwen35Model: CBv2RecurrentCaptureMTPForwardable {
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenCaptured(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }
}

extension Qwen35Model: DFlash2TapTarget {
    public var dFlash2VocabularySize: Int { languageModel.dFlash2VocabularySize }
    public var dFlash2HiddenSize: Int { languageModel.dFlash2HiddenSize }
    public var dFlash2LayerCount: Int { languageModel.dFlash2LayerCount }

    public var dFlash2TapLayerIds: [Int]? {
        get { languageModel.dFlash2TapLayerIds }
        set { languageModel.dFlash2TapLayerIds = newValue }
    }

    public var dFlash2TappedHidden: MLXArray? { languageModel.dFlash2TappedHidden }

    public func embedTokensForDFlash2(_ tokens: MLXArray) -> MLXArray {
        languageModel.embedTokensForDFlash2(tokens)
    }

    public func logitsForDFlash2Hidden(_ hidden: MLXArray) -> MLXArray {
        languageModel.logitsForDFlash2Hidden(hidden)
    }
}

extension Qwen35Model: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        languageModel.cbv2MTPTopTwo(logits)
    }

    public func cbv2MTPTakeVerifyHeadTopTwo() -> (ids: MLXArray, values: MLXArray)? {
        languageModel.cbv2MTPTakeVerifyHeadTopTwo()
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Qwen35Model + MTPCapable

/// VLM-outer-wrapper pass-through for MTPCapable.
/// Forwards all MTP calls to the inner `languageModel` (a Qwen35TextModel).
/// omlx: patches/mlx_lm_mtp/qwen35_model.py `_patch_outer_model`
extension Qwen35Model: MTPCapable {
    public var hasMTPHead: Bool { languageModel.hasMTPHead }

    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        languageModel.callWithHidden(input: input, cache: cache, nConfirmed: nConfirmed)
    }

    public func replayRecurrentPrefix(
        cache: [any KVCache], committedRows: Int
    ) -> Bool {
        languageModel.replayRecurrentPrefix(
            cache: cache, committedRows: committedRows)
    }

    public func clearRecurrentPrefixReplay(cache: [any KVCache]) {
        languageModel.clearRecurrentPrefixReplay(cache: cache)
    }

    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        languageModel.mtpForward(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    public func makeMTPCache() -> [any KVCache] {
        languageModel.makeMTPCache()
    }
}
