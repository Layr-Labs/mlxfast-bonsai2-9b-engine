//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Dispatch
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

/// The dense FP32 `in_proj_b` and `in_proj_a` of a few-row window (MTP
/// verify) in one launch. Each simdgroup owns one output column and reads
/// that weight row once for all rows; FP32 FMAs, then `simd_sum`. Single-row
/// decode and wide prefill keep the stock matmul.
private enum Qwen35DenseGateProjection {
    static let maximumRows = 16

    /// The eligibility `apply` checks; the rows of `x` when it holds.
    static func eligibleRows(_ x: MLXArray, b: Linear, a: Linear) -> Int? {
        guard !(b is QuantizedLinear), !(a is QuantizedLinear), b.bias == nil, a.bias == nil,
            x.dtype == .float32, b.weight.dtype == .float32, a.weight.dtype == .float32,
            b.weight.ndim == 2, b.weight.shape == a.weight.shape, x.ndim == 3,
            b.weight.dim(1) == x.dim(2), x.dim(2) % 4 == 0,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.dim(0) * x.dim(1)
        return (2 ... maximumRows).contains(rows) ? rows : nil
    }

    static func apply(_ x: MLXArray, b: Linear, a: Linear) -> (MLXArray, MLXArray)? {
        guard let rows = eligibleRows(x, b: b, a: a) else { return nil }
        let N = b.weight.dim(0)
        let outputs = VerifyDenseBA.standalone(
            x.reshaped([rows, x.dim(2)]), wb: b.weight, wa: a.weight)
        let shape = [x.dim(0), x.dim(1), N]
        return (outputs.0.reshaped(shape), outputs.1.reshaped(shape))
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
            conv_in[t * C + c] = t < 3u ? state[t * C + c] : qkv[(t - 3u) * C + c];
            if (t >= 3u) {
              const uint p = t - 3u;
              float acc = 0.0f;
              for (uint tap = 0; tap < 4u; ++tap) {
                const uint r = p + tap;
                const float v = r < 3u ? state[r * C + c] : qkv[(r - 3u) * C + c];
                acc = fma(v, conv_weight[c * 4u + tap], acc);
              }
              const float e = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
              const float sig = acc < 0.0f ? e : 1.0f - e;
              conv_out[p * C + c] = acc * sig;
            }
            """,
        ensureRowContiguous: true)

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

private enum Qwen35RegisterGDN {
    private static func source(valueRead: String) -> String {
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
        for (int t = 0; t < T; ++t) {
          const float gt = g_[0];
          const float bt = beta_[0];
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ((const device float4*)k_)[j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            const float4 q4 = ((const device float4*)q_)[j];
            qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
          }
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
          q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;
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
        source: source(valueRead: "vt[d] = v_[d];"))

    private static let valueKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_value_conv",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "conv_weight"],
        outputNames: ["y", "state_out"],
        source: source(valueRead: """
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
            """))

    /// The prompt recurrence of `valueKernel` with the depthwise conv + SiLU
    /// value computed once per channel and step: every 32 steps the
    /// threadgroup fills a 32-step x 32-channel tile in parallel (each value
    /// by one thread, the same fma chain, exp and divide as `valueKernel`),
    /// instead of 8 lanes recomputing each value inside the serial step.
    /// The recurrence itself is `valueKernel`'s, op for op, so y and the
    /// state are bit-identical (custom kernels compile in Safe math mode).
    private static let stagedValueKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_value_conv_staged",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "conv_weight"],
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
        constexpr int TS = 32;
        threadgroup float vts[TS][32];
        const uint ltid = sg * 32u + lane;
        const uint vloc = sg * DVPS + (lane / LPD) * DVPL;
        const uint ch_tg = 2u * Hk * Dk + hv_idx * Dv + threadgroup_position_in_grid.y * DVPT;
        const uint v_stride = v_shape[2];
        for (int t = 0; t < T; ++t) {
          const int s = t % TS;
          if (s == 0) {
            const int tn = min(TS, int(T) - t);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = ltid; idx < uint(tn) * 32u; idx += 128u) {
              const uint ss = idx / 32u, c = idx % 32u;
              const uint channel = ch_tg + c;
              const float4 cw4 = *((const device float4*)(conv_weight + channel * 4u));
              const uint v_base = (b_idx * (T + 3u) + uint(t) + ss) * v_stride + channel;
              float cv = fma(v[v_base], cw4.x, 0.0f);
              cv = fma(v[v_base + v_stride], cw4.y, cv);
              cv = fma(v[v_base + 2u * v_stride], cw4.z, cv);
              cv = fma(v[v_base + 3u * v_stride], cw4.w, cv);
              const float sig = 1.0f / (1.0f + metal::exp(metal::abs(cv)));
              vts[ss][c] = cv * (cv < 0.0f ? sig : 1.0f - sig);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          const float gt = g_[0];
          const float bt = beta_[0];
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ((const device float4*)k_)[j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            const float4 q4 = ((const device float4*)q_)[j];
            qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
          }
          float kv[DVPL];
          float vt[DVPL];
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            vt[d] = vts[s][vloc + d];


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
          q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          device float4* s_out4 = (device float4*)(state_out + (n * Dv + dvbase + d) * Dk + dk0);
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            s_out4[j] = float4(state[d][4 * j + 0], state[d][4 * j + 1], state[d][4 * j + 2], state[d][4 * j + 3]);
          }
        }
        """)

    /// v2 of `stagedValueKernel`: every 16 steps the threadgroup also copies
    /// the k rows of the tile (16 x 128 floats) and the step's g and beta into
    /// threadgroup memory, so the serial step reads them from threadgroup
    /// memory instead of 16 threads (k) or 128 threads (g, beta) each loading
    /// the same device address. Loads only: the values and every arithmetic
    /// op are `valueKernel`'s, so y and the state are bit-identical.
    /// 10,368 bytes of threadgroup memory (v 2 KiB, k 8 KiB, g/beta 128 B).
    private static let stagedV2ValueKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_value_conv_staged_v2",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "conv_weight"],
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
        constexpr int TS = 16;
        threadgroup float vts[TS][32];
        threadgroup float4 kts[TS][Dk / 4];
        threadgroup float gts[TS];
        threadgroup float bts[TS];
        const uint ltid = sg * 32u + lane;
        const uint vloc = sg * DVPS + (lane / LPD) * DVPL;
        const uint ch_tg = 2u * Hk * Dk + hv_idx * Dv + threadgroup_position_in_grid.y * DVPT;
        const uint v_stride = v_shape[2];
        const device float4* k_tile4 = (const device float4*)(k + (b_idx * T * Hk + hk_idx) * Dk);
        for (int t = 0; t < T; ++t) {
          const int s = t % TS;
          if (s == 0) {
            const int tn = min(TS, int(T) - t);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = ltid; idx < uint(tn) * 32u; idx += 128u) {
              const uint ss = idx / 32u, c = idx % 32u;
              const uint channel = ch_tg + c;
              const float4 cw4 = *((const device float4*)(conv_weight + channel * 4u));
              const uint v_base = (b_idx * (T + 3u) + uint(t) + ss) * v_stride + channel;
              float cv = fma(v[v_base], cw4.x, 0.0f);
              cv = fma(v[v_base + v_stride], cw4.y, cv);
              cv = fma(v[v_base + 2u * v_stride], cw4.z, cv);
              cv = fma(v[v_base + 3u * v_stride], cw4.w, cv);
              const float sig = 1.0f / (1.0f + metal::exp(metal::abs(cv)));
              vts[ss][c] = cv * (cv < 0.0f ? sig : 1.0f - sig);
              kts[ss][c] = k_tile4[(uint(t) + ss) * (Hk * Dk / 4) + c];
            }
            if (ltid < uint(tn)) {
              gts[ltid] = g_[ltid * Hv];
              bts[ltid] = beta_[ltid * Hv];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          const float gt = gts[s];
          const float bt = bts[s];
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const uint kj = (lane % LPD) * 4u + uint(j);
            const float4 k4 = kts[s][kj];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            const float4 q4 = ((const device float4*)q_)[j];
            qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
          }
          float kv[DVPL];
          float vt[DVPL];
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            vt[d] = vts[s][vloc + d];
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
          q_ += Hk * Dk; y_ += Hv * Dv;
          if (s == TS - 1) { g_ += TS * Hv; beta_ += TS * Hv; }
        }
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          device float4* s_out4 = (device float4*)(state_out + (n * Dv + dvbase + d) * Dk + dk0);
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            s_out4[j] = float4(state[d][4 * j + 0], state[d][4 * j + 1], state[d][4 * j + 2], state[d][4 * j + 3]);
          }
        }
        """)

    private static func dispatchValue(
        _ k: MLXFast.MLXFastKernel, q: MLXArray, kk: MLXArray, input: MLXArray,
        weight: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let B = q.dim(0), T = q.dim(1), Hk = q.dim(2), Dk = q.dim(3), Hv = g.dim(2)
        let result = k(
            [q, kk, input, g, beta, state, weight],
            template: [("Dk", Dk), ("Dv", 128), ("Hk", Hk), ("Hv", Hv)],
            grid: (128, 4, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [[B, T, Hv, 128], state.shape], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }

    /// Load-time check, once per process per shape and kernel: a staged kernel
    /// must reproduce `valueKernel`'s y and state bit for bit on a synthetic
    /// input of the served shape (first call is the unmeasured warm prefill).
    /// Any mismatch keeps the next kernel down (v2 -> v1 -> `valueKernel`).
    /// `MLXFAST_GDN_PIPELINED=0` disables both staged kernels;
    /// `MLXFAST_GDN_STAGED_V2=0` disables v2 only.
    nonisolated(unsafe) private static var pipelinedVerdict: [String: Bool] = [:]
    private static let pipelinedLock = NSLock()
    private static func envEnabled(_ name: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }
    private static let pipelinedEnabled: Bool = envEnabled("MLXFAST_GDN_PIPELINED")
    private static let stagedV2Enabled: Bool = envEnabled("MLXFAST_GDN_STAGED_V2")

    private static func stagedMatches(
        _ candidate: MLXFast.MLXFastKernel, tag: String, Hk: Int, Hv: Int, convDim: Int
    ) -> Bool {
        let key = "\(tag):\(Hk)x\(Hv)x\(convDim)"
        pipelinedLock.lock()
        defer { pipelinedLock.unlock() }
        if let verdict = pipelinedVerdict[key] { return verdict }
        let T = 67
        let key0 = MLXRandom.key(0x6d6e)
        let parts = MLXRandom.split(key: key0, into: 7)
        let q = MLXRandom.uniform(low: -0.2, high: 0.2, [1, T, Hk, 128], key: parts[0])
        let kk = MLXRandom.uniform(low: -0.2, high: 0.2, [1, T, Hk, 128], key: parts[1])
        let input = MLXRandom.uniform(low: -3, high: 3, [1, T + 3, convDim], key: parts[2])
        let weight = MLXRandom.uniform(low: -0.7, high: 0.7, [convDim, 4, 1], key: parts[3])
        let g = MLXRandom.uniform(low: 0.6, high: 1.0, [1, T, Hv], key: parts[4])
        let beta = MLXRandom.uniform(low: 0.0, high: 1.0, [1, T, Hv], key: parts[5])
        let state = MLXRandom.uniform(low: -1, high: 1, [1, Hv, 128, 128], key: parts[6])
        let (y0, s0) = dispatchValue(
            valueKernel, q: q, kk: kk, input: input, weight: weight, g: g, beta: beta, state: state)
        let (y1, s1) = dispatchValue(
            candidate, q: q, kk: kk, input: input, weight: weight, g: g, beta: beta,
            state: state)
        let same = MLX.all(y0.view(dtype: .uint32) .== y1.view(dtype: .uint32))
            .&& MLX.all(s0.view(dtype: .uint32) .== s1.view(dtype: .uint32))
        let verdict = same.item(Bool.self)
        pipelinedVerdict[key] = verdict
        if !verdict {
            FileHandle.standardError.write(
                Data("qwen35 gdn: \(tag) prompt recurrence mismatch, falling back\n".utf8))
        }
        return verdict
    }

    private static func promptKernel(Hk: Int, Hv: Int, convDim: Int) -> MLXFast.MLXFastKernel {
        guard pipelinedEnabled else { return valueKernel }
        if stagedV2Enabled
            && stagedMatches(stagedV2ValueKernel, tag: "staged_v2", Hk: Hk, Hv: Hv, convDim: convDim)
        {
            return stagedV2ValueKernel
        }
        return stagedMatches(stagedValueKernel, tag: "staged", Hk: Hk, Hv: Hv, convDim: convDim)
            ? stagedValueKernel : valueKernel
    }


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
        let kernel = promptKernel(Hk: q.dim(2), Hv: g.dim(2), convDim: input.dim(2))
        return dispatchValue(
            kernel, q: q, kk: k, input: input, weight: weight, g: g, beta: beta, state: state)
    }

}

/// Prompt-width GDN convolution, SiLU and q/k RMSNorm in one kernel: the
/// depthwise `conv1d` over the retained three-row state followed by the new
/// QKV rows, read in place (no concatenation), with the arithmetic of
/// `Qwen35FusedConvSilu` (taps in order, one FP32 multiply-add each from
/// zero; MLX's stable `silu`), then for each q/k head the RMSNorm and scale
/// of `Qwen35FusedQKNorm`. Value channels are written as SiLU(conv), the
/// `v` the register recurrence reads. Each simdgroup forms one 128-channel
/// head of one row; outputs are the bits of the concatenate, `conv1d`,
/// `silu` and q/k-norm graph.
private enum Qwen35PromptConvQKNorm {
    private static let source = """
            #pragma clang fp reassociate(off)
            const uint groups = uint(2 * HK + HV);
            const uint t0 = 2u * (threadgroup_position_in_grid.y / groups);
            const uint T = qkv_shape[1];
            const uint gi = threadgroup_position_in_grid.y % groups;
            const uint lane = thread_index_in_simdgroup;
            const uint C = uint(CONV_DIM);
            const uint c0 = gi * 128u + lane * 4u;
            float vals[2][4];
            for (int i = 0; i < 4; ++i) {
              const uint c = c0 + uint(i);
              const float4 wt = ((const device float4*)conv_weight)[c];
              float xv[5];
              #pragma clang loop unroll(full)
              for (uint j = 0; j < 5u; ++j) {
                const uint rr = t0 + j;
                xv[j] = rr < 3u ? state[rr * C + c] :
                    (rr < T + 3u ? qkv[(rr - 3u) * C + c] : 0.0f);
              }
              float ac0 = 0.0f, ac1 = 0.0f;
              #pragma clang loop unroll(full)
              for (uint tap = 0; tap < 4u; ++tap) {
                ac0 = fma(xv[tap], wt[tap], ac0);
                ac1 = fma(xv[tap + 1u], wt[tap], ac1);
              }
              const float e0 = 1.0f / (1.0f + metal::exp(metal::abs(ac0)));
              const float e1 = 1.0f / (1.0f + metal::exp(metal::abs(ac1)));
              vals[0][i] = ac0 * (ac0 < 0.0f ? e0 : 1.0f - e0);
              vals[1][i] = ac1 * (ac1 < 0.0f ? e1 : 1.0f - e1);
            }
            for (uint dt = 0; dt < 2u; ++dt) {
              const uint t = t0 + dt;
              if (t >= T) break;
              const float val[4] = {vals[dt][0], vals[dt][1], vals[dt][2], vals[dt][3]};
            if (gi >= uint(2 * HK)) {
              const uint o = (t * uint(HV) + (gi - uint(2 * HK))) * 128u + lane * 4u;
              ((device float4*)(v_out + o))[0] = float4(val[0], val[1], val[2], val[3]);
              continue;
            }
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
              sum += val[i] * val[i];
            }
            const float eps = as_type<float>(uint(EPS_BITS));
            const float inv = metal::precise::rsqrt(simd_sum(sum) / 128.0f + eps);
            const bool is_q = gi < uint(HK);
            const float scale = as_type<float>(uint(is_q ? Q_SCALE_BITS : K_SCALE_BITS));
            const uint o = (t * uint(HK) + (is_q ? gi : gi - uint(HK))) * 128u + lane * 4u;
            ((device float4*)((is_q ? q_out : k_out) + o))[0] = float4(
              scale * (val[0] * inv),
              scale * (val[1] * inv),
              scale * (val[2] * inv),
              scale * (val[3] * inv)
            );
            }
            """

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prompt_conv_qk",
        inputNames: ["state", "qkv", "conv_weight"], outputNames: ["q_out", "k_out", "v_out"],
        source: source, ensureRowContiguous: true)

    private static let qkOnlyKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prompt_conv_qk_only",
        inputNames: ["state", "qkv", "conv_weight"], outputNames: ["q_out", "k_out"],
        source: source.replacingOccurrences(
            of: "const uint groups = uint(2 * HK + HV);", with: "const uint groups = uint(2 * HK);")
            .replacingOccurrences(of: """
            if (gi >= uint(2 * HK)) {
              const uint o = (t * uint(HV) + (gi - uint(2 * HK))) * 128u + lane * 4u;
              ((device float4*)(v_out + o))[0] = float4(val[0], val[1], val[2], val[3]);
              continue;
            }

            """, with: ""),
        ensureRowContiguous: true)

    static func apply(
        state: MLXArray, qkv: MLXArray, weight: MLXArray, keyHeads: Int, valueHeads: Int,
        materializeValue: Bool = true
    ) -> [MLXArray]? {
        let C = qkv.dim(-1)
        guard state.ndim == 3, state.dim(0) == 1, state.dim(1) == 3, qkv.ndim == 3, qkv.dim(0) == 1,
            qkv.dim(1) >= 3, C == state.dim(2), C == (2 * keyHeads + valueHeads) * 128,
            state.dtype == .float32, qkv.dtype == .float32, weight.dtype == .float32,
            weight.shape == [C, 4, 1], qkv.size <= Int(Int32.max),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let S = qkv.dim(1)
        let invScale = pow(Float(128), -0.5)
        return (materializeValue ? kernel : qkOnlyKernel)(
            [state, qkv, weight],
            template: [("HK", keyHeads), ("HV", valueHeads), ("CONV_DIM", C),
                ("EPS_BITS", Int(Float(1e-6).bitPattern)),
                ("Q_SCALE_BITS", Int(pow(invScale, 2).bitPattern)),
                ("K_SCALE_BITS", Int(invScale.bitPattern))],
            grid: (32, ((S + 1) / 2) * (2 * keyHeads + (materializeValue ? valueHeads : 0)), 1), threadGroup: (32, 1, 1),
            outputShapes: [[1, S, keyHeads, 128], [1, S, keyHeads, 128]]
                + (materializeValue ? [[1, S, valueHeads, 128]] : []),
            outputDTypes: Array(repeating: .float32, count: materializeValue ? 3 : 2))
    }
}

/// Fresh B1 prompt scan on the current q/k/V and stacked BA producers.
/// The FP32 matrix operations use the checked order for 8 rows. It differs
/// from the serial fallback; adoption requires the unchanged 2e-5 state gate
/// and the target-token gate. Retained and verify states use the fallback.
private enum Qwen35Fresh8PromptGDN {
    static func run(
        q: MLXArray, k: MLXArray, v: MLXArray, ba: MLXArray,
        aLog: MLXArray, dtBias: MLXArray
    ) -> (MLXArray, MLXArray)? {
        guard q.ndim == 4, q.shape == k.shape, q.dim(0) == 1,
            (64 ... 1365).contains(q.dim(1)), q.dim(1) % 8 == 0,
            q.dim(2) == 16, q.dim(3) == 128,
            v.shape == [1, q.dim(1), 32, 128], ba.shape == [1, q.dim(1), 64],
            aLog.size == 32, dtBias.size == 32,
            q.dtype == .float32, k.dtype == .float32, v.dtype == .float32,
            ba.dtype == .float32, aLog.dtype == .float32, dtBias.dtype == .float32,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let T = q.dim(1)
        let q = q.contiguous(), k = k.contiguous(), v = v.contiguous(), ba = ba.contiguous()
        let rows = MLXArray(Int32(T))
        let prepared = prepKernel(
            [q, k, ba, aLog, dtBias, rows],
            template: [("C", 8), ("Dk", 128), ("Hk", 16), ("Hv", 32)],
            grid: (32, T / 8, 16), threadGroup: (32, 1, 1),
            outputShapes: [[1, 32, T / 8, 8, 8], [1, 32, T / 8, 8, 8], [1, 32, T / 8, 2, 8]],
            outputDTypes: [.float32, .float32, .float32])
        let result = scanKernel(
            [q, k, v, prepared[0], prepared[1], prepared[2], rows],
            template: [("C", 8), ("Dk", 128), ("Dv", 128), ("Hk", 16), ("Hv", 32), ("NS", scanNS)],
            grid: (32, 16, 32), threadGroup: (32, scanNS, 1),
            outputShapes: [[1, T, 32, 128], [1, 32, 128, 128]], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }

    /// The QK-only producer and this scan share contiguous convolution inputs.
    /// Each V element is computed once at its sole 8x8 scan-tile load.
    static func runConvolvedValue(
        q: MLXArray, k: MLXArray, convState: MLXArray, qkv: MLXArray, weight: MLXArray,
        ba: MLXArray, aLog: MLXArray, dtBias: MLXArray
    ) -> (MLXArray, MLXArray)? {
        guard q.ndim == 4, q.shape == k.shape, q.dim(0) == 1,
            (64 ... 1365).contains(q.dim(1)), q.dim(1) % 8 == 0,
            q.dim(2) == 16, q.dim(3) == 128,
            convState.shape == [1, 3, 8192], qkv.shape == [1, q.dim(1), 8192],
            weight.shape == [8192, 4, 1], ba.shape == [1, q.dim(1), 64],
            aLog.size == 32, dtBias.size == 32,
            q.dtype == .float32, k.dtype == .float32, convState.dtype == .float32,
            qkv.dtype == .float32, weight.dtype == .float32, ba.dtype == .float32,
            aLog.dtype == .float32, dtBias.dtype == .float32,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let T = q.dim(1)
        let q = q.contiguous(), k = k.contiguous(), ba = ba.contiguous()
        let rows = MLXArray(Int32(T))
        let prepared = prepKernel(
            [q, k, ba, aLog, dtBias, rows],
            template: [("C", 8), ("Dk", 128), ("Hk", 16), ("Hv", 32)],
            grid: (32, T / 8, 16), threadGroup: (32, 1, 1),
            outputShapes: [[1, 32, T / 8, 8, 8], [1, 32, T / 8, 8, 8], [1, 32, T / 8, 2, 8]],
            outputDTypes: [.float32, .float32, .float32])
        let result = convolvedValueScanKernel(
            [q, k, convState, qkv, weight, prepared[0], prepared[1], prepared[2], rows],
            template: [("C", 8), ("Dk", 128), ("Dv", 128), ("Hk", 16), ("Hv", 32), ("NS", scanNS)],
            grid: (32, 16, 32), threadGroup: (32, scanNS, 1),
            outputShapes: [[1, T, 32, 128], [1, 32, 128, 128]], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }

    private static let prepKernel = MLXFast.metalKernel(
        name: "qwen35_fresh8_prompt_prep",
        inputNames: ["q", "k", "ba", "a_log", "dt_bias", "T"], outputNames: ["tp", "pm", "gf"],
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
            const uint gate_row = ((uint)b_idx * T_ + t0 + row) * (2u * Hv);
            const float negative_a = -metal::precise::exp(a_log[hv]);
            const float gv = qwen35_simd_gate_decay(
                ba[gate_row + Hv + hv], dt_bias[hv], negative_a);
            zero = !(gv >= FLT_MIN);
            gam = zero ? 0.0f : metal::precise::log(zero ? 1.0f : gv);
            bet = qwen35_simd_gate_beta(ba[gate_row + hv]);
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
        """, header: Qwen35RegisterGatedSIMD.gateHeader)

    private static let scanSource = """
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
        
        const device float* vbase = v + (uint)b_idx * T_ * vs + hv * Dv + r0;
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
          const device float* v_ = vbase + (uint)t0 * vs;
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
            const float2 vv = *(const device float2*)(v_ + row * vs + fn);
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
        """

    private static let scanKernel = MLXFast.metalKernel(
        name: "qwen35_fresh8_prompt_scan",
        inputNames: ["q", "k", "v", "tp", "pm", "gf", "T"], outputNames: ["y", "state_out"],
        source: scanSource, ensureRowContiguous: true)

    private static let convolvedValueScanKernel = MLXFast.metalKernel(
        name: "qwen35_fresh8_prompt_scan_vfold",
        inputNames: ["q", "k", "conv_state", "qkv", "w", "tp", "pm", "gf", "T"],
        outputNames: ["y", "state_out"],
        source: convolvedValueScanSource, ensureRowContiguous: true)

    private static let convolvedValueScanSource = scanSource
        .replacingOccurrences(
            of: "const device float* vbase = v + (uint)b_idx * T_ * vs + hv * Dv + r0;",
            with: """
            constexpr uint CD = 2 * Hk * Dk + Hv * Dv;
            const uint vcol0 = 2 * Hk * Dk + hv * Dv + r0 + fn;
            const float4 vw0 = *(const device float4*)(w + vcol0 * 4u);
            const float4 vw1 = *(const device float4*)(w + (vcol0 + 1u) * 4u);
            auto vconv = [&](int t, uint col, float4 wt) -> float {
              #pragma clang fp reassociate(off)
              float acc = 0.0f;
              for (uint tap = 0; tap < 4u; ++tap) {
                const uint r = uint(t) + tap;
                const float xv = r < 3u ? conv_state[r * CD + col] : qkv[(r - 3u) * CD + col];
                acc = fma(xv, wt[tap], acc);
              }
              const float e = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
              const float sig = acc < 0.0f ? e : 1.0f - e;
              return acc * sig;
            };
            """)
        .replacingOccurrences(of: "const device float* v_ = vbase + (uint)t0 * vs;", with: "")
        .replacingOccurrences(
            of: "const float2 vv = *(const device float2*)(v_ + row * vs + fn);",
            with: "const float2 vv = float2(vconv(t0 + row, vcol0, vw0), vconv(t0 + row, vcol0 + 1u, vw1));")

    /// Simdgroups per threadgroup for the two chunk scans. `r0` is
    /// `thread_position_in_grid.y * 8`, so the block a simdgroup owns does not depend on
    /// this count; only how many threadgroups share one staged chunk does. The check
    /// below compares both launch shapes bit for bit on both scans at the served shape,
    /// and 8 serves whenever it passes, so no boot-time sample picks the route.
    /// `DARKBLOOM_BONSAI_PSCAN8=0` keeps 4.
    private static let scanNS: Int = resolveScanNS()

    private static func resolveScanNS() -> Int {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_PSCAN8"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["0", "false", "no", "off"].contains(raw ?? "") {
            FileHandle.standardError.write(
                Data("qwen35 prompt scan simdgroups: 4 (DARKBLOOM_BONSAI_PSCAN8=0)\n".utf8))
            return 4
        }
        let passed = scanShapesAgree()
        let ns = passed ? 8 : 4
        let reason = passed ? "bits equal at the served shape" : "bit mismatch or error, 4 kept"
        FileHandle.standardError.write(
            Data("qwen35 prompt scan simdgroups: \(ns) (\(reason))\n".utf8))
        return ns
    }

    private static func scanProbe(_ shape: [Int], seed: Float, scale: Float) -> MLXArray {
        let n = shape.reduce(1, *)
        let i = MLXArray(0 ..< n).asType(.float32)
        let amplitude = 1 + 0.5 * MLX.cos(i * 0.0031 + seed)
        return (MLX.sin(i * 0.017 + seed) * amplitude * scale).reshaped(shape)
    }

    /// Runs both launch shapes of both chunk scans at the served shape (T = 512, 16 key
    /// heads, 32 value heads) and compares every output word.
    private static func scanShapesAgree() -> Bool {
        let T = 512
        let rows = MLXArray(Int32(T))
        let q = scanProbe([1, T, 16, 128], seed: 0.01, scale: 0.5)
        let k = scanProbe([1, T, 16, 128], seed: 0.02, scale: 0.5)
        let v = scanProbe([1, T, 32, 128], seed: 0.03, scale: 0.5)
        let ba = scanProbe([1, T, 64], seed: 0.04, scale: 0.2)
        let aLog = scanProbe([32], seed: 0.05, scale: 0.1) - 0.5
        let dtBias = scanProbe([32], seed: 0.06, scale: 0.5)
        let convState = scanProbe([1, 3, 8192], seed: 0.07, scale: 0.5)
        let qkv = scanProbe([1, T, 8192], seed: 0.08, scale: 0.5)
        let weight = scanProbe([8192, 4, 1], seed: 0.09, scale: 0.1)
        let prepared = prepKernel(
            [q, k, ba, aLog, dtBias, rows],
            template: [("C", 8), ("Dk", 128), ("Hk", 16), ("Hv", 32)],
            grid: (32, T / 8, 16), threadGroup: (32, 1, 1),
            outputShapes: [[1, 32, T / 8, 8, 8], [1, 32, T / 8, 8, 8], [1, 32, T / 8, 2, 8]],
            outputDTypes: [.float32, .float32, .float32])
        func folded(_ ns: Int) -> [MLXArray] {
            convolvedValueScanKernel(
                [q, k, convState, qkv, weight, prepared[0], prepared[1], prepared[2], rows],
                template: [("C", 8), ("Dk", 128), ("Dv", 128), ("Hk", 16), ("Hv", 32), ("NS", ns)],
                grid: (32, 16, 32), threadGroup: (32, ns, 1),
                outputShapes: [[1, T, 32, 128], [1, 32, 128, 128]],
                outputDTypes: [.float32, .float32])
        }
        func plain(_ ns: Int) -> [MLXArray] {
            scanKernel(
                [q, k, v, prepared[0], prepared[1], prepared[2], rows],
                template: [("C", 8), ("Dk", 128), ("Dv", 128), ("Hk", 16), ("Hv", 32), ("NS", ns)],
                grid: (32, 16, 32), threadGroup: (32, ns, 1),
                outputShapes: [[1, T, 32, 128], [1, 32, 128, 128]],
                outputDTypes: [.float32, .float32])
        }
        do {
            return try withError {
                let four = folded(4) + plain(4)
                let eight = folded(8) + plain(8)
                eval(four + eight)
                for (a, b) in zip(four, eight) {
                    let same = (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
                        .item(Bool.self)
                    if !same { return false }
                }
                return true
            }
        } catch {
            FileHandle.standardError.write(Data("qwen35 prompt scan check error: \(error)\n".utf8))
            return false
        }
    }

}

/// Prompt-width register recurrence: `qwen35_gdn_register_v3` (same per-lane
/// layout, sums and shuffles) with the step decay and beta formed in the
/// kernel with the arithmetic of `Qwen35RegisterGatedSIMD` (the bits of
/// `computeGatedDeltaG` and `sigmoid`) from the merged `[b | a]` projection
/// rows. Each threadgroup (two simdgroups, 16 value columns of one head)
/// forms the gates of all steps into threadgroup memory before the scan.
private enum Qwen35PromptGatedGDN {
    static let maximumSteps = 2048
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prompt_gated",
        inputNames: ["q", "k", "v", "ba", "a_log", "dt_bias", "state_in"],
        outputNames: ["y", "state_out"],
        source: """
        const uint T = q_shape[1];
        constexpr int R = 16;
        constexpr int DVPL = 2;
        constexpr int LPD = Dk / R;
        constexpr int DVPS = (32 / LPD) * DVPL;
        constexpr int DVPT = DVPS * 2;
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float gate_g[MAX_T];
        threadgroup float gate_beta[MAX_T];
        {
          const float negative_a = -metal::precise::exp(a_log[hv_idx]);
          const float bias = dt_bias[hv_idx];
          const device float* ba_ = ba + b_idx * T * (2 * Hv);
          for (uint t = thread_index_in_threadgroup; t < T; t += 64u) {
            gate_g[t] = qwen35_simd_gate_decay(ba_[t * (2 * Hv) + Hv + hv_idx], bias, negative_a);
            gate_beta[t] = qwen35_simd_gate_beta(ba_[t * (2 * Hv) + hv_idx]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint dk0 = (lane % LPD) * R;
        const uint dvbase = threadgroup_position_in_grid.y * DVPT + sg * DVPS + (lane / LPD) * DVPL;
        const device float* q_ = q + (b_idx * T * Hk + hk_idx) * Dk + dk0;
        const device float* k_ = k + (b_idx * T * Hk + hk_idx) * Dk + dk0;
        const device float* v_ = v + (b_idx * T * Hv + hv_idx) * Dv + dvbase;
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
        for (uint t = 0; t < T; ++t) {
          const float gt = gate_g[t];
          const float bt = gate_beta[t];
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ((const device float4*)k_)[j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            const float4 q4 = ((const device float4*)q_)[j];
            qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
          }
          float kv[DVPL];
          float vt[DVPL];
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            vt[d] = v_[d];
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
          q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv;
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
        header: Qwen35RegisterGatedSIMD.gateHeader)

    static func run(
        q: MLXArray, k: MLXArray, v: MLXArray, ba: MLXArray, aLog: MLXArray, dtBias: MLXArray,
        state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let B = k.dim(0), T = k.dim(1), Hk = k.dim(2), Dk = k.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)
        let result = kernel(
            [q, k, v, ba, aLog, dtBias, state],
            template: [("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                ("MAX_T", T <= 512 ? 512 : maximumSteps)],
            grid: (64, Dv / 16, B * Hv), threadGroup: (64, 1, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }
}

/// The `in_proj_b` and `in_proj_a` weights stacked `[b; a]` for the
/// prompt-width projection, outside the module topology. One matmul over the
/// stacked weight gives each output column the bits of its own matmul (the
/// same split-K partition and tile code per column; checked once on this
/// device before first use).
final class Qwen35PromptGateWeights {
    private var sources: (MLXArray, MLXArray)?
    private var stacked: MLXArray?
    nonisolated(unsafe) private static var checked: Bool?

    func prepared(b: MLXArray, a: MLXArray) -> MLXArray? {
        guard let (sb, sa) = sources, sb === b, sa === a else { return nil }
        return stacked
    }

    func get(b: MLXArray, a: MLXArray) -> MLXArray? {
        guard Self.columnsMatch(rows: 512, inputs: b.dim(1), outputs: b.dim(0)) else { return nil }
        if let (sb, sa) = sources, sb === b, sa === a, let stacked { return stacked }
        let merged = concatenated([b, a], axis: 0)
        eval(merged)
        (sources, stacked) = ((b, a), merged)
        return merged
    }

    private static func columnsMatch(rows: Int, inputs: Int, outputs: Int) -> Bool {
        if let checked { return checked }
        let x = sin(MLXArray(0 ..< Int32(rows * inputs)).asType(.float32) * 0.37).reshaped([1, rows, inputs])
        let wb = cos(MLXArray(0 ..< Int32(outputs * inputs)).asType(.float32) * 0.11).reshaped([outputs, inputs]) * 0.02
        let wa = sin(MLXArray(0 ..< Int32(outputs * inputs)).asType(.float32) * 0.29).reshaped([outputs, inputs]) * 0.02
        let both = matmul(x, concatenated([wb, wa], axis: 0).T)
        let parts = [matmul(x, wb.T), matmul(x, wa.T)]
        let same = [both[0..., 0..., 0 ..< outputs], both[0..., 0..., outputs...]].enumerated().allSatisfy {
            (contiguous($0.element).view(dtype: .uint32) .== parts[$0.offset].view(dtype: .uint32)).all().item(Bool.self)
        }
        checked = same
        return same
    }
}

enum Qwen35RegisterGatedSIMD {
    static let gateHeader = """
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
        """

    private static let source = """
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
        if (lane == 0) {
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
            local_g = qwen35_simd_gate_decay(a_[0], bias, negative_a);
            local_beta = qwen35_simd_gate_beta(bb_[0]);
          }
          const float gt = simd_broadcast(local_g, 0);
          const float bt = simd_broadcast(local_beta, 0);
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ((const device float4*)k_)[j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            if (!STATE_ONLY) {
              const float4 q4 = ((const device float4*)q_)[j];
              qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
            }
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
              if (!STATE_ONLY) {
                o0 = fma(state[d][4 * j], qr[4 * j], o0);
                o1 = fma(state[d][4 * j + 1], qr[4 * j + 1], o1);
                o2 = fma(state[d][4 * j + 2], qr[4 * j + 2], o2);
                o3 = fma(state[d][4 * j + 3], qr[4 * j + 3], o3);
              }
            }
            if (!STATE_ONLY) {
              out[d] = (o0 + o1) + (o2 + o3);
            }
          }
          if (!STATE_ONLY) {
            out[0] += simd_shuffle_xor(out[0], 4);
            out[1] += simd_shuffle_xor(out[1], 4);
            out[0] += simd_shuffle_xor(out[0], 2);
            out[1] += simd_shuffle_xor(out[1], 2);
            out[0] += simd_shuffle_xor(out[0], 1);
            out[1] += simd_shuffle_xor(out[1], 1);
            if (lane % LPD == 0) {
              *((device float2*)y_) = float2(out[0], out[1]);
            }
            q_ += Hk * Dk;
            y_ += Hv * Dv;
          }
          k_ += Hk * Dk; v_ += Hv * Dv; a_ += Hv; bb_ += Hv;
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

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_register_gate_simd",
        inputNames: ["q", "k", "v", "a", "b", "a_log", "dt_bias", "state_in"],
        outputNames: ["y", "state_out"],
        source: source,
        header: gateHeader)

    /// The prefix-replay twin of the state-only kernel. It reads `k`, `v`,
    /// `a`, `b` and `state_in` through their strides, so the strided value
    /// slice of the verify conv output needs no contiguous copy, and it also
    /// writes the boundary conv rows `conv_in[:, T ..< T + 3]` as bit copies
    /// to their own buffer. The recurrence text is the record kernel's.
    private static let replaySource: String = replaySource(hoistedGates: false)

    /// `hoistedGates` forms every replayed row's gates once, before the scan
    /// (as `qwen35_gdn_verify_fused` does), instead of lane 0 forming one per
    /// step inside the serial chain. Same function, same inputs, same bits.
    private static func replaySource(hoistedGates: Bool) -> String {
        var s = source
        func swap(_ from: String, _ to: String) {
            precondition(
                s.components(separatedBy: from).count == 2,
                "qwen35 replay kernel source: \(from)")
            s = s.replacingOccurrences(of: from, with: to)
        }
        swap(
            "const device float* k_ = k + (b_idx * T * Hk + hk_idx) * Dk + dk0;",
            "const device float* k_ = k + b_idx * k_strides[0] + hk_idx * k_strides[2] + dk0;")
        swap(
            "const device float* v_ = v + (b_idx * T * Hv + hv_idx) * Dv + dvbase;",
            "const device float* v_ = v + b_idx * v_strides[0] + hv_idx * v_strides[2] + dvbase;")
        swap(
            "const device float* a_ = a + b_idx * T * Hv + hv_idx;",
            "const device float* a_ = a + b_idx * a_strides[0] + hv_idx * a_strides[2];")
        swap(
            "const device float* bb_ = b + b_idx * T * Hv + hv_idx;",
            "const device float* bb_ = b + b_idx * b_strides[0] + hv_idx * b_strides[2];")
        swap(
            "(const device float4*)(state_in + (n * Dv + dvbase + d) * Dk + dk0);",
            "(const device float4*)(state_in + b_idx * state_in_strides[0] + hv_idx * state_in_strides[1] + (dvbase + d) * state_in_strides[2] + dk0);")
        swap(
            "k_ += Hk * Dk; v_ += Hv * Dv; a_ += Hv; bb_ += Hv;",
            "k_ += k_strides[1]; v_ += v_strides[1]; a_ += a_strides[1]; bb_ += b_strides[1];")
        // Replay is state-only and has no output rows.
        swap(
            "device float* y_ = y + (b_idx * T * Hv + hv_idx) * Dv + dvbase;",
            "")
        swap(
            "if (lane % LPD == 0) {\n      *((device float2*)y_) = float2(out[0], out[1]);\n    }",
            "")
        swap("y_ += Hv * Dv;", "")
        if hoistedGates {
            swap(
                "float state[DVPL][R];",
                """
                threadgroup float gate_g[16];
                threadgroup float gate_beta[16];
                if (thread_index_in_threadgroup < uint(T)) {
                  const uint gt_row = thread_index_in_threadgroup;
                  const float gate_negative_a = -metal::precise::exp(a_log[hv_idx]);
                  const float gate_bias = dt_bias[hv_idx];
                  gate_g[gt_row] = qwen35_simd_gate_decay(a_[gt_row * a_strides[1]], gate_bias, gate_negative_a);
                  gate_beta[gt_row] = qwen35_simd_gate_beta(bb_[gt_row * b_strides[1]]);
                }
                float state[DVPL][R];
                """)
            swap(
                "float kr[R];",
                "threadgroup_barrier(mem_flags::mem_threadgroup);\nfloat kr[R];")
            swap("local_g = qwen35_simd_gate_decay(a_[0], bias, negative_a);", "")
            swap("local_beta = qwen35_simd_gate_beta(bb_[0]);", "")
            swap("const float gt = simd_broadcast(local_g, 0);", "const float gt = gate_g[t];")
            swap("const float bt = simd_broadcast(local_beta, 0);", "const float bt = gate_beta[t];")
        }
        return s + """
            {
              // Boundary conv rows T ..< T + 3 of conv_in, copied as raw bits.
              const uint rows = 3u;
              const uint cdim = uint(conv_in_shape[2]);
              const uint per_batch = rows * cdim;
              const uint total = uint(conv_in_shape[0]) * per_batch;
              const uint nthreads = threads_per_grid.x * threads_per_grid.y * threads_per_grid.z;
              const uint gid = (thread_position_in_grid.z * threads_per_grid.y + thread_position_in_grid.y)
                  * threads_per_grid.x + thread_position_in_grid.x;
              const device uint* src = (const device uint*)conv_in;
              device uint* dst = (device uint*)conv_tail;
              for (uint i = gid; i < total; i += nthreads) {
                const uint bi = i / per_batch;
                const uint rem = i - bi * per_batch;
                const uint r = rem / cdim;
                const uint c = rem - r * cdim;
                dst[i] = src[bi * conv_in_strides[0] + (T + r) * conv_in_strides[1] + c * conv_in_strides[2]];
              }
            }

            """
    }

    private static let replayKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_register_gate_simd_replay",
        inputNames: ["q", "k", "v", "a", "b", "a_log", "dt_bias", "state_in", "conv_in"],
        outputNames: ["state_out", "conv_tail"],
        source: replaySource,
        header: gateHeader,
        ensureRowContiguous: false)

    private static let hoistedReplayKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_register_gate_simd_replay_hg",
        inputNames: ["q", "k", "v", "a", "b", "a_log", "dt_bias", "state_in", "conv_in"],
        outputNames: ["state_out", "conv_tail"],
        source: replaySource(hoistedGates: true),
        header: gateHeader,
        ensureRowContiguous: false)

    /// Row-list twin of a replay source: step `t` replays tape row `rows[t]` (`T` = rows' length)
    /// and the boundary conv rows are the path's (logical row `L` of `conv_in` is `L` below 3,
    /// else `3 + rows[L - 3]`). `rows = 0 ..< T` is the record replay.
    static func treeReplaySource(_ record: String) -> String? {
        var s = record
        func swap(_ from: String, _ to: String) -> Bool {
            guard s.components(separatedBy: from).count == 2 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        let ok = swap("const uint T = q_shape[1];", "const uint T = uint(tree_rows_shape[0]);")
            && swap(
                "for (int t = 0; t < T; ++t) {",
                """
                for (int t = 0; t < T; ++t) {
                const int tr_row = tree_rows[t];
                q_ = q + b_idx * q_strides[0] + hk_idx * q_strides[2] + dk0 + tr_row * q_strides[1];
                k_ = k + b_idx * k_strides[0] + hk_idx * k_strides[2] + dk0 + tr_row * k_strides[1];
                v_ = v + b_idx * v_strides[0] + hv_idx * v_strides[2] + dvbase + tr_row * v_strides[1];
                a_ = a + b_idx * a_strides[0] + hv_idx * a_strides[2] + tr_row * a_strides[1];
                bb_ = b + b_idx * b_strides[0] + hv_idx * b_strides[2] + tr_row * b_strides[1];
                """)
            && swap("q_ += Hk * Dk;", "")
            && swap("k_ += k_strides[1]; v_ += v_strides[1]; a_ += a_strides[1]; bb_ += b_strides[1];", "")
            && swap(
                "(T + r) * conv_in_strides[1]",
                "((T + r) < 3u ? (T + r) : 3u + uint(tree_rows[T + r - 3u])) * conv_in_strides[1]")
        guard ok else { return nil }
        if s.contains("a_[gt_row * a_strides[1]]") {
            guard swap("a_[gt_row * a_strides[1]]", "a_[tree_rows[gt_row] * a_strides[1]]"),
                swap("bb_[gt_row * b_strides[1]]", "bb_[tree_rows[gt_row] * b_strides[1]]")
            else { return nil }
        }
        return s
    }

    private static let treeReplayKernels: [MLXFast.MLXFastKernel?] = [false, true].map { hoisted in
        treeReplaySource(replaySource(hoistedGates: hoisted)).map {
            MLXFast.metalKernel(
                name: hoisted ? "qwen35_gdn_register_gate_simd_replay_hg_tree" : "qwen35_gdn_register_gate_simd_replay_tree",
                inputNames: ["q", "k", "v", "a", "b", "a_log", "dt_bias", "state_in", "conv_in", "tree_rows"],
                outputNames: ["state_out", "conv_tail"],
                source: $0,
                header: gateHeader,
                ensureRowContiguous: false)
        }
    }

    /// `runReplay` along tape rows `rows` (int32 [m]); nil when the twin was not derived.
    static func runTreeReplay(q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray,
                              b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
                              state: MLXArray, convInput: MLXArray, rows: MLXArray) -> (MLXArray, MLXArray)? {
        guard let kernel = treeReplayKernels[hoistedReplayPassed ? 1 : 0] else { return nil }
        let B = k.dim(0), Hk = k.dim(2), Dk = k.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)
        let result = kernel(
            [q, k, v, a, b, aLog, dtBias, state, convInput, rows],
            template: [("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("STATE_ONLY", 1)],
            grid: (128, Dv / 32, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [[B, Hv, Dv, Dk], [B, 3, convInput.dim(2)]],
            outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }

    /// `DARKBLOOM_BONSAI_REPLAY_GATE_HOIST=0` keeps the record replay kernel.
    static let hoistedReplayOffered =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_REPLAY_GATE_HOIST"] != "0"
    /// Written once by `prepareHoistedReplay` at load; read-only afterwards.
    nonisolated(unsafe) static var hoistedReplayPassed = false
    nonisolated(unsafe) private static var hoistedReplayPrepared = false

    /// Load-time check: the hoisted replay against the record replay at 1, 2,
    /// 5, 8, 11 and 15 rows, every output bit. Any mismatch or error keeps
    /// the record kernel.
    static func prepareHoistedReplay() {
        guard !hoistedReplayPrepared else { return }
        hoistedReplayPrepared = true
        guard hoistedReplayOffered else {
            FileHandle.standardError.write(Data("qwen35 replay gate hoist: off\n".utf8))
            return
        }
        var verdict = "mismatch"
        do {
            if try withError({ hoistedReplayCheck() }) {
                hoistedReplayPassed = true
                verdict = "bitwise equal at rows 1, 2, 5, 8, 11, 15"
            }
        } catch {
            verdict = "error \(error)"
        }
        FileHandle.standardError.write(Data("qwen35 replay gate hoist: \(verdict)\n".utf8))
    }

    private static func checkRows(_ shape: [Int], seed: Float, scale: Float) -> MLXArray {
        let n = shape.reduce(1, *)
        let i = MLXArray(0 ..< n).asType(.float32)
        let firstAngle = i * (0.7548776 + seed)
        let first = MLX.sin(firstAngle)
        let secondAngle = i * 0.0137 + seed
        let second = MLX.cos(secondAngle)
        let amplitude = 1 + 0.5 * second
        let w = first * amplitude
        return (w * scale).reshaped(shape)
    }

    private static func hoistedReplayCheck() -> Bool {
        let hk = 16, hv = 32, d = 128, width = 16, convDim = 2 * hk * d + hv * d
        let qkv = checkRows([1, width, hk, d], seed: 0.01, scale: 0.1)
        let kk = checkRows([1, width, hk, d], seed: 0.02, scale: 0.1)
        let vv = checkRows([1, width, hv, d], seed: 0.03, scale: 1)
        // Gate inputs span both softplus branches and the large-|x| paths.
        let aa = checkRows([1, width, hv], seed: 0.04, scale: 30)
        let bb = checkRows([1, width, hv], seed: 0.05, scale: 8)
        let aLog = checkRows([hv], seed: 0.06, scale: 2)
        let dtBias = checkRows([hv], seed: 0.07, scale: 3)
        let state = checkRows([1, hv, d, d], seed: 0.08, scale: 0.5)
        let conv = checkRows([1, width + 3, convDim], seed: 0.09, scale: 1)
        var eqs: [MLXArray] = []
        for rows in [1, 2, 5, 8, 11, 15] {
            let q = qkv[0..., 0 ..< rows, 0...]
            let inputs = [q, kk, vv, aa, bb, aLog, dtBias, state, conv]
            let template: [(String, any KernelTemplateArg)] = [
                ("Dk", d), ("Dv", d), ("Hk", hk), ("Hv", hv), ("STATE_ONLY", 1),
            ]
            let shapes = [state.shape, [1, 3, convDim]]
            let record = replayKernel(
                inputs, template: template, grid: (128, d / 32, hv), threadGroup: (128, 1, 1),
                outputShapes: shapes, outputDTypes: [.float32, .float32])
            let hoisted = hoistedReplayKernel(
                inputs, template: template, grid: (128, d / 32, hv), threadGroup: (128, 1, 1),
                outputShapes: shapes, outputDTypes: [.float32, .float32])
            for j in 0 ..< 2 {
                let x = record[j].view(dtype: .uint32)
                let y = hoisted[j].view(dtype: .uint32)
                eqs.append(MLX.all(x .== y))
            }
        }
        return MLX.all(MLX.stacked(eqs)).item(Bool.self)
    }

    static func run(q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray,
                    b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
                    state: MLXArray) -> (MLXArray, MLXArray) {
        let B = k.dim(0), T = k.dim(1), Hk = k.dim(2), Dk = k.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)
        let result = kernel(
            [q, k, v, a, b, aLog, dtBias, state],
            template: [("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("STATE_ONLY", 0)],
            grid: (128, Dv / 32, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [[B, T, Hv, Dv], [B, Hv, Dv, Dk]], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }

    static func runStateOnly(q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray,
                             b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
                             state: MLXArray) -> MLXArray {
        let B = k.dim(0), Hk = k.dim(2), Dk = k.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)
        let result = kernel(
            [q, k, v, a, b, aLog, dtBias, state],
            template: [("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("STATE_ONLY", 1)],
            grid: (128, Dv / 32, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [[1], [B, Hv, Dv, Dk]], outputDTypes: [.float32, .float32])
        return result[1]
    }

    /// `(state after T rows, conv rows T ..< T + 3 of convInput)`: the bits of
    /// `runStateOnly` and of `contiguous(convInput[:, T ..< T + 3])`, with every
    /// input read in place through its strides. `T` is `q.dim(1)`.
    static func runReplay(q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray,
                          b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
                          state: MLXArray, convInput: MLXArray) -> (MLXArray, MLXArray) {
        let B = k.dim(0), Hk = k.dim(2), Dk = k.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)
        let result = (hoistedReplayPassed ? hoistedReplayKernel : replayKernel)(
            [q, k, v, a, b, aLog, dtBias, state, convInput],
            template: [("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("STATE_ONLY", 1)],
            grid: (128, Dv / 32, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [[B, Hv, Dv, Dk], [B, 3, convInput.dim(2)]],
            outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }
}

/// Verify-width GDN middle in one kernel: `Qwen35FusedConvSilu`,
/// `Qwen35FusedQKNorm`, the contiguous copy of the value slice and
/// `Qwen35RegisterGatedSIMD`. Each threadgroup of the recurrence (one value
/// head, 32 value columns) first forms, with the same code, the conv and
/// SiLU of its key head's q/k channels and of its 32 value channels, then
/// the q/k RMSNorm, and keeps them in threadgroup memory for the scan.
/// Every output (conv input, packed V, normalized q/k, y, final state) is
/// the bits of the four-kernel graph; each element is written by exactly
/// one threadgroup.
enum Qwen35FusedGDNVerify {
    static let maximumRows = 16
    private static let invScale = pow(Float(128), -0.5)
    static let source = """
        const uint T = qkv_shape[1];
        constexpr int R = 16;
        constexpr int DVPL = 2;
        constexpr int LPD = Dk / R;
        constexpr int DVPS = (32 / LPD) * DVPL;
        constexpr int DVPT = DVPS * 8;
        constexpr uint C = uint(CONV_DIM);
        threadgroup float qk_raw[2 * MAX_T * Dk];
        threadgroup float v_tile[MAX_T * DVPT];
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_index_in_threadgroup;
        const uint dvblock = threadgroup_position_in_grid.y;
        const bool owns_qk = (hv_idx % (Hv / Hk)) == 0 && dvblock == 0;
        const uint dk0 = (lane % LPD) * R;
        const uint dvlocal = sg * DVPS + (lane / LPD) * DVPL;
        const uint dvbase = dvblock * DVPT + dvlocal;
        // Form every row's gates concurrently, before the existing conv/norm
        // barrier. All four simdgroups reuse the same bits in the scan.
        threadgroup float gate_g[MAX_T];
        threadgroup float gate_beta[MAX_T];
        if (tid < T) {
          const float negative_a = -metal::precise::exp(a_log[hv_idx]);
          const float bias = dt_bias[hv_idx];
          const uint gate_idx = (b_idx * T + tid) * Hv + hv_idx;
          gate_g[tid] = qwen35_simd_gate_decay(a[gate_idx], bias, negative_a);
          gate_beta[tid] = qwen35_simd_gate_beta(b[gate_idx]);
        }
        // The scan state loads first, so they overlap the conv and norm below.
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
        {
          #pragma clang fp reassociate(off)
          // Conv + SiLU (Qwen35FusedConvSilu) of the q/k channels of key head hk:
          // all 256 threads run in parallel (tid 0..127 -> Q, tid 128..255 -> K).
          {
            const uint part = tid >> 7;
            const uint qk_col = tid & 127u;
            const uint c = part * uint(Hk * Dk) + hk_idx * uint(Dk) + qk_col;
            // All loads first; the taps then run in Qwen35FusedConvSilu's order.
            float xin[3 + MAX_T];
            #pragma clang loop unroll(full)
            for (uint t = 0; t < 3u + uint(MAX_T); ++t) {
              if (t < 3u + T) xin[t] = t < 3u ? conv_state[t * C + c] : qkv[(t - 3u) * C + c];
            }
            const float4 w4 = ((const device float4*)conv_weight)[c];
            const float w[4] = {w4.x, w4.y, w4.z, w4.w};
            if (owns_qk) {
              #pragma clang loop unroll(full)
              for (uint t = 0; t < 3u + uint(MAX_T); ++t) {
                if (t < 3u + T) {
                  conv_in[t * C + c] = xin[t];
                  // The retained tail, detached: the bits of `x + 0` (-0 -> +0).
                  if (t >= T) {
                    const uint xb = as_type<uint>(xin[t]);
                    conv_tail[(t - T) * C + c] = as_type<float>(xb == 0x80000000u ? 0u : xb);
                  }
                }
              }
            }
            #pragma clang loop unroll(full)
            for (uint p = 0; p < uint(MAX_T); ++p) {
              if (p < T) {
                float acc = 0.0f;
                #pragma clang loop unroll(full)
                for (uint tap = 0; tap < 4u; ++tap) {
                  acc = fma(xin[p + tap], w[tap], acc);
                }
                const float e = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
                const float sig = acc < 0.0f ? e : 1.0f - e;
                const float out = acc * sig;
                qk_raw[(part * MAX_T + p) * Dk + qk_col] = out;
              }
            }
          }
          // Eight simdgroups share Q/K for 64 value channels, without increasing per-thread state.
          {
            const uint col = tid % uint(DVPT);
            const uint c = 2u * uint(Hk * Dk) + hv_idx * uint(Dv) + dvblock * uint(DVPT) + col;
            const float4 vw4 = ((const device float4*)conv_weight)[c];
            const float vw[4] = {vw4.x, vw4.y, vw4.z, vw4.w};
            for (uint t = tid / uint(DVPT); t < 3u + T; t += 256u / uint(DVPT)) {
              const float xv = t < 3u ? conv_state[t * C + c] : qkv[(t - 3u) * C + c];
              conv_in[t * C + c] = xv;
              if (t >= T) {
                const uint xb = as_type<uint>(xv);
                conv_tail[(t - T) * C + c] = as_type<float>(xb == 0x80000000u ? 0u : xb);
              }
            }
            for (uint p = tid / uint(DVPT); p < T; p += 256u / uint(DVPT)) {
              float acc = 0.0f;
              #pragma clang loop unroll(full)
              for (uint tap = 0; tap < 4u; ++tap) {
                const uint r = p + tap;
                const float v = r < 3u ? conv_state[r * C + c] : qkv[(r - 3u) * C + c];
                acc = fma(v, vw[tap], acc);
              }
              const float e = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
              const float sig = acc < 0.0f ? e : 1.0f - e;
              const float out = acc * sig;
              v_tile[p * DVPT + col] = out;
              v_out[(p * uint(Hv) + hv_idx) * uint(Dv) + dvblock * uint(DVPT) + col] = out;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
          #pragma clang fp reassociate(off)
          // q/k RMSNorm (Qwen35FusedQKNorm): one simdgroup per (row, q or k).
          for (uint job = sg; job < 2u * T; job += 8u) {
            const uint part = job % 2u;
            const uint t = job / 2u;
            threadgroup float* row = qk_raw + (part * MAX_T + t) * Dk;
            const float4 x4 = ((threadgroup const float4*)(row + lane * 4u))[0];
            const float x[4] = {x4.x, x4.y, x4.z, x4.w};
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
              sum += x[i] * x[i];
            }
            const float eps = as_type<float>(uint(EPS_BITS));
            const float inv = metal::precise::rsqrt(simd_sum(sum) / 128.0f + eps);
            const float scale = as_type<float>(uint(part == 0u ? Q_SCALE_BITS : K_SCALE_BITS));
            const float4 normed = float4(
              scale * (x[0] * inv),
              scale * (x[1] * inv),
              scale * (x[2] * inv),
              scale * (x[3] * inv)
            );
            ((threadgroup float4*)(row + lane * 4u))[0] = normed;
            if (owns_qk) {
              device float* dst = (part == 0u ? q_out : k_out) + (t * uint(Hk) + hk_idx) * uint(Dk) + lane * 4u;
              ((device float4*)dst)[0] = normed;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Qwen35RegisterGatedSIMD with q/k/v read from threadgroup memory.
        threadgroup const float* q_ = qk_raw + dk0;
        threadgroup const float* k_ = qk_raw + MAX_T * Dk + dk0;
        threadgroup const float* v_ = v_tile + dvlocal;
        device float* y_ = y + (b_idx * T * Hv + hv_idx) * Dv + dvbase;
        float kr[R];
        float qr[R];
        for (int t = 0; t < int(T); ++t) {
          const float gt = gate_g[t];
          const float bt = gate_beta[t];
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ((threadgroup const float4*)k_)[j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            const float4 q4 = ((threadgroup const float4*)q_)[j];
            qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
          }
          float kv[DVPL];
          float vt[DVPL];
          const float2 v2 = *((threadgroup const float2*)v_);
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
          q_ += Dk;
          y_ += Hv * Dv;
          k_ += Dk; v_ += DVPT;
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

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_verify_fused",
        inputNames: ["conv_state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias", "state_in"],
        outputNames: ["conv_in", "v_out", "q_out", "k_out", "y", "state_out", "conv_tail"],
        source: source,
        header: Qwen35RegisterGatedSIMD.gateHeader)

    private static let deltaKernel: MLXFast.MLXFastKernel? = {
        guard let text = capturingDelta(source) else { return nil }
        return MLXFast.metalKernel(
            name: "qwen35_gdn_verify_fused_delta",
            inputNames: ["conv_state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias", "state_in"],
            outputNames: ["conv_in", "k_out", "y", "state_out", "conv_tail", "delta_out"],
            source: text, header: Qwen35RegisterGatedSIMD.gateHeader)
    }()

    private static func tunedSourceFor(_ record: String) -> String? {
        var s = record
        func change(_ from: String, _ to: String, count: Int = 1) -> Bool {
            guard s.components(separatedBy: from).count == count + 1 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        guard change("const uint T = qkv_shape[1];", "constexpr uint T = 16u;") else { return nil }
        guard change("// all 256 threads run in parallel (tid 0..127 -> Q, tid 128..255 -> K).\n  {",
            "// The first 256 threads form Q/K; the remaining threads form V.\n  if (tid < 256u) {") else { return nil }
        guard change("constexpr int DVPT = DVPS * 8;", "constexpr int DVPT = DVPS * 16;") else { return nil }
        guard change(
            "  // Eight simdgroups share Q/K for 64 value channels, without increasing per-thread state.\n  {\n    const uint col = tid % uint(DVPT);",
            "  // Eight simdgroups share Q/K for 64 value channels, without increasing per-thread state.\n  if (tid >= 256u) {\n    const uint vtid = tid - 256u;\n    const uint col = vtid % uint(DVPT);"
        ) else { return nil }
        guard change("uint t = tid / uint(DVPT);", "uint t = vtid / uint(DVPT);") else { return nil }
        guard change("uint p = tid / uint(DVPT);", "uint p = vtid / uint(DVPT);") else { return nil }
        guard change("job += 8u", "job += 16u") else { return nil }
        guard let beginRange = s.range(of: "// The scan state loads first, so they overlap the conv and norm below.")
        else { return nil }
        let begin = beginRange.lowerBound
        guard let finishRange = s.range(of: "\n{\n  #pragma clang fp reassociate(off)", range: begin..<s.endIndex)
        else { return nil }
        let finish = finishRange.lowerBound
        var load = String(s[begin..<finish])
        load = load.replacingOccurrences(
            of: "// The scan state loads first, so they overlap the conv and norm below.",
            with: "// Keep state registers out of the convolution frontend.")
        s.removeSubrange(begin..<finish)
        let scan = "// Qwen35RegisterGatedSIMD with q/k/v read from threadgroup memory."
        guard s.components(separatedBy: scan).count == 2 else { return nil }
        return s.replacingOccurrences(of: scan, with: load + "\n" + scan)
    }

    private static let tunedSource = tunedSourceFor(source)

    private static func capturingDelta(_ s: String, dead: Bool = false) -> String? {
        let from = "const float delta = (vt[d] - kv[d]) * bt;"
        let vStore = "      v_out[(p * uint(Hv) + hv_idx) * uint(Dv) + dvblock * uint(DVPT) + col] = out;\n"
        let qkStore = """
                if (owns_qk) {
                  device float* dst = (part == 0u ? q_out : k_out) + (t * uint(Hk) + hk_idx) * uint(Dk) + lane * 4u;
                  ((device float4*)dst)[0] = normed;
                }
            """
        let kOnlyStore = """
                if (owns_qk && part != 0u) {
                  device float* dst = k_out + (t * uint(Hk) + hk_idx) * uint(Dk) + lane * 4u;
                  ((device float4*)dst)[0] = normed;
                }
            """
        // The window copy is read only through its three boundary rows, which `conv_tail`
        // and the replay already take from the inputs.
        let qkConvStore = "          conv_in[t * C + c] = xin[t];\n"
        let vConvStore = "      conv_in[t * C + c] = xv;\n"
        guard s.components(separatedBy: from).count == 2,
            s.components(separatedBy: vStore).count == 2,
            s.components(separatedBy: qkStore).count == 2,
            !dead || (s.components(separatedBy: qkConvStore).count == 2
                && s.components(separatedBy: vConvStore).count == 2)
        else { return nil }
        let kept = dead
            ? s.replacingOccurrences(of: qkConvStore, with: "")
                .replacingOccurrences(of: vConvStore, with: "")
            : s
        return kept.replacingOccurrences(of: vStore, with: "")
            .replacingOccurrences(of: qkStore, with: kOnlyStore)
            .replacingOccurrences(of: from, with: """
            const float delta = (vt[d] - kv[d]) * bt;
            if (lane % LPD == 0) {
              delta_out[(t * Hv + hv_idx) * (Dv + 1) + dvbase + d] = delta;
            }
            if (dvbase == 0 && lane % LPD == 0 && d == 0) {
              delta_out[(t * Hv + hv_idx) * (Dv + 1) + Dv] = gt;
            }
            """)
    }

    private static let deltaKernelDead: MLXFast.MLXFastKernel? = {
        guard deadOutputsOffered, let text = capturingDelta(source, dead: true) else { return nil }
        return MLXFast.metalKernel(
            name: "qwen35_gdn_verify_fused_delta_nocv",
            inputNames: ["conv_state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias", "state_in"],
            outputNames: ["k_out", "y", "state_out", "conv_tail", "delta_out"],
            source: text, header: Qwen35RegisterGatedSIMD.gateHeader)
    }()

    private static let tunedKernel: MLXFast.MLXFastKernel? = {
        guard let text = tunedSource else { return nil }
        return MLXFast.metalKernel(
            name: "qwen35_gdn_verify_fused_sg16",
            inputNames: ["conv_state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias", "state_in"],
            outputNames: ["conv_in", "v_out", "q_out", "k_out", "y", "state_out", "conv_tail"],
            source: text, header: Qwen35RegisterGatedSIMD.gateHeader)
    }()

    private static let tunedDeltaKernel: MLXFast.MLXFastKernel? = {
        guard let record = tunedSource, let text = capturingDelta(record) else { return nil }
        return MLXFast.metalKernel(
            name: "qwen35_gdn_verify_fused_sg16_delta",
            inputNames: ["conv_state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias", "state_in"],
            outputNames: ["conv_in", "k_out", "y", "state_out", "conv_tail", "delta_out"],
            source: text, header: Qwen35RegisterGatedSIMD.gateHeader)
    }()

    private static let tunedDeltaKernelDead: MLXFast.MLXFastKernel? = {
        guard deadOutputsOffered, let record = tunedSource,
            let text = capturingDelta(record, dead: true) else { return nil }
        return MLXFast.metalKernel(
            name: "qwen35_gdn_verify_fused_sg16_delta_nocv",
            inputNames: ["conv_state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias", "state_in"],
            outputNames: ["k_out", "y", "state_out", "conv_tail", "delta_out"],
            source: text, header: Qwen35RegisterGatedSIMD.gateHeader)
    }()

    static let deadOutputsOffered =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DEAD_OUTPUTS"] != "0"
    private static let tuneOffered = ProcessInfo.processInfo.environment["GDN_VERIFY_TUNE"] != "0"
    private static let deltaOffered = ProcessInfo.processInfo.environment["GDN_DELTA_REPLAY"] != "0"
    nonisolated(unsafe) private static var tunePassed = false
    nonisolated(unsafe) private static var deltaPassed = false
    nonisolated(unsafe) private static var deadPassed = false
    nonisolated(unsafe) private static var prepared = false

    typealias VerifyResult = (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray?)



    /// Tree twin of `source`. `tree` is the draft tree descriptor (int32, at least 288):
    /// parent row [16], depth [16] (row 0 = anchor, depth 0), ancestor row by depth [16][16].
    /// Each row's conv window reads its own path (taps past the anchor read `conv_state`), and a
    /// row whose parent is not the previous row restarts the scan from `state_in` and re-runs
    /// its ancestors in path order (no output for them), so every row is the bits of a chain
    /// verify of its path. `conv_in` is the physical window (the replay tape); `state_out` and
    /// `conv_tail` follow the last physical row, which is the full acceptance of a chain.
    static func treeSource(_ record: String) -> String? {
        var s = record
        func swap(_ from: String, _ to: String) -> Bool {
            guard s.components(separatedBy: from).count == 2 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        let reload = """
            if (tb_back) {
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                const device float4* s_re4 = (const device float4*)(state_in + (n * Dv + dvbase + d) * Dk + dk0);
                #pragma clang loop unroll(full)
                for (int j = 0; j < R / 4; ++j) {
                  const float4 s4 = s_re4[j];
                  state[d][4 * j + 0] = s4.x;
                  state[d][4 * j + 1] = s4.y;
                  state[d][4 * j + 2] = s4.z;
                  state[d][4 * j + 3] = s4.w;
                }
              }
            }
            """
        let ok = swap(
            "acc = fma(xin[p + tap], w[tap], acc);",
            """
            const uint tb_l = uint(tree[16 + p]) + tap;
            acc = fma(xin[tb_l < 3u ? tb_l : 3u + uint(tree[32 + 16 * p + tb_l - 3u])], w[tap], acc);
            """)
            && swap(
                "const uint r = p + tap;",
                """
                const uint tb_l = uint(tree[16 + p]) + tap;
                const uint r = tb_l < 3u ? tb_l : 3u + uint(tree[32 + 16 * p + tb_l - 3u]);
                """)
            && swap(
                "for (int t = 0; t < int(T); ++t) {",
                """
                for (int t = 0; t < int(T); ++t) {
                const bool tb_back = t > 0 && tree[t] != t - 1;
                \(reload)
                const int tb_d = tree[16 + t];
                for (int tb_i = tb_back ? 0 : tb_d; tb_i <= tb_d; ++tb_i) {
                const int u = tb_i == tb_d ? t : tree[32 + 16 * t + tb_i];
                q_ = qk_raw + dk0 + u * Dk;
                k_ = qk_raw + MAX_T * Dk + dk0 + u * Dk;
                v_ = v_tile + dvlocal + u * DVPT;
                y_ = y + (b_idx * T * Hv + hv_idx) * Dv + dvbase + uint(u) * Hv * Dv;
                """)
            && swap("const float gt = gate_g[t];", "const float gt = gate_g[u];")
            && swap("const float bt = gate_beta[t];", "const float bt = gate_beta[u];")
            && swap("if (lane % LPD == 0) {", "if (lane % LPD == 0 && u == t) {")
            && swap("q_ += Dk;", "}")
            && swap("y_ += Hv * Dv;", "")
            && swap("k_ += Dk; v_ += DVPT;", "")
        return ok ? s : nil
    }

    /// `treeSource`'s outputs, cheaper. The descriptor and each row's four conv tap rows
    /// (`tb_tab`, physical `conv_in` rows) are read into threadgroup memory once. Each q/k
    /// column thread stages its rows' raw inputs in its own `qk_raw` slots and convolves the
    /// rows in descending order, so a row's ancestors (always lower rows) are still raw when it
    /// reads them: tap 3 is the row itself, tap 2 its parent, taps 0 and 1 come from the table.
    /// A row whose parent is not the previous row reloads `state_in` and re-runs only the
    /// state update of its ancestors. Same values in the same fma order as `treeSource`.
    /// `DARKBLOOM_BONSAI_TREE_FAST=0` keeps `treeSource`.
    static func fastTreeSource(_ record: String) -> String? {
        var s = record
        func swap(_ from: String, _ to: String) -> Bool {
            guard s.components(separatedBy: from).count == 2 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        let table = """
            threadgroup float gate_g[MAX_T];
            threadgroup int tb_desc[288];
            threadgroup uint tb_tab[MAX_T * 4];
            for (uint tb_i = thread_index_in_threadgroup; tb_i < 288u; tb_i += 256u) { tb_desc[tb_i] = tree[tb_i]; }
            if (thread_index_in_threadgroup < MAX_T * 4u) {
              const uint tb_p = thread_index_in_threadgroup / 4u, tb_t = thread_index_in_threadgroup % 4u;
              const uint tb_l = uint(tree[16 + tb_p]) + tb_t;
              tb_tab[tb_p * 4u + tb_t] = tb_l < 3u ? tb_l : 3u + uint(tree[32 + 16 * tb_p + tb_l - 3u]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """
        let descending = """
            for (uint tb_s = 0; tb_s < uint(MAX_T); ++tb_s) {
              if (tb_s < T) { qk_raw[(part * MAX_T + tb_s) * Dk + qk_col] = xin[3u + tb_s]; }
            }
            #pragma clang loop unroll(full)
            for (uint tb_q = 0; tb_q < uint(MAX_T); ++tb_q) {
            const uint p = uint(MAX_T) - 1u - tb_q;
            """
        let tap = """
            float tb_v;
            if (p == 0u || tap == 3u) {
              tb_v = xin[p + tap];
            } else if (tap == 2u) {
              tb_v = qk_raw[(part * MAX_T + uint(tb_desc[p])) * Dk + qk_col];
            } else {
              const uint tb_r = tb_tab[p * 4u + tap];
              const float tb_cs = tb_r == 0u ? xin[0] : (tb_r == 1u ? xin[1] : xin[2]);
              const float tb_qv = qk_raw[(part * MAX_T + (tb_r < 3u ? 0u : tb_r - 3u)) * Dk + qk_col];
              tb_v = tb_r < 3u ? tb_cs : tb_qv;
            }
            acc = fma(tb_v, w[tap], acc);
            """
        let restart = """
            for (int t = 0; t < int(T); ++t) {
            if (t > 0 && tb_desc[t] != t - 1) {
              const int sp_d = tb_desc[16 + t];
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                const device float4* s_re4 = (const device float4*)(state_in + (n * Dv + dvbase + d) * Dk + dk0);
                #pragma clang loop unroll(full)
                for (int j = 0; j < R / 4; ++j) {
                  const float4 s4 = s_re4[j];
                  state[d][4 * j + 0] = s4.x;
                  state[d][4 * j + 1] = s4.y;
                  state[d][4 * j + 2] = s4.z;
                  state[d][4 * j + 3] = s4.w;
                }
              }
              for (int sp_i = 0; sp_i < sp_d; ++sp_i) {
                const int u = tb_desc[32 + 16 * t + sp_i];
                const float gu = gate_g[u];
                const float bu = gate_beta[u];
                threadgroup const float* ku = qk_raw + MAX_T * Dk + dk0 + u * Dk;
                float kq[R];
                #pragma clang loop unroll(full)
                for (int j = 0; j < R / 4; ++j) {
                  const float4 k4 = ((threadgroup const float4*)ku)[j];
                  kq[4 * j] = k4.x; kq[4 * j + 1] = k4.y; kq[4 * j + 2] = k4.z; kq[4 * j + 3] = k4.w;
                }
                const float2 vu = *((threadgroup const float2*)(v_tile + dvlocal + u * DVPT));
                const float vq[DVPL] = {vu.x, vu.y};
                float kvq[DVPL];
                #pragma clang loop unroll(full)
                for (int d = 0; d < DVPL; ++d) {
                  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                  #pragma clang loop unroll(full)
                  for (int j = 0; j < R / 4; ++j) {
                    state[d][4 * j] *= gu; state[d][4 * j + 1] *= gu;
                    state[d][4 * j + 2] *= gu; state[d][4 * j + 3] *= gu;
                    a0 = fma(state[d][4 * j], kq[4 * j], a0);
                    a1 = fma(state[d][4 * j + 1], kq[4 * j + 1], a1);
                    a2 = fma(state[d][4 * j + 2], kq[4 * j + 2], a2);
                    a3 = fma(state[d][4 * j + 3], kq[4 * j + 3], a3);
                  }
                  kvq[d] = (a0 + a1) + (a2 + a3);
                }
                kvq[0] += simd_shuffle_xor(kvq[0], 4);
                kvq[1] += simd_shuffle_xor(kvq[1], 4);
                kvq[0] += simd_shuffle_xor(kvq[0], 2);
                kvq[1] += simd_shuffle_xor(kvq[1], 2);
                kvq[0] += simd_shuffle_xor(kvq[0], 1);
                kvq[1] += simd_shuffle_xor(kvq[1], 1);
                #pragma clang loop unroll(full)
                for (int d = 0; d < DVPL; ++d) {
                  const float delta = (vq[d] - kvq[d]) * bu;
                  #pragma clang loop unroll(full)
                  for (int j = 0; j < R / 4; ++j) {
                    state[d][4 * j] = fma(kq[4 * j], delta, state[d][4 * j]);
                    state[d][4 * j + 1] = fma(kq[4 * j + 1], delta, state[d][4 * j + 1]);
                    state[d][4 * j + 2] = fma(kq[4 * j + 2], delta, state[d][4 * j + 2]);
                    state[d][4 * j + 3] = fma(kq[4 * j + 3], delta, state[d][4 * j + 3]);
                  }
                }
              }
            }
            """
        let ok = swap("threadgroup float gate_g[MAX_T];", table)
            && swap("for (uint p = 0; p < uint(MAX_T); ++p) {", descending)
            && swap("acc = fma(xin[p + tap], w[tap], acc);", tap)
            && swap("const uint r = p + tap;", "const uint r = tb_tab[p * 4u + tap];")
            && swap("for (int t = 0; t < int(T); ++t) {", restart)
        guard ok else { return nil }
        return preloadedTreeValues(s)
    }

    /// In SG16 the scan state is loaded after the frontend. Keep V row work
    /// parallel, but read raw path values once per column-owner parity into
    /// a scoped private array instead of reloading four taps for every row.
    static func preloadedTreeValues(_ fast: String) -> String? {
        guard fast.contains("const uint vtid = tid - 256u;") else { return fast }
        var s = fast
        let before = "const float vw[4] = {vw4.x, vw4.y, vw4.z, vw4.w};"
        let after = before + """

          float vx[3 + MAX_T];
          #pragma clang loop unroll(full)
          for (uint vr = 0u; vr < 3u + uint(MAX_T); ++vr) {
            if (vr < 3u + T) {
              vx[vr] = vr < 3u ? conv_state[vr * C + c] : qkv[(vr - 3u) * C + c];
            }
          }
        """
        let load = "const float v = r < 3u ? conv_state[r * C + c] : qkv[(r - 3u) * C + c];"
        let raw = "const float xv = t < 3u ? conv_state[t * C + c] : qkv[(t - 3u) * C + c];"
        guard s.components(separatedBy: before).count == 2,
            s.components(separatedBy: load).count == 2,
            s.components(separatedBy: raw).count == 2 else { return nil }
        s = s.replacingOccurrences(of: before, with: after)
        s = s.replacingOccurrences(of: load, with: "const float v = vx[r];")
        s = s.replacingOccurrences(of: raw, with: "const float xv = vx[t];")
        return s
    }

    /// Reuse the strict-commit delta output instead of keeping a second cache.
    /// Only the lane that already wrote this word reads it; SIMD shuffle then
    /// broadcasts its own-thread device load, so no cross-lane memory fence is needed.
    /// This transform is offered only after capturingDelta, not to non-delta scans.
    static func cachedTreeDeltas(_ fast: String) -> String? {
        var s = fast
        func swap(_ from: String, _ to: String) -> Bool {
            guard s.components(separatedBy: from).count == 2 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        let old = """
              const float2 vu = *((threadgroup const float2*)(v_tile + dvlocal + u * DVPT));
              const float vq[DVPL] = {vu.x, vu.y};
              float kvq[DVPL];
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                #pragma clang loop unroll(full)
                for (int j = 0; j < R / 4; ++j) {
                  state[d][4 * j] *= gu; state[d][4 * j + 1] *= gu;
                  state[d][4 * j + 2] *= gu; state[d][4 * j + 3] *= gu;
                  a0 = fma(state[d][4 * j], kq[4 * j], a0);
                  a1 = fma(state[d][4 * j + 1], kq[4 * j + 1], a1);
                  a2 = fma(state[d][4 * j + 2], kq[4 * j + 2], a2);
                  a3 = fma(state[d][4 * j + 3], kq[4 * j + 3], a3);
                }
                kvq[d] = (a0 + a1) + (a2 + a3);
              }
              kvq[0] += simd_shuffle_xor(kvq[0], 4);
              kvq[1] += simd_shuffle_xor(kvq[1], 4);
              kvq[0] += simd_shuffle_xor(kvq[0], 2);
              kvq[1] += simd_shuffle_xor(kvq[1], 2);
              kvq[0] += simd_shuffle_xor(kvq[0], 1);
              kvq[1] += simd_shuffle_xor(kvq[1], 1);
          """
        let update = """
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                #pragma clang loop unroll(full)
                for (int j = 0; j < R / 4; ++j) {
                  state[d][4 * j] *= gu; state[d][4 * j + 1] *= gu;
                  state[d][4 * j + 2] *= gu; state[d][4 * j + 3] *= gu;
                }
              }
          """
        let registerDeltaLoad = """
            float delta = 0.0f;
            if (lane % LPD == 0u) {
              delta = delta_out[(u * Hv + hv_idx) * (Dv + 1) + dvbase + d];
            }
            delta = simd_shuffle(delta, lane - lane % LPD);
            """
        let ok = swap(old, update)
            && swap("const float bu = gate_beta[u];", "")
            && swap("const float delta = (vq[d] - kvq[d]) * bu;", registerDeltaLoad)
        return ok ? s : nil
    }

    /// `fastTreeSource(source)` with one saved state: the prologue (during the conv) picks the row
    /// whose state, reused by the restarts below it, saves the most ancestor re-runs (each
    /// restart row t with that row s at depth d on its path skips d + 1 re-runs). The scan
    /// writes that row's state to the thread's own `state_out` words when it reaches it (the
    /// final state overwrites them at the end), and such a restart reloads it and re-runs only
    /// the ancestors below s. A re-run is the same update on the same values, so the bits are
    /// those of a restart from `state_in`. `DARKBLOOM_BONSAI_TREE_SNAPSHOT=0` keeps `fastTreeSource`.
    static func snapshotTreeSource(_ fast: String) -> String? {
        var s = fast
        func swap(_ from: String, _ to: String) -> Bool {
            guard s.components(separatedBy: from).count == 2 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        let pick = """
            {
              const uint sn_s = tid >> 4, sn_t = tid & 15u;
              uint sn_v = 0u;
              if (sn_t > sn_s && sn_t < T) {
                const int sn_ds = tb_desc[16 + sn_s];
                if (tb_desc[sn_t] != int(sn_t) - 1 && tb_desc[16 + sn_t] > sn_ds
                    && tb_desc[32 + 16 * sn_t + sn_ds] == int(sn_s)) {
                  sn_v = uint(sn_ds) + 1u;
                }
              }
              sn_v += simd_shuffle_xor(sn_v, 8u);
              sn_v += simd_shuffle_xor(sn_v, 4u);
              sn_v += simd_shuffle_xor(sn_v, 2u);
              sn_v += simd_shuffle_xor(sn_v, 1u);
              if (sn_t == 0u) {
                sn_score[sn_s] = sn_v != 0u ? (sn_v << 4) | sn_s : 0u;
              }
            }
            """
        let save = """
            if (t == sn_row) {
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                device float4* sn_out4 = (device float4*)(state_out + (n * Dv + dvbase + d) * Dk + dk0);
                #pragma clang loop unroll(full)
                for (int j = 0; j < R / 4; ++j) {
                  sn_out4[j] = float4(state[d][4 * j + 0], state[d][4 * j + 1], state[d][4 * j + 2], state[d][4 * j + 3]);
                }
              }
            }
            """
        let ok = swap("threadgroup uint tb_tab[MAX_T * 4];", "threadgroup uint tb_tab[MAX_T * 4];\nthreadgroup uint sn_score[16];")
            && swap(
                "// Eight simdgroups share Q/K for 64 value channels, without increasing per-thread state.",
                pick + "\n// Eight simdgroups share Q/K for 64 value channels, without increasing per-thread state.")
            && swap(
                "float qr[R];",
                "float qr[R];\nconst uint sn_k = simd_max(lane < 16u ? sn_score[lane] : 0u);\n"
                    + "const int sn_row = sn_k != 0u ? int(sn_k & 15u) : -1;\nconst int sn_d = sn_row >= 0 ? tb_desc[16 + sn_row] : 0;")
            && swap("q_ += Dk;", save + "\nq_ += Dk;")
            && swap(
                "const int sp_d = tb_desc[16 + t];",
                "const int sp_d = tb_desc[16 + t];\n"
                    + "const bool sn_use = sn_row >= 0 && sp_d > sn_d && tb_desc[32 + 16 * t + sn_d] == sn_row;\n"
                    + "const device float* sn_src = sn_use ? (const device float*)state_out : state_in;")
            && swap(
                "const device float4* s_re4 = (const device float4*)(state_in + ",
                "const device float4* s_re4 = (const device float4*)(sn_src + ")
            && swap("for (int sp_i = 0; sp_i < sp_d; ++sp_i) {", "for (int sp_i = sn_use ? sn_d + 1 : 0; sp_i < sp_d; ++sp_i) {")
        return ok ? s : nil
    }

    static let fastTreeOffered =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE_FAST"] != "0"
    static let snapshotOffered =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TREE_SNAPSHOT"] != "0"

    private static let treeKernel: MLXFast.MLXFastKernel? = {
        let plainFast = fastTreeOffered ? fastTreeSource(source) : nil
        let snapshot = snapshotOffered ? plainFast.flatMap(Qwen35FusedGDNVerify.snapshotTreeSource) : nil
        let fast = snapshot ?? plainFast
        guard let text = fast ?? treeSource(source) else { return nil }
        let name = snapshot != nil ? "qwen35_gdn_verify_fused_tree_fast_snapshot"
            : fast != nil ? "qwen35_gdn_verify_fused_tree_fast" : "qwen35_gdn_verify_fused_tree"
        return MLXFast.metalKernel(
            name: name,
            inputNames: ["conv_state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias", "state_in", "tree"],
            outputNames: ["conv_in", "v_out", "q_out", "k_out", "y", "state_out", "conv_tail"],
            source: text,
            header: Qwen35RegisterGatedSIMD.gateHeader)
    }()

    /// The final `state_out` and `conv_tail` are read only by a full-acceptance commit, which needs
    /// an accepted path through all T rows. Rows come in DFS preorder (parents are lower rows), so
    /// such a path exists only when every row's parent is the row before it; on any other descriptor
    /// both stores are skipped. The snapshot's mid-scan `state_out` store is scratch and stays.
    static func chainStores(_ text: String) -> String? {
        var s = text
        func swap(_ from: String, _ to: String, count: Int) -> Bool {
            guard s.components(separatedBy: from).count == count + 1 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        let ok = swap(
            "const uint n = threadgroup_position_in_grid.z;\n",
            "const uint n = threadgroup_position_in_grid.z;\n"
                + "bool tb_chain = true;\n"
                + "for (uint tb_c = 1u; tb_c < T; ++tb_c) { tb_chain = tb_chain && tree[tb_c] == int(tb_c) - 1; }\n",
            count: 1)
            && swap("s_out4[j] = float4(state[d]", "if (tb_chain) s_out4[j] = float4(state[d]", count: 1)
            && swap("conv_tail[(t - T) * C + c] =", "if (tb_chain) conv_tail[(t - T) * C + c] =", count: 2)
        return ok ? s : nil
    }

    static let chainStoresOffered =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DEAD_OUTPUTS2"] != "0"
    nonisolated(unsafe) private static var chainStoresPassed = false

    private static func treeVariant(tuned: Bool, delta: Bool, dead: Bool = false, chain: Bool = false)
        -> MLXFast.MLXFastKernel?
    {
        guard !dead || (delta && deadOutputsOffered) else { return nil }
        guard !chain || chainStoresOffered else { return nil }
        let record: String
        if tuned {
            guard let text = tunedSource else { return nil }
            record = text
        } else { record = source }
        let plainFast = fastTreeOffered ? fastTreeSource(record) : nil
        let snapshot = snapshotOffered ? plainFast.flatMap(Qwen35FusedGDNVerify.snapshotTreeSource) : nil
        guard var text = snapshot ?? plainFast ?? treeSource(record) else { return nil }
        if tuned && plainFast != nil {
            // The descriptor covers 288 words; SG16 has 512 threads rather than 256.
            guard text.components(separatedBy: "tb_i += 256u").count == 2 else { return nil }
            text = text.replacingOccurrences(of: "tb_i += 256u", with: "tb_i += 512u")
        }
        if tuned && snapshot != nil {
            // Only the first 256 threads score the 16 by 16 snapshot candidates.
            guard text.components(separatedBy: "{\n  const uint sn_s = tid >> 4, sn_t = tid & 15u;").count == 2
            else { return nil }
            text = text.replacingOccurrences(
                of: "{\n  const uint sn_s = tid >> 4, sn_t = tid & 15u;",
                with: "if (tid < 256u) {\n  const uint sn_s = tid >> 4, sn_t = tid & 15u;")
        }
        if delta {
            guard let captured = capturingDelta(text, dead: dead) else { return nil }
            text = captured
        }
        if chain {
            guard let guarded = chainStores(text) else { return nil }
            text = guarded
        }
        if delta && plainFast != nil {
            guard let cached = cachedTreeDeltas(text) else { return nil }
            text = cached
        }
        let route = snapshot != nil ? "snapshot" : plainFast != nil ? "fast" : "record"
        return MLXFast.metalKernel(
            name: "qwen35_gdn_verify_tree_" + route + (tuned ? "_sg16" : "") + (delta ? "_delta" : "")
                + (dead ? "_nocv" : "") + (chain ? "_cs" : ""),
            inputNames: ["conv_state", "qkv", "conv_weight", "a", "b", "a_log", "dt_bias", "state_in", "tree"],
            outputNames: dead
                ? ["k_out", "y", "state_out", "conv_tail", "delta_out"]
                : delta
                ? ["conv_in", "k_out", "y", "state_out", "conv_tail", "delta_out"]
                : ["conv_in", "v_out", "q_out", "k_out", "y", "state_out", "conv_tail"],
            source: text, header: Qwen35RegisterGatedSIMD.gateHeader)
    }

    private static let deltaTreeKernel = treeVariant(tuned: false, delta: true)
    private static let tunedTreeKernel = treeVariant(tuned: true, delta: false)
    private static let tunedDeltaTreeKernel = treeVariant(tuned: true, delta: true)
    private static let deltaTreeKernelDead = treeVariant(tuned: false, delta: true, dead: true)
    private static let tunedDeltaTreeKernelDead = treeVariant(tuned: true, delta: true, dead: true)
    private static let treeKernelChain = treeVariant(tuned: false, delta: false, chain: true)
    private static let tunedTreeKernelChain = treeVariant(tuned: true, delta: false, chain: true)
    private static let deltaTreeKernelChain = treeVariant(tuned: false, delta: true, chain: true)
    private static let tunedDeltaTreeKernelChain = treeVariant(tuned: true, delta: true, chain: true)
    private static let deltaTreeKernelDeadChain = treeVariant(tuned: false, delta: true, dead: true, chain: true)
    private static let tunedDeltaTreeKernelDeadChain = treeVariant(tuned: true, delta: true, dead: true, chain: true)

    /// `apply` for a draft tree window (`tree`: see `treeSource`); nil when `apply` would be.
    static func applyTree(
        convState: MLXArray, qkv: MLXArray, convWeight: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray, state: MLXArray, tree: MLXArray,
        keyHeads: Int, valueHeads: Int
    ) -> VerifyResult? {
        guard let treeKernel, tree.dtype == .int32, tree.size >= 288,
            convState.ndim == 3, qkv.ndim == 3, convWeight.ndim == 3,
            a.ndim == 3, b.ndim == 3, state.ndim == 4,
            convState.dtype == .float32, qkv.dtype == .float32, convWeight.dtype == .float32,
            a.dtype == .float32, b.dtype == .float32, aLog.dtype == .float32,
            dtBias.dtype == .float32, state.dtype == .float32
        else { return nil }
        let qkvShape = qkv.shape3
        let S = qkvShape.1
        let C = qkvShape.2
        guard qkvShape.0 == 1, (3 ... maximumRows).contains(S),
            convState.shape3 == (1, 3, C), C == 2 * keyHeads * 128 + valueHeads * 128,
            valueHeads % keyHeads == 0, convWeight.shape3 == (C, 4, 1),
            a.shape3 == (1, S, valueHeads), b.shape3 == (1, S, valueHeads),
            state.shape4 == (1, valueHeads, 128, 128),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let eligible = keyHeads == 16 && valueHeads == 32
        let wantsDelta = eligible && deltaPassed
        let wantsTuned = eligible && S == 16 && tunePassed
        let tuned = wantsTuned && (wantsDelta ? tunedDeltaTreeKernel : tunedTreeKernel) != nil
        let captureDelta = wantsDelta && (tuned ? tunedDeltaTreeKernel : deltaTreeKernel) != nil
        func stores(_ full: MLXFast.MLXFastKernel?, _ chain: MLXFast.MLXFastKernel?) -> MLXFast.MLXFastKernel? {
            chainStoresPassed ? (chain ?? full) : full
        }
        let selected: MLXFast.MLXFastKernel?
        if tuned {
            selected = captureDelta
                ? stores(tunedDeltaTreeKernel, tunedDeltaTreeKernelChain) : stores(tunedTreeKernel, tunedTreeKernelChain)
        } else {
            selected = captureDelta ? stores(deltaTreeKernel, deltaTreeKernelChain) : stores(treeKernel, treeKernelChain)
        }
        let deadKernel = captureDelta && deadPassed
            ? (tuned ? stores(tunedDeltaTreeKernelDead, tunedDeltaTreeKernelDeadChain)
                : stores(deltaTreeKernelDead, deltaTreeKernelDeadChain))
            : nil
        guard let selected = deadKernel ?? selected else { return nil }
        let qShape = [1, S, keyHeads, 128]
        let vShape = [1, S, valueHeads, 128]
        let shapes = deadKernel != nil
            ? [qShape, vShape, [1, valueHeads, 128, 128], [1, 3, C], [1, S, valueHeads, 129]]
            : captureDelta
            ? [[1, S + 3, C], qShape, vShape, [1, valueHeads, 128, 128], [1, 3, C], [1, S, valueHeads, 129]]
            : [[1, S + 3, C], vShape, qShape, qShape, vShape, [1, valueHeads, 128, 128], [1, 3, C]]
        let o = selected(
            [convState, qkv, convWeight, a, b, aLog, dtBias, state, tree],
            template: [("Dk", 128), ("Dv", 128), ("Hk", keyHeads), ("Hv", valueHeads),
                ("CONV_DIM", C), ("MAX_T", maximumRows),
                ("EPS_BITS", Int(Float(1e-6).bitPattern)),
                ("Q_SCALE_BITS", Int(pow(invScale, 2).bitPattern)),
                ("K_SCALE_BITS", Int(invScale.bitPattern))],
            grid: (tuned ? 512 : 256, tuned ? 1 : 2, valueHeads), threadGroup: (tuned ? 512 : 256, 1, 1),
            outputShapes: shapes,
            outputDTypes: deadKernel != nil ? Array(repeating: .float32, count: 5)
                : captureDelta ? [.float32, .float32, .float32, .float32, .float32, .float32] : [.float32, .float32, .float32, .float32, .float32, .float32, .float32])
        if deadKernel != nil {
            // `qkv` stands for the window rows of the conv input; the three rows before them are `convState`.
            return (qkv, o[0], o[0], o[0], o[1], o[2], o[3], o[4])
        }
        if captureDelta {
            return (o[0], o[1], o[1], o[1], o[2], o[3], o[4], o[5])
        }
        return (o[0], o[1], o[2], o[3], o[4], o[5], o[6], nil)
    }

    /// `(convIn, v, q, k, y, finalState, convTail, deltaTape)` of the verify
    /// window, or nil when the shapes do not take the fused path. `convTail`
    /// is `convIn[:, S...] + 0` (the full-acceptance conv state) in its own
    /// buffer.
    static func apply(
        convState: MLXArray, qkv: MLXArray, convWeight: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray, state: MLXArray, keyHeads: Int, valueHeads: Int
    ) -> VerifyResult? {
        guard convState.ndim == 3, qkv.ndim == 3, convWeight.ndim == 3,
            a.ndim == 3, b.ndim == 3, state.ndim == 4,
            convState.dtype == .float32, qkv.dtype == .float32, convWeight.dtype == .float32,
            a.dtype == .float32, b.dtype == .float32, aLog.dtype == .float32,
            dtBias.dtype == .float32, state.dtype == .float32
        else { return nil }
        let qkvShape = qkv.shape3
        let S = qkvShape.1
        let C = qkvShape.2
        guard qkvShape.0 == 1, (3 ... maximumRows).contains(S),
            convState.shape3 == (1, 3, C), C == 2 * keyHeads * 128 + valueHeads * 128,
            valueHeads % keyHeads == 0, convWeight.shape3 == (C, 4, 1),
            a.shape3 == (1, S, valueHeads), b.shape3 == (1, S, valueHeads),
            state.shape4 == (1, valueHeads, 128, 128),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let eligible = keyHeads == 16 && valueHeads == 32
        let wantsDelta = eligible && deltaPassed
        let wantsTuned = eligible && S == 16 && tunePassed
        let tuned = wantsTuned && (wantsDelta ? tunedDeltaKernel : tunedKernel) != nil
        let captureDelta = wantsDelta && (tuned ? tunedDeltaKernel : deltaKernel) != nil
        let deadKernel = captureDelta && deadPassed ? (tuned ? tunedDeltaKernelDead : deltaKernelDead) : nil
        let selectedKernel: MLXFast.MLXFastKernel
        if let deadKernel {
            selectedKernel = deadKernel
        } else if tuned {
            let selected = captureDelta ? tunedDeltaKernel : tunedKernel
            selectedKernel = selected ?? kernel
        } else {
            selectedKernel = captureDelta ? (deltaKernel ?? kernel) : kernel
        }
        let qShape = [1, S, keyHeads, 128]
        let vShape = [1, S, valueHeads, 128]
        let convShape = [1, S + 3, C]
        let tailShape = [1, 3, C]
        let shapes = deadKernel != nil
            ? [qShape, vShape, [1, valueHeads, 128, 128], tailShape, [1, S, valueHeads, 129]]
            : captureDelta
            ? [convShape, qShape, vShape, [1, valueHeads, 128, 128], tailShape, [1, S, valueHeads, 129]]
            : [convShape, vShape, qShape, qShape, vShape, [1, valueHeads, 128, 128], tailShape]
        let outputs = selectedKernel(
            [convState, qkv, convWeight, a, b, aLog, dtBias, state],
            template: [("Dk", 128), ("Dv", 128), ("Hk", keyHeads), ("Hv", valueHeads),
                ("CONV_DIM", C), ("MAX_T", maximumRows),
                ("EPS_BITS", Int(Float(1e-6).bitPattern)),
                ("Q_SCALE_BITS", Int(pow(invScale, 2).bitPattern)),
                ("K_SCALE_BITS", Int(invScale.bitPattern))],
            grid: (tuned ? 512 : 256, tuned ? 1 : 2, valueHeads),
            threadGroup: (tuned ? 512 : 256, 1, 1),
            outputShapes: shapes,
            outputDTypes: deadKernel != nil ? Array(repeating: .float32, count: 5)
                : captureDelta ? [.float32, .float32, .float32, .float32, .float32, .float32] : [.float32, .float32, .float32, .float32, .float32, .float32, .float32])
        if deadKernel != nil {
            return (qkv, outputs[0], outputs[0], outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
        }
        if captureDelta {
            return (outputs[0], outputs[1], outputs[1], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5])
        }
        return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], nil)
    }
    /// Compile/check fixed synthetic tensors at model load, before request data exists.
    static func prepare() {
        guard !prepared, StreamOrDevice.default == StreamOrDevice.gpu else { return }
        prepared = true
        guard tuneOffered || deltaOffered else { return }
        Qwen35RegisterGatedSIMD.prepareHoistedReplay()
        var verdict = (false, false, false, false)
        do {
            verdict = try withError { checkPreparedKernels() }
        } catch {
            FileHandle.standardError.write(Data("gdn16 check error: \(error)\n".utf8))
        }
        tunePassed = tuneOffered && verdict.0
        deltaPassed = deltaOffered && verdict.1
        Qwen35DeltaReplay.unrolledPassed = deltaPassed && Qwen35DeltaReplay.unrolledOffered && verdict.2
        deadPassed = deltaPassed && deadOutputsOffered && verdict.3
        var chainOK = false
        if chainStoresOffered {
            do {
                chainOK = try withError { checkChainStores() }
            } catch {
                FileHandle.standardError.write(Data("gdn16 chain-store check error: \(error)\n".utf8))
            }
        }
        chainStoresPassed = chainOK
        let message = "gdn16 checked: tune=\(tunePassed) delta=\(deltaPassed)"
            + " unroll=\(Qwen35DeltaReplay.unrolledPassed) dead=\(deadPassed) dead2=\(chainStoresPassed)\n"
        FileHandle.standardError.write(Data(message.utf8))
        guard Qwen35SnapshotCommit.offered else { return }
        var snapOK = false
        if snapshotOffered && fastTreeOffered && deltaPassed && chainStoresPassed {
            do {
                snapOK = try withError { checkSnapshotCommit() }
            } catch {
                FileHandle.standardError.write(Data("snap commit check error: \(error)\n".utf8))
            }
        }
        Qwen35SnapshotCommit.mode = !snapOK ? .off : Qwen35SnapshotCommit.audit ? .auditing : .on
        FileHandle.standardError.write(Data("snap commit checked: \(snapOK)\n".utf8))
    }

    /// The tree kernel the checked flags select, on fixed and pseudo-random descriptors: for every
    /// path `CBv2BranchVerify.snapshotPrefix` accepts, a path ending at the snapshot row has the
    /// kernel's `state_out` and `Qwen35SnapshotCommit.convTail` as its replay's bits, and a longer
    /// path replayed from `state_out` past that row has the full replay's bits. Each skip width
    /// compiles and matches on a chain. The parent row's replay must differ from `state_out`.
    private static func checkSnapshotCommit() -> Bool {
        let flags = (tunePassed, deltaPassed, deadPassed, chainStoresPassed)
        defer { (tunePassed, deltaPassed, deadPassed, chainStoresPassed) = flags }
        var trees: [[Int32]] = [
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 11, 4, 13, 14],
            [-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12],
        ]
        var seed: UInt32 = 12345
        for _ in 0 ..< 8 {
            var parents: [Int32] = [-1]
            for t in 1 ..< 16 {
                seed = seed &* 1_664_525 &+ 1_013_904_223
                parents.append(Int32(t - 1 - Int(seed >> 16) % min(t, 4)))
            }
            trees.append(parents)
        }
        var ok = true
        var distinct = false
        var (ends, passes) = (0, 0)
        for kind in 0 ..< 2 {
            let convState = probeRows([1, 3, 8192], seed: 0.01, scale: 0.5)
            let qkv = probeRows([1, 16, 8192], seed: 0.02, scale: 0.5)
            let weight = probeRows([8192, 4, 1], seed: 0.03, scale: 0.1)
            let a = probeRows([1, 16, 32], seed: 0.04, scale: 30)
            let b = probeRows([1, 16, 32], seed: 0.05, scale: 8)
            let aLog = probeRows([32], seed: 0.06, scale: 2)
            let bias = probeRows([32], seed: 0.07, scale: 3)
            let state = kind == 1 ? probeWords([1, 32, 128, 128])
                : probeRows([1, 32, 128, 128], seed: 0.08, scale: 0.5)
            let inputs = [convState, qkv, weight, a, b, aLog, bias, state]
            for parents in trees {
                let paths = probeTree(parents.map(Int.init)).1.map { $0.map(Int32.init) }
                guard let out = probeForward(inputs, tuned: flags.0, delta: true, tree: probeTree(parents.map(Int.init)).0,
                        dead: flags.2, chain: true),
                    let tape = out.7
                else { return false }
                let split = out.0.dim(1) == 16
                func replay(_ path: [Int32], from: MLXArray, skip: Int) -> (MLXArray, MLXArray)? {
                    Qwen35DeltaReplay.run(committedRows: path.count, k: out.3, delta: tape, state: from,
                        convInput: out.0, treeRows: MLXArray(path), stateIsRowContiguous: true,
                        convState: split ? convState : nil, skip: skip)
                }
                for path in paths {
                    guard let prefix = CBv2BranchVerify.snapshotPrefix(parents: parents, path: path) else { continue }
                    let (row, rows) = prefix
                    guard let full = replay(path, from: state, skip: 0) else { return false }
                    if rows == path.count {
                        let tail = Qwen35SnapshotCommit.convTail(
                            path: path, convInput: out.0, convState: split ? convState : nil)
                        ok = sameBits([full.0, full.1], [out.5, tail]) && ok
                        ends += 1
                        if kind == 0, row > 0, let other = replay(paths[Int(parents[row])], from: state, skip: 0) {
                            distinct = distinct || !sameBits([other.0], [out.5])
                        }
                    } else {
                        guard let resumed = replay(path, from: out.5, skip: rows) else { return false }
                        ok = sameBits([full.0, full.1], [resumed.0, resumed.1]) && ok
                        passes += 1
                    }
                }
                if kind == 0, parents == trees[0] {
                    for n in 2 ... 16 {
                        let chain = (0 ..< Int32(n)).map { $0 }
                        guard let first = replay([0], from: state, skip: 0),
                            let full = replay(chain, from: state, skip: 0),
                            let resumed = replay(chain, from: first.0, skip: 1)
                        else { return false }
                        ok = sameBits([full.0, full.1], [resumed.0, resumed.1]) && ok
                    }
                }
            }
        }
        return ok && distinct && ends > 0 && passes > 0
    }

    private static func probeRows(_ shape: [Int], seed: Float, scale: Float) -> MLXArray {
        let n = shape.reduce(1, *)
        let i = MLXArray(0..<n).asType(.float32)
        let firstAngle = i * (0.7548776 + seed)
        let first = MLX.sin(firstAngle)
        let secondAngle = i * 0.0137 + seed
        let second = MLX.cos(secondAngle)
        let amplitude = 1 + 0.5 * second
        let values = first * amplitude
        return (values * scale).reshaped(shape)
    }

    private static func probeWords(_ shape: [Int]) -> MLXArray {
        let pattern: [UInt32] = [0x7fc00001, 0x7fc12345, 0xffc00042, 0x7f800001,
            0x7f800000, 0xff800000, 0, 0x80000000]
        let n = shape.reduce(1, *)
        let words = (0..<n).map { pattern[$0 % pattern.count] }
        return MLXArray(words).view(dtype: .float32).reshaped(shape)
    }

    private static func sameBits(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
        guard a.count == b.count else { return false }
        for j in a.indices {
            guard a[j].shape == b[j].shape else { return false }
            let lhs = a[j].view(dtype: .uint32)
            let rhs = b[j].view(dtype: .uint32)
            let equal = lhs .== rhs
            if !equal.all().item(Bool.self) { return false }
        }
        return true
    }

    private static func seven(_ a: VerifyResult) -> [MLXArray] {
        [a.0, a.1, a.2, a.3, a.4, a.5, a.6]
    }

    private static func five(_ a: VerifyResult) -> [MLXArray] {
        [a.0, a.3, a.4, a.5, a.6]
    }

    /// The tree kernel the checked flags select, with and without `chainStores`, on a chain and two
    /// branching descriptors: every output equal, except `state_out` and `conv_tail` of a branching
    /// descriptor, which no commit reads. `DARKBLOOM_BONSAI_DEAD_OUTPUTS2_NEGCTL=1` compares those
    /// too, so the check fails and the full stores stay.
    private static func checkChainStores() -> Bool {
        let (tuned, delta, dead) = (tunePassed, deltaPassed, deadPassed)
        defer { (tunePassed, deltaPassed, deadPassed) = (tuned, delta, dead); chainStoresPassed = false }
        let kernels = [treeKernelChain, tunedTreeKernelChain, deltaTreeKernelChain, tunedDeltaTreeKernelChain]
            + (deadOutputsOffered ? [deltaTreeKernelDeadChain, tunedDeltaTreeKernelDeadChain] : [])
        guard treeKernel != nil, kernels.allSatisfy({ $0 != nil }) else { return false }
        let negativeControl = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DEAD_OUTPUTS2_NEGCTL"] == "1"
        let identity = Array(-1..<15)
        let parents: [[Int]] = [identity,
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 11, 4, 13, 14],
            [-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        var ok = true
        for kind in 0..<2 {
            let convState = probeRows([1, 3, 8192], seed: 0.01, scale: 0.5)
            let qkv = probeRows([1, 16, 8192], seed: 0.02, scale: 0.5)
            let weight = probeRows([8192, 4, 1], seed: 0.03, scale: 0.1)
            let a = probeRows([1, 16, 32], seed: 0.04, scale: 30)
            let b = probeRows([1, 16, 32], seed: 0.05, scale: 8)
            let aLog = probeRows([32], seed: 0.06, scale: 2)
            let bias = probeRows([32], seed: 0.07, scale: 3)
            let state = kind == 1 ? probeWords([1, 32, 128, 128])
                : probeRows([1, 32, 128, 128], seed: 0.08, scale: 0.5)
            let inputs = [convState, qkv, weight, a, b, aLog, bias, state]
            for parent in parents {
                let tree = probeTree(parent).0
                guard let full = probeForward(inputs, tuned: tuned, delta: delta, tree: tree, dead: dead),
                    let cut = probeForward(inputs, tuned: tuned, delta: delta, tree: tree, dead: dead, chain: true)
                else { return false }
                var lhs = [full.0, full.1, full.2, full.3, full.4]
                var rhs = [cut.0, cut.1, cut.2, cut.3, cut.4]
                switch (full.7, cut.7) {
                case let (fullTape?, cutTape?):
                    lhs.append(fullTape)
                    rhs.append(cutTape)
                case (nil, nil):
                    break
                default:
                    return false
                }
                if parent == identity || negativeControl {
                    lhs += [full.5, full.6]
                    rhs += [cut.5, cut.6]
                }
                ok = sameBits(lhs, rhs) && ok
            }
        }
        return ok
    }

    private static func probeForward(
        _ inputs: [MLXArray], tuned: Bool, delta: Bool, tree: MLXArray? = nil, dead: Bool = false,
        chain: Bool = false
    ) -> VerifyResult? {
        tunePassed = tuned
        deltaPassed = delta
        deadPassed = dead
        chainStoresPassed = chain
        if let tree {
            return applyTree(convState: inputs[0], qkv: inputs[1], convWeight: inputs[2],
                a: inputs[3], b: inputs[4], aLog: inputs[5], dtBias: inputs[6],
                state: inputs[7], tree: tree, keyHeads: 16, valueHeads: 32)
        }
        return apply(convState: inputs[0], qkv: inputs[1], convWeight: inputs[2],
            a: inputs[3], b: inputs[4], aLog: inputs[5], dtBias: inputs[6],
            state: inputs[7], keyHeads: 16, valueHeads: 32)
    }

    private static func probeTree(_ parents: [Int]) -> (MLXArray, [[Int]]) {
        var paths = [[Int]]()
        for row in parents.indices {
            var path = parents[row] < 0 ? [] : paths[parents[row]]
            path.append(row)
            paths.append(path)
        }
        var descriptor = parents.map { Int32($0) }
        descriptor += paths.map { Int32($0.count - 1) }
        for path in paths {
            descriptor += path.map { Int32($0) }
            descriptor += Array(repeating: Int32(0), count: 16 - path.count)
        }
        return (MLXArray(descriptor), paths)
    }

    private static func checkPreparedKernels() -> (Bool, Bool, Bool, Bool) {
        defer { tunePassed = false; deltaPassed = false; deadPassed = false }
        guard tunedKernel != nil, tunedDeltaKernel != nil, deltaKernel != nil,
            deltaTreeKernel != nil, tunedTreeKernel != nil, tunedDeltaTreeKernel != nil
        else { return (false, false, false, false) }
        var tuneOK = true
        var deltaOK = true
        var unrollOK = true
        var deadOK = deadOutputsOffered && deltaKernelDead != nil && tunedDeltaKernelDead != nil
            && deltaTreeKernelDead != nil && tunedDeltaTreeKernelDead != nil
        let negativeControl = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DEAD_OUTPUTS_NEGCTL"] == "1"
        var splitUnrollOK = true
        for kind in 0..<3 {
            let convState = probeRows([1, 3, 8192], seed: 0.01, scale: 0.5)
            let qkv = kind == 2 ? probeWords([1, 16, 8192])
                : probeRows([1, 16, 8192], seed: 0.02, scale: 0.5)
            let weight = probeRows([8192, 4, 1], seed: 0.03, scale: 0.1)
            let a = probeRows([1, 16, 32], seed: 0.04, scale: 30)
            let b = probeRows([1, 16, 32], seed: 0.05, scale: 8)
            let aLog = probeRows([32], seed: 0.06, scale: 2)
            let bias = probeRows([32], seed: 0.07, scale: 3)
            let state = kind == 1 ? probeWords([1, 32, 128, 128])
                : probeRows([1, 32, 128, 128], seed: 0.08, scale: 0.5)
            let inputs = [convState, qkv, weight, a, b, aLog, bias, state]
            let parents: [[Int]?] = [nil,
                [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 11, 4, 13, 14],
                [-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                Array(-1..<15)]
            for parent in parents {
            let descriptor = parent.map { probeTree($0) }
            let tree = descriptor.map { $0.0 }
            guard let record = probeForward(inputs, tuned: false, delta: false, tree: tree),
                let tuned = probeForward(inputs, tuned: true, delta: false, tree: tree),
                let plainCapture = probeForward(inputs, tuned: false, delta: true, tree: tree),
                let tunedCapture = probeForward(inputs, tuned: true, delta: true, tree: tree)
            else { return (false, false, false, false) }
            tuneOK = sameBits(seven(record), seven(tuned)) && tuneOK
            deltaOK = sameBits(five(record), five(plainCapture)) && deltaOK
            deltaOK = sameBits(five(record), five(tunedCapture)) && deltaOK
            for capture in [plainCapture, tunedCapture] {
                guard let tape = capture.7 else { return (false, false, false, false) }
                let paths = descriptor.map { $0.1 }
                    ?? (1...16).map { (steps: Int) -> [Int] in Array(0..<steps) }
                for path in paths {
                    let q = record.2[0..., 0..<path.count, 0...]
                    let treeRows = tree == nil ? nil : MLXArray(path)
                    let old: (MLXArray, MLXArray)
                    if let treeRows {
                        guard let replay = Qwen35RegisterGatedSIMD.runTreeReplay(q: record.2,
                            k: record.3, v: record.1, a: a, b: b, aLog: aLog, dtBias: bias,
                            state: state, convInput: record.0, rows: treeRows)
                        else { return (false, false, false, false) }
                        old = replay
                    } else {
                        old = Qwen35RegisterGatedSIMD.runReplay(q: q, k: record.3, v: record.1,
                            a: a, b: b, aLog: aLog, dtBias: bias, state: state, convInput: record.0)
                    }
                    guard let new = Qwen35DeltaReplay.run(committedRows: path.count, k: capture.3, delta: tape,
                        state: state, convInput: capture.0, treeRows: treeRows, stateIsRowContiguous: true,
                        unrolled: false)
                    else { return (false, false, false, false) }
                    deltaOK = sameBits([old.0, old.1], [new.0, new.1]) && deltaOK
                    // A failure of the unrolled variant only keeps the record replay kernels.
                    let unrolledSame = (try? withError { () -> Bool in
                        guard let unrolled = Qwen35DeltaReplay.run(committedRows: path.count, k: capture.3,
                            delta: tape, state: state, convInput: capture.0, treeRows: treeRows,
                            stateIsRowContiguous: true, unrolled: true)
                        else { return false }
                        return sameBits([new.0, new.1], [unrolled.0, unrolled.1])
                    }) ?? false
                    unrollOK = unrolledSame && unrollOK
                }
            }
            for tuned in [false, true] where deadOK {
                // The dead-output kernels hand back the window rows and leave the three boundary
                // rows to `convState`; everything else is the record's bits.
                guard let dead = probeForward(inputs, tuned: tuned, delta: true, tree: tree, dead: true),
                    let deadTape = dead.7, let capturedTape = plainCapture.7
                else { deadOK = false; break }
                let probeWindow = negativeControl ? dead.0 + 1 : dead.0
                let probeState = negativeControl ? convState + 1 : convState
                deadOK = sameBits([probeWindow, dead.3, dead.4, dead.5, dead.6, deadTape],
                    [qkv, record.3, record.4, record.5, record.6, capturedTape]) && deadOK
                deadOK = sameBits([record.0], [concatenated([convState, qkv], axis: 1)]) && deadOK
                let paths = descriptor.map { $0.1 }
                    ?? (1...16).map { (steps: Int) -> [Int] in Array(0..<steps) }
                for path in paths {
                    let treeRows = tree == nil ? nil : MLXArray(path)
                    let old: (MLXArray, MLXArray)
                    if let treeRows {
                        guard let replay = Qwen35RegisterGatedSIMD.runTreeReplay(q: record.2,
                            k: record.3, v: record.1, a: a, b: b, aLog: aLog, dtBias: bias,
                            state: state, convInput: record.0, rows: treeRows)
                        else { deadOK = false; break }
                        old = replay
                    } else {
                        old = Qwen35RegisterGatedSIMD.runReplay(
                            q: record.2[0..., 0..<path.count, 0...], k: record.3, v: record.1,
                            a: a, b: b, aLog: aLog, dtBias: bias, state: state, convInput: record.0)
                    }
                    guard let new = Qwen35DeltaReplay.run(committedRows: path.count, k: dead.3, delta: deadTape,
                        state: state, convInput: dead.0, treeRows: treeRows, stateIsRowContiguous: true,
                        unrolled: false, convState: probeState)
                    else { deadOK = false; break }
                    deadOK = sameBits([old.0, old.1], [new.0, new.1]) && deadOK
                    // A failure of the unrolled split variant only keeps the record kernels.
                    let splitUnrolledSame = (try? withError { () -> Bool in
                        guard let unrolled = Qwen35DeltaReplay.run(committedRows: path.count, k: dead.3,
                            delta: deadTape, state: state, convInput: dead.0, treeRows: treeRows,
                            stateIsRowContiguous: true, unrolled: true, convState: probeState)
                        else { return false }
                        return sameBits([new.0, new.1], [unrolled.0, unrolled.1])
                    }) ?? false
                    splitUnrollOK = splitUnrolledSame && splitUnrollOK
                }
            }
        }
        }
        return (tuneOK, deltaOK, unrollOK, deadOK && (splitUnrollOK || !unrollOK))
    }

}

/// B and B': on a tree commit whose accepted path passes the verify's snapshot row
/// (`snapshotTreeSource`; with `chainStores` a branching descriptor leaves that snapshot in
/// `state_out`), a path ending at the row commits `state_out` and its conv rows, and a longer path
/// replays only the rows after it. The load check (`checkSnapshotCommit`) must pass first; a
/// failure keeps the record replay, and so does `DARKBLOOM_BONSAI_SNAP_COMMIT=0`. Nothing checks
/// on the round path: the boot warm runs no engine round, so a check there would sit in a timed window.
enum Qwen35SnapshotCommit {
    static let offered = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_SNAP_COMMIT"] != "0"
    /// Diagnostic only: every snapshot commit keeps the record replay's result and checks the
    /// snapshot path against it, reading the checks back one commit later.
    static let audit = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_SNAP_COMMIT_AUDIT"] == "1"

    enum Mode { case off, auditing, on }
    nonisolated(unsafe) static var mode = Mode.off
    nonisolated(unsafe) private static var pending: [(same: MLXArray, ends: Bool)] = []
    nonisolated(unsafe) private static var audited = (ends: 0, endsSame: 0, passes: 0, passesSame: 0)

    /// The mode; while auditing, first reads the queued checks of earlier commits.
    static func current() -> Mode {
        guard mode == .auditing, pending.count >= 24 else { return mode }
        let same = stacked(pending.map { $0.same }).asArray(Bool.self)
        for (entry, equal) in zip(pending, same) {
            if entry.ends {
                audited.ends += 1
                audited.endsSame += equal ? 1 : 0
            } else {
                audited.passes += 1
                audited.passesSame += equal ? 1 : 0
            }
        }
        FileHandle.standardError.write(Data(("snap commit audit: ends \(audited.endsSame)/\(audited.ends)"
            + " passes \(audited.passesSame)/\(audited.passes) layer commits equal\n").utf8))
        pending.removeAll()
        return mode
    }

    static func note(record: CBv2RecurrentLayerState, snapshot: CBv2RecurrentLayerState, ends: Bool) {
        func same(_ a: MLXArray?, _ b: MLXArray?) -> MLXArray {
            guard let a, let b, a.shape == b.shape, a.dtype == .float32, b.dtype == .float32 else {
                return MLXArray(false)
            }
            return (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
        }
        pending.append((same(record.ssm, snapshot.ssm) .&& same(record.conv, snapshot.conv), ends))
    }

    /// The conv rows after `path`: the replay's `conv_tail` bits. `convState` set: `convInput` is
    /// the window alone.
    static func convTail(path: [Int32], convInput: MLXArray, convState: MLXArray?) -> MLXArray {
        let rows = (0 ..< 3).map { path.count + $0 }
        let index: [Int32]
        let source: MLXArray
        if let convState, rows[0] < 3 {
            index = rows.map { $0 < 3 ? Int32($0) : 3 + path[$0 - 3] }
            source = concatenated([convState, convInput], axis: 1)
        } else if convState != nil {
            index = rows.map { path[$0 - 3] }
            source = convInput
        } else {
            index = rows.map { $0 < 3 ? Int32($0) : 3 + path[$0 - 3] }
            source = convInput
        }
        return source.view(dtype: .uint32).take(HostConstants.recentInt32(index), axis: 1).view(dtype: .float32)
    }
}

/// Accepted-prefix state reconstruction from the exact delta captured by verify.
/// The multiply and FMA remain separate, in the original recurrence order.
private enum Qwen35DeltaReplay {
    private static let source = """
        #pragma clang fp reassociate(off)
        #pragma clang fp contract(off)
        const uint T = uint(t_in[0]);
        const uint tid = thread_position_in_grid.x;
        const uint hv = threadgroup_position_in_grid.z;
        const uint dv = tid / 8u;
        const uint dk = (tid % 8u) * 16u;
        float s[16];
        const uint so = (hv * uint(Dv) + dv) * uint(Dk) + dk;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
          const float4 sv = ((const device float4*)(state_in + so))[j];
          s[j * 4] = sv.x; s[j * 4 + 1] = sv.y;
          s[j * 4 + 2] = sv.z; s[j * 4 + 3] = sv.w;
        }
        const uint hk_base = (hv / uint(Hv / Hk)) * uint(Dk) + dk;
        for (uint t = 0; t < T; ++t) {
          const uint d_base = (t * uint(Hv) + hv) * uint(Dv + 1);
          const float g = delta[d_base + uint(Dv)];
          const float d = delta[d_base + dv];
          const uint ki = t * uint(Hk * Dk) + hk_base;
          const device float4* kp = (const device float4*)(k + ki);
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; ++j) {
            const float4 kv = kp[j];
            s[j * 4] *= g; s[j * 4 + 1] *= g;
            s[j * 4 + 2] *= g; s[j * 4 + 3] *= g;
            s[j * 4] = fma(kv.x, d, s[j * 4]);
            s[j * 4 + 1] = fma(kv.y, d, s[j * 4 + 1]);
            s[j * 4 + 2] = fma(kv.z, d, s[j * 4 + 2]);
            s[j * 4 + 3] = fma(kv.w, d, s[j * 4 + 3]);
          }
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
          ((device float4*)(state_out + so))[j] =
              float4(s[j * 4], s[j * 4 + 1], s[j * 4 + 2], s[j * 4 + 3]);
        }
        // Match the old replay's raw-bit conv tail, including signed zero.
        const uint cdim = uint(conv_in_shape[2]);
        const uint gid = hv * threads_per_grid.x + tid;
        const uint nthreads = uint(Hv) * threads_per_grid.x;
        const device uint* src = (const device uint*)conv_in;
        device uint* dst = (device uint*)conv_tail;
        for (uint i = gid; i < 3u * cdim; i += nthreads) {
          const uint r = i / cdim;
          const uint c = i - r * cdim;
          dst[i] = src[(T + r) * cdim + c];
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_delta_replay",
        inputNames: ["t_in", "k", "delta", "state_in", "conv_in"],
        outputNames: ["state_out", "conv_tail"], source: source, ensureRowContiguous: true)

    private static let treeSource = source
        .replacingOccurrences(of: "const uint T = uint(t_in[0]);", with: "const uint T = uint(tree_rows_shape[0]);")
        .replacingOccurrences(of: "for (uint t = 0; t < T; ++t) {", with: "for (uint t = 0; t < T; ++t) {\nconst uint step = uint(tree_rows[t]);")
        .replacingOccurrences(of: "(t * uint(Hv)", with: "(step * uint(Hv)")
        .replacingOccurrences(of: "t * uint(Hk * Dk)", with: "step * uint(Hk * Dk)")
        .replacingOccurrences(of: "(T + r) * cdim", with: "((T + r) < 3u ? (T + r) : 3u + uint(tree_rows[T + r - 3u])) * cdim")

    private static let treeKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_delta_tree_replay",
        inputNames: ["k", "delta", "state_in", "conv_in", "tree_rows"],
        outputNames: ["state_out", "conv_tail"],
        source: treeSource,
        ensureRowContiguous: true)

    /// The same recurrence with the step count as a template constant: the loop unrolls, the
    /// gate and delta loads of every step issue before the scan, and the key rows of the
    /// window reach threadgroup memory in one cooperative load. Each state element sees the
    /// same multiply and FMA in the same order as `source`.
    private static let unrolledBody = """
        #pragma clang fp reassociate(off)
        #pragma clang fp contract(off)
        constexpr uint NT = uint(NSTEP);
        const uint tid = thread_position_in_grid.x;
        const uint tl = tid & 127u;
        const uint hv = threadgroup_position_in_grid.z;
        const uint dv = tid / 8u;
        const uint dk = (tid % 8u) * 16u;
        const uint kh = hv / (Hv / Hk);
        threadgroup float4 ks[NT * 32u];
        float s[16];
        const uint so = (hv * uint(Dv) + dv) * uint(Dk) + dk;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
          const float4 sv = ((const device float4*)(state_in + so))[j];
          s[j * 4] = sv.x; s[j * 4 + 1] = sv.y;
          s[j * 4 + 2] = sv.z; s[j * 4 + 3] = sv.w;
        }
        float gs[NT];
        float ds[NT];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < NT; ++t) {
          const uint row = STEP_ROW(t);
          const uint d_base = (row * uint(Hv) + hv) * uint(Dv + 1);
          gs[t] = delta[d_base + uint(Dv)];
          ds[t] = delta[d_base + dv];
        }
        #pragma clang loop unroll(full)
        for (uint u = 0; u < (NT + 3u) / 4u; ++u) {
          const uint i = u * 128u + tl;
          if (i < NT * 32u) {
            const uint row = STEP_ROW(i >> 5);
            ks[i] = ((const device float4*)(k + row * uint(Hk * Dk) + kh * uint(Dk)))[i & 31u];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint kc = (tid % 8u) * 4u;
        #pragma clang loop unroll(full)
        for (uint t = 0; t < NT; ++t) {
          const float g = gs[t];
          const float d = ds[t];
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; ++j) {
            const float4 kv = ks[t * 32u + kc + uint(j)];
            s[j * 4] *= g; s[j * 4 + 1] *= g;
            s[j * 4 + 2] *= g; s[j * 4 + 3] *= g;
            s[j * 4] = fma(kv.x, d, s[j * 4]);
            s[j * 4 + 1] = fma(kv.y, d, s[j * 4 + 1]);
            s[j * 4 + 2] = fma(kv.z, d, s[j * 4 + 2]);
            s[j * 4 + 3] = fma(kv.w, d, s[j * 4 + 3]);
          }
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
          ((device float4*)(state_out + so))[j] =
              float4(s[j * 4], s[j * 4 + 1], s[j * 4 + 2], s[j * 4 + 3]);
        }
        // Match the old replay's raw-bit conv tail, including signed zero.
        const uint cdim = uint(conv_in_shape[2]);
        const uint gid = hv * threads_per_grid.x + tid;
        const uint nthreads = uint(Hv) * threads_per_grid.x;
        const device uint* src = (const device uint*)conv_in;
        device uint* dst = (device uint*)conv_tail;
        for (uint i = gid; i < 3u * cdim; i += nthreads) {
          const uint r = i / cdim;
          const uint c = i - r * cdim;
          dst[i] = src[TAIL_ROW(r) * cdim + c];
        }
        """

    private static let unrolledChainSource = """
        #define STEP_ROW(t) (t)
        #define TAIL_ROW(r) (NT + (r))

        """ + unrolledBody

    private static let unrolledTreeSource = """
        #define STEP_ROW(t) uint(tree_rows[t])
        #define TAIL_ROW(r) ((NT + (r)) < 3u ? (NT + (r)) : 3u + uint(tree_rows[NT + (r) - 3u]))

        """ + unrolledBody

    private static let unrolledKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_delta_replay_unrolled",
        inputNames: ["k", "delta", "state_in", "conv_in"],
        outputNames: ["state_out", "conv_tail"], source: unrolledChainSource,
        ensureRowContiguous: true)

    private static let unrolledTreeKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_delta_tree_replay_unrolled",
        inputNames: ["k", "delta", "state_in", "conv_in", "tree_rows"],
        outputNames: ["state_out", "conv_tail"], source: unrolledTreeSource,
        ensureRowContiguous: true)

    /// `GDN_REPLAY_UNROLL=0` keeps the record replay kernels.
    static let unrolledOffered = ProcessInfo.processInfo.environment["GDN_REPLAY_UNROLL"] != "0"
    /// Written once by `Qwen35FusedGDNVerify.prepare`; read-only afterwards.
    nonisolated(unsafe) static var unrolledPassed = false

    /// The same kernels with the conv window split in two inputs: `conv_state` holds logical
    /// rows 0 ..< 3 and `conv_in` the window rows after them. The tail copy takes the same bits.
    private static let splitSource = source
        .replacingOccurrences(
            of: "const device uint* src = (const device uint*)conv_in;",
            with: "const device uint* src = (const device uint*)conv_in;\n        const device uint* sst = (const device uint*)conv_state;")
        .replacingOccurrences(
            of: "dst[i] = src[(T + r) * cdim + c];",
            with: "const uint row = T + r;\n          dst[i] = row < 3u ? sst[row * cdim + c] : src[(row - 3u) * cdim + c];")

    private static let splitKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_delta_replay_split",
        inputNames: ["t_in", "k", "delta", "state_in", "conv_state", "conv_in"],
        outputNames: ["state_out", "conv_tail"], source: splitSource, ensureRowContiguous: true)

    private static let splitTreeSource = splitSource
        .replacingOccurrences(of: "const uint T = uint(t_in[0]);", with: "const uint T = uint(tree_rows_shape[0]);")
        .replacingOccurrences(of: "for (uint t = 0; t < T; ++t) {", with: "for (uint t = 0; t < T; ++t) {\nconst uint step = uint(tree_rows[t]);")
        .replacingOccurrences(of: "(t * uint(Hv)", with: "(step * uint(Hv)")
        .replacingOccurrences(of: "t * uint(Hk * Dk)", with: "step * uint(Hk * Dk)")
        .replacingOccurrences(of: "const uint row = T + r;", with: "const uint row = (T + r) < 3u ? (T + r) : 3u + uint(tree_rows[T + r - 3u]);")

    private static let splitTreeKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_delta_tree_replay_split",
        inputNames: ["k", "delta", "state_in", "conv_state", "conv_in", "tree_rows"],
        outputNames: ["state_out", "conv_tail"],
        source: splitTreeSource,
        ensureRowContiguous: true)

    private static func splitUnrolled(_ text: String) -> String {
        text.replacingOccurrences(
                of: "const device uint* src = (const device uint*)conv_in;",
                with: "const device uint* src = (const device uint*)conv_in;\n        const device uint* sst = (const device uint*)conv_state;")
            .replacingOccurrences(
                of: "dst[i] = src[TAIL_ROW(r) * cdim + c];",
                with: "const uint row = TAIL_ROW(r);\n          dst[i] = row < 3u ? sst[row * cdim + c] : src[(row - 3u) * cdim + c];")
    }

    private static let unrolledSplitKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_delta_replay_unrolled_split",
        inputNames: ["k", "delta", "state_in", "conv_state", "conv_in"],
        outputNames: ["state_out", "conv_tail"], source: splitUnrolled(unrolledChainSource),
        ensureRowContiguous: true)

    private static let unrolledSplitTreeKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_delta_tree_replay_unrolled_split",
        inputNames: ["k", "delta", "state_in", "conv_state", "conv_in", "tree_rows"],
        outputNames: ["state_out", "conv_tail"], source: splitUnrolled(unrolledTreeSource),
        ensureRowContiguous: true)

    /// The tree replays with the state scan starting at path row `t_skip[0]`: `state_in` is
    /// then the state after the path's first `t_skip[0]` rows (the verify's snapshot), and the
    /// conv tail still follows the whole path. The steps that run are the record's.
    private static func skipping(_ text: String, unrolled: Bool) -> String? {
        let (from, to) = unrolled
            ? ("const float g = gs[t];", "if (t < uint(t_skip[0])) continue;\nconst float g = gs[t];")
            : ("for (uint t = 0; t < T; ++t) {", "for (uint t = uint(t_skip[0]); t < T; ++t) {")
        guard text.components(separatedBy: from).count == 2 else { return nil }
        return text.replacingOccurrences(of: from, with: to)
    }

    private static func skipKernel(_ name: String, _ text: String, unrolled: Bool, split: Bool)
        -> MLXFast.MLXFastKernel?
    {
        skipping(text, unrolled: unrolled).map {
            MLXFast.metalKernel(
                name: name + "_skip",
                inputNames: ["t_skip", "k", "delta", "state_in"] + (split ? ["conv_state"] : [])
                    + ["conv_in", "tree_rows"],
                outputNames: ["state_out", "conv_tail"], source: $0, ensureRowContiguous: true)
        }
    }

    private static let skipTreeKernel = skipKernel(
        "qwen35_gdn_delta_tree_replay", treeSource, unrolled: false, split: false)
    private static let skipUnrolledTreeKernel = skipKernel(
        "qwen35_gdn_delta_tree_replay_unrolled", unrolledTreeSource, unrolled: true, split: false)
    private static let skipSplitTreeKernel = skipKernel(
        "qwen35_gdn_delta_tree_replay_split", splitTreeSource, unrolled: false, split: true)
    private static let skipUnrolledSplitTreeKernel = skipKernel(
        "qwen35_gdn_delta_tree_replay_unrolled_split", splitUnrolled(unrolledTreeSource),
        unrolled: true, split: true)

    /// `convState` set: `convInput` is the verify window alone (`Qwen35FusedGDNVerify` without
    /// its conv copy) and `convState` the three rows before it. `skip` > 0 (tree rows only):
    /// `state` holds the state after the first `skip` path rows, and the scan runs the rest.
    static func run(
        committedRows: Int, k: MLXArray, delta: MLXArray, state: MLXArray, convInput: MLXArray,
        treeRows: MLXArray? = nil, stateIsRowContiguous: Bool = false, unrolled: Bool? = nil,
        convState: MLXArray? = nil, skip: Int = 0
    ) -> (MLXArray, MLXArray)? {
        guard (1...16).contains(committedRows), (0 ..< committedRows).contains(skip),
            skip == 0 || treeRows != nil,
            k.ndim == 4, k.dim(0) == 1, (1...16).contains(k.dim(1)), k.dim(3) == 128,
            state.ndim == 4, state.dim(0) == 1, state.dim(2) == 128,
            delta.shape == [1, k.dim(1), state.dim(1), 129],
            delta.dtype == .float32, k.dtype == .float32,
            state.dim(3) == 128, state.dtype == .float32
        else { return nil }
        // D2: producer layout metadata is valid before GPU completion.
        // Unknown views still need observed row-contiguous backing, or decline.
        if !stateIsRowContiguous {
            guard let stateInfo = try? state.evaluatedBufferInfo(), stateInfo.isRowContiguous
            else { return nil }
        }
        let inputs: [MLXArray]
        let selected: MLXFast.MLXFastKernel
        let unroll = unrolled ?? unrolledPassed
        if let convState {
            guard convState.shape == [1, 3, convInput.dim(2)], convState.dtype == .float32,
                convInput.dim(1) == k.dim(1), convInput.dtype == .float32 else { return nil }
        }
        if let treeRows {
            guard treeRows.dtype == .int32, treeRows.ndim == 1,
                treeRows.size == committedRows else { return nil }
            let rest = convState.map { [k, delta, state, $0, convInput, treeRows] }
                ?? [k, delta, state, convInput, treeRows]
            if skip > 0 {
                guard let skipped = convState == nil
                    ? (unroll ? skipUnrolledTreeKernel : skipTreeKernel)
                    : (unroll ? skipUnrolledSplitTreeKernel : skipSplitTreeKernel)
                else { return nil }
                inputs = [HostConstants.int32([Int32(skip)])] + rest
                selected = skipped
            } else {
                inputs = rest
                selected = convState == nil
                    ? (unroll ? unrolledTreeKernel : treeKernel)
                    : (unroll ? unrolledSplitTreeKernel : splitTreeKernel)
            }
        } else {
            let rows = HostConstants.int32([Int32(committedRows)])
            let lead = unroll ? [k, delta, state] : [rows, k, delta, state]
            inputs = lead + (convState.map { [$0, convInput] } ?? [convInput])
            selected = convState == nil
                ? (unroll ? unrolledKernel : kernel)
                : (unroll ? unrolledSplitKernel : splitKernel)
        }
        var template: [(String, Int)] = [("Dk", 128), ("Dv", 128), ("Hk", k.dim(2)), ("Hv", state.dim(1))]
        if unroll { template.append(("NSTEP", committedRows)) }
        let result = selected(
            inputs,
            template: template,
            grid: (1024, 1, state.dim(1)), threadGroup: (128, 1, 1),
            outputShapes: [state.shape, [1, 3, convInput.dim(2)]],
            outputDTypes: [.float32, .float32])
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
    private var fusedInputSourceSignature: [MLXArray]?
    private var fusedInputPermanentlyIneligible = false
    private let promptGateWeights = Qwen35PromptGateWeights()

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
        prompt: [MLXArray]? = nil,
        promptGates: MLXArray? = nil
    ) -> (
        qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
    ) {
        guard prepareFusedInputProjection(), let fusedInProj else {
            if prompt == nil, let boundary,
                Qwen35DenseGateProjection.eligibleRows(inputs, b: inProjB, a: inProjA) != nil,
                let fused = HadamardQuantizedLinear.boundaryProjectionsWithDenseBA(
                    boundary, [inProjQKV, inProjZ], x: inputs, wb: inProjB.weight, wa: inProjA.weight)
            {
                return (
                    fused.outputs[0], fused.outputs[1].reshaped(B, S, numVHeads, headVDim),
                    fused.b, fused.a)
            }
            let shared = prompt ?? boundary.flatMap {
                HadamardQuantizedLinear.boundaryProjections($0, [inProjQKV, inProjZ])
            } ?? (S >= 128
                ? HadamardQuantizedLinear.sharedPromptProjections(inputs, [inProjQKV, inProjZ])
                : HadamardQuantizedLinear.sharedShortProjections(inputs, [inProjQKV, inProjZ]))
            let qkv = shared?[0] ?? inProjQKV(inputs)
            let z = (shared?[1] ?? inProjZ(inputs)).reshaped(B, S, numVHeads, headVDim)
            let ba = projectBA(inputs, promptGates: promptGates)
            return (qkv, z, ba.b, ba.a)
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

    /// b and a beside the unfused input projections: the halves of the
    /// prompt gate GEMM when it ran, else the dense or the packed projections.
    private func projectBA(_ inputs: MLXArray, promptGates: MLXArray?) -> (b: MLXArray, a: MLXArray) {
        if let promptGates {
            return (promptGates[0..., 0..., 0 ..< numVHeads], promptGates[0..., 0..., numVHeads...])
        }
        if let ba = Qwen35DenseGateProjection.apply(inputs, b: inProjB, a: inProjA) { return ba }
        return (inProjB(inputs), inProjA(inputs))
    }

    /// `projectInputs` on the verify boundary with z out of the qkv launch
    /// (`VerifySplitZ`): z feeds only the gated norm, so `z(dependency)`
    /// makes it after `dependency` (the recurrence) and its launch runs beside
    /// the recurrence instead of before it. The same values, bit for bit; nil
    /// wherever `projectInputs` would not take the boundary route with the
    /// dense b/a.
    private func projectInputsSplittingZ(
        _ inputs: MLXArray, B: Int, S: Int, boundary: HadamardQuantizedLinear.VerifyBoundary?
    ) -> (qkv: MLXArray, b: MLXArray, a: MLXArray, z: (MLXArray) -> MLXArray?)? {
        guard let boundary, !prepareFusedInputProjection(),
            Qwen35DenseGateProjection.eligibleRows(inputs, b: inProjB, a: inProjA) != nil,
            let split = HadamardQuantizedLinear.boundarySplitProjections(
                boundary, lead: inProjQKV, deferred: inProjZ, x: inputs,
                wb: inProjB.weight, wa: inProjA.weight),
            let ba = split.b.flatMap({ b in split.a.map { (b, $0) } })
                ?? Qwen35DenseGateProjection.apply(inputs, b: inProjB, a: inProjA)
        else { return nil }
        let heads = numVHeads, width = headVDim
        return (split.lead, ba.0, ba.1, { split.deferred($0)?.reshaped(B, S, heads, width) })
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

    private func updateRecurrence(
        q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray, state: MLXArray?, mask: MLXArray?
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
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                state: floatState)
        }
        let beta = sigmoid(b).asType(.float32)
        let g = computeGatedDeltaG(aLog, a, dtBias)
        let initial = state ?? MLXArray.zeros(
            [q.dim(0), numVHeads, headVDim, headKDim], dtype: .float32)
        let floatState = initial.dtype == .float32 ? initial : initial.asType(.float32)
        return Qwen35RegisterGDN.run(
            q: q, k: k, v: v, g: g, beta: beta, state: floatState)
    }

    private func updateRecurrenceStateOnly(
        q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray, state: MLXArray?, mask: MLXArray?
    ) -> MLXArray {
        if mask == nil, headKDim == 128, headVDim == 128,
            q.dtype == .float32, k.dtype == .float32, v.dtype == .float32,
            q.dim(1) <= 16, a.dtype == .float32, b.dtype == .float32,
            aLog.dtype == .float32, dtBias.dtype == .float32
        {
            let initial = state ?? MLXArray.zeros(
                [q.dim(0), numVHeads, headVDim, headKDim], dtype: .float32)
            let floatState = initial.dtype == .float32 ? initial : initial.asType(.float32)
            return Qwen35RegisterGatedSIMD.runStateOnly(
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                state: floatState)
        }
        return updateRecurrence(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: state, mask: mask
        ).1
    }

    private func canUsePromptMiddle(_ inputs: MLXArray, ssmState: MLXArray) -> Bool {
        guard inputs.ndim == 3 else { return false }
        let S = inputs.dim(1)
        guard inputs.dim(0) == 1, (64 ... 1365).contains(S),
            S <= Qwen35PromptGatedGDN.maximumSteps, inputs.dtype == .float32
        else { return false }
        guard headKDim == 128, headVDim == 128, convKernelSize == 4 else { return false }
        guard !(inProjB is QuantizedLinear), !(inProjA is QuantizedLinear),
            inProjB.bias == nil, inProjA.bias == nil
        else { return false }
        guard inProjB.weight.dtype == .float32, inProjA.weight.dtype == .float32 else { return false }
        guard inProjB.weight.shape == [numVHeads, inputs.dim(2)],
            inProjA.weight.shape == inProjB.weight.shape
        else { return false }
        guard conv1d.bias == nil, conv1d.weight.dtype == .float32 else { return false }
        guard aLog.dtype == .float32, dtBias.dtype == .float32,
            aLog.size == numVHeads, dtBias.size == numVHeads
        else { return false }
        return ssmState.dtype == .float32 && ssmState.shape == [1, numVHeads, headVDim, headKDim]
    }

    /// The prompt-width (S >= 64, one row) middle of `cbv2Forward`: the
    /// current Q/K math is unchanged; eligible fresh scans fold V convolution.
    /// If the stacked b/a matmul
    /// does not preserve each column's bits, an eligible fresh scan uses the
    /// separate b/a values already made by `projectInputs`.
    /// An eligible empty B1 state uses the FP32 fresh8 matrix scan. Other
    /// requests keep `Qwen35PromptGatedGDN` and its serial operation order.
    /// The retained convolution tail is the last three QKV rows. Nil
    /// when the shapes, dtypes or layers do not qualify.
    private func promptMiddle(
        _ inputs: MLXArray, qkv: MLXArray, a: MLXArray, b: MLXArray,
        convState: MLXArray, ssmState: MLXArray,
        freshState: Bool = false, promptGates: MLXArray? = nil
    ) -> (out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray)? {
        let S = qkv.dim(1)
        guard promptGates != nil || canUsePromptMiddle(inputs, ssmState: ssmState) else { return nil }
        let joined = promptGates ?? promptGateWeights.get(b: inProjB.weight, a: inProjA.weight)
            .map { matmul(inputs, $0.T) }
        guard joined != nil || (freshState && S % 8 == 0 && numKHeads == 16 && numVHeads == 32
            && a.dtype == .float32 && b.dtype == .float32
            && a.shape == [1, S, numVHeads] && b.shape == a.shape)
        else { return nil }
        let ba = joined ?? concatenated([b, a], axis: -1)
        if freshState && S % 8 == 0 && numKHeads == 16 && numVHeads == 32 {
            let foldState = convState.contiguous(), foldQKV = qkv.contiguous()
            let foldWeight = conv1d.weight.contiguous()
            if let qk = Qwen35PromptConvQKNorm.apply(
                    state: foldState, qkv: foldQKV, weight: foldWeight,
                    keyHeads: numKHeads, valueHeads: numVHeads, materializeValue: false),
                let folded = Qwen35Fresh8PromptGDN.runConvolvedValue(
                    q: qk[0], k: qk[1], convState: foldState, qkv: foldQKV, weight: foldWeight,
                    ba: ba, aLog: aLog, dtBias: dtBias)
            {
                return (folded.0, contiguous(qkv[0..., (S - 3)...]), folded.1)
            }
        }
        guard let qkvOut = Qwen35PromptConvQKNorm.apply(
                state: convState, qkv: qkv, weight: conv1d.weight,
                keyHeads: numKHeads, valueHeads: numVHeads)
        else { return nil }
        let q = qkvOut[0], k = qkvOut[1], v = qkvOut[2]
        let fresh = freshState ? Qwen35Fresh8PromptGDN.run(
            q: q, k: k, v: v, ba: ba, aLog: aLog, dtBias: dtBias) : nil
        // A declined stack check can use only the eligible fresh scan.
        // If that scan declines, retain the exact processChunk fallback.
        guard joined != nil || fresh != nil else { return nil }
        let (out, newSsmState) = fresh ?? Qwen35PromptGatedGDN.run(
            q: q, k: k, v: v, ba: ba, aLog: aLog, dtBias: dtBias, state: ssmState)
        return (out, contiguous(qkv[0..., (S - 3)...]), newSsmState)
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
        mask: MLXArray?
    ) -> (out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray) {
        let B = qkv.dim(0)
        let S = qkv.dim(1)

        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = retainedConvTail(of: convInput, keeping: nKeep, chunkWidth: S)
        let foldShape = mask == nil && S >= 64 && headKDim == 128 && headVDim == 128
        let foldInput = foldShape && convKernelSize == 4 && convInput.dtype == .float32
        let foldWeight = foldInput && conv1d.weight.dtype == .float32
            && conv1d.weight.shape == [convDim, 4, 1]
        let foldValue = foldWeight && conv1d.bias == nil
            && aLog.dtype == .float32 && dtBias.dtype == .float32
        let convOut = foldValue
            ? silu(MLX.conv1d(
                convInput[0..., 0..., 0 ..< (2 * keyDim)],
                conv1d.weight[0 ..< (2 * keyDim), 0..., 0...], groups: 2 * keyDim))
            : silu(conv1d(convInput))
        let (qNormed, kNormed) = normalizedQK(convOut, B: B, S: S)
        let out: MLXArray
        let newSsmState: MLXArray
        if foldValue {
            let beta = sigmoid(b).asType(.float32)
            let g = computeGatedDeltaG(aLog, a, dtBias)
            let initial = ssmState ?? MLXArray.zeros(
                [B, numVHeads, headVDim, headKDim], dtype: .float32)
            (out, newSsmState) = Qwen35RegisterGDN.runConvolvedValue(
                q: qNormed, k: kNormed, input: convInput, weight: conv1d.weight,
                g: g, beta: beta, state: initial.asType(.float32))
        } else {
            let v = convOut[0..., 0..., (2 * keyDim)...].reshaped(B, S, numVHeads, headVDim)
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
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = retainedConvTail(of: convInput, keeping: nKeep, chunkWidth: S)
        let convOut = silu(conv1d(convInput))

        let (qNormed, kNormed) = normalizedQK(convOut, B: B, S: S)
        let v = convOut[0..., 0..., (2 * keyDim)...].reshaped(B, S, numVHeads, headVDim)

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
              tape.convStateRows == convKernelSize - 1
        else { return false }

        let batch = tape.k.dim(0)
        let windowRows = tape.convState == nil ? tape.convStateRows + tape.rowCount : tape.rowCount
        // Delta tapes may omit q/v when a boundary state lacks a producer layout
        // certificate. Check the backing, as Qwen35DeltaReplay.run already does.
        var deltaStateIsRowContiguous = tape.ssmPreIsRowContiguous
        if tape.delta != nil && !deltaStateIsRowContiguous, let state = tape.ssmPre {
            deltaStateIsRowContiguous = (try? state.evaluatedBufferInfo())?.isRowContiguous == true
            if ProcessInfo.processInfo.environment["GDN_REPLAY_LAYOUT_DIAGNOSTIC"] == "1" {
                FileHandle.standardError.write(Data("replay layout observed: contiguous=\(deltaStateIsRowContiguous) q=\(tape.q.shape) v=\(tape.v.shape) state=\(state.shape) rows=\(committedRows)/\(tape.rowCount)\n".utf8))
            }
        }
        guard tape.convInput.shape3 == (batch, windowRows, convDim),
              tape.convState.map({ $0.shape3 == (batch, tape.convStateRows, convDim) }) ?? true,
              tape.k.shape4 == (batch, tape.rowCount, numKHeads, headKDim),
              tape.a.shape3 == (batch, tape.rowCount, numVHeads),
              tape.b.shape3 == (batch, tape.rowCount, numVHeads),
              (tape.delta != nil && deltaStateIsRowContiguous) || (
                  tape.q.shape4 == (batch, tape.rowCount, numKHeads, headKDim) &&
                  tape.v.shape4 == (batch, tape.rowCount, numVHeads, headVDim))
        else { return false }

        if let ssmPre = tape.ssmPre,
           ssmPre.shape4 != (batch, numVHeads, headVDim, headKDim)
        {
            return false
        }
        if let mask = tape.mask,
           mask.shape2 != (batch, tape.rowCount)
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

    /// `treeStateOut`: the tree verify's `state_out`, which holds its snapshot on a branching
    /// descriptor (`Qwen35SnapshotCommit`).
    private func replayedPrefixState(
        tape: ArraysCache.PrefixReplayTape, committedRows: Int, treeStateOut: MLXArray? = nil
    ) -> CBv2RecurrentLayerState {
        precondition(
            canReplayPrefix(tape: tape, committedRows: committedRows),
            "Qwen35 invalid compact recurrent prefix replay")
        guard let treeStateOut, Qwen35SnapshotCommit.current() != .off,
            let snapshot = snapshotCommitState(tape: tape, committedRows: committedRows, snapshot: treeStateOut)
        else { return recordReplayedPrefixState(tape: tape, committedRows: committedRows) }
        if Qwen35SnapshotCommit.mode == .on { return snapshot.state }
        let record = recordReplayedPrefixState(tape: tape, committedRows: committedRows)
        Qwen35SnapshotCommit.note(record: record, snapshot: snapshot.state, ends: snapshot.ends)
        return record
    }

    private func snapshotCommitState(
        tape: ArraysCache.PrefixReplayTape, committedRows: Int, snapshot: MLXArray
    ) -> (state: CBv2RecurrentLayerState, ends: Bool)? {
        guard let parents = CBv2BranchVerify.committedParents, let delta = tape.delta,
            tape.ssmPreIsRowContiguous, tape.mask == nil, tape.rowCount == 16, tape.convStateRows == 3,
            headKDim == 128, headVDim == 128, tape.convInput.dtype == .float32, tape.k.dtype == .float32,
            snapshot.shape == [1, numVHeads, headVDim, headKDim], snapshot.dtype == .float32
        else { return nil }
        let path = CBv2BranchVerify.committedRows ?? (0 ..< Int32(committedRows)).map { $0 }
        guard path.count == committedRows,
            let rows = CBv2BranchVerify.snapshotPrefix(parents: parents, path: path)?.rows
        else { return nil }
        if rows == committedRows {
            let conv = Qwen35SnapshotCommit.convTail(path: path, convInput: tape.convInput, convState: tape.convState)
            return (CBv2RecurrentLayerState(conv: conv, ssm: snapshot, ssmIsRowContiguous: true), true)
        }
        guard let replay = Qwen35DeltaReplay.run(committedRows: committedRows, k: tape.k, delta: delta,
            state: snapshot, convInput: tape.convInput, treeRows: HostConstants.recentInt32(path),
            stateIsRowContiguous: true, convState: tape.convState, skip: rows)
        else { return nil }
        return (CBv2RecurrentLayerState(conv: replay.1, ssm: replay.0, ssmIsRowContiguous: true), false)
    }

    private func recordReplayedPrefixState(
        tape: ArraysCache.PrefixReplayTape, committedRows: Int
    ) -> CBv2RecurrentLayerState {
        if CBv2BranchVerify.committedRowsIsBranch,
            let path = CBv2BranchVerify.committedRows, path.count == committedRows,
            let pathArray = CBv2BranchVerify.committedRowsArray
        {
            guard tape.mask == nil, headKDim == 128, headVDim == 128, tape.rowCount == 16,
                tape.convStateRows == 3, tape.convInput.dtype == .float32,
                tape.q.dtype == .float32, tape.k.dtype == .float32, tape.v.dtype == .float32,
                tape.a.dtype == .float32, tape.b.dtype == .float32,
                aLog.dtype == .float32, dtBias.dtype == .float32, let ssmPre = tape.ssmPre
            else { preconditionFailure("Qwen35 tree commit: path replay unavailable") }
            if let delta = tape.delta,
                let replay = Qwen35DeltaReplay.run(committedRows: committedRows, k: tape.k, delta: delta,
                    state: ssmPre.dtype == .float32 ? ssmPre : ssmPre.asType(.float32),
                    convInput: tape.convInput, treeRows: pathArray,
                    stateIsRowContiguous: tape.ssmPreIsRowContiguous, convState: tape.convState)
            { return CBv2RecurrentLayerState(
                conv: replay.1, ssm: replay.0, ssmIsRowContiguous: true) }
            precondition(tape.convState == nil, "Qwen35 tree commit: split conv window needs the delta replay")
            guard let replayed = Qwen35RegisterGatedSIMD.runTreeReplay(
                    q: tape.q, k: tape.k, v: tape.v, a: tape.a, b: tape.b, aLog: aLog, dtBias: dtBias,
                    state: ssmPre.dtype == .float32 ? ssmPre : ssmPre.asType(.float32),
                    convInput: tape.convInput, rows: pathArray)
            else { preconditionFailure("Qwen35 tree commit: path replay unavailable") }
            return CBv2RecurrentLayerState(
                conv: replayed.1, ssm: replayed.0, ssmIsRowContiguous: true)
        }
        let rows = 0 ..< committedRows
        // One kernel reads the tape in place (no copy of the strided value
        // slice) and writes the boundary conv rows: the bits of the path below.
        if tape.mask == nil, headKDim == 128, headVDim == 128, committedRows <= 16,
            tape.convStateRows == 3, tape.convInput.dtype == .float32,
            tape.q.dtype == .float32, tape.k.dtype == .float32, tape.v.dtype == .float32,
            tape.a.dtype == .float32, tape.b.dtype == .float32,
            aLog.dtype == .float32, dtBias.dtype == .float32
        {
            let initial = tape.ssmPre ?? MLXArray.zeros(
                [tape.k.dim(0), numVHeads, headVDim, headKDim], dtype: .float32)
            let floatState = initial.dtype == .float32 ? initial : initial.asType(.float32)
            if let delta = tape.delta,
                let replay = Qwen35DeltaReplay.run(
                    committedRows: committedRows, k: tape.k, delta: delta,
                    state: floatState, convInput: tape.convInput,
                    stateIsRowContiguous: tape.ssmPreIsRowContiguous, convState: tape.convState)
            { return CBv2RecurrentLayerState(
                conv: replay.1, ssm: replay.0, ssmIsRowContiguous: true) }
            precondition(tape.convState == nil, "Qwen35 commit: split conv window needs the delta replay")
            let (boundarySsm, boundaryConv) = Qwen35RegisterGatedSIMD.runReplay(
                q: tape.q[0..., rows, 0...],
                k: tape.k,
                v: tape.v,
                a: tape.a,
                b: tape.b,
                aLog: aLog,
                dtBias: dtBias,
                state: floatState,
                convInput: tape.convInput)
            return CBv2RecurrentLayerState(
                conv: boundaryConv, ssm: boundarySsm, ssmIsRowContiguous: true)
        }
        let boundarySsm = updateRecurrenceStateOnly(
            q: tape.q[0..., rows, 0...],
            k: tape.k[0..., rows, 0...],
            v: tape.v[0..., rows, 0...],
            a: tape.a[0..., rows, 0...],
            b: tape.b[0..., rows, 0...],
            aLog: aLog,
            dtBias: dtBias,
            state: tape.ssmPre,
            mask: tape.mask.map { $0[0..., rows] }
        )
        let boundaryConvView = tape.convInput[
            0...,
            committedRows ..< (committedRows + tape.convStateRows),
            0...]
        let boundaryConv = contiguous(boundaryConvView)
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
                mask: mask
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
        promptProjections: [MLXArray]? = nil,
        promptZ: ((MLXArray) -> MLXArray?)? = nil,
        fold: PromptResidualFold? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        precondition(recurrentState.count == B, "Qwen35 CBv2 recurrent row count mismatch")

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
        let promptGates = canUsePromptMiddle(inputs, ssmState: ssmState)
            ? promptGateWeights.prepared(b: inProjB.weight, a: inProjA.weight).map { matmul(inputs, $0.T) }
            : nil
        // With `promptZ`, `promptProjections` holds qkv only and z is made
        // after the recurrence, so its launch runs beside it (`PromptSplitZ`).
        let qkv: MLXArray, projectedZ: MLXArray?, b: MLXArray, a: MLXArray
        if promptZ != nil, !prepareFusedInputProjection(), let lead = promptProjections?.first {
            let ba = projectBA(inputs, promptGates: promptGates)
            (qkv, projectedZ, b, a) = (lead, nil, ba.b, ba.a)
        } else {
            let projected = projectInputs(
                inputs, B: B, S: S, prompt: promptProjections, promptGates: promptGates)
            (qkv, projectedZ, b, a) = (projected.qkv, projected.z, projected.b, projected.a)
        }
        let middle = promptMiddle(
            inputs, qkv: qkv, a: a, b: b, convState: convState, ssmState: ssmState,
            freshState: B == 1 && recurrentState[0].inputState(modelLayerIndex: modelLayerIndex) == nil,
            promptGates: promptGates)
        let (out, newConvState, newSsmState) = middle ?? processChunk(
                qkv: qkv, a: promptGates == nil ? a : inProjA(inputs),
                b: promptGates == nil ? b : inProjB(inputs),
                convState: convState, ssmState: ssmState, mask: nil)

        if B == 1 {
            do {
                try recurrentState[0].stage(
                    modelLayerIndex: modelLayerIndex,
                    conv: newConvState,
                    ssm: newSsmState, ssmIsRowContiguous: middle != nil)
            } catch {
                preconditionFailure(
                    "Qwen35 CBv2 recurrent stage failed at layer \(modelLayerIndex): \(error)")
            }
        } else {
            for (row, evaluation) in recurrentState.enumerated() {
                do {
                    try evaluation.stage(
                        modelLayerIndex: modelLayerIndex,
                        conv: newConvState[row ..< row + 1],
                        ssm: newSsmState[row ..< row + 1], ssmIsRowContiguous: middle != nil)
                } catch {
                    preconditionFailure(
                        "Qwen35 CBv2 recurrent stage failed at layer \(modelLayerIndex): \(error)")
                }
            }
        }

        let z = projectedZ ?? (promptZ.flatMap { $0(out) } ?? inProjZ(inputs))
            .reshaped(B, S, numVHeads, headVDim)
        if out.ndim == 4, out.dim(-1) == 128, S >= 64,
            let packed = outProj as? HadamardQuantizedLinear,
            let gated = packed.callAfterGatedNormPrompt(
                out, gate: z, normWeight: norm.weight, eps: norm.eps, fold: fold)
        {
            return gated
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
        boundary: HadamardQuantizedLinear.VerifyBoundary? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        precondition(recurrentState.count == B, "Qwen35 CBv2 recurrent row count mismatch")
        precondition(S >= 1, "Qwen35 capture-verify window must be non-empty")

        let qkv: MLXArray
        let projectedZ: MLXArray?
        var deferredZ: ((MLXArray) -> MLXArray?)?
        let b: MLXArray
        let a: MLXArray
        if exactTargetVerify {
            let exact = qwen35A3BExactW4G64ProjectionQuad(
                inProjQKV, inProjZ, inProjB, inProjA, inputs)
            qkv = exact.0
            projectedZ = exact.1.reshaped(B, S, numVHeads, headVDim)
            b = exact.2
            a = exact.3
        } else if let split = projectInputsSplittingZ(inputs, B: B, S: S, boundary: boundary) {
            (qkv, b, a) = (split.qkv, split.b, split.a)
            projectedZ = nil
            deferredZ = split.z
        } else {
            // Preserve main's fused GDN projection construction and graph.
            let projected = projectInputs(inputs, B: B, S: S, boundary: boundary)
            (qkv, b, a) = (projected.qkv, projected.b, projected.a)
            projectedZ = projected.z
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
        // Verify windows of 3 ... 8 rows: conv, SiLU, q/k norm and the scan
        // in one kernel (`Qwen35FusedGDNVerify`), the same bits.
        let tree = S == 16 && B == 1 ? CBv2BranchVerify.tree : nil
        let fusedMiddle = exactTargetVerify || conv1d.bias != nil || convKernelSize != 4
            || headKDim != 128 || headVDim != 128
            ? nil : (tree.map {
                Qwen35FusedGDNVerify.applyTree(
                    convState: convState, qkv: qkv, convWeight: conv1d.weight, a: a, b: b,
                    aLog: aLog, dtBias: dtBias, state: ssmState, tree: $0,
                    keyHeads: numKHeads, valueHeads: numVHeads)
            } ?? Qwen35FusedGDNVerify.apply(
                convState: convState, qkv: qkv, convWeight: conv1d.weight, a: a, b: b,
                aLog: aLog, dtBias: dtBias, state: ssmState,
                keyHeads: numKHeads, valueHeads: numVHeads))
        precondition(tree == nil || fusedMiddle != nil, "Qwen35 tree verify: fused GDN tree kernel unavailable")
        let fusedConv = fusedMiddle != nil || exactTargetVerify || conv1d.bias != nil
            || convKernelSize != 4
            ? nil : Qwen35FusedConvSilu.apply(state: convState, qkv: qkv, weight: conv1d.weight)
        let convInput = fusedMiddle?.0 ?? fusedConv?.0 ?? concatenated([convState, qkv], axis: 1)
        let convOut: MLXArray
        if let fusedMiddle {
            convOut = fusedMiddle.1
        } else if let fusedConv {
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

        let (qNormed, kNormed) = fusedMiddle.map { ($0.2, $0.3) }
            ?? normalizedQK(convOut, B: B, S: S)
        let v = fusedMiddle?.1
            ?? convOut[0..., 0..., (2 * keyDim)...].reshaped(B, S, numVHeads, headVDim)

        let out: MLXArray
        if S >= 3 {
            let recurrence = fusedMiddle.map { ($0.4, $0.5) } ?? updateRecurrence(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: ssmState,
                mask: nil)
            out = recurrence.0
            let finalSsmState = recurrence.1

            // The fused path has one row; capture only its tail buffer.
            let fusedTail = fusedMiddle?.6
            let finalSSMIsRowContiguous = fusedMiddle != nil
            let capturedDelta = fusedMiddle.flatMap { $0.7 }
            // The dead-output verify kernel hands back the window rows alone, not a copy with
            // the three conv state rows in front.
            let splitConvWindow = capturedDelta != nil && convInput.dim(1) == S
            for (row, evaluation) in recurrentState.enumerated() {
                let finalConv: MLXArray
                let finalSSM: MLXArray
                let tape: ArraysCache.PrefixReplayTape
                let convOutBacking: MLXArray
                if B == 1 {
                    finalConv = fusedTail ?? convInput[0..., S ..< (S + nKeep), 0...]
                    finalSSM = finalSsmState
                    tape = ArraysCache.PrefixReplayTape(
                        convInput: convInput,
                        q: qNormed,
                        k: kNormed,
                        v: v,
                        a: a,
                        b: b,
                        ssmPre: ssmState,
                        mask: nil,
                        rowCount: S,
                        convStateRows: nKeep,
                        delta: capturedDelta,
                        ssmPreIsRowContiguous: evaluation.inputState(
                            modelLayerIndex: modelLayerIndex)?.ssmIsRowContiguous == true,
                        convState: splitConvWindow ? convState : nil)
                    convOutBacking = convOut
                } else {
                    let rowRange = row ..< (row + 1)
                    finalConv = convInput[rowRange, S ..< (S + nKeep), 0...]
                    finalSSM = finalSsmState[rowRange]
                    tape = ArraysCache.PrefixReplayTape(
                        convInput: convInput[rowRange],
                        q: qNormed[rowRange],
                        k: kNormed[rowRange],
                        v: v[rowRange],
                        a: a[rowRange],
                        b: b[rowRange],
                        ssmPre: ssmState[rowRange],
                        mask: nil,
                        rowCount: S,
                        convStateRows: nKeep,
                        delta: capturedDelta.map { $0[rowRange] })
                    convOutBacking = convOut[rowRange]
                }
                var roots: [MLXArray]
                if let delta = tape.delta, tape.ssmPreIsRowContiguous {
                    roots = [tape.convInput, tape.k, delta]
                    if let convState = tape.convState { roots.append(convState) }
                } else {
                    roots = [
                        tape.convInput, tape.q, tape.k, convOutBacking, tape.a, tape.b,
                    ]
                    if let delta = tape.delta { roots.append(delta) }
                }
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
                        ssmIsRowContiguous: finalSSMIsRowContiguous,
                        materializedByteCount: materializedBytes,
                        evaluationRoots: strictReplayRoots,
                        strictReplayRetainedByteCount: strictReplayRetainedBytes,
                        strictReplayRetainedRoots: strictReplayRoots,
                        fullAcceptanceRetainedByteCount: fullAcceptanceRetainedBytes,
                        fullAcceptanceRetainedRoots: [tape.convInput],
                        fullAcceptance: {
                            // The fused kernel wrote these bits to their own
                            // buffer; otherwise add zero to detach the tail.
                            let detachedConv = fusedTail ?? finalConv + MLXArray.zeros(
                                finalConv.shape, dtype: finalConv.dtype)
                            return CBv2RecurrentLayerState(
                                conv: detachedConv, ssm: finalSSM,
                                ssmIsRowContiguous: finalSSMIsRowContiguous)
                        },
                        replay: { [unowned self] keepPositions in
                            self.replayedPrefixState(
                                tape: tape, committedRows: keepPositions,
                                treeStateOut: tree != nil && B == 1 ? finalSSM : nil)
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
                    mask: nil)
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
        let z = projectedZ ?? deferredZ.flatMap { $0(out) } ?? projectInputs(
            inputs, B: B, S: S, boundary: boundary).z
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

/// Verify-width attention q/k preparation in one kernel: MLX `RMSNorm` of
/// each 256-wide q head (read in place from the gated query projection) and
/// k head, then MLX `RoPE` (non-traditional, `ROPE_DIMS` leading dims, array
/// offset) into the `[1, heads, L, 256]` layout the attention takes. One
/// simdgroup per (head, row): lane l holds dims 4l ... 4l + 3 of each half,
/// the layout of MLX's `rms_single_row` (64 threads × 4 reads, two simdgroup
/// sums), and the rotation partner of dim p < ROPE_DIMS / 2 is in lane
/// l ^ (ROPE_DIMS / 8). Outputs are the bits of `rmsNorm` + `transposed` +
/// `RoPE` as MLX's compiled kernels form them.
enum Qwen35VerifyQKNormRoPE {
    private static let source = """
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            constexpr uint D = 256;
            constexpr uint HALF_ROT = uint(ROPE_DIMS) / 2u;
            const uint lane = thread_index_in_simdgroup;
            const uint block = threadgroup_position_in_grid.z;
            const uint L = LAST_QUERY
                ? (threadgroups_per_grid.z - uint(Q_HEADS)) / uint(KV_HEADS)
                : threadgroups_per_grid.z;
            const uint head = LAST_QUERY ? block : threadgroup_position_in_grid.y;
            const bool is_q = head < uint(Q_HEADS);
            const uint h = LAST_QUERY && !is_q ? (head - uint(Q_HEADS)) / L
                : (is_q ? head : head - uint(Q_HEADS));
            const uint t = LAST_QUERY ? (is_q ? L - 1u : (head - uint(Q_HEADS)) % L) : block;
            const uint qt = LAST_QUERY ? 0u : t;
            const device float* src = is_q
                ? q_proj + (qt * uint(Q_HEADS) + h) * (2u * D)
                : k_proj + (t * uint(KV_HEADS) + h) * D;
            const device float* w = is_q ? q_weight : k_weight;
            const float epsilon = eps[is_q ? 0 : 1];
            float x[2][4];
            float acc[2] = {0.0f, 0.0f};
            for (uint part = 0; part < 2u; ++part) {
              for (uint j = 0; j < 4u; ++j) {
                x[part][j] = src[part * 128u + lane * 4u + j];
                acc[part] = metal::fma(x[part][j], x[part][j], acc[part]);
              }
            }
            const float total = simd_sum(acc[0]) + simd_sum(acc[1]);
            const float inv = metal::precise::rsqrt(total / float(D) + epsilon);
            float y[2][4];
            for (uint part = 0; part < 2u; ++part) {
              for (uint j = 0; j < 4u; ++j) {
                y[part][j] = w[part * 128u + lane * 4u + j] * (x[part][j] * inv);
              }
            }
            const float position = 1.0f * static_cast<float>(int(t) + offset[0]);
            const float base = eps[2];
            for (uint j = 0; j < 4u; ++j) {
              const float partner = simd_shuffle_xor(y[0][j], ushort(HALF_ROT / 4u));
              const uint dim = lane * 4u + j;
              if (dim < uint(ROPE_DIMS)) {
                const uint p = dim < HALF_ROT ? dim : dim - HALF_ROT;
                const float d = static_cast<float>(p) / static_cast<float>(HALF_ROT);
                const float inv_freq = metal::exp2(-d * base);
                const float theta = position * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                y[0][j] = dim < HALF_ROT
                    ? metal::fma(y[0][j], costheta, -(partner * sintheta))
                    : metal::fma(partner, sintheta, y[0][j] * costheta);
              }
            }
            const uint outL = LAST_QUERY && is_q ? 1u : L;
            const uint outT = LAST_QUERY && is_q ? 0u : t;
            device float* dst = (is_q ? q_out : k_out) + (h * outL + outT) * D;
            #pragma clang loop unroll(full)
            for (uint part = 0; part < 2u; ++part) {
              *((device float4*)(dst + part * 128u + lane * 4u)) =
                  float4(y[part][0], y[part][1], y[part][2], y[part][3]);
            }
            """

    private static let inputNames = ["q_proj", "k_proj", "q_weight", "k_weight", "eps", "offset"]
    private static let outputNames = ["q_out", "k_out"]

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_verify_qk_norm_rope",
        inputNames: inputNames, outputNames: outputNames, source: source)

    /// The norm and rotation of `source`: the norm weight is read after the
    /// row's simdgroup sum, and the RoPE angle (which depends only on the
    /// position, the offset and the dim) is formed after the norm, so both
    /// wait behind the row loads and the reduction.
    private static let normRopeParent = """
        float x[2][4];
        float acc[2] = {0.0f, 0.0f};
        for (uint part = 0; part < 2u; ++part) {
          for (uint j = 0; j < 4u; ++j) {
            x[part][j] = src[part * 128u + lane * 4u + j];
            acc[part] = metal::fma(x[part][j], x[part][j], acc[part]);
          }
        }
        const float total = simd_sum(acc[0]) + simd_sum(acc[1]);
        const float inv = metal::precise::rsqrt(total / float(D) + epsilon);
        float y[2][4];
        for (uint part = 0; part < 2u; ++part) {
          for (uint j = 0; j < 4u; ++j) {
            y[part][j] = w[part * 128u + lane * 4u + j] * (x[part][j] * inv);
          }
        }
        const float position = 1.0f * static_cast<float>(int(t) + offset[0]);
        const float base = eps[2];
        for (uint j = 0; j < 4u; ++j) {
          const float partner = simd_shuffle_xor(y[0][j], ushort(HALF_ROT / 4u));
          const uint dim = lane * 4u + j;
          if (dim < uint(ROPE_DIMS)) {
            const uint p = dim < HALF_ROT ? dim : dim - HALF_ROT;
            const float d = static_cast<float>(p) / static_cast<float>(HALF_ROT);
            const float inv_freq = metal::exp2(-d * base);
            const float theta = position * inv_freq;
            const float costheta = metal::fast::cos(theta);
            const float sintheta = metal::fast::sin(theta);
            y[0][j] = dim < HALF_ROT
                ? metal::fma(y[0][j], costheta, -(partner * sintheta))
                : metal::fma(partner, sintheta, y[0][j] * costheta);
          }
        }
        """

    /// The same norm and rotation with the position-only work first: the
    /// angle's cos and sin and the norm weight are issued before the row
    /// loads, so they overlap the loads and the reduction. Every expression
    /// is the parent's text, so every output bit is the parent's.
    private static let normRopeHoisted = """
        const float position = 1.0f * static_cast<float>(int(t) + offset[0]);
        const float base = eps[2];
        float cos_j[4];
        float sin_j[4];
        if (lane * 4u < uint(ROPE_DIMS)) {
          #pragma clang loop unroll(full)
          for (uint j = 0; j < 4u; ++j) {
            const uint dim = lane * 4u + j;
            const uint p = dim < HALF_ROT ? dim : dim - HALF_ROT;
            const float d = static_cast<float>(p) / static_cast<float>(HALF_ROT);
            const float inv_freq = metal::exp2(-d * base);
            const float theta = position * inv_freq;
            cos_j[j] = metal::fast::cos(theta);
            sin_j[j] = metal::fast::sin(theta);
          }
        }
        float wv[2][4];
        #pragma clang loop unroll(full)
        for (uint part = 0; part < 2u; ++part) {
          const float4 w4 = *((const device float4*)(w + part * 128u + lane * 4u));
          wv[part][0] = w4.x;
          wv[part][1] = w4.y;
          wv[part][2] = w4.z;
          wv[part][3] = w4.w;
        }
        float x[2][4];
        float acc[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint part = 0; part < 2u; ++part) {
          const float4 x4 = *((const device float4*)(src + part * 128u + lane * 4u));
          x[part][0] = x4.x;
          x[part][1] = x4.y;
          x[part][2] = x4.z;
          x[part][3] = x4.w;
          #pragma clang loop unroll(full)
          for (uint j = 0; j < 4u; ++j) {
            acc[part] = metal::fma(x[part][j], x[part][j], acc[part]);
          }
        }
        const float total = simd_sum(acc[0]) + simd_sum(acc[1]);
        const float inv = metal::precise::rsqrt(total / float(D) + epsilon);
        float y[2][4];
        #pragma clang loop unroll(full)
        for (uint part = 0; part < 2u; ++part) {
          #pragma clang loop unroll(full)
          for (uint j = 0; j < 4u; ++j) {
            y[part][j] = wv[part][j] * (x[part][j] * inv);
          }
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4u; ++j) {
          const float partner = simd_shuffle_xor(y[0][j], ushort(HALF_ROT / 4u));
          const uint dim = lane * 4u + j;
          if (dim < uint(ROPE_DIMS)) {
            const float costheta = cos_j[j];
            const float sintheta = sin_j[j];
            y[0][j] = dim < HALF_ROT
                ? metal::fma(y[0][j], costheta, -(partner * sintheta))
                : metal::fma(partner, sintheta, y[0][j] * costheta);
          }
        }
        """

    private static func replacingOnce(_ text: String, _ old: String, _ new: String) -> String {
        guard text.components(separatedBy: old).count == 2 else {
            preconditionFailure("qwen35 verify q/k norm RoPE: hoist anchor not unique")
        }
        return text.replacingOccurrences(of: old, with: new)
    }

    private static let hoistKernel = MLXFast.metalKernel(
        name: "qwen35_verify_qk_norm_rope_h",
        inputNames: inputNames, outputNames: outputNames,
        source: replacingOnce(source, normRopeParent, normRopeHoisted))

    /// `DARKBLOOM_BONSAI_ROPE_HOIST=0` keeps the parent kernel.
    static let hoisted = Qwen35MTPFusedDraftStep.environmentFlag("DARKBLOOM_BONSAI_ROPE_HOIST")

    /// Tree twin of a kernel text (`source` or its hoisted form; `tree`: the draft tree
    /// descriptor, see `Qwen35FusedGDNVerify.treeSource`): row `t` takes the position
    /// `offset + depth[t]` of its path. Nil when the line is not found.
    static func treeSource(_ record: String) -> String? {
        let line = "const float position = 1.0f * static_cast<float>(int(t) + offset[0]);"
        guard record.components(separatedBy: line).count == 2 else { return nil }
        return record.replacingOccurrences(
            of: line,
            with: "const float position = 1.0f * static_cast<float>(tree[16 + t] + offset[0]);")
    }

    /// Built from the text `apply` launches (`hoisted` picks it), so a path row's
    /// q/k bits are the chain verify's bits under either kernel.
    private static let treeKernel: MLXFast.MLXFastKernel? = treeSource(
        hoisted ? replacingOnce(source, normRopeParent, normRopeHoisted) : source
    ).map {
        MLXFast.metalKernel(
            name: "qwen35_verify_qk_norm_rope_tree",
            inputNames: ["q_proj", "k_proj", "q_weight", "k_weight", "eps", "offset", "tree"],
            outputNames: ["q_out", "k_out"],
            source: $0)
    }

    nonisolated(unsafe) private static let defaultEpsRopeBase: MLXArray = {
        let a = MLXArray([Float(1e-6), Float(1e-6), log2(Float(10_000_000.0))])
        eval(a)
        return a
    }()

    @inline(__always)
    private static func epsRopeBaseArray(queryEps: Float, keyEps: Float, ropeBase: Float) -> MLXArray {
        if queryEps == Float(1e-6) && keyEps == Float(1e-6) && ropeBase == Float(10_000_000.0) {
            return defaultEpsRopeBase
        }
        return MLXArray([queryEps, keyEps, log2(ropeBase)])
    }

    /// `apply` (all query rows) for a draft tree window.
    static func applyTree(
        queryProjection: MLXArray, keyProjection: MLXArray, queryWeight: MLXArray,
        keyWeight: MLXArray, queryEps: Float, keyEps: Float, offsets: MLXArray,
        queryHeads: Int, keyHeads: Int, ropeDims: Int, ropeBase: Float, tree: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let qShape = queryProjection.shape3
        guard qShape.0 == 1,
            (2 ... 16).contains(qShape.1),
            qShape.2 == queryHeads * 512,
            keyProjection.shape3 == (1, qShape.1, keyHeads * 256),
            let treeKernel, tree.dtype == .int32, tree.size >= 288,
            queryProjection.dtype == .float32, keyProjection.dtype == .float32,
            queryWeight.dtype == .float32, keyWeight.dtype == .float32,
            queryWeight.ndim == 1, queryWeight.dim(0) == 256,
            keyWeight.ndim == 1, keyWeight.dim(0) == 256,
            offsets.size == 1, offsets.dtype == .int32,
            ropeDims % 8 == 0, (8 ... 128).contains(ropeDims),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let L = qShape.1
        let epsArray = epsRopeBaseArray(queryEps: queryEps, keyEps: keyEps, ropeBase: ropeBase)
        let outputs = treeKernel(
            [queryProjection, keyProjection, queryWeight, keyWeight,
                epsArray, offsets, tree],
            template: [("Q_HEADS", queryHeads), ("KV_HEADS", keyHeads), ("ROPE_DIMS", ropeDims),
                ("LAST_QUERY", false)],
            grid: (32, queryHeads + keyHeads, L),
            threadGroup: (32, 1, 1),
            outputShapes: [[1, queryHeads, L, 256], [1, keyHeads, L, 256]],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    /// `(queries [1, qHeads, L, 256], keys [1, kvHeads, L, 256])` or nil.
    static func apply(
        queryProjection: MLXArray, keyProjection: MLXArray, queryWeight: MLXArray,
        keyWeight: MLXArray, queryEps: Float, keyEps: Float, offsets: MLXArray,
        queryHeads: Int, keyHeads: Int, ropeDims: Int, ropeBase: Float, lastQueryOnly: Bool = false
    ) -> (MLXArray, MLXArray)? {
        let qShape = queryProjection.shape3
        let kShape = keyProjection.shape3
        guard qShape.0 == 1,
            kShape.0 == 1,
            lastQueryOnly
                ? (qShape.1 == 1 && kShape.1 >= 128 && kShape.1 % 64 == 0)
                : ((2 ... 16).contains(qShape.1) || (64 ... 4096).contains(qShape.1)),
            qShape.2 == queryHeads * 512,
            kShape.1 == (lastQueryOnly ? kShape.1 : qShape.1),
            kShape.2 == keyHeads * 256,
            queryProjection.dtype == .float32, keyProjection.dtype == .float32,
            queryWeight.dtype == .float32, keyWeight.dtype == .float32,
            queryWeight.ndim == 1, queryWeight.dim(0) == 256,
            keyWeight.ndim == 1, keyWeight.dim(0) == 256,
            offsets.size == 1, offsets.dtype == .int32,
            ropeDims % 8 == 0, (8 ... 128).contains(ropeDims),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let L = kShape.1
        let queryRows = lastQueryOnly ? 1 : L
        let epsArray = epsRopeBaseArray(queryEps: queryEps, keyEps: keyEps, ropeBase: ropeBase)
        let outputs = (hoisted ? hoistKernel : kernel)(
            [queryProjection, keyProjection, queryWeight, keyWeight,
                epsArray, offsets],
            template: [("Q_HEADS", queryHeads), ("KV_HEADS", keyHeads), ("ROPE_DIMS", ropeDims),
                ("LAST_QUERY", lastQueryOnly)],
            grid: lastQueryOnly ? (32, 1, queryHeads + keyHeads * L) : (32, queryHeads + keyHeads, L),
            threadGroup: (32, 1, 1),
            outputShapes: [[1, queryHeads, queryRows, 256], [1, keyHeads, L, 256]],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }
}

/// The prompt q/k norm and RoPE kernel with `q * scale` folded in. The scale is 2^-4, so the multiply that MLX's
/// composed attention graph makes on the whole query is exact and happens here on the register that is stored: the same
/// IEEE multiply of the same stored float. q leaves block-major (block b = rows 128b ..< 128b + n, laid out as
/// [heads, n, 256]) so each `[1, heads, n, 256]` block is one contiguous view, the layout that multiply's output
/// had. k is the record kernel's. Served only after `Qwen35AttentionCore.prepare` finds the block outputs
/// equal to the record's; `DARKBLOOM_BONSAI_ATTN_QFOLD=0` keeps the record kernel and the graph's multiply.
extension Qwen35VerifyQKNormRoPE {
    static let qFoldOffered = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_ATTN_QFOLD"] != "0"
    /// Written once by `prepare` before the resident serves; read-only afterwards.
    nonisolated(unsafe) static var qFoldVerified = false
    nonisolated(unsafe) private static var scaleArrays: [UInt32: MLXArray] = [:]

    private static let foldKernel = MLXFast.metalKernel(
        name: "qwen35_prompt_qk_norm_rope_qfold",
        inputNames: inputNames + ["qscale"], outputNames: outputNames,
        source: replacingOnce(
            replacingOnce(
                hoisted ? replacingOnce(source, normRopeParent, normRopeHoisted) : source,
                "device float* dst = (is_q ? q_out : k_out) + (h * outL + outT) * D;",
                """
                const uint blk = t >> 7u;
                const uint nb = min(128u, L - (blk << 7u));
                const float qs = qscale[0];
                device float* dst = is_q
                    ? q_out + (blk * uint(Q_HEADS) * 128u + h * nb + (t & 127u)) * D
                    : k_out + (h * L + t) * D;
                """),
            "float4(y[part][0], y[part][1], y[part][2], y[part][3]);",
            """
            is_q ? float4(y[part][0] * qs, y[part][1] * qs, y[part][2] * qs, y[part][3] * qs)
                : float4(y[part][0], y[part][1], y[part][2], y[part][3]);
            """))

    /// `(queryBlocks, keys, queries)`: the 128-row query blocks `[1, queryHeads, n, 256]` already multiplied by
    /// `scale`, the keys `[1, keyHeads, L, 256]`, and the record kernel's unscaled queries, a graph node nothing
    /// evaluates unless the blocks go unused. Nil when the prompt does not qualify.
    static func applyPrompt(
        queryProjection: MLXArray, keyProjection: MLXArray, queryWeight: MLXArray,
        keyWeight: MLXArray, queryEps: Float, keyEps: Float, offsets: MLXArray,
        queryHeads: Int, keyHeads: Int, ropeDims: Int, ropeBase: Float, scale: Float
    ) -> (blocks: [MLXArray], keys: MLXArray, queries: MLXArray)? {
        let L = keyProjection.shape3.1
        guard CBv2PromptAttentionCore.serves(rows: L), L <= 4096,
            let record = apply(
                queryProjection: queryProjection, keyProjection: keyProjection, queryWeight: queryWeight,
                keyWeight: keyWeight, queryEps: queryEps, keyEps: keyEps, offsets: offsets,
                queryHeads: queryHeads, keyHeads: keyHeads, ropeDims: ropeDims, ropeBase: ropeBase)
        else { return nil }
        let bits = scale.bitPattern
        let scaleArray: MLXArray
        if let cached = scaleArrays[bits] {
            scaleArray = cached
        } else {
            scaleArray = MLXArray([scale])
            scaleArrays[bits] = scaleArray
        }
        let epsArray = epsRopeBaseArray(queryEps: queryEps, keyEps: keyEps, ropeBase: ropeBase)
        let outputs = foldKernel(
            [queryProjection, keyProjection, queryWeight, keyWeight, epsArray, offsets, scaleArray],
            template: [("Q_HEADS", queryHeads), ("KV_HEADS", keyHeads), ("ROPE_DIMS", ropeDims),
                ("LAST_QUERY", false)],
            grid: (32, queryHeads + keyHeads, L), threadGroup: (32, 1, 1),
            outputShapes: [[queryHeads * L * 256], [1, keyHeads, L, 256]],
            outputDTypes: [.float32, .float32])
        let blocks = (0 ..< (L + 127) / 128).map { (index: Int) -> MLXArray in
            let n = min(128, L - 128 * index)
            let start = index * queryHeads * 128 * 256
            return outputs[0][start ..< (start + queryHeads * n * 256)].reshaped(1, queryHeads, n, 256)
        }
        return (blocks, outputs[1], record.0)
    }
}

final class Qwen35Attention: Module {
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
    /// RoPE dims and base when `rope` is a plain unscaled `RoPE`, for the
    /// fused verify q/k kernel; nil otherwise.
    let plainRope: (dims: Int, base: Float)?

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
        self.plainRope = self.rope is RoPE && args.ropeScaling?["factor"] == nil
            ? (max(1, ropeDims), args.ropeTheta) : nil

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
        guard x.ndim == 3, x.dim(0) == 1,
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
        promptProjections: [MLXArray]? = nil, lastQueryOnly: Bool = false,
        fold: PromptResidualFold? = nil, normalizedRows: (() -> MLXArray)? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        // With `normalizedRows` (`Qwen35DeadOutputs3`), `x` carries only the shape; a path that reads the rows
        // rebuilds them with the record boundary kernel.
        var rebuilt: MLXArray? = nil
        func rows() -> MLXArray {
            guard let normalizedRows else { return x }
            if let rebuilt { return rebuilt }
            let value = normalizedRows()
            rebuilt = value
            return value
        }
        let narrowsToLastQuery = lastQueryOnly && canUseLastQueryPrefill(
            x, cache: cache, positionIds: positionIds, exactTargetVerify: exactTargetVerify)
        let queryRows = narrowsToLastQuery ? 1 : L
        let projectionLayers = narrowsToLastQuery ? [kProj, vProj] : [qProj, kProj, vProj]

        // The gate halves of the query projection leave its launch and run
        // beside the attention (`VerifySplitGate`); nil keeps the record launch.
        var gateSplit: (query: MLXArray, key: MLXArray, value: MLXArray, gate: (MLXArray) -> MLXArray?)? = nil
        if !exactTargetVerify, !narrowsToLastQuery, promptProjections == nil, B == 1, let boundary {
            gateSplit = HadamardQuantizedLinear.boundarySplitQueryGate(
                boundary, query: qProj, key: kProj, value: vProj)
        }
        let shared: [MLXArray]?
        if let gateSplit {
            shared = [gateSplit.query, gateSplit.key, gateSplit.value]
        } else {
            shared = promptProjections ?? (!exactTargetVerify
                ? boundary.flatMap {
                    HadamardQuantizedLinear.boundaryProjections($0, projectionLayers)
                } ?? (L >= 128
                    ? HadamardQuantizedLinear.sharedPromptProjections(rows(), projectionLayers)
                    : HadamardQuantizedLinear.sharedShortProjections(rows(), projectionLayers))
                : nil)
        }
        let qProjOutput: MLXArray
        if narrowsToLastQuery {
            qProjOutput = promptProjections?[0] ?? (qProj as! HadamardQuantizedLinear).callPromptLastRow(rows())!
        } else {
            qProjOutput = exactTargetVerify
                ? qwen35A3BExactW4G64Projection(qProj, rows()) : (shared?[0] ?? qProj(rows()))
        }
        let keyIndex = narrowsToLastQuery && promptProjections == nil ? 0 : 1
        let valueIndex = narrowsToLastQuery && promptProjections == nil ? 1 : 2
        let kProjection = exactTargetVerify
            ? qwen35A3BExactW4G64Projection(kProj, rows()) : (shared?[keyIndex] ?? kProj(rows()))
        let vProjection = exactTargetVerify
            ? qwen35A3BExactW4G64Projection(vProj, rows()) : (shared?[valueIndex] ?? vProj(rows()))
        let values = vProjection.reshaped(B, L, kvHeads, -1)
            .transposed(0, 2, 1, 3)

        var cachedQSplit: [MLXArray]? = nil
        func qSplitParts() -> [MLXArray] {
            if let cachedQSplit { return cachedQSplit }
            let parts = qProjOutput.reshaped(B, queryRows, attentionHeads, -1).split(parts: 2, axis: -1)
            cachedQSplit = parts
            return parts
        }

        var queries: MLXArray
        var keys: MLXArray
        let promptCore = Qwen35VerifyQKNormRoPE.qFoldVerified && !exactTargetVerify && !narrowsToLastQuery
            && gateSplit == nil && B == 1 && L > 128
        var foldedBlocks: [MLXArray]? = nil
        let tree = L == 16 && !narrowsToLastQuery ? CBv2BranchVerify.tree : nil
        if let tree {
            guard !exactTargetVerify, positionIds == nil, let plainRope,
                let fused = Qwen35VerifyQKNormRoPE.applyTree(
                    queryProjection: qProjOutput, keyProjection: kProjection,
                    queryWeight: qNorm.weight, keyWeight: kNorm.weight,
                    queryEps: qNorm.eps, keyEps: kNorm.eps, offsets: cache.positionOffsets,
                    queryHeads: attentionHeads, keyHeads: kvHeads,
                    ropeDims: plainRope.dims, ropeBase: plainRope.base, tree: tree)
            else { preconditionFailure("Qwen35 tree verify: tree q/k RoPE unavailable") }
            (queries, keys) = fused
        } else if promptCore, positionIds == nil, let plainRope,
            let folded = Qwen35VerifyQKNormRoPE.applyPrompt(
                queryProjection: qProjOutput, keyProjection: kProjection,
                queryWeight: qNorm.weight, keyWeight: kNorm.weight,
                queryEps: qNorm.eps, keyEps: kNorm.eps, offsets: cache.positionOffsets,
                queryHeads: attentionHeads, keyHeads: kvHeads,
                ropeDims: plainRope.dims, ropeBase: plainRope.base, scale: scale)
        {
            // Prompt: the query blocks leave the kernel already scaled; `queries` is the record's
            // unscaled node, evaluated only if the blocks go unused.
            (queries, keys) = (folded.queries, folded.keys)
            foldedBlocks = folded.blocks
        } else if !exactTargetVerify, positionIds == nil, let plainRope,
            let fused = Qwen35VerifyQKNormRoPE.apply(
                queryProjection: qProjOutput, keyProjection: kProjection,
                queryWeight: qNorm.weight, keyWeight: kNorm.weight,
                queryEps: qNorm.eps, keyEps: kNorm.eps, offsets: cache.positionOffsets,
                queryHeads: attentionHeads, keyHeads: kvHeads,
                ropeDims: plainRope.dims, ropeBase: plainRope.base, lastQueryOnly: narrowsToLastQuery)
        {
            // Verify window: q/k RMSNorm and RoPE in one kernel, same bits.
            (queries, keys) = fused
        } else {
            queries = qNorm(qSplitParts()[0]).transposed(0, 2, 1, 3)
            keys = kNorm(kProjection.reshaped(B, L, kvHeads, -1))
                .transposed(0, 2, 1, 3)

            // Text-only Qwen positions are ordinary scalar-equivalent positions,
            // but histories differ across rows. Capture the per-row device offsets
            // before the cache advances and use the array RoPE overload.
            if let positionIds {
                (queries, keys) = mrope.apply(
                    queries: queries, keys: keys, positionIds: positionIds)
            } else {
                let offsets = cache.positionOffsets
                queries = rope(queries, offset: narrowsToLastQuery ? offsets + Int32(L - 1) : offsets)
                keys = rope(keys, offset: offsets)
            }
        }

        let attended: MLXArray
        var attentionBlocks: [MLXArray]?
        var gateProjection = qProjOutput
        if narrowsToLastQuery {
            attended = (cache as! any CBv2LastQueryPrefillLayerCache).updateAndAttendLastQuery(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        } else if let gateSplit, let boundary {
            let partial: MLXArray?
            (attended, partial) = CBv2VerifyAttentionPartial.capturing {
                cache.updateAndAttend(queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
            }
            gateProjection = gateSplit.gate(partial ?? attended)
                ?? HadamardQuantizedLinear.boundaryProjections(boundary, [qProj])?[0] ?? qProj(rows())
        } else if (Qwen35AttentionBlocks.verified || CBv2PromptAttentionCore.engaged), !exactTargetVerify, B == 1,
            L > 128
        {
            let attend = { [scale] in
                cache.updateAndAttend(queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
            }
            var blocks: [MLXArray]?
            if CBv2PromptAttentionCore.engaged {
                (attended, _) = CBv2PromptAttentionCore.capturing(prescaled: foldedBlocks) {
                    let (result, captured) = CBv2PromptAttentionBlocks.capturing(attend)
                    blocks = captured
                    return result
                }
            } else {
                (attended, blocks) = CBv2PromptAttentionBlocks.capturing(attend)
            }
            attentionBlocks = Qwen35AttentionBlocks.verified ? blocks : nil
        } else {
            attended = cache.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        }
        if narrowsToLastQuery, !exactTargetVerify, B == 1, let packed = oProj as? HadamardQuantizedLinear,
            let gated = packed.callAfterAttentionGateLastRow(attended, queryProjection: gateProjection)
        {
            return gated
        }
        if !narrowsToLastQuery, !exactTargetVerify, B == 1, let packed = oProj as? HadamardQuantizedLinear,
            let gated = packed.callAfterAttentionGate(attended, queryProjection: gateProjection)
        {
            return gated
        }
        if !narrowsToLastQuery, !exactTargetVerify, B == 1, L >= 64, let packed = oProj as? HadamardQuantizedLinear,
            let gated = packed.callAfterAttentionGatePrompt(
                attended, queryProjection: gateProjection, fold: fold,
                quantized: attentionBlocks.flatMap { (blocks: [MLXArray]) -> [MLXArray]? in
                    Qwen35AttentionBlocks.quantize(blocks, gate: gateProjection, transform: packed.transform)
                })
        {
            return gated
        }
        let gate = (gateSplit == nil
            ? qSplitParts()[1]
            : gateProjection.reshaped(B, queryRows, attentionHeads, -1).split(parts: 2, axis: -1)[1]
        ).reshaped(B, queryRows, -1)
        let output = attended
            .transposed(0, 2, 1, 3)
            .reshaped(B, queryRows, -1)
        let projectionInput = sigmoidMultiply(output, gate)
        if narrowsToLastQuery {
            let projected = (oProj as! HadamardQuantizedLinear).callPromptLastRow(projectionInput)!
            return projected
        }
        let projected = exactTargetVerify
            ? qwen35A3BExactW4G64Projection(oProj, projectionInput)
            : oProj(projectionInput)
        return projected
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
    func qwen35ProjectedGateUp(_ gate: MLXArray, _ up: MLXArray, fold: PromptResidualFold? = nil) -> MLXArray {
        if let down = downProj as? HadamardQuantizedLinear,
            let output = down.callAfterSwiGLU(gate: gate, up: up, fold: fold)
        {
            return output
        }
        return downProj(silu(gate) * up)
    }

    /// `qwen35ProjectedGateUp` over `sharedResidualNormProjections`' result: one
    /// projection is the gate/up product the matmul formed, two are the pair.
    func qwen35ProjectedGateUp(_ projections: [MLXArray], fold: PromptResidualFold? = nil) -> MLXArray {
        if projections.count == 1 {
            if let down = downProj as? HadamardQuantizedLinear,
                let output = down.callAfterPromptProduct(projections[0], fold: fold)
            {
                return output
            }
            return downProj(projections[0])
        }
        return qwen35ProjectedGateUp(projections[0], projections[1], fold: fold)
    }

    func qwen35TargetVerify(_ x: MLXArray, exact: Bool) -> MLXArray {
        if !exact, x.ndim >= 2, x.dim(-2) >= 128,
            let shared = HadamardQuantizedLinear.sharedPromptProjections(x, [gateProj, upProj])
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
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            precondition(attentionCache == nil, "Qwen35 recurrent layer received attention KV")
            if captureRecurrentWindow {
                r = linearAttn!.cbv2ForwardCaptured(
                    inputLayerNorm(x), modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState,
                    exactTargetVerify: exactTargetVerify)
            } else if x.ndim == 3, x.dim(1) >= 128, let linear = linearAttn,
                let prompt = HadamardQuantizedLinear.sharedNormPromptProjections(
                    x, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                    projections: [linear.inProjQKV, linear.inProjZ], normalized: true, deferLast: true),
                let normalized = prompt.normalized
            {
                // Prompt width: the input RMSNorm formed in the quantizing
                // kernel of the qkv/z projections (same bits as
                // `inputLayerNorm(x)` and its shared prompt codes).
                r = linear.cbv2Forward(
                    normalized, modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState, promptProjections: prompt.projections,
                    promptZ: prompt.deferred)
            } else {
                r = linearAttn!.cbv2Forward(
                    inputLayerNorm(x), modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState)
            }
        } else {
            guard let attentionCache else {
                preconditionFailure("Qwen35 full-attention layer is missing its CBv2 cache")
            }
            let lastQueryOnly = lastPositionOnly && !captureRecurrentWindow
                && selfAttn!.canUseLastQueryPrefill(
                    x, cache: attentionCache, positionIds: positionIds, exactTargetVerify: exactTargetVerify)
            if !exactTargetVerify, x.ndim == 3, x.dim(1) >= 128, let attention = selfAttn,
                let prompt = HadamardQuantizedLinear.sharedNormPromptProjections(
                    x, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                    projections: [attention.qProj, attention.kProj, attention.vProj], normalized: false,
                    firstProjectionLastRow: lastQueryOnly)
            {
                // Prompt width: the input RMSNorm formed in the quantizing
                // kernel of the q/k/v projections (their only consumer).
                r = attention.cbv2Forward(
                    x, cache: attentionCache, positionIds: positionIds,
                    exactTargetVerify: false, promptProjections: prompt.projections, lastQueryOnly: lastQueryOnly)
            } else {
                r = selfAttn!.cbv2Forward(
                    inputLayerNorm(x), cache: attentionCache, positionIds: positionIds,
                    exactTargetVerify: exactTargetVerify, lastQueryOnly: lastQueryOnly)
            }
        }
        let xPost = lastPositionOnly && x.dim(1) > 1
            ? x[0..., (x.dim(1) - 1)..., 0...] : x
        let rPost = lastPositionOnly && r.dim(1) > 1
            ? r[0..., (r.dim(1) - 1)..., 0...] : r
        if let dense = mlp as? Qwen3NextMLP,
            let boundary = HadamardQuantizedLinear.sharedResidualNormProjections(
                xPost, residual: rPost, weight: postAttentionLayerNorm.weight, eps: postAttentionLayerNorm.eps,
                projections: [dense.gateProj, dense.upProj],
                swigluDown: xPost.size == 16 * 4096 ? dense.downProj as? HadamardQuantizedLinear : nil)
        {
            return boundary.residual + dense.qwen35ProjectedGateUp(boundary.projections)
        }
        let h = xPost + rPost
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
        return h + feedForward
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
            guard let linearAttn else { return nil }
            if let boundary = HadamardQuantizedLinear.verifyNormBoundary(
                x, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                projections: [linearAttn.inProjQKV, linearAttn.inProjZ])
            {
                r = linearAttn.cbv2ForwardCaptured(
                    boundary.normalized, modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState, exactTargetVerify: false,
                    boundary: boundary)
            } else {
                r = linearAttn.cbv2ForwardCaptured(
                    inputLayerNorm(x), modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState, exactTargetVerify: false)
            }
        } else {
            guard let selfAttn, let attentionCache else { return nil }
            if let boundary = HadamardQuantizedLinear.verifyNormBoundary(
                x, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                projections: [selfAttn.qProj, selfAttn.kProj, selfAttn.vProj])
            {
                r = selfAttn.cbv2Forward(
                    boundary.normalized, cache: attentionCache, positionIds: positionIds,
                    exactTargetVerify: false, boundary: boundary)
            } else {
                r = selfAttn.cbv2Forward(
                    inputLayerNorm(x), cache: attentionCache, positionIds: positionIds,
                    exactTargetVerify: false)
            }
        }
        if let post = HadamardQuantizedLinear.sharedResidualNormProjections(
            x, residual: r, weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps, projections: [dense.gateProj, dense.upProj],
            swigluDown: x.size == 16 * 4096 ? dense.downProj as? HadamardQuantizedLinear : nil)
        {
            return (post.residual, dense.qwen35ProjectedGateUp(post.projections))
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
    ) -> (inputResidual: MLXArray, residual: MLXArray, delta: MLXArray)? {
        guard let dense = mlp as? Qwen3NextMLP else { return nil }
        let firstProjections: [Linear]
        if isLinear {
            guard let linearAttn else { return nil }
            firstProjections = [linearAttn.inProjQKV, linearAttn.inProjZ]
        } else {
            guard let selfAttn, attentionCache != nil else { return nil }
            firstProjections = [selfAttn.qProj, selfAttn.kProj, selfAttn.vProj]
        }
        let lean = !isLinear && Qwen35DeadOutputs3.verified
        guard
            let boundary = HadamardQuantizedLinear.verifyBoundary(
                x, delta: delta, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                projections: firstProjections, normalized: !lean)
        else { return nil }
        let r: MLXArray
        if isLinear {
            r = linearAttn!.cbv2ForwardCaptured(
                boundary.normalized, modelLayerIndex: modelLayerIndex,
                recurrentState: recurrentState, exactTargetVerify: false,
                boundary: boundary)
        } else {
            let norm = inputLayerNorm
            r = selfAttn!.cbv2Forward(
                boundary.normalized, cache: attentionCache!, positionIds: positionIds,
                exactTargetVerify: false, boundary: boundary,
                normalizedRows: lean ? {
                    HadamardQuantizedLinear.verifyBoundary(
                        x, delta: delta, weight: norm.weight, eps: norm.eps, projections: firstProjections
                    )?.normalized ?? norm(boundary.residual)
                } : nil)
        }
        if let post = HadamardQuantizedLinear.sharedResidualNormProjections(
            boundary.residual, residual: r, weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps, projections: [dense.gateProj, dense.upProj],
            swigluDown: x.size == 16 * 4096 ? dense.downProj as? HadamardQuantizedLinear : nil)
        {
            return (
                boundary.residual, post.residual,
                dense.qwen35ProjectedGateUp(post.projections))
        }
        let h = boundary.residual + r
        return (boundary.residual, h, dense.qwen35TargetVerify(postAttentionLayerNorm(h), exact: false))
    }

    /// Prompt-width layer returning its residual and MLP output unadded so the
    /// next layer can fuse the add into its input norm+quantize kernel.
    func cbv2ForwardPromptSplitOutput(
        _ x: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        foldDown: Bool = false
    ) -> (residual: MLXArray, delta: MLXArray?)? {
        guard let dense = mlp as? Qwen3NextMLP, x.ndim == 3, x.dim(1) >= 128 else { return nil }
        let r: MLXArray
        let outFold = PromptResidualFold(x)
        if isLinear {
            guard let linear = linearAttn,
                let prompt = HadamardQuantizedLinear.sharedNormPromptProjections(
                    x, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                    projections: [linear.inProjQKV, linear.inProjZ], normalized: true, deferLast: true),
                let normalized = prompt.normalized
            else { return nil }
            r = linear.cbv2Forward(
                normalized, modelLayerIndex: modelLayerIndex,
                recurrentState: recurrentState, promptProjections: prompt.projections,
                promptZ: prompt.deferred, fold: outFold)
        } else {
            guard let attention = selfAttn, let attentionCache,
                let prompt = HadamardQuantizedLinear.sharedNormPromptProjections(
                    x, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                    projections: [attention.qProj, attention.kProj, attention.vProj], normalized: false)
            else { return nil }
            r = attention.cbv2Forward(
                x, cache: attentionCache, positionIds: positionIds,
                exactTargetVerify: false, promptProjections: prompt.projections, fold: outFold)
        }
        guard let post = HadamardQuantizedLinear.sharedResidualNormProjections(
            outFold.folded ? r : x, residual: outFold.folded ? nil : r, weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps, projections: [dense.gateProj, dense.upProj],
            swigluDown: dense.downProj as? HadamardQuantizedLinear)
        else { return nil }
        let downFold = foldDown ? PromptResidualFold(post.residual) : nil
        let down = dense.qwen35ProjectedGateUp(post.projections, fold: downFold)
        return downFold?.folded == true ? (down, nil) : (post.residual, down)
    }

    /// Prompt-width layer whose input arrives as `(x, delta)` unadded, fusing
    /// `x + delta` into `sharedResidualNormPromptProjections`.
    func cbv2ForwardPromptFused(
        _ x: MLXArray, delta: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        lastPositionOnly: Bool = false,
        foldDown: Bool = false
    ) -> (inputResidual: MLXArray, residual: MLXArray, delta: MLXArray?)? {
        guard let dense = mlp as? Qwen3NextMLP, x.ndim == 3, x.dim(1) >= 128 else { return nil }
        let inputResidual: MLXArray
        let r: MLXArray
        var outFold: PromptResidualFold? = nil
        if isLinear {
            guard let linear = linearAttn,
                let prompt = HadamardQuantizedLinear.sharedResidualNormPromptProjections(
                    x, delta: delta, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                    projections: [linear.inProjQKV, linear.inProjZ], normalized: true, deferLast: true),
                let normalized = prompt.normalized
            else { return nil }
            inputResidual = prompt.residual
            outFold = lastPositionOnly ? nil : PromptResidualFold(inputResidual)
            r = linear.cbv2Forward(
                normalized, modelLayerIndex: modelLayerIndex,
                recurrentState: recurrentState, promptProjections: prompt.projections,
                promptZ: prompt.deferred, fold: outFold)
        } else {
            guard let attention = selfAttn, let attentionCache else { return nil }
            let lastQueryOnly = lastPositionOnly && attention.canUseLastQueryPrefill(
                x, cache: attentionCache, positionIds: positionIds)
            guard let prompt = HadamardQuantizedLinear.sharedResidualNormPromptProjections(
                    x, delta: delta, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                    projections: [attention.qProj, attention.kProj, attention.vProj], normalized: false,
                    firstProjectionLastRow: lastQueryOnly)
            else { return nil }
            inputResidual = prompt.residual
            outFold = lastPositionOnly ? nil : PromptResidualFold(inputResidual)
            r = attention.cbv2Forward(
                inputResidual, cache: attentionCache, positionIds: positionIds,
                exactTargetVerify: false, promptProjections: prompt.projections, lastQueryOnly: lastQueryOnly,
                fold: outFold)
        }
        let folded = outFold?.folded == true
        let xPost = lastPositionOnly && inputResidual.dim(1) > 1
            ? inputResidual[0..., (inputResidual.dim(1) - 1)..., 0...] : inputResidual
        let rPost = lastPositionOnly && r.dim(1) > 1
            ? r[0..., (r.dim(1) - 1)..., 0...] : r
        if let post = HadamardQuantizedLinear.sharedResidualNormProjections(
            folded ? rPost : xPost, residual: folded ? nil : rPost, weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps, projections: [dense.gateProj, dense.upProj],
            swigluDown: dense.downProj as? HadamardQuantizedLinear)
        {
            let downFold = foldDown ? PromptResidualFold(post.residual) : nil
            let down = dense.qwen35ProjectedGateUp(post.projections, fold: downFold)
            return downFold?.folded == true ? (inputResidual, down, nil) : (inputResidual, post.residual, down)
        }
        let h = folded ? rPost : xPost + rPost
        return (inputResidual, h, dense.qwen35TargetVerify(postAttentionLayerNorm(h), exact: false))
    }
}

// MARK: - Layer 0 on an FP32 embedding

extension Qwen35GatedDeltaNet {
    /// `projectInputs` as the layer-0 check (`Qwen35Layer0FP32`) needs it.
    fileprivate func layer0CheckInputs(
        _ x: MLXArray, boundary: HadamardQuantizedLinear.VerifyBoundary?, prompt: [MLXArray]?
    ) -> [MLXArray] {
        let (qkv, z, b, a) = projectInputs(
            x, B: x.dim(0), S: x.dim(1), boundary: boundary, prompt: prompt)
        return [qkv, z, b, a]
    }
}

/// The packed embedding returns F16 rows, so layer 0 misses the fused prompt
/// and verify boundaries that need an FP32 residual. Its unfused path casts
/// those rows to FP32 anyway (MLX `rmsNorm` and the residual add promote
/// them), and F16 to FP32 is exact, so an FP32 copy of the embedding changes
/// no input value; whether the fused boundaries then return the unfused bits
/// is checked once per width, at the resident's boot warm, on the real
/// layer-0 and layer-1 weights. A width switches only after its check
/// passes. `DARKBLOOM_BONSAI_LAYER0_FP32=0` keeps the F16 input.
enum Qwen35Layer0FP32 {
    static let offered =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_LAYER0_FP32"] != "0"
    /// Written once by `prepareIfBoot` before the resident serves; read-only afterwards.
    nonisolated(unsafe) static var promptPassed = false
    nonisolated(unsafe) static var verifyPassed = false
    nonisolated(unsafe) private static var prepared = false

    /// Runs from the first prompt forward of at least 1024 rows: the boot
    /// warm, which no timed phase reaches (the timed prompts are 512 rows).
    static func prepareIfBoot(_ model: Qwen35TextModelInner, promptRows: Int) {
        guard promptRows >= 1024, !prepared else { return }
        prepared = true
        Qwen35DeadOutputs3.prepare(model)
        Qwen35AttentionBlocks.prepare(model)
        Qwen35AttentionCore.prepare(model)
        guard offered else {
            FileHandle.standardError.write(Data("qwen35 layer0 fp32: off\n".utf8))
            return
        }
        let prompt = verdict { check(model, rows: [512], prompt: true) }
        let verify = verdict { check(model, rows: Array(2 ... 16), prompt: false) }
        promptPassed = prompt.passed
        verifyPassed = verify.passed
        FileHandle.standardError.write(Data(
            "qwen35 layer0 fp32: prompt \(prompt.note) (rows 512), verify \(verify.note) (rows 2-16)\n".utf8))
        Qwen35VerifyStartFusion.prepare(model)
    }

    private static func verdict(_ body: () -> Bool) -> (passed: Bool, note: String) {
        do {
            return try withError({ body() })
                ? (true, "bitwise equal") : (false, "mismatch, F16 input kept")
        } catch {
            return (false, "error, F16 input kept: \(error)")
        }
    }

    private static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        guard a.shape == b.shape, a.dtype == b.dtype else { return MLXArray(false) }
        let bits: DType? = a.dtype == .float32 ? .uint32
            : (a.dtype == .float16 || a.dtype == .bfloat16) ? .uint16 : nil
        guard let bits else { return MLX.all(a .== b) }
        return MLX.all(contiguous(a).view(dtype: bits) .== contiguous(b).view(dtype: bits))
    }

    /// Layer 0 (and, at prompt width, layer 1's input) on real embedding rows
    /// both ways: the unfused path on the F16 rows, as the record serves it,
    /// against the path `cbv2ForwardSplitFinal` takes on their FP32 copy,
    /// including its unfused fallbacks. Every array a later kernel reads is
    /// compared bit for bit.
    private static func check(_ model: Qwen35TextModelInner, rows: [Int], prompt: Bool) -> Bool {
        let layers = model.layers
        guard layers.count > 1, let gdn0 = layers[0].linearAttn,
            let gdn1 = layers[1].linearAttn, let mlp0 = layers[0].mlp as? Qwen3NextMLP
        else { return false }
        let norm0 = layers[0].inputLayerNorm
        let post0 = layers[0].postAttentionLayerNorm
        let norm1 = layers[1].inputLayerNorm
        let vocabulary = model.embedTokens.shape.0
        var eqs: [MLXArray] = []
        for count in rows {
            let ids = MLXArray((0 ..< count).map { Int32((100 + $0 &* 7919) % vocabulary) })
            let x16 = model.embedTokens(ids.reshaped([1, count]))
            guard x16.dtype == .float16 else { return false }
            let x32 = x16.asType(.float32)
            let r = MLX.sin(MLXArray(0 ..< x16.size).asType(.float32) * 0.37)
                .reshaped(x16.shape) * 0.05

            let unfused = norm0(x16)
            let oldInputs = gdn0.layer0CheckInputs(unfused, boundary: nil, prompt: nil)
            var normalized = norm0(x32)
            var newInputs: [MLXArray]? = nil
            if prompt {
                if let p = HadamardQuantizedLinear.sharedNormPromptProjections(
                    x32, weight: norm0.weight, eps: norm0.eps,
                    projections: [gdn0.inProjQKV, gdn0.inProjZ], normalized: true),
                    let n = p.normalized
                {
                    normalized = n
                    newInputs = gdn0.layer0CheckInputs(n, boundary: nil, prompt: p.projections)
                }
            } else if let boundary = HadamardQuantizedLinear.verifyNormBoundary(
                x32, weight: norm0.weight, eps: norm0.eps,
                projections: [gdn0.inProjQKV, gdn0.inProjZ])
            {
                normalized = boundary.normalized
                newInputs = gdn0.layer0CheckInputs(
                    boundary.normalized, boundary: boundary, prompt: nil)
            }
            let fusedInputs = newInputs
                ?? gdn0.layer0CheckInputs(normalized, boundary: nil, prompt: nil)
            eqs.append(bitsEqual(unfused, normalized))
            for (o, n) in zip(oldInputs, fusedInputs) { eqs.append(bitsEqual(o, n)) }

            let hOld = x16 + r
            let ffOld = mlp0.qwen35TargetVerify(post0(hOld), exact: false)
            let hNew: MLXArray
            let ffNew: MLXArray
            if let post = HadamardQuantizedLinear.sharedResidualNormProjections(
                x32, residual: r, weight: post0.weight, eps: post0.eps,
                projections: [mlp0.gateProj, mlp0.upProj])
            {
                hNew = post.residual
                ffNew = mlp0.qwen35ProjectedGateUp(post.projections[0], post.projections[1])
            } else {
                hNew = x32 + r
                ffNew = mlp0.qwen35TargetVerify(post0(hNew), exact: false)
            }
            eqs.append(bitsEqual(hOld, hNew))
            eqs.append(bitsEqual(ffOld, ffNew))

            // Prompt width only: layer 1 now receives layer 0's output unadded.
            // Verify already handed it over unadded on the record path.
            guard prompt,
                let fused1 = HadamardQuantizedLinear.sharedResidualNormPromptProjections(
                    hNew, delta: ffNew, weight: norm1.weight, eps: norm1.eps,
                    projections: [gdn1.inProjQKV, gdn1.inProjZ], normalized: true)
            else { continue }
            let y = hOld + ffOld
            guard let old1 = HadamardQuantizedLinear.sharedNormPromptProjections(
                    y, weight: norm1.weight, eps: norm1.eps,
                    projections: [gdn1.inProjQKV, gdn1.inProjZ], normalized: true),
                let oldNormalized = old1.normalized, let newNormalized = fused1.normalized,
                old1.projections.count == fused1.projections.count
            else { return false }
            eqs.append(bitsEqual(y, fused1.residual))
            eqs.append(bitsEqual(oldNormalized, newNormalized))
            for (o, n) in zip(old1.projections, fused1.projections) { eqs.append(bitsEqual(o, n)) }
        }
        return MLX.all(MLX.stacked(eqs)).item(Bool.self)
    }
}

/// The attention layers' verify boundary without its normalized rows (`verifyBoundary(normalized: false)`): on the
/// split-gate route those layers read the rows only for their shape, and any other path rebuilds them with the record
/// kernel (`cbv2Forward(normalizedRows:)`). Served only after a check at the boot warm, on every attention layer's
/// norm weight and q/k/v at verify rows 2 to 16, finds the codes, scales, offsets, sum and the split-gate route's
/// query, key, value and gate outputs bitwise equal to the record boundary's. `DARKBLOOM_BONSAI_DEAD_OUTPUTS3=0`
/// keeps the record boundary; `DARKBLOOM_BONSAI_DEAD_OUTPUTS3_NEGCTL=1` scales one norm weight of the lean side by
/// 1 + 2^-8, which the check must report.
enum Qwen35DeadOutputs3 {
    static let offered = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DEAD_OUTPUTS3"] != "0"
    static let negativeControl = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DEAD_OUTPUTS3_NEGCTL"] == "1"
    /// Set once by `prepare` before the resident serves; only read after that.
    nonisolated(unsafe) static var verified = false

    static func prepare(_ model: Qwen35TextModelInner) {
        let note: String
        if !offered {
            note = "off"
        } else {
            do {
                let (bad, cases) = try withError { check(model) }
                verified = bad == 0 && cases > 0
                note = verified ? "on (\(cases) cases bitwise equal)" : "off (\(bad) of \(cases) cases differ)"
            } catch {
                note = "off (error: \(error))"
            }
        }
        FileHandle.standardError.write(Data("qwen35 dead outputs3: \(note)\n".utf8))
    }

    private static func same(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        guard a.shape == b.shape, a.dtype == b.dtype else { return MLXArray(false) }
        let bits: DType = a.dtype == .float32 ? .uint32 : a.dtype
        return MLX.all(contiguous(a).view(dtype: bits) .== contiguous(b).view(dtype: bits))
    }

    /// Query (`index` 0) or gate (1) halves of each 512-wide head; the launch leaves the other half unwritten.
    private static func half(_ a: MLXArray, _ index: Int) -> MLXArray {
        let n = a.dim(-1)
        return a.reshaped([-1, n / 512, 2, 256])[0..., 0..., index, 0...]
    }

    private static func check(_ model: Qwen35TextModelInner) -> (bad: Int, cases: Int) {
        var bad = 0
        var cases = 0
        let bump = MLXArray((0 ..< 4096).map { (i: Int) -> Float in i == 7 ? Float(1) / 256 : 0 })
        for layer in model.layers where !layer.isLinear {
            guard let attention = layer.selfAttn else { return (bad + 1, cases + 1) }
            let projections: [Linear] = [attention.qProj, attention.kProj, attention.vProj]
            let norm = layer.inputLayerNorm
            let leanWeight = negativeControl ? norm.weight * (bump + 1) : norm.weight
            var results: [MLXArray] = []
            for rows in 2 ... 16 {
                let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x4430_0000 + rows)), into: 2)
                let scale = MLXArray((0 ..< rows).map { (r: Int) -> Float in r == 0 ? 0 : (r == 1 ? 30 : 1) })
                    .reshaped([1, rows, 1])
                let x = MLXRandom.uniform(low: -3, high: 3, [1, rows, 4096], key: keys[0]) * scale
                let delta = MLXRandom.uniform(low: -4, high: 4, [1, rows, 4096], key: keys[1]) * scale
                guard let full = HadamardQuantizedLinear.verifyBoundary(
                    x, delta: delta, weight: norm.weight, eps: norm.eps, projections: projections)
                else { continue }
                cases += 1
                guard let lean = HadamardQuantizedLinear.verifyBoundary(
                        x, delta: delta, weight: leanWeight, eps: norm.eps, projections: projections,
                        normalized: false),
                    lean.normalized.shape == full.normalized.shape,
                    let a = HadamardQuantizedLinear.boundarySplitQueryGate(
                        full, query: attention.qProj, key: attention.kProj, value: attention.vProj),
                    let b = HadamardQuantizedLinear.boundarySplitQueryGate(
                        lean, query: attention.qProj, key: attention.kProj, value: attention.vProj),
                    let gateA = a.gate(a.value), let gateB = b.gate(b.value)
                else {
                    bad += 1
                    continue
                }
                var eqs = zip(full.outputs, lean.outputs).map { (pair: (MLXArray, MLXArray)) -> MLXArray in
                    same(pair.0, pair.1)
                }
                eqs.append(same(half(a.query, 0), half(b.query, 0)))
                eqs.append(same(a.key, b.key))
                eqs.append(same(a.value, b.value))
                eqs.append(same(half(gateA, 1), half(gateB, 1)))
                results.append(MLX.all(MLX.stacked(eqs)))
            }
            guard !results.isEmpty else { continue }
            let passed = MLX.stacked(results).asType(.int32).sum().item(Int.self)
            bad += results.count - passed
        }
        return (bad, cases)
    }
}

/// The prompt attention's gate quantizer reading the query-block outputs (`CBv2PromptAttentionBlocks`) in place,
/// so their concatenation (a copy of the whole attention output) is never formed: the record kernel with
/// only its attention load moved from the concatenation to the block holding the row. Served only after a check at
/// the boot warm, on the real `o_proj` signs at 512, 448, 256 and 192 rows, finds the codes, ascale and rsb words
/// equal to the record kernel's on the concatenation, a lean input one word off unequal, and 128 rows (one block)
/// declined. `DARKBLOOM_BONSAI_ATTNBLK=0` keeps the concatenation; `DARKBLOOM_BONSAI_ATTNBLK_NEGCTL=1` adds 1 to
/// one attention value of every lean case, which the check must report.
enum Qwen35AttentionBlocks {
    static let offered = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_ATTNBLK"] != "0"
    static let negativeControl = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_ATTNBLK_NEGCTL"] == "1"
    /// Set once by `prepare` before the resident serves; only read after that.
    nonisolated(unsafe) static var verified = false

    private static func replacingOnce(_ text: String, _ old: String, _ new: String) -> String {
        guard text.components(separatedBy: old).count == 2 else {
            preconditionFailure("qwen35 attention blocks: anchor not unique")
        }
        return text.replacingOccurrences(of: old, with: new)
    }

    /// Prompt row r reads block r / 128 (inputs `x`, `x1`, `x2`, `x3`) at its row r % 128; a block holds
    /// min(128, rows - its first row) prompt rows.
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_attention_gate_prompt_blocks_q8",
        inputNames: ["x", "x1", "x2", "x3", "gate", "signs"],
        outputNames: ["codes", "ascale", "rsb"],
        source: replacingOnce(
            replacingOnce(
                SignedBlockHadamard.quantized4096Source(swiglu: false, attentionGate: true),
                "const float4 attn = *((const device float4*)(x",
                """
                const uint ag_block = ag_row >> 7;
                const device float* ag_x = ag_block == 0u ? x : ag_block == 1u ? x1 : ag_block == 2u ? x2 : x3;
                const float4 attn = *((const device float4*)(ag_x
                """),
            "* uint(VERIFY_ROWS) + ag_row", "* min(128u, uint(ATTENTION_ROWS) - (ag_row & ~127u)) + (ag_row & 127u)"),
        header: SignedBlockHadamard.fused4096Header.replacingOccurrences(
            of: "hadamard_radix16(", with: "hadamard_radix16_q8("),
        ensureRowContiguous: true)

    /// Codes, ascale and rsb of `transform.quantizedForPromptAttentionGate` on the blocks' concatenation, read from
    /// the 2 to 4 blocks of 128 rows (the last 64 to 128). Nil when the blocks or shapes do not qualify.
    static func quantize(_ blocks: [MLXArray], gate: MLXArray, transform: SignedBlockHadamard) -> [MLXArray]? {
        guard CBv2PromptAttentionBlocks.blockSize == 128, (2 ... 4).contains(blocks.count),
            transform.blockSize == 4096, transform.width == 4096, gate.dtype == .float32,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let last = blocks[blocks.count - 1].dim(2)
        let rows = 128 * (blocks.count - 1) + last
        guard last > 0, last <= 128, rows % 64 == 0, gate.size == rows * 8192 else { return nil }
        for (index, block) in blocks.enumerated() {
            let n = index == blocks.count - 1 ? last : 128
            guard block.dtype == .float32, block.shape == [1, 16, n, 256] else { return nil }
        }
        let filled = blocks + Array(repeating: blocks[blocks.count - 1], count: 4 - blocks.count)
        return kernel(
            filled + [gate.reshaped([rows, 8192]), transform.signs],
            template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0), ("ATTENTION_ROWS", rows)],
            grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, 4096], [rows * 32], [rows * 32]],
            outputDTypes: [.int8, .float32, .float32])
    }

    static func prepare(_ model: Qwen35TextModelInner) {
        let note: String
        if !offered {
            note = "off"
        } else {
            do {
                let (bad, cases) = try withError { check(model) }
                verified = bad == 0 && cases > 0
                note = verified ? "on (\(cases) cases bitwise equal)" : "off (\(bad) of \(cases) cases differ)"
            } catch {
                note = "off (error: \(error))"
            }
        }
        FileHandle.standardError.write(Data("qwen35 attention blocks: \(note)\n".utf8))
    }

    private static func same(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        guard a.shape == b.shape, a.dtype == b.dtype else { return MLXArray(false) }
        let bits: DType = a.dtype == .float32 ? .uint32 : a.dtype
        return MLX.all(contiguous(a).view(dtype: bits) .== contiguous(b).view(dtype: bits))
    }

    /// Each case: blocks of uniform attention values (a zero row and a x30 row in each), the
    /// `[1, rows, 8192]` query projection, record kernel on the concatenation against `quantize` on the
    /// blocks. Returns the failed and the total case count; a case also fails when a lean input one word
    /// off still matches.
    private static func check(_ model: Qwen35TextModelInner) -> (bad: Int, cases: Int) {
        guard let packed = model.layers.lazy.compactMap({ (layer: Qwen35DecoderLayer) -> HadamardQuantizedLinear? in
            layer.selfAttn?.oProj as? HadamardQuantizedLinear
        }).first else { return (1, 1) }
        let transform = packed.transform
        var bad = 0
        var cases = 0
        for rows in [512, 448, 256, 192, 128] {
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x4154_0000 + rows)), into: 5)
            var blocks: [MLXArray] = []
            var start = 0
            while start < rows {
                let n = min(128, rows - start)
                let scale = MLXArray((0 ..< n).map { (r: Int) -> Float in r == 0 ? 0 : (r == 1 ? 30 : 1) })
                    .reshaped([1, 1, n, 1])
                blocks.append(
                    MLXRandom.uniform(low: -3, high: 3, [1, 16, n, 256], key: keys[blocks.count]) * scale)
                start += n
            }
            let gate = MLXRandom.uniform(low: -6, high: 6, [1, rows, 8192], key: keys[4])
            let whole = concatenated(blocks, axis: 2)
            cases += 1
            guard rows > 128 else {
                if quantize(blocks, gate: gate, transform: transform) != nil { bad += 1 }
                continue
            }
            // One attention value of the last block (head 3, its row 16, dim 5) one up.
            let tail = blocks[blocks.count - 1]
            let offset = MLXArray((0 ..< tail.size).map { (i: Int) -> Float in
                i == (3 * tail.dim(2) + 16) * 256 + 5 ? 1 : 0
            }).reshaped(tail.shape)
            let off = Array(blocks.dropLast()) + [tail + offset]
            guard let record = transform.quantizedForPromptAttentionGate(whole, queryProjection: gate),
                let lean = quantize(negativeControl ? off : blocks, gate: gate, transform: transform),
                let control = quantize(off, gate: gate, transform: transform)
            else {
                bad += 1
                continue
            }
            let expected = [record.codes, record.ascale, record.rsb]
            let equal = MLX.all(MLX.stacked(zip(expected, lean).map { (pair: (MLXArray, MLXArray)) -> MLXArray in
                same(pair.0, pair.1)
            }))
            let controlEqual = MLX.all(MLX.stacked(zip(expected, control).map { (pair: (MLXArray, MLXArray)) -> MLXArray in
                same(pair.0, pair.1)
            }))
            if !equal.item(Bool.self) || controlEqual.item(Bool.self) { bad += 1 }
        }
        return (bad, cases)
    }
}

/// Boot check of the prompt attention core (`CBv2PromptAttentionCore`, `Qwen35VerifyQKNormRoPE.applyPrompt`): at 512,
/// 448, 256, 192 and 137 rows (every query block the scored shapes cut, the last of 9 to 128 rows) on the first full
/// attention layer's real norm weights, rope and scale, the record's per-block `scaledDotProductAttention` on the
/// cache's strided K/V against the core's blocks, bit for bit, for each enabled part alone and, when both are,
/// together; k from the folded kernel must equal the record's; a lean run with its scale (fold) or its causal limit
/// (softmax) one step off must differ; 129 rows must decline. The softmax is also compared on score blocks of every
/// kL with a spread of magnitudes. `DARKBLOOM_BONSAI_ATTN_QFOLD=0` and `DARKBLOOM_BONSAI_ATTN_SMAX=0` switch a part
/// off; `..._NEGCTL=1` on either makes its lean run one step off, which the check must report.
enum Qwen35AttentionCore {
    private static let foldNegativeControl =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_ATTN_QFOLD_NEGCTL"] == "1"

    static func prepare(_ model: Qwen35TextModelInner) {
        let fold = Qwen35VerifyQKNormRoPE.qFoldOffered
        let smax = CBv2PromptAttentionCore.softmaxOffered
        var foldOK = false
        var smaxOK = false
        var note = "off"
        var fill = "off"
        if CBv2FullSequenceKV.firstFillOffered {
            do {
                let (bad, cases) = try withError { CBv2FullSequenceKV.prepareFirstFill() }
                fill = CBv2FullSequenceKV.firstFillVerified ? "on (\(cases) cases)" : "off (\(bad) of \(cases) cases differ)"
            } catch {
                fill = "off (error: \(error))"
            }
        }
        if fold || smax {
            do {
                var cases = 0
                if fold {
                    let (bad, n) = try withError { check(model, fold: true, smax: false) }
                    foldOK = bad == 0
                    cases += n
                }
                if smax {
                    let (bad, n) = try withError { check(model, fold: false, smax: true) }
                    smaxOK = bad == 0
                    cases += n
                }
                if foldOK && smaxOK {
                    let (bad, n) = try withError { check(model, fold: true, smax: true) }
                    smaxOK = bad == 0
                    cases += n
                }
                note = "q fold \(foldOK ? "on" : "off"), softmax \(smaxOK ? "on" : "off") (\(cases) cases)"
            } catch {
                note = "off (error: \(error))"
            }
        }
        Qwen35VerifyQKNormRoPE.qFoldVerified = foldOK
        CBv2PromptAttentionCore.softmaxVerified = smaxOK
        CBv2PromptAttentionCore.engaged = foldOK || smaxOK
        FileHandle.standardError.write(Data("qwen35 attention core: \(note); first KV fill \(fill)\n".utf8))
    }

    private static func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        guard a.shape == b.shape, a.dtype == b.dtype else { return false }
        return MLX.all(contiguous(a).view(dtype: .uint32) .== contiguous(b).view(dtype: .uint32)).item(Bool.self)
    }

    private static func check(_ model: Qwen35TextModelInner, fold: Bool, smax: Bool) -> (bad: Int, cases: Int) {
        guard let attention = model.layers.lazy.compactMap({ (layer: Qwen35DecoderLayer) -> Qwen35Attention? in
            layer.selfAttn
        }).first, let rope = attention.plainRope, attention.attentionHeads == 16, attention.kvHeads == 4
        else { return (1, 1) }
        let scale = attention.scale
        var bad = 0
        var cases = 0
        cases += 1
        if !CBv2PromptAttentionCore.serves(rows: 512) || !CBv2PromptAttentionCore.serves(rows: 137)
            || CBv2PromptAttentionCore.serves(rows: 129) || CBv2PromptAttentionCore.serves(rows: 136)
            || CBv2PromptAttentionCore.serves(rows: 128)
        {
            bad += 1
        }
        for rows in [512, 448, 256, 192, 137] {
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x4143_0000 + rows)), into: 6)
            let queryProjection = MLXRandom.normal([1, rows, 16 * 512], key: keys[0])
            let keyProjection = MLXRandom.normal([1, rows, 4 * 256], key: keys[1])
            let values = MLXRandom.normal([1, 4, rows, 256], key: keys[2])
            let offsets = MLXArray([Int32(rows == 256 ? 1000 : 0)])
            guard
                let record = Qwen35VerifyQKNormRoPE.apply(
                    queryProjection: queryProjection, keyProjection: keyProjection,
                    queryWeight: attention.qNorm.weight, keyWeight: attention.kNorm.weight,
                    queryEps: attention.qNorm.eps, keyEps: attention.kNorm.eps, offsets: offsets,
                    queryHeads: 16, keyHeads: 4, ropeDims: rope.dims, ropeBase: rope.base)
            else {
                bad += 1
                cases += 1
                continue
            }
            let store = CBv2FullSequenceKV(promptLength: rows, maxLength: rows + 512, kvHeads: 4, headDim: 256)
            let (cachedKeys, cachedValues) = store.update(keys: record.1, values: values)
            let ranges = (0 ..< (rows + 127) / 128).map { (index: Int) -> (start: Int, count: Int) in
                (128 * index, min(128, rows - 128 * index))
            }
            let expected = ranges.map { (range: (start: Int, count: Int)) -> MLXArray in
                MLXFast.scaledDotProductAttention(
                    queries: record.0[0..., 0..., range.start ..< (range.start + range.count), 0...],
                    keys: cachedKeys[0..., 0..., 0 ..< (range.start + range.count), 0...],
                    values: cachedValues[0..., 0..., 0 ..< (range.start + range.count), 0...],
                    scale: scale, mask: .causal, sinks: nil)
            }
            func lean(foldScale: Float, shift: Int32) -> (outputs: [MLXArray], keys: MLXArray)? {
                var prescaled: [MLXArray]? = nil
                var leanKeys = record.1
                if fold {
                    guard let folded = Qwen35VerifyQKNormRoPE.applyPrompt(
                        queryProjection: queryProjection, keyProjection: keyProjection,
                        queryWeight: attention.qNorm.weight, keyWeight: attention.kNorm.weight,
                        queryEps: attention.qNorm.eps, keyEps: attention.kNorm.eps, offsets: offsets,
                        queryHeads: 16, keyHeads: 4, ropeDims: rope.dims, ropeBase: rope.base, scale: foldScale)
                    else { return nil }
                    prescaled = folded.blocks
                    leanKeys = folded.keys
                }
                let outputs = ranges.enumerated().map { (index: Int, range: (start: Int, count: Int)) -> MLXArray in
                    CBv2PromptAttentionCore.composed(
                        query: prescaled?[index]
                            ?? record.0[0..., 0..., range.start ..< (range.start + range.count), 0...],
                        keys: cachedKeys[0..., 0..., 0 ..< (range.start + range.count), 0...],
                        values: cachedValues[0..., 0..., 0 ..< (range.start + range.count), 0...],
                        scale: scale, scaled: fold, fusedSoftmax: smax, softmaxShift: shift)
                }
                return (outputs, leanKeys)
            }
            cases += 1
            let stepped = scale * (1 + 1.0 / 4_194_304)
            guard let run = lean(
                foldScale: foldNegativeControl ? stepped : scale,
                shift: CBv2PromptAttentionCore.negativeControl ? 1 : 0)
            else {
                bad += 1
                continue
            }
            var equal = same(run.keys, record.1)
            for (a, b) in zip(expected, run.outputs) where !same(a, b) { equal = false }
            var controlDiffers = true
            if fold, let control = lean(foldScale: stepped, shift: 0) {
                controlDiffers = zip(expected, control.outputs).contains { (pair: (MLXArray, MLXArray)) -> Bool in
                    !same(pair.0, pair.1)
                }
            }
            if smax, let control = lean(foldScale: scale, shift: 1) {
                controlDiffers = controlDiffers && zip(expected, control.outputs).contains {
                    (pair: (MLXArray, MLXArray)) -> Bool in !same(pair.0, pair.1)
                }
            }
            if !equal || !controlDiffers { bad += 1 }
            guard smax, !fold else { continue }
            for range in ranges {
                let kL = range.start + range.count
                cases += 1
                let draw = MLXRandom.normal([1, 4, 4, range.count, kL], key: keys[3])
                let spread = MLXRandom.uniform(low: -1, high: 5, [1, 4, 4, range.count, 1], key: keys[4])
                let scores = draw * (MLXArray(1) + spread * 12) + MLXRandom.uniform(
                    low: -30, high: 30, [1, 4, 4, range.count, 1], key: keys[5])
                let reference = softmax(
                    MLX.where(
                        greaterEqual(
                            expandedDimensions(MLXArray.arange(kL - range.count, kL), axis: 1),
                            expandedDimensions(MLXArray.arange(0, kL), axis: 0)),
                        scores, MLXArray(-Float.greatestFiniteMagnitude)),
                    axes: [-1], precise: true)
                let fused = CBv2PromptAttentionCore.causalSoftmax(scores, rows: range.count, keyCount: kL)
                let off = CBv2PromptAttentionCore.causalSoftmax(scores, rows: range.count, keyCount: kL, shift: 1)
                if !same(reference, fused) || same(reference, off) { bad += 1 }
            }
        }
        return (bad, cases)
    }
}

enum Qwen35VerifyStartFusion {
    static let offered =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VSTART_FUSE"] != "0"
    static let negativeControl =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VSTART_FUSE_NEGCTL"] == "1"
    nonisolated(unsafe) static var verified = false

    static let source: String = {
        let record = SignedBlockHadamard.fused4096Source(residualNorm: false)
        let head = record.components(separatedBy: "if (sizeof(T) == 4) {")
        precondition(head.count == 2)
        let branches = head[1].components(separatedBy: "} else {")
        precondition(branches.count == 2)
        let read = "float val = float(x[in_offset + r]);"
        let store = "y[batch_idx + idx] = T(val);"
        let buffer = "threadgroup float buf[256 * 17];"
        let offset = "const uint in_offset = batch_idx + uint(base0);"
        let fused = head[0] + "{" + branches.dropFirst().joined(separator: "} else {")
        precondition([read, store, buffer, offset].allSatisfy { fused.components(separatedBy: $0).count == 2 })
        let dequant = "const uint8_t d = (uint(wb[r >> 2]) >> (2 * (r & 3))) & 0x03; const T q = scale * d + bias; float val = float(q);"
        return fused
            .replacingOccurrences(of: offset, with: "")
            .replacingOccurrences(of: buffer, with: """
                const int64_t raw = int64_t(ids[block_idx]); const int64_t tok = raw < 0 ? raw + int64_t(w_shape[0]) : raw;
                const device uint8_t* wb = (const device uint8_t*)(w + tok * 256) + (int(i) << 2);
                const T scale = scales[tok * 32 + (i >> 3)];
                const T bias = biases[tok * 32 + (i >> 3)];
                \(buffer)
                """)
            .replacingOccurrences(of: read, with: dequant)
            .replacingOccurrences(of: store, with: negativeControl ? "y[batch_idx + idx] = val;" : "y[batch_idx + idx] = float(T(val));")
    }()

    static let kernel = MLXFast.metalKernel(
        name: negativeControl ? "qwen35_verify_start_embed_negctl" : "qwen35_verify_start_embed",
        inputNames: ["ids", "w", "scales", "biases", "signs"], outputNames: ["y"],
        source: source, header: SignedBlockHadamard.fused4096Header)

    static func packed(_ model: Qwen35TextModelInner) -> HadamardQuantizedEmbedding? {
        guard let e = model.embedTokens as? HadamardQuantizedEmbedding, e.bits == 2, e.groupSize == 128,
            e.transform.blockSize == 4096, e.transform.width == 4096, e.weight.dtype == .uint32,
            e.weight.ndim == 2, e.weight.dim(1) == 256, e.scales.dtype == .float16, e.scales.dim(1) == 32,
            let biases = e.biases, biases.dtype == .float16, biases.shape == e.scales.shape,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        return e
    }

    static func rows(_ e: HadamardQuantizedEmbedding, _ ids: MLXArray) -> MLXArray {
        kernel(
            [ids, e.weight, e.scales, e.biases!, e.transform.signs],
            template: [("T", DType.float16), ("BLOCKS_PER_ROW", 1), ("INVERSE", true)],
            grid: (256, ids.size, 1), threadGroup: (256, 1, 1),
            outputShapes: [[ids.size, 4096]], outputDTypes: [.float32])[0]
    }

    static func embed(_ model: Qwen35TextModelInner, _ inputs: MLXArray) -> MLXArray? {
        guard verified, inputs.ndim == 2, inputs.dim(0) == 1, (2 ... 16).contains(inputs.dim(1)),
            inputs.dtype == .int32, let e = packed(model)
        else { return nil }
        return rows(e, inputs.reshaped([-1])).reshaped([1, inputs.dim(1), 4096])
    }

    /// `DARKBLOOM_BONSAI_PSTART_FUSE=0` keeps the record chain at prompt width.
    static let promptOffered =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_PSTART_FUSE"] != "0"

    /// The same rows at the prompt widths whose F16 embedding the forward casts to FP32
    /// (`fusePromptBoundaries` with `promptPassed`); the check covers every id.
    static func promptEmbed(_ model: Qwen35TextModelInner, _ inputs: MLXArray) -> MLXArray? {
        guard promptOffered, verified, Qwen35Layer0FP32.promptPassed, inputs.ndim == 2,
            inputs.dim(0) == 1, inputs.dim(1) >= 128, inputs.dim(1) % 64 == 0,
            inputs.dtype == .int32, let e = packed(model)
        else { return nil }
        return rows(e, inputs.reshaped([-1])).reshaped([1, inputs.dim(1), 4096])
    }

    static func prepare(_ model: Qwen35TextModelInner) {
        let note: String
        if !offered {
            note = "off"
        } else if !Qwen35Layer0FP32.verifyPassed {
            note = "off (layer0 fp32 verify off)"
        } else if let e = packed(model) {
            do {
                let (bad, cases) = try withError { check(e) }
                verified = bad == 0
                note = verified
                    ? "on (\(cases) cases bitwise equal: rows 2-16 chain and tree, negative ids, all \(e.weight.dim(0)) ids)"
                    : "off (mismatch: \(bad) of \(cases) cases differ, record chain kept)"
            } catch {
                note = "off (error, record chain kept: \(error))"
            }
        } else {
            note = "off (embedding not 2-bit g128 WHT-4096 F16)"
        }
        FileHandle.standardError.write(Data("qwen35 verify start fusion: \(note)\n".utf8))
        let promptNote = !promptOffered ? "off"
            : !Qwen35Layer0FP32.promptPassed ? "off (layer0 fp32 prompt off)"
            : verified ? "on (same kernel and check)" : "off (verify start fusion off)"
        FileHandle.standardError.write(Data("qwen35 prompt start fusion: \(promptNote)\n".utf8))
    }

    private static func check(_ e: HadamardQuantizedEmbedding) -> (Int, Int) {
        let vocabulary = e.weight.dim(0)
        var batches: [[Int32]] = []
        for count in 2 ... 16 {
            let chain = (0 ..< count).map { Int32((11 + $0 &* 7919 &+ count &* 104_729) % vocabulary) }
            var tree = chain
            for j in 1 ..< count { tree[j] = chain[(j - 1) / 2] }
            tree[count - 1] = Int32(vocabulary - 1)
            tree[0] = 0
            batches += [chain, tree]
        }
        batches.append([-1, Int32(-vocabulary), -2, -7919])
        var start = 0
        while start < vocabulary {
            let n = min(2048, vocabulary - start)
            batches.append((start ..< start + n).map { Int32($0) })
            start += n
        }
        var bad = 0
        for ids in batches {
            let shaped = MLXArray(ids).reshaped([1, ids.count])
            let record = e(shaped).asType(.float32)
            let fused = rows(e, shaped.reshaped([-1])).reshaped(record.shape)
            let same = MLX.all(
                contiguous(record).view(dtype: .uint32) .== contiguous(fused).view(dtype: .uint32))
            if !same.item(Bool.self) { bad += 1 }
        }
        return (bad, batches.count)
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
    let fullAttentionLayerCount: Int

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        let constructedLayers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }
        self.layers = constructedLayers
        self.fullAttentionLayerCount = constructedLayers.reduce(into: 0) { count, layer in
            if !layer.isLinear { count += 1 }
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
        lastPositionOnly: Bool = false,
        deferFinalDelta: Bool = false
    ) -> MLXArray {
        let (hiddenStates, pendingDelta) = cbv2ForwardSplitFinal(
            inputs, inputEmbeddings: inputEmbeddings, caches: caches,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: captureRecurrentWindow,
            lastPositionOnly: lastPositionOnly, deferFinalDelta: deferFinalDelta)
        if let delta = pendingDelta {
            return hiddenStates + delta
        }
        return hiddenStates
    }

    func cbv2ForwardSplitFinal(
        _ inputs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false,
        lastPositionOnly: Bool = false,
        deferFinalDelta: Bool = false
    ) -> (hiddenStates: MLXArray, pendingDelta: MLXArray?) {
        precondition(
            caches.count == fullAttentionLayerCount,
            "Qwen35 CBv2 requires only full-attention caches")
        let shapeCall = CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(liveBatchRows: inputs.dim(0), sequenceWidth: inputs.dim(1)) : nil
        defer { shapeCall?.end() }
        var hiddenStates = inputEmbeddings
            ?? (captureRecurrentWindow && !exactTargetVerify && !lastPositionOnly
                && Qwen35Layer0FP32.verifyPassed ? Qwen35VerifyStartFusion.embed(self, inputs) : nil)
            ?? (!captureRecurrentWindow && !exactTargetVerify
                ? Qwen35VerifyStartFusion.promptEmbed(self, inputs) : nil)
            ?? embedTokens(inputs)
        // Read the tap ONCE. A nil list costs one comparison per layer and
        // allocates nothing; the drafter is not attached on a serial leg.
        let tapLayerIds = dFlash2Tap.layerIds
        var tapped = [MLXArray?](
            repeating: nil, count: tapLayerIds?.count ?? 0)
        var attentionIndex = 0
        // Verify: carry each layer's output as (residual, delta) so the next
        // layer forms the add, its input norm and its codes in one kernel.
        let fuseVerifyBoundaries = captureRecurrentWindow && !exactTargetVerify
            && !lastPositionOnly
            && hiddenStates.ndim == 3 && hiddenStates.dim(0) == 1
            && (2 ... 16).contains(hiddenStates.dim(1))
        let fusePromptBoundaries = !captureRecurrentWindow && !exactTargetVerify
            && hiddenStates.ndim == 3 && hiddenStates.dim(0) == 1
            && hiddenStates.dim(1) >= 128 && hiddenStates.dim(1) % 64 == 0
        let promptRows = fusePromptBoundaries ? hiddenStates.dim(1) : 0
        if hiddenStates.dtype == .float16,
            fusePromptBoundaries
                ? Qwen35Layer0FP32.promptPassed
                : fuseVerifyBoundaries && Qwen35Layer0FP32.verifyPassed
        {
            hiddenStates = hiddenStates.asType(.float32)
        }
        var pendingDelta: MLXArray? = nil
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
            let layerLastOnly = lastPositionOnly && !captureRecurrentWindow
                && modelLayerIndex == layers.count - 1
                && !(tapLayerIds?.contains(modelLayerIndex) ?? false)
            if fuseVerifyBoundaries, let delta = pendingDelta,
                let fused = layer.cbv2ForwardVerifyFused(
                    hiddenStates, delta: delta, modelLayerIndex: modelLayerIndex,
                    attentionCache: attentionCache, recurrentState: recurrentState,
                    positionIds: positionIds)
            {
                if let tapLayerIds, let prevSlot = tapLayerIds.firstIndex(of: modelLayerIndex - 1) {
                    tapped[prevSlot] = fused.inputResidual
                }
                (hiddenStates, pendingDelta) = (fused.residual, fused.delta)
                if Self.submitsVerify(after: modelLayerIndex, of: layers.count) {
                    asyncEval(hiddenStates, fused.delta)
                }
                continue
            }
            if fusePromptBoundaries, let delta = pendingDelta,
                let fused = layer.cbv2ForwardPromptFused(
                    hiddenStates, delta: delta, modelLayerIndex: modelLayerIndex,
                    attentionCache: attentionCache, recurrentState: recurrentState,
                    positionIds: positionIds, lastPositionOnly: layerLastOnly,
                    foldDown: modelLayerIndex + 2 < layers.count)
            {
                if let tapLayerIds, let prevSlot = tapLayerIds.firstIndex(of: modelLayerIndex - 1) {
                    tapped[prevSlot] = fused.inputResidual
                }
                (hiddenStates, pendingDelta) = (fused.residual, fused.delta)
                if Self.submitsVerify(after: modelLayerIndex, of: layers.count) {
                    if let delta = fused.delta { asyncEval(hiddenStates, delta) } else { asyncEval(hiddenStates) }
                }
                if fused.delta == nil, let tapLayerIds, let slot = tapLayerIds.firstIndex(of: modelLayerIndex) {
                    tapped[slot] = hiddenStates
                }
                continue
            }
            if let delta = pendingDelta {
                hiddenStates = hiddenStates + delta
                pendingDelta = nil
                if let tapLayerIds, let prevSlot = tapLayerIds.firstIndex(of: modelLayerIndex - 1) {
                    tapped[prevSlot] = hiddenStates
                }
            }
            if fuseVerifyBoundaries, modelLayerIndex + 1 < layers.count,
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
            if fusePromptBoundaries, modelLayerIndex + 1 < layers.count,
                let split = layer.cbv2ForwardPromptSplitOutput(
                    hiddenStates, modelLayerIndex: modelLayerIndex,
                    attentionCache: attentionCache, recurrentState: recurrentState,
                    positionIds: positionIds, foldDown: modelLayerIndex + 2 < layers.count)
            {
                (hiddenStates, pendingDelta) = (split.residual, split.delta)
                if Self.submitsVerify(after: modelLayerIndex, of: layers.count) {
                    if let delta = split.delta { asyncEval(hiddenStates, delta) } else { asyncEval(hiddenStates) }
                }
                if split.delta == nil, let tapLayerIds, let slot = tapLayerIds.firstIndex(of: modelLayerIndex) {
                    tapped[slot] = hiddenStates
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
                lastPositionOnly: layerLastOnly)
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
            if (captureRecurrentWindow || !lastPositionOnly || inputs.dim(1) >= 64),
                Self.submitsVerify(after: modelLayerIndex, of: layers.count)
            {
                asyncEval(hiddenStates)
            }
        }
        // After the layers are built, so the wide-row load checks this boot
        // pass triggers have already chosen the verify kernels being checked.
        Qwen35Layer0FP32.prepareIfBoot(self, promptRows: promptRows)
        if let delta = pendingDelta {
            if let tapLayerIds, let lastSlot = tapLayerIds.firstIndex(of: layers.count - 1) {
                hiddenStates = hiddenStates + delta
                pendingDelta = nil
                tapped[lastSlot] = hiddenStates
            } else if !deferFinalDelta {
                hiddenStates = hiddenStates + delta
                pendingDelta = nil
            }
        }
        if tapLayerIds == nil {
            dFlash2Tap.tappedHidden = nil
        } else {
            let unpacked = tapped.map { $0! }
            if unpacked.count == 8, unpacked[0].ndim == 3, unpacked[0].dim(0) == 1,
                unpacked[0].dim(2) == 4096, unpacked.allSatisfy({ $0.dtype == .float32 })
            {
                let s = unpacked[0].dim(1)
                dFlash2Tap.tappedHidden = Self.packTapsBF16Kernel(
                    unpacked,
                    grid: (8192, s, 1),
                    threadGroup: (256, 1, 1),
                    outputShapes: [[1, s, 32768]],
                    outputDTypes: [.bfloat16]
                )[0]
            } else {
                dFlash2Tap.tappedHidden = concatenated(unpacked, axis: -1)
            }
        }
        return (hiddenStates, pendingDelta)
    }

    private static let packTapsBF16Kernel = MLXFast.metalKernel(
        name: "qwen35_pack_taps8_bf16",
        inputNames: ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"],
        outputNames: ["out"],
        source: """
            const uint col4 = thread_position_in_grid.x;
            const uint s = thread_position_in_grid.y;
            const uint slot = col4 >> 10;
            const uint c4 = col4 & 1023u;
            const size_t srcIdx = (size_t)s * 1024u + c4;
            const device float4* src = slot == 0 ? (const device float4*)t0
                : slot == 1 ? (const device float4*)t1
                : slot == 2 ? (const device float4*)t2
                : slot == 3 ? (const device float4*)t3
                : slot == 4 ? (const device float4*)t4
                : slot == 5 ? (const device float4*)t5
                : slot == 6 ? (const device float4*)t6
                : (const device float4*)t7;
            const float4 v = src[srcIdx];
            device bfloat4* dst = (device bfloat4*)out;
            dst[(size_t)s * 8192u + col4] = bfloat4(bfloat(v.x), bfloat(v.y), bfloat(v.z), bfloat(v.w));
            """
    )
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// The fused output-head top-two of the last captured verify forward.
    let verifyHeadTopTwo = Qwen35VerifyHeadTopTwoSlot()

    /// The drafter's prefix head (`DFlash2DraftHeadSlot`), off the module tree.
    let dFlash2DraftHead = DFlash2DraftHeadSlot()

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
        let (hiddenRaw, pendingDelta) = model.cbv2ForwardSplitFinal(
            inputs, inputEmbeddings: inputEmbedding, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            lastPositionOnly: lastPositionOnly, deferFinalDelta: lastPositionOnly && lmHead != nil)
        if let delta = pendingDelta, let lmHead {
            let xLast = lastPositionOnly && hiddenRaw.dim(1) > 1
                ? hiddenRaw[0..., (hiddenRaw.dim(1) - 1)..., 0...] : hiddenRaw
            let dLast = lastPositionOnly && delta.dim(1) > 1
                ? delta[0..., (delta.dim(1) - 1)..., 0...] : delta
            if let fused = HadamardQuantizedLinear.sharedResidualNormProjections(
                xLast, residual: dLast, weight: model.norm.weight, eps: model.norm.eps,
                projections: [lmHead])
            {
                return fused.projections[0]
            }
        }
        let hidden = pendingDelta.map { hiddenRaw + $0 } ?? hiddenRaw
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
/// exactly one row. The trunk — every K/V write, every recurrent-state
/// stage, positions, embeddings — is byte-identical to `positionedForward`;
/// final RMSNorm is row-independent, so norm-after-slice equals
/// slice-after-norm for the surviving row. Decode, MTP draft/verify, and
/// capture paths keep their existing full contracts.
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
        let (hiddenRaw, pendingDelta) = model.cbv2ForwardSplitFinal(
            inputs, inputEmbeddings: inputEmbedding, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            lastPositionOnly: true, deferFinalDelta: lmHead != nil)
        switch requirement {
        case .evaluationOnly:
            let hidden = pendingDelta.map { hiddenRaw + $0 } ?? hiddenRaw
            // Small handle whose graph depends on the whole trunk — forcing
            // it commits every layer's K/V write and recurrent stage.
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            if let delta = pendingDelta, let lmHead, hiddenRaw.dim(1) == 1,
                let fused = HadamardQuantizedLinear.sharedResidualNormProjections(
                    hiddenRaw, residual: delta, weight: model.norm.weight, eps: model.norm.eps,
                    projections: [lmHead])
            {
                return fused.projections[0][0..., 0, 0...]
            }
            let hidden = pendingDelta.map { hiddenRaw + $0 } ?? hiddenRaw
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
        let lastOnly = model.dFlash2Tap.layerIds != nil
        let (hiddenRaw, pendingDelta) = model.cbv2ForwardSplitFinal(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            lastPositionOnly: lastOnly, deferFinalDelta: lastOnly && lmHead != nil)
        switch requirement {
        case .evaluationOnly:
            let hidden = pendingDelta.map { hiddenRaw + $0 } ?? hiddenRaw
            return (hidden[0..., (hidden.dim(1) - 1)..., 0 ..< 1], hidden)
        case .lastPositionLogits:
            if let delta = pendingDelta, let lmHead, hiddenRaw.dim(1) == 1,
                let fused = HadamardQuantizedLinear.sharedResidualNormProjections(
                    hiddenRaw, residual: delta, weight: model.norm.weight, eps: model.norm.eps,
                    projections: [lmHead])
            {
                return (fused.projections[0], fused.residual)
            }
            let hidden = pendingDelta.map { hiddenRaw + $0 } ?? hiddenRaw
            let last = model.norm(hidden[0..., (hidden.dim(1) - 1)..., 0...])
            let logits = lmHead.map { $0(last) } ?? model.embedTokens.asLinear(last)
            return (logits, hidden)
        }
    }
}

extension Qwen35TextModel: CBv2RecurrentCaptureMTPForwardable {
    /// `MLXFAST_MTP_DIAGNOSTICS` unset, read once at first use instead of
    /// copying the process environment on every verify forward
    /// (`DARKBLOOM_BONSAI_HOSTCUT=0` reads it per call again).
    private static let mtpDiagnosticsUnset: Bool =
        ProcessInfo.processInfo.environment["MLXFAST_MTP_DIAGNOSTICS"] == nil

    fileprivate static func mtpDiagnosticsOff() -> Bool {
        HostConstants.enabled
            ? mtpDiagnosticsUnset
            : ProcessInfo.processInfo.environment["MLXFAST_MTP_DIAGNOSTICS"] == nil
    }

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
        let canDeferFinal = !model.exactTargetVerify && (lmHead is HadamardQuantizedLinear)
        let (hiddenRaw, pendingDelta) = model.cbv2ForwardSplitFinal(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: true, deferFinalDelta: canDeferFinal)
        verifyHeadTopTwo.value = nil
        if let delta = pendingDelta,
            !model.exactTargetVerify,
            let head = lmHead as? HadamardQuantizedLinear,
            Qwen35InlineMTPAssistant.shortlistSize == nil,
            Qwen35TextModel.mtpDiagnosticsOff(),
            let fusedTopTwo = head.topTwoRowsAfterResidualNorm(
                hiddenRaw, delta: delta, weight: model.norm.weight, eps: model.norm.eps)
        {
            verifyHeadTopTwo.value = (fusedTopTwo.ids, fusedTopTwo.values)
            let hidden = fusedTopTwo.residual
            let logits = broadcast(
                hidden[0..., 0..., 0 ..< 1],
                to: [hidden.dim(0), hidden.dim(1), head.weight.dim(0)])
            return (logits, hidden)
        }
        let hidden = pendingDelta.map { hiddenRaw + $0 } ?? hiddenRaw
        let normalized = model.norm(hidden)
        let logits: MLXArray
        if let lmHead {
            if !model.exactTargetVerify,
                let head = lmHead as? HadamardQuantizedLinear,
                let topTwo = head.topTwoRows(normalized)
            {
                verifyHeadTopTwo.value = topTwo
                if Qwen35InlineMTPAssistant.shortlistSize == nil,
                    Qwen35TextModel.mtpDiagnosticsOff()
                {
                    logits = broadcast(
                        normalized[0..., 0..., 0 ..< 1],
                        to: [normalized.dim(0), normalized.dim(1), head.weight.dim(0)])
                } else {
                    logits = lmHead(normalized)
                }
            } else {
                logits = model.exactTargetVerify
                    ? qwen35A3BExactW4G64Projection(lmHead, normalized)
                    : lmHead(normalized)
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

    public func draftLogitsForDFlash2Hidden(_ hidden: MLXArray, rows: Int) -> MLXArray {
        guard let full = lmHead as? HadamardQuantizedLinear,
            let prefix = dFlash2DraftHead.head(of: full, rows: rows)
        else { return logitsForDFlash2Hidden(hidden) }
        return prefix(hidden)
    }

    public func draftShortlistForDFlash2Hidden(
        _ hidden: MLXArray, rows: Int
    ) -> (ids: MLXArray, values: MLXArray)? {
        guard let full = lmHead as? HadamardQuantizedLinear else { return nil }
        return dFlash2DraftHead.shortlist(hidden, of: full, rows: rows)
    }

    /// One exact top-2 at `rows` rows over a fixed normalized activation: the
    /// head's lazily built constants and its one-time bit-exact self-check at
    /// that row count run here, at load, instead of inside the first round.
    public func prepareDFlash2Verify(rows: Int) {
        Qwen35RegisterGatedSIMD.prepareHoistedReplay()
        if rows == 16 {
            Qwen35FusedGDNVerify.prepare()
            CBv2VerifyAttentionPartial.prepareLean()
        }
        guard let head = lmHead as? HadamardQuantizedLinear else { return }
        let x = MLXRandom.normal([1, rows, configuration.hiddenSize], key: MLXRandom.key(0xDF2))
        if let topTwo = head.topTwoRows(model.norm(x)) { eval(topTwo.ids, topTwo.values) }
        if let fused = head.topTwoRowsAfterResidualNorm(
            x, delta: x, weight: model.norm.weight, eps: model.norm.eps)
        {
            eval(fused.ids, fused.values, fused.residual)
        }
        HadamardQuantizedLinear.prepareSharedVerifyWeights()
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

    public func draftLogitsForDFlash2Hidden(_ hidden: MLXArray, rows: Int) -> MLXArray {
        languageModel.draftLogitsForDFlash2Hidden(hidden, rows: rows)
    }

    public func draftShortlistForDFlash2Hidden(
        _ hidden: MLXArray, rows: Int
    ) -> (ids: MLXArray, values: MLXArray)? {
        languageModel.draftShortlistForDFlash2Hidden(hidden, rows: rows)
    }

    public func prepareDFlash2Verify(rows: Int) {
        languageModel.prepareDFlash2Verify(rows: rows)
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
