// Adapted from PrismML-Eng/mlx-swift 6d3a84de28225d1f5bc0a56f5c781596997242f9 (MIT).
// Preserve the published Bonsai pack's FP32 transform / original output dtype contract.
import Foundation
import MLX
import os


/// Invalid transform metadata or incompatible packed weights.
public enum HadamardError: Error {
    case invalidBlockSize
    case invalidSigns
    case incompatibleShape
    case unsupportedContract
    case missingSignWidth(Int)
}

public struct SignedBlockHadamard {
    public let blockSize: Int
    public let width: Int
    private let signs: MLXArray
    private let signValues: [Float]

    public init(blockSize: Int, signs: [Float]) throws {
        guard blockSize > 0, blockSize <= 8192,
            blockSize & (blockSize - 1) == 0
        else { throw HadamardError.invalidBlockSize }
        guard !signs.isEmpty, signs.count % blockSize == 0,
            signs.allSatisfy({ $0 == -1 || $0 == 1 })
        else { throw HadamardError.invalidSigns }
        self.blockSize = blockSize
        self.width = signs.count
        self.signs = MLXArray(signs)
        self.signValues = signs
    }

    public func matches(signs values: [Float]) -> Bool { values == signValues }

    func isIdentical(to other: SignedBlockHadamard) -> Bool {
        blockSize == other.blockSize && width == other.width && signValues == other.signValues
    }

    // Fused transform from Yukon 9727ebfe (9B PR #35); native MLX radix order.
    private static let residualNormSource = """

            threadgroup float norm_sums[32];
            const uint norm_lane = thread_index_in_simdgroup;
            const uint norm_sg = simdgroup_index_in_threadgroup;
            for (uint part = 0; part < 4; ++part) {
              const uint at = batch_idx + (uint(i) + part * 256u) * 4u;
              const float4 value = *((const device float4*)(x + at))
                  + *((const device float4*)(residual + at));
              float square_sum = 0.0f;
              #pragma clang loop unroll(full)
              for (int j = 0; j < 4; ++j) square_sum += value[j] * value[j];
              square_sum = simd_sum(square_sum);
              if (norm_lane == 0) norm_sums[norm_sg + part * 8u] = square_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Every simdgroup reduces the same 32 partials in the same lanes,
            // so no second barrier is needed to share the result.
            const float norm_total = simd_sum(norm_sums[norm_lane]);
            const float norm_inv = metal::precise::rsqrt(norm_total / 4096.0f + epsilon[0]);
            """

    private static func fused4096Source(residualNorm: Bool) -> String {
        let prefix = residualNorm ? residualNormSource : ""
        let valueRead = residualNorm ? """
            const float4 raw = in4[v] + *((const device float4*)(residual + in_offset) + v);
            *((device float4*)(residual_out + in_offset) + v) = raw;
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + v);
            float4 val = (weight * (raw * norm_inv)) * s4[v];
            """ : """
            float4 val = in4[v];
            if (!INVERSE) {
              val *= s4[v];
            }
            """
        return """
            #pragma clang fp reassociate(off)
            const short i = short(thread_position_in_threadgroup.x);
            const uint block_idx = threadgroup_position_in_grid.y;
            const uint batch_idx = block_idx * 4096u;
            const uint sign_offset = (BLOCKS_PER_ROW == 1)
                ? 0u : (block_idx % uint(BLOCKS_PER_ROW)) * 4096u;

            \(prefix)
            threadgroup float buf[256 * 17];
            float x_reg[16];

            const short base0 = i << 4;
            const uint in_offset = batch_idx + uint(base0);
            const uint s_offset = sign_offset + uint(base0);
            if (sizeof(T) == 4) {
              const device float4* in4 = (const device float4*)(x + in_offset);
              const device float4* s4 = (const device float4*)(signs + s_offset);
              #pragma clang loop unroll(full)
              for (short v = 0; v < 4; v++) {
                \(valueRead)
                x_reg[4 * v + 0] = val.x;
                x_reg[4 * v + 1] = val.y;
                x_reg[4 * v + 2] = val.z;
                x_reg[4 * v + 3] = val.w;
              }
            } else {
              #pragma clang loop unroll(full)
              for (short r = 0; r < 16; r++) {
                float val = float(x[in_offset + r]);
                if (!INVERSE) {
                  val *= signs[s_offset + r];
                }
                x_reg[r] = val;
              }
            }
            hadamard_radix16(x_reg);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              buf[i * 17 + r] = x_reg[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short k1 = i & 15;
            const short r_base2 = (i - k1);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              x_reg[r] = buf[(r_base2 + r) * 17 + k1];
            }
            hadamard_radix16(x_reg);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              buf[(r_base2 + r) * 17 + k1] = x_reg[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short i_hi = i >> 4;
            const short i_lo = i & 15;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              x_reg[r] = buf[(i_hi + 16 * r) * 17 + i_lo];
            }
            hadamard_radix16(x_reg);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              const uint idx = uint(i + 256 * r);
              float val = x_reg[r] * 0.015625f;
              if (INVERSE) {
                val *= signs[sign_offset + idx];
              }
              y[batch_idx + idx] = T(val);
            }
            """
    }

    private static let fused4096Header = """
            using namespace metal;
            METAL_FUNC void hadamard_radix16(thread float* x) {
              #pragma clang loop unroll(full)
              for (short j = 0; j < 16; j += 2) {
                float a = x[j], b = x[j+1];
                x[j] = a + b; x[j+1] = a - b;
              }
              #pragma clang loop unroll(full)
              for (short j = 0; j < 16; j += 4) {
                #pragma clang loop unroll(full)
                for (short k = 0; k < 2; k++) {
                  float a = x[j+k], b = x[j+k+2];
                  x[j+k] = a + b; x[j+k+2] = a - b;
                }
              }
              #pragma clang loop unroll(full)
              for (short j = 0; j < 16; j += 8) {
                #pragma clang loop unroll(full)
                for (short k = 0; k < 4; k++) {
                  float a = x[j+k], b = x[j+k+4];
                  x[j+k] = a + b; x[j+k+4] = a - b;
                }
              }
              #pragma clang loop unroll(full)
              for (short k = 0; k < 8; k++) {
                float a = x[k], b = x[k+8];
                x[k] = a + b; x[k+8] = a - b;
              }
            }
            """

    private static let fused4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_4096", inputNames: ["x", "signs"], outputNames: ["y"],
        source: fused4096Source(residualNorm: false), header: fused4096Header)

    private static let residualNorm4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_residual_norm_4096",
        inputNames: ["x", "residual", "norm_weight", "epsilon", "signs"],
        outputNames: ["y", "residual_out"],
        source: fused4096Source(residualNorm: true), header: fused4096Header)

    private func fused4096(_ x: MLXArray, inverse: Bool) -> MLXArray? {
        guard blockSize == 4096, x.size > 0, x.size <= Int(UInt32.max),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        return Self.fused4096Kernel(
            [x, signs], template: [("T", x.dtype), ("BLOCKS_PER_ROW", width / 4096),
                ("INVERSE", inverse)], grid: (256, x.size / 4096, 1),
            threadGroup: (256, 1, 1), outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }

    // GDN output gate formed inside the quantizing rotation: per-head (128
    // wide) RMSNorm of `x` with the arithmetic of MLX `rms_single_row` at
    // axis 128 (one simdgroup; lane l sums elements 4l..4l+3 in order, one
    // `simd_sum`, `precise::rsqrt(total / 128 + eps)`, `w * (x * inv)`),
    // times MLX `silu(gate)` (`gate * sigmoid(gate)` with MLX's stable
    // sigmoid), the product taken as `silu * normed`. Each head's 32 lane
    // partials are placed in the lanes MLX gives them before the `simd_sum`.
    private static let gatedNormSource = """

            threadgroup float head_part[32 * 32];
            {
              const device float4* nx4 = (const device float4*)(x + batch_idx + (uint(i) << 4));
              const uint part_base = (uint(i) >> 3) * 32u + (uint(i) & 7u) * 4u;
              #pragma clang loop unroll(full)
              for (short w = 0; w < 4; w++) {
                const float4 nv = nx4[w];
                float acc = 0.0f;
                acc += nv.x * nv.x;
                acc += nv.y * nv.y;
                acc += nv.z * nv.z;
                acc += nv.w * nv.w;
                head_part[part_base + uint(w)] = acc;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Simdgroup s reduces its own four heads 4s ... 4s + 3 (same
            // partials, same lanes) and each thread keeps its head's factor.
            float head_inv_value = 0.0f;
            {
              const uint norm_lane = thread_index_in_simdgroup;
              const uint head_base = simdgroup_index_in_threadgroup * 4u;
              for (uint k = 0; k < 4u; ++k) {
                const float total = simd_sum(head_part[(head_base + k) * 32u + norm_lane]);
                const float inv = metal::precise::rsqrt(total / 128.0f + epsilon[0]);
                if ((norm_lane >> 3) == k) head_inv_value = inv;
              }
            }
            """

    private static func quantized4096Source(
        swiglu: Bool, residualNorm: Bool = false, gatedNorm: Bool = false,
        attentionGate: Bool = false, normalizedOutput: Bool = false,
        rotatedOutput: Bool = false, normOnly: Bool = false, headError2: Bool = false
    ) -> String {
        if normOnly {
            return normQuantized4096Source(normalizedOutput: normalizedOutput)
        }
        let valueRead = swiglu ? """
            const float4 g = in4[w];
            const float4 u = *((const device float4*)(up + in_offset) + w);
            const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(g)));
            const float4 sigmoid_g = select(1.0f - e, e, g < 0.0f);
            float4 val = ((g * sigmoid_g) * u) * s4[w];
            """ : "float4 val = in4[w] * s4[w];"
        let prefix = residualNorm ? residualNormSource : gatedNorm ? gatedNormSource : ""
        let rotatedWrite = rotatedOutput ? """
            device float4* rot4 = (device float4*)(rotated_out + batch_idx + uint(base0));
            #pragma clang loop unroll(full)
            for (short w = 0; w < 4; w++) {
              rot4[w] = float4(v[4 * w + 0], v[4 * w + 1], v[4 * w + 2], v[4 * w + 3]);
            }
            """ : ""
        let inputRead = residualNorm && normalizedOutput ? """
            const float4 raw = in4[w] + *((const device float4*)(residual + in_offset) + w);
            *((device float4*)(residual_out + in_offset) + w) = raw;
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
            const float4 normed = weight * (raw * norm_inv);
            *((device float4*)(normed_out + in_offset) + w) = normed;
            float4 val = normed * s4[w];
            """ : residualNorm ? """
            const float4 raw = in4[w] + *((const device float4*)(residual + in_offset) + w);
            *((device float4*)(residual_out + in_offset) + w) = raw;
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
            float4 val = (weight * (raw * norm_inv)) * s4[w];
            """ : attentionGate ? """
            const uint ag_e = uint(base0) + 4u * uint(w);
            const uint ag_row = block_idx;
            const float4 attn = *((const device float4*)(x
                + ((ag_e >> 8) * uint(VERIFY_ROWS) + ag_row) * 256u + (ag_e & 255u)));
            const float4 g = *((const device float4*)(gate
                + ag_row * 8192u + (ag_e >> 8) * 512u + 256u + (ag_e & 255u)));
            const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(g)));
            const float4 sigmoid_g = select(1.0f - e, e, g < 0.0f);
            float4 val = (attn * sigmoid_g) * s4[w];
            """ : gatedNorm ? """
            const float4 g = *((const device float4*)(gate + in_offset) + w);
            const float4 nw = *((const device float4*)(norm_weight + ((uint(i) & 7u) << 4)) + w);
            const float4 normed = nw * (in4[w] * head_inv_value);
            const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(g)));
            const float4 sigmoid_g = select(1.0f - e, e, g < 0.0f);
            float4 val = ((g * sigmoid_g) * normed) * s4[w];
            """ : valueRead
        return """
            #pragma clang fp reassociate(off)
            const short i = short(thread_position_in_threadgroup.x);
            const uint block_idx = threadgroup_position_in_grid.y;
            const uint batch_idx = block_idx * 4096u;
            const uint sign_offset = (BLOCKS_PER_ROW == 1)
                ? 0u : (block_idx % uint(BLOCKS_PER_ROW)) * 4096u;
            if (VERIFY_ROWS > 0 && block_idx >= uint(VERIFY_ROWS * BLOCKS_PER_ROW)) {
              device char4* pad = (device char4*)(codes + batch_idx + uint(i) * 16u);
              pad[0] = char4(0); pad[1] = char4(0); pad[2] = char4(0); pad[3] = char4(0);
              return;
            }

            \(prefix)
            threadgroup float buf[256 * 17];
            float x_reg[16];

            // Same passes and buffer layout as fused4096Kernel: element e
            // lives at buf[(e >> 4) * 17 + (e & 15)] between passes.
            const short base0 = i << 4;
            const uint in_offset = batch_idx + uint(base0);
            const uint s_offset = sign_offset + uint(base0);
            const device float4* in4 = (const device float4*)(x + in_offset);
            const device float4* s4 = (const device float4*)(signs + s_offset);
            #pragma clang loop unroll(full)
            for (short w = 0; w < 4; w++) {
              \(inputRead)
              x_reg[4 * w + 0] = val.x;
              x_reg[4 * w + 1] = val.y;
              x_reg[4 * w + 2] = val.z;
              x_reg[4 * w + 3] = val.w;
            }
            hadamard_radix16_q8(x_reg);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              buf[i * 17 + r] = x_reg[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short k1 = i & 15;
            const short r_base2 = (i - k1);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              x_reg[r] = buf[(r_base2 + r) * 17 + k1];
            }
            hadamard_radix16_q8(x_reg);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              buf[(r_base2 + r) * 17 + k1] = x_reg[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short i_hi = i >> 4;
            const short i_lo = i & 15;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              x_reg[r] = buf[(i_hi + 16 * r) * 17 + i_lo];
            }
            hadamard_radix16_q8(x_reg);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              buf[(i_hi + 16 * r) * 17 + i_lo] = x_reg[r] * 0.015625f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float v[16];
            float amax = 0.0f;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              v[r] = buf[i * 17 + r];
              amax = max(amax, fabs(v[r]));
            }
            \(rotatedWrite)
            amax = max(amax, simd_shuffle_xor(amax, ushort(1)));
            amax = max(amax, simd_shuffle_xor(amax, ushort(2)));
            amax = max(amax, simd_shuffle_xor(amax, ushort(4)));
            const float qs = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
            const float iqs = amax > 0.0f ? metal::precise::divide(127.0f, amax) : 0.0f;
            float q[16];
            float part = 0.0f;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              q[r] = rint(v[r] * iqs);
              part += q[r];
            }
            part += simd_shuffle_xor(part, ushort(1));
            part += simd_shuffle_xor(part, ushort(2));
            part += simd_shuffle_xor(part, ushort(4));
            \(headError2 ? ExactVerifyHead.error2Source(elements: 16, cluster: 8) : "")
            device char4* dst = (device char4*)(codes + batch_idx + uint(base0));
            #pragma clang loop unroll(full)
            for (short w = 0; w < 4; w++) {
              dst[w] = char4(char(q[w]), char(q[4 + w]), char(q[8 + w]), char(q[12 + w]));
            }
            if ((i & 7) == 0) {
              const uint row = block_idx / uint(BLOCKS_PER_ROW);
              const uint kg = uint(BLOCKS_PER_ROW) * 32u;
              const uint g = (block_idx % uint(BLOCKS_PER_ROW)) * 32u + uint(i >> 3);
              const uint ml = row & 63u;
              const size_t qidx = VERIFY_ROWS > 0 ? size_t(row) * size_t(kg) + size_t(g)
                  : size_t(row >> 6) * size_t(kg) * 64 + size_t(g) * 64
                  + size_t(((ml >> 4) & 1u) * 32u + (ml & 7u) * 4u + ((ml >> 5) & 1u) * 2u + ((ml >> 3) & 1u));
              ascale[qidx] = qs;
              rsb[qidx] = qs * part;
              \(headError2 ? "qerror2[qidx] = head_error2;" : "")
            }
            """
    }

    private static func normQuantized4096Source(normalizedOutput: Bool) -> String {
        let prefix = residualNormSource.replacingOccurrences(
            of: "(x + at))\n      + *((const device float4*)(residual + at));", with: "(x + at));")
        precondition(prefix != residualNormSource, "norm-only prefix not formed")
        let read = normalizedOutput ? """
            const float4 raw = in4[w];
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
            const float4 normed = weight * (raw * norm_inv);
            *((device float4*)(normed_out + in_offset) + w) = normed;
            float4 val = normed * s4[w];
            """ : """
            const float4 raw = in4[w];
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
            float4 val = (weight * (raw * norm_inv)) * s4[w];
            """
        let residualSource = quantized4096Source(
            swiglu: false, residualNorm: true, normalizedOutput: normalizedOutput)
        let residualRead = normalizedOutput ? """
            const float4 raw = in4[w] + *((const device float4*)(residual + in_offset) + w);
            *((device float4*)(residual_out + in_offset) + w) = raw;
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
            const float4 normed = weight * (raw * norm_inv);
            *((device float4*)(normed_out + in_offset) + w) = normed;
            float4 val = normed * s4[w];
            """ : """
            const float4 raw = in4[w] + *((const device float4*)(residual + in_offset) + w);
            *((device float4*)(residual_out + in_offset) + w) = raw;
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
            float4 val = (weight * (raw * norm_inv)) * s4[w];
            """
        let source = residualSource
            .replacingOccurrences(of: residualNormSource, with: prefix)
            .replacingOccurrences(of: residualRead, with: read)
        precondition(!source.contains("residual"), "norm-only source still reads the residual")
        return source
    }

    private static let quantized4096Header = """
            using namespace metal;
            METAL_FUNC void hadamard_radix16_q8(thread float* x) {
              #pragma clang loop unroll(full)
              for (short j = 0; j < 16; j += 2) {
                float a = x[j], b = x[j+1];
                x[j] = a + b; x[j+1] = a - b;
              }
              #pragma clang loop unroll(full)
              for (short j = 0; j < 16; j += 4) {
                #pragma clang loop unroll(full)
                for (short k = 0; k < 2; k++) {
                  float a = x[j+k], b = x[j+k+2];
                  x[j+k] = a + b; x[j+k+2] = a - b;
                }
              }
              #pragma clang loop unroll(full)
              for (short j = 0; j < 16; j += 8) {
                #pragma clang loop unroll(full)
                for (short k = 0; k < 4; k++) {
                  float a = x[j+k], b = x[j+k+4];
                  x[j+k] = a + b; x[j+k+4] = a - b;
                }
              }
              #pragma clang loop unroll(full)
              for (short k = 0; k < 8; k++) {
                float a = x[k], b = x[k+8];
                x[k] = a + b; x[k+8] = a - b;
              }
            }
            """

    private static let quantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_4096_q8",
        inputNames: ["x", "signs"], outputNames: ["codes", "ascale", "rsb"],
        source: quantized4096Source(swiglu: false), header: quantized4096Header,
        ensureRowContiguous: true)

    private static let quantized4096RotatedKernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_4096_q8_rot",
        inputNames: ["x", "signs"], outputNames: ["codes", "ascale", "rsb", "rotated_out"],
        source: quantized4096Source(swiglu: false, rotatedOutput: true),
        header: quantized4096Header, ensureRowContiguous: true)

    private static let quantizedSwiGLU4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_swiglu_4096_q8",
        inputNames: ["x", "up", "signs"], outputNames: ["codes", "ascale", "rsb"],
        source: quantized4096Source(swiglu: true), header: quantized4096Header,
        ensureRowContiguous: true)

    private static let gatedNormQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_gated_norm_4096_q8",
        inputNames: ["x", "gate", "norm_weight", "epsilon", "signs"],
        outputNames: ["codes", "ascale", "rsb"],
        source: quantized4096Source(swiglu: false, gatedNorm: true),
        header: quantized4096Header, ensureRowContiguous: true)

    private static let attentionGateQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_attention_gate_4096_q8",
        inputNames: ["x", "gate", "signs"],
        outputNames: ["codes", "ascale", "rsb"],
        source: quantized4096Source(swiglu: false, attentionGate: true),
        header: quantized4096Header, ensureRowContiguous: true)

    func quantizedForVerifyAttentionGate(
        _ attention: MLXArray, queryProjection: MLXArray
    ) -> PromptInt8Activation? {
        guard blockSize == 4096, width == 4096, attention.dtype == .float32,
            queryProjection.dtype == .float32, attention.ndim == 4, attention.dim(0) == 1,
            attention.dim(1) == 16, attention.dim(3) == 256,
            queryProjection.size == attention.dim(2) * 8192,
            VerifyInt8Matmul.rowRange.contains(attention.dim(2)),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = attention.dim(2)
        let outputs = Self.wideRowVerify(
            .attentionGate, [attention, queryProjection.reshaped([rows, 8192]), signs], rows: rows)
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    private static let attentionGatePromptQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_attention_gate_prompt_4096_q8",
        inputNames: ["x", "gate", "signs"],
        outputNames: ["codes", "ascale", "rsb"],
        source: quantized4096Source(swiglu: false, attentionGate: true).replacingOccurrences(
            of: "* uint(VERIFY_ROWS) + ag_row", with: "* uint(ATTENTION_ROWS) + ag_row"),
        header: quantized4096Header, ensureRowContiguous: true)

    func quantizedForPromptAttentionGate(
        _ attention: MLXArray, queryProjection: MLXArray
    ) -> PromptInt8Activation? {
        guard blockSize == 4096, width == 4096, attention.dtype == .float32,
            queryProjection.dtype == .float32, attention.ndim == 4, attention.dim(0) == 1,
            attention.dim(1) == 16, attention.dim(3) == 256,
            queryProjection.size == attention.dim(2) * 8192,
            attention.dim(2) > 0, attention.dim(2) % 64 == 0, queryProjection.size <= Int(Int32.max),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = attention.dim(2)
        let groups = rows * (width / 128)
        let outputs = Self.attentionGatePromptQuantized4096Kernel(
            [attention, queryProjection.reshaped([rows, 8192]), signs],
            template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0), ("ATTENTION_ROWS", rows)],
            grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, width], [groups], [groups]],
            outputDTypes: [.int8, .float32, .float32])
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    func quantizedForVerifyGatedNorm(
        _ x: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float
    ) -> PromptInt8Activation? {
        let rows = x.size / width
        guard blockSize == 4096, width == 4096, x.dtype == .float32, gate.dtype == .float32,
            x.size == gate.size, x.dim(-1) == 128 || x.dim(-1) == 4096,
            weight.dtype == .float32, weight.shape == [128],
            VerifyInt8Matmul.rowRange.contains(rows), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let outputs = Self.wideRowVerify(
            .gatedNorm,
            [x.reshaped([rows, width]), gate.reshaped([rows, width]), weight, MLXArray([eps]), signs],
            rows: rows)
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    func quantizedForPromptGatedNorm(
        _ x: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float
    ) -> PromptInt8Activation? {
        let rows = x.size / width
        guard blockSize == 4096, width == 4096, x.dtype == .float32, gate.dtype == .float32,
            x.size == gate.size, x.dim(-1) == 128 || x.dim(-1) == 4096,
            weight.dtype == .float32, weight.shape == [128], rows > 0, rows % 64 == 0,
            x.size <= Int(Int32.max), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let groups = rows * (width / 128)
        let outputs = Self.gatedNormQuantized4096Kernel(
            [x.reshaped([rows, width]), gate.reshaped([rows, width]), weight, MLXArray([eps]), signs],
            template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0)],
            grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, width], [groups], [groups]],
            outputDTypes: [.int8, .float32, .float32])
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    private static let residualNormNormedQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_residual_norm_normed_4096_q8",
        inputNames: ["x", "residual", "norm_weight", "epsilon", "signs"],
        outputNames: ["codes", "ascale", "rsb", "residual_out", "normed_out"],
        source: quantized4096Source(swiglu: false, residualNorm: true, normalizedOutput: true),
        header: quantized4096Header, ensureRowContiguous: true)

    func residualNormVerifyNormalized(
        _ x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float
    ) -> [MLXArray]? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32,
            residual.dtype == .float32, residual.shape == x.shape,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / width
        guard VerifyInt8Matmul.rowRange.contains(rows) else { return nil }
        return Self.wideRowVerify(
            .residualNormNormed, [x, residual, weight, MLXArray([eps]), signs], rows: rows)
    }

    private static let residualNormQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_residual_norm_4096_q8",
        inputNames: ["x", "residual", "norm_weight", "epsilon", "signs"],
        outputNames: ["codes", "ascale", "rsb", "residual_out"],
        source: quantized4096Source(swiglu: false, residualNorm: true),
        header: quantized4096Header, ensureRowContiguous: true)

    private static let normQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_norm_4096_q8",
        inputNames: ["x", "norm_weight", "epsilon", "signs"],
        outputNames: ["codes", "ascale", "rsb"],
        source: quantized4096Source(swiglu: false, normOnly: true),
        header: quantized4096Header, ensureRowContiguous: true)

    private static let normNormedQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_norm_normed_4096_q8",
        inputNames: ["x", "norm_weight", "epsilon", "signs"],
        outputNames: ["codes", "ascale", "rsb", "normed_out"],
        source: quantized4096Source(swiglu: false, normalizedOutput: true, normOnly: true),
        header: quantized4096Header, ensureRowContiguous: true)

    private static let residualNormRotatedQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_residual_norm_rot_4096_q8",
        inputNames: ["x", "residual", "norm_weight", "epsilon", "signs"],
        outputNames: ["codes", "ascale", "rsb", "residual_out", "rotated_out"],
        source: quantized4096Source(swiglu: false, residualNorm: true, rotatedOutput: true),
        header: quantized4096Header, ensureRowContiguous: true)

    private static let headResidualNormRotatedKernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_head_residual_rot_error2_v1",
        inputNames: ["x", "residual", "norm_weight", "epsilon", "signs"],
        outputNames: ["codes", "ascale", "rsb", "residual_out", "rotated_out", "qerror2"],
        source: quantized4096Source(swiglu: false, residualNorm: true, rotatedOutput: true, headError2: true),
        header: quantized4096Header + ExactVerifyHead.l2Header, ensureRowContiguous: true)
    private static let headRotatedKernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_head_rot_error2_v1", inputNames: ["x", "signs"],
        outputNames: ["codes", "ascale", "rsb", "rotated_out", "qerror2"],
        source: quantized4096Source(swiglu: false, rotatedOutput: true, headError2: true),
        header: quantized4096Header + ExactVerifyHead.l2Header, ensureRowContiguous: true)

    func residualNormVerifyRotated(
        _ x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float, headError2: Bool = false
    ) -> (activation: PromptInt8Activation, residual: MLXArray, rotated: MLXArray, error2: MLXArray?)? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32,
            residual.dtype == .float32, residual.shape == x.shape,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / width
        guard VerifyInt8Matmul.rowRange.contains(rows) else { return nil }
        let outputs = Self.wideRowVerify(
            .residualNormRot, [x, residual, weight, MLXArray([eps]), signs], rows: rows, headError2: headError2 && rows == 16)
        return (
            PromptInt8Activation(
                codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width),
            outputs[3], outputs[4], headError2 && rows == 16 ? outputs[5] : nil)
    }

    func normVerifyNormalized(_ x: MLXArray, weight: MLXArray, eps: Float) -> [MLXArray]? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32, x.dim(-1) == 4096,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / width
        guard VerifyInt8Matmul.rowRange.contains(rows) else { return nil }
        return Self.wideRowVerify(.normNormed, [x, weight, MLXArray([eps]), signs], rows: rows)
    }

    func residualNormPrompt(
        _ x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float, normalized: Bool
    ) -> [MLXArray]? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32, x.dim(-1) == 4096,
            residual.dtype == .float32, residual.shape == x.shape,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            (x.size / 4096) % 64 == 0, x.size <= Int(Int32.max),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / 4096
        Self.wideRowPrepareIfBoot(rows: rows)
        let groups = rows * 32
        let inputs = [x, residual, weight, MLXArray([eps]), signs]
        let template: [(String, any KernelTemplateArg)] = [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0)]
        if normalized {
            return Self.residualNormNormedQuantized4096Kernel(
                inputs, template: template, grid: (256, rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, width], [groups], [groups], x.shape, x.shape],
                outputDTypes: [.int8, .float32, .float32, .float32, .float32])
        }
        return Self.residualNormQuantized4096Kernel(
            inputs, template: template, grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, width], [groups], [groups], x.shape],
            outputDTypes: [.int8, .float32, .float32, .float32])
    }

    func normPrompt(_ x: MLXArray, weight: MLXArray, eps: Float, normalized: Bool) -> [MLXArray]? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32, x.dim(-1) == 4096,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            (x.size / 4096) % 64 == 0, x.size <= Int(Int32.max),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / 4096
        Self.wideRowPrepareIfBoot(rows: rows)
        let groups = rows * 32
        let inputs = [x, weight, MLXArray([eps]), signs]
        let template: [(String, any KernelTemplateArg)] = [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0)]
        if normalized {
            return Self.normNormedQuantized4096Kernel(
                inputs, template: template, grid: (256, rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, width], [groups], [groups], x.shape],
                outputDTypes: [.int8, .float32, .float32, .float32])
        }
        return Self.normQuantized4096Kernel(
            inputs, template: template, grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, width], [groups], [groups]],
            outputDTypes: [.int8, .float32, .float32])
    }

    func residualNorm(
        _ x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float, quantized: Bool,
        verifyRows: Int = 0
    ) -> [MLXArray]? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32,
            residual.dtype == .float32, residual.shape == x.shape,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            x.size <= Int(UInt32.max), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / width
        let inputs = [x, residual, weight, MLXArray([eps]), signs]
        if verifyRows > 0 {
            guard verifyRows == rows, VerifyInt8Matmul.rowRange.contains(rows) else { return nil }
            return Self.wideRowVerify(.residualNorm, inputs, rows: rows)
        }
        Self.wideRowPrepareIfBoot(rows: rows)
        if quantized {
            return Self.residualNormQuantized4096Kernel(
                inputs, template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0)],
                grid: (256, rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, width], [rows * 32], [rows * 32], x.shape],
                outputDTypes: [.int8, .float32, .float32, .float32])
        }
        return Self.residualNorm4096Kernel(
            inputs, template: [("T", DType.float32), ("BLOCKS_PER_ROW", 1), ("INVERSE", false)],
            grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [x.shape, x.shape], outputDTypes: [.float32, .float32])
    }

    func quantizedForPrompt(_ x: MLXArray, up: MLXArray? = nil) -> PromptInt8Activation? {
        validate(x)
        let rows = x.size / width
        guard blockSize == 4096, x.dtype == .float32, rows % 64 == 0, rows > 0,
            x.size <= Int(Int32.max), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        if let up, up.shape != x.shape || up.dtype != x.dtype { return nil }
        Self.wideRowPrepareIfBoot(rows: rows)
        let groups = rows * (width / 128)
        let kernel = up == nil ? Self.quantized4096Kernel : Self.quantizedSwiGLU4096Kernel
        let inputs = up.map { [x, $0, signs] } ?? [x, signs]
        let outputs = kernel(
            inputs,
            template: [("BLOCKS_PER_ROW", width / 4096), ("VERIFY_ROWS", 0)],
            grid: (256, rows * (width / 4096), 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, width], [groups], [groups]],
            outputDTypes: [.int8, .float32, .float32])
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    func quantizedForPromptLastRow(_ x: MLXArray) -> PromptInt8Activation? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) > 0, x.dim(2) == 4096,
            StreamOrDevice.default == StreamOrDevice.gpu else { return nil }
        let row = x[0..., (x.dim(1) - 1)..., 0...]
        let outputs = Self.quantized4096Kernel(
            [row, signs], template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0)],
            grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 4096], [64 * 32], [64 * 32]],
            outputDTypes: [.int8, .float32, .float32])
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: 1, k: 4096)
    }

    func quantizedForVerify(_ x: MLXArray, up: MLXArray? = nil) -> PromptInt8Activation? {
        validate(x)
        let rows = x.size / width
        guard blockSize == 4096, x.dtype == .float32, (2 ... 16).contains(rows),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        if let up, up.shape != x.shape || up.dtype != x.dtype { return nil }
        let kind: WideRowKind = up == nil ? .plain : .swiglu
        let blocks = width / 4096
        guard (blocks == kind.checkBlocks && Self.wideRowUsesWide(kind))
            || VerifyInt8Matmul.rowRange.contains(rows)
        else { return nil }
        let outputs = Self.wideRowVerify(
            kind,
            [x.reshaped([rows, width])] + (up.map { [$0.reshaped([rows, width])] } ?? []) + [signs],
            rows: rows, blocks: blocks)
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    func quantizedForVerifySwiGLUValue(_ x: MLXArray) -> PromptInt8Activation? {
        validate(x)
        guard blockSize == 4096, width == 12288, x.dtype == .float32,
            x.size / width == 16, StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let lanes = Self.wideRowLanes(.swiglu)
        guard lanes == 0 || Self.wideRowWideKernels[lanes]?[.plain] != nil else { return nil }
        let outputs = Self.wideRowDispatch(
            .plain, [x.reshaped([16, width]), signs], rows: 16, blocks: 3, lanes: lanes)
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: 16, k: width)
    }

    static let swiGLUPartial4Header = """
        inline float4 bonsai_swiglu_signed_radix4(float4 value, float4 signs) {
          const float4 x = value * signs;
          const float a = x.x + x.y, b = x.x - x.y;
          const float c = x.z + x.w, d = x.z - x.w;
          return float4(a + c, b + d, a - c, b - d);
        }
        """

    var swiGLUPartialSigns: MLXArray? {
        blockSize == 4096 && width == 12288 ? signs : nil
    }

    static func swiGLUPartialFinish(lanes: Int) -> (header: String, source: String)? {
        guard [0,512,1024].contains(lanes) else { return nil }
        let elements: Int = lanes == 0 ? 16 : 4096 / lanes
        let originalHeader: String = lanes == 0 ? quantized4096Header : WideRowSource.header(threads: lanes)
        var source: String = lanes == 0
            ? quantized4096Source(swiglu: false)
            : WideRowSource.source(threads: lanes, form: WideRowSource.Form())
        let read: String = "float4 val = in4[w] * s4[w];"
        let call: String = lanes == 0 ? "hadamard_radix16_q8(x_reg);" : "widerow_radix\(elements)(x_reg);"
        guard source.components(separatedBy: read).count == 2, let first = source.range(of: call) else { return nil }
        source = source.replacingCharacters(in: first, with: "swiglu_finish_radix\(elements)(x_reg);")
            .replacingOccurrences(of: read, with: "float4 val = in4[w];")
            .replacingOccurrences(of: "const device float4* s4 = (const device float4*)(signs + s_offset);", with: "")
            .replacingOccurrences(of: "const uint s_offset = sign_offset + base0;", with: "")
            .replacingOccurrences(of: "const uint s_offset = sign_offset + uint(base0);", with: "")
            .replacingOccurrences(of: "const uint sign_offset = (BLOCKS_PER_ROW == 1)\n    ? 0u : (block_idx % uint(BLOCKS_PER_ROW)) * 4096u;", with: "")
        var stages: String = ""
        var stride: Int = 4
        while stride < elements {
            stages += """
                #pragma clang loop unroll(full)
                for (short j = 0; j < \(elements); j += \(2 * stride)) {
                  #pragma clang loop unroll(full)
                  for (short k = 0; k < \(stride); k++) {
                    float a = x[j+k], b = x[j+k+\(stride)];
                    x[j+k] = a + b; x[j+k+\(stride)] = a - b;
                  }
                }

                """
            stride *= 2
        }
        let header: String = originalHeader + "\nMETAL_FUNC void swiglu_finish_radix\(elements)(thread float* x) {\n" + stages + "}\n"
        return (header,source)
    }

    private static let swiGLUPartialFinishKernels: [Int: MLXFast.MLXFastKernel] = {
        var kernels: [Int: MLXFast.MLXFastKernel] = [:]
        for lanes in [0,512,1024] {
            guard let generated = swiGLUPartialFinish(lanes: lanes) else { continue }
            kernels[lanes] = MLXFast.metalKernel(
                name: "signed_block_hadamard_partial4_finish_\(lanes)",
                inputNames: lanes == 0 ? ["x"] : ["x","verify_rows"],
                outputNames: ["codes","ascale","rsb"], source: generated.source,
                header: generated.header, ensureRowContiguous: true)
        }
        return kernels
    }()

    func quantizedForVerifySwiGLUPartial4(_ x: MLXArray) -> PromptInt8Activation? {
        guard blockSize == 4096, width == 12288, x.dtype == .float32, x.size == 16 * width,
            StreamOrDevice.default == StreamOrDevice.gpu else { return nil }
        let lanes: Int = Self.wideRowLanes(.swiglu)
        guard let kernel = Self.swiGLUPartialFinishKernels[lanes] else { return nil }
        var inputs: [MLXArray] = [x.reshaped([16,width])]
        if lanes > 0 { inputs.append(MLXArray([Int32(16)])) }
        let arguments: [(String, any KernelTemplateArg)] = lanes == 0
            ? [("BLOCKS_PER_ROW",3),("VERIFY_ROWS",16)] : [("BLOCKS_PER_ROW",3)]
        let threads: Int = lanes == 0 ? 256 : lanes
        let output: [MLXArray] = kernel(inputs, template: arguments,
            grid: (threads,48,1), threadGroup: (threads,1,1),
            outputShapes: [[16,width],[1536],[1536]], outputDTypes: [.int8,.float32,.float32])
        return PromptInt8Activation(codes: output[0], ascale: output[1], rsb: output[2], rows: 16, k: width)
    }

    private static let swiGLUPartialReferenceKernel = MLXFast.metalKernel(
        name: "bonsai_swiglu_signed_radix4_reference", inputNames: ["x","signs"], outputNames: ["out"],
        source: """
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            const uint i = thread_position_in_grid.x;
            const float4 value = ((const device float4*)x)[i];
            const float4 sign = ((const device float4*)signs)[i % 3072u];
            ((device float4*)out)[i] = bonsai_swiglu_signed_radix4(value, sign);
            """, header: swiGLUPartial4Header, ensureRowContiguous: true)

    func swiGLUPartialReference(_ x: MLXArray) -> MLXArray? {
        guard let signs = swiGLUPartialSigns, x.dtype == .float32, x.size == 16 * width else { return nil }
        return Self.swiGLUPartialReferenceKernel([x,signs], grid: (16 * width / 4,1,1), threadGroup: (128,1,1),
            outputShapes: [[16,width]], outputDTypes: [.float32])[0]
    }

    func verifySwiGLUValueMatches(gate: MLXArray, up: MLXArray, value: MLXArray) -> Bool {
        guard let actual = quantizedForVerifySwiGLUValue(value) else { return false }
        let reference = Self.wideRowDispatch(
            .swiglu, [gate, up, signs], rows: 16, blocks: 3, lanes: Self.wideRowLanes(.swiglu))
        let checks = [MLX.all(reference[0] .== actual.codes),
            MLX.all(reference[1].view(dtype: .uint32) .== actual.ascale.view(dtype: .uint32)),
            MLX.all(reference[2].view(dtype: .uint32) .== actual.rsb.view(dtype: .uint32))]
        eval(checks)
        return checks.allSatisfy { $0.item(Bool.self) }
    }

    func quantizedForVerifyWithRotated(_ x: MLXArray, headError2: Bool = false) -> (activation: PromptInt8Activation, rotated: MLXArray, error2: MLXArray?)? {
        validate(x)
        let rows = x.size / width
        guard blockSize == 4096, x.dtype == .float32, (2 ... 16).contains(rows),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let blocks = width / 4096
        guard (blocks == WideRowKind.plainRot.checkBlocks && Self.wideRowUsesWide(.plainRot))
            || VerifyInt8Matmul.rowRange.contains(rows)
        else { return nil }
        let outputs = Self.wideRowVerify(
            .plainRot, [x.reshaped([rows, width]), signs], rows: rows, blocks: blocks,
            sideShape: x.shape, headError2: headError2 && rows == 16)
        return (
            PromptInt8Activation(
                codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width),
            outputs[3], headError2 && rows == 16 ? outputs[4] : nil)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        validate(x)
        if let fused = fused4096(x, inverse: false) { return fused }
        return hadamardTransform((x.asType(.float32) * signs).reshaped([-1, blockSize]))
            .reshaped(x.shape).asType(x.dtype)
    }

    public func inverse(_ x: MLXArray) -> MLXArray {
        validate(x)
        if let fused = fused4096(x, inverse: true) { return fused }
        return (hadamardTransform(x.asType(.float32).reshaped([-1, blockSize])).reshaped(x.shape)
            * signs).asType(x.dtype)
    }

    private func validate(_ x: MLXArray) {
        precondition(x.ndim > 0 && x.dim(-1) == width, "Hadamard input width mismatch")
        precondition(
            [DType.float32, .float16, .bfloat16].contains(x.dtype),
            "Hadamard input must have a real floating-point dtype")
    }
}

// WIDEROW-GEN BEGIN
enum WideRowSource {
    struct Form: Hashable {
        var swiglu = false, residualNorm = false, normOnly = false, gatedNorm = false
        var attentionGate = false, normalizedOutput = false, rotatedOutput = false, headError2 = false
        var hoist = false
    }

    static func header(threads: Int) -> String {
        let e = 4096 / threads
        var stages = ""
        var h = 1
        while h < e {
            stages += """
                  #pragma clang loop unroll(full)
                  for (short j = 0; j < \(e); j += \(2 * h)) {
                    #pragma clang loop unroll(full)
                    for (short k = 0; k < \(h); k++) {
                      float a = x[j+k], b = x[j+k+\(h)];
                      x[j+k] = a + b; x[j+k+\(h)] = a - b;
                    }
                  }

                """
            h *= 2
        }
        return """
            using namespace metal;
            METAL_FUNC void widerow_radix\(e)(thread float* x) {
            \(stages)}

            """
    }

    static func source(threads: Int, form f: Form) -> String {
        precondition(threads == 512 || threads == 1024, "wide row threads must be 512 or 1024")
        let e = 4096 / threads, v = e / 4, b = e == 8 ? 3 : 2, passes = 12 / b
        let cluster = 128 / e
        let pad = { (idx: String) in "(\(idx)) + ((\(idx)) >> \(b)u)" }
        let residualAdd = f.normOnly ? ";" : "\n      + *((const device float4*)(residual + at));"
        let normPrefix = f.residualNorm || f.normOnly ? """
            threadgroup float norm_sums[32];
            {
              const uint norm_lane = thread_index_in_simdgroup;
              for (uint part = 0; part < \(1024 / threads)u; ++part) {
                const uint f = i + part * \(threads)u;
                const uint at = batch_idx + f * 4u;
                const float4 value = *((const device float4*)(x + at))\(residualAdd)
                float square_sum = 0.0f;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) square_sum += value[j] * value[j];
                square_sum = simd_sum(square_sum);
                if (norm_lane == 0) norm_sums[f >> 5] = square_sum;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float norm_total = simd_sum(norm_sums[thread_index_in_simdgroup]);
            const float norm_inv = metal::precise::rsqrt(norm_total / 4096.0f + epsilon[0]);
            """ : ""
        let gatedPrefix = f.gatedNorm ? """
            threadgroup float head_part[32 * 32];
            {
              const device float4* nx4 = (const device float4*)(x + batch_idx + base0);
              #pragma clang loop unroll(full)
              for (short w = 0; w < \(v); w++) {
                const float4 nv = nx4[w];
                float acc = 0.0f;
                acc += nv.x * nv.x;
                acc += nv.y * nv.y;
                acc += nv.z * nv.z;
                acc += nv.w * nv.w;
                head_part[(base0 >> 2) + uint(w)] = acc;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float head_inv_value = 0.0f;
            {
              const uint norm_lane = thread_index_in_simdgroup;
              const uint head_base = simdgroup_index_in_threadgroup * \(v)u;
              for (uint k = 0; k < \(v)u; ++k) {
                const float total = simd_sum(head_part[(head_base + k) * 32u + norm_lane]);
                const float inv = metal::precise::rsqrt(total / 128.0f + epsilon[0]);
                if ((norm_lane / \(32 / v)u) == k) head_inv_value = inv;
              }
            }
            """ : ""
        let read: String
        if f.residualNorm && f.normalizedOutput {
            read = """
                const float4 raw = in4[w] + *((const device float4*)(residual + in_offset) + w);
                *((device float4*)(residual_out + in_offset) + w) = raw;
                const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
                const float4 normed = weight * (raw * norm_inv);
                *((device float4*)(normed_out + in_offset) + w) = normed;
                float4 val = normed * s4[w];
                """
        } else if f.residualNorm {
            read = """
                const float4 raw = in4[w] + *((const device float4*)(residual + in_offset) + w);
                *((device float4*)(residual_out + in_offset) + w) = raw;
                const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
                float4 val = (weight * (raw * norm_inv)) * s4[w];
                """
        } else if f.normOnly && f.normalizedOutput {
            read = """
                const float4 raw = in4[w];
                const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
                const float4 normed = weight * (raw * norm_inv);
                *((device float4*)(normed_out + in_offset) + w) = normed;
                float4 val = normed * s4[w];
                """
        } else if f.normOnly {
            read = """
                const float4 raw = in4[w];
                const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
                float4 val = (weight * (raw * norm_inv)) * s4[w];
                """
        } else if f.attentionGate {
            read = """
                const uint ag_e = uint(base0) + 4u * uint(w);
                const uint ag_row = block_idx;
                const float4 attn = *((const device float4*)(x
                    + ((ag_e >> 8) * uint(VERIFY_ROWS) + ag_row) * 256u + (ag_e & 255u)));
                const float4 g = *((const device float4*)(gate
                    + ag_row * 8192u + (ag_e >> 8) * 512u + 256u + (ag_e & 255u)));
                const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(g)));
                const float4 sigmoid_g = select(1.0f - e, e, g < 0.0f);
                float4 val = (attn * sigmoid_g) * s4[w];
                """
        } else if f.gatedNorm {
            read = """
                const float4 g = *((const device float4*)(gate + in_offset) + w);
                const float4 nw = *((const device float4*)(norm_weight + (base0 & 127u)) + w);
                const float4 normed = nw * (in4[w] * head_inv_value);
                const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(g)));
                const float4 sigmoid_g = select(1.0f - e, e, g < 0.0f);
                float4 val = ((g * sigmoid_g) * normed) * s4[w];
                """
        } else if f.swiglu {
            read = """
                const float4 g = in4[w];
                const float4 u = *((const device float4*)(up + in_offset) + w);
                const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(g)));
                const float4 sigmoid_g = select(1.0f - e, e, g < 0.0f);
                float4 val = ((g * sigmoid_g) * u) * s4[w];
                """
        } else {
            read = "float4 val = in4[w] * s4[w];"
        }
        let hoist = f.hoist && (f.residualNorm || f.normOnly || f.gatedNorm)
        let mainDecl = """
            const uint in_offset = batch_idx + base0;
            const uint s_offset = sign_offset + base0;
            const device float4* in4 = (const device float4*)(x + in_offset);
            const device float4* s4 = (const device float4*)(signs + s_offset);
            """
        var hoistDecl = ""
        var mainRead = read
        var gatedPrefixRead = gatedPrefix
        if hoist {
            let residualLoad = f.residualNorm
                ? "hr[w] = *((const device float4*)(residual + in_offset) + w);\n" : ""
            let gateLoad = f.gatedNorm
                ? "hg[w] = *((const device float4*)(gate + in_offset) + w);\n" : ""
            let weightAt = f.gatedNorm ? "(base0 & 127u)" : "s_offset"
            hoistDecl = mainDecl + "\n" + """
                float4 hx[\(v)], hr[\(v)], hg[\(v)], hw[\(v)], hs[\(v)];
                #pragma clang loop unroll(full)
                for (short w = 0; w < \(v); w++) {
                hx[w] = in4[w];
                \(residualLoad)\(gateLoad)hw[w] = *((const device float4*)(norm_weight + \(weightAt)) + w);
                hs[w] = s4[w];
                }

                """
            mainRead = read
                .replacingOccurrences(of: "*((const device float4*)(residual + in_offset) + w)", with: "hr[w]")
                .replacingOccurrences(of: "*((const device float4*)(gate + in_offset) + w)", with: "hg[w]")
                .replacingOccurrences(of: "*((const device float4*)(norm_weight + s_offset) + w)", with: "hw[w]")
                .replacingOccurrences(
                    of: "*((const device float4*)(norm_weight + (base0 & 127u)) + w)", with: "hw[w]")
                .replacingOccurrences(of: "in4[w]", with: "hx[w]")
                .replacingOccurrences(of: "s4[w]", with: "hs[w]")
            precondition(
                !mainRead.contains("const device float4*") && !mainRead.contains("in4[")
                    && !mainRead.contains("s4["), "hoisted read still loads from device")
            if f.gatedNorm {
                gatedPrefixRead = gatedPrefix.replacingOccurrences(
                    of: "const float4 nv = nx4[w];", with: "const float4 nv = hx[w];")
                precondition(gatedPrefixRead != gatedPrefix, "gated prefix not hoisted")
            }
        }
        var passSource = ""
        for k in 1 ..< passes {
            let s = k * b
            let last = k == passes - 1
            passSource += """
                threadgroup_barrier(mem_flags::mem_threadgroup);
                {
                  const uint e0 = ((i >> \(s)u) << \(s + b)u) | (i & \((1 << s) - 1)u);
                  #pragma clang loop unroll(full)
                  for (uint r = 0; r < \(e)u; r++) {
                    x_reg[r] = buf[\(pad("e0 + (r << \(s)u)"))];
                  }
                  widerow_radix\(e)(x_reg);
                  #pragma clang loop unroll(full)
                  for (uint r = 0; r < \(e)u; r++) {
                    buf[\(pad("e0 + (r << \(s)u)"))] = x_reg[r]\(last ? " * 0.015625f" : "");
                  }
                }

                """
        }
        var amaxShuffle = "", partShuffle = ""
        var m = 1
        while m < cluster {
            amaxShuffle += "amax = max(amax, simd_shuffle_xor(amax, ushort(\(m))));\n"
            partShuffle += "part += simd_shuffle_xor(part, ushort(\(m)));\n"
            m *= 2
        }
        let rotatedWrite = f.rotatedOutput ? """
            device float4* rot4 = (device float4*)(rotated_out + batch_idx + base0);
            #pragma clang loop unroll(full)
            for (short w = 0; w < \(v); w++) {
              rot4[w] = float4(v[4 * w + 0], v[4 * w + 1], v[4 * w + 2], v[4 * w + 3]);
            }
            """ : ""
        return """
            #pragma clang fp reassociate(off)
            const uint i = thread_position_in_threadgroup.x;
            const uint block_idx = threadgroup_position_in_grid.y;
            const uint batch_idx = block_idx * 4096u;
            const uint sign_offset = (BLOCKS_PER_ROW == 1)
                ? 0u : (block_idx % uint(BLOCKS_PER_ROW)) * 4096u;
            const int VERIFY_ROWS = verify_rows[0];
            const uint base0 = i * \(e)u;
            if (block_idx >= uint(VERIFY_ROWS * BLOCKS_PER_ROW)) {
              device char* pad = (device char*)(codes + batch_idx + base0);
              #pragma clang loop unroll(full)
              for (uint r = 0; r < \(e)u; r++) pad[r] = char(0);
              return;
            }

            \(hoistDecl)\(normPrefix)\(gatedPrefixRead)
            threadgroup float buf[4096 + \(4096 / e)];
            float x_reg[\(e)];

            \(hoist ? "" : mainDecl)
            #pragma clang loop unroll(full)
            for (short w = 0; w < \(v); w++) {
              \(mainRead)
              x_reg[4 * w + 0] = val.x;
              x_reg[4 * w + 1] = val.y;
              x_reg[4 * w + 2] = val.z;
              x_reg[4 * w + 3] = val.w;
            }
            widerow_radix\(e)(x_reg);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < \(e)u; r++) {
              buf[\(pad("base0 + r"))] = x_reg[r];
            }
            \(passSource)threadgroup_barrier(mem_flags::mem_threadgroup);

            float v[\(e)];
            float amax = 0.0f;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < \(e)u; r++) {
              v[r] = buf[\(pad("base0 + r"))];
              amax = max(amax, fabs(v[r]));
            }
            \(rotatedWrite)
            \(amaxShuffle)const float qs = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
            const float iqs = amax > 0.0f ? metal::precise::divide(127.0f, amax) : 0.0f;
            float q[\(e)];
            float part = 0.0f;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < \(e)u; r++) {
              q[r] = rint(v[r] * iqs);
              part += q[r];
            }
            \(partShuffle)\(f.headError2 ? ExactVerifyHead.error2Source(elements: e, cluster: cluster) : "")device char* dst = (device char*)(codes + batch_idx + (base0 & ~15u));
            #pragma clang loop unroll(full)
            for (uint r = 0; r < \(e)u; r++) {
              const uint c = (base0 & 15u) + r;
              dst[4u * (c & 3u) + (c >> 2)] = char(q[r]);
            }
            if ((i & \(cluster - 1)u) == 0) {
              const uint row = block_idx / uint(BLOCKS_PER_ROW);
              const uint kg = uint(BLOCKS_PER_ROW) * 32u;
              const uint g = (block_idx % uint(BLOCKS_PER_ROW)) * 32u + (base0 >> 7);
              const size_t qidx = size_t(row) * size_t(kg) + size_t(g);
              ascale[qidx] = qs;
              rsb[qidx] = qs * part;
              \(f.headError2 ? "qerror2[qidx] = head_error2;" : "")
            }
            """
    }
}
// WIDEROW-GEN END

extension SignedBlockHadamard {
    enum WideRowKind: String, CaseIterable {
        case residualNorm, residualNormNormed, residualNormRot, normNormed, gatedNorm
        case plain, plainRot, swiglu, attentionGate

        var form: WideRowSource.Form {
            var f = WideRowSource.Form()
            switch self {
            case .residualNorm: f.residualNorm = true
            case .residualNormNormed: f.residualNorm = true; f.normalizedOutput = true
            case .residualNormRot: f.residualNorm = true; f.rotatedOutput = true
            case .normNormed: f.normOnly = true; f.normalizedOutput = true
            case .gatedNorm: f.gatedNorm = true
            case .plain: break
            case .plainRot: f.rotatedOutput = true
            case .swiglu: f.swiglu = true
            case .attentionGate: f.attentionGate = true
            }
            return f
        }

        var elementwise: Bool {
            switch self {
            case .plain, .plainRot, .swiglu, .attentionGate: return true
            default: return false
            }
        }

        var checkBlocks: Int { self == .swiglu ? 3 : 1 }

        var index: Int {
            switch self {
            case .residualNorm: return 0
            case .residualNormNormed: return 1
            case .residualNormRot: return 2
            case .normNormed: return 3
            case .gatedNorm: return 4
            case .plain: return 5
            case .plainRot: return 6
            case .swiglu: return 7
            case .attentionGate: return 8
            }
        }

        var sideOutputCount: Int {
            switch self {
            case .residualNorm, .normNormed, .plainRot: return 1
            case .residualNormNormed, .residualNormRot: return 2
            case .gatedNorm, .plain, .swiglu, .attentionGate: return 0
            }
        }
    }

    static let wideRowEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_WIDE_ROWS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    static let wideRowElementwiseEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_WIDE_ROWS_ELEMENTWISE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static func wideRowKindEnabled(_ kind: WideRowKind) -> Bool {
        wideRowEnabled && (!kind.elementwise || wideRowElementwiseEnabled)
    }
    static let wideRowThreadsSetting: String? = ProcessInfo.processInfo
        .environment["MLXFAST_WIDE_ROWS_THREADS"]?.trimmingCharacters(in: .whitespacesAndNewlines)
    static let wideRow1024Enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_WIDE_ROWS_1024"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static func wideRowLaneCandidates(_ kind: WideRowKind) -> [Int] {
        if wideRowThreadsSetting == "512" || !wideRow1024Enabled { return [512] }
        if wideRowThreadsSetting == "1024" || !kind.elementwise { return [1024, 512] }
        return [512]
    }

    private static func wideRowRecordKernel(_ kind: WideRowKind, headError2: Bool = false) -> MLXFast.MLXFastKernel {
        if headError2 { return kind == .plainRot ? headRotatedKernel : headResidualNormRotatedKernel }
        switch kind {
        case .residualNorm: return residualNormQuantized4096Kernel
        case .residualNormNormed: return residualNormNormedQuantized4096Kernel
        case .residualNormRot: return residualNormRotatedQuantized4096Kernel
        case .normNormed: return normNormedQuantized4096Kernel
        case .gatedNorm: return gatedNormQuantized4096Kernel
        case .plain: return quantized4096Kernel
        case .plainRot: return quantized4096RotatedKernel
        case .swiglu: return quantizedSwiGLU4096Kernel
        case .attentionGate: return attentionGateQuantized4096Kernel
        }
    }

    private static func wideRowInputNames(_ kind: WideRowKind) -> [String] {
        switch kind {
        case .residualNorm, .residualNormNormed, .residualNormRot:
            return ["x", "residual", "norm_weight", "epsilon", "signs"]
        case .normNormed: return ["x", "norm_weight", "epsilon", "signs"]
        case .gatedNorm: return ["x", "gate", "norm_weight", "epsilon", "signs"]
        case .plain, .plainRot: return ["x", "signs"]
        case .swiglu: return ["x", "up", "signs"]
        case .attentionGate: return ["x", "gate", "signs"]
        }
    }

    private static func wideRowOutputNames(_ kind: WideRowKind) -> [String] {
        switch kind {
        case .residualNorm: return ["codes", "ascale", "rsb", "residual_out"]
        case .residualNormNormed: return ["codes", "ascale", "rsb", "residual_out", "normed_out"]
        case .residualNormRot: return ["codes", "ascale", "rsb", "residual_out", "rotated_out"]
        case .normNormed: return ["codes", "ascale", "rsb", "normed_out"]
        case .gatedNorm, .plain, .swiglu, .attentionGate: return ["codes", "ascale", "rsb"]
        case .plainRot: return ["codes", "ascale", "rsb", "rotated_out"]
        }
    }

    static let wideRowLoadHoist: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_LOAD_HOIST"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static func wideRowMakeKernels(hoist: Bool, threads: Int, headError2: Bool = false)
        -> [WideRowKind: MLXFast.MLXFastKernel]
    {
        var kernels: [WideRowKind: MLXFast.MLXFastKernel] = [:]
        for kind in WideRowKind.allCases
        where wideRowKindEnabled(kind) && wideRowLaneCandidates(kind).contains(threads) {
            if headError2 && kind != .plainRot && kind != .residualNormRot { continue }
            var form = kind.form
            form.hoist = hoist
            form.headError2 = headError2
            let hoisted = hoist && (form.residualNorm || form.normOnly || form.gatedNorm)
            kernels[kind] = MLXFast.metalKernel(
                name: "signed_block_hadamard_wide_\(kind.rawValue)_w\(threads)\(hoisted ? "_h" : "")\(headError2 ? "_head_error2_v1" : "")",
                inputNames: wideRowInputNames(kind) + ["verify_rows"],
                outputNames: wideRowOutputNames(kind) + (headError2 ? ["qerror2"] : []),
                source: WideRowSource.source(threads: threads, form: form),
                header: WideRowSource.header(threads: threads) + (headError2 ? ExactVerifyHead.l2Header : ""), ensureRowContiguous: true)
        }
        return kernels
    }

    private static let wideRowWideKernels: [Int: [WideRowKind: MLXFast.MLXFastKernel]] = [
        512: wideRowMakeKernels(hoist: wideRowLoadHoist, threads: 512),
        1024: wideRowMakeKernels(hoist: wideRowLoadHoist, threads: 1024),
    ]
    private static let wideRowHeadKernels: [Int: [WideRowKind: MLXFast.MLXFastKernel]] = [
        512: wideRowMakeKernels(hoist: wideRowLoadHoist, threads: 512, headError2: true),
        1024: wideRowMakeKernels(hoist: wideRowLoadHoist, threads: 1024, headError2: true),
    ]

    nonisolated(unsafe) private static var wideRowVerdict: [WideRowKind: Int] = [:]
    nonisolated(unsafe) private static var wideRowResolvedLanes: [Int] = [Int](repeating: 0, count: 9)
    nonisolated(unsafe) private static var wideRowResolvedKernels: [MLXFast.MLXFastKernel?] = [MLXFast.MLXFastKernel?](repeating: nil, count: 9)
    nonisolated(unsafe) private static var wideRowResolvedHeadKernels: [MLXFast.MLXFastKernel?] = [MLXFast.MLXFastKernel?](repeating: nil, count: 9)
    nonisolated(unsafe) private static var wideRowChecked = false
    nonisolated(unsafe) private static var wideRowReady = false
    private static let wideRowLock = NSLock()

    static func wideRowPrepareIfBoot(rows: Int) {
        guard rows >= 1024, wideRowEnabled, !wideRowReady else { return }
        wideRowLock.lock()
        defer { wideRowLock.unlock() }
        guard !wideRowChecked else { return }
        wideRowChecked = true
        for kind in WideRowKind.allCases where wideRowKindEnabled(kind) {
            let candidates = wideRowLaneCandidates(kind)
            for threads in candidates {
                // An MLX error in the check (a pipeline that cannot run this
                // many threads per threadgroup on this GPU, a compile error)
                // falls to the next lane count, then to the record kernel,
                // instead of stopping the process.
                var note = "mismatch"
                do {
                    if try withError({ wideRowCheck(kind, threads: threads) }) {
                        wideRowVerdict[kind] = threads
                        wideRowResolvedLanes[kind.index] = threads
                        wideRowResolvedKernels[kind.index] = wideRowWideKernels[threads]?[kind]
                        wideRowResolvedHeadKernels[kind.index] = wideRowHeadKernels[threads]?[kind]
                        note = "passed"
                    }
                } catch {
                    note = "error: \(error)"
                }
                let next = threads == candidates.last ? "record kernel kept" : "next lane count"
                FileHandle.standardError.write(Data(
                    "hadamard wide rows: \(kind.rawValue) w\(threads) load check \(note == "passed" ? note : note + ", " + next) (rows 3, 11, 16)\n".utf8))
                if wideRowVerdict[kind] != nil { break }
            }
        }
        wideRowReady = true
    }

    private static func wideRowUsesWide(_ kind: WideRowKind) -> Bool {
        wideRowLanes(kind) > 0
    }

    private static func wideRowLanes(_ kind: WideRowKind) -> Int {
        guard wideRowKindEnabled(kind) else { return 0 }
        if wideRowReady { return wideRowResolvedLanes[kind.index] }
        wideRowLock.lock()
        defer { wideRowLock.unlock() }
        return wideRowVerdict[kind] ?? 0
    }

    private static func wideRowCheck(_ kind: WideRowKind, threads: Int) -> Bool {
        var same: [MLXArray] = []
        for rows in [3, 11, 16] {
            let parts = MLXRandom.split(key: MLXRandom.key(UInt64(0x4163_7400 + rows)), into: 5)
            let scale = MLXArray((0 ..< rows).map { Float($0 == 0 ? 0 : ($0 == 1 ? 30 : 1)) })
                .reshaped([rows, 1])
            let rowWidth = 4096 * kind.checkBlocks
            let x = MLXRandom.uniform(low: -3, high: 3, [rows, rowWidth], key: parts[0]) * scale
            let other = MLXRandom.uniform(low: -4, high: 4, [rows, rowWidth], key: parts[1])
            let signs = MLX.floor(MLXRandom.uniform(low: 0, high: 2, [rowWidth], key: parts[2])) * 2 - 1
            let width = kind == .gatedNorm ? 128 : 4096
            let weight = MLXRandom.uniform(low: 0.5, high: 1.5, [width], key: parts[3])
            let eps = MLXArray([Float(1e-6)])
            let inputs: [MLXArray]
            switch kind {
            case .residualNorm, .residualNormNormed, .residualNormRot:
                inputs = [x, other, weight, eps, signs]
            case .normNormed: inputs = [x, weight, eps, signs]
            case .gatedNorm: inputs = [x, other, weight, eps, signs]
            case .plain, .plainRot: inputs = [x, signs]
            case .swiglu: inputs = [x, other, signs]
            case .attentionGate:
                let attention = x.reshaped([rows, 16, 256]).transposed(1, 0, 2)
                    .reshaped([1, 16, rows, 256])
                let gate = MLXRandom.uniform(low: -4, high: 4, [rows, 8192], key: parts[4])
                    * scale
                inputs = [attention, gate, signs]
            }
            let record = wideRowDispatch(
                kind, inputs, rows: rows, blocks: kind.checkBlocks, lanes: 0)
            let wide = wideRowDispatch(
                kind, inputs, rows: rows, blocks: kind.checkBlocks, lanes: threads)
            same.append(MLX.all(record[0] .== wide[0]))
            for (r, w) in zip(record.dropFirst(), wide.dropFirst()) {
                same.append(MLX.all(r.view(dtype: .uint32) .== w.view(dtype: .uint32)))
            }
        }
        return MLX.all(MLX.stacked(same)).item(Bool.self)
    }

    private static func wideRowDispatch(
        _ kind: WideRowKind, _ inputs: [MLXArray], rows: Int, blocks: Int = 1,
        sideShape: [Int]? = nil, lanes: Int, headError2: Bool = false
    ) -> [MLXArray] {
        let tile = VerifyInt8Matmul.tileRows
        var shapes = [[tile, 4096 * blocks], [rows * 32 * blocks], [rows * 32 * blocks]]
            + Array(
                repeating: sideShape ?? inputs[0].shape, count: kind.sideOutputCount)
        if headError2 { shapes.append([rows * 32 * blocks]) }
        let dtypes = [DType.int8] + Array(repeating: DType.float32, count: shapes.count - 1)
        let resolvedKernel: MLXFast.MLXFastKernel? = wideRowReady
            ? (headError2 ? wideRowResolvedHeadKernels[kind.index] : wideRowResolvedKernels[kind.index])
            : (headError2 ? wideRowHeadKernels : wideRowWideKernels)[lanes]?[kind]
        if lanes > 0, let kernel = resolvedKernel {
            return kernel(
                inputs + [MLXArray([Int32(rows)])], template: [("BLOCKS_PER_ROW", blocks)],
                grid: (lanes, tile * blocks, 1), threadGroup: (lanes, 1, 1),
                outputShapes: shapes, outputDTypes: dtypes)
        }
        return wideRowRecordKernel(kind, headError2: headError2)(
            inputs, template: [("BLOCKS_PER_ROW", blocks), ("VERIFY_ROWS", rows)],
            grid: (256, tile * blocks, 1), threadGroup: (256, 1, 1),
            outputShapes: shapes, outputDTypes: dtypes)
    }

    static func wideRowVerify(
        _ kind: WideRowKind, _ inputs: [MLXArray], rows: Int, blocks: Int = 1,
        sideShape: [Int]? = nil, headError2: Bool = false
    ) -> [MLXArray] {
        wideRowDispatch(
            kind, inputs, rows: rows, blocks: blocks, sideShape: sideShape,
            lanes: blocks == kind.checkBlocks ? wideRowLanes(kind) : 0, headError2: headError2)
    }
}

public struct PrismHadamardConfiguration: Decodable {
    public let blockSize: Int
    public let weightNames: [String]
    public let inverseWeightNames: [String]
    public let gdnVGrouped: Bool
    private let transforms: [Int: SignedBlockHadamard]

    private enum CodingKeys: String, CodingKey {
        case version = "prism.hadamard.version"
        case blockSize = "prism.hadamard.block_size"
        case transform = "prism.hadamard.transform"
        case axis = "prism.hadamard.axis"
        case signMode = "prism.hadamard.sign_mode"
        case weightNames = "prism.hadamard.weight_names"
        case inverseWeightNames = "prism.hadamard.inverse_weight_names"
        case signWidths = "prism.hadamard.sign_widths"
        case signValues = "prism.hadamard.sign_values"
        case gdnVGrouped = "prism.hadamard.gdn_v_grouped"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard try c.decode(Int.self, forKey: .version) == 1,
            try c.decode(String.self, forKey: .transform) == "normalized-sylvester-walsh-hadamard",
            try c.decode(String.self, forKey: .axis) == "input-last-dimension",
            try c.decode(String.self, forKey: .signMode) == "explicit"
        else { throw HadamardError.unsupportedContract }
        blockSize = try c.decode(Int.self, forKey: .blockSize)
        weightNames = try c.decode([String].self, forKey: .weightNames)
        inverseWeightNames = try c.decodeIfPresent([String].self, forKey: .inverseWeightNames) ?? []
        gdnVGrouped = try c.decodeIfPresent(Bool.self, forKey: .gdnVGrouped) ?? false
        guard Set(weightNames).count == weightNames.count,
            Set(inverseWeightNames).count == inverseWeightNames.count,
            Set(weightNames).isDisjoint(with: inverseWeightNames),
            (weightNames + inverseWeightNames).allSatisfy({ !$0.isEmpty })
        else { throw HadamardError.unsupportedContract }
        let widths = try c.decode([Int].self, forKey: .signWidths)
        let values = try c.decode([Float].self, forKey: .signValues)
        guard !widths.isEmpty, Set(widths).count == widths.count else {
            throw HadamardError.invalidSigns
        }
        var offset = 0
        var transforms = [Int: SignedBlockHadamard]()
        for width in widths {
            guard width > 0, width <= values.count - offset else {
                throw HadamardError.invalidSigns
            }
            transforms[width] = try SignedBlockHadamard(
                blockSize: blockSize, signs: Array(values[offset ..< offset + width]))
            offset += width
        }
        guard offset == values.count else { throw HadamardError.invalidSigns }
        self.transforms = transforms
    }

    public func transform(forWidth width: Int) throws -> SignedBlockHadamard {
        guard let transform = transforms[width] else {
            throw HadamardError.missingSignWidth(width)
        }
        return transform
    }
}

private func validateHadamardWeights(
    _ weight: MLXArray, scales: MLXArray, biases: MLXArray?,
    groupSize: Int, bits: Int, transform: SignedBlockHadamard
) throws {
    guard [2, 3, 4, 5, 6, 8].contains(bits), [32, 64, 128].contains(groupSize),
        weight.ndim == 2, weight.dtype == .uint32,
        weight.dim(0) > 0, weight.dim(1) == transform.width / 32 * bits,
        transform.width % 32 == 0, transform.width % groupSize == 0,
        scales.shape == [weight.dim(0), transform.width / groupSize],
        [DType.float32, .float16, .bfloat16].contains(scales.dtype),
        biases == nil || (biases!.shape == scales.shape && biases!.dtype == scales.dtype)
    else { throw HadamardError.incompatibleShape }
}

public struct HadamardGDNLayout {
    public let width: Int
    public let keyHeads: Int
    public let valueHeads: Int

    public init(width: Int, keyHeads: Int, valueHeads: Int) throws {
        guard width > 0, keyHeads > 0, valueHeads > 0,
            valueHeads % keyHeads == 0, width % valueHeads == 0
        else { throw HadamardError.incompatibleShape }
        self.width = width
        self.keyHeads = keyHeads
        self.valueHeads = valueHeads
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim > 0 && x.dim(-1) == width, "GDN input width mismatch")
        let repeats = valueHeads / keyHeads
        if repeats == 1 { return x }
        return x.reshaped([-1, repeats, keyHeads, width / valueHeads])
            .transposed(0, 2, 1, 3).reshaped(x.shape)
    }
}

private enum Float16ConstantQMV {
    static let fast = MLXFast.metalKernel(
        name: "hadamard_qmv_fast_f16_constants",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const int K = x_shape[x_ndim - 1];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int out_row = tid.y * 8 + simd_gid * 4;
            const device uint8_t* ws = (const device uint8_t*)w + out_row * in_w + simd_lid * 4;
            const device half* sc = scales + out_row * in_g + simd_lid / 8;
            const device half* bi = biases + out_row * in_g + simd_lid / 8;
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float x_thread[16];
            float result[4] = {0, 0, 0, 0};
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += float(xv.x) + float(xv.y) + float(xv.z) + float(xv.w);
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y * 0.25f;
                x_thread[4 * i + 2] = xv.z * 0.0625f;
                x_thread[4 * i + 3] = xv.w * 0.015625f;
              }
              for (int r = 0; r < 4; r++) {
                const uint packed_w = *(const device uint*)(ws + r * in_w);
                const float s = float(sc[r * in_g]);
                const float b = NEG ? -s : float(bi[r * in_g]);
                float accum = 0;
                for (int i = 0; i < 4; i++) {
                  const uint8_t w_byte = (packed_w >> (8 * i)) & 0xFF;
                  accum += (x_thread[4 * i] * (w_byte & 0x03) + x_thread[4 * i + 1] * (w_byte & 0x0c)
                      + x_thread[4 * i + 2] * (w_byte & 0x30) + x_thread[4 * i + 3] * (w_byte & 0xc0));
                }
                result[r] += s * accum + sum * b;
              }
              ws += 128;
              sc += 4;
              bi += 4;
              xp4 += 128;
            }
            for (int r = 0; r < 4; r++) {
              const float total = simd_sum(result[r]);
              if (simd_lid == 0) y[out_row + r] = total;
            }
            """)

    static let rows = MLXFast.metalKernel(
        name: "hadamard_qmv_rows_f16_constants",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const int K = x_shape[x_ndim - 1];
            const int N = w_shape[0];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int out_row = tid.y * (2 * outputs) + simd_gid * outputs;
            """ + rowsAccumulate + """
            for (int v = 0; v < rows; v++) {
              for (int r = 0; r < outputs; r++) {
                const float total = simd_sum(result[v][r]);
                if (simd_lid == 0) y[v * N + out_row + r] = total;
              }
            }
            """)

    static let rowsAccumulate = """
            const device uint8_t* ws = (const device uint8_t*)w + out_row * in_w + simd_lid * 4;
            const device half* sc = scales + out_row * in_g + simd_lid / 8;
            const device half* bi = biases + out_row * in_g + simd_lid / 8;
            const device float4* xp = (const device float4*)(x + simd_lid * 16);
            float result[rows][outputs];
            for (int v = 0; v < rows; v++) for (int r = 0; r < outputs; r++) result[v][r] = 0;
            for (int k = 0; k < K; k += 512) {
              float s4[outputs][4], b[outputs];
              uint packed[outputs];
              for (int r = 0; r < outputs; r++) {
                const float s = float(sc[r * in_g]);
                s4[r][0] = s;
                s4[r][1] = s * 0.25f;
                s4[r][2] = s * 0.0625f;
                s4[r][3] = s * 0.015625f;
                b[r] = float(bi[r * in_g]);
                packed[r] = *(const device uint*)(ws + r * in_w);
              }
              for (int i = 0; i < 4; i++) {
                float wd[outputs][4];
                for (int r = 0; r < outputs; r++) {
                  const uint q = packed[r] >> (8 * i);
                  wd[r][0] = fma(s4[r][0], float(q & 0x03u), b[r]);
                  wd[r][1] = fma(s4[r][1], float(q & 0x0cu), b[r]);
                  wd[r][2] = fma(s4[r][2], float(q & 0x30u), b[r]);
                  wd[r][3] = fma(s4[r][3], float(q & 0xc0u), b[r]);
                }
                for (int v = 0; v < rows; v++) {
                  const float4 xv = xp[v * in_w + i];
                  for (int r = 0; r < outputs; r++) {
                    result[v][r] = fma(xv.x, wd[r][0], result[v][r]);
                    result[v][r] = fma(xv.y, wd[r][1], result[v][r]);
                    result[v][r] = fma(xv.z, wd[r][2], result[v][r]);
                    result[v][r] = fma(xv.w, wd[r][3], result[v][r]);
                  }
                }
              }
              ws += 128;
              sc += 4;
              bi += 4;
              xp += 128;
            }
            """

    static let maximumRows = 7

    static func outputsPerSimdgroup(rows: Int) -> Int { rows <= 4 ? 4 : 2 }

    /// Top-2 order: value descending, token id ascending on exact ties, NaNs
    /// last. The same order as the CBv2 top-2 reduction over stored logits.
    static let topTwoHeader = """
        struct hadamard_top2 {
          float first_value;
          float second_value;
          uint first_id;
          uint second_id;
          uint count;
        };

        inline hadamard_top2 hadamard_top2_empty() {
          hadamard_top2 state;
          state.first_value = 0.0f;
          state.second_value = 0.0f;
          state.first_id = 0;
          state.second_id = 0;
          state.count = 0;
          return state;
        }

        inline bool hadamard_top2_better(float value, uint id, float current_value, uint current_id) {
          bool nan = isnan(value);
          bool current_nan = isnan(current_value);
          if (nan != current_nan) return !nan;
          if (value > current_value) return true;
          if (value < current_value) return false;
          return id < current_id;
        }

        inline void hadamard_top2_insert(thread hadamard_top2 &state, float value, uint id) {
          if (state.count > 0 && state.first_id == id) return;
          if (state.count > 1 && state.second_id == id) return;
          if (state.count == 0
              || hadamard_top2_better(value, id, state.first_value, state.first_id)) {
            if (state.count > 0) {
              state.second_value = state.first_value;
              state.second_id = state.first_id;
            }
            state.first_value = value;
            state.first_id = id;
            state.count = min(state.count + 1, 2u);
            return;
          }
          if (state.count == 1
              || hadamard_top2_better(value, id, state.second_value, state.second_id)) {
            state.second_value = value;
            state.second_id = id;
            state.count = 2;
          }
        }

        inline void hadamard_top2_merge(thread hadamard_top2 &state, hadamard_top2 other) {
          if (other.count > 0) hadamard_top2_insert(state, other.first_value, other.first_id);
          if (other.count > 1) hadamard_top2_insert(state, other.second_value, other.second_id);
        }
        """

    static let topTwoPartialSource = """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const int K = x_shape[1];
            const int in_w = K / 4;
            const int in_g = K / 128;
            hadamard_top2 best[rows];
            for (int v = 0; v < rows; v++) best[v] = hadamard_top2_empty();
            for (int block = 0; block < blocks; block++) {
            const int out_row = (tid.y * blocks + block) * (2 * outputs) + simd_gid * outputs;
            """ + rowsAccumulate + """
              for (int v = 0; v < rows; v++) {
                for (int r = 0; r < outputs; r++) {
                  const float total = simd_sum(result[v][r]);
                  hadamard_top2_insert(best[v], total, uint(out_row + r));
                }
              }
            }
            threadgroup hadamard_top2 scratch[rows];
            if (simd_gid == 1 && simd_lid == 0) {
              for (int v = 0; v < rows; v++) scratch[v] = best[v];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_gid == 0 && simd_lid == 0) {
              for (int v = 0; v < rows; v++) {
                hadamard_top2_merge(best[v], scratch[v]);
                const uint base = (uint(v) * uint(groups) + tid.y) * 2;
                partial_ids[base] = int(best[v].first_id);
                partial_ids[base + 1] = int(best[v].second_id);
                partial_values[base] = best[v].first_value;
                partial_values[base + 1] = best[v].second_value;
              }
            }
            """

    static let topTwoPartial = MLXFast.metalKernel(
        name: "hadamard_qmv_rows_f16_constants_top2",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["partial_ids", "partial_values"],
        source: topTwoPartialSource,
        header: topTwoHeader)

    static let topTwoFinalizeSource = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint groups = partial_ids_shape[1];
            hadamard_top2 local = hadamard_top2_empty();
            for (uint g = lane; g < groups; g += 256) {
              const uint base = (row * groups + g) * 2;
              hadamard_top2_insert(local, partial_values[base], uint(partial_ids[base]));
              hadamard_top2_insert(local, partial_values[base + 1], uint(partial_ids[base + 1]));
            }
            for (uint stride = 16; stride > 0; stride >>= 1) {
              hadamard_top2 other;
              other.first_value = simd_shuffle_down(local.first_value, stride);
              other.second_value = simd_shuffle_down(local.second_value, stride);
              other.first_id = simd_shuffle_down(local.first_id, stride);
              other.second_id = simd_shuffle_down(local.second_id, stride);
              other.count = simd_shuffle_down(local.count, stride);
              if (simd_lid < stride) hadamard_top2_merge(local, other);
            }
            threadgroup hadamard_top2 scratch[8];
            if (simd_lid == 0) scratch[simd_gid] = local;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              for (uint s = 1; s < 8; s++) hadamard_top2_merge(local, scratch[s]);
              top_ids[row * 2] = int(local.first_id);
              top_ids[row * 2 + 1] = int(local.second_id);
              top_values[row * 2] = local.first_value;
              top_values[row * 2 + 1] = local.second_value;
            }
            """

    static let topTwoFinalize = MLXFast.metalKernel(
        name: "hadamard_qmv_top2_finalize",
        inputNames: ["partial_ids", "partial_values"],
        outputNames: ["top_ids", "top_values"],
        source: topTwoFinalizeSource,
        header: topTwoHeader)

    static let topTwoBlocks = 8

    static func topTwo(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray
    ) -> (ids: MLXArray, values: MLXArray)? {
        let K = x.dim(-1)
        let rows = x.size / K
        let N = weight.dim(0)
        let outputs = 4
        let span = 2 * outputs * topTwoBlocks
        guard (2 ... 4).contains(rows), outputsPerSimdgroup(rows: rows) == outputs,
            N % span == 0
        else { return nil }
        let groups = N / span
        let partials = topTwoPartial(
            [x.reshaped([rows, K]), weight, scales, biases],
            template: [
                ("rows", rows), ("outputs", outputs), ("blocks", topTwoBlocks),
                ("groups", groups),
            ],
            grid: (32, groups * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[rows, groups, 2], [rows, groups, 2]],
            outputDTypes: [.int32, .float32])
        let top = topTwoFinalize(
            partials, grid: (rows * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, 2], [rows, 2]], outputDTypes: [.int32, .float32])
        return (top[0], top[1])
    }

    static func matmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        negative: Bool = false
    ) -> MLXArray {
        let K = x.dim(-1)
        let rows = x.size / K
        let N = weight.dim(0)
        // Shape metadata supplies K and N without allocating scalar tensor inputs. The kernels
        // index x and y as flat row-major [rows, K] and [rows, N], so x and y keep their
        // leading dimensions and the graph needs no reshape nodes around the call.
        let outputs = rows == 1 ? 4 : outputsPerSimdgroup(rows: rows)
        return (rows == 1 ? fast : Self.rows)(
            [x, weight, scales, biases],
            template: rows == 1 ? [("NEG", negative ? 1 : 0)] : [("rows", rows), ("outputs", outputs)],
            grid: (32, N / (2 * outputs) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [Array(x.shape.dropLast()) + [N]], outputDTypes: [.float32])[0]
    }
}

enum ExactVerifyHead {
    static let cap = 256
    static let threads = 512


    /// Positive normal-or-zero arithmetic, rounded outwards after each step.
    /// Unsafe ranges return infinity and retain the existing screen bound.
    static let l2Header = """
        inline bool head_l2_safe(float x) {
          const uint u = as_type<uint>(x);
          return u == 0u || (u >= 0x00800000u && u < 0x7f800000u);
        }
        inline float head_l2_up(float x) {
          const uint u = as_type<uint>(x);
          if (u == 0u) return 0.0f;
          return u >= 0x00800000u && u < 0x7f7fffffu ? as_type<float>(u + 1u) : INFINITY;
        }
        inline float head_l2_mul(float a, float b) {
          #pragma clang fp contract(off)
          #pragma clang fp reassociate(off)
          if (!head_l2_safe(a) || !head_l2_safe(b)) return INFINITY;
          if (a == 0.0f || b == 0.0f) return 0.0f;
          const float p = a * b;
          return p == 0.0f ? INFINITY : head_l2_up(p);
        }
        inline float head_l2_add(float a, float b) {
          #pragma clang fp contract(off)
          #pragma clang fp reassociate(off)
          if (!head_l2_safe(a) || !head_l2_safe(b)) return INFINITY;
          if (a == 0.0f) return b;
          if (b == 0.0f) return a;
          return head_l2_up(a + b);
        }
        inline float head_l2_root(float p) {
          if (!head_l2_safe(p)) return INFINITY;
          if (p == 0.0f) return 0.0f;
          // AM >= GM on each exponent interval; +1 rounds the bit shift up.
          return as_type<float>((as_type<uint>(p) >> 1) + 0x1fc00001u);
        }
        inline float head_l2_error(float old_e, const device float* ascale,
            const device float* bounds, const device float* error2, uint row, int groups) {
          float total = 0.0f;
          for (int g = 0; g < groups; g++) {
            const float a = ascale[row * uint(groups) + uint(g)];
            const float d2 = error2[row * uint(groups) + uint(g)];
            const float c2 = bounds[2 * groups + g];
            if (!head_l2_safe(d2) || !head_l2_safe(c2)) return old_e;
            const float radius = head_l2_root(head_l2_mul(d2, c2));
            if (!head_l2_safe(radius)) return old_e;
            const float worst = head_l2_mul(head_l2_mul(0.5f, a), bounds[g]);
            const float rounding = head_l2_mul(a, head_l2_add(
                head_l2_mul(head_l2_mul(127.0f, 3.0e-5f), bounds[g]),
                head_l2_mul(1.0e-5f, bounds[groups + g])));
            total = head_l2_add(total, head_l2_add(min(worst, radius), rounding));
          }
          const float upper = head_l2_mul(total, 1.001f);
          return head_l2_safe(upper) ? min(old_e, upper) : old_e;
        }
        """

    static let partitionL2Header = l2Header + """
        inline float head_l2_partition_error(float old_e, const device float* ascale,
            const device float* bounds, const device float* error2, uint row, int groups, uint block, thread float& local_e) {
          local_e = old_e;
          float total = 0.0f, part_total = 0.0f;
          bool part_ok = true;
          for (int g = 0; g < groups; g++) {
            const float a = ascale[row * uint(groups) + uint(g)];
            const float d2 = error2[row * uint(groups) + uint(g)];
            const float c2 = bounds[2 * groups + g];
            if (!head_l2_safe(d2) || !head_l2_safe(c2)) return old_e;
            const float radius = head_l2_root(head_l2_mul(d2, c2));
            if (!head_l2_safe(radius)) return old_e;
            const float worst = head_l2_mul(head_l2_mul(0.5f, a), bounds[g]);
            const float rounding = head_l2_mul(a, head_l2_add(
                head_l2_mul(head_l2_mul(127.0f, 3.0e-5f), bounds[g]),
                head_l2_mul(1.0e-5f, bounds[groups + g])));
            total = head_l2_add(total, head_l2_add(min(worst, radius), rounding));
            if (part_ok) {
              const float part_c2 = bounds[(3 + int(block)) * groups + g];
              const float part_radius = head_l2_root(head_l2_mul(d2, part_c2));
              part_ok = head_l2_safe(part_c2) && head_l2_safe(part_radius);
              if (part_ok) part_total = head_l2_add(part_total, head_l2_add(min(worst, part_radius), rounding));
            }
          }
          const float upper = head_l2_mul(total, 1.001f);
          const float global_e = head_l2_safe(upper) ? min(old_e, upper) : old_e;
          local_e = global_e;
          if (head_l2_safe(upper) && part_ok) {
            const float part_upper = head_l2_mul(part_total, 1.001f);
            if (head_l2_safe(part_upper)) local_e = min(global_e, part_upper);
          }
          return global_e;
        }
        """

    static let jointL2Header: String = {
        let start = partitionL2Header.range(of: "inline float head_l2_partition_error")!.lowerBound
        let function = String(partitionL2Header[start...])
            .replacingOccurrences(of: "head_l2_partition_error(", with: "head_l2_joint_partition_error(")
            .replacingOccurrences(of: "thread float& local_e)", with: "thread float& local_e, const device float* joint_c2)")
            .replacingOccurrences(of: "bool part_ok = true;", with: "bool part_ok = true, joint_ok = true; float segment_d2 = 0.0f, joint_radius = 0.0f, joint_rounding = 0.0f;")
            .replacingOccurrences(of: "if (part_ok) {", with: """
              if (joint_ok) {
                segment_d2 = head_l2_add(segment_d2, d2);
                joint_rounding = head_l2_add(joint_rounding, rounding);
                if ((g & 7) == 7) {
                  const float segment_c2 = joint_c2[block * 4u + uint(g >> 3)];
                  const float segment_radius = head_l2_root(head_l2_mul(segment_d2, segment_c2));
                  joint_ok = head_l2_safe(segment_c2) && head_l2_safe(segment_radius);
                  if (joint_ok) joint_radius = head_l2_add(joint_radius, segment_radius);
                  segment_d2 = 0.0f;
                }
              }
              if (part_ok) {
            """)
            .replacingOccurrences(of: "return global_e;", with: """
            if (joint_ok) {
              const float joint_upper = head_l2_mul(head_l2_add(joint_radius, joint_rounding), 1.001f);
              if (head_l2_safe(joint_upper)) local_e = min(local_e, joint_upper);
            }
            return global_e;
            """)
        return partitionL2Header + function
    }()

    static func error2Source(elements: Int, cluster: Int) -> String {
        var reduce = ""
        var stride = 1
        while stride < cluster {
            reduce += "head_error2 = head_l2_add(head_error2, simd_shuffle_xor(head_error2, ushort(\(stride))));\n"
            stride *= 2
        }
        return """
            float head_error2 = 0.0f;
            const uint head_qs = as_type<uint>(qs);
            for (uint r = 0; r < \(elements)u; r++) {
              const uint vm = as_type<uint>(v[r]) & 0x7fffffffu;
              const uint qm = as_type<uint>(q[r]) & 0x7fffffffu;
              if (head_qs < 0x2f800000u || head_qs > 0x4f800000u
                  || (vm != 0u && vm < 0x21800000u) || vm > 0x53800000u
                  || qm >= 0x7f800000u || fabs(q[r]) > 127.0f) {
                head_error2 = INFINITY; continue;
              }
              const float d = fabs(fma(-q[r], qs, v[r]));
              const uint dm = as_type<uint>(d);
              if (dm != 0u && dm < 0x21800000u) { head_error2 = INFINITY; continue; }
              const float upper = head_l2_up(d);
              head_error2 = head_l2_add(head_error2, head_l2_mul(upper, upper));
            }
            \(reduce)
            """
    }

    static let boundsSource = """
            const uint g = threadgroup_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const int N = w_shape[0];
            const int words = w_shape[1];
            const int G = scales_shape[1];
            float amax = 0.0f, bmax = 0.0f, cmax = 0.0f;
            for (int n = int(lane); n < N; n += 256) {
              const device uint32_t* wr = w + size_t(n) * size_t(words) + g * 8;
              int n1 = 0, n2 = 0, n3 = 0;
              for (int j = 0; j < 8; j++) {
                const uint v = wr[j];
                const uint lo = v & 0x55555555u;
                const uint hi = (v >> 1) & 0x55555555u;
                n3 += popcount(lo & hi);
                n1 += popcount(lo & ~hi);
                n2 += popcount(hi & ~lo);
              }
              const int n0 = 128 - n1 - n2 - n3;
              const float s = float(scales[size_t(n) * size_t(G) + g]);
              const float b = float(biases[size_t(n) * size_t(G) + g]);
              const float a = float(n0) * fabs(b) + float(n1) * fabs(s + b)
                  + float(n2) * fabs(2.0f * s + b) + float(n3) * fabs(3.0f * s + b);
              const float bb = 127.0f * (fabs(s) * float(n1 + 2 * n2 + 3 * n3) + 128.0f * fabs(b));
              const float w0 = fabs(b), w1 = fabs(fma(s, 1.0f, b));
              const float w2 = fabs(fma(s, 2.0f, b)), w3 = fabs(fma(s, 3.0f, b));
              const float cc = head_l2_add(head_l2_add(
                  head_l2_mul(float(n0), head_l2_mul(w0, w0)), head_l2_mul(float(n1), head_l2_mul(w1, w1))),
                  head_l2_add(head_l2_mul(float(n2), head_l2_mul(w2, w2)), head_l2_mul(float(n3), head_l2_mul(w3, w3))));
              cmax = max(cmax, cc);
              amax = max(amax, a);
              bmax = max(bmax, bb);
            }
            amax = simd_max(amax);
            bmax = simd_max(bmax);
            cmax = simd_max(cmax);
            threadgroup float sa[8], sb[8], sc2[8];
            if (simd_lid == 0) { sa[simd_gid] = amax; sb[simd_gid] = bmax; sc2[simd_gid] = cmax; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              for (int i = 1; i < 8; i++) { amax = max(amax, sa[i]); bmax = max(bmax, sb[i]); cmax = max(cmax, sc2[i]); }
              bounds[g] = amax;
              bounds[G + g] = bmax;
              bounds[2 * G + g] = cmax;
            }
            """
    static let bounds = MLXFast.metalKernel(
        name: "hadamard_xhead_bounds_l2_v1",
        inputNames: ["w", "scales", "biases"], outputNames: ["bounds"],
        source: boundsSource, header: l2Header)

    static let partitionBoundsSource = boundsSource
        .replacingOccurrences(of: "float amax = 0.0f, bmax = 0.0f, cmax = 0.0f;", with: "threadgroup float part_scratch[16 * 8]; float amax = 0.0f, bmax = 0.0f, cmax = 0.0f;")
        .replacingOccurrences(of: "for (int n = int(lane); n < N; n += 256) {", with: "for (int partition = 0; partition < 16; partition++) { float part_max = 0.0f; const int first = partition * (N / 16); const int end = first + N / 16; for (int n = first + int(lane); n < end; n += 256) {")
        .replacingOccurrences(of: "cmax = max(cmax, cc);", with: "cmax = max(cmax, cc); part_max = max(part_max, cc);")
        .replacingOccurrences(of: "amax = simd_max(amax);", with: "part_max = simd_max(part_max); if (simd_lid == 0) part_scratch[partition * 8 + simd_gid] = part_max; } amax = simd_max(amax);")
        .replacingOccurrences(of: "bounds[2 * G + g] = cmax;", with: "bounds[2 * G + g] = cmax; for (int partition = 0; partition < 16; partition++) { float part_max = part_scratch[partition * 8]; for (int i = 1; i < 8; i++) part_max = max(part_max, part_scratch[partition * 8 + i]); bounds[(3 + partition) * G + g] = part_max; }")
    static let partitionBounds = MLXFast.metalKernel(
        name: "hadamard_xhead_partition_bounds16_l2_v1",
        inputNames: ["w", "scales", "biases"], outputNames: ["bounds"],
        source: partitionBoundsSource, header: l2Header)

    static let jointBoundsSource: String = {
        let begin = boundsSource.range(of: "const device uint32_t* wr")!.lowerBound
        let end = boundsSource.range(of: "const float a = float(n0)")!.lowerBound
        let normBegin = boundsSource.range(of: "const float w0 = fabs(b)")!.lowerBound
        let normEnd = boundsSource.range(of: "cmax = max(cmax, cc);")!.lowerBound
        return """
            const uint lane = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint segment = threadgroup_position_in_grid.x % 4u;
            const uint partition = threadgroup_position_in_grid.x / 4u;
            const int N = w_shape[0], words = w_shape[1], G = scales_shape[1];
            const int first = int(partition) * (N / 16), end = first + N / 16;
            float maximum = 0.0f;
            for (int n = first + int(lane); n < end; n += 256) {
              float sum = 0.0f;
              for (uint g = segment * 8u; g < (segment + 1u) * 8u; g++) {
            """ + String(boundsSource[begin..<end]) + String(boundsSource[normBegin..<normEnd]) + """
                sum = head_l2_add(sum, cc);
              }
              maximum = max(maximum, sum);
            }
            maximum = simd_max(maximum);
            threadgroup float scratch[8];
            if (simd_lid == 0) scratch[simd_gid] = maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              for (int i = 1; i < 8; i++) maximum = max(maximum, scratch[i]);
              joint_c2[partition * 4u + segment] = maximum;
            }
            """
    }()
    static let jointBounds = MLXFast.metalKernel(
        name: "hadamard_xhead_joint_k4_bounds_v1",
        inputNames: ["w", "scales", "biases"], outputNames: ["joint_c2"],
        source: jointBoundsSource, header: l2Header)

    static let selectSource = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const int K = x_shape[1];
            const int N = w_shape[0];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const device float* lrow = logits + size_t(row) * size_t(N);
            threadgroup atomic_int any_bad;
            threadgroup atomic_int cand_count;
            threadgroup int cand_ids[CAP];
            threadgroup float cand_values[CAP];
            threadgroup hadamard_top2 scratch[THREADS / 32];
            threadgroup float shared_tau[2];
            if (lane == 0) {
              atomic_store_explicit(&any_bad, 0, memory_order_relaxed);
              atomic_store_explicit(&cand_count, 0, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            hadamard_top2 local = hadamard_top2_empty();
            bool bad = false;
            for (int n = int(lane); n < N; n += THREADS) {
              const float v = lrow[n];
              bad = bad || !isfinite(v);
              if (local.count == 2 && !(v > local.second_value)) continue;
              hadamard_top2_insert(local, v, uint(n));
            }
            if (bad) atomic_store_explicit(&any_bad, 1, memory_order_relaxed);
            for (uint stride = 16; stride > 0; stride >>= 1) {
              hadamard_top2 other;
              other.first_value = simd_shuffle_down(local.first_value, stride);
              other.second_value = simd_shuffle_down(local.second_value, stride);
              other.first_id = simd_shuffle_down(local.first_id, stride);
              other.second_id = simd_shuffle_down(local.second_id, stride);
              other.count = simd_shuffle_down(local.count, stride);
              if (simd_lid < stride) hadamard_top2_merge(local, other);
            }
            if (simd_lid == 0) scratch[simd_gid] = local;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              for (uint s = 1; s < THREADS / 32; s++) hadamard_top2_merge(local, scratch[s]);
              float e = 0.0f;
              for (int g = 0; g < in_g; g++) {
                e += ascale[row * uint(in_g) + uint(g)]
                    * (bounds[g] * (0.5f + 127.0f * 3.0e-5f) + bounds[in_g + g] * 1.0e-5f);
              }
              e *= 1.001f;
              const float t = local.second_value - 2.0f * e;
              shared_tau[0] = t - (fabs(t) * 2.4e-7f + 1.0e-30f);
              shared_tau[1] = e;
              if (local.count < 2 || !isfinite(e) || !isfinite(t))
                atomic_store_explicit(&any_bad, 1, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tau = shared_tau[0];
            for (int n = int(lane); n < N; n += THREADS) {
              if (lrow[n] >= tau) {
                const int slot = atomic_fetch_add_explicit(&cand_count, 1, memory_order_relaxed);
                if (slot < CAP) cand_ids[slot] = n;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int count = atomic_load_explicit(&cand_count, memory_order_relaxed);
            const int used = min(count, CAP);
            for (int c = int(simd_gid); c < used; c += THREADS / 32) {
              const int out_row = cand_ids[c];
              constexpr int rows = 1;
              constexpr int outputs = 1;
            """ + Float16ConstantQMV.rowsAccumulate.replacingOccurrences(
                of: "(x + simd_lid", with: "(x + size_t(row) * size_t(K) + simd_lid"
            ).replacingOccurrences(
                of: "b[r] = float(bi[r * in_g]);", with: "b[r] = NEG ? -s : float(bi[r * in_g]);"
            ) + """
              const float total = simd_sum(result[0][0]);
              if (simd_lid == 0) cand_values[c] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              hadamard_top2 best = hadamard_top2_empty();
              bool nonfinite = false;
              for (int c = 0; c < used; c++) {
                nonfinite = nonfinite || !isfinite(cand_values[c]);
                hadamard_top2_insert(best, cand_values[c], uint(cand_ids[c]));
              }
              const bool fallback = atomic_load_explicit(&any_bad, memory_order_relaxed) != 0
                  || count > CAP || nonfinite || best.count < 2;
              sel_ids[row * 2] = int(best.first_id);
              sel_ids[row * 2 + 1] = int(best.second_id);
              sel_values[row * 2] = best.first_value;
              sel_values[row * 2 + 1] = best.second_value;
              flag[row] = fallback ? 1 : 0;
              info[row * 2] = shared_tau[1];
              info[row * 2 + 1] = float(count);
            }
            """

    static let select = MLXFast.metalKernel(
        name: "hadamard_xhead_select",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases"],
        outputNames: ["sel_ids", "sel_values", "flag", "info"],
        source: selectSource,
        header: Float16ConstantQMV.topTwoHeader)


    static let compactSelectSource = selectSource
        .replacingOccurrences(of: "const int N = w_shape[0];", with: "const int N = logits_shape[1]; const int TILES = N / 4;")
        .replacingOccurrences(of: "threadgroup atomic_int any_bad;", with: "threadgroup atomic_int any_bad; threadgroup atomic_int omitted_overflow;")
        .replacingOccurrences(of: "atomic_store_explicit(&any_bad, 0, memory_order_relaxed);", with: "atomic_store_explicit(&any_bad, 0, memory_order_relaxed); atomic_store_explicit(&omitted_overflow, 0, memory_order_relaxed);")
        .replacingOccurrences(of: "bool bad = false;", with: "bool bad = false; for (int tile = int(lane); tile < TILES; tile += THREADS) bad = bad || part_bad[row * uint(TILES) + uint(tile)] != 0;")
        .replacingOccurrences(of: "hadamard_top2_insert(local, v, uint(n));", with: "hadamard_top2_insert(local, v, uint(part_ids[row * uint(N) + uint(n)]));")
        .replacingOccurrences(of: "if (local.count == 2 && !(v > local.second_value)) continue;", with: "if (local.count == 2 && !hadamard_top2_better(v, uint(part_ids[row * uint(N) + uint(n)]), local.second_value, local.second_id)) continue;")
        .replacingOccurrences(of: "const float tau = shared_tau[0];", with: "const float tau = shared_tau[0]; for (int tile = int(lane); tile < TILES; tile += THREADS) { if (omitted[row * uint(TILES) + uint(tile)] >= tau) atomic_store_explicit(&omitted_overflow, 1, memory_order_relaxed); }")
        .replacingOccurrences(of: "if (slot < CAP) cand_ids[slot] = n;", with: "if (slot < CAP) cand_ids[slot] = part_ids[row * uint(N) + uint(n)];")
        .replacingOccurrences(of: "|| count > CAP || nonfinite || best.count < 2;", with: "|| atomic_load_explicit(&omitted_overflow, memory_order_relaxed) != 0 || count > CAP || nonfinite || best.count < 2;")
        .replacingOccurrences(of: "info[row * 2 + 1] = float(count);", with: "info[row * 2 + 1] = atomic_load_explicit(&omitted_overflow, memory_order_relaxed) != 0 ? -1.0f : float(count);")

    static let compactSelect = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_select",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "omitted", "part_bad"],
        outputNames: ["sel_ids", "sel_values", "flag", "info"],
        source: compactSelectSource, header: Float16ConstantQMV.topTwoHeader)

    static func runCompact(
        rotated: MLXArray, summary: (ids: MLXArray, values: MLXArray, omitted: MLXArray, bad: MLXArray),
        ascale: MLXArray, bounds: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        rows: Int, negative: Bool
    ) -> (ids: MLXArray, values: MLXArray) {
        let x = rotated.reshaped([rows, rotated.dim(-1)])
        let selected = compactSelect(
            [summary.values.reshaped([rows, -1]), ascale, bounds, x, weight, scales, biases,
             summary.ids.reshaped([rows, -1]), summary.omitted, summary.bad],
            template: [("CAP", cap), ("THREADS", threads), ("NEG", negative ? 1 : 0)],
            grid: (rows * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, 2], [rows, 2], [rows], [rows, 2]],
            outputDTypes: [.int32, .float32, .int32, .float32])
        let groups = weight.dim(0) / (8 * Float16ConstantQMV.topTwoBlocks)
        let partials = guardedPartial(
            [x, weight, scales, biases, selected[2]],
            template: [("rows", rows), ("outputs", 4), ("blocks", Float16ConstantQMV.topTwoBlocks), ("groups", groups)],
            grid: (32, groups * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[rows, groups, 2], [rows, groups, 2]],
            outputDTypes: [.int32, .float32])
        let top = finalize(
            [partials[0], partials[1], selected[2], selected[0], selected[1]],
            grid: (rows * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, 2], [rows, 2]], outputDTypes: [.int32, .float32])
        return (top[0], top[1])
    }

    static let guardedPartial = MLXFast.metalKernel(
        name: "hadamard_xhead_guarded_top2_masked_rows_v1",
        inputNames: ["x", "w", "scales", "biases", "flag"],
        outputNames: ["partial_ids", "partial_values"],
        source: """
            uint active_rows = 0u;
            for (int v = 0; v < rows; v++) {
              if (flag[v] != 0) active_rows |= 1u << uint(v);
            }
            if (active_rows == 0u) return;
            """ + Float16ConstantQMV.topTwoPartialSource.replacingOccurrences(
                of: "for (int v = 0; v < rows; v++) {",
                with: "for (int v = 0; v < rows; v++) { if ((active_rows & (1u << uint(v))) == 0u) continue;"),
        header: Float16ConstantQMV.topTwoHeader)

    static let finalize = MLXFast.metalKernel(
        name: "hadamard_xhead_finalize",
        inputNames: ["partial_ids", "partial_values", "flag", "sel_ids", "sel_values"],
        outputNames: ["top_ids", "top_values"],
        source: """
            if (flag[threadgroup_position_in_grid.x] == 0) {
              if (thread_position_in_threadgroup.x == 0) {
                const uint r2 = threadgroup_position_in_grid.x * 2;
                top_ids[r2] = sel_ids[r2];
                top_ids[r2 + 1] = sel_ids[r2 + 1];
                top_values[r2] = sel_values[r2];
                top_values[r2 + 1] = sel_values[r2 + 1];
              }
              return;
            }
            """ + Float16ConstantQMV.topTwoFinalizeSource,
        header: Float16ConstantQMV.topTwoHeader)

    static let blocks = 64
    static let blockThreads = 256

    static let selectScanSource = """
            const uint block = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lane = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const int quads = NV / (4 * BLOCKS);
            const uint first = block * uint(quads) * 4u;
            const device float4* lq = (const device float4*)(logits + size_t(row) * size_t(NV) + first);
            threadgroup hadamard_top2 scratch[THREADS / 32];
            threadgroup int scratch_bad[THREADS / 32];
            hadamard_top2 local = hadamard_top2_empty();
            bool bad = false;
            for (int q = int(lane); q < quads; q += THREADS) {
              const float4 v4 = lq[q];
              for (int i = 0; i < 4; i++) {
                const float v = v4[i];
                bad = bad || !isfinite(v);
                if (local.count == 2 && !(v > local.second_value)) continue;
                hadamard_top2_insert(local, v, first + uint(q) * 4u + uint(i));
              }
            }
            for (uint stride = 16; stride > 0; stride >>= 1) {
              hadamard_top2 other;
              other.first_value = simd_shuffle_down(local.first_value, stride);
              other.second_value = simd_shuffle_down(local.second_value, stride);
              other.first_id = simd_shuffle_down(local.first_id, stride);
              other.second_id = simd_shuffle_down(local.second_id, stride);
              other.count = simd_shuffle_down(local.count, stride);
              if (simd_lid < stride) hadamard_top2_merge(local, other);
            }
            const int any_bad = simd_max(bad ? 1 : 0);
            if (simd_lid == 0) { scratch[simd_gid] = local; scratch_bad[simd_gid] = any_bad; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              int b = scratch_bad[0];
              for (uint s = 1; s < THREADS / 32; s++) {
                hadamard_top2_merge(local, scratch[s]);
                b |= scratch_bad[s];
              }
              const uint p = row * uint(BLOCKS) + block;
              part_ids[p * 2] = int(local.first_id);
              part_ids[p * 2 + 1] = int(local.second_id);
              part_values[p * 2] = local.first_value;
              part_values[p * 2 + 1] = local.second_value;
              part_meta[p] = b | int(local.count << 1);
            }
            """

    static let selectScan = MLXFast.metalKernel(
        name: "hadamard_xhead_select_scan",
        inputNames: ["logits"],
        outputNames: ["part_ids", "part_values", "part_meta"],
        source: selectScanSource,
        header: Float16ConstantQMV.topTwoHeader)

    static let selectRescoreSource = """
            const uint block = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lane = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const int K = x_shape[1];
            const int in_w = K / 4;
            const int in_g = K / 128;
            threadgroup atomic_int cand_count;
            threadgroup int cand_ids[CAP];
            threadgroup float cand_values[CAP];
            threadgroup float shared_tau[2];
            threadgroup int shared_bad;
            if (lane == 0) atomic_store_explicit(&cand_count, 0, memory_order_relaxed);
            if (simd_gid == 0) {
              hadamard_top2 local = hadamard_top2_empty();
              int pbad = 0;
              for (uint pb = simd_lid; pb < uint(BLOCKS); pb += 32) {
                const uint p = row * uint(BLOCKS) + pb;
                const int meta = part_meta[p];
                hadamard_top2 o;
                o.first_value = part_values[p * 2];
                o.second_value = part_values[p * 2 + 1];
                o.first_id = uint(part_ids[p * 2]);
                o.second_id = uint(part_ids[p * 2 + 1]);
                o.count = uint(meta >> 1);
                hadamard_top2_merge(local, o);
                pbad |= meta & 1;
              }
              for (uint stride = 16; stride > 0; stride >>= 1) {
                hadamard_top2 other;
                other.first_value = simd_shuffle_down(local.first_value, stride);
                other.second_value = simd_shuffle_down(local.second_value, stride);
                other.first_id = simd_shuffle_down(local.first_id, stride);
                other.second_id = simd_shuffle_down(local.second_id, stride);
                other.count = simd_shuffle_down(local.count, stride);
                if (simd_lid < stride) hadamard_top2_merge(local, other);
              }
              pbad = simd_max(pbad);
              if (simd_lid == 0) {
                float e = 0.0f;
                for (int g = 0; g < in_g; g++) {
                  e += ascale[row * uint(in_g) + uint(g)]
                      * (bounds[g] * (0.5f + 127.0f * 3.0e-5f) + bounds[in_g + g] * 1.0e-5f);
                }
                e *= 1.001f;
                const float t = local.second_value - 2.0f * e;
                shared_tau[0] = t - (fabs(t) * 2.4e-7f + 1.0e-30f);
                shared_tau[1] = e;
                shared_bad = (pbad != 0 || local.count < 2 || !isfinite(e) || !isfinite(t)) ? 1 : 0;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tau = shared_tau[0];
            const int quads = NV / (4 * BLOCKS);
            const uint first = block * uint(quads) * 4u;
            const device float4* lq = (const device float4*)(logits + size_t(row) * size_t(NV) + first);
            for (int q = int(lane); q < quads; q += THREADS) {
              const float4 v4 = lq[q];
              for (int i = 0; i < 4; i++) {
                if (v4[i] >= tau) {
                  const int slot = atomic_fetch_add_explicit(&cand_count, 1, memory_order_relaxed);
                  if (slot < CAP) cand_ids[slot] = int(first + uint(q) * 4u + uint(i));
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int count = atomic_load_explicit(&cand_count, memory_order_relaxed);
            const int used = count > CAP ? 0 : count;
            for (int c = int(simd_gid); c < used; c += THREADS / 32) {
              const int out_row = cand_ids[c];
              constexpr int rows = 1;
              constexpr int outputs = 1;
            """ + Float16ConstantQMV.rowsAccumulate.replacingOccurrences(
                of: "(x + simd_lid", with: "(x + size_t(row) * size_t(K) + simd_lid"
            ).replacingOccurrences(
                of: "b[r] = float(bi[r * in_g]);", with: "b[r] = NEG ? -s : float(bi[r * in_g]);"
            ) + """
              const float total = simd_sum(result[0][0]);
              if (simd_lid == 0) cand_values[c] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              hadamard_top2 best = hadamard_top2_empty();
              bool nonfinite = false;
              for (int c = 0; c < used; c++) {
                nonfinite = nonfinite || !isfinite(cand_values[c]);
                hadamard_top2_insert(best, cand_values[c], uint(cand_ids[c]));
              }
              const uint p = row * uint(BLOCKS) + block;
              blk_ids[p * 2] = int(best.first_id);
              blk_ids[p * 2 + 1] = int(best.second_id);
              blk_values[p * 2] = best.first_value;
              blk_values[p * 2 + 1] = best.second_value;
              blk_meta[p * 4] = count;
              blk_meta[p * 4 + 1] = int(best.count);
              blk_meta[p * 4 + 2] = nonfinite ? 1 : 0;
              blk_meta[p * 4 + 3] = shared_bad;
              if (block == 0) row_e[row] = shared_tau[1];
            }
            """

    static let selectRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_select_rescore",
        inputNames: [
            "logits", "ascale", "bounds", "x", "w", "scales", "biases",
            "part_ids", "part_values", "part_meta",
        ],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: selectRescoreSource,
        header: Float16ConstantQMV.topTwoHeader)

    static let selectMergeSource = """
            const uint row = threadgroup_position_in_grid.x;
            const uint simd_lid = thread_index_in_simdgroup;
            hadamard_top2 best = hadamard_top2_empty();
            int count = 0, nonfinite = 0, bad = 0;
            for (uint pb = simd_lid; pb < uint(BLOCKS); pb += 32) {
              const uint p = row * uint(BLOCKS) + pb;
              count += blk_meta[p * 4];
              const int valid = blk_meta[p * 4 + 1];
              nonfinite |= blk_meta[p * 4 + 2];
              bad |= blk_meta[p * 4 + 3];
              if (valid > 0) hadamard_top2_insert(best, blk_values[p * 2], uint(blk_ids[p * 2]));
              if (valid > 1) hadamard_top2_insert(best, blk_values[p * 2 + 1], uint(blk_ids[p * 2 + 1]));
            }
            for (uint stride = 16; stride > 0; stride >>= 1) {
              hadamard_top2 other;
              other.first_value = simd_shuffle_down(best.first_value, stride);
              other.second_value = simd_shuffle_down(best.second_value, stride);
              other.first_id = simd_shuffle_down(best.first_id, stride);
              other.second_id = simd_shuffle_down(best.second_id, stride);
              other.count = simd_shuffle_down(best.count, stride);
              if (simd_lid < stride) hadamard_top2_merge(best, other);
            }
            count = simd_sum(count);
            nonfinite = simd_max(nonfinite);
            bad = simd_max(bad);
            if (simd_lid == 0) {
              const bool fallback = bad != 0 || count > CAP || nonfinite != 0 || best.count < 2;
              sel_ids[row * 2] = int(best.first_id);
              sel_ids[row * 2 + 1] = int(best.second_id);
              sel_values[row * 2] = best.first_value;
              sel_values[row * 2 + 1] = best.second_value;
              flag[row] = fallback ? 1 : 0;
              info[row * 2] = row_e[row];
              info[row * 2 + 1] = float(count);
            }
            """

    static let selectMerge = MLXFast.metalKernel(
        name: "hadamard_xhead_select_merge",
        inputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        outputNames: ["sel_ids", "sel_values", "flag", "info"],
        source: selectMergeSource,
        header: Float16ConstantQMV.topTwoHeader)

    static func parallelSelect(
        logits: MLXArray, ascale: MLXArray, bounds: MLXArray, x: MLXArray,
        weight: MLXArray, scales: MLXArray, biases: MLXArray, rows: Int, negative: Bool
    ) -> [MLXArray] {
        let N = weight.dim(0)
        let template: [(String, any KernelTemplateArg)] = [
            ("CAP", cap), ("THREADS", blockThreads), ("BLOCKS", blocks), ("NV", N),
            ("NEG", negative ? 1 : 0),
        ]
        let parts = selectScan(
            [logits], template: [("THREADS", blockThreads), ("BLOCKS", blocks), ("NV", N)],
            grid: (blocks * blockThreads, rows, 1), threadGroup: (blockThreads, 1, 1),
            outputShapes: [[rows, blocks, 2], [rows, blocks, 2], [rows, blocks]],
            outputDTypes: [.int32, .float32, .int32])
        let blk = selectRescore(
            [logits, ascale, bounds, x, weight, scales, biases, parts[0], parts[1], parts[2]],
            template: template,
            grid: (blocks * blockThreads, rows, 1), threadGroup: (blockThreads, 1, 1),
            outputShapes: [[rows, blocks, 2], [rows, blocks, 2], [rows, blocks, 4], [rows]],
            outputDTypes: [.int32, .float32, .int32, .float32])
        return selectMerge(
            blk, template: [("CAP", cap), ("BLOCKS", blocks)],
            grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, 2], [rows, 2], [rows], [rows, 2]],
            outputDTypes: [.int32, .float32, .int32, .float32])
    }


    static let compactBlocks = 16
    static let compactParallelScanSource = selectScanSource
        .replacingOccurrences(of: "const float4 v4 = lq[q];", with: "const float4 v4 = lq[q]; bad = bad || compact_bad[row * uint(NV / 4) + first / 4u + uint(q)] != 0;")
        .replacingOccurrences(of: "if (local.count == 2 && !(v > local.second_value)) continue;", with: "if (local.count == 2 && !hadamard_top2_better(v, uint(compact_ids[row * uint(NV) + first + uint(q) * 4u + uint(i)]), local.second_value, local.second_id)) continue;")
        .replacingOccurrences(of: "hadamard_top2_insert(local, v, first + uint(q) * 4u + uint(i));", with: "hadamard_top2_insert(local, v, uint(compact_ids[row * uint(NV) + first + uint(q) * 4u + uint(i)]));")

    static let compactParallelAnchorScanSource: String = {
        let merge = """
            if (lane == 0) {
              int b = scratch_bad[0];
              for (uint s = 1; s < THREADS / 32; s++) {
                hadamard_top2_merge(local, scratch[s]);
                b |= scratch_bad[s];
              }
              const uint p = row * uint(BLOCKS) + block;
              part_ids[p * 2] = int(local.first_id);
              part_ids[p * 2 + 1] = int(local.second_id);
              part_values[p * 2] = local.first_value;
              part_values[p * 2 + 1] = local.second_value;
              part_meta[p] = b | int(local.count << 1);
            }
            """
        let scoredMerge = """
            if (simd_gid == 0) {
              int anchor_bad = 0;
              if (simd_lid == 0) {
                int b = scratch_bad[0];
                for (uint s = 1; s < THREADS / 32; s++) {
                  hadamard_top2_merge(local, scratch[s]);
                  b |= scratch_bad[s];
                }
                const uint p = row * uint(BLOCKS) + block;
                part_ids[p * 2] = int(local.first_id);
                part_ids[p * 2 + 1] = int(local.second_id);
                part_values[p * 2] = local.first_value;
                part_values[p * 2 + 1] = local.second_value;
                part_meta[p] = b | int(local.count << 1);
                anchor_bad = b;
              }
              anchor_bad = simd_shuffle(anchor_bad, ushort(0));
              const uint id0 = simd_shuffle(local.first_id, ushort(0));
              const uint id1 = simd_shuffle(local.second_id, ushort(0));
              const uint span = uint(w_shape[0]) / uint(BLOCKS);
              const uint begin = block * span;
              bool anchor_ok = anchor_bad == 0 && simd_shuffle(local.count, ushort(0)) == 2u
                  && id0 != id1 && id0 >= begin && id0 < begin + span && id1 >= begin && id1 < begin + span
                  && isfinite(simd_shuffle(local.first_value, ushort(0))) && isfinite(simd_shuffle(local.second_value, ushort(0)));
              float exact0 = as_type<float>(0x7fc00000u), exact1 = exact0;
              if (anchor_ok) {
                for (int anchor = 0; anchor < 2; anchor++) {
                  const int out_row = int(anchor == 0 ? id0 : id1);
                  constexpr int rows = 1;
                  constexpr int outputs = 1;
            """ + Float16ConstantQMV.rowsAccumulate.replacingOccurrences(
                of: "(x + simd_lid", with: "(x + size_t(row) * size_t(K) + simd_lid"
            ).replacingOccurrences(
                of: "b[r] = float(bi[r * in_g]);", with: "b[r] = NEG ? -s : float(bi[r * in_g]);"
            ) + """
                  const float exact_value = simd_sum(result[0][0]);
                  if (anchor == 0) exact0 = exact_value; else exact1 = exact_value;
                  anchor_ok = anchor_ok && isfinite(exact_value);
                }
              }
              if (simd_lid == 0) {
                const uint p = row * uint(BLOCKS) + block;
                part_exact[p * 2] = anchor_ok ? exact0 : as_type<float>(0x7fc00000u);
                part_exact[p * 2 + 1] = anchor_ok ? exact1 : as_type<float>(0x7fc00000u);
              }
            }
            """
        precondition(compactParallelScanSource.components(separatedBy: merge).count == 2)
        return compactParallelScanSource
            .replacingOccurrences(of: "const uint block = threadgroup_position_in_grid.x;", with: "const int K = x_shape[1]; const int in_w = K / 4; const int in_g = K / 128;\nconst uint block = threadgroup_position_in_grid.x;")
            .replacingOccurrences(of: merge, with: scoredMerge)
    }()

    static let compactParallelRescoreSource = selectRescoreSource
        .replacingOccurrences(of: "e *= 1.001f;", with: "e *= 1.001f; if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);")
        .replacingOccurrences(of: "threadgroup atomic_int cand_count;", with: "threadgroup atomic_int cand_count; threadgroup atomic_int omitted_overflow;")
        .replacingOccurrences(of: "if (lane == 0) atomic_store_explicit(&cand_count, 0, memory_order_relaxed);", with: "if (lane == 0) { atomic_store_explicit(&cand_count, 0, memory_order_relaxed); atomic_store_explicit(&omitted_overflow, 0, memory_order_relaxed); }")
        .replacingOccurrences(of: "const float4 v4 = lq[q];", with: "const float4 v4 = lq[q]; if (omitted[row * uint(NV / 4) + first / 4u + uint(q)] >= tau) atomic_store_explicit(&omitted_overflow, 1, memory_order_relaxed);")
        .replacingOccurrences(of: "if (slot < CAP) cand_ids[slot] = int(first + uint(q) * 4u + uint(i));", with: "if (slot < CAP) cand_ids[slot] = compact_ids[row * uint(NV) + first + uint(q) * 4u + uint(i)];")
        .replacingOccurrences(of: "blk_meta[p * 4 + 3] = shared_bad;", with: "blk_meta[p * 4 + 3] = shared_bad | (atomic_load_explicit(&omitted_overflow, memory_order_relaxed) != 0 ? 2 : 0);")

    static let compactParallelAnchorRescoreSource: String = {
        let anchorDots = """
              const uint anchor_id0 = simd_shuffle(local.first_id, ushort(0));
              const uint anchor_id1 = simd_shuffle(local.second_id, ushort(0));
              bool anchor_ok = pbad == 0 && simd_shuffle(local.count, ushort(0)) >= 2u
                  && anchor_id0 != anchor_id1 && anchor_id0 < uint(w_shape[0]) && anchor_id1 < uint(w_shape[0])
                  && isfinite(simd_shuffle(local.first_value, ushort(0)))
                  && isfinite(simd_shuffle(local.second_value, ushort(0)));
              float anchor_floor = INFINITY;
              if (simd_lid == 0) { shared_anchor_ids[0] = -1; shared_anchor_ids[1] = -1; }
              if (anchor_ok) {
                for (int anchor = 0; anchor < 2; anchor++) {
                  const int out_row = int(anchor == 0 ? anchor_id0 : anchor_id1);
                  constexpr int rows = 1;
                  constexpr int outputs = 1;
            """ + Float16ConstantQMV.rowsAccumulate.replacingOccurrences(
                of: "(x + simd_lid", with: "(x + size_t(row) * size_t(K) + simd_lid"
            ).replacingOccurrences(
                of: "b[r] = float(bi[r * in_g]);", with: "b[r] = NEG ? -s : float(bi[r * in_g]);"
            ) + """
                  const float anchor_value = simd_sum(result[0][0]);
                  if (simd_lid == 0) shared_anchor_values[anchor] = anchor_value;
                  anchor_ok = anchor_ok && isfinite(anchor_value);
                  anchor_floor = min(anchor_floor, anchor_value);
                }
                if (simd_lid == 0 && anchor_ok) {
                  const uint anchor_span = uint(w_shape[0]) / uint(BLOCKS);
                  if (anchor_id0 / anchor_span == block) shared_anchor_ids[0] = int(anchor_id0);
                  if (anchor_id1 / anchor_span == block) {
                    if (shared_anchor_ids[0] >= 0) shared_anchor_ids[1] = int(anchor_id1);
                    else {
                      shared_anchor_ids[0] = int(anchor_id1);
                      shared_anchor_values[0] = shared_anchor_values[1];
                    }
                  }
                }
              }
            """
        let originalTau = "const float t = local.second_value - 2.0f * e;"
        let anchoredTau = """
                const float base_t = local.second_value - 2.0f * e;
                float t = base_t;
                if (anchor_ok && isfinite(e) && isfinite(base_t)) {
                  const float anchor_t = anchor_floor - e;
                  if (isfinite(anchor_t)) t = max(t, anchor_t);
                }
            """
        precondition(compactParallelRescoreSource.components(separatedBy: "pbad = simd_max(pbad);").count == 2)
        precondition(compactParallelRescoreSource.components(separatedBy: originalTau).count == 2)
        return compactParallelRescoreSource
            .replacingOccurrences(of: "threadgroup float shared_tau[2];", with: "threadgroup float shared_tau[2]; threadgroup int shared_anchor_ids[2]; threadgroup float shared_anchor_values[2];")
            .replacingOccurrences(of: "const int out_row = cand_ids[c];", with: """
              const int out_row = cand_ids[c];
              const int anchor0 = shared_anchor_ids[0];
              const int anchor = anchor0 < 0 ? -1 : (out_row == anchor0 ? 0 : (out_row == shared_anchor_ids[1] ? 1 : -1));
              if (anchor >= 0) {
                if (simd_lid == 0) cand_values[c] = shared_anchor_values[anchor];
                continue;
              }
            """)
            .replacingOccurrences(of: "pbad = simd_max(pbad);", with: "pbad = simd_max(pbad);\n" + anchorDots)
            .replacingOccurrences(of: originalTau, with: anchoredTau)
    }()

    static let compactParallelPartitionRescoreSource = compactParallelAnchorRescoreSource
        .replacingOccurrences(of: "if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);", with: "float local_e = e; if (USE_L2) e = head_l2_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e);")
        .replacingOccurrences(of: "const float anchor_t = anchor_floor - e;", with: "const float anchor_t = anchor_floor - local_e;")

    static let compactParallelDistributedAnchorRescoreSource: String = {
        let partial = """
                const uint anchor_span = uint(w_shape[0]) / uint(BLOCKS);
                const uint anchor_begin = pb * anchor_span;
                const float f0 = part_exact[p * 2], f1 = part_exact[p * 2 + 1];
                const bool valid = (meta & 1) == 0 && o.count == 2u && o.first_id != o.second_id
                    && o.first_id >= anchor_begin && o.first_id < anchor_begin + anchor_span
                    && o.second_id >= anchor_begin && o.second_id < anchor_begin + anchor_span
                    && isfinite(f0) && isfinite(f1);
                distributed_bad |= valid ? 0 : 1;
                if (valid) {
                  hadamard_top2_insert(distributed, f0, o.first_id);
                  hadamard_top2_insert(distributed, f1, o.second_id);
                }
            """
        let reduce = """
                hadamard_top2 distributed_other;
                distributed_other.first_value = simd_shuffle_down(distributed.first_value, stride);
                distributed_other.second_value = simd_shuffle_down(distributed.second_value, stride);
                distributed_other.first_id = simd_shuffle_down(distributed.first_id, stride);
                distributed_other.second_id = simd_shuffle_down(distributed.second_id, stride);
                distributed_other.count = simd_shuffle_down(distributed.count, stride);
                if (simd_lid < stride) hadamard_top2_merge(distributed, distributed_other);
            """
        let cached = """
              const bool distributed_ok = anchor_ok && distributed_bad == 0
                  && simd_shuffle(distributed.count, ushort(0)) == 2u;
              if (distributed_ok) {
                anchor_floor = simd_shuffle(distributed.second_value, ushort(0));
                if (simd_lid == 0) {
                  const uint p = row * uint(BLOCKS) + block;
                  shared_anchor_ids[0] = part_ids[p * 2];
                  shared_anchor_ids[1] = part_ids[p * 2 + 1];
                  shared_anchor_values[0] = part_exact[p * 2];
                  shared_anchor_values[1] = part_exact[p * 2 + 1];
                }
              } else if (anchor_ok) {
            """
        return compactParallelAnchorRescoreSource
            .replacingOccurrences(of: "int pbad = 0;", with: "int pbad = 0; hadamard_top2 distributed = hadamard_top2_empty(); int distributed_bad = 0;")
            .replacingOccurrences(of: "pbad |= meta & 1;", with: "pbad |= meta & 1;\n" + partial)
            .replacingOccurrences(of: "if (simd_lid < stride) hadamard_top2_merge(local, other);", with: "if (simd_lid < stride) hadamard_top2_merge(local, other);\n" + reduce)
            .replacingOccurrences(of: "pbad = simd_max(pbad);", with: "pbad = simd_max(pbad); distributed_bad = simd_max(distributed_bad);")
            .replacingOccurrences(of: "if (anchor_ok) {", with: cached)
    }()

    static let compactParallelDistributedPartitionRescoreSource = compactParallelDistributedAnchorRescoreSource
        .replacingOccurrences(of: "if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);", with: "float local_e = e; if (USE_L2) e = head_l2_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e);")
        .replacingOccurrences(of: "const float anchor_t = anchor_floor - e;", with: "const float anchor_t = anchor_floor - local_e;")

    static let compactParallelSingleAnchorScanSource = compactParallelAnchorScanSource
        .replacingOccurrences(of: "float exact0 = as_type<float>(0x7fc00000u), exact1 = exact0;", with: "float exact0 = as_type<float>(0x7fc00000u);")
        .replacingOccurrences(of: "for (int anchor = 0; anchor < 2; anchor++)", with: "for (int anchor = 0; anchor < 1; anchor++)")
        .replacingOccurrences(of: "const int out_row = int(anchor == 0 ? id0 : id1);", with: "const int out_row = int(id0);")
        .replacingOccurrences(of: "if (anchor == 0) exact0 = exact_value; else exact1 = exact_value;", with: "exact0 = exact_value;")
        .replacingOccurrences(of: "part_exact[p * 2] = anchor_ok ? exact0 : as_type<float>(0x7fc00000u);", with: "part_exact[p] = anchor_ok ? exact0 : as_type<float>(0x7fc00000u);")
        .replacingOccurrences(of: "part_exact[p * 2 + 1] = anchor_ok ? exact1 : as_type<float>(0x7fc00000u);", with: "")

    static let compactParallelSingleAnchorRescoreSource = compactParallelDistributedAnchorRescoreSource
        .replacingOccurrences(of: "const float f0 = part_exact[p * 2], f1 = part_exact[p * 2 + 1];", with: "const float f0 = part_exact[p];")
        .replacingOccurrences(of: "&& isfinite(f0) && isfinite(f1);", with: "&& isfinite(f0);")
        .replacingOccurrences(of: "hadamard_top2_insert(distributed, f1, o.second_id);", with: "")
        .replacingOccurrences(of: "const bool distributed_ok = anchor_ok && distributed_bad == 0", with: "const bool distributed_ok = anchor_ok && distributed_bad == 0 && anchor_id0 / (uint(w_shape[0]) / uint(BLOCKS)) != anchor_id1 / (uint(w_shape[0]) / uint(BLOCKS))")
        .replacingOccurrences(of: "shared_anchor_ids[1] = part_ids[p * 2 + 1];", with: "")
        .replacingOccurrences(of: "shared_anchor_values[0] = part_exact[p * 2];", with: "shared_anchor_values[0] = part_exact[p];")
        .replacingOccurrences(of: "shared_anchor_values[1] = part_exact[p * 2 + 1];", with: "")

    static let compactParallelSinglePartitionRescoreSource = compactParallelSingleAnchorRescoreSource
        .replacingOccurrences(of: "if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);", with: "float local_e = e; if (USE_L2) e = head_l2_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e);")
        .replacingOccurrences(of: "const float anchor_t = anchor_floor - e;", with: "const float anchor_t = anchor_floor - local_e;")

    static let compactParallelJointPartitionRescoreSource = compactParallelSinglePartitionRescoreSource
        .replacingOccurrences(of: "head_l2_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e)", with: "head_l2_joint_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e, joint_c2)")
    static let compactParallelJointPartitionRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_joint_k4_partition16_v1",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2", "part_exact", "joint_c2"],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: compactParallelJointPartitionRescoreSource, header: Float16ConstantQMV.topTwoHeader + jointL2Header)

    static let compactParallelSingleAnchorScan = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_single_anchor_scan16_v1",
        inputNames: ["logits", "compact_ids", "compact_bad", "x", "w", "scales", "biases"],
        outputNames: ["part_ids", "part_values", "part_meta", "part_exact"],
        source: compactParallelSingleAnchorScanSource, header: Float16ConstantQMV.topTwoHeader)
    static let compactParallelSingleAnchorRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_single_anchor16_v1",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2", "part_exact"],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: compactParallelSingleAnchorRescoreSource, header: Float16ConstantQMV.topTwoHeader + l2Header)
    static let compactParallelSinglePartitionRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_single_partition16_v1",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2", "part_exact"],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: compactParallelSinglePartitionRescoreSource, header: Float16ConstantQMV.topTwoHeader + partitionL2Header)

    static let compactParallelAnchorScan = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_anchor_scan16_v1",
        inputNames: ["logits", "compact_ids", "compact_bad", "x", "w", "scales", "biases"],
        outputNames: ["part_ids", "part_values", "part_meta", "part_exact"],
        source: compactParallelAnchorScanSource, header: Float16ConstantQMV.topTwoHeader)
    static let compactParallelDistributedAnchorRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_distributed_anchors16_v1",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2", "part_exact"],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: compactParallelDistributedAnchorRescoreSource, header: Float16ConstantQMV.topTwoHeader + l2Header)
    static let compactParallelDistributedPartitionRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_distributed_partition16_v1",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2", "part_exact"],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: compactParallelDistributedPartitionRescoreSource, header: Float16ConstantQMV.topTwoHeader + partitionL2Header)

    static let compactParallelScan = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_scan16", inputNames: ["logits", "compact_ids", "compact_bad"],
        outputNames: ["part_ids", "part_values", "part_meta"],
        source: compactParallelScanSource, header: Float16ConstantQMV.topTwoHeader)
    static let compactParallelRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_rescore16_l2_v1",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2"],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: compactParallelRescoreSource, header: Float16ConstantQMV.topTwoHeader + l2Header)
    static let compactParallelAnchorRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_rescore16_l2_anchors_v1",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2"],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: compactParallelAnchorRescoreSource, header: Float16ConstantQMV.topTwoHeader + l2Header)
    static let compactParallelPartitionRescore = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_rescore16_partition_l2_anchors_v1",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2"],
        outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        source: compactParallelPartitionRescoreSource, header: Float16ConstantQMV.topTwoHeader + partitionL2Header)
    static let compactParallelMerge = MLXFast.metalKernel(
        name: "hadamard_xhead_compact_parallel_merge16", inputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
        outputNames: ["sel_ids", "sel_values", "flag", "info"],
        source: selectMergeSource.replacingOccurrences(of: "info[row * 2 + 1] = float(count);", with: "info[row * 2 + 1] = (bad & 2) != 0 ? -1.0f : float(count);"),
        header: Float16ConstantQMV.topTwoHeader)

    static func compactParallelSelect(
        summary: (ids: MLXArray, values: MLXArray, omitted: MLXArray, bad: MLXArray),
        ascale: MLXArray, bounds: MLXArray, x: MLXArray,
        weight: MLXArray, scales: MLXArray, biases: MLXArray, rows: Int, negative: Bool, error2: MLXArray? = nil, anchors: Bool = false, partitionL2: Bool = false, distributedAnchors: Bool = false, singleLocalAnchors: Bool = false, jointC2: MLXArray? = nil
    ) -> [MLXArray] {
        let values = summary.values.reshaped([rows, -1])
        let ids = summary.ids.reshaped([rows, -1])
        let n = values.dim(1)
        let b = compactBlocks
        let distributed = (distributedAnchors || singleLocalAnchors) && anchors && rows == 16 && b == 16
            && x.dim(1) == 4096 && scales.dim(1) == 32 && weight.dim(0) % 512 == 0 && n == weight.dim(0) / 8
        let single = singleLocalAnchors && distributed
        let parts = (single ? compactParallelSingleAnchorScan : (distributed ? compactParallelAnchorScan : compactParallelScan))(
            [values, ids, summary.bad] + (distributed ? [x, weight, scales, biases] : []),
            template: [("THREADS", blockThreads), ("BLOCKS", b), ("NV", n)] + (distributed ? [("NEG", negative ? 1 : 0)] : []),
            grid: (b * blockThreads, rows, 1), threadGroup: (blockThreads, 1, 1),
            outputShapes: [[rows, b, 2], [rows, b, 2], [rows, b]] + (distributed ? (single ? [[rows, b]] : [[rows, b, 2]]) : []),
            outputDTypes: [.int32, .float32, .int32] + (distributed ? [.float32] : []))
        let local = partitionL2 && anchors && error2 != nil && rows == 16 && b == 16
            && x.dim(1) == 4096 && scales.dim(1) == 32 && weight.dim(0) % 512 == 0
            && n == weight.dim(0) / 8 && bounds.size == 19 * 32
        let joint = single && local && jointC2?.size == 64
        let kernel = joint ? compactParallelJointPartitionRescore
            : single ? (local ? compactParallelSinglePartitionRescore : compactParallelSingleAnchorRescore)
            : distributed ? (local ? compactParallelDistributedPartitionRescore : compactParallelDistributedAnchorRescore)
            : (local ? compactParallelPartitionRescore : (anchors ? compactParallelAnchorRescore : compactParallelRescore))
        var inputs: [MLXArray] = [values, ascale, bounds, x, weight, scales, biases, parts[0], parts[1], parts[2], ids, summary.omitted, error2 ?? ascale]
        if distributed { inputs.append(parts[3]) }
        if joint { inputs.append(jointC2!) }
        let blk = kernel(
            inputs,
            template: [("CAP", cap), ("THREADS", blockThreads), ("BLOCKS", b), ("NV", n), ("NEG", negative ? 1 : 0), ("USE_L2", error2 != nil)],
            grid: (b * blockThreads, rows, 1), threadGroup: (blockThreads, 1, 1),
            outputShapes: [[rows, b, 2], [rows, b, 2], [rows, b, 4], [rows]], outputDTypes: [.int32, .float32, .int32, .float32])
        return compactParallelMerge(
            blk, template: [("CAP", cap), ("BLOCKS", b)],
            grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, 2], [rows, 2], [rows], [rows, 2]], outputDTypes: [.int32, .float32, .int32, .float32])
    }

    static func runCompactParallel(
        rotated: MLXArray, summary: (ids: MLXArray, values: MLXArray, omitted: MLXArray, bad: MLXArray),
        ascale: MLXArray, bounds: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        rows: Int, negative: Bool, error2: MLXArray? = nil, anchors: Bool = false, partitionL2: Bool = false, distributedAnchors: Bool = false, singleLocalAnchors: Bool = false, jointC2: MLXArray? = nil
    ) -> (ids: MLXArray, values: MLXArray) {
        let x = rotated.reshaped([rows, rotated.dim(-1)])
        let selected = compactParallelSelect(
            summary: summary, ascale: ascale, bounds: bounds, x: x,
            weight: weight, scales: scales, biases: biases, rows: rows, negative: negative, error2: error2, anchors: anchors, partitionL2: partitionL2, distributedAnchors: distributedAnchors, singleLocalAnchors: singleLocalAnchors, jointC2: jointC2)
        let groups = weight.dim(0) / (8 * Float16ConstantQMV.topTwoBlocks)
        let partials = guardedPartial(
            [x, weight, scales, biases, selected[2]],
            template: [("rows", rows), ("outputs", 4), ("blocks", Float16ConstantQMV.topTwoBlocks), ("groups", groups)],
            grid: (32, groups * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[rows, groups, 2], [rows, groups, 2]], outputDTypes: [.int32, .float32])
        let top = finalize(
            [partials[0], partials[1], selected[2], selected[0], selected[1]],
            grid: (rows * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, 2], [rows, 2]], outputDTypes: [.int32, .float32])
        return (top[0], top[1])
    }

    static let parallelOn: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_XHEAD_PARALLEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    nonisolated(unsafe) static var parallelVerified: [Int: Bool] = [:]

    static func parallelApplies(rows: Int, N: Int) -> Bool {
        parallelOn && parallelVerified[rows] != false && N % (4 * blocks) == 0
    }

    static let rowRange: ClosedRange<Int>? = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_XHEAD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["0", "false", "no", "off"].contains(value ?? "") { return nil }
        let high = 16
        if let value, let low = Int(value), (2 ... 8).contains(low) { return low ... high }
        return 2 ... high
    }()

    static let checkEveryCall: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_XHEAD_CHECK"] == "1"

    nonisolated(unsafe) static var verified: [Int: Bool] = [:]
    nonisolated(unsafe) static var stats: [Int: (calls: Int, fallbacks: Int, maxRatio: Float, maxCount: Float)] = [:]

    static func log(_ message: String) {
        FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
        os_log("%{public}@", log: .default, type: .default, message as NSString)
    }

    static func run(
        rotated: MLXArray, logits: MLXArray, ascale: MLXArray, bounds: MLXArray,
        weight: MLXArray, scales: MLXArray, biases: MLXArray, rows: Int,
        negative: Bool = false, force: Bool = false, parallel: Bool = false
    ) -> (ids: MLXArray, values: MLXArray, flag: MLXArray, info: MLXArray) {
        let K = rotated.dim(-1)
        let N = weight.dim(0)
        let x = rotated.reshaped([rows, K])
        let selected = parallel
            ? parallelSelect(
                logits: logits, ascale: ascale, bounds: bounds, x: x, weight: weight,
                scales: scales, biases: biases, rows: rows, negative: negative)
            : select(
                [logits, ascale, bounds, x, weight, scales, biases],
                template: [("CAP", cap), ("THREADS", threads), ("NEG", negative ? 1 : 0)],
                grid: (rows * threads, 1, 1), threadGroup: (threads, 1, 1),
                outputShapes: [[rows, 2], [rows, 2], [rows], [rows, 2]],
                outputDTypes: [.int32, .float32, .int32, .float32])
        let flag = force ? MLXArray(Array(repeating: Int32(1), count: rows)) : selected[2]
        let outputs = 4
        let span = 2 * outputs * Float16ConstantQMV.topTwoBlocks
        let groups = N / span
        let partials = guardedPartial(
            [x, weight, scales, biases, flag],
            template: [
                ("rows", rows), ("outputs", outputs), ("blocks", Float16ConstantQMV.topTwoBlocks),
                ("groups", groups),
            ],
            grid: (32, groups * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[rows, groups, 2], [rows, groups, 2]],
            outputDTypes: [.int32, .float32])
        let top = finalize(
            [partials[0], partials[1], flag, selected[0], selected[1]],
            grid: (rows * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, 2], [rows, 2]], outputDTypes: [.int32, .float32])
        return (top[0], top[1], selected[2], selected[3])
    }

    static func sameBits(_ a: (ids: MLXArray, values: MLXArray), _ b: (ids: MLXArray, values: MLXArray)) -> Bool {
        let ai = a.ids.asArray(Int32.self), bi = b.ids.asArray(Int32.self)
        let av = a.values.asArray(Float.self).map(\.bitPattern)
        let bv = b.values.asArray(Float.self).map(\.bitPattern)
        return ai == bi && av == bv
    }
}

final class ExactVerifyHeadBounds {
    var compactVerified: Bool?
    var compactParallelVerified: Bool?
    var residualL2Verified: Bool?
    var exactAnchorsVerified: Bool?
    var partitionL2Verified: Bool?
    var distributedAnchorsVerified: Bool?
    var singleLocalAnchorsVerified: Bool?
    var jointBoundsVerified: Bool?
    private var jointCached: MLXArray?
    private var cached: MLXArray?

    func getJoint(weight: MLXArray, scales: MLXArray, biases: MLXArray) -> MLXArray? {
        guard weight.dim(1) == 256, scales.dim(1) == 32, weight.dim(0) % 512 == 0 else { return nil }
        if let jointCached { return jointCached }
        let value = ExactVerifyHead.jointBounds(
            [weight, scales, biases], grid: (64 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[16, 4]], outputDTypes: [.float32])[0]
        eval(value)
        jointCached = value
        return value
    }

    func get(weight: MLXArray, scales: MLXArray, biases: MLXArray) -> MLXArray {
        if let cached { return cached }
        let groups = scales.dim(1)
        let partitioned = groups == 32 && weight.dim(0) % 512 == 0
        let value = (partitioned ? ExactVerifyHead.partitionBounds : ExactVerifyHead.bounds)(
            [weight, scales, biases], grid: (groups * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[(partitioned ? 19 : 3) * groups]], outputDTypes: [.float32])[0]
        eval(value)
        cached = value
        return value
    }
}

private enum DraftScreenQMV {
    static let partial = MLXFast.metalKernel(
        name: "hadamard_qmv_screen_f16_constants",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const int K = x_shape[1];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int out_row = tid.y * 8 + simd_gid * 4;
            const device uint8_t* ws = (const device uint8_t*)w + out_row * in_w + simd_lid * 4;
            const device half* sc = scales + out_row * in_g + simd_lid / 8;
            const device half* bi = biases + out_row * in_g + simd_lid / 8;
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float x_thread[16];
            float result[4] = {0, 0, 0, 0};
            for (int k = 0; k < P; k += 512) {
              float sum = 0;
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += xv.x + xv.y + xv.z + xv.w;
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y * 0.25f;
                x_thread[4 * i + 2] = xv.z * 0.0625f;
                x_thread[4 * i + 3] = xv.w * 0.015625f;
              }
              for (int r = 0; r < 4; r++) {
                const uint packed_w = *(const device uint*)(ws + r * in_w);
                const float s = float(sc[r * in_g]);
                const float b = NEG ? -s : float(bi[r * in_g]);
                float accum = 0;
                for (int i = 0; i < 4; i++) {
                  const uint8_t w_byte = (packed_w >> (8 * i)) & 0xFF;
                  accum += (x_thread[4 * i] * (w_byte & 0x03) + x_thread[4 * i + 1] * (w_byte & 0x0c)
                      + x_thread[4 * i + 2] * (w_byte & 0x30) + x_thread[4 * i + 3] * (w_byte & 0xc0));
                }
                result[r] += s * accum + sum * b;
              }
              ws += 128;
              sc += 4;
              bi += 4;
              xp4 += 128;
            }
            for (int r = 0; r < 4; r++) {
              const float total = simd_sum(result[r]);
              if (simd_lid == 0) y[out_row + r] = total;
            }
            """)

    static let rescore = MLXFast.metalKernel(
        name: "hadamard_qmv_screen_rescore_f16_constants",
        inputNames: ["x", "w", "scales", "biases", "pick"], outputNames: ["y", "row"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const int K = x_shape[1];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int c = threadgroup_position_in_grid.y * 2 + simd_gid;
            const int r = c * CH + int(pick[c]);
            const device uint8_t* ws = (const device uint8_t*)w + r * in_w + simd_lid * 4;
            const device half* sc = scales + r * in_g + simd_lid / 8;
            const device half* bi = biases + r * in_g + simd_lid / 8;
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float result = 0;
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              float x_thread[16];
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += xv.x + xv.y + xv.z + xv.w;
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y * 0.25f;
                x_thread[4 * i + 2] = xv.z * 0.0625f;
                x_thread[4 * i + 3] = xv.w * 0.015625f;
              }
              const uint packed_w = *(const device uint*)ws;
              float accum = 0;
              for (int i = 0; i < 4; i++) {
                const uint8_t w_byte = (packed_w >> (8 * i)) & 0xFF;
                accum += (x_thread[4 * i] * (w_byte & 0x03) + x_thread[4 * i + 1] * (w_byte & 0x0c)
                    + x_thread[4 * i + 2] * (w_byte & 0x30) + x_thread[4 * i + 3] * (w_byte & 0xc0));
              }
              const float s = float(*sc);
              const float b = NEG ? -s : float(*bi);
              result += s * accum + sum * b;
              ws += 128;
              sc += 4;
              bi += 4;
              xp4 += 128;
            }
            const float total = simd_sum(result);
            if (simd_lid == 0) {
              y[c] = total;
              row[c] = r;
            }
            """)

    static let rescoreFast32 = MLXFast.metalKernel(
        name: "hadamard_qmv_screen_rescore_fast32_f16",
        inputNames: ["x", "w", "scales", "partial"], outputNames: ["y", "row"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const int K = x_shape[1];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int c = threadgroup_position_in_grid.y * 2 + simd_gid;
            float best_v = partial[c * 32 + int(simd_lid)];
            int best_i = int(simd_lid);
            for (ushort offset = 16; offset > 0; offset >>= 1) {
              const float other_v = simd_shuffle_xor(best_v, offset);
              const int other_i = simd_shuffle_xor(best_i, offset);
              if (other_v > best_v || (other_v == best_v && other_i < best_i)) {
                best_v = other_v;
                best_i = other_i;
              }
            }
            const int r = c * 32 + best_i;
            const device uint8_t* ws = (const device uint8_t*)w + r * in_w + simd_lid * 4;
            const device half* sc = scales + r * in_g + simd_lid / 8;
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float result = 0;
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              float x_thread[16];
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += xv.x + xv.y + xv.z + xv.w;
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y * 0.25f;
                x_thread[4 * i + 2] = xv.z * 0.0625f;
                x_thread[4 * i + 3] = xv.w * 0.015625f;
              }
              const uint packed_w = *(const device uint*)ws;
              float accum = 0;
              for (int i = 0; i < 4; i++) {
                const uint8_t w_byte = (packed_w >> (8 * i)) & 0xFF;
                accum += (x_thread[4 * i] * (w_byte & 0x03) + x_thread[4 * i + 1] * (w_byte & 0x0c)
                    + x_thread[4 * i + 2] * (w_byte & 0x30) + x_thread[4 * i + 3] * (w_byte & 0xc0));
              }
              const float s = float(*sc);
              result += s * accum + sum * (-s);
              ws += 128;
              sc += 4;
              xp4 += 128;
            }
            const float total = simd_sum(result);
            if (simd_lid == 0) {
              y[c] = total;
              row[c] = r;
            }
            """)

    static let rescoreFastChunk = MLXFast.metalKernel(
        name: "hadamard_qmv_screen_rescore_fast_chunk_f16",
        inputNames: ["x", "w", "scales", "partial"], outputNames: ["y", "row"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const int K = x_shape[1];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int c = threadgroup_position_in_grid.y * 2 + simd_gid;
            float best_v = int(simd_lid) < CH ? partial[c * CH + int(simd_lid)] : -INFINITY;
            int best_i = int(simd_lid);
            for (ushort offset = 16; offset > 0; offset >>= 1) {
              const float other_v = simd_shuffle_xor(best_v, offset);
              const int other_i = simd_shuffle_xor(best_i, offset);
              if (other_v > best_v || (other_v == best_v && other_i < best_i)) {
                best_v = other_v;
                best_i = other_i;
              }
            }
            const int r = c * CH + best_i;
            const device uint8_t* ws = (const device uint8_t*)w + r * in_w + simd_lid * 4;
            const device half* sc = scales + r * in_g + simd_lid / 8;
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float result = 0;
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              float x_thread[16];
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += xv.x + xv.y + xv.z + xv.w;
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y * 0.25f;
                x_thread[4 * i + 2] = xv.z * 0.0625f;
                x_thread[4 * i + 3] = xv.w * 0.015625f;
              }
              const uint packed_w = *(const device uint*)ws;
              float accum = 0;
              for (int i = 0; i < 4; i++) {
                const uint8_t w_byte = (packed_w >> (8 * i)) & 0xFF;
                accum += (x_thread[4 * i] * (w_byte & 0x03) + x_thread[4 * i + 1] * (w_byte & 0x0c)
                    + x_thread[4 * i + 2] * (w_byte & 0x30) + x_thread[4 * i + 3] * (w_byte & 0xc0));
              }
              const float s = float(*sc);
              result += s * accum + sum * (-s);
              ws += 128;
              sc += 4;
              xp4 += 128;
            }
            const float total = simd_sum(result);
            if (simd_lid == 0) {
              y[c] = total;
              row[c] = r;
            }
            """)

    static let finalPick = MLXFast.metalKernel(
        name: "hadamard_qmv_screen_final_pick",
        inputNames: ["y", "row", "ids"], outputNames: ["out"],
        source: """
            const uint lane = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            float best_v = -INFINITY;
            int best_c = 0x7fffffff;
            for (int c = int(lane); c < BLOCKS; c += 256) {
              const float v = y[c];
              if (v > best_v || (v == best_v && c < best_c)) {
                best_v = v;
                best_c = c;
              }
            }
            for (ushort offset = 16; offset > 0; offset >>= 1) {
              const float other_v = simd_shuffle_xor(best_v, offset);
              const int other_c = simd_shuffle_xor(best_c, offset);
              if (other_v > best_v || (other_v == best_v && other_c < best_c)) {
                best_v = other_v;
                best_c = other_c;
              }
            }
            threadgroup float shared_v[8];
            threadgroup int shared_c[8];
            if (simd_lid == 0) {
              shared_v[simd_gid] = best_v;
              shared_c[simd_gid] = best_c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              float fv = shared_v[0];
              int fc = shared_c[0];
              for (int s = 1; s < 8; s++) {
                const float sv = shared_v[s];
                const int sc_idx = shared_c[s];
                if (sv > fv || (sv == fv && sc_idx < fc)) {
                  fv = sv;
                  fc = sc_idx;
                }
              }
              const int win_row = row[fc];
              out[0] = USE_IDS ? ids[win_row] : win_row;
            }
            """)
}

struct PromptInt8Activation {
    let codes: MLXArray
    let ascale: MLXArray
    let rsb: MLXArray
    let rows: Int
    let k: Int
}

enum PromptInt8Matmul {
    enum Route { case nax, plain, off }

    static let constantIndexSource = """
        inline size_t bonsai_q8_cidx(int m, int g, int Kg) {
          const uint ml = uint(m) & 63u;
          return size_t(m >> 6) * size_t(Kg) * 64 + size_t(g) * 64
              + size_t(((ml >> 4) & 1u) * 32u + (ml & 7u) * 4u + ((ml >> 5) & 1u) * 2u + ((ml >> 3) & 1u));
        }

        """

    // 32 x 32 outputs per threadgroup of four simdgroups (2 x 2, 16 x 16
    // each). Per 128-group: the codes and the unpacked 2-bit weights are
    // staged as FP32 in threadgroup memory 64 k at a time and multiplied on
    // FP32 simdgroup matrices (exact, see above); the epilogue reads each
    // lane's two elements per fragment in place (MLX steel's fragment
    // layout, `BaseMMAFrag::get_coord`). The weight codes are unpacked in
    // the staged order the activation codes use.
    private static let plainSource = """
        const int K = ksz[0]; const int N = ksz[2];
        const int Kg = K / 128;
        const int KW = K / 16;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const int m0 = int(threadgroup_position_in_grid.y) * 32;
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int sm = int(sg >> 1) * 16;
        const int sn = int(sg & 1) * 16;
        const int qid = int(lane >> 2);
        const int fm = (qid & 4) + int((lane >> 1) & 3);
        const int fn = (qid & 2) * 2 + int(lane & 1) * 2;
        threadgroup float As[32 * 68];
        threadgroup float Bs[32 * 68];
        float acc[2][2][2];
        #pragma clang loop unroll(full)
        for (int a = 0; a < 2; a++) {
          #pragma clang loop unroll(full)
          for (int b = 0; b < 2; b++) { acc[a][b][0] = 0.0f; acc[a][b][1] = 0.0f; }
        }
        const int arow = int(tid >> 2);
        const int acol = int(tid & 3) * 16;
        const device char* asrc = (const device char*)xq + (size_t)(m0 + arow) * K + acol;
        const int bcol = int(tid >> 2);
        const int bword = int(tid & 3);
        const device uint32_t* bsrc = w + (size_t)(n0 + bcol) * KW + bword;
        for (int g = 0; g < Kg; g++) {
          simdgroup_float8x8 c[2][2];
          #pragma clang loop unroll(full)
          for (int a = 0; a < 2; a++) {
            #pragma clang loop unroll(full)
            for (int b = 0; b < 2; b++) { c[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
          }
          for (int kb = 0; kb < 2; kb++) {
            const int k0 = g * 128 + kb * 64;
            threadgroup float* adst = As + arow * 68 + acol;
            #pragma clang loop unroll(full)
            for (int q = 0; q < 4; q++) {
              const char4 av = *(const device char4*)(asrc + k0 + 4 * q);
              adst[4 * q + 0] = float(av.x); adst[4 * q + 1] = float(av.y);
              adst[4 * q + 2] = float(av.z); adst[4 * q + 3] = float(av.w);
            }
            const uint wv = bsrc[k0 / 16];
            threadgroup float* bdst = Bs + bcol * 68 + bword * 16;
            #pragma clang loop unroll(full)
            for (int p = 0; p < 16; p++) {
              bdst[p] = float((wv >> uint(8 * (p & 3) + 2 * (p >> 2))) & 3u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (int kk = 0; kk < 64; kk += 8) {
              simdgroup_float8x8 am[2], bm[2];
              simdgroup_load(am[0], As + sm * 68 + kk, 68);
              simdgroup_load(am[1], As + (sm + 8) * 68 + kk, 68);
              simdgroup_load(bm[0], Bs + sn * 68 + kk, 68, ulong2(0, 0), true);
              simdgroup_load(bm[1], Bs + (sn + 8) * 68 + kk, 68, ulong2(0, 0), true);
              #pragma clang loop unroll(full)
              for (int a = 0; a < 2; a++) {
                #pragma clang loop unroll(full)
                for (int b = 0; b < 2; b++) { simdgroup_multiply_accumulate(c[a][b], am[a], bm[b], c[a][b]); }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          float as[2], rb[2];
          #pragma clang loop unroll(full)
          for (int a = 0; a < 2; a++) {
            const size_t ci = bonsai_q8_cidx(m0 + sm + 8 * a + fm, g, Kg);
            as[a] = ascale[ci]; rb[a] = rsb[ci];
          }
          #pragma clang loop unroll(full)
          for (int b = 0; b < 2; b++) {
            const size_t nidx = (size_t)g * N + n0 + sn + 8 * b + fn;
            const float2 s2 = float2(*(const device half2*)(scalesT + nidx));
            const float2 b2 = NEG ? -s2 : float2(*(const device half2*)(biasesT + nidx));
            #pragma clang loop unroll(full)
            for (int a = 0; a < 2; a++) {
              const auto e = c[a][b].thread_elements();
              #pragma clang loop unroll(full)
              for (int j = 0; j < 2; j++) {
                const float t = s2[j] * float(e[j]);
                acc[a][b][j] = fma(b2[j], rb[a], fma(as[a], t, acc[a][b][j]));
              }
            }
          }
        }
        #pragma clang loop unroll(full)
        for (int a = 0; a < 2; a++) {
          #pragma clang loop unroll(full)
          for (int b = 0; b < 2; b++) {
            const size_t o = (size_t)(m0 + sm + 8 * a + fm) * N + n0 + sn + 8 * b + fn;
            out[o] = acc[a][b][0];
            out[o + 1] = acc[a][b][1];
          }
        }
        """

    private static let lastRowKernel = MLXFast.metalKernel(
        name: "bonsai_q8_prompt_matmul_last_row",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: plainSource
            .replacingOccurrences(of: "(m0 + arow)", with: "ksz[3]")
            .replacingOccurrences(of: "bonsai_q8_cidx(m0 + sm + 8 * a + fm,", with: "bonsai_q8_cidx(ksz[3],")
            .replacingOccurrences(of: "out[o] = acc[a][b][0];", with: "if (m0 + sm + 8 * a + fm == 0) { out[o] = acc[a][b][0];")
            .replacingOccurrences(of: "out[o + 1] = acc[a][b][1];", with: "out[o + 1] = acc[a][b][1]; }"),
        header: constantIndexSource,
        ensureRowContiguous: true)

    private static let plainKernel = MLXFast.metalKernel(
        name: "bonsai_q8_prompt_matmul_plain",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: plainSource,
        header: constantIndexSource,
        ensureRowContiguous: true)

    static let naxHeader = """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """

    // The 27b staged8 prompt kernel (b402ddf2, SIGNED = 1, MPERM = 1): 64 x 64
    // outputs per threadgroup of four simdgroups, one 64 x 64 x 128 int8
    // tensor op per group into an int32 cooperative tensor, the 2-bit codes
    // expanded to int8 bytes in threadgroup memory (double-buffered, staged
    // order), the epilogue above per element. grid (N / 64 * 128, M / 64).
    static let naxSource = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, M));
        threadgroup uint32_t bs[2][64 * 128 / 4];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)bs[0], dextents<int, 2>(128, 64));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B1((threadgroup int8_t*)bs[1], dextents<int, 2>(128, 64));
        auto tA0 = A.template slice<128, 64>(0, m0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const int NQ = N / 4;
        const size_t tbase = (size_t)(m0 / 64) * (size_t)Kg * 64 + (size_t)((8 * int(sg >> 1) + fm) * 4);
        const int sc = int(tid >> 1); const int sh = int(tid & 1);
        const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + sh * 4;
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + g * 8);
          const uint4 v = *src;
          threadgroup uint32_t* dst = bs[buf] + sc * 32 + sh * 16;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const uint32_t wv = v[j];
            dst[4 * j + 0] = wv & 0x03030303u;
            dst[4 * j + 1] = (wv >> 2) & 0x03030303u;
            dst[4 * j + 2] = (wv >> 4) & 0x03030303u;
            dst[4 * j + 3] = (wv >> 6) & 0x03030303u;
          }
        };
        stage(0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < Kg; g++) {
          const int cur = g & 1;
          if (g + 1 < Kg) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 64>(g * 128, m0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = float4(bp0[g * NQ]), b1 = float4(bp1[g * NQ]);
          const float4 as4 = *(const device float4*)(ascale + tbase + (size_t)g * 64);
          const float4 rb4 = *(const device float4*)(rsb + tbase + (size_t)g * 64);
          float as[4], rb[4];
          as[0] = as4.x; as[1] = as4.y; as[2] = as4.z; as[3] = as4.w;
          rb[0] = rb4.x; rb[1] = rb4.y; rb[2] = rb4.z; rb[3] = rb4.w;
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float t = s * float(cT[i]);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          const int c = i & 3; const int nh = (i >> 3) & 1;
          const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
          out[(size_t)mm * N + nb + c + 32 * nh] = acc[i];
        }
        """

    // `naxSource` with the same values reached by fewer instructions: each
    // word's four planes go to threadgroup memory as one uint4 (the base is
    // 16-word aligned; the same bytes at the same addresses), the four
    // consecutive outputs i ... i + 3 of one row (c = 0 ... 3) leave as one
    // float4 (nb and N are multiples of 4), and with `NEG = 1` (a module
    // whose FP16 offsets are bit for bit its negated scales, checked at
    // init) the offsets are `-s` instead of a second load, the same FP32
    // value. The epilogue's arithmetic and order are unchanged.
    static let naxVectorSource = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, M));
        threadgroup uint32_t bs[2][64 * 128 / 4];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)bs[0], dextents<int, 2>(128, 64));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B1((threadgroup int8_t*)bs[1], dextents<int, 2>(128, 64));
        auto tA0 = A.template slice<128, 64>(0, m0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const int NQ = N / 4;
        const size_t tbase = (size_t)(m0 / 64) * (size_t)Kg * 64 + (size_t)((8 * int(sg >> 1) + fm) * 4);
        const int sc = int(tid >> 1); const int sh = int(tid & 1);
        const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + sh * 4;
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + g * 8);
          const uint4 v = *src;
          threadgroup uint32_t* dst = bs[buf] + sc * 32 + sh * 16;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const uint32_t wv = v[j];
            *(threadgroup uint4*)(dst + 4 * j) = uint4(
                wv & 0x03030303u, (wv >> 2) & 0x03030303u, (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
          }
        };
        stage(0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < Kg; g++) {
          const int cur = g & 1;
          if (g + 1 < Kg) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 64>(g * 128, m0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = NEG ? -s0 : float4(bp0[g * NQ]);
          const float4 b1 = NEG ? -s1 : float4(bp1[g * NQ]);
          const float4 as4 = *(const device float4*)(ascale + tbase + (size_t)g * 64);
          const float4 rb4 = *(const device float4*)(rsb + tbase + (size_t)g * 64);
          float as[4], rb[4];
          as[0] = as4.x; as[1] = as4.y; as[2] = as4.z; as[3] = as4.w;
          rb[0] = rb4.x; rb[1] = rb4.y; rb[2] = rb4.z; rb[3] = rb4.w;
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float t = s * float(cT[i]);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i += 4) {
          const int nh = (i >> 3) & 1;
          const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
          *(device float4*)(out + (size_t)mm * N + nb + 32 * nh) = float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
        }
        """

    private static let naxKernel = MLXFast.metalKernel(
        name: "bonsai_q8_prompt_matmul_nax",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxSource,
        header: naxHeader,
        ensureRowContiguous: true)

    private static let naxVectorKernel = MLXFast.metalKernel(
        name: "bonsai_q8_prompt_matmul_nax_vector",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxVectorSource,
        header: naxHeader,
        ensureRowContiguous: true)

    // `naxVectorSource` re-tiled for the M5 (same bits): the 64 x 64 tile's
    // four 32 x 32 blocks each run their own per-simdgroup 32 x 32 x 128 int8
    // tensor op (`execution_simdgroup`) on the shared double-buffered weights,
    // and each lane applies the epilogue to its own 32 elements. The integer C
    // per group is the same exact integer, the epilogue is the same three
    // roundings in the same order with groups ascending from `acc = 0`, so
    // every output is the vector kernel's bits. Lane layout of one block (row
    // fm + 8 i2 + 16 i4, column fn + 16 i3 + (i & 3)) is the one measured on
    // the M5; each simdgroup checks it against `get_multidimensional_index`
    // and writes NaN on any mismatch, so the load-time check then fails.
    // Body of `naxSgSource`: reads K, M, N, bx, w, scalesT, biasesT and out.
    static let naxSgBody = """
        using elem_index = metal::conditional_t<INDEX32, uint, size_t>;
        const int Kg = K / 128;
        const int n0 = bx * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        const int q = int(sg & 3);
        const int sm = 32 * (q >> 1), sn = 32 * (q & 1);
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        threadgroup uint32_t bs[2][64 * 128 / 4];
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, M));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)bs[0], dextents<int, 2>(128, 64));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B1((threadgroup int8_t*)bs[1], dextents<int, 2>(128, 64));
        auto B0s = B0.template slice<128, 32>(0, sn);
        auto B1s = B1.template slice<128, 32>(0, sn);
        auto tA0 = A.template slice<128, 32>(0, m0 + sm);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0s)>, int32_t>();
        bool ok = int(cT.get_capacity()) == 32;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; i++) {
          auto ids = cT.get_multidimensional_index(i);
          ok = ok && int(ids[1]) == fm + 8 * ((i >> 2) & 1) + 16 * ((i >> 4) & 1)
                  && int(ids[0]) == fn + 16 * ((i >> 3) & 1) + (i & 3);
        }
        const bool good = simd_all(ok);
        const int sc = int(tid >> 1); const int sh = int(tid & 1);
        const device uint32_t* wrow = w + (elem_index)(n0 + sc) * (K / 16) + sh * 4;
        auto stage = [&](int g, int buf) {
          const uint4 v = *(const device uint4*)(wrow + g * 8);
          threadgroup uint32_t* dst = bs[buf] + sc * 32 + sh * 16;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const uint32_t wv = v[j];
            *(threadgroup uint4*)(dst + 4 * j) = uint4(
                wv & 0x03030303u, (wv >> 2) & 0x03030303u, (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
          }
        };
        const int nb = n0 + sn + fn;
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 16);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 16);
        const int NQ = N / 4;
        const elem_index tbase = (elem_index)(m0 / 64) * (elem_index)Kg * 64 + (elem_index)(4 * fm + 2 * (q >> 1));
        float acc[32];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; i++) { acc[i] = 0.0f; }
        stage(0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < Kg; g++) {
          if (g + 1 < Kg) { stage(g + 1, (g + 1) & 1); }
          auto tA = A.template slice<128, 32>(g * 128, m0 + sm);
          if ((g & 1) == 0) { op.run(tA, B0s, cT); } else { op.run(tA, B1s, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = NEG ? -s0 : float4(bp0[g * NQ]);
          const float4 b1 = NEG ? -s1 : float4(bp1[g * NQ]);
          const float2 a0 = *(const device float2*)(ascale + tbase + (elem_index)g * 64);
          const float2 a1 = *(const device float2*)(ascale + tbase + 32 + (elem_index)g * 64);
          const float2 r0 = *(const device float2*)(rsb + tbase + (elem_index)g * 64);
          const float2 r1 = *(const device float2*)(rsb + tbase + 32 + (elem_index)g * 64);
          float as[4], rb[4];
          as[0] = a0.x; as[1] = a0.y; as[2] = a1.x; as[3] = a1.y;
          rb[0] = r0.x; rb[1] = r0.y; rb[2] = r1.x; rb[3] = r1.y;
          #pragma clang loop unroll(full)
          for (int i = 0; i < 32; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float t = s * float(cT[i]);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const int mb = m0 + sm + fm;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; i += 4) {
          const int mm = mb + 8 * ((i >> 2) & 1) + 16 * ((i >> 4) & 1);
          const float4 v = float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
          *(device float4*)(out + (elem_index)mm * N + nb + 16 * ((i >> 3) & 1)) = good ? v : float4(as_type<float>(0x7fc00000u));
        }
        """

    static func naxSgSource(layers: Int, body: String = naxSgBody) -> String {
        var pick = """
            const int K = ksz[0]; const int M = ksz[1];
            int bx = int(threadgroup_position_in_grid.x);
            const device uint32_t* w = w0;
            const device half* scalesT = s0;
            const device half* biasesT = b0;
            device float* out = out0;
            int N = ksz[2];

            """
        for l in 1 ..< layers {
            pick += "if (bx >= N / 64) { bx -= N / 64; w = w\(l); scalesT = s\(l); biasesT = b\(l); out = out\(l); N = ksz[\(2 + l)]; }\n"
        }
        return pick + body
    }

    private static let naxSgKernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_prompt_matmul_nax_sg\(layers)_checked_offsets_v1",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxSgSource(layers: layers),
            header: naxHeader,
            ensureRowContiguous: true)
    }

    // `naxSgBody` with the right operand built in registers (27b 0043ea42
    // part B, `sourceStaged8Reg`): a 32 x 64 tile per threadgroup of two
    // simdgroups, each running its own 32 x 32 x 128 op. Lane l builds its
    // slice of the right cooperative tensor straight from the packed words:
    // columns nl + 8c (c = 0...3) and k-quad kq of every 16-block of the
    // group, word j of column nl + 8c shifted by 2 kq and masked, which is
    // the byte group `naxSgBody` stages at k = 16 j + 4 kq for that column.
    // No threadgroup memory, no staging stores, no barriers. Same integer C
    // per group, the same epilogue in the same order, the same layout check
    // and stores, so every output is `naxSgBody`'s bits (checked at load on
    // the prompt shapes, `checkRegister`). The words are read in place
    // ([N, K / 16]); 27b read a tiled copy. grid (sum N / 64 * 64, M / 32).
    static let naxRegBody = """
        const int Kg = K / 128;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int ms = int(threadgroup_position_in_grid.y) * 32;
        const int ns = bx * 64 + 32 * int(sg);
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, M));
        auto aT = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        auto bT = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        auto tA0 = A.template slice<128, 32>(0, ms);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aT)>, metal::remove_addrspace_t<decltype(bT)>, int32_t>();
        const int nb = ns + fn;
        const int mb = ms + fm;
        // The operand words are written straight into bT's storage, so the
        // storage must be 128 int8 elements at consecutive bytes in index
        // order from a 4-byte-aligned address; the destination must have the
        // element layout the stores assume. Otherwise the simdgroup stores
        // NaN (failing the load check) and touches no tensor.
        thread int8_t* b0p = &bT[0];
        bool ok = int(bT.get_capacity()) == 128 && int(cT.get_capacity()) == 32
            && ((uintptr_t)(b0p) & 3ul) == 0;
        if (ok) {
          for (int i = 1; i < 128; i++) { ok = ok && &bT[i] == b0p + i; }
          #pragma clang loop unroll(full)
          for (int i = 0; i < 32; i++) {
            auto ids = cT.get_multidimensional_index(i);
            ok = ok && int(ids[1]) == fm + 8 * ((i >> 2) & 1) + 16 * ((i >> 4) & 1)
                    && int(ids[0]) == fn + 16 * ((i >> 3) & 1) + (i & 3);
          }
        }
        if (!simd_all(ok)) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 32; i += 4) {
            const int mm = mb + 8 * ((i >> 2) & 1) + 16 * ((i >> 4) & 1);
            *(device float4*)(out + (size_t)mm * N + nb + 16 * ((i >> 3) & 1)) = float4(as_type<float>(0x7fc00000u));
          }
          return;
        }
        thread uint32_t* bw = (thread uint32_t*)b0p;
        const int nl = int(((lane >> 1) & 3) + 4 * ((lane >> 4) & 1));
        const uint sh = 2 * ((lane & 1) + 2 * ((lane >> 3) & 1));
        const size_t cs = (size_t)(K / 8);
        const device uint4* wcol = (const device uint4*)(w + (size_t)(ns + nl) * (size_t)(K / 16));
        uint4 wv[8];
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 16);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 16);
        const int NQ = N / 4;
        const size_t tbase = (size_t)(ms / 64) * (size_t)Kg * 64 + (size_t)(4 * fm) + (size_t)((ms & 32) >> 4);
        float acc[32];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; i++) { acc[i] = 0.0f; }
        for (int g = 0; g < Kg; g++) {
          const device uint4* src = wcol + (size_t)g * 2;
          #pragma clang loop unroll(full)
          for (int c = 0; c < 4; c++) { wv[2 * c] = src[c * cs]; wv[2 * c + 1] = src[c * cs + 1]; }
          #pragma clang loop unroll(full)
          for (int c = 0; c < 4; c++) {
            const uint4 lo = wv[2 * c]; const uint4 hi = wv[2 * c + 1];
            bw[c + 0] = (lo.x >> sh) & 0x03030303u; bw[c + 4] = (lo.y >> sh) & 0x03030303u;
            bw[c + 8] = (lo.z >> sh) & 0x03030303u; bw[c + 12] = (lo.w >> sh) & 0x03030303u;
            bw[c + 16] = (hi.x >> sh) & 0x03030303u; bw[c + 20] = (hi.y >> sh) & 0x03030303u;
            bw[c + 24] = (hi.z >> sh) & 0x03030303u; bw[c + 28] = (hi.w >> sh) & 0x03030303u;
          }
          auto tA = A.template slice<128, 32>(g * 128, ms);
          op.run(tA, bT, cT);
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = NEG ? -s0 : float4(bp0[g * NQ]);
          const float4 b1 = NEG ? -s1 : float4(bp1[g * NQ]);
          const float2 a0 = *(const device float2*)(ascale + tbase + (size_t)g * 64);
          const float2 a1 = *(const device float2*)(ascale + tbase + 32 + (size_t)g * 64);
          const float2 r0 = *(const device float2*)(rsb + tbase + (size_t)g * 64);
          const float2 r1 = *(const device float2*)(rsb + tbase + 32 + (size_t)g * 64);
          float as[4], rb[4];
          as[0] = a0.x; as[1] = a0.y; as[2] = a1.x; as[3] = a1.y;
          rb[0] = r0.x; rb[1] = r0.y; rb[2] = r1.x; rb[3] = r1.y;
          #pragma clang loop unroll(full)
          for (int i = 0; i < 32; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float t = s * float(cT[i]);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; i += 4) {
          const int mm = mb + 8 * ((i >> 2) & 1) + 16 * ((i >> 4) & 1);
          const float4 v = float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
          *(device float4*)(out + (size_t)mm * N + nb + 16 * ((i >> 3) & 1)) = v;
        }
        """

    private static let naxRegKernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_prompt_matmul_nax_reg\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxSgSource(layers: layers, body: naxRegBody),
            header: naxHeader,
            ensureRowContiguous: true)
    }

    private static func swapped(_ body: String, _ swaps: [(of: String, with: String)]) -> String {
        swaps.reduce(body) { text, swap in
            precondition(text.components(separatedBy: swap.of).count == 2, "fragment not found once: \(swap.of)")
            return text.replacingOccurrences(of: swap.of, with: swap.with)
        }
    }

    // `naxRegBody` on a 32 x 128 tile per threadgroup of four simdgroups, all
    // on the same 32 rows, so one activation slice serves 128 columns instead
    // of 64. Simdgroups 2 h and 2 h + 1 take the 64-column unit 2 x + h through
    // `naxSgSource`'s layer pick, each its own 32-column half, so the pick's
    // boundaries stay at N / 64: a threadgroup may span two layers, and a unit
    // past the last layer (an odd unit count) returns before any access. Each
    // simdgroup runs `naxRegBody`'s guard, operand build, op, epilogue and
    // stores on its own 32 x 32 tile, so every output is `naxSgBody`'s bits
    // (checked at load, `checkRegisterCols128`). grid (ceil(sum N / 128) *
    // 128, M / 32).
    static func naxRegCols128Source(layers: Int, operandBody: String = naxRegBody) -> String {
        let body = "using elem_index = metal::conditional_t<INDEX32, uint, size_t>;\n"
            + operandBody.replacingOccurrences(of: "(size_t)", with: "(elem_index)")
                .replacingOccurrences(of: "const size_t ", with: "const elem_index ")
        return swapped(naxSgSource(layers: layers, body: body), [
            (of: "int bx = int(threadgroup_position_in_grid.x);",
             with: "int bx = 2 * int(threadgroup_position_in_grid.x) + int(simdgroup_index_in_threadgroup >> 1);"),
            (of: "const int Kg = K / 128;", with: "if (bx >= N / 64) { return; }\nconst int Kg = K / 128;"),
            (of: "const int ns = bx * 64 + 32 * int(sg);", with: "const int ns = bx * 64 + 32 * int(sg & 1);"),
        ])
    }

    private static let naxRegCols128Kernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_prompt_matmul_nax_reg_cols128_\(layers)_checked_offsets_v1",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxRegCols128Source(layers: layers),
            header: naxHeader,
            ensureRowContiguous: true)
    }

    // Read the same immutable C16 copies as verify. Each uint4 still expands
    // into exactly the canonical staged B bytes. Independent allocations keep
    // MLX's command-buffer grouping by input buffer unchanged.
    private static let streamedSgBody: String? = {
        guard naxSgBody.components(separatedBy: "const int sc =").count == 2,
            naxSgBody.components(separatedBy: "const int nb =").count == 2,
            let start = naxSgBody.range(of: "const int sc ="),
            let end = naxSgBody.range(of: "const int nb =",
                range: start.lowerBound ..< naxSgBody.endIndex) else { return nil }
        let stage = """
            const uint pl = tid & 31u;
            const uint tile = tid >> 5;
            auto stage = [&](int g, int buf) {
              const elem_index at = ((elem_index)(n0 / 16 + int(tile)) * Kg + g) * 128
                  + pl * 4u;
              const uint4 v = *(const device uint4*)(w + at);
              #pragma clang loop unroll(full)
              for (int j = 0; j < 4; j++) {
                const uint col = tile * 16u + (pl >> 1);
                const uint word = (pl & 1u) * 4u + uint(j);
                threadgroup uint32_t* dst = bs[buf] + col * 32u + word * 4u;
                const uint wv = v[j];
                *(threadgroup uint4*)dst = uint4(wv & 0x03030303u,
                    (wv >> 2) & 0x03030303u, (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
              }
            };

            """
        return naxSgBody.replacingCharacters(in: start.lowerBound ..< end.lowerBound, with: stage)
    }()

    private static let streamedSgKernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_prompt_ms1b_c16_sg\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxSgSource(layers: layers, body: streamedSgBody ?? naxSgBody),
            header: naxHeader, ensureRowContiguous: true)
    }

    private static func replacingStreamFragments(
        _ body: String, _ swaps: [(of: String, with: String)]
    ) -> String? {
        var result = body
        for swap in swaps {
            guard result.components(separatedBy: swap.of).count == 2 else { return nil }
            result = result.replacingOccurrences(of: swap.of, with: swap.with)
        }
        return result
    }

    // Only the word loads differ; the cooperative operand and all math stay native.
    private static let streamedRegBody: String? = replacingStreamFragments(naxRegBody, [
        (of: "const size_t cs = (size_t)(K / 8);\nconst device uint4* wcol = (const device uint4*)(w + (size_t)(ns + nl) * (size_t)(K / 16));\nuint4 wv[8];", with: "const device uint4* wcol = (const device uint4*)(w + (size_t)(ns / 16) * Kg * 128 + (size_t)nl * 8);\nuint4 wv[8];"),
        (of: "  const device uint4* src = wcol + (size_t)g * 2;\n  #pragma clang loop unroll(full)\n  for (int c = 0; c < 4; c++) { wv[2 * c] = src[c * cs]; wv[2 * c + 1] = src[c * cs + 1]; }\n  #pragma clang loop unroll(full)\n  for (int c = 0; c < 4; c++) {\n    const uint4 lo = wv[2 * c]; const uint4 hi = wv[2 * c + 1];\n    bw[c + 0] = (lo.x >> sh) & 0x03030303u; bw[c + 4] = (lo.y >> sh) & 0x03030303u;\n    bw[c + 8] = (lo.z >> sh) & 0x03030303u; bw[c + 12] = (lo.w >> sh) & 0x03030303u;\n    bw[c + 16] = (hi.x >> sh) & 0x03030303u; bw[c + 20] = (hi.y >> sh) & 0x03030303u;\n    bw[c + 24] = (hi.z >> sh) & 0x03030303u; bw[c + 28] = (hi.w >> sh) & 0x03030303u;\n  }", with: "  const device uint4* src = wcol + (size_t)g * 32;\n  #pragma clang loop unroll(full)\n  for (int c = 0; c < 4; c++) {\n    const size_t off = (size_t)(c >> 1) * Kg * 32 + (c & 1) * 16;\n    wv[2*c] = src[off]; wv[2*c+1] = src[off+1];\n  }\n  #pragma clang loop unroll(full)\n  for (int c = 0; c < 4; c++) {\n    const uint4 lo = wv[2 * c]; const uint4 hi = wv[2 * c + 1];\n    bw[c + 0] = (lo.x >> sh) & 0x03030303u; bw[c + 4] = (lo.y >> sh) & 0x03030303u;\n    bw[c + 8] = (lo.z >> sh) & 0x03030303u; bw[c + 12] = (lo.w >> sh) & 0x03030303u;\n    bw[c + 16] = (hi.x >> sh) & 0x03030303u; bw[c + 20] = (hi.y >> sh) & 0x03030303u;\n    bw[c + 24] = (hi.z >> sh) & 0x03030303u; bw[c + 28] = (hi.w >> sh) & 0x03030303u;\n  }"),
    ])

    private static let streamedRegKernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_prompt_c16col_reg_\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxSgSource(layers: layers, body: streamedRegBody ?? naxRegBody),
            header: naxHeader, ensureRowContiguous: true)
    }

    private static let streamedRegCols128Kernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_prompt_c16col_reg_cols128_\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxRegCols128Source(layers: layers, operandBody: streamedRegBody ?? naxRegBody),
            header: naxHeader, ensureRowContiguous: true)
    }

    // No tensor operation: compare the 32 packed operand words in each lane.
    private static let registerOperandProofSource = "const uint lane = thread_index_in_simdgroup;\nconst uint3 pos = threadgroup_position_in_grid;\nconst int Kg = K / 128;\nconst int ns = int(pos.x) * 32;\nconst int g = int(pos.y);\nconst int nl = int(((lane >> 1) & 3) + 4 * ((lane >> 4) & 1));\nconst uint sh = 2 * ((lane & 1) + 2 * ((lane >> 3) & 1));\nuint32_t reference[32], candidate[32];\n{\n  const device uint32_t* w = canonical;\n  thread uint32_t* bw = reference;\n  const size_t cs = (size_t)(K / 8);\n  const device uint4* wcol = (const device uint4*)(w + (size_t)(ns + nl) * (size_t)(K / 16));\n  uint4 wv[8];\n  const device uint4* src = wcol + (size_t)g * 2;\n  #pragma clang loop unroll(full)\n  for (int c = 0; c < 4; c++) { wv[2 * c] = src[c * cs]; wv[2 * c + 1] = src[c * cs + 1]; }\n  #pragma clang loop unroll(full)\n  for (int c = 0; c < 4; c++) {\n    const uint4 lo = wv[2 * c]; const uint4 hi = wv[2 * c + 1];\n    bw[c + 0] = (lo.x >> sh) & 0x03030303u; bw[c + 4] = (lo.y >> sh) & 0x03030303u;\n    bw[c + 8] = (lo.z >> sh) & 0x03030303u; bw[c + 12] = (lo.w >> sh) & 0x03030303u;\n    bw[c + 16] = (hi.x >> sh) & 0x03030303u; bw[c + 20] = (hi.y >> sh) & 0x03030303u;\n    bw[c + 24] = (hi.z >> sh) & 0x03030303u; bw[c + 28] = (hi.w >> sh) & 0x03030303u;\n  }\n}\n{\n  const device uint32_t* w = packed;\n  thread uint32_t* bw = candidate;\n  const device uint4* wcol = (const device uint4*)(w + (size_t)(ns / 16) * Kg * 128 + (size_t)nl * 8);\n  uint4 wv[8];\n  const device uint4* src = wcol + (size_t)g * 32;\n  #pragma clang loop unroll(full)\n  for (int c = 0; c < 4; c++) {\n    const size_t off = (size_t)(c >> 1) * Kg * 32 + (c & 1) * 16;\n    wv[2*c] = src[off]; wv[2*c+1] = src[off+1];\n  }\n  #pragma clang loop unroll(full)\n  for (int c = 0; c < 4; c++) {\n    const uint4 lo = wv[2 * c]; const uint4 hi = wv[2 * c + 1];\n    bw[c + 0] = (lo.x >> sh) & 0x03030303u; bw[c + 4] = (lo.y >> sh) & 0x03030303u;\n    bw[c + 8] = (lo.z >> sh) & 0x03030303u; bw[c + 12] = (lo.w >> sh) & 0x03030303u;\n    bw[c + 16] = (hi.x >> sh) & 0x03030303u; bw[c + 20] = (hi.y >> sh) & 0x03030303u;\n    bw[c + 24] = (hi.z >> sh) & 0x03030303u; bw[c + 28] = (hi.w >> sh) & 0x03030303u;\n  }\n}\nuint diff = 0;\nfor (int i = 0; i < 32; ++i) diff |= reference[i] ^ candidate[i];\nbad[(size_t(pos.x) * Kg + pos.y) * 32 + lane] = diff != 0 ? 1 : 0;\n"
    private static let registerOperandProofKernels = [false, true].map { index32 in
        let source = index32 ? registerOperandProofSource
            .replacingOccurrences(of: "(size_t)", with: "(uint)")
            .replacingOccurrences(of: "const size_t ", with: "const uint ") : registerOperandProofSource
        return MLXFast.metalKernel(
            name: "bonsai_c16_register_operand_proof_\(index32 ? 32 : 64)",
            inputNames: ["canonical", "packed"], outputNames: ["bad"], source: source,
            ensureRowContiguous: true)
    }

    static func checkRegisterStreamOperands(weight: MLXArray, packed: MLXArray, k: Int) -> Bool {
        let n = weight.dim(0), kg = k / 128
        var matched = true
        let failed = OSAllocatedUnfairLock(initialState: false)
        withErrorHandler({ _ in failed.withLock { $0 = true } }) {
            for kernel in registerOperandProofKernels where matched {
                let bad = kernel([weight, packed], template: [("K", k)],
                    grid: (n, kg, 1), threadGroup: (32, 1, 1),
                    outputShapes: [[n / 32, kg, 32]], outputDTypes: [.uint8])[0]
                let equal = MLX.all(bad .== MLXArray(UInt8(0)))
                eval(equal)
                let didFail = failed.withLock { $0 }
                matched = !didFail && equal.item(Bool.self)
            }
        }
        let didFail = failed.withLock { $0 }
        let passed = matched && !didFail
        FileHandle.standardError.write(Data("C16 register operand: \(passed ? "passed" : "FAILED") K\(k) N\(n); all tiles/groups/lanes/32 words; index32/index64\n".utf8))
        return passed
    }

    private final class StreamPair {
        weak var canonical: MLXArray?
        weak var packed: MLXArray?
        init(_ canonical: MLXArray, _ packed: MLXArray) {
            self.canonical = canonical
            self.packed = packed
        }
    }
    private final class StreamTable: @unchecked Sendable {
        let lock = NSLock()
        var pairs: [ObjectIdentifier: StreamPair] = [:]
    }
    private static let streamTable = StreamTable()
    private static let streamPromptAllowed = ProcessInfo.processInfo.environment[
        "DARKBLOOM_BONSAI_PREFILL_STREAM"] != "0"
    private static let checkStreamActual = ProcessInfo.processInfo.environment[
        "DARKBLOOM_BONSAI_PREFILL_STREAM_CHECK"] == "1"
    nonisolated(unsafe) private static var streamedForm = SgForm.staged
    nonisolated(unsafe) private static var streamPromptReady = false
    nonisolated(unsafe) private static var streamPromptPrepared = false
    nonisolated(unsafe) private static var streamPromptError = false
    nonisolated(unsafe) private static var streamPromptNoted = false

    static func registerStream(_ weight: MLXArray, packed: MLXArray) {
        streamTable.lock.lock()
        defer { streamTable.lock.unlock() }
        if streamTable.pairs.count > 1024 {
            streamTable.pairs = streamTable.pairs.filter { $0.value.canonical != nil && $0.value.packed != nil }
        }
        streamTable.pairs[ObjectIdentifier(weight)] = StreamPair(weight, packed)
    }

    private static func promptStream(_ weight: MLXArray, m: Int, k: Int) -> MLXArray? {
        guard streamPromptReady, m >= 64, m % 64 == 0, k >= 128, k % 128 == 0,
            weight.dim(0) > 0, weight.dim(0) % 64 == 0
        else { return nil }
        streamTable.lock.lock()
        defer { streamTable.lock.unlock() }
        guard let pair = streamTable.pairs[ObjectIdentifier(weight)], pair.canonical === weight else { return nil }
        return pair.packed
    }

    private static func streamOutputs(_ outputs: [MLXArray], reference: () -> [MLXArray]) -> [MLXArray] {
        if !streamPromptNoted {
            streamPromptNoted = true
            FileHandle.standardError.write(Data("prompt shared verify weights: active\n".utf8))
        }
        guard checkStreamActual else { return outputs }
        let original = reference()
        let checks: [MLXArray] = zip(original, outputs).map { a, b in
            MLX.all(a.view(dtype: .uint32) .== b.view(dtype: .uint32))
        }
        eval(checks)
        let matched = checks.allSatisfy { $0.item(Bool.self) }
        FileHandle.standardError.write(Data("prompt streamed real: \(matched ? "passed" : "FAILED") \(outputs.map { $0.shape })\n".utf8))
        if !matched { streamPromptReady = false }
        return matched ? outputs : original
    }

    static func prepareStreaming() {
        guard !streamPromptPrepared else { return }
        streamPromptPrepared = true
        guard streamPromptAllowed, VerifyInt8Matmul.streamReady, route == .nax, naxSg else { return }
        guard streamedRegBody != nil, streamedSgBody != nil else {
            VerifyInt8Matmul.streamReady = false
            FileHandle.standardError.write(Data("prompt C16 source drift: canonical route kept\n".utf8))
            return
        }
        var allMatched = true
        for k in [4096, 12288] where allMatched {
            let n = 128, kg = k / 128
            let ws: [UInt32] = (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                let v: UInt64 = UInt64(i + 1) &* 2_654_435_761
                return UInt32(truncatingIfNeeded: v)
            }
            let sv: [Float16] = (0 ..< kg * n).map { (i: Int) -> Float16 in
                let v: Float = Float(i % 13 + 1) / 1024
                return Float16(v)
            }
            let weight = MLXArray(ws, [n, k / 16])
            let scales = MLXArray(sv, [kg, n])
            let bias = (scales.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
            let ranges: [[Range<Int>]] = [[0 ..< 128], [0 ..< 64, 64 ..< 128], [0 ..< 64, 64 ..< 128, 0 ..< 64]]
            let copied: [[MLXArray]] = ranges.map { rs in rs.map { VerifyInt8Matmul.streamPack(weight[$0], k: k, columns: 16) } }
            eval(copied.flatMap { $0 })
            var matched = checkRegisterStreamOperands(weight: weight, packed: copied[0][0], k: k)
            streamPromptError = false
            withErrorHandler({ _ in PromptInt8Matmul.streamPromptError = true }) {
                for m in [64, 128] where matched {
                    let qs: [Int8] = (0 ..< m * k).map { (i: Int) -> Int8 in
                        let v: Int = (i * 37 + 11) % 255 - 127
                        return Int8(truncatingIfNeeded: v)
                    }
                    let av: [Float] = (0 ..< m * kg).map { (i: Int) -> Float in Float(i % 11 + 1) / 256 }
                    let rv: [Float] = (0 ..< m * kg).map { (i: Int) -> Float in Float(i % 17 - 8) / 64 }
                    let codes = MLXArray(qs, [m, k]), ascale = MLXArray(av), rsb = MLXArray(rv)
                    for negative in [false, true] where matched {
                        for count in 1 ... 3 where matched {
                            let canonical: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)] = ranges[count - 1].map { r in
                                (weight[r], scales[0..., r], bias[0..., r])
                            }
                            let packed: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)] = canonical.enumerated().map { i, entry in
                                (copied[count - 1][i], entry.scalesT, entry.biasesT)
                            }
                            let a = runSg(codes: codes, ascale: ascale, rsb: rsb, layers: canonical,
                                m: m, k: k, negative: negative, form: .staged)
                            let b = runSg(codes: codes, ascale: ascale, rsb: rsb, layers: packed,
                                m: m, k: k, negative: negative, form: .staged, streamed: true)
                            let checks: [MLXArray] = zip(a, b).map { x, y in MLX.all(x.view(dtype: .uint32) .== y.view(dtype: .uint32)) }
                            eval(checks)
                            matched = !streamPromptError && checks.allSatisfy { $0.item(Bool.self) }
                        }
                    }
                }
            }
            allMatched = matched && !streamPromptError
        }
        streamedForm = .staged
        if allMatched, sgForm != .staged {
            // Keep the canonical prompt form's performance contract. A C16
            // register failure disables both streams instead of downgrading.
            let qualified = matchesStaged(sgForm, streamed: true)
            allMatched = qualified.same
            if allMatched { streamedForm = sgForm }
            FileHandle.standardError.write(Data("prompt C16 register qualification: \(qualified.same ? "passed" : "FAILED; canonical buffers kept") \(sgForm)\n".utf8))
        }
        streamPromptReady = allMatched
        FileHandle.standardError.write(Data("prompt C16 selected form: \(streamedForm); canonical form: \(sgForm)\n".utf8))
        // A failed prefill qualification keeps the canonical layout on both
        // paths, rather than enabling copies with an unqualified consumer.
        if !allMatched { VerifyInt8Matmul.streamReady = false }
        if allMatched { prepareGroupedM(streamed: true) }
        FileHandle.standardError.write(Data("prompt streamed loadcheck: \(streamPromptReady ? "passed" : "FAILED")\n".utf8))
    }

    private static func groupedMSource(_ source: String, form: SgForm) -> String? {
        guard source.components(separatedBy: "threadgroup_position_in_grid.x").count == 2,
            source.components(separatedBy: "threadgroup_position_in_grid.y").count == 2 else { return nil }
        let rows = form == .staged ? 64 : 32
        let columns = form == .registerCols128 ? 128 : 64
        return """
            const uint3 pm_pos = threadgroup_position_in_grid;
            const uint pm_nt = (uint(ksz[2]) + uint(ksz[3]) + \(columns - 1)u) / \(columns)u;
            const uint pm_mt = uint(ksz[1]) / \(rows)u;
            const uint pm_p = pm_pos.x + pm_nt * pm_pos.y;
            const uint pm_span = min(4u, pm_mt) * pm_nt;
            const uint pm_group = pm_p / pm_span;
            const uint pm_first = 4u * pm_group;
            const uint pm_height = min(4u, pm_mt - pm_first);
            const uint pm_local = pm_p - pm_group * pm_span;
            {
            const uint3 threadgroup_position_in_grid = uint3(pm_local / pm_height, pm_first + pm_local % pm_height, pm_pos.z);

            """ + source + "\n}\n"
    }

    private static let groupedMKernels: [[MLXFast.MLXFastKernel?]] = [false, true].map { streamed in
        [SgForm.staged, SgForm.register, SgForm.registerCols128].map { form in
            let body: String?
            if form == .staged { body = streamed ? streamedSgBody : naxSgBody }
            else { body = streamed ? streamedRegBody : naxRegBody }
            guard let body else { return nil }
            let original = form == .registerCols128
                ? naxRegCols128Source(layers: 2, operandBody: body) : naxSgSource(layers: 2, body: body)
            guard let source = groupedMSource(original, form: form) else { return nil }
            return MLXFast.metalKernel(name: "bonsai_prompt_m4_\(streamed ? 1 : 0)_\(form.index)_v1",
                inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "rsb", "ksz"],
                outputNames: ["out0", "out1"], source: source, header: naxHeader, ensureRowContiguous: true)
        }
    }

    nonisolated(unsafe) private static var groupedMReady = [false, false]

    private static func groupedMShape(k: Int, ns: [Int]) -> Bool {
        k == 4096 && ns.count == 2 && ns[0] == 12288 && ns[1] == 12288
    }

    private static func prepareGroupedM(streamed: Bool) {
        let slot = streamed ? 1 : 0
        groupedMReady[slot] = false
        let form = streamed ? streamedForm : sgForm
        guard naxSg, naxSgShared, !streamed || streamPromptReady,
            groupedMKernels[slot][form.index] != nil else { return }
        let qualified = matchesStaged(form, streamed: streamed, groupedM: true)
        groupedMReady[slot] = qualified.same
        FileHandle.standardError.write(Data("prompt grouped-M4 streamed=\(streamed) form=\(form) selected=\(qualified.same): \(qualified.detail)\n".utf8))
    }

    enum SgForm {
        case staged, register, registerCols128

        var index: Int { self == .staged ? 0 : self == .register ? 1 : 2 }

        var streamedLaunch: (kernels: [MLXFast.MLXFastKernel], threads: Int, rows: Int, columns: Int) {
            switch self {
            case .staged: return (PromptInt8Matmul.streamedSgKernels, 128, 64, 64)
            case .register: return (PromptInt8Matmul.streamedRegKernels, 64, 32, 64)
            case .registerCols128: return (PromptInt8Matmul.streamedRegCols128Kernels, 128, 32, 128)
            }
        }

        var launch: (kernels: [MLXFast.MLXFastKernel], threads: Int, rows: Int, columns: Int) {
            switch self {
            case .staged: return (PromptInt8Matmul.naxSgKernels, 128, 64, 64)
            case .register: return (PromptInt8Matmul.naxRegKernels, 64, 32, 64)
            case .registerCols128: return (PromptInt8Matmul.naxRegCols128Kernels, 128, 32, 128)
            }
        }
    }

    nonisolated(unsafe) private static var naxSg = false
    nonisolated(unsafe) private static var naxSgShared = false
    nonisolated(unsafe) private static var sgForm = SgForm.staged

    private static func runSg(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)],
        m: Int, k: Int, negative: Bool, form: SgForm, streamed: Bool = false, groupedM: Bool? = nil
    ) -> [MLXArray] {
        let ns = layers.map { $0.weight.dim(0) }
        let dims = MLXArray([Int32(k), Int32(m)] + ns.map { Int32($0) })
        let inputs = [codes] + layers.map { $0.weight } + layers.map { $0.scalesT } + layers.map { $0.biasesT }
            + [ascale, rsb, dims]
        let limit = Int(Int32.max)
        let index32 = m > 0 && m % 64 == 0 && k >= 128 && k % 128 == 0
            && m <= limit / k && ns.allSatisfy { n in
                n > 0 && n % 64 == 0 && n <= limit / (k / 16) && m <= limit / n
            }
        var template: [(String, any KernelTemplateArg)] = [("NEG", negative ? 1 : 0)]
        if form == .staged || form == .registerCols128 { template.append(("INDEX32", index32)) }
        let launch = streamed ? form.streamedLaunch : form.launch
        let kernels = launch.kernels
        let slot = streamed ? 1 : 0
        let grouped = (groupedM ?? groupedMReady[slot]) && m >= 128 && index32 && groupedMShape(k: k, ns: ns)
        let kernel = grouped ? (groupedMKernels[slot][form.index] ?? kernels[layers.count - 1])
            : kernels[layers.count - 1]
        return kernel(
            inputs, template: template,
            grid: ((ns.reduce(0, +) + launch.columns - 1) / launch.columns * launch.threads, m / launch.rows, 1),
            threadGroup: (launch.threads, 1, 1),
            outputShapes: ns.map { [m, $0] }, outputDTypes: ns.map { _ in .float32 })
    }

    nonisolated(unsafe) private static var naxVector = false
    nonisolated(unsafe) private static var negativeForms = false

    private static func run(
        _ route: Route, codes: MLXArray, ascale: MLXArray, rsb: MLXArray, weight: MLXArray,
        scalesT: MLXArray, biasesT: MLXArray, m: Int, k: Int, n: Int,
        vector: Bool, negative: Bool
    ) -> MLXArray {
        let dims = MLXArray([Int32(k), Int32(m), Int32(n)])
        let inputs = [codes, weight, scalesT, biasesT, ascale, rsb, dims]
        let neg = [("NEG", negative ? 1 : 0)] as [(String, any KernelTemplateArg)]
        if route == .nax {
            if vector {
                return naxVectorKernel(
                    inputs, template: neg, grid: (n / 64 * 128, m / 64, 1), threadGroup: (128, 1, 1),
                    outputShapes: [[m, n]], outputDTypes: [.float32])[0]
            }
            return naxKernel(
                inputs, grid: (n / 64 * 128, m / 64, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [.float32])[0]
        }
        return plainKernel(
            inputs, template: neg, grid: (n / 32 * 128, m / 32, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [.float32])[0]
    }

    static let registerCheckShapes: [(k: Int, ns: [Int])] = [
        (4096, [8192, 1024, 1024]), (4096, [8192, 4096]), (4096, [12288, 12288]),
        (4096, [1024, 1024]), (4096, [4096]), (12288, [4096]),
    ]
    static let registerCheckRows = 512

    nonisolated(unsafe) private static var registerFailed = false

    private static func matchesStaged(_ form: SgForm, streamed: Bool = false, groupedM: Bool = false) -> (same: Bool, detail: String) {
        let m = registerCheckRows
        var same = true
        var compared = 0
        registerFailed = false
        withErrorHandler({ _ in PromptInt8Matmul.registerFailed = true }) {
            for (index, shape) in registerCheckShapes.enumerated() where same {
                if groupedM && !groupedMShape(k: shape.k, ns: shape.ns) { continue }
                let k = shape.k, kg = k / 128
                let seed = UInt64(0x5245_4700 + 16 * index)
                let codes = MLXRandom.randInt(Int32(-127) ..< Int32(128), [m, k], key: MLXRandom.key(seed))
                    .asType(.int8)
                let ascale = MLXRandom.uniform(
                    low: Float(0.0001), high: Float(0.05), [m * kg], key: MLXRandom.key(seed + 1))
                let rsb = MLXRandom.normal([m * kg], key: MLXRandom.key(seed + 2)) * Float(50)
                var offsets: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)] = []
                var negated: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)] = []
                for (layer, n) in shape.ns.enumerated() {
                    let key = seed + 3 + UInt64(4 * layer)
                    let weight = MLXRandom.randInt(Int32(0) ..< Int32(65536), [n, k / 8], key: MLXRandom.key(key))
                        .asType(.uint16).view(dtype: .uint32)
                    let scalesT = MLXRandom.uniform(
                        low: Float(-0.05), high: Float(0.05), [kg, n], key: MLXRandom.key(key + 1)
                    ).asType(.float16)
                    let biasesT = MLXRandom.uniform(
                        low: Float(-0.05), high: Float(0.05), [kg, n], key: MLXRandom.key(key + 2)
                    ).asType(.float16)
                    let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
                    offsets.append((weight, scalesT, biasesT))
                    negated.append((weight, scalesT, negatedT))
                }
                var cases = [(offsets, false)]
                if negativeForms { cases.append((negated, true)) }
                for (layers, negative) in cases where same {
                    let copies = streamed ? layers.map { VerifyInt8Matmul.streamPack($0.weight, k: k, columns: 16) } : []
                    if streamed { eval(copies) }
                    let packed: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)] = streamed
                        ? layers.enumerated().map { i, layer in (copies[i], layer.scalesT, layer.biasesT) } : layers
                    func outputs(_ which: SgForm, grouped: Bool = false) -> [MLXArray] {
                        let useCopies = streamed && which == form
                        return runSg(
                            codes: codes, ascale: ascale, rsb: rsb, layers: useCopies ? packed : layers,
                            m: m, k: k, negative: negative, form: which, streamed: useCopies, groupedM: grouped
                        ).map { $0.view(dtype: .uint32) }
                    }
                    if groupedM {
                        let kind = 100 + 2 * form.index + (streamed ? 1 : 0)
                        VerifyInt8Matmul.qualifyMetadata(kind: kind, k: k, rows: m, negative: negative, ns: shape.ns,
                            label: "prompt grouped-M4 form=\(form)", methods: "[old,grouped_M4]", verdict: "selected") { grouped in
                            outputs(form, grouped: grouped).map { $0.view(dtype: .float32) }
                        }
                        same = VerifyInt8Matmul.metadataUsesShuffle(kind: kind, k: k, rows: m, negative: negative, ns: shape.ns)
                    } else {
                        let checks = zip(outputs(.staged), outputs(form)).map { MLX.all($0 .== $1) }
                        eval(checks)
                        same = !registerFailed && checks.allSatisfy { $0.item(Bool.self) }
                    }
                    compared += m * shape.ns.reduce(0, +)
                }
            }
        }
        same = same && !registerFailed
        registerFailed = false
        let shapes = registerCheckShapes.filter { !groupedM || groupedMShape(k: $0.k, ns: $0.ns) }.map { shape in
            "K \(shape.k) N \(shape.ns.map { String($0) }.joined(separator: "+"))"
        }.joined(separator: ", ")
        return (same, same
            ? "bitwise equal to the per-simdgroup kernel at rows \(m): \(shapes); NEG 0\(negativeForms ? " and 1" : ""); \(compared) values"
            : groupedM ? "exact/finite/strict timing check declined" : "a check differed or failed to compile")
    }

    private static func switchedOff(_ variable: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[variable]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["0", "false", "no", "off"].contains(value ?? "")
    }

    private static func checkRegister() -> String {
        sgForm = .staged
        if switchedOff("DARKBLOOM_BONSAI_PROMPT_REG") { return "off (DARKBLOOM_BONSAI_PROMPT_REG=0)" }
        guard naxSg, naxSgShared else { return "off (the per-simdgroup shared kernel is not in use)" }
        let (same, detail) = matchesStaged(.register)
        if same { sgForm = .register }
        return same ? "on (\(detail))" : "off (\(detail); the per-simdgroup kernel is kept)"
    }

    private static func checkRegisterCols128() -> String {
        if switchedOff("DARKBLOOM_BONSAI_PROMPT_REG_COLS128") { return "off (DARKBLOOM_BONSAI_PROMPT_REG_COLS128=0)" }
        guard sgForm != .staged else { return "off (the register-weight kernel is not in use)" }
        let (same, detail) = matchesStaged(.registerCols128)
        if same { sgForm = .registerCols128 }
        return same ? "on (\(detail))" : "off (\(detail); the \(sgForm) form is kept)"
    }

    static func architectureHasNAX(_ name: String) -> Bool {
        guard let range = name.range(of: "applegpu_g") else { return false }
        let tail = name[range.upperBound...]
        let digits = tail.prefix(while: { $0.isNumber })
        guard let gen = Int(digits) else { return false }
        let variant = tail.dropFirst(digits.count).first
        return gen >= (variant == "p" ? 18 : 17)
    }

    nonisolated(unsafe) private static var checkFailed = false
    nonisolated(unsafe) private static var checkNote = "not run"

    @discardableResult
    private static func checkKernels(nax: Bool) -> Bool {
        checkFailed = false
        let m = 128, n = 128, k = 512, kg = k / 128
        let codes = MLXArray((0 ..< m * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [m, k])
        let weight = MLXArray(
            (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                let value: UInt64 = UInt64(i + 1) &* 2_654_435_761
                return UInt32(truncatingIfNeeded: value)
            }, [n, k / 16])
        let scaleValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
            let value: Float = Float(1 + index % 13) / 1024
            return Float16(value)
        }
        let scalesT = MLXArray(scaleValues, [kg, n])
        let biasValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
            let value: Float = -Float(index % 7) / 512
            return Float16(value)
        }
        let biasesT = MLXArray(biasValues, [kg, n])
        let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
        let ascaleValues: [Float] = (0 ..< m * kg).map { (index: Int) -> Float in
            let value: Float = Float(1 + index % 11) / 256
            return value
        }
        let ascale = MLXArray(ascaleValues, [m * kg])
        let rsbValues: [Float] = (0 ..< m * kg).map { (index: Int) -> Float in
            let centered: Int = index % 17 - 8
            let value: Float = Float(centered) / 64
            return value
        }
        let rsb = MLXArray(rsbValues, [m * kg])
        func output(_ route: Route, _ offsets: MLXArray, vector: Bool, negative: Bool) -> MLXArray {
            run(route, codes: codes, ascale: ascale, rsb: rsb, weight: weight,
                scalesT: scalesT, biasesT: offsets, m: m, k: k, n: n, vector: vector, negative: negative)
                .view(dtype: .uint32)
        }
        func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
            let equal = MLX.all(a .== b)
            eval(equal)
            return !checkFailed && equal.item(Bool.self)
        }
        func sg(_ offsets: MLXArray, negative: Bool, _ columns: [Range<Int>]) -> MLXArray {
            let outputs = runSg(
                codes: codes, ascale: ascale, rsb: rsb,
                layers: columns.map { (weight: weight[$0], scalesT: scalesT[0..., $0], biasesT: offsets[0..., $0]) },
                m: m, k: k, negative: negative, form: .staged)
            return concatenated(outputs, axis: 1).view(dtype: .uint32)
        }
        var matched = false
        naxVector = false
        negativeForms = false
        naxSg = false
        naxSgShared = false
        withErrorHandler({ message in
            if !PromptInt8Matmul.checkFailed { PromptInt8Matmul.checkNote = String(message.prefix(160)) }
            PromptInt8Matmul.checkFailed = true
        }) {
            let plain = output(.plain, biasesT, vector: false, negative: false)
            let plainNegated = output(.plain, negatedT, vector: false, negative: false)
            let plainNeg = same(output(.plain, negatedT, vector: false, negative: true), plainNegated)
            guard nax else {
                matched = !checkFailed
                negativeForms = plainNeg
                return
            }
            if same(sg(biasesT, negative: false, [0 ..< n]), plain) {
                matched = true
                naxSg = true
                negativeForms = plainNeg && same(sg(negatedT, negative: true, [0 ..< n]), plainNegated)
                naxSgShared = same(sg(biasesT, negative: false, [0 ..< 64, 64 ..< 128]), plain)
                    && same(
                        sg(biasesT, negative: false, [0 ..< 64, 64 ..< 128, 0 ..< 64]),
                        concatenated([plain, plain[0..., 0 ..< 64]], axis: 1))
                if checkFailed {
                    checkFailed = false
                    naxSgShared = false
                }
                return
            }
            checkFailed = false
            if same(output(.nax, biasesT, vector: true, negative: false), plain) {
                matched = true
                naxVector = true
                negativeForms = plainNeg && same(output(.nax, negatedT, vector: true, negative: true), plainNegated)
            } else {
                checkFailed = false
                matched = same(output(.nax, biasesT, vector: false, negative: false), plain)
            }
        }
        if !checkFailed {
            checkNote = !matched ? "outputs differ"
                : "bit-identical (\(nax ? (naxSg ? (naxSgShared ? "per-simdgroup, shared" : "per-simdgroup") : naxVector ? "vector" : "record") : "plain") kernel, offset skip \(negativeForms ? "on" : "off"))"
        }
        return matched
    }

    static let route: Route = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_INT8_PREFILL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let architecture = GPU.deviceInfo().architecture
        let chosen: Route
        if ["0", "false", "no", "off"].contains(value ?? "") {
            chosen = .off
        } else if architectureHasNAX(architecture) {
            chosen = checkKernels(nax: true) ? .nax : .off
        } else {
            checkKernels(nax: false)
            chosen = .plain
        }
        FileHandle.standardError.write(
            "int8 prompt matmul: \(chosen) (\(architecture); nax check: \(checkNote))\n"
                .data(using: .utf8)!)
        if chosen == .nax {
            FileHandle.standardError.write(
                "int8 prompt register-weight kernel: \(checkRegister())\n".data(using: .utf8)!)
            FileHandle.standardError.write(
                "int8 prompt register-weight 128-column kernel: \(checkRegisterCols128())\n".data(using: .utf8)!)
            prepareGroupedM(streamed: false)
        }
        return chosen
    }()

    static func applies(rows: Int) -> Bool { rows >= 128 && rows % 64 == 0 }

    static func matmulLastRow(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray,
        biasesT: MLXArray, negative: Bool
    ) -> MLXArray? {
        guard route != .off, activation.rows == 1 || applies(rows: activation.rows),
            weight.dim(0) % 32 == 0 else { return nil }
        let n = weight.dim(0)
        let dims = MLXArray([Int32(activation.k), 1, Int32(n), Int32(activation.rows - 1)])
        return lastRowKernel(
            [activation.codes, weight, scalesT, biasesT, activation.ascale, activation.rsb, dims],
            template: [("NEG", negative && negativeForms ? 1 : 0)],
            grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[1, n]], outputDTypes: [.float32])[0]
    }

    static func matmul(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        negative: Bool
    ) -> MLXArray? {
        guard route != .off else { return nil }
        if let copied = promptStream(weight, m: activation.rows, k: activation.k) {
            let output = runSg(codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                layers: [(copied, scalesT, biasesT)], m: activation.rows, k: activation.k,
                negative: negative && negativeForms, form: streamedForm, streamed: true)
            return streamOutputs(output) {
                runSg(codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                    layers: [(weight, scalesT, biasesT)], m: activation.rows, k: activation.k,
                    negative: negative && negativeForms, form: sgForm)
            }[0]
        }
        if route == .nax && naxSg {
            return runSg(
                codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                layers: [(weight, scalesT, biasesT)], m: activation.rows, k: activation.k,
                negative: negative && negativeForms, form: sgForm)[0]
        }
        return run(
            route, codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
            weight: weight, scalesT: scalesT, biasesT: biasesT,
            m: activation.rows, k: activation.k, n: weight.dim(0),
            vector: naxVector, negative: negative && negativeForms)
    }

    static func matmul(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)]
    ) -> [MLXArray]? {
        guard route != .off else { return nil }
        // Siblings share one streamed dispatch only where the canonical shared
        // forms matched at load; otherwise each layer runs on its own, as below.
        if (2 ... 3).contains(layers.count), naxSgShared {
            let copies = layers.compactMap { promptStream($0.weight, m: activation.rows, k: activation.k) }
            if copies.count == layers.count {
                let packed: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)] = layers.enumerated().map { i, entry in
                    (copies[i], entry.scalesT, entry.biasesT)
                }
                let negative = negativeForms && layers.allSatisfy { $0.negative }
                let output = runSg(codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                    layers: packed, m: activation.rows, k: activation.k, negative: negative,
                    form: streamedForm, streamed: true)
                return streamOutputs(output) {
                    runSg(codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                        layers: layers.map { ($0.weight, $0.scalesT, $0.biasesT) },
                        m: activation.rows, k: activation.k, negative: negative, form: sgForm)
                }
            }
        }
        if route == .nax, naxSg, naxSgShared, (2 ... 3).contains(layers.count) {
            return runSg(
                codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                layers: layers.map { ($0.weight, $0.scalesT, $0.biasesT) },
                m: activation.rows, k: activation.k,
                negative: negativeForms && layers.allSatisfy { $0.negative }, form: sgForm)
        }
        let outputs = layers.compactMap {
            matmul(activation, weight: $0.weight, scalesT: $0.scalesT, biasesT: $0.biasesT, negative: $0.negative)
        }
        return outputs.count == layers.count ? outputs : nil
    }
}

enum VerifyInt8Matmul {
    static let streamAllowed: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_STREAM"]?.lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    nonisolated(unsafe) static var streamReady = false
    nonisolated(unsafe) private static var streamFailed = false

    private static let streamRepack = MLXFast.metalKernel(
        name: "bonsai_verify_stream_repack", inputNames: ["w"], outputNames: ["packed"],
        source: """
            const uint i = thread_position_in_grid.x;
            const uint Kg = K / 128;
            if (i >= N * K / 16) return;
            const uint bx = i / (Kg * C * 8), local = i % (Kg * C * 8);
            const uint g = local / (C * 8), v = local % (C * 8);
            const uint col = bx * C + v / 8, j = v % 8;
            packed[i] = w[col * (K / 16) + g * 8 + j];
            """, ensureRowContiguous: true)

    static func streamPack(_ weight: MLXArray, k: Int, columns: Int) -> MLXArray {
        let n = weight.dim(0)
        return streamRepack(
            [weight], template: [("K", k), ("N", n), ("C", columns)],
            grid: (n * k / 16, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[n, k / 16]], outputDTypes: [.uint32])[0]
    }

    private struct MetadataShape: Hashable {
        let kind: Int
        let k: Int
        let rows: Int
        let negative: Bool
        let ns: [Int]
    }
    nonisolated(unsafe) private static var metadataChoices: [MetadataShape: Bool] = [:]

    static func metadataUsesShuffle(kind: Int, k: Int, rows: Int, negative: Bool, ns: [Int]) -> Bool {
        metadataChoices[MetadataShape(kind: kind, k: k, rows: rows, negative: negative, ns: ns)] ?? false
    }

    fileprivate static func qualifyMetadata(
        kind: Int, k: Int, rows: Int, negative: Bool, ns: [Int],
        label: String = "verify streamed metadata", methods: String = "[direct,shuffle]", verdict: String = "shuffle",
        build: (Bool) -> [MLXArray]?
    ) {
        let key = MetadataShape(kind: kind, k: k, rows: rows, negative: negative, ns: ns)
        guard metadataChoices[key] == nil else { return }
        metadataChoices[key] = false
        let coldStart = DispatchTime.now().uptimeNanoseconds
        let exact = (try? withError { () -> Bool in
            guard let direct = build(false), let shuffled = build(true), direct.count == shuffled.count else { return false }
            var checks: [MLXArray] = []
            for (a, b) in zip(direct, shuffled) {
                checks.append(MLX.all(a.view(dtype: .uint8) .== b.view(dtype: .uint8)))
                if a.dtype == .float32 {
                    checks.append(MLX.all(MLX.isFinite(a)))
                    checks.append(MLX.all(MLX.isFinite(b)))
                }
            }
            eval(checks)
            return checks.allSatisfy { $0.item(Bool.self) }
        }) ?? false
        func time(_ shared: Bool) -> UInt64? {
            try? withError {
                Stream.gpu.synchronize()
                let start = DispatchTime.now().uptimeNanoseconds
                guard let roots = build(shared) else { return nil }
                eval(roots)
                Stream.gpu.synchronize()
                let end = DispatchTime.now().uptimeNanoseconds
                withExtendedLifetime(roots) {}
                return end > start ? end - start : nil
            }
        }
        var complete = exact
        var warm: [UInt64] = [0, 0]
        var samples: [[UInt64]] = [[], []]
        var legs: [[[UInt64]]] = [[], []]
        if complete {
            for method in 0 ..< 2 {
                if let ns = time(method == 1) { warm[method] = ns }
                else { complete = false; break }
            }
        }
        if complete {
            for order in [[0, 1], [1, 0], [0, 1]] {
                var round: [[UInt64]] = [[], []]
                for mirrored in [order, Array(order.reversed())] {
                    for method in mirrored {
                        if let ns = time(method == 1) { round[method].append(ns) }
                        else { complete = false; break }
                    }
                    if !complete { break }
                }
                for method in 0 ..< 2 {
                    legs[method].append(round[method])
                    if round[method].count == 2 { samples[method].append(round[method][0] + round[method][1]) }
                }
                if !complete { break }
            }
        }
        let shared = HadamardQuantizedLinear.streamedSwiGLUChoicePassed(complete: complete, samples: samples)
        metadataChoices[key] = shared
        let cold = DispatchTime.now().uptimeNanoseconds - coldStart
        FileHandle.standardError.write(Data(
            "\(label) kind=\(kind) R\(rows) K\(k) N\(ns) NEG=\(negative) exact=\(exact) complete=\(complete) methods=\(methods) policy=all3_strict warm_ns=\(warm) legs_ns=\(legs) round_sums_ns=\(samples) \(verdict)=\(shared) calibration_ns=\(cold)\n".utf8))
    }

    @discardableResult
    static func prepareMetadata(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        streamedWeights: [MLXArray], swiglu: Bool = false,
        dense: (x: MLXArray, wb: MLXArray, wa: MLXArray)? = nil
    ) -> (negative: Bool, shuffle: Bool)? {
        guard route == .nax, naxFast, streamReady, activation.rows == 16,
            usesForms(k: activation.k, layers: layers.map { ($0.weight, $0.scalesT, $0.biasesT) }),
            layers.count == streamedWeights.count
        else { return nil }
        var choice: (negative: Bool, shuffle: Bool)? = nil
        let plain = layers.map { (weight: $0.weight, scalesT: $0.scalesT, biasesT: $0.biasesT) }
        let ns = layers.map { $0.weight.dim(0) }
        let k = activation.k, negative = negativeForms && layers.allSatisfy { $0.negative }
        qualifyMetadata(kind: 0, k: k, rows: 16, negative: negative, ns: ns) { shared in
            runFast(codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                layers: plain, m: 16, k: k, negative: negative, forms: true,
                streamedWeights: streamedWeights, metadataShuffle: shared)
        }
        if swiglu, k == 4096, ns == [12288, 12288], canStreamedSwiGLU {
            qualifyMetadata(kind: 1, k: k, rows: 16, negative: negative, ns: ns) { shared in
                [runStreamedSwiGLU(codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                    layers: plain, streamedWeights: streamedWeights, negative: negative, metadataShuffle: shared)]
            }
            choice = (negative, metadataUsesShuffle(kind: 1, k: k, rows: 16, negative: negative, ns: ns))
        }
        if let dense, k == 4096, layers.count == 2, VerifyDenseBA.fusedMatched {
            qualifyMetadata(kind: 2, k: k, rows: 16, negative: negative, ns: ns + [dense.wb.dim(0)]) { shared in
                VerifyDenseBA.fused(codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                    layers: plain, m: 16, k: k, negative: negative, x: dense.x, wb: dense.wb, wa: dense.wa,
                    streamedWeights: streamedWeights, metadataShuffle: shared)
            }
        }
        return choice
    }

    static func prepareHeadMetadata(
        _ activation: PromptInt8Activation, weight: MLXArray, copied: MLXArray,
        scalesT: MLXArray, biasesT: MLXArray
    ) {
        guard streamReady, screenAvailable, activation.rows == 15, activation.k == 4096,
            weight.dim(0) == 98304 else { return }
        qualifyMetadata(kind: 3, k: 4096, rows: 15, negative: true, ns: [98304, 16]) { shared in
            screen(activation, weight: copied, scalesT: scalesT, biasesT: biasesT,
                streamed: true, groups: 16, metadataShuffle: shared).map { [$0] }
        }
        qualifyMetadata(kind: 4, k: 4096, rows: 15, negative: true, ns: [98304]) { shared in
            draftHeadPartials(activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
                negative: true, streamedWeight: copied, metadataShuffle: shared).map { [$0.values, $0.tags] }
        }
        qualifyMetadata(kind: 5, k: 4096, rows: 15, negative: true, ns: [98304, 16]) { shared in
            draftScreenPartials(activation, weight: copied, scalesT: scalesT, biasesT: biasesT,
                streamed: true, groups: 16, metadataShuffle: shared).map { [$0.values, $0.tags] }
        }
    }

    static let wideStoreAllowed: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_WIDESTORE"]?.lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    nonisolated(unsafe) private static var wideStore = false

    private static func streamLoad(columns: Int) -> String {
        "for (int t = 0; t < \(columns / 4); t += 4) { const uint4 u = *((const device uint4*)(wcol + (t / 4) * Kg * 128 + gl * 128 + lane * 4)); wv[t] = u.x; wv[t + 1] = u.y; wv[t + 2] = u.z; wv[t + 3] = u.w; }"
    }

    private static func streamStore(columns: Int, wide: Bool) -> String {
        if wide && (columns == 32 || columns == 64 || columns == 128) {
            return "if ((t & 1) != 0) { const uint col = (lane >> 1) + 16u * uint(t / 4); const uint word = (lane & 1u) * 4u + uint(t % 4) - 1u; const uint32_t p = wv[t - 1]; const uint4 lo = uint4(p & 0x03030303u, (p >> 2) & 0x03030303u, (p >> 4) & 0x03030303u, (p >> 6) & 0x03030303u); const uint4 hi = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u); threadgroup ulong4* target = (threadgroup ulong4*)(bs[sg] + col * 32u + word * 4u); *target = ulong4(as_type<ulong2>(lo), as_type<ulong2>(hi)); }"
        }
        return "const uint col = (lane >> 1) + 16u * uint(t / 4); const uint word = (lane & 1u) * 4u + uint(t % 4); threadgroup uint4* target = (threadgroup uint4*)(bs[sg] + col * 32u + word * 4u); *target = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);"
    }

    private static func stagingTwinSource(wide: Bool) -> String {
        let store = streamStore(columns: 32, wide: wide)
            .replacingOccurrences(of: "threadgroup ", with: "device ")
            .replacingOccurrences(of: "bs[sg]", with: "tile")
        return """
            const uint lane = thread_index_in_simdgroup;
            const int bx = int(threadgroup_position_in_grid.x);
            const int gl = int(threadgroup_position_in_grid.y);
            const int Kg = K / 128;
            const device uint32_t* wcol = w + (size_t)bx * (2) * Kg * 128;
            device uint32_t* tile = out + ((size_t)bx * Kg + gl) * 1024;
            uint32_t wv[8];
            \(streamLoad(columns: 32))
            for (int t = 0; t < 8; t++) {
              const uint32_t v = wv[t];
              \(store)
            }
            """
    }

    private static let stagingTwinKernels = [false, true].map { (wide: Bool) -> MLXFast.MLXFastKernel in
        MLXFast.metalKernel(
            name: wide ? "bonsai_verify_stream_staging_wide" : "bonsai_verify_stream_staging_pair",
            inputNames: ["w"], outputNames: ["out"], source: stagingTwinSource(wide: wide),
            ensureRowContiguous: true)
    }

    private static func checkWideStore() -> Bool {
        let n = 64
        var matched = true
        let failed = OSAllocatedUnfairLock(initialState: false)
        withErrorHandler({ _ in failed.withLock { $0 = true } }) {
            for k in [4096, 12288] where matched {
                let kg = k / 128
                let words: [UInt32] = (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                    let value: UInt64 = UInt64(i + 7) &* 2_654_435_761
                    return UInt32(truncatingIfNeeded: value)
                }
                let packed = streamPack(MLXArray(words, [n, k / 16]), k: k, columns: 16)
                let tiles: [MLXArray] = stagingTwinKernels.map { kernel in
                    kernel([packed], template: [("K", k)], grid: (n / 32 * 32, kg, 1), threadGroup: (32, 1, 1),
                        outputShapes: [[n / 32 * kg * 1024]], outputDTypes: [.uint32], initValue: 255)[0]
                }
                let same = MLX.all(tiles[0] .== tiles[1])
                let full = MLX.all(tiles[0] .!= MLXArray(UInt32(255)))
                eval(same, full)
                let didFail = failed.withLock { $0 }
                matched = !didFail && same.item(Bool.self) && full.item(Bool.self)
            }
        }
        let didFail = failed.withLock { $0 }
        return matched && !didFail
    }

    static func prepareWideStore() {
        guard wideStoreAllowed else {
            FileHandle.standardError.write(Data("int8 verify streamed 32-B tile stores: off (DARKBLOOM_BONSAI_VERIFY_WIDESTORE=0)\n".utf8))
            return
        }
        wideStore = checkWideStore()
        FileHandle.standardError.write(Data("int8 verify streamed 32-B tile stores: \(wideStore ? "passed" : "FAILED; 16-B stores kept") staging twin K 4096/12288 all tiles/groups/lanes\n".utf8))
    }

    private static func makeStreamedBody(_ body: String, columns: Int) -> String? {
        let words = columns / 4
        let oldPointer = "const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;"
        let pointer = "const device uint32_t* wcol = w + (size_t)bx * (\(columns / 16)) * Kg * 128;"
        let oldLoad = "for (int t = 0; t < \(words); t++) { wv[t] = wcol[(size_t)(4 * t) * KW + gl * 8]; }"
        let load = streamLoad(columns: columns)
        let oldStore = "dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);"
        let store = streamStore(columns: columns, wide: wideStore)
        guard body.components(separatedBy: oldPointer).count == 2,
            body.components(separatedBy: oldLoad).count == 3,
            body.components(separatedBy: oldStore).count == 2 else { return nil }
        var result = body.replacingOccurrences(of: oldPointer, with: pointer)
            .replacingOccurrences(of: oldLoad, with: load)
            .replacingOccurrences(of: oldStore, with: store)
        if wideStore && columns == 32 {
            let tile = "threadgroup uint32_t bs[4][32 * 32];"
            guard result.components(separatedBy: tile).count == 2 else { return nil }
            result = result.replacingOccurrences(of: tile, with: "alignas(32) " + tile)
        }
        let prefetchStart = "  if (g + 4 < Kg) {"
        let barrier = "  simdgroup_barrier(mem_flags::mem_threadgroup);"
        let prefetchEnd = barrier + "\n  auto tA ="
        // Two native barriers, but exactly one prefetch boundary. Reject
        // duplicated bare markers as well as duplicated boundary context.
        guard result.components(separatedBy: prefetchStart).count == 2,
            result.components(separatedBy: prefetchEnd).count == 2,
            result.components(separatedBy: barrier).count == 3,
            let start = result.range(of: prefetchStart),
            let end = result.range(of: prefetchEnd,
                range: start.lowerBound ..< result.endIndex) else { return nil }
        let prefetch = String(result[start.lowerBound ..< end.lowerBound])
        result.removeSubrange(start.lowerBound ..< end.lowerBound)
        let issue = "  op.run(tA, B, cT);"
        guard result.components(separatedBy: issue).count == 2 else { return nil }
        return result.replacingOccurrences(of: issue, with: issue + "\n" + prefetch)
    }

    private static let streamedSourcesReady = makeStreamedBody(naxFastBody, columns: 32) != nil
        && makeStreamedBody(naxNarrow16Body, columns: 16) != nil

    static func streamedBody(_ body: String, columns: Int) -> String {
        makeStreamedBody(body, columns: columns) ?? body
    }

    static func streamedFormsSource(layers: Int, metadataShuffle: Bool? = nil) -> String {
        let prefix = naxFastPick(layers: layers).replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
        var body = streamedBody(naxFastBody, columns: 32).replacingOccurrences(of: "(size_t)", with: "(uint)")
        if metadataShuffle == false { return prefix + body }
        let oldMetadata = """
        sA = *(const device half4*)(scol + (uint)gl * N);
        sB = *(const device half4*)(scol + (uint)gl * N + 16);
        if (!NEG) {
          bA = *(const device half4*)(bcol + (uint)gl * N);
          bB = *(const device half4*)(bcol + (uint)gl * N + 16);
        }
        """
        let sharedMetadata = """
        sA = half4(0.0h); sB = half4(0.0h);
        if ((lane & 22u) == 0u) {
          sA = *(const device half4*)(scol + (uint)gl * N);
          sB = *(const device half4*)(scol + (uint)gl * N + 16);
        }
        sA = as_type<half4>(simd_shuffle(as_type<uint2>(sA), ushort(lane & 9u)));
        sB = as_type<half4>(simd_shuffle(as_type<uint2>(sB), ushort(lane & 9u)));
        if (!NEG) {
          bA = half4(0.0h); bB = half4(0.0h);
          if ((lane & 22u) == 0u) {
            bA = *(const device half4*)(bcol + (uint)gl * N);
            bB = *(const device half4*)(bcol + (uint)gl * N + 16);
          }
          bA = as_type<half4>(simd_shuffle(as_type<uint2>(bA), ushort(lane & 9u)));
          bB = as_type<half4>(simd_shuffle(as_type<uint2>(bB), ushort(lane & 9u)));
        }
        """
        let selectedMetadata = metadataShuffle == true ? sharedMetadata : """
        if constexpr (METADATA_SHUFFLE) {
        \(sharedMetadata)
        } else {
        \(oldMetadata)
        }
        """
        for indent in ["  ", "    "] {
            let load = oldMetadata.replacingOccurrences(of: "\n", with: "\n" + indent)
            let shared = selectedMetadata.replacingOccurrences(of: "\n", with: "\n" + indent)
            precondition(body.components(separatedBy: load).count == 2)
            body = body.replacingOccurrences(of: load, with: shared)
        }
        return prefix + body
    }

    private static let streamedKernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_verify_streamed_col\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: streamedFormsSource(layers: layers), header: PromptInt8Matmul.naxHeader,
            ensureRowContiguous: true)
    }
    static func streamedSingle64Source(splitK: Bool = false, pairK: Bool = false, wide: Bool = false) -> String? {
        var body: String = naxWide64Body.replacingOccurrences(of: "const int KW = K / 16;\n", with: "")
        let pointer: String = "const device uint32_t* wcol = w + (uint)(n0 + c) * (uint)KW + j;"
        guard body.components(separatedBy: pointer).count == 2 else { return nil }
        body = body.replacingOccurrences(of: pointer, with: "const device uint32_t* wcol = w + (uint)bx * (uint)Kg * 512u;")
        for group in ["sg", "gl"] {
            let old: String = "for (int t = 0; t < 16; t++) wv[t] = wcol[(uint)(4 * t) * (uint)KW + \(group) * 8u];"
            let load: String = "for (int t = 0; t < 16; t += 4) { const uint4 u = *(const device uint4*)(wcol + (uint)(t / 4) * (uint)Kg * 128u + \(group) * 128u + lane * 4u); wv[t] = u.x; wv[t + 1] = u.y; wv[t + 2] = u.z; wv[t + 3] = u.w; }"
            guard body.components(separatedBy: old).count == 2 else { return nil }
            body = body.replacingOccurrences(of: old, with: load)
        }
        let stage: String = "dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);"
        let columnStore: String = streamStore(columns: wide ? 128 : 64, wide: wideStore)
        guard body.components(separatedBy: stage).count == 2 else { return nil }
        body = body.replacingOccurrences(of: stage,with: columnStore)
            .replacingOccurrences(of: "  const int j = int(lane & 7u);\n",with: "")
            .replacingOccurrences(of: "  const int c = int(lane >> 3);\n",with: "")
            .replacingOccurrences(of: "  threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 8 + j;\n",with: "")
        guard let start = body.range(of: "    if (g + 4 < Kg) {"),
            let end = body.range(of: "    simdgroup_barrier(mem_flags::mem_threadgroup);", range: start.lowerBound ..< body.endIndex)
        else { return nil }
        let prefetch: String = String(body[start.lowerBound ..< end.lowerBound])
        body.removeSubrange(start.lowerBound ..< end.lowerBound)
        body = body.replacingOccurrences(of: "    op.run(tA, B, cT);", with: "    op.run(tA, B, cT);\n" + String(prefetch.dropLast()))
        let pairs: String = """
            const uint meta_pair = uint(g) * uint(N) + uint(n0) + 2u * lane;
            const uint scale_pair = as_type<uint>(*(const device half2*)(scalesT + meta_pair));
            const uint bias_pair = NEG ? 0u : as_type<uint>(*(const device half2*)(biasesT + meta_pair));
        """
        let read: String = """
              const uint shift = 16u * (col & 1u);
              const half s = as_type<half>(ushort(simd_shuffle(scale_pair, ushort(col >> 1)) >> shift));
              const half b = NEG ? -s : as_type<half>(ushort(simd_shuffle(bias_pair, ushort(col >> 1)) >> shift));
        """
        let direct: String = """
              const uint meta = uint(g) * uint(N) + uint(n0) + col;
              const half s = scalesT[meta];
              const half b = NEG ? -s : biasesT[meta];
        """
        guard body.components(separatedBy: pairs).count == 2,
            body.components(separatedBy: read).count == 2 else { return nil }
        body = body.replacingOccurrences(of: pairs + "\n", with: "")
            .replacingOccurrences(of: read, with: direct)
        if splitK {
            guard let end = body.range(of: "  // The tensor op and epilogue no longer read B.") else { return nil }
            body = String(body[..<end.lowerBound])
                .replacingOccurrences(of: "const uint sg = simdgroup_index_in_threadgroup;\nconst uint tid = thread_position_in_threadgroup.x;", with: pairK ? "const uint sg = 2u * threadgroup_position_in_grid.y + simdgroup_index_in_threadgroup;" : "const uint sg = threadgroup_position_in_grid.y;")
                .replacingOccurrences(of: "threadgroup uint32_t bs[4][64 * 32];", with: pairK ? "threadgroup uint32_t bs[2][64 * 32];" : "threadgroup uint32_t bs[1][64 * 32];")
                .replacingOccurrences(of: "bs[sg]", with: pairK ? "bs[simdgroup_index_in_threadgroup]" : "bs[0]")
            body += """
              for (int i = 0; i < 32; i++) {
                const uint p = output_index[i];
                out[((size_t)sg * 16 + (p >> 6)) * N + n0 + (p & 63u)] = acc[i];
              }
            } else {
              for (uint p = lane; p < 1024u; p += 32u)
                out[((size_t)sg * 16 + (p >> 6)) * N + n0 + (p & 63u)] = as_type<float>(0x7fc00000u);
            }
            """
        }
        if wide {
            let widths: [(String,String)] = [
                ("bx * 64","bx * 128"),("(16, 64, 128,","(16, 128, 128,"),
                ("bs[1][64 * 32]","bs[1][128 * 32]"),("(128, 64)","(128, 128)"),
                ("output_index[32]","output_index[64]"),("cap == 32","cap == 64"),
                ("i < 32","i < 64"),("col < 64","col < 128"),("row * 64 + col","row * 128 + col"),
                ("acc[32]","acc[64]"),("wv[16]","wv[32]"),("t < 16","t < 32"),
                ("Kg * 512u","Kg * 1024u"),("position >> 6","position >> 7"),("position & 63u","position & 127u"),
                ("16 + (p >> 6)","16 + (p >> 7)"),("p & 63u","p & 127u"),("p < 1024u","p < 2048u")]
            for (a,b) in widths {
                guard body.contains(a) else { return nil }
                body = body.replacingOccurrences(of: a,with: b)
            }
        }
        if wideStore {
            let arena: String = "threadgroup uint32_t bs["
            guard body.components(separatedBy: arena).count == 2 else { return nil }
            body = body.replacingOccurrences(of: arena, with: "alignas(32) " + arena)
        }
        return naxFastPick(layers: 1)
            .replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;") + body
    }

    private static let streamedSingle64Kernels: [MLXFast.MLXFastKernel?] = [0,1,2,3].map { (variant: Int) -> MLXFast.MLXFastKernel? in
        guard let source: String = streamedSingle64Source(splitK: variant > 0, pairK: variant == 2, wide: variant == 3) else { return nil }
        return MLXFast.metalKernel(
            name: variant == 0 ? "bonsai_q8_verify_streamed_col_single_n64_v2" : variant == 1 ? "bonsai_q8_verify_streamed_col_single_n64_split_v1" : variant == 2 ? "bonsai_q8_verify_streamed_col_single_n64_pair_v1" : "bonsai_q8_verify_streamed_n128_split_v1",
            inputNames: ["xq","w0","s0","b0","ascale","rsb","ksz"], outputNames: ["out0"],
            source: source, header: PromptInt8Matmul.naxHeader + naxWide64ValidityHeader,
            ensureRowContiguous: true)
    }

    private static let streamedSingle64Fold = MLXFast.metalKernel(
        name: "bonsai_q8_verify_streamed_n64_fold_v1", inputNames: ["partial"], outputNames: ["out"],
        source: """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const uint e = thread_position_in_grid.x;
        if (e >= 16384u) return;
        const device float4* p = (const device float4*)partial;
        float4 v = p[e] + p[e + 16384u];
        v = v + p[e + 32768u];
        v = v + p[e + 49152u];
        ((device float4*)out)[e] = v;
        """)

    static func streamedSingle64(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        negative: Bool, streamedWeight: MLXArray, variant: Int = 0
    ) -> MLXArray? {
        let splitK: Bool = variant > 0
        let k: Int = activation.k
        guard route == .nax, naxFast, streamReady, activation.rows == 16, [4096,12288].contains(k),
            weight.shape == [4096,k/16], streamedWeight.shape == weight.shape, streamedWeight.dtype == .uint32,
            scalesT.shape == [k/128,4096], biasesT.shape == scalesT.shape,
            scalesT.dtype == .float16, biasesT.dtype == .float16,
            usesForms(k: k,layers: [(weight,scalesT,biasesT)]), let kernel = streamedSingle64Kernels[variant] else { return nil }
        let shape: MLXArray = MLXArray([Int32(k),Int32(16),Int32(4096)])
        let inputs: [MLXArray] = [activation.codes,streamedWeight,scalesT,biasesT,activation.ascale,activation.rsb,shape]
        let arguments: [(String,any KernelTemplateArg)] = [("ROWS",16),("NEG",negative && negativeForms ? 1 : 0),("KC",k)]
        let columns: Int = variant == 3 ? 128 : 64
        let threads: Int = variant == 2 ? 64 : splitK ? 32 : 128
        let pairs: Int = variant == 2 ? 2 : splitK ? 4 : 1
        let output: MLXArray = kernel(inputs,template: arguments,
            grid: (4096/columns*threads,pairs,1),threadGroup: (threads,1,1),
            outputShapes: [splitK ? [4,16,4096] : [16,4096]],outputDTypes: [.float32])[0]
        if !splitK { return output }
        return streamedSingle64Fold([output],grid: (16384,1,1),threadGroup: (128,1,1),
            outputShapes: [[16,4096]],outputDTypes: [.float32])[0]
    }

    static func streamedSwiGLUSource(metadataShuffle: Bool? = nil) -> String {
        let source = streamedFormsSource(layers: 1, metadataShuffle: metadataShuffle)
        let start = source.range(of: "const int Kg = K / 128;")!
        var body = String(source[start.lowerBound...])
        let scratch = "threadgroup uint32_t bs[4][32 * 32];"
        let selectedScratch = wideStore ? "alignas(32) " + scratch : scratch
        let store = "*(device float4*)(out + (uint)(e >> 3) * N + n0 + 4 * (e & 7)) = good ? v : float4(as_type<float>(0x7fc00000u));"
        precondition(body.components(separatedBy: selectedScratch).count == 2)
        precondition(body.components(separatedBy: store).count == 2)
        body = body.replacingOccurrences(of: selectedScratch, with: "")
            .replacingOccurrences(of: store, with: """
            const float4 value = good ? v : float4(as_type<float>(0x7fc00000u));
            if (layer == 0) {
              gateValue = value;
            } else {
              const float4 g = gateValue, u = value;
              const float4 sigmoid_e = 1.0f / (1.0f + metal::exp(metal::abs(g)));
              const float4 sigmoid_g = select(1.0f - sigmoid_e, sigmoid_e, g < 0.0f);
              *(device float4*)(out + (uint)(e >> 3) * N + n0 + 4 * (e & 7)) = (g * sigmoid_g) * u;
            }
            """)
        return """
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            const int K = KC, N = ksz[2];
            const int bx = int(threadgroup_position_in_grid.x);
            \(selectedScratch)
            float4 gateValue = float4(0.0f);
            for (int layer = 0; layer < 2; layer++) {
              const device uint32_t* w = layer == 0 ? w0 : w1;
              const device half* scalesT = layer == 0 ? s0 : s1;
              const device half* biasesT = layer == 0 ? b0 : b1;

            """ + body + "\n  if (layer == 0) threadgroup_barrier(mem_flags::mem_threadgroup);\n}\n"
    }

    private static let streamedSwiGLUKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_streamed_swiglu",
        inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "rsb", "ksz"],
        outputNames: ["out"], source: streamedSwiGLUSource(),
        header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)

    private static let streamedSwiGLUReference = MLXFast.metalKernel(
        name: "bonsai_verify_swiglu_reference", inputNames: ["gate", "up"], outputNames: ["out"],
        source: """
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            const uint i = thread_position_in_grid.x;
            const float4 g = ((const device float4*)gate)[i], u = ((const device float4*)up)[i];
            const float4 e = 1.0f / (1.0f + metal::exp(metal::abs(g)));
            const float4 sigmoid_g = select(1.0f - e, e, g < 0.0f);
            ((device float4*)out)[i] = (g * sigmoid_g) * u;
            """, ensureRowContiguous: true)

    nonisolated(unsafe) private static var streamedSwiGLUReady = false

    static var canStreamedSwiGLU: Bool {
        route == .nax && naxFast && naxShared && streamReady && streamedSwiGLUReady
            && formsAllowed && (formsMatched & formBit(k: 4096, layers: 2)) != 0
    }

    private static func runStreamedSwiGLU(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)],
        streamedWeights: [MLXArray], negative: Bool, metadataShuffle: Bool? = nil
    ) -> MLXArray {
        let n = layers[0].weight.dim(0)
        let dims = MLXArray([Int32(4096), 16, Int32(n)])
        var inputs: [MLXArray] = [codes]
        inputs.append(contentsOf: streamedWeights)
        inputs.append(contentsOf: layers.map { $0.scalesT })
        inputs.append(contentsOf: layers.map { $0.biasesT })
        inputs.append(contentsOf: [ascale, rsb, dims])
        let shared = metadataShuffle ?? metadataUsesShuffle(kind: 1, k: 4096, rows: 16, negative: negative, ns: layers.map { $0.weight.dim(0) })
        return streamedSwiGLUKernel(
            inputs, template: [("ROWS", 16), ("NEG", negative ? 1 : 0), ("KC", 4096), ("METADATA_SHUFFLE", shared)],
            grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[16, n]], outputDTypes: [.float32])[0]
    }

    static func streamedSwiGLU(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        streamedWeights: [MLXArray], metadataChoice: (negative: Bool, shuffle: Bool)? = nil
    ) -> MLXArray? {
        guard canStreamedSwiGLU, activation.rows == 16, activation.k == 4096,
            layers.count == 2, streamedWeights.count == 2,
            layers.allSatisfy({ $0.weight.dim(0) == 12288 && formsFit(k: 4096, n: $0.weight.dim(0)) })
        else { return nil }
        let negative = negativeForms && layers.allSatisfy { $0.negative }
        let shared = metadataChoice?.negative == negative ? metadataChoice?.shuffle : nil
        return runStreamedSwiGLU(
            codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
            layers: layers.map { ($0.weight, $0.scalesT, $0.biasesT) },
            streamedWeights: streamedWeights, negative: negative, metadataShuffle: shared)
    }

    private static func checkStreamedSwiGLU() -> Bool {
        guard naxShared, formsAllowed, (formsMatched & formBit(k: 4096, layers: 2)) != 0 else { return false }
        return (try? withError {
            let k = 4096, n = 64, kg = 32
            let q: [Int8] = (0 ..< 16 * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }
            let w: [UInt32] = (0 ..< n * k / 16).map { UInt32(truncatingIfNeeded: UInt64($0 + 1) &* 2_654_435_761) }
            let sv: [Float16] = (0 ..< kg * n).map { Float16(Float(1 + $0 % 13) / 1024) }
            let bv: [Float16] = (0 ..< kg * n).map { Float16(-Float($0 % 7) / 512) }
            let av: [Float] = (0 ..< 16 * kg).map { Float(1 + $0 % 11) / 256 }
            let rv: [Float] = (0 ..< 16 * kg).map { Float($0 % 17 - 8) / 64 }
            let codes = MLXArray(q, [16, k]), weight = MLXArray(w, [n, k / 16])
            let otherWeight = weight ^ MLXArray(UInt32(0x5555_5555))
            let scales = MLXArray(sv, [kg, n]), biases = MLXArray(bv, [kg, n])
            let ascale = MLXArray(av), rsb = MLXArray(rv)
            let copies = [streamPack(weight, k: k, columns: 16), streamPack(otherWeight, k: k, columns: 16)]
            let signs: [Float] = (0 ..< 12288).map { $0 % 3 == 0 ? -1 : 1 }
            let transform = try SignedBlockHadamard(blockSize: 4096, signs: signs)
            for negative in negativeForms ? [false, true] : [false] {
                var layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)] = []
                for l in 0 ..< 2 {
                    let factor = MLXArray(Float16(l + 1))
                    let scaled = (scales * factor).asType(.float16)
                    let offsets = negative
                        ? (scaled.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
                        : (biases * factor).asType(.float16)
                    layers.append((l == 0 ? weight : otherWeight, scaled, offsets))
                }
                let old = runFast(codes: codes, ascale: ascale, rsb: rsb, layers: layers,
                    m: 16, k: k, negative: negative, forms: true, streamedWeights: copies)
                let actual = runStreamedSwiGLU(codes: codes, ascale: ascale, rsb: rsb,
                    layers: layers, streamedWeights: copies, negative: negative)
                let expected = streamedSwiGLUReference(
                    old, grid: (16 * n / 4, 1, 1), threadGroup: (128, 1, 1),
                    outputShapes: [[16, n]], outputDTypes: [.float32])[0]
                let finite = (old + [actual, expected]).map { MLX.all(MLX.isFinite($0)) }
                let same = MLX.all(actual.view(dtype: .uint32) .== expected.view(dtype: .uint32))
                eval(finite + [same])
                guard finite.allSatisfy({ $0.item(Bool.self) }), same.item(Bool.self) else { return false }
                // Reject all nonfinite output before either quantizer can cast to int8.
                guard transform.verifySwiGLUValueMatches(
                    gate: tiled(old[0], repetitions: [1, 192]), up: tiled(old[1], repetitions: [1, 192]),
                    value: tiled(actual, repetitions: [1, 192])) else { return false }
            }
            return true
        }) ?? false
    }

    static func streamedSwiGLUPartialSource() -> String? {
        let original: String = streamedSwiGLUSource()
        let store: String = "*(device float4*)(out + (uint)(e >> 3) * N + n0 + 4 * (e & 7)) = (g * sigmoid_g) * u;"
        guard original.components(separatedBy: store).count == 2 else { return nil }
        return original.replacingOccurrences(of: store, with: """
            const float4 value4 = (g * sigmoid_g) * u;
            const float4 sign4 = *(const device float4*)(down_signs + n0 + 4 * (e & 7));
            *(device float4*)(out + (uint)(e >> 3) * N + n0 + 4 * (e & 7)) = bonsai_swiglu_signed_radix4(value4, sign4);
            """)
    }

    private static let streamedSwiGLUPartialKernel: MLXFast.MLXFastKernel? = {
        guard let source = streamedSwiGLUPartialSource() else { return nil }
        return MLXFast.metalKernel(
            name: "bonsai_q8_verify_streamed_swiglu_partial4",
            inputNames: ["xq","w0","w1","s0","s1","b0","b1","ascale","rsb","ksz","down_signs"],
            outputNames: ["out"], source: source,
            header: PromptInt8Matmul.naxHeader + SignedBlockHadamard.swiGLUPartial4Header, ensureRowContiguous: true)
    }()

    static func streamedSwiGLUPartial4(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        streamedWeights: [MLXArray], signs: MLXArray, metadataChoice: (negative: Bool, shuffle: Bool)? = nil
    ) -> MLXArray? {
        guard canStreamedSwiGLU, activation.rows == 16, activation.k == 4096,
            layers.count == 2, streamedWeights.count == 2, signs.dtype == .float32, signs.shape == [12288],
            layers.allSatisfy({ $0.weight.dim(0) == 12288 && formsFit(k: 4096,n: 12288) }),
            let kernel = streamedSwiGLUPartialKernel else { return nil }
        let dimensions: MLXArray = MLXArray([Int32(4096),Int32(16),Int32(12288)])
        var inputs: [MLXArray] = [activation.codes]
        inputs.append(contentsOf: streamedWeights)
        inputs.append(contentsOf: layers.map { $0.scalesT })
        inputs.append(contentsOf: layers.map { $0.biasesT })
        inputs.append(contentsOf: [activation.ascale,activation.rsb,dimensions,signs])
        let negative: Bool = negativeForms && layers.allSatisfy { $0.negative }
        let shared = (metadataChoice?.negative == negative ? metadataChoice?.shuffle : nil)
            ?? metadataUsesShuffle(kind: 1,k: 4096,rows: 16,negative: negative,ns: [12288,12288])
        let arguments: [(String, any KernelTemplateArg)] = [("ROWS",16),("NEG",negative ? 1 : 0),("KC",4096),("METADATA_SHUFFLE",shared)]
        return kernel(inputs, template: arguments, grid: (384 * 128,1,1), threadGroup: (128,1,1),
            outputShapes: [[16,12288]], outputDTypes: [.float32])[0]
    }

    private static let streamedNarrowKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_streamed_col_n16", inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxNarrowPick.replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
            + streamedBody(naxNarrow16Body, columns: 16).replacingOccurrences(of: "(size_t)", with: "(uint)"),
        header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)

    static func streamColumns(k: Int, n: Int) -> Int { 16 }

    private static func checkStreamed() -> Bool {
        guard streamAllowed, naxShared, formsAllowed, streamedSourcesReady else { return false }
        var matched = true
        streamFailed = false
        withErrorHandler({ _ in VerifyInt8Matmul.streamFailed = true }) {
            for k in formsK where matched {
                let kg = k / 128, n = 64
                let q: [Int8] = (0 ..< tileRows * k).map { (i: Int) -> Int8 in
                    let centered: Int = (i * 37 + 11) % 255 - 127
                    return Int8(truncatingIfNeeded: centered)
                }
                let w: [UInt32] = (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                    let value: UInt64 = UInt64(i + 1) &* 2_654_435_761
                    return UInt32(truncatingIfNeeded: value)
                }
                let sv: [Float16] = (0 ..< kg * n).map { (i: Int) -> Float16 in
                    let value: Float = Float(1 + i % 13) / 1024
                    return Float16(value)
                }
                let bv: [Float16] = (0 ..< kg * n).map { (i: Int) -> Float16 in
                    let value: Float = -Float(i % 7) / 512
                    return Float16(value)
                }
                let av: [Float] = (0 ..< tileRows * kg).map { (i: Int) -> Float in
                    let value: Float = Float(1 + i % 11) / 256
                    return value
                }
                let rv: [Float] = (0 ..< tileRows * kg).map { (i: Int) -> Float in
                    let centered: Int = i % 17 - 8
                    return Float(centered) / 64
                }
                let codes = MLXArray(q, [tileRows, k]), weight = MLXArray(w, [n, k / 16])
                let scales = MLXArray(sv, [kg, n]), biases = MLXArray(bv, [kg, n])
                let ascale = MLXArray(av), rsb = MLXArray(rv)
                let negativeBias = (scales.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
                let ranges: [[Range<Int>]] = [[0 ..< 64], [0 ..< 32, 32 ..< 64], [0 ..< 32, 32 ..< 64, 0 ..< 32]]
                let packed: [[MLXArray]] = ranges.map { ranges in
                    ranges.map { streamPack(weight[$0], k: k, columns: 16) }
                }
                let narrowWeight = streamPack(weight, k: k, columns: 16)
                eval(packed.flatMap { $0 } + [narrowWeight])
                func equal(_ lhs: [MLXArray], _ rhs: [MLXArray]) -> Bool {
                    let checks: [MLXArray] = zip(lhs, rhs).map { a, b in MLX.all(a.view(dtype: .uint32) .== b.view(dtype: .uint32)) }
                    eval(checks)
                    return !streamFailed && checks.allSatisfy { $0.item(Bool.self) }
                }
                let full = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_STREAM_FULL_CHECK"] == "1"
                let checkRows: [Int] = full ? Array((1 ... tileRows).reversed()) : [tileRows]
                for m in checkRows where matched {
                    for negative in [false, true] where matched {
                        let offsets = negative ? negativeBias : biases
                        for count in 1 ... 3 where matched {
                            guard (formsMatched & formBit(k: k, layers: count)) != 0 else { matched = false; break }
                            let layers = ranges[count - 1].map { (weight: weight[$0], scalesT: scales[0..., $0], biasesT: offsets[0..., $0]) }
                            let a = runFast(codes: codes, ascale: ascale, rsb: rsb, layers: layers, m: m, k: k, negative: negative, forms: true)
                            let b = runFast(codes: codes, ascale: ascale, rsb: rsb, layers: layers, m: m, k: k, negative: negative, forms: true, streamedWeights: packed[count - 1])
                            matched = equal(a, b)
                            if k == 4096, count == 2, matched {
                                let x = codes[0 ..< m].asType(.float32)
                                let wb = MLXArray.ones([32, k], dtype: .float32) / 1024
                                let wa = wb * 2
                                let ref = VerifyDenseBA.fused(codes: codes, ascale: ascale, rsb: rsb, layers: layers, m: m, k: k, negative: negative, x: x, wb: wb, wa: wa)
                                let out = VerifyDenseBA.fused(codes: codes, ascale: ascale, rsb: rsb, layers: layers, m: m, k: k, negative: negative, x: x, wb: wb, wa: wa, streamedWeights: packed[count - 1])
                                matched = equal(ref, out)
                            }
                        }
                        if matched, narrow16FormsK.contains(k) {
                            let a = runNarrow(codes: codes, ascale: ascale, rsb: rsb, weight: weight, scalesT: scales, biasesT: offsets, m: m, k: k, negative: negative, forms: true, compact: true)
                            let b = runNarrow(codes: codes, ascale: ascale, rsb: rsb, weight: weight, scalesT: scales, biasesT: offsets, m: m, k: k, negative: negative, forms: true, compact: true, streamedWeight: narrowWeight)
                            matched = equal([a], [b])
                        }
                    }
                }
            }
        }
        return matched && !streamFailed
    }


    // An opt-in validation pass over the pinned weights, once per real shape.
    // It is load-only and input-independent; timing legs leave it disabled.
    static let checkRealStreams = ProcessInfo.processInfo.environment[
        "DARKBLOOM_BONSAI_VERIFY_STREAM_FULL_CHECK"] == "1"
    private static let checkedStreamShapes = OSAllocatedUnfairLock(initialState: Set<String>())

    static func checkRealStream(
        weight: MLXArray, copied: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        k: Int, negative: Bool
    ) -> Bool {
        let n = weight.dim(0), kg = k / 128
        let shape = "K\(k) N\(n)"
        let first = checkedStreamShapes.withLock { $0.insert(shape).inserted }
        guard first else { return true }
        guard PromptInt8Matmul.checkRegisterStreamOperands(weight: weight, packed: copied, k: k)
        else { return false }
        let q: [Int8] = (0 ..< tileRows * k).map { (i: Int) -> Int8 in
            let v: Int = (i * 37 + 11) % 255 - 127
            return Int8(truncatingIfNeeded: v)
        }
        let av: [Float] = (0 ..< tileRows * kg).map { (i: Int) -> Float in Float(i % 11 + 1) / 256 }
        let rv: [Float] = (0 ..< tileRows * kg).map { (i: Int) -> Float in Float(i % 17 - 8) / 64 }
        let codes = MLXArray(q, [tileRows, k])
        let ascale = MLXArray(av), rsb = MLXArray(rv)
        let layers = [(weight: weight, scalesT: scalesT, biasesT: biasesT)]
        var matched = true
        streamFailed = false
        func equal(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
            let checks: [MLXArray] = zip(a, b).map { x, y in MLX.all(x.view(dtype: .uint32) .== y.view(dtype: .uint32)) }
            eval(checks)
            return !streamFailed && checks.allSatisfy { $0.item(Bool.self) }
        }
        withErrorHandler({ _ in VerifyInt8Matmul.streamFailed = true }) {
            for m in (1 ... tileRows).reversed() where matched {
                let a = runFast(codes: codes, ascale: ascale, rsb: rsb, layers: layers,
                    m: m, k: k, negative: negative, forms: true)
                let b = runFast(codes: codes, ascale: ascale, rsb: rsb, layers: layers,
                    m: m, k: k, negative: negative, forms: true, streamedWeights: [copied])
                matched = equal(a, b)
                if matched, n <= narrowColumns, narrow16FormsK.contains(k) {
                    let a = runNarrow(codes: codes, ascale: ascale, rsb: rsb, weight: weight,
                        scalesT: scalesT, biasesT: biasesT, m: m, k: k, negative: negative,
                        forms: true, compact: true)
                    let b = runNarrow(codes: codes, ascale: ascale, rsb: rsb, weight: weight,
                        scalesT: scalesT, biasesT: biasesT, m: m, k: k, negative: negative,
                        forms: true, compact: true, streamedWeight: copied)
                    matched = equal([a], [b])
                }
            }
        }
        matched = matched && !streamFailed
        FileHandle.standardError.write(Data("verify streamed real: \(matched ? "passed" : "FAILED") \(shape) rows 1...16\n".utf8))
        return matched
    }

    enum Route { case nax, plain, off }

    // The plain kernel maps one row to lane & 7.
    static var rowRange: ClosedRange<Int> { 2 ... (route == .plain ? 8 : 16) }
    static let tileRows = 16

    private static let plainSource = """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const int K = ksz[0]; const int N = ksz[2];
        const int Kg = K / 128;
        const int KW = K / 16;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        const int j = int(lane & 7u);
        const int c = int(lane >> 3);
        threadgroup float red[4][ROWS * 32];
        float acc[8];
        #pragma clang loop unroll(full)
        for (int t = 0; t < 8; t++) { acc[t] = 0.0f; }
        const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;
        for (int g = int(sg); g < Kg; g += 4) {
          float q[ROWS][16];
          #pragma clang loop unroll(full)
          for (int m = 0; m < ROWS; m++) {
            const device char4* src = (const device char4*)(xq + (size_t)m * K + g * 128 + 16 * j);
            #pragma clang loop unroll(full)
            for (int u = 0; u < 4; u++) {
              const char4 v = src[u];
              q[m][4 * u + 0] = float(v.x); q[m][4 * u + 1] = float(v.y);
              q[m][4 * u + 2] = float(v.z); q[m][4 * u + 3] = float(v.w);
            }
          }
          const int jr = min(j, ROWS - 1);
          const float a = j < ROWS ? ascale[jr * Kg + g] : 0.0f;
          const float rb = j < ROWS ? rsb[jr * Kg + g] : 0.0f;
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) {
            const int n = c + 4 * t;
            const uint32_t wv = wcol[(size_t)(4 * t) * KW + g * 8];
            float wf[16];
            #pragma clang loop unroll(full)
            for (int u = 0; u < 4; u++) {
              #pragma clang loop unroll(full)
              for (int b = 0; b < 4; b++) { wf[4 * u + b] = float((wv >> uint(8 * b + 2 * u)) & 3u); }
            }
            float cm = 0.0f;
            #pragma clang loop unroll(full)
            for (int m = 0; m < ROWS; m++) {
              float d = 0.0f;
              #pragma clang loop unroll(full)
              for (int p = 0; p < 16; p++) { d = fma(q[m][p], wf[p], d); }
              d += simd_shuffle_xor(d, ushort(1));
              d += simd_shuffle_xor(d, ushort(2));
              d += simd_shuffle_xor(d, ushort(4));
              cm = (j == m) ? d : cm;
            }
            if (j < ROWS) {
              const float s = float(scalesT[(size_t)g * N + n0 + n]);
              const float bb = NEG ? -s : float(biasesT[(size_t)g * N + n0 + n]);
              const float tt = s * cm;
              acc[t] = fma(bb, rb, fma(a, tt, acc[t]));
            }
          }
        }
        if (j < ROWS) {
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) { red[sg][j * 32 + c + 4 * t] = acc[t]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = int(tid); e < ROWS * 32; e += 128) {
          float v = red[0][e] + red[1][e];
          v = v + red[2][e];
          v = v + red[3][e];
          out[(size_t)(e >> 5) * N + n0 + (e & 31)] = v;
        }
        """

    private static let plainKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_plain",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: plainSource,
        ensureRowContiguous: true)

    // Each lane stages word `lane & 7` of columns `lane >> 3` + 4t (t < 8):
    // the 16 codes of a word become one uint4 of int8 bytes in the staged
    // order. The next group's words are loaded while this group runs.
    static let naxSource = """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const int K = ksz[0]; const int N = ksz[2];
        const int Kg = K / 128;
        const int KW = K / 16;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
        threadgroup uint32_t bs[4][32 * 32];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B((threadgroup int8_t*)bs[sg], dextents<int, 2>(128, 32));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        constexpr int CAP = 16;
        const int cap = int(cT.get_capacity());
        short em[CAP], en[CAP];
        bool live[CAP];
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          em[i] = 16; en[i] = 0;
          if (i < cap) {
            auto ids = cT.get_multidimensional_index(i);
            en[i] = short(ids[0]); em[i] = short(ids[1]);
          }
          live[i] = simd_any(em[i] < ROWS);
          acc[i] = 0.0f;
        }
        const int j = int(lane & 7u);
        const int c = int(lane >> 3);
        const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;
        threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 8 + j;
        uint32_t wv[8];
        #pragma clang loop unroll(full)
        for (int t = 0; t < 8; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + int(sg) * 8]; }
        for (int g = int(sg); g < Kg; g += 4) {
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) {
            const uint32_t v = wv[t];
            dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);
          }
          const half2 sb = half2(scalesT[(size_t)g * N + n0 + lane], biasesT[(size_t)g * N + n0 + lane]);
          const int lr = min(int(lane), ROWS - 1);
          const float2 ar = lane < uint(ROWS) ? float2(ascale[lr * Kg + g], rsb[lr * Kg + g]) : float2(0.0f);
          if (g + 4 < Kg) {
            #pragma clang loop unroll(full)
            for (int t = 0; t < 8; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + (g + 4) * 8]; }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          auto tA = A.template slice<128, 16>(g * 128, 0);
          op.run(tA, B, cT);
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            if (live[i]) {
              const half2 s2 = simd_shuffle(sb, ushort(en[i]));
              const float2 a2 = simd_shuffle(ar, ushort(em[i] & 15));
              const float t = float(s2.x) * float(cT[i]);
              acc[i] = fma(float(s2.y), a2.y, fma(a2.x, t, acc[i]));
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float* red = (threadgroup float*)bs[sg];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          if (em[i] < ROWS) { red[em[i] * 32 + en[i]] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup float* r0 = (const threadgroup float*)bs[0];
        const threadgroup float* r1 = (const threadgroup float*)bs[1];
        const threadgroup float* r2 = (const threadgroup float*)bs[2];
        const threadgroup float* r3 = (const threadgroup float*)bs[3];
        for (int e = int(tid); e < ROWS * 32; e += 128) {
          float v = r0[e] + r1[e];
          v = v + r2[e];
          v = v + r3[e];
          out[(size_t)(e >> 5) * N + n0 + (e & 31)] = cap == CAP ? v : as_type<float>(0x7fc00000u);
        }
        """

    // `naxSource` with the same values reached by fewer instructions: the
    // simdgroup partials are summed and stored four columns at a time (the
    // same `((p0 + p1) + p2) + p3` per element, contraction off; n0 and N
    // are multiples of 32), and with `NEG = 1` (offsets bit for bit the
    // negated scales, checked at init) each lane's offset is `-s` instead
    // of a second load, the same FP16 value.
    static let naxVectorSource = """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const int K = ksz[0]; const int N = ksz[2];
        const int Kg = K / 128;
        const int KW = K / 16;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
        threadgroup uint32_t bs[4][32 * 32];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B((threadgroup int8_t*)bs[sg], dextents<int, 2>(128, 32));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        constexpr int CAP = 16;
        const int cap = int(cT.get_capacity());
        short em[CAP], en[CAP];
        bool live[CAP];
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          em[i] = 16; en[i] = 0;
          if (i < cap) {
            auto ids = cT.get_multidimensional_index(i);
            en[i] = short(ids[0]); em[i] = short(ids[1]);
          }
          live[i] = simd_any(em[i] < ROWS);
          acc[i] = 0.0f;
        }
        const int j = int(lane & 7u);
        const int c = int(lane >> 3);
        const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;
        threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 8 + j;
        uint32_t wv[8];
        #pragma clang loop unroll(full)
        for (int t = 0; t < 8; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + int(sg) * 8]; }
        for (int g = int(sg); g < Kg; g += 4) {
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) {
            const uint32_t v = wv[t];
            dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);
          }
          const half sv = scalesT[(size_t)g * N + n0 + lane];
          const half2 sb = half2(sv, NEG ? -sv : biasesT[(size_t)g * N + n0 + lane]);
          const int lr = min(int(lane), ROWS - 1);
          const float2 ar = lane < uint(ROWS) ? float2(ascale[lr * Kg + g], rsb[lr * Kg + g]) : float2(0.0f);
          if (g + 4 < Kg) {
            #pragma clang loop unroll(full)
            for (int t = 0; t < 8; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + (g + 4) * 8]; }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          auto tA = A.template slice<128, 16>(g * 128, 0);
          op.run(tA, B, cT);
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            if (live[i]) {
              const half2 s2 = simd_shuffle(sb, ushort(en[i]));
              const float2 a2 = simd_shuffle(ar, ushort(em[i] & 15));
              const float t = float(s2.x) * float(cT[i]);
              acc[i] = fma(float(s2.y), a2.y, fma(a2.x, t, acc[i]));
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float* red = (threadgroup float*)bs[sg];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          if (em[i] < ROWS) { red[em[i] * 32 + en[i]] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup float4* r0 = (const threadgroup float4*)bs[0];
        const threadgroup float4* r1 = (const threadgroup float4*)bs[1];
        const threadgroup float4* r2 = (const threadgroup float4*)bs[2];
        const threadgroup float4* r3 = (const threadgroup float4*)bs[3];
        for (int e = int(tid); e < ROWS * 8; e += 128) {
          float4 v = r0[e] + r1[e];
          v = v + r2[e];
          v = v + r3[e];
          *(device float4*)(out + (size_t)(e >> 3) * N + n0 + 4 * (e & 7)) = cap == CAP ? v : float4(as_type<float>(0x7fc00000u));
        }
        """

    // `naxVectorSource` with the destination layout fixed (tuned on the M5
    // Max). On the M5, element i of lane l is (row, column) = (mrow + 8 *
    // ((i >> 2) & 1), nb + (i & 3) + 16 * (i >> 3)) with mrow = ((l >> 1) & 3)
    // + 4 * ((l >> 4) & 1) and nb = 4 * (l & 1) + 8 * ((l >> 3) & 1). Each
    // simdgroup compares that layout with `get_multidimensional_index` once
    // and writes NaN on any difference, so the load-time check rejects the
    // kernel wherever the tensor is laid out otherwise. Rows 8 and up
    // (elements 4-7 and 12-15) are padding for ROWS <= 8 and skipped. Each
    // lane loads its own two four-column scale (and offset) vectors and its
    // row's ascale/rsb a group ahead instead of two shuffles per element; the
    // FP32 steps per element are the vector kernel's, in its order.
    static let naxFastBody = """
        const int Kg = K / 128;
        const int KW = K / 16;
        const int n0 = bx * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
        threadgroup uint32_t bs[4][32 * 32];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B((threadgroup int8_t*)bs[sg], dextents<int, 2>(128, 32));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        const int mrow = int((lane >> 1) & 3u) + 4 * int((lane >> 4) & 1u);
        const int nb = 4 * int(lane & 1u) + 8 * int((lane >> 3) & 1u);
        const int cap = int(cT.get_capacity());
        bool ok = cap == 16;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) {
          if (i < cap) {
            auto ids = cT.get_multidimensional_index(i);
            ok = ok && int(ids[1]) == mrow + 8 * ((i >> 2) & 1) && int(ids[0]) == nb + (i & 3) + 16 * (i >> 3);
          }
        }
        const bool good = simd_all(ok);
        float acc[16];
        #pragma clang loop unroll(full)
        for (int e = 0; e < 16; e++) { acc[e] = 0.0f; }
        const int mr = min(mrow, ROWS - 1);
        const int mr1 = min(mrow + 8, ROWS - 1);
        const device half* scol = scalesT + n0 + nb;
        const device half* bcol = biasesT + n0 + nb;
        const int j = int(lane & 7u);
        const int c = int(lane >> 3);
        const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;
        threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 8 + j;
        uint32_t wv[8];
        half4 sA, sB, bA = half4(0.0h), bB = half4(0.0h);
        float fa, frb, fa1 = 0.0f, frb1 = 0.0f;
        {
          const int gl = int(sg);
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + gl * 8]; }
          sA = *(const device half4*)(scol + (size_t)gl * N);
          sB = *(const device half4*)(scol + (size_t)gl * N + 16);
          if (!NEG) {
            bA = *(const device half4*)(bcol + (size_t)gl * N);
            bB = *(const device half4*)(bcol + (size_t)gl * N + 16);
          }
          fa = ascale[mr * Kg + gl];
          frb = rsb[mr * Kg + gl];
          if (ROWS > 8) { fa1 = ascale[mr1 * Kg + gl]; frb1 = rsb[mr1 * Kg + gl]; }
        }
        for (int g = int(sg); g < Kg; g += 4) {
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) {
            const uint32_t v = wv[t];
            dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);
          }
          const half4 cA = sA, cB = sB;
          const half4 dA = NEG ? -sA : bA, dB = NEG ? -sB : bB;
          const float ca = fa, crb = frb, ca1 = fa1, crb1 = frb1;
          if (g + 4 < Kg) {
            const int gl = g + 4;
            #pragma clang loop unroll(full)
            for (int t = 0; t < 8; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + gl * 8]; }
            sA = *(const device half4*)(scol + (size_t)gl * N);
            sB = *(const device half4*)(scol + (size_t)gl * N + 16);
            if (!NEG) {
              bA = *(const device half4*)(bcol + (size_t)gl * N);
              bB = *(const device half4*)(bcol + (size_t)gl * N + 16);
            }
            fa = ascale[mr * Kg + gl];
            frb = rsb[mr * Kg + gl];
            if (ROWS > 8) { fa1 = ascale[mr1 * Kg + gl]; frb1 = rsb[mr1 * Kg + gl]; }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          auto tA = A.template slice<128, 16>(g * 128, 0);
          op.run(tA, B, cT);
          #pragma clang loop unroll(full)
          for (int e = 0; e < 4; e++) {
            const float t0 = float(cA[e]) * float(cT[e]);
            acc[e] = fma(float(dA[e]), crb, fma(ca, t0, acc[e]));
            const float t1 = float(cB[e]) * float(cT[8 + e]);
            acc[4 + e] = fma(float(dB[e]), crb, fma(ca, t1, acc[4 + e]));
            if (ROWS > 8) {
              const float t2 = float(cA[e]) * float(cT[4 + e]);
              acc[8 + e] = fma(float(dA[e]), crb1, fma(ca1, t2, acc[8 + e]));
              const float t3 = float(cB[e]) * float(cT[12 + e]);
              acc[12 + e] = fma(float(dB[e]), crb1, fma(ca1, t3, acc[12 + e]));
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float* red = (threadgroup float*)bs[sg];
        if (mrow < ROWS) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < 4; e++) {
            red[mrow * 32 + nb + e] = acc[e];
            red[mrow * 32 + nb + 16 + e] = acc[4 + e];
          }
        }
        if (ROWS > 8 && mrow + 8 < ROWS) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < 4; e++) {
            red[(mrow + 8) * 32 + nb + e] = acc[8 + e];
            red[(mrow + 8) * 32 + nb + 16 + e] = acc[12 + e];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup float4* r0 = (const threadgroup float4*)bs[0];
        const threadgroup float4* r1 = (const threadgroup float4*)bs[1];
        const threadgroup float4* r2 = (const threadgroup float4*)bs[2];
        const threadgroup float4* r3 = (const threadgroup float4*)bs[3];
        for (int e = int(tid); e < ROWS * 8; e += 128) {
          float4 v = r0[e] + r1[e];
          v = v + r2[e];
          v = v + r3[e];
          *(device float4*)(out + (size_t)(e >> 3) * N + n0 + 4 * (e & 7)) = good ? v : float4(as_type<float>(0x7fc00000u));
        }
        """

    static let naxWide64Body = """
        const int Kg = K / 128;
        const int KW = K / 16;
        const int n0 = bx * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
        threadgroup uint32_t bs[4][64 * 32];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B((threadgroup int8_t*)bs[sg], dextents<int, 2>(128, 64));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        const int cap = int(cT.get_capacity());
        ushort output_index[32];
        bool ok = cap == 32;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; i++) {
          output_index[i] = 0;
          if (i < cap && bonsai_verify_n64_valid_v1(cT, i, 0)) {
            const auto xy = cT.get_multidimensional_index(i);
            const int col = int(xy[0]), row = int(xy[1]);
            const bool inside = col >= 0 && col < 64 && row >= 0 && row < 16;
            ok = ok && inside;
            if (inside) output_index[i] = ushort(row * 64 + col);
          } else {
            ok = false;
          }
        }
        const bool simd_ok = simd_all(ok);
        if (simd_ok) {
          float acc[32];
          #pragma clang loop unroll(full)
          for (int i = 0; i < 32; i++) acc[i] = 0.0f;
          const int j = int(lane & 7u);
          const int c = int(lane >> 3);
          const device uint32_t* wcol = w + (uint)(n0 + c) * (uint)KW + j;
          threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 8 + j;
          uint32_t wv[16];
          #pragma clang loop unroll(full)
          for (int t = 0; t < 16; t++) wv[t] = wcol[(uint)(4 * t) * (uint)KW + sg * 8u];
          for (int g = int(sg); g < Kg; g += 4) {
            #pragma clang loop unroll(full)
            for (int t = 0; t < 16; t++) {
              const uint32_t v = wv[t];
              dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);
            }
            if (g + 4 < Kg) {
              const uint gl = uint(g + 4);
              #pragma clang loop unroll(full)
              for (int t = 0; t < 16; t++) wv[t] = wcol[(uint)(4 * t) * (uint)KW + gl * 8u];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto tA = A.template slice<128, 16>(g * 128, 0);
            op.run(tA, B, cT);
            const uint meta_pair = uint(g) * uint(N) + uint(n0) + 2u * lane;
            const uint scale_pair = as_type<uint>(*(const device half2*)(scalesT + meta_pair));
            const uint bias_pair = NEG ? 0u : as_type<uint>(*(const device half2*)(biasesT + meta_pair));
            #pragma clang loop unroll(full)
            for (int i = 0; i < 32; i++) {
              const uint position = uint(output_index[i]);
              const uint row = position >> 6, col = position & 63u;
              const uint shift = 16u * (col & 1u);
              const half s = as_type<half>(ushort(simd_shuffle(scale_pair, ushort(col >> 1)) >> shift));
              const half b = NEG ? -s : as_type<half>(ushort(simd_shuffle(bias_pair, ushort(col >> 1)) >> shift));
              const float t = float(s) * float(cT[i]);
              acc[i] = fma(float(b), rsb[row * uint(Kg) + uint(g)], fma(ascale[row * uint(Kg) + uint(g)], t, acc[i]));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }
          // The tensor op and epilogue no longer read B. Its first half holds partials.
          threadgroup float* red = (threadgroup float*)bs[sg];
          #pragma clang loop unroll(full)
          for (int i = 0; i < 32; i++) red[output_index[i]] = acc[i];
        } else {
          threadgroup float* red = (threadgroup float*)bs[sg];
          for (int e = int(lane); e < 16 * 64; e += 32) red[e] = as_type<float>(0x7fc00000u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup float4* r0 = (const threadgroup float4*)bs[0];
        const threadgroup float4* r1 = (const threadgroup float4*)bs[1];
        const threadgroup float4* r2 = (const threadgroup float4*)bs[2];
        const threadgroup float4* r3 = (const threadgroup float4*)bs[3];
        for (int e = int(tid); e < ROWS * 16; e += 128) {
          float4 v = r0[e] + r1[e];
          v = v + r2[e];
          v = v + r3[e];
          *(device float4*)(out + (uint)(e >> 4) * (uint)N + n0 + 4 * (e & 15)) = v;
        }
        """

    static let compactHeadEpilogue = """
        for (int e = int(tid); e < ROWS * 8; e += 128) {
          float4 v = r0[e] + r1[e];
          v = v + r2[e];
          v = v + r3[e];
          if (!good) v = float4(as_type<float>(0x7fc00000u));
          const uint row = uint(e >> 3);
          const uint tile = uint(n0 / 32);
          const uint sublane = lane & 7u;
          const uint tiles = uint(N / 32);
          float v0 = v.x, v1 = v.y, v2 = v.z, v3 = v.w;
          uint i0 = uint(n0 + 4 * (e & 7)), i1 = i0 + 1, i2 = i0 + 2, i3 = i0 + 3;
          uint bad = all(isfinite(v)) ? 0u : 1u;
          for (uint stride = 4; stride > 0; stride >>= 1)
            bad |= simd_shuffle_xor(bad, ushort(stride));
          // Five compare-swaps sort each lane's four values once.
          if (hadamard_top2_better(v1, i1, v0, i0)) {
            const float t = v0; v0 = v1; v1 = t; const uint j = i0; i0 = i1; i1 = j;
          }
          if (hadamard_top2_better(v3, i3, v2, i2)) {
            const float t = v2; v2 = v3; v3 = t; const uint j = i2; i2 = i3; i3 = j;
          }
          if (hadamard_top2_better(v2, i2, v0, i0)) {
            const float t = v0; v0 = v2; v2 = t; const uint j = i0; i0 = i2; i2 = j;
          }
          if (hadamard_top2_better(v3, i3, v1, i1)) {
            const float t = v1; v1 = v3; v3 = t; const uint j = i1; i1 = i3; i3 = j;
          }
          if (hadamard_top2_better(v2, i2, v1, i1)) {
            const float t = v1; v1 = v2; v2 = t; const uint j = i1; i1 = i2; i2 = j;
          }
          uint remaining = 4;
          for (int rank = 0; rank < 5; rank++) {
            bool have = remaining != 0;
            float best_value = v0;
            uint best_id = i0;
            for (uint stride = 4; stride > 0; stride >>= 1) {
              const bool other_have = simd_shuffle_xor(uint(have), ushort(stride)) != 0;
              const float other_value = simd_shuffle_xor(best_value, ushort(stride));
              const uint other_id = simd_shuffle_xor(best_id, ushort(stride));
              if (other_have && (!have || hadamard_top2_better(other_value, other_id, best_value, best_id))) {
                have = true; best_value = other_value; best_id = other_id;
              }
            }
            if (sublane == 0) {
              const uint p = row * tiles + tile;
              if (rank < 4) { out0[p * 4 + uint(rank)] = best_value; part_ids[p * 4 + uint(rank)] = int(best_id); }
              else { omitted[p] = best_value; part_bad[p] = bad ? 1 : 0; }
            }
            if (remaining != 0 && best_id == i0) {
              v0 = v1; v1 = v2; v2 = v3;
              i0 = i1; i1 = i2; i2 = i3;
              remaining--;
            }
          }
        }
        """

    static let draftHeadRetainSource = """
        for (int e = int(tid); e < ROWS * 8; e += 128) {
          float4 v = r0[e] + r1[e];
          v = v + r2[e];
          v = v + r3[e];
          v = good ? v : float4(as_type<float>(0x7fc00000u));
          for (int column = 0; column < 4; column++) {
            const float value = v[column];
            if (value > retained[3][column]) {
              int rank = 3;
              while (rank > 0 && value > retained[rank - 1][column]) {
                retained[rank][column] = retained[rank - 1][column];
                stepTags[rank][column] = stepTags[rank - 1][column];
                rank--;
              }
              retained[rank][column] = value;
              stepTags[rank][column] = uchar(step);
            }
          }
        }
        """

    static func draftHeadPartialsSource(streamed: Bool, metadataShuffle: Bool? = nil) -> String {
        let original = streamed ? streamedFormsSource(layers: 1, metadataShuffle: metadataShuffle) : naxFormsSource(layers: 1)
        let store = """
            for (int e = int(tid); e < ROWS * 8; e += 128) {
              float4 v = r0[e] + r1[e];
              v = v + r2[e];
              v = v + r3[e];
              *(device float4*)(out + (uint)(e >> 3) * N + n0 + 4 * (e & 7)) = good ? v : float4(as_type<float>(0x7fc00000u));
            }
            """
        let start = "int bx = int(threadgroup_position_in_grid.x);"
        precondition(original.components(separatedBy: start).count == 2)
        precondition(original.components(separatedBy: store).count == 2)
        let setup = """
            const int owner = int(threadgroup_position_in_grid.x);
            const int stripe = owner % 64, segment = owner / 64;
            float4 retained[4] = {float4(-INFINITY), float4(-INFINITY), float4(-INFINITY), float4(-INFINITY)};
            uchar4 stepTags[4] = {uchar4(255), uchar4(255), uchar4(255), uchar4(255)};
            for (int step = 0; step < 12; step++) {
            const int bx = stripe + (segment * 12 + step) * 64;
            """
        let body = original.replacingOccurrences(of: start, with: setup)
            .replacingOccurrences(of: store, with: draftHeadRetainSource)
        return body + """
            if (step < 11) { threadgroup_barrier(mem_flags::mem_threadgroup); }
            }
            const uint outputThread = thread_position_in_threadgroup.x;
            if (outputThread < ROWS * 8u) {
              const uint row = outputThread >> 3u;
              const uint first = uint(stripe) * 32u + 4u * (outputThread & 7u);
              for (uint column = 0; column < 4u; column++) {
                const size_t at = ((size_t)row * 2048u + first + column) * 16u + uint(segment) * 4u;
                *(device float4*)(out0 + at) = float4(retained[0][column], retained[1][column], retained[2][column], retained[3][column]);
                *(device uchar4*)(tags + at) = uchar4(stepTags[0][column], stepTags[1][column], stepTags[2][column], stepTags[3][column]);
              }
            }
            """
    }

    private static let draftHeadPartialsKernels: [MLXFast.MLXFastKernel] = [false, true].map { streamed in
        MLXFast.metalKernel(
            name: streamed ? "bonsai_draft_head_top4_segments_stream" : "bonsai_draft_head_top4_segments",
            inputNames: ["xq", "w0", "s0", "b0", "ascale", "rsb", "ksz"],
            outputNames: ["out0", "tags"], source: draftHeadPartialsSource(streamed: streamed),
            header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)
    }

    static func draftHeadPartials(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        negative: Bool, streamedWeight: MLXArray?, metadataShuffle: Bool? = nil
    ) -> (values: MLXArray, tags: MLXArray)? {
        guard route == .nax, naxFast, activation.rows == 15, activation.k == 4096,
            weight.dim(0) == 98304, formsAllowed,
            (formsMatched & formBit(k: 4096, layers: 1)) != 0,
            formsFit(k: 4096, n: 98304)
        else { return nil }
        let dims: MLXArray = MLXArray([Int32(4096), Int32(15), Int32(98304)])
        let inputs: [MLXArray] = [activation.codes, streamedWeight ?? weight, scalesT, biasesT,
            activation.ascale, activation.rsb, dims]
        let template: [(String, any KernelTemplateArg)] = [
            ("ROWS", 15), ("NEG", negative && negativeForms ? 1 : 0), ("KC", 4096),
            ("METADATA_SHUFFLE", metadataShuffle ?? metadataUsesShuffle(kind: 4, k: 4096, rows: 15, negative: negative && negativeForms, ns: [98304]))]
        let outputs: [MLXArray] = draftHeadPartialsKernels[streamedWeight == nil ? 0 : 1](
            inputs, template: template, grid: (256 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[15, 2048, 16], [15, 2048, 16]], outputDTypes: [.float32, .uint8])
        return (outputs[0], outputs[1])
    }

    private static let compactHeadKernel = {
        let store = """
            for (int e = int(tid); e < ROWS * 8; e += 128) {
              float4 v = r0[e] + r1[e];
              v = v + r2[e];
              v = v + r3[e];
              *(device float4*)(out + (size_t)(e >> 3) * N + n0 + 4 * (e & 7)) = good ? v : float4(as_type<float>(0x7fc00000u));
            }
            """
        let original = naxFastSource(layers: 1)
        precondition(original.components(separatedBy: store).count == 2)
        let source = original.replacingOccurrences(of: store, with: compactHeadEpilogue)
            .replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
            .replacingOccurrences(of: "(size_t)", with: "(uint)")
        return MLXFast.metalKernel(
            name: "bonsai_q8_verify_head_top4_nax_front_v1_forms",
            inputNames: ["xq", "w0", "s0", "b0", "ascale", "rsb", "ksz"],
            outputNames: ["out0", "part_ids", "omitted", "part_bad"], source: source,
            header: PromptInt8Matmul.naxHeader + Float16ConstantQMV.topTwoHeader,
            ensureRowContiguous: true)
    }()

    static func compactHeadAvailable(rows: Int, k: Int, n: Int) -> Bool {
        route == .nax && naxFast && rows == 16 && k == 4096 && n % 64 == 0
            && formsAllowed && (formsMatched & formBit(k: k, layers: 1)) != 0 && formsFit(k: k, n: n)
    }

    static func compactHead(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        negative: Bool
    ) -> (ids: MLXArray, values: MLXArray, omitted: MLXArray, bad: MLXArray)? {
        guard compactHeadAvailable(rows: activation.rows, k: activation.k, n: weight.dim(0))
        else { return nil }
        let tiles = weight.dim(0) / 32
        let result = compactHeadKernel(
            [activation.codes, weight, scalesT, biasesT, activation.ascale, activation.rsb,
             MLXArray([Int32(activation.k), Int32(activation.rows), Int32(weight.dim(0))])],
            template: [("ROWS", 16), ("NEG", negative && negativeForms ? 1 : 0), ("KC", activation.k)],
            grid: (tiles * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[16, tiles, 4], [16, tiles, 4], [16, tiles], [16, tiles]],
            outputDTypes: [.float32, .int32, .float32, .int32])
        return (result[1], result[0], result[2], result[3])
    }

    static func naxFastSource(layers: Int) -> String {
        naxFastPick(layers: layers) + naxFastBody
    }

    static func naxFormsSource(layers: Int) -> String {
        naxFastPick(layers: layers).replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
            + naxFastBody.replacingOccurrences(of: "(size_t)", with: "(uint)")
    }

    private static func naxFastPick(layers: Int) -> String {
        var pick = """
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            const int K = ksz[0];
            int bx = int(threadgroup_position_in_grid.x);
            const device uint32_t* w = w0;
            const device half* scalesT = s0;
            const device half* biasesT = b0;
            device float* out = out0;
            int N = ksz[2];

            """
        for l in 1 ..< layers {
            pick += "if (bx >= N / 32) { bx -= N / 32; w = w\(l); scalesT = s\(l); biasesT = b\(l); out = out\(l); N = ksz[\(2 + l)]; }\n"
        }
        return pick
    }

    // `naxFastBody` over 16 columns per threadgroup instead of 32, for one
    // layer at most `narrowColumns` wide: an N = 4096 call then launches 256
    // threadgroups (6.4 per core on the M5 Max's 40) instead of 128 (3.2).
    // The tensor op, its layout guard and the FP32 steps per output, in
    // order, are `naxFastBody`'s; columns 16-31 of the B tile and their
    // destination elements (8-15) are dead.
    static let naxNarrowBody = """
        const int Kg = K / 128;
        const int KW = K / 16;
        const int n0 = bx * 16;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
        threadgroup uint32_t bs[4][32 * 32];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B((threadgroup int8_t*)bs[sg], dextents<int, 2>(128, 32));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        const int mrow = int((lane >> 1) & 3u) + 4 * int((lane >> 4) & 1u);
        const int nb = 4 * int(lane & 1u) + 8 * int((lane >> 3) & 1u);
        const int cap = int(cT.get_capacity());
        bool ok = cap == 16;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) {
          if (i < cap) {
            auto ids = cT.get_multidimensional_index(i);
            ok = ok && int(ids[1]) == mrow + 8 * ((i >> 2) & 1) && int(ids[0]) == nb + (i & 3) + 16 * (i >> 3);
          }
        }
        const bool good = simd_all(ok);
        float acc[8];
        #pragma clang loop unroll(full)
        for (int e = 0; e < 8; e++) { acc[e] = 0.0f; }
        const int mr = min(mrow, ROWS - 1);
        const int mr1 = min(mrow + 8, ROWS - 1);
        const device half* scol = scalesT + n0 + nb;
        const device half* bcol = biasesT + n0 + nb;
        const int j = int(lane & 7u);
        const int c = int(lane >> 3);
        const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;
        threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 8 + j;
        uint32_t wv[4];
        half4 sA, bA = half4(0.0h);
        float fa, frb, fa1 = 0.0f, frb1 = 0.0f;
        {
          const int gl = int(sg);
          #pragma clang loop unroll(full)
          for (int t = 0; t < 4; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + gl * 8]; }
          sA = *(const device half4*)(scol + (size_t)gl * N);
          if (!NEG) { bA = *(const device half4*)(bcol + (size_t)gl * N); }
          fa = ascale[mr * Kg + gl];
          frb = rsb[mr * Kg + gl];
          if (ROWS > 8) { fa1 = ascale[mr1 * Kg + gl]; frb1 = rsb[mr1 * Kg + gl]; }
        }
        for (int g = int(sg); g < Kg; g += 4) {
          #pragma clang loop unroll(full)
          for (int t = 0; t < 4; t++) {
            const uint32_t v = wv[t];
            dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);
          }
          const half4 cA = sA;
          const half4 dA = NEG ? -sA : bA;
          const float ca = fa, crb = frb, ca1 = fa1, crb1 = frb1;
          if (g + 4 < Kg) {
            const int gl = g + 4;
            #pragma clang loop unroll(full)
            for (int t = 0; t < 4; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + gl * 8]; }
            sA = *(const device half4*)(scol + (size_t)gl * N);
            if (!NEG) { bA = *(const device half4*)(bcol + (size_t)gl * N); }
            fa = ascale[mr * Kg + gl];
            frb = rsb[mr * Kg + gl];
            if (ROWS > 8) { fa1 = ascale[mr1 * Kg + gl]; frb1 = rsb[mr1 * Kg + gl]; }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          auto tA = A.template slice<128, 16>(g * 128, 0);
          op.run(tA, B, cT);
          #pragma clang loop unroll(full)
          for (int e = 0; e < 4; e++) {
            const float t0 = float(cA[e]) * float(cT[e]);
            acc[e] = fma(float(dA[e]), crb, fma(ca, t0, acc[e]));
            if (ROWS > 8) {
              const float t2 = float(cA[e]) * float(cT[4 + e]);
              acc[4 + e] = fma(float(dA[e]), crb1, fma(ca1, t2, acc[4 + e]));
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float* red = (threadgroup float*)bs[sg];
        if (mrow < ROWS) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < 4; e++) { red[mrow * 16 + nb + e] = acc[e]; }
        }
        if (ROWS > 8 && mrow + 8 < ROWS) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < 4; e++) { red[(mrow + 8) * 16 + nb + e] = acc[4 + e]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup float4* r0 = (const threadgroup float4*)bs[0];
        const threadgroup float4* r1 = (const threadgroup float4*)bs[1];
        const threadgroup float4* r2 = (const threadgroup float4*)bs[2];
        const threadgroup float4* r3 = (const threadgroup float4*)bs[3];
        for (int e = int(tid); e < ROWS * 4; e += 128) {
          float4 v = r0[e] + r1[e];
          v = v + r2[e];
          v = v + r3[e];
          *(device float4*)(out + (size_t)(e >> 2) * N + n0 + 4 * (e & 3)) = good ? v : float4(as_type<float>(0x7fc00000u));
        }
        """

    static let naxNarrow16Body = naxNarrowBody
        .replacingOccurrences(of: "matmul2d_descriptor(16, 32, 128,", with: "matmul2d_descriptor(16, 16, 128,")
        .replacingOccurrences(of: "threadgroup uint32_t bs[4][32 * 32];", with: "threadgroup uint32_t bs[4][16 * 32];")
        .replacingOccurrences(of: "dextents<int, 2>(128, 32));", with: "dextents<int, 2>(128, 16));")
        .replacingOccurrences(of: "bool ok = cap == 16;", with: "bool ok = cap == 8;")
        .replacingOccurrences(of: "for (int i = 0; i < 16; i++) {", with: "for (int i = 0; i < 8; i++) {")
        .replacingOccurrences(of: "  #pragma clang loop unroll(full)\n  for (int e = 0; e < 4; e++) {\n    const float t0 = float(cA[e]) * float(cT[e]);\n    acc[e] = fma(float(dA[e]), crb, fma(ca, t0, acc[e]));\n    if (ROWS > 8) {\n      const float t2 = float(cA[e]) * float(cT[4 + e]);\n      acc[4 + e] = fma(float(dA[e]), crb1, fma(ca1, t2, acc[4 + e]));\n    }\n  }\n", with: "  if (good) {\n    #pragma clang loop unroll(full)\n    for (int e = 0; e < 4; e++) {\n      const float t0 = float(cA[e]) * float(cT[e]);\n      acc[e] = fma(float(dA[e]), crb, fma(ca, t0, acc[e]));\n      if (ROWS > 8) {\n        const float t2 = float(cA[e]) * float(cT[4 + e]);\n        acc[4 + e] = fma(float(dA[e]), crb1, fma(ca1, t2, acc[4 + e]));\n      }\n    }\n  }\n")

    static let naxNarrowPick = """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const int K = ksz[0];
        const int bx = int(threadgroup_position_in_grid.x);
        const int N = ksz[2];

        """

    private static let naxKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_nax",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxSource,
        header: PromptInt8Matmul.naxHeader,
        ensureRowContiguous: true)

    private static let naxVectorKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_nax_vector",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxVectorSource,
        header: PromptInt8Matmul.naxHeader,
        ensureRowContiguous: true)

    private static let naxFastKernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_verify_matmul_nax_fast\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxFastSource(layers: layers),
            header: PromptInt8Matmul.naxHeader,
            ensureRowContiguous: true)
    }

    private static let naxFormsKernels = (1 ... 3).map { layers in
        MLXFast.metalKernel(
            name: "bonsai_q8_verify_matmul_nax_forms\(layers)",
            inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxFormsSource(layers: layers),
            header: PromptInt8Matmul.naxHeader,
            ensureRowContiguous: true)
    }

    static let formsK = [4096, 12288]
    nonisolated(unsafe) private static var formsMatched: UInt8 = 0

    private static func formBit(k: Int, layers: Int) -> UInt8 {
        guard (1 ... 3).contains(layers) else { return 0 }
        switch k {
        case 4096: return 1 << (layers - 1)
        case 12288: return 1 << (layers + 2)
        default: return 0
        }
    }

    static let formsAllowed: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_FORMS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static func formsFit(k: Int, n: Int) -> Bool {
        n * (k / 16) < 1 << 31 && (k / 128) * n < 1 << 31 && 16 * n < 1 << 31
    }

    private static func usesForms(k: Int, layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)]) -> Bool {
        formsAllowed && (formsMatched & formBit(k: k, layers: layers.count)) != 0
            && layers.allSatisfy { formsFit(k: k, n: $0.weight.dim(0)) }
    }

    private static let naxNarrowKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_nax_narrow",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxNarrowPick + naxNarrowBody,
        header: PromptInt8Matmul.naxHeader,
        ensureRowContiguous: true)

    static var naxNarrowFormsSource: String {
        naxNarrowPick.replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
            + naxNarrowBody.replacingOccurrences(of: "(size_t)", with: "(uint)")
    }

    private static let naxNarrowFormsKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_nax_narrow_forms",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxNarrowFormsSource,
        header: PromptInt8Matmul.naxHeader,
        ensureRowContiguous: true)

    private static let naxNarrow16Kernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_nax_narrow16",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxNarrowPick + naxNarrow16Body,
        header: PromptInt8Matmul.naxHeader,
        ensureRowContiguous: true)

    private static let naxNarrow16FormsKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_nax_narrow16_forms",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxNarrowPick.replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
            + naxNarrow16Body.replacingOccurrences(of: "(size_t)", with: "(uint)"),
        header: PromptInt8Matmul.naxHeader,
        ensureRowContiguous: true)

    static let narrowColumns = 4096
    static let narrowDepths = [4096, 12288]

    nonisolated(unsafe) private static var narrowK = Set<Int>()
    nonisolated(unsafe) private static var narrowFormsK = Set<Int>()
    nonisolated(unsafe) private static var narrow16K = Set<Int>()
    nonisolated(unsafe) private static var narrow16FormsK = Set<Int>()
    nonisolated(unsafe) private static var narrowMask: UInt32 = 0
    @inline(__always)
    private static func narrowShift(k: Int) -> UInt32? {
        if k == 4096 { return 0 }
        if k == 12288 { return 4 }
        return nil
    }
    @inline(__always)
    private static func hasNarrow(k: Int) -> Bool {
        if let s = narrowShift(k: k) { return (narrowMask & (1 << s)) != 0 }
        return narrowK.contains(k)
    }
    @inline(__always)
    private static func hasNarrow16(k: Int) -> Bool {
        if let s = narrowShift(k: k) { return (narrowMask & (1 << (s + 2))) != 0 }
        return narrow16K.contains(k)
    }
    @inline(__always)
    private static func hasNarrowForms(k: Int, compact: Bool) -> Bool {
        if let s = narrowShift(k: k) {
            return (narrowMask & (1 << (s + (compact ? 3 : 1)))) != 0
        }
        return (compact ? narrow16FormsK : narrowFormsK).contains(k)
    }

    nonisolated(unsafe) private static var naxVector = false
    nonisolated(unsafe) private static var negativeForms = false
    nonisolated(unsafe) private static var naxFast = false
    nonisolated(unsafe) private static var naxShared = false

    private static func run(
        _ route: Route, codes: MLXArray, ascale: MLXArray, rsb: MLXArray, weight: MLXArray,
        scalesT: MLXArray, biasesT: MLXArray, m: Int, k: Int, n: Int,
        vector: Bool, negative: Bool
    ) -> MLXArray {
        let dims = MLXArray([Int32(k), Int32(m), Int32(n)])
        let inputs = [codes, weight, scalesT, biasesT, ascale, rsb, dims]
        let rows = [("ROWS", m)] as [(String, any KernelTemplateArg)]
        let neg = rows + [("NEG", negative ? 1 : 0)] as [(String, any KernelTemplateArg)]
        let kernel = route == .nax ? (vector ? naxVectorKernel : naxKernel) : plainKernel
        return kernel(
            inputs, template: route == .nax && !vector ? rows : neg,
            grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [.float32])[0]
    }

    static let naxWide64ValidityHeader = """
        template<typename T>
        inline auto bonsai_verify_n64_valid_v1(const thread T& tensor, int i, int)
            -> decltype(tensor.is_valid_element(i), bool()) {
          return bool(tensor.is_valid_element(i));
        }
        template<typename T>
        inline bool bonsai_verify_n64_valid_v1(const thread T&, int, long) {
          return false;
        }
        """

    static func naxWide64Source(layers: Int) -> String {
        naxFastPick(layers: layers)
            .replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
            .replacingOccurrences(of: "N / 32", with: "N / 64")
            + naxWide64Body
    }

    private static let naxWide64Kernels = (1 ... 3).map { layers in
        var names: [String] = ["xq"]
        names.append(contentsOf: (0 ..< layers).map { "w\($0)" })
        names.append(contentsOf: (0 ..< layers).map { "s\($0)" })
        names.append(contentsOf: (0 ..< layers).map { "b\($0)" })
        names.append(contentsOf: ["ascale", "rsb", "ksz"])
        return MLXFast.metalKernel(
            name: "bonsai_q8_verify_matmul_nax_n64_metadata_pair_v1_l\(layers)",
            inputNames: names,
            outputNames: (0 ..< layers).map { "out\($0)" },
            source: naxWide64Source(layers: layers),
            header: PromptInt8Matmul.naxHeader + naxWide64ValidityHeader,
            ensureRowContiguous: true)
    }

    nonisolated(unsafe) private static var wide64Matched: UInt8 = 0
    nonisolated(unsafe) private static var wide64Failed = false

    private static func runWide64(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)], k: Int, negative: Bool
    ) -> [MLXArray] {
        let ns: [Int] = layers.map { $0.weight.dim(0) }
        var shapeValues: [Int32] = [Int32(k), 16]
        shapeValues.append(contentsOf: ns.map { Int32($0) })
        let dims: MLXArray = MLXArray(shapeValues)
        var inputs: [MLXArray] = [codes]
        inputs.append(contentsOf: layers.map { $0.weight })
        inputs.append(contentsOf: layers.map { $0.scalesT })
        inputs.append(contentsOf: layers.map { $0.biasesT })
        inputs.append(contentsOf: [ascale, rsb, dims])
        let template: [(String, any KernelTemplateArg)] = [("ROWS", 16), ("NEG", negative ? 1 : 0), ("KC", k)]
        let shapes: [[Int]] = ns.map { [16, $0] }
        let dtypes: [DType] = ns.map { _ in .float32 }
        return naxWide64Kernels[layers.count - 1](
            inputs, template: template,
            grid: (ns.reduce(0, +) / 64 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: shapes, outputDTypes: dtypes)
    }

    private static func qualifyWide64(
        k: Int, codes: MLXArray, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        ascale: MLXArray, rsb: MLXArray
    ) {
        for count in 1 ... 3 where (formsMatched & formBit(k: k, layers: count)) != 0 {
            var matched = true
            wide64Failed = false
            withErrorHandler({ _ in VerifyInt8Matmul.wide64Failed = true }) {
                let cases: [Bool] = negativeForms ? [false, true] : [false]
                for negative in cases where matched {
                    var layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)] = []
                    for l in 0 ..< count {
                        let factor: MLXArray = MLXArray(Float16(l + 1))
                        let scaled: MLXArray = (scalesT * factor).asType(.float16)
                        let offsets: MLXArray
                        if negative {
                            offsets = (scaled.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
                        } else {
                            offsets = (biasesT * factor).asType(.float16)
                        }
                        layers.append((weight: weight, scalesT: scaled, biasesT: offsets))
                    }
                    // All fixture Ns are 64, so runFast keeps the original N32 path.
                    let reference: [MLXArray] = runFast(
                        codes: codes, ascale: ascale, rsb: rsb, layers: layers,
                        m: 16, k: k, negative: negative, forms: true)
                    let candidate: [MLXArray] = runWide64(
                        codes: codes, ascale: ascale, rsb: rsb, layers: layers, k: k, negative: negative)
                    let checks: [MLXArray] = zip(reference, candidate).map { pair in
                        let a: MLXArray = pair.0.view(dtype: .uint32)
                        let b: MLXArray = pair.1.view(dtype: .uint32)
                        let same: MLXArray = MLX.all(a .== b)
                        let finiteA: MLXArray = MLX.all(MLX.isFinite(pair.0))
                        let finiteB: MLXArray = MLX.all(MLX.isFinite(pair.1))
                        let finite: MLXArray = finiteA .&& finiteB
                        return same .&& finite
                    }
                    eval(checks)
                    matched = !wide64Failed && checks.count == count && checks.allSatisfy { $0.item(Bool.self) }
                }
            }
            if matched && !wide64Failed { wide64Matched |= formBit(k: k, layers: count) }
        }
        wide64Failed = false
    }


    private static func runFast(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)],
        m: Int, k: Int, negative: Bool, forms: Bool? = nil, streamedWeights: [MLXArray]? = nil, metadataShuffle: Bool? = nil
    ) -> [MLXArray] {
        let ns = layers.map { $0.weight.dim(0) }
        let dims = MLXArray([Int32(k), Int32(m)] + ns.map { Int32($0) })
        let useForms = forms ?? usesForms(k: k, layers: layers)
        if useForms, streamedWeights == nil, m == 16, (wide64Matched & formBit(k: k, layers: layers.count)) != 0,
            ns.allSatisfy({ $0 > 0 && $0 % 64 == 0 && formsFit(k: k, n: $0) }), ns.contains(where: { $0 >= 8192 })
        {
            return runWide64(
                codes: codes, ascale: ascale, rsb: rsb, layers: layers, k: k, negative: negative)
        }
        if useForms {
            var inputs: [MLXArray] = [codes]
            inputs.append(contentsOf: streamedWeights ?? layers.map { $0.weight })
            inputs.append(contentsOf: layers.map { $0.scalesT })
            inputs.append(contentsOf: layers.map { $0.biasesT })
            inputs.append(contentsOf: [ascale, rsb, dims])
            let kernels = streamedWeights == nil ? naxFormsKernels : streamedKernels
            let shared = metadataShuffle ?? metadataUsesShuffle(kind: 0, k: k, rows: m, negative: negative, ns: ns)
            let template = [("ROWS", m), ("NEG", negative ? 1 : 0), ("KC", k), ("METADATA_SHUFFLE", shared)] as [(String, any KernelTemplateArg)]
            return kernels[layers.count - 1](
                inputs, template: template,
                grid: (ns.reduce(0, +) / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: ns.map { [m, $0] }, outputDTypes: ns.map { _ in .float32 })
        }
        let inputs = [codes] + layers.map { $0.weight } + layers.map { $0.scalesT } + layers.map { $0.biasesT }
            + [ascale, rsb, dims]
        let template = [("ROWS", m), ("NEG", negative ? 1 : 0)] as [(String, any KernelTemplateArg)]
        return naxFastKernels[layers.count - 1](
            inputs, template: template,
            grid: (ns.reduce(0, +) / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: ns.map { [m, $0] }, outputDTypes: ns.map { _ in .float32 })
    }

    private static func runNarrow(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray, weight: MLXArray,
        scalesT: MLXArray, biasesT: MLXArray, m: Int, k: Int, negative: Bool, forms: Bool? = nil, compact: Bool = false, streamedWeight: MLXArray? = nil
    ) -> MLXArray {
        let n = weight.dim(0)
        let dims = MLXArray([Int32(k), Int32(m), Int32(n)])
        if forms ?? (formsAllowed && hasNarrowForms(k: k, compact: compact) && formsFit(k: k, n: n)) {
            let stream = compact && streamedWeight != nil
            let kernel = stream ? streamedNarrowKernel : compact ? naxNarrow16FormsKernel : naxNarrowFormsKernel
            let kernelWeight = stream ? streamedWeight! : weight
            return kernel(
                [codes, kernelWeight, scalesT, biasesT, ascale, rsb, dims],
                template: [("ROWS", m), ("NEG", negative ? 1 : 0), ("KC", k)] as [(String, any KernelTemplateArg)],
                grid: (n / 16 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [.float32])[0]
        }
        let kernel = compact ? naxNarrow16Kernel : naxNarrowKernel
        return kernel(
            [codes, weight, scalesT, biasesT, ascale, rsb, dims],
            template: [("ROWS", m), ("NEG", negative ? 1 : 0)] as [(String, any KernelTemplateArg)],
            grid: (n / 16 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [.float32])[0]
    }

    private static func checkNarrow(compact: Bool = false) -> (narrow: Set<Int>, forms: Set<Int>) {
        let n = 128
        var passed = Set<Int>()
        var formsPassed = Set<Int>()
        for k in narrowDepths where !compact || narrowK.contains(k) {
            let kg = k / 128
            let codes = MLXArray(
                (0 ..< tileRows * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [tileRows, k])
            let weight = MLXArray(
                (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                    let value: UInt64 = UInt64(i + 1) &* 2_654_435_761
                    return UInt32(truncatingIfNeeded: value)
                }, [n, k / 16])
            let scaleValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
                let value: Float = Float(1 + index % 13) / 1024
                return Float16(value)
            }
            let scalesT = MLXArray(scaleValues, [kg, n])
            let biasValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
                let value: Float = -Float(index % 7) / 512
                return Float16(value)
            }
            let biasesT = MLXArray(biasValues, [kg, n])
            let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
            let ascaleValues: [Float] = (0 ..< tileRows * kg).map { (index: Int) -> Float in
                let value: Float = Float(1 + index % 11) / 256
                return value
            }
            let ascale = MLXArray(ascaleValues, [tileRows * kg])
            let rsbValues: [Float] = (0 ..< tileRows * kg).map { (index: Int) -> Float in
                let centered: Int = index % 17 - 8
                let value: Float = Float(centered) / 64
                return value
            }
            let rsb = MLXArray(rsbValues, [tileRows * kg])
            checkFailed = false
            var matched = true
            var formsOK = formsAllowed && formsFit(k: k, n: n)
            withErrorHandler({ _ in VerifyInt8Matmul.checkFailed = true }) {
                for m in 2 ... 16 {
                    for negative in [false, true] where matched {
                        let offsets = negative ? negatedT : biasesT
                        let fast = runFast(
                            codes: codes, ascale: ascale, rsb: rsb, layers: [(weight, scalesT, offsets)],
                            m: m, k: k, negative: negative, forms: false)[0]
                        let narrow = runNarrow(
                            codes: codes, ascale: ascale, rsb: rsb, weight: weight,
                            scalesT: scalesT, biasesT: offsets, m: m, k: k, negative: negative, forms: false, compact: compact)
                        let equal = MLX.all(fast.view(dtype: .uint32) .== narrow.view(dtype: .uint32))
                        eval(equal)
                        matched = !checkFailed && equal.item(Bool.self)
                        if matched && formsOK {
                            let narrowForms = runNarrow(
                                codes: codes, ascale: ascale, rsb: rsb, weight: weight,
                                scalesT: scalesT, biasesT: offsets, m: m, k: k, negative: negative, forms: true, compact: compact)
                            let same = MLX.all(fast.view(dtype: .uint32) .== narrowForms.view(dtype: .uint32))
                            eval(same)
                            formsOK = !checkFailed && same.item(Bool.self)
                            checkFailed = false
                        }
                    }
                }
            }
            if matched {
                passed.insert(k)
                if formsOK { formsPassed.insert(k) }
            }
        }
        checkFailed = false
        return (passed, formsPassed)
    }

    nonisolated(unsafe) private static var checkFailed = false
    nonisolated(unsafe) private static var checkNote = "not run"

    @discardableResult
    private static func checkKernels(nax: Bool) -> Bool {
        checkFailed = false
        let m = 4, n = 64, k = 1024, kg = k / 128
        let codes = MLXArray(
            (0 ..< tileRows * k).map { $0 < m * k ? Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) : 0 },
            [tileRows, k])
        let weight = MLXArray(
            (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                let value: UInt64 = UInt64(i + 1) &* 2_654_435_761
                return UInt32(truncatingIfNeeded: value)
            }, [n, k / 16])
        let scaleValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
            let value: Float = Float(1 + index % 13) / 1024
            return Float16(value)
        }
        let scalesT = MLXArray(scaleValues, [kg, n])
        let biasValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
            let value: Float = -Float(index % 7) / 512
            return Float16(value)
        }
        let biasesT = MLXArray(biasValues, [kg, n])
        let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
        let ascaleValues: [Float] = (0 ..< m * kg).map { (index: Int) -> Float in
            let value: Float = Float(1 + index % 11) / 256
            return value
        }
        let ascale = MLXArray(ascaleValues, [m * kg])
        let rsbValues: [Float] = (0 ..< m * kg).map { (index: Int) -> Float in
            let centered: Int = index % 17 - 8
            let value: Float = Float(centered) / 64
            return value
        }
        let rsb = MLXArray(rsbValues, [m * kg])
        func output(_ route: Route, _ offsets: MLXArray, vector: Bool, negative: Bool) -> MLXArray {
            run(route, codes: codes, ascale: ascale, rsb: rsb, weight: weight,
                scalesT: scalesT, biasesT: offsets, m: m, k: k, n: n, vector: vector, negative: negative)
                .view(dtype: .uint32)
        }
        func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
            let equal = MLX.all(a .== b)
            eval(equal)
            return !checkFailed && equal.item(Bool.self)
        }
        func fast(_ offsets: MLXArray, negative: Bool, _ columns: [Range<Int>]) -> MLXArray {
            let outputs = runFast(
                codes: codes, ascale: ascale, rsb: rsb,
                layers: columns.map { (weight: weight[$0], scalesT: scalesT[0..., $0], biasesT: offsets[0..., $0]) },
                m: m, k: k, negative: negative)
            return concatenated(outputs, axis: 1).view(dtype: .uint32)
        }
        var matched = false
        naxVector = false
        negativeForms = false
        naxFast = false
        naxShared = false
        withErrorHandler({ message in
            if !VerifyInt8Matmul.checkFailed { VerifyInt8Matmul.checkNote = String(message.prefix(160)) }
            VerifyInt8Matmul.checkFailed = true
        }) {
            let plain = output(.plain, biasesT, vector: false, negative: false)
            let plainNegated = output(.plain, negatedT, vector: false, negative: false)
            let plainNeg = same(output(.plain, negatedT, vector: false, negative: true), plainNegated)
            guard nax else {
                matched = !checkFailed
                negativeForms = plainNeg
                return
            }
            if same(fast(biasesT, negative: false, [0 ..< n]), plain) {
                matched = true
                naxFast = true
                negativeForms = plainNeg && same(fast(negatedT, negative: true, [0 ..< n]), plainNegated)
                naxShared = same(fast(biasesT, negative: false, [0 ..< 32, 32 ..< 64]), plain)
                    && same(
                        fast(biasesT, negative: false, [0 ..< 32, 32 ..< 64, 0 ..< 32]),
                        concatenated([plain, plain[0..., 0 ..< 32]], axis: 1))
                if checkFailed {
                    checkFailed = false
                    naxShared = false
                }
                return
            }
            checkFailed = false
            if same(output(.nax, biasesT, vector: true, negative: false), plain) {
                matched = true
                naxVector = true
                negativeForms = plainNeg && same(output(.nax, negatedT, vector: true, negative: true), plainNegated)
            } else {
                checkFailed = false
                matched = same(output(.nax, biasesT, vector: false, negative: false), plain)
            }
        }
        if !checkFailed {
            checkNote = !matched ? "outputs differ"
                : "bit-identical (\(nax ? (naxFast ? (naxShared ? "fast, shared" : "fast") : naxVector ? "vector" : "record") : "plain") kernel, offset skip \(negativeForms ? "on" : "off"))"
        }
        return matched
    }

    nonisolated(unsafe) private static var formsFailed = false

    private static func checkForms() -> String {
        formsMatched = 0
        wide64Matched = 0
        guard formsAllowed else { return "off (DARKBLOOM_BONSAI_VERIFY_FORMS=0)" }
        let m = tileRows, n = 64
        for k in formsK {
            let kg = k / 128
            let codes = MLXArray(
                (0 ..< tileRows * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [tileRows, k])
            let weight = MLXArray(
                (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                    let value: UInt64 = UInt64(i + 1) &* 2_654_435_761
                    return UInt32(truncatingIfNeeded: value)
                }, [n, k / 16])
            let scaleValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
                let value: Float = Float(1 + index % 13) / 1024
                return Float16(value)
            }
            let scalesT = MLXArray(scaleValues, [kg, n])
            let biasValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
                let value: Float = -Float(index % 7) / 512
                return Float16(value)
            }
            let biasesT = MLXArray(biasValues, [kg, n])
            let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
            let ascaleValues: [Float] = (0 ..< m * kg).map { (index: Int) -> Float in
                let value: Float = Float(1 + index % 11) / 256
                return value
            }
            let ascale = MLXArray(ascaleValues, [m * kg])
            let rsbValues: [Float] = (0 ..< m * kg).map { (index: Int) -> Float in
                let centered: Int = index % 17 - 8
                let value: Float = Float(centered) / 64
                return value
            }
            let rsb = MLXArray(rsbValues, [m * kg])
            func bits(_ offsets: MLXArray, negative: Bool, _ columns: [Range<Int>], forms: Bool) -> MLXArray {
                let outputs = runFast(
                    codes: codes, ascale: ascale, rsb: rsb,
                    layers: columns.map { (weight: weight[$0], scalesT: scalesT[0..., $0], biasesT: offsets[0..., $0]) },
                    m: m, k: k, negative: negative, forms: forms)
                return concatenated(outputs, axis: 1).view(dtype: .uint32)
            }
            func matches(_ columns: [Range<Int>]) -> Bool {
                formsFailed = false
                var cases = [(biasesT, false)]
                if negativeForms { cases.append((negatedT, true)) }
                for (offsets, negative) in cases {
                    let equal = MLX.all(
                        bits(offsets, negative: negative, columns, forms: true)
                            .== bits(offsets, negative: negative, columns, forms: false))
                    eval(equal)
                    if formsFailed || !equal.item(Bool.self) { return false }
                }
                return !formsFailed
            }
            withErrorHandler({ _ in VerifyInt8Matmul.formsFailed = true }) {
                if matches([0 ..< n]) { formsMatched |= formBit(k: k, layers: 1) }
                if naxShared, matches([0 ..< 32, 32 ..< 64]) { formsMatched |= formBit(k: k, layers: 2) }
                if naxShared, matches([0 ..< 32, 32 ..< 64, 0 ..< 32]) { formsMatched |= formBit(k: k, layers: 3) }
            }
            qualifyWide64(
                k: k, codes: codes, weight: weight, scalesT: scalesT, biasesT: biasesT,
                ascale: ascale, rsb: rsb)
        }
        let wide64Note = formsK.map { k in
            let counts = (1 ... 3).filter { (wide64Matched & formBit(k: k, layers: $0)) != 0 }.map { String($0) }
            return "K \(k): layers " + (counts.isEmpty ? "none" : counts.joined(separator: ","))
        }.joined(separator: "; ")
        FileHandle.standardError.write("int8 verify N64 forms: \(wide64Note)\n".data(using: .utf8)!)
        formsFailed = false
        return "matched " + formsK.map { k in
            let layers = (1 ... 3).filter { (formsMatched & formBit(k: k, layers: $0)) != 0 }.map { String($0) }
            return "K \(k): layers " + (layers.isEmpty ? "none" : layers.joined(separator: ","))
        }.joined(separator: "; ")
    }

    static let route: Route = {
        prepareWideStore()
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_INT8_VERIFY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let architecture = GPU.deviceInfo().architecture
        let chosen: Route
        if ["0", "false", "no", "off"].contains(value ?? "") {
            chosen = .off
        } else if value == "nax" || PromptInt8Matmul.architectureHasNAX(architecture) {
            chosen = checkKernels(nax: true) ? .nax : .off
        } else {
            checkKernels(nax: false)
            chosen = .plain
        }
        let forms = chosen == .nax && naxFast ? "; compile-time K forms: \(checkForms())" : ""
        if chosen == .nax && naxFast {
            let eligible = naxShared && formsAllowed && (formsMatched & formBit(k: 4096, layers: 2)) != 0
            FileHandle.standardError.write(
                "int8 verify dense b/a fold: \(VerifyDenseBA.check(eligible: eligible, negativeForms: negativeForms))\n"
                    .data(using: .utf8)!)
        }
        FileHandle.standardError.write(
            "int8 verify matmul: \(chosen) (\(architecture); nax check: \(checkNote)\(forms))\n"
                .data(using: .utf8)!)
        let narrowValue = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_NARROW"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if chosen == .nax, naxFast, !["0", "false", "no", "off"].contains(narrowValue ?? "") {
            (narrowK, narrowFormsK) = checkNarrow()
            (narrow16K, narrow16FormsK) = checkNarrow(compact: true)
            var mask: UInt32 = 0
            for k in narrowDepths {
                if let s = narrowShift(k: k) {
                    if narrowK.contains(k) { mask |= (1 << s) }
                    if narrowFormsK.contains(k) { mask |= (1 << (s + 1)) }
                    if narrow16K.contains(k) { mask |= (1 << (s + 2)) }
                    if narrow16FormsK.contains(k) { mask |= (1 << (s + 3)) }
                }
            }
            narrowMask = mask
            FileHandle.standardError.write(
                "int8 verify narrow: K \(narrowK.sorted()) of \(narrowDepths); compile-time K forms: K \(narrowFormsK.sorted())\n"
                    .data(using: .utf8)!)
            FileHandle.standardError.write(
                "int8 verify narrow N16: K \(narrow16K.sorted()); compile-time K forms: K \(narrow16FormsK.sorted())\n"
                    .data(using: .utf8)!)
        }
        streamReady = chosen == .nax && naxFast && checkStreamed()
        FileHandle.standardError.write(
            "int8 verify streamed: \(streamReady ? "on (M16 guard, fast/shared/N16/dense-b/a bit-identical)" : "off")\n".data(using: .utf8)!)
        streamedSwiGLUReady = streamReady && checkStreamedSwiGLU()
        FileHandle.standardError.write(
            "int8 verify streamed SwiGLU: \(streamedSwiGLUReady ? "on (M16 K4096 N12288 exact)" : "off")\n".data(using: .utf8)!)
        return chosen
    }()

    static func matmul(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        negative: Bool, streamedWeight: MLXArray? = nil
    ) -> MLXArray? {
        guard route != .off, route != .plain || activation.rows <= 8 else { return nil }
        if route == .nax, naxFast, weight.dim(0) <= narrowColumns, hasNarrow(k: activation.k) {
            return runNarrow(
                codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb, weight: weight,
                scalesT: scalesT, biasesT: biasesT, m: activation.rows, k: activation.k,
                negative: negative && negativeForms, compact: hasNarrow16(k: activation.k), streamedWeight: streamedWeight)
        }
        if route == .nax && naxFast {
            return runFast(
                codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                layers: [(weight, scalesT, biasesT)], m: activation.rows, k: activation.k,
                negative: negative && negativeForms, streamedWeights: streamedWeight.map { [$0] })[0]
        }
        return run(
            route, codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
            weight: weight, scalesT: scalesT, biasesT: biasesT,
            m: activation.rows, k: activation.k, n: weight.dim(0),
            vector: naxVector, negative: negative && negativeForms)
    }

    static func matmul(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        streamedWeights: [MLXArray]? = nil
    ) -> [MLXArray]? {
        guard route != .off, route != .plain || activation.rows <= 8 else { return nil }
        if route == .nax, naxFast, naxShared, (2 ... 3).contains(layers.count) {
            return runFast(
                codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                layers: layers.map { ($0.weight, $0.scalesT, $0.biasesT) },
                m: activation.rows, k: activation.k,
                negative: negativeForms && layers.allSatisfy { $0.negative }, streamedWeights: streamedWeights)
        }
        let outputs = layers.compactMap {
            matmul(activation, weight: $0.weight, scalesT: $0.scalesT, biasesT: $0.biasesT, negative: $0.negative)
        }
        return outputs.count == layers.count ? outputs : nil
    }
    // MARK: The draft-head screen

    static var screenAvailable: Bool { route == .nax && naxFast && negativeForms }

    static func screenSource(streamed: Bool, metadataShuffle: Bool? = nil) -> String? {
        var source: String = streamed ? streamedFormsSource(layers: 1, metadataShuffle: metadataShuffle) : naxFormsSource(layers: 1)
        let swaps: [(String, String)] = [
            ("for (int g = int(sg); g < Kg; g += 4) {", "for (int g = int(sg); g < SG; g += 4) {"),
            ("if (g + 4 < Kg) {", "if (g + 4 < SG) {"),
        ]
        for (old, new) in swaps {
            guard source.components(separatedBy: old).count == 2 else { return nil }
            source = source.replacingOccurrences(of: old, with: new)
        }
        return source
    }

    private static let screenKernels: [MLXFast.MLXFastKernel?] = [false, true].map { streamed in
        screenSource(streamed: streamed).map { source in
            MLXFast.metalKernel(
                name: streamed ? "bonsai_q8_head_screen_streamed" : "bonsai_q8_head_screen",
                inputNames: ["xq", "w0", "s0", "b0", "ascale", "rsb", "ksz"], outputNames: ["out0"],
                source: source, header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)
        }
    }

    static func screen(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        streamed: Bool, groups: Int, metadataShuffle: Bool? = nil
    ) -> MLXArray? {
        guard let kernel = screenKernels[streamed ? 1 : 0] else { return nil }
        let n: Int = weight.dim(0)
        let dims = MLXArray([Int32(activation.k), Int32(activation.rows), Int32(n)])
        let template = [("ROWS", activation.rows), ("NEG", 1), ("KC", activation.k), ("SG", groups),
            ("METADATA_SHUFFLE", metadataShuffle ?? metadataUsesShuffle(kind: 3, k: activation.k, rows: activation.rows, negative: true, ns: [n, groups]))]
            as [(String, any KernelTemplateArg)]
        let inputs: [MLXArray] = [
            activation.codes, weight, scalesT, biasesT, activation.ascale, activation.rsb, dims,
        ]
        return kernel(
            inputs, template: template, grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[activation.rows, n]], outputDTypes: [.float32])[0]
    }

    static let headGatherSource = """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const int K = ksz[0]; const int N = ksz[1]; const int P = ksz[2];
        const int Kg = K / 128;
        const int KW = K / 16;
        const uint lane = thread_index_in_simdgroup;
        const uint pair = thread_position_in_grid.x >> 5;
        if (pair >= uint(ROWS * P)) return;
        const int m = int(pair) / P;
        const int id = ALL ? int(pair) % P : ids[pair];
        if (id < 0 || id >= N) {
          if (lane == 0) { out[pair] = -INFINITY; }
          return;
        }
        const int s = int(lane >> 3);
        const int j = int(lane & 7u);
        const device uint32_t* wrow = w + (size_t)id * KW + j;
        const device char4* xrow = (const device char4*)(xq + (size_t)m * K) + 4 * j;
        float p = 0.0f;
        for (int g = s; g < GL; g += 4) {
          const uint32_t v = wrow[g * 8];
          const device char4* src = xrow + g * 32;
          int d = 0;
          #pragma clang loop unroll(full)
          for (int u = 0; u < 4; u++) {
            const char4 a = src[u];
            const uint c = (v >> uint(2 * u)) & 0x03030303u;
            d += int(a.x) * int(c & 255u) + int(a.y) * int((c >> 8) & 255u)
              + int(a.z) * int((c >> 16) & 255u) + int(a.w) * int(c >> 24);
          }
          d += simd_shuffle_xor(d, ushort(1));
          d += simd_shuffle_xor(d, ushort(2));
          d += simd_shuffle_xor(d, ushort(4));
          const float sc = float(scales[(size_t)id * Kg + g]);
          const float off = NEG ? -sc : float(biases[(size_t)id * Kg + g]);
          const float t = sc * float(d);
          p = fma(off, rsb[m * Kg + g], fma(ascale[m * Kg + g], t, p));
        }
        const float p0 = simd_shuffle(p, ushort(0));
        const float p1 = simd_shuffle(p, ushort(8));
        const float p2 = simd_shuffle(p, ushort(16));
        const float p3 = simd_shuffle(p, ushort(24));
        if (lane == 0) { out[pair] = ((p0 + p1) + p2) + p3; }
        """

    private static let headGatherKernel = MLXFast.metalKernel(
        name: "bonsai_q8_head_gather",
        inputNames: ["xq", "w", "scales", "biases", "ascale", "rsb", "ids", "ksz"], outputNames: ["out"],
        source: headGatherSource, ensureRowContiguous: true)

    static func gather(
        _ activation: PromptInt8Activation, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        ids: MLXArray?, groups: Int, negative: Bool
    ) -> MLXArray {
        let n: Int = weight.dim(0)
        let p: Int = ids?.dim(-1) ?? n
        let dims = MLXArray([Int32(activation.k), Int32(n), Int32(p)])
        let template = [("ROWS", activation.rows), ("NEG", negative), ("GL", groups), ("ALL", ids == nil)]
            as [(String, any KernelTemplateArg)]
        let inputs: [MLXArray] = [
            activation.codes, weight, scales, biases, activation.ascale, activation.rsb,
            ids ?? MLXArray([Int32(0)]), dims,
        ]
        return headGatherKernel(
            inputs, template: template, grid: (activation.rows * p * 32, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[activation.rows, p]], outputDTypes: [.float32])[0]
    }


    static let headGatherTop16Source = """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const int K = ksz[0]; const int N = ksz[1]; const int P = ksz[2];
        const int Kg = K / 128;
        const int KW = K / 16;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int m = int(threadgroup_position_in_grid.x);
        threadgroup float scores[128];
        for (int position = int(sg); position < P; position += 16) {
          const int id = ids[m * P + position];
          if (id >= 0 && id < N) {
            const int s = int(lane >> 3);
            const int j = int(lane & 7u);
            const device uint32_t* wrow = w + (size_t)id * KW + j;
            const device char4* xrow = (const device char4*)(xq + (size_t)m * K) + 4 * j;
            float p = 0.0f;
            for (int g = s; g < GL; g += 4) {
              const uint32_t v = wrow[g * 8];
              const device char4* src = xrow + g * 32;
              int d = 0;
              #pragma clang loop unroll(full)
              for (int u = 0; u < 4; u++) {
                const char4 a = src[u];
                const uint c = (v >> uint(2 * u)) & 0x03030303u;
                d += int(a.x) * int(c & 255u) + int(a.y) * int((c >> 8) & 255u)
                  + int(a.z) * int((c >> 16) & 255u) + int(a.w) * int(c >> 24);
              }
              d += simd_shuffle_xor(d, ushort(1));
              d += simd_shuffle_xor(d, ushort(2));
              d += simd_shuffle_xor(d, ushort(4));
              const float sc = float(scales[(size_t)id * Kg + g]);
              const float off = NEG ? -sc : float(biases[(size_t)id * Kg + g]);
              const float t = sc * float(d);
              p = fma(off, rsb[m * Kg + g], fma(ascale[m * Kg + g], t, p));
            }
            const float p0 = simd_shuffle(p, ushort(0));
            const float p1 = simd_shuffle(p, ushort(8));
            const float p2 = simd_shuffle(p, ushort(16));
            const float p3 = simd_shuffle(p, ushort(24));
            if (lane == 0) { scores[position] = ((p0 + p1) + p2) + p3; }
          } else if (lane == 0) { scores[position] = -INFINITY; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          const int row = m;
          const float NEG = -INFINITY;
          float cv[4]; int ci[4];
          const size_t base = (size_t)row * 128u + lane * 4u;
          for (int k = 0; k < 4; k++) {
            cv[k] = scores[lane * 4u + uint(k)];
            ci[k] = ids[base + k];
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
              top_ids[row * 16 + r] = best_id;
              top_values[row * 16 + r] = m;
              cv[bk] = NEG;
            }
          }
        }

        """
    private static let headGatherTop16Kernel = MLXFast.metalKernel(
        name: "bonsai_q8_head_gather_top16", inputNames: ["xq", "w", "scales", "biases", "ascale", "rsb", "ids", "ksz"],
        outputNames: ["top_ids", "top_values"], source: headGatherTop16Source, ensureRowContiguous: true)

    static func gatherTop16(
        _ activation: PromptInt8Activation, weight: MLXArray, scales: MLXArray, biases: MLXArray, ids: MLXArray
    ) -> (ids: MLXArray, values: MLXArray)? {
        guard activation.rows == 15, activation.k == 4096, weight.shape == [98304, 256],
            weight.dtype == .uint32, scales.shape == [98304, 32], scales.dtype == .float16,
            biases.shape == [98304, 32], biases.dtype == .float16,
            ids.shape == [15, 128], ids.dtype == .int32 else { return nil }
        let dimensions: MLXArray = MLXArray([Int32(4096), Int32(98304), Int32(128)])
        let inputs: [MLXArray] = [activation.codes, weight, scales, biases, activation.ascale, activation.rsb, ids, dimensions]
        let arguments: [(String, any KernelTemplateArg)] = [("ROWS",15),("NEG",true),("GL",32),("ALL",false)]
        let outputs: [MLXArray] = headGatherTop16Kernel(inputs, template: arguments,
            grid: (15 * 512,1,1), threadGroup: (512,1,1),
            outputShapes: [[15,16],[15,16]], outputDTypes: [.int32,.float32])
        return (outputs[0],outputs[1])
    }

    static let headIndexedNAXHeader = """
        inline uint bonsai_head_indexed_id(const device int* ids, int column, int N) {
          const int id = ids[column];
          return uint(id >= 0 && id < N ? id : 0);
        }
        inline half4 bonsai_head_indexed_metadata(
            const device half* values, const device int* ids, int column, int Kg, int g, int N) {
          return half4(values[(size_t)bonsai_head_indexed_id(ids, column, N) * Kg + g],
                       values[(size_t)bonsai_head_indexed_id(ids, column + 1, N) * Kg + g],
                       values[(size_t)bonsai_head_indexed_id(ids, column + 2, N) * Kg + g],
                       values[(size_t)bonsai_head_indexed_id(ids, column + 3, N) * Kg + g]);
        }
        """

    static func headIndexedNAXSource() -> String? {
        var source: String = naxFastBody
        let swaps: [(String, String)] = [
            ("""
                const int n0 = bx * 32;
                """, """
                const int owner = int(threadgroup_position_in_grid.x);
                const int query = owner / 4;
                const int first = (owner % 4) * 32;
                const device int* selected = ids + query * 128 + first;
                """),
            ("""
                const device half* scol = scalesT + n0 + nb;
                const device half* bcol = biasesT + n0 + nb;
                """, """
                // Metadata follows the four candidate IDs, not consecutive vocabulary rows.
                """),
            ("""
                const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;
                """, """
                size_t rowWord[8];
                #pragma clang loop unroll(full)
                for (int t = 0; t < 8; t++) {
                  rowWord[t] = (size_t)bonsai_head_indexed_id(selected, c + 4 * t, N) * KW + j;
                }
                """),
            ("""
                wv[t] = wcol[(size_t)(4 * t) * KW + gl * 8];
                """, """
                wv[t] = w[rowWord[t] + gl * 8];
                """),
            ("""
                sA = *(const device half4*)(scol + (size_t)gl * N);
                """, """
                sA = bonsai_head_indexed_metadata(scales, selected, nb, Kg, gl, N);
                """),
            ("""
                sB = *(const device half4*)(scol + (size_t)gl * N + 16);
                """, """
                sB = bonsai_head_indexed_metadata(scales, selected, nb + 16, Kg, gl, N);
                """),
            ("""
                bA = *(const device half4*)(bcol + (size_t)gl * N);
                """, """
                bA = bonsai_head_indexed_metadata(biases, selected, nb, Kg, gl, N);
                """),
            ("""
                bB = *(const device half4*)(bcol + (size_t)gl * N + 16);
                """, """
                bB = bonsai_head_indexed_metadata(biases, selected, nb + 16, Kg, gl, N);
                """),
            ("""
                  for (int e = 0; e < 4; e++) {
                    const float t0 = float(cA[e]) * float(cT[e]);
                    acc[e] = fma(float(dA[e]), crb, fma(ca, t0, acc[e]));
                    const float t1 = float(cB[e]) * float(cT[8 + e]);
                    acc[4 + e] = fma(float(dB[e]), crb, fma(ca, t1, acc[4 + e]));
                    if (ROWS > 8) {
                      const float t2 = float(cA[e]) * float(cT[4 + e]);
                      acc[8 + e] = fma(float(dA[e]), crb1, fma(ca1, t2, acc[8 + e]));
                      const float t3 = float(cB[e]) * float(cT[12 + e]);
                      acc[12 + e] = fma(float(dB[e]), crb1, fma(ca1, t3, acc[12 + e]));
                    }
                  }
                """, """
                  for (int e = 0; e < 4; e++) {
                    if (good && mrow == query) {
                      const float t0 = float(cA[e]) * float(cT[e]);
                      acc[e] = fma(float(dA[e]), crb, fma(ca, t0, acc[e]));
                      const float t1 = float(cB[e]) * float(cT[8 + e]);
                      acc[4 + e] = fma(float(dB[e]), crb, fma(ca, t1, acc[4 + e]));
                    }
                    if (good && mrow + 8 == query) {
                      const float t2 = float(cA[e]) * float(cT[4 + e]);
                      acc[8 + e] = fma(float(dA[e]), crb1, fma(ca1, t2, acc[8 + e]));
                      const float t3 = float(cB[e]) * float(cT[12 + e]);
                      acc[12 + e] = fma(float(dB[e]), crb1, fma(ca1, t3, acc[12 + e]));
                    }
                  }
                """),
            ("""
                if (mrow < ROWS) {
                """, """
                if (mrow == query) {
                """),
            ("""
                if (ROWS > 8 && mrow + 8 < ROWS) {
                """, """
                if (mrow + 8 == query) {
                """),
            ("""
                = acc[e];
                """, """
                = good ? acc[e] : as_type<float>(0x7fc00000u);
                """),
            ("""
                = acc[4 + e];
                """, """
                = good ? acc[4 + e] : as_type<float>(0x7fc00000u);
                """),
            ("""
                = acc[8 + e];
                """, """
                = good ? acc[8 + e] : as_type<float>(0x7fc00000u);
                """),
            ("""
                = acc[12 + e];
                """, """
                = good ? acc[12 + e] : as_type<float>(0x7fc00000u);
                """),
            ("""
                for (int e = int(tid); e < ROWS * 8; e += 128) {
                  float4 v = r0[e] + r1[e];
                  v = v + r2[e];
                  v = v + r3[e];
                  *(device float4*)(out + (size_t)(e >> 3) * N + n0 + 4 * (e & 7)) = good ? v : float4(as_type<float>(0x7fc00000u));
                }
                """, """
                for (int e = int(tid); e < 8; e += 128) {
                  const int at = query * 8 + e;
                  float4 v = r0[at] + r1[at];
                  v = v + r2[at];
                  v = v + r3[at];
                  #pragma clang loop unroll(full)
                  for (int component = 0; component < 4; component++) {
                    const int id = selected[4 * e + component];
                    out[query * 128 + first + 4 * e + component] = id >= 0 && id < N ? v[component] : -INFINITY;
                  }
                }
                """)
        ]
        let counts: [Int] = [1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        for (index, change) in swaps.enumerated() {
            guard source.components(separatedBy: change.0).count == counts[index] + 1 else { return nil }
            source = source.replacingOccurrences(of: change.0, with: change.1)
        }
        return """
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            const int K = KC;
            const int N = ksz[1];

            """ + "\n" + source
    }

    private static let headIndexedNAXKernel: MLXFast.MLXFastKernel? = {
        guard let source: String = headIndexedNAXSource() else { return nil }
        return MLXFast.metalKernel(
            name: "bonsai_q8_head_indexed_n32",
            inputNames: ["xq", "w", "scales", "biases", "ascale", "rsb", "ids", "ksz"],
            outputNames: ["out"], source: source,
            header: PromptInt8Matmul.naxHeader + headIndexedNAXHeader, ensureRowContiguous: true)
    }()

    static func gatherIndexedNAX(
        _ activation: PromptInt8Activation, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        ids: MLXArray
    ) -> MLXArray? {
        guard screenAvailable, formsAllowed, (formsMatched & formBit(k: 4096, layers: 1)) != 0,
            formsFit(k: 4096, n: 98304), activation.rows == 15, activation.k == 4096,
            activation.codes.dtype == .int8, activation.codes.shape == [16, 4096],
            weight.dtype == .uint32, weight.shape == [98304, 256],
            scales.dtype == .float16, scales.shape == [98304, 32],
            biases.dtype == .float16, biases.shape == [98304, 32],
            ids.dtype == .int32, ids.shape == [15, 128], let kernel = headIndexedNAXKernel
        else { return nil }
        let dims: MLXArray = MLXArray([Int32(4096), Int32(98304), Int32(128)])
        let inputs: [MLXArray] = [activation.codes, weight, scales, biases,
            activation.ascale, activation.rsb, ids, dims]
        let template: [(String, any KernelTemplateArg)] = [("ROWS", 15), ("NEG", 1), ("KC", 4096)]
        let outputs: [MLXArray] = kernel(inputs, template: template,
            grid: (60 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[15, 128]], outputDTypes: [.float32])
        return outputs[0]
    }

    static func draftScreenPartialsSource(streamed: Bool, metadataShuffle: Bool? = nil) -> String? {
        var source: String = draftHeadPartialsSource(streamed: streamed, metadataShuffle: metadataShuffle)
        let swaps: [(String, String)] = [
            ("for (int g = int(sg); g < Kg; g += 4) {", "for (int g = int(sg); g < SG; g += 4) {"),
            ("if (g + 4 < Kg) {", "if (g + 4 < SG) {"),
        ]
        for (old, new) in swaps {
            guard source.components(separatedBy: old).count == 2 else { return nil }
            source = source.replacingOccurrences(of: old, with: new)
        }
        return source
    }

    private static let draftScreenPartialsKernels: [MLXFast.MLXFastKernel?] = [false, true].map { streamed in
        draftScreenPartialsSource(streamed: streamed).map { source in
            MLXFast.metalKernel(
                name: streamed ? "bonsai_draft_screen_top4_segments_stream" : "bonsai_draft_screen_top4_segments",
                inputNames: ["xq", "w0", "s0", "b0", "ascale", "rsb", "ksz"],
                outputNames: ["out0", "tags"], source: source,
                header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)
        }
    }

    static func draftScreenPartials(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        streamed: Bool, groups: Int, metadataShuffle: Bool? = nil
    ) -> (values: MLXArray, tags: MLXArray)? {
        guard screenAvailable, activation.rows == 15, activation.k == 4096,
            weight.dim(0) == 98304, groups == 16,
            let kernel = draftScreenPartialsKernels[streamed ? 1 : 0]
        else { return nil }
        let dims: MLXArray = MLXArray([Int32(4096), Int32(15), Int32(98304)])
        let template: [(String, any KernelTemplateArg)] = [
            ("ROWS", 15), ("NEG", 1), ("KC", 4096), ("SG", groups),
            ("METADATA_SHUFFLE", metadataShuffle ?? metadataUsesShuffle(kind: 5, k: 4096, rows: 15, negative: true, ns: [98304, groups]))]
        let inputs: [MLXArray] = [activation.codes, weight, scalesT, biasesT,
            activation.ascale, activation.rsb, dims]
        let output: [MLXArray] = kernel(
            inputs, template: template, grid: (256 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[15, 2048, 16], [15, 2048, 16]], outputDTypes: [.float32, .uint8])
        return (output[0], output[1])
    }


}

// MARK: - The dense GDN b/a projection in the qkv/z verify dispatch

public enum VerifyDenseBA {
    static let columnBody = """
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
          if (lane == 0) (is_a ? oa : ob)[v * N + col] = total;
        }

        """

    public static let standaloneSource = """
        const uint lane = thread_index_in_simdgroup;
        const uint n = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
        const int K = x_shape[1];
        const uint N = wb_shape[0];

        """ + columnBody

    static func fusedSource(streamed: Bool = false, metadataShuffle: Bool? = nil) -> String {
        let forms = streamed ? VerifyInt8Matmul.streamedFormsSource(layers: 2, metadataShuffle: metadataShuffle) : VerifyInt8Matmul.naxFormsSource(layers: 2)
        let anchor = "int bx = int(threadgroup_position_in_grid.x);"
        guard forms.components(separatedBy: anchor).count == 2 else { return "" }
        return forms.replacingOccurrences(
            of: anchor,
            with: """
                if (int(threadgroup_position_in_grid.x) < BAG) {
                const uint lane = thread_index_in_simdgroup;
                const uint n = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
                const uint N = BAN;
                \(columnBody)
                return;
                }
                int bx = int(threadgroup_position_in_grid.x) - BAG;
                """)
    }

    static let standaloneKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_dense_ba_rows",
        inputNames: ["x", "wb", "wa"], outputNames: ["ob", "oa"],
        source: standaloneSource)

    static let fusedKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_nax_forms2_dense_ba",
        inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "rsb", "ksz", "x", "wb", "wa"],
        outputNames: ["out0", "out1", "ob", "oa"],
        source: fusedSource(),
        header: PromptInt8Matmul.naxHeader,
        ensureRowContiguous: true)

    static let streamedKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_streamed_col_dense_ba",
        inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "rsb", "ksz", "x", "wb", "wa"],
        outputNames: ["out0", "out1", "ob", "oa"], source: fusedSource(streamed: true),
        header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)

    public static func standalone(_ x: MLXArray, wb: MLXArray, wa: MLXArray) -> (MLXArray, MLXArray) {
        let rows = x.dim(0)
        let N = wb.dim(0)
        let outputs = standaloneKernel(
            [x, wb, wa], template: [("ROWS", rows)],
            grid: (32, 2 * N, 1), threadGroup: (32, 2, 1),
            outputShapes: [[rows, N], [rows, N]], outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    static func fused(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)],
        m: Int, k: Int, negative: Bool, x: MLXArray, wb: MLXArray, wa: MLXArray, streamedWeights: [MLXArray]? = nil, metadataShuffle: Bool? = nil
    ) -> [MLXArray] {
        let ns = layers.map { $0.weight.dim(0) }
        let N = wb.dim(0)
        let dims = MLXArray([Int32(k), Int32(m)] + ns.map { Int32($0) })
        let weightInputs: [MLXArray] = streamedWeights ?? layers.map { $0.weight }
        let scaleInputs: [MLXArray] = layers.map { $0.scalesT }
        let biasInputs: [MLXArray] = layers.map { $0.biasesT }
        let codeAndWeights: [MLXArray] = [codes] + weightInputs
        let withScales: [MLXArray] = codeAndWeights + scaleInputs
        let withBiases: [MLXArray] = withScales + biasInputs
        let inputs: [MLXArray] = withBiases + [ascale, rsb, dims, x, wb, wa]
        let template = [
            ("ROWS", m), ("NEG", negative ? 1 : 0), ("KC", k), ("BAG", N / 2), ("BAN", N),
            ("METADATA_SHUFFLE", metadataShuffle ?? VerifyInt8Matmul.metadataUsesShuffle(kind: 2, k: k, rows: m, negative: negative, ns: ns + [N])),
        ] as [(String, any KernelTemplateArg)]
        let kernel = streamedWeights == nil ? fusedKernel : streamedKernel
        return kernel(
            inputs, template: template,
            grid: ((N / 2 + ns.reduce(0, +) / 32) * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: ns.map { [m, $0] } + [[m, N], [m, N]],
            outputDTypes: [.float32, .float32, .float32, .float32])
    }

    static let allowed: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_DENSE_BA"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    nonisolated(unsafe) private static var failed = false

    nonisolated(unsafe) static var fusedMatched = false

    static func check(eligible: Bool, negativeForms: Bool) -> String {
        fusedMatched = false
        guard allowed else { return "off (DARKBLOOM_BONSAI_VERIFY_DENSE_BA=0)" }
        guard eligible else { return "off (two-layer K 4096 form not in use)" }
        let k = 4096, n = 64, N = 32, kg = k / 128
        let maxRows = VerifyInt8Matmul.tileRows
        let codes = MLXArray(
            (0 ..< maxRows * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [maxRows, k])
        let weight = MLXArray(
            (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                let value: UInt64 = UInt64(i + 1) &* 2_654_435_761
                return UInt32(truncatingIfNeeded: value)
            }, [n, k / 16])
        let scaleValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
            let value: Float = Float(1 + index % 13) / 1024
            return Float16(value)
        }
        let scalesT = MLXArray(scaleValues, [kg, n])
        let biasValues: [Float16] = (0 ..< kg * n).map { (index: Int) -> Float16 in
            let value: Float = -Float(index % 7) / 512
            return Float16(value)
        }
        let biasesT = MLXArray(biasValues, [kg, n])
        let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
        let ascaleValues: [Float] = (0 ..< maxRows * kg).map { (index: Int) -> Float in
            let value: Float = Float(1 + index % 11) / 256
            return value
        }
        let ascale = MLXArray(ascaleValues, [maxRows * kg])
        let rsbValues: [Float] = (0 ..< maxRows * kg).map { (index: Int) -> Float in
            let centered: Int = index % 17 - 8
            let value: Float = Float(centered) / 64
            return value
        }
        let rsb = MLXArray(rsbValues, [maxRows * kg])
        func pattern(_ count: Int, _ mul: Int, _ add: Int, _ mod: Int, _ center: Int, _ scale: Float) -> [Float] {
            var values: [Float] = []
            values.reserveCapacity(count)
            for i in 0 ..< count {
                let v: Int = (i * mul + add) % mod - center
                values.append(Float(v) / scale)
            }
            return values
        }
        let wb = MLXArray(pattern(N * k, 29, 3, 97, 48, 3072), [N, k])
        let wa = MLXArray(pattern(N * k, 53, 7, 89, 44, 2048), [N, k])
        let xAll = MLXArray(pattern(maxRows * k, 41, 5, 211, 105, 64), [maxRows, k])
        func u(_ a: MLXArray) -> MLXArray { a.view(dtype: .uint32) }
        var matched = true
        withErrorHandler({ _ in VerifyDenseBA.failed = true }) {
            failed = false
            for m in (2 ... maxRows).reversed() where matched {
                let x = xAll[0 ..< m]
                var cases = [(biasesT, false)]
                if negativeForms { cases.append((negatedT, true)) }
                for (offsets, negative) in cases where matched {
                    let layers = [0 ..< 32, 32 ..< 64].map {
                        (weight: weight[$0], scalesT: scalesT[0..., $0], biasesT: offsets[0..., $0])
                    }
                    let fusedOut = fused(
                        codes: codes, ascale: ascale, rsb: rsb, layers: layers, m: m, k: k,
                        negative: negative, x: x, wb: wb, wa: wa)
                    let formsOut = VerifyInt8Matmul.denseBAForms(
                        codes: codes, ascale: ascale, rsb: rsb, layers: layers, m: m, k: k, negative: negative)
                    let ba = standalone(x, wb: wb, wa: wa)
                    let pairs: [(MLXArray, MLXArray)] = [
                        (fusedOut[0], formsOut[0]), (fusedOut[1], formsOut[1]),
                        (fusedOut[2], ba.0), (fusedOut[3], ba.1),
                    ]
                    let checks: [MLXArray] = pairs.map { pair in MLX.all(u(pair.0) .== u(pair.1)) }
                    eval(checks)
                    matched = !failed && checks.allSatisfy { $0.item(Bool.self) }
                }
            }
        }
        if failed { matched = false }
        failed = false
        fusedMatched = matched
        return matched ? "on (rows 2...\(maxRows) bit-identical)" : "off (a check differed or failed to compile)"
    }
}

extension VerifyInt8Matmul {
    static func denseBAEligible(k: Int) -> Bool {
        route == .nax && naxFast && naxShared && formsAllowed && (formsMatched & formBit(k: k, layers: 2)) != 0
    }

    static func denseBAForms(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)],
        m: Int, k: Int, negative: Bool
    ) -> [MLXArray] {
        runFast(codes: codes, ascale: ascale, rsb: rsb, layers: layers, m: m, k: k, negative: negative, forms: true)
    }

    static func matmulWithDenseBA(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        x: MLXArray, wb: MLXArray, wa: MLXArray, streamedWeights: [MLXArray]? = nil
    ) -> (outputs: [MLXArray], b: MLXArray, a: MLXArray)? {
        let plain = layers.map { ($0.weight, $0.scalesT, $0.biasesT) }
        guard layers.count == 2, activation.k == 4096, denseBAEligible(k: activation.k),
            usesForms(k: activation.k, layers: plain),
            (2 ... tileRows).contains(activation.rows),
            x.ndim == 2, x.dim(0) == activation.rows, x.dim(1) == activation.k, x.dtype == .float32,
            wb.ndim == 2, wb.shape == wa.shape, wb.dim(1) == activation.k, wb.dim(0) % 2 == 0,
            wb.dtype == .float32, wa.dtype == .float32,
            VerifyDenseBA.fusedMatched
        else { return nil }
        let outputs = VerifyDenseBA.fused(
            codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb, layers: plain,
            m: activation.rows, k: activation.k,
            negative: negativeForms && layers.allSatisfy { $0.negative }, x: x, wb: wb, wa: wa, streamedWeights: streamedWeights)
        return ([outputs[0], outputs[1]], outputs[2], outputs[3])
    }
}

final class PromptInt8Constants {
    private var streamed: [Int: MLXArray] = [:]
    private var transposed: (MLXArray, MLXArray)?
    let negative: Bool

    private var allowStream: Bool

    init(negative: Bool, allowStream: Bool) {
        self.negative = negative
        self.allowStream = allowStream
    }

    func streamWeight(_ weight: MLXArray, k: Int, columns: Int) -> MLXArray? {
        guard allowStream && VerifyInt8Matmul.streamReady else { return nil }
        if let value = streamed[columns] { return value }
        let value = VerifyInt8Matmul.streamPack(weight, k: k, columns: columns)
        eval(value)
        streamed[columns] = value
        if columns == 16 { PromptInt8Matmul.registerStream(weight, packed: value) }
        return value
    }

    func get(scales: MLXArray, biases: MLXArray) -> (MLXArray, MLXArray) {
        if let transposed { return transposed }
        let pair = (scales.transposed(1, 0).contiguous(), biases.transposed(1, 0).contiguous())
        transposed = pair
        return pair
    }
}

final class DraftHeadScreenState {
    var streamed: Bool?
    var retained = false
    var indexedRescore: Bool?
    var fusedRescoreTop16: Bool?
    nonisolated(unsafe) static var failure: String?
}

public final class HadamardQuantizedLinear: QuantizedLinear {
    public static func prepareSharedVerifyWeights() {
        PromptInt8Matmul.prepareStreaming()
    }

    public let transform: SignedBlockHadamard
    public let gdnLayout: HadamardGDNLayout?
    private static let reuseFloat16Constants = ProcessInfo.processInfo.environment[
        "DARKBLOOM_BONSAI_F16_CONSTANT_CACHE"] == "1"

    var permitsFloat16ConstantReuse: Bool {
        bits == 2 && groupSize == 128 && scales.dtype == .float16
            && (transform.blockSize == 4096
                || (Self.reuseFloat16Constants && transform.blockSize == 1024))
    }

    public init(
        weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, transform: SignedBlockHadamard,
        gdnLayout: HadamardGDNLayout? = nil, verifyStreaming: Bool = true
    ) throws {
        try validateHadamardWeights(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, transform: transform)
        guard
            bias == nil
                || (bias!.shape == [weight.dim(0)]
                    && [DType.float32, .float16, .bfloat16].contains(bias!.dtype))
        else {
            throw HadamardError.incompatibleShape
        }
        guard gdnLayout == nil || gdnLayout!.width == transform.width else {
            throw HadamardError.incompatibleShape
        }
        self.gdnLayout = gdnLayout
        self.transform = transform
        let inPlace =
            bits == 2 && groupSize == 128 && scales.dtype == .float16
            && biases?.dtype == .float16 && transform.width % 512 == 0
            && weight.dim(0) % 8 == 0
        self.readsFloat16ConstantsInPlace = inPlace
        var negative = false
        if inPlace, transform.blockSize == 4096, let biases {
            let flipped = scales.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))
            negative = MLX.all(biases.view(dtype: .uint16) .== flipped).item(Bool.self)
        }
        self.promptConstants = PromptInt8Constants(negative: negative, allowStream: verifyStreaming)
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
        freeze()
        if readsFloat16ConstantsInPlace && transform.blockSize == 4096 {
            _ = PromptInt8Matmul.route
            _ = VerifyInt8Matmul.route
            if weight.dim(0) % 32 == 0, VerifyInt8Matmul.formsK.contains(transform.width) {
                if let copied = promptConstants.streamWeight(weight, k: transform.width, columns: 16),
                    VerifyInt8Matmul.checkRealStreams, let biases
                {
                    let pair = promptConstants.get(scales: scales, biases: biases)
                    let matched = VerifyInt8Matmul.checkRealStream(
                        weight: weight, copied: copied, scalesT: pair.0, biasesT: pair.1,
                        k: transform.width, negative: negative)
                    if !matched { VerifyInt8Matmul.streamReady = false }
                }
            }
        }
    }


    let readsFloat16ConstantsInPlace: Bool

    private let promptConstants: PromptInt8Constants
    let exactHeadBounds = ExactVerifyHeadBounds()
    let draftScreenState = DraftHeadScreenState()
    private var streamedSwiGLUChoice: Bool?
    private var streamedSingle64Choice: Bool?
    private var streamedSingle64SplitChoice: Bool?
    private var streamedSingle64PairChoice: Bool?
    private var streamedSingle128Choice: Bool?
    private var streamedSwiGLUPartialChoice: Bool?
    private var streamedSwiGLUMetadataChoice: (gate: HadamardQuantizedLinear, up: HadamardQuantizedLinear, choice: (negative: Bool, shuffle: Bool))?

    private func promptInt8Forward(_ x: MLXArray, up: MLXArray? = nil) -> MLXArray? {
        guard readsFloat16ConstantsInPlace, transform.blockSize == 4096, x.dtype == .float32,
            x.ndim >= 2, PromptInt8Matmul.applies(rows: x.size / x.dim(-1)),
            weight.dim(0) % 64 == 0, transform.width % 512 == 0, PromptInt8Matmul.route != .off,
            let activation = transform.quantizedForPrompt(gdnLayout.map { $0(x) } ?? x, up: up)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = PromptInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative)
        else { return nil }
        return output.reshaped(Array(x.shape.dropLast()) + [weight.dim(0)])
    }

    public func canCallPromptLastRow(_ x: MLXArray) -> Bool {
        guard readsFloat16ConstantsInPlace && gdnLayout == nil && transform.blockSize == 4096
            && transform.width == 4096 && weight.dim(0) % 32 == 0
        else { return false }
        guard x.ndim == 3 && x.dim(0) == 1 && x.dim(2) == 4096 && x.dtype == .float32
        else { return false }
        guard x.dim(1) == 1 || PromptInt8Matmul.applies(rows: x.dim(1)) else { return false }
        return x.size <= Int(Int32.max) && StreamOrDevice.default == StreamOrDevice.gpu
            && PromptInt8Matmul.route != .off
    }

    private func promptLastRow(_ activation: PromptInt8Activation) -> MLXArray {
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        let output = PromptInt8Matmul.matmulLastRow(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative)!.reshaped([1, 1, weight.dim(0)])
        return bias.map { output + $0 } ?? output
    }

    public func callPromptLastRow(_ x: MLXArray) -> MLXArray? {
        guard canCallPromptLastRow(x), let activation = transform.quantizedForPromptLastRow(x)
        else { return nil }
        return promptLastRow(activation)
    }

    public func callAfterSwiGLU(gate: MLXArray, up: MLXArray) -> MLXArray? {
        guard gdnLayout == nil,
            let output = promptInt8Forward(gate, up: up) ?? verifyInt8Forward(gate, up: up)
        else { return nil }
        return bias.map { output + $0 } ?? output
    }

    private func verifyStreamedSingleOutput(
        _ activation: PromptInt8Activation, scalesT: MLXArray, biasesT: MLXArray
    ) -> MLXArray? {
        let stream: MLXArray? = streamedVerifyWeight
        let variant: Int = streamedSingle128Choice == true ? 3 : streamedSingle64PairChoice == true ? 2 : streamedSingle64SplitChoice == true ? 1 : 0
        if variant > 0 || streamedSingle64Choice == true, let stream,
            let output = VerifyInt8Matmul.streamedSingle64(activation,weight: weight,scalesT: scalesT,biasesT: biasesT,
                negative: promptConstants.negative,streamedWeight: stream,variant: variant) { return output }
        return VerifyInt8Matmul.matmul(activation,weight: weight,scalesT: scalesT,biasesT: biasesT,
            negative: promptConstants.negative,streamedWeight: stream)
    }

    private func prepareStreamedSingle64Choice(
        _ activation: PromptInt8Activation, streamedWeight: MLXArray, variant: Int = 0
    ) {
        guard (variant == 3 ? streamedSingle128Choice : variant == 2 ? streamedSingle64PairChoice : variant == 1 ? streamedSingle64SplitChoice : streamedSingle64Choice) == nil else { return }
        if variant == 3 { streamedSingle128Choice = false }
        else if variant == 2 { streamedSingle64PairChoice = false }
        else if variant == 1 { streamedSingle64SplitChoice = false } else { streamedSingle64Choice = false }
        let k: Int = activation.k
        guard [4096,12288].contains(k), transform.width == k, weight.shape == [4096,k/16],
            activation.rows == 16 else { return }
        let pair = promptConstants.get(scales: scales,biases: biases!)
        let negative: Bool = promptConstants.negative
        let start: UInt64 = DispatchTime.now().uptimeNanoseconds
        var complete: Bool = true
        var warm: [UInt64] = [0,0]
        var samples: [[UInt64]] = [[],[]]
        var legs: [[[UInt64]]] = [[],[]]
        func build(_ method: Int) -> [MLXArray]? {
            let result: MLXArray?
            if method == 0 {
                result = variant > 0 ? verifyStreamedSingleOutput(activation,scalesT: pair.0,biasesT: pair.1)
                    : VerifyInt8Matmul.matmul(activation,weight: weight,scalesT: pair.0,biasesT: pair.1,
                        negative: negative,streamedWeight: streamedWeight)
            } else {
                result = VerifyInt8Matmul.streamedSingle64(activation,weight: weight,scalesT: pair.0,biasesT: pair.1,
                    negative: negative,streamedWeight: streamedWeight,variant: variant)
            }
            return result.map { (value: MLXArray) -> [MLXArray] in [value] }
        }
        do {
            try withError {
                guard let old = build(0), let proposed = build(1) else { complete = false; return }
                let checks: [MLXArray] = [MLX.all(MLX.isFinite(old[0])),MLX.all(MLX.isFinite(proposed[0])),
                    MLX.all(old[0].view(dtype: .uint32) .== proposed[0].view(dtype: .uint32))]
                eval(checks)
                guard checks.allSatisfy({ $0.item(Bool.self) }) else { complete = false; return }
                for method in 0 ..< 2 {
                    if let ns = Self.streamedSwiGLUTime({ build(method) }) { warm[method] = ns }
                    else { complete = false; break }
                }
                if complete {
                    for order in [[0,1],[1,0],[0,1]] {
                        var round: [[UInt64]] = [[],[]]
                        for mirrored in [order,Array(order.reversed())] {
                            for method in mirrored {
                                if let ns = Self.streamedSwiGLUTime({ build(method) }) { round[method].append(ns) }
                                else { complete = false; break }
                            }
                            if !complete { break }
                        }
                        for method in 0 ..< 2 {
                            legs[method].append(round[method])
                            if round[method].count == 2 { samples[method].append(round[method][0]+round[method][1]) }
                        }
                        if !complete { break }
                    }
                }
                withExtendedLifetime((activation,streamedWeight,pair,old,proposed)) {}
            }
        } catch { complete = false }
        let selected: Bool = Self.streamedSwiGLUChoicePassed(complete: complete,samples: samples)
        if variant == 3 { streamedSingle128Choice = selected }
        else if variant == 2 { streamedSingle64PairChoice = selected }
        else if variant == 1 { streamedSingle64SplitChoice = selected } else { streamedSingle64Choice = selected }
        let cold: UInt64 = DispatchTime.now().uptimeNanoseconds-start
        FileHandle.standardError.write(Data(
            "int8 verify streamed single N64 select variant=\(variant) M16 K\(k) N4096 exact=\(complete) policy=all3_strict methods=[current_single,n64] warm_ns=\(warm) legs_ns=\(legs) round_sums_ns=\(samples) n64=\(selected) calibration_ns=\(cold)\n".utf8))
    }

    public static func prepareStreamedMetadataChoice(
        projections: [Linear], wb: MLXArray? = nil, wa: MLXArray? = nil
    ) {
        guard VerifyInt8Matmul.route == .nax, VerifyInt8Matmul.streamReady,
            (1 ... 3).contains(projections.count) else { return }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({ $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil && $0.bias == nil
                && $0.transform.blockSize == 4096 && $0.transform.isIdentical(to: first.transform)
                && $0.weight.dim(0) % 32 == 0 })
        else { return }
        var streams: [MLXArray] = []
        for layer in packed {
            guard let stream = layer.streamedVerifyWeight else { return }
            streams.append(stream)
        }
        _ = try? withError {
            let k: Int = first.transform.width
            let count: Int = 16 * k
            let values: [Float] = (0 ..< count).map { (index: Int) -> Float in
                let product: Int = index * 37
                let shifted: Int = product + 11
                let wrapped: Int = shifted % 255
                let centered: Int = wrapped - 127
                let value: Float = Float(centered)
                return value / Float(256)
            }
            let x = MLXArray(values, [16, k])
            guard let activation = first.transform.quantizedForVerify(x) else { return }
            let operands = packed.map(\.verifyOperands)
            var inputs = [activation.codes, activation.ascale, activation.rsb]
            inputs.append(contentsOf: streams)
            inputs.append(contentsOf: operands.map { $0.scalesT })
            inputs.append(contentsOf: operands.map { $0.biasesT })
            var dense: (x: MLXArray, wb: MLXArray, wa: MLXArray)? = nil
            if let wb, let wa, wb.dtype == .float32, wa.dtype == .float32,
                wb.ndim == 2, wb.shape == wa.shape, wb.dim(1) == k, wb.dim(0) % 2 == 0
            {
                inputs.append(contentsOf: [x, wb, wa])
                dense = (x, wb, wa)
            }
            eval(inputs)
            VerifyInt8Matmul.prepareMetadata(activation, layers: operands, streamedWeights: streams, dense: dense)
            if packed.count == 1, [4096,12288].contains(k), first.weight.shape == [4096,k/16] {
                first.prepareStreamedSingle64Choice(activation,streamedWeight: streams[0])
                first.prepareStreamedSingle64Choice(activation,streamedWeight: streams[0],variant: 1)
                first.prepareStreamedSingle64Choice(activation,streamedWeight: streams[0],variant: 2)
                first.prepareStreamedSingle64Choice(activation,streamedWeight: streams[0],variant: 3)
            }
            withExtendedLifetime((x, activation, operands, streams, inputs)) {}
        }
    }

    private func streamedSwiGLUInput(
        _ activation: PromptInt8Activation,
        operands: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)], streams: [MLXArray],
        metadataChoice: (negative: Bool, shuffle: Bool)?
    ) -> PromptInt8Activation? {
        if streamedSwiGLUPartialChoice == true, let signs = transform.swiGLUPartialSigns,
            let partial = VerifyInt8Matmul.streamedSwiGLUPartial4(activation, layers: operands, streamedWeights: streams, signs: signs, metadataChoice: metadataChoice),
            let result = transform.quantizedForVerifySwiGLUPartial4(partial) { return result }
        guard let value = VerifyInt8Matmul.streamedSwiGLU(activation, layers: operands, streamedWeights: streams, metadataChoice: metadataChoice) else { return nil }
        return transform.quantizedForVerifySwiGLUValue(value)
    }

    private func prepareSwiGLUPartialChoice(
        _ activation: PromptInt8Activation,
        operands: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)], streams: [MLXArray],
        metadataChoice: (negative: Bool, shuffle: Bool)?
    ) {
        guard streamedSwiGLUPartialChoice == nil else { return }
        streamedSwiGLUPartialChoice = false
        guard let signs = transform.swiGLUPartialSigns else { return }
        let start: UInt64 = DispatchTime.now().uptimeNanoseconds
        var complete: Bool = true
        var warm: [UInt64] = [0,0]
        var sums: [[UInt64]] = [[],[]]
        var legs: [[[UInt64]]] = [[],[]]
        do {
            try withError {
                guard let old = VerifyInt8Matmul.streamedSwiGLU(activation, layers: operands, streamedWeights: streams, metadataChoice: metadataChoice),
                    let partial = VerifyInt8Matmul.streamedSwiGLUPartial4(activation, layers: operands, streamedWeights: streams, signs: signs, metadataChoice: metadataChoice),
                    let expected = transform.swiGLUPartialReference(old) else { complete = false; return }
                let finite: [MLXArray] = [MLX.all(MLX.isFinite(old)),MLX.all(MLX.isFinite(partial)),
                    MLX.all(partial.view(dtype: .uint32) .== expected.view(dtype: .uint32))]
                eval(finite)
                guard finite.allSatisfy({ $0.item(Bool.self) }) else { complete = false; return }
                guard let oldQ = transform.quantizedForVerifySwiGLUValue(old),
                    let newQ = transform.quantizedForVerifySwiGLUPartial4(partial) else { complete = false; return }
                let exact: [MLXArray] = [MLX.all(oldQ.codes .== newQ.codes),
                    MLX.all(oldQ.ascale.view(dtype: .uint32) .== newQ.ascale.view(dtype: .uint32)),
                    MLX.all(oldQ.rsb.view(dtype: .uint32) .== newQ.rsb.view(dtype: .uint32)),
                    MLX.all(MLX.isFinite(oldQ.ascale)),MLX.all(MLX.isFinite(newQ.ascale)),
                    MLX.all(MLX.isFinite(oldQ.rsb)),MLX.all(MLX.isFinite(newQ.rsb))]
                eval(exact)
                guard exact.allSatisfy({ $0.item(Bool.self) }) else { complete = false; return }
                func build(_ method: Int) -> [MLXArray]? {
                    let result: PromptInt8Activation?
                    if method == 0 {
                        guard let value = VerifyInt8Matmul.streamedSwiGLU(activation, layers: operands, streamedWeights: streams, metadataChoice: metadataChoice) else { return nil }
                        result = transform.quantizedForVerifySwiGLUValue(value)
                    } else {
                        guard let value = VerifyInt8Matmul.streamedSwiGLUPartial4(activation, layers: operands, streamedWeights: streams, signs: signs, metadataChoice: metadataChoice) else { return nil }
                        result = transform.quantizedForVerifySwiGLUPartial4(value)
                    }
                    guard let result else { return nil }
                    return [result.codes,result.ascale,result.rsb]
                }
                for method in 0 ..< 2 {
                    if let ns = Self.streamedSwiGLUTime({ build(method) }) { warm[method] = ns } else { complete = false; break }
                }
                if complete {
                    for order in [[0,1],[1,0],[0,1]] {
                        var round: [[UInt64]] = [[],[]]
                        for mirror in [order,Array(order.reversed())] {
                            for method in mirror {
                                if let ns = Self.streamedSwiGLUTime({ build(method) }) { round[method].append(ns) } else { complete = false; break }
                            }
                            if !complete { break }
                        }
                        for method in 0 ..< 2 {
                            legs[method].append(round[method])
                            if round[method].count == 2 { sums[method].append(round[method][0] + round[method][1]) }
                        }
                        if !complete { break }
                    }
                }
                withExtendedLifetime((activation,operands,streams,signs,old,partial,expected,oldQ,newQ)) {}
            }
        } catch { complete = false }
        let selected: Bool = Self.streamedSwiGLUChoicePassed(complete: complete,samples: sums)
        streamedSwiGLUPartialChoice = selected
        let cold: UInt64 = DispatchTime.now().uptimeNanoseconds - start
        FileHandle.standardError.write(Data(
            "int8 verify SwiGLU partial4 select M16 K4096 N12288 exact=\(complete) policy=all3_strict warm_ns=\(warm) legs_ns=\(legs) round_sums_ns=\(sums) partial=\(selected) calibration_ns=\(cold)\n".utf8))
    }

    public static func prepareStreamedSwiGLUChoice(projections: [Linear], down: Linear) {
        guard let down = down as? HadamardQuantizedLinear,
            down.streamedSwiGLUChoice == nil, VerifyInt8Matmul.canStreamedSwiGLU
        else { return }
        down.streamedSwiGLUChoice = false
        guard projections.count == 2 else { return }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == 2, let first = packed.first, first.transform.width == 4096,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.bias == nil && $0.gdnLayout == nil
                    && $0.weight.dim(0) == 12288 && $0.transform.blockSize == 4096
                    && $0.transform.isIdentical(to: first.transform)
            }), down.readsFloat16ConstantsInPlace, down.gdnLayout == nil,
            down.transform.blockSize == 4096, down.transform.width == 12288, down.weight.dim(0) == 4096,
            let streams = Self.streamedWeights(packed)
        else { return }
        let coldStart: UInt64 = DispatchTime.now().uptimeNanoseconds
        let values: [Float] = (0 ..< 16 * 4096).map { (i: Int) -> Float in
            let t: Double = Double(i)
            return Float(Foundation.sin(t * 0.6180339) * (1 + 0.5 * Foundation.cos(t * 0.0211)))
        }
        let fixed = MLXArray(values, [16, 4096])
        let operands = packed.map(\.verifyOperands)
        var complete: Bool = true
        var warm: [UInt64] = [0, 0]
        var samples: [[UInt64]] = [[], []]
        var legs: [[[UInt64]]] = [[], []]
        do {
            try withError {
                guard let activation = first.transform.quantizedForVerify(fixed) else {
                    complete = false; return
                }
                var inputs: [MLXArray] = [activation.codes, activation.ascale, activation.rsb]
                inputs.append(contentsOf: streams)
                inputs.append(contentsOf: operands.map { $0.scalesT })
                inputs.append(contentsOf: operands.map { $0.biasesT })
                eval(inputs)
                down.streamedSwiGLUMetadataChoice = VerifyInt8Matmul.prepareMetadata(activation, layers: operands, streamedWeights: streams, swiglu: true)
                    .map { (packed[0], packed[1], $0) }
                let metadataChoice = down.streamedSwiGLUMetadataChoice?.choice
                down.prepareSwiGLUPartialChoice(activation, operands: operands, streams: streams, metadataChoice: metadataChoice)
                func build(_ method: Int) -> [MLXArray]? {
                    let result: PromptInt8Activation?
                    if method == 0 {
                        guard let pair = VerifyInt8Matmul.matmul(
                            activation, layers: operands, streamedWeights: streams)
                        else { return nil }
                        result = down.transform.quantizedForVerify(pair[0], up: pair[1])
                    } else {
                        result = down.streamedSwiGLUInput(activation, operands: operands, streams: streams, metadataChoice: metadataChoice)
                    }
                    guard let result else { return nil }
                    return [result.codes, result.ascale, result.rsb]
                }
                for method in 0 ..< 2 {
                    if let ns = Self.streamedSwiGLUTime({ build(method) }) { warm[method] = ns }
                    else { complete = false; break }
                }
                if complete {
                    for order in [[0, 1], [1, 0], [0, 1]] {
                        var round: [[UInt64]] = [[], []]
                        for mirrored in [order, Array(order.reversed())] {
                            for method in mirrored {
                                if let ns = Self.streamedSwiGLUTime({ build(method) }) { round[method].append(ns) }
                                else { complete = false; break }
                            }
                            if !complete { break }
                        }
                        for method in 0 ..< 2 {
                            legs[method].append(round[method])
                            if round[method].count == 2 {
                                samples[method].append(round[method][0] + round[method][1])
                            }
                        }
                        if !complete { break }
                    }
                }
                withExtendedLifetime((fixed, activation, streams, operands, inputs)) {}
            }
        } catch { complete = false }
        let fused = Self.streamedSwiGLUChoicePassed(complete: complete, samples: samples)
        down.streamedSwiGLUChoice = fused
        let cold: UInt64 = DispatchTime.now().uptimeNanoseconds - coldStart
        FileHandle.standardError.write(Data(
            "int8 verify streamed SwiGLU select M16 K4096 N12288 methods=[original,fused] policy=all3_strict complete=\(complete) warm_ns=\(warm) legs_ns=\(legs) round_sums_ns=\(samples) fused=\(fused) calibration_ns=\(cold)\n".utf8))
    }

    static func streamedSwiGLUChoicePassed(complete: Bool, samples: [[UInt64]]) -> Bool {
        guard complete, samples.count == 2, samples[0].count == 3, samples[1].count == 3 else { return false }
        return zip(samples[0], samples[1]).allSatisfy {
            $0.0 > 0 && $0.1 > 0 && $0.1 < $0.0
        }
    }

    private static func streamedSwiGLUTime(_ build: () -> [MLXArray]?) -> UInt64? {
        do {
            return try withError { () -> UInt64? in
                Stream.gpu.synchronize()
                let start: UInt64 = DispatchTime.now().uptimeNanoseconds
                guard let outputs = build() else { return nil }
                eval(outputs)
                Stream.gpu.synchronize()
                let end: UInt64 = DispatchTime.now().uptimeNanoseconds
                withExtendedLifetime(outputs) {}
                return end > start ? end - start : nil
            }
        } catch { return nil }
    }

    public static func sharedResidualNormSwiGLUProjection(
        _ x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float,
        projections: [Linear], down: Linear
    ) -> (residual: MLXArray, output: MLXArray)? {
        guard x.ndim >= 2, x.dtype == .float32, x.dim(-1) == 4096, x.size / 4096 == 16,
            VerifyInt8Matmul.canStreamedSwiGLU, projections.count == 2
        else { return nil }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == 2, let first = packed.first, first.transform.width == 4096,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.bias == nil && $0.gdnLayout == nil
                    && $0.weight.dim(0) == 12288 && $0.transform.blockSize == 4096
                    && $0.transform.isIdentical(to: first.transform)
            }), let down = down as? HadamardQuantizedLinear, down.streamedSwiGLUChoice == true,
            down.readsFloat16ConstantsInPlace, down.gdnLayout == nil,
            down.transform.blockSize == 4096, down.transform.width == 12288, down.weight.dim(0) == 4096,
            let streams = Self.streamedWeights(packed),
            let activation = first.transform.residualNorm(
                x, residual: residual, weight: weight, eps: eps, quantized: false, verifyRows: 16)
        else { return nil }
        let input = PromptInt8Activation(codes: activation[0], ascale: activation[1], rsb: activation[2], rows: 16, k: 4096)
        let ready = down.streamedSwiGLUMetadataChoice
        let metadataChoice = ready?.gate === packed[0] && ready?.up === packed[1] ? ready?.choice : nil
        if down.streamedSwiGLUPartialChoice == true,
            let codes = down.streamedSwiGLUInput(input, operands: packed.map(\.verifyOperands), streams: streams, metadataChoice: metadataChoice)
        {
            let pair = down.promptConstants.get(scales: down.scales,biases: down.biases!)
            if let value = down.verifyStreamedSingleOutput(codes,scalesT: pair.0,biasesT: pair.1) {
                let output = value.reshaped(Array(x.shape.dropLast()) + [down.weight.dim(0)])
                return (activation.last!,down.bias.map { output + $0 } ?? output)
            }
        }
        guard let value = VerifyInt8Matmul.streamedSwiGLU(input, layers: packed.map(\.verifyOperands), streamedWeights: streams, metadataChoice: metadataChoice),
            let output = down.callSwiGLUValue(value.reshaped(Array(x.shape.dropLast()) + [12288])) else { return nil }
        return (activation.last!,output)
    }

    private func callSwiGLUValue(_ x: MLXArray) -> MLXArray? {
        guard let output = verifyInt8Forward(x, precomputedSwiGLU: true) else { return nil }
        return bias.map { output + $0 } ?? output
    }

    public static func sharedResidualNormProjections(
        _ x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float,
        projections: [Linear]
    ) -> (residual: MLXArray, projections: [MLXArray])? {
        guard x.ndim >= 2, x.dim(-1) == 4096 else { return nil }
        let rows = x.size / 4096
        let quantized = PromptInt8Matmul.applies(rows: rows) && PromptInt8Matmul.route != .off
        let verify = !quantized && x.dtype == .float32 && VerifyInt8Matmul.rowRange.contains(rows)
            && VerifyInt8Matmul.route != .off
        guard (1 ... 16).contains(rows) || quantized else { return nil }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil
                    && $0.weight.dim(0) % 64 == 0 && $0.transform.isIdentical(to: first.transform)
            }),
            let activation = first.transform.residualNorm(
                x, residual: residual, weight: weight, eps: eps, quantized: quantized,
                verifyRows: verify ? rows : 0)
        else { return nil }
        let promptOutputs = quantized && !verify
            ? PromptInt8Matmul.matmul(
                PromptInt8Activation(
                    codes: activation[0], ascale: activation[1], rsb: activation[2], rows: rows, k: 4096),
                layers: packed.map(\.verifyOperands))
            : nil
        let verifyOutputs = verify
            ? VerifyInt8Matmul.matmul(
                PromptInt8Activation(
                    codes: activation[0], ascale: activation[1], rsb: activation[2], rows: rows, k: 4096),
                layers: packed.map(\.verifyOperands), streamedWeights: Self.streamedWeights(packed))
            : nil
        let outputs = packed.enumerated().map { index, layer in
            if verify {
                guard let output = verifyOutputs?[index] else {
                    let (scalesT, biasesT) = layer.promptConstants.get(scales: layer.scales, biases: layer.biases!)
                    let codes = PromptInt8Activation(
                        codes: activation[0], ascale: activation[1], rsb: activation[2], rows: rows, k: 4096)
                    let single = VerifyInt8Matmul.matmul(
                        codes, weight: layer.weight, scalesT: scalesT, biasesT: biasesT,
                        negative: layer.promptConstants.negative, streamedWeight: layer.streamedVerifyWeight)!
                        .reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
                    return layer.bias.map { single + $0 } ?? single
                }
                let reshaped = output.reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
                return layer.bias.map { reshaped + $0 } ?? reshaped
            }
            if !quantized { return layer.forwardRotated(activation[0]) }
            let output = promptOutputs![index]
                .reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
        return (activation.last!, outputs)
    }

    public static func sharedResidualNormPromptProjections(
        _ x: MLXArray, delta: MLXArray, weight: MLXArray, eps: Float,
        projections: [Linear], normalized: Bool, firstProjectionLastRow: Bool = false
    ) -> (residual: MLXArray, normalized: MLXArray?, projections: [MLXArray])? {
        guard x.ndim >= 2, x.dtype == .float32, x.dim(-1) == 4096,
            PromptInt8Matmul.applies(rows: x.size / x.dim(-1)),
            PromptInt8Matmul.route != .off
        else { return nil }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil
                    && $0.transform.blockSize == 4096 && $0.weight.dim(0) % 64 == 0
                    && $0.transform.isIdentical(to: first.transform)
            }), !firstProjectionLastRow || first.canCallPromptLastRow(x),
            let codes = first.transform.residualNormPrompt(
                x, residual: delta, weight: weight, eps: eps, normalized: normalized)
        else { return nil }
        let rows = x.size / 4096
        let activation = PromptInt8Activation(
            codes: codes[0], ascale: codes[1], rsb: codes[2], rows: rows, k: 4096)
        let full = firstProjectionLastRow ? Array(packed.dropFirst()) : packed
        guard let outputs = PromptInt8Matmul.matmul(activation, layers: full.map(\.verifyOperands))
        else { return nil }
        var projected = zip(full, outputs).map { layer, output in
            let output = output.reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
        if firstProjectionLastRow { projected.insert(first.promptLastRow(activation), at: 0) }
        return (codes[3], normalized ? codes[4] : nil, projected)
    }

    public static func sharedNormPromptProjections(
        _ x: MLXArray, weight: MLXArray, eps: Float, projections: [Linear], normalized: Bool, firstProjectionLastRow: Bool = false
    ) -> (normalized: MLXArray?, projections: [MLXArray])? {
        guard x.ndim >= 2, x.dtype == .float32, x.dim(-1) == 4096,
            PromptInt8Matmul.applies(rows: x.size / x.dim(-1)),
            PromptInt8Matmul.route != .off
        else { return nil }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil
                    && $0.transform.blockSize == 4096 && $0.weight.dim(0) % 64 == 0
                    && $0.transform.isIdentical(to: first.transform)
            }), !firstProjectionLastRow || first.canCallPromptLastRow(x),
            let codes = first.transform.normPrompt(x, weight: weight, eps: eps, normalized: normalized)
        else { return nil }
        let rows = x.size / 4096
        let activation = PromptInt8Activation(
            codes: codes[0], ascale: codes[1], rsb: codes[2], rows: rows, k: 4096)
        let full = firstProjectionLastRow ? Array(packed.dropFirst()) : packed
        guard let outputs = PromptInt8Matmul.matmul(activation, layers: full.map(\.verifyOperands))
        else { return nil }
        var projected = zip(full, outputs).map { layer, output in
            let output = output.reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
        if firstProjectionLastRow { projected.insert(first.promptLastRow(activation), at: 0) }
        return (normalized ? codes[3] : nil, projected)
    }

    public static func sharedPromptProjections(
        _ x: MLXArray, _ projections: [Linear]
    ) -> [MLXArray]? {
        guard x.ndim >= 2, x.dtype == .float32,
            PromptInt8Matmul.applies(rows: x.size / x.dim(-1)),
            PromptInt8Matmul.route != .off
        else { return nil }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil
                    && $0.transform.blockSize == 4096 && $0.weight.dim(0) % 64 == 0
                    && $0.transform.isIdentical(to: first.transform)
            }),
            let activation = first.transform.quantizedForPrompt(x),
            let outputs = PromptInt8Matmul.matmul(activation, layers: packed.map(\.verifyOperands))
        else { return nil }
        return zip(packed, outputs).map { layer, output in
            let output = output.reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
    }

    public struct VerifyBoundary {
        let activation: PromptInt8Activation
        let transform: SignedBlockHadamard
        public let residual: MLXArray
        public let normalized: MLXArray
    }

    public static func verifyNormBoundary(
        _ x: MLXArray, weight: MLXArray, eps: Float,
        projections: [Linear]
    ) -> VerifyBoundary? {
        guard x.ndim >= 2, x.dim(-1) == 4096, x.dtype == .float32, VerifyInt8Matmul.route != .off,
            VerifyInt8Matmul.rowRange.contains(x.size / 4096)
        else { return nil }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil && $0.bias == nil
                    && $0.transform.blockSize == 4096 && $0.weight.dim(0) % 32 == 0
                    && $0.transform.isIdentical(to: first.transform)
            }),
            let outputs = first.transform.normVerifyNormalized(x, weight: weight, eps: eps)
        else { return nil }
        let rows = x.size / 4096
        return VerifyBoundary(
            activation: PromptInt8Activation(
                codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: 4096),
            transform: first.transform, residual: x, normalized: outputs[3])
    }

    public static func verifyBoundary(
        _ x: MLXArray, delta: MLXArray, weight: MLXArray, eps: Float,
        projections: [Linear]
    ) -> VerifyBoundary? {
        guard x.ndim >= 2, x.dim(-1) == 4096, x.dtype == .float32, VerifyInt8Matmul.route != .off,
            VerifyInt8Matmul.rowRange.contains(x.size / 4096)
        else { return nil }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil && $0.bias == nil
                    && $0.transform.blockSize == 4096 && $0.weight.dim(0) % 32 == 0
                    && $0.transform.isIdentical(to: first.transform)
            }),
            let outputs = first.transform.residualNormVerifyNormalized(
                x, residual: delta, weight: weight, eps: eps)
        else { return nil }
        let rows = x.size / 4096
        return VerifyBoundary(
            activation: PromptInt8Activation(
                codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: 4096),
            transform: first.transform, residual: outputs[3], normalized: outputs[4])
    }

    public static func boundaryProjections(
        _ boundary: VerifyBoundary, _ projections: [Linear]
    ) -> [MLXArray]? {
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil && $0.bias == nil
                    && $0.transform.blockSize == 4096 && $0.weight.dim(0) % 32 == 0
                    && $0.transform.isIdentical(to: boundary.transform)
            })
        else { return nil }
        guard let outputs = VerifyInt8Matmul.matmul(
            boundary.activation, layers: packed.map(\.verifyOperands), streamedWeights: Self.streamedWeights(packed))
        else { return nil }
        return zip(packed, outputs).map { layer, output in
            output.reshaped(Array(boundary.normalized.shape.dropLast()) + [layer.weight.dim(0)])
        }
    }

    public static func boundaryProjectionsWithDenseBA(
        _ boundary: VerifyBoundary, _ projections: [Linear], x: MLXArray, wb: MLXArray, wa: MLXArray
    ) -> (outputs: [MLXArray], b: MLXArray, a: MLXArray)? {
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == 2, projections.count == 2,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil && $0.bias == nil
                    && $0.transform.blockSize == 4096 && $0.weight.dim(0) % 32 == 0
                    && $0.transform.isIdentical(to: boundary.transform)
            })
        else { return nil }
        let rows = boundary.activation.rows
        guard boundary.normalized.dim(-1) == 4096, boundary.normalized.size == rows * 4096,
            x.shape == boundary.normalized.shape,
            let fused = VerifyInt8Matmul.matmulWithDenseBA(
                boundary.activation, layers: packed.map(\.verifyOperands),
                x: x.reshaped([rows, 4096]), wb: wb, wa: wa, streamedWeights: Self.streamedWeights(packed))
        else { return nil }
        let lead = Array(boundary.normalized.shape.dropLast())
        return (
            zip(packed, fused.outputs).map { layer, output in
                output.reshaped(lead + [layer.weight.dim(0)])
            },
            fused.b.reshaped(lead + [wb.dim(0)]), fused.a.reshaped(lead + [wb.dim(0)]))
    }

    public static func sharedShortProjections(
        _ x: MLXArray, _ projections: [Linear]
    ) -> [MLXArray]? {
        guard x.ndim >= 2, x.dtype == .float32 else { return nil }
        let rows = x.size / x.dim(-1)
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil
                    && $0.transform.blockSize == 4096
                    && $0.transform.isIdentical(to: first.transform)
            })
        else { return nil }

        // Route 1: VerifyInt8Matmul when rows in 2 ... 8 and route != .off
        if VerifyInt8Matmul.rowRange.contains(rows), VerifyInt8Matmul.route != .off,
            packed.allSatisfy({ $0.weight.dim(0) % 32 == 0 }),
            let activation = first.transform.quantizedForVerify(x)
        {
            let outputs = VerifyInt8Matmul.matmul(activation, layers: packed.map(\.verifyOperands), streamedWeights: Self.streamedWeights(packed))
            return packed.enumerated().map { index, layer in
                guard let output = outputs?[index] else { return layer(x) }
                let reshaped = output.reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
                return layer.bias.map { reshaped + $0 } ?? reshaped
            }
        }

        // Route 2: Shared Hadamard rotation for Float16ConstantQMV (1 ... 16 rows)
        if (1 ... Float16ConstantQMV.maximumRows).contains(rows) {
            let rotated = first.transform(x)
            return packed.map { $0.forwardRotated(rotated) }
        }

        return nil
    }

    private var streamedVerifyWeight: MLXArray? {
        let columns = VerifyInt8Matmul.streamColumns(k: transform.width, n: weight.dim(0))
        return promptConstants.streamWeight(weight, k: transform.width, columns: columns)
    }

    private static func streamedWeights(_ layers: [HadamardQuantizedLinear]) -> [MLXArray]? {
        guard (2 ... 3).contains(layers.count), VerifyInt8Matmul.streamReady else { return nil }
        var weights: [MLXArray] = []
        for layer in layers {
            guard let weight = layer.promptConstants.streamWeight(layer.weight, k: layer.transform.width, columns: 16)
            else { return nil }
            weights.append(weight)
        }
        return weights
    }

    private var verifyOperands: (weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool) {
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        return (weight, scalesT, biasesT, promptConstants.negative)
    }

    private func verifyInt8Forward(_ x: MLXArray, up: MLXArray? = nil, precomputedSwiGLU: Bool = false) -> MLXArray? {
        guard readsFloat16ConstantsInPlace, transform.blockSize == 4096, x.dtype == .float32,
            x.ndim >= 1, VerifyInt8Matmul.rowRange.contains(x.size / x.dim(-1)),
            weight.dim(0) % 32 == 0, transform.width % 512 == 0, VerifyInt8Matmul.route != .off,
            let activation = precomputedSwiGLU
                ? transform.quantizedForVerifySwiGLUValue(x)
                : transform.quantizedForVerify(gdnLayout.map { $0(x) } ?? x, up: up)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = verifyStreamedSingleOutput(activation,scalesT: scalesT,biasesT: biasesT)
        else { return nil }
        return output.reshaped(Array(x.shape.dropLast()) + [weight.dim(0)])
    }

    public func screenedArgMax(
        _ x: MLXArray, width: Int, chunk: Int, ids: MLXArray? = nil
    ) -> MLXArray? {
        let K = transform.width
        let N = weight.dim(0)
        guard readsFloat16ConstantsInPlace, bias == nil, gdnLayout == nil, let biases,
            x.dtype == .float32, x.size == K, width > 0, width < K, width % 512 == 0,
            chunk > 0, N % chunk == 0, (N / chunk) % 2 == 0
        else { return nil }
        let neg = promptConstants.negative ? 1 : 0
        let rotated = transform(x).reshaped([1, K])
        let partial = DraftScreenQMV.partial(
            [rotated, weight, scales, biases], template: [("P", width), ("NEG", neg)],
            grid: (32, N / 8 * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[N]], outputDTypes: [.float32])[0]
        let blocks = N / chunk
        let rescored: [MLXArray]
        if chunk < 32 && 32 % chunk == 0 && promptConstants.negative {
            rescored = DraftScreenQMV.rescoreFastChunk(
                [rotated, weight, scales, partial], template: [("CH", chunk)],
                grid: (32, blocks, 1), threadGroup: (32, 2, 1),
                outputShapes: [[blocks], [blocks]], outputDTypes: [.float32, .int32])
        } else if chunk == 32 && promptConstants.negative {
            rescored = DraftScreenQMV.rescoreFast32(
                [rotated, weight, scales, partial],
                grid: (32, blocks, 1), threadGroup: (32, 2, 1),
                outputShapes: [[blocks], [blocks]], outputDTypes: [.float32, .int32])
        } else {
            let pick = argMax(partial.reshaped([blocks, chunk]), axis: -1)
            rescored = DraftScreenQMV.rescore(
                [rotated, weight, scales, biases, pick],
                template: [("CH", chunk), ("NEG", neg)],
                grid: (32, blocks, 1), threadGroup: (32, 2, 1),
                outputShapes: [[blocks], [blocks]], outputDTypes: [.float32, .int32])
        }
        let idsInput = ids ?? rescored[1]
        let useIds = (ids != nil && ids!.dtype == .int32) ? 1 : 0
        let picked = DraftScreenQMV.finalPick(
            [rescored[0], rescored[1], idsInput],
            template: [("BLOCKS", blocks), ("USE_IDS", useIds)],
            grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
        if let ids, useIds == 0 {
            return ids[picked]
        }
        return picked
    }

    public func callAfterGatedNorm(
        _ x: MLXArray, gate: MLXArray, normWeight: MLXArray, eps: Float
    ) -> MLXArray? {
        guard gdnLayout == nil, readsFloat16ConstantsInPlace, transform.blockSize == 4096,
            weight.dim(0) % 32 == 0, transform.width % 512 == 0, VerifyInt8Matmul.route != .off,
            let activation = transform.quantizedForVerifyGatedNorm(
                x, gate: gate, weight: normWeight, eps: eps)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = verifyStreamedSingleOutput(activation,scalesT: scalesT,biasesT: biasesT)
        else { return nil }
        let shaped = output.reshaped(Array(x.shape.dropLast(2)) + [weight.dim(0)])
        return bias.map { shaped + $0 } ?? shaped
    }

    public func callAfterGatedNormPrompt(
        _ x: MLXArray, gate: MLXArray, normWeight: MLXArray, eps: Float
    ) -> MLXArray? {
        guard gdnLayout == nil, readsFloat16ConstantsInPlace, transform.blockSize == 4096,
            x.ndim >= 3, PromptInt8Matmul.applies(rows: x.size / 4096),
            weight.dim(0) % 64 == 0, transform.width % 512 == 0, PromptInt8Matmul.route != .off,
            let activation = transform.quantizedForPromptGatedNorm(
                x, gate: gate, weight: normWeight, eps: eps)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = PromptInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative)
        else { return nil }
        let shaped = output.reshaped(Array(x.shape.dropLast(2)) + [weight.dim(0)])
        return bias.map { shaped + $0 } ?? shaped
    }

    public func callAfterAttentionGate(
        _ attention: MLXArray, queryProjection: MLXArray
    ) -> MLXArray? {
        guard gdnLayout == nil, readsFloat16ConstantsInPlace, transform.blockSize == 4096,
            weight.dim(0) % 32 == 0, transform.width % 512 == 0, VerifyInt8Matmul.route != .off,
            let activation = transform.quantizedForVerifyAttentionGate(
                attention, queryProjection: queryProjection)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = verifyStreamedSingleOutput(activation,scalesT: scalesT,biasesT: biasesT)
        else { return nil }
        let shaped = output.reshaped([1, activation.rows, weight.dim(0)])
        return bias.map { shaped + $0 } ?? shaped
    }

    public func callAfterAttentionGatePrompt(
        _ attention: MLXArray, queryProjection: MLXArray
    ) -> MLXArray? {
        guard gdnLayout == nil, readsFloat16ConstantsInPlace, transform.blockSize == 4096,
            attention.ndim == 4, PromptInt8Matmul.applies(rows: attention.dim(2)),
            weight.dim(0) % 64 == 0, transform.width % 512 == 0, PromptInt8Matmul.route != .off,
            let activation = transform.quantizedForPromptAttentionGate(
                attention, queryProjection: queryProjection)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = PromptInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative)
        else { return nil }
        let shaped = output.reshaped([1, activation.rows, weight.dim(0)])
        return bias.map { shaped + $0 } ?? shaped
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let output = promptInt8Forward(x) ?? verifyInt8Forward(x) {
            return bias.map { output + $0 } ?? output
        }
        return forwardRotated(transform(gdnLayout.map { $0(x) } ?? x))
    }

    public func draftTop16Partials(_ x: MLXArray) -> (values: MLXArray, tags: MLXArray)? {
        guard readsFloat16ConstantsInPlace, bias == nil, gdnLayout == nil,
            transform.blockSize == 4096, transform.width == 4096, x.dtype == .float32,
            x.ndim >= 1, x.dim(-1) == 4096, x.size / 4096 == 15,
            weight.dim(0) == 98304, VerifyInt8Matmul.route == .nax,
            let activation = transform.quantizedForVerify(x)
        else { return nil }
        let pair = promptConstants.get(scales: scales, biases: biases!)
        return VerifyInt8Matmul.draftHeadPartials(
            activation, weight: weight, scalesT: pair.0, biasesT: pair.1,
            negative: promptConstants.negative, streamedWeight: streamedVerifyWeight)
    }

    private func forwardRotated(_ rotated: MLXArray) -> MLXArray {
        if readsFloat16ConstantsInPlace, rotated.dtype == .float32,
            (1 ... Float16ConstantQMV.maximumRows).contains(rotated.size / rotated.dim(-1))
        {
            let output = Float16ConstantQMV.matmul(
                rotated, weight: weight, scales: scales, biases: biases!,
                negative: promptConstants.negative)
            return bias.map { output + $0 } ?? output
        }
        if permitsFloat16ConstantReuse && rotated.dtype == .float32 {
            return constantCachedForward(rotated, allowFloat16: true)
        }
        return super.callAsFunction(rotated)
    }
}

extension HadamardQuantizedLinear {
    public func topTwoRows(_ x: MLXArray) -> (ids: MLXArray, values: MLXArray)? {
        guard readsFloat16ConstantsInPlace, bias == nil else { return nil }
        if x.ndim >= 1, x.dim(-1) == transform.width {
            let rows = x.size / x.dim(-1)
            if let exact = exactHeadTopTwo(x, rows: rows) { return exact }
        }
        let rotated = transform(gdnLayout.map { $0(x) } ?? x)
        guard rotated.dtype == .float32 else { return nil }
        let rows = rotated.size / rotated.dim(-1)
        if (5 ... 8).contains(rows), ExactVerifyHead.verified[rows] == false { return nil }
        return Float16ConstantQMV.topTwo(
            rotated, weight: weight, scales: scales, biases: biases!)
    }

    private func needsHeadResidualL2(rows: Int) -> Bool {
        !ExactVerifyHead.checkEveryCall && exactHeadBounds.compactVerified != false
            && exactHeadBounds.compactParallelVerified != false && exactHeadBounds.residualL2Verified != false
            && weight.dim(0) % 512 == 0
            && VerifyInt8Matmul.compactHeadAvailable(rows: rows, k: transform.width, n: weight.dim(0))
    }

    public func topTwoRowsAfterResidualNorm(
        _ x: MLXArray, delta: MLXArray, weight: MLXArray, eps: Float
    ) -> (ids: MLXArray, values: MLXArray, residual: MLXArray)? {
        guard readsFloat16ConstantsInPlace, bias == nil,
            x.ndim >= 1, x.dim(-1) == transform.width
        else { return nil }
        let rows = x.size / x.dim(-1)
        guard let range = ExactVerifyHead.rowRange, range.contains(rows),
            ExactVerifyHead.verified[rows] != false,
            gdnLayout == nil, transform.blockSize == 4096, transform.width == 4096,
            self.weight.dim(0) % 64 == 0, x.dtype == .float32, VerifyInt8Matmul.route != .off,
            let fused = transform.residualNormVerifyRotated(
                x, residual: delta, weight: weight, eps: eps, headError2: needsHeadResidualL2(rows: rows)),
            let topTwo = exactHeadTopTwo(
                activation: fused.activation, rotated: fused.rotated, rows: rows, error2: fused.error2)
        else { return nil }
        return (topTwo.ids, topTwo.values, fused.residual)
    }

    private func exactHeadTopTwo(
        _ x: MLXArray, rows: Int
    ) -> (ids: MLXArray, values: MLXArray)? {
        guard let range = ExactVerifyHead.rowRange, range.contains(rows),
            ExactVerifyHead.verified[rows] != false,
            gdnLayout == nil, transform.blockSize == 4096, transform.width == 4096,
            weight.dim(0) % 64 == 0, x.dtype == .float32, VerifyInt8Matmul.route != .off,
            let (activation, rotated, error2) = transform.quantizedForVerifyWithRotated(
                x, headError2: needsHeadResidualL2(rows: rows))
        else { return nil }
        return exactHeadTopTwo(activation: activation, rotated: rotated, rows: rows, error2: error2)
    }

    private func exactHeadTopTwo(
        activation: PromptInt8Activation, rotated: MLXArray, rows: Int, error2: MLXArray? = nil
    ) -> (ids: MLXArray, values: MLXArray)? {
        var compactParallelCandidate: (ids: MLXArray, values: MLXArray)?
        var residualL2Candidate: (ids: MLXArray, values: MLXArray)?
        var anchorCandidate: (ids: MLXArray, values: MLXArray)?
        var partitionCandidate: (ids: MLXArray, values: MLXArray)?
        var distributedCandidate: (ids: MLXArray, values: MLXArray)?
        var singleLocalCandidate: (ids: MLXArray, values: MLXArray)?
        var jointCandidate: (ids: MLXArray, values: MLXArray)?
        func compact() -> (ids: MLXArray, values: MLXArray)? {
            guard rows == 16, !ExactVerifyHead.checkEveryCall else { return nil }
            let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
            guard let summary = VerifyInt8Matmul.compactHead(
                activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
                negative: promptConstants.negative)
            else { return nil }
            let bounds = exactHeadBounds.get(weight: weight, scales: scales, biases: biases!)
            func parallel(tight: Bool = false, anchors: Bool = false, partitionL2: Bool = false, distributedAnchors: Bool = false, singleLocalAnchors: Bool = false, jointC2: MLXArray? = nil) -> (ids: MLXArray, values: MLXArray) {
                ExactVerifyHead.runCompactParallel(
                    rotated: rotated, summary: summary, ascale: activation.ascale, bounds: bounds,
                    weight: weight, scales: scales, biases: biases!, rows: rows,
                    negative: promptConstants.negative, error2: tight ? error2 : nil, anchors: anchors, partitionL2: partitionL2, distributedAnchors: distributedAnchors, singleLocalAnchors: singleLocalAnchors, jointC2: jointC2)
            }
            let canUseParallel = summary.values.size / rows % (4 * ExactVerifyHead.compactBlocks) == 0
            if canUseParallel, exactHeadBounds.compactParallelVerified == true {
                return parallel(tight: exactHeadBounds.residualL2Verified == true,
                    anchors: exactHeadBounds.exactAnchorsVerified == true,
                    partitionL2: exactHeadBounds.partitionL2Verified == true,
                    distributedAnchors: exactHeadBounds.distributedAnchorsVerified == true,
                    singleLocalAnchors: exactHeadBounds.singleLocalAnchorsVerified == true,
                    jointC2: exactHeadBounds.jointBoundsVerified == true
                        ? exactHeadBounds.getJoint(weight: weight, scales: scales, biases: biases!) : nil)
            }
            let original = ExactVerifyHead.runCompact(
                rotated: rotated, summary: summary, ascale: activation.ascale, bounds: bounds,
                weight: weight, scales: scales, biases: biases!, rows: rows,
                negative: promptConstants.negative)
            if canUseParallel, exactHeadBounds.compactParallelVerified == nil {
                compactParallelCandidate = parallel()
                if error2 != nil { residualL2Candidate = parallel(tight: true) }
                anchorCandidate = parallel(tight: error2 != nil, anchors: true)
                if error2 != nil, bounds.size == 19 * 32 {
                    partitionCandidate = parallel(tight: true, anchors: true, partitionL2: true)
                }
                distributedCandidate = parallel(tight: error2 != nil, anchors: true,
                    partitionL2: error2 != nil && bounds.size == 19 * 32, distributedAnchors: true)
                singleLocalCandidate = parallel(tight: error2 != nil, anchors: true,
                    partitionL2: error2 != nil && bounds.size == 19 * 32, singleLocalAnchors: true)
                if error2 != nil, bounds.size == 19 * 32,
                    let joint = exactHeadBounds.getJoint(weight: weight, scales: scales, biases: biases!)
                {
                    jointCandidate = parallel(tight: true, anchors: true, partitionL2: true,
                        singleLocalAnchors: true, jointC2: joint)
                }
            }
            return original
        }
        if ExactVerifyHead.verified[rows] == true, exactHeadBounds.compactVerified == true,
            let result = compact()
        { return result }
        let original = originalExactHeadTopTwo(activation: activation, rotated: rotated, rows: rows)
        if let original, ExactVerifyHead.verified[rows] == true,
            exactHeadBounds.compactVerified == nil, let result = compact()
        {
            var targets = [original.ids, original.values, result.ids, result.values]
            if let compactParallelCandidate { targets += [compactParallelCandidate.ids, compactParallelCandidate.values] }
            if let residualL2Candidate { targets += [residualL2Candidate.ids, residualL2Candidate.values] }
            if let anchorCandidate { targets += [anchorCandidate.ids, anchorCandidate.values] }
            if let partitionCandidate { targets += [partitionCandidate.ids, partitionCandidate.values] }
            if let distributedCandidate { targets += [distributedCandidate.ids, distributedCandidate.values] }
            if let singleLocalCandidate { targets += [singleLocalCandidate.ids, singleLocalCandidate.values] }
            if let jointCandidate { targets += [jointCandidate.ids, jointCandidate.values] }
            eval(targets)
            let match = ExactVerifyHead.sameBits(original, result)
            if let compactParallelCandidate {
                let parallelMatch = match && ExactVerifyHead.sameBits(result, compactParallelCandidate)
                exactHeadBounds.compactParallelVerified = parallelMatch
                if let residualL2Candidate {
                    let tightMatch = parallelMatch && ExactVerifyHead.sameBits(compactParallelCandidate, residualL2Candidate)
                    exactHeadBounds.residualL2Verified = tightMatch
                    ExactVerifyHead.log("XHEAD residual L2 rows \(rows): \(tightMatch ? "on" : "MISMATCH, existing bound kept")")
                }
                if let anchorCandidate {
                    let baseline = residualL2Candidate ?? compactParallelCandidate
                    let anchorMatch = parallelMatch
                        && (residualL2Candidate == nil || exactHeadBounds.residualL2Verified == true)
                        && ExactVerifyHead.sameBits(baseline, anchorCandidate)
                    exactHeadBounds.exactAnchorsVerified = anchorMatch
                    ExactVerifyHead.log("XHEAD exact anchors rows \(rows): \(anchorMatch ? "on" : "MISMATCH, existing cutoff kept")")
                    if let partitionCandidate {
                        let localMatch = anchorMatch && ExactVerifyHead.sameBits(anchorCandidate, partitionCandidate)
                        exactHeadBounds.partitionL2Verified = localMatch
                        ExactVerifyHead.log("XHEAD partition L2 rows \(rows): \(localMatch ? "on; partitions16" : "MISMATCH, global anchors kept")")
                    }
                    if let distributedCandidate {
                        let baseline = partitionCandidate ?? anchorCandidate
                        let distributedMatch = anchorMatch
                            && (partitionCandidate == nil || exactHeadBounds.partitionL2Verified == true)
                            && ExactVerifyHead.sameBits(baseline, distributedCandidate)
                        exactHeadBounds.distributedAnchorsVerified = distributedMatch
                        ExactVerifyHead.log("XHEAD distributed anchors rows \(rows): \(distributedMatch ? "on; scan leaders32" : "MISMATCH, existing anchors kept")")
                        if let singleLocalCandidate {
                            let singleMatch = distributedMatch && ExactVerifyHead.sameBits(distributedCandidate, singleLocalCandidate)
                            exactHeadBounds.singleLocalAnchorsVerified = singleMatch
                            ExactVerifyHead.log("XHEAD single local anchor rows \(rows): \(singleMatch ? "on; scan leaders16" : "MISMATCH, distributed anchors kept")")
                            if let jointCandidate {
                                let jointMatch = singleMatch && exactHeadBounds.partitionL2Verified == true
                                    && ExactVerifyHead.sameBits(singleLocalCandidate, jointCandidate)
                                exactHeadBounds.jointBoundsVerified = jointMatch
                                ExactVerifyHead.log("XHEAD joint K4 bound rows \(rows): \(jointMatch ? "on; cache64" : "MISMATCH, group bound kept")")
                            }
                        }
                    }
                }
                ExactVerifyHead.log("XHEAD compact parallel rows \(rows): \(parallelMatch ? "on; blocks16" : "MISMATCH, compact single-group path kept")")
            }
            exactHeadBounds.compactVerified = match
            ExactVerifyHead.log("XHEAD compact rows \(rows): \(match ? "on; top4/32, omitted >= tau uses full float, info count -1" : "MISMATCH, original exact path kept")")
        }
        return original
    }

    private func originalExactHeadTopTwo(
        activation: PromptInt8Activation, rotated: MLXArray, rows: Int
    ) -> (ids: MLXArray, values: MLXArray)? {
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let logits = VerifyInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative, streamedWeight: streamedVerifyWeight)
        else { return nil }
        let bounds = exactHeadBounds.get(weight: weight, scales: scales, biases: biases!)
        let parallel = ExactVerifyHead.parallelApplies(rows: rows, N: weight.dim(0))
        var result = ExactVerifyHead.run(
            rotated: rotated, logits: logits, ascale: activation.ascale, bounds: bounds,
            weight: weight, scales: scales, biases: biases!, rows: rows,
            negative: promptConstants.negative, parallel: parallel)
        let first = ExactVerifyHead.verified[rows] == nil
        let firstParallel = parallel && ExactVerifyHead.parallelVerified[rows] == nil
        guard first || firstParallel || ExactVerifyHead.checkEveryCall else {
            return (result.ids, result.values)
        }
        if parallel {
            // The block select against the one-threadgroup select: the same
            // flags and (E, |S|) on every row, the same pair on every row
            // that is not flagged, and the same final top-2.
            let serial = ExactVerifyHead.run(
                rotated: rotated, logits: logits, ascale: activation.ascale, bounds: bounds,
                weight: weight, scales: scales, biases: biases!, rows: rows,
                negative: promptConstants.negative)
            eval([result.ids, result.values, result.flag, result.info,
                  serial.ids, serial.values, serial.flag, serial.info])
            let match = ExactVerifyHead.sameBits((result.ids, result.values), (serial.ids, serial.values))
                && result.flag.asArray(Int32.self) == serial.flag.asArray(Int32.self)
                && result.info.asArray(Float.self).map(\.bitPattern)
                    == serial.info.asArray(Float.self).map(\.bitPattern)
            if !match, ExactVerifyHead.checkEveryCall {
                ExactVerifyHead.log("XHEAD CHECK FAILED rows \(rows): parallel select differs from select")
                preconditionFailure("XHEAD parallel select check failed at rows \(rows)")
            }
            if firstParallel {
                ExactVerifyHead.parallelVerified[rows] = match
                ExactVerifyHead.log(
                    "XHEAD parallel select rows \(rows): \(match ? "on" : "MISMATCH, select kept")")
            }
            if !match { result = serial }
            if !first && !ExactVerifyHead.checkEveryCall { return (result.ids, result.values) }
        }
        let full = ExactVerifyHead.run(
            rotated: rotated, logits: logits, ascale: activation.ascale, bounds: bounds,
            weight: weight, scales: scales, biases: biases!, rows: rows,
            negative: promptConstants.negative, force: true)
        let fused = rows <= 4
            ? Float16ConstantQMV.topTwo(rotated, weight: weight, scales: scales, biases: biases!)
            : nil
        var ratio: MLXArray? = nil
        if ExactVerifyHead.checkEveryCall {
            // Full float logits (rows kernel; 8 rows as 4 + 4) against the int8 logits.
            let flat = rotated.reshaped([rows, rotated.dim(-1)])
            let full32 = rows <= Float16ConstantQMV.maximumRows
                ? Float16ConstantQMV.matmul(
                    flat, weight: weight, scales: scales, biases: biases!,
                    negative: promptConstants.negative)
                : concatenated([
                    Float16ConstantQMV.matmul(flat[0 ..< 4], weight: weight, scales: scales, biases: biases!),
                    Float16ConstantQMV.matmul(flat[4 ..< rows], weight: weight, scales: scales, biases: biases!),
                ], axis: 0)
            let bound = result.info.reshaped([rows, 2])[0 ..< rows, 0 ..< 1].reshaped([rows])
            ratio = abs(full32 - logits).max(axis: -1) / bound
        }
        var arrays = [result.ids, result.values, result.flag, result.info, full.ids, full.values]
        if let fused { arrays += [fused.ids, fused.values] }
        if let ratio { arrays.append(ratio) }
        eval(arrays)
        var same = ExactVerifyHead.sameBits((result.ids, result.values), (full.ids, full.values))
        if let fused { same = same && ExactVerifyHead.sameBits((result.ids, result.values), fused) }
        let flags = result.flag.asArray(Int32.self)
        let info = result.info.asArray(Float.self)
        let worst = ratio.map { $0.asArray(Float.self).max() ?? 0 } ?? 0
        var stat = ExactVerifyHead.stats[rows] ?? (0, 0, 0, 0)
        stat.calls += 1
        stat.fallbacks += flags.filter { $0 != 0 }.count
        stat.maxRatio = max(stat.maxRatio, worst)
        stat.maxCount = max(stat.maxCount, stride(from: 1, to: info.count, by: 2).map { info[$0] }.max() ?? 0)
        ExactVerifyHead.stats[rows] = stat
        if first {
            ExactVerifyHead.verified[rows] = same
            ExactVerifyHead.log("XHEAD verify head rows \(rows): \(same ? "exact path on" : "MISMATCH, record head kept")")
        }
        if ExactVerifyHead.checkEveryCall {
            if !same || worst > 1 || !worst.isFinite {
                ExactVerifyHead.log("XHEAD CHECK FAILED rows \(rows) same \(same) ratio \(worst) info \(info)")
                preconditionFailure("XHEAD check failed at rows \(rows): same \(same), ratio \(worst)")
            }
            if stat.calls % 25 == 1 || flags.contains(where: { $0 != 0 }) {
                ExactVerifyHead.log(
                    "XHEAD stats rows \(rows) calls \(stat.calls) fallback_rows \(stat.fallbacks) "
                        + "max_err_over_E \(stat.maxRatio) max_candidates \(stat.maxCount) last_info \(info)")
            }
        }
        if !same { return nil }
        return (result.ids, result.values)
    }
}

public final class HadamardQuantizedEmbedding: Embedding, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode = .affine
    public let scales: MLXArray
    public let biases: MLXArray?
    public let transform: SignedBlockHadamard

    public override var shape: (Int, Int) { (weight.dim(0), transform.width) }

    public init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, transform: SignedBlockHadamard
    ) throws {
        try validateHadamardWeights(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, transform: transform)
        self.groupSize = groupSize
        self.bits = bits
        self.scales = scales
        self.biases = biases
        self.transform = transform
        super.init(weight: weight)
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let indices = x.flattened()
        let rows = dequantized(
            weight[indices], scales: scales[indices],
            biases: biases.map { $0[indices] }, groupSize: groupSize, bits: bits)
        return transform.inverse(rows).reshaped(x.shape + [transform.width])
    }

    public override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            transform(x), weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
    }
}

// MARK: - Draft-step front row and screened token (single row)

extension SignedBlockHadamard {
    fileprivate var draftSigns: MLXArray { signs }
}

private enum DraftStepRowKernels {
    static let header = """
        METAL_FUNC void dsr_radix16(thread float* x) {
          #pragma clang loop unroll(full)
          for (short j = 0; j < 16; j += 2) { float a = x[j], b = x[j+1]; x[j] = a + b; x[j+1] = a - b; }
          #pragma clang loop unroll(full)
          for (short j = 0; j < 16; j += 4) {
            #pragma clang loop unroll(full)
            for (short k = 0; k < 2; k++) { float a = x[j+k], b = x[j+k+2]; x[j+k] = a + b; x[j+k+2] = a - b; }
          }
          #pragma clang loop unroll(full)
          for (short j = 0; j < 16; j += 8) {
            #pragma clang loop unroll(full)
            for (short k = 0; k < 4; k++) { float a = x[j+k], b = x[j+k+4]; x[j+k] = a + b; x[j+k+4] = a - b; }
          }
          #pragma clang loop unroll(full)
          for (short k = 0; k < 8; k++) { float a = x[k], b = x[k+8]; x[k] = a + b; x[k+8] = a - b; }
        }
        // Unscaled 4096-point WHT over a 256-thread threadgroup (the passes of
        // signed_block_hadamard_4096): x_reg holds elements 16i..16i+15 on entry
        // and elements i + 256r on exit.
        METAL_FUNC void dsr_wht4096(thread float* x_reg, threadgroup float* buf, short i) {
          dsr_radix16(x_reg);
          #pragma clang loop unroll(full)
          for (short r = 0; r < 16; r++) buf[i * 17 + r] = x_reg[r];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          const short k1 = i & 15;
          const short r_base2 = (i - k1);
          #pragma clang loop unroll(full)
          for (short r = 0; r < 16; r++) x_reg[r] = buf[(r_base2 + r) * 17 + k1];
          dsr_radix16(x_reg);
          #pragma clang loop unroll(full)
          for (short r = 0; r < 16; r++) buf[(r_base2 + r) * 17 + k1] = x_reg[r];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          const short i_hi = i >> 4;
          const short i_lo = i & 15;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 16; r++) x_reg[r] = buf[(i_hi + 16 * r) * 17 + i_lo];
          dsr_radix16(x_reg);
        }
        // Sum over a 256-thread threadgroup; every thread receives the total.
        METAL_FUNC float dsr_sum256(float v, threadgroup float* red, uint sg, uint lane) {
          v = simd_sum(v);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (lane == 0) red[sg] = v;
          threadgroup_barrier(mem_flags::mem_threadgroup);
          float t = 0;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 8; j++) t += red[j];
          return t;
        }
        """

    static let front = MLXFast.metalKernel(
        name: "draft_step_front_row",
        inputNames: ["tok", "ew", "es", "eb", "signs", "hid", "ge", "gh", "eps"], outputNames: ["y"],
        source: """
            threadgroup float buf[256 * 17];
            threadgroup float red[8];
            const short i = short(thread_position_in_threadgroup.x);
            const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
            const int64_t t = int64_t(tok[0]);
            const uint q = ew[t * 256 + i];
            const float s = float(es[t * 32 + i / 8]);
            const float b = float(eb[t * 32 + i / 8]);
            float x_reg[16];
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) x_reg[r] = s * float((q >> (2 * r)) & 3u) + b;
            dsr_wht4096(x_reg, buf, i);
            float ss = 0;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              x_reg[r] *= 0.015625f * signs[i + 256 * r];
              ss += x_reg[r] * x_reg[r];
            }
            const float inv_e = metal::precise::rsqrt(dsr_sum256(ss, red, sg, lane) / 4096.0f + eps[0]);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) {
              const int idx = i + 256 * r;
              y[idx] = float(ge[idx]) * (x_reg[r] * inv_e);
            }
            float4 h4[4];
            float hs = 0;
            #pragma clang loop unroll(full)
            for (short v = 0; v < 4; v++) { h4[v] = *((const device float4*)(hid + 16 * i) + v); hs += dot(h4[v], h4[v]); }
            const float inv_h = metal::precise::rsqrt(dsr_sum256(hs, red, sg, lane) / 4096.0f + eps[0]);
            #pragma clang loop unroll(full)
            for (short v = 0; v < 4; v++) {
              const float4 gg = float4(*((const device half4*)(gh + 16 * i) + v));
              *((device float4*)(y + 4096 + 16 * i) + v) = gg * (h4[v] * inv_h);
            }
            """, header: header)

    static let finalNormRotate = MLXFast.metalKernel(
        name: "draft_step_final_norm_rotate",
        inputNames: ["z", "g", "signs", "eps"], outputNames: ["hidden", "rot"],
        source: """
            threadgroup float buf[256 * 17];
            threadgroup float red[8];
            const short i = short(thread_position_in_threadgroup.x);
            const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
            float4 z4[4];
            float ss = 0;
            #pragma clang loop unroll(full)
            for (short v = 0; v < 4; v++) { z4[v] = *((const device float4*)(z + 16 * i) + v); ss += dot(z4[v], z4[v]); }
            const float inv = metal::precise::rsqrt(dsr_sum256(ss, red, sg, lane) / 4096.0f + eps[0]);
            float x_reg[16];
            #pragma clang loop unroll(full)
            for (short v = 0; v < 4; v++) {
              const float4 n = float4(*((const device half4*)(g + 16 * i) + v)) * (z4[v] * inv);
              *((device float4*)(hidden + 16 * i) + v) = n;
              const float4 sv = *((const device float4*)(signs + 16 * i) + v);
              x_reg[4 * v] = n.x * sv.x; x_reg[4 * v + 1] = n.y * sv.y;
              x_reg[4 * v + 2] = n.z * sv.z; x_reg[4 * v + 3] = n.w * sv.w;
            }
            dsr_wht4096(x_reg, buf, i);
            #pragma clang loop unroll(full)
            for (short r = 0; r < 16; r++) rot[i + 256 * r] = x_reg[r] * 0.015625f;
            """, header: header)

    static let rescorePick = MLXFast.metalKernel(
        name: "draft_step_screen_rescore_pick",
        inputNames: ["x", "w", "scales", "biases", "partial"], outputNames: ["y", "row"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const int K = x_shape[1];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int c = threadgroup_position_in_grid.y * 2 + simd_gid;
            const float p = partial[c * 32 + simd_lid];
            const float m = simd_max(p);
            const uint pick = simd_min(p == m ? simd_lid : 32u);
            const int r = c * 32 + int(min(pick, 31u));
            const device uint8_t* ws = (const device uint8_t*)w + r * in_w + simd_lid * 4;
            const device half* sc = scales + r * in_g + simd_lid / 8;
            const device half* bi = biases + r * in_g + simd_lid / 8;
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float result = 0;
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              float x_thread[16];
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += xv.x + xv.y + xv.z + xv.w;
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y / 4.0f;
                x_thread[4 * i + 2] = xv.z / 16.0f;
                x_thread[4 * i + 3] = xv.w / 64.0f;
              }
              const uint packed_w = *(const device uint*)ws;
              float accum = 0;
              for (int i = 0; i < 4; i++) {
                const uint8_t w_byte = (packed_w >> (8 * i)) & 0xFF;
                accum += (x_thread[4 * i] * (w_byte & 0x03) + x_thread[4 * i + 1] * (w_byte & 0x0c)
                    + x_thread[4 * i + 2] * (w_byte & 0x30) + x_thread[4 * i + 3] * (w_byte & 0xc0));
              }
              result += float(*sc) * accum + sum * float(*bi);
              ws += 128;
              sc += 4;
              bi += 4;
              xp4 += 128;
            }
            const float total = simd_sum(result);
            if (simd_lid == 0) {
              y[c] = total;
              row[c] = r;
            }
            """)

    static let pickToken = MLXFast.metalKernel(
        name: "draft_step_screen_pick_token",
        inputNames: ["y", "row", "ids"], outputNames: ["tok", "local"],
        source: """
            threadgroup float bv[32];
            threadgroup int bi[32];
            const uint t = thread_position_in_threadgroup.x;
            const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
            const int n = y_shape[0];
            float best = -INFINITY;
            int bidx = 0x7fffffff;
            for (int j = int(t); j < n; j += 1024) {
              const float v = y[j];
              if (bidx == 0x7fffffff || v > best) { best = v; bidx = j; }
            }
            float m = simd_max(best);
            int cand = simd_min(best == m ? bidx : 0x7fffffff);
            if (lane == 0) { bv[sg] = m; bi[sg] = cand; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              best = bv[lane];
              bidx = bi[lane];
              m = simd_max(best);
              cand = simd_min(best == m ? bidx : 0x7fffffff);
              if (lane == 0) {
                const int r = row[min(cand, n - 1)];
                tok[0] = ids[r];
                local[0] = r;
              }
            }
            """)
}

extension HadamardQuantizedEmbedding {
    public func draftFrontRow(
        token: MLXArray, embedNorm: MLXArray, hidden: MLXArray, hiddenNorm: MLXArray, eps: MLXArray
    ) -> MLXArray? {
        guard bits == 2, groupSize == 128, transform.blockSize == 4096, transform.width == 4096,
            weight.dtype == .uint32, weight.ndim == 2, weight.dim(1) == 256,
            scales.dtype == .float16, biases?.dtype == .float16, let biases,
            token.dtype == .int32, token.size == 1,
            hidden.dtype == .float32, hidden.size == 4096,
            embedNorm.dtype == .float16, embedNorm.size == 4096,
            hiddenNorm.dtype == .float16, hiddenNorm.size == 4096,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        return DraftStepRowKernels.front(
            [token.reshaped([1]), weight, scales, biases, transform.draftSigns, hidden.reshaped([4096]),
             embedNorm, hiddenNorm, eps],
            grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[8192]], outputDTypes: [.float32])[0]
    }
}

extension HadamardQuantizedLinear {
    public func draftScreenedToken(
        _ z: MLXArray, normWeight: MLXArray, eps: MLXArray, ids: MLXArray, width: Int, chunk: Int
    ) -> (token: MLXArray, hidden: MLXArray, row: MLXArray)? {
        let K = transform.width
        let N = weight.dim(0)
        guard readsFloat16ConstantsInPlace, bias == nil, gdnLayout == nil, let biases,
            K == 4096, transform.blockSize == 4096, chunk == 32,
            width > 0, width < K, width % 512 == 0, N % chunk == 0, (N / chunk) % 2 == 0,
            z.dtype == .float32, z.size == K,
            normWeight.dtype == .float16, normWeight.size == K,
            ids.dtype == .int32, ids.size == N,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let normed = DraftStepRowKernels.finalNormRotate(
            [z.reshaped([K]), normWeight, transform.draftSigns, eps],
            grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 1, K], [1, K]], outputDTypes: [.float32, .float32])
        let rotated = normed[1]
        let partial = DraftScreenQMV.partial(
            [rotated, weight, scales, biases],
            template: [("P", width), ("NEG", promptConstants.negative ? 1 : 0)],
            grid: (32, N / 8 * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[N]], outputDTypes: [.float32])[0]
        let blocks = N / chunk
        let rescored = DraftStepRowKernels.rescorePick(
            [rotated, weight, scales, biases, partial],
            grid: (32, blocks, 1), threadGroup: (32, 2, 1),
            outputShapes: [[blocks], [blocks]], outputDTypes: [.float32, .int32])
        let picked = DraftStepRowKernels.pickToken(
            [rescored[0], rescored[1], ids], grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1], [1]], outputDTypes: [.int32, .int32])
        return (picked[0], normed[0], picked[1])
    }
}

// MARK: - The DFlash 2 draft-head screen

extension HadamardQuantizedLinear {
    public static let draftScreenGroups = 16

    private var draftScreenPacking: Bool {
        readsFloat16ConstantsInPlace && bits == 2 && groupSize == 128 && weight.dtype == .uint32
            && scales.dtype == .float16 && biases?.dtype == .float16 && bias == nil && gdnLayout == nil
            && transform.blockSize == 4096 && transform.width == 4096 && promptConstants.negative
            && weight.dim(0) % 32 == 0 && VerifyInt8Matmul.screenAvailable
    }

    private func draftScreenActivation(_ x: MLXArray) -> PromptInt8Activation? {
        guard draftScreenPacking, x.dtype == .float32, x.shape == [1, 15, transform.width],
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        return transform.quantizedForVerify(x)
    }

    public func draftScreen(
        _ x: MLXArray, shortlist: (MLXArray) -> MLXArray,
        retainedShortlist: ((MLXArray, MLXArray) -> MLXArray)? = nil
    ) -> (ids: MLXArray, values: MLXArray)? {
        guard let streamed = draftScreenState.streamed, let biases,
            let activation = draftScreenActivation(x)
        else { return nil }
        let copy: MLXArray? = streamedVerifyWeight
        guard (copy != nil) == streamed else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases)
        let ids: MLXArray
        if draftScreenState.retained, let retainedShortlist,
            let retained = VerifyInt8Matmul.draftScreenPartials(
                activation, weight: copy ?? weight, scalesT: scalesT, biasesT: biasesT,
                streamed: streamed, groups: Self.draftScreenGroups)
        {
            ids = retainedShortlist(retained.values, retained.tags)
        } else {
            guard let partial = VerifyInt8Matmul.screen(
                activation, weight: copy ?? weight, scalesT: scalesT, biasesT: biasesT,
                streamed: streamed, groups: Self.draftScreenGroups)
            else { return nil }
            ids = shortlist(partial)
        }
        if draftScreenState.fusedRescoreTop16 == true, draftScreenState.indexedRescore != true,
            let fused = VerifyInt8Matmul.gatherTop16(
                activation, weight: weight, scales: scales, biases: biases, ids: ids)
        { return (fused.ids, fused.values) }
        let values: MLXArray
        if draftScreenState.indexedRescore == true,
            let indexed = VerifyInt8Matmul.gatherIndexedNAX(
                activation, weight: weight, scales: scales, biases: biases, ids: ids)
        {
            values = indexed
        } else {
            values = VerifyInt8Matmul.gather(
                activation, weight: weight, scales: scales, biases: biases, ids: ids,
                groups: transform.width / 128, negative: true)
        }
        return (ids, values)
    }

    public func qualifyDraftScreen(shortlist: (MLXArray) -> MLXArray) -> String? {
        draftScreenState.streamed = nil
        draftScreenState.retained = false
        draftScreenState.indexedRescore = nil
        draftScreenState.fusedRescoreTop16 = nil
        let k: Int = transform.width
        let values: [Float] = (0 ..< 16 * k).map { (i: Int) -> Float in
            let t = Double(i)
            return Float(Foundation.sin(t * 0.6180339) * (1 + 0.5 * Foundation.cos(t * 0.0211)))
        }
        let x: MLXArray = MLXArray(values, [1, 16, k])[0..., 1..., 0...]
        guard x.dtype == .float32, x.shape == [1, 15, k], let biases, draftScreenPacking else {
            return "route, packing or shape does not qualify"
        }
        let streamed: Bool = streamedVerifyWeight != nil
        if streamed, let copied = streamedVerifyWeight, let activation = draftScreenActivation(x) {
            let pair = promptConstants.get(scales: scales, biases: biases)
            _ = try? withError {
                eval(activation.codes, activation.ascale, activation.rsb, copied, pair.0, pair.1)
                VerifyInt8Matmul.prepareHeadMetadata(
                    activation, weight: weight, copied: copied, scalesT: pair.0, biasesT: pair.1)
            }
        }
        DraftHeadScreenState.failure = nil
        var reason: String? = nil
        withErrorHandler({ message in DraftHeadScreenState.failure = String(message.prefix(160)) }) {
            reason = draftScreenMismatch(x, biases: biases, streamed: streamed, shortlist: shortlist)
        }
        if let failure = DraftHeadScreenState.failure { return "error: \(failure)" }
        if reason == nil { draftScreenState.streamed = streamed }
        return reason
    }

    public func qualifyDraftScreenRetention(
        shortlist: (MLXArray) -> [MLXArray],
        retainedShortlist: (MLXArray, MLXArray) -> [MLXArray]
    ) -> String? {
        let coldStart: UInt64 = DispatchTime.now().uptimeNanoseconds
        draftScreenState.retained = false
        guard let streamed = draftScreenState.streamed, let biases, weight.dim(0) == 98304 else {
            return "screen has not qualified"
        }
        let k: Int = transform.width
        let values: [Float] = (0 ..< 16 * k).map { (i: Int) -> Float in
            let t: Double = Double(i)
            return Float(Foundation.sin(t * 0.6180339) * (1 + 0.5 * Foundation.cos(t * 0.0211)))
        }
        let block: MLXArray = MLXArray(values, [1, 16, k])
        let x: MLXArray = block[0..., 1..., 0...]
        guard let activation = draftScreenActivation(x) else { return "no screen activation" }
        let copy: MLXArray? = streamedVerifyWeight
        let pair = promptConstants.get(scales: scales, biases: biases)
        guard (copy != nil) == streamed else { return "screen operand form changed" }
        DraftHeadScreenState.failure = nil
        var reason: String? = nil
        withErrorHandler({ message in DraftHeadScreenState.failure = String(message.prefix(160)) }) {
            guard let partial = VerifyInt8Matmul.screen(
                    activation, weight: copy ?? weight, scalesT: pair.0, biasesT: pair.1,
                    streamed: streamed, groups: Self.draftScreenGroups),
                let retained = VerifyInt8Matmul.draftScreenPartials(
                    activation, weight: copy ?? weight, scalesT: pair.0, biasesT: pair.1,
                    streamed: streamed, groups: Self.draftScreenGroups)
            else { reason = "retained screen kernel unavailable"; return }
            let reference: [MLXArray] = shortlist(partial)
            let candidate: [MLXArray] = retainedShortlist(retained.values, retained.tags)
            let idsSame: MLXArray = MLX.all(reference[0] .== candidate[0])
            let valuesSame: MLXArray = MLX.all(reference[1].view(dtype: .uint32) .== candidate[1].view(dtype: .uint32))
            let referenceFinite: MLXArray = MLX.all(MLX.isFinite(partial))
            let candidateFinite: MLXArray = MLX.all(MLX.isFinite(retained.values))
            let checks: [(String, MLXArray)] = [
                ("retained screen ids differ", idsSame),
                ("retained screen values differ", valuesSame),
                ("screen not finite", referenceFinite),
                ("retained screen not finite", candidateFinite),
            ]
            eval(checks.map { $0.1 })
            reason = checks.first { !$0.1.item(Bool.self) }?.0
        }
        if let failure = DraftHeadScreenState.failure { return "error: \(failure)" }
        if let reason { return reason }
        func build(_ method: Int) -> [MLXArray]? {
            if method == 0 {
                guard let partial = VerifyInt8Matmul.screen(
                    activation, weight: copy ?? weight, scalesT: pair.0, biasesT: pair.1,
                    streamed: streamed, groups: Self.draftScreenGroups)
                else { return nil }
                return shortlist(partial)
            }
            guard let retained = VerifyInt8Matmul.draftScreenPartials(
                activation, weight: copy ?? weight, scalesT: pair.0, biasesT: pair.1,
                streamed: streamed, groups: Self.draftScreenGroups)
            else { return nil }
            return retainedShortlist(retained.values, retained.tags)
        }
        var complete: Bool = true
        var warm: [UInt64] = [0, 0]
        var samples: [[UInt64]] = [[], []]
        var legs: [[[UInt64]]] = [[], []]
        for method in 0 ..< 2 {
            if let elapsed = Self.draftScreenRetainedTime({ build(method) }) { warm[method] = elapsed }
            else { complete = false; break }
        }
        if complete {
            let orders: [[Int]] = [[0, 1], [1, 0], [0, 1]]
            for order in orders {
                var round: [[UInt64]] = [[], []]
                for mirrored in [order, Array(order.reversed())] {
                    for method in mirrored {
                        if let elapsed = Self.draftScreenRetainedTime({ build(method) }) { round[method].append(elapsed) }
                        else { complete = false; break }
                    }
                    if !complete { break }
                }
                for method in 0 ..< 2 {
                    legs[method].append(round[method])
                    if round[method].count == 2 {
                        samples[method].append(round[method][0] + round[method][1])
                    }
                }
                if !complete { break }
            }
        }
        let retained: Bool = Self.draftScreenRetainedChoice(exact: complete, samples: samples)
        draftScreenState.retained = retained
        withExtendedLifetime((block, x, activation, copy, pair)) {}
        let cold: UInt64 = DispatchTime.now().uptimeNanoseconds - coldStart
        FileHandle.standardError.write(Data(
            "DF2 half-K retain select M15 K4096 SG16 N98304 streamed=\(streamed) exact=true complete=\(complete) methods=[materialized,retained] policy=all3_strict warm_ns=\(warm) legs_ns=\(legs) round_sums_ns=\(samples) retained=\(retained) calibration_ns=\(cold)\n".utf8))
        return retained ? nil : "no consistent producer/selection gain; materialized screen kept"
    }

    public func qualifyDraftIndexedRescore(shortlist: (MLXArray) -> MLXArray) -> String? {
        if let cached = draftScreenState.indexedRescore {
            return cached ? nil : "cached ALU rescore kept"
        }
        draftScreenState.indexedRescore = false
        let coldStart: UInt64 = DispatchTime.now().uptimeNanoseconds
        guard let streamed = draftScreenState.streamed, let biases, weight.dim(0) == 98304 else {
            return "screen has not qualified"
        }
        do {
            return try withError { () -> String? in
                let k: Int = transform.width
                let values: [Float] = (0 ..< 16 * k).map { (index: Int) -> Float in
                    let t: Double = Double(index)
                    return Float(Foundation.sin(t * 0.6180339) * (1 + 0.5 * Foundation.cos(t * 0.0211)))
                }
                let block: MLXArray = MLXArray(values, [1, 16, k])
                let x: MLXArray = block[0..., 1..., 0...]
                guard let activation = draftScreenActivation(x) else { return "no screen activation" }
                let copy: MLXArray? = streamedVerifyWeight
                let pair: (MLXArray, MLXArray) = promptConstants.get(scales: scales, biases: biases)
                guard (copy != nil) == streamed,
                    let partial = VerifyInt8Matmul.screen(
                        activation, weight: copy ?? weight, scalesT: pair.0, biasesT: pair.1,
                        streamed: streamed, groups: Self.draftScreenGroups)
                else { return "screen operand form changed" }
                let ids: MLXArray = shortlist(partial)
                eval([activation.codes, activation.ascale, activation.rsb, ids])
                Stream.gpu.synchronize()
                func build(_ method: Int) -> [MLXArray]? {
                    if method == 0 {
                        let output: MLXArray = VerifyInt8Matmul.gather(
                            activation, weight: weight, scales: scales, biases: biases, ids: ids,
                            groups: k / 128, negative: true)
                        return [output]
                    }
                    guard let output = VerifyInt8Matmul.gatherIndexedNAX(
                        activation, weight: weight, scales: scales, biases: biases, ids: ids)
                    else { return nil }
                    return [output]
                }
                guard let reference = build(0), let candidate = build(1) else { return "indexed N32 kernel unavailable" }
                let same: MLXArray = MLX.all(reference[0].view(dtype: .uint32) .== candidate[0].view(dtype: .uint32))
                let referenceFinite: MLXArray = MLX.all(MLX.isFinite(reference[0]))
                let candidateFinite: MLXArray = MLX.all(MLX.isFinite(candidate[0]))
                eval([same, referenceFinite, candidateFinite])
                guard referenceFinite.item(Bool.self), candidateFinite.item(Bool.self), same.item(Bool.self) else {
                    return "indexed N32 finite/exact check failed"
                }
                var complete: Bool = true
                var warm: [UInt64] = [0, 0]
                var samples: [[UInt64]] = [[], []]
                var legs: [[[UInt64]]] = [[], []]
                for method in 0 ..< 2 {
                    if let elapsed = Self.draftScreenRetainedTime({ build(method) }) { warm[method] = elapsed }
                    else { complete = false; break }
                }
                if complete {
                    let orders: [[Int]] = [[0, 1], [1, 0], [0, 1]]
                    for order in orders {
                        var round: [[UInt64]] = [[], []]
                        for mirrored in [order, Array(order.reversed())] {
                            for method in mirrored {
                                if let elapsed = Self.draftScreenRetainedTime({ build(method) }) { round[method].append(elapsed) }
                                else { complete = false; break }
                            }
                            if !complete { break }
                        }
                        for method in 0 ..< 2 {
                            legs[method].append(round[method])
                            if round[method].count == 2 {
                                samples[method].append(round[method][0] + round[method][1])
                            }
                        }
                        if !complete { break }
                    }
                }
                let selected: Bool = Self.draftScreenRetainedChoice(exact: complete, samples: samples)
                draftScreenState.indexedRescore = selected
                withExtendedLifetime((block, x, activation, copy, pair, ids, reference, candidate)) {}
                let cold: UInt64 = DispatchTime.now().uptimeNanoseconds - coldStart
                FileHandle.standardError.write(Data(
                    "DF2 full-K rescore select M15 K4096 N98304 P128 exact=true complete=\(complete) methods=[ALU,indexed_N32] policy=all3_strict warm_ns=\(warm) legs_ns=\(legs) round_sums_ns=\(samples) indexed=\(selected) calibration_ns=\(cold)\n".utf8))
                return selected ? nil : "no consistent indexed N32 gain; ALU rescore kept"
            }
        } catch { return "indexed N32 check error: \(error)" }
    }

    public func qualifyDraftRescoreTop16(
        shortlist: (MLXArray) -> MLXArray, reduce: ([MLXArray]) -> [MLXArray],
        tail: ([MLXArray]) -> [MLXArray]
    ) -> String? {
        if let cached = draftScreenState.fusedRescoreTop16 { return cached ? nil : "cached original kept" }
        draftScreenState.fusedRescoreTop16 = false
        guard draftScreenState.indexedRescore == false,
            let streamed = draftScreenState.streamed, let biases, weight.dim(0) == 98304
        else { return "screen/ALU rescore not selected; original kept" }
        let coldStart: UInt64 = DispatchTime.now().uptimeNanoseconds
        do {
            return try withError { () -> String? in
                let values: [Float] = (0 ..< 16 * 4096).map { (index: Int) -> Float in
                    let t: Double = Double(index)
                    return Float(Foundation.sin(t * 0.6180339) * (1 + 0.5 * Foundation.cos(t * 0.0211)))
                }
                let block: MLXArray = MLXArray(values, [1,16,4096])
                let x: MLXArray = block[0...,1...,0...]
                guard let activation = draftScreenActivation(x) else { return "no activation" }
                let copy: MLXArray? = streamedVerifyWeight
                let pair: (MLXArray, MLXArray) = promptConstants.get(scales: scales, biases: biases)
                guard (copy != nil) == streamed,
                    let partial = VerifyInt8Matmul.screen(
                        activation, weight: copy ?? weight, scalesT: pair.0, biasesT: pair.1,
                        streamed: streamed, groups: Self.draftScreenGroups)
                else { return "screen operand form changed" }
                let ids: MLXArray = shortlist(partial)
                eval(activation.codes, activation.ascale, activation.rsb, ids)
                Stream.gpu.synchronize()
                func pairs(_ method: Int) -> [MLXArray]? {
                    if method == 0 {
                        let output: MLXArray = VerifyInt8Matmul.gather(
                            activation, weight: weight, scales: scales, biases: biases,
                            ids: ids, groups: 32, negative: true)
                        return [ids,output]
                    }
                    guard let output = VerifyInt8Matmul.gatherTop16(
                        activation, weight: weight, scales: scales, biases: biases, ids: ids) else { return nil }
                    return [output.ids,output.values]
                }
                guard let original = pairs(0), let fused = pairs(1) else { return "fused kernel unavailable" }
                let reference: [MLXArray] = reduce(original)
                let oldTail: [MLXArray] = tail(original), newTail: [MLXArray] = tail(fused)
                guard reference.count == 2, oldTail.count == newTail.count else { return "selector output shape changed" }
                var checks: [MLXArray] = [MLX.all(reference[0] .== fused[0]),
                    MLX.all(reference[1].view(dtype: .uint32) .== fused[1].view(dtype: .uint32)),
                    MLX.all(MLX.isFinite(original[1])), MLX.all(MLX.isFinite(fused[1]))]
                for index in oldTail.indices { checks.append(MLX.all(oldTail[index] .== newTail[index])) }
                eval(checks)
                guard checks.allSatisfy({ $0.item(Bool.self) }) else { return "TOP16 or selector tail not finite/exact" }
                func build(_ method: Int) -> [MLXArray]? { pairs(method).map { tail($0) } }
                var complete: Bool = true
                var warm: [UInt64] = [0,0]
                var legs: [[[UInt64]]] = [[],[]]
                var sums: [[UInt64]] = [[],[]]
                for method in 0 ..< 2 {
                    if let elapsed = Self.draftScreenRetainedTime({ build(method) }) { warm[method] = elapsed }
                    else { complete = false; break }
                }
                if complete {
                    let orders: [[Int]] = [[0,1],[1,0],[0,1]]
                    for order in orders {
                        var round: [[UInt64]] = [[],[]]
                        for mirror in [order,Array(order.reversed())] {
                            for method in mirror {
                                if let elapsed = Self.draftScreenRetainedTime({ build(method) }) { round[method].append(elapsed) }
                                else { complete = false; break }
                            }
                            if !complete { break }
                        }
                        for method in 0 ..< 2 {
                            legs[method].append(round[method])
                            if round[method].count == 2 { sums[method].append(round[method][0] + round[method][1]) }
                        }
                        if !complete { break }
                    }
                }
                let selected: Bool = Self.draftScreenRetainedChoice(exact: complete, samples: sums)
                draftScreenState.fusedRescoreTop16 = selected
                withExtendedLifetime((block,x,activation,copy,pair,ids,original,fused,reference,oldTail,newTail)) {}
                let cold: UInt64 = DispatchTime.now().uptimeNanoseconds - coldStart
                FileHandle.standardError.write(Data(
                    "DF2 rescore TOP16 select M15 K4096 N98304 P128 exact=true complete=\(complete) methods=[ALU_tree,fused_TOP16_tree] policy=all3_strict warm_ns=\(warm) legs_ns=\(legs) round_sums_ns=\(sums) selected=\(selected) calibration_ns=\(cold)\n".utf8))
                return selected ? nil : "complete sequence has no consistent gain; original kept"
            }
        } catch { return "fused TOP16 check error: \(error)" }
    }

    static func draftScreenRetainedChoice(exact: Bool, samples: [[UInt64]]) -> Bool {
        guard exact, samples.count == 2, samples[0].count == 3, samples[1].count == 3 else { return false }
        return zip(samples[0], samples[1]).allSatisfy {
            $0.0 > 0 && $0.1 > 0 && $0.1 < $0.0
        }
    }

    private static func draftScreenRetainedTime(_ build: () -> [MLXArray]?) -> UInt64? {
        do {
            return try withError { () -> UInt64? in
                Stream.gpu.synchronize()
                let start: UInt64 = DispatchTime.now().uptimeNanoseconds
                guard let outputs = build() else { return nil }
                eval(outputs)
                Stream.gpu.synchronize()
                let end: UInt64 = DispatchTime.now().uptimeNanoseconds
                withExtendedLifetime(outputs) {}
                return end > start ? end - start : nil
            }
        } catch { return nil }
    }

    private func draftScreenMismatch(
        _ x: MLXArray, biases: MLXArray, streamed: Bool, shortlist: (MLXArray) -> MLXArray
    ) -> String? {
        guard let activation = draftScreenActivation(x) else { return "no verify activation" }
        let n: Int = weight.dim(0)
        let rows: Int = activation.rows
        let copy: MLXArray? = streamedVerifyWeight
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases)
        guard (copy != nil) == streamed,
            let partial = VerifyInt8Matmul.screen(
                activation, weight: copy ?? weight, scalesT: scalesT, biasesT: biasesT,
                streamed: streamed, groups: Self.draftScreenGroups)
        else { return "screen kernel unavailable" }
        let reference: MLXArray = VerifyInt8Matmul.gather(
            activation, weight: weight, scales: scales, biases: biases, ids: nil,
            groups: Self.draftScreenGroups, negative: true)
        let full: MLXArray = self(x)
        guard full.dtype == .float32, full.shape == [1, rows, n] else { return "full head shape or dtype" }
        let fullRows: MLXArray = full.reshaped([rows, n])
        let ids: MLXArray = shortlist(partial)
        let referenceIds: MLXArray = shortlist(reference)
        let columns: [Int32] = stride(from: 0, to: n, by: 97).map { Int32($0) } + [Int32(n - 1)]
        let spread: MLXArray = contiguous(
            broadcast(MLXArray(columns).reshaped([1, columns.count]), to: [rows, columns.count]))
        let groups: Int = transform.width / 128
        func exact(_ at: MLXArray) -> MLXArray {
            let mine: MLXArray = VerifyInt8Matmul.gather(
                activation, weight: weight, scales: scales, biases: biases, ids: at,
                groups: groups, negative: true)
            let head: MLXArray = takeAlong(fullRows, at, axis: -1)
            return MLX.all(mine.view(dtype: .uint32) .== head.view(dtype: .uint32))
        }
        let samePartial: MLXArray = MLX.all(partial.view(dtype: .uint32) .== reference.view(dtype: .uint32))
        let finite: MLXArray = MLX.all(abs(partial) .< MLXArray(Float.infinity))
        let checks: [(String, MLXArray)] = [
            ("screen differs from the reference partial", samePartial),
            ("screen not finite", finite),
            ("shortlists differ", MLX.all(ids .== referenceIds)),
            ("rescore differs at the shortlist", exact(ids)),
            ("rescore differs at the spread ids", exact(spread)),
        ]
        eval(checks.map { $0.1 })
        return checks.first { !$0.1.item(Bool.self) }?.0
    }
}
