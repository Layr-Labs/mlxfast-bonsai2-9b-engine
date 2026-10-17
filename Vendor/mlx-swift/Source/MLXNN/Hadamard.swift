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

/// A normalized block Walsh-Hadamard transform with explicit input signs.
///
/// Forward evaluation computes `(x * signs) H`; inverse evaluation computes
/// `(x H) * signs`. Signs cover the entire final dimension, not just one block.
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

    /// Check serialized signs against the independently decoded metadata.
    public func matches(signs values: [Float]) -> Bool { values == signValues }

    func isIdentical(to other: SignedBlockHadamard) -> Bool {
        blockSize == other.blockSize && width == other.width && signValues == other.signValues
    }

    // Fused transform from Yukon 9727ebfe (9B PR #35); native MLX radix order.
    private static let residualNormSource = """

            threadgroup float norm_sums[32];
            threadgroup float norm_inv[1];
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
            if (norm_sg == 0) {
              const float total = simd_sum(norm_sums[norm_lane]);
              if (norm_lane == 0) norm_inv[0] = metal::precise::rsqrt(total / 4096.0f + epsilon[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """

    private static func fused4096Source(residualNorm: Bool) -> String {
        let prefix = residualNorm ? residualNormSource : ""
        let valueRead = residualNorm ? """
            const float4 raw = in4[v] + *((const device float4*)(residual + in_offset) + v);
            *((device float4*)(residual_out + in_offset) + v) = raw;
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + v);
            float4 val = (weight * (raw * norm_inv[0])) * s4[v];
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

    /// The forward transform of `fused4096Kernel` (same passes, same order,
    /// same 1/64 scale), quantized per row and 128-group for the prompt-width
    /// integer matmul (27b 5fbfa003 / 0e1314b2). Each lane re-reads 16
    /// consecutive outputs, so a 128-group is an 8-lane cluster: `amax` is
    /// the cluster max of |v|, `q = rint(v * (127 / amax))` as int8, written
    /// in the staged weight order (position p of each 16-block holds element
    /// 4 * (p % 4) + p / 4), and `ascale = amax / 127`, `rsb = ascale *
    /// sum(q)` per group in the row-tiled order the matmuls read
    /// (`PromptInt8Matmul.constantIndexSource`). Every operation is exact or a
    /// single IEEE rounding (sign multiply, butterfly adds in a fixed order, a
    /// power-of-two scale, max, a precise divide, rint, an integer sum), so the
    /// codes and constants do not depend on the GPU that computes them.
    /// `VERIFY_ROWS > 0` is the verify-width form (`VerifyInt8Matmul`): the
    /// codes fill a 16-row tile whose rows past `VERIFY_ROWS` are zero, and the
    /// constants are `[rows, groups]`. The values are the same either way.
    // GDN output gate formed inside the quantizing rotation: per-head (128
    // wide) RMSNorm of `x` with the arithmetic of MLX `rms_single_row` at
    // axis 128 (one simdgroup; lane l sums elements 4l..4l+3 in order, one
    // `simd_sum`, `precise::rsqrt(total / 128 + eps)`, `w * (x * inv)`),
    // times MLX `silu(gate)` (`gate * sigmoid(gate)` with MLX's stable
    // sigmoid), the product taken as `silu * normed`. Each head's 32 lane
    // partials are placed in the lanes MLX gives them before the `simd_sum`.
    private static let gatedNormSource = """

            threadgroup float head_part[32 * 32];
            threadgroup float head_inv[32];
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
            {
              const uint norm_lane = thread_index_in_simdgroup;
              for (uint h = simdgroup_index_in_threadgroup; h < 32u; h += 8u) {
                const float total = simd_sum(head_part[h * 32u + norm_lane]);
                if (norm_lane == 0) head_inv[h] = metal::precise::rsqrt(total / 128.0f + epsilon[0]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """

    private static func quantized4096Source(
        swiglu: Bool, residualNorm: Bool = false, gatedNorm: Bool = false,
        attentionGate: Bool = false, normalizedOutput: Bool = false,
        rotatedOutput: Bool = false
    ) -> String {
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
            const float4 normed = weight * (raw * norm_inv[0]);
            *((device float4*)(normed_out + in_offset) + w) = normed;
            float4 val = normed * s4[w];
            """ : residualNorm ? """
            const float4 raw = in4[w] + *((const device float4*)(residual + in_offset) + w);
            *((device float4*)(residual_out + in_offset) + w) = raw;
            const float4 weight = *((const device float4*)(norm_weight + s_offset) + w);
            float4 val = (weight * (raw * norm_inv[0])) * s4[w];
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
            const float4 normed = nw * (in4[w] * head_inv[uint(i) >> 3]);
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
            }
            """
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

    /// `forward(attention * sigmoid(gate))` quantized for `VerifyInt8Matmul`,
    /// for a gated full-attention output: `attention` is the `[1, 16, rows,
    /// 256]` attention result in its head-major layout and the gate is the
    /// second half of each 512-wide head of the `[1, rows, 8192]` query
    /// projection. The kernel reads both in place, so the head transpose and
    /// the gate slice need no copies, and forms MLX's `x * sigmoid(gate)`
    /// (stable sigmoid) bit for bit. Nil when the shapes do not qualify.
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
        let tile = VerifyInt8Matmul.tileRows
        let groups = rows * (width / 128)
        let outputs = Self.attentionGateQuantized4096Kernel(
            [attention, queryProjection.reshaped([rows, 8192]), signs],
            template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", rows)],
            grid: (256, tile, 1), threadGroup: (256, 1, 1),
            outputShapes: [[tile, width], [groups], [groups]],
            outputDTypes: [.int8, .float32, .float32])
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    /// `forward(silu(gate) * rmsNorm(x, weight))` quantized for
    /// `VerifyInt8Matmul`, with the RMSNorm over 128-wide heads: the GDN
    /// output gate, its norm and the out-projection rotation in one kernel.
    /// The values entering the rotation are bit-identical to MLX `rmsNorm`,
    /// `silu` and the multiply. Nil when the shape is not a verify-width
    /// FP32 4096-wide input.
    func quantizedForVerifyGatedNorm(
        _ x: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float
    ) -> PromptInt8Activation? {
        let rows = x.size / width
        guard blockSize == 4096, width == 4096, x.dtype == .float32, gate.dtype == .float32,
            x.size == gate.size, x.dim(-1) == 128 || x.dim(-1) == 4096,
            weight.dtype == .float32, weight.shape == [128],
            VerifyInt8Matmul.rowRange.contains(rows), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let tile = VerifyInt8Matmul.tileRows
        let groups = rows * (width / 128)
        let outputs = Self.gatedNormQuantized4096Kernel(
            [x.reshaped([rows, width]), gate.reshaped([rows, width]), weight, MLXArray([eps]), signs],
            template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", rows)],
            grid: (256, tile, 1), threadGroup: (256, 1, 1),
            outputShapes: [[tile, width], [groups], [groups]],
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

    /// `residualNorm` at verify width that also returns the normalized
    /// (unrotated) rows: `[codes, ascale, rsb, x + residual, rmsNorm(x +
    /// residual, weight)]`. The sum and the normalized values are the bits of
    /// MLX's add and `rmsNorm`; the codes are those of `quantizedForVerify`
    /// on the normalized rows.
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
        let tile = VerifyInt8Matmul.tileRows
        return Self.residualNormNormedQuantized4096Kernel(
            [x, residual, weight, MLXArray([eps]), signs],
            template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", rows)],
            grid: (256, tile, 1), threadGroup: (256, 1, 1),
            outputShapes: [[tile, width], [rows * 32], [rows * 32], x.shape, x.shape],
            outputDTypes: [.int8, .float32, .float32, .float32, .float32])
    }

    /// `residualNormVerifyNormalized` at prompt width (`PromptInt8Matmul`
    /// rows): the same kernel, one threadgroup per row, with the prompt
    /// constant order (`VERIFY_ROWS` 0). Returns `[codes, ascale, rsb, x +
    /// residual, rmsNorm(x + residual, weight)]`; the codes and constants are
    /// those of `quantizedForPrompt` on the normalized rows.
    func residualNormPromptNormalized(
        _ x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float
    ) -> [MLXArray]? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32,
            residual.dtype == .float32, residual.shape == x.shape,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            x.size <= Int(Int32.max), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / width
        guard PromptInt8Matmul.applies(rows: rows) else { return nil }
        return Self.residualNormNormedQuantized4096Kernel(
            [x, residual, weight, MLXArray([eps]), signs],
            template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0)],
            grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, width], [rows * 32], [rows * 32], x.shape, x.shape],
            outputDTypes: [.int8, .float32, .float32, .float32, .float32])
    }

    /// Whether `residualNormPromptNormalized` gives, bit for bit, MLX's add,
    /// `MLXFast.rmsNorm` and `quantizedForPrompt` of the normalized rows, on
    /// synthetic rows at widths 512 and 128. Run once, at load.
    nonisolated(unsafe) private static var boundaryCheckFailed = false
    nonisolated(unsafe) private static var boundaryCheckNote = "ok"

    func promptBoundaryMatchesSeparate() -> (Bool, String) {
        Self.boundaryCheckFailed = false
        Self.boundaryCheckNote = "ok"
        var note: String? = nil
        var matched = false
        withErrorHandler({ message in
            if !SignedBlockHadamard.boundaryCheckFailed {
                SignedBlockHadamard.boundaryCheckNote = String(message.prefix(160))
            }
            SignedBlockHadamard.boundaryCheckFailed = true
        }) {
            for (trial, rows) in [512, 128].enumerated() {
                let key = MLXRandom.key(UInt64(41 + trial))
                let keys = MLXRandom.split(key: key, into: 3)
                let x = MLXRandom.normal([1, rows, 4096], key: keys[0]) * Float(4 + 20 * trial)
                let delta = MLXRandom.normal([1, rows, 4096], key: keys[1]) * 0.75
                let weight = MLXRandom.uniform(Float(0.05) ..< Float(2.5), [4096], key: keys[2])
                let eps: Float = 1e-6
                guard let fused = residualNormPromptNormalized(x, residual: delta, weight: weight, eps: eps)
                else { note = "route not taken at rows \(rows)"; return }
                let sum = x + delta
                let normed = MLXFast.rmsNorm(sum, weight: weight, eps: eps)
                guard let separate = quantizedForPrompt(normed)
                else { note = "separate route not taken at rows \(rows)"; return }
                let pairs: [(MLXArray, MLXArray)] = [
                    (fused[0].view(dtype: .uint8), separate.codes.view(dtype: .uint8)),
                    (fused[1].view(dtype: .uint32), separate.ascale.view(dtype: .uint32)),
                    (fused[2].view(dtype: .uint32), separate.rsb.view(dtype: .uint32)),
                    (fused[3].view(dtype: .uint32), sum.view(dtype: .uint32)),
                    (fused[4].view(dtype: .uint32), normed.view(dtype: .uint32)),
                ]
                for (index, (a, b)) in pairs.enumerated() {
                    let equal = a.shape == b.shape ? MLX.all(a .== b) : MLXArray(false)
                    eval(equal)
                    if Self.boundaryCheckFailed { return }
                    if !equal.item(Bool.self) {
                        note = "output \(index) differs at rows \(rows)"
                        return
                    }
                }
            }
            matched = !Self.boundaryCheckFailed
        }
        return (matched, note ?? Self.boundaryCheckNote)
    }

    private static let residualNormQuantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_residual_norm_4096_q8",
        inputNames: ["x", "residual", "norm_weight", "epsilon", "signs"],
        outputNames: ["codes", "ascale", "rsb", "residual_out"],
        source: quantized4096Source(swiglu: false, residualNorm: true),
        header: quantized4096Header, ensureRowContiguous: true)

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
            let tile = VerifyInt8Matmul.tileRows
            return Self.residualNormQuantized4096Kernel(
                inputs, template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", rows)],
                grid: (256, tile, 1), threadGroup: (256, 1, 1),
                outputShapes: [[tile, width], [rows * 32], [rows * 32], x.shape],
                outputDTypes: [.int8, .float32, .float32, .float32])
        }
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

    /// `forward(x)` quantized for `PromptInt8Matmul`: int8 codes `[rows,
    /// width]` and the FP32 group constants `[rows * width / 128]`. Nil when
    /// the shape is not a prompt-width FP32 block-4096 input.
    func quantizedForPrompt(_ x: MLXArray, up: MLXArray? = nil) -> PromptInt8Activation? {
        validate(x)
        let rows = x.size / width
        guard blockSize == 4096, x.dtype == .float32, rows % 64 == 0, rows > 0,
            x.size <= Int(Int32.max), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        if let up, up.shape != x.shape || up.dtype != x.dtype { return nil }
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

    /// `forward(x)` quantized for `VerifyInt8Matmul`, with the codes and
    /// constants of `quantizedForPrompt`: codes `[16, width]` (rows past
    /// `rows` zero) and constants `[rows, width / 128]`. Nil when the shape is
    /// not a verify-width FP32 block-4096 input.
    /// With `up`, the input is `silu(x) * up`, formed inside the rotation as
    /// in `quantizedForPrompt`.
    func quantizedForVerify(_ x: MLXArray, up: MLXArray? = nil) -> PromptInt8Activation? {
        validate(x)
        let rows = x.size / width
        guard blockSize == 4096, x.dtype == .float32, VerifyInt8Matmul.rowRange.contains(rows),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        if let up, up.shape != x.shape || up.dtype != x.dtype { return nil }
        let tile = VerifyInt8Matmul.tileRows
        let groups = rows * (width / 128)
        let kernel = up == nil ? Self.quantized4096Kernel : Self.quantizedSwiGLU4096Kernel
        let outputs = kernel(
            [x.reshaped([rows, width])] + (up.map { [$0.reshaped([rows, width])] } ?? []) + [signs],
            template: [("BLOCKS_PER_ROW", width / 4096), ("VERIFY_ROWS", rows)],
            grid: (256, tile * (width / 4096), 1), threadGroup: (256, 1, 1),
            outputShapes: [[tile, width], [groups], [groups]],
            outputDTypes: [.int8, .float32, .float32])
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    /// `quantizedForVerify` that also emits the exact FP32 rotated tensor
    /// `[rows, width]` from the same 3-pass Hadamard butterfly.
    func quantizedForVerifyWithRotated(_ x: MLXArray) -> (activation: PromptInt8Activation, rotated: MLXArray)? {
        validate(x)
        let rows = x.size / width
        guard blockSize == 4096, x.dtype == .float32, VerifyInt8Matmul.rowRange.contains(rows),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let tile = VerifyInt8Matmul.tileRows
        let groups = rows * (width / 128)
        let outputs = Self.quantized4096RotatedKernel(
            [x.reshaped([rows, width]), signs],
            template: [("BLOCKS_PER_ROW", width / 4096), ("VERIFY_ROWS", rows)],
            grid: (256, tile * (width / 4096), 1), threadGroup: (256, 1, 1),
            outputShapes: [[tile, width], [groups], [groups], x.shape],
            outputDTypes: [.int8, .float32, .float32, .float32])
        return (
            PromptInt8Activation(
                codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width),
            outputs[3])
    }

    /// Transform activations before multiplication by folded weights.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        validate(x)
        if let fused = fused4096(x, inverse: false) { return fused }
        return hadamardTransform((x.asType(.float32) * signs).reshaped([-1, blockSize]))
            .reshaped(x.shape).asType(x.dtype)
    }

    /// Recover the original basis after looking up folded embedding rows.
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

/// The version-1 `prism.hadamard.*` metadata exported as a JSON object.
///
/// This decodes extracted metadata, not GGUF bytes. Tensor names remain in the
/// source namespace. A model loader must map them to its modules and honor
/// `gdnVGrouped` when arranging GDN values before an output projection.
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

/// Reorders tiled GDN values into the grouped order of folded output weights.
///
/// The final axis changes from `[repeat, keyHead, headDimension]` to
/// `[keyHead, repeat, headDimension]` before signs and Hadamard are applied.
/// Use only when the upstream GDN produces tiled values; already grouped values
/// must not be permuted again.
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

/// Packed 2-bit, group-128 affine matmul of FP32 activations against F16
/// scales and biases that the kernels read in place.
///
/// `quantizedMM` casts the scales and biases to the activation dtype before it
/// dispatches. With FP32 activations and F16 constants, every call therefore
/// runs two cast kernels, and the matmul then reads the widened copies. The
/// Bonsai 2 9B pack has exactly that pairing on all 202 packed modules, so a
/// decode step and an MTP verify forward paid the casts on every projection.
///
/// One row (a decode step) runs MLX's `qmv_fast` with only the constant loads
/// changed: each F16 value is widened in a register. The widening is exact,
/// and the arithmetic and its order are MLX's, so the output is bit-identical
/// to `quantizedMM`.
///
/// Two to seven rows (an MTP verify) run `rows` below. At four rows a 2-bit
/// matvec is bound by FP32 FMAs, not by the weight read: the M5 Max reads the
/// 14 MB gate projection in 24 us, and its 4 x 50M FMAs take 26 us at the
/// measured 7.6 T FMA/s peak. So the kernel spends as few instructions per
/// weight as it can. It dequantizes each weight once (`s * q + b`, the
/// dequantization `qmv_wide` and `qmm` use) and then does one FMA per weight
/// per row. Each lane keeps `qmv_fast`'s layout: 16 values of each 512-wide
/// step for 4 output rows, so one float4 activation load serves 4 rows. The
/// reduction order is not `qmv_fast`'s, which `qmv_wide` did not keep either.
/// A row's result does not depend on the other rows in the call.
private enum Float16ConstantQMV {
    /// MLX `qmv_fast_impl` for 2 bits: 16 values per lane, 4 rows per
    /// simdgroup, 2 simdgroups per threadgroup.
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

    /// `rows` activation rows against `outputs` output rows per simdgroup, 2
    /// simdgroups per threadgroup. Each lane dequantizes its 16 values of a
    /// 512-wide step for every output row once, then accumulates them into
    /// all `rows` results.
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

    /// One simdgroup's `result[rows][outputs]` for the output rows starting at
    /// `out_row`. `rows` and `topTwoPartial` share this text, so a fused top-2
    /// sees the very values `rows` would store.
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

    /// The widest MTP verify: a draft chain of at most six tokens (the Qwen
    /// 3.5 drafter's cap) plus its seed.
    static let maximumRows = 7

    /// Output rows per simdgroup for `rows` activation rows: four, so one
    /// activation load serves four output rows, and two at five to seven
    /// rows. Five rows at four outputs need 20 + 20 live accumulators and
    /// dequantized values; that spills and runs 2.2x slower on box 2. At six
    /// and seven rows, one output row per simdgroup runs 1.9x slower than
    /// two, and two beats `quantizedMM` by 20 % per verify.
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

    /// `rows` with the store replaced by a running top-2 per activation row.
    /// A threadgroup covers `blocks` consecutive spans of `2 * outputs` output
    /// rows and writes one partial pair per activation row, so the
    /// `rows x N` logits are never written.
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

    /// One threadgroup per activation row merges its partial pairs.
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

    /// Output spans of `2 * outputs` rows per top-2 threadgroup.
    static let topTwoBlocks = 8

    /// Exact top-2 of `x @ dequantized(w).T` for 2 to 4 activation rows,
    /// equal in ids and values to running `rows` and reducing its output; nil
    /// when the shape has no fused path.
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

/// Exact verify-head top-2 at 2 to 8 rows decided by float arithmetic only,
/// at about the cost of one int8 head pass.
///
/// The int8 verify route (`VerifyInt8Matmul`) computes the head logits L8
/// from rounded activation codes, so a near tie can order two tokens
/// differently than the float logits F that `Float16ConstantQMV.topTwo`
/// reduces. Here L8 only screens: it picks the candidates, and float values
/// with exactly the per-row arithmetic of `rowsAccumulate` decide.
///
/// For a row, `L8[n] = sum_k xhat_k w_nk` with `xhat = ascale * q` and
/// `F[n] = sum_k v_k w_nk`, where v is the rotated row and `q = rint(v /
/// ascale)` per 128-group. So `|F[n] - L8[n]| <= E` with
///
///     E = 1.001 * sum_g ascale_g * (Amax_g * (0.5 + 127 gf) + Bmax_g * gi)
///
/// `Amax_g = max_n sum_{k in g} |s_ng c_nk + b_ng|` bounds the rounding of
/// the codes (`|v - xhat| <= ascale / 2`), `Bmax_g = max_n 127 (|s_ng| sum_k
/// c_nk + 128 |b_ng|)` bounds the size of the int8 route's terms, and gf, gi
/// cover the FP32 rounding of both sums with a wide margin. Amax and Bmax are
/// properties of the head weights, computed once.
///
/// With i2 the second-largest L8 value of the row and `tau <= i2 - 2E`, every
/// id outside `S = {n : L8[n] >= tau}` has `F[n] <= L8[n] + E < i2 - E`,
/// while the two int8 leaders (both in S) have `F >= i2 - E`. So the float
/// top-2 lies in S, and the top-2 of S's float values in the float top-2
/// order (value descending, id ascending) is the float top-2. A row whose S
/// holds more than `cap` ids, or that shows a non-finite value, runs the full
/// float top-2 instead (`guardedPartial`, a no-op for every other row).
enum ExactVerifyHead {
    static let cap = 256
    static let threads = 512

    /// One threadgroup per 128-group g: `bounds[g] = Amax_g`, `bounds[G + g]
    /// = Bmax_g`.
    static let bounds = MLXFast.metalKernel(
        name: "hadamard_xhead_bounds",
        inputNames: ["w", "scales", "biases"], outputNames: ["bounds"],
        source: """
            const uint g = threadgroup_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const int N = w_shape[0];
            const int words = w_shape[1];
            const int G = scales_shape[1];
            float amax = 0.0f, bmax = 0.0f;
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
              amax = max(amax, a);
              bmax = max(bmax, bb);
            }
            amax = simd_max(amax);
            bmax = simd_max(bmax);
            threadgroup float sa[8], sb[8];
            if (simd_lid == 0) { sa[simd_gid] = amax; sb[simd_gid] = bmax; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              for (int i = 1; i < 8; i++) { amax = max(amax, sa[i]); bmax = max(bmax, sb[i]); }
              bounds[g] = amax;
              bounds[G + g] = bmax;
            }
            """)

    /// One threadgroup of `threads` per row: int8 top-2, E, the candidate
    /// set, its float values (`rowsAccumulate` at one row and one output
    /// row, so each value is the one `topTwoPartial` computes for that id)
    /// and their top-2. `info[row] = (E, |S|)`.
    static let select = MLXFast.metalKernel(
        name: "hadamard_xhead_select",
        inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases"],
        outputNames: ["sel_ids", "sel_values", "flag", "info"],
        source: """
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
            """,
        header: Float16ConstantQMV.topTwoHeader)

    /// `topTwoPartial` for every row when any row is flagged; otherwise each
    /// threadgroup returns at once. Rows are independent in `rowsAccumulate`,
    /// so the values match the 2-4 row kernel for each row.
    static let guardedPartial = MLXFast.metalKernel(
        name: "hadamard_xhead_guarded_top2",
        inputNames: ["x", "w", "scales", "biases", "flag"],
        outputNames: ["partial_ids", "partial_values"],
        source: """
            {
              bool needed = false;
              for (int v = 0; v < rows; v++) needed = needed || (flag[v] != 0);
              if (!needed) return;
            }
            """ + Float16ConstantQMV.topTwoPartialSource,
        header: Float16ConstantQMV.topTwoHeader)

    /// Per row: the selected pair, or `topTwoFinalize` over the full-float
    /// partials for a flagged row.
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

    /// `DARKBLOOM_BONSAI_XHEAD=0` keeps the record's head (float top-2 at 2-4
    /// rows, int8 logits at 5-8); `=5` takes this path at 5-8 rows only.
    static let rowRange: ClosedRange<Int>? = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_XHEAD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["0", "false", "no", "off"].contains(value ?? "") { return nil }
        if let value, let low = Int(value), (2 ... 8).contains(low) { return low ... 8 }
        return 2 ... 8
    }()

    /// `DARKBLOOM_BONSAI_XHEAD_CHECK=1`: every call is also checked against
    /// the full float top-2 and the full float logits (development only).
    static let checkEveryCall: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_XHEAD_CHECK"] == "1"

    /// Row counts whose first call passed (true) or failed (false) the
    /// load-time comparison with the full float top-2.
    nonisolated(unsafe) static var verified: [Int: Bool] = [:]
    nonisolated(unsafe) static var stats: [Int: (calls: Int, fallbacks: Int, maxRatio: Float, maxCount: Float)] = [:]

    static func log(_ message: String) {
        FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
        os_log("%{public}@", log: .default, type: .default, message as NSString)
    }

    /// `(ids, values, flag, info)` for `rows` rows; `force` flags every row
    /// (the full float top-2 through the same kernels).
    static func run(
        rotated: MLXArray, logits: MLXArray, ascale: MLXArray, bounds: MLXArray,
        weight: MLXArray, scales: MLXArray, biases: MLXArray, rows: Int,
        negative: Bool = false, force: Bool = false
    ) -> (ids: MLXArray, values: MLXArray, flag: MLXArray, info: MLXArray) {
        let K = rotated.dim(-1)
        let N = weight.dim(0)
        let x = rotated.reshaped([rows, K])
        let selected = select(
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

/// The head's `Amax_g, Bmax_g` for `ExactVerifyHead`, made on first use.
final class ExactVerifyHeadBounds {
    private var cached: MLXArray?

    func get(weight: MLXArray, scales: MLXArray, biases: MLXArray) -> MLXArray {
        if let cached { return cached }
        let groups = scales.dim(1)
        let value = ExactVerifyHead.bounds(
            [weight, scales, biases], grid: (groups * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[2 * groups]], outputDTypes: [.float32])[0]
        eval(value)
        cached = value
        return value
    }
}

/// Draft-only screened argmax over the rows of a packed head for one rotated
/// activation row (`HadamardQuantizedLinear.screenedArgMax`).
private enum DraftScreenQMV {
    /// `Float16ConstantQMV.fast` over the leading `P` inputs of every row:
    /// the same per-lane layout and arithmetic, stopped after `P / 512` steps.
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

    /// One simdgroup per `CH`-row block: the full-width score of the block's
    /// screened row, and that row's index.
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

    /// Fused `argMax(partial.reshaped([blocks, 32]), axis: -1)` + `rescore`
    /// when `CH == 32` and `biases == -scales`: each 32-lane simdgroup reduces
    /// its 32 `partial` values in registers via 5 `simd_shuffle_xor` steps and
    /// immediately rescores the winning row `r` without reading `biases`.
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

    /// `rescoreFast32` for blocks of `CH <= 32` rows (lanes past `CH` hold
    /// no candidate).
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

    /// Single-threadgroup final argmax + gather: finds the winning block across
    /// `y[0 ..< BLOCKS]`, resolves `win_row = row[best_c]`, and writes either
    /// `win_row` (`USE_IDS == 0`) or `ids[win_row]` (`USE_IDS == 1`).
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

/// A rotated prompt-width activation quantized per row and 128-group by
/// `SignedBlockHadamard.quantizedForPrompt`.
struct PromptInt8Activation {
    let codes: MLXArray
    let ascale: MLXArray
    let rsb: MLXArray
    let rows: Int
    let k: Int
}

/// Prompt-width packed matmul on int8 activation codes (27b 5fbfa003, with
/// 31a73441's row-tiled constants and 0e1314b2's signed codes).
///
/// For output (m, n) the kernels compute, per 128-group g in ascending order
/// from `acc = 0`:
///
///     C   = sum_{k in g} q[m, k] * code[n, k]          (integer)
///     t   = s[n, g] * float(C)
///     acc = fma(b[n, g], rsb[m, g], fma(ascale[m, g], t, acc))
///
/// `q` is in [-127, 127] and `code` in [0, 3], so |C| <= 128 * 127 * 3 =
/// 48768 < 2^24: the integer is exact in int32 (the M5 tensor unit) and in
/// FP32 (every partial sum of integer products is an integer below 2^24, so
/// each FP32 multiply-add is exact whatever its order). `float(C)` is then the
/// same number on both, and the epilogue is the same three roundings in the
/// same order, with explicit `fma` so contraction cannot change it. The M2
/// path (`plain`) therefore produces the M5 path's (`nax`) bits;
/// `checkKernels` confirms it on the device before `nax` is used.
enum PromptInt8Matmul {
    enum Route { case nax, plain, off }

    /// Index of (row, group) in the row-tiled constant layout: per 64-row
    /// tile, per group, the 64 rows in the order the tensor kernel's lanes
    /// read them as one float4.
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

    /// The tensor kernel the `nax` route runs: the vector form when it
    /// matches the plain kernel at load, else the record form.
    nonisolated(unsafe) private static var naxVector = false
    /// The `NEG = 1` forms matched their `NEG = 0` twins at load.
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

    /// MLX's own NAX rule (device.cpp `is_nax_available`): GPU generation 17
    /// or later, 18 or later for a `p` part.
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

    /// Runs the kernels on a fixed, input-independent 128 x 128 x 512
    /// problem (2 x 2 tiles, 4 groups) and compares output bits. First the
    /// plain kernel's `NEG = 1` form against its `NEG = 0` form on offsets
    /// that are the negated scales. On a NAX device then the vector tensor
    /// kernel (both forms) against the plain kernel, else the record tensor
    /// kernel against the plain kernel. Sets `naxVector` and `negativeForms`;
    /// false when no tensor kernel matches. A compile error (caught by a
    /// scoped MLX error handler) counts as a mismatch.
    @discardableResult
    private static func checkKernels(nax: Bool) -> Bool {
        checkFailed = false
        let m = 128, n = 128, k = 512, kg = k / 128
        let codes = MLXArray((0 ..< m * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [m, k])
        let weight = MLXArray((0 ..< n * k / 16).map { UInt32(truncatingIfNeeded: UInt64($0 + 1) &* 2_654_435_761) }, [n, k / 16])
        let scalesT = MLXArray((0 ..< kg * n).map { Float16(Float(1 + $0 % 13) / 1024) }, [kg, n])
        let biasesT = MLXArray((0 ..< kg * n).map { Float16(-Float($0 % 7) / 512) }, [kg, n])
        let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
        let ascale = MLXArray((0 ..< m * kg).map { Float(1 + $0 % 11) / 256 }, [m * kg])
        let rsb = MLXArray((0 ..< m * kg).map { Float($0 % 17 - 8) / 64 }, [m * kg])
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
        var matched = false
        naxVector = false
        negativeForms = false
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
                : "bit-identical (\(nax ? (naxVector ? "vector" : "record") : "plain") kernel, offset skip \(negativeForms ? "on" : "off"))"
        }
        return matched
    }

    /// Decided once, at the first packed module's init. A NAX device takes
    /// the tensor kernel only when it compiles and matches the plain kernel
    /// bit for bit; otherwise it keeps the record's float path. Any other
    /// device takes the plain kernel. `DARKBLOOM_BONSAI_INT8_PREFILL=0` keeps
    /// the record's float path everywhere.
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
        return chosen
    }()

    /// Rows at which the route applies: prompt widths only (a verify window
    /// has at most 7 rows), in whole 64-row tiles.
    static func applies(rows: Int) -> Bool { rows >= 128 && rows % 64 == 0 }

    /// `negative`: the module's offsets are its negated scales
    /// (`PromptInt8Constants.negative`).
    static func matmul(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        negative: Bool
    ) -> MLXArray? {
        guard route != .off else { return nil }
        return run(
            route, codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
            weight: weight, scalesT: scalesT, biasesT: biasesT,
            m: activation.rows, k: activation.k, n: weight.dim(0),
            vector: naxVector, negative: negative && negativeForms)
    }
}

/// Verify-width packed matmul (an MTP verify: 2 to 8 rows) on the int8
/// activation codes of `PromptInt8Matmul`, split over K inside a threadgroup
/// (27b 320ec19a's verify split-K on tensor units, built integer-exact).
///
/// A threadgroup owns 32 output columns. Its four simdgroups take every fourth
/// 128-group (simdgroup s takes g = s, s + 4, ...). Per group each computes
/// `C[m, n] = sum_{k in g} q[m, k] * code[n, k]` (integer, |C| < 2^24, exact
/// in int32 and in FP32 as `PromptInt8Matmul` shows) and folds it into its
/// partial with `PromptInt8Matmul`'s epilogue, groups ascending from 0:
///
///     acc = fma(b[n, g], rsb[m, g], fma(ascale[m, g], s[n, g] * float(C), acc))
///
/// The four partials are then summed in simdgroup order, `((p0 + p1) + p2) +
/// p3`. `nax` gets C from one 16 x 32 x 128 int8 tensor op per group (the
/// rows padded to 16 with zero codes, the 2-bit codes expanded to int8 in the
/// simdgroup's threadgroup buffer) and finds each element's (m, n) through
/// `get_multidimensional_index`, so it assumes no fragment layout. `plain`
/// gets C from FP32 FMAs and lane shuffles. Same integers, same FP32 steps in
/// the same order: the same bits, which `checkKernels` checks on the device
/// before `nax` is used.
enum VerifyInt8Matmul {
    enum Route { case nax, plain, off }

    static let rowRange = 2 ... 8
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
        float acc[8];
        #pragma clang loop unroll(full)
        for (int e = 0; e < 8; e++) { acc[e] = 0.0f; }
        const int mr = min(mrow, ROWS - 1);
        const device half* scol = scalesT + n0 + nb;
        const device half* bcol = biasesT + n0 + nb;
        const int j = int(lane & 7u);
        const int c = int(lane >> 3);
        const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;
        threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 8 + j;
        uint32_t wv[8];
        half4 sA, sB, bA = half4(0.0h), bB = half4(0.0h);
        float fa, frb;
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
        }
        for (int g = int(sg); g < Kg; g += 4) {
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) {
            const uint32_t v = wv[t];
            dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);
          }
          const half4 cA = sA, cB = sB;
          const half4 dA = NEG ? -sA : bA, dB = NEG ? -sB : bB;
          const float ca = fa, crb = frb;
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

    /// `naxFastBody` for `layers` sibling projections of one activation in
    /// one dispatch: threadgroups x < N0 / 32 take layer 0, the next N1 / 32
    /// layer 1, and so on, each running the one-layer body unchanged. Inputs
    /// `xq`, `w0`..., `s0`..., `b0`..., `ascale`, `rsb`, `ksz` = [K, rows,
    /// N0, N1, ...]; outputs `out0`....
    static func naxFastSource(layers: Int) -> String {
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
        return pick + naxFastBody
    }

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

    /// `naxFastSource` for one, two and three layers (index `layers - 1`).
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

    /// As `PromptInt8Matmul`: the vector tensor kernel when it matches the
    /// plain kernel at load, and the `NEG = 1` forms when they match theirs.
    /// `naxFast`: the fast kernel matched (it then replaces the vector one);
    /// `naxShared`: its two- and three-layer forms matched too.
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

    /// The fast tensor kernel over `layers` (one to three) sharing `codes`.
    private static func runFast(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)],
        m: Int, k: Int, negative: Bool
    ) -> [MLXArray] {
        let ns = layers.map { $0.weight.dim(0) }
        let dims = MLXArray([Int32(k), Int32(m)] + ns.map { Int32($0) })
        let inputs = [codes] + layers.map { $0.weight } + layers.map { $0.scalesT } + layers.map { $0.biasesT }
            + [ascale, rsb, dims]
        let template = [("ROWS", m), ("NEG", negative ? 1 : 0)] as [(String, any KernelTemplateArg)]
        return naxFastKernels[layers.count - 1](
            inputs, template: template,
            grid: (ns.reduce(0, +) / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: ns.map { [m, $0] }, outputDTypes: ns.map { _ in .float32 })
    }

    nonisolated(unsafe) private static var checkFailed = false
    nonisolated(unsafe) private static var checkNote = "not run"

    /// Runs the kernels on a fixed, input-independent 4 x 64 x 1024 problem
    /// (two threadgroups, two groups per simdgroup) and compares output
    /// bits, as `PromptInt8Matmul.checkKernels`: the plain `NEG = 1` form
    /// against `NEG = 0` on negated-scale offsets, then on a NAX device the
    /// fast tensor kernel (both forms, then its two- and three-layer forms on
    /// column slices), else the vector tensor kernel (both forms), else the
    /// record tensor kernel, against the plain kernel.
    @discardableResult
    private static func checkKernels(nax: Bool) -> Bool {
        checkFailed = false
        let m = 4, n = 64, k = 1024, kg = k / 128
        let codes = MLXArray(
            (0 ..< tileRows * k).map { $0 < m * k ? Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) : 0 },
            [tileRows, k])
        let weight = MLXArray((0 ..< n * k / 16).map { UInt32(truncatingIfNeeded: UInt64($0 + 1) &* 2_654_435_761) }, [n, k / 16])
        let scalesT = MLXArray((0 ..< kg * n).map { Float16(Float(1 + $0 % 13) / 1024) }, [kg, n])
        let biasesT = MLXArray((0 ..< kg * n).map { Float16(-Float($0 % 7) / 512) }, [kg, n])
        let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
        let ascale = MLXArray((0 ..< m * kg).map { Float(1 + $0 % 11) / 256 }, [m * kg])
        let rsb = MLXArray((0 ..< m * kg).map { Float($0 % 17 - 8) / 64 }, [m * kg])
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

    /// Decided once, at the first packed module's init. A NAX device takes
    /// the tensor kernel only when it compiles and matches the plain kernel
    /// bit for bit; otherwise it keeps the record's float path. Any other
    /// device takes the plain kernel. `DARKBLOOM_BONSAI_INT8_VERIFY=0` keeps
    /// the record's float path everywhere; `=nax` runs the check and the
    /// tensor kernel on any device (the M2 emulates the tensor ops).
    static let route: Route = {
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
        FileHandle.standardError.write(
            "int8 verify matmul: \(chosen) (\(architecture); nax check: \(checkNote))\n"
                .data(using: .utf8)!)
        return chosen
    }()

    static func matmul(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        negative: Bool
    ) -> MLXArray? {
        guard route != .off else { return nil }
        if route == .nax && naxFast {
            return runFast(
                codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                layers: [(weight, scalesT, biasesT)], m: activation.rows, k: activation.k,
                negative: negative && negativeForms)[0]
        }
        return run(
            route, codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
            weight: weight, scalesT: scalesT, biasesT: biasesT,
            m: activation.rows, k: activation.k, n: weight.dim(0),
            vector: naxVector, negative: negative && negativeForms)
    }

    /// `matmul` for sibling projections of one activation: two or three
    /// layers in one dispatch when the fast kernel's shared forms matched at
    /// load, else one `matmul` per layer. Either way each output is that
    /// layer's `matmul` output bit for bit (`NEG = 0` reads offsets that are
    /// bit for bit the negated scales wherever `NEG = 1` would apply).
    static func matmul(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)]
    ) -> [MLXArray]? {
        guard route != .off else { return nil }
        if route == .nax, naxFast, naxShared, (2 ... 3).contains(layers.count) {
            return runFast(
                codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
                layers: layers.map { ($0.weight, $0.scalesT, $0.biasesT) },
                m: activation.rows, k: activation.k,
                negative: negativeForms && layers.allSatisfy { $0.negative })
        }
        let outputs = layers.compactMap {
            matmul(activation, weight: $0.weight, scalesT: $0.scalesT, biasesT: $0.biasesT, negative: $0.negative)
        }
        return outputs.count == layers.count ? outputs : nil
    }
}

/// Group-major copies of a packed module's F16 scales and offsets for
/// `PromptInt8Matmul`, made on first use. A plain class, so nothing enters the
/// module parameter tree.
final class PromptInt8Constants {
    private var transposed: (MLXArray, MLXArray)?
    /// Every FP16 offset is bit for bit its negated scale, so the int8
    /// kernels may use `-s` for the offset (`NEG = 1`). Read once at init.
    let negative: Bool

    init(negative: Bool) { self.negative = negative }

    func get(scales: MLXArray, biases: MLXArray) -> (MLXArray, MLXArray) {
        if let transposed { return transposed }
        let pair = (scales.transposed(1, 0).contiguous(), biases.transposed(1, 0).contiguous())
        transposed = pair
        return pair
    }
}

/// Affine packed linear weights with a signed Hadamard input transform.
///
/// Pass weights already folded and packed in MLX format. The initializer does
/// not requantize them and does not accept raw GGUF quantization blocks.
public final class HadamardQuantizedLinear: QuantizedLinear {
    public let transform: SignedBlockHadamard
    public let gdnLayout: HadamardGDNLayout?
    private static let reuseFloat16Constants = ProcessInfo.processInfo.environment[
        "DARKBLOOM_BONSAI_F16_CONSTANT_CACHE"] == "1"

    /// Qualification witness; this selects reuse, never a different precision
    /// or packed-matmul kernel. The process-wide generic cache kill switch also
    /// remains effective. No cached arrays enter the module parameter tree.
    var permitsFloat16ConstantReuse: Bool {
        bits == 2 && groupSize == 128 && scales.dtype == .float16
            && (transform.blockSize == 4096
                || (Self.reuseFloat16Constants && transform.blockSize == 1024))
    }

    public init(
        weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, transform: SignedBlockHadamard,
        gdnLayout: HadamardGDNLayout? = nil
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
        self.promptConstants = PromptInt8Constants(negative: negative)
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
        freeze()
        if readsFloat16ConstantsInPlace && transform.blockSize == 4096 {
            _ = PromptInt8Matmul.route
            _ = VerifyInt8Matmul.route
            if transform.width == 4096 { _ = Self.promptBoundaryEnabled(transform) }
        }
    }

    /// Fixed at init: the packing `Float16ConstantQMV` reads in place.
    let readsFloat16ConstantsInPlace: Bool

    private let promptConstants: PromptInt8Constants
    let exactHeadBounds = ExactVerifyHeadBounds()

    /// The prompt-width integer route (`PromptInt8Matmul`); nil when the
    /// input, the packing or the device does not take it.
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

    /// Form prompt SwiGLU values inside the existing quantizing transform.
    public func callAfterSwiGLU(gate: MLXArray, up: MLXArray) -> MLXArray? {
        guard gdnLayout == nil,
            let output = promptInt8Forward(gate, up: up) ?? verifyInt8Forward(gate, up: up)
        else { return nil }
        return bias.map { output + $0 } ?? output
    }

    /// Share the residual, RMSNorm and rotation for an eligible projection group.
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
        let outputs = packed.map { layer in
            if verify {
                let (scalesT, biasesT) = layer.promptConstants.get(scales: layer.scales, biases: layer.biases!)
                let codes = PromptInt8Activation(
                    codes: activation[0], ascale: activation[1], rsb: activation[2], rows: rows, k: 4096)
                let output = VerifyInt8Matmul.matmul(
                    codes, weight: layer.weight, scalesT: scalesT, biasesT: biasesT,
                    negative: layer.promptConstants.negative)!
                    .reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
                return layer.bias.map { output + $0 } ?? output
            }
            if !quantized { return layer.forwardRotated(activation[0]) }
            let (scalesT, biasesT) = layer.promptConstants.get(scales: layer.scales, biases: layer.biases!)
            let codes = PromptInt8Activation(
                codes: activation[0], ascale: activation[1], rsb: activation[2], rows: rows, k: 4096)
            let output = PromptInt8Matmul.matmul(
                codes, weight: layer.weight, scalesT: scalesT, biasesT: biasesT,
                negative: layer.promptConstants.negative)!
                .reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
        return (activation.last!, outputs)
    }

    /// Reuse prompt codes within this forward when all input transforms match.
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
            let activation = first.transform.quantizedForPrompt(x)
        else { return nil }
        return packed.map { layer in
            let (scalesT, biasesT) = layer.promptConstants.get(
                scales: layer.scales, biases: layer.biases!)
            let output = PromptInt8Matmul.matmul(
                activation, weight: layer.weight, scalesT: scalesT, biasesT: biasesT,
                negative: layer.promptConstants.negative)!
                .reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
    }

    /// Share verify-width integer or float rotations across a group of projections.
    /// A layer's verify-width input, formed in one kernel from the previous
    /// layer's residual and its last output: the sum, the input RMSNorm and
    /// the int8 codes of the rotated normalized rows.
    public struct VerifyBoundary {
        let activation: PromptInt8Activation
        let transform: SignedBlockHadamard
        /// Prompt-width codes (`PromptInt8Matmul` order) rather than verify.
        var prompt = false
        /// `x + delta`, the residual stream entering the layer.
        public let residual: MLXArray
        /// `rmsNorm(x + delta, weight)`.
        public let normalized: MLXArray
    }

    /// `x + delta`, its RMSNorm and the verify codes of the rotated norm in one
    /// kernel, with the rotation of `projections` (all must share it). Nil
    /// when the shapes, the packing or the route do not qualify; the caller
    /// then adds, normalizes and projects separately.
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

    nonisolated(unsafe) private static var promptBoundaryChecked: Bool?

    /// The prompt-width boundary kernel passed its load-time bit check
    /// against the separate add, RMSNorm and quantization (once per process;
    /// `DARKBLOOM_BONSAI_PROMPT_BOUNDARY=0` turns it off).
    static func promptBoundaryEnabled(_ transform: SignedBlockHadamard) -> Bool {
        if let promptBoundaryChecked { return promptBoundaryChecked }
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_PROMPT_BOUNDARY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let enabled: Bool
        let note: String
        if ["0", "false", "no", "off"].contains(value ?? "") {
            (enabled, note) = (false, "disabled")
        } else if PromptInt8Matmul.route == .off {
            (enabled, note) = (false, "int8 prompt route off")
        } else {
            (enabled, note) = transform.promptBoundaryMatchesSeparate()
        }
        promptBoundaryChecked = enabled
        FileHandle.standardError.write(
            "prompt boundary: \(enabled ? "fused" : "separate") (check: \(note))\n"
                .data(using: .utf8)!)
        return enabled
    }

    /// Rows at which `promptBoundary` can apply.
    public static func promptBoundaryApplies(rows: Int) -> Bool {
        PromptInt8Matmul.applies(rows: rows) && PromptInt8Matmul.route != .off
            && promptBoundaryChecked == true
    }

    /// `verifyBoundary` at prompt width: `x + delta`, its RMSNorm and the
    /// prompt codes of the rotated norm in one kernel, each the bits of the
    /// separate add, `rmsNorm` and `quantizedForPrompt`. Nil when the rows,
    /// the packing, the route or the load-time check do not qualify; the
    /// caller then adds, normalizes and projects separately.
    public static func promptBoundary(
        _ x: MLXArray, delta: MLXArray, weight: MLXArray, eps: Float,
        projections: [Linear]
    ) -> VerifyBoundary? {
        guard x.ndim >= 2, x.dim(-1) == 4096, x.dtype == .float32,
            promptBoundaryApplies(rows: x.size / 4096)
        else { return nil }
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count, let first = packed.first,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil && $0.bias == nil
                    && $0.transform.blockSize == 4096 && $0.weight.dim(0) % 64 == 0
                    && $0.transform.isIdentical(to: first.transform)
            }),
            let outputs = first.transform.residualNormPromptNormalized(
                x, residual: delta, weight: weight, eps: eps)
        else { return nil }
        let rows = x.size / 4096
        return VerifyBoundary(
            activation: PromptInt8Activation(
                codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: 4096),
            transform: first.transform, prompt: true, residual: outputs[3], normalized: outputs[4])
    }

    /// `projections` applied to `boundary.normalized` from the boundary's
    /// codes. Nil when a projection's packing or rotation differs.
    public static func boundaryProjections(
        _ boundary: VerifyBoundary, _ projections: [Linear]
    ) -> [MLXArray]? {
        let packed = projections.compactMap { $0 as? HadamardQuantizedLinear }
        guard packed.count == projections.count,
            packed.allSatisfy({
                $0.readsFloat16ConstantsInPlace && $0.gdnLayout == nil && $0.bias == nil
                    && $0.transform.blockSize == 4096
                    && $0.weight.dim(0) % (boundary.prompt ? 64 : 32) == 0
                    && $0.transform.isIdentical(to: boundary.transform)
            })
        else { return nil }
        if boundary.prompt {
            return packed.map { layer in
                let (scalesT, biasesT) = layer.promptConstants.get(
                    scales: layer.scales, biases: layer.biases!)
                return PromptInt8Matmul.matmul(
                    boundary.activation, weight: layer.weight, scalesT: scalesT, biasesT: biasesT,
                    negative: layer.promptConstants.negative)!
                    .reshaped(Array(boundary.normalized.shape.dropLast()) + [layer.weight.dim(0)])
            }
        }
        guard let outputs = VerifyInt8Matmul.matmul(
            boundary.activation, layers: packed.map(\.verifyOperands))
        else { return nil }
        return zip(packed, outputs).map { layer, output in
            output.reshaped(Array(boundary.normalized.shape.dropLast()) + [layer.weight.dim(0)])
        }
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
            let outputs = VerifyInt8Matmul.matmul(activation, layers: packed.map(\.verifyOperands))
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

    /// This layer's operands for `VerifyInt8Matmul.matmul(_:layers:)`.
    private var verifyOperands: (weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool) {
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        return (weight, scalesT, biasesT, promptConstants.negative)
    }

    /// The verify-width integer route (`VerifyInt8Matmul`); nil when the
    /// input, the packing or the device does not take it.
    private func verifyInt8Forward(_ x: MLXArray, up: MLXArray? = nil) -> MLXArray? {
        guard readsFloat16ConstantsInPlace, transform.blockSize == 4096, x.dtype == .float32,
            x.ndim >= 1, VerifyInt8Matmul.rowRange.contains(x.size / x.dim(-1)),
            weight.dim(0) % 32 == 0, transform.width % 512 == 0, VerifyInt8Matmul.route != .off,
            let activation = transform.quantizedForVerify(gdnLayout.map { $0(x) } ?? x, up: up)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = VerifyInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative)
        else { return nil }
        return output.reshaped(Array(x.shape.dropLast()) + [weight.dim(0)])
    }

    /// Draft-only screened argmax of `self(x)` for one input row. Every row is
    /// scored on the leading `width` rotated inputs only; each `chunk`-row
    /// block keeps its best-screened row, those rows are rescored on the full
    /// width, and the best of them wins. Returns the row index (`[1]`,
    /// int32) or mapped `ids[row]` when `ids` is supplied, or nil when the
    /// packing or the shape does not take it. The winner can differ from the
    /// exact argmax, so only a proposal may use it.
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

    /// `self(silu(gate) * rmsNorm(x, normWeight))` for a verify-width GDN
    /// output with 128-wide norm heads, with the gate, the norm and the input
    /// rotation formed in one quantizing kernel. Nil when the route, the
    /// packing or the shapes do not qualify; the caller then runs the
    /// separate operations.
    public func callAfterGatedNorm(
        _ x: MLXArray, gate: MLXArray, normWeight: MLXArray, eps: Float
    ) -> MLXArray? {
        guard gdnLayout == nil, readsFloat16ConstantsInPlace, transform.blockSize == 4096,
            weight.dim(0) % 32 == 0, transform.width % 512 == 0, VerifyInt8Matmul.route != .off,
            let activation = transform.quantizedForVerifyGatedNorm(
                x, gate: gate, weight: normWeight, eps: eps)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = VerifyInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative)
        else { return nil }
        let shaped = output.reshaped(Array(x.shape.dropLast(2)) + [weight.dim(0)])
        return bias.map { shaped + $0 } ?? shaped
    }

    /// `self(attention * sigmoid(gate))` for a verify-width gated attention
    /// output (see `quantizedForVerifyAttentionGate`), returned as `[1, rows,
    /// N]`. Nil when the route, the packing or the shapes do not qualify.
    public func callAfterAttentionGate(
        _ attention: MLXArray, queryProjection: MLXArray
    ) -> MLXArray? {
        guard gdnLayout == nil, readsFloat16ConstantsInPlace, transform.blockSize == 4096,
            weight.dim(0) % 32 == 0, transform.width % 512 == 0, VerifyInt8Matmul.route != .off,
            let activation = transform.quantizedForVerifyAttentionGate(
                attention, queryProjection: queryProjection)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = VerifyInt8Matmul.matmul(
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
    /// Exact top-2 output ids and values per activation row, ordered by value
    /// descending and id ascending on ties, without writing the `rows x N`
    /// outputs: the float values of `Float16ConstantQMV.rows` for 2 to 8
    /// FP32 rows (`ExactVerifyHead`; at 2 to 4 rows also the fused top-2).
    /// Returns `[rows, 2]` int32 ids and float32 values, or nil when this
    /// module or shape has no exact path.
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

    /// `ExactVerifyHead` for this module: int8 screen, float rescore, full
    /// float top-2 for a row the margin does not settle. The first call at
    /// each row count is compared bit for bit with the full float top-2
    /// (and, at 2-4 rows, with `Float16ConstantQMV.topTwo`); on a mismatch
    /// that row count keeps the record's head.
    private func exactHeadTopTwo(
        _ x: MLXArray, rows: Int
    ) -> (ids: MLXArray, values: MLXArray)? {
        guard let range = ExactVerifyHead.rowRange, range.contains(rows),
            ExactVerifyHead.verified[rows] != false,
            gdnLayout == nil, transform.blockSize == 4096, transform.width == 4096,
            weight.dim(0) % 64 == 0, x.dtype == .float32, VerifyInt8Matmul.route != .off,
            let (activation, rotated) = transform.quantizedForVerifyWithRotated(x)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let logits = VerifyInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative)
        else { return nil }
        let bounds = exactHeadBounds.get(weight: weight, scales: scales, biases: biases!)
        let result = ExactVerifyHead.run(
            rotated: rotated, logits: logits, ascale: activation.ascale, bounds: bounds,
            weight: weight, scales: scales, biases: biases!, rows: rows,
            negative: promptConstants.negative)
        let first = ExactVerifyHead.verified[rows] == nil
        guard first || ExactVerifyHead.checkEveryCall else { return (result.ids, result.values) }
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

/// Packed folded embeddings with an inverse transform after lookup.
///
/// `asLinear` applies the forward transform, allowing the same packed weights
/// to serve as a tied output projection without unfolding the full vocabulary.
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
