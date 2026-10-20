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

/// Small kernel inputs (problem dimensions, epsilons, scales, per-round key
/// counts and offsets), made once per distinct value instead of once per call.
/// Each call used to allocate a fresh device buffer and copy the same few bytes
/// into it; the kernels read identical values of the same shape and dtype
/// either way, so the GPU work is unchanged. Load-time constants live in a
/// value table; values that move with the round (key counts, cache offsets)
/// live in a small recent-value list, so the layers of one round share one
/// array. `DARKBLOOM_BONSAI_HOSTCUT=0` restores per-call arrays everywhere.
public enum HostConstants {
    public static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_HOSTCUT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    /// Entries per table before new values fall back to per-call arrays.
    private static let capacity = 4096
    /// Entries in the recent-value list (oldest replaced first).
    private static let recentCapacity = 8
    nonisolated(unsafe) private static var int32Arrays: [[Int32]: MLXArray] = [:]
    nonisolated(unsafe) private static var float32Arrays: [[UInt32]: MLXArray] = [:]
    nonisolated(unsafe) private static var recentValues: [[Int32]] = []
    nonisolated(unsafe) private static var recentArrays: [MLXArray] = []
    nonisolated(unsafe) private static var recentNext = 0
    nonisolated(unsafe) private static var lock = os_unfair_lock()

    /// `MLXArray(values)`, int32 `[values.count]`, for values fixed at load.
    public static func int32(_ values: [Int32]) -> MLXArray {
        guard enabled else { return MLXArray(values) }
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        if let array = int32Arrays[values] { return array }
        let array = MLXArray(values)
        if int32Arrays.count < capacity { int32Arrays[values] = array }
        return array
    }

    /// `MLXArray(values)`, float32 `[values.count]`, keyed by bit pattern.
    public static func float32(_ values: [Float]) -> MLXArray {
        guard enabled else { return MLXArray(values) }
        var key = [UInt32]()
        key.reserveCapacity(values.count)
        for value in values { key.append(value.bitPattern) }
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        if let array = float32Arrays[key] { return array }
        let array = MLXArray(values)
        if float32Arrays.count < capacity { float32Arrays[key] = array }
        return array
    }

    /// `MLXArray(values)`, int32 `[values.count]`, for values that change with
    /// the round but repeat across the layers of one round.
    public static func recentInt32(_ values: [Int32]) -> MLXArray {
        guard enabled else { return MLXArray(values) }
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        for index in 0 ..< recentValues.count where recentValues[index] == values {
            return recentArrays[index]
        }
        let array = MLXArray(values)
        if recentValues.count < recentCapacity {
            recentValues.append(values)
            recentArrays.append(array)
        } else {
            recentValues[recentNext] = values
            recentArrays[recentNext] = array
            recentNext = (recentNext + 1) % recentCapacity
        }
        return array
    }
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

    /// The residual-norm form of `quantized4096Source` without the residual:
    /// the input RMSNorm (MLX `rms_single_row` arithmetic at axis 4096, as
    /// `residualNormSource` forms it) and the codes of the rotated normalized
    /// row, optionally also writing the normalized row. The value entering the
    /// rotation is `(weight * (x * inv)) * sign`, as in the residual form.
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

    /// `quantizedForVerifyAttentionGate` at prompt width: the same kernel
    /// arithmetic, with the head-major attention rows given by
    /// `ATTENTION_ROWS`, the codes `[rows, width]` and the constants in the
    /// row-tiled order `PromptInt8Matmul` reads. Nil when the shapes do not
    /// qualify.
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

    /// `quantizedForPromptAttentionGate` for a single last-query prefill row (`rows == 1`),
    /// writing `ascale` and `rsb` in the 64-row tiled layout `PromptInt8Matmul.matmulLastRow` reads.
    func quantizedForPromptLastRowAttentionGate(
        _ attention: MLXArray, queryProjection: MLXArray
    ) -> PromptInt8Activation? {
        guard blockSize == 4096, width == 4096, attention.dtype == .float32,
            queryProjection.dtype == .float32, attention.ndim == 4, attention.dim(0) == 1,
            attention.dim(1) == 16, attention.dim(2) == 1, attention.dim(3) == 256,
            queryProjection.size == 8192, StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let outputs = Self.attentionGatePromptQuantized4096Kernel(
            [attention, queryProjection.reshaped([1, 8192]), signs],
            template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 0), ("ATTENTION_ROWS", 1)],
            grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 4096], [64 * 32], [64 * 32]],
            outputDTypes: [.int8, .float32, .float32])
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: 1, k: 4096)
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
        let outputs = Self.wideRowVerify(
            .gatedNorm,
            [x.reshaped([rows, width]), gate.reshaped([rows, width]), weight, HostConstants.float32([eps]), signs],
            rows: rows)
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
    }

    /// `quantizedForVerifyGatedNorm` at prompt width: the same kernel and
    /// values, with the codes `[rows, width]` and the constants in the
    /// row-tiled order `PromptInt8Matmul` reads (`quantizedForPrompt`). Nil
    /// when the shape is not a prompt-width FP32 4096-wide input.
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
            [x.reshaped([rows, width]), gate.reshaped([rows, width]), weight, HostConstants.float32([eps]), signs],
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
        return Self.wideRowVerify(
            .residualNormNormed, [x, residual, weight, HostConstants.float32([eps]), signs], rows: rows)
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

    /// `residualNorm` at verify width that also returns the rotated normalized
    /// rows: `[codes, ascale, rsb, x + residual, transform(rmsNorm(x + residual, weight))]`.
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
            .residualNormRot, [x, residual, weight, HostConstants.float32([eps]), signs], rows: rows, headError2: headError2 && rows == 16)
        return (
            PromptInt8Activation(
                codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width),
            outputs[3], outputs[4], headError2 && rows == 16 ? outputs[5] : nil)
    }

    /// `normPrompt` at verify width that also returns the normalized (unrotated)
    /// rows: `[codes, ascale, rsb, rmsNorm(x, weight)]`.
    func normVerifyNormalized(_ x: MLXArray, weight: MLXArray, eps: Float) -> [MLXArray]? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32, x.dim(-1) == 4096,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / width
        guard VerifyInt8Matmul.rowRange.contains(rows) else { return nil }
        return Self.wideRowVerify(.normNormed, [x, weight, HostConstants.float32([eps]), signs], rows: rows)
    }

    /// `quantizedForPrompt(rmsNorm(x + residual, weight))` at prompt width in one kernel:
    /// `[codes, ascale, rsb, x + residual]`, plus the normalized rows when `normalized`.
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
        let inputs = [x, residual, weight, HostConstants.float32([eps]), signs]
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

    /// `quantizedForPrompt(rmsNorm(x, weight))` at prompt width in one kernel:
    /// `[codes, ascale, rsb]`, plus the normalized rows (the bits of
    /// `MLXFast.rmsNorm`) when `normalized`. Nil when the shape is not a
    /// prompt-width FP32 4096-wide input.
    func normPrompt(_ x: MLXArray, weight: MLXArray, eps: Float, normalized: Bool) -> [MLXArray]? {
        guard blockSize == 4096, width == 4096, x.dtype == .float32, x.dim(-1) == 4096,
            weight.dtype == .float32, weight.shape == [4096], x.size > 0,
            (x.size / 4096) % 64 == 0, x.size <= Int(Int32.max),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let rows = x.size / 4096
        Self.wideRowPrepareIfBoot(rows: rows)
        let groups = rows * 32
        let inputs = [x, weight, HostConstants.float32([eps]), signs]
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
        let inputs = [x, residual, weight, HostConstants.float32([eps]), signs]
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

    /// Prompt codes for one consumed row; unused metadata lanes are not read.
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

    /// `forward(x)` quantized for `VerifyInt8Matmul`, with the codes and
    /// constants of `quantizedForPrompt`: codes `[16, width]` (rows past
    /// `rows` zero) and constants `[rows, width / 128]`. Nil when the shape is
    /// not a verify-width FP32 block-4096 input.
    /// With `up`, the input is `silu(x) * up`, formed inside the rotation as
    /// in `quantizedForPrompt`.
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

    /// `quantizedForVerify` that also emits the exact FP32 rotated tensor
    /// `[rows, width]` from the same 3-pass Hadamard butterfly.
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

// WIDEROW-GEN BEGIN
/// The verify-width forms of `quantized4096Source` (`VERIFY_ROWS > 0`), each
/// with `threads` lanes per 4096-wide row (512 or 1024) instead of 256, so
/// every row's threadgroup carries 2x or 4x the threads. Lane `i` holds the
/// `4096 / threads` consecutive elements from `i * (4096 / threads)`. The
/// arithmetic is the record kernel's, operand for operand:
/// - the 12 butterfly stages run in the record's bit order 0 ... 11 (the
///   record's three radix-16 passes are stages 0-3, 4-7 and 8-11), as
///   radix-8 or radix-4 passes over the same threadgroup values, with the
///   1/64 scale on the last store; each stage forms `a + b` and `a - b` from
///   the same two values, so every intermediate keeps its bits;
/// - the row RMS partials: lane `i`, part `p` sums the squares of float4 `f =
///   i + p * threads` in order, and the `simd_sum` of simdgroup `f >> 5`
///   holds float4 `f` in lane `f & 31`, as in the record, so both `simd_sum`
///   steps see the same values in the same lanes;
/// - the 128-wide head partials of the gated norm sit in `head_part[f]` and
///   are reduced over the same 32 lanes;
/// - the per-element value formulas are the record's text;
/// - the 128-group max and the sum of the integer codes are exact in any
///   order (|sum| <= 128 * 127).
/// `VERIFY_ROWS` is read from the last input, so one pipeline serves every
/// verify row count.
enum WideRowSource {
    struct Form: Hashable {
        var swiglu = false, residualNorm = false, normOnly = false, gatedNorm = false
        var attentionGate = false, normalizedOutput = false, rotatedOutput = false, headError2 = false
        /// The norm forms' main-phase loads (x, residual, gate, norm weight,
        /// signs) issued into registers before the norm reduction's barrier
        /// (`DARKBLOOM_BONSAI_LOAD_HOIST=0` keeps the parent text). Same
        /// addresses, same values, same arithmetic text: only the loads move.
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
    /// The verify-width per-row forms: those whose rows carry an RMS
    /// reduction before the rotation (the residual and input norms, the GDN
    /// head norm), and the elementwise ones (the plain rotation and its
    /// rotated-output twin, the SwiGLU down-projection input at three blocks
    /// per row, the attention output gate). Each one runs `WideRowSource` at
    /// the widest lane count of `wideRowLaneCandidates` whose load check
    /// passed, and the record kernel otherwise.
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

        /// The elementwise forms, switched by `MLXFAST_WIDE_ROWS_ELEMENTWISE`.
        var elementwise: Bool {
            switch self {
            case .plain, .plainRot, .swiglu, .attentionGate: return true
            default: return false
            }
        }

        /// 4096-wide blocks per row in the load check (the SwiGLU input is
        /// the 12288-wide down-projection input).
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

    /// `MLXFAST_WIDE_ROWS=0` keeps every record kernel.
    static let wideRowEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_WIDE_ROWS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    /// `MLXFAST_WIDE_ROWS_ELEMENTWISE=0` keeps the record kernel for the
    /// elementwise forms only (`WideRowKind.elementwise`).
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

    /// Lane counts tried in order. The norm forms try 1024 lanes (four
    /// elements per lane, half the per-thread chain of 512) before 512; the
    /// elementwise forms keep 512, where a wider row showed no gain.
    /// `DARKBLOOM_BONSAI_WIDE_ROWS_1024=0` or `MLXFAST_WIDE_ROWS_THREADS=512`
    /// keeps 512 for every form; `MLXFAST_WIDE_ROWS_THREADS=1024` tries 1024
    /// for every form.
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

    /// `DARKBLOOM_BONSAI_LOAD_HOIST=0` keeps the parent text of the norm forms
    /// (`WideRowSource.Form.hoist`).
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

    /// The lane count each form serves (absent: the record kernel).
    nonisolated(unsafe) private static var wideRowVerdict: [WideRowKind: Int] = [:]
    nonisolated(unsafe) private static var wideRowResolvedLanes: [Int] = [Int](repeating: 0, count: 9)
    nonisolated(unsafe) private static var wideRowResolvedKernels: [MLXFast.MLXFastKernel?] = [MLXFast.MLXFastKernel?](repeating: nil, count: 9)
    nonisolated(unsafe) private static var wideRowResolvedHeadKernels: [MLXFast.MLXFastKernel?] = [MLXFast.MLXFastKernel?](repeating: nil, count: 9)
    nonisolated(unsafe) private static var wideRowChecked = false
    nonisolated(unsafe) private static var wideRowReady = false
    private static let wideRowLock = NSLock()

    /// Runs the load check once, from the first prompt-width call of at
    /// least 1024 rows: the resident's boot warm, which runs before the
    /// socket serves and which no timed phase reaches (the timed prompts are
    /// 512 tokens). Until then, and without it, every form keeps the record
    /// kernel.
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

    /// The record kernel's outputs bit for bit (`WideRowSource`): codes,
    /// constants and every FP32 side output, on synthetic rows (one all-zero
    /// row, one at 30x scale) at 3, 11 and 16 verify rows, at each form's
    /// real shapes (the SwiGLU input 12288 wide, the attention output in its
    /// `[1, 16, rows, 256]` head-major layout beside the `[rows, 8192]`
    /// query projection).
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

    /// One verify-width launch of `kind` over `inputs` (`x` first, `[rows,
    /// 4096 * blocks]` in any leading shape): `[codes [16, 4096 * blocks],
    /// ascale, rsb, side outputs shaped as x]` (or as `sideShape`).
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
                inputs + [HostConstants.int32([Int32(rows)])], template: [("BLOCKS_PER_ROW", blocks)],
                grid: (lanes, tile * blocks, 1), threadGroup: (lanes, 1, 1),
                outputShapes: shapes, outputDTypes: dtypes)
        }
        return wideRowRecordKernel(kind, headError2: headError2)(
            inputs, template: [("BLOCKS_PER_ROW", blocks), ("VERIFY_ROWS", rows)],
            grid: (256, tile * blocks, 1), threadGroup: (256, 1, 1),
            outputShapes: shapes, outputDTypes: dtypes)
    }

    /// `wideRowDispatch` with the wide kernel when its load check passed at
    /// this many blocks per row.
    static func wideRowVerify(
        _ kind: WideRowKind, _ inputs: [MLXArray], rows: Int, blocks: Int = 1,
        sideShape: [Int]? = nil, headError2: Bool = false
    ) -> [MLXArray] {
        wideRowDispatch(
            kind, inputs, rows: rows, blocks: blocks, sideShape: sideShape,
            lanes: blocks == kind.checkBlocks ? wideRowLanes(kind) : 0, headError2: headError2)
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

    /// The global operations keep their order; local arithmetic shares the group loop.
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

    /// v and q are still in registers. The scale/value restrictions make
    /// nonzero cancellation normal; tiny squares and all unsafe cases keep
    /// the old bound. One outward ULP covers the residual FMA rounding.
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

    /// One threadgroup per 128-group g: `bounds[g] = Amax_g`, `bounds[G + g]
    /// = Bmax_g`.
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

    /// One threadgroup of `threads` per row: int8 top-2, E, the candidate
    /// set, its float values (`rowsAccumulate` at one row and one output
    /// row, so each value is the one `topTwoPartial` computes for that id)
    /// and their top-2. `info[row] = (E, |S|)`.
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


    /// Four values per 32-column tile. A fifth value at or above tau makes
    /// the row use full-float fallback. info.count == -1 marks this overflow;
    /// otherwise it is the exact candidate count for a finite row.
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

    /// `topTwoPartial` for flagged rows, with shared weight loads unchanged.
    /// The uniform row mask skips other activation loads and arithmetic;
    /// `finalize` reads partial pairs only for flagged rows.
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

    /// `select` spread over `blocks` threadgroups per row. The top-2 order is
    /// a strict total order and the blocks partition the ids, so the merged
    /// top-2 of the block top-2s is the row's top-2; each candidate's float
    /// value is one simdgroup's `rowsAccumulate` at one row and one output
    /// row, as in `select`. So `selectScan`, `selectRescore` and
    /// `selectMerge` write `select`'s outputs: the same `flag` and `info`,
    /// and the same pair on every row that is not flagged (a flagged row's
    /// pair is not read).
    static let blocks = 64
    static let blockThreads = 256

    /// Per (row, block): the int8 top-2 of the block's logits and whether
    /// one is not finite. Each thread visits its ids in ascending order.
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

    /// Per (row, block): merges the row's block top-2s into its int8 top-2,
    /// forms E and tau with `select`'s arithmetic, collects the block's ids
    /// with `L8 >= tau`, rescores them with `select`'s float text and writes
    /// their float top-2, the block's candidate count, a non-finite flag and
    /// the row's bad flag. A block over `CAP` candidates rescores nothing:
    /// its row is flagged.
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

    /// One simdgroup per row: `select`'s pair, `flag` and `info` from the
    /// block results.
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

    /// `select`'s outputs through the three block kernels.
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


    /// 31,040 retained values at the target geometry: 16 fixed blocks give
    /// 1,940 values per block. The old three stages and float rescore remain.
    static let compactBlocks = 16
    static let compactParallelScanSource = selectScanSource
        .replacingOccurrences(of: "const float4 v4 = lq[q];", with: "const float4 v4 = lq[q]; bad = bad || compact_bad[row * uint(NV / 4) + first / 4u + uint(q)] != 0;")
        .replacingOccurrences(of: "if (local.count == 2 && !(v > local.second_value)) continue;", with: "if (local.count == 2 && !hadamard_top2_better(v, uint(compact_ids[row * uint(NV) + first + uint(q) * 4u + uint(i)]), local.second_value, local.second_id)) continue;")
        .replacingOccurrences(of: "hadamard_top2_insert(local, v, first + uint(q) * 4u + uint(i));", with: "hadamard_top2_insert(local, v, uint(compact_ids[row * uint(NV) + first + uint(q) * 4u + uint(i)]));")

    static let compactParallelRescoreSource = selectRescoreSource
        .replacingOccurrences(of: "e *= 1.001f;", with: "e *= 1.001f; if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);")
        .replacingOccurrences(of: "threadgroup atomic_int cand_count;", with: "threadgroup atomic_int cand_count; threadgroup atomic_int omitted_overflow;")
        .replacingOccurrences(of: "if (lane == 0) atomic_store_explicit(&cand_count, 0, memory_order_relaxed);", with: "if (lane == 0) { atomic_store_explicit(&cand_count, 0, memory_order_relaxed); atomic_store_explicit(&omitted_overflow, 0, memory_order_relaxed); }")
        .replacingOccurrences(of: "const float4 v4 = lq[q];", with: "const float4 v4 = lq[q]; if (omitted[row * uint(NV / 4) + first / 4u + uint(q)] >= tau) atomic_store_explicit(&omitted_overflow, 1, memory_order_relaxed);")
        .replacingOccurrences(of: "if (slot < CAP) cand_ids[slot] = int(first + uint(q) * 4u + uint(i));", with: "if (slot < CAP) cand_ids[slot] = compact_ids[row * uint(NV) + first + uint(q) * 4u + uint(i)];")
        .replacingOccurrences(of: "blk_meta[p * 4 + 3] = shared_bad;", with: "blk_meta[p * 4 + 3] = shared_bad | (atomic_load_explicit(&omitted_overflow, memory_order_relaxed) != 0 ? 2 : 0);")

    /// Two distinct exact leader scores lower-bound the true second score.
    /// Only simdgroup0 computes them; all candidate dots keep the same text.
    static let compactParallelAnchorRescoreSource: String = {
        let anchorDots = """
              const uint anchor_id0 = simd_shuffle(local.first_id, ushort(0));
              const uint anchor_id1 = simd_shuffle(local.second_id, ushort(0));
              bool anchor_ok = pbad == 0 && simd_shuffle(local.count, ushort(0)) >= 2u
                  && anchor_id0 != anchor_id1 && anchor_id0 < uint(w_shape[0]) && anchor_id1 < uint(w_shape[0])
                  && isfinite(simd_shuffle(local.first_value, ushort(0)))
                  && isfinite(simd_shuffle(local.second_value, ushort(0)));
              float anchor_floor = INFINITY;
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
                  anchor_ok = anchor_ok && isfinite(anchor_value);
                  anchor_floor = min(anchor_floor, anchor_value);
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
            .replacingOccurrences(of: "pbad = simd_max(pbad);", with: "pbad = simd_max(pbad);\n" + anchorDots)
            .replacingOccurrences(of: originalTau, with: anchoredTau)
    }()

    static let compactParallelPartitionRescoreSource = compactParallelAnchorRescoreSource
        .replacingOccurrences(of: "if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);", with: "float local_e = e; if (USE_L2) e = head_l2_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e);")
        .replacingOccurrences(of: "const float anchor_t = anchor_floor - e;", with: "const float anchor_t = anchor_floor - local_e;")

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
        weight: MLXArray, scales: MLXArray, biases: MLXArray, rows: Int, negative: Bool, error2: MLXArray? = nil, anchors: Bool = false, partitionL2: Bool = false
    ) -> [MLXArray] {
        let values = summary.values.reshaped([rows, -1])
        let ids = summary.ids.reshaped([rows, -1])
        let n = values.dim(1)
        let b = compactBlocks
        let parts = compactParallelScan(
            [values, ids, summary.bad], template: [("THREADS", blockThreads), ("BLOCKS", b), ("NV", n)],
            grid: (b * blockThreads, rows, 1), threadGroup: (blockThreads, 1, 1),
            outputShapes: [[rows, b, 2], [rows, b, 2], [rows, b]], outputDTypes: [.int32, .float32, .int32])
        let local = partitionL2 && anchors && error2 != nil && rows == 16 && b == 16
            && x.dim(1) == 4096 && scales.dim(1) == 32 && weight.dim(0) % 512 == 0
            && n == weight.dim(0) / 8 && bounds.size == 19 * 32
        let kernel = local ? compactParallelPartitionRescore
            : (anchors ? compactParallelAnchorRescore : compactParallelRescore)
        let blk = kernel(
            [values, ascale, bounds, x, weight, scales, biases, parts[0], parts[1], parts[2], ids, summary.omitted, error2 ?? ascale],
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
        rows: Int, negative: Bool, error2: MLXArray? = nil, anchors: Bool = false, partitionL2: Bool = false
    ) -> (ids: MLXArray, values: MLXArray) {
        let x = rotated.reshaped([rows, rotated.dim(-1)])
        let selected = compactParallelSelect(
            summary: summary, ascale: ascale, bounds: bounds, x: x,
            weight: weight, scales: scales, biases: biases, rows: rows, negative: negative, error2: error2, anchors: anchors, partitionL2: partitionL2)
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

    /// `DARKBLOOM_BONSAI_XHEAD_PARALLEL=0` keeps the one-threadgroup-per-row
    /// `select`.
    static let parallelOn: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_XHEAD_PARALLEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Row counts whose first parallel select matched `select` (true) or
    /// not (false: that row count keeps `select`).
    nonisolated(unsafe) static var parallelVerified: [Int: Bool] = [:]

    static func parallelApplies(rows: Int, N: Int) -> Bool {
        parallelOn && parallelVerified[rows] != false && N % (4 * blocks) == 0
    }

    /// `DARKBLOOM_BONSAI_XHEAD=0` keeps the record's head (float top-2 at 2-4
    /// rows, int8 logits at 5-8); `=5` takes this path at 5-8 rows only.
    static let rowRange: ClosedRange<Int>? = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_XHEAD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["0", "false", "no", "off"].contains(value ?? "") { return nil }
        let high = 16
        if let value, let low = Int(value), (2 ... 8).contains(low) { return low ... high }
        return 2 ... high
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

/// The head's `Amax_g, Bmax_g` for `ExactVerifyHead`, made on first use.
final class ExactVerifyHeadBounds {
    var compactVerified: Bool?
    var compactParallelVerified: Bool?
    var residualL2Verified: Bool?
    var exactAnchorsVerified: Bool?
    var partitionL2Verified: Bool?
    private var cached: MLXArray?

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
        const int m0 = ty * 64;
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

    /// `naxSgBody` for `layers` sibling projections of one activation in one
    /// dispatch: threadgroups x < N0 / 64 take layer 0, the next N1 / 64
    /// layer 1, and so on, each running the one-layer body unchanged. Inputs
    /// `xq`, `w0`..., `s0`..., `b0`..., `ascale`, `rsb`, `ksz` = [K, rows,
    /// N0, N1, ...]; outputs `out0`....
    static func naxSgSource(layers: Int, body: String = naxSgBody) -> String {
        var pick = """
            const int K = ksz[0]; const int M = ksz[1];
            int tx = int(threadgroup_position_in_grid.x);
            int ty = int(threadgroup_position_in_grid.y);
            if (MFIRST) {
              const uint lin = threadgroup_position_in_grid.y * threadgroups_per_grid.x + threadgroup_position_in_grid.x;
              tx = int(lin / threadgroups_per_grid.y);
              ty = int(lin % threadgroups_per_grid.y);
            }
            int bx = tx;
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

    /// `naxSgSource` for one, two and three layers (index `layers - 1`).
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
        const int ms = ty * 32;
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

    /// `naxRegBody` for one, two and three layers (index `layers - 1`).
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

    /// `body` with each swap applied; a fragment that is not in the text
    /// exactly once is a programming error.
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
            (of: "int bx = tx;",
             with: "int bx = 2 * tx + int(simdgroup_index_in_threadgroup >> 1);"),
            (of: "const int Kg = K / 128;", with: "if (bx >= N / 64) { return; }\nconst int Kg = K / 128;"),
            (of: "const int ns = bx * 64 + 32 * int(sg);", with: "const int ns = bx * 64 + 32 * int(sg & 1);"),
        ])
    }

    /// `naxRegCols128Source` for one, two and three layers (index `layers - 1`).
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

    /// The per-simdgroup kernel forms `runSg` launches: `naxSgBody`, and its
    /// register-weight forms `naxRegBody` and `naxRegCols128Source`. Each names its kernels (index `layers - 1`), its
    /// threads per threadgroup and its rows and columns per threadgroup.
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

    /// Source drift disables the shared route before any copied weights are used.
    /// The native register sources and their qualification remain unchanged.
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
        if allMatched, traversalMFirst {
            for rows in [registerCheckRows, 192] where traversalMFirst {
                let (same, text) = matchesStaged(streamedForm, streamed: true, rows: rows, traversal: true)
                if !same {
                    traversalMFirst = false
                    FileHandle.standardError.write(Data("int8 prompt M-first traversal: off for the streamed form (\(text))\n".utf8))
                }
            }
        }
        FileHandle.standardError.write(Data("prompt C16 selected form: \(streamedForm); canonical form: \(sgForm)\n".utf8))
        // A failed prefill qualification keeps the canonical layout on both
        // paths, rather than enabling copies with an unqualified consumer.
        if !allMatched { VerifyInt8Matmul.streamReady = false }
        FileHandle.standardError.write(Data("prompt streamed loadcheck: \(streamPromptReady ? "passed" : "FAILED")\n".utf8))
        if streamPromptReady { PromptSplitZ.qualifyStreamed() }
    }

    enum SgForm {
        case staged, register, registerCols128

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

    /// `naxSg`: the per-simdgroup kernel matched at load (it then replaces
    /// the vector one); `naxSgShared`: its two- and three-layer forms did too;
    /// `sgForm`: the form of it the prompt route runs, the last one that
    /// matched it on the prompt shapes.
    nonisolated(unsafe) private static var naxSg = false
    nonisolated(unsafe) private static var naxSgShared = false
    nonisolated(unsafe) private static var sgForm = SgForm.staged

    /// Launch order of `runSg`'s threadgroups: M-first (consecutive threadgroups
    /// take the same weight columns on successive 32 or 64 row tiles, so the
    /// packed weights stream through the caches once) instead of N-first. The
    /// tile each threadgroup computes is unchanged; only which threadgroup
    /// position computes it. Set by `checkTraversal` and `prepareStreaming`.
    nonisolated(unsafe) private static var traversalMFirst = false

    /// The per-simdgroup tensor kernel over `layers` (one to three) sharing
    /// one activation, in `form`; every N a multiple of 64, `m` of 64. Every
    /// form has the same outputs.
    private static func runSg(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)],
        m: Int, k: Int, negative: Bool, form: SgForm, streamed: Bool = false, mFirst: Bool? = nil
    ) -> [MLXArray] {
        let limit = Int(Int32.max)
        let dims: MLXArray
        let inputs: [MLXArray]
        let outputShapes: [[Int]]
        let outputDTypes: [DType]
        let totalN: Int
        let allNValid: Bool
        switch layers.count {
        case 1:
            let l0 = layers[0]
            let n0 = l0.weight.dim(0)
            totalN = n0
            allNValid = n0 > 0 && n0 % 64 == 0 && n0 <= limit / (k / 16) && m <= limit / n0
            dims = HostConstants.int32([Int32(k), Int32(m), Int32(n0)])
            inputs = [codes, l0.weight, l0.scalesT, l0.biasesT, ascale, rsb, dims]
            outputShapes = [[m, n0]]
            outputDTypes = [.float32]
        case 2:
            let l0 = layers[0], l1 = layers[1]
            let n0 = l0.weight.dim(0), n1 = l1.weight.dim(0)
            totalN = n0 + n1
            let maxN = limit / (k / 16)
            allNValid = n0 > 0 && n0 % 64 == 0 && n0 <= maxN && m <= limit / n0
                && n1 > 0 && n1 % 64 == 0 && n1 <= maxN && m <= limit / n1
            dims = HostConstants.int32([Int32(k), Int32(m), Int32(n0), Int32(n1)])
            inputs = [codes, l0.weight, l1.weight, l0.scalesT, l1.scalesT, l0.biasesT, l1.biasesT, ascale, rsb, dims]
            outputShapes = [[m, n0], [m, n1]]
            outputDTypes = [.float32, .float32]
        case 3:
            let l0 = layers[0], l1 = layers[1], l2 = layers[2]
            let n0 = l0.weight.dim(0), n1 = l1.weight.dim(0), n2 = l2.weight.dim(0)
            totalN = n0 + n1 + n2
            let maxN = limit / (k / 16)
            allNValid = n0 > 0 && n0 % 64 == 0 && n0 <= maxN && m <= limit / n0
                && n1 > 0 && n1 % 64 == 0 && n1 <= maxN && m <= limit / n1
                && n2 > 0 && n2 % 64 == 0 && n2 <= maxN && m <= limit / n2
            dims = HostConstants.int32([Int32(k), Int32(m), Int32(n0), Int32(n1), Int32(n2)])
            inputs = [
                codes, l0.weight, l1.weight, l2.weight,
                l0.scalesT, l1.scalesT, l2.scalesT,
                l0.biasesT, l1.biasesT, l2.biasesT,
                ascale, rsb, dims,
            ]
            outputShapes = [[m, n0], [m, n1], [m, n2]]
            outputDTypes = [.float32, .float32, .float32]
        default:
            let ns = layers.map { $0.weight.dim(0) }
            totalN = ns.reduce(0, +)
            allNValid = ns.allSatisfy { n in
                n > 0 && n % 64 == 0 && n <= limit / (k / 16) && m <= limit / n
            }
            dims = HostConstants.int32([Int32(k), Int32(m)] + ns.map { Int32($0) })
            inputs = [codes] + layers.map { $0.weight } + layers.map { $0.scalesT } + layers.map { $0.biasesT }
                + [ascale, rsb, dims]
            outputShapes = ns.map { [m, $0] }
            outputDTypes = ns.map { _ in .float32 }
        }
        let index32 = m > 0 && m % 64 == 0 && k >= 128 && k % 128 == 0
            && m <= limit / k && allNValid
        let mFirstOrder = mFirst ?? traversalMFirst
        let template: [(String, any KernelTemplateArg)] =
            (form == .staged || form == .registerCols128)
            ? [("NEG", negative ? 1 : 0), ("INDEX32", index32), ("MFIRST", mFirstOrder)]
            : [("NEG", negative ? 1 : 0), ("MFIRST", mFirstOrder)]
        let launch = streamed ? form.streamedLaunch : form.launch
        let kernels = launch.kernels
        return kernels[layers.count - 1](
            inputs, template: template,
            grid: ((totalN + launch.columns - 1) / launch.columns * launch.threads, m / launch.rows, 1),
            threadGroup: (launch.threads, 1, 1),
            outputShapes: outputShapes, outputDTypes: outputDTypes)
    }

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
        let dims = HostConstants.int32([Int32(k), Int32(m), Int32(n)])
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

    /// The prompt call shapes of the Qwen 3.5 9B tower at the 512-row seed:
    /// q/k/v, GDN qkv/z, gate/up, the last layer's k/v, o/out and down.
    static let registerCheckShapes: [(k: Int, ns: [Int])] = [
        (4096, [8192, 1024, 1024]), (4096, [8192, 4096]), (4096, [12288, 12288]),
        (4096, [1024, 1024]), (4096, [4096]), (12288, [4096]),
    ]
    static let registerCheckRows = 512

    nonisolated(unsafe) private static var registerFailed = false

    /// `form` against `naxSgBody` (the kernel the prompt route runs) at
    /// `registerCheckRows` rows on every `registerCheckShapes` call, on fixed
    /// synthetic operands, `NEG = 0` and, where the offset skip is on,
    /// `NEG = 1`; every output bit compared. A compile or run error counts as
    /// a mismatch. Returns whether every bit matched, and the detail for the
    /// stderr line.
    private static func matchesStaged(
        _ form: SgForm, streamed: Bool = false, rows: Int = registerCheckRows, traversal: Bool = false
    ) -> (same: Bool, detail: String) {
        let m = rows
        var same = true
        var compared = 0
        registerFailed = false
        withErrorHandler({ _ in PromptInt8Matmul.registerFailed = true }) {
            for (index, shape) in registerCheckShapes.enumerated() where same {
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
                    func outputs(_ which: SgForm, candidate: Bool) -> [MLXArray] {
                        let useCopies = streamed && candidate
                        return runSg(
                            codes: codes, ascale: ascale, rsb: rsb, layers: useCopies ? packed : layers,
                            m: m, k: k, negative: negative, form: which, streamed: useCopies,
                            mFirst: traversal && candidate
                        ).map { $0.view(dtype: .uint32) }
                    }
                    let checks = zip(outputs(.staged, candidate: false), outputs(form, candidate: true))
                        .map { MLX.all($0 .== $1) }
                    eval(checks)
                    same = !registerFailed && checks.allSatisfy { $0.item(Bool.self) }
                    compared += m * shape.ns.reduce(0, +)
                }
            }
        }
        same = same && !registerFailed
        registerFailed = false
        let shapes = registerCheckShapes.map { shape in
            "K \(shape.k) N \(shape.ns.map { String($0) }.joined(separator: "+"))"
        }.joined(separator: ", ")
        return (same, same
            ? "bitwise equal to the per-simdgroup kernel at rows \(m): \(shapes); NEG 0\(negativeForms ? " and 1" : ""); \(compared) values"
            : "a check differed or failed to compile")
    }

    /// `true` when the environment switches `variable` off.
    private static func switchedOff(_ variable: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[variable]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["0", "false", "no", "off"].contains(value ?? "")
    }

    /// `naxRegBody` against `naxSgBody`: `sgForm` becomes `.register` when
    /// every bit matched. Returns the verdict for the stderr line.
    private static func checkRegister() -> String {
        sgForm = .staged
        if switchedOff("DARKBLOOM_BONSAI_PROMPT_REG") { return "off (DARKBLOOM_BONSAI_PROMPT_REG=0)" }
        guard naxSg, naxSgShared else { return "off (the per-simdgroup shared kernel is not in use)" }
        let (same, detail) = matchesStaged(.register)
        if same { sgForm = .register }
        return same ? "on (\(detail))" : "off (\(detail); the per-simdgroup kernel is kept)"
    }

    /// `naxRegCols128Source` against `naxSgBody`, once a register-weight form
    /// matched: `sgForm` becomes `.registerCols128` when every bit matched, else
    /// the form selected before it stays. Returns the verdict for the stderr
    /// line.
    private static func checkRegisterCols128() -> String {
        if switchedOff("DARKBLOOM_BONSAI_PROMPT_REG_COLS128") { return "off (DARKBLOOM_BONSAI_PROMPT_REG_COLS128=0)" }
        guard sgForm != .staged else { return "off (the register-weight kernel is not in use)" }
        let (same, detail) = matchesStaged(.registerCols128)
        if same { sgForm = .registerCols128 }
        return same ? "on (\(detail))" : "off (\(detail); the \(sgForm) form is kept)"
    }

    /// M-first launch order against the record's N-first order, bit for bit, on
    /// every `registerCheckShapes` call at the 512-row prompt and at 192 rows (a
    /// tile count that is no power of two), in the form the prompt route runs.
    /// `DARKBLOOM_BONSAI_PREFILL_TRAVERSAL=0` keeps the record order.
    private static func checkTraversal() -> String {
        traversalMFirst = false
        if switchedOff("DARKBLOOM_BONSAI_PREFILL_TRAVERSAL") { return "off (DARKBLOOM_BONSAI_PREFILL_TRAVERSAL=0)" }
        guard naxSg else { return "off (the per-simdgroup kernel is not in use)" }
        var detail = ""
        for rows in [registerCheckRows, 192] {
            let (same, text) = matchesStaged(sgForm, rows: rows, traversal: true)
            guard same else { return "off (\(text); record order kept)" }
            detail += (detail.isEmpty ? "" : "; ") + text
        }
        traversalMFirst = true
        return "on (\(detail))"
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
    /// that are the negated scales. On a NAX device then the per-simdgroup
    /// tensor kernel (both forms, and its two- and three-layer forms on
    /// column slices) against the plain kernel; failing that the vector
    /// tensor kernel (both forms), else the record tensor kernel. Sets
    /// `naxSg`, `naxSgShared`, `naxVector` and `negativeForms`; false when no
    /// tensor kernel matches. A compile error (caught by a
    /// scoped MLX error handler) counts as a mismatch.
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
        if chosen == .nax {
            FileHandle.standardError.write(
                "int8 prompt register-weight kernel: \(checkRegister())\n".data(using: .utf8)!)
            FileHandle.standardError.write(
                "int8 prompt register-weight 128-column kernel: \(checkRegisterCols128())\n".data(using: .utf8)!)
            FileHandle.standardError.write(
                "int8 prompt M-first traversal: \(checkTraversal())\n".data(using: .utf8)!)
        }
        return chosen
    }()

    /// Rows at which the route applies: prompt widths only (a verify window
    /// has at most 7 rows), in whole 64-row tiles.
    static func applies(rows: Int) -> Bool { rows >= 128 && rows % 64 == 0 }

    /// One logical row with the plain Prompt kernel's ordered group fold.
    static func matmulLastRow(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray,
        biasesT: MLXArray, negative: Bool
    ) -> MLXArray? {
        guard route != .off, activation.rows == 1 || applies(rows: activation.rows),
            weight.dim(0) % 32 == 0 else { return nil }
        let n = weight.dim(0)
        let dims = HostConstants.int32([Int32(activation.k), 1, Int32(n), Int32(activation.rows - 1)])
        return lastRowKernel(
            [activation.codes, weight, scalesT, biasesT, activation.ascale, activation.rsb, dims],
            template: [("NEG", negative && negativeForms ? 1 : 0)],
            grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[1, n]], outputDTypes: [.float32])[0]
    }

    /// `negative`: the module's offsets are its negated scales
    /// (`PromptInt8Constants.negative`).
    static func matmul(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        negative: Bool, streamedWeight: MLXArray? = nil
    ) -> MLXArray? {
        guard route != .off else { return nil }
        let copied: MLXArray?
        if let streamedWeight, streamPromptReady,
            activation.rows >= 64, activation.rows % 64 == 0,
            activation.k >= 128, activation.k % 128 == 0,
            weight.dim(0) > 0, weight.dim(0) % 64 == 0
        {
            copied = streamedWeight
        } else {
            copied = promptStream(weight, m: activation.rows, k: activation.k)
        }
        if let copied {
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

    /// `matmul` for sibling projections of one activation (two or three):
    /// one dispatch when the per-simdgroup kernel's shared forms matched at
    /// load, else one call per layer. Each output is that layer's `matmul`
    /// output bit for bit (`NEG = 0` reads offsets that are bit for bit the
    /// negated scales wherever `NEG = 1` would apply).
    static func matmul(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        streamedWeights: [MLXArray]? = nil
    ) -> [MLXArray]? {
        guard route != .off else { return nil }
        // Siblings share one streamed dispatch only where the canonical shared
        // forms matched at load; otherwise each layer runs on its own, as below.
        if (2 ... 3).contains(layers.count), naxSgShared {
            let copies: [MLXArray]
            if let streamedWeights, streamedWeights.count == layers.count,
                streamPromptReady, activation.rows >= 64, activation.rows % 64 == 0,
                activation.k >= 128, activation.k % 128 == 0,
                layers.allSatisfy({ $0.weight.dim(0) > 0 && $0.weight.dim(0) % 64 == 0 })
            {
                copies = streamedWeights
            } else {
                copies = layers.compactMap { promptStream($0.weight, m: activation.rows, k: activation.k) }
            }
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

extension PromptInt8Matmul {
    /// `matmul(_:layers:)` of a pair in two launches, each in the kernel
    /// form, word layout and offset form the pair's own launch takes, one
    /// layer each: `first` now, and `second(activation)`. Nil when the pair
    /// would not run, or would read streamed words `streamedQualified` does
    /// not admit. The per-layer fallback runs only while no streamed words
    /// serve, since its one-layer calls may find registered copies.
    static func splitPair(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        streamedWeights: [MLXArray]?, streamedQualified: Bool
    ) -> (first: MLXArray, second: (PromptInt8Activation) -> MLXArray?)? {
        guard route != .off, layers.count == 2 else { return nil }
        let negative = negativeForms && layers.allSatisfy { $0.negative }
        if naxSgShared {
            let copies: [MLXArray]
            if let streamedWeights, streamedWeights.count == 2,
                streamPromptReady, activation.rows >= 64, activation.rows % 64 == 0,
                activation.k >= 128, activation.k % 128 == 0,
                layers.allSatisfy({ $0.weight.dim(0) > 0 && $0.weight.dim(0) % 64 == 0 })
            {
                copies = streamedWeights
            } else {
                copies = layers.compactMap { promptStream($0.weight, m: activation.rows, k: activation.k) }
            }
            if copies.count == 2 {
                guard streamedQualified else { return nil }
                func launch(_ a: PromptInt8Activation, _ index: Int) -> MLXArray {
                    let layer = layers[index]
                    let output = runSg(codes: a.codes, ascale: a.ascale, rsb: a.rsb,
                        layers: [(copies[index], layer.scalesT, layer.biasesT)], m: a.rows, k: a.k,
                        negative: negative, form: streamedForm, streamed: true)
                    return streamOutputs(output) {
                        runSg(codes: a.codes, ascale: a.ascale, rsb: a.rsb,
                            layers: [(layer.weight, layer.scalesT, layer.biasesT)], m: a.rows, k: a.k,
                            negative: negative, form: sgForm)
                    }[0]
                }
                return (launch(activation, 0), { launch($0, 1) })
            }
        }
        if route == .nax, naxSg, naxSgShared {
            func launch(_ a: PromptInt8Activation, _ index: Int) -> MLXArray {
                let layer = layers[index]
                return runSg(codes: a.codes, ascale: a.ascale, rsb: a.rsb,
                    layers: [(layer.weight, layer.scalesT, layer.biasesT)], m: a.rows, k: a.k,
                    negative: negative, form: sgForm)[0]
            }
            return (launch(activation, 0), { launch($0, 1) })
        }
        guard !streamPromptReady else { return nil }
        let (lead, late) = (layers[0], layers[1])
        guard let first = matmul(
            activation, weight: lead.weight, scalesT: lead.scalesT, biasesT: lead.biasesT, negative: lead.negative)
        else { return nil }
        return (first, {
            matmul($0, weight: late.weight, scalesT: late.scalesT, biasesT: late.biasesT, negative: late.negative)
        })
    }

    nonisolated(unsafe) private static var splitCheckFailed = false

    /// `splitPair`, its second launch ordered after its first, against
    /// `matmul(_:layers:)` on fixed synthetic operands: every output bit at
    /// 512, 192 and 128 rows, for each pair of offset forms, on the
    /// canonical words or, with `streamed`, on their streamed copies. A
    /// compile or run error counts as a mismatch. Returns the number of
    /// cases, nil when one differed.
    static func checkSplitPair(streamed: Bool) -> Int? {
        let k = 4096, kg = k / 128, maxRows = 512
        let codeValues: [Int8] = (0 ..< maxRows * k).map { (i: Int) -> Int8 in
            let centered: Int = (i * 37 + 11) % 255 - 127
            return Int8(truncatingIfNeeded: centered)
        }
        let codes = MLXArray(codeValues, [maxRows, k])
        let ascaleValues: [Float] = (0 ..< maxRows * kg).map { (i: Int) -> Float in Float(i % 11 + 1) / 256 }
        let rsbValues: [Float] = (0 ..< maxRows * kg).map { (i: Int) -> Float in Float(i % 17 - 8) / 64 }
        let ascale = MLXArray(ascaleValues), rsb = MLXArray(rsbValues)
        let operands: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negatedT: MLXArray)] =
            [256, 128].enumerated().map { layer, n in
                let words: [UInt32] = (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
                    let value: UInt64 = UInt64(i + 1 + 7919 * layer) &* 2_654_435_761
                    return UInt32(truncatingIfNeeded: value)
                }
                let scaleValues: [Float16] = (0 ..< kg * n).map { (i: Int) -> Float16 in
                    let value: Float = Float(1 + (i + layer) % 13) / 1024
                    return Float16(value)
                }
                let biasValues: [Float16] = (0 ..< kg * n).map { (i: Int) -> Float16 in
                    let value: Float = -Float((i + 3 * layer) % 7) / 512
                    return Float16(value)
                }
                let scalesT = MLXArray(scaleValues, [kg, n])
                let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
                return (MLXArray(words, [n, k / 16]), scalesT, MLXArray(biasValues, [kg, n]), negatedT)
            }
        let noted = streamPromptNoted
        defer { streamPromptNoted = noted }
        let streamReadyBefore = streamPromptReady
        func u(_ a: MLXArray) -> MLXArray { a.view(dtype: .uint32) }
        var matched = true
        var cases = 0
        var copies: [MLXArray]?
        splitCheckFailed = false
        withErrorHandler({ _ in PromptInt8Matmul.splitCheckFailed = true }) {
            if streamed {
                let packed = operands.map { VerifyInt8Matmul.streamPack($0.weight, k: k, columns: 16) }
                eval(packed)
                copies = packed
                matched = !splitCheckFailed
            }
            for m in [512, 192, 128] where matched {
                let activation = PromptInt8Activation(
                    codes: codes[0 ..< m], ascale: ascale[0 ..< m * kg], rsb: rsb[0 ..< m * kg], rows: m, k: k)
                for negatives in [(false, false), (true, true), (true, false), (false, true)] where matched {
                    let layers = zip(operands, [negatives.0, negatives.1]).map { operand, negative in
                        (weight: operand.weight, scalesT: operand.scalesT,
                         biasesT: negative ? operand.negatedT : operand.biasesT, negative: negative)
                    }
                    guard let reference = matmul(activation, layers: layers, streamedWeights: copies),
                        let split = splitPair(
                            activation, layers: layers, streamedWeights: copies, streamedQualified: true),
                        let second = split.second(VerifySplitZ.ordered(activation, after: split.first)),
                        reference.count == 2
                    else { matched = false; break }
                    let checks = zip([split.first, second], reference).map { MLX.all(u($0) .== u($1)) }
                    eval(checks)
                    matched = !splitCheckFailed && checks.allSatisfy { $0.item(Bool.self) }
                    cases += 1
                }
            }
        }
        // `DARKBLOOM_BONSAI_PREFILL_STREAM_CHECK=1` makes `streamOutputs` hand back the canonical result and clear
        // `streamPromptReady` on a streamed mismatch; that is a failed case, not a pass.
        matched = matched && !splitCheckFailed && streamPromptReady == streamReadyBefore
        splitCheckFailed = false
        return matched ? cases : nil
    }
}

/// The prompt GDN z projection out of the qkv launch. In the record the
/// prompt qkv and z run as one launch (one per layer on the plain route)
/// before the convolution, the b/a GEMM and the fresh8 scan, though z feeds
/// only the gated norm after the scan. Split, z runs in its own launch
/// whose codes are ordered after the scan output, with no barrier between
/// them, so z's weight stream overlaps the latency-bound scan. The split
/// serves only after `PromptInt8Matmul.checkSplitPair` found every output
/// equal, bit for bit, to the record route's; otherwise the record launch
/// stays. `DARKBLOOM_BONSAI_PROMPT_Z_SPLIT=0` keeps the record launch.
enum PromptSplitZ {
    static let offered: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_PROMPT_Z_SPLIT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The split qualified on the canonical words. Resolved once, at the
    /// first packed module's init, after `PromptInt8Matmul.route` and
    /// before any timed work.
    static let ready: Bool = {
        let passed: Bool
        let note: String
        if !offered {
            (passed, note) = (false, "off (DARKBLOOM_BONSAI_PROMPT_Z_SPLIT=0)")
        } else if PromptInt8Matmul.route == .off {
            (passed, note) = (false, "off (int8 prompt off)")
        } else if let cases = PromptInt8Matmul.checkSplitPair(streamed: false) {
            (passed, note) = (true, "on (\(PromptInt8Matmul.route), rows 512/192/128, \(cases) cases bit-identical)")
        } else {
            (passed, note) = (false, "off (a case differed or failed to compile)")
        }
        FileHandle.standardError.write(Data("int8 prompt z split: \(note)\n".utf8))
        return passed
    }()

    /// The split qualified on the streamed words too. Set once, at load,
    /// by `PromptInt8Matmul.prepareStreaming` after those words qualify.
    nonisolated(unsafe) private(set) static var streamedReady = false

    static func qualifyStreamed() {
        guard ready else { return }
        let cases = PromptInt8Matmul.checkSplitPair(streamed: true)
        streamedReady = cases != nil
        let note = cases.map { "on (\($0) cases bit-identical)" } ?? "off (a case differed or failed to compile)"
        FileHandle.standardError.write(Data("int8 prompt z split, streamed words: \(note)\n".utf8))
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
    /// The canonical 2-bit codes remain model parameters. This private working
    /// copy only permutes uint32 words into [tile, group, lane, word].
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

    /// `DARKBLOOM_BONSAI_VERIFY_WIDESTORE=0` keeps 16-B tile stores in the
    /// 32-column streamed body. Otherwise each lane writes each pair of its
    /// staged words as one 32-B store (four instead of eight per group) once
    /// the staging twin below writes the same tile words as the 16-B form at
    /// load. The tile buffer is declared 32-B aligned for that body.
    static let wideStoreAllowed: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_WIDESTORE"]?.lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    nonisolated(unsafe) private static var wideStore = false
    nonisolated(unsafe) private static var wideStore16 = false

    private static func streamLoad(columns: Int) -> String {
        "for (int t = 0; t < \(columns / 4); t += 4) { const uint4 u = *((const device uint4*)(wcol + (t / 4) * Kg * 128 + gl * 128 + lane * 4)); wv[t] = u.x; wv[t + 1] = u.y; wv[t + 2] = u.z; wv[t + 3] = u.w; }"
    }

    private static func streamStore(columns: Int, wide: Bool) -> String {
        if wide && (columns == 32 || columns == 16) {
            return "if ((t & 1) != 0) { const uint col = (lane >> 1) + 16u * uint(t / 4); const uint word = (lane & 1u) * 4u + uint(t % 4) - 1u; const uint32_t p = wv[t - 1]; const uint4 lo = uint4(p & 0x03030303u, (p >> 2) & 0x03030303u, (p >> 4) & 0x03030303u, (p >> 6) & 0x03030303u); const uint4 hi = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u); threadgroup ulong4* target = (threadgroup ulong4*)(bs[sg] + col * 32u + word * 4u); *target = ulong4(as_type<ulong2>(lo), as_type<ulong2>(hi)); }"
        }
        return "const uint col = (lane >> 1) + 16u * uint(t / 4); const uint word = (lane & 1u) * 4u + uint(t % 4); threadgroup uint4* target = (threadgroup uint4*)(bs[sg] + col * 32u + word * 4u); *target = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);"
    }

    /// The streamed staging alone (no tensor op): one simdgroup per (tile,
    /// group) loads the copy's words and writes the `columns x 32`-word B tile
    /// to device memory with the body's own load and store statements.
    private static func stagingTwinSource(columns: Int, wide: Bool) -> String {
        let words = columns / 4
        let tileWords = columns * 32
        let store = streamStore(columns: columns, wide: wide)
            .replacingOccurrences(of: "threadgroup ", with: "device ")
            .replacingOccurrences(of: "bs[sg]", with: "tile")
        return """
            const uint lane = thread_index_in_simdgroup;
            const int bx = int(threadgroup_position_in_grid.x);
            const int gl = int(threadgroup_position_in_grid.y);
            const int Kg = K / 128;
            const device uint32_t* wcol = w + (size_t)bx * (\(columns / 16)) * Kg * 128;
            device uint32_t* tile = out + ((size_t)bx * Kg + gl) * \(tileWords);
            uint32_t wv[\(words)];
            \(streamLoad(columns: columns))
            for (int t = 0; t < \(words); t++) {
              const uint32_t v = wv[t];
              \(store)
            }
            """
    }

    private static let stagingTwinKernels = [false, true].map { (wide: Bool) -> MLXFast.MLXFastKernel in
        MLXFast.metalKernel(
            name: wide ? "bonsai_verify_stream_staging_wide" : "bonsai_verify_stream_staging_pair",
            inputNames: ["w"], outputNames: ["out"], source: stagingTwinSource(columns: 32, wide: wide),
            ensureRowContiguous: true)
    }

    private static let stagingTwin16Kernels = [false, true].map { (wide: Bool) -> MLXFast.MLXFastKernel in
        MLXFast.metalKernel(
            name: wide ? "bonsai_verify_stream_staging_n16_wide" : "bonsai_verify_stream_staging_n16_pair",
            inputNames: ["w"], outputNames: ["out"], source: stagingTwinSource(columns: 16, wide: wide),
            ensureRowContiguous: true)
    }

    /// Every word of every tile and group at each streamed K, both forms over
    /// one real copy of hashed words; 255 marks an unwritten word (a staged
    /// word holds bytes 0...3 only). A compile or run error is a mismatch.
    private static func checkWideStore(columns: Int) -> Bool {
        let n = 64
        let tileWords = columns * 32
        let kernels = columns == 16 ? stagingTwin16Kernels : stagingTwinKernels
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
                let tiles: [MLXArray] = kernels.map { kernel in
                    kernel([packed], template: [("K", k)], grid: (n / columns * 32, kg, 1), threadGroup: (32, 1, 1),
                        outputShapes: [[n / columns * kg * tileWords]], outputDTypes: [.uint32], initValue: 255)[0]
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

    /// Runs before any streamed kernel is built, so a failed twin keeps the
    /// 16-B stores in every streamed source.
    static func prepareWideStore() {
        guard wideStoreAllowed else {
            FileHandle.standardError.write(Data("int8 verify streamed 32-B tile stores: off (DARKBLOOM_BONSAI_VERIFY_WIDESTORE=0)\n".utf8))
            return
        }
        wideStore = checkWideStore(columns: 32)
        wideStore16 = checkWideStore(columns: 16)
        FileHandle.standardError.write(Data("int8 verify streamed 32-B tile stores: N32 \(wideStore ? "passed" : "FAILED"), N16 \(wideStore16 ? "passed" : "FAILED") staging twin K 4096/12288 all tiles/groups/lanes\n".utf8))
    }

    /// Same staged B tile, int32 tensor result and FP32 epilogue. Only the
    /// word addresses/load width and the position of next-group loads change.
    private static func makeStreamedBody(_ body: String, columns: Int) -> String? {
        let words = columns / 4
        let oldPointer = "const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;"
        let pointer = "const device uint32_t* wcol = w + (size_t)bx * (\(columns / 16)) * Kg * 128;"
        let oldLoad = "for (int t = 0; t < \(words); t++) { wv[t] = wcol[(size_t)(4 * t) * KW + gl * 8]; }"
        let load = streamLoad(columns: columns)
        let oldStore = "dst[32 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);"
        let useWide = columns == 16 ? wideStore16 : (columns == 32 ? wideStore : false)
        let store = streamStore(columns: columns, wide: useWide)
        guard body.components(separatedBy: oldPointer).count == 2,
            body.components(separatedBy: oldLoad).count == 3,
            body.components(separatedBy: oldStore).count == 2 else { return nil }
        var result = body.replacingOccurrences(of: oldPointer, with: pointer)
            .replacingOccurrences(of: oldLoad, with: load)
            .replacingOccurrences(of: oldStore, with: store)
        if useWide {
            let tile = "threadgroup uint32_t bs[4][\(columns) * 32];"
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

    static func streamedFormsSource(layers: Int) -> String {
        let prefix = naxFastPick(layers: layers).replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
        let body = streamedBody(naxFastBody, columns: 32).replacingOccurrences(of: "(size_t)", with: "(uint)")
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
    private static let streamedNarrowKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_streamed_col_n16", inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxNarrowPick.replacingOccurrences(of: "const int K = ksz[0];", with: "const int K = KC;")
            + streamedBody(naxNarrow16Body, columns: 16).replacingOccurrences(of: "(size_t)", with: "(uint)"),
        header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)

    static func streamColumns(k: Int, n: Int) -> Int { 16 }

    /// Input-independent guard at load, before any request. Compare new
    /// pipelines with the unchanged canonical forms, including dense b/a.
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
                            guard formsMatched.contains("\(k) \(count)") else { matched = false; break }
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

    /// Same Q8 values as naxFastBody; five stable selections per 32-column
    /// tile. The fifth value bounds all omitted values. No target changes.
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
            && formsAllowed && hasMatchedForm(k: k, layers: 1) && formsFit(k: k, n: n)
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
             HostConstants.int32([Int32(activation.k), Int32(activation.rows), Int32(weight.dim(0))])],
            template: [("ROWS", 16), ("NEG", negative && negativeForms ? 1 : 0), ("KC", activation.k)],
            grid: (tiles * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[16, tiles, 4], [16, tiles, 4], [16, tiles], [16, tiles]],
            outputDTypes: [.float32, .int32, .float32, .int32])
        return (result[1], result[0], result[2], result[3])
    }

    /// `naxFastBody` for `layers` sibling projections of one activation in
    /// one dispatch: threadgroups x < N0 / 32 take layer 0, the next N1 / 32
    /// layer 1, and so on, each running the one-layer body unchanged. Inputs
    /// `xq`, `w0`..., `s0`..., `b0`..., `ascale`, `rsb`, `ksz` = [K, rows,
    /// N0, N1, ...]; outputs `out0`....
    static func naxFastSource(layers: Int) -> String {
        naxFastPick(layers: layers) + naxFastBody
    }

    /// `naxFastSource` with K a compile-time constant (template `KC`, the
    /// loop bound and the word and group strides) and every `size_t` index
    /// computed in 32 bits. Each index is below 2^32 on every shape this
    /// route takes (`formsFit`), so each address, and so each value read and
    /// each FP32 step, is the same: the same bits, which `checkForms` checks
    /// at each real K before a form is used.
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

    /// The narrow output uses only 16 columns. Keep the accepted N32
    /// operation as fallback; use this N16 operation only after its check.
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

    /// `naxFormsSource` for one, two and three layers (index `layers - 1`).
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

    /// The model's verify K values (hidden 4096, MLP 12288); each gets its
    /// own compile-time form, used only after `checkForms` matched it at
    /// that K. Any other K keeps the record kernel.
    static let formsK = [4096, 12288]
    /// `"\(k) \(layers)"` for each form that matched the record kernel at
    /// load, both offset forms included.
    nonisolated(unsafe) private static var formsMatched: Set<String> = []
    /// Bitmask for fast hot-path lookup without string formatting:
    /// bit `(layers - 1)` for `k == 4096`, bit `(layers + 3)` for `k == 12288`.
    nonisolated(unsafe) private static var formsMatchedMask: UInt32 = 0
    @inline(__always)
    private static func formBit(k: Int, layers: Int) -> UInt32 {
        guard (1 ... 3).contains(layers) else { return 0 }
        if k == 4096 { return 1 << UInt32(layers - 1) }
        if k == 12288 { return 1 << UInt32(layers + 3) }
        return 0
    }
    @inline(__always)
    private static func hasMatchedForm(k: Int, layers: Int) -> Bool {
        let bit = formBit(k: k, layers: layers)
        if bit != 0 { return (formsMatchedMask & bit) != 0 }
        return formsMatched.contains("\(k) \(layers)")
    }
    /// `DARKBLOOM_BONSAI_VERIFY_FORMS=0` keeps the record kernel everywhere.
    static let formsAllowed: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_FORMS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Every 32-bit index the forms compute stays below 2^31: word offsets
    /// below N * K / 16, scale and offset offsets below K / 128 * N, output
    /// offsets below 16 * N (the head, N 248320 at K 4096, needs 63.6M).
    static func formsFit(k: Int, n: Int) -> Bool {
        n * (k / 16) < 1 << 31 && (k / 128) * n < 1 << 31 && 16 * n < 1 << 31
    }

    private static func usesForms(k: Int, layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)]) -> Bool {
        formsAllowed && hasMatchedForm(k: k, layers: layers.count)
            && layers.allSatisfy { formsFit(k: k, n: $0.weight.dim(0)) }
    }

    private static let naxNarrowKernel = MLXFast.metalKernel(
        name: "bonsai_q8_verify_matmul_nax_narrow",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxNarrowPick + naxNarrowBody,
        header: PromptInt8Matmul.naxHeader,
        ensureRowContiguous: true)

    /// The narrow kernel in the compile-time form `naxFormsSource` gives the
    /// fast one: K from template `KC` and every `size_t` index in 32 bits.
    /// Its indices are bounded as the fast kernel's (`formsFit`).
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

    /// The widest one-layer call that takes the narrow kernel, and the
    /// inner sizes of the model's calls that fit: attention o and GDN out
    /// (4096) and MLP down (12288).
    static let narrowColumns = 4096
    static let narrowDepths = [4096, 12288]

    /// Inner sizes whose narrow kernel matched the fast one at load.
    /// `DARKBLOOM_BONSAI_VERIFY_NARROW=0` leaves it empty.
    nonisolated(unsafe) private static var narrowK = Set<Int>()
    /// Inner sizes in `narrowK` whose compile-time narrow form also matched
    /// the record fast kernel at load. `DARKBLOOM_BONSAI_VERIFY_FORMS=0`
    /// leaves it empty (the narrow kernel then runs in its record form).
    nonisolated(unsafe) private static var narrowFormsK = Set<Int>()
    nonisolated(unsafe) private static var narrow16K = Set<Int>()
    nonisolated(unsafe) private static var narrow16FormsK = Set<Int>()
    /// Bitmask for O(1) hot-path narrow kernel checks without Set<Int> hashing:
    /// bits 0..3 for k == 4096 (narrow, narrowForms, narrow16, narrow16Forms),
    /// bits 4..7 for k == 12288 (narrow, narrowForms, narrow16, narrow16Forms).
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
        let dims = HostConstants.int32([Int32(k), Int32(m), Int32(n)])
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
        m: Int, k: Int, negative: Bool, forms: Bool? = nil, streamedWeights: [MLXArray]? = nil
    ) -> [MLXArray] {
        let useForms = forms ?? usesForms(k: k, layers: layers)
        let dims: MLXArray
        let inputs: [MLXArray]
        let outputShapes: [[Int]]
        let outputDTypes: [DType]
        let totalN: Int
        switch layers.count {
        case 1:
            let l0 = layers[0]
            let n0 = l0.weight.dim(0)
            totalN = n0
            dims = HostConstants.int32([Int32(k), Int32(m), Int32(n0)])
            let w0 = (useForms ? streamedWeights?[0] : nil) ?? l0.weight
            inputs = [codes, w0, l0.scalesT, l0.biasesT, ascale, rsb, dims]
            outputShapes = [[m, n0]]
            outputDTypes = [.float32]
        case 2:
            let l0 = layers[0], l1 = layers[1]
            let n0 = l0.weight.dim(0), n1 = l1.weight.dim(0)
            totalN = n0 + n1
            dims = HostConstants.int32([Int32(k), Int32(m), Int32(n0), Int32(n1)])
            let w0 = (useForms ? streamedWeights?[0] : nil) ?? l0.weight
            let w1 = (useForms ? streamedWeights?[1] : nil) ?? l1.weight
            inputs = [codes, w0, w1, l0.scalesT, l1.scalesT, l0.biasesT, l1.biasesT, ascale, rsb, dims]
            outputShapes = [[m, n0], [m, n1]]
            outputDTypes = [.float32, .float32]
        case 3:
            let l0 = layers[0], l1 = layers[1], l2 = layers[2]
            let n0 = l0.weight.dim(0), n1 = l1.weight.dim(0), n2 = l2.weight.dim(0)
            totalN = n0 + n1 + n2
            dims = HostConstants.int32([Int32(k), Int32(m), Int32(n0), Int32(n1), Int32(n2)])
            let w0 = (useForms ? streamedWeights?[0] : nil) ?? l0.weight
            let w1 = (useForms ? streamedWeights?[1] : nil) ?? l1.weight
            let w2 = (useForms ? streamedWeights?[2] : nil) ?? l2.weight
            inputs = [
                codes, w0, w1, w2,
                l0.scalesT, l1.scalesT, l2.scalesT,
                l0.biasesT, l1.biasesT, l2.biasesT,
                ascale, rsb, dims,
            ]
            outputShapes = [[m, n0], [m, n1], [m, n2]]
            outputDTypes = [.float32, .float32, .float32]
        default:
            let ns = layers.map { $0.weight.dim(0) }
            totalN = ns.reduce(0, +)
            dims = HostConstants.int32([Int32(k), Int32(m)] + ns.map { Int32($0) })
            let ws = (useForms ? streamedWeights : nil) ?? layers.map { $0.weight }
            inputs = [codes] + ws + layers.map { $0.scalesT } + layers.map { $0.biasesT } + [ascale, rsb, dims]
            outputShapes = ns.map { [m, $0] }
            outputDTypes = ns.map { _ in .float32 }
        }
        if useForms {
            let kernels = streamedWeights == nil ? naxFormsKernels : streamedKernels
            let template = [("ROWS", m), ("NEG", negative ? 1 : 0), ("KC", k)] as [(String, any KernelTemplateArg)]
            return kernels[layers.count - 1](
                inputs, template: template,
                grid: (totalN / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: outputShapes, outputDTypes: outputDTypes)
        }
        let template = [("ROWS", m), ("NEG", negative ? 1 : 0)] as [(String, any KernelTemplateArg)]
        return naxFastKernels[layers.count - 1](
            inputs, template: template,
            grid: (totalN / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: outputShapes, outputDTypes: outputDTypes)
    }

    private static func runNarrow(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray, weight: MLXArray,
        scalesT: MLXArray, biasesT: MLXArray, m: Int, k: Int, negative: Bool, forms: Bool? = nil, compact: Bool = false, streamedWeight: MLXArray? = nil
    ) -> MLXArray {
        let n = weight.dim(0)
        let dims = HostConstants.int32([Int32(k), Int32(m), Int32(n)])
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

    /// Compares the narrow kernel's output bits with the fast kernel's at
    /// each of `narrowDepths`, every verify row count and both offset forms,
    /// on 128 columns (eight narrow threadgroups) with all 16 code rows
    /// filled. An inner size joins `narrowK` only when every case matches.
    /// Both sides are the record forms (`forms: false`), so this check does
    /// not depend on `checkForms`. Where `formsAllowed`, the compile-time
    /// narrow form is compared with the same record fast output in each case
    /// too; an inner size joins `narrowFormsK` only when it is in `narrowK`
    /// and every such case matches (a compile error fails the form only).
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

    /// Runs each compile-time form against the record's fast kernel on a
    /// fixed, input-independent 16 x 64 x K problem at each real K (16 rows,
    /// the verify width, so these pipelines are also built here, before any
    /// timed work) and compares output bits: one layer with `NEG = 0` and,
    /// where the offset skip is on, `NEG = 1`; two and three layers on
    /// column slices where the shared forms are on. A form is used only at
    /// a K and layer count where every comparison matched; a compile error
    /// or any differing bit keeps the record kernel there.
    private static func checkForms() -> String {
        formsMatched = []
        formsMatchedMask = 0
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
                if matches([0 ..< n]) {
                    formsMatched.insert("\(k) 1")
                    formsMatchedMask |= formBit(k: k, layers: 1)
                }
                if naxShared, matches([0 ..< 32, 32 ..< 64]) {
                    formsMatched.insert("\(k) 2")
                    formsMatchedMask |= formBit(k: k, layers: 2)
                }
                if naxShared, matches([0 ..< 32, 32 ..< 64, 0 ..< 32]) {
                    formsMatched.insert("\(k) 3")
                    formsMatchedMask |= formBit(k: k, layers: 3)
                }
            }
        }
        formsFailed = false
        return "matched " + formsK.map { k in
            let layers = (1 ... 3).filter { hasMatchedForm(k: k, layers: $0) }.map { String($0) }
            return "K \(k): layers " + (layers.isEmpty ? "none" : layers.joined(separator: ","))
        }.joined(separator: "; ")
    }

    /// Decided once, at the first packed module's init. A NAX device takes
    /// the tensor kernel only when it compiles and matches the plain kernel
    /// bit for bit; otherwise it keeps the record's float path. Any other
    /// device takes the plain kernel. `DARKBLOOM_BONSAI_INT8_VERIFY=0` keeps
    /// the record's float path everywhere; `=nax` runs the check and the
    /// tensor kernel on any device (the M2 emulates the tensor ops).
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
            let eligible = naxShared && formsAllowed && hasMatchedForm(k: 4096, layers: 2)
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

    /// `matmul` for sibling projections of one activation: two or three
    /// layers in one dispatch when the fast kernel's shared forms matched at
    /// load, else one `matmul` per layer. Either way each output is that
    /// layer's `matmul` output bit for bit (`NEG = 0` reads offsets that are
    /// bit for bit the negated scales wherever `NEG = 1` would apply).
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

    /// The fast tensor kernel takes the screen when its route and its offset
    /// skip matched at load.
    static var screenAvailable: Bool { route == .nax && naxFast && negativeForms }

    /// The one-layer compile-time form (stored or streamed words) over the
    /// first `SG` 128-groups only. Physical K, every stride and every operand
    /// stay as they are; only the group loop and its next-group load stop at
    /// `SG`. The streaming conversion runs first, since it searches for the
    /// prefetch bound.
    static func screenSource(streamed: Bool) -> String? {
        var source: String = streamed ? streamedFormsSource(layers: 1) : naxFormsSource(layers: 1)
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

    /// `[rows, N]` FP32 sums over the first `groups` 128-groups, `NEG = 1`.
    static func screen(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray,
        streamed: Bool, groups: Int
    ) -> MLXArray? {
        guard let kernel = screenKernels[streamed ? 1 : 0] else { return nil }
        let n: Int = weight.dim(0)
        let dims = HostConstants.int32([Int32(activation.k), Int32(activation.rows), Int32(n)])
        let template = [("ROWS", activation.rows), ("NEG", 1), ("KC", activation.k), ("SG", groups)]
            as [(String, any KernelTemplateArg)]
        let inputs: [MLXArray] = [
            activation.codes, weight, scalesT, biasesT, activation.ascale, activation.rsb, dims,
        ]
        return kernel(
            inputs, template: template, grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[activation.rows, n]], outputDTypes: [.float32])[0]
    }

    /// One simdgroup per (row, id) over canonical words, on the ALU. Lanes
    /// 8s ..< 8s + 8 hold chain s (groups s, s + 4, ... below `GL`), lane & 7
    /// a word of each group. The group sum is an integer (|C| < 2^16), so its
    /// order is free; the FP32 steps per group, the four chains and their
    /// final order are `naxFastBody`'s, so at `GL = Kg` a value is that
    /// kernel's output bit for bit. `ALL` scores every row (`ids` unused).
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

    /// `headGatherSource` with each lane's integer group sum taken in FP16 pairs. An activation
    /// byte and a 2-bit code each become an exact half through the 1024 exponent
    /// (`0x6400 | byte`), and each half lane sums the four products of two `u` steps, at most
    /// 4 x 128 x 3 = 1536 < 2048 in magnitude, so every FP16 result is an exact integer and `d`
    /// is the record's integer. A lane's 16 activation bytes load as one `packed_uint4`, which
    /// needs only the 4-byte alignment of the record's `char4` loads. The FP32 steps per group,
    /// the four chains and their final order are the record text. Nil unless the cut matches once.
    static let headGatherFastSource: String? = {
        let dot = """
              int d = 0;
              #pragma clang loop unroll(full)
              for (int u = 0; u < 4; u++) {
                const char4 a = src[u];
                const uint c = (v >> uint(2 * u)) & 0x03030303u;
                d += int(a.x) * int(c & 255u) + int(a.y) * int((c >> 8) & 255u)
                  + int(a.z) * int((c >> 16) & 255u) + int(a.w) * int(c >> 24);
              }
            """
        let halves = """
              const packed_uint4 xw = *((const device packed_uint4*)src);
              int d = 0;
              #pragma clang loop unroll(full)
              for (int uu = 0; uu < 4; uu += 2) {
                half2 acc = half2(0.0h);
                #pragma clang loop unroll(full)
                for (int u = uu; u < uu + 2; u++) {
                  const uint A = xw[u] ^ 0x80808080u;
                  const half2 a02 = as_type<half2>((A & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h);
                  const half2 a13 = as_type<half2>(((A >> 8) & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h);
                  const uint c = v >> uint(2 * u);
                  const half2 c02 = as_type<half2>((c & 0x00030003u) | 0x64006400u) - half2(1024.0h);
                  const half2 c13 = as_type<half2>(((c >> 8) & 0x00030003u) | 0x64006400u) - half2(1024.0h);
                  acc = fma(a02, c02, acc);
                  acc = fma(a13, c13, acc);
                }
                d += int(acc.x) + int(acc.y);
              }
            """
        guard headGatherSource.components(separatedBy: dot).count == 2 else { return nil }
        return headGatherSource.replacingOccurrences(of: dot, with: halves)
    }()

    private static let headGatherFastKernel: MLXFast.MLXFastKernel? = headGatherFastSource.map { source in
        MLXFast.metalKernel(
            name: "bonsai_q8_head_gather_h16",
            inputNames: ["xq", "w", "scales", "biases", "ascale", "rsb", "ids", "ksz"], outputNames: ["out"],
            source: source, ensureRowContiguous: true)
    }

    /// `DARKBLOOM_BONSAI_DF2_GATHER_FAST=0` keeps the record gather.
    static let gatherFastAllowed = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DF2_GATHER_FAST"] != "0"
    /// Written once by `prepareGatherFast` at drafter bind; read-only afterwards.
    nonisolated(unsafe) static var gatherFastPassed = false
    nonisolated(unsafe) private static var gatherFastPrepared = false

    /// Runs once at drafter bind, before any request: `headGatherFastKernel` against
    /// `headGatherKernel` at 32 and 16 groups, `NEG` on and off, on `gatherCheckInputs`.
    /// `gatherFastPassed` is set only when every output bit matched; a mismatch or an MLX error
    /// keeps the record kernel. Only gathers at `ids` take the fast kernel.
    static func prepareGatherFast() {
        guard !gatherFastPrepared else { return }
        gatherFastPrepared = true
        var note = "off (DARKBLOOM_BONSAI_DF2_GATHER_FAST=0)"
        if gatherFastAllowed, let fast = headGatherFastKernel {
            do {
                let detail: String? = try withError { try gatherFastMismatch(fast) }
                gatherFastPassed = detail == nil
                note = detail.map { "off (\($0))" }
                    ?? "on (bitwise equal to the record kernel at 32 and 16 groups, NEG on and off, 15 rows x 128 ids"
                    + " with extreme codes, all-3 and all-0 rows, out-of-range and repeated ids)"
            } catch {
                note = "off (error: \(error))"
            }
        } else if gatherFastAllowed {
            note = "off (a cut of the record source did not match once)"
        }
        FileHandle.standardError.write(Data("DF2 head gather fast: \(note)\n".utf8))
    }

    private static func gatherFastMismatch(_ fast: MLXFast.MLXFastKernel) throws -> String? {
        let (activation, weight, scales, biases, ids) = gatherCheckInputs()
        for groups in [32, 16] {
            for negative in [true, false] {
                // A kernel application that MLX refused returns no output: throw, never index it.
                guard
                    let record = gatherOutputs(
                        activation, weight: weight, scales: scales, biases: biases, ids: ids, groups: groups,
                        negative: negative, kernel: headGatherKernel
                    ).first,
                    let mine = gatherOutputs(
                        activation, weight: weight, scales: scales, biases: biases, ids: ids, groups: groups,
                        negative: negative, kernel: fast
                    ).first
                else { throw MLXError.caught("head gather check: a kernel returned no output") }
                let same: MLXArray = MLX.all(record.view(dtype: .uint32) .== mine.view(dtype: .uint32))
                eval(same)
                if !same.item(Bool.self) { return "mismatch at \(groups) groups, NEG \(negative)" }
            }
        }
        return nil
    }

    /// 15 rows of int8 codes (11 hashed over -128 ... 127, then all -128, all 127, alternating
    /// -128 and 127, all 0), 512 weight rows (row 0 all codes 3, row 1 all 0, the rest hashed),
    /// FP16 scales and biases over [-2, 2) with zeros, and 128 ids per row with -1, 512, 0, 511
    /// and a repeated id among hashed ones.
    private static func gatherCheckInputs() -> (PromptInt8Activation, MLXArray, MLXArray, MLXArray, MLXArray) {
        let k = 4096, kg = k / 128, n = 512, rows = 15, p = 128
        func hash(_ i: Int, _ salt: Int) -> Int {
            Int(UInt32(truncatingIfNeeded: (UInt64(i) &* 2_654_435_761 &+ UInt64(salt) &* 40_503) >> 5))
        }
        let codes: [Int8] = (0 ..< rows * k).map { (i: Int) -> Int8 in
            switch i / k {
            case 11: return -128
            case 12: return 127
            case 13: return i % 2 == 0 ? -128 : 127
            case 14: return 0
            default: return Int8(truncatingIfNeeded: hash(i, 1))
            }
        }
        let words: [UInt32] = (0 ..< n * k / 16).map { (i: Int) -> UInt32 in
            switch i / (k / 16) {
            case 0: return 0xFFFF_FFFF
            case 1: return 0
            default: return UInt32(truncatingIfNeeded: hash(i, 2) &* 2_246_822_519)
            }
        }
        func constants(_ salt: Int) -> [Float16] {
            (0 ..< n * kg).map { (i: Int) -> Float16 in
                let h = hash(i, salt) % 4096
                return h % 17 == 0 ? 0 : Float16(Float(h - 2048) / 1024)
            }
        }
        let ascale: [Float] = (0 ..< rows * kg).map { Float(hash($0, 5) % 1000 + 1) / 1.0e5 }
        let rsb: [Float] = (0 ..< rows * kg).map { Float(hash($0, 6) % 20001 - 10000) / 100 }
        let ids: [Int32] = (0 ..< rows * p).map { (i: Int) -> Int32 in
            switch i % 13 {
            case 0: return -1
            case 1: return Int32(n)
            case 2: return 0
            case 3: return Int32(n - 1)
            case 4: return 5
            default: return Int32(hash(i, 7) % n)
            }
        }
        let activation = PromptInt8Activation(
            codes: MLXArray(codes, [rows, k]), ascale: MLXArray(ascale), rsb: MLXArray(rsb), rows: rows, k: k)
        return (
            activation, MLXArray(words, [n, k / 16]), MLXArray(constants(3), [n, kg]),
            MLXArray(constants(4), [n, kg]), MLXArray(ids, [rows, p])
        )
    }

    /// `[rows, P]` values at `ids` `[rows, P]` int32 (every row when nil, P = N)
    /// over the first `groups` 128-groups; `scales` and `biases` are `[N, Kg]`.
    static func gather(
        _ activation: PromptInt8Activation, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        ids: MLXArray?, groups: Int, negative: Bool
    ) -> MLXArray {
        let fast: MLXFast.MLXFastKernel? = ids != nil && gatherFastPassed ? headGatherFastKernel : nil
        return gatherOutputs(
            activation, weight: weight, scales: scales, biases: biases, ids: ids, groups: groups,
            negative: negative, kernel: fast ?? headGatherKernel)[0]
    }

    /// The gather's output vector `[[rows, P]]`, unchecked.
    private static func gatherOutputs(
        _ activation: PromptInt8Activation, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        ids: MLXArray?, groups: Int, negative: Bool, kernel: MLXFast.MLXFastKernel
    ) -> [MLXArray] {
        let n: Int = weight.dim(0)
        let p: Int = ids?.dim(-1) ?? n
        let dims = HostConstants.int32([Int32(activation.k), Int32(n), Int32(p)])
        let template = [("ROWS", activation.rows), ("NEG", negative), ("GL", groups), ("ALL", ids == nil)]
            as [(String, any KernelTemplateArg)]
        let inputs: [MLXArray] = [
            activation.codes, weight, scales, biases, activation.ascale, activation.rsb,
            ids ?? HostConstants.int32([Int32(0)]), dims,
        ]
        return kernel(
            inputs, template: template, grid: (activation.rows * p * 32, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[activation.rows, p]], outputDTypes: [.float32])
    }
}

// MARK: - The dense GDN b/a projection in the qkv/z verify dispatch

/// The dense FP32 `in_proj_b` / `in_proj_a` of a few-row verify window. Each
/// simdgroup owns one output column (the b columns, then the a columns),
/// reads that weight row once for all `ROWS` rows with FP32 FMAs, then
/// `simd_sum`. The model's standalone kernel (`qwen35_gdn_dense_ba_rows`)
/// compiles `standaloneSource`; the two-layer verify form below runs the same
/// `columnBody` in leading threadgroups of the qkv/z dispatch, which saves
/// one launch per GDN layer per round. The fused form is used only where it
/// matched both the standalone kernel and the two-layer form bit for bit at
/// load (`fusedMatched`).
public enum VerifyDenseBA {
    /// Needs `lane`, `n`, `K`, `N`, inputs `x`, `wb`, `wa`, outputs `ob`, `oa`.
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

    /// Grid `(32, 2N, 1)`, threadgroup `(32, 2, 1)`: two columns per threadgroup.
    public static let standaloneSource = """
        const uint lane = thread_index_in_simdgroup;
        const uint n = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
        const int K = x_shape[1];
        const uint N = wb_shape[0];

        """ + columnBody

    /// `naxFormsSource(layers:)` with `BAG` leading 128-thread threadgroups
    /// that each run four b/a columns and return; every later threadgroup
    /// runs the layers' body unchanged, `BAG` threadgroups later.
    static func fusedSource(streamed: Bool = false, layers: Int = 2) -> String {
        let forms = streamed
            ? VerifyInt8Matmul.streamedFormsSource(layers: layers) : VerifyInt8Matmul.naxFormsSource(layers: layers)
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

    /// The one-layer forms of `fusedKernel` and `streamedKernel`, for the
    /// qkv projection when z leaves its launch (`VerifySplitZ`); nil when
    /// the one-layer source lacks the anchor.
    static let fusedOneKernels: [MLXFast.MLXFastKernel?] = [false, true].map { streamed in
        let source = fusedSource(streamed: streamed, layers: 1)
        guard !source.isEmpty else { return nil }
        return MLXFast.metalKernel(
            name: streamed ? "bonsai_q8_verify_streamed_col1_dense_ba" : "bonsai_q8_verify_matmul_nax_forms1_dense_ba",
            inputNames: ["xq", "w0", "s0", "b0", "ascale", "rsb", "ksz", "x", "wb", "wa"],
            outputNames: ["out0", "ob", "oa"], source: source,
            header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)
    }

    /// `fused` for one layer: `[out0, ob, oa]`; nil when its kernel is missing.
    static func fusedOne(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layer: (weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray),
        m: Int, k: Int, negative: Bool, x: MLXArray, wb: MLXArray, wa: MLXArray, streamedWeight: MLXArray? = nil
    ) -> [MLXArray]? {
        guard let kernel = fusedOneKernels[streamedWeight == nil ? 0 : 1] else { return nil }
        let n0 = layer.weight.dim(0)
        let N = wb.dim(0)
        let inputs: [MLXArray] = [
            codes, streamedWeight ?? layer.weight, layer.scalesT, layer.biasesT, ascale, rsb,
            HostConstants.int32([Int32(k), Int32(m), Int32(n0)]), x, wb, wa,
        ]
        let template = [
            ("ROWS", m), ("NEG", negative ? 1 : 0), ("KC", k), ("BAG", N / 2), ("BAN", N),
        ] as [(String, any KernelTemplateArg)]
        return kernel(
            inputs, template: template,
            grid: ((N / 2 + n0 / 32) * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n0], [m, N], [m, N]],
            outputDTypes: [.float32, .float32, .float32])
    }

    /// The standalone kernel: `x` `[rows, K]` FP32, `wb` and `wa` `[N, K]` FP32.
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
        m: Int, k: Int, negative: Bool, x: MLXArray, wb: MLXArray, wa: MLXArray, streamedWeights: [MLXArray]? = nil
    ) -> [MLXArray] {
        let l0 = layers[0], l1 = layers[1]
        let n0 = l0.weight.dim(0), n1 = l1.weight.dim(0)
        let N = wb.dim(0)
        let dims = HostConstants.int32([Int32(k), Int32(m), Int32(n0), Int32(n1)])
        let w0 = streamedWeights?[0] ?? l0.weight
        let w1 = streamedWeights?[1] ?? l1.weight
        let inputs: [MLXArray] = [
            codes, w0, w1, l0.scalesT, l1.scalesT, l0.biasesT, l1.biasesT,
            ascale, rsb, dims, x, wb, wa,
        ]
        let template = [
            ("ROWS", m), ("NEG", negative ? 1 : 0), ("KC", k), ("BAG", N / 2), ("BAN", N),
        ] as [(String, any KernelTemplateArg)]
        let kernel = streamedWeights == nil ? fusedKernel : streamedKernel
        return kernel(
            inputs, template: template,
            grid: ((N / 2 + (n0 + n1) / 32) * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n0], [m, n1], [m, N], [m, N]],
            outputDTypes: [.float32, .float32, .float32, .float32])
    }

    /// `DARKBLOOM_BONSAI_VERIFY_DENSE_BA=0` keeps the separate b/a launch.
    static let allowed: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_DENSE_BA"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    nonisolated(unsafe) private static var failed = false

    /// Set once by `check`, which `VerifyInt8Matmul.route` runs at load,
    /// after the compile-time forms check and before any timed work.
    nonisolated(unsafe) static var fusedMatched = false

    /// Fixed, input-independent problems at K 4096 with N 32 b/a columns and
    /// 64 matmul columns as two layers, at every verify row count 2 ... 16
    /// (each its own pipeline, all built here) and both offset forms: every
    /// output of the fused kernel must equal, bit for bit, the two-layer
    /// form's outputs and the standalone b/a kernel's outputs.
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
    /// The two-layer compile-time form is what the qkv/z pair takes at `k`.
    static func denseBAEligible(k: Int) -> Bool {
        route == .nax && naxFast && naxShared && formsAllowed && hasMatchedForm(k: k, layers: 2)
    }

    static func denseBAForms(
        codes: MLXArray, ascale: MLXArray, rsb: MLXArray,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray)],
        m: Int, k: Int, negative: Bool
    ) -> [MLXArray] {
        runFast(codes: codes, ascale: ascale, rsb: rsb, layers: layers, m: m, k: k, negative: negative, forms: true)
    }

    /// `matmul(activation, layers:)` for exactly two layers, plus the dense
    /// b/a of `x` in the same dispatch; nil wherever `matmul` would not take
    /// the two-layer form, or the fused form did not match at load.
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
            wb.ndim == 2, wb.shape2 == wa.shape2, wb.dim(1) == activation.k, wb.dim(0) % 2 == 0,
            wb.dtype == .float32, wa.dtype == .float32,
            VerifyDenseBA.fusedMatched
        else { return nil }
        let outputs = VerifyDenseBA.fused(
            codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb, layers: plain,
            m: activation.rows, k: activation.k,
            negative: negativeForms && layers.allSatisfy { $0.negative }, x: x, wb: wb, wa: wa, streamedWeights: streamedWeights)
        return ([outputs[0], outputs[1]], outputs[2], outputs[3])
    }

    /// The qkv/z pair as the record's boundary route returns it, `[qkv, z,
    /// b, a]`: the fused dense b/a form where it applies, else the shared form
    /// and the standalone b/a. `VerifySplitZ.check` compares against it.
    static func splitReference(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        x: MLXArray, wb: MLXArray, wa: MLXArray, streamedWeights: [MLXArray]?
    ) -> [MLXArray]? {
        if let fused = matmulWithDenseBA(
            activation, layers: layers, x: x, wb: wb, wa: wa, streamedWeights: streamedWeights)
        {
            return fused.outputs + [fused.b, fused.a]
        }
        guard let outputs = matmul(activation, layers: layers, streamedWeights: streamedWeights) else { return nil }
        let ba = VerifyDenseBA.standalone(x, wb: wb, wa: wa)
        return outputs + [ba.0, ba.1]
    }

    /// The record runs the qkv/z pair in one launch whose offset form is the
    /// pair's (`NEG` only when both layers take it); elsewhere each layer
    /// runs alone with its own.
    static var pairSharesOffsetForm: Bool { route == .nax && naxFast && naxShared }

    /// The first launch of the split: the qkv projection, with the dense b/a
    /// in the same dispatch on the tensor route (`b`, `a` nil elsewhere: the
    /// caller then runs the record's standalone b/a). `negative` is the
    /// offset form the record's launch gives this layer.
    static func splitLead(
        _ activation: PromptInt8Activation,
        layer: (weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray),
        negative: Bool, x: MLXArray, wb: MLXArray, wa: MLXArray, streamedWeight: MLXArray?
    ) -> (out: MLXArray, b: MLXArray?, a: MLXArray?)? {
        guard route != .off, rowRange.contains(activation.rows) else { return nil }
        guard route == .nax else {
            guard let out = matmul(
                activation, weight: layer.weight, scalesT: layer.scalesT, biasesT: layer.biasesT,
                negative: negative)
            else { return nil }
            return (out, nil, nil)
        }
        guard activation.k == 4096, naxFast, usesForms(k: activation.k, layers: [layer]),
            x.ndim == 2, x.dim(0) == activation.rows, x.dim(1) == activation.k, x.dtype == .float32,
            wb.ndim == 2, wb.shape2 == wa.shape2, wb.dim(1) == activation.k, wb.dim(0) % 2 == 0,
            wb.dtype == .float32, wa.dtype == .float32,
            let outputs = VerifyDenseBA.fusedOne(
                codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb, layer: layer,
                m: activation.rows, k: activation.k, negative: negativeForms && negative,
                x: x, wb: wb, wa: wa, streamedWeight: streamedWeight)
        else { return nil }
        return (outputs[0], outputs[1], outputs[2])
    }

    /// The second launch of the split: the z projection alone, through the
    /// record's one-layer route (the narrow kernel the GDN out projection
    /// takes at the same shape).
    static func splitDeferred(
        _ activation: PromptInt8Activation,
        layer: (weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray),
        negative: Bool, streamedWeight: MLXArray?
    ) -> MLXArray? {
        matmul(
            activation, weight: layer.weight, scalesT: layer.scalesT, biasesT: layer.biasesT,
            negative: negative, streamedWeight: streamedWeight)
    }
}

/// The GDN z projection out of the qkv/z verify launch. z feeds only the
/// gated norm after the recurrence, but in the record it shares the launch
/// the recurrence waits on. Split, qkv (with the dense b/a) runs first and z
/// runs after the recurrence kernel is encoded, with no barrier between
/// them, so z's weight stream overlaps the latency-bound recurrence. The
/// split serves only after `check` found every output equal, bit for bit,
/// to the record route's on synthetic operands at every verify row count,
/// both offset forms and both weight layouts; otherwise the record launch
/// stays. `DARKBLOOM_BONSAI_VERIFY_Z_SPLIT=0` keeps the record launch.
enum VerifySplitZ {
    static let offered: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_Z_SPLIT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    nonisolated(unsafe) private static var failed = false

    /// Resolved once, at the first packed module's init, after
    /// `VerifyInt8Matmul.route` and before any timed work.
    static let ready: Bool = {
        let (passed, note) = check()
        FileHandle.standardError.write(Data("int8 verify z split: \(note)\n".utf8))
        return passed
    }()

    /// `ordered` with its codes made after `dependency` in the graph: the
    /// same buffer, so the launch that reads them waits on no barrier for it.
    static func ordered(_ activation: PromptInt8Activation, after dependency: MLXArray) -> PromptInt8Activation {
        PromptInt8Activation(
            codes: depends(input: activation.codes, dependencies: [dependency]),
            ascale: activation.ascale, rsb: activation.rsb, rows: activation.rows, k: activation.k)
    }

    private static func check() -> (Bool, String) {
        guard offered else { return (false, "off (DARKBLOOM_BONSAI_VERIFY_Z_SPLIT=0)") }
        guard VerifyInt8Matmul.route != .off else { return (false, "off (int8 verify off)") }
        let k = 4096, n = 32, N = 32, kg = k / 128
        let maxRows = VerifyInt8Matmul.tileRows
        func pattern(_ count: Int, _ mul: Int, _ add: Int, _ mod: Int, _ center: Int, _ scale: Float) -> [Float] {
            var values: [Float] = []
            values.reserveCapacity(count)
            for i in 0 ..< count {
                let v: Int = (i * mul + add) % mod - center
                values.append(Float(v) / scale)
            }
            return values
        }
        let codeValues: [Int8] = (0 ..< maxRows * k).map { (i: Int) -> Int8 in
            let centered: Int = (i * 37 + 11) % 255 - 127
            return Int8(truncatingIfNeeded: centered)
        }
        let codes = MLXArray(codeValues, [maxRows, k])
        let words: [UInt32] = (0 ..< 2 * n * k / 16).map { (i: Int) -> UInt32 in
            let value: UInt64 = UInt64(i + 1) &* 2_654_435_761
            return UInt32(truncatingIfNeeded: value)
        }
        let weight = MLXArray(words, [2 * n, k / 16])
        let scaleValues: [Float16] = (0 ..< kg * 2 * n).map { (i: Int) -> Float16 in
            let value: Float = Float(1 + i % 13) / 1024
            return Float16(value)
        }
        let scalesT = MLXArray(scaleValues, [kg, 2 * n])
        let biasValues: [Float16] = (0 ..< kg * 2 * n).map { (i: Int) -> Float16 in
            let value: Float = -Float(i % 7) / 512
            return Float16(value)
        }
        let biasesT = MLXArray(biasValues, [kg, 2 * n])
        let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
        let ascale = MLXArray(pattern(maxRows * kg, 1, 11, 11, -1, 256))
        let rsb = MLXArray(pattern(maxRows * kg, 1, 0, 17, 8, 64))
        let wb = MLXArray(pattern(N * k, 29, 3, 97, 48, 3072), [N, k])
        let wa = MLXArray(pattern(N * k, 53, 7, 89, 44, 2048), [N, k])
        let xAll = MLXArray(pattern(maxRows * k, 41, 5, 211, 105, 64), [maxRows, k])
        let halves = [0 ..< n, n ..< 2 * n]
        let packed: [MLXArray]? = VerifyInt8Matmul.streamReady
            ? halves.map { VerifyInt8Matmul.streamPack(weight[$0], k: k, columns: 16) } : nil
        if let packed { eval(packed) }
        func u(_ a: MLXArray) -> MLXArray { a.view(dtype: .uint32) }
        var matched = true
        var cases = 0
        failed = false
        withErrorHandler({ _ in VerifySplitZ.failed = true }) {
            for m in VerifyInt8Matmul.rowRange.reversed() where matched {
                let x = xAll[0 ..< m]
                for negative in [false, true] where matched {
                    let offsets = negative ? negatedT : biasesT
                    let layers = halves.map {
                        (weight: weight[$0], scalesT: scalesT[0..., $0], biasesT: offsets[0..., $0], negative: negative)
                    }
                    for streamed in packed == nil ? [nil] : [nil, packed] where matched {
                        let activation = PromptInt8Activation(codes: codes, ascale: ascale, rsb: rsb, rows: m, k: k)
                        guard let reference = VerifyInt8Matmul.splitReference(
                                activation, layers: layers, x: x, wb: wb, wa: wa, streamedWeights: streamed),
                            let lead = VerifyInt8Matmul.splitLead(
                                activation, layer: (layers[0].weight, layers[0].scalesT, layers[0].biasesT),
                                negative: negative, x: x, wb: wb, wa: wa, streamedWeight: streamed?[0]),
                            let z = VerifyInt8Matmul.splitDeferred(
                                ordered(activation, after: lead.out),
                                layer: (layers[1].weight, layers[1].scalesT, layers[1].biasesT),
                                negative: negative, streamedWeight: streamed?[1])
                        else { matched = false; break }
                        let ba = lead.b.flatMap { b in lead.a.map { (b, $0) } } ?? VerifyDenseBA.standalone(x, wb: wb, wa: wa)
                        let split = [lead.out, z, ba.0, ba.1]
                        let checks = zip(split, reference).map { MLX.all(u($0) .== u($1)) }
                        eval(checks)
                        matched = !failed && reference.count == 4 && checks.allSatisfy { $0.item(Bool.self) }
                        cases += 1
                    }
                }
            }
        }
        if failed { matched = false }
        failed = false
        let rows = VerifyInt8Matmul.rowRange
        return matched
            ? (true, "on (\(VerifyInt8Matmul.route), rows \(rows.lowerBound)...\(rows.upperBound), \(cases) cases bit-identical)")
            : (false, "off (a case differed or failed to compile)")
    }
}

/// The attention output gate out of the q/k/v verify launch. The query
/// projection interleaves each 512-wide head as [query 256 | gate 256]; the
/// gate feeds only the attention-gate rotation after the attention, but in
/// the record it shares the launch the q/k RoPE waits on. Split, the q/k/v
/// launch runs the query halves only, and the gate halves run in a second
/// launch ordered after the attention's partial kernel, with no barrier
/// between them. Both launches run the record kernels with only the column
/// tile map changed, and write into a full-width `[rows, 8192]` buffer whose
/// other halves stay unwritten and unread. The split serves only after
/// `check` found every read half equal, bit for bit, to the record route's
/// on synthetic operands at every verify row count, both offset forms and
/// both weight layouts; otherwise the record launch stays.
/// `DARKBLOOM_BONSAI_VERIFY_GATE_SPLIT=0` keeps the record launch.
enum VerifySplitGate {
    static let offered: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_GATE_SPLIT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    nonisolated(unsafe) private static var failed = false

    /// Resolved once, at the first packed module's init, after
    /// `VerifyInt8Matmul.route` and before any timed work.
    static let ready: Bool = {
        let (passed, note) = check()
        FileHandle.standardError.write(Data("int8 verify attention gate split: \(note)\n".utf8))
        return passed
    }()

    private static func check() -> (Bool, String) {
        guard offered else { return (false, "off (DARKBLOOM_BONSAI_VERIFY_GATE_SPLIT=0)") }
        guard VerifyInt8Matmul.route != .off else { return (false, "off (int8 verify off)") }
        let k = 4096, kg = k / 128
        let widths = [1024, 64, 64]
        let total = widths.reduce(0, +)
        let maxRows = VerifyInt8Matmul.tileRows
        func pattern(_ count: Int, _ mul: Int, _ add: Int, _ mod: Int, _ center: Int, _ scale: Float) -> [Float] {
            var values: [Float] = []
            values.reserveCapacity(count)
            for i in 0 ..< count {
                let v: Int = (i * mul + add) % mod - center
                values.append(Float(v) / scale)
            }
            return values
        }
        let codeValues: [Int8] = (0 ..< maxRows * k).map { (i: Int) -> Int8 in
            let centered: Int = (i * 37 + 11) % 255 - 127
            return Int8(truncatingIfNeeded: centered)
        }
        let codes = MLXArray(codeValues, [maxRows, k])
        let words: [UInt32] = (0 ..< total * k / 16).map { (i: Int) -> UInt32 in
            let value: UInt64 = UInt64(i + 1) &* 2_654_435_761
            return UInt32(truncatingIfNeeded: value)
        }
        let weight = MLXArray(words, [total, k / 16])
        let scaleValues: [Float16] = (0 ..< kg * total).map { (i: Int) -> Float16 in
            let value: Float = Float(1 + i % 13) / 1024
            return Float16(value)
        }
        let scalesT = MLXArray(scaleValues, [kg, total])
        let biasValues: [Float16] = (0 ..< kg * total).map { (i: Int) -> Float16 in
            let value: Float = -Float(i % 7) / 512
            return Float16(value)
        }
        let biasesT = MLXArray(biasValues, [kg, total])
        let negatedT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
        let ascale = MLXArray(pattern(maxRows * kg, 1, 11, 11, -1, 256))
        let rsb = MLXArray(pattern(maxRows * kg, 1, 0, 17, 8, 64))
        let ranges = [0 ..< widths[0], widths[0] ..< widths[0] + widths[1], widths[0] + widths[1] ..< total]
        let packed: [MLXArray]? = VerifyInt8Matmul.streamReady
            ? ranges.map { VerifyInt8Matmul.streamPack(weight[$0], k: k, columns: 16) } : nil
        if let packed { eval(packed) }
        func u(_ a: MLXArray) -> MLXArray { a.view(dtype: .uint32) }
        func half(_ a: MLXArray, _ index: Int, rows: Int) -> MLXArray {
            a.reshaped([rows, widths[0] / 512, 2, 256])[0..., 0..., index, 0...]
        }
        var matched = true
        var cases = 0
        failed = false
        withErrorHandler({ _ in VerifySplitGate.failed = true }) {
            for m in VerifyInt8Matmul.rowRange.reversed() where matched {
                for negative in [false, true] where matched {
                    let offsets = negative ? negatedT : biasesT
                    let layers = ranges.map {
                        (weight: weight[$0], scalesT: scalesT[0..., $0], biasesT: offsets[0..., $0], negative: negative)
                    }
                    for streamed in packed == nil ? [nil] : [nil, packed] where matched {
                        let activation = PromptInt8Activation(codes: codes, ascale: ascale, rsb: rsb, rows: m, k: k)
                        guard let reference = VerifyInt8Matmul.matmul(
                                activation, layers: layers, streamedWeights: streamed),
                            let lead = VerifyInt8Matmul.gateSplitLead(
                                activation, layers: layers, streamedWeights: streamed),
                            let gate = VerifyInt8Matmul.gateSplitGate(
                                VerifySplitZ.ordered(activation, after: lead[0]), layers: layers,
                                streamedWeights: streamed)
                        else { matched = false; break }
                        let checks = [
                            MLX.all(u(half(lead[0], 0, rows: m)) .== u(half(reference[0], 0, rows: m))),
                            MLX.all(u(half(gate, 1, rows: m)) .== u(half(reference[0], 1, rows: m))),
                            MLX.all(u(lead[1]) .== u(reference[1])),
                            MLX.all(u(lead[2]) .== u(reference[2])),
                        ]
                        eval(checks)
                        matched = !failed && reference.count == 3 && lead.count == 3
                            && checks.allSatisfy { $0.item(Bool.self) }
                        cases += 1
                    }
                }
            }
        }
        if failed { matched = false }
        failed = false
        let rows = VerifyInt8Matmul.rowRange
        return matched
            ? (true, "on (\(VerifyInt8Matmul.route), rows \(rows.lowerBound)...\(rows.upperBound), \(cases) cases bit-identical)")
            : (false, "off (a case differed or failed to compile)")
    }
}

extension VerifyInt8Matmul {
    /// Layer 0's 32-column tiles in the split: threadgroup `t` of a half
    /// takes tile `(t / 8) * 16 + t % 8`, plus 8 for the gate half, so the
    /// query (gate) half of each 512-wide head lands where the record writes
    /// it. Nil when an anchor is missing.
    static func gateSplitSource(_ source: String, gate: Bool) -> String? {
        let anchor = "int N = ksz[2];\n"
        guard source.components(separatedBy: anchor).count == 2 else { return nil }
        let remap = gate
            ? "bx = (bx >> 3) * 16 + (bx & 7) + 8;\n"
            : "if (bx < N / 64) { bx = (bx >> 3) * 16 + (bx & 7); } else { bx += N / 64; }\n"
        return source.replacingOccurrences(of: anchor, with: anchor + remap)
    }

    static func gateSplitPlainSource() -> String? {
        let anchor = "const int n0 = int(threadgroup_position_in_grid.x) * 32;"
        guard plainSource.components(separatedBy: anchor).count == 2 else { return nil }
        return plainSource.replacingOccurrences(
            of: anchor,
            with: "const int n0 = ((int(threadgroup_position_in_grid.x) >> 3) * 16 + (int(threadgroup_position_in_grid.x) & 7) + 8 * HALF) * 32;")
    }

    /// `[lead canonical, lead streamed, gate canonical, gate streamed]`.
    private static let gateSplitKernels: [MLXFast.MLXFastKernel?] = [(false, false), (false, true), (true, false), (true, true)]
        .map { gate, streamed in
            let layers = gate ? 1 : 3
            let record = streamed ? streamedFormsSource(layers: layers) : naxFormsSource(layers: layers)
            return gateSplitSource(record, gate: gate).map { source in
                MLXFast.metalKernel(
                    name: "bonsai_q8_verify_\(streamed ? "streamed_col" : "matmul_nax_forms")\(layers)_\(gate ? "gate" : "query")_half",
                    inputNames: ["xq"] + (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" }
                        + (0 ..< layers).map { "b\($0)" } + ["ascale", "rsb", "ksz"],
                    outputNames: (0 ..< layers).map { "out\($0)" },
                    source: source, header: PromptInt8Matmul.naxHeader, ensureRowContiguous: true)
            }
        }

    private static let gateSplitPlainKernel: MLXFast.MLXFastKernel? = gateSplitPlainSource().map {
        MLXFast.metalKernel(
            name: "bonsai_q8_verify_matmul_plain_half",
            inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
            outputNames: ["out"], source: $0, ensureRowContiguous: true)
    }

    /// The record's three-layer launch takes the compile-time forms here.
    @inline(__always)
    private static func gateSplitTakesForms(
        k: Int, layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)]
    ) -> Bool {
        naxFast && naxShared && formsAllowed && hasMatchedForm(k: k, layers: 1)
            && hasMatchedForm(k: k, layers: layers.count)
            && layers.allSatisfy { formsFit(k: k, n: $0.weight.dim(0)) }
    }

    /// Layer 0 of the split on the plain route: one half of each head.
    private static func gateSplitPlain(
        _ activation: PromptInt8Activation,
        layer: (weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool), gate: Bool
    ) -> MLXArray? {
        guard let kernel = gateSplitPlainKernel else { return nil }
        let m = activation.rows, k = activation.k, n = layer.weight.dim(0)
        return kernel(
            [activation.codes, layer.weight, layer.scalesT, layer.biasesT, activation.ascale, activation.rsb,
             HostConstants.int32([Int32(k), Int32(m), Int32(n)])],
            template: [("ROWS", m), ("NEG", layer.negative && negativeForms ? 1 : 0), ("HALF", gate ? 1 : 0)],
            grid: (n / 64 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [.float32])[0]
    }

    /// The first launch of the split: `[query halves, k, v]` as the record's
    /// q/k/v launch computes them, layer 0 `[rows, N0]` with only the query
    /// half of each 512-wide head written. Nil where the split does not apply.
    static func gateSplitLead(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        streamedWeights: [MLXArray]?
    ) -> [MLXArray]? {
        guard route != .off, rowRange.contains(activation.rows), layers.count == 3,
            layers[0].weight.dim(0) % 512 == 0
        else { return nil }
        let m = activation.rows, k = activation.k
        guard route == .nax else {
            guard let query = gateSplitPlain(activation, layer: layers[0], gate: false),
                let key = matmul(
                    activation, weight: layers[1].weight, scalesT: layers[1].scalesT,
                    biasesT: layers[1].biasesT, negative: layers[1].negative),
                let value = matmul(
                    activation, weight: layers[2].weight, scalesT: layers[2].scalesT,
                    biasesT: layers[2].biasesT, negative: layers[2].negative)
            else { return nil }
            return [query, key, value]
        }
        guard gateSplitTakesForms(k: k, layers: layers),
            let kernel = gateSplitKernels[streamedWeights == nil ? 0 : 1]
        else { return nil }
        let l0 = layers[0], l1 = layers[1], l2 = layers[2]
        let n0 = l0.weight.dim(0), n1 = l1.weight.dim(0), n2 = l2.weight.dim(0)
        let w0 = streamedWeights?[0] ?? l0.weight
        let w1 = streamedWeights?[1] ?? l1.weight
        let w2 = streamedWeights?[2] ?? l2.weight
        let dims = HostConstants.int32([Int32(k), Int32(m), Int32(n0), Int32(n1), Int32(n2)])
        let inputs: [MLXArray] = [
            activation.codes, w0, w1, w2,
            l0.scalesT, l1.scalesT, l2.scalesT,
            l0.biasesT, l1.biasesT, l2.biasesT,
            activation.ascale, activation.rsb, dims,
        ]
        let negative = negativeForms && l0.negative && l1.negative && l2.negative
        return kernel(
            inputs, template: [("ROWS", m), ("NEG", negative ? 1 : 0), ("KC", k)],
            grid: ((n0 / 2 + n1 + n2) / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n0], [m, n1], [m, n2]], outputDTypes: [.float32, .float32, .float32])
    }

    /// The second launch of the split: layer 0 `[rows, N0]` with only the
    /// gate half of each 512-wide head written, in the offset form the
    /// record's q/k/v launch gives it.
    static func gateSplitGate(
        _ activation: PromptInt8Activation,
        layers: [(weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray, negative: Bool)],
        streamedWeights: [MLXArray]?
    ) -> MLXArray? {
        guard route != .off, rowRange.contains(activation.rows), layers.count == 3,
            layers[0].weight.dim(0) % 512 == 0
        else { return nil }
        guard route == .nax else { return gateSplitPlain(activation, layer: layers[0], gate: true) }
        let m = activation.rows, k = activation.k, n = layers[0].weight.dim(0)
        guard gateSplitTakesForms(k: k, layers: layers),
            let kernel = gateSplitKernels[streamedWeights == nil ? 2 : 3]
        else { return nil }
        let negative = negativeForms && layers[0].negative && layers[1].negative && layers[2].negative
        return kernel(
            [activation.codes, streamedWeights?[0] ?? layers[0].weight, layers[0].scalesT, layers[0].biasesT,
             activation.ascale, activation.rsb, HostConstants.int32([Int32(k), Int32(m), Int32(n)])],
            template: [("ROWS", m), ("NEG", negative ? 1 : 0), ("KC", k)],
            grid: (n / 64 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [.float32])[0]
    }
}

/// Group-major copies of a packed module's F16 scales and offsets for
/// `PromptInt8Matmul`, made on first use. A plain class, so nothing enters the
/// module parameter tree.
final class PromptInt8Constants {
    private var streamed16: MLXArray?
    private var streamed8: MLXArray?
    private var streamed: [Int: MLXArray] = [:]
    private var transposed: (MLXArray, MLXArray)?
    /// Every FP16 offset is bit for bit its negated scale, so the int8
    /// kernels may use `-s` for the offset (`NEG = 1`). Read once at init.
    let negative: Bool

    private var allowStream: Bool

    init(negative: Bool, allowStream: Bool) {
        self.negative = negative
        self.allowStream = allowStream
    }

    func streamWeight(_ weight: MLXArray, k: Int, columns: Int) -> MLXArray? {
        guard allowStream && VerifyInt8Matmul.streamReady else { return nil }
        if columns == 16, let streamed16 { return streamed16 }
        if columns == 8, let streamed8 { return streamed8 }
        if columns != 16, columns != 8, let value = streamed[columns] { return value }
        let value = VerifyInt8Matmul.streamPack(weight, k: k, columns: columns)
        eval(value)
        if columns == 16 {
            streamed16 = value
            PromptInt8Matmul.registerStream(weight, packed: value)
        } else if columns == 8 {
            streamed8 = value
        } else {
            streamed[columns] = value
        }
        return value
    }

    func get(scales: MLXArray, biases: MLXArray) -> (MLXArray, MLXArray) {
        if let transposed { return transposed }
        let pair = (scales.transposed(1, 0).contiguous(), biases.transposed(1, 0).contiguous())
        transposed = pair
        return pair
    }
}

/// The operand form a packed module's draft-head screen qualified with at
/// bind (`streamed` words or the stored ones); nil until it qualifies.
final class DraftHeadScreenState {
    var streamed: Bool?
    nonisolated(unsafe) static var failure: String?
}

/// Affine packed linear weights with a signed Hadamard input transform.
///
/// Pass weights already folded and packed in MLX format. The initializer does
/// not requantize them and does not accept raw GGUF quantization blocks.
public final class HadamardQuantizedLinear: QuantizedLinear {
    /// Qualify the shared immutable weight layout before serving requests.
    /// Prefill and verify then consume the same copies; no allocator policy or
    /// request scratch is changed, and no request work is moved into begin.
    public static func prepareSharedVerifyWeights() {
        PromptInt8Matmul.prepareStreaming()
    }

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
            _ = VerifySplitZ.ready
            _ = VerifySplitGate.ready
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
            _ = PromptSplitZ.ready
        }
    }


    /// Fixed at init: the packing `Float16ConstantQMV` reads in place.
    let readsFloat16ConstantsInPlace: Bool

    private let promptConstants: PromptInt8Constants
    let exactHeadBounds = ExactVerifyHeadBounds()
    let draftScreenState = DraftHeadScreenState()

    /// The prompt-width integer route (`PromptInt8Matmul`); nil when the
    /// input, the packing or the device does not take it.
    private func promptInt8Forward(_ x: MLXArray, up: MLXArray? = nil) -> MLXArray? {
        guard readsFloat16ConstantsInPlace, transform.blockSize == 4096, x.dtype == .float32,
            x.ndim >= 2, PromptInt8Matmul.applies(rows: x.size / x.dim(-1)),
            weight.dim(0) % 64 == 0, transform.width % 512 == 0, PromptInt8Matmul.route != .off,
            let activation = transform.quantizedForPrompt(gdnLayout.map { $0(x) } ?? x, up: up)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        let streamed = promptConstants.streamWeight(weight, k: transform.width, columns: 16)
        guard let output = PromptInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative, streamedWeight: streamed)
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
        let promptOutputs = quantized && !verify
            ? PromptInt8Matmul.matmul(
                PromptInt8Activation(
                    codes: activation[0], ascale: activation[1], rsb: activation[2], rows: rows, k: 4096),
                layers: packed.map(\.verifyOperands), streamedWeights: Self.streamedWeights(packed))
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

    /// `sharedPromptProjections(rmsNorm(x + delta, weight), projections)` with the
    /// residual add and RMSNorm formed in the quantizing kernel; returns `x + delta`
    /// and, with `normalized`, the normalized rows as well.
    public static func sharedResidualNormPromptProjections(
        _ x: MLXArray, delta: MLXArray, weight: MLXArray, eps: Float,
        projections: [Linear], normalized: Bool, firstProjectionLastRow: Bool = false, deferLast: Bool = false
    ) -> (residual: MLXArray, normalized: MLXArray?, projections: [MLXArray], deferred: ((MLXArray) -> MLXArray?)?)? {
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
        if deferLast, !firstProjectionLastRow,
            let split = splitPromptPair(activation, full, shape: Array(x.shape.dropLast()))
        {
            return (codes[3], normalized ? codes[4] : nil, [split.lead], split.deferred)
        }
        guard let outputs = PromptInt8Matmul.matmul(
            activation, layers: full.map(\.verifyOperands), streamedWeights: Self.streamedWeights(full))
        else { return nil }
        var projected = zip(full, outputs).map { layer, output in
            let output = output.reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
        if firstProjectionLastRow { projected.insert(first.promptLastRow(activation), at: 0) }
        return (codes[3], normalized ? codes[4] : nil, projected, nil)
    }

    /// `sharedPromptProjections(rmsNorm(x, weight), projections)` with the
    /// RMSNorm formed in the quantizing kernel; with `normalized` the
    /// normalized rows are returned as well (for other consumers of the
    /// layer input). Nil when the norm, the shapes, the packing or the route
    /// do not qualify; the caller then normalizes and projects separately.
    public static func sharedNormPromptProjections(
        _ x: MLXArray, weight: MLXArray, eps: Float, projections: [Linear], normalized: Bool, firstProjectionLastRow: Bool = false,
        deferLast: Bool = false
    ) -> (normalized: MLXArray?, projections: [MLXArray], deferred: ((MLXArray) -> MLXArray?)?)? {
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
        if deferLast, !firstProjectionLastRow,
            let split = splitPromptPair(activation, full, shape: Array(x.shape.dropLast()))
        {
            return (normalized ? codes[3] : nil, [split.lead], split.deferred)
        }
        guard let outputs = PromptInt8Matmul.matmul(
            activation, layers: full.map(\.verifyOperands), streamedWeights: Self.streamedWeights(full))
        else { return nil }
        var projected = zip(full, outputs).map { layer, output in
            let output = output.reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
        if firstProjectionLastRow { projected.insert(first.promptLastRow(activation), at: 0) }
        return (normalized ? codes[3] : nil, projected, nil)
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
            let activation = first.transform.quantizedForPrompt(x),
            let outputs = PromptInt8Matmul.matmul(
                activation, layers: packed.map(\.verifyOperands), streamedWeights: Self.streamedWeights(packed))
        else { return nil }
        return zip(packed, outputs).map { layer, output in
            let output = output.reshaped(Array(x.shape.dropLast()) + [layer.weight.dim(0)])
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
        /// `x + delta`, the residual stream entering the layer.
        public let residual: MLXArray
        /// `rmsNorm(x + delta, weight)`.
        public let normalized: MLXArray
    }

    /// `x`, its RMSNorm and the verify codes of the rotated norm in one kernel.
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

    /// `projections` applied to `boundary.normalized` from the boundary's
    /// codes. Nil when a projection's packing or rotation differs.
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

    /// `boundaryProjections(boundary, projections)` for two projections, plus
    /// `VerifyDenseBA.standalone` of the FP32 `x` (`[..., 4096]`, one row
    /// per boundary row) over the dense FP32 `wb` and `wa`, in one dispatch,
    /// each output bit for bit what the separate calls return. Nil when that
    /// dispatch does not qualify; the caller then makes the separate calls.
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

    /// The qkv/z pair of the boundary route in two launches (`VerifySplitZ`):
    /// `lead` now, with the dense b/a of `x` on the tensor route (`b`, `a`
    /// nil elsewhere), and `deferred(dependency)`, the z launch ordered after
    /// `dependency` without reading it. Each output is the record route's,
    /// bit for bit. Nil when the split did not qualify at load or the shapes
    /// do not take it; the caller then runs the record route.
    public static func boundarySplitProjections(
        _ boundary: VerifyBoundary, lead: Linear, deferred: Linear, x: MLXArray, wb: MLXArray, wa: MLXArray
    ) -> (lead: MLXArray, b: MLXArray?, a: MLXArray?, deferred: (MLXArray) -> MLXArray?)? {
        guard VerifySplitZ.ready, let first = lead as? HadamardQuantizedLinear,
            let second = deferred as? HadamardQuantizedLinear,
            first.readsFloat16ConstantsInPlace, first.gdnLayout == nil, first.bias == nil,
            first.transform.blockSize == 4096, first.weight.dim(0) % 32 == 0,
            first.transform.isIdentical(to: boundary.transform),
            second.readsFloat16ConstantsInPlace, second.gdnLayout == nil, second.bias == nil,
            second.transform.blockSize == 4096, second.weight.dim(0) % 32 == 0,
            second.transform.isIdentical(to: boundary.transform)
        else { return nil }
        let rows = boundary.activation.rows
        let streamed: [MLXArray]?
        if VerifyInt8Matmul.streamReady,
            let w0 = first.promptConstants.streamWeight(first.weight, k: first.transform.width, columns: 16),
            let w1 = second.promptConstants.streamWeight(second.weight, k: second.transform.width, columns: 16)
        {
            streamed = [w0, w1]
        } else {
            streamed = nil
        }
        let operands = (first.verifyOperands, second.verifyOperands)
        let pair = operands.0.negative && operands.1.negative
        let shared = VerifyInt8Matmul.pairSharesOffsetForm
        guard boundary.normalized.dim(-1) == 4096, boundary.normalized.size == rows * 4096,
            x.shape == boundary.normalized.shape,
            let out = VerifyInt8Matmul.splitLead(
                boundary.activation, layer: (operands.0.weight, operands.0.scalesT, operands.0.biasesT),
                negative: shared ? pair : operands.0.negative, x: x.reshaped([rows, 4096]), wb: wb, wa: wa,
                streamedWeight: streamed?[0])
        else { return nil }
        let shape = Array(boundary.normalized.shape.dropLast())
        let activation = boundary.activation
        return (
            out.out.reshaped(shape + [first.weight.dim(0)]),
            out.b?.reshaped(shape + [wb.dim(0)]), out.a?.reshaped(shape + [wb.dim(0)]),
            { dependency in
                VerifyInt8Matmul.splitDeferred(
                    VerifySplitZ.ordered(activation, after: dependency),
                    layer: (operands.1.weight, operands.1.scalesT, operands.1.biasesT),
                    negative: shared ? pair : operands.1.negative, streamedWeight: streamed?[1]
                )?.reshaped(shape + [second.weight.dim(0)])
            })
    }

    /// The query/key/value projections of the boundary route in two launches
    /// (`VerifySplitGate`): `query` holds the query half of each 512-wide
    /// head of the query projection, `key` and `value` are the record's, and
    /// `gate(dependency)` is the launch of the gate halves ordered after
    /// `dependency` without reading it. Each half the attention reads is the
    /// record route's, bit for bit; the other halves are unwritten. Nil when
    /// the split did not qualify at load or the shapes do not take it; the
    /// caller then runs the record route.
    public static func boundarySplitQueryGate(
        _ boundary: VerifyBoundary, query: Linear, key: Linear, value: Linear
    ) -> (query: MLXArray, key: MLXArray, value: MLXArray, gate: (MLXArray) -> MLXArray?)? {
        guard VerifySplitGate.ready,
            let q0 = query as? HadamardQuantizedLinear,
            let k0 = key as? HadamardQuantizedLinear,
            let v0 = value as? HadamardQuantizedLinear,
            q0.readsFloat16ConstantsInPlace, q0.gdnLayout == nil, q0.bias == nil,
            q0.transform.blockSize == 4096, q0.weight.dim(0) % 32 == 0,
            q0.transform.isIdentical(to: boundary.transform),
            k0.readsFloat16ConstantsInPlace, k0.gdnLayout == nil, k0.bias == nil,
            k0.transform.blockSize == 4096, k0.weight.dim(0) % 32 == 0,
            k0.transform.isIdentical(to: boundary.transform),
            v0.readsFloat16ConstantsInPlace, v0.gdnLayout == nil, v0.bias == nil,
            v0.transform.blockSize == 4096, v0.weight.dim(0) % 32 == 0,
            v0.transform.isIdentical(to: boundary.transform)
        else { return nil }
        let layers = [q0.verifyOperands, k0.verifyOperands, v0.verifyOperands]
        let streamed: [MLXArray]?
        if VerifyInt8Matmul.streamReady,
            let w0 = q0.promptConstants.streamWeight(q0.weight, k: q0.transform.width, columns: 16),
            let w1 = k0.promptConstants.streamWeight(k0.weight, k: k0.transform.width, columns: 16),
            let w2 = v0.promptConstants.streamWeight(v0.weight, k: v0.transform.width, columns: 16)
        {
            streamed = [w0, w1, w2]
        } else {
            streamed = nil
        }
        let activation = boundary.activation
        guard let lead = VerifyInt8Matmul.gateSplitLead(activation, layers: layers, streamedWeights: streamed)
        else { return nil }
        let shape = Array(boundary.normalized.shape.dropLast())
        let n0 = q0.weight.dim(0)
        return (
            lead[0].reshaped(shape + [n0]),
            lead[1].reshaped(shape + [k0.weight.dim(0)]),
            lead[2].reshaped(shape + [v0.weight.dim(0)]),
            { dependency in
                VerifyInt8Matmul.gateSplitGate(
                    VerifySplitZ.ordered(activation, after: dependency), layers: layers, streamedWeights: streamed
                )?.reshaped(shape + [n0])
            })
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

    /// A prompt projection pair over one activation in two launches
    /// (`PromptSplitZ`): `lead` now, and `deferred(dependency)`, the second
    /// launch ordered after `dependency` without reading it. Each output is
    /// the record route's, bit for bit. Nil when the split did not qualify at
    /// load or the pair's launch does not take it; the caller then runs the
    /// record route.
    private static func splitPromptPair(
        _ activation: PromptInt8Activation, _ pair: [HadamardQuantizedLinear], shape: [Int]
    ) -> (lead: MLXArray, deferred: (MLXArray) -> MLXArray?)? {
        guard PromptSplitZ.ready, pair.count == 2,
            let split = PromptInt8Matmul.splitPair(
                activation, layers: pair.map(\.verifyOperands), streamedWeights: Self.streamedWeights(pair),
                streamedQualified: PromptSplitZ.streamedReady)
        else { return nil }
        func shaped(_ layer: HadamardQuantizedLinear, _ output: MLXArray) -> MLXArray {
            let output = output.reshaped(shape + [layer.weight.dim(0)])
            return layer.bias.map { output + $0 } ?? output
        }
        return (shaped(pair[0], split.first), { dependency in
            split.second(VerifySplitZ.ordered(activation, after: dependency)).map { shaped(pair[1], $0) }
        })
    }

    private static func streamedWeights(_ layers: [HadamardQuantizedLinear]) -> [MLXArray]? {
        guard VerifyInt8Matmul.streamReady else { return nil }
        switch layers.count {
        case 2:
            let l0 = layers[0], l1 = layers[1]
            guard let w0 = l0.promptConstants.streamWeight(l0.weight, k: l0.transform.width, columns: 16),
                let w1 = l1.promptConstants.streamWeight(l1.weight, k: l1.transform.width, columns: 16)
            else { return nil }
            return [w0, w1]
        case 3:
            let l0 = layers[0], l1 = layers[1], l2 = layers[2]
            guard let w0 = l0.promptConstants.streamWeight(l0.weight, k: l0.transform.width, columns: 16),
                let w1 = l1.promptConstants.streamWeight(l1.weight, k: l1.transform.width, columns: 16),
                let w2 = l2.promptConstants.streamWeight(l2.weight, k: l2.transform.width, columns: 16)
            else { return nil }
            return [w0, w1, w2]
        default:
            return nil
        }
    }

    /// This layer's operands for `VerifyInt8Matmul.matmul(_:layers:)` and
    /// `PromptInt8Matmul.matmul(_:layers:)`.
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
            negative: promptConstants.negative, streamedWeight: streamedVerifyWeight)
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
            negative: promptConstants.negative, streamedWeight: streamedVerifyWeight)
        else { return nil }
        let shaped = output.reshaped(Array(x.shape.dropLast(2)) + [weight.dim(0)])
        return bias.map { shaped + $0 } ?? shaped
    }

    /// `callAfterGatedNorm` at prompt width (`PromptInt8Matmul` rows): the
    /// GDN output gate and norm formed in the quantizing rotation, then the
    /// prompt integer matmul, as `self(norm(x, gate))` computes it. Nil when
    /// the route, the packing or the shapes do not qualify.
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
        let streamed = promptConstants.streamWeight(weight, k: transform.width, columns: 16)
        guard let output = PromptInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative, streamedWeight: streamed)
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
            negative: promptConstants.negative, streamedWeight: streamedVerifyWeight)
        else { return nil }
        let shaped = output.reshaped([1, activation.rows, weight.dim(0)])
        return bias.map { shaped + $0 } ?? shaped
    }

    /// `callAfterAttentionGate` at prompt width (`PromptInt8Matmul` rows):
    /// `self(attention * sigmoid(gate))` with the gate formed in the
    /// quantizing rotation, as `[1, rows, N]`. Nil when the route, the
    /// packing or the shapes do not qualify.
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
        let streamed = promptConstants.streamWeight(weight, k: transform.width, columns: 16)
        guard let output = PromptInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
            negative: promptConstants.negative, streamedWeight: streamed)
        else { return nil }
        let shaped = output.reshaped([1, activation.rows, weight.dim(0)])
        return bias.map { shaped + $0 } ?? shaped
    }

    /// `callAfterAttentionGatePrompt` for a single last-query prefill row (`rows == 1`),
    /// fusing the head-major attention transpose, gate slice, stable sigmoid multiply,
    /// signed Hadamard rotation, and int8 quantization before `PromptInt8Matmul.matmulLastRow`.
    public func callAfterAttentionGateLastRow(
        _ attention: MLXArray, queryProjection: MLXArray
    ) -> MLXArray? {
        guard gdnLayout == nil, readsFloat16ConstantsInPlace, transform.blockSize == 4096,
            transform.width == 4096, weight.dim(0) % 32 == 0, PromptInt8Matmul.route != .off,
            let activation = transform.quantizedForPromptLastRowAttentionGate(
                attention, queryProjection: queryProjection)
        else { return nil }
        return promptLastRow(activation)
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

    private func needsHeadResidualL2(rows: Int) -> Bool {
        !ExactVerifyHead.checkEveryCall && exactHeadBounds.compactVerified != false
            && exactHeadBounds.compactParallelVerified != false && exactHeadBounds.residualL2Verified != false
            && weight.dim(0) % 512 == 0
            && VerifyInt8Matmul.compactHeadAvailable(rows: rows, k: transform.width, n: weight.dim(0))
    }

    /// `topTwoRows(rmsNorm(x + delta, weight))` with the residual add, RMSNorm,
    /// Hadamard rotation and int8 verify quantization fused into one kernel.
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
        func compact() -> (ids: MLXArray, values: MLXArray)? {
            guard rows == 16, !ExactVerifyHead.checkEveryCall else { return nil }
            let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
            guard let summary = VerifyInt8Matmul.compactHead(
                activation, weight: weight, scalesT: scalesT, biasesT: biasesT,
                negative: promptConstants.negative)
            else { return nil }
            let bounds = exactHeadBounds.get(weight: weight, scales: scales, biases: biases!)
            func parallel(tight: Bool = false, anchors: Bool = false, partitionL2: Bool = false) -> (ids: MLXArray, values: MLXArray) {
                ExactVerifyHead.runCompactParallel(
                    rotated: rotated, summary: summary, ascale: activation.ascale, bounds: bounds,
                    weight: weight, scales: scales, biases: biases!, rows: rows,
                    negative: promptConstants.negative, error2: tight ? error2 : nil, anchors: anchors, partitionL2: partitionL2)
            }
            let canUseParallel = summary.values.size / rows % (4 * ExactVerifyHead.compactBlocks) == 0
            if canUseParallel, exactHeadBounds.compactParallelVerified == true {
                return parallel(tight: exactHeadBounds.residualL2Verified == true,
                    anchors: exactHeadBounds.exactAnchorsVerified == true,
                    partitionL2: exactHeadBounds.partitionL2Verified == true)
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

// MARK: - Draft-step front row and screened token (single row)

extension SignedBlockHadamard {
    fileprivate var draftSigns: MLXArray { signs }
}

/// Single-row kernels around a draft head step (`Qwen35MTP` fused draft step).
/// Draft-only: they change draft numerics (FP32 throughout, different
/// summation order), never anything the target verifies.
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

    /// `[rmsNorm(embed(tok)) · ge | rmsNorm(hid) · gh]` for one 2-bit g128
    /// folded-WHT embedding row. One 256-thread threadgroup.
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

    /// Final RMSNorm of `z` (the hidden output) and the head's forward signed
    /// WHT of it (the screen input, `[1, 4096]`). One 256-thread threadgroup.
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

    /// One simdgroup per 32-row block: the block's best screened row (first
    /// maximum, as `argMax`), rescored on the full width (the arithmetic of
    /// `DraftScreenQMV.rescore`).
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

    /// First maximum over the rescored blocks; writes the winner's id and its
    /// row in the head.
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
    /// Draft-only front row of an MTP head step: `[rmsNorm(self(token)) ·
    /// embedNorm | rmsNorm(hidden) · hiddenNorm]` as one FP32 `[2 * 4096]`
    /// row, computed in FP32 (the module path rounds the looked-up row to the
    /// FP16 constants' dtype first). Nil when the packing or shapes do not
    /// take it.
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
    /// Draft-only screened token for the hidden `z` before its final RMSNorm:
    /// the normalized hidden (`[1, 1, 4096]`, FP32) and `ids[screenedArgMax]`
    /// (`[1]`, int32): the norm, the rotation, the screen, the rescore of
    /// each 32-row block's best row and the final pick in four kernels. Same
    /// rule as `screenedArgMax` with `chunk == 32`. Nil when the packing or
    /// the shapes do not take it. `row` is the winner's row in this head.
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
    /// Resolves the fast head gather once (drafter bind, before any request) and logs its state.
    public static func prepareDraftGather() { VerifyInt8Matmul.prepareGatherFast() }

    /// The leading ten 128-feature groups (1280 of 4096) used to screen every row.
    public static let draftScreenGroups = 10

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

    /// The draft head's 128 candidates per row with their exact values. The
    /// partial over `draftScreenGroups` groups picks the ids (`shortlist`:
    /// `[15, N]` to `[15, 128]` int32), then each id is rescored over every
    /// group, bit for bit `self(x)` at that id. Nil (the caller runs the full
    /// head) unless `qualifyDraftScreen` passed for the operand form in use
    /// and `x` is `[1, 15, 4096]` FP32 on the GPU.
    public func draftScreen(
        _ x: MLXArray, shortlist: (MLXArray) -> MLXArray
    ) -> (ids: MLXArray, values: MLXArray)? {
        guard let streamed = draftScreenState.streamed, let biases,
            let activation = draftScreenActivation(x)
        else { return nil }
        let copy: MLXArray? = streamedVerifyWeight
        guard (copy != nil) == streamed else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases)
        guard
            let partial = VerifyInt8Matmul.screen(
                activation, weight: copy ?? weight, scalesT: scalesT, biasesT: biasesT,
                streamed: streamed, groups: Self.draftScreenGroups)
        else { return nil }
        let ids: MLXArray = shortlist(partial)
        let values: MLXArray = VerifyInt8Matmul.gather(
            activation, weight: weight, scales: scales, biases: biases, ids: ids,
            groups: transform.width / 128, negative: true)
        return (ids, values)
    }

    /// Resolved once at bind, before any request, on a fixed FP32 15-row
    /// strided input: (i) the screen equals the ALU gather over the same
    /// groups on every value, all finite; (ii) `shortlist` picks the same ids
    /// from both; (iii) the full-width gather equals `self(x)` at those ids
    /// and at every 97th id and the last. Nil when it qualified, else why not.
    public func qualifyDraftScreen(shortlist: (MLXArray) -> MLXArray) -> String? {
        draftScreenState.streamed = nil
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
        DraftHeadScreenState.failure = nil
        var reason: String? = nil
        withErrorHandler({ message in DraftHeadScreenState.failure = String(message.prefix(160)) }) {
            reason = draftScreenMismatch(x, biases: biases, streamed: streamed, shortlist: shortlist)
        }
        if let failure = DraftHeadScreenState.failure { return "error: \(failure)" }
        if reason == nil { draftScreenState.streamed = streamed }
        return reason
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
