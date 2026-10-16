// Adapted from PrismML-Eng/mlx-swift 6d3a84de28225d1f5bc0a56f5c781596997242f9 (MIT).
// Preserve the published Bonsai pack's FP32 transform / original output dtype contract.
import Foundation
import MLX

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

    // Fused transform from Yukon 9727ebfe (9B PR #35); native MLX radix order.
    private static let fused4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_4096",
        inputNames: ["x", "signs"],
        outputNames: ["y"],
        source: """
            #pragma clang fp reassociate(off)
            const short i = short(thread_position_in_threadgroup.x);
            const uint block_idx = threadgroup_position_in_grid.y;
            const uint batch_idx = block_idx * 4096u;
            const uint sign_offset = (BLOCKS_PER_ROW == 1)
                ? 0u : (block_idx % uint(BLOCKS_PER_ROW)) * 4096u;

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
                float4 val = in4[v];
                if (!INVERSE) {
                  val *= s4[v];
                }
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
            """,
        header: """
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
            """)

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
    private static let quantized4096Kernel = MLXFast.metalKernel(
        name: "signed_block_hadamard_4096_q8",
        inputNames: ["x", "signs"],
        outputNames: ["codes", "ascale", "rsb"],
        source: """
            #pragma clang fp reassociate(off)
            const short i = short(thread_position_in_threadgroup.x);
            const uint block_idx = threadgroup_position_in_grid.y;
            const uint batch_idx = block_idx * 4096u;
            const uint sign_offset = (BLOCKS_PER_ROW == 1)
                ? 0u : (block_idx % uint(BLOCKS_PER_ROW)) * 4096u;

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
              float4 val = in4[w] * s4[w];
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
              const size_t qidx = size_t(row >> 6) * size_t(kg) * 64 + size_t(g) * 64
                  + size_t(((ml >> 4) & 1u) * 32u + (ml & 7u) * 4u + ((ml >> 5) & 1u) * 2u + ((ml >> 3) & 1u));
              ascale[qidx] = qs;
              rsb[qidx] = qs * part;
            }
            """,
        header: """
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
            """,
        ensureRowContiguous: true)

    /// `forward(x)` quantized for `PromptInt8Matmul`: int8 codes `[rows,
    /// width]` and the FP32 group constants `[rows * width / 128]`. Nil when
    /// the shape is not a prompt-width FP32 block-4096 input.
    func quantizedForPrompt(_ x: MLXArray) -> PromptInt8Activation? {
        validate(x)
        let rows = x.size / width
        guard blockSize == 4096, x.dtype == .float32, rows % 64 == 0, rows > 0,
            x.size <= Int(Int32.max), StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let groups = rows * (width / 128)
        let outputs = Self.quantized4096Kernel(
            [x.reshaped([rows, width]), signs],
            template: [("BLOCKS_PER_ROW", width / 4096)],
            grid: (256, rows * (width / 4096), 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, width], [groups], [groups]],
            outputDTypes: [.int8, .float32, .float32])
        return PromptInt8Activation(
            codes: outputs[0], ascale: outputs[1], rsb: outputs[2], rows: rows, k: width)
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
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += float(xv.x) + float(xv.y) + float(xv.z) + float(xv.w);
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y / 4.0f;
                x_thread[4 * i + 2] = xv.z / 16.0f;
                x_thread[4 * i + 3] = xv.w / 64.0f;
              }
              for (int r = 0; r < 4; r++) {
                const uint packed_w = *(const device uint*)(ws + r * in_w);
                const float s = float(sc[r * in_g]);
                const float b = float(bi[r * in_g]);
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
            const int K = x_shape[1];
            const int N = w_shape[0];
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int out_row = tid.y * (2 * outputs) + simd_gid * outputs;
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
                s4[r][1] = s / 4.0f;
                s4[r][2] = s / 16.0f;
                s4[r][3] = s / 64.0f;
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
                  const float4 xv = xp[v * (K / 4) + i];
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
            for (int v = 0; v < rows; v++) {
              for (int r = 0; r < outputs; r++) {
                const float total = simd_sum(result[v][r]);
                if (simd_lid == 0) y[v * N + out_row + r] = total;
              }
            }
            """)

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

    static func matmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray
    ) -> MLXArray {
        let K = x.dim(-1)
        let rows = x.size / K
        let N = weight.dim(0)
        // Shape metadata supplies K and N without allocating scalar tensor inputs.
        let inputs = [x.reshaped([rows, K]), weight, scales, biases]
        let outputs = rows == 1 ? 4 : outputsPerSimdgroup(rows: rows)
        let output = (rows == 1 ? fast : Self.rows)(
            inputs, template: rows == 1 ? [] : [("rows", rows), ("outputs", outputs)],
            grid: (32, N / (2 * outputs) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[rows, N]], outputDTypes: [.float32])[0]
        return output.reshaped(Array(x.shape.dropLast()) + [N])
    }
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
/// `naxMatchesPlain` confirms it on the device before `nax` is used.
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
            const float2 b2 = float2(*(const device half2*)(biasesT + nidx));
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

    private static let naxHeader = """
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

    private static let naxKernel = MLXFast.metalKernel(
        name: "bonsai_q8_prompt_matmul_nax",
        inputNames: ["xq", "w", "scalesT", "biasesT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: naxSource,
        header: naxHeader,
        ensureRowContiguous: true)

    private static func run(
        _ route: Route, codes: MLXArray, ascale: MLXArray, rsb: MLXArray, weight: MLXArray,
        scalesT: MLXArray, biasesT: MLXArray, m: Int, k: Int, n: Int
    ) -> MLXArray {
        let dims = MLXArray([Int32(k), Int32(m), Int32(n)])
        let inputs = [codes, weight, scalesT, biasesT, ascale, rsb, dims]
        if route == .nax {
            return naxKernel(
                inputs, grid: (n / 64 * 128, m / 64, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [.float32])[0]
        }
        return plainKernel(
            inputs, grid: (n / 32 * 128, m / 32, 1), threadGroup: (128, 1, 1),
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

    /// Compiles and runs the tensor kernel on a fixed, input-independent
    /// 128 x 128 x 512 problem (2 x 2 tiles, 4 groups) and compares it bit for bit with the plain
    /// kernel. False on a compile error (caught by a scoped MLX error
    /// handler) or on any differing output.
    private static func naxMatchesPlain() -> Bool {
        checkFailed = false
        let m = 128, n = 128, k = 512, kg = k / 128
        let codes = MLXArray((0 ..< m * k).map { Int8(truncatingIfNeeded: ($0 * 37 + 11) % 255 - 127) }, [m, k])
        let weight = MLXArray((0 ..< n * k / 16).map { UInt32(truncatingIfNeeded: UInt64($0 + 1) &* 2_654_435_761) }, [n, k / 16])
        let scalesT = MLXArray((0 ..< kg * n).map { Float16(Float(1 + $0 % 13) / 1024) }, [kg, n])
        let biasesT = MLXArray((0 ..< kg * n).map { Float16(-Float($0 % 7) / 512) }, [kg, n])
        let ascale = MLXArray((0 ..< m * kg).map { Float(1 + $0 % 11) / 256 }, [m * kg])
        let rsb = MLXArray((0 ..< m * kg).map { Float($0 % 17 - 8) / 64 }, [m * kg])
        var same = false
        withErrorHandler({ message in
            if !PromptInt8Matmul.checkFailed { PromptInt8Matmul.checkNote = String(message.prefix(160)) }
            PromptInt8Matmul.checkFailed = true
        }) {
            let plain = run(
                .plain, codes: codes, ascale: ascale, rsb: rsb, weight: weight,
                scalesT: scalesT, biasesT: biasesT, m: m, k: k, n: n)
            let nax = run(
                .nax, codes: codes, ascale: ascale, rsb: rsb, weight: weight,
                scalesT: scalesT, biasesT: biasesT, m: m, k: k, n: n)
            let equal = MLX.all(plain .== nax)
            eval(equal)
            if !checkFailed { same = equal.item(Bool.self) }
        }
        if !checkFailed { checkNote = same ? "bit-identical" : "outputs differ" }
        return !checkFailed && same
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
            chosen = naxMatchesPlain() ? .nax : .off
        } else {
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

    static func matmul(
        _ activation: PromptInt8Activation, weight: MLXArray, scalesT: MLXArray, biasesT: MLXArray
    ) -> MLXArray? {
        guard route != .off else { return nil }
        return run(
            route, codes: activation.codes, ascale: activation.ascale, rsb: activation.rsb,
            weight: weight, scalesT: scalesT, biasesT: biasesT,
            m: activation.rows, k: activation.k, n: weight.dim(0))
    }
}

/// Group-major copies of a packed module's F16 scales and offsets for
/// `PromptInt8Matmul`, made on first use. A plain class, so nothing enters the
/// module parameter tree.
final class PromptInt8Constants {
    private var transposed: (MLXArray, MLXArray)?

    func get(scales: MLXArray, biases: MLXArray) -> (MLXArray, MLXArray) {
        if let transposed { return transposed }
        let pair = (scales.transposed(1, 0).contiguous(), biases.transposed(1, 0).contiguous())
        transposed = pair
        return pair
    }
}

/// The packed 2-bit, group-128 matmul for a 4 to 7 row MTP verify, flat in
/// rows. Each lane turns two weight codes into exact halves inside a
/// simdgroup matrix fragment, and one 8x8x8 multiply covers 8 output rows
/// against every verify row, so a weight is dequantized once for all rows
/// and no activation is reloaded per output row. Scales and biases apply
/// once per group to the FP32 partial sums, the bias through the group's
/// activation sum. The simdgroups of a threadgroup split K and add their
/// partials through threadgroup memory in a fixed order, so the split costs
/// no second launch. A row's result does not depend on the other rows.
enum FewRowPackedMatmul {
    static let rows = 4 ... 7
    static let rowBlocks = 4
    static let occupancy = 2048

    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_FEW_ROW_VERIFY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let warmed = NSLock()
    nonisolated(unsafe) private static var warmedShapes = Set<[Int]>()

    static func supports(k: Int, n: Int) -> Bool {
        enabled && k % 128 == 0 && n % (8 * rowBlocks) == 0
    }

    /// Simdgroups per threadgroup: the fewest that reach `occupancy`
    /// simdgroups in the grid, each taking whole groups of K.
    static func split(k: Int, n: Int) -> Int {
        let groups = k / 128
        let tiles = n / (8 * rowBlocks)
        let options = [1, 2, 4, 8, 16].filter { groups % $0 == 0 }
        return options.first { tiles * $0 >= occupancy } ?? options.last!
    }

    /// Compiles the kernel for one weight shape before the first verify, so
    /// no speculative round pays a Metal library build.
    static func warm(k: Int, n: Int) {
        guard supports(k: k, n: n) else { return }
        warmed.lock()
        let fresh = warmedShapes.insert([k, n]).inserted
        warmed.unlock()
        guard fresh else { return }
        let y = matmul(
            MLXArray.zeros([rows.lowerBound, k], dtype: .float32),
            weight: MLXArray.zeros([n, k / 16], dtype: .uint32),
            scales: MLXArray.zeros([n, k / 128], dtype: .float16),
            biases: MLXArray.zeros([n, k / 128], dtype: .float16))
        eval(y)
    }

    static func matmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray
    ) -> MLXArray {
        let K = x.dim(-1)
        let m = x.size / K
        let N = weight.dim(0)
        let S = split(k: K, n: N)
        let y = kernel(
            [x.reshaped([m, K]), weight, scales, biases],
            template: [("K", K), ("N", N), ("RB", rowBlocks), ("S", S)],
            grid: (N / (8 * rowBlocks) * 32 * S, 1, 1), threadGroup: (32 * S, 1, 1),
            outputShapes: [[m, N]], outputDTypes: [.float32])[0]
        return y.reshaped(Array(x.shape.dropLast()) + [N])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "hadamard_few_row_flat_f16_constants",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            constexpr uint CH = 32;
            constexpr uint TK = 128;
            constexpr uint row_bytes = K / 4;
            constexpr uint G = K / 128;
            constexpr uint KS = K / S;
            const uint M = x_shape[0];
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint qid = lane / 4;
            const uint fm = (qid & 4) + ((lane / 2) % 4);
            const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
            const uint row_base = threadgroup_position_in_grid.x * 8 * RB;
            threadgroup float red[S * RB * 64];
            simdgroup_float8x8 acc[RB];
            simdgroup_float8x8 part[RB];
            UNROLL for (uint b = 0; b < RB; ++b) {
              acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
              part[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            const device uchar* wb = (const device uchar*)w;
            const uint kl = (fn / 2) * CH;
            const uint kb = (fm / 2) * CH + (fm % 2) * 8;
            const device float* x0 = x + min(fn, M - 1) * K;
            const device float* x1 = x + min(fn + 1, M - 1) * K;
            for (uint k0 = sg * KS; k0 < (sg + 1) * KS; k0 += TK) {
              uint words[RB][2];
              UNROLL for (uint b = 0; b < RB; ++b) {
                const device uint* p =
                    (const device uint*)(wb + (row_base + b * 8 + fm) * row_bytes + (k0 + kl) / 4);
                words[b][0] = p[0];
                words[b][1] = p[1];
              }
              float sum0 = 0.0f, sum1 = 0.0f;
              UNROLL for (uint j = 0; j < CH / 2; ++j) {
                const uint k = k0 + kb + (j % 8) + 16 * (j / 8);
                const float v0 = x0[k];
                const float v1 = x1[k];
                sum0 += v0;
                sum1 += v1;
                simdgroup_float8x8 bx;
                bx.thread_elements()[0] = v0;
                bx.thread_elements()[1] = v1;
                UNROLL for (uint b = 0; b < RB; ++b) {
                  const uint bitsv = (words[b][j / 8] >> (2 * (j % 8))) & 0x00030003u;
                  const half2 q = as_type<half2>(bitsv | 0x64006400u) - half2(1024.0h);
                  simdgroup_half8x8 a;
                  a.thread_elements()[0] = q.x;
                  a.thread_elements()[1] = q.y;
                  simdgroup_multiply_accumulate(part[b], a, bx, part[b]);
                }
              }
              float s0 = sum0 + simd_shuffle_xor(sum0, 2);
              s0 += simd_shuffle_xor(s0, 4);
              s0 += simd_shuffle_xor(s0, 16);
              float s1 = sum1 + simd_shuffle_xor(sum1, 2);
              s1 += simd_shuffle_xor(s1, 4);
              s1 += simd_shuffle_xor(s1, 16);
              const uint g = k0 / TK;
              UNROLL for (uint b = 0; b < RB; ++b) {
                const uint gi = (row_base + b * 8 + fm) * G + g;
                const float sc = float(scales[gi]);
                const float bi = float(biases[gi]);
                acc[b].thread_elements()[0] += sc * part[b].thread_elements()[0] + bi * s0;
                acc[b].thread_elements()[1] += sc * part[b].thread_elements()[1] + bi * s1;
                part[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
              }
            }
            UNROLL for (uint b = 0; b < RB; ++b) {
              red[(sg * RB + b) * 64 + fm * 8 + fn] = acc[b].thread_elements()[0];
              red[(sg * RB + b) * 64 + fm * 8 + fn + 1] = acc[b].thread_elements()[1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = sg * 32 + lane; e < RB * 64; e += S * 32) {
              const uint r = e % 8;
              if (r < M) {
                float total = red[e];
                for (uint s = 1; s < S; ++s) total += red[s * RB * 64 + e];
                y[r * N + row_base + e / 8] = total;
              }
            }
            """,
        header: """
            #include <metal_simdgroup_matrix>
            #define UNROLL _Pragma("clang loop unroll(full)")

            """)
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
        self.readsFloat16ConstantsInPlace =
            bits == 2 && groupSize == 128 && scales.dtype == .float16
            && biases?.dtype == .float16 && transform.width % 512 == 0
            && weight.dim(0) % 8 == 0
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
        freeze()
        if readsFloat16ConstantsInPlace && transform.blockSize == 4096 {
            _ = PromptInt8Matmul.route
        }
        if readsFloat16ConstantsInPlace {
            FewRowPackedMatmul.warm(k: weight.dim(1) * 16, n: weight.dim(0))
        }
    }

    /// Fixed at init: the packing `Float16ConstantQMV` reads in place.
    let readsFloat16ConstantsInPlace: Bool

    private let promptConstants = PromptInt8Constants()

    /// The prompt-width integer route (`PromptInt8Matmul`); nil when the
    /// input, the packing or the device does not take it.
    private func promptInt8Forward(_ x: MLXArray) -> MLXArray? {
        guard readsFloat16ConstantsInPlace, transform.blockSize == 4096, x.dtype == .float32,
            x.ndim >= 2, PromptInt8Matmul.applies(rows: x.size / x.dim(-1)),
            weight.dim(0) % 64 == 0, transform.width % 512 == 0, PromptInt8Matmul.route != .off,
            let activation = transform.quantizedForPrompt(gdnLayout.map { $0(x) } ?? x)
        else { return nil }
        let (scalesT, biasesT) = promptConstants.get(scales: scales, biases: biases!)
        guard let output = PromptInt8Matmul.matmul(
            activation, weight: weight, scalesT: scalesT, biasesT: biasesT)
        else { return nil }
        return output.reshaped(Array(x.shape.dropLast()) + [weight.dim(0)])
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let output = promptInt8Forward(x) {
            return bias.map { output + $0 } ?? output
        }
        let rotated = transform(gdnLayout.map { $0(x) } ?? x)
        if readsFloat16ConstantsInPlace, rotated.dtype == .float32,
            (1 ... Float16ConstantQMV.maximumRows).contains(rotated.size / rotated.dim(-1))
        {
            let rows = rotated.size / rotated.dim(-1)
            let output =
                FewRowPackedMatmul.rows.contains(rows)
                    && FewRowPackedMatmul.supports(k: rotated.dim(-1), n: weight.dim(0))
                ? FewRowPackedMatmul.matmul(
                    rotated, weight: weight, scales: scales, biases: biases!)
                : Float16ConstantQMV.matmul(
                    rotated, weight: weight, scales: scales, biases: biases!)
            return bias.map { output + $0 } ?? output
        }
        if permitsFloat16ConstantReuse && rotated.dtype == .float32 {
            return constantCachedForward(rotated, allowFloat16: true)
        }
        return super.callAsFunction(rotated)
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
