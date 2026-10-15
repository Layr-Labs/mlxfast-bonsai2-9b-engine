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

    /// CPU-only qualification for sharing one computed activation transform.
    /// Signs are compared exactly; matching widths alone is not sufficient.
    public func hasSameTransform(as other: SignedBlockHadamard) -> Bool {
        blockSize == other.blockSize && width == other.width
            && signValues == other.signValues
    }

    /// Transform activations before multiplication by folded weights.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        validate(x)
        return hadamardTransform((x.asType(.float32) * signs).reshaped([-1, blockSize]))
            .reshaped(x.shape).asType(x.dtype)
    }

    /// Recover the original basis after looking up folded embedding rows.
    public func inverse(_ x: MLXArray) -> MLXArray {
        validate(x)
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
        inputNames: ["x", "w", "scales", "biases", "K", "N"], outputNames: ["y"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const int in_w = K / 4;
            const int in_g = K / 128;
            const int out_row = tid.y * 8 + simd_gid * 4;
            const device uint8_t* ws = (const device uint8_t*)w + out_row * in_w + simd_lid * 4;
            const device half* sc = scales + out_row * in_g + simd_lid / 8;
            const device half* bi = biases + out_row * in_g + simd_lid / 8;
            const device float* xp = x + simd_lid * 16;
            float x_thread[16];
            float result[4] = {0, 0, 0, 0};
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              for (int i = 0; i < 16; i += 4) {
                sum += float(xp[i]) + float(xp[i + 1]) + float(xp[i + 2]) + float(xp[i + 3]);
                x_thread[i] = xp[i];
                x_thread[i + 1] = xp[i + 1] / 4.0f;
                x_thread[i + 2] = xp[i + 2] / 16.0f;
                x_thread[i + 3] = xp[i + 3] / 64.0f;
              }
              for (int r = 0; r < 4; r++) {
                const device uint8_t* wl = ws + r * in_w;
                const float s = float(sc[r * in_g]);
                const float b = float(bi[r * in_g]);
                float accum = 0;
                for (int i = 0; i < 4; i++) {
                  accum += (x_thread[4 * i] * (wl[i] & 0x03) + x_thread[4 * i + 1] * (wl[i] & 0x0c)
                      + x_thread[4 * i + 2] * (wl[i] & 0x30) + x_thread[4 * i + 3] * (wl[i] & 0xc0));
                }
                result[r] += s * accum + sum * b;
              }
              ws += 128;
              sc += 4;
              bi += 4;
              xp += 512;
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
        inputNames: ["x", "w", "scales", "biases", "K", "N"], outputNames: ["y"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
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
        let inputs = [x.reshaped([rows, K]), weight, scales, biases,
                      MLXArray(Int32(K)), MLXArray(Int32(N))]
        let outputs = rows == 1 ? 4 : outputsPerSimdgroup(rows: rows)
        let output = (rows == 1 ? fast : Self.rows)(
            inputs, template: rows == 1 ? [] : [("rows", rows), ("outputs", outputs)],
            grid: (32, N / (2 * outputs) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[rows, N]], outputDTypes: [.float32])[0]
        return output.reshaped(Array(x.shape.dropLast()) + [N])
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
        Self.reuseFloat16Constants && bits == 2 && groupSize == 128
            && transform.blockSize == 1024 && scales.dtype == .float16
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
    }

    /// Fixed at init: the packing `Float16ConstantQMV` reads in place.
    let readsFloat16ConstantsInPlace: Bool

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rotated = transform(gdnLayout.map { $0(x) } ?? x)
        return callWithTransformedInput(rotated)
    }

    /// Qualify only the input preprocessing; weights and output rows stay separate.
    /// The first experiment intentionally excludes GDN output-layout permutations.
    public func canShareInputTransform(with other: HadamardQuantizedLinear) -> Bool {
        bits == 2 && other.bits == bits && groupSize == 128
            && other.groupSize == groupSize && transform.blockSize == 4096
            && gdnLayout == nil && other.gdnLayout == nil
            && scales.dtype == .float16 && other.scales.dtype == scales.dtype
            && biases?.dtype == .float16 && other.biases?.dtype == biases?.dtype
            && transform.hasSameTransform(as: other.transform)
    }

    /// Apply this module's unchanged matmul after an exactly matching signed
    /// transform has already run. No concatenation or kernel selection change.
    public func callWithTransformedInput(_ rotated: MLXArray) -> MLXArray {
        precondition(rotated.ndim > 0 && rotated.dim(-1) == transform.width)
        if readsFloat16ConstantsInPlace, rotated.dtype == .float32,
            (1 ... Float16ConstantQMV.maximumRows).contains(rotated.size / rotated.dim(-1))
        {
            let output = Float16ConstantQMV.matmul(
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
