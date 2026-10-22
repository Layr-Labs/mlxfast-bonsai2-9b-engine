// Copyright © 2026 Eigen Labs.
import Dispatch
import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Text factory for Prism's folded Qwen 3.5-family packs: the Ternary Bonsai 2
/// 27B pack (Hadamard block 1,024) and the Ternary Bonsai 2 9B pack (block 4,096).
///
/// The pack declares a vision tower. This track never serves an image, so the
/// load filter drops the 333 `vision_tower.*` tensors and the tower is never
/// built. `Qwen35Runner` declares `multimodal: false` for the same reason, so
/// the LLM factory is the only factory that resolves this model type.
///
/// The pack's tensors carry no MTP head (`mtp_num_hidden_layers: 0`). The track
/// attaches a separate head instead (the 9B pack ships it in `mtp/`), so the speculative capability
/// stays as `Qwen35Model` declares it and the runner decides whether a head is
/// present. See `docs/bonsai2.md`.
public final class PrismHadamardQwen35TextModel: Qwen35Model, PrismHadamardLoading,
    WeightNameFiltering
{
    public let prismCheckpoint: PrismHadamardCheckpointConfiguration

    public init(configurationData: Data) throws {
        _ = VerifyK64.install
        prismCheckpoint = try JSONDecoder().decode(PrismHadamardCheckpointConfiguration.self, from: configurationData)
        _ = M5K128Product.install
        super.init(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData))
    }
    public func shouldLoadWeight(named name: String) -> Bool {
        !name.hasPrefix("vision_tower.")
    }
}

enum M5K128Product {
    private static let m5: Bool = {
        let arch = GPU.deviceInfo().architecture
        return arch.hasPrefix("applegpu_g17") && !arch.hasPrefix("applegpu_g17p")
    }()

    static let install: Void = {
        guard m5 else { return }
        HadamardQuantizedLinear.verifyTile = { M5Int4Tile.make($0) }
        HadamardQuantizedLinear.verifyHeadRescore = M5HeadGroupedRescore.make
    }()
}

enum M5Int4Tile {
    static func make(_ source: String) -> String {
        let start = source.range(of: "constexpr auto desc =")!
        let end = source.range(of: "threadgroup_barrier(mem_flags::mem_threadgroup);", options: .backwards)!
        let alignment = source.contains("alignas(32) threadgroup uint32_t bs") ? 32 : 4
        var old = String(source[start.lowerBound ..< end.lowerBound])
        old = old.replacingOccurrences(of: "alignas(32) ", with: "")
            .replacingOccurrences(of: "threadgroup uint32_t bs[4][32 * 32];", with: "")
            .replacingOccurrences(of: "const bool good = simd_all(ok);", with: "good = simd_all(ok);")
        let streamed = source.contains("w + (size_t)bx * (2) * Kg * 128")
        let pointer = streamed ? "const device uint32_t* wcol = w + (size_t)bx * (2) * Kg * 128;"
                : "const device uint32_t* wcol = w + (size_t)(n0 + c) * KW + j;"
        let load = streamed ? "for (int t = 0; t < 8; t += 4) { const uint4 u = *((const device uint4*)(wcol + (t / 4) * Kg * 128 + gl * 128 + lane * 4)); wv[t] = u.x; wv[t + 1] = u.y; wv[t + 2] = u.z; wv[t + 3] = u.w; }"
                : "for (int t = 0; t < 8; t++) { wv[t] = wcol[(size_t)(4 * t) * KW + gl * 8]; }"
        let address = streamed
            ? "const int col = int(lane >> 1) + 16 * (t / 4);\nconst int word = int(lane & 1u) * 4 + t % 4;"
            : "const int col = c + 4 * t;\nconst int word = j;"
        let metadata = "const half s = scalesT[(size_t)g * N + n0 + col];\nconst half b = NEG ? -s : biasesT[(size_t)g * N + n0 + col];"
        let body = """
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
        tensor<threadgroup int4b_format, extents<int, 128, 32>, tensor_inline> B((threadgroup uchar*)bs[sg], extents<int, 128, 32>{}, array<int, 2>{1, 256});
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        auto accT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        const int cap = int(cT.get_capacity());
        #pragma clang loop unroll(full)
        for (int i = 0; i < cap; i++) {
          if (cT.is_valid_element(ushort(i))) accT[i] = as_type<int32_t>(0.0f);
        }
        const int j = int(lane & 7u);
        const int c = int(lane >> 3);
        \(pointer)
        uint32_t wv[8];
        {
          const int gl = int(sg);
          \(load)
        }
        for (int g = int(sg); g < Kg; g += 4) {
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) {
            const uint32_t v = wv[t];
            \(address)
            uint2 packed = uint2(0u);
            #pragma clang loop unroll(full)
            for (int u = 0; u < 16; u++) packed[u >> 3] |= ((v >> (2 * (u / 4) + 8 * (u % 4))) & 3u) << (4 * (u & 7));
            *(threadgroup uint2*)((threadgroup uchar*)bs[sg] + col * 128 + word * 8) = packed;
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          auto tA = A.template slice<128, 16>(g * 128, 0);
          op.run(tA, B, cT);
          #pragma clang loop unroll(full)
          for (int i = 0; i < cap; i++) {
            if (cT.is_valid_element(ushort(i))) {
              const auto ids = cT.get_multidimensional_index(i);
              const int col = int(ids[0]);
              const int row = int(ids[1]);
              \(metadata)
              const float t = float(s) * float(cT[i]);
              const float a = ascale[row * Kg + g], rb = rsb[row * Kg + g];
              accT[i] = as_type<int32_t>(fma(float(b), rb, fma(a, t, as_type<float>(accT[i]))));
            }
          }
          if (g + 4 < Kg) {
            const int gl = g + 4;
            \(load)
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        tensor<threadgroup int32_t, extents<int, 32, 16>, tensor_inline> R((threadgroup int32_t*)bs[sg], extents<int, 32, 16>{});
        accT.store(R);
        """
        return String(source[..<start.lowerBound]) + "alignas(ROWS == 16 ? 128 : \(alignment)) threadgroup uint32_t bs[4][32 * 32];\nbool good = true;\nif constexpr (ROWS == 16) {\n" + body + "\n} else {\n" + old + "\n}\n" + String(source[end.lowerBound...])
    }
}

/// Current one-row float head law with32 vocabulary outputs per group.
enum M5R1Head {
    private static let m5: Bool = {
        let arch = GPU.deviceInfo().architecture
        return arch.hasPrefix("applegpu_g17") && !arch.hasPrefix("applegpu_g17p")
    }()

    static func accepts(_ head: HadamardQuantizedLinear) -> Bool {
        m5 && StreamOrDevice.default == StreamOrDevice.gpu
            && head.bits == 2 && head.groupSize == 128 && head.bias == nil && head.gdnLayout == nil
            && head.transform.width == 4096 && head.transform.blockSize == 4096
            && head.transform.signs.dtype == .float32 && head.transform.signs.shape == [4096]
            && head.weight.dtype == .uint32 && head.weight.shape == [248320, 256]
            && head.scales.dtype == .float16 && head.scales.shape == [248320, 32]
            && head.biases?.dtype == .float16 && head.biases?.shape == [248320, 32]
    }

    static func run(_ head: HadamardQuantizedLinear, _ normalized: MLXArray)
        -> (ids: MLXArray, values: MLXArray)
    {
        let partials = producer(
            [head.transform(normalized), head.weight, head.scales, head.biases!],
            template: [("NEG", head.neg ? 1 : 0)],
            grid: (32, 15520, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, 7760, 2], [1, 7760, 2]], outputDTypes: [.int32, .float32])
        let top = Float16ConstantQMV.topTwoFinalize(
            partials, grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 2], [1, 2]], outputDTypes: [.int32, .float32])
        return (top[0], top[1])
    }

    private static let producer = MLXFast.metalKernel(
        name: "bonsai_head_r1_fast32_top2",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["partial_ids", "partial_values"],
        source: #"""
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const uint3 tid = threadgroup_position_in_grid;
        const int K = x_shape[x_ndim - 1];
        const int in_w = K / 4;
        const int in_g = K / 128;
        const int out_row = tid.y * 32 + simd_gid * 4;
        const device uint8_t* ws = (const device uint8_t*)w + out_row * in_w + simd_lid * 4;
        const device half* sc = scales + out_row * in_g + simd_lid / 8;
        const device half* bi = biases + out_row * in_g + simd_lid / 8;
        const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
        float x_thread[16];
        float r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0;
        #define HEAD_ONE(RESULT, COL) { \
          const uint packed_w = *(const device uint*)(ws + (COL) * in_w); \
          const float s = float(sc[(COL) * in_g]); \
          const float b = NEG ? -s : float(bi[(COL) * in_g]); \
          float accum = 0; \
          for (int i = 0; i < 4; i++) { \
            const uint8_t w_byte = (packed_w >> (8 * i)) & 0xFF; \
            accum += (x_thread[4 * i] * (w_byte & 0x03) + x_thread[4 * i + 1] * (w_byte & 0x0c) \
                + x_thread[4 * i + 2] * (w_byte & 0x30) + x_thread[4 * i + 3] * (w_byte & 0xc0)); \
          } \
          RESULT += s * accum + sum * b; \
        }
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
          HEAD_ONE(r0, 0)
          HEAD_ONE(r1, 1)
          HEAD_ONE(r2, 2)
          HEAD_ONE(r3, 3)
          HEAD_ONE(r4, 8)
          HEAD_ONE(r5, 9)
          HEAD_ONE(r6, 10)
          HEAD_ONE(r7, 11)
          HEAD_ONE(r8, 16)
          HEAD_ONE(r9, 17)
          HEAD_ONE(r10, 18)
          HEAD_ONE(r11, 19)
          HEAD_ONE(r12, 24)
          HEAD_ONE(r13, 25)
          HEAD_ONE(r14, 26)
          HEAD_ONE(r15, 27)
          ws += 128;
          sc += 4;
          bi += 4;
          xp4 += 128;
        }
        #undef HEAD_ONE
        hadamard_top2 best = hadamard_top2_empty();
        const float total0 = simd_sum(r0);
        if (simd_lid == 0) hadamard_top2_insert(best, total0, uint(out_row + 0));
        const float total1 = simd_sum(r1);
        if (simd_lid == 0) hadamard_top2_insert(best, total1, uint(out_row + 1));
        const float total2 = simd_sum(r2);
        if (simd_lid == 0) hadamard_top2_insert(best, total2, uint(out_row + 2));
        const float total3 = simd_sum(r3);
        if (simd_lid == 0) hadamard_top2_insert(best, total3, uint(out_row + 3));
        const float total4 = simd_sum(r4);
        if (simd_lid == 0) hadamard_top2_insert(best, total4, uint(out_row + 8));
        const float total5 = simd_sum(r5);
        if (simd_lid == 0) hadamard_top2_insert(best, total5, uint(out_row + 9));
        const float total6 = simd_sum(r6);
        if (simd_lid == 0) hadamard_top2_insert(best, total6, uint(out_row + 10));
        const float total7 = simd_sum(r7);
        if (simd_lid == 0) hadamard_top2_insert(best, total7, uint(out_row + 11));
        const float total8 = simd_sum(r8);
        if (simd_lid == 0) hadamard_top2_insert(best, total8, uint(out_row + 16));
        const float total9 = simd_sum(r9);
        if (simd_lid == 0) hadamard_top2_insert(best, total9, uint(out_row + 17));
        const float total10 = simd_sum(r10);
        if (simd_lid == 0) hadamard_top2_insert(best, total10, uint(out_row + 18));
        const float total11 = simd_sum(r11);
        if (simd_lid == 0) hadamard_top2_insert(best, total11, uint(out_row + 19));
        const float total12 = simd_sum(r12);
        if (simd_lid == 0) hadamard_top2_insert(best, total12, uint(out_row + 24));
        const float total13 = simd_sum(r13);
        if (simd_lid == 0) hadamard_top2_insert(best, total13, uint(out_row + 25));
        const float total14 = simd_sum(r14);
        if (simd_lid == 0) hadamard_top2_insert(best, total14, uint(out_row + 26));
        const float total15 = simd_sum(r15);
        if (simd_lid == 0) hadamard_top2_insert(best, total15, uint(out_row + 27));
        threadgroup hadamard_top2 scratch;
        if (simd_gid == 1 && simd_lid == 0) scratch = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid == 0 && simd_lid == 0) {
          hadamard_top2_merge(best, scratch);
          const uint base = tid.y * 2;
          partial_ids[base] = int(best.first_id);
          partial_ids[base + 1] = int(best.second_id);
          partial_values[base] = best.first_value;
          partial_values[base + 1] = best.second_value;
        }
        """#,
        header: Float16ConstantQMV.topTwoHeader)
}

enum M5HeadGroupedRescore {
    static func make(_ original: String, _ header: String) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "bonsai_verify_head_partition_rescore_group4_v1",
            inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2"],
            outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
            source: source(original), header: header)
    }

    static func source(_ original: String) -> String {
        var text = original
        func replace(_ old: String, _ new: String, count: Int = 1) {
            precondition(text.components(separatedBy: old).count == count + 1)
            text = text.replacingOccurrences(of: old, with: new)
        }
        replace("const uint block = threadgroup_position_in_grid.x;", """
            const uint partition = thread_position_in_threadgroup.x / uint(THREADS);
            const uint block = threadgroup_position_in_grid.x * 4u + partition;
            """)
        replace("const uint lane = thread_position_in_threadgroup.x;",
            "const uint lane = thread_position_in_threadgroup.x % uint(THREADS);")
        replace("const uint simd_gid = simdgroup_index_in_threadgroup;",
            "const uint simd_gid = simdgroup_index_in_threadgroup % uint(THREADS / 32);")
        replace("""
            threadgroup atomic_int cand_count; threadgroup atomic_int omitted_overflow;
            threadgroup int cand_ids[CAP];
            threadgroup float cand_values[CAP];
            threadgroup float shared_tau[2];
            threadgroup int shared_bad;
            """, """
            threadgroup atomic_int counts4[4], overflows4[4];
            threadgroup int ids4[4 * CAP], bads4[4];
            threadgroup float values4[4 * CAP], taus4[8];
            threadgroup atomic_int* cand_count = counts4 + partition;
            threadgroup atomic_int* omitted_overflow = overflows4 + partition;
            threadgroup int* cand_ids = ids4 + partition * uint(CAP);
            threadgroup float* cand_values = values4 + partition * uint(CAP);
            threadgroup float* shared_tau = taus4 + partition * 2u;
            threadgroup int* shared_bad = bads4 + partition;
            """)
        replace("&cand_count", "cand_count", count: 3)
        replace("&omitted_overflow", "omitted_overflow", count: 3)
        replace("if (simd_gid == 0) {", "if (simdgroup_index_in_threadgroup == 0) {")
        replace("""
              if (simd_lid == 0) {
                float e = 0.0f;
            """, """
              const float leader_second = simd_shuffle(local.second_value, ushort(0));
              const uint leader_count = simd_shuffle(local.count, ushort(0));
              if (simd_lid < 4) {
                const uint cutoff_block = threadgroup_position_in_grid.x * 4u + simd_lid;
                float e = 0.0f;
            """)
        replace("if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);",
            "float local_e = e; if (USE_L2) e = head_l2_partition_error(e, ascale, bounds, error2, row, in_g, cutoff_block, local_e);")
        replace("const float base_t = local.second_value - 2.0f * e;",
            "const float base_t = leader_second - 2.0f * e;")
        replace("const float anchor_t = anchor_floor - e;", "const float anchor_t = anchor_floor - local_e;")
        replace("shared_tau[0] =", "taus4[simd_lid * 2u] =")
        replace("shared_tau[1] =", "taus4[simd_lid * 2u + 1u] =")
        replace("shared_bad = (pbad != 0 || local.count < 2",
            "bads4[simd_lid] = (pbad != 0 || leader_count < 2")
        replace("= shared_bad |", "= shared_bad[0] |")
        return text
    }
}

/// Fixed loaded9B M5 one-row GDN input norm and shared rotation.
enum M5R1GDNInput {
    private static let m5: Bool = {
        let arch = GPU.deviceInfo().architecture
        return arch.hasPrefix("applegpu_g17") && !arch.hasPrefix("applegpu_g17p")
    }()

    static func apply(_ x: MLXArray, weight: MLXArray, eps: Float, qkv: Linear, z: Linear)
        -> (normalized: MLXArray, projections: [MLXArray])?
    {
        guard m5, StreamOrDevice.default == StreamOrDevice.gpu,
            x.dtype == .float32, x.shape == [1, 1, 4096],
            weight.dtype == .float32, weight.shape == [4096],
            let first = qkv as? HadamardQuantizedLinear, let second = z as? HadamardQuantizedLinear,
            first.weight.shape == [8192, 256], second.weight.shape == [4096, 256]
        else { return nil }
        let projections: [HadamardQuantizedLinear] = [first, second]
        for projection in projections {
            guard projection.bits == 2, projection.groupSize == 128,
                projection.bias == nil, projection.gdnLayout == nil,
                projection.weight.dtype == .uint32
            else { return nil }
            guard projection.transform.width == 4096, projection.transform.blockSize == 4096,
                projection.transform.signs.dtype == .float32,
                projection.transform.signs.shape == [4096]
            else { return nil }
            let metadataShape: [Int] = [projection.weight.dim(0), 32]
            guard projection.scales.dtype == .float16, projection.scales.shape == metadataShape,
                let offsets = projection.biases, offsets.dtype == .float16,
                offsets.shape == metadataShape
            else { return nil }
        }
        guard first.transform.isIdentical(to: second.transform) else { return nil }
        let output = prepare([x, weight, HostConstants.float32([eps]), first.transform.signs],
            template: [("T", DType.float32), ("BLOCKS_PER_ROW", 1), ("INVERSE", false)],
            grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [x.shape, x.shape], outputDTypes: [.float32, .float32])
        return (output[1], [first.forwardRotated(output[0]), second.forwardRotated(output[0])])
    }

    private static let prepare: MLXFast.MLXFastKernel = {
        var source = SignedBlockHadamard.fused4096Source(residualNorm: true)
        let oldNorm = """
        const float4 value = *((const device float4*)(x + at))
              + *((const device float4*)(residual + at));
        """
        let newNorm = "const float4 value = *((const device float4*)(x + at));"
        let oldRead = """
        const float4 raw = in4[v] + *((const device float4*)(residual + in_offset) + v);
        *((device float4*)(residual_out + in_offset) + v) = raw;
        const float4 weight = *((const device float4*)(norm_weight + s_offset) + v);
        float4 val = (weight * (raw * norm_inv)) * s4[v];
        """
        let newRead = """
        const float4 raw = in4[v];
        const float4 weight = *((const device float4*)(norm_weight + s_offset) + v);
        const float4 normed = weight * (raw * norm_inv);
        *((device float4*)(normed_out + in_offset) + v) = normed;
        float4 val = normed * s4[v];
        """
        precondition(source.components(separatedBy: oldNorm).count == 2)
        precondition(source.components(separatedBy: oldRead).count == 2)
        source = source.replacingOccurrences(of: oldNorm, with: newNorm)
            .replacingOccurrences(of: oldRead, with: newRead)
        return MLXFast.metalKernel(
            name: "bonsai_m5_r1_gdn_norm_rotation_v1",
            inputNames: ["x", "norm_weight", "epsilon", "signs"], outputNames: ["y", "normed_out"],
            source: source, header: SignedBlockHadamard.fused4096Header)
    }()
}

private enum VerifyK64 {
    nonisolated(unsafe) private static var kernels: [Bool: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var wideKernels: [Bool: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var productKernels: [Bool: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var productSelected: UInt8 = 0
    nonisolated(unsafe) private static var productUsed: UInt8 = 0
    nonisolated(unsafe) private static var wideSelected: UInt8 = 0
    nonisolated(unsafe) private static var selected: UInt8 = 0
    nonisolated(unsafe) private static var used: UInt8 = 0
    nonisolated(unsafe) private static var prepared = false

    static let install: Void = {
        HadamardQuantizedLinear.verifyK64Prepare = { prepare() }
        HadamardQuantizedLinear.verifyK64Run = { inputs, negative, streamed, product in
            let bit = UInt8(1 << ((streamed ? 2 : 0) + (negative ? 1 : 0)))
            if product, productSelected & bit != 0, let kernel = productKernels[streamed] {
                if productUsed & bit == 0 {
                    productUsed |= bit
                    FileHandle.standardError.write(Data("int8 verify K64 product: use streamed=\(streamed) NEG=\(negative) R16 K4096 N12288\n".utf8))
                }
                return runProduct(kernel, inputs, negative)
            }
            let wide = wideSelected & bit != 0
            guard let kernel = wide ? wideKernels[streamed] : (selected & bit != 0 ? kernels[streamed] : nil) else { return nil }
            let columns = wide ? 64 : 32
            if used & bit == 0 {
                used |= bit
                FileHandle.standardError.write(Data("int8 verify K64: use columns=\(columns) streamed=\(streamed) NEG=\(negative) R16 K4096 N12288x2\n".utf8))
            }
            return run(kernel, inputs, negative, columns)
        }
    }()

    // Return nil when any expected source fragment changes.
    static func source(_ original: String, streamed: Bool) -> String? {
        var text = original
        func swap(_ old: String, _ new: String) -> Bool {
            guard text.components(separatedBy: old).count == 2 else { return false }
            text = text.replacingOccurrences(of: old, with: new)
            return true
        }
        let loop = "for (int g = int(sg); g < Kg; g += 4) {\n"
        let meta = "  const half4 cA = sA, cB = sB;"
        guard let lo = text.range(of: loop),
            let hi = text.range(of: meta, range: lo.upperBound ..< text.endIndex) else { return nil }
        var stage = String(text[lo.upperBound ..< hi.lowerBound])
        text.removeSubrange(lo.upperBound ..< hi.lowerBound)
        if streamed {
            let old = "col * 32u + word * 4u"
            guard stage.components(separatedBy: old).count == 2 else { return nil }
            stage = stage.replacingOccurrences(of: old, with: "col * 16u + (word & 3u) * 4u")
        } else {
            guard stage.components(separatedBy: "dst[32 * t]").count == 2,
                swap("+ c * 8 + j;", "+ c * 4 + (j & 3);") else { return nil }
            stage = stage.replacingOccurrences(of: "dst[32 * t]", with: "dst[16 * t]")
        }
        let pre = "  if (g + 4 < Kg) {"
        guard text.components(separatedBy: pre).count == 2,
            let begin = text.range(of: pre) else { return nil }
        var end = begin.upperBound, depth = 1
        while end < text.endIndex && depth > 0 {
            if text[end] == "{" { depth += 1 }
            if text[end] == "}" { depth -= 1 }
            end = text.index(after: end)
        }
        guard depth == 0, end < text.endIndex, text[end] == "\n" else { return nil }
        end = text.index(after: end)
        let prefetch = String(text[begin.lowerBound ..< end])
        text.removeSubrange(begin.lowerBound ..< end)
        let issue = "  simdgroup_barrier(mem_flags::mem_threadgroup);\n  auto tA = A.template slice<128, 16>(g * 128, 0);\n  op.run(tA, B, cT);\n"
        let snapshot = "  const float ca = fa, crb = frb, ca1 = fa1, crb1 = frb1;\n"
        let predicate = streamed ? "int(lane & 1u) == h" : "(j >> 2) == h"
        let staged = stage.split(separator: "\n", omittingEmptySubsequences: false).dropLast()
            .map { "    " + $0 }.joined(separator: "\n")
        let halves = """
          #pragma clang loop unroll(full)
          for (int i = 0; i < 16; i++) { cT[i] = 0; }
          #pragma clang loop unroll(full)
          for (int h = 0; h < 2; h++) {
            if (\(predicate)) {
        \(staged)
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto tA = A.template slice<64, 16>(g * 128 + h * 64, 0);
            op.run(tA, B, cT);
            if (h == 0) { simdgroup_barrier(mem_flags::mem_threadgroup); }
          }

        """
        guard swap(issue, ""),
            swap(snapshot, snapshot + halves + prefetch),
            swap("matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply)",
                 "matmul2d_descriptor(16, 32, 64, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate)"),
            swap("threadgroup uint32_t bs[4][32 * 32];", "threadgroup uint32_t bs[4][16 * 32];"),
            swap("dextents<int, 2>(128, 32));", "dextents<int, 2>(64, 32));"),
            swap("auto tA0 = A.template slice<128, 16>(0, 0);", "auto tA0 = A.template slice<64, 16>(0, 0);"),
            swap("for (int e = 0; e < 16; e++) { acc[e] = 0.0f; }",
                 "for (int e = 0; e < 16; e++) { acc[e] = good ? 0.0f : as_type<float>(0x7fc00000u); }"),
            swap(loop, "if (good) {\n" + loop),
            swap("threadgroup float* red = (threadgroup float*)bs[sg];", "}\nthreadgroup float* red = (threadgroup float*)bs[sg];")
        else { return nil }
        return text
    }

    static func wideSource(_ original: String, streamed: Bool) -> String? {
        guard source(original, streamed: streamed) != nil,
            let begin = original.range(of: "const int Kg = K / 128;"),
            original.components(separatedBy: "const int Kg = K / 128;").count == 2 else { return nil }
        let prefix = String(original[..<begin.lowerBound])
        guard prefix.components(separatedBy: "N / 32").count == 3 else { return nil }
        let pointer: String
        let load: String
        let stage: String
        let predicate: String
        if streamed {
            guard original.contains("w + (uint)bx * (2) * Kg * 128;"),
                original.contains("(t / 4) * Kg * 128 + gl * 128 + lane * 4") else { return nil }
            pointer = "w + (uint)bx * 4u * (uint)Kg * 128u"
            load = "for (int t = 0; t < 16; t += 4) { const uint4 u = *((const device uint4*)(wcol + (t / 4) * Kg * 128 + gl * 128 + lane * 4)); wv[t] = u.x; wv[t + 1] = u.y; wv[t + 2] = u.z; wv[t + 3] = u.w; }"
            predicate = "int(lane & 1u) == h"
            if original.contains("threadgroup ulong4* target") {
                stage = "if ((t & 1) != 0) { const uint col = (lane >> 1) + 16u * uint(t / 4); const uint word = (lane & 1u) * 4u + uint(t % 4) - 1u; const uint32_t p = wv[t - 1]; const uint4 lo = uint4(p & 0x03030303u, (p >> 2) & 0x03030303u, (p >> 4) & 0x03030303u, (p >> 6) & 0x03030303u); const uint4 hi = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u); threadgroup ulong4* target = (threadgroup ulong4*)(bs[sg] + col * 16u + (word & 3u) * 4u); *target = ulong4(as_type<ulong2>(lo), as_type<ulong2>(hi)); }"
            } else {
                stage = "const uint col = (lane >> 1) + 16u * uint(t / 4); const uint word = (lane & 1u) * 4u + uint(t % 4); threadgroup uint4* target = (threadgroup uint4*)(bs[sg] + col * 16u + (word & 3u) * 4u); *target = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);"
            }
        } else {
            pointer = "w + (uint)(n0 + c) * (uint)KW + (uint)j"
            load = "for (int t = 0; t < 16; t++) { wv[t] = wcol[(uint)(4 * t) * (uint)KW + (uint)gl * 8u]; }"
            predicate = "(j >> 2) == h"
            stage = "dst[16 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);"
        }
        return prefix.replacingOccurrences(of: "N / 32", with: "N / 64") + """
        const int Kg = K / 128;
        const int KW = K / 16;
        const int n0 = bx * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 64, 64, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, 16));
        alignas(32) threadgroup uint32_t bs[4][64 * 16];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B((threadgroup int8_t*)bs[sg], dextents<int, 2>(64, 64));
        auto tA0 = A.template slice<64, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        const int cap = int(cT.get_capacity());
        ushort output_index[32];
        bool ok = cap == 32;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; i++) {
          output_index[i] = 0;
          if (i < cap && cT.is_valid_element(ushort(i))) {
            const auto xy = cT.get_multidimensional_index(i);
            const int col = int(xy[0]), row = int(xy[1]);
            const bool inside = col >= 0 && col < 64 && row >= 0 && row < 16;
            ok = ok && inside;
            if (inside) output_index[i] = ushort(row * 64 + col);
          } else { ok = false; }
        }
        const bool good = simd_all(ok);
        float acc[32];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; i++) { acc[i] = 0.0f; }
        const int j = int(lane & 7u), c = int(lane >> 3);
        const device uint32_t* wcol = \(pointer);
        threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 4 + (j & 3);
        uint32_t wv[16];
        if (good) {
          { const int gl = int(sg); \(load) }
          for (int g = int(sg); g < Kg; g += 4) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < 32; i++) { cT[i] = 0; }
            #pragma clang loop unroll(full)
            for (int h = 0; h < 2; h++) {
              if (\(predicate)) {
                #pragma clang loop unroll(full)
                for (int t = 0; t < 16; t++) {
                  const uint32_t v = wv[t];
                  \(stage)
                }
              }
              simdgroup_barrier(mem_flags::mem_threadgroup);
              auto tA = A.template slice<64, 16>(g * 128 + h * 64, 0);
              op.run(tA, B, cT);
              if (h == 0) { simdgroup_barrier(mem_flags::mem_threadgroup); }
            }
            if (g + 4 < Kg) { const int gl = g + 4; \(load) }
            #pragma clang loop unroll(full)
            for (int i = 0; i < 32; i++) {
              const uint p = uint(output_index[i]);
              const uint row = p >> 6, col = uint(n0) + (p & 63u);
              const half sh = scalesT[(uint)g * (uint)N + col];
              const half bh = NEG ? -sh : biasesT[(uint)g * (uint)N + col];
              const float a = ascale[row * (uint)Kg + (uint)g];
              const float rb = rsb[row * (uint)Kg + (uint)g];
              const float t = float(sh) * float(cT[i]);
              acc[i] = fma(float(bh), rb, fma(a, t, acc[i]));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }
        }
        threadgroup float* red = (threadgroup float*)bs[sg];
        if (good) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 32; i++) { red[output_index[i]] = acc[i]; }
        } else {
          for (int e = int(lane); e < 16 * 64; e += 32) { red[e] = as_type<float>(0x7fc00000u); }
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
    }

    static func productSource(_ original: String, streamed: Bool) -> String? {
        guard var text = wideSource(original, streamed: streamed),
            let begin = text.range(of: "const int Kg = K / 128;") else { return nil }
        text.replaceSubrange(text.startIndex ..< begin.lowerBound, with: """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const int K = KC;
        const int N = ksz[2];
        const int bx = int(threadgroup_position_in_grid.x);

        """)
        func swap(_ old: String, _ new: String, count: Int = 1) -> Bool {
            guard text.components(separatedBy: old).count == count + 1 else { return false }
            text = text.replacingOccurrences(of: old, with: new)
            return true
        }
        let pointer: String, replacement: String, load: String, pairedLoad: String
        if streamed {
            pointer = "const device uint32_t* wcol = w + (uint)bx * 4u * (uint)Kg * 128u;"
            replacement = "const device uint32_t* wcol0 = w0 + (uint)bx * 2u * (uint)Kg * 128u;\nconst device uint32_t* wcol1 = w1 + (uint)bx * 2u * (uint)Kg * 128u;"
            load = "for (int t = 0; t < 16; t += 4) { const uint4 u = *((const device uint4*)(wcol + (t / 4) * Kg * 128 + gl * 128 + lane * 4)); wv[t] = u.x; wv[t + 1] = u.y; wv[t + 2] = u.z; wv[t + 3] = u.w; }"
            pairedLoad = "for (int t = 0; t < 16; t += 4) { const device uint32_t* wc = t < 8 ? wcol0 : wcol1; const uint4 u = *((const device uint4*)(wc + ((t / 4) & 1) * Kg * 128 + gl * 128 + lane * 4)); wv[t] = u.x; wv[t + 1] = u.y; wv[t + 2] = u.z; wv[t + 3] = u.w; }"
        } else {
            pointer = "const device uint32_t* wcol = w + (uint)(n0 + c) * (uint)KW + (uint)j;"
            replacement = "const device uint32_t* wcol0 = w0 + (uint)(n0 + c) * (uint)KW + (uint)j;\nconst device uint32_t* wcol1 = w1 + (uint)(n0 + c) * (uint)KW + (uint)j;"
            load = "for (int t = 0; t < 16; t++) { wv[t] = wcol[(uint)(4 * t) * (uint)KW + (uint)gl * 8u]; }"
            pairedLoad = "for (int t = 0; t < 16; t++) { const device uint32_t* wc = t < 8 ? wcol0 : wcol1; wv[t] = wc[(uint)(4 * (t & 7)) * (uint)KW + (uint)gl * 8u]; }"
        }
        let oldFold = """
        for (int e = int(tid); e < ROWS * 16; e += 128) {
          float4 v = r0[e] + r1[e];
          v = v + r2[e];
          v = v + r3[e];
          *(device float4*)(out + (uint)(e >> 4) * (uint)N + n0 + 4 * (e & 15)) = v;
        }
        """
        let productFold = """
        for (int e = int(tid); e < ROWS * 8; e += 128) {
          const int p = (e >> 3) * 16 + (e & 7);
          float4 g = r0[p] + r1[p];
          g = g + r2[p];
          g = g + r3[p];
          float4 u = r0[p + 8] + r1[p + 8];
          u = u + r2[p + 8];
          u = u + r3[p + 8];
          const float4 z = 1.0f / (1.0f + metal::exp(metal::abs(g)));
          const float4 sig = select(1.0f - z, z, g < 0.0f);
          *(device float4*)(out0 + (uint)(e >> 3) * (uint)N + n0 + 4 * (e & 7)) = (g * sig) * u;
        }
        """
        guard swap("const int n0 = bx * 64;", "const int n0 = bx * 32;"),
            swap(pointer, replacement), swap(load, pairedLoad, count: 2),
            swap("const uint row = p >> 6, col = uint(n0) + (p & 63u);",
                 "const uint row = p >> 6, col = uint(n0) + (p & 31u);"),
            swap("const half sh = scalesT[(uint)g * (uint)N + col];",
                 "const half sh = ((p & 32u) != 0 ? s1 : s0)[(uint)g * (uint)N + col];"),
            swap("const half bh = NEG ? -sh : biasesT[(uint)g * (uint)N + col];",
                 "const half bh = NEG ? -sh : ((p & 32u) != 0 ? b1 : b0)[(uint)g * (uint)N + col];"),
            swap(oldFold, productFold) else { return nil }
        return text
    }

    private static func runProduct(
        _ kernel: MLXFast.MLXFastKernel, _ inputs: [MLXArray], _ negative: Bool
    ) -> [MLXArray] {
        kernel(inputs, template: [("ROWS", 16), ("NEG", negative ? 1 : 0), ("KC", 4096)],
               grid: (12288 / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
               outputShapes: [[16, 12288]], outputDTypes: [.float32])
    }

    private static let productReference = MLXFast.metalKernel(
        name: "bonsai_verify_k64_product_reference", inputNames: ["gate", "up"], outputNames: ["product"],
        source: """
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            const uint i = thread_position_in_grid.x;
            if (i >= 16u * 12288u / 4u) return;
            const float4 g = ((const device float4*)gate)[i];
            const float4 u = ((const device float4*)up)[i];
            const float4 z = 1.0f / (1.0f + metal::exp(metal::abs(g)));
            const float4 sig = select(1.0f - z, z, g < 0.0f);
            ((device float4*)product)[i] = (g * sig) * u;
            """, ensureRowContiguous: true)

    private static func chooseProduct(
        _ candidate: MLXFast.MLXFastKernel, current: MLXFast.MLXFastKernel,
        columns: Int, inputs: [MLXArray], negative: Bool, streamed: Bool, currentK64: Bool,
        down: SignedBlockHadamard
    ) -> Bool {
        let exact = (try? withError {
            let old = run(current, inputs, negative, columns), new = runProduct(candidate, inputs, negative)
            let reference = productReference(
                old, grid: (16 * 12288 / 4, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[16, 12288]], outputDTypes: [.float32])[0]
            var checks = old.map { MLX.all(isFinite($0)) }
                + [MLX.all(isFinite(reference)), MLX.all(isFinite(new[0])),
                   MLX.all(reference.view(dtype: .uint32) .== new[0].view(dtype: .uint32))]
            eval(checks)
            guard checks.allSatisfy({ $0.item(Bool.self) }),
                let a = down.quantizedForVerify(old[0], up: old[1]),
                let b = down.quantizedForVerify(new[0]) else { return false }
            checks = [MLX.all(a.codes .== b.codes)]
            for (x, y) in [(a.ascale, b.ascale), (a.rsb, b.rsb)] {
                checks.append(MLX.all(isFinite(x)))
                checks.append(MLX.all(isFinite(y)))
                checks.append(MLX.all(x.view(dtype: .uint32) .== y.view(dtype: .uint32)))
            }
            eval(checks)
            return checks.allSatisfy { $0.item(Bool.self) }
        }) ?? false
        func time(_ product: Bool) -> UInt64? {
            try? withError {
                Stream.gpu.synchronize()
                let start = DispatchTime.now().uptimeNanoseconds
                let projections = product ? runProduct(candidate, inputs, negative) : run(current, inputs, negative, columns)
                guard let a = down.quantizedForVerify(projections[0], up: product ? nil : projections[1]) else { return nil }
                let outputs = [a.codes, a.ascale, a.rsb]
                eval(outputs)
                Stream.gpu.synchronize()
                let end = DispatchTime.now().uptimeNanoseconds
                withExtendedLifetime((projections, outputs)) {}
                return end - start
            }
        }
        var oldNS: [UInt64] = [], newNS: [UInt64] = []
        if exact, time(false) != nil, time(true) != nil {
            for round in 0 ..< 3 {
                var pair = [UInt64](repeating: 0, count: 2)
                for method in round == 1 ? [1, 0] : [0, 1] { pair[method] = time(method == 1) ?? 0 }
                oldNS.append(pair[0]); newNS.append(pair[1])
            }
        }
        let chosen = exact && oldNS.count == 3 && zip(oldNS, newNS).allSatisfy { $0 > 0 && $1 > 0 && $1 < $0 }
        FileHandle.standardError.write(Data("int8 verify K64 product: streamed=\(streamed) NEG=\(negative) currentK64=\(currentK64) columns=\(columns) selected=\(chosen) exact=\(exact) current_ns=\(oldNS) product_ns=\(newNS)\n".utf8))
        return chosen
    }

    private static func chooseWide(
        _ candidate: MLXFast.MLXFastKernel, current: MLXFast.MLXFastKernel,
        inputs: [MLXArray], negative: Bool, streamed: Bool, currentK64: Bool
    ) -> Bool {
        let exact = (try? withError {
            let old = run(current, inputs, negative), new = run(candidate, inputs, negative, 64)
            var checks: [MLXArray] = []
            for i in 0 ..< 2 {
                checks.append(MLX.all(isFinite(old[i])))
                checks.append(MLX.all(isFinite(new[i])))
                checks.append(MLX.all(old[i].view(dtype: .uint32) .== new[i].view(dtype: .uint32)))
            }
            eval(checks)
            return checks.allSatisfy { $0.item(Bool.self) }
        }) ?? false
        func time(_ kernel: MLXFast.MLXFastKernel, _ columns: Int) -> UInt64? {
            try? withError {
                Stream.gpu.synchronize()
                let start = DispatchTime.now().uptimeNanoseconds
                let outputs = run(kernel, inputs, negative, columns)
                eval(outputs)
                Stream.gpu.synchronize()
                let end = DispatchTime.now().uptimeNanoseconds
                withExtendedLifetime(outputs) {}
                return end - start
            }
        }
        var oldNS: [UInt64] = [], newNS: [UInt64] = []
        if exact, time(current, 32) != nil, time(candidate, 64) != nil {
            for round in 0 ..< 3 {
                var pair = [UInt64](repeating: 0, count: 2)
                for method in round == 1 ? [1, 0] : [0, 1] {
                    pair[method] = time(method == 0 ? current : candidate, method == 0 ? 32 : 64) ?? 0
                }
                oldNS.append(pair[0]); newNS.append(pair[1])
            }
        }
        let chosen = exact && oldNS.count == 3 && zip(oldNS, newNS).allSatisfy { $0 > 0 && $1 > 0 && $1 < $0 }
        FileHandle.standardError.write(Data("int8 verify K64 N64: streamed=\(streamed) NEG=\(negative) currentK64=\(currentK64) selected=\(chosen) exact=\(exact) current_ns=\(oldNS) n64_ns=\(newNS)\n".utf8))
        return chosen
    }

    private static func run(_ kernel: MLXFast.MLXFastKernel, _ inputs: [MLXArray], _ negative: Bool, _ columns: Int = 32) -> [MLXArray] {
        kernel(inputs, template: [("ROWS", 16), ("NEG", negative ? 1 : 0), ("KC", 4096)],
               grid: (24576 / columns * 128, 1, 1), threadGroup: (128, 1, 1),
               outputShapes: [[16, 12288], [16, 12288]], outputDTypes: [.float32, .float32])
    }

    private static func fixture() -> [MLXArray] {
        let k = 4096, n = 12288, kg = k / 128
        var q = [Int8](repeating: 0, count: 16 * k)
        for i in q.indices {
            let row = i / k
            q[i] = row == 12 ? -128 : row == 13 ? 127 : row == 14 ? (i % 2 == 0 ? -128 : 127) : Int8((i * 37 + 11) % 255 - 127)
        }
        var w = [UInt32](repeating: 0, count: 2 * n * k / 16)
        for i in w.indices {
            let col = i / (k / 16)
            w[i] = col % n == 0 ? 0xFFFF_FFFF : col % n == 1 ? 0 : UInt32(truncatingIfNeeded: UInt64(i + 1) &* 2_654_435_761)
        }
        var s = [Float16](repeating: 0, count: 2 * kg * n)
        var b = s
        for i in s.indices { s[i] = Float16(Float(1 + i % 13) / 1024); b[i] = Float16(-Float(i % 7) / 512) }
        var a = [Float](repeating: 0, count: 16 * kg), r = a
        for i in a.indices { a[i] = Float(1 + i % 11) / 256; r[i] = Float(i % 17 - 8) / 64 }
        let words = MLXArray(w, [2, n, k / 16]), scales = MLXArray(s, [2, kg, n]), biases = MLXArray(b, [2, kg, n])
        return [MLXArray(q, [16, k]), words[0], words[1], scales[0], scales[1], biases[0], biases[1],
                MLXArray(a), MLXArray(r), HostConstants.int32([4096, 16, 12288, 12288])]
    }

    private static let repack = MLXFast.metalKernel(
        name: "bonsai_verify_k64_fixture_repack", inputNames: ["w"], outputNames: ["packed"],
        source: """
            const uint i = thread_position_in_grid.x;
            const uint Kg = 32, C = 16;
            if (i >= 12288 * 4096 / 16) return;
            const uint bx = i / (Kg * C * 8), local = i % (Kg * C * 8);
            const uint g = local / (C * 8), v = local % (C * 8);
            const uint col = bx * C + v / 8, j = v % 8;
            packed[i] = w[col * 256 + g * 8 + j];
            """, ensureRowContiguous: true)

    private static func prepare() {
        guard !prepared else { return }
        prepared = true
        var data: [MLXArray]? = nil
        var wideCases: [(Bool, MLXFast.MLXFastKernel, MLXFast.MLXFastKernel, [MLXArray])] = []
        var productCases: [(Bool, MLXFast.MLXFastKernel, MLXFast.MLXFastKernel, [MLXArray])] = []
        for streamed in [false, true] {
            guard let (original, control) = HadamardQuantizedLinear.verifyK64Source(streamed) else {
                FileHandle.standardError.write(Data("int8 verify K64: streamed=\(streamed) off (current N32 forms unavailable)\n".utf8))
                continue
            }
            guard let body = source(original, streamed: streamed) else {
                FileHandle.standardError.write(Data("int8 verify K64: streamed=\(streamed) off (source changed)\n".utf8))
                continue
            }
            let candidate = MLXFast.metalKernel(
                name: "bonsai_q8_verify_k64_gateup_\(streamed ? "streamed" : "canonical")_v1",
                inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "rsb", "ksz"],
                outputNames: ["out0", "out1"], source: body,
                header: "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
                ensureRowContiguous: true)
            let wideCandidate = wideSource(original, streamed: streamed).map { body in
                MLXFast.metalKernel(
                    name: "bonsai_q8_verify_k64_n64_gateup_\(streamed ? "streamed" : "canonical")_v1",
                    inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "rsb", "ksz"],
                    outputNames: ["out0", "out1"], source: body,
                    header: "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
                    ensureRowContiguous: true)
            }
            let productCandidate = productSource(original, streamed: streamed).map { body in
                MLXFast.metalKernel(
                    name: "bonsai_q8_verify_k64_product_\(streamed ? "streamed" : "canonical")_v1",
                    inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "rsb", "ksz"],
                    outputNames: ["out0"], source: body,
                    header: "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
                    ensureRowContiguous: true)
            }
            if productCandidate == nil { FileHandle.standardError.write(Data("int8 verify K64 product: streamed=\(streamed) off (source changed)\n".utf8)) }
            if wideCandidate == nil { FileHandle.standardError.write(Data("int8 verify K64 N64: streamed=\(streamed) off (source changed)\n".utf8)) }
            if data == nil { data = fixture(); eval(data!) }
            var inputs = data!
            if streamed {
                for i in 1 ... 2 {
                    inputs[i] = repack([inputs[i]], grid: (12288 * 256, 1, 1), threadGroup: (256, 1, 1),
                                       outputShapes: [[12288, 256]], outputDTypes: [.uint32])[0]
                }
                eval(inputs)
            }
            for negative in [false, true] {
                var values = inputs
                if negative {
                    for i in 5 ... 6 { values[i] = (inputs[i - 2].view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16) }
                    eval(values)
                }
                let exact = (try? withError {
                    let old = run(control, values, negative), new = run(candidate, values, negative)
                    var checks: [MLXArray] = []
                    for i in 0 ..< 2 {
                        checks.append(MLX.all(isFinite(old[i])))
                        checks.append(MLX.all(isFinite(new[i])))
                        checks.append(MLX.all(old[i].view(dtype: .uint32) .== new[i].view(dtype: .uint32)))
                    }
                    eval(checks)
                    return checks.allSatisfy { $0.item(Bool.self) }
                }) ?? false
                func time(_ kernel: MLXFast.MLXFastKernel) -> UInt64? {
                    try? withError {
                        Stream.gpu.synchronize()
                        let start = DispatchTime.now().uptimeNanoseconds
                        let outputs = run(kernel, values, negative)
                        eval(outputs)
                        Stream.gpu.synchronize()
                        let end = DispatchTime.now().uptimeNanoseconds
                        withExtendedLifetime(outputs) {}
                        return end - start
                    }
                }
                var oldNS: [UInt64] = [], newNS: [UInt64] = []
                if exact, time(control) != nil, time(candidate) != nil {
                    for round in 0 ..< 3 {
                        var pair = [UInt64](repeating: 0, count: 2)
                        for method in round == 1 ? [1, 0] : [0, 1] { pair[method] = time(method == 0 ? control : candidate) ?? 0 }
                        oldNS.append(pair[0]); newNS.append(pair[1])
                    }
                }
                let chosen = exact && oldNS.count == 3 && zip(oldNS, newNS).allSatisfy { $0 > 0 && $1 > 0 && $1 < $0 }
                FileHandle.standardError.write(Data("int8 verify K64: streamed=\(streamed) NEG=\(negative) selected=\(chosen) exact=\(exact) current_ns=\(oldNS) k64_ns=\(newNS)\n".utf8))
                if chosen {
                    selected |= UInt8(1 << ((streamed ? 2 : 0) + (negative ? 1 : 0)))
                    kernels[streamed] = candidate
                }
            }
            if let wideCandidate { wideCases.append((streamed, wideCandidate, control, inputs)) }
            if let productCandidate { productCases.append((streamed, productCandidate, control, inputs)) }
        }
        for (streamed, candidate, control, inputs) in wideCases {
            for negative in [false, true] {
                var values = inputs
                if negative {
                    for i in 5 ... 6 { values[i] = (inputs[i - 2].view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16) }
                    eval(values)
                }
                let bit = UInt8(1 << ((streamed ? 2 : 0) + (negative ? 1 : 0)))
                let currentK64 = selected & bit != 0
                let incumbent = currentK64 ? kernels[streamed]! : control
                if chooseWide(candidate, current: incumbent, inputs: values,
                              negative: negative, streamed: streamed, currentK64: currentK64) {
                    wideSelected |= bit
                    wideKernels[streamed] = candidate
                }
            }
        }
        if !productCases.isEmpty, let down = try? SignedBlockHadamard(
            blockSize: 4096, signs: (0 ..< 12288).map { ($0 * 11 + 3) % 7 < 3 ? Float(-1) : Float(1) }) {
            SignedBlockHadamard.wideRowPrepareIfBoot(rows: 1024)
            eval(down.signs)
            for (streamed, candidate, control, inputs) in productCases {
                for negative in [false, true] {
                    var values = inputs
                    if negative {
                        for i in 5 ... 6 { values[i] = (inputs[i - 2].view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16) }
                        eval(values)
                    }
                    let bit = UInt8(1 << ((streamed ? 2 : 0) + (negative ? 1 : 0)))
                    let wide = wideSelected & bit != 0
                    let currentK64 = wide || selected & bit != 0
                    let incumbent = wide ? wideKernels[streamed]! : (currentK64 ? kernels[streamed]! : control)
                    if chooseProduct(candidate, current: incumbent, columns: wide ? 64 : 32,
                                     inputs: values, negative: negative, streamed: streamed, currentK64: currentK64, down: down) {
                        productSelected |= bit
                        productKernels[streamed] = candidate
                    }
                }
            }
        }
    }
}
