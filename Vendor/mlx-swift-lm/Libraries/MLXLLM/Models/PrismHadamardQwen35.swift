// Copyright © 2026 Eigen Labs.
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
        HadamardQuantizedLinear.verifyProduct = { x, residual, weight, eps, packed, down in
            guard HadamardQuantizedLinear.verifyProductReady,
                x.dtype == .float32, x.size == 16 * 4096,
                residual.dtype == .float32, residual.shape == x.shape,
                weight.dtype == .float32, weight.shape == [4096], packed.count == 2,
                packed.allSatisfy({ valid($0, k: 4096, n: 12288) }),
                valid(down, k: 12288, n: 4096), StreamOrDevice.default == StreamOrDevice.gpu,
                let activation = packed[0].transform.residualNorm(
                    x, residual: residual, weight: weight, eps: eps, quantized: false, verifyRows: 16)
            else { return nil }
            let g = packed[0].verifyOperands, u = packed[1].verifyOperands
            let output = product(
                [activation[0], g.weight, u.weight, g.scalesT, u.scalesT, g.biasesT, u.biasesT,
                 activation[1], activation[2], HostConstants.int32([4096, 16, 12288, 12288])],
                template: [("ROWS", 16), ("NEG", 0), ("KC", 4096)],
                grid: (768 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[16, 12288]], outputDTypes: [.float32])[0]
            return (activation.last!, [output.reshaped(Array(x.shape.dropLast()) + [12288])])
        }
        FileHandle.standardError.write(Data("M5 verify product: installed canonical K128 N32; product WHT512; accepted norm/down dispatch\n".utf8))
    }()

    private static func valid(_ layer: HadamardQuantizedLinear, k: Int, n: Int) -> Bool {
        layer.bits == 2 && layer.groupSize == 128 && layer.bias == nil && layer.gdnLayout == nil
            && layer.transform.blockSize == 4096 && layer.transform.width == k
            && layer.weight.dtype == .uint32 && layer.weight.shape == [n, k / 16]
            && layer.scales.dtype == .float16 && layer.scales.shape == [n, k / 128]
            && layer.biases?.dtype == .float16 && layer.biases?.shape == [n, k / 128]
    }

    static func down(_ projection: Linear, _ x: MLXArray) -> MLXArray? {
        guard m5, HadamardQuantizedLinear.verifyProductReady,
            let layer = projection as? HadamardQuantizedLinear,
            x.ndim >= 2, x.dim(-1) == 12288, x.size == 16 * 12288,
            x.dtype == .float32, valid(layer, k: 12288, n: 4096),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let activation = productQuantizer(
            [x.reshaped([16, 12288]), layer.transform.signs, HostConstants.int32([16])],
            template: [("BLOCKS_PER_ROW", 3)],
            grid: (512, 48, 1), threadGroup: (512, 1, 1),
            outputShapes: [[16, 12288], [1536], [1536]], outputDTypes: [.int8, .float32, .float32])
        return layer.verifyProductDown(activation).reshaped(Array(x.shape.dropLast()) + [4096])
    }

    private static let productQuantizer: MLXFast.MLXFastKernel = {
        let text = SignedBlockHadamard.verifyProductText
        return MLXFast.metalKernel(
            name: "bonsai_verify_product_input_w512_v1",
            inputNames: ["x", "signs", "verify_rows"], outputNames: ["codes", "ascale", "rsb"],
            source: text.0, header: text.1, ensureRowContiguous: true)
    }()

    private static let product = MLXFast.metalKernel(
        name: "bonsai_verify_fixed_m5_k128_n32_product_v1",
        inputNames: ["xq", "w0", "w1", "s0", "s1", "b0", "b1", "ascale", "rsb", "ksz"],
        outputNames: ["out0"],
        source: """
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        const int K = KC;
        const int N = ksz[2];
        const int bx = int(threadgroup_position_in_grid.x);
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
          if (i < cap && cT.is_valid_element(ushort(i))) {
            auto ids = cT.get_multidimensional_index(i);
            ok = ok && int(ids[1]) == mrow + 8 * ((i >> 2) & 1) && int(ids[0]) == nb + (i & 3) + 16 * (i >> 3);
          } else { ok = false; }
        }
        const bool good = simd_all(ok);
        float acc[16];
        #pragma clang loop unroll(full)
        for (int e = 0; e < 16; e++) { acc[e] = 0.0f; }
        const int mr = min(mrow, ROWS - 1);
        const int mr1 = min(mrow + 8, ROWS - 1);
        const device half* scol0 = s0 + n0 + nb;
        const device half* scol1 = s1 + n0 + nb;
        const device half* bcol0 = b0 + n0 + nb;
        const device half* bcol1 = b1 + n0 + nb;
        const int j = int(lane & 7u);
        const int c = int(lane >> 3);
        const device uint32_t* wcol0 = w0 + (uint)(n0 + c) * KW + j;
        const device uint32_t* wcol1 = w1 + (uint)(n0 + c) * KW + j;
        threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 8 + j;
        uint32_t wv[8];
        half4 sA, sB, bA = half4(0.0h), bB = half4(0.0h);
        float fa, frb, fa1 = 0.0f, frb1 = 0.0f;
        threadgroup float* red = (threadgroup float*)bs[sg];
        if (good) {
        {
          const int gl = int(sg);
          #pragma clang loop unroll(full)
          for (int t = 0; t < 8; t++) { const device uint32_t* wc = t < 4 ? wcol0 : wcol1; wv[t] = wc[(uint)(4 * (t & 3)) * KW + gl * 8]; }
          sA = *(const device half4*)(scol0 + (uint)gl * N);
          sB = *(const device half4*)(scol1 + (uint)gl * N);
          if (!NEG) {
            bA = *(const device half4*)(bcol0 + (uint)gl * N);
            bB = *(const device half4*)(bcol1 + (uint)gl * N);
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
            for (int t = 0; t < 8; t++) { const device uint32_t* wc = t < 4 ? wcol0 : wcol1; wv[t] = wc[(uint)(4 * (t & 3)) * KW + gl * 8]; }
            sA = *(const device half4*)(scol0 + (uint)gl * N);
            sB = *(const device half4*)(scol1 + (uint)gl * N);
            if (!NEG) {
              bA = *(const device half4*)(bcol0 + (uint)gl * N);
              bB = *(const device half4*)(bcol1 + (uint)gl * N);
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
        } else {
          for (int e = int(lane); e < 16 * 32; e += 32) { red[e] = as_type<float>(0x7fc00000u); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup float4* r0 = (const threadgroup float4*)bs[0];
        const threadgroup float4* r1 = (const threadgroup float4*)bs[1];
        const threadgroup float4* r2 = (const threadgroup float4*)bs[2];
        const threadgroup float4* r3 = (const threadgroup float4*)bs[3];
        for (int e = int(tid); e < ROWS * 4; e += 128) {
          const int p = (e >> 2) * 8 + (e & 3);
          float4 g = r0[p] + r1[p];
          g = g + r2[p];
          g = g + r3[p];
          float4 u = r0[p + 4] + r1[p + 4];
          u = u + r2[p + 4];
          u = u + r3[p + 4];
          const float4 z = 1.0f / (1.0f + metal::exp(metal::abs(g)));
          const float4 sig = select(1.0f - z, z, g < 0.0f);
          *(device float4*)(out0 + (uint)(e >> 2) * N + n0 + 4 * (e & 3)) = (g * sig) * u;
        }
        """,
        header: "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        ensureRowContiguous: true)
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
            template: [("NEG", head.negative ? 1 : 0)],
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
