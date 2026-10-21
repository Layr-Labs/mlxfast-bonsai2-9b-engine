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
        _ = FixedM5Product.install
        super.init(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData))
    }
    public func shouldLoadWeight(named name: String) -> Bool {
        !name.hasPrefix("vision_tower.")
    }
}

enum FixedM5Product {
    private static let m5: Bool = {
        let arch = GPU.deviceInfo().architecture
        return arch.hasPrefix("applegpu_g17") && !arch.hasPrefix("applegpu_g17p")
    }()
    static let install: Void = {
        guard m5 else {
            FileHandle.standardError.write(Data("M5 fixed product: non-M5 fallback\n".utf8))
            return
        }
        HadamardQuantizedLinear.fixedProduct = { x, residual, weight, eps, packed, down in
            guard x.dtype == .float32, x.size == 16 * 4096,
                residual.dtype == .float32, residual.shape == x.shape,
                weight.dtype == .float32, weight.shape == [4096], packed.count == 2,
                packed.allSatisfy({ valid($0, k: 4096, n: 12288) }),
                valid(down, k: 12288, n: 4096), StreamOrDevice.default == StreamOrDevice.gpu
            else { return nil }
            let activation = SignedBlockHadamard.fixedProductKernels.0(
                [x, residual, weight, HostConstants.float32([eps]), packed[0].transform.signs],
                template: [("BLOCKS_PER_ROW", 1), ("VERIFY_ROWS", 16)],
                grid: (256, 16, 1), threadGroup: (256, 1, 1),
                outputShapes: [[16, 4096], [512], [512], x.shape],
                outputDTypes: [.int8, .float32, .float32, .float32])
            let g = packed[0].verifyOperands, u = packed[1].verifyOperands
            let output = product(
                [activation[0], g.weight, u.weight, g.scalesT, u.scalesT, g.biasesT, u.biasesT,
                 activation[1], activation[2], HostConstants.int32([4096, 16, 12288, 12288])],
                template: [("ROWS", 16), ("NEG", 0), ("KC", 4096)],
                grid: (384 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[16, 12288]], outputDTypes: [.float32])[0]
            return (activation[3], [output.reshaped(Array(x.shape.dropLast()) + [12288])])
        }
        FileHandle.standardError.write(Data("M5 fixed product: installed canonical K64 N64 NEG0; norm256 plain3-256 downN16; R16 K4096 N12288\n".utf8))
    }()

    private static func valid(_ layer: HadamardQuantizedLinear, k: Int, n: Int) -> Bool {
        layer.bits == 2 && layer.groupSize == 128 && layer.bias == nil && layer.gdnLayout == nil
            && layer.transform.blockSize == 4096 && layer.transform.width == k
            && layer.weight.dtype == .uint32 && layer.weight.shape == [n, k / 16]
            && layer.scales.dtype == .float16 && layer.scales.shape == [n, k / 128]
            && layer.biases?.dtype == .float16 && layer.biases?.shape == [n, k / 128]
    }

    static func down(_ layer: HadamardQuantizedLinear, _ x: MLXArray) -> MLXArray? {
        guard m5, x.ndim >= 2, x.dim(-1) == 12288, x.size == 16 * 12288,
            x.dtype == .float32, valid(layer, k: 12288, n: 4096),
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let activation = SignedBlockHadamard.fixedProductKernels.1(
            [x.reshaped([16, 12288]), layer.transform.signs],
            template: [("BLOCKS_PER_ROW", 3), ("VERIFY_ROWS", 16)],
            grid: (256, 48, 1), threadGroup: (256, 1, 1),
            outputShapes: [[16, 12288], [1536], [1536]], outputDTypes: [.int8, .float32, .float32])
        return layer.fixedProductDown(activation).reshaped(Array(x.shape.dropLast()) + [4096])
    }

    private static let product = MLXFast.metalKernel(
        name: "bonsai_verify_fixed_m5_k64_n64_product_v1",
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
        const int n0 = bx * 32;
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
        const device uint32_t* wcol0 = w0 + (uint)(n0 + c) * (uint)KW + (uint)j;
        const device uint32_t* wcol1 = w1 + (uint)(n0 + c) * (uint)KW + (uint)j;
        threadgroup uint4* dst = (threadgroup uint4*)bs[sg] + c * 4 + (j & 3);
        uint32_t wv[16];
        if (good) {
          { const int gl = int(sg); for (int t = 0; t < 16; t++) { const device uint32_t* wc = t < 8 ? wcol0 : wcol1; wv[t] = wc[(uint)(4 * (t & 7)) * (uint)KW + (uint)gl * 8u]; } }
          for (int g = int(sg); g < Kg; g += 4) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < 32; i++) { cT[i] = 0; }
            #pragma clang loop unroll(full)
            for (int h = 0; h < 2; h++) {
              if ((j >> 2) == h) {
                #pragma clang loop unroll(full)
                for (int t = 0; t < 16; t++) {
                  const uint32_t v = wv[t];
                  dst[16 * t] = uint4(v & 0x03030303u, (v >> 2) & 0x03030303u, (v >> 4) & 0x03030303u, (v >> 6) & 0x03030303u);
                }
              }
              simdgroup_barrier(mem_flags::mem_threadgroup);
              auto tA = A.template slice<64, 16>(g * 128 + h * 64, 0);
              op.run(tA, B, cT);
              if (h == 0) { simdgroup_barrier(mem_flags::mem_threadgroup); }
            }
            if (g + 4 < Kg) { const int gl = g + 4; for (int t = 0; t < 16; t++) { const device uint32_t* wc = t < 8 ? wcol0 : wcol1; wv[t] = wc[(uint)(4 * (t & 7)) * (uint)KW + (uint)gl * 8u]; } }
            #pragma clang loop unroll(full)
            for (int i = 0; i < 32; i++) {
              const uint p = uint(output_index[i]);
              const uint row = p >> 6, col = uint(n0) + (p & 31u);
              const half sh = ((p & 32u) != 0 ? s1 : s0)[(uint)g * (uint)N + col];
              const half bh = NEG ? -sh : ((p & 32u) != 0 ? b1 : b0)[(uint)g * (uint)N + col];
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
        """,
        header: "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        ensureRowContiguous: true)
}
