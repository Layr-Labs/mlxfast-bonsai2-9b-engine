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
        super.init(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData))
    }
    public func shouldLoadWeight(named name: String) -> Bool {
        !name.hasPrefix("vision_tower.")
    }
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
        SignedBlockHadamard.headTail = HeadQuantRowBounds.tail
        HadamardQuantizedLinear.verifyHeadRescore = M5HeadGroupedRescore.make
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

private enum M5HeadGroupedRescore {
    static func make(_ original: String, _ header: String) -> (MLXFast.MLXFastKernel, Int) {
        let arch = GPU.deviceInfo().architecture
        let grouped = arch.hasPrefix("applegpu_g17") && !arch.hasPrefix("applegpu_g17p")
        let body = grouped ? HeadQuantRowBounds.rescore(source(original)) : original
            .replacingOccurrences(of: "if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);", with: "float local_e = e; if (USE_L2) e = head_l2_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e);")
            .replacingOccurrences(of: "const float anchor_t = anchor_floor - e;", with: "const float anchor_t = anchor_floor - local_e;")
        return (MLXFast.metalKernel(
            name: "bonsai_verify_head_partition_rescore_group2_shared_error_v1",
            inputNames: ["logits", "ascale", "bounds", "x", "w", "scales", "biases", "part_ids", "part_values", "part_meta", "compact_ids", "omitted", "error2"],
            outputNames: ["blk_ids", "blk_values", "blk_meta", "row_e"],
            source: body, header: header), grouped ? 512 : 256)
    }

    static func source(_ original: String) -> String {
        var text = original
        func replace(_ old: String, _ new: String, count: Int = 1) {
            precondition(text.components(separatedBy: old).count == count + 1)
            text = text.replacingOccurrences(of: old, with: new)
        }
        replace("const uint block = threadgroup_position_in_grid.x;", """
            const uint partition = thread_position_in_threadgroup.x / uint(THREADS);
            const uint block = threadgroup_position_in_grid.x * 2u + partition;
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
            threadgroup atomic_int counts2[2], overflows2[2];
            threadgroup int ids2[2 * CAP], bads2[2];
            threadgroup float values2[2 * CAP], taus2[4];
            threadgroup float leaders[2], leader_second;
            threadgroup uint leader_ok[2], leader_count;
            threadgroup int leader_bad;
            threadgroup atomic_int* cand_count = counts2 + partition;
            threadgroup atomic_int* omitted_overflow = overflows2 + partition;
            threadgroup int* cand_ids = ids2 + partition * uint(CAP);
            threadgroup float* cand_values = values2 + partition * uint(CAP);
            threadgroup float* shared_tau = taus2 + partition * 2u;
            threadgroup int* shared_bad = bads2 + partition;
            """)
        replace("&cand_count", "cand_count", count: 3)
        replace("&omitted_overflow", "omitted_overflow", count: 3)
        replace("for (int anchor = 0; anchor < 2; anchor++) {", "{ const int anchor = int(partition);")
        replace("""
              if (simd_lid == 0) {
                float e = 0.0f;
                for (int g = 0; g < in_g; g++) {
                  e += ascale[row * uint(in_g) + uint(g)]
                      * (bounds[g] * (0.5f + 127.0f * 3.0e-5f) + bounds[in_g + g] * 1.0e-5f);
                }
                e *= 1.001f; if (USE_L2) e = head_l2_error(e, ascale, bounds, error2, row, in_g);
            """, """
              if (simd_lid == 0) {
                leaders[partition] = anchor_floor;
                leader_ok[partition] = anchor_ok ? 1u : 0u;
                if (partition == 0) {
                  leader_second = local.second_value;
                  leader_count = local.count;
                  leader_bad = pbad;
                }
                float e = 0.0f;
                for (int g = 0; g < in_g; g++) {
                  e += ascale[row * uint(in_g) + uint(g)]
                      * (bounds[g] * (0.5f + 127.0f * 3.0e-5f) + bounds[in_g + g] * 1.0e-5f);
                }
                e *= 1.001f; float local_e = e; if (USE_L2) e = head_l2_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e);
                shared_tau[0] = e;
                shared_tau[1] = local_e;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float cutoff_tau = 0.0f;
            if (simd_lid == 0) {
              const bool anchor_ok = leader_ok[0] != 0u && leader_ok[1] != 0u;
              const float anchor_floor = min(min(INFINITY, leaders[0]), leaders[1]);
              const float e = shared_tau[0];
              const float local_e = shared_tau[1];
            """)
        replace("const float base_t = local.second_value - 2.0f * e;",
            "const float base_t = leader_second - 2.0f * e;")
        replace("const float anchor_t = anchor_floor - e;", "const float anchor_t = anchor_floor - local_e;")
        replace("""
                shared_tau[0] = t - (fabs(t) * 2.4e-7f + 1.0e-30f);
                shared_tau[1] = e;
                shared_bad = (pbad != 0 || local.count < 2 || !isfinite(e) || !isfinite(t)) ? 1 : 0;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tau = shared_tau[0];
            """, """
              cutoff_tau = t - (fabs(t) * 2.4e-7f + 1.0e-30f);
              if (simd_gid == 0) {
                shared_bad[0] = (leader_bad != 0 || leader_count < 2 || !isfinite(e) || !isfinite(t)) ? 1 : 0;
              }
            }
            const float tau = simd_shuffle(cutoff_tau, ushort(0));
            """)
        replace("= shared_bad |", "= shared_bad[0] |")
        replace("if (block == 0) row_e[row] = shared_tau[1];", "if (block == 0) row_e[row] = shared_tau[0];")
        replace("""
            for (int q = int(lane); q < quads; q += THREADS) {
              const float4 v4 = lq[q]; if (omitted[row * uint(NV / 4) + first / 4u + uint(q)] >= tau) atomic_store_explicit(omitted_overflow, 1, memory_order_relaxed);
              for (int i = 0; i < 4; i++) {
                if (v4[i] >= tau) {
                  const int slot = atomic_fetch_add_explicit(cand_count, 1, memory_order_relaxed);
                  if (slot < CAP) cand_ids[slot] = head_id(compact_ids,row,NV,first+4*q+i);
                }
              }
            }
            """, """
            for (int q = int(lane); q < quads; q += THREADS) {
              const uint tile = first / 4u + uint(q);
              if (omitted[row * uint(NV / 4) + tile] >= tau) {
                const int slot = atomic_fetch_add_explicit(cand_count, 32, memory_order_relaxed);
                if (slot + 32 > CAP) atomic_store_explicit(omitted_overflow, 1, memory_order_relaxed);
                for (int i = 0; i < 32; i++) {
                  if (slot + i < CAP) cand_ids[slot + i] = int(tile * 32u + uint(i));
                }
              } else {
                const float4 v4 = lq[q];
                for (int i = 0; i < 4; i++) {
                  if (v4[i] >= tau) {
                    const int slot = atomic_fetch_add_explicit(cand_count, 1, memory_order_relaxed);
                    if (slot < CAP) cand_ids[slot] = head_id(compact_ids,row,NV,first+4*q+i);
                  }
                }
              }
            }
            """)
        return text
    }
}

private enum HeadQuantRowBounds {
    static let tail = """
        threadgroup_barrier(mem_flags::mem_device);
        if (i < 32) {
          const uint row = block_idx;
          const uint bound_lane = thread_index_in_simdgroup;
          constexpr int in_g = 32;
          const float group_a = ascale[row * uint(in_g) + bound_lane];
          const float group_amax = head_bounds[bound_lane];
          const float group_bmax = head_bounds[uint(in_g) + bound_lane];
          const float group_d2 = qerror2[row * uint(in_g) + bound_lane];
          const float group_c2 = head_bounds[2u * uint(in_g) + bound_lane];
          const float group_inner = group_amax * (0.5f + 127.0f * 3.0e-5f) + group_bmax * 1.0e-5f;
          float group_term = 0.0f;
          uint group_ok = 0u;
          if (head_l2_safe(group_d2) && head_l2_safe(group_c2)) {
            const float radius = head_l2_root(head_l2_mul(group_d2, group_c2));
            if (head_l2_safe(radius)) {
              const float worst = head_l2_mul(head_l2_mul(0.5f, group_a), group_amax);
              const float rounding = head_l2_mul(group_a, head_l2_add(
                  head_l2_mul(head_l2_mul(127.0f, 3.0e-5f), group_amax),
                  head_l2_mul(1.0e-5f, group_bmax)));
              group_term = head_l2_add(min(worst, radius), rounding);
              group_ok = 1u;
            }
          }
          float e = 0.0f;
          for (int g = 0; g < in_g; g++) {
            const float a = simd_shuffle(group_a, ushort(g));
            const float inner = simd_shuffle(group_inner, ushort(g));
            if (bound_lane == 0u) {
              e += a * inner;
            }
          }
          e *= 1.001f;
          const float raw_e = e;
          float total = 0.0f;
          uint status = 0u;
          bool safe = true;
          for (int g = 0; g < in_g; g++) {
            const float term = simd_shuffle(group_term, ushort(g));
            const uint ok = simd_shuffle(group_ok, ushort(g));
            if (bound_lane == 0u && safe) {
              if (ok == 0u) safe = false;
              else total = head_l2_add(total, term);
            }
          }
          if (bound_lane == 0u) {
            if (safe) {
              const float upper = head_l2_mul(total, 1.001f);
              status = head_l2_safe(upper) ? 2u : 1u;
              e = status == 2u ? min(raw_e, upper) : raw_e;
            }
            const uint at = uint(VERIFY_ROWS) * 32u + row * 3u;
            qerror2[at] = raw_e;
            qerror2[at + 1u] = e;
            qerror2[at + 2u] = float(status);
          }
        }
        """
    static func rescore(_ original: String) -> String {
        let old = """
            float e = 0.0f;
            for (int g = 0; g < in_g; g++) {
              e += ascale[row * uint(in_g) + uint(g)]
                  * (bounds[g] * (0.5f + 127.0f * 3.0e-5f) + bounds[in_g + g] * 1.0e-5f);
            }
            e *= 1.001f; float local_e = e; if (USE_L2) e = head_l2_partition_error(e, ascale, bounds, error2, row, in_g, block, local_e);
            shared_tau[0] = e;
            shared_tau[1] = local_e;
        """
        let new = """
            const uint bound_at = uint(x_shape[0]) * uint(in_g) + row * 3u;
            const float e = error2[bound_at + 1u];
            float local_e = e;
            if (error2[bound_at + 2u] == 2.0f) {
              float part_total = 0.0f;
              bool part_ok = true;
              for (int g = 0; g < in_g; g++) {
                const float a = ascale[row * uint(in_g) + uint(g)];
                const float d2 = error2[row * uint(in_g) + uint(g)];
                const float worst = head_l2_mul(head_l2_mul(0.5f, a), bounds[g]);
                const float rounding = head_l2_mul(a, head_l2_add(
                    head_l2_mul(head_l2_mul(127.0f, 3.0e-5f), bounds[g]),
                    head_l2_mul(1.0e-5f, bounds[in_g + g])));
                const float part_c2 = bounds[(3 + int(block)) * in_g + g];
                const float part_radius = head_l2_root(head_l2_mul(d2, part_c2));
                if (!head_l2_safe(part_c2) || !head_l2_safe(part_radius)) { part_ok = false; break; }
                part_total = head_l2_add(part_total, head_l2_add(min(worst, part_radius), rounding));
              }
              if (part_ok) {
                const float part_upper = head_l2_mul(part_total, 1.001f);
                if (head_l2_safe(part_upper)) local_e = min(e, part_upper);
              }
            }
            shared_tau[0] = e;
            shared_tau[1] = local_e;
        """
        precondition(original.components(separatedBy: old).count == 2)
        return original.replacingOccurrences(of: old, with: new)
    }
}
