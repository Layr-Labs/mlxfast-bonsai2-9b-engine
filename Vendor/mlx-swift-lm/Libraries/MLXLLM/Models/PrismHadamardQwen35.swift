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
    nonisolated(unsafe) private static var selected: UInt8 = 0
    nonisolated(unsafe) private static var used: UInt8 = 0
    nonisolated(unsafe) private static var prepared = false

    static let install: Void = {
        HadamardQuantizedLinear.verifyK64Prepare = { prepare() }
        HadamardQuantizedLinear.verifyK64Run = { inputs, negative, streamed in
            let bit = UInt8(1 << ((streamed ? 2 : 0) + (negative ? 1 : 0)))
            guard selected & bit != 0, let kernel = kernels[streamed] else { return nil }
            if used & bit == 0 {
                used |= bit
                FileHandle.standardError.write(Data("int8 verify K64: use streamed=\(streamed) NEG=\(negative) R16 K4096 N12288x2\n".utf8))
            }
            return run(kernel, inputs, negative)
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

    private static func run(_ kernel: MLXFast.MLXFastKernel, _ inputs: [MLXArray], _ negative: Bool) -> [MLXArray] {
        kernel(inputs, template: [("ROWS", 16), ("NEG", negative ? 1 : 0), ("KC", 4096)],
               grid: (24576 / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
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
        }
    }
}
