// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import CoreFoundation
import Foundation
@_spi(QuantizedConstantCache) import MLX
import MLXLMCommon
import MLXNN

private enum Qwen35MTPFloat16ConstantQMV {
    static let maximumWideRows = 8

    static let fast = MLXFast.metalKernel(
        name: "qwen35_mtp_qmv_fast_w4_f16_constants",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const int K = x_shape[x_ndim - 1];
            const int in_w = K / 2;
            const int in_g = K / GS;
            const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
            const device uint8_t* ws = (const device uint8_t*)w + out_row * in_w + simd_lid * 8;
            const device half* sc = scales + out_row * in_g + simd_lid / (GS / 16);
            const device half* bi = biases + out_row * in_g + simd_lid / (GS / 16);
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float x_thread[16];
            float result[4] = {0, 0, 0, 0};
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += xv.x + xv.y + xv.z + xv.w;
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y * 0.0625f;
                x_thread[4 * i + 2] = xv.z * 0.00390625f;
                x_thread[4 * i + 3] = xv.w * 0.000244140625f;
              }
              for (int r = 0; r < 4; r++) {
                const device uint16_t* wl = (const device uint16_t*)(ws + r * in_w);
                const float s = float(sc[r * in_g]);
                const float b = float(bi[r * in_g]);
                float accum = 0;
                for (int i = 0; i < 4; i++) {
                  const uint16_t packed_w = wl[i];
                  accum += (x_thread[4 * i] * float(packed_w & 0x000fu)
                      + x_thread[4 * i + 1] * float(packed_w & 0x00f0u)
                      + x_thread[4 * i + 2] * float(packed_w & 0x0f00u)
                      + x_thread[4 * i + 3] * float(packed_w & 0xf000u));
                }
                result[r] += s * accum + sum * b;
              }
              ws += 256;
              sc += 512 / GS;
              bi += 512 / GS;
              xp4 += 128;
            }
            for (int r = 0; r < 4; r++) {
              const float total = simd_sum(result[r]);
              if (simd_lid == 0) y[out_row + r] = total;
            }
            """)

    static let wide = MLXFast.metalKernel(
        name: "qwen35_mtp_qmv_wide_w4_f16_constants",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            constexpr int k_lanes = 8;
            constexpr int results_per_sg = 32 / k_lanes;
            constexpr int sub = 8;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const short k_lane = simd_lid % k_lanes;
            const short sg_row = simd_lid / k_lanes;
            const int K = x_shape[x_ndim - 1];
            const int N = w_shape[0];
            const int out_row = int(tid.y) * 8 + int(simd_gid) * results_per_sg + int(sg_row);
            const int row = min(out_row, N - 1);
            const int in_w = K / 2;
            const int in_g = K / GS;
            const device uint8_t* wrow = (const device uint8_t*)w + row * in_w;
            const device half* srow = scales + row * in_g;
            const device half* brow = biases + row * in_g;
            const device float* xv[ROWS];
            #pragma clang loop unroll(full)
            for (int v = 0; v < ROWS; v++) {
              xv[v] = x + v * K;
            }
            float result[ROWS];
            #pragma clang loop unroll(full)
            for (int v = 0; v < ROWS; v++) result[v] = 0.0f;
            for (int g = k_lane; g < in_g; g += k_lanes) {
              const float scale = float(srow[g]);
              const float bias = float(brow[g]);
              const float sc1 = scale / 16.0f;
              #pragma clang loop unroll(full)
              for (int sc = 0; sc < GS / sub; sc++) {
                const int k0 = g * GS + sc * sub;
                const device uint8_t* wc = wrow + k0 / 2;
                float w_dq[sub];
                #pragma clang loop unroll(full)
                for (int i = 0; i < 4; i++) {
                  w_dq[2 * i] = scale * float(wc[i] & 0x0fu) + bias;
                  w_dq[2 * i + 1] = sc1 * float(wc[i] & 0xf0u) + bias;
                }
                #pragma clang loop unroll(full)
                for (int v = 0; v < ROWS; v++) {
                  const device float* xc = xv[v] + k0;
                  float acc = 0.0f;
                  #pragma clang loop unroll(full)
                  for (int i = 0; i < sub; i++) {
                    acc += xc[i] * w_dq[i];
                  }
                  result[v] += acc;
                }
              }
            }
            #pragma clang loop unroll(full)
            for (int v = 0; v < ROWS; v++) {
              result[v] += simd_shuffle_down(result[v], 4);
              result[v] += simd_shuffle_down(result[v], 2);
              result[v] += simd_shuffle_down(result[v], 1);
            }
            if (k_lane == 0 && out_row < N) {
              #pragma clang loop unroll(full)
              for (int v = 0; v < ROWS; v++) {
                y[v * N + out_row] = result[v];
              }
            }
            """)

    static let fastGateUpSilu = MLXFast.metalKernel(
        name: "qwen35_mtp_qmv_fast_gate_up_silu_w4_f16",
        inputNames: ["x", "wg", "sg", "bg", "wu", "su", "bu"], outputNames: ["y"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const int K = x_shape[x_ndim - 1];
            const int in_w = K / 2;
            const int in_g = K / GS;
            const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
            const device uint8_t* wgs = (const device uint8_t*)wg + out_row * in_w + simd_lid * 8;
            const device half* scg = sg + out_row * in_g + simd_lid / (GS / 16);
            const device half* big = bg + out_row * in_g + simd_lid / (GS / 16);
            const device uint8_t* wus = (const device uint8_t*)wu + out_row * in_w + simd_lid * 8;
            const device half* scu = su + out_row * in_g + simd_lid / (GS / 16);
            const device half* biu = bu + out_row * in_g + simd_lid / (GS / 16);
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float x_thread[16];
            float res_g[4] = {0, 0, 0, 0};
            float res_u[4] = {0, 0, 0, 0};
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += xv.x + xv.y + xv.z + xv.w;
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y * 0.0625f;
                x_thread[4 * i + 2] = xv.z * 0.00390625f;
                x_thread[4 * i + 3] = xv.w * 0.000244140625f;
              }
              for (int r = 0; r < 4; r++) {
                const device uint16_t* wlg = (const device uint16_t*)(wgs + r * in_w);
                const device uint16_t* wlu = (const device uint16_t*)(wus + r * in_w);
                const float s_g = float(scg[r * in_g]);
                const float b_g = float(big[r * in_g]);
                const float s_u = float(scu[r * in_g]);
                const float b_u = float(biu[r * in_g]);
                float acc_g = 0;
                float acc_u = 0;
                for (int i = 0; i < 4; i++) {
                  const uint16_t pg = wlg[i];
                  const uint16_t pu = wlu[i];
                  acc_g += (x_thread[4 * i] * float(pg & 0x000fu)
                      + x_thread[4 * i + 1] * float(pg & 0x00f0u)
                      + x_thread[4 * i + 2] * float(pg & 0x0f00u)
                      + x_thread[4 * i + 3] * float(pg & 0xf000u));
                  acc_u += (x_thread[4 * i] * float(pu & 0x000fu)
                      + x_thread[4 * i + 1] * float(pu & 0x00f0u)
                      + x_thread[4 * i + 2] * float(pu & 0x0f00u)
                      + x_thread[4 * i + 3] * float(pu & 0xf000u));
                }
                res_g[r] += s_g * acc_g + sum * b_g;
                res_u[r] += s_u * acc_u + sum * b_u;
              }
              wgs += 256; scg += 512 / GS; big += 512 / GS;
              wus += 256; scu += 512 / GS; biu += 512 / GS;
              xp4 += 128;
            }
            for (int r = 0; r < 4; r++) {
              const float g = simd_sum(res_g[r]);
              const float u = simd_sum(res_u[r]);
              if (simd_lid == 0) {
                #pragma clang fp contract(off)
                const float y_sig = 1.0f / (1.0f + metal::exp(metal::abs(g)));
                const float sig = (g < 0.0f) ? y_sig : (1.0f - y_sig);
                const float sg_val = g * sig;
                y[out_row + r] = sg_val * u;
              }
            }
            """)

    static let fastQKV = MLXFast.metalKernel(
        name: "qwen35_mtp_qmv_fast_qkv_w4_f16",
        inputNames: ["x", "wq", "sq", "bq", "wk", "sk", "bk", "wv", "sv", "bv"],
        outputNames: ["yq", "yk", "yv"],
        source: """
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const int K = x_shape[x_ndim - 1];
            const int in_w = K / 2;
            const int in_g = K / GS;
            const int Nq = wq_shape[0];
            const int Nk = wk_shape[0];
            const int global_row = int(tid.y) * 8 + int(simd_gid) * 4;
            const device uint8_t* w_base;
            const device half* s_base;
            const device half* b_base;
            device float* y_base;
            int out_row;
            if (global_row < Nq) {
              out_row = global_row;
              w_base = (const device uint8_t*)wq;
              s_base = sq; b_base = bq; y_base = yq;
            } else if (global_row < Nq + Nk) {
              out_row = global_row - Nq;
              w_base = (const device uint8_t*)wk;
              s_base = sk; b_base = bk; y_base = yk;
            } else {
              out_row = global_row - (Nq + Nk);
              w_base = (const device uint8_t*)wv;
              s_base = sv; b_base = bv; y_base = yv;
            }
            const device uint8_t* ws = w_base + out_row * in_w + simd_lid * 8;
            const device half* sc = s_base + out_row * in_g + simd_lid / (GS / 16);
            const device half* bi = b_base + out_row * in_g + simd_lid / (GS / 16);
            const device float4* xp4 = (const device float4*)(x + simd_lid * 16);
            float x_thread[16];
            float result[4] = {0, 0, 0, 0};
            for (int k = 0; k < K; k += 512) {
              float sum = 0;
              for (int i = 0; i < 4; ++i) {
                const float4 xv = xp4[i];
                sum += xv.x + xv.y + xv.z + xv.w;
                x_thread[4 * i] = xv.x;
                x_thread[4 * i + 1] = xv.y * 0.0625f;
                x_thread[4 * i + 2] = xv.z * 0.00390625f;
                x_thread[4 * i + 3] = xv.w * 0.000244140625f;
              }
              for (int r = 0; r < 4; r++) {
                const device uint16_t* wl = (const device uint16_t*)(ws + r * in_w);
                const float s = float(sc[r * in_g]);
                const float b = float(bi[r * in_g]);
                float accum = 0;
                for (int i = 0; i < 4; i++) {
                  const uint16_t packed_w = wl[i];
                  accum += (x_thread[4 * i] * float(packed_w & 0x000fu)
                      + x_thread[4 * i + 1] * float(packed_w & 0x00f0u)
                      + x_thread[4 * i + 2] * float(packed_w & 0x0f00u)
                      + x_thread[4 * i + 3] * float(packed_w & 0xf000u));
                }
                result[r] += s * accum + sum * b;
              }
              ws += 256;
              sc += 512 / GS;
              bi += 512 / GS;
              xp4 += 128;
            }
            for (int r = 0; r < 4; r++) {
              const float total = simd_sum(result[r]);
              if (simd_lid == 0) y_base[out_row + r] = total;
            }
            """)

    static let wideKV = MLXFast.metalKernel(
        name: "qwen35_mtp_qmv_wide_kv_w4_f16",
        inputNames: ["x", "wk", "sk", "bk", "wv", "sv", "bv"],
        outputNames: ["yk", "yv"],
        source: """
            constexpr int k_lanes = 8;
            constexpr int results_per_sg = 32 / k_lanes;
            constexpr int sub = 8;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint3 tid = threadgroup_position_in_grid;
            const short k_lane = simd_lid % k_lanes;
            const short sg_row = simd_lid / k_lanes;
            const int K = x_shape[x_ndim - 1];
            const int Nk = wk_shape[0];
            const int Nv = wv_shape[0];
            const int global_row = int(tid.y) * 8 + int(simd_gid) * results_per_sg + int(sg_row);
            const bool is_v = global_row >= Nk;
            const int N = is_v ? Nv : Nk;
            const int out_row = is_v ? (global_row - Nk) : global_row;
            const int row = min(out_row, N - 1);
            const int in_w = K / 2;
            const int in_g = K / GS;
            const device uint8_t* wrow = (const device uint8_t*)(is_v ? wv : wk) + row * in_w;
            const device half* srow = (is_v ? sv : sk) + row * in_g;
            const device half* brow = (is_v ? bv : bk) + row * in_g;
            device float* y_base = is_v ? yv : yk;
            const device float* xv[ROWS];
            #pragma clang loop unroll(full)
            for (int v = 0; v < ROWS; v++) {
              xv[v] = x + v * K;
            }
            float result[ROWS];
            #pragma clang loop unroll(full)
            for (int v = 0; v < ROWS; v++) result[v] = 0.0f;
            for (int g = k_lane; g < in_g; g += k_lanes) {
              const float scale = float(srow[g]);
              const float bias = float(brow[g]);
              const float sc1 = scale / 16.0f;
              #pragma clang loop unroll(full)
              for (int sc = 0; sc < GS / sub; sc++) {
                const int k0 = g * GS + sc * sub;
                const device uint8_t* wc = wrow + k0 / 2;
                float w_dq[sub];
                #pragma clang loop unroll(full)
                for (int i = 0; i < 4; i++) {
                  w_dq[2 * i] = scale * float(wc[i] & 0x0fu) + bias;
                  w_dq[2 * i + 1] = sc1 * float(wc[i] & 0xf0u) + bias;
                }
                #pragma clang loop unroll(full)
                for (int v = 0; v < ROWS; v++) {
                  const device float* xc = xv[v] + k0;
                  float acc = 0.0f;
                  #pragma clang loop unroll(full)
                  for (int i = 0; i < sub; i++) {
                    acc += xc[i] * w_dq[i];
                  }
                  result[v] += acc;
                }
              }
            }
            #pragma clang loop unroll(full)
            for (int v = 0; v < ROWS; v++) {
              result[v] += simd_shuffle_down(result[v], 4);
              result[v] += simd_shuffle_down(result[v], 2);
              result[v] += simd_shuffle_down(result[v], 1);
            }
            if (k_lane == 0 && out_row < N) {
              #pragma clang loop unroll(full)
              for (int v = 0; v < ROWS; v++) {
                y_base[v * N + out_row] = result[v];
              }
            }
            """)

    static func matmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int
    ) -> MLXArray {
        let N = weight.dim(0)
        return fast(
            [x, weight, scales, biases],
            template: [("GS", groupSize)],
            grid: (32, N / 8 * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [Array(x.shape.dropLast()) + [N]], outputDTypes: [.float32])[0]
    }

    static func matmulWide(
        _ x: MLXArray, rows: Int, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        groupSize: Int
    ) -> MLXArray {
        let N = weight.dim(0)
        return wide(
            [x, weight, scales, biases],
            template: [("GS", groupSize), ("ROWS", rows)],
            grid: (32, N / 8 * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [Array(x.shape.dropLast()) + [N]], outputDTypes: [.float32])[0]
    }

    static func gateUpSilu(
        _ x: MLXArray, gate: Qwen35MTPQuantizedLinear, up: Qwen35MTPQuantizedLinear
    ) -> MLXArray? {
        let K = x.dim(-1)
        let rows = x.size / K
        let N = gate.weight.dim(0)
        guard rows == 1, N == up.weight.dim(0), gate.groupSize == up.groupSize,
            gate.eligibleForFast(rows: 1), up.eligibleForFast(rows: 1),
            gate.bias == nil, up.bias == nil,
            let bg = gate.biases, let bu = up.biases
        else { return nil }
        return fastGateUpSilu(
            [x, gate.weight, gate.scales, bg, up.weight, up.scales, bu],
            template: [("GS", gate.groupSize)],
            grid: (32, N / 8 * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [Array(x.shape.dropLast()) + [N]], outputDTypes: [.float32])[0]
    }

    static func qkv(
        _ x: MLXArray, q: Qwen35MTPQuantizedLinear, k: Qwen35MTPQuantizedLinear,
        v: Qwen35MTPQuantizedLinear
    ) -> (MLXArray, MLXArray, MLXArray)? {
        let K = x.dim(-1)
        let rows = x.size / K
        guard rows == 1,
            q.groupSize == k.groupSize, q.groupSize == v.groupSize,
            q.eligibleForFast(rows: 1), k.eligibleForFast(rows: 1), v.eligibleForFast(rows: 1),
            q.bias == nil, k.bias == nil, v.bias == nil,
            let bq = q.biases, let bk = k.biases, let bv = v.biases
        else { return nil }
        let Nq = q.weight.dim(0)
        let Nk = k.weight.dim(0)
        let Nv = v.weight.dim(0)
        let prefix = Array(x.shape.dropLast())
        let outs = fastQKV(
            [x, q.weight, q.scales, bq, k.weight, k.scales, bk, v.weight, v.scales, bv],
            template: [("GS", q.groupSize)],
            grid: (32, (Nq + Nk + Nv) / 8 * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [prefix + [Nq], prefix + [Nk], prefix + [Nv]],
            outputDTypes: [.float32, .float32, .float32])
        return (outs[0], outs[1], outs[2])
    }

    static func kvWide(
        _ x: MLXArray, k: Qwen35MTPQuantizedLinear, v: Qwen35MTPQuantizedLinear
    ) -> (MLXArray, MLXArray)? {
        let K = x.dim(-1)
        let rows = x.size / K
        guard (2 ... maximumWideRows).contains(rows),
            k.groupSize == v.groupSize,
            k.eligibleForFast(rows: rows), v.eligibleForFast(rows: rows),
            k.bias == nil, v.bias == nil,
            let bk = k.biases, let bv = v.biases
        else { return nil }
        let Nk = k.weight.dim(0)
        let Nv = v.weight.dim(0)
        let prefix = Array(x.shape.dropLast())
        let outs = wideKV(
            [x, k.weight, k.scales, bk, v.weight, v.scales, bv],
            template: [("GS", k.groupSize), ("ROWS", rows)],
            grid: (32, (Nk + Nv) / 8 * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [prefix + [Nk], prefix + [Nv]],
            outputDTypes: [.float32, .float32])
        return (outs[0], outs[1])
    }
}

final class Qwen35MTPRMSNorm: RMSNorm {
    private let weightCastCache = ConstantArrayCastCache()

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer { weightCastCache.clear() }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = x.dtype == .float32 && weight.dtype == .float16
            ? (weightCastCache.cachedCast(weight, to: .float32, allowFloat16: true) ?? weight)
            : weight
        return MLXFast.rmsNorm(x, weight: w, eps: eps)
    }
}

private enum Qwen35MTPNormPair {
    nonisolated(unsafe) static var verified: Bool?

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_mtp_norm_concat_4096",
        inputNames: ["e", "we", "h", "wh"], outputNames: ["out"],
        source: """
            constexpr int N = 4096;
            const uint tid = thread_position_in_threadgroup.x;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint row = threadgroup_position_in_grid.y;
            const device float4* e4 = (const device float4*)(e + row * N);
            const device float4* h4 = (const device float4*)(h + row * N);
            const device half4* we4 = (const device half4*)we;
            const device half4* wh4 = (const device half4*)wh;
            device float4* out_e = (device float4*)(out + row * (2 * N));
            device float4* out_h = out_e + (N / 4);
            #pragma clang fp contract(off)
            const float4 ev = e4[tid];
            const float4 hv = h4[tid];
            float se = ((ev.x * ev.x + ev.y * ev.y) + ev.z * ev.z) + ev.w * ev.w;
            float sh = ((hv.x * hv.x + hv.y * hv.y) + hv.z * hv.z) + hv.w * hv.w;
            se = simd_sum(se);
            sh = simd_sum(sh);
            threadgroup float lse[32];
            threadgroup float lsh[32];
            if (simd_lid == 0) {
                lse[simd_gid] = se;
                lsh[simd_gid] = sh;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_gid == 0) {
                se = simd_sum(lse[simd_lid]);
                sh = simd_sum(lsh[simd_lid]);
                if (simd_lid == 0) {
                    lse[0] = metal::precise::rsqrt(se * (1.0f / 4096.0f) + 1e-6f);
                    lsh[0] = metal::precise::rsqrt(sh * (1.0f / 4096.0f) + 1e-6f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv_e = lse[0];
            const float inv_h = lsh[0];
            out_e[tid] = float4(we4[tid]) * (ev * inv_e);
            out_h[tid] = float4(wh4[tid]) * (hv * inv_h);
            """)

    static func fusedConcat(
        embeds: MLXArray, normEmbeds: RMSNorm,
        hidden: MLXArray, normHidden: RMSNorm
    ) -> MLXArray? {
        guard verified != false,
            embeds.ndim == 3, hidden.ndim == 3,
            embeds.shape == hidden.shape,
            embeds.dim(-1) == 4096,
            embeds.dtype == .float32, hidden.dtype == .float32,
            normEmbeds.weight.dtype == .float16, normHidden.weight.dtype == .float16,
            abs(normEmbeds.eps - 1e-6) < 1e-12, normEmbeds.eps == normHidden.eps
        else { return nil }
        let rows = embeds.dim(0) * embeds.dim(1)
        guard rows >= 1 else { return nil }
        let fastOut = kernel(
            [embeds, normEmbeds.weight, hidden, normHidden.weight],
            grid: (1024, rows, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[embeds.dim(0), embeds.dim(1), 8192]],
            outputDTypes: [.float32])[0]
        if verified == true {
            return fastOut
        }
        let refOut = concatenated([normEmbeds(embeds), normHidden(hidden)], axis: -1)
        eval(fastOut, refOut)
        let same = fastOut.asArray(Float.self).map(\Float.bitPattern)
            == refOut.asArray(Float.self).map(\Float.bitPattern)
        verified = same
        return same ? fastOut : refOut
    }
}

// The pinned MTP head carries FP16 affine metadata while the target feeds it
// FP32 activations. Reuse only the exact constant widening that quantizedMM
// would otherwise repeat at every draft step, or read FP16 scales/biases in
// place for 1-row and short-window steps once verified bit-identical. This
// cache is input-independent; the packed weights and the module's original
// parameter tree remain unchanged.
final class Qwen35MTPQuantizedLinear: QuantizedLinear {
    private let mtpScaleCastCache = ConstantArrayCastCache()
    private let mtpOffsetCastCache = ConstantArrayCastCache()
    private var verifiedOneRowFast: Bool?
    private var verifiedWideFast: [Int: Bool] = [:]

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer {
            mtpScaleCastCache.clear()
            mtpOffsetCastCache.clear()
            verifiedOneRowFast = nil
            verifiedWideFast.removeAll()
        }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    func eligibleForFast(rows: Int) -> Bool {
        guard mode == .affine, scales.dtype == .float16,
            let biases, biases.dtype == .float16,
            bits == 4, (groupSize == 64 || groupSize == 128),
            (weight.dim(1) * 8) % 512 == 0, weight.dim(0) % 8 == 0
        else { return false }
        if rows == 1 { return verifiedOneRowFast == true }
        return verifiedWideFast[rows] == true
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard mode == .affine, x.dtype == .float32, scales.dtype == .float16 else {
            return super.callAsFunction(x)
        }
        let K = x.dim(-1)
        let rows = x.size / K
        if rows == 1, bits == 4, (groupSize == 64 || groupSize == 128),
            K % 512 == 0, weight.dim(0) % 8 == 0,
            let biases, biases.dtype == .float16,
            verifiedOneRowFast != false
        {
            let fastOut = Qwen35MTPFloat16ConstantQMV.matmul(
                x, weight: weight, scales: scales, biases: biases, groupSize: groupSize)
            if verifiedOneRowFast == true {
                return bias.map { fastOut + $0 } ?? fastOut
            }
            let widenedScales = mtpScaleCastCache.cachedCast(
                scales, to: x.dtype, allowFloat16: true) ?? scales
            let widenedOffsets = mtpOffsetCastCache.cachedCast(
                biases, to: x.dtype, allowFloat16: true) ?? biases
            let refOut = quantizedMM(
                x, weight, scales: widenedScales, biases: widenedOffsets, transpose: true,
                groupSize: groupSize, bits: bits, mode: mode)
            eval(fastOut, refOut)
            let same = fastOut.asArray(Float.self).map(\.bitPattern)
                == refOut.asArray(Float.self).map(\.bitPattern)
            verifiedOneRowFast = same
            let chosen = same ? fastOut : refOut
            return bias.map { chosen + $0 } ?? chosen
        }
        if (2 ... Qwen35MTPFloat16ConstantQMV.maximumWideRows).contains(rows),
            bits == 4, (groupSize == 64 || groupSize == 128),
            K % 512 == 0, weight.dim(0) % 8 == 0,
            let biases, biases.dtype == .float16,
            verifiedWideFast[rows] != false
        {
            let fastOut = Qwen35MTPFloat16ConstantQMV.matmulWide(
                x, rows: rows, weight: weight, scales: scales, biases: biases,
                groupSize: groupSize)
            if verifiedWideFast[rows] == true {
                return bias.map { fastOut + $0 } ?? fastOut
            }
            let widenedScales = mtpScaleCastCache.cachedCast(
                scales, to: x.dtype, allowFloat16: true) ?? scales
            let widenedOffsets = mtpOffsetCastCache.cachedCast(
                biases, to: x.dtype, allowFloat16: true) ?? biases
            let refOut = quantizedMM(
                x, weight, scales: widenedScales, biases: widenedOffsets, transpose: true,
                groupSize: groupSize, bits: bits, mode: mode)
            eval(fastOut, refOut)
            let same = fastOut.asArray(Float.self).map(\.bitPattern)
                == refOut.asArray(Float.self).map(\.bitPattern)
            verifiedWideFast[rows] = same
            let chosen = same ? fastOut : refOut
            return bias.map { chosen + $0 } ?? chosen
        }
        let widenedScales = mtpScaleCastCache.cachedCast(
            scales, to: x.dtype, allowFloat16: true) ?? scales
        let widenedOffsets = biases.map {
            mtpOffsetCastCache.cachedCast($0, to: x.dtype, allowFloat16: true) ?? $0
        }
        var output = quantizedMM(
            x, weight, scales: widenedScales, biases: widenedOffsets, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
        if let bias { output = output + bias }
        return output
    }
}

// MARK: - Module-level MTP flag

/// Controls whether Qwen3.5/3.6 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _qwen35MTPEnabled: Bool = false

/// Fail-loud errors for the production-safe inline Qwen MTP loader.
///
/// Unlike the legacy process-global attachment flag above, this loader builds
/// an assistant explicitly for one already-loaded target and never changes
/// model construction behavior process-wide.
public enum Qwen35InlineMTPError: Error, LocalizedError, Sendable, Equatable {
    case invalidConfiguration(String)
    case incompatibleTarget(field: String, artifact: Int, target: Int)
    case invalidWeightIndex(String)
    case missingWeights
    case duplicateWeight(String)
    case missingQuantization(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid inline Qwen MTP configuration: \(detail)."
        case .incompatibleTarget(let field, let artifact, let target):
            return "Inline Qwen MTP target mismatch at \(field): artifact=\(artifact), target=\(target)."
        case .invalidWeightIndex(let detail):
            return "Invalid inline Qwen MTP weight index: \(detail)."
        case .missingWeights:
            return "The checkpoint declares inline Qwen MTP but contains no matching tensors."
        case .duplicateWeight(let key):
            return "The inline Qwen MTP tensor \(key) appears more than once."
        case .missingQuantization(let path):
            return "The quantized inline Qwen MTP module \(path) has no matching quantization entry."
        }
    }
}

/// Parsed, bounded metadata for an inline Qwen MTP assistant.
/// Internal so focused tests can validate checkpoint interpretation without
/// constructing hundreds of MiB of weights.
struct Qwen35InlineMTPMetadata: Sendable {
    let textConfiguration: Qwen35TextConfiguration
    let prefix: String
    let blockSize: Int
    let quantization: BaseConfiguration.PerLayerQuantization?

    func resolvedQuantization(
        for path: String
    ) -> BaseConfiguration.Quantization? {
        quantization?.quantization(layer: path)
    }
}

// MARK: - MTP history KV

extension Qwen35Attention {
    /// Append committed proposal-head history without computing unused query,
    /// gate, attention, or output-projection rows.
    func appendMTPHistoryKV(_ x: MLXArray, cache: any KVCache) {
        let batch = x.dim(0)
        let length = x.dim(1)
        let kRaw: MLXArray
        let vRaw: MLXArray
        if let kp = kProj as? Qwen35MTPQuantizedLinear,
            let vp = vProj as? Qwen35MTPQuantizedLinear,
            let fused = Qwen35MTPFloat16ConstantQMV.kvWide(x, k: kp, v: vp)
        {
            (kRaw, vRaw) = fused
        } else {
            kRaw = kProj(x)
            vRaw = vProj(x)
        }
        var keys = kNorm(kRaw.reshaped(batch, length, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vRaw.reshaped(batch, length, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        _ = cache.update(keys: keys, values: values)
    }

    func callMTP1Row(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let qProjOutput: MLXArray
        var keys: MLXArray
        var values: MLXArray
        if L == 1,
            let qp = qProj as? Qwen35MTPQuantizedLinear,
            let kp = kProj as? Qwen35MTPQuantizedLinear,
            let vp = vProj as? Qwen35MTPQuantizedLinear,
            let fused = Qwen35MTPFloat16ConstantQMV.qkv(x, q: qp, k: kp, v: vp)
        {
            (qProjOutput, keys, values) = fused
        } else {
            return callAsFunction(x, mask: mask, cache: cache)
        }
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        let gate = qSplit[1].reshaped(B, L, -1)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return oProj(sigmoidMultiply(output, gate))
    }

    func callMTPLastPositionOnly(
        _ normed: MLXArray, normedLast: MLXArray, cache: any KVCache
    ) -> MLXArray {
        let B = normed.dim(0)
        let L = normed.dim(1)
        let qProjOutput = qProj(normedLast)
        let qSplit = qProjOutput.reshaped(B, 1, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        let gate = qSplit[1].reshaped(B, 1, -1)

        let kRaw: MLXArray
        let vRaw: MLXArray
        if let kp = kProj as? Qwen35MTPQuantizedLinear,
            let vp = vProj as? Qwen35MTPQuantizedLinear,
            let fused = Qwen35MTPFloat16ConstantQMV.kvWide(normed, k: kp, v: vp)
        {
            (kRaw, vRaw) = fused
        } else {
            kRaw = kProj(normed)
            vRaw = vProj(normed)
        }
        var keys = kNorm(kRaw.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vRaw.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let baseOffset = cache.offset
        queries = rope(queries, offset: baseOffset + L - 1)
        keys = rope(keys, offset: baseOffset)
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, 1, -1)
        return oProj(sigmoidMultiply(output, gate))
    }
}

// MARK: - MTPDecoderLayer

/// Full-attention transformer layer used inside the Qwen3.5/3.6 MTP head.
/// Unlike `Qwen35DecoderLayer`, this always uses full attention (never SSM/linear).
/// MoE config is honoured when `num_experts > 0`.
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPDecoderLayer
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module
    private var verifiedMLP1Row: Bool?

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        let attn = Qwen35Attention(args)
        try? attn.update(
            modules: ModuleChildren(values: [
                "q_norm": .value(Qwen35MTPRMSNorm(dimensions: headDim, eps: args.rmsNormEps)),
                "k_norm": .value(Qwen35MTPRMSNorm(dimensions: headDim, eps: args.rmsNormEps)),
            ]),
            verify: [])
        _selfAttn.wrappedValue = attn
        if args.numExperts > 0 {
            // Split gate/up: the assistant's quantization table
            // (`mtplx_mtp_quantization`) and the checkpoint's mtp.* tensors
            // are keyed on the split module paths, and the MTP head loads
            // outside the target sanitizers that perform gate/up fusion.
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args, fuseGateUp: false)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = Qwen35MTPRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Qwen35MTPRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    private func callMLP(_ x: MLXArray) -> MLXArray {
        if x.dim(1) == 1, verifiedMLP1Row != false,
            let denseMLP = mlp as? Qwen3NextMLP,
            let gate = denseMLP.gateProj as? Qwen35MTPQuantizedLinear,
            let up = denseMLP.upProj as? Qwen35MTPQuantizedLinear,
            let gated = Qwen35MTPFloat16ConstantQMV.gateUpSilu(x, gate: gate, up: up)
        {
            if verifiedMLP1Row == true {
                return denseMLP.downProj(gated)
            }
            let refGated = silu(denseMLP.gateProj(x)) * denseMLP.upProj(x)
            eval(gated, refGated)
            let same = gated.asArray(Float.self).map(\.bitPattern)
                == refGated.asArray(Float.self).map(\.bitPattern)
            verifiedMLP1Row = same
            return denseMLP.downProj(same ? gated : refGated)
        }
        return (mlp as! UnaryLayer)(x)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        let normed = inputLayerNorm(x)
        let r = x.dim(1) == 1
            ? selfAttn.callMTP1Row(normed, mask: mask, cache: cache)
            : selfAttn(normed, mask: mask, cache: cache)
        let h = x + r
        return h + callMLP(postAttentionLayerNorm(h))
    }

    func callLastPositionOnly(_ x: MLXArray, cache: any KVCache) -> MLXArray {
        let L = x.dim(1)
        let normed = inputLayerNorm(x)
        let normedLast = normed[0..., (L - 1)..., 0...]
        let xLast = x[0..., (L - 1)..., 0...]
        let r = selfAttn.callMTPLastPositionOnly(normed, normedLast: normedLast, cache: cache)
        let h = xLast + r
        return h + callMLP(postAttentionLayerNorm(h))
    }

    /// Populate this layer's K/V history without computing a dead decoder
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendMTPHistoryKV(inputLayerNorm(x), cache: cache)
    }
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's final-normalized hidden state at position t with the
/// embedding of the sampled main token (t+1) to predict the draft token at (t+2).
///
/// Architecture (port of PR #990):
/// ```
/// pre_fc_norm_hidden:    RMSNorm(hidden_size)
/// pre_fc_norm_embedding: RMSNorm(hidden_size)
/// fc:                    Linear(hidden_size * 2 → hidden_size, bias: false)
/// layers:                [MTPDecoderLayer]  × mtp_num_hidden_layers
/// norm:                  RMSNorm(hidden_size)
/// ```
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPModule
final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // `layers` uses the default ModuleInfo key derived from the property name.
    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = Qwen35MTPRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = Qwen35MTPRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        self.norm = Qwen35MTPRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    /// An UNQUANTIZED head computes in its own weight dtype. The Bonsai 2 9B
    /// `mtp/` drafter is FP16 and the Prism trunk hands it FP32 activations;
    /// a mixed-dtype matmul promotes, and so copies, every head weight on
    /// every draft step. The activations entering the head are cast instead.
    /// The weights are never cast, and a quantized head keeps the incoming
    /// dtype. Only the draft numerics move; the target verifies every token.
    private func entering(_ x: MLXArray) -> MLXArray {
        guard !(fc is QuantizedLinear), x.dtype != fc.weight.dtype else { return x }
        return x.asType(fc.weight.dtype)
    }

    private func fuseNormed(embeds: MLXArray, hidden: MLXArray) -> MLXArray {
        let eIn = entering(embeds)
        let hIn = entering(hidden)
        if let pair = Qwen35MTPNormPair.fusedConcat(
            embeds: eIn, normEmbeds: preFcNormEmbedding,
            hidden: hIn, normHidden: preFcNormHidden)
        {
            return fc(pair)
        }
        let e = preFcNormEmbedding(eIn)
        let h = preFcNormHidden(hIn)
        return fc(concatenated([e, h], axis: -1))
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        let embeds = embedTokens(nextTokenIds)
        var fused = fuseNormed(embeds: embeds, hidden: hidden)

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }

        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        return norm(fused)
    }

    /// Append every leading committed row through a K/V-only path and compute
    /// a full decoder output only for the final proposal row. Multi-layer heads
    /// fail closed before any cache mutation because later layers require the
    /// omitted leading outputs.
    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
            hidden.dim(1) > 1,
            nextTokenIds.dim(1) == hidden.dim(1)
        else { return nil }

        let embeds = embedTokens(nextTokenIds)
        let fused = fuseNormed(embeds: embeds, hidden: hidden)
        if (2 ... Qwen35MTPFloat16ConstantQMV.maximumWideRows).contains(fused.dim(1)) {
            return norm(layers[0].callLastPositionOnly(fused, cache: cache[0]))
        }
        let historyCount = fused.dim(1) - 1
        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }
}

// MARK: - Artifact-scoped inline assistant

/// An explicitly loaded Qwen3.5/3.6 MTP assistant bound to one target.
///
/// The assistant owns only the tensors below `mtp.*`. Target embeddings and
/// the LM head are called through the bound target, so loading this object does
/// not duplicate the target checkpoint and does not require
/// `_qwen35MTPEnabled`.
public final class Qwen35InlineMTPAssistant: Module, @unchecked Sendable {
    static let cacheAllocationStep = 256
    /// The longest draft chain: the head is applied to its own output at most
    /// this many times in one round. David ruled 6 for the Bonsai 2 9B MLX
    /// track (2026-10-14), the depth the CUDA track serves. A fixed declared
    /// depth above it is refused by the round driver, never clamped.
    public static let maximumDepth = 6

    private let mtp: Qwen35MTPModule
    private let target: Qwen35TextModel
    private let installedVerificationMode: CBv2MTPVerificationMode
    /// Draft-only output head over a fixed slice of the target vocabulary.
    /// nil (small fixtures, tied or unpacked heads) keeps the full head.
    private let draftVocabulary: Qwen35MTPDraftVocabulary?

    public let blockSize: Int
    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }

    var prefixCheckpointGeometry: (width: Int, dtype: DType, vocabulary: Int, verification: String) {
        (target.configuration.hiddenSize, target.model.norm.weight.dtype,
         target.vocabularySize, installedVerificationMode.rawValue)
    }

    private init(
        configuration: Qwen35TextConfiguration,
        blockSize: Int,
        target: Qwen35TextModel,
        verificationMode: CBv2MTPVerificationMode?
    ) {
        self.mtp = Qwen35MTPModule(configuration)
        self.blockSize = blockSize
        self.target = target
        self.draftVocabulary = Qwen35MTPDraftVocabulary(target: target)
        self.installedVerificationMode = Self.resolvedVerificationMode(
            requested: verificationMode,
            forceSerialEnvironment: Self.forceSerialVerification)
        super.init()
    }

    static func resolvedVerificationMode(
        requested: CBv2MTPVerificationMode?, forceSerialEnvironment: Bool
    ) -> CBv2MTPVerificationMode {
        if forceSerialEnvironment { return .serialTarget }
        return requested ?? .rectangular
    }

    /// Allocate the assistant's own autoregressive full-attention KV.
    /// Target recurrent and attention state is never reused as assistant KV.
    public func makeCache() -> [any KVCache] {
        mtp.layers.map { _ in
            let cache = KVCacheSimple()
            cache.step = Self.cacheAllocationStep
            return cache as any KVCache
        }
    }

    /// Advance the assistant by one or more already-chosen target/draft tokens.
    /// Returns logits from the target's shared output projection and the
    /// assistant hidden used to continue a draft chain.
    public func forward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let output = mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
        return (headLogits(output), output)
    }

    /// The target's shared output projection over assistant hidden states.
    func headLogits(_ output: MLXArray) -> MLXArray {
        if target.configuration.tieWordEmbeddings {
            return target.model.embedTokens.asLinear(output)
        }
        return target.lmHead!(output)
    }

    /// Apply the exact final norm of the bound target once at the boundary
    /// where target-derived hidden enters assistant history. Target hidden
    /// capture is deliberately pre-norm; recursive assistant hidden is already
    /// post-`mtp.norm` and must never pass through this helper.
    func targetFinalNorm(_ hidden: MLXArray) -> MLXArray {
        target.model.norm(hidden)
    }

    /// Assistant hidden states WITHOUT the output projection — the draft
    /// path applies the head to the last position only (or to a shortlist).
    func moduleForward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray {
        mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
    }

    /// History-flush specialization. Returns one final hidden row while every
    /// leading trusted row contributes only K/V state. nil means the head
    /// geometry cannot safely omit intermediate layer outputs.
    func moduleLastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray? {
        mtp.lastHiddenWithKVOnlyHistory(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
    }

    /// Draft logits over an engine-provided shortlist of token ids. The
    /// draft needs only an argmax, so score the gathered head rows instead
    /// of streaming the full `[V, H]` output projection (~K/V of the bytes;
    /// the 4-bit Qwen3.6 lm_head alone is ~286 MB per read). Quantized
    /// heads gather packed rows plus their scales/biases and stay on the
    /// quantized-matmul path; float heads (small test fixtures) gather
    /// plain rows.
    ///
    /// A PACKED HADAMARD head folds the transform into its rows, so the
    /// rotation the module applies in `callAsFunction` has to be applied to
    /// the hidden state here as well. `HadamardQuantizedLinear` IS a
    /// `QuantizedLinear`, so without the rotation the cast below succeeds and
    /// scores folded rows against an unrotated hidden state.
    /// `HadamardQuantizedEmbedding` is NOT a `QuantizedEmbedding`, so without
    /// its own branch the tied path reads packed integers as floats.
    func shortlistLogits(hidden: MLXArray, ids: MLXArray) -> MLXArray {
        if target.configuration.tieWordEmbeddings {
            let embed = target.model.embedTokens
            if let packed = embed as? HadamardQuantizedEmbedding {
                return quantizedMM(
                    packed.transform(hidden), packed.weight[ids],
                    scales: packed.scales[ids],
                    biases: packed.biases.map { $0[ids] },
                    transpose: true,
                    groupSize: packed.groupSize, bits: packed.bits,
                    mode: packed.mode)
            }
            if let quantized = embed as? QuantizedEmbedding {
                return quantizedMM(
                    hidden, quantized.weight[ids],
                    scales: quantized.scales[ids],
                    biases: quantized.biases.map { $0[ids] },
                    transpose: true,
                    groupSize: quantized.groupSize, bits: quantized.bits,
                    mode: quantized.mode)
            }
            return matmul(hidden, embed.weight[ids].transposed(1, 0))
        }
        let head = target.lmHead!
        let projected = (head as? HadamardQuantizedLinear).map { $0.transform(hidden) } ?? hidden
        if let quantized = head as? QuantizedLinear {
            var logits = quantizedMM(
                projected, quantized.weight[ids],
                scales: quantized.scales[ids],
                biases: quantized.biases.map { $0[ids] },
                transpose: true,
                groupSize: quantized.groupSize, bits: quantized.bits,
                mode: quantized.mode)
            if let bias = quantized.bias { logits = logits + bias[ids] }
            return logits
        }
        var logits = matmul(projected, head.weight[ids].transposed(1, 0))
        if let bias = head.bias { logits = logits + bias[ids] }
        return logits
    }

    /// Strictly load either an inline assistant declared by a combined
    /// checkpoint or a standalone `qwen3_5_mtp` artifact. Inline artifacts
    /// read only indexed keys below their declared prefix; standalone
    /// artifacts read every tensor from their bounded safetensors directory.
    public static func load(
        from modelDirectory: URL,
        target: any LanguageModel,
        verificationMode: CBv2MTPVerificationMode? = nil
    ) throws -> Qwen35InlineMTPAssistant {
        let serving = target
        let target = try qwen35TextTarget(target)
        let resolvedVerificationMode = Self.resolvedVerificationMode(
            requested: verificationMode,
            forceSerialEnvironment: Self.forceSerialVerification)
        if resolvedVerificationMode == .rectangularExact && !target.model.exactTargetVerify {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "rectangular_exact verification requires exact target arithmetic")
        }
        let metadata = try loadMetadata(from: modelDirectory)
        try validate(metadata.textConfiguration, against: target.configuration)

        let indexed: [String: MLXArray]
        if metadata.prefix.isEmpty {
            indexed = try loadStandaloneWeights(from: modelDirectory)
        } else {
            indexed = try loadIndexedWeights(
                from: modelDirectory, prefix: metadata.prefix)
        }
        let scaledPaths = Set(indexed.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        })
        for path in scaledPaths
        where metadata.resolvedQuantization(for: path) == nil {
            throw Qwen35InlineMTPError.missingQuantization(path)
        }

        var configuration = metadata.textConfiguration
        configuration.mtpNumHiddenLayers = try decoderLayerCount(
            in: indexed.keys, declared: metadata.textConfiguration.mtpNumHiddenLayers)
        let assistant = Qwen35InlineMTPAssistant(
            configuration: configuration,
            blockSize: metadata.blockSize,
            target: target,
            verificationMode: verificationMode)
        if !scaledPaths.isEmpty {
            quantize(
                model: assistant.mtp,
                filter: { path, _ in
                    guard scaledPaths.contains(path) else { return nil }
                    return metadata.resolvedQuantization(for: path)?.asTuple
                },
                apply: { module, groupSize, bits, mode in
                    guard let linear = module as? Linear, !(linear is QuantizedLinear) else {
                        return quantizeSingle(
                            layer: module, groupSize: groupSize, bits: bits, mode: mode)
                    }
                    return Qwen35MTPQuantizedLinear(
                        weight: linear.weight, bias: linear.bias,
                        groupSize: groupSize, bits: bits, mode: mode)
                })
        }

        try assistant.mtp.update(
            parameters: ModuleParameters.unflattened(indexed), verify: [.all])
        eval(assistant.mtp)
        requantizeHeadLinears(in: assistant.mtp)
        // Warm after re-quantization, so the warm round compiles the
        // kernels of the head geometry the scored rounds run.
        assistant.warmEngineRound(serving: serving)
        // Once more at the resident's boot warm, after the runner has adopted
        // the model (27b 76841fed: a load-time round alone left part of the
        // first timed round's cost in place in some processes).
        CBv2DeferredLoadWarm.register { [weak assistant] in
            assistant?.warmEngineRound(serving: serving)
        }
        return assistant
    }

    /// On-load head re-encoding, permitted by the head re-quantization
    /// exception (participant contract section 4.4): in memory only, within
    /// the declaration cap, after the export's tensors were bound at their
    /// own geometry. The export on disk and every target tensor — the shared
    /// embedding, the lm_head, and the draft-vocabulary copy of its rows —
    /// stay untouched. The head only proposes; re-encoding can move draft
    /// choices and therefore acceptance, never bypass target verification.
    static let headRequantGeometry: (groupSize: Int, bits: Int)? = (128, 4)

    private static func requantizeHeadLinears(in head: Qwen35MTPModule) {
        guard let geometry = headRequantGeometry else { return }
        quantize(
            model: head,
            filter: { _, module in
                guard let linear = module as? Qwen35MTPQuantizedLinear,
                    linear.mode == .affine,
                    linear.groupSize != geometry.groupSize || linear.bits != geometry.bits,
                    (linear.weight.dim(1) * 32 / linear.bits) % geometry.groupSize == 0
                else { return nil }
                return (geometry.groupSize, geometry.bits, QuantizationMode.affine)
            },
            apply: { module, groupSize, bits, mode in
                guard let linear = module as? Qwen35MTPQuantizedLinear else { return nil }
                let dense = dequantized(
                    linear.weight, scales: linear.scales, biases: linear.biases,
                    groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode,
                    dtype: .float16)
                let (weight, scales, biases) = quantized(
                    dense, groupSize: groupSize, bits: bits, mode: mode)
                let replacement = Qwen35MTPQuantizedLinear(
                    weight: weight, bias: linear.bias, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: mode)
                // Materialize the packed replacement now so the dense FP16
                // intermediate frees before the next module dequantizes.
                eval(replacement)
                return replacement
            })
    }

    private static func qwen35TextTarget(
        _ target: any LanguageModel
    ) throws -> Qwen35TextModel {
        if let target = target as? Qwen35TextModel { return target }
        if let target = target as? Qwen35Model { return target.languageModel }
        throw Qwen35InlineMTPError.invalidConfiguration(
            "target type \(String(describing: type(of: target))) is not Qwen3.5/3.6")
    }

    private static let globalQuantizationKeys: Set<String> = [
        "group_size", "bits", "mode"
    ]

    /// Scalar keys understood or deliberately ignored by
    /// `BaseConfiguration.QuantizationContainer`; all other keys name modules.
    private static let quantizationMetadataKeys =
        globalQuantizationKeys.union([
            "quant_method", "linear_class", "quantization_mode"
        ])

    private static func selectedJSONObject(
        in root: [String: Any],
        primaryKey: String,
        fallbackKey: String
    ) throws -> [String: Any]? {
        for key in [primaryKey, fallbackKey] {
            guard let raw = root[key], !(raw is NSNull) else { continue }
            guard let object = raw as? [String: Any] else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "standalone MTP quantization must be an object")
            }
            return object
        }
        return nil
    }

    private static func decodeQuantization(
        _ rawQuantization: [String: Any]
    ) throws -> BaseConfiguration.PerLayerQuantization {
        var perLayer = [String: BaseConfiguration.QuantizationOption]()
        for (path, raw) in rawQuantization
        where !quantizationMetadataKeys.contains(path) {
            if CFGetTypeID(raw as CFTypeRef) == CFBooleanGetTypeID() {
                guard (raw as? Bool) == false else {
                    throw Qwen35InlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                perLayer[path] = .skip
            } else {
                guard let object = raw as? [String: Any] else {
                    throw Qwen35InlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                let data = try JSONSerialization.data(withJSONObject: object)
                perLayer[path] = .quantize(
                    try JSONDecoder().decode(
                        BaseConfiguration.Quantization.self, from: data))
            }
        }

        let hasGlobalQuantization = rawQuantization.keys.contains {
            globalQuantizationKeys.contains($0)
        }
        let global: BaseConfiguration.Quantization?
        if hasGlobalQuantization {
            let data = try JSONSerialization.data(withJSONObject: rawQuantization)
            global = try JSONDecoder().decode(
                BaseConfiguration.Quantization.self, from: data)
        } else {
            global = nil
        }
        return BaseConfiguration.PerLayerQuantization(
            quantization: global, perLayerQuantization: perLayer)
    }

    static func loadMetadata(from directory: URL) throws -> Qwen35InlineMTPMetadata {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = root["text_config"] as? [String: Any]
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "text_config is required")
        }

        let prefix: String
        let blockSize: Int
        let rawQuantization: [String: Any]?
        if let inline = root["mtplx_mtp"] as? [String: Any],
            inline["included"] as? Bool == true
        {
            prefix = (inline["prefix"] as? String) ?? "mtp."
            guard prefix == "mtp." else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "only the mtp. prefix is supported")
            }
            blockSize = (inline["block_size"] as? NSNumber)?.intValue ?? 3
            guard let quantization = root["mtplx_mtp_quantization"] as? [String: Any]
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "mtplx_mtp_quantization is required")
            }
            rawQuantization = quantization
        } else {
            guard (root["model_type"] as? String)?.lowercased() == "qwen3_5_mtp"
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "mtplx_mtp.included=true or model_type=qwen3_5_mtp is required")
            }
            prefix = ""
            blockSize = (root["block_size"] as? NSNumber)?.intValue ?? 3
            rawQuantization = try selectedJSONObject(
                in: root,
                primaryKey: "quantization",
                fallbackKey: "quantization_config")
        }
        guard (2...8).contains(blockSize) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "block_size \(blockSize) is outside 2...8")
        }
        let textData = try JSONSerialization.data(withJSONObject: text)
        let configuration = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: textData)
        // Zero is legal here: a standalone export can restate the TRUNK's
        // text_config, whose mtp_num_hidden_layers is 0 (the prism-ml Bonsai
        // 2 9B `mtp/` drafter does). `load` reads the depth off the tensors.
        guard (0...4).contains(configuration.mtpNumHiddenLayers) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "mtp_num_hidden_layers must be within 0...4")
        }

        let quantization: BaseConfiguration.PerLayerQuantization?
        if let rawQuantization, !rawQuantization.isEmpty {
            quantization = try decodeQuantization(rawQuantization)
        } else {
            quantization = nil
        }
        return Qwen35InlineMTPMetadata(
            textConfiguration: configuration,
            prefix: prefix,
            blockSize: blockSize,
            quantization: quantization)
    }

    /// The head's decoder depth, read off its tensors: the `layers.<i>.`
    /// indices must be exactly `0..<n`, with n within 1...4. A non-zero
    /// `mtp_num_hidden_layers` must agree with them. Zero means the export
    /// restated the trunk's text_config and declares no head depth.
    static func decoderLayerCount(
        in keys: some Sequence<String>, declared: Int
    ) throws -> Int {
        var indices = Set<Int>()
        for key in keys where key.hasPrefix("layers.") {
            let rest = key.dropFirst("layers.".count)
            guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "tensor \(key) names no decoder layer index")
            }
            indices.insert(index)
        }
        guard (1...4).contains(indices.count), indices == Set(0 ..< indices.count) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "the head's decoder layers must be 0..<n with n within 1...4")
        }
        guard declared == 0 || declared == indices.count else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "mtp_num_hidden_layers \(declared) disagrees with the "
                    + "\(indices.count) decoder layer(s) in the tensors")
        }
        return indices.count
    }

    private struct WeightIndex: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    private static func loadIndexedWeights(
        from directory: URL,
        prefix: String
    ) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let index: WeightIndex
        do {
            index = try JSONDecoder().decode(WeightIndex.self, from: Data(contentsOf: indexURL))
        } catch {
            throw Qwen35InlineMTPError.invalidWeightIndex(String(describing: error))
        }

        var byFile: [String: [(source: String, destination: String)]] = [:]
        for (key, file) in index.weightMap where key.hasPrefix(prefix) {
            guard file == URL(fileURLWithPath: file).lastPathComponent,
                file.hasSuffix(".safetensors")
            else {
                throw Qwen35InlineMTPError.invalidWeightIndex(
                    "unsafe shard path for \(key)")
            }
            let destination = String(key.dropFirst(prefix.count))
            guard !destination.isEmpty else {
                throw Qwen35InlineMTPError.invalidWeightIndex("empty stripped key")
            }
            byFile[file, default: []].append((key, destination))
        }
        guard !byFile.isEmpty else { throw Qwen35InlineMTPError.missingWeights }

        var weights: [String: MLXArray] = [:]
        for file in byFile.keys.sorted() {
            let url = directory.appendingPathComponent(file)
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for entry in byFile[file]! {
                guard let value = shard[entry.source] else {
                    throw Qwen35InlineMTPError.invalidWeightIndex(
                        "indexed tensor \(entry.source) is absent from \(file)")
                }
                guard weights.updateValue(value, forKey: entry.destination) == nil else {
                    throw Qwen35InlineMTPError.duplicateWeight(entry.destination)
                }
            }
        }
        return weights
    }

    static func loadStandaloneWeights(
        from directory: URL
    ) throws -> [String: MLXArray] {
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            throw Qwen35InlineMTPError.invalidWeightIndex(String(describing: error))
        }
        guard !urls.isEmpty, urls.count <= 64 else {
            throw Qwen35InlineMTPError.missingWeights
        }

        var weights: [String: MLXArray] = [:]
        for url in urls {
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in shard {
                guard !key.isEmpty, key.utf8.count <= 1024 else {
                    throw Qwen35InlineMTPError.invalidWeightIndex(
                        "standalone tensor key is empty or oversized")
                }
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw Qwen35InlineMTPError.duplicateWeight(key)
                }
            }
        }
        guard !weights.isEmpty else { throw Qwen35InlineMTPError.missingWeights }
        return weights
    }

    private static func validate(
        _ artifact: Qwen35TextConfiguration,
        against target: Qwen35TextConfiguration
    ) throws {
        let fields: [(String, Int, Int)] = [
            ("hidden_size", artifact.hiddenSize, target.hiddenSize),
            ("vocab_size", artifact.vocabularySize, target.vocabularySize),
            ("num_attention_heads", artifact.attentionHeads, target.attentionHeads),
            ("num_key_value_heads", artifact.kvHeads, target.kvHeads),
            ("head_dim", artifact.headDim ?? 0, target.headDim ?? 0),
            ("num_experts", artifact.numExperts, target.numExperts),
            ("num_experts_per_tok", artifact.numExpertsPerTok, target.numExpertsPerTok),
        ]
        for (field, artifactValue, targetValue) in fields
        where artifactValue != targetValue {
            throw Qwen35InlineMTPError.incompatibleTarget(
                field: field, artifact: artifactValue, target: targetValue)
        }
    }
}

extension Qwen35InlineMTPAssistant: CBv2MTPRequestStatefulDrafter {
    final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]

        /// Trusted target transitions not yet appended to the persistent head KV.
        /// Each hidden row at position t is paired with the token at t+1.
        var backlogHidden: [MLXArray] = []
        var backlogTokens: [MLXArray] = []
        /// Last trusted target hidden in an observed chunk. It becomes the
        /// preceding row when the next observed target chunk crosses a boundary.
        var targetHiddenFrontier: MLXArray?

        /// Cache geometry captured before and after the round's trusted flush.
        /// Every later head-chain input is speculative and is trimmed to
        /// `roundValidHistoryOffset` at finalize.
        var roundBaseOffset = 0
        var roundValidHistoryOffset = 0
        var roundDraftSteps = 0
        var roundInFlight = false
        var isReleased = false

        /// Trusted inputs moved out of the backlog for this round. Retaining
        /// their original roots both fences lazy concatenation and lets discard
        /// restore them without a host read.
        var roundTrustedHidden: [MLXArray] = []
        var roundTrustedTokens: [MLXArray] = []
        /// Final proposal hidden rows and draft ids that retain each lazy
        /// head-step graph until the engine's existing finalize synchronization.
        var roundRoots: [MLXArray] = []

        var cacheOffset: Int {
            guard let first = caches.first else { return 0 }
            precondition(
                caches.dropFirst().allSatisfy { $0.offset == first.offset },
                "inline Qwen MTP cache offsets diverged")
            return first.offset
        }

        private var backlogInputCount: Int {
            backlogTokens.reduce(0) { total, tokens in
                let (next, overflow) = total.addingReportingOverflow(tokens.dim(1))
                return overflow ? Int.max : next
            }
        }

        var committedInputCount: Int {
            let committedCache =
                roundInFlight ? roundValidHistoryOffset : cacheOffset
            let (total, overflow) = committedCache.addingReportingOverflow(
                backlogInputCount)
            return overflow ? Int.max : total
        }

        var stagedInputCount: Int {
            guard roundInFlight else { return 0 }
            return max(0, cacheOffset - roundValidHistoryOffset)
        }

        var materializedBytes: Int {
            let arrays =
                caches.flatMap { $0.innerState() }
                + backlogHidden + backlogTokens
                + [targetHiddenFrontier].compactMap { $0 }
                + roundTrustedHidden + roundTrustedTokens + roundRoots
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        init(caches: [any KVCache]) { self.caches = caches }

        func clearRound() {
            roundBaseOffset = cacheOffset
            roundValidHistoryOffset = cacheOffset
            roundDraftSteps = 0
            roundInFlight = false
            roundTrustedHidden.removeAll(keepingCapacity: true)
            roundTrustedTokens.removeAll(keepingCapacity: true)
            roundRoots.removeAll(keepingCapacity: true)
        }

        func clearAll() {
            caches.removeAll(keepingCapacity: false)
            backlogHidden.removeAll(keepingCapacity: false)
            backlogTokens.removeAll(keepingCapacity: false)
            targetHiddenFrontier = nil
            roundTrustedHidden.removeAll(keepingCapacity: false)
            roundTrustedTokens.removeAll(keepingCapacity: false)
            roundRoots.removeAll(keepingCapacity: false)
            roundBaseOffset = 0
            roundValidHistoryOffset = 0
            roundDraftSteps = 0
            roundInFlight = false
            isReleased = true
        }
    }

    private final class UnusedPreparedCapture: CBv2MTPPreparedCapture {}

    public var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    /// Verification policy. Rectangular scores all 1+k columns in one
    /// `[B, 1+k]` target forward. Widths one and two stage ordinary captured
    /// recurrent states. At S>=3 each GatedDeltaNet layer runs one full-window
    /// recurrence and retains compact transformed inputs: full acceptance
    /// installs the final state directly, while a strict accepted prefix lazily
    /// replays only that prefix. This replaces the serial per-column loop whose
    /// k+1 full-model re-reads made the round ≤1.0x by construction.
    ///
    /// NUMERICS POLICY: bitwise greedy parity with serial decode is NOT the
    /// bar here — batched verify changes accumulation geometry exactly like
    /// every other CBv2 batch-shape change (chunked prefill, B>1 decode).
    /// Distribution-exactness is the invariant: committed tokens are always
    /// target-authoritative (argmax when greedy, genuine target samples
    /// under target-prefix acceptance). The serial oracle remains available
    /// via `DARKBLOOM_QWEN_MTP_SERIAL=1` for A/B and certification runs.
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        installedVerificationMode
    }
    /// Production Qwen drafting self-applies the recurrent MTP head up to
    /// `maximumDepth` times. The legacy double-forward oracle stays a k=1 A/B
    /// control.
    public var maximumDraftTokens: Int? { Self.forceDoubleForward ? 1 : Self.maximumDepth }
    public var maximumSpeculativeBatch: Int? { 1 }
    /// Target-prefix acceptance is exact at any temperature (every committed
    /// token IS a target sample), so this drafter lifts the greedy gate when
    /// the engine sampler supports verify pre-sampling.
    public var supportsTargetPrefixAcceptance: Bool { true }

    /// `DARKBLOOM_QWEN_MTP_SERIAL=1/true/yes/on` pins the serial per-column
    /// verification oracle (the pre-capture-verify behavior).
    static let forceSerialVerification: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SERIAL"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()
    /// `DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD=1/true/yes/on` pins the pre-trim
    /// draft step (proposal forward + staging re-forward, both through the
    /// full lm_head) as the paired A/B control arm.
    static let forceDoubleForward: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// `DARKBLOOM_QWEN_MTP_SHORTLIST=<K>` opts the draft head into an
    /// engine-provided top-K shortlist (score K gathered lm_head rows
    /// instead of streaming all 248,320). DEFAULT OFF: measured on M4 Max
    /// (release, B=1 greedy, 256 tok, paired session 2026-08-13) the
    /// shortlist is a net LOSS at every practical K because the ids come
    /// from the target's distribution one position BEHIND the draft:
    ///   K=256   acceptance 0.79→0.58,  95.5 tok/s (full head: 116.5)
    ///   K=2048  acceptance      0.68, 110.3 tok/s
    ///   K=16384 acceptance      0.77, 115.2 tok/s — coverage recovered,
    ///           but per-round verify-side top-K + row gather costs more
    ///           than the ~1.3 ms full 286 MB 4-bit head read it replaces.
    /// Kept env-gated for deeper draft chains (k≥2 amortizes the top-K
    /// cost across several head evaluations per round).
    static let shortlistSize: Int? = {
        guard
            let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SHORTLIST"],
            let value = Int(raw), value > 0
        else { return nil }
        return value
    }()

    /// Screened draft argmax (`HadamardQuantizedLinear.screenedArgMax`):
    /// the leading `width` rotated inputs of every draft-vocabulary row, then
    /// the full width for the best row of each `chunk`-row block. Local
    /// experiments override it with `QWEN35_DRAFT_SCREEN=<width>,<chunk>` or
    /// `off`.
    static let draftScreen: (Int, Int)? = {
        guard let raw = ProcessInfo.processInfo.environment["QWEN35_DRAFT_SCREEN"] else {
            return (1536, 32)
        }
        let parts = raw.split(separator: ",").compactMap { Int($0) }
        return parts.count == 2 ? (parts[0], parts[1]) : nil
    }()

    /// Prompt-lookup drafting: the n-gram the engine matches against the
    /// request's own prompt and committed tokens to propose a round's chain
    /// without running this head. Default 6. Local experiments override it
    /// with `QWEN35_PLD_NGRAM=<n>`; `0` or `off` disables it.
    static let promptLookupNgram: Int? = {
        guard let raw = ProcessInfo.processInfo.environment["QWEN35_PLD_NGRAM"] else {
            return 6
        }
        guard let value = Int(raw), value > 0 else { return nil }
        return value
    }()

    public var lookupDraftNgram: Int? { Self.promptLookupNgram }

    /// The head's chain depth on rounds whose prompt lookup misses, when the
    /// declared depth is deeper: lookup rounds propose the declared depth.
    /// Default 4: with the verify body and the exact screened head both flat
    /// for 2 to 8 rows, a fourth head step commits more per round than it
    /// costs. Local experiments override it with
    /// `QWEN35_PLD_MISS_DEPTH=<k>`; `0` or `off` keeps the declared depth.
    static let promptLookupMissDepth: Int? = {
        guard let raw = ProcessInfo.processInfo.environment["QWEN35_PLD_MISS_DEPTH"] else {
            return 4
        }
        guard let value = Int(raw), value > 0 else { return nil }
        return value
    }()

    public var lookupMissDraftDepth: Int? { Self.promptLookupMissDepth }

    /// Drafted rounds whose carried target top-two logit margin is at least
    /// 4 draft one token deeper (4 instead of 3). On the public prompt at a
    /// uniform depth 4, such rounds (46% of drafted rounds) accepted all four
    /// drafts 55% of the time, the others 27%. `QWEN35_PLD_MISS_DEEP=0` or
    /// `off` keeps one drafted depth (local experiments only).
    static let promptLookupMissDeep: (depth: Int, minimumMargin: Double)? = {
        let raw = ProcessInfo.processInfo.environment["QWEN35_PLD_MISS_DEEP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let raw = raw, ["0", "false", "no", "off"].contains(raw) { return nil }
        guard let missDepth = promptLookupMissDepth else { return nil }
        return (depth: missDepth + 1, minimumMargin: 4)
    }()

    public var lookupMissDeepDraft: (depth: Int, minimumMargin: Double)? {
        Self.promptLookupMissDeep
    }

    /// Shortlisting and the double-forward oracle are intentionally separate
    /// controls: the oracle scores the complete shared output head.
    public var draftShortlistSize: Int? {
        Self.forceDoubleForward ? nil : Self.shortlistSize
    }
    public var requestStateBytesPerToken: Int {
        Self.stateBytesPerToken(
            configuration: target.configuration,
            layerCount: mtp.layers.count,
            cacheElementBytes: mtp.norm.weight.dtype.size,
            hiddenElementBytes: target.model.norm.weight.dtype.size)
    }
    public var requestStateTokenGranularity: Int { Self.cacheAllocationStep }
    /// A round stages at most `maximumDepth` head inputs past the committed
    /// history before finalize trims them.
    public var requestStateTokenAllocationPadding: Int { Self.maximumDepth }

    public var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        let config = target.configuration
        let (elements, geometryOverflow) = config.kvHeads.multipliedReportingOverflow(by: config.headDim ?? 0)
        let (cacheRow, cacheOverflow) = elements.multipliedReportingOverflow(by: mtp.norm.weight.dtype.size)
        let (cacheCount, countOverflow) = mtp.layers.count.multipliedReportingOverflow(by: 2)
        let (hiddenRow, hiddenOverflow) = config.hiddenSize.multipliedReportingOverflow(
            by: target.model.norm.weight.dtype.size)
        guard !geometryOverflow, !cacheOverflow, !countOverflow, !hiddenOverflow else { return nil }
        return [
            .init(bytesPerToken: cacheRow, allocationCount: cacheCount,
                  tokenGranularity: Self.cacheAllocationStep, tokenPadding: Self.maximumDepth),
            // Backlog and frontier are views of normalized committed chunks;
            // the bound covers every possible partition of the retained rows.
            .init(bytesPerToken: hiddenRow, tokenPadding: Self.maximumDepth, partitioned: true),
            .init(bytesPerToken: MemoryLayout<Int32>.stride, tokenPadding: Self.maximumDepth, partitioned: true),
        ]
    }

    static func cacheBytesPerToken(
        configuration: Qwen35TextConfiguration,
        layerCount: Int,
        elementBytes: Int
    ) -> Int {
        let (headsByDimension, geometryOverflow) = configuration.kvHeads
            .multipliedReportingOverflow(by: configuration.headDim ?? 0)
        let (kvElements, kvOverflow) = headsByDimension.multipliedReportingOverflow(by: 2)
        let (layerElements, layerOverflow) = kvElements.multipliedReportingOverflow(
            by: layerCount)
        let (bytes, byteOverflow) = layerElements.multipliedReportingOverflow(
            by: elementBytes)
        return geometryOverflow || kvOverflow || layerOverflow || byteOverflow ? Int.max : bytes
    }

    static func stateBytesPerToken(
        configuration: Qwen35TextConfiguration,
        layerCount: Int,
        cacheElementBytes: Int,
        hiddenElementBytes: Int
    ) -> Int {
        let cacheBytes = cacheBytesPerToken(
            configuration: configuration, layerCount: layerCount,
            elementBytes: cacheElementBytes)
        let (hiddenBytes, hiddenOverflow) = configuration.hiddenSize
            .multipliedReportingOverflow(by: hiddenElementBytes)
        guard cacheBytes != Int.max, !hiddenOverflow else { return Int.max }
        let (withHidden, hiddenAdditionOverflow) = cacheBytes.addingReportingOverflow(
            hiddenBytes)
        let (withToken, tokenAdditionOverflow) = withHidden.addingReportingOverflow(
            MemoryLayout<Int32>.stride)
        return hiddenAdditionOverflow || tokenAdditionOverflow ? Int.max : withToken
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        RequestState(caches: makeCache())
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP observed released request state")
        precondition(!state.roundInFlight, "inline Qwen MTP observed target during a round")
        precondition(
            observation.tokens.ndim == 2 && observation.hidden.ndim == 3
                && observation.tokens.dim(0) == 1 && observation.hidden.dim(0) == 1
                && observation.tokens.dim(1) == observation.hidden.dim(1),
            "inline Qwen MTP target observation shape mismatch")

        let count = observation.tokens.dim(1)
        guard count > 0 else { return }
        let normalizedHidden = targetFinalNorm(observation.hidden)

        // Cross-chunk transition: the preceding chunk's final normalized
        // target hidden pairs with this chunk's first target input.
        if let frontier = state.targetHiddenFrontier {
            state.backlogHidden.append(frontier)
            state.backlogTokens.append(observation.tokens[0..., 0 ..< 1])
        }
        // Intra-chunk transitions: finalNorm(hidden[t]) conditions token[t+1].
        if count > 1 {
            state.backlogHidden.append(normalizedHidden[0..., 0 ..< count - 1, 0...])
            state.backlogTokens.append(observation.tokens[0..., 1 ..< count])
        }
        state.targetHiddenFrontier =
            normalizedHidden[0..., (count - 1) ..< count, 0...]
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        UnusedPreparedCapture()
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("inline Qwen MTP requires request-owned assistant state")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP drafted with released request state")
        precondition(
            tokens.ndim == 2 && hidden.ndim == 3
                && tokens.dim(0) == 1 && tokens.dim(1) == 1
                && hidden.dim(0) == 1 && hidden.dim(1) == 1,
            "inline Qwen MTP draft input shape mismatch")

        if Self.forceDoubleForward {
            return legacyDoubleForwardDraftStep(tokens: tokens, hidden: hidden, state: state)
        }

        let isFirstStep = !state.roundInFlight
        let feed: (tokens: MLXArray, hidden: MLXArray)
        if isFirstStep {
            feed = beginRound(tokens: tokens, hidden: hidden, state: state)
        } else {
            precondition(
                state.roundDraftSteps < Self.maximumDepth,
                "inline Qwen MTP exceeded its \(Self.maximumDepth)-step draft chain")
            feed = (tokens, hidden)
        }

        let output =
            isFirstStep
            ? (moduleLastHiddenWithKVOnlyHistory(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
                ?? moduleForward(
                    hidden: feed.hidden, tokens: feed.tokens, cache: state.caches))
            : moduleForward(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        let lastHidden = output[0..., (output.dim(1) - 1)..., 0...]
        let draft = draftToken(hidden: lastHidden, shortlist: shortlist)
        state.roundRoots.append(contentsOf: [lastHidden, draft])
        state.roundDraftSteps += 1

        if isFirstStep {
            // The engine immediately submits evaluationTargets for this first
            // cache generation before constructing a deeper draft step.
            state.roundValidHistoryOffset = state.cacheOffset
        }
        return (draft, lastHidden)
    }

    private func beginRound(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "inline Qwen MTP round already in flight")
        precondition(
            state.backlogHidden.count == state.backlogTokens.count,
            "inline Qwen MTP trusted backlog diverged")

        state.roundBaseOffset = state.cacheOffset
        state.roundValidHistoryOffset = state.cacheOffset
        state.roundDraftSteps = 0
        state.roundInFlight = true
        state.roundTrustedHidden = state.backlogHidden
        state.roundTrustedTokens = state.backlogTokens
        state.backlogHidden.removeAll(keepingCapacity: true)
        state.backlogTokens.removeAll(keepingCapacity: true)

        // The current target carry is trusted and completes the frontier
        // transition. Normalize it exactly once before it enters head history.
        state.roundTrustedHidden.append(targetFinalNorm(hidden))
        state.roundTrustedTokens.append(tokens)
        state.targetHiddenFrontier = nil

        if state.roundTrustedTokens.count == 1 {
            return (state.roundTrustedTokens[0], state.roundTrustedHidden[0])
        }
        let feedTokens = concatenated(state.roundTrustedTokens, axis: 1)
        let feedHidden = concatenated(state.roundTrustedHidden, axis: 1)
        return (feedTokens, feedHidden)
    }

    private func draftToken(hidden: MLXArray, shortlist: MLXArray?) -> MLXArray {
        if let shortlist {
            let logits = shortlistLogits(hidden: hidden, ids: shortlist)
            return shortlist[argMax(logits[0..., -1, 0...], axis: -1)]
                .asType(.int32)
        }
        if let draftVocabulary {
            if let (width, chunk) = Self.draftScreen,
                let picked = draftVocabulary.head.screenedArgMax(
                    hidden, width: width, chunk: chunk, ids: draftVocabulary.ids)
            {
                return picked
            }
            let local = argMax(draftVocabulary.head(hidden)[0..., -1, 0...], axis: -1)
            return draftVocabulary.ids[local]
        }
        return argMax(headLogits(hidden)[0..., -1, 0...], axis: -1)
            .asType(.int32)
    }

    /// Explicit pre-cutover A/B oracle. It shares production history upkeep,
    /// but stages the proposed draft through a second complete MTP/lm-head
    /// forward and performs the historical blocking mid-round evaluation.
    private func legacyDoubleForwardDraftStep(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "inline Qwen MTP round already in flight")
        let feed = beginRound(tokens: tokens, hidden: hidden, state: state)
        let output =
            moduleLastHiddenWithKVOnlyHistory(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
            ?? moduleForward(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        let lastHidden = output[0..., (output.dim(1) - 1)..., 0...]
        let logits = headLogits(lastHidden)
        let draft = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
        state.roundRoots.append(contentsOf: [lastHidden, draft])
        state.roundDraftSteps = 1
        state.roundValidHistoryOffset = state.cacheOffset

        eval([draft, lastHidden] + state.caches.flatMap { $0.innerState() })
        _ = forward(
            hidden: lastHidden, tokens: draft.reshaped([1, 1]), cache: state.caches)
        return (draft, lastHidden)
    }

    public func evaluationTargets(
        for requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return []
        }
        return state.caches.flatMap { $0.innerState() }
            + state.backlogHidden + state.backlogTokens
            + [state.targetHiddenFrontier].compactMap { $0 }
            + state.roundTrustedHidden + state.roundTrustedTokens + state.roundRoots
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP finalized released request state")
        precondition(state.roundInFlight, "inline Qwen MTP finalized without a round")
        precondition(
            (0 ... state.roundDraftSteps + 1).contains(confirmedInputTokens),
            "inline Qwen MTP confirmed prefix exceeds the draft round")
        precondition(
            committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1
                && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "inline Qwen MTP committed target rows mismatch")
        let committedDraftCount = committedDraftTokens.dim(1)
        precondition(
            committedDraftCount <= state.roundDraftSteps
                && committedDraftCount <= max(0, confirmedInputTokens - 1),
            "inline Qwen MTP committed drafts exceed confirmed target inputs")

        trim(state: state, to: state.roundValidHistoryOffset)
        if committedDraftCount > 0 {
            // These are pre-final-norm target verify hiddens, never speculative
            // assistant hiddens. Normalize exactly once, retain lazily, and
            // flush them with the next carry.
            state.backlogTokens.append(committedDraftTokens)
            state.backlogHidden.append(targetFinalNorm(committedTargetHidden))
        }
        state.clearRound()
    }

    public func commitLookupRound(
        requestState: any CBv2MTPRequestState,
        seedToken: MLXArray,
        seedHidden: MLXArray,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP committed released request state")
        precondition(!state.roundInFlight, "inline Qwen MTP lookup round met a staged round")
        precondition(
            seedToken.ndim == 2 && seedHidden.ndim == 3
                && seedToken.dim(0) == 1 && seedToken.dim(1) == 1
                && seedHidden.dim(0) == 1 && seedHidden.dim(1) == 1
                && committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1
                && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "inline Qwen MTP lookup round rows mismatch")

        // The history `beginRound` + `finalizeRound` would leave, minus the
        // head forward: the carry transition completes the frontier (which
        // therefore never enters history itself), then the accepted drafts
        // with their target verify hiddens. Everything stays queued for the
        // next drafted round's trusted flush, each row exactly once.
        state.backlogHidden.append(targetFinalNorm(seedHidden))
        state.backlogTokens.append(seedToken)
        state.targetHiddenFrontier = nil
        if committedDraftTokens.dim(1) > 0 {
            state.backlogTokens.append(committedDraftTokens)
            state.backlogHidden.append(targetFinalNorm(committedTargetHidden))
        }
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState,
            !state.isReleased, state.roundInFlight
        else { return }

        trim(state: state, to: state.roundBaseOffset)
        // Restore all trusted transitions consumed by the abandoned graph in
        // original order. No speculative assistant hidden enters the backlog.
        state.backlogHidden =
            state.roundTrustedHidden + state.backlogHidden
        state.backlogTokens =
            state.roundTrustedTokens + state.backlogTokens
        state.clearRound()
    }

    private func trim(state: RequestState, to offset: Int) {
        let rollback = state.cacheOffset - offset
        precondition(rollback >= 0, "inline Qwen MTP cache checkpoint moved forward")
        guard rollback > 0 else { return }
        for cache in state.caches {
            precondition(cache.trim(rollback) == rollback)
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return
        }
        state.clearAll()
    }
}

/// The draft chain proposes tokens only; the pinned target verifies and
/// decides every emitted token. So the draft head may score a fixed subset of
/// the vocabulary: a draft outside the subset is a rejected proposal, never a
/// wrong output. Byte-level BPE ids follow merge order, so the leading rows
/// hold the most frequent tokens, and added/special tokens sit at the tail.
/// The subset is fixed at load from the head's geometry alone (input- and
/// prompt-independent): the leading `leadingRows` rows plus the trailing
/// `trailingRows` rows of the target's own packed `lm_head`, copied once,
/// byte-identical, into one contiguous draft-only head. The target's head and
/// quantization are untouched; each draft step reads only the subset rows.
final class Qwen35MTPDraftVocabulary: @unchecked Sendable {
    /// Half draft subset (leading 49,152 + trailing 512) — officially
    /// validated sweet spot (9ff5316e sealed 2.30167, acceptance pinned
    /// 0.8378, mismatch 0). Quarter cut falsified (d15bec94 2.24069, acc
    /// 0.7583): [24_576, 49_152) carries accepted tokens. Halving N halves
    /// the screened screen's row stream and every neural draft step's head
    /// traffic. Guards hold: 49_664 % 32 == 0, 1,552 blocks even.
    /// Out-of-subset proposals are rejected drafts, never wrong outputs.
    static let leadingRows = 49_152
    static let trailingRows = 512

    let head: HadamardQuantizedLinear
    /// Target token id of each draft-head row (int32).
    let ids: MLXArray

    init?(target: Qwen35TextModel) {
        guard !target.configuration.tieWordEmbeddings,
            let full = target.lmHead as? HadamardQuantizedLinear,
            full.bias == nil, let biases = full.biases
        else { return nil }
        let vocabulary = full.weight.dim(0)
        let leading = Self.leadingRows
        let tailStart = vocabulary - Self.trailingRows
        guard leading < tailStart, (leading + Self.trailingRows) % 8 == 0 else { return nil }
        func rows(_ array: MLXArray) -> MLXArray {
            concatenated([array[0 ..< leading], array[tailStart ..< vocabulary]], axis: 0)
        }
        let (weight, scales, offsets) = (rows(full.weight), rows(full.scales), rows(biases))
        var tokenIds = [Int32](0 ..< Int32(leading))
        tokenIds.append(contentsOf: Int32(tailStart) ..< Int32(vocabulary))
        let ids = MLXArray(tokenIds)
        eval(weight, scales, offsets, ids)
        guard
            let head = try? HadamardQuantizedLinear(
                weight: weight, scales: scales, biases: offsets,
                groupSize: full.groupSize, bits: full.bits, transform: full.transform)
        else { return nil }
        self.head = head
        self.ids = ids
    }
}

// MARK: - Load-time engine round warm

extension Qwen35InlineMTPAssistant {
    /// `MLXFAST_MTP_ROUND_WARM=0` skips `warmEngineRound`.
    static let engineRoundWarmEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_MTP_ROUND_WARM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The declared depth of this track's chain (mtp-head.manifest.json) and
    /// the scored seed width. At the declared depth the engine runs two round
    /// shapes: a prompt-lookup hit verifies 1 + 6 rows, a miss drafts a
    /// depth-3 chain with the head and verifies 1 + 3 rows
    /// (`promptLookupMissDepth`), or a depth-4 chain and 1 + 4 rows when the
    /// target's carried margin is high (`promptLookupMissDeep`).
    static let warmDepth = 6
    static let warmPromptRows = 512
    /// The period of the second warm prompt: its newest n-gram always recurs,
    /// so the lookup hits and the rounds verify at the declared width.
    static let warmLookupPeriod = 64

    /// Runs two short requests through a real `EngineV2` on throwaway state:
    /// the seed prefill seam, the seed step and a few rounds of each verify
    /// width the declared depth runs (draft chain, draft head, rectangular
    /// verify, top-two, finalize, rollback; and lookup-proposed rounds).
    ///
    /// Nothing before the timed decode window runs the engine's MTP round:
    /// the warmup and timed prefills go through the teacher-forced stepper.
    /// So a process paid each first-use cost (Metal pipeline builds, lazily
    /// widened head constants, allocator growth) inside the window's first
    /// round, and the window is only 128 tokens. The prompt is the resident
    /// warm's fixed token pattern, greedy, so nothing depends on any
    /// request's input; the engine, its caches and the request state are
    /// released and the buffer cache is drained before anything is served.
    func warmEngineRound(serving: any LanguageModel) {
        guard Self.engineRoundWarmEnabled else { return }
        let layerKinds: [CBv2LayerKind]
        let caches: [any CBv2AttendingLayerCache]
        let make: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache = {
            index, kind in CBv2LayerCache(layerIndex: index, kind: kind)
        }
        if let model = serving as? Qwen35Model, let built = try? model.newCacheV2(makeLayerCache: make) {
            (layerKinds, caches) = (model.cbv2LayerKinds, built)
        } else if let model = serving as? Qwen35TextModel,
            let built = try? model.newCacheV2(makeLayerCache: make)
        {
            (layerKinds, caches) = (model.cbv2LayerKinds, built)
        } else {
            return
        }
        let depth = Self.warmDepth
        let rows = Self.warmPromptRows
        weak var released: EngineV2?
        var warmMTPConfig = CBv2MTPConfig(
            enabled: true,
            maxDraftTokens: depth,
            maxSpeculativeBatch: 1,
            fixedDraftTokens: depth,
            verificationMode: .automatic,
            maxAutomaticRectangularTokens: 1 + depth,
            draftTokenCeiling: CBv2MTPConfig.testedMaxDraftTokens)
        // The first request's first two drafted rounds run the deep chain
        // (5-row verify) and the shallow one (4-row verify), whatever the warm
        // prompt's margins are, so both drafted widths are built here.
        if let deep = Self.promptLookupMissDeep, let shallow = Self.promptLookupMissDepth {
            warmMTPConfig.missDepthSchedule = [deep.depth, shallow]
        }
        do {
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(serving),
                layerKinds: layerKinds,
                backend: CBv2ContiguousKVBackend(
                    config: CBv2ContiguousBackendConfig(bytesCapacity: 1 << 30)),
                cacheProvider: CBv2LayerCacheBank(caches: caches),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 1,
                    prefillChunkSize: max(CBv2SchedulerConfig().prefillChunkSize, rows),
                    maxWaiting: 1,
                    enablePrefixCache: false),
                mtpDrafter: self,
                mtpConfig: warmMTPConfig)
            released = engine
            // Two greedy requests, one after the other: a non-repeating
            // pattern, whose rounds miss the lookup and run the head's chain
            // (4-row verify), then a periodic one, whose rounds hit it
            // (7-row verify). Each request has its own lookup index.
            let period = Self.warmLookupPeriod
            let prompts: [[Int]] = [
                (0 ..< rows).map { 100 + ($0 &* 7919) % 20_000 },
                (0 ..< rows).map { 100 + (($0 % period) &* 7919) % 20_000 },
            ]
            let warmRequests: [CBv2Request] = prompts.enumerated().map { index, prompt in
                var request = CBv2Request(
                    id: CBv2RequestID(UInt64(index + 1)),
                    promptTokens: prompt,
                    maxTokens: 2 + 4 * (1 + depth))
                request.sampling = CBv2SamplingParams(temperature: 0, topP: 1, topK: 0)
                request.stopTokens = []
                return request
            }
            let done = DispatchSemaphore(value: 0)
            Task.detached {
                for warmRequest in warmRequests {
                    if let events = try? engine.submit(warmRequest) {
                        for await _ in events {}
                    }
                }
                await engine.shutdown()
                done.signal()
            }
            done.wait()
        }
        var waited = 0
        while released != nil, waited < 500 {
            usleep(1_000)
            waited += 1
        }
        for r in 2 ... Qwen35MTPFloat16ConstantQMV.maximumWideRows {
            let warmCaches = makeCache()
            let count = r * target.configuration.hiddenSize
            let warmH = (MLXArray(0 ..< count).asType(.float32) * Float(1e-4))
                .reshaped(1, r, target.configuration.hiddenSize)
            let warmT = MLXArray.zeros([1, r], dtype: .int32)
            for _ in 0 ..< 2 {
                if let out = moduleLastHiddenWithKVOnlyHistory(
                    hidden: warmH, tokens: warmT, cache: warmCaches)
                {
                    eval([out] + warmCaches.flatMap { $0.innerState() })
                }
            }
        }
        let warmQ = MLXArray.zeros([1, 1, 16, 128], dtype: .float32)
        let warmV = MLXArray.zeros([1, 1, 32, 128], dtype: .float32)
        let warmA = MLXArray.zeros([1, 1, 32], dtype: .float32)
        let warmLog = MLXArray.zeros([32], dtype: .float32)
        let warmState = MLXArray.zeros([1, 32, 128, 128], dtype: .float32)
        eval(
            Qwen35RegisterGatedSIMD.runStateOnly(
                q: warmQ, k: warmQ, v: warmV, a: warmA, b: warmA,
                aLog: warmLog, dtBias: warmLog, state: warmState))
        Stream().synchronize()
        Memory.clearCache()
    }
}
