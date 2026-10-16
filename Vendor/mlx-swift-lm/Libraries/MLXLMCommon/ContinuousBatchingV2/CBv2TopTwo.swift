// Copyright © 2026 Eigen Labs.

import MLX

// MARK: - Hierarchical top-2

// Metal entry-point names remain stable so extraction preserves pipeline identity.
/// Shared exact ordering for the two-stage candidate-only top-2 reduction:
/// value descending, token id ascending on exact ties, and NaNs last.
private let cbv2TopTwoHeader = """
    struct darkbloom_qwen35_mtp_top2_state {
        float first_value;
        float second_value;
        uint first_id;
        uint second_id;
        uint count;
    };

    inline darkbloom_qwen35_mtp_top2_state darkbloom_qwen35_mtp_top2_empty() {
        darkbloom_qwen35_mtp_top2_state state;
        state.first_value = 0.0f;
        state.second_value = 0.0f;
        state.first_id = 0;
        state.second_id = 0;
        state.count = 0;
        return state;
    }

    inline bool darkbloom_qwen35_mtp_top2_better(
        float candidate_value,
        uint candidate_id,
        float current_value,
        uint current_id
    ) {
        bool candidate_nan = isnan(candidate_value);
        bool current_nan = isnan(current_value);
        if (candidate_nan != current_nan) {
            return !candidate_nan;
        }
        if (candidate_value > current_value) {
            return true;
        }
        if (candidate_value < current_value) {
            return false;
        }
        return candidate_id < current_id;
    }

    inline void darkbloom_qwen35_mtp_top2_insert(
        thread darkbloom_qwen35_mtp_top2_state &state,
        float value,
        uint id
    ) {
        if (state.count > 0 && state.first_id == id) {
            return;
        }
        if (state.count > 1 && state.second_id == id) {
            return;
        }
        if (state.count == 0
            || darkbloom_qwen35_mtp_top2_better(
                value, id, state.first_value, state.first_id)) {
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
            || darkbloom_qwen35_mtp_top2_better(
                value, id, state.second_value, state.second_id)) {
            state.second_value = value;
            state.second_id = id;
            state.count = 2;
        }
    }

    inline darkbloom_qwen35_mtp_top2_state darkbloom_qwen35_mtp_top2_simd_reduce(
        darkbloom_qwen35_mtp_top2_state local,
        uint lane
    ) {
        for (uint stride = 16; stride > 0; stride >>= 1) {
            darkbloom_qwen35_mtp_top2_state other;
            other.first_value = simd_shuffle_down(local.first_value, stride);
            other.second_value = simd_shuffle_down(local.second_value, stride);
            other.first_id = simd_shuffle_down(local.first_id, stride);
            other.second_id = simd_shuffle_down(local.second_id, stride);
            other.count = simd_shuffle_down(local.count, stride);
            if (lane < stride) {
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        local, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        local, other.second_value, other.second_id);
                }
            }
        }
        return local;
    }
"""

/// Stage one: SIMD reductions, then one shared-memory exchange per stripe.
private let cbv2TopTwoPartialKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_partial",
    inputNames: ["logits"],
    outputNames: ["partial_ids", "partial_values"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();

        const ulong row_offset = ulong(row) * ulong(logits_strides[1]);
        const ulong stride2 = ulong(logits_strides[2]);

        for (uint index = group * 256 + lane;
             index < vocab;
             index += 32 * 256) {
            ulong offset = row_offset + ulong(index) * stride2;
            float val = float(logits[offset]);
            if (local.count == 2 && val <= local.second_value) {
                continue;
            }
            darkbloom_qwen35_mtp_top2_insert(local, val, index);
        }

        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        local = darkbloom_qwen35_mtp_top2_simd_reduce(local, simd_lane);
        threadgroup darkbloom_qwen35_mtp_top2_state scratch[8];
        if (simd_lane == 0) {
            scratch[simd_group] = local;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            local = simd_lane < 8
                ? scratch[simd_lane] : darkbloom_qwen35_mtp_top2_empty();
            local = darkbloom_qwen35_mtp_top2_simd_reduce(local, simd_lane);
            if (simd_lane == 0) {
                uint base = (row * 32 + group) * 2;
                uint sentinel_id = vocab + group * 2;
                float sentinel_value = as_type<float>(0x7fc00000u);
                partial_ids[base] = local.count > 0
                    ? int(local.first_id) : int(sentinel_id);
                partial_ids[base + 1] = local.count > 1
                    ? int(local.second_id) : int(sentinel_id + 1);
                partial_values[base] = local.count > 0
                    ? local.first_value : sentinel_value;
                partial_values[base + 1] = local.count > 1
                    ? local.second_value : sentinel_value;
            }
        }
    """,
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// Stage two: one SIMD group per row merges the partial pairs in registers.
private let cbv2TopTwoFinalizeKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_finalize",
    inputNames: ["partial_ids", "partial_values"],
    outputNames: ["top_ids", "top_values"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint base = (row * 32 + lane) * 2;
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base], uint(partial_ids[base]));
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base + 1], uint(partial_ids[base + 1]));

        local = darkbloom_qwen35_mtp_top2_simd_reduce(local, lane);
        if (lane == 0) {
            uint output_base = row * 2;
            top_ids[output_base] = int(local.first_id);
            top_ids[output_base + 1] = int(local.second_id);
            top_values[output_base] = local.first_value;
            top_values[output_base + 1] = local.second_value;
        }
    """,
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// Exact top-2 token ids and logit values for every row of `[1, rows, vocab]`.
///
/// Returns lazy device arrays shaped `[rows, 2]`: ids are `int32`, values are
/// `float32`. No evaluation or host read occurs here. Results are ordered by
/// value descending, then token id ascending on exact ties, with NaNs last.
package func cbv2TopTwoRows(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
    precondition(logits.ndim == 3 && logits.dim(0) == 1)
    let rows = logits.dim(1)
    let vocabularySize = logits.dim(2)
    precondition(rows > 0 && vocabularySize >= 2)

    let partials = cbv2TopTwoPartialKernel(
        [logits],
        grid: (rows * 32 * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[rows, 32, 2], [rows, 32, 2]],
        outputDTypes: [.int32, .float32]
    )
    let outputs = cbv2TopTwoFinalizeKernel(
        partials,
        grid: (rows * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[rows, 2], [rows, 2]],
        outputDTypes: [.int32, .float32]
    )
    return (outputs[0], outputs[1])
}
