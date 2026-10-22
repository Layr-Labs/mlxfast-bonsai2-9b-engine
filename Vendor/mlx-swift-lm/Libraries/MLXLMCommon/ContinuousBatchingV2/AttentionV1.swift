// AttentionV1.swift
//
// The v1 attention dispatch used by `CBv2LayerCache.updateAndAttend`:
// per-row `MLXFast.scaledDotProductAttention` against each sequence's own
// contiguous KV. The paged backend (workstream C) replaces this behind the
// same `CBv2AttendingLayerCache` protocol.
//
// ## Why this path cannot have the left-padding bug class
// - Decode is rectangular [B, 1]: each row attends EXACTLY its own KV, with
//   no mask at all — a fully-masked row cannot exist by construction, so
//   NaN poisoning (ee2a921) is impossible.
// - Prefill is per-request [1, chunk]: masks are per-request, derived only
//   from that request's own lengths — batch composition cannot influence
//   them.
// - The mask mode is a PURE FUNCTION of (L, returned KV length, window):
//   never data-dependent, never drifting across steps for the same logical
//   computation (MLX #3384 / report 10 §4 invariant 5).

import Cmlx
import Foundation
import MLX
import MLXNN

/// Namespace for the v1 (per-row SDPA) attention dispatch.
enum CBv2AttentionV1 {

    /// Query-block width for multi-token prompt attention (see
    /// `attendQueryBlocks`). Smaller blocks execute strictly less attention
    /// work and hold a smaller score tensor, but cost one dispatch set each;
    /// the composed path is ~4-5 Metal dispatches per call, so very small
    /// blocks trade GPU efficiency for FLOPs already saved.
    ///
    /// 128 keeps >97% of the achievable work reduction at a quarter of the
    /// launch overhead of 32. `0` disables blocking entirely (one call for the
    /// whole chunk — the pre-2026-07 behavior), which is the kill switch if
    /// this is ever implicated in a numerics or latency regression.
    static let queryBlockSize: Int = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_CBV2_ATTN_QUERY_BLOCK"],
            let value = Int(raw), value >= 0
        else { return 128 }
        return value
    }()

    /// Whether a chunk of `L` queries should be split into blocks. Single
    /// queries (decode) and chunks already at or below the block width take
    /// the unchanged single-call path, so decode is provably untouched.
    @inline(__always)
    static func shouldBlockQueries(_ L: Int) -> Bool {
        queryBlockSize > 0 && L > queryBlockSize
    }

    /// The sink logits an `MLXFast.scaledDotProductAttention` call may be
    /// handed for `queryDType` activations.
    ///
    /// MLX requires the sink dtype to PROMOTE to the SDPA output dtype. fp16
    /// queries with fp32 sinks are therefore not a rounding wart but a
    /// process abort: MLX throws `[scaled_dot_product_attention] Type of
    /// sinks must promote to output type float16` in C++, mlx-c routes it to
    /// the installed error handler, and mlx-swift's handler calls
    /// `fatalError` (`MLX/ErrorHandler.swift`) — SIGTRAP, not a Swift error
    /// anything upstream can catch. A daemon hosting several models loses
    /// every in-flight request and emits nothing.
    ///
    /// Every SDPA terminal on a SERVING path funnels through here: the two
    /// in this file via `dispatchSinks`, and the paged prefill terminal
    /// (`PagedLayerCache.attendQueryBlock`) via that file's `prefillSinks`.
    /// Both callers hoist the cast to their top-level dispatch, so the
    /// per-row / per-block / per-token loops below them re-use one array.
    ///
    /// The remaining SDPA calls in the module are in `PagedDecodeProfiler`
    /// and `PagedBackendBenchmark`, which mint their own fixtures in the
    /// dtype they then query with; they are measurement rigs, not a path any
    /// request reaches.
    ///
    /// NOT for `PagedAttentionReference.composedAttention`: it runs in fp32
    /// throughout and widens the sinks itself, so narrowing them first would
    /// be a real precision loss. Both callers carve that case out on
    /// `softcap != nil`, which is what selects the composed path.
    @inline(__always)
    static func sdpaSinks(_ sinks: MLXArray?, queryDType: DType) -> MLXArray? {
        // `asType` returns `self` when the dtypes already match, so the
        // models shipping today (gpt-oss loads `sinks` in checkpoint dtype,
        // matching the activations) add no operation at all.
        sinks?.asType(queryDType)
    }

    /// The sinks every terminal reached by ONE top-level dispatch receives.
    ///
    /// Computed once per `updateAndAttend` / `attendBorrowing` /
    /// `updateAndAttendLastQuery` call rather than at the terminal: eager
    /// decode enters `attend` once per row, query-blocked prefill once per
    /// block and the MTP serial path once per query, so casting at the
    /// terminal rebuilds the same one-element conversion for every row,
    /// block, layer and generated token on latency-sensitive paths.
    ///
    /// Slicing preserves dtype, so the top-level `queries.dtype` is exactly
    /// the dtype every per-row / per-block slice below presents.
    @inline(__always)
    private static func dispatchSinks(
        _ sinks: MLXArray?, kind: CBv2LayerKind, queries: MLXArray, softcap: Float?
    ) -> MLXArray? {
        guard kind.hasSinks, let sinks else { return nil }
        // A softcap sends BOTH phases to the composed fp32 reference, which
        // wants the model's own (possibly wider) sinks — see `sdpaSinks`.
        return softcap == nil ? sdpaSinks(sinks, queryDType: queries.dtype) : sinks
    }

    /// Gathered-key columns ONE query block may see: from the EARLIEST
    /// query's window floor to the LATEST query's own position, so the
    /// block's queries are exactly the trailing `count` entries of the span
    /// — the invariant `maskMode` assumes.
    ///
    /// `historyCount` is the number of keys that predate the block's chunk;
    /// it means the same thing on both backends. Contiguous derives it as
    /// `keys.dim(2) - newTokenCount`, paged reads it off the assembled
    /// `PrefillKV`.
    ///
    /// **This arithmetic IS the activation-reserve bound.** Blocking is what
    /// pins the materialized score tensor at `[B, heads, count, kL]` instead
    /// of `[B, heads, L, kL]`, which is what lets `prefillChunkSize` grow
    /// without the flat 3 GiB reserve in the provider's `UnifiedMemoryCap`
    /// becoming a lie. On a sliding layer the span saturates at
    /// `window - 1 + blockSize` however long the chunk is; on a full layer
    /// only the query extent is capped. Both backends therefore have to
    /// compute the SAME span — if the two copies drift, one of them silently
    /// attends a different set of keys, and the memory bound stops holding
    /// on whichever side widened.
    ///
    /// It is shared rather than copied for exactly that reason. The paged
    /// track originally copied it because `attendQueryBlocks` is `private`
    /// and returns a symbolic mask mode — true, but a reason that applies to
    /// the attention CALL, not to the bounds.
    static func queryBlockBounds(
        historyCount: Int, offset: Int, count: Int, window: Int?
    ) -> (visibleStart: Int, visibleEnd: Int) {
        let visibleEnd = historyCount + offset + count
        let visibleStart = window.map { max(0, historyCount + offset + 1 - $0) } ?? 0
        return (visibleStart, visibleEnd)
    }

    /// Mask mode for a single-request attention call.
    ///
    /// - `L == 1` (decode): `.none`. The row's retained KV IS its window —
    ///   sliding-window eviction already dropped everything outside it.
    /// - `L > 1` (prefill chunk) against `kL` returned KV entries:
    ///   - `.causal` when no window is configured, or when `kL <= window`
    ///     (the window cannot bind: the oldest returned entry is inside
    ///     every query's window).
    ///   - causal ∧ window ARRAY mask when `kL > window` (a windowed layer's
    ///     multi-token update returned pre-eviction history so early chunk
    ///     tokens see their full window; later tokens must not over-attend).
    ///
    /// Pure in (L, kL, window): the same request produces the same mask mode
    /// at the same point in its lifetime regardless of batchmates.
    static func maskMode(L: Int, kL: Int, window: Int?, bidirectional: Bool = false)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        if L == 1 { return .none }
        if bidirectional {
            return .array(boolMask(L: L, kL: kL, window: window, bidirectional: true)!)
        }
        if let window, kL > window {
            // Relative coordinates: keys span [0, kL), queries are the last
            // L positions. Window comparisons are translation-invariant, so
            // the absolute offset is irrelevant.
            return .array(createCausalMask(n: L, offset: kL - L, windowSize: window))
        }
        return .causal
    }

    /// Update each row with this step's K/V and attend.
    ///
    /// - queries/keys/values: `[B, heads, L, headDim]` with `L == 1` for
    ///   decode, `B == 1` for a singleton prefill chunk, or rectangular
    ///   `[B > 1, L > 1]` packed prefill / MTP verification. Every row
    ///   dispatches against its own KV and optional span context.
    /// - softcap: construction-time attention-logit soft cap
    ///   (`cap * tanh(qk / cap)` before softmax, Gemma-2 style). When set,
    ///   BOTH phases take the composed reference path (SDPA cannot express
    ///   softcapping) — still one pinned path per phase.
    /// - spanContexts: when bound, one optional context per row. Non-nil
    ///   entries select that row's vision overlay; nil entries retain q=128
    ///   query-block causal/window semantics in the same rectangular call.
    /// - Returns `[B, queryHeads, L, headDim]`.
    static func updateAndAttend(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float? = nil,
        spanContexts: [CBv2SpanChunkContext?]? = nil,
        serializeQueries: Bool = false, keepMask: MLXArray? = nil,
        metadata: CBv2AttentionMetadataObservation? = nil,
        packet: CBv2AttentionPacketObservation? = nil
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        if keepMask != nil {
            precondition(
                B == 1,
                "CBv2AttentionV1: a keep mask serves one row per call, got batch \(B)")
            precondition(
                spanContexts == nil,
                "CBv2AttentionV1: a keep mask cannot be combined with vision span masks")
            precondition(
                window(of: kind) == nil && !kind.isBidirectional,
                "CBv2AttentionV1: a keep mask serves causal full attention only")
        }
        precondition(!rows.isEmpty, "CBv2AttentionV1: no rows")
        precondition(
            B == rows.count,
            "CBv2AttentionV1: batch \(B) != bound cache rows \(rows.count)")
        if let spanContexts {
            precondition(
                spanContexts.count == B,
                "CBv2AttentionV1: span contexts \(spanContexts.count) != batch \(B)")
            precondition(
                L > 1,
                "CBv2AttentionV1: decode and one-token chunks never bind span masks")
            precondition(
                !serializeQueries || spanContexts.allSatisfy { $0 == nil },
                "CBv2AttentionV1: serialized MTP queries cannot carry span contexts")
        }
        let effectiveSinks = dispatchSinks(
            sinks, kind: kind, queries: queries, softcap: softcap)

        if B == 1 {
            if serializeQueries, L > 1 {
                return updateAndAttendRowSerialQueries(
                    row: rows[0], kind: kind,
                    queries: queries, keys: keys, values: values,
                    scale: scale, sinks: effectiveSinks, softcap: softcap,
                    keepMask: keepMask, allowNativeVectorBatch: true)
            }
            return updateAndAttendRow(
                row: rows[0], kind: kind,
                queries: queries, keys: keys, values: values,
                scale: scale, sinks: effectiveSinks, softcap: softcap,
                spanContext: spanContexts?[0], keepMask: keepMask,
                metadata: metadata, packet: packet)
        }

        if L == 1 {
            // Batched decode: split queries per row, per-row update + SDPA
            // against that row's own KV, then concatenate. No masks — each row
            // sees exactly its own KV, so batch-composition invariance holds by
            // construction and fully-masked rows cannot exist.
            var outputs: [MLXArray] = []
            outputs.reserveCapacity(B)
            for (index, row) in rows.enumerated() {
                let (cachedKeys, cachedValues) = row.update(
                    keys: keys[index ..< (index + 1)],
                    values: values[index ..< (index + 1)])
                outputs.append(
                    attend(
                        queries: queries[index ..< (index + 1)],
                        keys: cachedKeys, values: cachedValues, scale: scale,
                        L: 1, kL: cachedKeys.dim(2), window: nil,
                        sinks: effectiveSinks, softcap: softcap))
            }
            return concatenated(outputs, axis: 0)
        }

        // Rectangular [B > 1, L > 1] packed prefill or MTP verify: every row
        // takes the same per-row path as a singleton chunk and attends
        // exactly its own KV. Serialized queries remain MTP-only; ordinary
        // packed prefill keeps q-blocking and optional row-local span masks.
        return packedPerRow(batch: B) { index, slice in
            if serializeQueries {
                return updateAndAttendRowSerialQueries(
                    row: rows[index], kind: kind,
                    queries: slice(queries), keys: slice(keys), values: slice(values),
                    scale: scale, sinks: effectiveSinks, softcap: softcap)
            }
            return updateAndAttendRow(
                row: rows[index], kind: kind,
                queries: slice(queries), keys: slice(keys), values: slice(values),
                scale: scale, sinks: effectiveSinks, softcap: softcap,
                spanContext: spanContexts?[index])
        }
    }

    /// The batch-axis decomposition of a PACKED `[B, ...]` prompt pass —
    /// the one definition both backends share, so "packed" cannot come to
    /// mean two different things depending on storage.
    ///
    /// Row `index` sees the `index`-th batch slice of every tensor and
    /// nothing else; the per-row results are rejoined on axis 0 in row
    /// order. What is deliberately NOT shared is the body: the contiguous
    /// per-row path updates the row and masks in RELATIVE coordinates
    /// (`maskMode`, which may return symbolic `.causal`), while the paged
    /// one gathers pages before writing and masks in ABSOLUTE ones (always
    /// `.array` — MLX #3384). Those are genuinely different computations
    /// over genuinely different storage; forcing one body on both would
    /// break the paged file's pinned mask contract. The decomposition is
    /// the part that must not drift, so the decomposition is what is
    /// factored out.
    ///
    /// `batch == 1` passes the caller's tensors through UNSLICED and skips
    /// the concatenate, so a singleton call is provably the same graph it
    /// was before packing existed.
    @inline(__always)
    static func packedPerRow(
        batch: Int, _ row: (_ index: Int, _ slice: (MLXArray) -> MLXArray) -> MLXArray
    ) -> MLXArray {
        precondition(batch >= 1, "CBv2AttentionV1: packed batch must be >= 1")
        if batch == 1 { return row(0, { $0 }) }
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(batch)
        for index in 0 ..< batch {
            outputs.append(row(index, { $0[index ..< (index + 1)] }))
        }
        return concatenated(outputs, axis: 0)
    }

    /// Commit a full-attention prefill chunk's K/V while evaluating only its
    /// newest query (see LastQueryPrefillV2.swift). The newest causal query
    /// sees every key the chunk just wrote, so this is exactly the final row
    /// of ordinary chunk attention — mask-free by construction, which is why
    /// a bound span overlay cannot change it either.
    static func updateAndAttendLastQuery(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float? = nil
    ) -> MLXArray {
        precondition(
            kind.attention == .full,
            "CBv2AttentionV1: last-query prefill requires full attention")
        precondition(
            kind.sharesKVWithLayer == nil,
            "CBv2AttentionV1: last-query prefill cannot own KV for a shared layer")
        precondition(
            queries.ndim == 4 && keys.ndim == 4 && values.ndim == 4,
            "CBv2AttentionV1: last-query prefill tensors must be rank 4")
        let batch = queries.dim(0)
        precondition(
            batch > 0 && rows.count == batch
                && keys.dim(0) == batch && values.dim(0) == batch,
            "CBv2AttentionV1: last-query prefill rows and tensors must share batch B")
        precondition(
            queries.dim(2) == 1,
            "CBv2AttentionV1: last-query prefill requires qL=1")
        let kvLength = keys.dim(2)
        precondition(kvLength > 1, "CBv2AttentionV1: last-query prefill requires kvL>1")
        precondition(
            values.dim(2) == kvLength,
            "CBv2AttentionV1: last-query prefill K/V lengths must match")
        precondition(
            queries.dim(1) == kind.queryHeads && queries.dim(3) == kind.headDim,
            "CBv2AttentionV1: last-query prefill Q shape does not match the layer kind")
        precondition(
            keys.dim(1) == kind.kvHeads && values.dim(1) == kind.kvHeads
                && keys.dim(3) == kind.headDim && values.dim(3) == kind.headDim,
            "CBv2AttentionV1: last-query prefill K/V shape does not match the layer kind")

        let effectiveSinks = dispatchSinks(
            sinks, kind: kind, queries: queries, softcap: softcap)
        return packedPerRow(batch: batch) { index, slice in
            let (cachedKeys, cachedValues) = rows[index].update(
                keys: slice(keys),
                values: slice(values))
            return attend(
                queries: slice(queries),
                keys: cachedKeys, values: cachedValues,
                scale: scale, L: 1, kL: cachedKeys.dim(2), window: nil,
                sinks: effectiveSinks, softcap: softcap)
        }
    }

    /// One row's update + attention — verbatim the single-request ([1, L])
    /// logic, shared by the B == 1 path and the rectangular [B > 1, L > 1]
    /// verify loop so a batched row is bit-identical to running alone.
    private static func updateAndAttendRow(
        row: CBv2SequenceKV, kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?,
        spanContext: CBv2SpanChunkContext?, keepMask: MLXArray? = nil,
        metadata: CBv2AttentionMetadataObservation? = nil,
        packet: CBv2AttentionPacketObservation? = nil
    ) -> MLXArray {
        let L = queries.dim(2)
        precondition(CBv2BranchVerify.tree == nil || L != 16, "CBv2 tree verify: attention took the chain path")
        let (cachedKeys, cachedValues) = row.update(keys: keys, values: values)
        if let keepMask {
            precondition(
                keepMask.dim(2) == L && keepMask.dim(3) == cachedKeys.dim(2),
                "CBv2AttentionV1: keep mask \(keepMask.shape) does not match "
                    + "\(L) queries over \(cachedKeys.dim(2)) keys")
        }
        if shouldBlockQueries(L) && !kind.isBidirectional {
            return attendQueryBlocks(
                queries: queries, keys: cachedKeys, values: cachedValues,
                newTokenCount: L, window: window(of: kind), scale: scale,
                sinks: sinks, softcap: softcap, blockSize: queryBlockSize,
                spanContext: spanContext, keepMask: keepMask)
        }
        if let spanContext {
            return attendSpanChunk(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                L: L, kL: cachedKeys.dim(2), window: window(of: kind),
                context: spanContext, sinks: sinks, softcap: softcap)
        }
        return attend(
            queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
            L: L, kL: cachedKeys.dim(2), window: window(of: kind),
            sinks: sinks, softcap: softcap, bidirectional: kind.isBidirectional,
            keepMask: keepMask, metadata: metadata, packet: packet)
    }

    /// The native-vector verify attention over the stored prefix and the window's new rows, whose
    /// launch also stores the new rows into the cache, so no slice update stands between the rows'
    /// producers and the attention. Same attention bits and same cache bytes as `row.update`
    /// then attention; nil (and no state changed) when the row or the inputs do not qualify.
    private static func attendBeforeWrite(
        row: CBv2SequenceKV, kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?, keepMask: MLXArray?,
        allowNativeVectorBatch: Bool
    ) -> MLXArray? {
        guard allowNativeVectorBatch, CBv2VerifyVectorAttention.kvwriteEnabled,
            let stored = row as? CBv2FullSequenceKV,
            canBatchSerialQueriesWithNativeVector(
                kind: kind, queries: queries, keys: keys, values: values,
                sinks: sinks, softcap: softcap, keepMask: keepMask),
            let (prefixKeys, prefixValues) = stored.prefixViews(appending: keys, values),
            let attended = CBv2VerifyVectorAttention.attendNewRows(
                queries: queries, prefixKeys: prefixKeys, prefixValues: prefixValues,
                newKeys: keys, newValues: values, scale: scale,
                tree: queries.dim(2) == 16 ? CBv2BranchVerify.tree : nil)
        else { return nil }
        stored.commitStoredRows(keys.dim(2))
        return attended
    }

    private static func updateAndAttendRowSerialQueries(
        row: CBv2SequenceKV, kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?,
        keepMask: MLXArray? = nil, allowNativeVectorBatch: Bool = false
    ) -> MLXArray {
        let L = queries.dim(2)
        if let shared = attendBeforeWrite(
            row: row, kind: kind, queries: queries, keys: keys, values: values, scale: scale,
            sinks: sinks, softcap: softcap, keepMask: keepMask,
            allowNativeVectorBatch: allowNativeVectorBatch)
        {
            return shared
        }
        let (cachedKeys, cachedValues) = row.update(keys: keys, values: values)
        if let tree = CBv2BranchVerify.tree, L == 16 {
            precondition(keepMask == nil && allowNativeVectorBatch, "CBv2 tree verify: unsupported attention form")
            if canBatchSerialQueriesWithNativeVector(
                kind: kind, queries: queries, keys: cachedKeys, values: cachedValues,
                sinks: sinks, softcap: softcap, keepMask: nil)
            {
                guard let shared = CBv2VerifyVectorAttention.attendTree(
                    queries: queries, keys: cachedKeys, values: cachedValues, scale: scale, tree: tree)
                else { preconditionFailure("CBv2 tree verify: tree attention kernel unavailable") }
                return shared
            }
            return attendTreeSerialQueries(
                queries: queries, keys: cachedKeys, values: cachedValues, tree: tree,
                window: window(of: kind), scale: scale, sinks: sinks, softcap: softcap)
        }
        if let keepMask {
            precondition(
                keepMask.dim(2) == L && keepMask.dim(3) == cachedKeys.dim(2),
                "CBv2AttentionV1: keep mask \(keepMask.shape) does not match "
                    + "\(L) queries over \(cachedKeys.dim(2)) keys")
        }
        if allowNativeVectorBatch,
            canBatchSerialQueriesWithNativeVector(
                kind: kind, queries: queries, keys: cachedKeys, values: cachedValues,
                sinks: sinks, softcap: softcap, keepMask: keepMask)
        {
            if let shared = CBv2VerifyVectorAttention.attend(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale)
            {
                return shared
            }
            return attend(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                L: L, kL: cachedKeys.dim(2), window: nil, sinks: nil, softcap: nil)
        }
        return attendSerialQueries(
            queries: queries, keys: cachedKeys, values: cachedValues,
            newTokenCount: L, window: window(of: kind), scale: scale,
            sinks: sinks, softcap: softcap, keepMask: keepMask)
    }

    /// The one-pass vector kernel visits each query's visible keys in the
    /// same SIMD-lane order as the serial prefix calls. Keep the total cached
    /// length below every Metal architecture's two-pass selection threshold.
    private static func canBatchSerialQueriesWithNativeVector(
        kind: CBv2LayerKind, queries: MLXArray, keys: MLXArray, values: MLXArray,
        sinks: MLXArray?, softcap: Float?, keepMask: MLXArray?
    ) -> Bool {
        guard kind.attention == .full, !kind.isBidirectional, !kind.hasSinks,
            sinks == nil, softcap == nil, keepMask == nil,
            queries.ndim == 4, keys.ndim == 4, values.ndim == 4,
            queries.dtype == .float32, keys.dtype == .float32, values.dtype == .float32,
            queries.dim(0) == 1, queries.dim(1) == 16, queries.dim(3) == 256,
            (2...16).contains(queries.dim(2)),
            keys.dim(0) == 1, keys.dim(1) == 4, keys.dim(3) == 256,
            values.shape == keys.shape,
            keys.dim(2) >= queries.dim(2), keys.dim(2) < 1024,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return false }

        // This read-only query excludes tracing and retained autodiff graphs.
        // The descriptor identity is discarded; no tensor or result is cached.
        var identity: UInt = 0
        var canCache = false
        return _mlx_array_constant_cache_identity(&identity, &canCache, queries.ctx) == 0
            && canCache
    }

    /// Attend against `sourceRows`' KV WITHOUT updating (Gemma-4 cross-layer
    /// KV sharing: shared layers project Q only and borrow the source
    /// layer's K/V — the source layer already appended this step's tokens
    /// earlier in the forward pass).
    ///
    /// View selection is a pure function of L (pinned-path discipline):
    ///  - CHUNK borrow (`L > 1`): `chunkBorrowViews(of:)`, NOT `snapshot()`.
    ///    After a windowed source's multi-token update, `snapshot()` is the
    ///    POST-eviction ring (≤ window entries), while the source layer's
    ///    own attention saw the PRE-eviction history + chunk
    ///    (`window - 1 + n` entries) so the chunk's earliest queries keep
    ///    their full window. Borrowing must attend the same view, with the
    ///    same `kL > window` array-mask path (the paged backend is already
    ///    exact here by construction — this mirrors its semantics).
    ///  - DECODE borrow (`L == 1`): `snapshot()`. The retained ring IS the
    ///    window (the source's same-step decode update already evicted),
    ///    so the mask-free decode path stays exact.
    static func attendBorrowing(
        sourceRows: [CBv2SequenceKV], sourceKind: CBv2LayerKind, kind: CBv2LayerKind,
        queries: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float? = nil,
        spanContexts: [CBv2SpanChunkContext?]? = nil,
        serializeQueries: Bool = false
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(!sourceRows.isEmpty, "CBv2AttentionV1: no source rows to borrow from")
        precondition(
            B == sourceRows.count,
            "CBv2AttentionV1: batch \(B) != source rows \(sourceRows.count)")
        if let spanContexts {
            precondition(
                spanContexts.count == B,
                "CBv2AttentionV1: span contexts \(spanContexts.count) != batch \(B)")
            precondition(
                L > 1,
                "CBv2AttentionV1: decode and one-token chunks never bind span masks")
            precondition(
                !serializeQueries || spanContexts.allSatisfy { $0 == nil },
                "CBv2AttentionV1: serialized MTP queries cannot carry span contexts")
        }
        let effectiveSinks = dispatchSinks(
            sinks, kind: kind, queries: queries, softcap: softcap)

        if L > 1 {
            if B == 1 {
                if serializeQueries {
                    let (keys, values) = chunkBorrowViews(of: sourceRows[0])
                    return attendSerialQueries(
                        queries: queries, keys: keys, values: values,
                        newTokenCount: L, window: window(of: sourceKind), scale: scale,
                        sinks: effectiveSinks, softcap: softcap)
                }
                return borrowAndAttendRow(
                    sourceRow: sourceRows[0], sourceKind: sourceKind,
                    queries: queries, scale: scale, sinks: effectiveSinks, softcap: softcap,
                    spanContext: spanContexts?[0])
            }
            // Rectangular [B > 1, L > 1] packed prefill / MTP verify:
            // independently borrow each source row's current chunk views.
            var outputs: [MLXArray] = []
            outputs.reserveCapacity(B)
            for (index, row) in sourceRows.enumerated() {
                if serializeQueries {
                    let (keys, values) = chunkBorrowViews(of: row)
                    outputs.append(
                        attendSerialQueries(
                            queries: queries[index ..< (index + 1)],
                            keys: keys, values: values, newTokenCount: L,
                            window: window(of: sourceKind), scale: scale,
                            sinks: effectiveSinks, softcap: softcap))
                } else {
                    outputs.append(
                        borrowAndAttendRow(
                            sourceRow: row, sourceKind: sourceKind,
                            queries: queries[index ..< (index + 1)],
                            scale: scale, sinks: effectiveSinks, softcap: softcap,
                            spanContext: spanContexts?[index]))
                }
            }
            return concatenated(outputs, axis: 0)
        }

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (index, row) in sourceRows.enumerated() {
            let cachedKeys: MLXArray
            let cachedValues: MLXArray
            if let windowed = row as? CBv2WindowedSequenceKV {
                // Ordinary decode borrows the retained ring. A staged serial
                // MTP transaction has not written that ring yet, so borrow
                // the source layer's logical post-update view instead.
                (cachedKeys, cachedValues) = windowed.decodeBorrowableViews()
            } else {
                (cachedKeys, cachedValues, _) = row.snapshot()
            }
            outputs.append(
                attend(
                    queries: B == 1 ? queries : queries[index ..< (index + 1)],
                    keys: cachedKeys, values: cachedValues, scale: scale,
                    L: 1, kL: cachedKeys.dim(2), window: nil,
                    sinks: effectiveSinks, softcap: softcap))
        }
        return B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    /// One row's chunk borrow + attention — verbatim the single-request
    /// ([1, chunk]) borrow logic, shared by the B == 1 path and the
    /// rectangular [B > 1, L > 1] verify loop.
    private static func borrowAndAttendRow(
        sourceRow: CBv2SequenceKV, sourceKind: CBv2LayerKind,
        queries: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float?,
        spanContext: CBv2SpanChunkContext?
    ) -> MLXArray {
        let L = queries.dim(2)
        let (cachedKeys, cachedValues) = chunkBorrowViews(of: sourceRow)
        if shouldBlockQueries(L) && !sourceKind.isBidirectional {
            return attendQueryBlocks(
                queries: queries, keys: cachedKeys, values: cachedValues,
                newTokenCount: L, window: window(of: sourceKind), scale: scale,
                sinks: sinks, softcap: softcap, blockSize: queryBlockSize,
                spanContext: spanContext)
        }
        if let spanContext {
            // Same overlay as the storage-owning path: the MLXVLM reference
            // applies it to the masks KV-shared layers reuse.
            return attendSpanChunk(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                L: L, kL: cachedKeys.dim(2), window: window(of: sourceKind),
                context: spanContext, sinks: sinks, softcap: softcap)
        }
        return attend(
            queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
            L: L, kL: cachedKeys.dim(2), window: window(of: sourceKind),
            sinks: sinks, softcap: softcap, bidirectional: sourceKind.isBidirectional)
    }

    /// The views a borrowing layer must attend for the current PREFILL-CHUNK
    /// step: the windowed source's step-scoped pre-eviction views (see
    /// `CBv2WindowedSequenceKV.borrowableViews()`), else `snapshot()`
    /// (full-attention sources retain everything).
    private static func chunkBorrowViews(of row: CBv2SequenceKV) -> (MLXArray, MLXArray) {
        if let windowed = row as? CBv2WindowedSequenceKV {
            return windowed.borrowableViews()
        }
        let (keys, values, _) = row.snapshot()
        return (keys, values)
    }

    // MARK: - Private

    private static func window(of kind: CBv2LayerKind) -> Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    /// Attend a prompt chunk in QUERY BLOCKS, slicing K/V to each block's own
    /// visible span instead of computing the whole `[L, kL]` rectangle and
    /// masking the excess away.
    ///
    /// Two things this buys, both of which matter:
    ///
    /// 1. WORK. A sliding-window layer's chunk attention today evaluates
    ///    `L x (window - 1 + L)` score positions to use `L x window` of them:
    ///    a `(window - 1 + L)/window` ratio, i.e. 1.499x at L=512/window=1024
    ///    and 2.67x at L=2048. Per block the ratio collapses to
    ///    `(window - 1 + q)/window` — 1.062x at q=128 — because a block only
    ///    ever reads the keys its own queries can see.
    /// 2. MEMORY. The composed attention path (which Gemma 4 always takes:
    ///    MLX's fused kernel supports head_dim 64/80/128 and Gemma 4 uses
    ///    256/512) MATERIALIZES the `[B, heads, L, kL]` score tensor. Blocking
    ///    pins that at `[B, heads, q, kL]`, making it O(1) in chunk length —
    ///    which is what lets `prefillChunkSize` grow without the flat 3 GiB
    ///    activation reserve in `UnifiedMemoryCap` becoming a lie. At 124k
    ///    context a full-attention layer drops from 2.04 GB to 0.51 GB.
    ///
    /// Numerics: NOT bit-identical to the single-call path. The same non-zero
    /// terms are summed (masked entries contribute `exp(-inf) = 0`), but a
    /// different `kL` changes the reduction tiling and may select a different
    /// kernel specialization, so results can differ in the last ulp.
    ///
    /// Scope: this applies to every multi-token prompt call that reaches
    /// `updateAndAttendRow` / `borrowAndAttendRow`, including rectangular
    /// packed prefill. Vision rows keep the same q=128 blocking; each query
    /// block expands its K/V slice only as needed to include complete image
    /// spans touched by that block, then composes causal/window and
    /// bidirectional-span masks. Decode (`L == 1`) remains outside this path.
    ///
    /// `blockSize == 1` reproduces the MTP serial-query path exactly.
    private static func attendQueryBlocks(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        newTokenCount: Int, window: Int?, scale: Float,
        sinks: MLXArray?, softcap: Float?, blockSize: Int,
        spanContext: CBv2SpanChunkContext? = nil,
        keepMask: MLXArray? = nil
    ) -> MLXArray {
        precondition(blockSize >= 1, "CBv2AttentionV1: query block size must be >= 1")
        precondition(
            keepMask == nil || spanContext == nil,
            "CBv2AttentionV1: a keep mask cannot be combined with vision span masks")
        let keyCount = keys.dim(2)
        let historyCount = keyCount - newTokenCount
        precondition(historyCount >= 0)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity((newTokenCount + blockSize - 1) / blockSize)
        let core = CBv2PromptAttentionCore.plan(
            queries: queries, keys: keys, values: values, newTokenCount: newTokenCount,
            window: window, sinks: sinks, softcap: softcap, blockSize: blockSize,
            spanContext: spanContext, keepMask: keepMask)
        var offset = 0
        while offset < newTokenCount {
            let count = min(blockSize, newTokenCount - offset)
            let bounds = queryBlockBounds(
                historyCount: historyCount, offset: offset, count: count, window: window)
            var visibleStart = bounds.visibleStart
            var visibleEnd = bounds.visibleEnd
            var intersectingBlocks: ArraySlice<CBv2ImageSpan>?
            let queryAbsoluteStart =
                spanContext.map { $0.chunkEnd - newTokenCount + offset }
            let keyAbsoluteStart = spanContext.map { $0.chunkEnd - keyCount }
            if let context = spanContext,
                let qStart = queryAbsoluteStart,
                let kStart = keyAbsoluteStart
            {
                let blocks = spanBlocksIntersectingQueryRange(
                    queryAbsoluteStart: qStart, queryCount: count, context: context)
                intersectingBlocks = blocks
                for block in blocks {
                    // Bidirectional span attention can see future keys and,
                    // on windowed layers, older same-span keys. Retain the
                    // complete touched span while keeping all other keys at
                    // the ordinary q-block bounds.
                    visibleStart = min(visibleStart, max(0, block.tokenOffset - kStart))
                    visibleEnd = max(visibleEnd, min(keyCount, block.end - kStart))
                }
            }

            let querySlice = queries[0..., 0..., offset ..< (offset + count), 0...]
            let keySlice = keys[0..., 0..., visibleStart ..< visibleEnd, 0...]
            let valueSlice = values[0..., 0..., visibleStart ..< visibleEnd, 0...]
            if let blocks = intersectingBlocks, !blocks.isEmpty,
                let qStart = queryAbsoluteStart,
                let kStart = keyAbsoluteStart
            {
                outputs.append(
                    attendSpanSlice(
                        queries: querySlice, keys: keySlice, values: valueSlice,
                        scale: scale, queryAbsoluteStart: qStart,
                        keyAbsoluteStart: kStart + visibleStart,
                        window: window, blocks: blocks,
                        sinks: sinks, softcap: softcap))
            } else {
                // The block reads keys `[visibleStart, visibleEnd)` and
                // queries `[offset, offset + count)`, so the mask is sliced
                // to the same window. Column coordinates are absolute over
                // the retained tape on both sides.
                let blockKeepMask = keepMask.map {
                    $0[0..., 0..., offset ..< (offset + count), visibleStart ..< visibleEnd]
                }
                outputs.append(
                    core?.attend(
                        blockIndex: offset / blockSize, queries: querySlice, keys: keySlice,
                        values: valueSlice, scale: scale)
                        ?? attend(
                            queries: querySlice, keys: keySlice, values: valueSlice,
                            scale: scale, L: count, kL: visibleEnd - visibleStart,
                            window: window, sinks: sinks, softcap: softcap,
                            keepMask: blockKeepMask))
            }
            offset += count
        }
        return outputs.count == 1
            ? outputs[0] : CBv2PromptAttentionBlocks.noted(outputs, concatenated(outputs, axis: 2))
    }

    /// `attendSerialQueries` for a draft tree window: query `t` attends to the history and its
    /// own path's keys in path order (the keys a chain verify of that path shows its row).
    private static func attendTreeSerialQueries(
        queries: MLXArray, keys: MLXArray, values: MLXArray, tree: MLXArray,
        window: Int?, scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        precondition(window == nil, "CBv2 tree verify: windowed attention")
        let L = queries.dim(2)
        let history = keys.dim(2) - L
        let host = CBv2BranchVerify.host(tree)
        var outputs: [MLXArray] = []
        for t in 0 ..< L {
            let depth = Int(host[16 + t])
            let rowBase = 32 + 16 * t
            let path = (0 ... depth).map { (d: Int) -> Int32 in
                let ancestor: Int32 = host[rowBase + d]
                return Int32(history) + ancestor
            }
            let index = MLXArray(Array(0 ..< Int32(history)) + path)
            outputs.append(
                attend(
                    queries: queries[0..., 0..., t ..< (t + 1), 0...],
                    keys: keys.take(index, axis: 2), values: values.take(index, axis: 2),
                    scale: scale, L: 1, kL: history + depth + 1, window: nil, sinks: sinks,
                    softcap: softcap, keepMask: nil))
        }
        return concatenated(outputs, axis: 2)
    }

    /// One query at a time — the pinned MTP serial-verification path.
    private static func attendSerialQueries(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        newTokenCount: Int, window: Int?, scale: Float,
        sinks: MLXArray?, softcap: Float?, keepMask: MLXArray? = nil
    ) -> MLXArray {
        attendQueryBlocks(
            queries: queries, keys: keys, values: values,
            newTokenCount: newTokenCount, window: window, scale: scale,
            sinks: sinks, softcap: softcap, blockSize: 1, keepMask: keepMask)
    }

    /// Single-request attention dispatch. Without a softcap this is MLXFast
    /// SDPA with `maskMode(L:kL:window:)`; with a softcap it is the composed
    /// fp32 reference (SDPA cannot express logit softcapping) with the
    /// EQUIVALENT boolean mask — both are pure functions of (L, kL, window),
    /// so each configuration keeps exactly one pinned path per phase.
    private static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        L: Int, kL: Int, window: Int?, sinks: MLXArray?, softcap: Float?,
        bidirectional: Bool = false, keepMask: MLXArray? = nil,
        metadata: CBv2AttentionMetadataObservation? = nil,
        packet: CBv2AttentionPacketObservation? = nil
    ) -> MLXArray {
        // A model may widen Q for safer attention math while retaining compact
        // K/V storage. SDPA requires one dtype, so widen only these views.
        let attentionKeys = keys.dtype == queries.dtype ? keys : keys.asType(queries.dtype)
        let attentionValues =
            values.dtype == queries.dtype ? values : values.asType(queries.dtype)
        if let keepMask {
            // EXACT: the keep mask can only remove keys the causal/window
            // mask already admitted, so the composed mask is the AND of the
            // two. `boolMask` returns nil at L == 1 (the retained KV IS the
            // visible set), where the keep mask stands alone.
            let composed =
                boolMask(L: L, kL: kL, window: window, bidirectional: bidirectional)
                .map { $0 .&& keepMask } ?? keepMask
            guard let softcap else {
                return MLXFast.scaledDotProductAttention(
                    queries: queries, keys: attentionKeys, values: attentionValues,
                    scale: scale, mask: .array(composed), sinks: sinks)
            }
            return PagedAttentionReference.composedAttention(
                queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
                boolMask: composed, sinks: sinks, softcap: softcap)
        }
        guard let softcap else {
            // Sinks arrive ALREADY normalized to the query dtype: see
            // `sdpaSinks` (MLX aborts the process on a wider sink) and
            // `dispatchSinks` (the conversion is hoisted to the top-level
            // call, so this terminal never rebuilds it per row or block).
            assert(
                sinks == nil || sinks!.dtype == queries.dtype,
                "CBv2AttentionV1: sinks must be normalized to the query dtype before SDPA")
            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
                mask: maskMode(
                    L: L, kL: kL, window: window, bidirectional: bidirectional),
                sinks: sinks)
            if let metadata {
                metadata.finish(
                    storage: ["visible_keys": .init(keys), "visible_values": .init(values)],
                    kernelOutputDType: output.dtype, output: output, dispatch: "contiguous_sdpa")
            }
            packet?.finishContiguous(keys: keys, values: values, output: output)
            return output
        }
        return PagedAttentionReference.composedAttention(
            queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
            boolMask: boolMask(
                L: L, kL: kL, window: window, bidirectional: bidirectional),
            sinks: sinks, softcap: softcap)
    }

    /// Boolean causal(∧window) mask (true == attend) equivalent to
    /// `maskMode(L:kL:window:)`, for the composed softcap path. nil for
    /// decode (L == 1: the retained KV IS the window).
    static func boolMask(
        L: Int, kL: Int, window: Int?, bidirectional: Bool = false
    ) -> MLXArray? {
        guard L > 1 else { return nil }
        // Relative coordinates: keys span [0, kL), queries are the last L.
        let qpos = MLXArray(Int32(kL - L) ..< Int32(kL)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(kL)).expandedDimensions(axis: 0)
        var mask = kpos .<= qpos
        if let window, kL > window {
            mask = mask .&& (kpos .> (qpos - Int32(window)))
        }
        if bidirectional {
            // Complete only the trailing current-chunk square. Cached prefix
            // columns keep their established causal/window visibility.
            var reverse = (kpos .>= qpos) .&& (kpos .>= Int32(kL - L))
            if let window {
                reverse = reverse .&& (kpos .< (qpos + Int32(window)))
            }
            mask = mask .|| reverse
        }
        return mask
    }

    // MARK: - Vision span-chunk path (pinned; span-containing chunks only)

    /// Zero-copy selection of the coalesced image blocks that can affect one
    /// query slice. Context blocks are sorted/non-overlapping by resolution,
    /// so intersections form one contiguous `ArraySlice`.
    static func spanBlocksIntersectingQueryRange(
        queryAbsoluteStart: Int, queryCount: Int,
        context: CBv2SpanChunkContext
    ) -> ArraySlice<CBv2ImageSpan> {
        precondition(queryCount >= 0)
        let queryEnd = queryAbsoluteStart + queryCount
        var lower = context.blocks.startIndex
        while lower < context.blocks.endIndex,
            context.blocks[lower].end <= queryAbsoluteStart
        {
            lower += 1
        }
        var upper = lower
        while upper < context.blocks.endIndex,
            context.blocks[upper].tokenOffset < queryEnd
        {
            upper += 1
        }
        return context.blocks[lower ..< upper]
    }

    /// Boolean causal(∧window) mask over arbitrary absolute query/key slices,
    /// overlaid only with the image blocks intersecting this query slice.
    private static func spanMask(
        queryAbsoluteStart: Int, queryCount: Int,
        keyAbsoluteStart: Int, keyCount: Int,
        window: Int?, blocks: ArraySlice<CBv2ImageSpan>
    ) -> MLXArray {
        let qAbs =
            MLXArray(
                Int32(queryAbsoluteStart) ..< Int32(queryAbsoluteStart + queryCount)
            ).expandedDimensions(axis: 1)
        let kAbs =
            MLXArray(
                Int32(keyAbsoluteStart) ..< Int32(keyAbsoluteStart + keyCount)
            ).expandedDimensions(axis: 0)
        var mask = kAbs .<= qAbs
        if let window {
            mask = mask .&& (kAbs .> (qAbs - Int32(window)))
        }
        for block in blocks {
            let lo = Int32(block.tokenOffset)
            let hi = Int32(block.end)
            let qIn = (qAbs .>= lo) .&& (qAbs .< hi)
            let kIn = (kAbs .>= lo) .&& (kAbs .< hi)
            mask = mask .|| (qIn .&& kIn)
        }
        return mask
    }

    /// Whole-chunk form retained for focused mask contracts.
    static func spanChunkMask(
        L: Int, kL: Int, window: Int?, context: CBv2SpanChunkContext
    ) -> MLXArray {
        precondition(L > 1, "span chunks are multi-token by construction")
        return spanMask(
            queryAbsoluteStart: context.chunkEnd - L, queryCount: L,
            keyAbsoluteStart: context.chunkEnd - kL, keyCount: kL,
            window: window, blocks: context.blocks[...])
    }

    /// Span-chunk attention dispatch: always an explicit boolean array mask
    /// (the bidirectional overlay cannot ride the symbolic `.causal` mode).
    /// One pinned path per configuration — plain SDPA, or the composed fp32
    /// reference when a softcap is configured (same split as `attend`).
    private static func attendSpanChunk(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        L: Int, kL: Int, window: Int?, context: CBv2SpanChunkContext,
        sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        attendSpanSlice(
            queries: queries, keys: keys, values: values, scale: scale,
            queryAbsoluteStart: context.chunkEnd - L,
            keyAbsoluteStart: context.chunkEnd - kL,
            window: window, blocks: context.blocks[...],
            sinks: sinks, softcap: softcap)
    }

    private static func attendSpanSlice(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        queryAbsoluteStart: Int, keyAbsoluteStart: Int,
        window: Int?, blocks: ArraySlice<CBv2ImageSpan>,
        sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        let attentionKeys = keys.dtype == queries.dtype ? keys : keys.asType(queries.dtype)
        let attentionValues =
            values.dtype == queries.dtype ? values : values.asType(queries.dtype)
        let mask = spanMask(
            queryAbsoluteStart: queryAbsoluteStart, queryCount: queries.dim(2),
            keyAbsoluteStart: keyAbsoluteStart, keyCount: keys.dim(2),
            window: window, blocks: blocks)
        guard let softcap else {
            // Same contract as `attend`: normalized upstream by
            // `dispatchSinks`, because MLX traps on fp16 queries + fp32 sinks.
            assert(
                sinks == nil || sinks!.dtype == queries.dtype,
                "CBv2AttentionV1: sinks must be normalized to the query dtype before SDPA")
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
                mask: .array(mask), sinks: sinks)
        }
        return PagedAttentionReference.composedAttention(
            queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
            boolMask: mask, sinks: sinks, softcap: softcap)
    }
}

/// Bit-exact stand-in for MLX's one-pass `sdpa_vector` (float32, D = V = 256,
/// causal, no mask, sinks or softcap) on the MTP verify batch: 2 ... 7 queries,
/// 16 query heads over 4 KV heads, fewer than 1024 keys.
///
/// MLX runs one 1024-thread threadgroup per (head, query); its simdgroup g
/// scans keys g, g + 32, ... with an online softmax and the 32 partial states
/// are then combined across simdgroups. Each threadgroup reads the whole KV
/// of its head, so the 4 heads × L queries that share a KV head read it
/// 4L times. Here simdgroup (class g, query t) scans the same keys in the
/// same order for all 4 heads of a KV head, with the per-key arithmetic of
/// MLX's kernel as compiled in `mlx.metallib` (FMA dot product, `fma(o,
/// factor, exp·v)`, `fma(sum, factor, exp)`), and the queries of a class
/// share its keys through one threadgroup. The 32 partial states per (head,
/// query) go to memory and a second kernel combines them with MLX's
/// cross-simdgroup code (lane l = class l). Outputs are the bits of
/// `MLXFast.scaledDotProductAttention(..., mask: .causal)`.
enum CBv2VerifyVectorAttention {
    /// Key classes per partial threadgroup: 2, except 1 on a GPU whose
    /// parsed generation is below MLX's `is_nax_available` rule (as in
    /// `PromptInt8Matmul.architectureHasNAX`), where 64 * L threads can
    /// exceed the pipeline's limit (384 on an M2 at L = 7). A name that
    /// does not parse keeps 2. Each simdgroup computes the same (class,
    /// query) state either way, so the outputs are the same bits.
    private static let classesPerGroup: Int = {
        let name = GPU.deviceInfo().architecture
        guard let range = name.range(of: "applegpu_g") else { return 2 }
        let tail = name[range.upperBound...]
        let digits = tail.prefix(while: { $0.isNumber })
        guard let gen = Int(digits) else { return 2 }
        let variant = tail.dropFirst(digits.count).first
        return gen >= (variant == "p" ? 18 : 17) ? 2 : 1
    }()

    private static let partial = MLXFast.metalKernel(
        name: "cbv2_verify_sdpa_partial",
        inputNames: ["queries", "keys", "values", "n_keys"],
        outputNames: ["part_o", "part_m", "part_s"],
        source: partialSource,
        ensureRowContiguous: false)

    /// Tree twin of a partial source. `tree` is the draft tree descriptor (int32: parent row
    /// [16], depth [16], ancestor row by depth [16][16]; see `DF2Fast.treeSource`). Query row
    /// `t` visits logical keys `0 ... base + depth[t]` (`base` = keys before the window) in the
    /// record's class and order, reading in-window key `base + i` from its path's row
    /// `base + ancestor[t][i]`, so each (class, query) state is the bits of a chain verify of
    /// its path.
    static func treeSource(_ record: String) -> String? {
        var s = record
        func swap(_ from: String, _ to: String, count: Int = 1) -> Bool {
            guard s.components(separatedBy: from).count == count + 1 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        let ok = swap(
            "const int limit = N - int(L) + int(t);",
            """
            const int tv_base = N - int(L);
            const int tree_depth = tree[16 + t];
            const int limit = tv_base + tree_depth;
            const int tree_d = (c - tv_base) & 31;
            const int tree_pi = tv_base + (tree_d <= tree_depth ? tree[32 + 16 * t + tree_d] : 0);
            """)
            && swap(
                "for (int i = c; i <= limit; i += 32) {",
                "for (int i = c; i <= limit; i += 32) {\nconst int pi = i < tv_base ? i : tree_pi;")
            && swap("(kp + i * ks + ", "(kp + pi * ks + ", count: 2)
            && swap("(vp + i * vs + ", "(vp + pi * vs + ", count: 2)
        return ok ? s : nil
    }

    private static let partialTree: MLXFast.MLXFastKernel? = treeSource(partialSource).map {
        MLXFast.metalKernel(
            name: "cbv2_verify_sdpa_partial_tree",
            inputNames: ["queries", "keys", "values", "n_keys", "tree"],
            outputNames: ["part_o", "part_m", "part_s"],
            source: $0,
            ensureRowContiguous: false)
    }

    private static let partialSource = """
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            constexpr int D = 256;
            constexpr int G = 4;
            const uint L = uint(N_QUERIES);
            const uint t = simdgroup_index_in_threadgroup / uint(CPT);
            const uint kvh = threadgroup_position_in_grid.z;
            const uint h0 = kvh * G;
            const uint simd_lid = thread_index_in_simdgroup;
            const int c = int(threadgroup_position_in_grid.x * uint(CPT)
                + simdgroup_index_in_threadgroup % uint(CPT));
            const int N = n_keys[0];
            const float scale = as_type<float>(n_keys[1]);
            const int limit = N - int(L) + int(t);
            float q[G][8];
            float o[G][8];
            float max_score[G];
            float sum_exp_score[G];
            #pragma clang loop unroll(full)
            for (int h = 0; h < G; h++) {
              const device float* qp = queries + (h0 + h) * queries_strides[1]
                  + t * queries_strides[2] + simd_lid * 8;
              const packed_float4 q0 = *((const device packed_float4*)(qp + 0));
              const packed_float4 q1 = *((const device packed_float4*)(qp + 4));
              q[h][0] = static_cast<float>(scale) * q0.x;
              q[h][1] = static_cast<float>(scale) * q0.y;
              q[h][2] = static_cast<float>(scale) * q0.z;
              q[h][3] = static_cast<float>(scale) * q0.w;
              q[h][4] = static_cast<float>(scale) * q1.x;
              q[h][5] = static_cast<float>(scale) * q1.y;
              q[h][6] = static_cast<float>(scale) * q1.z;
              q[h][7] = static_cast<float>(scale) * q1.w;
              #pragma clang loop unroll(full)
              for (int j = 0; j < 8; j++) {
                o[h][j] = 0;
              }
              max_score[h] = -metal::numeric_limits<float>::max();
              sum_exp_score[h] = 0;
            }
            const device float* kp = keys + kvh * keys_strides[1] + simd_lid * 8;
            const device float* vp = values + kvh * values_strides[1] + simd_lid * 8;
            const int64_t ks = keys_strides[2];
            const int64_t vs = values_strides[2];
            for (int i = c; i <= limit; i += 32) {
              const packed_float4 k0 = *((const device packed_float4*)(kp + i * ks + 0));
              const packed_float4 k1 = *((const device packed_float4*)(kp + i * ks + 4));
              const packed_float4 v0 = *((const device packed_float4*)(vp + i * vs + 0));
              const packed_float4 v1 = *((const device packed_float4*)(vp + i * vs + 4));
              const float k[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
              const float v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
              #pragma clang loop unroll(full)
              for (int h = 0; h < G; h++) {
                float score = 0;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 8; j++) {
                  score = metal::fma(q[h][j], k[j], score);
                }
                score = simd_sum(score);
                float new_max = max(max_score[h], score);
                float factor = fast::exp(max_score[h] - new_max);
                float exp_score = fast::exp(score - new_max);
                max_score[h] = new_max;
                sum_exp_score[h] = metal::fma(sum_exp_score[h], factor, exp_score);
                #pragma clang loop unroll(full)
                for (int j = 0; j < 8; j++) {
                  o[h][j] = metal::fma(o[h][j], factor, exp_score * v[j]);
                }
              }
            }
            #pragma clang loop unroll(full)
            for (int h = 0; h < G; h++) {
              const uint slot = ((h0 + h) * L + t) * 32u + uint(c);
              device float* op = part_o + slot * D + simd_lid * 8;
              *((device packed_float4*)(op + 0)) = packed_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
              *((device packed_float4*)(op + 4)) = packed_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
              if (simd_lid == 0) {
                part_m[slot] = max_score[h];
                part_s[slot] = sum_exp_score[h];
              }
            }
            """

    /// One threadgroup per (head, query, half of the output dims); the 32
    /// partials pass through threadgroup memory so lane l holds partial l.
    private static let combine = MLXFast.metalKernel(
        name: "cbv2_verify_sdpa_combine",
        inputNames: ["part_o", "part_m", "part_s"],
        outputNames: ["out"],
        source: """
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            constexpr int v_per_thread = 16;
            threadgroup float staged[32 * 129];
            const uint head = threadgroup_position_in_grid.x / 2u;
            const uint half_ = threadgroup_position_in_grid.x % 2u;
            const uint t = threadgroup_position_in_grid.y;
            const uint L = threadgroups_per_grid.y;
            const uint tid = thread_index_in_threadgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint base = (head * L + t) * 32u;
            #pragma clang loop unroll(full)
            for (uint idx = tid; idx < 32u * 32u; idx += 256u) {
              const uint c = idx / 32u, x = (idx % 32u) * 4u;
              const packed_float4 packet = *((const device packed_float4*)(part_o + (base + c) * 256u + half_ * 128u + x));
              *((threadgroup packed_float4*)(staged + c * 129u + x)) = packet;
            }
            float max_score = part_m[base + simd_lid];
            float new_max = simd_max(max_score);
            float factor = fast::exp(max_score - new_max);
            float sum_exp_score = simd_sum(part_s[base + simd_lid] * factor);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float o[v_per_thread];
            #pragma clang loop unroll(full)
            for (int i = 0; i < v_per_thread; i++) {
              o[i] = simd_sum(staged[simd_lid * 129u + simd_gid * v_per_thread + i] * factor);
              o[i] = sum_exp_score == 0 ? o[i] : metal::precise::divide(o[i], sum_exp_score);
            }
            if (simd_lid == 0) {
              device float* op = out + (head * L + t) * 256u + half_ * 128u
                  + simd_gid * v_per_thread;
              *((device packed_float4*)(op + 0)) = packed_float4(o[0], o[1], o[2], o[3]);
              *((device packed_float4*)(op + 4)) = packed_float4(o[4], o[5], o[6], o[7]);
              *((device packed_float4*)(op + 8)) = packed_float4(o[8], o[9], o[10], o[11]);
              *((device packed_float4*)(op + 12)) = packed_float4(o[12], o[13], o[14], o[15]);
            }
            """)


    /// Same lane/reduction order as combine, but read partials directly.
    /// Removes the threadgroup transpose and barrier, never changes key scanning.
    private static let combineDirect = MLXFast.metalKernel(
        name: "cbv2_verify_sdpa_combine_direct",
        inputNames: ["part_o", "part_m", "part_s"], outputNames: ["out"],
        source: """
        #pragma clang fp reassociate(off)
        #pragma clang fp contract(off)
        constexpr int v_per_thread = 16;

        const uint head = threadgroup_position_in_grid.x / 2u;
        const uint half_ = threadgroup_position_in_grid.x % 2u;
        const uint t = threadgroup_position_in_grid.y;
        const uint L = threadgroups_per_grid.y;
        const uint tid = thread_index_in_threadgroup;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const uint base = (head * L + t) * 32u;
        float max_score = part_m[base + simd_lid];
        float new_max = simd_max(max_score);
        float factor = fast::exp(max_score - new_max);
        float sum_exp_score = simd_sum(part_s[base + simd_lid] * factor);

        float o[v_per_thread];
        #pragma clang loop unroll(full)
        for (int i = 0; i < v_per_thread; i++) {
          o[i] = simd_sum(part_o[(base + simd_lid) * 256u + half_ * 128u + simd_gid * v_per_thread + i] * factor);
          o[i] = sum_exp_score == 0 ? o[i] : metal::precise::divide(o[i], sum_exp_score);
        }
        if (simd_lid == 0) {
          device float* op = out + (head * L + t) * 256u + half_ * 128u
              + simd_gid * v_per_thread;
          *((device packed_float4*)(op + 0)) = packed_float4(o[0], o[1], o[2], o[3]);
          *((device packed_float4*)(op + 4)) = packed_float4(o[4], o[5], o[6], o[7]);
          *((device packed_float4*)(op + 8)) = packed_float4(o[8], o[9], o[10], o[11]);
          *((device packed_float4*)(op + 12)) = packed_float4(o[12], o[13], o[14], o[15]);
        }
        """)

    private static let combineDirectOffered: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VATTN_DIRECT_COMBINE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let combineDirectPassed: Bool = {
        guard combineDirectOffered else { return false }
        var passed = false
        do { passed = try withError { checkDirectCombine() } }
        catch { FileHandle.standardError.write(Data("vattn direct combine check error: \(error)\n".utf8)) }
        FileHandle.standardError.write(Data("vattn direct combine checked: \(passed) (18 finite bitwise cases and negative control)\n".utf8))
        return passed
    }()

    private static func checkDirectCombine() -> Bool {
        var negativeControl = false
        for L in [2, 7, 16] {
            for seed: Float in [1, 17, 33] {
                let o = probe([16 * L * 32, 256], seed: seed, scale: 0.2)
                let m = probe([16 * L * 32], seed: seed + 1, scale: 3)
                let sums = abs(probe([16 * L * 32], seed: seed + 2, scale: 20)) + 0.1
                for empty in [false, true] {
                    let inputs = [o, m, empty ? MLXArray.zeros(like: sums) : sums]
                    let reference = combine(inputs, grid: (32 * 256, L, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[1, 16, L, 256]], outputDTypes: [.float32])[0]
                    let direct = combineDirect(inputs, grid: (32 * 256, L, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[1, 16, L, 256]], outputDTypes: [.float32])[0]
                    let finite = isFinite(reference).all() .&& isFinite(direct).all()
                    let same = (reference.view(dtype: .uint32) .== direct.view(dtype: .uint32)).all()
                    eval(finite, same)
                    guard finite.item(Bool.self), same.item(Bool.self) else { return false }
                    if !negativeControl {
                        let changed = direct + 1
                        negativeControl = !(reference.view(dtype: .uint32) .== changed.view(dtype: .uint32)).all().item(Bool.self)
                    }
                }
            }
        }
        return negativeControl
    }

    private static let pairedPartial = MLXFast.metalKernel(
        name: "cbv2_verify_sdpa_paired",
        inputNames: ["queries", "keys", "values", "n_keys"],
        outputNames: ["part_o", "part_m", "part_s"],
        source: pairedSource,
        ensureRowContiguous: false)

    private static let pairedTree: MLXFast.MLXFastKernel? = treeSource(pairedSource).map {
        MLXFast.metalKernel(
            name: "cbv2_verify_sdpa_paired_tree",
            inputNames: ["queries", "keys", "values", "n_keys", "tree"],
            outputNames: ["part_o", "part_m", "part_s"],
            source: $0,
            ensureRowContiguous: false)
    }

    private static let pairedSource = """
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            constexpr int D = 256;
            constexpr int G = GH;
            const uint L = uint(N_QUERIES);
            const uint t = simdgroup_index_in_threadgroup / uint(CPT);
            const uint kvh = threadgroup_position_in_grid.z;
            const uint h0 = kvh * 4u + threadgroup_position_in_grid.y * G;
            const uint simd_lid = thread_index_in_simdgroup;
            const int c = int(threadgroup_position_in_grid.x * uint(CPT)
                + simdgroup_index_in_threadgroup % uint(CPT));
            const int N = n_keys[0];
            const float scale = as_type<float>(n_keys[1]);
            const int limit = N - int(L) + int(t);
            float q[G][8];
            float o[G][8];
            float max_score[G];
            float sum_exp_score[G];
            #pragma clang loop unroll(full)
            for (int h = 0; h < G; h++) {
              const device float* qp = queries + (h0 + h) * queries_strides[1]
                  + t * queries_strides[2] + simd_lid * 8;
              const packed_float4 q0 = *((const device packed_float4*)(qp + 0));
              const packed_float4 q1 = *((const device packed_float4*)(qp + 4));
              q[h][0] = static_cast<float>(scale) * q0.x;
              q[h][1] = static_cast<float>(scale) * q0.y;
              q[h][2] = static_cast<float>(scale) * q0.z;
              q[h][3] = static_cast<float>(scale) * q0.w;
              q[h][4] = static_cast<float>(scale) * q1.x;
              q[h][5] = static_cast<float>(scale) * q1.y;
              q[h][6] = static_cast<float>(scale) * q1.z;
              q[h][7] = static_cast<float>(scale) * q1.w;
              #pragma clang loop unroll(full)
              for (int j = 0; j < 8; j++) {
                o[h][j] = 0;
              }
              max_score[h] = -metal::numeric_limits<float>::max();
              sum_exp_score[h] = 0;
            }
            const device float* kp = keys + kvh * keys_strides[1] + simd_lid * 8;
            const device float* vp = values + kvh * values_strides[1] + simd_lid * 8;
            const int64_t ks = keys_strides[2];
            const int64_t vs = values_strides[2];
            for (int i = c; i <= limit; i += 32) {
              const packed_float4 k0 = *((const device packed_float4*)(kp + i * ks + 0));
              const packed_float4 k1 = *((const device packed_float4*)(kp + i * ks + 4));
              const packed_float4 v0 = *((const device packed_float4*)(vp + i * vs + 0));
              const packed_float4 v1 = *((const device packed_float4*)(vp + i * vs + 4));
              const float k[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
              const float v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
              #pragma clang loop unroll(full)
              for (int h = 0; h < G; h++) {
                float score = 0;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 8; j++) {
                  score = metal::fma(q[h][j], k[j], score);
                }
                score = simd_sum(score);
                float new_max = max(max_score[h], score);
                float factor = fast::exp(max_score[h] - new_max);
                float exp_score = fast::exp(score - new_max);
                max_score[h] = new_max;
                sum_exp_score[h] = metal::fma(sum_exp_score[h], factor, exp_score);
                #pragma clang loop unroll(full)
                for (int j = 0; j < 8; j++) {
                  o[h][j] = metal::fma(o[h][j], factor, exp_score * v[j]);
                }
              }
            }
            #pragma clang loop unroll(full)
            for (int h = 0; h < G; h++) {
              const uint slot = ((h0 + h) * L + t) * 32u + uint(c);
              device float* op = part_o + slot * D + simd_lid * 8;
              *((device packed_float4*)(op + 0)) = packed_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
              *((device packed_float4*)(op + 4)) = packed_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
              if (simd_lid == 0) {
                part_m[slot] = max_score[h];
                part_s[slot] = sum_exp_score[h];
              }
            }
            """

    /// `DARKBLOOM_BONSAI_VATTN_LEAN=0` keeps the record partial kernels.
    private static let leanSwitch: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VATTN_LEAN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Lean twin of a record partial source (same keys in the same order, same floating-point
    /// operations on the same values). The key and value rows come from running pointers stepped
    /// by 32 rows, in 32-bit row strides, where the record multiplies the row by the 64-bit
    /// stride at every key. The tree twin runs the chain keys (rows before the window) in that
    /// loop and peels the window key: with a depth below 32 a class has at most one, it is the
    /// class's last key, and it sits at `tree_d` rows into the window, so the per-key window test
    /// and row select go away. nil when an anchor is missing.
    static func leanSource(_ record: String, tree: Bool) -> String? {
        let header = "for (int i = c; i <= limit; i += 32) {"
        let footer = "\n}\n#pragma clang loop unroll(full)\nfor (int h = 0; h < G; h++) {\n  const uint slot"
        let limitLine = "const int limit = N - int(L) + int(t);"
        func once(_ text: String, _ piece: String) -> Range<String.Index>? {
            guard text.components(separatedBy: piece).count == 2 else { return nil }
            return text.range(of: piece)
        }
        guard let head = once(record, header), let foot = once(record, footer),
            head.upperBound <= foot.lowerBound
        else { return nil }
        var prefix = String(record[..<head.lowerBound])
        let suffix = String(record[foot.lowerBound...])
        var body = String(record[head.upperBound ..< foot.lowerBound])
        func swap(_ text: inout String, _ from: String, _ to: String, count: Int = 1) -> Bool {
            guard text.components(separatedBy: from).count == count + 1 else { return false }
            text = text.replacingOccurrences(of: from, with: to)
            return true
        }
        var ok = swap(&prefix, "const int64_t ks = keys_strides[2];",
                "const int ks = int(keys_strides[2]);\nconst int kstep = 32 * ks;")
            && swap(&prefix, "const int64_t vs = values_strides[2];",
                "const int vs = int(values_strides[2]);\nconst int vstep = 32 * vs;")
            && swap(&body, "(kp + i * ks + ", "(kr + ", count: 2)
            && swap(&body, "(vp + i * vs + ", "(vr + ", count: 2)
        guard ok else { return nil }
        if !tree {
            return prefix
                + "const device float* kr = kp + c * ks;\nconst device float* vr = vp + c * vs;\n"
                + "for (int i = c; i <= limit; i += 32, kr += kstep, vr += vstep) {"
                + body + suffix
        }
        ok = swap(&prefix, limitLine, """
            const int tv_base = N - int(L);
            const int tree_depth = tree[16 + t];
            const int tree_d = (c - tv_base) & 31;
            const int tree_pi = tv_base + (tree_d <= tree_depth ? tree[32 + 16 * t + tree_d] : 0);
            """)
        guard ok else { return nil }
        return prefix
            + "const device float* kr = kp + c * ks;\nconst device float* vr = vp + c * vs;\n"
            + "for (int i = c; i < tv_base; i += 32, kr += kstep, vr += vstep) {"
            + body
            + "\n}\nif (tree_d <= tree_depth) {\n"
            + "const device float* kr = kp + tree_pi * ks;\nconst device float* vr = vp + tree_pi * vs;"
            + body + suffix
    }

    private static func leanKernel(_ name: String, _ source: String?, tree: Bool)
        -> MLXFast.MLXFastKernel?
    {
        guard leanSwitch, let source else { return nil }
        return MLXFast.metalKernel(
            name: name,
            inputNames: tree ? ["queries", "keys", "values", "n_keys", "tree"]
                : ["queries", "keys", "values", "n_keys"],
            outputNames: ["part_o", "part_m", "part_s"],
            source: source,
            ensureRowContiguous: false)
    }

    private static let partialLean = leanKernel(
        "cbv2_verify_sdpa_partial_lean", leanSource(partialSource, tree: false), tree: false)
    private static let partialTreeLean = leanKernel(
        "cbv2_verify_sdpa_partial_tree_lean", leanSource(partialSource, tree: true), tree: true)
    private static let pairedLean = leanKernel(
        "cbv2_verify_sdpa_paired_lean", leanSource(pairedSource, tree: false), tree: false)
    private static let pairedTreeLean = leanKernel(
        "cbv2_verify_sdpa_paired_tree_lean", leanSource(pairedSource, tree: true), tree: true)

    /// `DARKBLOOM_BONSAI_KVWRITE=0` keeps the write-then-attend order of the target K/V rows.
    private static let kvwriteSwitch: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_KVWRITE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Twin of a lean source whose window keys come from the round's new rows (`new_k`, `new_v`,
    /// any strides) instead of cache rows `tv_base ..< N`, and which stores those rows into the
    /// cache buffer behind `keys` and `values` itself (the cache is mutated through its input
    /// pointer; the rows `tv_base ..< N` are read by no thread of this launch). `keys` and
    /// `values` are read only before the window. The chain twin splits the key loop at the
    /// window start: the same classes, the same key order, the same loads and floating-point
    /// operations. The tree twin reads its peeled window key from the new row
    /// `tree_pi - tv_base`. nil when an anchor is missing.
    static func newRowsSource(_ lean: String, tree: Bool) -> String? {
        func once(_ text: String, _ piece: String) -> Range<String.Index>? {
            guard text.components(separatedBy: piece).count == 2 else { return nil }
            return text.range(of: piece)
        }
        guard let anchor = once(lean, "const int vstep = 32 * vs;") else { return nil }
        var s = lean
        s.replaceSubrange(anchor, with: """
            const int vstep = 32 * vs;
            const device float* nkp = new_k + kvh * new_k_strides[1] + simd_lid * 8;
            const device float* nvp = new_v + kvh * new_v_strides[1] + simd_lid * 8;
            const int nks = int(new_k_strides[2]);
            const int nvs = int(new_v_strides[2]);
            if (c == 0 && threadgroup_position_in_grid.y == 0) {
              const int64_t row = int64_t(N - int(L)) + int64_t(t);
              device float* kw = const_cast<device float*>(keys)
                  + int64_t(kvh) * keys_strides[1] + row * keys_strides[2] + simd_lid * 8;
              device float* vw = const_cast<device float*>(values)
                  + int64_t(kvh) * values_strides[1] + row * values_strides[2] + simd_lid * 8;
              const device float* ksrc = new_k + int64_t(kvh) * new_k_strides[1]
                  + int64_t(t) * new_k_strides[2] + simd_lid * 8;
              const device float* vsrc = new_v + int64_t(kvh) * new_v_strides[1]
                  + int64_t(t) * new_v_strides[2] + simd_lid * 8;
              *((device packed_float4*)(kw + 0)) = *((const device packed_float4*)(ksrc + 0));
              *((device packed_float4*)(kw + 4)) = *((const device packed_float4*)(ksrc + 4));
              *((device packed_float4*)(vw + 0)) = *((const device packed_float4*)(vsrc + 0));
              *((device packed_float4*)(vw + 4)) = *((const device packed_float4*)(vsrc + 4));
            }
            """)
        if tree {
            let from = "const device float* kr = kp + tree_pi * ks;\nconst device float* vr = vp + tree_pi * vs;"
            guard let at = once(s, from) else { return nil }
            s.replaceSubrange(at, with: """
                const device float* kr = nkp + (tree_pi - tv_base) * nks;
                const device float* vr = nvp + (tree_pi - tv_base) * nvs;
                """)
            return s
        }
        let header = "const device float* kr = kp + c * ks;\nconst device float* vr = vp + c * vs;\n"
            + "for (int i = c; i <= limit; i += 32, kr += kstep, vr += vstep) {"
        let footer = "\n}\n#pragma clang loop unroll(full)\nfor (int h = 0; h < G; h++) {\n  const uint slot"
        guard let head = once(s, header), let foot = once(s, footer), head.upperBound <= foot.lowerBound
        else { return nil }
        let body = String(s[head.upperBound ..< foot.lowerBound])
        return String(s[..<head.lowerBound]) + """
            const int tv_base = N - int(L);
            const device float* kr = kp + c * ks;
            const device float* vr = vp + c * vs;
            int i = c;
            for (; i < tv_base; i += 32, kr += kstep, vr += vstep) {
            """ + body + """

            }
            for (; i <= limit; i += 32) {
            const device float* kr = nkp + (i - tv_base) * nks;
            const device float* vr = nvp + (i - tv_base) * nvs;
            """ + body + String(s[foot.lowerBound...])
    }

    private static func newRowsKernel(_ name: String, _ lean: String?, tree: Bool)
        -> MLXFast.MLXFastKernel?
    {
        guard kvwriteSwitch, let lean, let source = newRowsSource(lean, tree: tree) else { return nil }
        return MLXFast.metalKernel(
            name: name,
            inputNames: tree ? ["queries", "keys", "values", "n_keys", "tree", "new_k", "new_v"]
                : ["queries", "keys", "values", "n_keys", "new_k", "new_v"],
            outputNames: ["part_o", "part_m", "part_s"],
            source: source,
            ensureRowContiguous: false)
    }

    private static let partialNew = newRowsKernel(
        "cbv2_verify_sdpa_partial_newrows", leanSource(partialSource, tree: false), tree: false)
    private static let partialTreeNew = newRowsKernel(
        "cbv2_verify_sdpa_partial_tree_newrows", leanSource(partialSource, tree: true), tree: true)
    private static let pairedNew = newRowsKernel(
        "cbv2_verify_sdpa_paired_newrows", leanSource(pairedSource, tree: false), tree: false)
    private static let pairedTreeNew = newRowsKernel(
        "cbv2_verify_sdpa_paired_tree_newrows", leanSource(pairedSource, tree: true), tree: true)

    /// The partial states `[part_o, part_m, part_s]` of the record kernel this device dispatches,
    /// or of its lean twin when `lean` and the twin exists. `tree` is nil for a chain verify.
    /// With `newRows`, `keys` and `values` hold only the rows before the window and the window's
    /// rows come from `newRows` (see `newRowsSource`).
    private static func launchStates(
        queries: MLXArray, keys: MLXArray, values: MLXArray, params: MLXArray,
        tree: MLXArray?, L: Int, lean: Bool, newRows: (MLXArray, MLXArray)? = nil
    ) -> [MLXArray]? {
        var inputs = tree.map { [queries, keys, values, params, $0] } ?? [queries, keys, values, params]
        if let newRows { inputs += [newRows.0, newRows.1] }
        let outputShapes = [[16 * L * 32, 256], [16 * L * 32], [16 * L * 32]]
        let outputDTypes: [DType] = [.float32, .float32, .float32]
        let isTree = tree != nil
        if L > 8 && classesPerGroup == 2 {
            let twin = newRows != nil ? (isTree ? pairedTreeNew : pairedNew)
                : lean ? (isTree ? pairedTreeLean : pairedLean) : nil
            guard let kernel = twin ?? (newRows != nil ? nil : isTree ? pairedTree : pairedPartial)
            else { return nil }
            return kernel(
                inputs, template: [("N_QUERIES", L), ("CPT", 1), ("GH", 2)],
                grid: (32 * 32 * L, 2, 4), threadGroup: (32 * 1 * L, 1, 1),
                outputShapes: outputShapes, outputDTypes: outputDTypes)
        }
        let twin = newRows != nil ? (isTree ? partialTreeNew : partialNew)
            : lean ? (isTree ? partialTreeLean : partialLean) : nil
        guard let kernel = twin ?? (newRows != nil ? nil : isTree ? partialTree : partial)
        else { return nil }
        return kernel(
            inputs, template: [("N_QUERIES", L), ("CPT", classesPerGroup)],
            grid: (32 * 32 * L, 1, 4), threadGroup: (32 * classesPerGroup * L, 1, 1),
            outputShapes: outputShapes, outputDTypes: outputDTypes)
    }

    /// True when the 32-bit row offsets of the lean twins cannot overflow for `array`.
    private static func leanStrideFits(_ array: MLXArray, keys nKeys: Int) -> Bool {
        guard let s = mlx_array_strides(array.ctx) else { return false }
        let rowStride = Int(s[2])
        return rowStride >= 0 && (nKeys + 64) * rowStride < Int(Int32.max)
    }

    private static func leanApplies(keys: MLXArray, values: MLXArray, nKeys: Int) -> Bool {
        leanVerdict && leanStrideFits(keys, keys: nKeys) && leanStrideFits(values, keys: nKeys)
    }

    /// Whether this device's lean twins gave the record kernels' bits on fixed probes (chain and
    /// tree, the two row counts the engine verifies, window-edge key counts), checked once. A GPU
    /// with one class per group keeps the record kernels, since the check's
    /// largest launch does not exist there.
    private static let leanVerdict: Bool = {
        guard leanSwitch else { return false }
        var passed = false
        do {
            passed = try withError { checkLean() }
        } catch {
            FileHandle.standardError.write(Data("vattn lean check error: \(error)\n".utf8))
        }
        FileHandle.standardError.write(Data("vattn lean checked: \(passed)\n".utf8))
        return passed
    }()

    static func prepareLean() { _ = leanVerdict; _ = combineDirectPassed; _ = kvwriteVerdict }

    private static func probe(_ shape: [Int], seed: Float, scale: Float) -> MLXArray {
        let n = shape.reduce(1, *)
        let i = MLXArray(0 ..< n).asType(.float32)
        let amplitude = 1 + 0.5 * MLX.cos(i * 0.0137 + seed)
        return (MLX.sin(i * (0.7548776 + seed)) * amplitude * scale).reshaped(shape)
    }

    private static func probeTree(_ parents: [Int]) -> MLXArray {
        var paths = [[Int]]()
        for row in parents.indices {
            var path = parents[row] < 0 ? [] : paths[parents[row]]
            path.append(row)
            paths.append(path)
        }
        var descriptor = parents.map { Int32($0) }
        descriptor += paths.map { Int32($0.count - 1) }
        for path in paths {
            descriptor += path.map { Int32($0) }
            descriptor += Array(repeating: Int32(0), count: 16 - path.count)
        }
        return MLXArray(descriptor)
    }

    private static func checkLean() -> Bool {
        guard classesPerGroup == 2 else { return false }
        let trees: [[Int]] = [
            Array(-1 ..< 15),
            [-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 11, 4, 13, 14],
        ]
        let treeChoices: [MLXArray?] = [nil] + trees.map { probeTree($0) }
        for L in [16, 7] {
            let queries = probe([1, 16, L, 256], seed: 0.01, scale: 1.5)
            for nKeys in [L, L + 1, L + 31, L + 33, 512, 640] {
                let keys = probe([1, 4, nKeys, 256], seed: 0.02, scale: 1.5)
                let values = probe([1, 4, nKeys, 256], seed: 0.03, scale: 1.0)
                let params = MLXArray([Int32(nKeys), Int32(bitPattern: Float(0.0625).bitPattern)])
                for tree in treeChoices {
                    guard
                        let record = launchStates(
                            queries: queries, keys: keys, values: values, params: params,
                            tree: tree, L: L, lean: false),
                        let lean = launchStates(
                            queries: queries, keys: keys, values: values, params: params,
                            tree: tree, L: L, lean: true)
                    else { return false }
                    for (a, b) in zip(record, lean) {
                        let same = (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all().item(Bool.self)
                        if !same { return false }
                    }
                }
            }
        }
        return true
    }

    /// Whether the new-row kernels gave the lean kernels' partial states, and left the cache
    /// holding the bytes a slice update of the same rows leaves (the whole buffer, rows after the
    /// window included), bit for bit, for every verify row count 2 ... 16 (chain) and for tree
    /// windows at 16, over window-edge prefix lengths, with row-contiguous and transposed new
    /// rows; and whether new rows shifted by one row differ in both (negative control). Checked
    /// once; only a GPU the lean twins serve (two classes per group) uses them.
    private static let kvwriteVerdict: Bool = {
        guard kvwriteSwitch, classesPerGroup == 2, leanVerdict else { return false }
        var passed = false
        do {
            passed = try withError { checkNewRows() }
        } catch {
            FileHandle.standardError.write(Data("kvwrite check error: \(error)\n".utf8))
        }
        FileHandle.standardError.write(Data("kvwrite checked: \(passed)\n".utf8))
        return passed
    }()

    static var kvwriteEnabled: Bool { kvwriteVerdict }

    private static func checkNewRows() -> Bool {
        let trees: [[Int]] = [
            Array(-1 ..< 15),
            [-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 11, 4, 13, 14],
        ]
        func equal(_ a: MLXArray, _ b: MLXArray) -> Bool {
            (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all().item(Bool.self)
        }
        func same(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
            zip(a, b).allSatisfy { (x: MLXArray, y: MLXArray) -> Bool in equal(x, y) }
        }
        var negativeControls = 0
        var cases = 0
        for L in 2 ... 16 {
            let treeChoices: [MLXArray?] = L == 16
                ? [nil] + trees.map { (parents: [Int]) -> MLXArray? in probeTree(parents) } : [nil]
            let queries = probe([1, 16, L, 256], seed: 0.01, scale: 1.5)
            for base in [1, 33, 512, 640] {
                let params = MLXArray([Int32(base + L), Int32(bitPattern: Float(0.0625).bitPattern)])
                for transposed in [false, true] {
                    let newK = transposed
                        ? probe([1, L, 4, 256], seed: 0.04, scale: 1.5).transposed(0, 2, 1, 3)
                        : probe([1, 4, L, 256], seed: 0.04, scale: 1.5)
                    let newV = transposed
                        ? probe([1, L, 4, 256], seed: 0.05, scale: 1.0).transposed(0, 2, 1, 3)
                        : probe([1, 4, L, 256], seed: 0.05, scale: 1.0)
                    let shiftedK = concatenated([newK[.ellipsis, 1..., 0...], newK[.ellipsis, ..<1, 0...]], axis: 2)
                    let shiftedV = concatenated([newV[.ellipsis, 1..., 0...], newV[.ellipsis, ..<1, 0...]], axis: 2)
                    for tree in treeChoices {
                        func fresh(_ seed: Float, _ scale: Float) -> MLXArray {
                            let cache = probe([1, 4, base + L + 24, 256], seed: seed, scale: scale)
                            eval(cache)
                            return cache
                        }
                        let cacheK = fresh(0.02, 1.5)
                        let cacheV = fresh(0.03, 1.0)
                        let shiftedCacheK = fresh(0.02, 1.5)
                        let shiftedCacheV = fresh(0.03, 1.0)
                        let writtenK = concatenated(
                            [cacheK[.ellipsis, ..<base, 0...], newK, cacheK[.ellipsis, (base + L)..., 0...]], axis: 2)
                        let writtenV = concatenated(
                            [cacheV[.ellipsis, ..<base, 0...], newV, cacheV[.ellipsis, (base + L)..., 0...]], axis: 2)
                        eval(writtenK, writtenV)
                        guard
                            let reference = launchStates(
                                queries: queries, keys: writtenK[.ellipsis, ..<(base + L), 0...],
                                values: writtenV[.ellipsis, ..<(base + L), 0...], params: params,
                                tree: tree, L: L, lean: true),
                            let moved = launchStates(
                                queries: queries, keys: cacheK[.ellipsis, ..<base, 0...],
                                values: cacheV[.ellipsis, ..<base, 0...], params: params,
                                tree: tree, L: L, lean: true, newRows: (newK, newV)),
                            let shifted = launchStates(
                                queries: queries, keys: shiftedCacheK[.ellipsis, ..<base, 0...],
                                values: shiftedCacheV[.ellipsis, ..<base, 0...], params: params,
                                tree: tree, L: L, lean: true, newRows: (shiftedK, shiftedV))
                        else { return false }
                        eval(reference + moved + shifted)
                        guard same(reference, moved), equal(cacheK, writtenK), equal(cacheV, writtenV)
                        else { return false }
                        cases += 1
                        if !same(reference, shifted) && !equal(shiftedCacheK, writtenK)
                            && !equal(shiftedCacheV, writtenV)
                        {
                            negativeControls += 1
                        }
                    }
                }
            }
        }
        return cases > 0 && negativeControls == cases
    }

    /// `[1, 16, L, 256]`, or nil when the inputs do not qualify.
    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float
    ) -> MLXArray? {
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4,
            queries.dtype == .float32, keys.dtype == .float32, values.dtype == .float32
        else { return nil }
        let qShape = queries.shape4
        let kShape = keys.shape4
        guard qShape.0 == 1, qShape.1 == 16, qShape.3 == 256,
            (2 ... 16).contains(qShape.2),
            kShape.0 == 1, kShape.1 == 4, kShape.3 == 256, values.shape4 == kShape,
            kShape.2 >= qShape.2, kShape.2 < 1024,
            queries.hasUnitInnerStride4, keys.hasUnitInnerStride4, values.hasUnitInnerStride4,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let L = qShape.2
        let nKeys = kShape.2
        let params = HostConstants.recentInt32([Int32(nKeys), Int32(bitPattern: scale.bitPattern)])
        guard let states = launchStates(
            queries: queries, keys: keys, values: values, params: params, tree: nil, L: L,
            lean: leanApplies(keys: keys, values: values, nKeys: nKeys))
        else { return nil }
        let combineKernel = combineDirectPassed ? combineDirect : combine
        return combineKernel(
            CBv2VerifyAttentionPartial.noted(states), grid: (32 * 256, L, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 16, L, 256]], outputDTypes: [.float32])[0]
    }

    /// `attend` for a draft tree window (`tree`: see `treeSource`).
    static func attendTree(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, tree: MLXArray
    ) -> MLXArray? {
        guard tree.dtype == .int32, tree.size >= 288,
            queries.ndim == 4, keys.ndim == 4, values.ndim == 4,
            queries.dtype == .float32, keys.dtype == .float32, values.dtype == .float32
        else { return nil }
        let qShape = queries.shape4
        let kShape = keys.shape4
        guard qShape.0 == 1, qShape.1 == 16, qShape.3 == 256,
            (2 ... 16).contains(qShape.2),
            kShape.0 == 1, kShape.1 == 4, kShape.3 == 256, values.shape4 == kShape,
            kShape.2 >= qShape.2, kShape.2 < 1024,
            queries.hasUnitInnerStride4, keys.hasUnitInnerStride4, values.hasUnitInnerStride4,
            StreamOrDevice.default == StreamOrDevice.gpu
        else { return nil }
        let L = qShape.2
        let params = HostConstants.recentInt32([Int32(kShape.2), Int32(bitPattern: scale.bitPattern)])
        guard let states = launchStates(
            queries: queries, keys: keys, values: values, params: params, tree: tree, L: L,
            lean: leanApplies(keys: keys, values: values, nKeys: kShape.2))
        else { return nil }
        let combineKernel = combineDirectPassed ? combineDirect : combine
        return combineKernel(
            CBv2VerifyAttentionPartial.noted(states), grid: (32 * 256, L, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 16, L, 256]], outputDTypes: [.float32])[0]
    }

    /// `attend` (chain) or `attendTree` (`tree`) over the cached rows before the window
    /// (`prefixKeys`, `prefixValues`) and the window's rows `newKeys`, `newValues` in place, so
    /// the attention does not wait for a cache write; the launch stores the rows into the cache
    /// behind `prefixKeys` and `prefixValues` (which must be views of the cache's leading rows
    /// at offset 0, owned by the caller alone). Returns the attention and its partial states.
    /// nil when the inputs do not qualify or the load-time check failed.
    static func attendNewRows(
        queries: MLXArray, prefixKeys: MLXArray, prefixValues: MLXArray,
        newKeys: MLXArray, newValues: MLXArray, scale: Float, tree: MLXArray?
    ) -> MLXArray? {
        guard kvwriteVerdict,
            queries.ndim == 4, prefixKeys.ndim == 4, prefixValues.ndim == 4,
            newKeys.ndim == 4, newValues.ndim == 4,
            queries.dtype == .float32, prefixKeys.dtype == .float32, prefixValues.dtype == .float32,
            newKeys.dtype == .float32, newValues.dtype == .float32
        else { return nil }
        if let tree { guard tree.dtype == .int32, tree.size >= 288 else { return nil } }
        let qShape = queries.shape4
        let pShape = prefixKeys.shape4
        let nShape = newKeys.shape4
        let L = qShape.2
        guard qShape.0 == 1, qShape.1 == 16, qShape.3 == 256, (2 ... 16).contains(L),
            tree == nil || L == 16,
            pShape.0 == 1, pShape.1 == 4, pShape.2 > 0, pShape.3 == 256,
            prefixValues.shape4 == pShape, nShape == (1, 4, L, 256), newValues.shape4 == nShape,
            pShape.2 + L < 1024,
            queries.hasUnitInnerStride4, prefixKeys.hasUnitInnerStride4, prefixValues.hasUnitInnerStride4,
            newKeys.hasUnitInnerStride4, newValues.hasUnitInnerStride4,
            StreamOrDevice.default == StreamOrDevice.gpu,
            leanStrideFits(prefixKeys, keys: pShape.2 + L), leanStrideFits(prefixValues, keys: pShape.2 + L),
            leanStrideFits(newKeys, keys: L), leanStrideFits(newValues, keys: L)
        else { return nil }
        let params = HostConstants.recentInt32([Int32(pShape.2 + L), Int32(bitPattern: scale.bitPattern)])
        guard let states = launchStates(
            queries: queries, keys: prefixKeys, values: prefixValues, params: params, tree: tree, L: L,
            lean: true, newRows: (newKeys, newValues))
        else { return nil }
        let combineKernel = combineDirectPassed ? combineDirect : combine
        let attended = combineKernel(
            CBv2VerifyAttentionPartial.noted(states), grid: (32 * 256, L, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 16, L, 256]], outputDTypes: [.float32])[0]
        return attended
    }
}

extension MLXArray {
    /// True when this 4D array has inner stride 1, without allocating a Swift `[Int]` array.
    public var hasUnitInnerStride4: Bool {
        guard let s = mlx_array_strides(ctx) else { return false }
        return s[3] == 1
    }

    /// True when this 4D array has inner stride 1 and all non-negative strides, without allocating `[Int]`.
    public var hasNonNegativeUnitInnerStride4: Bool {
        guard let s = mlx_array_strides(ctx) else { return false }
        return s[3] == 1 && s[0] >= 0 && s[1] >= 0 && s[2] >= 0
    }
}

/// The partial-state launch of the verify attention (`CBv2VerifyVectorAttention`),
/// handed to a caller that orders a launch after it: a launch made to depend on
/// the partial states runs beside the partial kernel instead of behind the
/// combine. Graph build only; nothing is held outside `capturing`.
public enum CBv2VerifyAttentionPartial {
    nonisolated(unsafe) private static var active = false
    nonisolated(unsafe) private static var captured: MLXArray?

    /// `body()` and the partial states its verify attention launched, if any.
    public static func capturing<T>(_ body: () -> T) -> (T, MLXArray?) {
        active = true
        captured = nil
        defer {
            active = false
            captured = nil
        }
        let result = body()
        return (result, captured)
    }

    static func noted(_ states: [MLXArray]) -> [MLXArray] {
        if active { captured = states[0] }
        return states
    }

    /// Runs the lean twins' one-time bit check at load.
    public static func prepareLean() { CBv2VerifyVectorAttention.prepareLean() }
}

/// The query-block outputs (`attendQueryBlocks`) of a prompt attention, handed to a caller that
/// reads them in place of their concatenation: the concatenation is then never evaluated. Graph
/// build only; nothing is held outside `capturing`.
public enum CBv2PromptAttentionBlocks {
    nonisolated(unsafe) private static var active = false
    nonisolated(unsafe) private static var captured: (whole: MLXArray, blocks: [MLXArray])?

    /// The query-block width the blocks were cut at.
    public static var blockSize: Int { CBv2AttentionV1.queryBlockSize }

    /// `body()` and, when it returned an `attendQueryBlocks` concatenation itself, the blocks.
    public static func capturing(_ body: () -> MLXArray) -> (MLXArray, [MLXArray]?) {
        active = true
        captured = nil
        defer {
            active = false
            captured = nil
        }
        let result = body()
        guard let blocks = captured, blocks.whole === result else { return (result, nil) }
        return (result, blocks.blocks)
    }

    static func noted(_ blocks: [MLXArray], _ whole: MLXArray) -> MLXArray {
        if active { captured = (whole, blocks) }
        return whole
    }
}

/// The prompt attention of a full-attention layer, composed in Swift from MLX's own fallback graph
/// (`fast.cpp`'s composed attention branch, the one the 128-row blocks take on every
/// GPU) with two optional substitutions, each served only after a boot check finds the block outputs equal
/// to the record's bit for bit: the queries arrive already multiplied by the scale (`Qwen35VerifyQKNormRoPE.applyPrompt`
/// folds the power-of-two multiply into the q/k norm and RoPE kernel), and `where(causal mask)` + `softmax`
/// is one kernel with `softmax_single_row`'s thread layout and reduction order. Graph build only; the
/// per-call state is set by `capturing` and cleared on exit.
public enum CBv2PromptAttentionCore {
    /// `DARKBLOOM_BONSAI_ATTN_SMAX=0` keeps `where` + `softmax`.
    public static let softmaxOffered = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_ATTN_SMAX"] != "0"
    public static let negativeControl =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_ATTN_SMAX_NEGCTL"] == "1"
    /// Written once by the boot check before the resident serves; read-only afterwards.
    nonisolated(unsafe) public static var softmaxVerified = false
    /// Whether the composed path serves at all (the q fold or the softmax is verified), set with the flags.
    nonisolated(unsafe) public static var engaged = false

    nonisolated(unsafe) private static var capturingNow = false
    nonisolated(unsafe) private static var prescaled: [MLXArray]?
    nonisolated(unsafe) private static var consumed = false
    nonisolated(unsafe) private static var dimsCache: [Int: MLXArray] = [:]

    /// `body()` with the composed path armed, and whether `attendQueryBlocks` took it. `prescaled`: the query
    /// blocks already multiplied by the scale, or nil to multiply here.
    public static func capturing(prescaled blocks: [MLXArray]?, _ body: () -> MLXArray) -> (MLXArray, Bool) {
        capturingNow = true
        prescaled = blocks
        consumed = false
        defer {
            capturingNow = false
            prescaled = nil
            consumed = false
        }
        let result = body()
        return (result, consumed)
    }

    struct Plan {
        let prescaled: [MLXArray]?
        let fusedSoftmax: Bool

        func attend(
            blockIndex: Int, queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float
        ) -> MLXArray {
            let query = prescaled.map { (blocks: [MLXArray]) -> MLXArray in blocks[blockIndex] } ?? queries
            return CBv2PromptAttentionCore.composed(
                query: query, keys: keys, values: values, scale: scale,
                scaled: prescaled != nil, fusedSoftmax: fusedSoftmax)
        }
    }

    /// Whether a prompt of `rows` rows is cut into 128-row blocks that all take MLX's composed attention branch:
    /// a block of 8 rows or fewer takes the fused vector kernel, so such a prompt keeps the
    /// record's call.
    public static func serves(rows: Int) -> Bool {
        rows > 128 && (rows % 128 == 0 || rows % 128 > 8)
    }

    /// The plan for one `attendQueryBlocks` call, or nil to keep the record's attention call. Only the plain
    /// causal float32 full-attention prompt qualifies.
    static func plan(
        queries: MLXArray, keys: MLXArray, values: MLXArray, newTokenCount: Int, window: Int?,
        sinks: MLXArray?, softcap: Float?, blockSize: Int, spanContext: CBv2SpanChunkContext?,
        keepMask: MLXArray?
    ) -> Plan? {
        guard capturingNow, engaged, serves(rows: newTokenCount), blockSize == 128, window == nil,
            sinks == nil, softcap == nil, spanContext == nil, keepMask == nil,
            queries.ndim == 4, keys.ndim == 4, values.ndim == 4, queries.dim(0) == 1,
            queries.dtype == .float32, keys.dtype == .float32, values.dtype == .float32,
            queries.dim(2) == newTokenCount, keys.dim(1) == values.dim(1),
            keys.dim(1) > 0, queries.dim(1) % keys.dim(1) == 0
        else { return nil }
        if let blocks = prescaled {
            let count = (newTokenCount + blockSize - 1) / blockSize
            guard blocks.count == count else { return nil }
            for (index, block) in blocks.enumerated() {
                let rows = min(blockSize, newTokenCount - index * blockSize)
                guard block.dtype == .float32, block.shape == [1, queries.dim(1), rows, queries.dim(3)]
                else { return nil }
            }
        }
        guard prescaled != nil || softmaxVerified else { return nil }
        consumed = true
        return Plan(prescaled: prescaled, fusedSoftmax: softmaxVerified)
    }

    /// Row `r` of a `[.., n, kL]` score block reads keys `0 ... r + kL - n + shift` (the causal mask of
    /// `fast.cpp`: `q_idx = r + kL - n >= k_idx`), then `softmax_single_row` (`softmax.h`) unchanged:
    /// N_READS 4, one thread per 4 keys, a threadgroup of ceil(kL / 4) rounded up to 32, the same simd and
    /// threadgroup reductions, `fast::exp`, `1 / sum`. A masked value is the `where` fill, -FLT_MAX.
    /// `dims` = [n, kL, shift]; `shift` is 0 outside the boot check's negative control.
    private static let softmaxKernel = MLXFast.metalKernel(
        name: "cbv2_prompt_causal_softmax",
        inputNames: ["scores", "dims"], outputNames: ["probs"],
        source: """
            constexpr int N_READS = 4;
            const int axis_size = dims[1];
            const uint gid = threadgroup_position_in_grid.x;
            const int lid = int(thread_position_in_threadgroup.x);
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const int last_visible = int(gid % uint(dims[0])) + (axis_size - dims[0]) + dims[2];
            threadgroup float local_max[32];
            threadgroup float local_normalizer[32];
            const device float* in = scores + size_t(gid) * size_t(axis_size);
            float ld[N_READS];
            for (int i = 0; i < N_READS; i++) {
              const int key = lid * N_READS + i;
              ld[i] = key < axis_size
                  ? (key <= last_visible ? in[key] : -metal::numeric_limits<float>::max())
                  : -metal::numeric_limits<float>::infinity();
            }
            if (simd_group_id == 0) {
              local_max[simd_lane_id] = -metal::numeric_limits<float>::infinity();
              local_normalizer[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float maxval = -metal::numeric_limits<float>::max();
            for (int i = 0; i < N_READS; i++) {
              maxval = (maxval < ld[i]) ? ld[i] : maxval;
            }
            maxval = simd_max(maxval);
            if (simd_lane_id == 0) {
              local_max[simd_group_id] = maxval;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
              maxval = simd_max(local_max[simd_lane_id]);
              if (simd_lane_id == 0) {
                local_max[0] = maxval;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maxval = local_max[0];
            float normalizer = 0;
            for (int i = 0; i < N_READS; i++) {
              const float exp_x = metal::fast::exp(ld[i] - maxval);
              ld[i] = exp_x;
              normalizer += exp_x;
            }
            normalizer = simd_sum(normalizer);
            if (simd_lane_id == 0) {
              local_normalizer[simd_group_id] = normalizer;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
              normalizer = simd_sum(local_normalizer[simd_lane_id]);
              if (simd_lane_id == 0) {
                local_normalizer[0] = normalizer;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            normalizer = 1 / local_normalizer[0];
            device float* out_row = probs + size_t(gid) * size_t(axis_size);
            for (int i = 0; i < N_READS; i++) {
              const int key = lid * N_READS + i;
              if (key < axis_size) {
                out_row[key] = ld[i] * normalizer;
              }
            }
            """)

    /// `where(causal mask, scores, min)` then `softmax` over the last axis, of a `[.., n, kL]` float32 block
    /// with kL <= 4096 (above that MLX's softmax is the looped kernel).
    public static func causalSoftmax(_ scores: MLXArray, rows n: Int, keyCount kL: Int, shift: Int32 = 0) -> MLXArray {
        let key = n << 32 | kL << 8 | Int(shift + 128)
        let dims: MLXArray
        if let cached = dimsCache[key] {
            dims = cached
        } else {
            dims = MLXArray([Int32(n), Int32(kL), shift])
            dimsCache[key] = dims
        }
        let threads = 32 * (((kL + 3) / 4 + 31) / 32)
        return softmaxKernel(
            [scores, dims], grid: (scores.size / kL * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [scores.shape], outputDTypes: [.float32])[0]
    }

    /// MLX's composed attention graph for a causal float32 block: `query` `[1, qHeads, n, d]`
    /// against `keys`, `values` `[1, kvHeads, kL, d]` (strided views are fine), every op the fallback's.
    /// `scaled`: `query` already carries the scale.
    public static func composed(
        query: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, scaled: Bool,
        fusedSoftmax: Bool, softmaxShift: Int32 = 0
    ) -> MLXArray {
        let n = query.dim(2)
        let kL = keys.dim(2)
        var q = scaled ? query : multiply(MLXArray(scale), query)
        var k = keys
        var v = values
        let repeats = query.dim(1) / keys.dim(1)
        if repeats > 1 {
            q = unflatten(q, axis: 1, shape: [keys.dim(1), repeats])
            k = expandedDimensions(k, axis: 2)
            v = expandedDimensions(v, axis: 2)
        }
        var scores = matmul(q, swappedAxes(k, -1, -2))
        let probs: MLXArray
        if fusedSoftmax && kL <= 4096 {
            probs = causalSoftmax(scores, rows: n, keyCount: kL, shift: softmaxShift)
        } else {
            let offset = kL - n
            let queryIndex = expandedDimensions(MLXArray.arange(offset, n + offset), axis: 1)
            let keyIndex = expandedDimensions(MLXArray.arange(0, kL), axis: 0)
            scores = MLX.where(
                greaterEqual(queryIndex, keyIndex), scores, MLXArray(-Float.greatestFiniteMagnitude))
            probs = softmax(scores, axes: [-1], precise: true)
        }
        let out = matmul(probs, v)
        return repeats > 1 ? flattened(out, start: 1, end: 2) : out
    }
}

/// Draft tree verify: the round's tree descriptor (int32 [333], `DF2Fast.treeSource`) for
/// the target verify, set by the engine around the verify forward and read by the attention,
/// RoPE and GDN layers; nil keeps every record kernel.
public enum CBv2BranchVerify {
    nonisolated(unsafe) public static var tree: MLXArray?
    nonisolated(unsafe) public private(set) static var committedRowsIsBranch = false
    nonisolated(unsafe) public private(set) static var committedRowsArray: MLXArray?
    /// The accepted path's verify rows (anchor first) while a tree round commits; nil or the
    /// prefix 0 ..< n keeps the record replay.
    nonisolated(unsafe) public static var committedRows: [Int32]? {
        didSet {
            if let rows = committedRows, isBranch(rows) {
                committedRowsIsBranch = true
                committedRowsArray = MLXArray(rows)
            } else {
                committedRowsIsBranch = false
                committedRowsArray = nil
            }
        }
    }

    /// The parent rows (`treeDesc[0 ..< 16]`) of the tree round that commits, set with
    /// `committedRows` on every tree round, a chain-prefix path included.
    nonisolated(unsafe) public static var committedParents: [Int32]?

    static let snapshotNegativeControl =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_SNAP_COMMIT_NEGCTL"] == "1"

    /// Host mirror of the GDN tree verify's snapshot pick (`Qwen35FusedGDNVerify.snapshotTreeSource`):
    /// row s scores depth(s) + 1 for each restart row t > s (parent not t - 1) that has s as an
    /// ancestor; the highest score wins, then the higher row. nil when no row scores or `parents`
    /// is not a 16-row tree with every parent above its child.
    public static func snapshotRow(parents: [Int32]) -> Int? {
        guard parents.count == 16, parents[0] < 0 else { return nil }
        var depth = [Int](repeating: 0, count: 16)
        for t in 1 ..< 16 {
            let p = Int(parents[t])
            guard (0 ..< t).contains(p) else { return nil }
            depth[t] = depth[p] + 1
        }
        var best = 0
        for s in 0 ..< 15 {
            var score = 0
            for t in (s + 1) ..< 16 where Int(parents[t]) != t - 1 {
                var a = Int(parents[t])
                while a > s { a = Int(parents[a]) }
                if a == s { score += depth[s] + 1 }
            }
            if score != 0 { best = max(best, (score << 4) | s) }
        }
        guard best != 0 else { return nil }
        let row = best & 15
        return snapshotNegativeControl && row > 0 ? Int(parents[row]) : row
    }

    /// `(row, rows)` when the committed `path` (anchor first) passes through the snapshot row:
    /// the snapshot holds the state after the path's first `rows` rows.
    public static func snapshotPrefix(parents: [Int32], path: [Int32]) -> (row: Int, rows: Int)? {
        guard !path.isEmpty, path[0] == 0, parents.count == 16 else { return nil }
        for i in 1 ..< path.count {
            guard (1 ..< 16).contains(Int(path[i])), parents[Int(path[i])] == path[i - 1] else { return nil }
        }
        guard let row = snapshotRow(parents: parents),
            let index = path.firstIndex(of: Int32(row))
        else { return nil }
        return (row, index + 1)
    }

    /// Recent tree proposals: the drafter registers each tree's tokens with its descriptor,
    /// and the engine finds the descriptor of the block it verifies by identity.
    nonisolated(unsafe) private static var proposals: [(tokens: MLXArray, desc: MLXArray)] = []

    public static func register(tokens: MLXArray, desc: MLXArray) {
        proposals.append((tokens, desc))
        if proposals.count > 4 { proposals.removeFirst() }
    }

    public static func desc(for tokens: MLXArray) -> MLXArray? {
        proposals.last(where: { $0.tokens === tokens })?.desc
    }

    /// True when `rows` is not `0 ..< rows.count`.
    public static func isBranch(_ rows: [Int32]) -> Bool {
        for i in 0 ..< rows.count where Int(rows[i]) != i { return true }
        return false
    }

    /// Host copy of the descriptor, read once per tree (a sync; only the serial attention
    /// fallback past 1024 keys needs it).
    nonisolated(unsafe) private static var hostDesc: (desc: MLXArray, host: [Int32])?
    static func host(_ desc: MLXArray) -> [Int32] {
        if let h = hostDesc, h.desc === desc { return h.host }
        let host = desc.asArray(Int32.self)
        hostDesc = (desc, host)
        return host
    }
}
