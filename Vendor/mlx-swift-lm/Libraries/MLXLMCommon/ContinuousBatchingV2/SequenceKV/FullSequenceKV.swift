// FullSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for FULL attention layers.
//
// One instance owns the K/V for ONE sequence at ONE layer. There is no shared
// batch frontier, no left padding, and no batch-wide trim: joining or leaving
// a batch never touches this object (batch membership is just list membership
// in `CBv2LayerCache.rows`).

import Cmlx
import Foundation
import MLX

/// Counters for the v2 core runtime's own host-interaction points.
///
/// The engine step loop must never force a host sync (`.item()`, `asArray`,
/// blocking `eval`) and must never rebuild per-row metadata arrays from host
/// integers outside membership changes. Any CBv2 core code that *does* touch
/// the host goes through these counters so tests can assert the step loop is
/// clean (see `CBv2CoreTests`). This deliberately does not instrument MLX
/// itself — only our own sync points.
public enum CBv2CoreInstrumentation {
    private static let lock = NSLock()

    nonisolated(unsafe) private static var _hostSyncs = 0
    nonisolated(unsafe) private static var _positionOffsetsHostRebuilds = 0

    /// Number of host syncs performed by CBv2 core code.
    public static var hostSyncs: Int {
        lock.lock()
        defer { lock.unlock() }
        return _hostSyncs
    }

    /// Number of times a layer cache rebuilt `positionOffsets` from host
    /// integers. Must only ever increase on batch membership changes.
    public static var positionOffsetsHostRebuilds: Int {
        lock.lock()
        defer { lock.unlock() }
        return _positionOffsetsHostRebuilds
    }

    /// Gate for `recordHostSync`: the engine's finalize readbacks count
    /// only while a test has switched counting on. Off (the default) the
    /// step path pays one static bool read per readback and never touches
    /// the lock — the same pattern as the engine's signposter gate. Tests
    /// flip it on before stepping and reset it afterwards.
    nonisolated(unsafe) static var countingEnabled = false

    static func recordHostSync() {
        guard countingEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        _hostSyncs += 1
    }

    static func recordPositionOffsetsHostRebuild() {
        lock.lock()
        defer { lock.unlock() }
        _positionOffsetsHostRebuilds += 1
    }
}

/// Internal hook so `CBv2LayerCache` can hand a row's lazily-mutated storage
/// arrays to the engine loop's `asyncEval` (graph/metadata hygiene: no
/// unconsumed lazy chain may grow O(steps) — DAR-325).
protocol CBv2InnerStateProviding {
    func cbv2InnerState() -> [MLXArray]
}

/// `CBv2SequenceKV` for full (non-windowed) attention.
///
/// Storage is one contiguous `[1, kvHeads, capacity, headDim]` buffer per
/// K and V, grown by doubling (initial capacity = promptLength + 256, capped
/// at `maxLength`). Appends are slice assignments — `mlx_slice_update`
/// donates the input buffer when refcount permits, so an append is O(n), not
/// O(cache). `update` returns temporal-order zero-copy strided views
/// `[..., 0..<retained, :]`; MLX SDPA accepts strided K/V.
public final class CBv2FullSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {

    /// Extra slots allocated beyond the prompt so the first decode steps
    /// don't immediately grow the buffer.
    static let initialSlack = 256

    public private(set) var absoluteOffset: Int = 0
    public var retainedCount: Int { absoluteOffset }

    /// Hard cap on this sequence's length; growth beyond it is an engine
    /// admission bug and traps.
    public let maxLength: Int

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?
    private var capacity: Int

    /// - Parameters:
    ///   - promptLength: expected prompt length, used to size the initial
    ///     allocation (`promptLength + 256`, capped at `maxLength`).
    ///   - maxLength: maximum total tokens this sequence may ever hold.
    ///   - kvHeads/headDim: from the layer's `CBv2LayerKind`; validated
    ///     against the arrays passed to `update`.
    public init(promptLength: Int, maxLength: Int, kvHeads: Int, headDim: Int) {
        precondition(maxLength > 0, "CBv2FullSequenceKV: maxLength must be > 0")
        precondition(
            promptLength <= maxLength,
            "CBv2FullSequenceKV: promptLength \(promptLength) exceeds maxLength \(maxLength)")
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacity = min(maxLength, max(1, promptLength + Self.initialSlack))
    }

    public var byteCount: Int {
        (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
    }

    /// Transfer an exclusively owned, fully authenticated native destination.
    /// No prefix copy or lazy assignment may retain the staging buffers.
    init(
        restoredKeys: MLXArray, restoredValues: MLXArray, offset: Int,
        maxLength: Int, kvHeads: Int, headDim: Int
    ) throws {
        let shape = [1, kvHeads, maxLength, headDim]
        guard maxLength > 0, offset > 0, offset <= maxLength,
            restoredKeys.shape == shape, restoredValues.shape == shape,
            restoredKeys.dtype == restoredValues.dtype
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacity = maxLength
        self.absoluteOffset = offset
        self.keys = restoredKeys
        self.values = restoredValues
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2FullSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2FullSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newValues.dim(2) == n,
            "CBv2FullSequenceKV: keys/values token count mismatch")
        precondition(
            absoluteOffset + n <= maxLength,
            "CBv2FullSequenceKV: append past maxLength (\(absoluteOffset) + \(n) > \(maxLength)) — admission bug"
        )

        if keys == nil, absoluteOffset == 0, Self.firstFillVerified,
            let (filledKeys, filledValues) = Self.firstFill(
                newKeys, newValues, capacity: min(maxLength, max(capacity, n)))
        {
            capacity = min(maxLength, max(capacity, n))
            keys = filledKeys
            values = filledValues
        } else {
            ensureCapacity(absoluteOffset + n, keyTemplate: newKeys, valueTemplate: newValues)

            if keys!.ndim == 4, newKeys.ndim == 4, values!.ndim == 4, newValues.ndim == 4 {
                keys = Self.sliceUpdate(keys!, newKeys, absoluteOffset ..< (absoluteOffset + n))
                values = Self.sliceUpdate(values!, newValues, absoluteOffset ..< (absoluteOffset + n))
            } else {
                keys![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = newKeys
                values![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = newValues
            }
        }
        absoluteOffset += n

        return (
            keys![.ellipsis, ..<absoluteOffset, 0...],
            values![.ellipsis, ..<absoluteOffset, 0...]
        )
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        guard let keys, let values else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            keys[.ellipsis, ..<absoluteOffset, 0...],
            values[.ellipsis, ..<absoluteOffset, 0...],
            absoluteOffset
        )
    }

    /// Plain rollback is already value-exact (see `rollback`: the offset
    /// decrement makes the un-confirmed tail structurally unreachable and
    /// the confirmed prefix is untouched), so speculative begin/commit are
    /// the contract's default no-ops.
    public var supportsSpeculativeWrites: Bool { true }

    /// Rollback the last `n` tokens (speculative rejection). The un-confirmed
    /// tail is structurally unreachable afterwards: every view this class
    /// hands out is sliced to `..<absoluteOffset`, and the tail slots are
    /// overwritten by the next `update` before they can ever be exposed —
    /// so no zeroing pass is needed.
    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2FullSequenceKV.rollback: negative n")
        precondition(
            n <= absoluteOffset,
            "CBv2FullSequenceKV.rollback(\(n)) exceeds retained \(absoluteOffset)")
        absoluteOffset -= n
    }

    public func compactTree(base: Int, rows: [Int32]) {
        guard let k = keys, let v = values, k.ndim == 4, v.ndim == 4 else {
            preconditionFailure("CBv2FullSequenceKV.compactTree: no 4-D storage")
        }
        let index = MLXArray(rows.map { Int32(base) + $0 })
        let range = base ..< (base + rows.count)
        keys = Self.sliceUpdate(k, k.take(index, axis: 2), range)
        values = Self.sliceUpdate(v, v.take(index, axis: 2), range)
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    // MARK: - First fill

    /// `DARKBLOOM_BONSAI_ATTN_KVFILL=0` keeps the zero fill and the slice update of a sequence's first append.
    public static let firstFillOffered = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_ATTN_KVFILL"] != "0"
    static let firstFillNegativeControl =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_ATTN_KVFILL_NEGCTL"] == "1"
    /// Written once by `prepareFirstFill` before the resident serves; read-only afterwards.
    nonisolated(unsafe) public private(set) static var firstFillVerified = false

    /// The first append as one kernel: the whole `[1, heads, capacity, headDim]` buffer, the source's rows
    /// first (read through its strides, so a transposed value view needs no copy) and zeros after, where the
    /// record allocates `zeros` and slice-updates the rows in. The same stored values, one pass less.
    private static let firstFillKernel = MLXFast.metalKernel(
        name: "cbv2_first_kv_fill",
        inputNames: ["src"], outputNames: ["dst"],
        source: """
            const uint x = thread_position_in_grid.x;
            const uint row = thread_position_in_grid.y;
            const uint head = thread_position_in_grid.z;
            float4 value = float4(0.0f);
            if (row < uint(src_shape[2])) {
              const device float* p = src + head * src_strides[1] + row * src_strides[2] + x * 4 * src_strides[3];
              value = src_strides[3] == 1
                  ? float4(*((const device packed_float4*)p))
                  : float4(p[0], p[src_strides[3]], p[2 * src_strides[3]], p[3 * src_strides[3]]);
            }
            *((device float4*)(dst + (head * uint(CAPACITY) + row) * uint(HEAD_DIM) + x * 4)) = value;
            """,
        ensureRowContiguous: false)

    private static func firstFill(
        _ newKeys: MLXArray, _ newValues: MLXArray, capacity: Int
    ) -> (MLXArray, MLXArray)? {
        guard StreamOrDevice.default == StreamOrDevice.gpu else { return nil }
        guard let filledKeys = fill(newKeys, capacity: capacity),
            let filledValues = fill(newValues, capacity: capacity)
        else { return nil }
        return (filledKeys, filledValues)
    }

    private static func fill(_ source: MLXArray, capacity: Int) -> MLXArray? {
        guard source.ndim == 4, source.dtype == .float32, source.dim(0) == 1, source.dim(3) % 4 == 0,
            source.dim(3) <= 1024, source.dim(2) <= capacity
        else { return nil }
        let heads = source.dim(1)
        let headDim = source.dim(3)
        return firstFillKernel(
            [source],
            template: [("CAPACITY", capacity), ("HEAD_DIM", headDim)],
            grid: (headDim / 4, capacity, heads), threadGroup: (headDim / 4, 1, 1),
            outputShapes: [[1, heads, capacity, headDim]], outputDTypes: [.float32])[0]
    }

    /// Boot check: 512, 137 and 64 rows of a contiguous key block and a transposed (strided) value view, 4 heads
    /// of 256, capacity rows + 256 as the cache allocates: the first fill against `zeros` + slice update, word for
    /// word; the same fill of a source one word off must differ. Returns the failed and the total case count.
    public static func prepareFirstFill() -> (bad: Int, cases: Int) {
        var bad = 0
        var cases = 0
        guard firstFillOffered else { return (0, 0) }
        for rows in [512, 137, 64] {
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x4B56_0000 + rows)), into: 2)
            let keySource = MLXRandom.normal([1, 4, rows, 256], key: keys[0])
            let valueSource = MLXRandom.normal([1, rows, 4, 256], key: keys[1]).transposed(0, 2, 1, 3)
            let capacity = rows + initialSlack
            func record(_ source: MLXArray) -> MLXArray {
                sliceUpdate(MLXArray.zeros([1, 4, capacity, 256], dtype: .float32), source, 0 ..< rows)
            }
            func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
                a.shape == b.shape
                    && MLX.all(
                        contiguous(a).view(dtype: .uint32) .== contiguous(b).view(dtype: .uint32)
                    ).item(Bool.self)
            }
            let off = MLXArray((0 ..< keySource.size).map { (i: Int) -> Float in
                i == (3 * rows + rows / 2) * 256 + 5 ? 1 : 0
            }).reshaped(keySource.shape)
            guard let leanKeys = fill(firstFillNegativeControl ? keySource + off : keySource, capacity: capacity),
                let leanValues = fill(valueSource, capacity: capacity),
                let control = fill(keySource + off, capacity: capacity)
            else {
                bad += 1
                cases += 1
                continue
            }
            cases += 1
            if !same(leanKeys, record(keySource)) || !same(leanValues, record(valueSource))
                || same(control, record(keySource))
            {
                bad += 1
            }
        }
        firstFillVerified = bad == 0 && cases > 0
        return (bad, cases)
    }

    // MARK: - Private

    /// `dst[..., range, :] = update` as one SliceUpdate on the 4-D update.
    /// The subscript setter first drops the update's leading singleton axis
    /// with a reshape, and a reshape of a strided update (the transposed
    /// verify values) is a full copy before the slice-update copy. Same
    /// primitive, same source elements, same bits.
    private static func sliceUpdate(
        _ dst: MLXArray, _ update: MLXArray, _ range: Range<Int>
    ) -> MLXArray {
        let start: [Int32] = [0, 0, Int32(range.lowerBound), 0]
        let stop: [Int32] = [
            Int32(dst.dim(0)), Int32(dst.dim(1)), Int32(range.upperBound), Int32(dst.dim(3)),
        ]
        let strides: [Int32] = [1, 1, 1, 1]
        var result = mlx_array_new()
        let rc = mlx_slice_update(
            &result, dst.ctx, update.ctx, start, start.count, stop, stop.count,
            strides, strides.count, StreamOrDevice.default.ctx)
        precondition(rc == 0, "CBv2FullSequenceKV: mlx_slice_update failed (\(rc))")
        return MLXArray(result)
    }

    private func ensureCapacity(_ needed: Int, keyTemplate: MLXArray, valueTemplate: MLXArray) {
        if keys == nil {
            capacity = min(maxLength, max(capacity, needed))
            keys = MLXArray.zeros(
                [1, kvHeads, capacity, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
            values = MLXArray.zeros(
                [1, kvHeads, capacity, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
            return
        }
        guard needed > capacity else { return }

        // Grow by doubling, capped at maxLength. The concat copies the old
        // buffer once per doubling — amortized O(1) per appended token.
        let newCapacity = min(maxLength, max(capacity * 2, needed))
        let growth = newCapacity - capacity
        keys = concatenated(
            [keys!, MLXArray.zeros([1, kvHeads, growth, keys!.dim(3)], dtype: keys!.dtype)],
            axis: 2)
        values = concatenated(
            [values!, MLXArray.zeros([1, kvHeads, growth, values!.dim(3)], dtype: values!.dtype)],
            axis: 2)
        capacity = newCapacity
    }
}
