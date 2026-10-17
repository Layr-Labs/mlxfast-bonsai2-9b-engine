// Read-only residency scheduling adapted from the final Bonsai 2 27B engine,
// 0c7c39233afa8aa3e22ac9f36f113b1d2389c3af, PR #1167.
import Foundation
import Cmlx
import MLX

public enum CBv2SideQueue {
    public static let enabled: Bool = !["0", "false", "no", "off"].contains(
        ProcessInfo.processInfo.environment["DARKBLOOM_SEEDGAP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "")

    /// Submits from the dedicated queue rather than the engine thread.
    public static let threaded: Bool = enabled && !["0", "false", "no", "off"].contains(
        ProcessInfo.processInfo.environment["DARKBLOOM_SEEDGAP_THREAD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "")

    private static let lock = NSLock()
    nonisolated(unsafe) private static var stream: mlx_stream?
    nonisolated(unsafe) private static var kernels: [String: mlx_fast_metal_kernel] = [:]
    nonisolated(unsafe) private static var queue: DispatchQueue?
    /// Set at exit: later submissions are dropped (touches only).
    nonisolated(unsafe) private static var closed = false
    nonisolated(unsafe) private static var waitNoted = false
    nonisolated(unsafe) private static var settleNoted = false
    /// Submissions handed to the queue and not yet finished, and whether exit
    /// has begun (no handoff after it); both under `idle`.
    private static let idle = NSCondition()
    nonisolated(unsafe) private static var pending = 0
    nonisolated(unsafe) private static var exiting = false
    /// The touch outputs of committed submissions, kept until their command
    /// buffer has completed (submission queue only). Released to the cache
    /// while the side kernel could still write them, a buffer could be handed
    /// to an engine array of the same size and then written by that kernel:
    /// the two queues are not ordered and MLX's buffers are hazard-untracked.
    nonisolated(unsafe) private static var retained: [mlx_vector_array] = []

    /// Creates the side queue (once), and under `threaded` the dispatch queue
    /// that alone submits to it. Idempotent; call at load.
    public static func prepare() {
        guard enabled else { return }
        _ = sideStream()
        if threaded { _ = submissionQueue() }
    }

    private static func sideStream() -> mlx_stream {
        lock.withLock {
            if let stream { return stream }
            let created = mlx_thread_unsafe_gpu_stream_new()
            stream = created
            return created
        }
    }

    private static func submissionQueue() -> DispatchQueue {
        lock.withLock {
            if let queue { return queue }
            let made = DispatchQueue(
                label: "mlxfast.seedgap.side-queue", qos: .userInitiated,
                autoreleaseFrequency: .workItem)
            queue = made
            // Registered after MLX's global encoder map was built: mlx-swift's
            // default `Stream.gpu` registers there at load, and `prepare()`
            // makes the side stream (also there) before this queue. So `exit`,
            // which runs atexit hooks and static destructors in reverse
            // registration order, runs this before that map's destructor
            // commits the side stream's encoder.
            atexit { CBv2SideQueue.closeAtExit() }
            return made
        }
    }

    private static func kernel(name: String, inputs: Int, source: String) -> mlx_fast_metal_kernel {
        lock.withLock {
            let key = "\(name)/\(inputs)"
            if let cached = kernels[key] { return cached }
            let inputNames = mlx_vector_string_new()
            defer { mlx_vector_string_free(inputNames) }
            for i in 0 ..< inputs { mlx_vector_string_append_value(inputNames, "w\(i)") }
            let outputNames = mlx_vector_string_new()
            defer { mlx_vector_string_free(outputNames) }
            mlx_vector_string_append_value(outputNames, "out")
            let made = mlx_fast_metal_kernel_new(name, inputNames, outputNames, source, "", false, false)
            kernels[key] = made
            return made
        }
    }

    /// One single-thread launch per chunk (each chunk exactly `inputs` arrays),
    /// all committed together on the side queue. The scalar outputs are dropped,
    /// exactly as the default-queue touches drop theirs.
    public static func submitTouches(
        _ chunks: [[MLXArray]], name: String, inputs: Int, source: String
    ) {
        guard !chunks.isEmpty else { return }
        let stream = sideStream()
        let kernel = kernel(name: name, inputs: inputs, source: source)
        releaseCompleted()
        let outputs = mlx_vector_array_new()
        defer { retained.append(outputs) }
        for chunk in chunks {
            precondition(chunk.count == inputs, "CBv2SideQueue: a touch chunk must hold \(inputs) arrays")
            let config = mlx_fast_metal_kernel_config_new()
            defer { mlx_fast_metal_kernel_config_free(config) }
            mlx_fast_metal_kernel_config_set_grid(config, 1, 1, 1)
            mlx_fast_metal_kernel_config_set_thread_group(config, 1, 1, 1)
            let shape: [Int32] = [1]
            mlx_fast_metal_kernel_config_add_output_arg(config, shape, 1, MLX_FLOAT32)
            let ins = mlx_vector_array_new()
            defer { mlx_vector_array_free(ins) }
            for array in chunk { mlx_vector_array_append_value(ins, array.ctx) }
            var result = mlx_vector_array_new()
            defer { mlx_vector_array_free(result) }
            mlx_fast_metal_kernel_apply(&result, kernel, ins, config, stream)
            for i in 0 ..< mlx_vector_array_size(result) {
                var out = mlx_array_new()
                mlx_vector_array_get(&out, result, i)
                mlx_vector_array_append_value(outputs, out)
                mlx_array_free(out)
            }
        }
        mlx_async_eval(outputs)
    }

    /// The `threaded` form of `submitTouches`: `groups` (each group's arrays,
    /// in order) are handed to the submission queue, which forms each group's
    /// chunks with `chunk` and commits them all as one `submitTouches` would.
    ///
    /// On the calling (engine) thread only: one retained C handle per array
    /// (so the queue never reads a Swift `MLXArray`, and every array stays
    /// alive until it has been submitted), and an availability check per array
    /// (`array::is_available`). The check settles an evaluated array to
    /// `available` with its event detached HERE, on the thread that owns the
    /// arrays, so the queue's eval and any later engine eval of the same
    /// arrays only read their status and event. Should an array still be
    /// unscheduled or in flight (not expected for these weights), it is
    /// scheduled here by the engine's usual `asyncEval`, and the group is still
    /// committed by the queue but the engine thread waits for it (the original
    /// stall, correctness first; noted once on stderr).
    public static func submitTouchesOnQueue(
        _ groups: [[MLXArray]], name: String, inputs: Int, source: String,
        chunk: @escaping ([mlx_array]) -> [[mlx_array]]
    ) {
        guard !groups.isEmpty else { return }
        let queue = submissionQueue()
        var handles: [[mlx_array]] = []
        handles.reserveCapacity(groups.count)
        var unsettled: [MLXArray] = []
        for group in groups {
            var row: [mlx_array] = []
            row.reserveCapacity(group.count)
            for array in group {
                var available = false
                _mlx_array_is_available(&available, array.ctx)
                if !available { unsettled.append(array) }
                var handle = mlx_array_new()
                mlx_array_set(&handle, array.ctx)
                row.append(handle)
            }
            handles.append(row)
        }
        idle.lock()
        if exiting {
            idle.unlock()
            for row in handles { for handle in row { mlx_array_free(handle) } }
            return
        }
        pending += 1
        idle.unlock()
        let submission = Submission(
            handles: handles, chunk: chunk, name: name, inputs: inputs, source: source)
        let work: @Sendable () -> Void = { submission.run() }
        if unsettled.isEmpty {
            queue.async(execute: work)
        } else {
            // Whatever is not yet scheduled is scheduled here, on its own
            // stream under the engine's usual `asyncEval` (as the touch's eval
            // would have), and the engine thread waits for the queue, so no
            // thread writes these arrays while the queue reads them.
            asyncEval(unsettled)
            let note = lock.withLock { () -> Bool in
                defer { waitNoted = true }
                return !waitNoted
            }
            if note {
                FileHandle.standardError.write(
                    Data("seedgap side queue: a touched array was still in flight; that group waited\n".utf8))
            }
            queue.sync(execute: work)
        }
    }

    /// One group set handed to the submission queue: its retained handles
    /// (released once submitted, or dropped after exit closed the queue) and
    /// the chunking.
    private final class Submission: @unchecked Sendable {
        let handles: [[mlx_array]]
        let chunk: ([mlx_array]) -> [[mlx_array]]
        let name: String
        let inputs: Int
        let source: String

        init(
            handles: [[mlx_array]], chunk: @escaping ([mlx_array]) -> [[mlx_array]],
            name: String, inputs: Int, source: String
        ) {
            self.handles = handles
            self.chunk = chunk
            self.name = name
            self.inputs = inputs
            self.source = source
        }

        /// On the submission queue only.
        func run() {
            defer {
                for row in handles { for handle in row { mlx_array_free(handle) } }
                idle.lock()
                pending -= 1
                if pending == 0 { idle.broadcast() }
                idle.unlock()
            }
            releaseCompleted()
            guard !lock.withLock({ closed }) else { return }
            commit(handles.flatMap { chunk($0) }, name: name, inputs: inputs, source: source)
        }
    }

    /// Releases the retained outputs whose command buffer has completed (each
    /// output's event signalled). Submission queue only: nothing else
    /// references these outputs. What is still retained at exit is left to it.
    private static func releaseCompleted() {
        retained.removeAll { outputs in
            var done = true
            for i in 0 ..< mlx_vector_array_size(outputs) where done {
                var out = mlx_array_new()
                mlx_vector_array_get(&out, outputs, i)
                var available = false
                _mlx_array_is_available(&available, out)
                mlx_array_free(out)
                done = available
            }
            if done { mlx_vector_array_free(outputs) }
            return done
        }
    }

    /// `submitTouches` over C handles, on the submission queue. The outputs are
    /// retained until their command buffer completes (`retained`).
    private static func commit(_ chunks: [[mlx_array]], name: String, inputs: Int, source: String) {
        guard !chunks.isEmpty else { return }
        let stream = sideStream()
        let kernel = kernel(name: name, inputs: inputs, source: source)
        let outputs = mlx_vector_array_new()
        defer { retained.append(outputs) }
        for chunk in chunks {
            precondition(chunk.count == inputs, "CBv2SideQueue: a touch chunk must hold \(inputs) arrays")
            let config = mlx_fast_metal_kernel_config_new()
            defer { mlx_fast_metal_kernel_config_free(config) }
            mlx_fast_metal_kernel_config_set_grid(config, 1, 1, 1)
            mlx_fast_metal_kernel_config_set_thread_group(config, 1, 1, 1)
            let shape: [Int32] = [1]
            mlx_fast_metal_kernel_config_add_output_arg(config, shape, 1, MLX_FLOAT32)
            let ins = mlx_vector_array_new()
            defer { mlx_vector_array_free(ins) }
            for handle in chunk { mlx_vector_array_append_value(ins, handle) }
            var result = mlx_vector_array_new()
            defer { mlx_vector_array_free(result) }
            mlx_fast_metal_kernel_apply(&result, kernel, ins, config, stream)
            for i in 0 ..< mlx_vector_array_size(result) {
                var out = mlx_array_new()
                mlx_vector_array_get(&out, result, i)
                mlx_vector_array_append_value(outputs, out)
                mlx_array_free(out)
            }
        }
        mlx_async_eval(outputs)
    }

    /// The raw dtype of a handle (the queue's chunking key).
    public static func dtypeKey(_ handle: mlx_array) -> UInt32 {
        mlx_array_dtype(handle).rawValue
    }

    /// Waits until every submission handed to the queue so far has finished
    /// (its touches committed), bounded by `timeout` seconds. Called by the
    /// engine's finalize after a step's sampled tokens are read back and before
    /// they are emitted: by then the prompt's GPU work is done and the queue,
    /// which committed alongside it, is normally idle, so this is one lock. A
    /// no-op unless `threaded`.
    public static func settle(timeout: Double = 5) {
        guard threaded else { return }
        idle.lock()
        defer { idle.unlock() }
        guard pending > 0 else { return }
        let deadline = Date().addingTimeInterval(timeout)
        while pending > 0 {
            if !idle.wait(until: deadline) {
                if !settleNoted {
                    settleNoted = true
                    FileHandle.standardError.write(
                        Data("seedgap side queue: settle timed out after \(timeout) s\n".utf8))
                }
                return
            }
        }
    }

    /// The `atexit` hook: refuse any later handoff (checked under the same
    /// `idle` lock that counts `pending`, so an engine thread still mid-prompt
    /// on a fault exit cannot slip one in after the wait), let queued blocks
    /// skip their touches, then wait (bounded) for the queue, so `exit`'s
    /// static destructors (the global command-encoder map with the side
    /// stream's encoder) never meet a block mid-encode.
    static func closeAtExit() {
        idle.lock()
        exiting = true
        idle.unlock()
        lock.withLock { closed = true }
        settle(timeout: 5)
    }
}


private enum CBv2ResidencyTouch {
    /// Inputs per kernel; with the output well inside Metal's 31 buffer slots.
    static let inputs = 24

    static let name = "dflash2_residency_touch"
    static let source: String = {
        let reads = (0 ..< inputs).map { "acc += static_cast<float>(w\($0)[0]);" }.joined(separator: "\n")
        return "float acc = 0.0f;\n" + reads + "\nout[0] = acc;\n"
    }()

    private static let kernel: MLXFast.MLXFastKernel = {
        let names = (0 ..< inputs).map { "w\($0)" }
        return MLXFast.metalKernel(
            name: name, inputNames: names, outputNames: ["out"],
            source: source,
            ensureRowContiguous: false)
    }()

    private static func launch(_ chunk: [MLXArray]) -> MLXArray {
        kernel(
            chunk, grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[1]], outputDTypes: [.float32])[0]
    }

    /// The launches' operands for `arrays`: per dtype (first-seen order),
    /// `inputs` at a time, the last chunk padded with its own last array.
    static func chunks(_ arrays: [MLXArray]) -> [[MLXArray]] {
        chunks(arrays, dtype: { $0.dtype })
    }

    /// `chunks` over any element with a dtype key (the side queue's retained
    /// C handles key on the raw dtype); the same order, chunks and padding.
    static func chunks<Element, Key: Hashable>(
        _ arrays: [Element], dtype: (Element) -> Key
    ) -> [[Element]] {
        var order: [Key] = []
        var byType: [Key: [Element]] = [:]
        for array in arrays {
            let key = dtype(array)
            if byType[key] == nil { order.append(key) }
            byType[key, default: []].append(array)
        }
        return order.flatMap { key -> [[Element]] in
            let same = byType[key]!
            return stride(from: 0, to: same.count, by: inputs).map { start in
                var chunk = Array(same[start ..< min(start + inputs, same.count)])
                chunk += Array(repeating: chunk[chunk.count - 1], count: inputs - chunk.count)
                return chunk
            }
        }
    }

    /// The touches of `arrays`: per dtype (first-seen order), `inputs` at a time.
    static func touch(_ arrays: [MLXArray]) -> [MLXArray] {
        chunks(arrays).map { launch($0) }
    }

    /// Submits the touches of `groups` (each group's chunks, in order) on the
    /// side queue (`CBv2SideQueue`, `MLXFAST_SEEDGAP`): the same launches the
    /// default queue would take, committed on their own queue.
    /// Under `MLXFAST_SEEDGAP_THREAD` (default on) the engine thread only hands
    /// `groups` to the side queue's own dispatch queue, which forms the same
    /// chunks and commits them (`CBv2SideQueue.submitTouchesOnQueue`).
    static func submitOnSideQueue(_ groups: [[MLXArray]]) {
        if CBv2SideQueue.threaded {
            CBv2SideQueue.submitTouchesOnQueue(
                groups, name: name, inputs: inputs, source: source,
                chunk: { chunks($0, dtype: CBv2SideQueue.dtypeKey) })
            return
        }
        CBv2SideQueue.submitTouches(
            groups.flatMap { chunks($0) }, name: name, inputs: inputs, source: source)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var warmed = false

    /// Compiles the touch for every dtype a weight can have, once.
    static func prewarm() {
        guard lock.withLock({ () -> Bool in
            defer { warmed = true }
            return !warmed
        }) else { return }
        let types: [DType] = [.bfloat16, .float16, .float32, .uint32, .uint16, .uint8, .int8, .int32]
        eval(types.map { launch(Array(repeating: MLXArray.zeros([1], dtype: $0), count: inputs)) })
        // The side queue is made here, at load, with its own first touches.
        if CBv2SideQueue.enabled {
            CBv2SideQueue.prepare()
            let zeros = types.map { MLXArray.zeros([1], dtype: $0) }
            eval(zeros)
            if CBv2SideQueue.threaded {
                // The same eight launches (one dtype each, padded with itself),
                // from the side queue's own thread like every later touch.
                submitOnSideQueue([zeros])
            } else {
                CBv2SideQueue.submitTouches(
                    zeros.map { Array(repeating: $0, count: inputs) }, name: name, inputs: inputs,
                    source: source)
            }
        }
    }
}


/// Read-only weight residency preparation during a DFlash prompt forward.
public enum CBv2ResidencyPrefetch {
    public static let enabled: Bool = !["0", "false", "no", "off"].contains(
        ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH_RESIDENCY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "")
    private static let due = [4, 8, 16, 24]
    private static let lock = NSLock()
    nonisolated(unsafe) private static var groups: [[MLXArray]] = []
    nonisolated(unsafe) private static var next = 0

    public static var isArmed: Bool { lock.withLock { next < groups.count } }

    public static func prepare() {
        if enabled { CBv2ResidencyTouch.prewarm() }
    }

    public static func arm(_ arrays: [MLXArray]) {
        guard enabled else { return }
        var seen = Set<ObjectIdentifier>()
        let arrays = arrays.filter { $0.size > 0 && seen.insert(ObjectIdentifier($0)).inserted }
        let total = arrays.reduce(0) { $0 + $1.nbytes }
        let share = max(1, total / due.count)
        var built: [[MLXArray]] = [], group: [MLXArray] = []
        var bytes = 0
        for array in arrays {
            group.append(array); bytes += array.nbytes
            if bytes >= share * (built.count + 1), built.count < due.count - 1 {
                built.append(group); group = []
            }
        }
        if !group.isEmpty { built.append(group) }
        lock.withLock { groups = built; next = 0 }
    }

    public static func disarm() { lock.withLock { groups = []; next = 0 } }

    public static func submitDue(completedLayers: Int) {
        let selected = lock.withLock { () -> [[MLXArray]] in
            var selected: [[MLXArray]] = []
            while next < groups.count, completedLayers >= due[next] {
                selected.append(groups[next]); next += 1
            }
            if next == groups.count { groups = []; next = 0 }
            return selected
        }
        guard !selected.isEmpty else { return }
        if CBv2SideQueue.enabled { CBv2ResidencyTouch.submitOnSideQueue(selected) }
        else { asyncEval(selected.flatMap { CBv2ResidencyTouch.touch($0) }) }
    }
}
