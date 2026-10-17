import Darwin
import Foundation
import os

/// Keeps each round's host work on a performance core.
///
/// Measured on an M4 Max with per-thread, per-performance-level counters: the
/// round's host work is the same in every round (55 M instructions for the
/// verify build, the same allocator traffic), yet for whole windows the
/// scheduler runs the engine thread on the efficiency cluster (1.0-2.5 GHz,
/// IPC 2.7) instead of a performance core (3.7-4.5 GHz, IPC 4.1). The verify
/// build then takes 11.4 ms instead of 3.8 ms, and the committed recurrent
/// state takes 5.3 ms instead of 3.1 ms to encode. The thread sleeps on the GPU
/// for most of every round, so its thread group looks light and the scheduler
/// recommends the efficiency cluster for it. The queue's QoS (userInitiated or
/// userInteractive) does not change that, and a time-constraint policy is
/// silently ignored on a dispatch worker thread.
///
/// A work interval with a deadline changes it. From the readback to the end of
/// the step, the host work is one interval of the engine's own workgroup, due a
/// few milliseconds after the readback. The scheduler then runs it on a
/// performance core at full clock: the verify build took 3.1 ms in every
/// round, and GPU idle per round fell from 2.0 ms to 0.6 ms with unchanged GPU
/// busy time. The workgroup is an audio work interval (`AudioWorkIntervalCreate`,
/// public since macOS 11). Swift does not export it, so it is looked up at run
/// time, and it serves only as the scheduling hint. The thread's policy stays
/// untouched and nothing spins. Every failure leaves the plain dispatch thread.
/// `MLXFAST_ENGINE_WORK_INTERVAL=0` turns the hint off.
///
/// A prompt forward's host work outlives one deadline: building and encoding
/// a 512-row forward takes longer than 4 ms, and once the deadline passes the
/// thread drifts back to the efficiency cluster (the seed step's engine CPU
/// ran 94% there, the step's interval notwithstanding). So a prompt-width
/// forward inside an engine step renews the step's interval at its start and
/// at each early submission of its prompt plan (`promptForwardBegan`,
/// `promptSubmitted`): the running interval finishes and the next one starts,
/// due one deadline later (M4 Max: the seed step's engine CPU 78% on the
/// efficiency cluster, 59 ms instead of 82 ms). A forward on a thread that
/// joined no interval is left alone: the timed `prefill` verb's worker thread
/// runs its forward on the performance cluster by itself (13% efficiency),
/// and joining it to an interval of its own moved it there (76%). Rounds and
/// verify windows are untouched. `MLXFAST_PROMPT_WORK_INTERVAL=0` turns only
/// this prompt renewal off; a positive integer sets its deadline in
/// milliseconds (default 4).
///
/// `stepBegan`, `hostWorkBegan` and `stepEnded` run on the serial engine
/// queue, one step at a time; the prompt entry points act only on an interval
/// the calling thread itself joined.
public final class CBv2EngineWorkInterval: @unchecked Sendable {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ENGINE_WORK_INTERVAL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `milliseconds` in mach absolute time.
    private static func ticks(milliseconds: UInt64) -> UInt64 {
        var info = mach_timebase_info_data_t()
        guard mach_timebase_info(&info) == KERN_SUCCESS, info.numer > 0 else {
            return 24_000 * milliseconds
        }
        return milliseconds * 1_000_000 * UInt64(info.denom) / UInt64(info.numer)
    }

    /// The interval's deadline after the readback, in mach absolute time.
    private static let deadlineTicks: UInt64 = ticks(milliseconds: 4)

    /// The deadline of a prompt forward's renewed intervals, or nil when the
    /// prompt coverage is off (`MLXFAST_PROMPT_WORK_INTERVAL`).
    private static let promptDeadlineTicks: UInt64? = {
        guard enabled else { return nil }
        // Off by default on this lineage (the box read it slower); `on` or a deadline in ms turns it on.
        let value = ProcessInfo.processInfo.environment["MLXFAST_PROMPT_WORK_INTERVAL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let value, !value.isEmpty, !["0", "false", "no", "off"].contains(value) else { return nil }
        let milliseconds = UInt64(value).flatMap { $0 > 0 ? $0 : nil } ?? 4
        return ticks(milliseconds: milliseconds)
    }()

    /// The interval the calling thread has joined (unretained; set by
    /// `stepBegan`, cleared by `stepEnded`).
    private static let currentKey: pthread_key_t? = {
        var key = pthread_key_t()
        return pthread_key_create(&key, nil) == 0 ? key : nil
    }()

    private typealias Create =
        @convention(c) (UnsafePointer<CChar>, UInt32, UnsafeMutableRawPointer?) ->
        UnsafeMutableRawPointer?
    /// The errno-returning C entry points (os/workgroup.h). Swift's own
    /// `WorkGroup` wrappers trap on a nonzero return, so a recoverable join
    /// or start failure would abort the process; these return the error
    /// instead and every failure path falls back to the plain thread.
    private typealias Join =
        @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Int32
    private typealias Leave =
        @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
    private typealias IntervalStart =
        @convention(c) (UnsafeMutableRawPointer, UInt64, UInt64, UnsafeMutableRawPointer?) -> Int32
    private typealias IntervalFinish =
        @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer?) -> Int32

    private static func symbol(_ name: String) -> UnsafeMutableRawPointer? {
        dlsym(UnsafeMutableRawPointer(bitPattern: -2), name)
    }

    private static let create: Create? = {
        guard enabled else { return nil }
        _ = dlopen("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox", RTLD_NOW)
        guard let symbol = symbol("AudioWorkIntervalCreate") else { return nil }
        return unsafeBitCast(symbol, to: Create.self)
    }()

    private static let join: Join? =
        symbol("os_workgroup_join").map { unsafeBitCast($0, to: Join.self) }
    private static let leave: Leave? =
        symbol("os_workgroup_leave").map { unsafeBitCast($0, to: Leave.self) }
    private static let intervalStart: IntervalStart? =
        symbol("os_workgroup_interval_start").map { unsafeBitCast($0, to: IntervalStart.self) }
    private static let intervalFinish: IntervalFinish? =
        symbol("os_workgroup_interval_finish").map { unsafeBitCast($0, to: IntervalFinish.self) }

    /// `OS_CLOCK_MACH_ABSOLUTE_TIME` from <os/clock.h>.
    private static let machAbsoluteClock: UInt32 = 32

    /// Retains the workgroup; the C calls take its opaque pointer.
    private let interval: WorkGroup?
    private let pointer: UnsafeMutableRawPointer?
    /// `os_workgroup_join_token_s` is 40 opaque bytes, invisible to Swift;
    /// 64 zeroed bytes hold it with room to spare.
    private let tokenStorage = UnsafeMutableRawPointer.allocate(byteCount: 64, alignment: 16)
    private var joined = false
    private var joinedThread: pthread_t?
    private var started = false

    init() {
        interval =
            Self.join != nil && Self.leave != nil && Self.intervalStart != nil
            && Self.intervalFinish != nil
            ? Self.create
                .flatMap { $0("com.eigen.cbv2.engine", Self.machAbsoluteClock, nil) }
                .map { Unmanaged<WorkGroup>.fromOpaque($0).takeRetainedValue() }
            : nil
        pointer = interval.map { Unmanaged.passUnretained($0).toOpaque() }
        tokenStorage.initializeMemory(as: UInt8.self, repeating: 0, count: 64)
    }

    deinit {
        tokenStorage.deallocate()
    }

    /// Joins the calling thread for one engine step. Returns whether it joined;
    /// only a step that joined calls `stepEnded`.
    func stepBegan() -> Bool {
        guard let pointer, let join = Self.join, !joined else { return false }
        guard join(pointer, tokenStorage) == 0 else { return false }
        joined = true
        joinedThread = pthread_self()
        if let key = Self.currentKey {
            pthread_setspecific(key, Unmanaged.passUnretained(self).toOpaque())
        }
        return true
    }

    /// The readback returned: the round's host work starts now.
    func hostWorkBegan() {
        start(deadlineTicks: Self.deadlineTicks)
    }

    private var onJoinedThread: Bool {
        guard joined, let joinedThread else { return false }
        return pthread_equal(joinedThread, pthread_self()) != 0
    }

    private func start(deadlineTicks: UInt64) {
        guard let pointer, let start = Self.intervalStart, !started, onJoinedThread else { return }
        let now = mach_absolute_time()
        if start(pointer, now, now &+ deadlineTicks, nil) == 0 {
            started = true
        }
    }

    /// Finishes the running interval and starts the next, due `deadlineTicks`
    /// from now; starts one if none runs. A finish the system refuses keeps
    /// the running interval (no new start over an unfinished one).
    private func renew(deadlineTicks: UInt64) {
        guard let pointer, let finish = Self.intervalFinish, onJoinedThread else { return }
        if started {
            guard finish(pointer, nil) == 0 else { return }
            started = false
        }
        start(deadlineTicks: deadlineTicks)
    }

    /// The interval the calling thread has joined, if any.
    private static func current() -> CBv2EngineWorkInterval? {
        guard let key = currentKey, let pointer = pthread_getspecific(key) else { return nil }
        return Unmanaged<CBv2EngineWorkInterval>.fromOpaque(pointer).takeUnretainedValue()
    }

    /// A prompt-width forward starts building on the calling thread: renew
    /// the interval the thread joined for its engine step (the step finishes
    /// it). A thread that joined none is left alone.
    public static func promptForwardBegan() {
        guard let promptDeadlineTicks, let current = current() else { return }
        current.renew(deadlineTicks: promptDeadlineTicks)
    }

    /// A prompt forward submits the layers built so far: renew the calling
    /// thread's running interval, so the encoding and the layers built next
    /// run within a deadline.
    public static func promptSubmitted() {
        guard let promptDeadlineTicks, let current = current(), current.started else { return }
        current.renew(deadlineTicks: promptDeadlineTicks)
    }

    func stepEnded() {
        guard let pointer, let leave = Self.leave, joined, onJoinedThread else { return }
        if started {
            // Best effort: the step is over either way, and leave is the only
            // cleanup that must happen (its token came from a successful join
            // on this thread).
            _ = Self.intervalFinish?(pointer, nil)
            started = false
        }
        leave(pointer, tokenStorage)
        joined = false
        joinedThread = nil
        if let key = Self.currentKey,
            pthread_getspecific(key) == Unmanaged.passUnretained(self).toOpaque()
        {
            pthread_setspecific(key, nil)
        }
    }
}
