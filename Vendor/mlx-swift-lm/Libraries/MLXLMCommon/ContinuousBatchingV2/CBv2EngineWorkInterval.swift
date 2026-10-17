import Darwin
import Foundation
import os

/// Keeps each round's host work on a performance core.
///
/// The engine thread sleeps on the GPU for most of every round, so its thread
/// group looks light and the scheduler may place it on the efficiency cluster,
/// where the round's host work (finalize, the next round's graph build and
/// encoding) runs at a fraction of the clock and the GPU waits for it. The
/// queue's QoS (`userInteractive`) does not change that placement, and a
/// time-constraint policy is ignored on a dispatch worker thread.
///
/// A work interval with a deadline does. From the readback to the end of the
/// step, the host work is one interval of the engine's own workgroup, due a
/// few milliseconds after the readback, and the scheduler runs it on a
/// performance core at full clock. The workgroup is an audio work interval
/// (`AudioWorkIntervalCreate`, public since macOS 11). Swift does not export
/// it, so it is looked up at run time, and it serves only as a scheduling
/// hint: the thread's policy stays untouched, nothing spins, and no value the
/// engine computes depends on it. Every failure leaves the plain dispatch
/// thread. `DARKBLOOM_BONSAI_WORK_INTERVAL=0` turns the hint off.
///
/// Engine-queue only: `stepBegan`, `hostWorkBegan` and `stepEnded` run on the
/// serial engine queue, one step at a time.
final class CBv2EngineWorkInterval: @unchecked Sendable {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_WORK_INTERVAL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The interval's deadline after the readback (4 ms), in mach absolute time.
    private static let deadlineTicks: UInt64 = {
        var info = mach_timebase_info_data_t()
        guard mach_timebase_info(&info) == KERN_SUCCESS, info.numer > 0 else { return 96_000 }
        return 4_000_000 * UInt64(info.denom) / UInt64(info.numer)
    }()

    private typealias Create =
        @convention(c) (UnsafePointer<CChar>, UInt32, UnsafeMutableRawPointer?) ->
        UnsafeMutableRawPointer?

    private static let create: Create? = {
        guard enabled else { return nil }
        _ = dlopen("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox", RTLD_NOW)
        guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "AudioWorkIntervalCreate")
        else { return nil }
        return unsafeBitCast(symbol, to: Create.self)
    }()

    /// `OS_CLOCK_MACH_ABSOLUTE_TIME` from <os/clock.h>.
    private static let machAbsoluteClock: UInt32 = 32

    private let interval: WorkGroup?
    private var token: WorkGroup.JoinToken?
    private var joinedThread: pthread_t?
    private var started = false

    init() {
        interval = Self.create
            .flatMap { $0("com.eigen.cbv2.engine", Self.machAbsoluteClock, nil) }
            .map { Unmanaged<WorkGroup>.fromOpaque($0).takeRetainedValue() }
    }

    /// Joins the calling thread for one engine step. Returns whether it joined;
    /// only a step that joined calls `stepEnded`.
    func stepBegan() -> Bool {
        guard let interval, token == nil else { return false }
        token = interval.join()
        joinedThread = pthread_self()
        return true
    }

    /// The readback returned (or the step has none): the host work starts now.
    /// A no-op once this step's interval has started.
    func hostWorkBegan() {
        guard let interval, token != nil, !started, let joinedThread,
            pthread_equal(joinedThread, pthread_self()) != 0
        else { return }
        let now = mach_absolute_time()
        interval.start(at: now, deadline: now &+ Self.deadlineTicks)
        started = true
    }

    func stepEnded() {
        guard let interval, let token else { return }
        if started {
            interval.finish()
            started = false
        }
        interval.leave(token: token)
        self.token = nil
        joinedThread = nil
    }
}
