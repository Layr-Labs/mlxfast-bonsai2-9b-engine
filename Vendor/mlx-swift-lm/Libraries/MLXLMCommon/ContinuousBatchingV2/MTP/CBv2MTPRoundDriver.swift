// CBv2MTPRoundDriver.swift
//
// ContinuousBatchingV2 — MTP (speculative decoding) round state.
//
// The driver is a passive state holder owned by `EngineLoopV2`: the drafter
// binding, per-row drafter carries, plan-scoped speculation marks, and the
// cumulative metrics. All round LOGIC (eligibility, execution, finalize)
// lives in `EngineLoopV2+MTP.swift` — it needs the loop's row state
// (`kvStates`, `multimodalByID`) and step-building helpers.
//
// Threading: everything except `metricsSnapshot()` is engine-thread
// confined. Metrics are mutated on the engine thread and snapshotted under
// a lock so the provider can poll them from any thread
// (`EngineV2.mtpMetricsSnapshot()`).

import Foundation
import MLX

// MARK: - Drafter carry

/// One row's drafter carry: the newest confirmed-but-unfed token (the round
/// seed) plus the target's pre-norm hidden at the position BEFORE it — the
/// pair the Gemma-4 drafter chains from. `tokensCount`/`kvOffset` fingerprint
/// the row state the carry was captured against; any mismatch at plan time
/// (a plain step appended a token, preemption reset progress, an id was
/// reused) invalidates the carry, costing exactly one seed step (prefer
/// simple and correct over carry salvage).
struct CBv2MTPCarry {
    let token: Int
    /// [1, 1, H], lazy slice of an already-evaluated step output.
    let hidden: MLXArray
    /// [K] int32 target top-K token ids at the carry position, lazy slice of
    /// an already-evaluated verify shortlist. Non-nil only when the drafter
    /// opted in AND the captured top-K probability mass cleared the coverage
    /// threshold at finalize; nil ⇒ the next draft scores the full head.
    let shortlist: MLXArray?
    /// Target-logit top-two margin selected at the carry position. It is
    /// finalized at the round's existing host readback and used only by the
    /// next adaptive depth decision.
    let previousTopTwoMargin: Double?
    /// True only for a verify-produced carry whose terminal transition was
    /// intentionally left outside committed assistant history.
    let needsHistoryTransition: Bool

    /// `rec.tokens.count` at capture — the carry token must still be
    /// `tokens.last` with the same count.
    let tokensCount: Int
    /// `rec.numComputedTokens` at capture (== the row's KV absoluteOffset,
    /// the round anchor).
    let kvOffset: Int
}

// MARK: - Declared depth

/// A fixed draft depth above the drafter's own draft-chain cap. The round
/// driver refuses it and never clamps it: a clamped run measures a shallower
/// depth than the one the caller declared and seals.
public struct CBv2MTPDepthAboveDrafterCap: Error, Equatable, CustomStringConvertible {
    public let declared: Int
    public let drafterCap: Int

    public var description: String {
        "the declared MTP depth \(declared) is above the drafter's draft-chain cap of "
            + "\(drafterCap); the round driver refuses a fixed depth above the cap "
            + "and does not run a shallower one"
    }
}

extension CBv2MTPConfig {
    /// Throws `CBv2MTPDepthAboveDrafterCap` when `drafter` cannot serve this
    /// configuration's fixed depth. An engine built from a declared depth
    /// calls this before it builds, so the refusal reaches the caller as an
    /// error before any timed work. The round driver stops on the same test.
    public func validateFixedDepth(for drafter: any CBv2MTPDrafter) throws {
        if let refusal = CBv2MTPRoundDriver.depthRefusal(config: self, drafter: drafter) {
            throw refusal
        }
    }
}

// MARK: - In-flight round payload

/// MTP payload riding a `CBv2InFlightStep`. Its presence marks the step as
/// an MTP round step: the chained-decode fast path must never build on top
/// of it (the chained finalize/deferred-release machinery assumes exactly
/// one sample per row), and `finalize` runs the accept-walk for the verify
/// rows at the step's one host-sync boundary.
final class CBv2MTPRoundInFlight {

    struct VerifyRow {
        let id: CBv2RequestID
        /// Storage-owning sequence states at launch, for finalize-time
        /// `rollback` + `commitSpeculativeWrite` (identical objects to
        /// `kvStates[id]` unless the row finished mid-flight, in which case
        /// the whole state is released via the deferred-release fence and
        /// this list is not touched).
        let storageRows: [CBv2SequenceKV]
        /// Request-owned autoregressive assistant state moved out of the
        /// driver for the duration of this in-flight round. This fences its
        /// release behind target/assistant evaluation on cancellation.
        let assistantState: (any CBv2MTPRequestState)?
        /// Non-nil when this row's drafts came from prompt lookup: the
        /// consumed carry. No draft step fed it to the assistant, so finalize
        /// hands it over through `commitLookupRound`.
        let lookupCarry: CBv2MTPCarry?
    }

    struct CommittedObservationRow {
        let id: CBv2RequestID
        /// Request-owned state detached until the target graph is fenced.
        let assistantState: any CBv2MTPRequestState
    }

    struct Verify {
        /// Draft tokens per row this round (uniform across the batch).
        let k: Int
        /// Verify-batch rows, in batch row order.
        let rows: [VerifyRow]
        /// Lazy flattened int32 packet: all [B, k] draft ids followed by all
        /// [B, 1+k] target argmaxes, then — iff `shortlistIDs` is non-nil —
        /// all [B, 1+k] shortlist probability masses in parts-per-million.
        /// One `asArray` at finalize reads everything, preserving the single
        /// host-sync boundary.
        let acceptancePacket: MLXArray
        /// Lazy [B,k] draft ids retained for exact accepted-prefix slicing
        /// into stateful assistant finalization.
        let draftIDs: MLXArray
        /// Lazy [B, 1+k, H] pre-norm hidden — the next carry is gathered
        /// from it at the finalize sync (index = accepted position).
        let lastHidden: MLXArray
        /// Lazy [B, 1+k, K] int32 target top-K ids per verify position, for
        /// shortlisted drafting next round. nil when the drafter did not opt
        /// in or verification ran on the serial oracle.
        let shortlistIDs: MLXArray?
        /// Serial recurrent target generations in target-column order. Empty
        /// for attention-only targets.
        let recurrentEvaluations: [CBv2RequestID: [CBv2RecurrentStateEvaluation]]
        /// Lazy exact Qwen policy top-two values [B, 1+k, 2], float32.
        /// IDs feed greedy scoring on device; values are read only after the
        /// existing acceptance-packet fence.
        let policyTopTwoValues: MLXArray?
        /// Lazy `[B, 1+k, taps * H]` fused tapped context the target produced
        /// for this verify window. A BLOCK drafter's next block reads columns
        /// `0..<confirmed` of it; nil for every chain drafter.
        let blockContext: MLXArray?
        var diagnostics: [CBv2LogitDiagnosticPacket] = []
        var includesAssistantPrefill = false
    }

    /// nil when this round only seeded (no row had a valid carry yet).
    let verify: Verify?
    /// Seed rows: (request, row index into the step's decode batch). Their
    /// bonus token rides the step's normal `sampledTokens`; the carry hidden
    /// is sliced from `seedHidden` at finalize.
    let seedRows: [(id: CBv2RequestID, decodeIndex: Int)]
    /// Lazy [B_decode, 1, H] pre-norm hidden of the step's decode batch
    /// (non-nil iff `seedRows` is non-empty).
    let seedHidden: MLXArray?
    /// Lazy float32 [B_decode, 1, 2] policy values for adaptive seed and
    /// temporary depth-zero carries.
    let seedPolicyTopTwoValues: MLXArray?
    /// Plain prompt/decode target observations whose request-owned assistant
    /// states stay detached until this step's target graph is fenced.
    let committedObservationRows: [CommittedObservationRow]
    /// Prompt rows whose first sampled token becomes a round carry at
    /// finalize (see `mtpPrefillCarryEnabled`).
    let prefillCarries: [(id: CBv2RequestID, hidden: MLXArray)]

    /// Finalization outcomes used by host-only controller attribution. These
    /// are populated at the existing host-sync boundary.
    var finalizedSeedIDs: Set<CBv2RequestID> = []
    var finalizedVerifyIDs: Set<CBv2RequestID> = []
    var claimedSeedCostNanos: UInt64 = 0
    /// Outputs actually kept after common-width, EOS, stop and budget handling.
    var committedVerifyTokenCount = 0
    /// Cancellation-owned assistant states released only after the target KV
    /// and recurrent deferred-release fence has retired.
    var deferredAssistantReleases: [any CBv2MTPRequestState] = []

    init(
        verify: Verify?,
        seedRows: [(id: CBv2RequestID, decodeIndex: Int)],
        seedHidden: MLXArray?,
        seedPolicyTopTwoValues: MLXArray?,
        committedObservationRows: [CommittedObservationRow],
        prefillCarries: [(id: CBv2RequestID, hidden: MLXArray)] = []
    ) {
        self.prefillCarries = prefillCarries
        self.verify = verify
        self.seedRows = seedRows
        self.seedHidden = seedHidden
        self.seedPolicyTopTwoValues = seedPolicyTopTwoValues
        self.committedObservationRows = committedObservationRows
    }
}

/// A seed step belongs to the row cohort it prepared, not to the depth that
/// happened to be selected on that plan. A later verification may run at a
/// different depth; exact-cohort matching prevents cancelled or invalidated
/// seed work from leaking into an unrelated probe.
struct CBv2MTPSeedCostLedger {
    private struct Pending {
        var nanos: UInt64
        let requestIDs: Set<CBv2RequestID>
    }

    private var byBucket: [Int: Pending] = [:]

    mutating func record(
        decodeRowBucket: Int, requestIDs: Set<CBv2RequestID>, nanos: UInt64
    ) {
        guard decodeRowBucket > 0, !requestIDs.isEmpty, nanos > 0 else { return }
        if var pending = byBucket[decodeRowBucket], pending.requestIDs == requestIDs {
            pending.nanos &+= nanos
            byBucket[decodeRowBucket] = pending
        } else {
            byBucket[decodeRowBucket] = Pending(nanos: nanos, requestIDs: requestIDs)
        }
    }

    /// Claim seed cost before verify-row completion can retire request ids.
    /// An overlapping but non-identical cohort is discarded rather than
    /// partially charging work whose rectangular batch shape changed.
    mutating func take(
        decodeRowBucket: Int, requestIDs: Set<CBv2RequestID>
    ) -> UInt64 {
        guard let pending = byBucket[decodeRowBucket], !requestIDs.isEmpty else { return 0 }
        if pending.requestIDs == requestIDs {
            byBucket.removeValue(forKey: decodeRowBucket)
            return pending.nanos
        }
        if !pending.requestIDs.isDisjoint(with: requestIDs) {
            byBucket.removeValue(forKey: decodeRowBucket)
        }
        return 0
    }

    mutating func invalidate(_ id: CBv2RequestID) {
        let matching = byBucket.compactMap { bucket, pending in
            pending.requestIDs.contains(id) ? bucket : nil
        }
        for bucket in matching {
            byBucket.removeValue(forKey: bucket)
        }
    }

    mutating func removeAll() {
        byBucket.removeAll(keepingCapacity: false)
    }

    var count: Int { byBucket.count }
}

// MARK: - Driver

/// Engine-side MTP state: drafter binding, carries, plan marks, metrics.
final class CBv2MTPRoundDriver {

    let config: CBv2MTPConfig
    let drafter: any CBv2MTPDrafter
    /// The engine's model, downcast once at build (verify forwards go
    /// through `forwardWithHidden`).
    let model: any CBv2MTPSteppableModel
    let captureLayers: CBv2MTPCaptureLayers?
    /// True when acceptance may use target-prefix pre-sampling (drafter
    /// opt-in). The planner additionally requires the installed sampler's
    /// `supportsMTPTargetPrefix` before lifting the greedy gate.
    let targetPrefixAcceptance: Bool
    /// Prompt-lookup n-gram length, or nil when lookup drafting is off.
    let lookupNgram: Int?
    /// Depth of a round whose lookup misses, or nil to keep the declared one.
    let lookupMissDepth: Int?
    /// Deeper drafted depth and the carried-margin threshold that selects it,
    /// or nil to keep one drafted depth.
    let lookupMissDeep: (depth: Int, minimumMargin: Double)?
    private var missDepthSchedule: [Int]
    /// The current plan took the schedule's first entry, and a round that
    /// plan launched consumed its carry. An entry is dropped only after a
    /// drafted round launched with it: the seed step and a re-plan of the
    /// same step plan without launching one.
    private var missDepthScheduleArmed = false
    private var missDepthScheduleLaunched = false
    private let depthController: CBv2MTPDepthController
    private var committedDecodeClock = CBv2MTPCommittedDecodeClock()
    private var committedGoodputClock = CBv2MTPCommittedGoodputClock()
    private var goodputPlanRows: [CBv2RequestID] = []
    /// Numeric request IDs can be reused. Stamp launch measurements with the
    /// current lifetime so late chained finalization cannot train a new one.
    private(set) var workloadGeneration: UInt64 = 0
    private var workloadInvalidated = false

    // Engine-thread confined.
    private var carries: [CBv2RequestID: CBv2MTPCarry] = [:]
    private var assistantStates: [CBv2RequestID: any CBv2MTPRequestState] = [:]
    /// Qwen acceptance is request-owned; raw nonchained wall cost is shared
    /// only within a compatible decode-row bucket.
    private var requestAcceptance: [CBv2RequestID: CBv2MTPRequestAcceptanceState] = [:]
    private var rawCostEstimators: [Int: CBv2MTPRawCostEstimator] = [:]
    /// Host token history is append-only, so an index outlives carries and
    /// preemption; it goes only with the request (ids are reusable).
    private var lookupIndexes: [CBv2RequestID: CBv2MTPPromptLookupIndex] = [:]

    /// Plan-scoped: rows the planner offered a 1+k round this plan.
    private(set) var roundMarks: [CBv2RequestID: Int] = [:]
    /// Plan-scoped: eligible rows without a valid carry — their decode step
    /// this plan is a SEED step (eager forwardWithHidden, hidden captured).
    private(set) var seedMarks: Set<CBv2RequestID> = []
    /// Step-global seed fallback without destroying persistent request
    /// history. Used when one rectangular row lacks a carry.
    private(set) var forceSeedPlan = false

    /// One selection for the whole scheduler plan. Every speculating row in
    /// that plan uses this depth, so target verification stays rectangular.
    private(set) var controllerDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)
    private(set) var planDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)
    private(set) var controllerMeasurementEligible = false

    private let metricsLock = NSLock()
    private var metrics = CBv2MTPMetrics()

    private var pendingSeedCosts = CBv2MTPSeedCostLedger()

    private init(
        config: CBv2MTPConfig, drafter: any CBv2MTPDrafter,
        model: any CBv2MTPSteppableModel, captureLayers: CBv2MTPCaptureLayers?
    ) {
        var config = config
        if let required = drafter.requiredVerificationMode {
            config.verificationMode = required
            if required != .rectangular { config.maxAutomaticRectangularTokens = 0 }
        }
        // A fixed depth is a declaration: a drafter that cannot reach it
        // stops the build here rather than run a shallower depth. The
        // adaptive ceiling (`maxDraftTokens`) is not a declaration and still
        // narrows to the drafter's cap.
        if let refusal = Self.depthRefusal(config: config, drafter: drafter) {
            preconditionFailure(refusal.description)
        }
        if let maximum = drafter.maximumDraftTokens {
            config.maxDraftTokens = min(config.maxDraftTokens, max(0, maximum))
        }
        if let maximum = drafter.maximumSpeculativeBatch {
            config.maxSpeculativeBatch = min(config.maxSpeculativeBatch, max(1, maximum))
        }
        let requestStatefulRecurrent =
            drafter is any CBv2MTPRequestStatefulDrafter
            && (model as? any CBv2RecurrentMTPSteppableModel)?.recurrentStateSpec != nil
        // The marginal-policy bound is the ADAPTIVE controller's table size,
        // not a verification limit. A block drafter runs at one fixed depth
        // and never consults that table, so the bound does not apply to it —
        // applying it would clamp a declared depth of 5...16 to 4.
        if requestStatefulRecurrent, !(drafter is any CBv2MTPBlockDrafter) {
            config.maxDraftTokens = min(
                config.maxDraftTokens, CBv2MTPMarginalDepthPolicy.maximumDepth)
            config.fixedDraftTokens = config.fixedDraftTokens.map {
                min($0, config.maxDraftTokens)
            }
        }
        if requestStatefulRecurrent {
            // Production recurrent verification requires the captured-window
            // seam. If it is absent, clamp to target-only before any assistant
            // state or draft graph exists; never turn a planned positive-depth
            // Qwen round into multiple serial target forwards.
            let capturedWindow =
                (model as? any CBv2RecurrentMTPSteppableModel)?
                .supportsCapturedVerifyWindow ?? false
            if config.verificationMode != .serialTarget && !capturedWindow {
                config.maxDraftTokens = 0
                config.fixedDraftTokens = config.fixedDraftTokens.map { _ in 0 }
                config.maxAutomaticRectangularTokens = 0
            }
            if config.verificationMode == .serialTarget {
                // The serial column loop is additionally the proven B=1
                // shape: one request-local assistant proposal and one
                // recurrent target transaction per column.
                config.maxSpeculativeBatch = 1
            }
        }
        // Prompt lookup stands in for a CHAIN drafter at a fixed declared
        // depth only. The adaptive controller and the marginal policy never
        // run there, so lookup rounds cannot skew what they learn.
        let lookupNgram =
            (config.fixedDraftTokens ?? 0) > 0 && !(drafter is any CBv2MTPBlockDrafter)
            ? (drafter as? any CBv2MTPRequestStatefulDrafter)?.lookupDraftNgram : nil
        self.lookupNgram = lookupNgram.flatMap { $0 > 0 ? $0 : nil }
        let missDepth =
            (lookupNgram ?? 0) <= 0
            ? nil
            : (drafter as? any CBv2MTPRequestStatefulDrafter)?.lookupMissDraftDepth
                .flatMap { $0 > 0 && $0 < (config.fixedDraftTokens ?? 0) ? $0 : nil }
        self.lookupMissDepth = missDepth
        self.lookupMissDeep =
            missDepth == nil
            ? nil
            : (drafter as? any CBv2MTPRequestStatefulDrafter)?.lookupMissDeepDraft
                .flatMap {
                    $0.depth > (missDepth ?? 0) && $0.depth < (config.fixedDraftTokens ?? 0)
                        && $0.minimumMargin.isFinite ? $0 : nil
                }
        self.missDepthSchedule = config.missDepthSchedule
        self.targetPrefixAcceptance = drafter.supportsTargetPrefixAcceptance
        self.config = config
        self.drafter = drafter
        self.model = model
        self.captureLayers = captureLayers
        self.depthController = CBv2MTPDepthController(
            maxDepth: self.config.maxDraftTokens, fixedDepth: self.config.fixedDraftTokens,
            useCommittedDecodeBaseline: drafter.supportsTargetPrefixAcceptance
                && !(drafter is any CBv2MTPRequestStatefulDrafter),
            ceiling: self.config.draftTokenCeiling)
        self.metrics.verificationMode = self.config.verificationMode
        self.metrics.maxAutomaticRectangularTokens = self.config.maxAutomaticRectangularTokens
    }

    /// The refusal for a fixed depth above the drafter's cap, or nil when the
    /// drafter can serve it (or the depth is adaptive, or the drafter states
    /// no cap).
    static func depthRefusal(
        config: CBv2MTPConfig, drafter: any CBv2MTPDrafter
    ) -> CBv2MTPDepthAboveDrafterCap? {
        guard let fixed = config.fixedDraftTokens, let maximum = drafter.maximumDraftTokens,
            fixed > max(0, maximum)
        else { return nil }
        return CBv2MTPDepthAboveDrafterCap(declared: fixed, drafterCap: max(0, maximum))
    }

    /// Build the driver, or nil when MTP cannot activate: config off (or the
    /// `DARKBLOOM_CBV2_MTP` kill switch), no drafter, or a model that cannot
    /// drive rounds. nil ⇒ the engine is byte-identical to MTP-less builds.
    static func build(
        model: CBv2SteppableModel, drafter: (any CBv2MTPDrafter)?,
        config: CBv2MTPConfig,
        supportsRectangularCacheBank: Bool = true
    ) -> CBv2MTPRoundDriver? {
        guard config.effectiveEnabled, let drafter else { return nil }
        guard let mtpModel = model as? (any CBv2MTPSteppableModel) else { return nil }
        let stateful = drafter is any CBv2MTPRequestStatefulDrafter
        let recurrent =
            (model as? any CBv2RecurrentMTPSteppableModel)?.recurrentStateSpec != nil
        let captureLayers = mtpModel.mtpCaptureLayers
        guard (stateful && recurrent && mtpModel.supportsRequestStatefulMTP)
            || (!stateful && !recurrent && captureLayers != nil)
        else { return nil }
        guard let modelTarget = mtpModel.mtpTargetIdentity,
            let drafterTarget = drafter.mtpTargetIdentity,
            modelTarget == drafterTarget
        else { return nil }
        var config = config
        let effectiveVerification = drafter.requiredVerificationMode
            ?? config.verificationMode
        if stateful, effectiveVerification != .serialTarget,
            !supportsRectangularCacheBank
        {
            config.maxDraftTokens = 0
            config.fixedDraftTokens = config.fixedDraftTokens.map { _ in 0 }
            config.maxAutomaticRectangularTokens = 0
        }
        let driver = CBv2MTPRoundDriver(
            config: config, drafter: drafter, model: mtpModel, captureLayers: captureLayers)
        if driver.usesMarginalPolicy {
            guard mtpModel is any CBv2MTPPolicyTopTwoProviding else { return nil }
            if let availability =
                mtpModel as? any CBv2MTPPolicyTopTwoCapabilityProviding,
                !availability.cbv2MTPPolicyTopTwoAvailable
            {
                return nil
            }
        }
        return driver
    }

    // MARK: Plan-scoped marks

    /// Reset speculation marks. Called immediately before every
    /// `scheduler.plan()` so marks can never leak across plans (a rolled-
    /// back plan's marks must not classify the next plan's rows).
    func beginPlan(
        plannedDecodeRows: Int, canSpeculate: Bool,
        rowIDs: [CBv2RequestID]? = nil
    ) {
        missDepthScheduleArmed = false
        if let rowIDs, depthController.usesCommittedDecodeBaseline {
            let changedCohort = Set(rowIDs) != Set(goodputPlanRows)
            if changedCohort {
                workloadGeneration &+= 1
                committedDecodeClock = CBv2MTPCommittedDecodeClock()
                pendingSeedCosts.removeAll()
            }
            if !canSpeculate || changedCohort {
                committedGoodputClock.reset()
                depthController.cancelCommittedWindow(
                    decodeRowBucket: CBv2MTPDepthController.decodeRowBucket(goodputPlanRows.count))
            }
            goodputPlanRows = rowIDs
            workloadInvalidated = false
            depthController.beginWorkload(rowIDs: rowIDs)
        }
        if !roundMarks.isEmpty { roundMarks = [:] }
        if !seedMarks.isEmpty { seedMarks = [] }
        forceSeedPlan = false
        controllerMeasurementEligible = canSpeculate
        controllerDecision = depthController.select(
            plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate)
        let offered =
            usesMarginalPolicy && canSpeculate && !controllerDecision.isExploration
            ? CBv2MTPDepthDecision(
                depth: config.maxDraftTokens,
                decodeRowBucket: controllerDecision.decodeRowBucket,
                reason: "marginal_offer",
                isExploration: false)
            : controllerDecision
        planDecision = verificationLimitedDecision(
            offered, plannedDecodeRows: plannedDecodeRows)
        guard plannedDecodeRows > 0 else { return }
        metricsLock.lock()
        metrics.selectedDepth = planDecision.depth
        metrics.decodeRowBucket = planDecision.decodeRowBucket
        metrics.depthSelections[planDecision.depth, default: 0] += 1
        metrics.controllerFallbacks[planDecision.reason, default: 0] += 1
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    func previewDecision(
        plannedDecodeRows: Int, canSpeculate: Bool
    ) -> CBv2MTPDepthDecision {
        verificationLimitedDecision(
            depthController.preview(
                plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate),
            plannedDecodeRows: plannedDecodeRows)
    }

    func maximumAutomaticDepth(plannedDecodeRows: Int) -> Int {
        guard config.verificationMode == .automatic, plannedDecodeRows > 0 else {
            return config.maxDraftTokens
        }
        let maxWidth = config.maxAutomaticRectangularTokens / plannedDecodeRows
        return min(config.maxDraftTokens, max(0, maxWidth - 1))
    }

    private func verificationLimitedDecision(
        _ decision: CBv2MTPDepthDecision, plannedDecodeRows: Int
    ) -> CBv2MTPDepthDecision {
        let limit = maximumAutomaticDepth(plannedDecodeRows: plannedDecodeRows)
        guard decision.depth > limit else { return decision }
        return CBv2MTPDepthDecision(
            depth: limit, decodeRowBucket: decision.decodeRowBucket,
            reason: "automatic_rectangular_limit", isExploration: false)
    }

    func requiresNonChainedDepthZeroProbe(_ decision: CBv2MTPDepthDecision) -> Bool {
        depthController.requiresNonChainedDepthZeroProbe(decision)
    }

    var planDepth: Int { planDecision.depth }
    var planDecodeRowBucket: Int { planDecision.decodeRowBucket }

    /// A plan boundary can discover that one otherwise eligible row cannot
    /// complete the selected full round (typically a max-token tail). Mixing
    /// that row's L=1 target decode with neighbors' L=1+k verify changes the
    /// target batch shape. Clamp the whole plan to depth zero so target-only
    /// and MTP execute the same rectangular tail batch.
    func clampPlanDepth(to requestedDepth: Int, reason: String) {
        let newDepth = min(max(requestedDepth, 0), planDecision.depth)
        guard newDepth != planDecision.depth else { return }
        let oldDepth = planDecision.depth
        depthController.cancelCommittedWindow(decodeRowBucket: planDecision.decodeRowBucket)
        planDecision = CBv2MTPDepthDecision(
            depth: newDepth, decodeRowBucket: planDecision.decodeRowBucket,
            reason: reason, isExploration: false)
        metricsLock.lock()
        metrics.selectedDepth = newDepth
        if let count = metrics.depthSelections[oldDepth], count > 0 {
            if count == 1 {
                metrics.depthSelections.removeValue(forKey: oldDepth)
            } else {
                metrics.depthSelections[oldDepth] = count - 1
            }
        }
        metrics.depthSelections[newDepth, default: 0] += 1
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func markRound(_ id: CBv2RequestID, k: Int) { roundMarks[id] = k }
    func markSeed(_ id: CBv2RequestID) { seedMarks.insert(id) }
    func roundMark(for id: CBv2RequestID) -> Int? { roundMarks[id] }
    func isSeedMarked(_ id: CBv2RequestID) -> Bool { seedMarks.contains(id) }


    func forceSynchronizedSeed() {
        forceSeedPlan = true
    }
    /// True when this plan produced any MTP work (round or seed).
    var planHasMTPWork: Bool { !roundMarks.isEmpty || !seedMarks.isEmpty }

    // MARK: Carries

    enum CarryStatus {
        case valid(CBv2MTPCarry)
        /// A carry existed but no longer matches the row (plain step,
        /// preemption, id reuse) — dropped by `validatedCarry`.
        case stale
        case none
    }

    /// Pure check (no mutation) — the chained-path pre-check uses it.
    func hasValidCarry(for rec: CBv2ScheduledRequest) -> Bool {
        guard let carry = carries[rec.id] else { return false }
        return carryMatches(carry, rec: rec)
    }

    /// Validate and return the row's carry; a stale carry is removed here
    /// (invalidate-on-mismatch — one seed step re-establishes it).
    func validatedCarry(for rec: CBv2ScheduledRequest) -> CarryStatus {
        guard let carry = carries[rec.id] else { return .none }
        guard carryMatches(carry, rec: rec) else {
            carries.removeValue(forKey: rec.id)
            return .stale
        }
        return .valid(carry)
    }

    private func carryMatches(_ carry: CBv2MTPCarry, rec: CBv2ScheduledRequest) -> Bool {
        rec.pendingSamples == 0
            && carry.tokensCount == rec.tokens.count
            && carry.token == rec.tokens.last
            && carry.kvOffset == rec.numComputedTokens
    }

    /// Take the row's carry for a launching round (a fresh one is stored at
    /// the round's finalize, or the row seeds again).
    func consumeCarry(for id: CBv2RequestID) -> CBv2MTPCarry? {
        let carry = carries.removeValue(forKey: id)
        if missDepthScheduleArmed, carry != nil { missDepthScheduleLaunched = true }
        return carry
    }

    func storeCarry(
        id: CBv2RequestID, token: Int, hidden: MLXArray,
        shortlist: MLXArray? = nil, previousTopTwoMargin: Double? = nil,
        needsHistoryTransition: Bool = false,
        tokensCount: Int, kvOffset: Int
    ) {
        if tracksPersistentHistory,
            let stateful = drafter as? any CBv2MTPRequestStatefulDrafter,
            assistantStates[id] == nil
        {
            assistantStates[id] = stateful.makeRequestState()
        }
        carries[id] = CBv2MTPCarry(
            token: token, hidden: hidden, shortlist: shortlist,
            previousTopTwoMargin: previousTopTwoMargin,
            needsHistoryTransition: needsHistoryTransition,
            tokensCount: tokensCount, kvOffset: kvOffset)
    }

    var tracksPersistentHistory: Bool {
        config.maxDraftTokens > 0
            && usesRequestStatefulDrafter && config.fixedDraftTokens != 0
    }

    func takeOrMakeAssistantState(
        for id: CBv2RequestID, maximumSequenceLength: Int
    ) throws -> (any CBv2MTPRequestState)? {
        guard tracksPersistentHistory,
            let stateful = drafter as? any CBv2MTPRequestStatefulDrafter
        else { return nil }
        let state = assistantStates.removeValue(forKey: id) ?? stateful.makeRequestState()
        do {
            try stateful.configureRequestState(
                state, maximumSequenceLength: maximumSequenceLength)
        } catch {
            // Configuration detached an existing owner from the map. Put it
            // back before propagating so the fenced cohort retirement path
            // remains the single authority that releases request state.
            assistantStates[id] = state
            throw error
        }
        return state
    }

    var usesMarginalPolicy: Bool {
        config.maxDraftTokens > 0
            && usesRequestStatefulDrafter && config.fixedDraftTokens == nil
    }


    var shouldApplyMarginalPolicyToPlan: Bool {
        usesMarginalPolicy && !planDecision.isExploration
    }
    func observeCommittedTarget(
        id: CBv2RequestID,
        observation: CBv2MTPCommittedTargetObservation,
        detachedState: any CBv2MTPRequestState
    ) {
        guard let stateful = drafter as? any CBv2MTPRequestStatefulDrafter else {
            preconditionFailure("CBv2 MTP committed observation reached a stateless drafter")
        }
        stateful.observeCommittedTarget(observation, requestState: detachedState)
    }

    /// `DARKBLOOM_BONSAI_DFLASH_PROMPT_IN_BEGIN=0` restores lazy absorption
    /// (the first proposal projects the prompt context inside the first
    /// decode round, the parent behavior).
    static let dflashPromptAbsorbEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DFLASH_PROMPT_IN_BEGIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Seed-prefill boundary: a block drafter absorbs its pending prompt
    /// context into its caches now, so the projection runs inside the begin
    /// step's work. Returns the arrays the step must evaluate; a chain
    /// drafter, a disabled switch, or an empty pending set return nothing.
    func absorbCommittedContext(detachedState: any CBv2MTPRequestState) -> [MLXArray] {
        guard Self.dflashPromptAbsorbEnabled,
            let block = drafter as? any CBv2MTPBlockDrafter
        else { return [] }
        return block.absorbPendingContext(requestState: detachedState)
    }

    func marginalDepth(
        for id: CBv2RequestID,
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int,
        decodeRowBucket: Int
    ) -> Int {
        guard usesRequestStatefulDrafter, config.fixedDraftTokens == nil else {
            return min(max(offeredDepth, 0), config.maxDraftTokens)
        }
        let acceptance = requestAcceptance[id] ?? CBv2MTPRequestAcceptanceState()
        let cost = rawCostEstimators[decodeRowBucket] ?? CBv2MTPRawCostEstimator()
        let cappedOffer = min(
            offeredDepth, CBv2MTPMarginalDepthPolicy.maximumDepth)
        if cost.needsSteadyStateProbe(depth: 1) {
            return CBv2MTPMarginalDepthPolicy.boundedProbeDepth(
                offeredDepth: cappedOffer,
                remainingTokens: remainingTokens,
                verificationLimit: verificationLimit)
        }
        return CBv2MTPMarginalDepthPolicy.selectDepth(
            offeredDepth: cappedOffer,
            remainingTokens: remainingTokens,
            verificationLimit: verificationLimit,
            acceptanceProbabilities: acceptance.probabilities,
            previousTargetTopTwoMargin: carries[id]?.previousTopTwoMargin,
            headStepCostRatio: cost.headStepCostRatio)
    }

    func observeRequestAcceptance(
        id: CBv2RequestID,
        draftedDepth: Int,
        acceptedDepth: Int,
        rejectionObserved: Bool,
        endedByTruncation: Bool
    ) {
        guard usesRequestStatefulDrafter, config.fixedDraftTokens == nil else { return }
        var state = requestAcceptance[id] ?? CBv2MTPRequestAcceptanceState()
        state.observe(
            draftedDepth: draftedDepth,
            acceptedDepth: acceptedDepth,
            rejectionObserved: rejectionObserved,
            endedByTruncation: endedByTruncation)
        requestAcceptance[id] = state
    }

    func pendingHistoryCarry(for id: CBv2RequestID) -> CBv2MTPCarry? {
        guard let carry = carries[id], carry.needsHistoryTransition else { return nil }
        return carry
    }

    func takeAssistantState(for id: CBv2RequestID) -> (any CBv2MTPRequestState)? {
        assistantStates.removeValue(forKey: id)
    }

    func restoreAssistantState(
        _ state: any CBv2MTPRequestState, for id: CBv2RequestID
    ) {
        precondition(assistantStates[id] == nil, "duplicate CBv2 MTP assistant state")
        assistantStates[id] = state
    }

    func releaseDetachedAssistantState(_ state: any CBv2MTPRequestState) {
        (drafter as? any CBv2MTPRequestStatefulDrafter)?.releaseRequestState(state)
    }

    func assistantStateCountsForTesting(
        _ id: CBv2RequestID
    ) -> (committed: Int, staged: Int)? {
        assistantStates[id].map { ($0.committedInputCount, $0.stagedInputCount) }
    }

    func materializedAssistantBytes(
        detachedStates: [any CBv2MTPRequestState] = []
    ) -> Int {
        var seen = Set<ObjectIdentifier>()
        var total = 0
        for state in Array(assistantStates.values) + detachedStates {
            guard seen.insert(ObjectIdentifier(state)).inserted else { continue }
            let (next, overflow) = total.addingReportingOverflow(state.materializedBytes)
            total = overflow ? Int.max : next
        }
        return total
    }

    var usesRequestStatefulDrafter: Bool {
        drafter is any CBv2MTPRequestStatefulDrafter
    }

    /// Whether any consumer reads a carried target top-two margin: the
    /// adaptive marginal depth policy or the prompt-lookup miss-deep choice.
    /// Fixed-depth block drafting (DFlash) has neither, so finalize skips the
    /// host readback of policy top-two values there.
    var consumesPolicyTopTwoMargin: Bool {
        usesMarginalPolicy || lookupMissDeep != nil
    }

    /// The block drafter this driver speculates with, if any. One `propose`
    /// per round replaces the k-step chain; everything else is the ordinary
    /// request-stateful path.
    var blockDrafter: (any CBv2MTPBlockDrafter)? {
        drafter as? any CBv2MTPBlockDrafter
    }

    var usesBlockDrafter: Bool { drafter is any CBv2MTPBlockDrafter }

    /// A verify-produced carry primes persistent assistant history for the
    /// one position the round did NOT feed: a chain drafter's finalization
    /// takes the ACCEPTED DRAFT columns, which stop one short of the
    /// correction token. A block drafter's finalization takes every CONFIRMED
    /// column, that one included, so it needs no transition — replaying it
    /// would feed the same context row twice.
    var carryNeedsHistoryTransition: Bool {
        tracksPersistentHistory && !usesBlockDrafter
    }

    /// The hidden rows one committed-history observation carries. A chain
    /// drafter takes the target's pre-norm last hidden; a BLOCK drafter takes
    /// the fused tapped context the target produced for the SAME positions.
    /// The caller must read this while the producing forward is still the
    /// target's last one.
    func committedObservationHidden(_ lastHidden: MLXArray) -> MLXArray {
        guard let block = blockDrafter else { return lastHidden }
        guard let context = block.blockContextHidden() else {
            preconditionFailure(
                "CBv2 block MTP: the target's context tap is not armed")
        }
        return context
    }

    // MARK: Prompt lookup

    /// Depth of a drafted round (the lookup missed): the deeper chain when
    /// every row's carried target top-two margin clears the threshold, else
    /// `missDepth`. The margin is the target's own, already on the host from
    /// the previous round's finalize, so the choice is deterministic and
    /// adds no sync. A row with no margin yet (its first round) stays shallow.
    func lookupMissRoundDepth(for rows: [CBv2ScheduledRequest], missDepth: Int) -> Int {
        let deepest = lookupMissDeep?.depth ?? missDepth
        if missDepthScheduleLaunched, !missDepthSchedule.isEmpty {
            missDepthSchedule.removeFirst()
        }
        missDepthScheduleLaunched = false
        if let depth = missDepthSchedule.first {
            missDepthScheduleArmed = true
            return min(max(depth, 1), deepest)
        }
        guard let deep = lookupMissDeep, !rows.isEmpty else { return missDepth }
        let allConfident = rows.allSatisfy { rec in
            guard let margin = carries[rec.id]?.previousTopTwoMargin else { return false }
            return margin >= deep.minimumMargin
        }
        return allConfident ? deep.depth : missDepth
    }

    /// The `depth` tokens that followed the most recent earlier occurrence of
    /// the row's newest `lookupNgram` committed tokens, or nil. Only the
    /// declared depth looks up; a clamped tail or budget round keeps the
    /// drafter's chain.
    func lookupDraft(for rec: CBv2ScheduledRequest, depth: Int) -> [Int]? {
        guard depth > 0, depth == config.fixedDraftTokens else { return nil }
        return lookupIndex(for: rec)?.draft(rec.tokens)
    }

    /// Index the row's committed history while a launched step runs, so the
    /// next round's lookup hashes only the tokens that step commits.
    func indexLookupHistory(for rec: CBv2ScheduledRequest) {
        lookupIndex(for: rec)?.catchUp(rec.tokens)
    }

    private func lookupIndex(for rec: CBv2ScheduledRequest) -> CBv2MTPPromptLookupIndex? {
        guard let ngram = lookupNgram, let depth = config.fixedDraftTokens else { return nil }
        if let index = lookupIndexes[rec.id] { return index }
        let index = CBv2MTPPromptLookupIndex(ngram: ngram, depth: depth)
        lookupIndexes[rec.id] = index
        return index
    }

    private func releaseAssistantState(_ id: CBv2RequestID) {
        guard let state = assistantStates.removeValue(forKey: id),
            let stateful = drafter as? any CBv2MTPRequestStatefulDrafter
        else { return }
        stateful.releaseRequestState(state)
    }

    /// Preemption / membership hygiene: the structural fingerprint would
    /// catch these lazily, but dropping eagerly keeps no stale device
    /// arrays alive.
    func invalidateCarry(_ id: CBv2RequestID) {
        carries.removeValue(forKey: id)
        releaseAssistantState(id)
        pendingSeedCosts.invalidate(id)
        requestAcceptance.removeValue(forKey: id)
    }

    /// The request left the engine for good — ids are legally reusable, so
    /// every per-id trace must go (a reused id must never inherit a carry).
    func requestDidFinish(_ id: CBv2RequestID) {
        if goodputPlanRows.contains(id) { invalidateWorkload() }
        carries.removeValue(forKey: id)
        releaseAssistantState(id)
        roundMarks.removeValue(forKey: id)
        seedMarks.remove(id)
        pendingSeedCosts.invalidate(id)
        requestAcceptance.removeValue(forKey: id)
        lookupIndexes.removeValue(forKey: id)
    }

    /// Invalidate only adaptive stateless learning. Other live rows retain
    /// their carries; the next plan calibrates its new request cohort.
    private func invalidateWorkload() {
        guard depthController.usesCommittedDecodeBaseline else { return }
        workloadGeneration &+= 1
        workloadInvalidated = true
        goodputPlanRows.removeAll(keepingCapacity: true)
        committedDecodeClock = CBv2MTPCommittedDecodeClock()
        committedGoodputClock.reset()
        pendingSeedCosts.removeAll()
        depthController.invalidateWorkload()
        metricsLock.lock()
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    private func acceptsWorkloadMeasurement(_ measurement: CBv2MTPStepMeasurement?) -> Bool {
        guard depthController.usesCommittedDecodeBaseline else { return true }
        guard !workloadInvalidated else { return false }
        // Older host-only tests construct unstamped measurements. Production
        // always stamps at plan capture and preserves the stamp at attachment.
        return measurement?.workloadGeneration.map { $0 == workloadGeneration } ?? true
    }

    /// Drain/shutdown drops every request trace and adaptive workload while
    /// retaining cumulative metrics for a final poll.
    func removeAllRequestState() {
        invalidateWorkload()
        carries.removeAll(keepingCapacity: false)
        rawCostEstimators.removeAll(keepingCapacity: false)
        for id in Array(assistantStates.keys) { releaseAssistantState(id) }
        roundMarks.removeAll(keepingCapacity: false)
        seedMarks.removeAll(keepingCapacity: false)
        pendingSeedCosts.removeAll()
        requestAcceptance.removeAll(keepingCapacity: false)
        lookupIndexes.removeAll(keepingCapacity: false)
    }

    var requestStateCountForTesting: Int {
        carries.count + assistantStates.count + requestAcceptance.count
            + roundMarks.count + seedMarks.count + pendingSeedCosts.count
            + lookupIndexes.count
    }

    // MARK: Metrics (lock-protected; polled cross-thread)

    func recordSkip(_ reason: String) {
        metricsLock.lock()
        metrics.skippedRows[reason, default: 0] += 1
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func recordControllerFallback(_ reason: String) {
        metricsLock.lock()
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func recordSeedSteps(_ count: Int) {
        guard count > 0 else { return }
        metricsLock.lock()
        metrics.seedSteps += count
        metricsLock.unlock()
    }

    func recordEarlyDraftSubmission() {
        metricsLock.lock()
        metrics.earlyDraftSubmissions += 1
        metricsLock.unlock()
    }

    func recordRound(
        drafted: Int, accepted: Int, emitted: Int, lookup: Bool = false,
        audit: CBv2MTPRoundAuditRecord? = nil
    ) {
        metricsLock.lock()
        metrics.rounds += 1
        metrics.draftedTokens += drafted
        metrics.acceptedTokens += accepted
        metrics.emittedTokens += emitted
        if lookup {
            metrics.lookupRounds += 1
            metrics.lookupAcceptedTokens += accepted
        }
        if let audit {
            metrics.roundAudits.append(audit)
            if metrics.roundAudits.count > CBv2MTPRoundAuditRecord.retainedRecordCap {
                metrics.roundAudits.removeFirst(
                    metrics.roundAudits.count - CBv2MTPRoundAuditRecord.retainedRecordCap)
            }
        }
        if metrics.perPositionAccepted.count < drafted {
            metrics.perPositionAccepted.append(
                contentsOf: Array(
                    repeating: 0, count: drafted - metrics.perPositionAccepted.count))
        }
        for position in 0 ..< accepted {
            metrics.perPositionAccepted[position] += 1
        }
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    /// The controller optimizes the synchronized rectangular step, so it
    /// learns the minimum accepted prefix that every participating verify
    /// row can commit together, once per step (not once per row).
    func recordStepAcceptance(
        drafted: Int, accepted: Int, observedDrafts: Int,
        decodeRowBucket: Int,
        measurement: CBv2MTPStepMeasurement? = nil
    ) {
        guard acceptsWorkloadMeasurement(measurement) else { return }
        depthController.observeAcceptance(
            decodeRowBucket: decodeRowBucket,
            drafted: observedDrafts,
            accepted: accepted)
        metricsLock.lock()
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    func claimPendingSeedCost(
        decodeRowBucket: Int, finalizedVerifyIDs: Set<CBv2RequestID>,
        measurement: CBv2MTPStepMeasurement? = nil
    ) -> UInt64 {
        guard acceptsWorkloadMeasurement(measurement) else { return 0 }
        return pendingSeedCosts.take(
            decodeRowBucket: decodeRowBucket, requestIDs: finalizedVerifyIDs)
    }

    /// The ordinary alternative to stateless MTP can pipeline target decode.
    /// Measure its actual commit cadence, with no extra clock or GPU readback.
    /// Reset on every nonqualifying finalize to exclude idle/cohort transitions.
    func recordCommittedDecodeBaseline(
        measurement: CBv2MTPStepMeasurement?, completedAtNanos: UInt64,
        sampledRows: [CBv2RequestID], finalizedPlainRowCount: Int,
        hasChainedSuccessor: Bool
    ) {
        guard depthController.usesCommittedDecodeBaseline,
            acceptsWorkloadMeasurement(measurement) else { return }
        let eligible = measurement.map {
            $0.costEligible && $0.chained && !$0.seedOnly
                && $0.actualDepth == 0 && $0.decision.depth == 0
                && $0.decision.decodeRowBucket == CBv2MTPDepthController.decodeRowBucket(sampledRows.count)
        } == true && hasChainedSuccessor && sampledRows.count == finalizedPlainRowCount
        guard let elapsed = committedDecodeClock.observe(
            completedAtNanos: completedAtNanos, rowIDs: sampledRows, eligible: eligible),
            let measurement else { return }
        depthController.observeCommittedDecodeInterval(
            decodeRowBucket: measurement.decision.decodeRowBucket, wallTimeNanos: elapsed)
    }

    func recordStepCost(
        _ measurement: CBv2MTPStepMeasurement,
        wallTimeNanos: UInt64,
        finalizedPlainWork: Bool,
        finalizedSeedIDs: Set<CBv2RequestID>,
        finalizedVerification: Bool,
        claimedSeedCostNanos: UInt64,
        completedAtNanos: UInt64 = 0,
        committedRows: [CBv2RequestID] = [],
        committedTokenCount: Int = 0
    ) {
        guard wallTimeNanos > 0 else { return }
        let decision = measurement.decision
        if depthController.usesCommittedDecodeBaseline {
            guard acceptsWorkloadMeasurement(measurement) else {
                // Finishing may precede this step's measurement callback.
                // Preserve executed-work telemetry without reviving learning.
                if measurement.actualDepth > 0, !measurement.seedOnly, finalizedVerification {
                    metricsLock.lock()
                    metrics.totalRoundWallTimeNanos &+= wallTimeNanos &+ claimedSeedCostNanos
                    metricsLock.unlock()
                }
                return
            }
            let sample = committedGoodputClock.observe(
                measurement: measurement, completedAtNanos: completedAtNanos,
                isolatedWallTimeNanos: wallTimeNanos, rowIDs: committedRows,
                committedTokens: committedTokenCount)
            if measurement.actualDepth > 0, !measurement.seedOnly,
                finalizedVerification
            {
                if let sample {
                    depthController.recordCommittedVerification(
                        decision: decision, wallTimeNanos: sample.wallTimeNanos,
                        committedTokens: sample.committedTokens, rowCount: committedRows.count)
                } else {
                    depthController.cancelCommittedWindow(decodeRowBucket: decision.decodeRowBucket)
                }
                metricsLock.lock()
                // Warmup is real work even when excluded from steady EWMA.
                metrics.totalRoundWallTimeNanos &+= sample?.wallTimeNanos
                    ?? (wallTimeNanos &+ claimedSeedCostNanos)
                refreshControllerMetricsLocked()
                metricsLock.unlock()
                return
            }
        }
        if measurement.seedOnly, decision.depth > 0 {
            guard measurement.costEligible, !finalizedSeedIDs.isEmpty else {
                depthController.cancelCommittedWindow(decodeRowBucket: decision.decodeRowBucket)
                return
            }
            pendingSeedCosts.record(
                decodeRowBucket: decision.decodeRowBucket,
                requestIDs: finalizedSeedIDs,
                nanos: wallTimeNanos)
            return
        }
        if measurement.actualDepth == 0 {
            depthController.cancelCommittedWindow(decodeRowBucket: decision.decodeRowBucket)
        }
        let rawCostEligible =
            usesMarginalPolicy
            && measurement.costEligible && !measurement.chained
            && measurement.actualDepth >= 0
            && (measurement.actualDepth == 0
                ? finalizedPlainWork : finalizedVerification)
        var steadyRawCostRecorded = true
        if rawCostEligible {
            var estimator =
                rawCostEstimators[decision.decodeRowBucket]
                ?? CBv2MTPRawCostEstimator()
            steadyRawCostRecorded = estimator.observe(
                depth: measurement.actualDepth,
                rawWallTimeNanos: Double(wallTimeNanos),
                chained: false)
            rawCostEstimators[decision.decodeRowBucket] = estimator
        }
        if rawCostEligible, measurement.actualDepth > 0,
            !steadyRawCostRecorded
        {
            // Keep the controller depth unsampled too: its next selection is
            // an immediate bounded retry that obtains steady-state C1.
            return
        }
        let attributed = wallTimeNanos &+ claimedSeedCostNanos
        let recorded = depthController.recordFinalizedStep(
            decision: decision,
            actualDepth: measurement.actualDepth,
            wallTimeNanos: attributed,
            costEligible: measurement.costEligible,
            chained: measurement.chained,
            finalizedPlainWork: finalizedPlainWork,
            finalizedVerification: finalizedVerification)
        guard recorded else { return }
        metricsLock.lock()
        if measurement.actualDepth > 0 {
            metrics.totalRoundWallTimeNanos &+= attributed
        }
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    var pendingSeedCostCountForTesting: Int { pendingSeedCosts.count }

    func activeDepthForTesting(decodeRowBucket: Int) -> Int {
        depthController.activeDepthForTesting(decodeRowBucket: decodeRowBucket)
    }

    func probeIntervalForTesting(decodeRowBucket: Int) -> Int {
        depthController.probeIntervalForTesting(decodeRowBucket: decodeRowBucket)
    }

    func metricsSnapshot() -> CBv2MTPMetrics {
        metricsLock.lock()
        defer { metricsLock.unlock() }
        return metrics
    }

    func recordVerificationStrategy(rectangular: Bool) {
        metricsLock.lock()
        if rectangular {
            metrics.rectangularVerificationRounds += 1
        } else {
            metrics.serialVerificationRounds += 1
        }
        metricsLock.unlock()
    }

    private func refreshControllerMetricsLocked() {
        let snapshot = depthController.snapshot()
        metrics.conditionalAcceptance = snapshot.conditionalAcceptance
        metrics.costInputs = snapshot.costInputs
    }
}

/// Official Gemma candidate generation carries the target hidden state that
/// predicted the newest accepted/unfed token. In a verify tensor whose input
/// columns are `[seed, d1, ...]`, that is exactly column `acceptedDrafts`.
/// Kept as a pure seam so indexing is deterministic and fixture-independent.
enum CBv2MTPHiddenIndex {
    static func carryColumn(targetOutputIndex: Int, draftDepth: Int) -> Int {
        precondition(targetOutputIndex >= 0 && targetOutputIndex <= draftDepth)
        return targetOutputIndex
    }
}

/// A BLOCK drafter's next block reads the tapped context of exactly the
/// columns the round CONFIRMED — the reference's
/// `hidden = hidden[:, :accepted + 1, :]`. That is ONE column wider than the
/// chain drafter's accepted-draft slice, because the correction token's own
/// position is context for the next block. Kept as a pure seam so the width
/// is deterministic and testable without a model.
enum CBv2MTPBlockContextIndex {
    static func confirmedColumns(confirmed: Int, verifyWidth: Int) -> Range<Int> {
        precondition(
            confirmed >= 1 && confirmed <= verifyWidth,
            "CBv2 block MTP: \(confirmed) confirmed columns of a \(verifyWidth)-wide verify")
        return 0 ..< confirmed
    }
}
