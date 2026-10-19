// EngineLoopV2+MTPFinalize.swift
//
// Finalize-time target-authoritative acceptance, streaming, and KV rollback.

import Cmlx
import Foundation
import MLX

extension EngineLoopV2 {
    /// Minimum target top-K probability mass (parts-per-million) at the
    /// carry position before the next draft may score only the shortlist
    /// rows. Below this the shortlist would too often miss the token the
    /// target actually produces next, costing acceptance for no byte win —
    /// the round simply reads the full head instead. 0.90 keeps greedy
    /// production traffic (mass typically ≥0.99 at K=256) shortlisted while
    /// flat/uncertain positions fall back.
    static let mtpShortlistMassThresholdPPM: Int32 = 900_000

    /// Wait for an array that `asyncEval` has already submitted by polling
    /// its completion event instead of sleeping in Metal's blocking
    /// `waitUntilSignaledValue`. The GPU has nothing queued behind the
    /// round's acceptance packet, so the time between the packet's completion
    /// and this thread resuming is idle GPU time on every round; a polled
    /// wait resumes within microseconds where the blocking wait sometimes
    /// takes a scheduler wake-up. The subsequent `asArray` then finds the
    /// array available and copies it out. The poll is bounded: past the cap
    /// the ordinary blocking read takes over (and reports any GPU error).
    static func mtpPollUntilAvailable(_ array: MLXArray) {
        var available = false
        let start = DispatchTime.now().uptimeNanoseconds
        while _mlx_array_is_available(&available, array.ctx) == 0, !available {
            if DispatchTime.now().uptimeNanoseconds &- start > 1_000_000_000 { return }
        }
    }

    /// `DARKBLOOM_BONSAI_DFLASH_ROUND1_IN_BEGIN=0` restores the parent: the
    /// first round proposes its block at its own launch.
    static let dflashRound1InBeginEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_DFLASH_ROUND1_IN_BEGIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Seed boundary, called for a prompt row's FIRST sampled token before
    /// that token is emitted. The token is the first round's anchor and the
    /// prompt context is already absorbed, so the block drafter proposes the
    /// first round's block here, through the same `prestageBlock` a verified
    /// round uses for its successor, and the step waits for it. The round
    /// then takes the prestaged block exactly as every later round does.
    /// Guards mirror that successor prestage: one row, no lookup chain, a
    /// fixed depth the next plan offers, and room for a full round.
    func mtpPrestageSeedRound(_ step: CBv2InFlightStep, rec: CBv2ScheduledRequest) {
        guard Self.dflashRound1InBeginEnabled, let mtp, let round = step.mtpRound,
            round.verify == nil, step.sampledRows.count == 1,
            let block = mtp.blockDrafter, mtp.lookupNgram == nil,
            mtp.config.fixedDraftTokens != nil,
            round.prefillCarries.contains(where: { $0.id == rec.id }),
            rec.pendingSamples == 0,
            rec.tokens.count > rec.request.promptTokens.count,
            let anchor = rec.tokens.last,
            let state = round.committedObservationRows.first(where: { $0.id == rec.id })?
                .assistantState
        else { return }
        let k = mtp.previewDecision(plannedDecodeRows: 1, canSpeculate: true).depth
        guard k > 0, k == mtp.config.fixedDraftTokens,
            rec.request.maxTokens - rec.generatedTokenCount > k
        else { return }
        block.prepareDraft(history: rec.tokens, depth: k, requestState: state)
        block.prestageBlock(anchor: anchor, depth: k, requestState: state)
        eval(block.evaluationTargets(for: state))
    }

    /// Runs at the step's existing host-sync boundary after ordinary sampled
    /// rows finalize and before deferred KV releases.
    func finalizeMTPRound(_ step: CBv2InFlightStep) {
        guard let mtp, let round = step.mtpRound else { return }

        // Plain prompt/decode observations may own request-state arrays that
        // the just-fenced target graph referenced. Restore or release them
        // only now, never at cancellation time while evaluation is live.
        for observation in round.committedObservationRows {
            if step.discard.contains(observation.id)
                || scheduler.record(for: observation.id) == nil
            {
                round.deferredAssistantReleases.append(
                    observation.assistantState)
            } else {
                mtp.restoreAssistantState(
                    observation.assistantState, for: observation.id)
            }
        }

        // The ordinary finalize loop has confirmed each seed row's bonus.
        if let seedHidden = round.seedHidden {
            let seedPolicyTopTwo =
                mtp.consumesPolicyTopTwoMargin
                ? round.seedPolicyTopTwoValues?.asArray(Float.self) : nil
            if seedPolicyTopTwo != nil { CBv2CoreInstrumentation.recordHostSync() }
            for (id, decodeIndex) in round.seedRows {
                guard !step.discard.contains(id),
                    let rec = scheduler.record(for: id)
                else { continue }
                let margin = seedPolicyTopTwo.map { values in
                    let base = decodeIndex * 2
                    return Double(values[base] - values[base + 1])
                }
                mtp.storeCarry(
                    id: id, token: rec.tokens.last!,
                    hidden: seedHidden[decodeIndex ..< (decodeIndex + 1), 0..., 0...],
                    previousTopTwoMargin: margin,
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
                round.finalizedSeedIDs.insert(id)
            }
        }

        // A prompt row that just sampled its first token carries into a round
        // directly: the token is the next round's anchor and the prompt's
        // last-position hidden is the drafter's chaining input; the prompt's
        // own observation absorbed the history in this same step. The carry
        // fingerprints the row exactly as a seed step's would.
        for carry in round.prefillCarries {
            guard !step.discard.contains(carry.id),
                let rec = scheduler.record(for: carry.id),
                rec.pendingSamples == 0,
                rec.tokens.count > rec.request.promptTokens.count,
                let token = rec.tokens.last
            else { continue }
            mtp.storeCarry(
                id: carry.id, token: token, hidden: carry.hidden,
                tokensCount: rec.tokens.count, kvOffset: rec.numComputedTokens)
        }

        guard let verify = round.verify else { return }
        let k = verify.k
        // Host readbacks of the MTP round, each counted: an MTP-round
        // finalize adds up to three syncs to the step's one (seed policy
        // margin above, acceptance packet, verify policy margin). Serial
        // target verification adds one blocking eval per column at launch
        // (`EngineLoopV2+MTPTargetVerification`); a logprob segment adds
        // three readbacks (`CBv2Logprobs.assemble`); a round whose capture
        // could not be fenced adds one blocking eval (`CBv2MTPCaptureFence`
        // fallback in `EngineLoopV2+MTPExecution`).
        Self.mtpPollUntilAvailable(verify.acceptancePacket)
        let host = verify.acceptancePacket.asArray(Int32.self)
        CBv2CoreInstrumentation.recordHostSync()
        let policyTopTwoHost =
            mtp.consumesPolicyTopTwoMargin
            ? verify.policyTopTwoValues?.asArray(Float.self) : nil
        if policyTopTwoHost != nil { CBv2CoreInstrumentation.recordHostSync() }
        let draftCount = verify.rows.count * k
        let targetWidth = 1 + k
        var anyRejected = false

        struct RowOutcome {
            let batchIndex: Int
            let metadata: CBv2MTPRoundInFlight.VerifyRow
            let rec: CBv2ScheduledRequest
            let targets: [Int]
            let accepted: Int
        }

        // Resolve each row's natural target-authoritative prefix, then choose
        // one committed width for the rectangular step.
        var outcomes: [RowOutcome] = []
        outcomes.reserveCapacity(verify.rows.count)
        var commonEmitted = targetWidth

        for (batchIndex, metadata) in verify.rows.enumerated() {
            let id = metadata.id
            if step.discard.contains(id) || scheduler.record(for: id) == nil {
                if let evaluations = verify.recurrentEvaluations[id] {
                    do {
                        for evaluation in evaluations.reversed() { try evaluation.rollback() }
                    } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP discard rollback failed for \(id): \(error)")
                    }
                }
                if let state = metadata.assistantState {
                    round.deferredAssistantReleases.append(state)
                }
                anyRejected = true
                continue
            }
            let rec = scheduler.record(for: id)!
            let drafts = (0 ..< k).map { Int(host[batchIndex * k + $0]) }
            let targets = (0 ..< targetWidth).map {
                Int(host[draftCount + batchIndex * targetWidth + $0])
            }

            var accepted = 0
            while accepted < k, targets[accepted] == drafts[accepted] { accepted += 1 }
            var naturalEmitted = accepted + 1
            naturalEmitted = min(
                naturalEmitted,
                rec.request.maxTokens - rec.generatedTokenCount)
            if let stopIndex = targets[..<naturalEmitted].firstIndex(where: {
                rec.request.stopTokens.contains($0)
            }) {
                naturalEmitted = stopIndex + 1
            }
            commonEmitted = min(commonEmitted, naturalEmitted)
            outcomes.append(
                RowOutcome(
                    batchIndex: batchIndex,
                    metadata: metadata,
                    rec: rec,
                    targets: targets,
                    accepted: accepted))
        }

        round.finalizedVerifyIDs = Set(outcomes.map { $0.metadata.id })
        round.claimedSeedCostNanos = mtp.claimPendingSeedCost(
            decodeRowBucket: mtp.planDecodeRowBucket,
            finalizedVerifyIDs: round.finalizedVerifyIDs,
            measurement: step.mtpMeasurement)

        if !outcomes.isEmpty {
            let stepAccepted = outcomes.map { min($0.accepted, commonEmitted) }.min() ?? 0
            let observedDrafts =
                commonEmitted <= stepAccepted
                ? commonEmitted : min(k, stepAccepted + 1)
            mtp.recordStepAcceptance(
                drafted: k,
                accepted: stepAccepted,
                observedDrafts: observedDrafts,
                decodeRowBucket: mtp.planDecodeRowBucket,
                measurement: step.mtpMeasurement)
        }

        for outcome in outcomes {
            let batchIndex = outcome.batchIndex
            let metadata = outcome.metadata
            let id = metadata.id
            let rec = outcome.rec
            let accepted = outcome.accepted
            let emitted = Array(outcome.targets.prefix(commonEmitted))

            // Confirm in order with the same stop and length semantics as the
            // ordinary finalize loop.
            let detokenizer = detokenizers[id]
            let hasStopStrings = !rec.request.stopStrings.isEmpty
            var kept: [Int] = []
            var textPieces: [String] = []
            var finishReason: CBv2FinishReason?
            for token in emitted {
                scheduler.recordSampled(id: id, token: token)
                kept.append(token)
                if rec.request.stopTokens.contains(token) {
                    finishReason = .stop
                    break
                }
                if hasStopStrings {
                    textPieces.append(detokenizer?.push([token]) ?? "")
                    if detokenizer?.matchedStopString == true {
                        finishReason = .stop
                        break
                    }
                }
                if rec.generatedTokenCount >= rec.request.maxTokens {
                    finishReason = .length
                    break
                }
            }

            // Correct KV and scheduler state before any terminal release.
            let confirmed = kept.count
            round.committedVerifyTokenCount += kept.filter {
                !rec.request.stopTokens.contains($0)
            }.count
            for packet in verify.diagnostics where packet.requestID == id {
                let drafts = (0..<k).map { Int(host[batchIndex * k + $0]) }
                packet.reconcile(
                    accepted: accepted, confirmed: confirmed, drafts: drafts, targets: outcome.targets)
            }
            let rejected = (1 + k) - confirmed
            let committedDraftCount = min(accepted, max(0, confirmed - 1))
            // The next round's drafter block is built and submitted only
            // after this round's tokens are handed to the stream and the
            // round is journaled (below): the host's graph build no longer
            // delays the emission, and the GPU sees the same submissions in
            // the same order.
            var prestage: (() -> Void)?
            if let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter,
                let state = metadata.assistantState
            {
                let committedHidden: MLXArray
                if let blockContext = verify.blockContext {
                    committedHidden = blockContext[
                        batchIndex ..< batchIndex + 1,
                        CBv2MTPBlockContextIndex.confirmedColumns(
                            confirmed: confirmed, verifyWidth: targetWidth),
                        0...]
                } else {
                    committedHidden = verify.lastHidden[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount,
                        0...]
                }
                let committedDraftTokens = verify.draftIDs[
                    batchIndex ..< batchIndex + 1,
                    0 ..< committedDraftCount]
                if let carry = metadata.lookupCarry {
                    stateful.commitLookupRound(
                        requestState: state,
                        seedToken: MLXArray([Int32(carry.token)]).reshaped([1, 1]),
                        seedHidden: carry.hidden,
                        committedDraftTokens: committedDraftTokens,
                        committedTargetHidden: committedHidden)
                } else {
                    stateful.finalizeRound(
                        requestState: state,
                        confirmedInputTokens: 1 + committedDraftCount,
                        committedDraftTokens: committedDraftTokens,
                        committedTargetHidden: committedHidden)
                    if confirmed == accepted + 1, let block = mtp.blockDrafter {
                        block.recordValidatedAcceptance(accepted, requestState: state)
                    }
                    if finishReason == nil, confirmed > 0, mtp.lookupNgram == nil,
                        verify.rows.count == 1,
                        rec.request.maxTokens - rec.generatedTokenCount > k,
                        let block = mtp.blockDrafter
                    {
                        let history = rec.tokens
                        let anchor = kept[confirmed - 1]
                        prestage = {
                            block.prepareDraft(history: history, depth: k, requestState: state)
                            block.prestageBlock(anchor: anchor, depth: k, requestState: state)
                        }
                    }
                }
            }
            if let evaluations = verify.recurrentEvaluations[id] {
                if evaluations.count == 1, evaluations[0].isCaptured {
                    // Capture-verify: one transaction spans the window. The
                    // accepted prefix commits by device-side selection of the
                    // captured state at position `confirmed`; zero confirmed
                    // tokens restore the pre-verify snapshot by rollback.
                    do {
                        if confirmed > 0 {
                            try evaluations[0].commit(keepPositions: confirmed)
                        } else {
                            try evaluations[0].rollback()
                        }
                    } catch {
                        preconditionFailure(
                            "CBv2 captured MTP finalization failed for \(id): \(error)")
                    }
                } else {
                    precondition(
                        evaluations.count == 1 + k,
                        "CBv2 recurrent MTP verification generation count mismatch")
                    do {
                        for evaluation in evaluations.suffix(rejected).reversed() {
                            try evaluation.rollback()
                        }
                        for evaluation in evaluations.prefix(confirmed) {
                            try evaluation.commit()
                        }
                    } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP finalization failed for \(id): \(error)")
                    }
                }
            }
            if rejected > 0 {
                for sequence in metadata.storageRows { sequence.rollback(rejected) }
                anyRejected = true
            }
            for sequence in metadata.storageRows { sequence.commitSpeculativeWrite() }
            if rejected > 0 {
                scheduler.discardPendingSamples(id: id, count: rejected)
                scheduler.rollbackComputed(id: id, tokens: rejected)
            }
            // The speculative suffix is now reconciled in BOTH the page tables
            // and scheduler record. Publish only the accepted frontier; doing
            // this before rollback would make a rejected block cache-visible.
            let launchedEnd = step.computedRanges[id]?.upperBound ?? rec.numComputedTokens
            publishFinalizedResidentBlocks(
                requestID: id,
                safeComputedEnd: min(launchedEnd, rec.numComputedTokens))

            if hasStopStrings {
                stream(for: id)?.emit(
                    .delta(text: textPieces.joined(), tokens: kept, logprobs: nil))
            } else {
                let stream = stream(for: id)
                stream?.reserveEmission()
                let endsWithStopToken = finishReason == .stop
                let pushTokens = endsWithStopToken ? Array(kept.dropLast()) : kept
                let allTokens = kept
                detokQueue.async {
                    let text = pushTokens.isEmpty ? "" : (detokenizer?.push(pushTokens) ?? "")
                    stream?.emit(
                        .delta(text: text, tokens: allTokens, logprobs: nil),
                        consumingReservation: true)
                }
            }

            let observedAccepted = min(accepted, confirmed)
            // Per-request timing: this verify row confirmed at the step's
            // readback-done instant (already read by `finalize`).
            rec.recordStepParticipation(step: step, batchRows: step.tokenProducingRows)
            rec.recordMTPRound(drafted: k, accepted: observedAccepted)
            if confirmed > 0 {
                rec.timing.decodeSteps &+= 1
                decodeRowsTotal = Self.saturatingAdd(decodeRowsTotal, 1)
            }
            // Acceptance/rollback audit record (observability): every value is
            // already on the host at this boundary. The scheduler fields are
            // read AFTER recordSampled/rollbackComputed above, so the record
            // states the row's post-round accounting — the boundary invariant
            // a consumer checks is
            // `numComputedAfter == tokensCountAfter - 1`.
            mtp.recordRound(
                drafted: k, accepted: observedAccepted, emitted: confirmed,
                lookup: metadata.lookupCarry != nil,
                audit: CBv2MTPRoundAuditRecord(
                    requestID: id.raw,
                    k: k,
                    draftTokens: Array(
                        host[batchIndex * k ..< (batchIndex + 1) * k].map(Int.init)),
                    targetTokens: outcome.targets,
                    accepted: accepted,
                    confirmed: confirmed,
                    rejected: rejected,
                    tokensCountAfter: rec.tokens.count,
                    numComputedAfter: rec.numComputedTokens,
                    generatedAfter: rec.generatedTokenCount,
                    finishReason: finishReason.map { String(describing: $0) }))
            let rejectionObserved = accepted < k && confirmed > accepted
            let acceptanceTruncated =
                !rejectionObserved && confirmed <= accepted && confirmed < k
            mtp.observeRequestAcceptance(
                id: id,
                draftedDepth: k,
                acceptedDepth: observedAccepted,
                rejectionObserved: rejectionObserved,
                endedByTruncation: acceptanceTruncated)
            // After the emission and this round's journal entry, so the
            // gap between them stays the record's (the stream's reader
            // snapshots the journal after draining).
            prestage?()

            if let finishReason {
                if let state = metadata.assistantState {
                    mtp.releaseDetachedAssistantState(state)
                }
                finishRequest(id, reason: finishReason)
            } else {
                if let state = metadata.assistantState {
                    mtp.restoreAssistantState(state, for: id)
                }
                // No inline deadline check: an MTP row that just confirmed
                // tokens is making progress and its decode lease is refreshed
                // in `refreshProgressLeases` (run at the end of the enclosing
                // `finalize`, after this round). Lease expiry is evaluated
                // centrally in `processLeaseExpiry` — identical typed-terminal
                // semantics to the ordinary decode path.
                let hiddenColumn = CBv2MTPHiddenIndex.carryColumn(
                    targetOutputIndex: confirmed - 1, draftDepth: k)
                // Shortlist coverage gate: hand the accepted position's
                // target top-K ids to the next draft only when their
                // probability mass (packet tail, parts-per-million) clears
                // the threshold; otherwise the draft falls back to the full
                // head for that round.
                var carryShortlist: MLXArray?
                if let shortlistIDs = verify.shortlistIDs {
                    let massBase = draftCount + verify.rows.count * targetWidth
                    let mass = host[massBase + batchIndex * targetWidth + hiddenColumn]
                    if mass >= Self.mtpShortlistMassThresholdPPM {
                        carryShortlist = shortlistIDs[batchIndex, hiddenColumn]
                    }
                }
                let previousTopTwoMargin: Double? = policyTopTwoHost.map { values in
                    let base =
                        (batchIndex * targetWidth + hiddenColumn) * 2
                    return Double(values[base] - values[base + 1])
                }
                mtp.storeCarry(
                    id: id, token: kept[confirmed - 1],
                    hidden: verify.lastHidden[
                        batchIndex ..< (batchIndex + 1),
                        hiddenColumn ..< (hiddenColumn + 1), 0...],
                    shortlist: carryShortlist,
                    previousTopTwoMargin: previousTopTwoMargin,
                    needsHistoryTransition: mtp.carryNeedsHistoryTransition,
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
            }
        }

        // Verify rows emitted tokens without passing through the sampler;
        // drop their configured row state so the next ordinary sample
        // reconfigures from confirmed history (exact penalty counts and
        // per-request RNG step indices).
        if !outcomes.isEmpty {
            sampler.mtpRoundDidCommit(requestIDs: outcomes.map { $0.metadata.id })
        }

        // Rejected suffixes advanced eager device offsets past host truth.
        if anyRejected {
            eagerCompositionStale = true
        }
    }
}
