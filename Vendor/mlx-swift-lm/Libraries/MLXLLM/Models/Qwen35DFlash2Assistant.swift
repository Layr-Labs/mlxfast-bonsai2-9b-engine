// Copyright © 2026 Eigen Labs.
//
// The CBv2 block-drafter adapter over `DFlash2DraftModel`, bound to a Qwen 3.5
// / Ternary Bonsai 2 target.
//
// `Qwen35InlineMTPAssistant` is the CHAIN drafter for this target: the engine
// asks it for one token at a time and threads the previous step's hidden state
// back in. DFlash 2 is a BLOCK drafter: one forward over `1 + depth` positions
// returns the whole block, and its context is the TARGET's hidden state at five
// named layers rather than anything the drafter itself produced. This file is
// the seam between those two facts and `CBv2MTPBlockDrafter`.
//
// WHERE THE CONTEXT COMES FROM. The engine hands the drafter every position the
// target has COMMITTED, through the two seams a request-stateful drafter
// already has:
//
//   * `observeCommittedTarget` — the prompt's positions, chunk by chunk, and
//     every plain decode position;
//   * `finalizeRound` — a verified round's CONFIRMED columns, which is the
//     reference's `hidden = hidden[:, :accepted + 1, :]`.
//
// Those rows queue here until the next `proposeBlock` feeds them to the block,
// which is where they enter the drafter's own cache as projected keys and
// values. Nothing else reaches the drafter: the block itself carries no state
// between rounds.

import Foundation
import MLX
import MLXLMCommon

public enum Qwen35DFlash2Error: LocalizedError, Sendable, Equatable {
    case invalidTarget(String)
    case targetLayerCountMismatch(drafter: Int, target: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidTarget(let type):
            return "The target \(type) is not a Qwen 3.5 text model."
        case .targetLayerCountMismatch(let drafter, let target):
            return
                "The DFlash 2 drafter was trained against \(drafter) target layers; "
                + "this target has \(target)."
        }
    }
}

/// A DFlash 2 drafter bound to one Qwen 3.5 target, as the engine sees it.
public final class Qwen35DFlash2Assistant: CBv2MTPBlockDrafter, @unchecked Sendable {

    public let drafter: DFlash2DraftModel
    private let target: Qwen35TextModel

    /// How many context rows the drafter can hold. A drafter whose layers all
    /// slide keeps `sliding_window - 1` of them, so older rows are dropped
    /// here rather than carried to a forward that would drop them anyway.
    private let contextRowLimit: Int?

    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }
    public var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }

    /// One drafter forward per round, over a rectangular `[1, 1 + k]` target
    /// window. Serial per-column verification would add k target forwards to a
    /// round whose whole point is that there is only one.
    public var requiredVerificationMode: CBv2MTPVerificationMode? { .rectangular }

    /// The engine's block ceiling. The drafter was trained at `block_size`,
    /// and the reference lets a caller ask for a larger block: the extra
    /// positions are legal and simply accept less. Capping at the trained
    /// depth would draft 7 while the echo named the depth that was asked for.
    public var maximumDraftTokens: Int? { CBv2MTPConfig.testedMaxBlockDraftTokens }

    /// Single stream. The block drafter keeps one context cache per request
    /// and the track measures one stream, so a wider rectangle is untested
    /// rather than declared.
    public var maximumSpeculativeBatch: Int? { 1 }

    /// Greedy only. The reference's rejection-sampling path is not ported, so
    /// this drafter never lifts the engine's `temperature == 0` gate.
    public var supportsTargetPrefixAcceptance: Bool { false }

    private init(drafter: DFlash2DraftModel, target: Qwen35TextModel) {
        self.drafter = drafter
        self.target = target
        self.contextRowLimit = drafter.contextRowLimit
    }

    // MARK: - Loading

    /// True when `directory` holds a DFlash 2 drafter. Read from the config's
    /// own declaration, never from the directory's name.
    public static func isDrafterDirectory(_ directory: URL) -> Bool {
        DFlash2DraftModel.isDFlash2Directory(directory)
    }

    /// Load the drafter and bind it to the target's tower.
    ///
    /// `bind` checks the vocabulary and the hidden size; the target's layer
    /// count is checked here, because a drafter trained against a different
    /// tower would read five layers that exist but mean something else, and
    /// the only symptom would be a poor accept rate.
    public static func load(
        from directory: URL, target: any LanguageModel
    ) throws -> Qwen35DFlash2Assistant {
        let text = try qwen35TextTarget(target)
        let drafter = try DFlash2DraftModel.load(from: directory)
        guard drafter.config.numTargetLayers == text.configuration.hiddenLayers else {
            throw Qwen35DFlash2Error.targetLayerCountMismatch(
                drafter: drafter.config.numTargetLayers,
                target: text.configuration.hiddenLayers)
        }
        try drafter.bind(target: text)
        return Qwen35DFlash2Assistant(drafter: drafter, target: text)
    }

    private static func qwen35TextTarget(
        _ target: any LanguageModel
    ) throws -> Qwen35TextModel {
        if let target = target as? Qwen35TextModel { return target }
        if let target = target as? Qwen35Model { return target.languageModel }
        throw Qwen35DFlash2Error.invalidTarget(String(describing: type(of: target)))
    }

    // MARK: - The context tap

    /// The tap is armed by the engine build that speculates with this drafter
    /// and disarmed by one that does not, so a serial leg in a process that
    /// also served a DFlash leg pays nothing for a resident drafter.
    public func setBlockContextArmed(_ armed: Bool) throws {
        if armed {
            try target.armDFlash2Tap(layerIds: drafter.config.targetLayerIds)
        } else {
            target.dFlash2TapLayerIds = nil
        }
    }

    public func blockContextHidden() -> MLXArray? {
        target.dFlash2TappedHidden
    }

    // MARK: - Request state

    /// One request's drafter cache and the committed context rows it has not
    /// absorbed yet.
    final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]
        var promptDraft: [Int32]?
        var preparedLookup: (history: [Int], depth: Int)?
        /// Committed context rows the next block will consume, oldest first.
        var pending: [MLXArray] = []
        var pendingRows = 0
        /// Absolute target position after the newest observed row; equal to
        /// the number of committed positions this state has ever seen.
        var observedRows = 0
        /// The caches start at the first pending row's position, which is not
        /// zero when the prompt was longer than the context window.
        var cacheSeeded = false
        /// Every pending row was absorbed ahead of the first block, so that
        /// block runs with no new context rows.
        var contextAbsorbed = false
        /// Lazy proposals retained until the engine's finalize fence.
        var roots: [MLXArray] = []
        var prestaged: (anchor: Int, depth: Int, tokens: MLXArray)?
        /// The in-flight verify's context projection (`projectRoundContext`),
        /// and how many of its columns the round committed (0 until then).
        var projection: DFlash2ContextProjection?
        var projectionRows = 0
        /// The finalized round's raw accepted count when it confirmed exactly
        /// `accepted + 1` columns (`recordValidatedAcceptance`): the
        /// projection's layer-0 front holds that anchor and serves only a
        /// block over that many plus one rows.
        var frontAccepted: Int?
        var lookupFC: (output: MLXArray, rows: Int)?
        /// A lookup round's absorption of the pending rows, not yet built:
        /// the engine submits it behind that round's verify
        /// (`submitDeferredContext`); every other entry point runs it first.
        var absorbDeferred = false
        /// The newest proposal came from the committed-history lookup.
        var lastProposalLookup = false
        var isReleased = false

        init(caches: [any KVCache]) { self.caches = caches }

        /// Absolute position of the first pending row.
        var firstPendingPosition: Int { observedRows - pendingRows }

        /// Every committed position this drafter knows about. Nothing here is
        /// ever speculative: a block absorbs only rows the target committed.
        var committedInputCount: Int { observedRows }
        var stagedInputCount: Int { 0 }

        var materializedBytes: Int {
            var arrays = caches.flatMap { $0.innerState() } + pending + roots
            if let retainedFC = lookupFC?.output ?? projection?.lookupFCOutput { arrays.append(retainedFC) }
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        func append(_ rows: MLXArray, limit: Int?) {
            let count = rows.dim(1)
            guard count > 0 else { return }
            lookupFC = nil
            pending.append(rows)
            pendingRows += count
            observedRows += count
            guard let limit, pendingRows > limit else { return }
            // Drop whole leading chunks, then the head of the next one. The
            // caches have not seen these rows, so dropping them only moves
            // where the block believes it sits.
            while let first = pending.first, pendingRows - first.dim(1) >= limit {
                pending.removeFirst()
                pendingRows -= first.dim(1)
            }
            if pendingRows > limit, let first = pending.first {
                let drop = pendingRows - limit
                pending[0] = first[0..., drop..., 0...]
                pendingRows -= drop
            }
        }

        func absorbPending() {
            lookupFC = nil
            pending.removeAll(keepingCapacity: true)
            pendingRows = 0
        }

        func clearAll() {
            promptDraft = nil
            preparedLookup = nil
            projection = nil
            projectionRows = 0
            frontAccepted = nil
            lookupFC = nil
            absorbDeferred = false
            lastProposalLookup = false
            caches.removeAll(keepingCapacity: false)
            pending.removeAll(keepingCapacity: false)
            roots.removeAll(keepingCapacity: false)
            prestaged = nil
            pendingRows = 0
            isReleased = true
        }
    }

    private func state(_ value: any CBv2MTPRequestState) -> RequestState {
        guard let state = value as? RequestState else {
            preconditionFailure("DFlash 2 request state of the wrong type")
        }
        return state
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        do {
            return RequestState(caches: try drafter.makeCache())
        } catch {
            preconditionFailure("DFlash 2 drafter cache construction failed: \(error)")
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        state(requestState).clearAll()
    }

    // MARK: - Committed history

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        append(observation.hidden, to: requestState)
    }

    /// THE DTYPE CROSSING. The Bonsai trunk promotes activations to FP32 after
    /// its FP32 norms; the drafter is BF16. The cast happens HERE rather than
    /// at the drafter's `fc` input: it is the same one cast per context row,
    /// and it additionally DETACHES the row from the whole prefill chunk it
    /// was sliced out of, so a 2047-row window retains 2047 rows and not the
    /// prompt. `DFlash2DraftModel.hiddenStates` casts again and that stays a
    /// no-op for an already-BF16 tensor.
    private func append(_ hidden: MLXArray, to requestState: any CBv2MTPRequestState) {
        let state = self.state(requestState)
        runDeferredAbsorption(state)
        guard hidden.dim(1) > 0 else { return }
        state.append(hidden.asType(drafter.dtype), limit: contextRowLimit)
    }

    // MARK: - Proposing

    public func prepareDraft(
        history: [Int], depth: Int, requestState: any CBv2MTPRequestState
    ) {
        let state = self.state(requestState)
        precondition(!state.isReleased)
        if let prepared = state.preparedLookup,
            prepared.depth == depth, prepared.history == history
        {
            return
        }
        state.promptDraft = DFlash2PromptLookup.continuation(history: history, depth: depth)
        state.preparedLookup = (history, depth)
    }

    public func proposeBlock(
        anchor: Int, depth: Int, requestState: any CBv2MTPRequestState
    ) throws -> MLXArray {
        let state = self.state(requestState)
        // Prestage restores this key until the engine consumes its proposal.
        state.preparedLookup = nil
        if let p = state.prestaged {
            state.prestaged = nil
            if p.anchor == anchor && p.depth == depth {
                return p.tokens
            }
        }
        runDeferredAbsorption(state)
        state.lastProposalLookup = false
        guard !state.pending.isEmpty || state.contextAbsorbed else {
            throw DFlash2Error.emptyBlockContext
        }
        // The round's own committed columns, projected while its verify ran.
        // Taken here so no later block can read a stale projection.
        let projection =
            state.projectionRows > 0 && state.pending.count == 1
                && state.pendingRows == state.projectionRows
            ? state.projection : nil
        let frontAccepted = state.frontAccepted
        state.projection = nil
        state.projectionRows = 0
        state.frontAccepted = nil
        if let projection, frontAccepted.map({ $0 + 1 }) != state.pendingRows {
            projection.front = nil
        }
        if let ids = state.promptDraft, ids.count == depth {
            if let output = projection?.lookupFCOutput {
                state.lookupFC = (output: output, rows: state.pendingRows)
            }
            // A lookup skips the block, but still advances committed context KV.
            // Nothing in this round reads that KV, so the absorption is built
            // after the round's verify (`submitDeferredContext`): the same
            // rows through the same operations into the same cache, off the
            // path from the readback to the verify.
            if Self.deferLookupAbsorption, absorptionWillTakeAllPending(state) {
                state.absorbDeferred = true
                state.lastProposalLookup = true
                let tokens = MLXArray(ids).reshaped([1, depth])
                state.roots.append(tokens)
                return tokens
            }
            _ = absorbPendingContext(requestState: requestState)
            if state.pending.isEmpty, state.contextAbsorbed {
                state.lastProposalLookup = true
                let tokens = MLXArray(ids).reshaped([1, depth])
                state.roots.append(tokens)
                return tokens
            }
        }
        let context =
            state.pending.isEmpty
            ? MLXArray.zeros([1, 0, drafter.config.targetHiddenSize], dtype: drafter.dtype)
            : state.pending.count == 1
            ? state.pending[0] : concatenated(state.pending, axis: 1)
        if !state.cacheSeeded {
            // The cache must sit where the retained context actually starts.
            // This is the reference's `cache.offset = prompt.size - rows`: a
            // prompt longer than the context window leaves the first retained
            // row at a positive position, and the block's rotations follow it.
            for cache in state.caches {
                guard let base = cache as? BaseKVCache else { continue }
                base.offset = state.firstPendingPosition
            }
            state.cacheSeeded = true
        }
        let tokens = try drafter.propose(
            anchor: [anchor], targetHidden: context, cache: state.caches,
            blockSize: depth + 1, projection: projection)
        state.absorbPending()
        state.contextAbsorbed = false
        state.roots.append(tokens)
        return tokens
    }

    public func prestageBlock(
        anchor: Int, depth: Int, requestState: any CBv2MTPRequestState
    ) {
        let state = self.state(requestState)
        guard state.prestaged == nil else { return }
        let preparedLookup = state.preparedLookup
        guard let tokens = try? proposeBlock(anchor: anchor, depth: depth, requestState: requestState)
        else { return }
        state.prestaged = (anchor: anchor, depth: depth, tokens: tokens)
        state.preparedLookup = preparedLookup
        asyncEval(evaluationTargets(for: requestState))
    }

    /// Absorb every pending committed row into the drafter's caches now,
    /// without proposing. The next `proposeBlock` then projects only the rows
    /// committed after this call — the same incremental continuation every
    /// steady-state round already performs. Called by the engine at the end
    /// of the seed prefill so the prompt-context projection runs inside
    /// `free_decode_begin`'s work instead of the first timed decode round.
    /// Returns the cache arrays the caller must evaluate within that step; an
    /// empty pending set returns nothing and changes nothing.
    public func absorbPendingContext(requestState: any CBv2MTPRequestState) -> [MLXArray] {
        let state = self.state(requestState)
        state.absorbDeferred = false
        let lookupFC = state.lookupFC
        state.lookupFC = nil
        // EVERY pending row is absorbed, in the one pass the first block would
        // have run over them: the same rows through the same projections, so
        // the caches hold the bits the lazy path would have put there, and
        // the first block then runs with no new context rows.
        guard state.pendingRows > 0 else { return [] }
        // A prompt at or past the sliding window keeps the parent's lazy
        // single-pass absorption.
        if let window = drafter.config.slidingWindow {
            guard state.pendingRows < window else { return [] }
        }
        let context =
            state.pending.count == 1
            ? state.pending[0] : concatenated(state.pending, axis: 1)
        if !state.cacheSeeded {
            // Identical seeding to `proposeBlock`: the cache sits where the
            // retained context actually starts.
            for cache in state.caches {
                guard let base = cache as? BaseKVCache else { continue }
                base.offset = state.firstPendingPosition
            }
            state.cacheSeeded = true
        }
        let fcPrefix = lookupFC.flatMap { retained -> MLXArray? in
            guard retained.rows == state.pendingRows, state.pending.count == 1,
                retained.output.ndim == 3, retained.output.dim(1) >= retained.rows
            else { return nil }
            return retained.output[0..., 0 ..< retained.rows, 0...]
        }
        do {
            try drafter.absorbContext(targetHidden: context, cache: state.caches, lookupFC: fcPrefix)
        } catch {
            // Fail open: the rows stay pending and the next proposal absorbs
            // them lazily, exactly the parent behavior.
            return []
        }
        state.absorbPending()
        state.contextAbsorbed = true
        let evalTargets = state.caches.flatMap { $0.innerState() }
        return evalTargets
    }

    public func trimBlockState(
        _ requestState: any CBv2MTPRequestState, toCommittedLength committed: Int
    ) {
        drafter.trimCache(state(requestState).caches, toCommittedLength: committed)
    }

    /// Queue the next block's acceptance-independent work behind this
    /// round's verify: every verify column's context projection and every
    /// candidate anchor's embedding (`DFlash2DraftModel.projectContext`). The
    /// GPU runs it while the host reads the verify back and accepts, and the
    /// next proposal reads the committed prefix of it.
    public func projectRoundContext(
        blockContext: MLXArray, targetTokens: MLXArray, draftTokens: MLXArray,
        requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        let state = self.state(requestState)
        state.projection = nil
        state.projectionRows = 0
        state.frontAccepted = nil
        // Lookup rounds come in runs, and the next lookup round would only
        // queue unused work in front of its verify.
        guard !state.lastProposalLookup, blockContext.ndim == 3, blockContext.dim(0) == 1,
            let projection = drafter.projectContext(
                targetHidden: blockContext.asType(drafter.dtype), anchors: targetTokens)
        else { return [] }
        state.projection = projection
        var arrays = projection.evaluationTargets
        if let offset = frontCacheOffset(state),
            let front = drafter.projectLayerZeroFront(
                projection, draftTokens: draftTokens, targetTokens: targetTokens,
                cacheOffset: offset)
        {
            projection.front = front
            arrays.append(contentsOf: front.evaluationTargets)
        }
        return arrays
    }

    /// The offset the next proposal's caches start from, when nothing can
    /// move them before it: seeded, no absorption or pending rows waiting,
    /// one shared offset, and the proposal's context and block inside the
    /// sliding window.
    private func frontCacheOffset(_ state: RequestState) -> Int? {
        guard DF2Fast.frontEnabled, DF2Fast.frontMatched, state.cacheSeeded,
            !state.absorbDeferred, state.pending.isEmpty, let first = state.caches.first,
            state.caches.allSatisfy({ $0.offset == first.offset })
        else { return nil }
        let offset: Int = first.offset
        if let window = drafter.config.slidingWindow, offset + 16 + 16 >= window {
            return nil
        }
        return offset
    }

    public func recordValidatedAcceptance(
        _ accepted: Int, requestState: any CBv2MTPRequestState
    ) {
        state(requestState).frontAccepted = accepted
    }

    /// Build and return a lookup round's deferred absorption (see
    /// `proposeBlock`); the engine evaluates it right behind the round's verify.
    public func submitDeferredContext(
        requestState: any CBv2MTPRequestState, after: MLXArray
    ) -> [MLXArray] {
        let state = self.state(requestState)
        guard state.absorbDeferred, after.size > 0 else { return [] }
        // An exact select that reads `after` puts the absorption behind it on
        // the GPU: an encoder waits only for the arrays it reads, and the
        // pending rows are long finished, so without it the absorption would
        // run beside the verify and slow it down.
        let gate = after[0 ..< 1] .== after[0 ..< 1]
        state.pending = state.pending.map { which(gate, $0, $0) }
        if let retained = state.lookupFC {
            state.lookupFC = (output: which(gate, retained.output, retained.output), rows: retained.rows)
        }
        return absorbPendingContext(requestState: requestState)
    }

    /// `DARKBLOOM_DFLASH_DEFER_LOOKUP_ABSORB=0` absorbs a lookup round's
    /// context before its verify, as the reference order does.
    static let deferLookupAbsorption: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH_DEFER_LOOKUP_ABSORB"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `absorbPendingContext` would take every pending row: the cases where it
    /// returns without absorbing (nothing pending, a window-sized backlog) are
    /// left to the immediate path.
    private func absorptionWillTakeAllPending(_ state: RequestState) -> Bool {
        guard state.pendingRows > 0,
            state.pending.allSatisfy({ $0.dim(-1) == drafter.config.targetHiddenSize })
        else { return false }
        if let window = drafter.config.slidingWindow {
            return state.pendingRows < window
        }
        return true
    }

    /// A deferred absorption that the engine has not submitted yet runs
    /// before anything else reads or extends the pending rows or the caches.
    private func runDeferredAbsorption(_ state: RequestState) {
        guard state.absorbDeferred else { return }
        let arrays = absorbPendingContext(requestState: state)
        if !arrays.isEmpty { asyncEval(arrays) }
    }

    // MARK: - Round lifecycle

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        let state = self.state(requestState)
        return state.caches.flatMap { $0.innerState() } + state.roots
    }

    /// A round's confirmed columns become the next block's context.
    ///
    /// `confirmedInputTokens` and `committedDraftTokens` are the CHAIN seam's
    /// accounting: they describe which speculative head inputs survived. A
    /// block drafter stages nothing speculative — every row it absorbed was
    /// already committed target history — so it reads neither, and the
    /// confirmed hidden rows are the whole of what it needs.
    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        let state = self.state(requestState)
        runDeferredAbsorption(state)
        state.promptDraft = nil
        state.preparedLookup = nil
        state.roots.removeAll(keepingCapacity: true)
        state.prestaged = nil
        // Armed by this round's `projectRoundContext` (no rows committed yet). A draft tree's
        // accepted path keeps the projection of its own columns.
        let treePath = CBv2BranchVerify.committedRows
        let projection = (state.projectionRows == 0 ? state.projection : nil).map { p in
            treePath.map { p.gathered($0) } ?? p
        }
        let pendingWasEmpty = state.pending.isEmpty
        state.projection = nil
        state.projectionRows = 0
        state.frontAccepted = nil
        append(committedTargetHidden, to: requestState)
        // The committed rows are the verify's leading columns, the ones the
        // projection covers; it stands in for them only if nothing else is
        // pending.
        let rows = committedTargetHidden.dim(1)
        if let projection, pendingWasEmpty, rows >= 1, rows <= projection.rows,
            state.pending.count == 1, state.pendingRows == rows
        {
            state.projection = projection
            state.projectionRows = rows
        }
    }

    /// Nothing to undo: the round wrote no speculative state here. The rows
    /// the proposal absorbed were committed before it ran, and the proposal
    /// itself is discarded with its graph.
    public func discardRound(requestState: any CBv2MTPRequestState) {
        let state = self.state(requestState)
        runDeferredAbsorption(state)
        state.promptDraft = nil
        state.preparedLookup = nil
        state.projection = nil
        state.projectionRows = 0
        state.frontAccepted = nil
        state.lookupFC = nil
        state.roots.removeAll(keepingCapacity: true)
        state.prestaged = nil
    }
}

// MARK: - Committed-history lookup

/// A longest suffix match must have one full, unambiguous committed continuation.
enum DFlash2PromptLookup {
    /// Set DARKBLOOM_DFLASH_PROMPT_LOOKUP=0 to use the neural drafter throughout.
    private static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH_PROMPT_LOOKUP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static func continuation(history: [Int], depth: Int) -> [Int32]? {
        let count = history.count
        guard enabled, depth >= 1, count >= 16 + depth else { return nil }
        let longest = min(64, count - depth)
        var best = 15
        var chosenEnd: Int?
        var ambiguous = false
        for end in 16 ... (count - depth) {
            guard history[end - 1] == history[count - 1] else { continue }
            var length = 1
            let limit = min(longest, end)
            while length < limit && history[end - length - 1] == history[count - length - 1] {
                length += 1
            }
            guard length >= 16, length >= best else { continue }
            if length > best {
                best = length
                chosenEnd = end
                ambiguous = false
            } else if let chosenEnd {
                for offset in 0 ..< depth where history[chosenEnd + offset] != history[end + offset] {
                    ambiguous = true
                    break
                }
            }
        }
        guard let chosenEnd, !ambiguous else { return nil }
        return history[chosenEnd ..< chosenEnd + depth].map(Int32.init)
    }
}
