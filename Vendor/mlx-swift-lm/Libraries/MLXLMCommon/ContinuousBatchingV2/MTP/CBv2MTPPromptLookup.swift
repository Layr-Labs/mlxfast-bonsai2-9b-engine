// CBv2MTPPromptLookup.swift
//
// ContinuousBatchingV2 — prompt-lookup draft source for MTP rounds.
//
// When a row's newest n committed tokens occurred earlier in its own history
// (prompt + committed output), the round proposes the tokens that followed
// that occurrence instead of running the drafter's chain. Target
// verification, acceptance and finalize are the drafted round's; only the
// proposal source differs, so the target still decides every emitted token.
// Engine-thread confined, one index per request.

/// Incremental n-gram index over one request's append-only host token
/// history. An n-gram enters the index once `depth` committed tokens follow
/// it, so a hit always proposes a full chain and never matches the row's own
/// current suffix.
final class CBv2MTPPromptLookupIndex {
    let ngram: Int
    let depth: Int
    /// n-gram hash -> exclusive end of its most recent indexed occurrence.
    private var ends: [UInt64: Int] = [:]
    /// Exclusive end of the newest occurrence indexed so far.
    private var indexedEnd: Int

    init(ngram: Int, depth: Int) {
        precondition(ngram > 0 && depth > 0, "CBv2 prompt lookup needs a positive n-gram and depth")
        self.ngram = ngram
        self.depth = depth
        self.indexedEnd = ngram - 1
    }

    /// Index every n-gram that now has `depth` tokens after it. Each history
    /// position is hashed once, in O(ngram); nothing indexed is revisited.
    func catchUp(_ tokens: [Int]) {
        let limit = tokens.count - depth
        guard limit > indexedEnd else { return }
        for end in (indexedEnd + 1) ... limit {
            ends[Self.hash(tokens, end: end, ngram: ngram)] = end
        }
        indexedEnd = limit
    }

    /// The `depth` tokens after the most recent earlier occurrence of the
    /// history's last `ngram` tokens, or nil.
    func draft(_ tokens: [Int]) -> [Int]? {
        catchUp(tokens)
        let count = tokens.count
        guard count >= ngram,
            let end = ends[Self.hash(tokens, end: count, ngram: ngram)],
            end + depth <= count,
            tokens[(end - ngram) ..< end].elementsEqual(tokens[(count - ngram) ..< count])
        else { return nil }
        return Array(tokens[end ..< end + depth])
    }

    /// FNV-1a over whole token ids. A collision only costs a miss: `draft`
    /// compares the tokens themselves before proposing.
    private static func hash(_ tokens: [Int], end: Int, ngram: Int) -> UInt64 {
        var value: UInt64 = 0xcbf2_9ce4_8422_2325
        for token in tokens[(end - ngram) ..< end] {
            value = (value ^ UInt64(truncatingIfNeeded: token)) &* 0x0000_0100_0000_01b3
        }
        return value
    }
}
