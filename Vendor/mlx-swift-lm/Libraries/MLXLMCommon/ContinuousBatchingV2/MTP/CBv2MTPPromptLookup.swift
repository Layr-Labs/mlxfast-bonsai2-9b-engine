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
/// current suffix. Matches hierarchically from maxNgram down to minNgram.
final class CBv2MTPPromptLookupIndex {
    let maxNgram: Int
    let minNgram: Int
    let depth: Int
    /// (n, hash) -> exclusive end of its most recent indexed occurrence.
    private var ends: [UInt64: Int] = [:]
    /// Exclusive end of the newest occurrence indexed so far.
    private var indexedEnd: Int

    var ngram: Int { maxNgram }

    init(ngram: Int, depth: Int, minNgram: Int = 4) {
        precondition(ngram > 0 && depth > 0, "CBv2 prompt lookup needs a positive n-gram and depth")
        self.maxNgram = ngram
        self.minNgram = max(3, min(minNgram, ngram))
        self.depth = depth
        self.indexedEnd = self.minNgram - 1
    }

    /// Index every n-gram (from minNgram to maxNgram) that now has `depth` tokens after it.
    func catchUp(_ tokens: [Int]) {
        let limit = tokens.count - depth
        guard limit > indexedEnd else { return }
        for end in (indexedEnd + 1) ... limit {
            for n in minNgram ... maxNgram {
                guard end >= n else { continue }
                let key = (UInt64(n) << 56) ^ Self.hash(tokens, end: end, ngram: n)
                ends[key] = end
            }
        }
        indexedEnd = limit
    }

    /// The `depth` tokens after the most recent earlier occurrence of the
    /// history's newest tokens, trying longest n-gram down to minNgram.
    func draft(_ tokens: [Int]) -> [Int]? {
        catchUp(tokens)
        let count = tokens.count
        for n in stride(from: maxNgram, through: minNgram, by: -1) {
            guard count >= n else { continue }
            let key = (UInt64(n) << 56) ^ Self.hash(tokens, end: count, ngram: n)
            if let end = ends[key],
                end + depth <= count,
                tokens[(end - n) ..< end].elementsEqual(tokens[(count - n) ..< count])
            {
                return Array(tokens[end ..< end + depth])
            }
        }
        return nil
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
