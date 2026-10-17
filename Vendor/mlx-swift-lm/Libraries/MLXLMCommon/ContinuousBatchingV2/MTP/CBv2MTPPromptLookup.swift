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

import MLX

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

/// Device-side continuation of a drafted chain from the row's own history.
///
/// A drafted round's chain ends with tokens the host has not seen yet, so the
/// host index above cannot extend it. This kernel does the same kind of
/// lookup on the device: over `history ++ chain` it finds the most recent
/// earlier position whose trailing 3-gram, else 2-gram, else 1-gram equals
/// the sequence's own last tokens, and proposes the tokens that followed it
/// (periodically, when the occurrence is closer to the end than the
/// extension is long). Without any occurrence it proposes the last token
/// again. The proposals are ordinary drafts: target verification decides
/// every emitted token, so a wrong guess costs only its verify row.
enum CBv2MTPChainExtension {
    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_mtp_chain_extension",
        inputNames: ["history", "chain"], outputNames: ["extension"],
        source: """
            const int L = history_shape[0];
            const int M = chain_shape[0];
            const int T = L + M;
            const uint lane = thread_position_in_threadgroup.x;
            const uint lanes = threads_per_threadgroup.x;
            #define CBV2_AT(i) ((i) < L ? history[(i)] : chain[(i) - L])
            const int k1 = CBV2_AT(T - 1);
            const int k2 = T >= 2 ? CBV2_AT(T - 2) : -1;
            const int k3 = T >= 3 ? CBV2_AT(T - 3) : -1;
            int best = -1;
            for (int j = int(lane); j <= T - 2; j += int(lanes)) {
                if (CBV2_AT(j) != k1) { continue; }
                int level = 1;
                if (j >= 1 && CBV2_AT(j - 1) == k2) {
                    level = 2;
                    if (j >= 2 && CBV2_AT(j - 2) == k3) { level = 3; }
                }
                best = max(best, level * T + j);
            }
            threadgroup int partial[32];
            best = simd_max(best);
            if (thread_index_in_simdgroup == 0) { partial[simdgroup_index_in_threadgroup] = best; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                const uint groups = (lanes + 31) / 32;
                for (uint g = 0; g < groups; ++g) { best = max(best, partial[g]); }
                if (best < 0) {
                    for (int e = 0; e < E; ++e) { extension[e] = k1; }
                } else {
                    const int j = best % T;
                    const int period = T - 1 - j;
                    for (int e = 0; e < E; ++e) { extension[e] = CBV2_AT(j + 1 + e % period); }
                }
            }
            #undef CBV2_AT
            """)

    /// `count` proposals [count] int32 continuing `history` (host tokens,
    /// oldest first, ending with the round's seed) followed by `chain`, the
    /// round's drafted tokens [M] int32.
    static func extend(history: [Int], chain: MLXArray, count: Int) -> MLXArray {
        precondition(count > 0 && !history.isEmpty, "CBv2 chain extension needs history and a count")
        let historyArray = MLXArray(history.map { Int32(truncatingIfNeeded: $0) })
        return kernel(
            [historyArray, chain.asType(.int32).reshaped([-1])],
            template: [("E", count)],
            grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[count]], outputDTypes: [.int32])[0]
    }
}
