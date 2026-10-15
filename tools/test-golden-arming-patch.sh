#!/usr/bin/env bash
# Unit test for tools/golden-arming-patch.py. Offline and hermetic: synthetic
# golden files in a temporary directory, a copy of the contract fixture for the
# --apply case, no R2, no box, no real golden.
#
# The synthetic set has the shape the per-depth rule must handle, over three
# live goldens: alpha's mtp1 is byte-identical to its serial tape and mtp2..6
# are identical to each other; bravo's mtp1..6 are identical to each other and
# differ from its tape; charlie's mtp1..6 all equal its tape. So 18 keys need
# 2 new objects, and 3 tapes + 2 oracles = 5 distinct uploads. The two public captures are not uploads: they ship in git.
#
# Usage: tools/test-golden-arming-patch.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOL="${SCRIPT_DIR}/tools/golden-arming-patch.py"
CONTRACT="${SCRIPT_DIR}/fixtures/bonsai2_9b_mlx_v1_track.json"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $1" >&2; failures=$((failures + 1)); }
pass() { echo "ok: $1"; }

# The marker is in every synthetic file. It must never reach the tool's output.
MARKER="SYNTHETIC-CONTENT-MARKER-7f3a"
golden() { printf '{"version":1,"marker":"%s","case":"%s"}\n' "${MARKER}" "$2" > "$1"; }

make_set() { # make_set DIR
  mkdir -p "$1"
  local n
  for n in alpha bravo charlie; do
    golden "$1/${n}.golden.json" "${n}"
  done
  cp "$1/alpha.golden.json" "$1/alpha.mtp1.golden.json"
  for n in 2 3 4 5 6; do golden "$1/alpha.mtp${n}.golden.json" "speculative-alpha"; done
  for n in 1 2 3 4 5 6; do golden "$1/bravo.mtp${n}.golden.json" "speculative-bravo"; done
  for n in 1 2 3 4 5 6; do cp "$1/charlie.golden.json" "$1/charlie.mtp${n}.golden.json"; done
  golden "$1/public-local-iterate.golden.json" "public-short"
  golden "$1/public-local-submit.golden.json" "public-long"
}

make_set "${WORK}/set"
q() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$@"; }

# --- case 1: the patch arms every key, and identical bytes share one object --
LIVES=(--live alpha --live bravo --live charlie)
if python3 "${TOOL}" --dir "${WORK}/set" "${LIVES[@]}" > "${WORK}/patch.json" 2> "${WORK}/case1.err"; then
  P="${WORK}/patch.json"
  pre="correctness_prompts/bonsai2-9b-mlx-v1"
  ok1=1
  [[ "$(q "${P}" 'd["official_scoring_enabled"]')" == "True" ]] || { fail "case 1: official_scoring_enabled is not true"; ok1=0; }
  [[ "$(q "${P}" 'len(d["timed_prompt_pool"])')" == "3" ]] || { fail "case 1: the pool is not 3 tapes"; ok1=0; }
  [[ "$(q "${P}" 'd["live_goldens"]')" == "['alpha', 'bravo', 'charlie']" ]] || { fail "case 1: live_goldens is not alpha, bravo, charlie"; ok1=0; }
  [[ "$(q "${P}" '[e for e in d["timed_prompt_pool"] if e["r2_path"].endswith("/alpha.golden.json")][0]["sha256"] == d["hidden_correctness_golden"]["sha256"]')" == "True" ]] \
    || { fail "case 1: the hidden oracle is not the live golden's own pin"; ok1=0; }
  [[ "$(q "${P}" 'sorted(d["live_golden_speculative"])')" == "['mtp1', 'mtp2', 'mtp3', 'mtp4', 'mtp5', 'mtp6']" ]] \
    || { fail "case 1: live_golden_speculative does not carry the 6 depths"; ok1=0; }
  [[ "$(q "${P}" 'all(sorted(v) == ["alpha", "bravo", "charlie"] for v in d["live_golden_speculative"].values())')" == "True" ]] \
    || { fail "case 1: a depth does not carry one pin per live golden"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["mtp1"]["alpha"]["r2_path"]')" == "${pre}/alpha.golden.json" ]] \
    || { fail "case 1: alpha mtp1 (serial-identical) does not point at the live golden"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["mtp6"]["alpha"]["r2_path"]')" == "${pre}/alpha.mtp2.golden.json" ]] \
    || { fail "case 1: alpha mtp6 does not share the mtp2 object"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["mtp6"]["bravo"]["r2_path"]')" == "${pre}/bravo.mtp1.golden.json" ]] \
    || { fail "case 1: bravo mtp6 does not share the mtp1 object"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["mtp4"]["charlie"]["r2_path"]')" == "${pre}/charlie.golden.json" ]] \
    || { fail "case 1: charlie mtp4 (serial-identical) does not point at its live golden"; ok1=0; }
  [[ "$(q "${P}" '"public_captures" in d')" == "False" ]] \
    || { fail "case 1: the patch carries public_captures; the captures ship in git, not in the fixture"; ok1=0; }
  grep -q "ship: public-local-submit.golden.json -> ${pre}/public-local-submit.golden.json" "${WORK}/case1.err" \
    || { fail "case 1: the local_submit capture is not listed to ship"; ok1=0; }
  grep -q "5 distinct object(s); 16 of 18 per-depth key(s) share an earlier object" "${WORK}/case1.err" \
    || { fail "case 1: the upload summary is wrong ($(tail -1 "${WORK}/case1.err"))"; ok1=0; }
  if grep -q "${MARKER}" "${P}" "${WORK}/case1.err"; then
    fail "case 1: the output carries golden CONTENT"; ok1=0
  fi
  [[ "${ok1}" == "1" ]] && pass "the patch arms all keys, dedupes identical bytes, and prints no content"
else
  fail "case 1: the tool refused a complete set ($(cat "${WORK}/case1.err"))"
fi

# --- case 2: a missing per-depth oracle is refused by key --------------------
make_set "${WORK}/missing"
rm "${WORK}/missing/alpha.mtp5.golden.json"
if python3 "${TOOL}" --dir "${WORK}/missing" "${LIVES[@]}" > /dev/null 2> "${WORK}/case2.err"; then
  fail "case 2: the tool accepted a set with no mtp5 oracle"
elif grep -q "no per-depth oracle for: mtp5 (alpha)" "${WORK}/case2.err"; then
  pass "a missing per-depth oracle is refused by key"
else
  fail "case 2: refused for the wrong reason ($(cat "${WORK}/case2.err"))"
fi

# --- case 3: a public capture with a hidden tape's bytes is refused ----------
make_set "${WORK}/leak"
cp "${WORK}/leak/charlie.golden.json" "${WORK}/leak/public-local-iterate.golden.json"
if python3 "${TOOL}" --dir "${WORK}/leak" "${LIVES[@]}" > /dev/null 2> "${WORK}/case3.err"; then
  fail "case 3: the tool accepted a public capture that is a hidden tape"
elif grep -q "same bytes as a hidden tape" "${WORK}/case3.err"; then
  pass "a public capture with a hidden tape's bytes is refused"
else
  fail "case 3: refused for the wrong reason ($(cat "${WORK}/case3.err"))"
fi

# --- case 4: a stored baseline pair is refused -------------------------------
make_set "${WORK}/pair"
printf '{"version":1,"benchmark":{"baseline_decode_seconds_per_token":0.03}}\n' > "${WORK}/pair/delta.golden.json"
if python3 "${TOOL}" --dir "${WORK}/pair" "${LIVES[@]}" > /dev/null 2> "${WORK}/case4.err"; then
  fail "case 4: the tool accepted a tape that stores a baseline pair"
elif grep -q "stores a baseline pair" "${WORK}/case4.err"; then
  pass "a tape that stores a baseline pair is refused"
else
  fail "case 4: refused for the wrong reason ($(cat "${WORK}/case4.err"))"
fi

# --- case 5: --apply leaves no pending golden sentinel -----------------------
# The ranked preflight refuses any PENDING-ORGANIZER in the contract, so an
# applied patch must leave none in the fields it owns. The target and head pins
# are not this tool's: while they are pending they carry the sentinel too, so
# the copy stands in synthetic values for them first, the way a re-stamp onto
# the published pack would.
python3 - "${CONTRACT}" "${WORK}/contract.json" <<'UNPENDEOF'
import json, sys
def unpend(node):
    if isinstance(node, dict):
        return {k: unpend(v) for k, v in node.items()}
    if isinstance(node, list):
        return [unpend(v) for v in node]
    if isinstance(node, str) and "PENDING-ORGANIZER" in node:
        return "synthetic-pin"
    return node
d = json.load(open(sys.argv[1]))
golden = d.pop("hidden_correctness_golden")
d = unpend(d)
d["hidden_correctness_golden"] = golden
json.dump(d, open(sys.argv[2], "w"), indent=2)
UNPENDEOF
if python3 "${TOOL}" --dir "${WORK}/set" "${LIVES[@]}" --apply --contract "${WORK}/contract.json" 2> "${WORK}/case5.err"; then
  if grep -q "PENDING-ORGANIZER" "${WORK}/contract.json"; then
    fail "case 5: the applied contract still carries a pending sentinel"
  elif [[ "$(q "${WORK}/contract.json" 'd["track_id"]')" != "bonsai2-9b-mlx-v1" ]]; then
    fail "case 5: --apply lost a fixture key it does not own"
  else
    pass "--apply arms the contract and leaves no pending sentinel"
  fi
else
  fail "case 5: --apply refused ($(cat "${WORK}/case5.err"))"
fi

echo
if (( failures > 0 )); then
  echo "${failures} case(s) failed" >&2
  exit 1
fi
echo "test-golden-arming-patch.sh: all cases passed"
