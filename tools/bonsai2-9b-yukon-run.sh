#!/usr/bin/env bash
#
# bonsai2-9b-yukon-run.sh -- benchmark.json's benchmarkCommand.
#
# `yukon run` executes benchmarkCommand on the participant's machine. The
# ranked script, tools/bonsai2-9b-measure-and-score.sh, needs inputs
# that only a ranked box has: the staged goldens, the reference workspace and
# the box calibration. Off a ranked box it can only refuse. This script picks
# the path:
#
#   RANKED  GITHUB_ACTIONS is "true", or one or more of
#           MLXFAST_QWEN38_GOLDEN_DIR, MLXFAST_BASELINE_WORKSPACE and
#           MLXFAST_BASELINE_CALIBRATION is set. This script then replaces
#           itself with the ranked script and passes all arguments. The ranked
#           behavior does not change. A partial ranked environment goes to the
#           ranked script, which refuses by name. It never falls back.
#   LOCAL   none of the above. This script transforms the weights when
#           weights/config.json is absent, then runs
#           ./benchmark.sh --local-submit at the depth that
#           mtp-head.manifest.json declares (tools/spec-declaration.sh).
#
# The ranked workflow (.github/workflows/benchmark.yml) calls the ranked script
# directly. It does not call this script.
#
# A LOCAL RESULT HAS NO SCORE. benchd seals a local run with "score": null,
# because a local run has no serial-control leg on this box. This script does
# not write a score of its own. Yukon reads the sealed file at scorePath.
#
# Env (LOCAL path only):
#   MLXFAST_SCORE_PATH     where benchd seals the result (benchmark.json sets
#                          score.json).
#   MLXFAST_REFERENCE_DIR  the downloaded checkpoint for the transform.
#                          Default: the reference_weights/ link that ./setup.sh
#                          writes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${GITHUB_ACTIONS:-}" == "true" \
  || -n "${MLXFAST_QWEN38_GOLDEN_DIR:-}" \
  || -n "${MLXFAST_BASELINE_WORKSPACE:-}" \
  || -n "${MLXFAST_BASELINE_CALIBRATION:-}" ]]; then
  exec "${ROOT}/tools/bonsai2-9b-measure-and-score.sh" "$@"
fi

if [[ $# -ne 0 ]]; then
  echo "bonsai2-9b-yukon-run.sh: the local path takes no arguments (got: $*)." >&2
  exit 2
fi

echo "bonsai2-9b-yukon-run.sh: no ranked inputs. This is a LOCAL run. It has no score; the ranked runner gives the score." >&2

if [[ ! -f weights/config.json ]]; then
  reference="${MLXFAST_REFERENCE_DIR:-reference_weights/bonsai2-9b-target}"
  echo "bonsai2-9b-yukon-run.sh: weights/ is absent. Transforming ${reference} into weights/." >&2
  .build/release/mlxfast-swift transform --reference "${reference}" --output weights
fi

depth="$(tools/spec-declaration.sh draft-len)"
depth_args=()
if [[ "${depth}" != "0" ]]; then
  depth_args=(--mtp-depth "${depth}")
fi
echo "bonsai2-9b-yukon-run.sh: declared MTP depth ${depth}." >&2

# The local mode defaults to the shipped long capture, so no golden variable
# is set here.
MLXFAST_ENGINE_BIN="${MLXFAST_ENGINE_BIN:-.build/release/bench-worker}" \
  exec ./benchmark.sh --local-submit ${depth_args[@]+"${depth_args[@]}"}
