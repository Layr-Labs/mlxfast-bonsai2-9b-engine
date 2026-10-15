#!/usr/bin/env bash
# Provision the organizer-pinned MTP HEAD for track bonsai2-9b-mlx-v1.
#
# This script provisions the HEAD and nothing else. The TARGET is the Bonsai 2
# 9B pack that ./setup.sh pins, downloads and verifies against
# fixtures/reference_bonsai2_9b_2bit.sha256, so there is no second target
# download here and no way for this script to point the track at a different
# pack.
#
# THE HEAD IS THE PACK'S OWN TRAINED DRAFTER. prism-ml ships it as the mtp/
# folder of the target's repository and revision: a standalone qwen3_5_mtp
# export (config.json and one model.safetensors, 31 tensors, 4-bit affine group-64). The head pin
# names those two files, with paths relative to mtp/, and only those are
# fetched, unmodified, into their own directory. The repository is private
# until prism-ml makes it public: set HF_TOKEN, exactly as for ./setup.sh.
#
# WHY THIS IS A WRAPPER AND NOT A DOWNLOADER. setup.sh already holds the
# resumable, hash-verified, stall-detecting download logic, and it is fully
# parameterised by environment (MLXFAST_REFERENCE_MODEL_REPO / _REVISION /
# _MANIFEST_PATH / _BASE_URL / _CACHE_DIR / _DIR / _COMPAT_LINK). This script
# points those at the head and delegates. One downloader, one verification
# path, one set of stall and resume semantics.
#
# The delegated run does NOT rebuild the Swift products: it passes
# MLXFAST_SKIP_SWIFT_BUILD=1, so an existing pair is reused. Nothing about the
# products can have changed between the two commands. The tool-installing legs
# and the metallib build are skipped for the same reason in the other
# direction: ./setup.sh owns those and this script must not mutate global state
# a second time.
#
# ./setup.sh runs this script for you as its last step. Run it by hand only to
# re-verify the head, or after MLXFAST_SKIP_MTP_HEAD=1 skipped it.
#
# Usage: ./setup-mtp-head.sh
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK_ID="bonsai2-9b-mlx-v1"
HEAD_REPO="${MLXFAST_MTP_HEAD_REPO:-prism-ml/Ternary-Bonsai-2-9B-mlx-2bit}"
HEAD_REVISION="${MLXFAST_MTP_HEAD_REVISION:-f4e28a59531731c41b6dfac88cf271bd58f0a287}"
HEAD_BASE_URL="${MLXFAST_MTP_HEAD_BASE_URL:-https://huggingface.co/${HEAD_REPO}/resolve/${HEAD_REVISION}/mtp}"
HEAD_MANIFEST="${MLXFAST_MTP_HEAD_MANIFEST_PATH:-fixtures/reference_bonsai2_9b_mtp_head.sha256}"
HEAD_DIR="${MLXFAST_MTP_HEAD_DIR:-reference_weights/bonsai2-9b-mtp-head}"
[[ -f "${HEAD_MANIFEST}" ]] || {
  echo "setup-mtp-head.sh: pin file is missing: ${HEAD_MANIFEST}" >&2
  exit 1
}

echo "setup-mtp-head.sh: track ${TRACK_ID}; head ${HEAD_REPO}@${HEAD_REVISION}/mtp -> ${HEAD_DIR}"

# ON THE RANKED BOX THE HEAD COMES FROM THE BOX'S VERIFIED COPY. The runner
# service exports MLXFAST_MTP_HEAD_REFERENCE_DIR, the copy the organizer staged
# and the reference tree's control leg loads. The repository is private and the
# ranked job holds no credential, so the job copies each pinned file of that
# copy into this checkout (an APFS clone where the volume allows it), and the
# pin check below verifies the copy. The head stays a real directory of this
# tree, because the resident's sandbox reads its own tree. An explicit
# MLXFAST_MTP_HEAD_DIR names another directory and copies nothing.
if [[ -z "${MLXFAST_MTP_HEAD_DIR:-}" && -n "${MLXFAST_MTP_HEAD_REFERENCE_DIR:-}" ]]; then
  echo "setup-mtp-head.sh: copying the head from the box's copy ${MLXFAST_MTP_HEAD_REFERENCE_DIR}"
  # A link left by an earlier run would point the removals below at the box's copy.
  [[ ! -L "${HEAD_DIR}" ]] || rm -f "${HEAD_DIR}"
  while read -r _sha _bytes path; do
    [[ -n "${path:-}" && "${_sha}" != \#* ]] || continue
    mkdir -p "$(dirname "${HEAD_DIR}/${path}")"
    rm -f "${HEAD_DIR}/${path}"
    cp -c "${MLXFAST_MTP_HEAD_REFERENCE_DIR}/${path}" "${HEAD_DIR}/${path}" 2>/dev/null \
      || cp "${MLXFAST_MTP_HEAD_REFERENCE_DIR}/${path}" "${HEAD_DIR}/${path}"
  done < "${HEAD_MANIFEST}"
fi

# The head has no compatibility link of its own: nothing resolves it by a
# legacy path. Point the link at the head directory so setup.sh does not
# create or check one for the target while provisioning the head.
MLXFAST_SKIP_SWIFT_BUILD=1 \
MLXFAST_SKIP_MACMON_INSTALL=1 \
MLXFAST_SKIP_MLX_METALLIB=1 \
MLXFAST_SKIP_MTP_HEAD=1 \
MLXFAST_SKIP_DFLASH_DRAFTER=1 \
MLXFAST_REFERENCE_MODEL_REPO="${HEAD_REPO}" \
MLXFAST_REFERENCE_REVISION="${HEAD_REVISION}" \
MLXFAST_REFERENCE_BASE_URL="${HEAD_BASE_URL}" \
MLXFAST_REFERENCE_MANIFEST_PATH="${HEAD_MANIFEST}" \
MLXFAST_REFERENCE_DIR="${HEAD_DIR}" \
MLXFAST_REFERENCE_COMPAT_LINK="${HEAD_DIR}" \
MLXFAST_REFERENCE_MIN_FREE_GIB="${MLXFAST_MTP_HEAD_MIN_FREE_GIB:-2}" \
  ./setup.sh "$@"

echo "setup-mtp-head.sh: the head is staged and verified at ${HEAD_DIR}"
echo "setup-mtp-head.sh: the engine reads it with --drafter ${HEAD_DIR}"
