#!/usr/bin/env bash
# Provision the organizer-pinned DFLASH 2 DRAFTER for track bonsai2-9b-mlx-v1.
#
# This script provisions the DRAFTER and nothing else. The TARGET is the Bonsai
# 2 9B pack that ./setup.sh pins, downloads and verifies, so there is no second
# target download here.
#
# THE SOURCE IS PRISM'S GGUF. prism-ml publishes the DFlash 2 drafter for this
# model once, as Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf in
# prism-ml/Ternary-Bonsai-2-9B-gguf. The CUDA track loads that file as is. The
# MLX engine loads an MLX directory, so this script converts the pinned file
# with tools/convert-dflash-gguf.py: the Q8_0 linears become MLX affine 8-bit
# group-32 linears without loss, and the norms, the convolution base kernels
# and the selector codebooks become BF16. David sanctioned that one conversion
# (2026-10-14) as a scoped waiver of the no-model-modification rule. Nothing is
# re-quantized and nothing is uploaded.
#
# THREE PINS.
#   fixtures/reference_bonsai2_9b_dflash2_source.sha256   the GGUF (download)
#   fixtures/reference_bonsai2_9b_dflash2_drafter.sha256  the converted
#                                                         directory (staged)
#   fixtures/dflash2_9b_drafter_config.json               its config.json
# The converted directory is staged only when every file matches its pin, so a
# converter or a source that changes is a refusal, never a different drafter.
#
# WHERE THE DIRECTORY COMES FROM.
#   * The ranked box: the runner service exports
#     MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR, the copy the organizer staged. The
#     job holds no credential, so it copies each pinned file of that copy into
#     this checkout (an APFS clone where the volume allows it) and verifies it.
#   * Everywhere else: the GGUF is downloaded (HF_TOKEN, else the token file,
#     exactly as for ./setup.sh) into MLXFAST_DFLASH_DRAFTER_SOURCE_DIR,
#     checked against its pin, and converted into the directory.
# An explicit MLXFAST_DFLASH_DRAFTER_DIR names another directory and copies
# nothing from the box's copy.
#
# ./setup.sh runs this script after the MTP head when the tree declares the
# dflash decoder (mtp-head.manifest.json spec.decoder "dflash") or when a
# drafter directory is named. Run it by hand after you change the declaration
# to dflash.
#
# DISK. The GGUF is 1.57 GB and the directory is 1.93 GB.
#
# Usage: ./setup-dflash-drafter.sh
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK_ID="bonsai2-9b-mlx-v1"
SOURCE_REPO="${MLXFAST_DFLASH_DRAFTER_REPO:-prism-ml/Ternary-Bonsai-2-9B-gguf}"
SOURCE_REVISION="${MLXFAST_DFLASH_DRAFTER_REVISION:-608e89c34258328a51061ebd313c924908747044}"
SOURCE_MANIFEST="fixtures/reference_bonsai2_9b_dflash2_source.sha256"
DRAFTER_MANIFEST="fixtures/reference_bonsai2_9b_dflash2_drafter.sha256"
DRAFTER_CONFIG="fixtures/dflash2_9b_drafter_config.json"
DRAFTER_DIR="${MLXFAST_DFLASH_DRAFTER_DIR:-reference_weights/bonsai2-9b-dflash2-drafter}"
SOURCE_DIR="${MLXFAST_DFLASH_DRAFTER_SOURCE_DIR:-reference_weights/bonsai2-9b-dflash2-gguf}"
CONVERTER="tools/convert-dflash-gguf.py"

log() { echo "setup-dflash-drafter.sh: $*"; }
die() { echo "setup-dflash-drafter.sh: $*" >&2; exit 1; }

for f in "${SOURCE_MANIFEST}" "${DRAFTER_MANIFEST}" "${DRAFTER_CONFIG}" "${CONVERTER}"; do
  [[ -f "${f}" ]] || die "pinned input is missing: ${f}"
done
command -v python3 >/dev/null 2>&1 || die "python3 is required (it runs the converter)"

# pin_rows <manifest>: "<sha256> <bytes> <path>" for every pinned file.
pin_rows() { awk '$0 !~ /^#/ && NF == 3 { print $1, $2, $3 }' "$1"; }

# drafter_ok: every pinned file of the directory matches its pin, and its
# config.json is the fixture's config byte for byte.
drafter_ok() {
  local sha bytes rel f
  while read -r sha bytes rel; do
    f="${DRAFTER_DIR}/${rel}"
    [[ -f "${f}" && ! -L "${f}" ]] || return 1
    [[ "$(wc -c < "${f}" | tr -d ' ')" == "${bytes}" ]] || return 1
    [[ "$(shasum -a 256 "${f}" | cut -d' ' -f1)" == "${sha}" ]] || return 1
  done < <(pin_rows "${DRAFTER_MANIFEST}")
  cmp -s "${DRAFTER_DIR}/config.json" "${DRAFTER_CONFIG}"
}

log "track ${TRACK_ID}; drafter source ${SOURCE_REPO}@${SOURCE_REVISION} -> ${DRAFTER_DIR}"

if [[ -z "${MLXFAST_DFLASH_DRAFTER_DIR:-}" && -n "${MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR:-}" ]]; then
  log "copying the drafter from the box's copy ${MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR}"
  # A link left by an earlier run would point the removals below at the box's copy.
  [[ ! -L "${DRAFTER_DIR}" ]] || rm -f "${DRAFTER_DIR}"
  mkdir -p "${DRAFTER_DIR}"
  while read -r _sha _bytes rel; do
    rm -f "${DRAFTER_DIR}/${rel}"
    cp -c "${MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR}/${rel}" "${DRAFTER_DIR}/${rel}" 2>/dev/null \
      || cp "${MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR}/${rel}" "${DRAFTER_DIR}/${rel}"
  done < <(pin_rows "${DRAFTER_MANIFEST}")
  drafter_ok || die "the box's copy ${MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR} does not match ${DRAFTER_MANIFEST}"
  log "the drafter is staged and verified at ${DRAFTER_DIR}"
  exit 0
fi

if drafter_ok; then
  log "the drafter at ${DRAFTER_DIR} already matches its pins; nothing to do"
  exit 0
fi

# The GGUF. setup.sh's downloader verifies MLX checkpoints (config.json and
# safetensors shards), so this one file is fetched here: resumable, and
# accepted only when its bytes and sha256 match the source pin. The token goes
# to curl on stdin, never on an argv.
read -r source_sha source_bytes source_rel < <(pin_rows "${SOURCE_MANIFEST}")
source_path="${SOURCE_DIR}/${source_rel}"
source_ok() {
  [[ -f "${source_path}" && "$(wc -c < "${source_path}" | tr -d ' ')" == "${source_bytes}" ]] \
    && [[ "$(shasum -a 256 "${source_path}" | cut -d' ' -f1)" == "${source_sha}" ]]
}
if ! source_ok; then
  token="${HF_TOKEN:-}"
  token_file="${HF_TOKEN_PATH:-${HOME:-}/.cache/huggingface/token}"
  if [[ -z "${token}" && -f "${token_file}" ]]; then IFS= read -r token < "${token_file}" || true; fi
  url="${MLXFAST_DFLASH_DRAFTER_BASE_URL:-https://huggingface.co/${SOURCE_REPO}/resolve/${SOURCE_REVISION}}/${source_rel}"
  log "downloading ${source_rel} (${source_bytes} bytes) from ${SOURCE_REPO}@${SOURCE_REVISION}"
  mkdir -p "${SOURCE_DIR}"
  rm -f "${source_path}"
  { [[ -z "${token}" ]] || printf 'header = "Authorization: Bearer %s"\n' "${token}"; } \
    | curl -fsSL --retry 10 --retry-all-errors --retry-delay 2 -C - \
        --speed-limit 1048576 --speed-time 60 -K - -o "${source_path}.part" "${url}" \
    || die "the download of ${source_rel} failed (the repository is private until prism-ml makes it public: set HF_TOKEN)"
  mv -f "${source_path}.part" "${source_path}"
  source_ok || die "the downloaded ${source_path} does not match ${SOURCE_MANIFEST}"
fi

log "converting ${SOURCE_DIR}/${source_rel} into ${DRAFTER_DIR}"
[[ ! -L "${DRAFTER_DIR}" ]] || rm -f "${DRAFTER_DIR}"
mkdir -p "${DRAFTER_DIR}"
python3 -I "${CONVERTER}" --gguf "${SOURCE_DIR}/${source_rel}" \
  --expect-source-sha256 "${source_sha}" --config "${DRAFTER_CONFIG}" --out-dir "${DRAFTER_DIR}"
cp "${DRAFTER_CONFIG}" "${DRAFTER_DIR}/config.json"
drafter_ok || die "the converted drafter at ${DRAFTER_DIR} does not match ${DRAFTER_MANIFEST}; refusing it"

log "the drafter is staged and verified at ${DRAFTER_DIR}"
log "the engine reads it with --drafter ${DRAFTER_DIR}"
