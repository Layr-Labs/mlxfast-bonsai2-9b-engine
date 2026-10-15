#!/usr/bin/env bash
# test-setup-hf-token.sh -- setup.sh's Hugging Face token: HF_TOKEN, else the
# first line of ${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}, trimmed.
#
# Hermetic: the test runs setup.sh's variable block (every line before
# print_help, which sets values only) with a test HOME, and compares the auth
# header that the downloads send with the expected value. The token is never
# printed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fail() { echo "test-setup-hf-token: FAIL -- $*" >&2; exit 1; }

sed -n '1,/^print_help() {$/p' "${ROOT}/setup.sh" | sed '$d' > "${WORK}/prelude.sh"
grep -q '^REFERENCE_AUTH_HEADER=' "${WORK}/prelude.sh" || fail "setup.sh sets REFERENCE_AUTH_HEADER after print_help"

token="hf-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')"
HOME_T="${WORK}/home"
mkdir -p "${HOME_T}/.cache/huggingface"

# header ENV...: "bearer" when the header is the bearer header of the token,
# "none" when there is no header, "other" for any other header.
header() {
  (cd "${WORK}" && env -u HF_TOKEN -u HF_TOKEN_PATH -u MLXFAST_REFERENCE_AUTH_HEADER -u MLXFAST_REFERENCE_BASE_URL \
    HOME="${HOME_T}" T="${token}" "$@" bash -c '. ./prelude.sh
      if [[ -z "${REFERENCE_AUTH_HEADER}" ]]; then echo none
      elif [[ "${REFERENCE_AUTH_HEADER}" == "Authorization: Bearer ${T}" ]]; then echo bearer
      else echo other; fi')
}

[[ "$(header)" == none ]] || fail "no HF_TOKEN and no token file: a header is set"
[[ "$(header HF_TOKEN="${token}")" == bearer ]] || fail "HF_TOKEN is not the bearer token"

printf '  %s \r\nsecond-line\n' "${token}" > "${HOME_T}/.cache/huggingface/token"
[[ "$(header)" == bearer ]] || fail "the token file (first line, trimmed) is not the bearer token"

printf 'hf-file-token-not-used\n' > "${HOME_T}/.cache/huggingface/token"
[[ "$(header HF_TOKEN="${token}")" == bearer ]] || fail "the token file wins over HF_TOKEN"
rm -f "${HOME_T}/.cache/huggingface/token"

printf '%s\n' "${token}" > "${WORK}/token-elsewhere"
[[ "$(header HF_TOKEN_PATH="${WORK}/token-elsewhere")" == bearer ]] || fail "HF_TOKEN_PATH is not read"
[[ "$(header HF_TOKEN_PATH="${WORK}/token-elsewhere" MLXFAST_REFERENCE_BASE_URL="https://mirror.example/x")" == none ]] \
  || fail "the token goes to a source that is not huggingface.co"
[[ "$(header HF_TOKEN_PATH="${WORK}/no-such-file")" == none ]] || fail "a missing token file sets a header"

echo "test-setup-hf-token: all cases passed"
