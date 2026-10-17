# mlxfast — Bonsai 2 9B MLX

This repository is the engine for the Bonsai 2 9B MLX speedup benchmark. The
track identifier is `bonsai2-9b-mlx-v1`.

> **NOTE — the target is the Ternary Bonsai 2 9B MLX pack.**
> The target is `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @
> `f4e28a59531731c41b6dfac88cf271bd58f0a287`. The pack is built on
> Qwen/Qwen3.5-9B. It is affine 2-bit at group 128, with a signed block-4096
> Walsh-Hadamard transform. The MTP head is the pack's own `mtp/` folder.
> The repository is private until prism-ml makes it public. You need read
> access to it. Set `HF_TOKEN` to a Hugging Face token that has that access.
> Official scoring is on. The ranked box is calibrated for the pack.
> Section [Current status](#current-status) states each fact.

## What this repository is

This repository holds the engine. The engine is the Swift and Metal code that
runs the target model on Apple Silicon. You optimize the engine. You make the
model do the same work in less time.

The benchmarker measures the engine. The benchmarker is a separate program
called `benchd`. It arrives as a verified prebuilt binary. It owns all
timing, all scoring, and all gates. Nothing in this repository measures or
scores anything.

The ranked run runs ONE STREAM AT A TIME. The ranked run measures your engine
and a serial control engine in the same session, on the same box. The score
compares the two. The control engine is the organizer's reference tree, and its
cost is measured in that same session. No file stores it.

> **NOTE — the track goldens are not in this repository.**
> A track's timed tapes and per-depth oracles are organizer material. They are
> published in R2 at the `r2_path` keys the contract pins. The ranked box
> stages them out of band into the directory its runner service exports as
> `MLXFAST_QWEN38_GOLDEN_DIR`. That environment name keeps its Qwen spelling
> because it is the fleet's contract with the runner service, not this track's
> own name. `tools/ranked-box-preflight.sh` verifies every file there against
> the contract's `{sha256, bytes}` and refuses an extra `*.json`. They are
> never in git, so your clone does not carry them. The two public captures for
> the local modes are different material: they ship in this repository under
> `correctness_prompts/bonsai2-9b-mlx-v1/`.
>
> The organizer recorded the goldens and the two public captures on M5 box 2
> against the pack. The fixture pins the goldens, and the tests pin the public
> captures. Section [Current status](#current-status) states what is still
> open.
>
> The organizer stages them with the signer this repository vendors:
>
> ```bash
> R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
>   tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
> tools/ranked-box-preflight.sh
> ```

> Official scoring is on. It was switched on in its own change, after the
> ranked box was calibrated for the pack. Section
> [Current status](#current-status) states each fact.
> `docs/bonsai2-9b-port-notes.md` is the engineering record.

### Lineage

This repository is a seed of the Ternary Bonsai 2 27B MLX engine
(`Layr-Labs/mlxfast-bonsai2-27b-engine`). That engine descends from the
Nemotron 3.5 30B A3B and Qwen 3.8 125B A6B MLX engines, and from
`Layr-Labs/mlxfast-challenge-dev`. Those repositories rank different models
under different rules. Only this track's rules apply here.

The `.gemma4` transform family in `Sources/MLXFastTransform/` is not a
leftover. It is the synthetic-fixture substrate the transform pipeline tests
run on. Those tests exercise the transform, not a Gemma model.

## Requirements

- An Apple Silicon Mac.
- Enough unified memory for the target model and its working set. The
  21 pinned files of the target are 3,475,218,619 bytes on disk, and the MTP
  head is 486,584,873 bytes. A run loads the target and the MTP head. Add the
  KV cache, the recurrent state and the decode buffers on top of that.
- Read access to the private repository
  `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit`, until prism-ml makes it public.
  Put a Hugging Face token with that access in `HF_TOKEN` before you run
  `./setup.sh`. Do not commit the token or print it.
- At least 22 GiB of free disk space before the download starts. Change this
  limit with `MLXFAST_REFERENCE_MIN_FREE_GIB`.
- macOS 14 or later. `Package.swift` sets that platform floor. The Metal
  toolchain policy in `setup.sh` treats macOS 26 as its own case.
- Swift 6, through Xcode or through the Xcode Command Line Tools.
  `Package.swift` declares `swift-tools-version: 6.3`.
- Xcode 26.4.1 (build 17E202) for a build check that matches the ranked box.
  The ranked runners build with Apple Swift 6.3.1 against the macOS 26.4 SDK,
  on macOS 26.6.2. The hosted CI uses a newer Swift. See "The ranked box
  toolchain".
- The Xcode Metal Toolchain. `./setup.sh` tries to download it. The Command
  Line Tools alone carry no `metal` compiler, so a box that builds the metallib
  needs full Xcode. Install it, open it once, and accept the license with
  `sudo xcodebuild -license accept`.
- CMake. `./setup.sh` installs it through Homebrew when it is missing.
- Git.

You do not need Rust. The benchmarker arrives as a prebuilt binary.
`./tools/fetch-benchd.sh` resolves it from the bench repository's `main` dist
channel and verifies it against that channel's `benchd.manifest.json`. The
channel is the public bench repository `Layr-Labs/benchd`. The fetch needs no
token.

## Quickstart

Run these commands in order. One sentence describes each command.

```bash
git clone <repository-url> mlxfast-bonsai2-9b-engine
```

This command copies the repository to your machine.

```bash
cd mlxfast-bonsai2-9b-engine
```

This command makes the repository your working directory.

```bash
./tools/fetch-benchd.sh
```

This command resolves the benchmarker binary from the dist channel into
`benchd-bin/` and verifies its sha256 and its byte count against the channel's
`benchd.manifest.json` (installed beside the binary).

```bash
./setup.sh
```

This command checks your toolchain, builds the two Swift binaries, builds
`mlx.metallib`, and downloads and verifies the target model and the MTP head.
A pin that carries the sentinel `BONSAI-MLX-PENDING-ORGANIZER` is refused by
name before anything is built.

The repository is private. Export `HF_TOKEN` in your shell before you run
`./setup.sh`, or log in with `huggingface-cli login`. When `HF_TOKEN` is unset,
the script reads the token from the Hugging Face token file of the user
(`${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}`). The script sends the
token to huggingface.co only. It never prints the token. Without a token, the
fetch is anonymous and fails while the repository is private.

```bash
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
```

This command builds the two Swift binaries and `mlx.metallib` and downloads
nothing.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/bonsai2-9b-target \
  --output weights
```

This command converts the downloaded checkpoint into the `weights/` tree that
the engine loads.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test against the shipped local-iterate capture,
with the engine `./setup.sh` built at `.build/release/bench-worker`.

The MTP head is a separate artifact. `./setup.sh` stages it for you with
`./setup-mtp-head.sh`. See [The MTP head is a separate
export](#the-mtp-head-is-a-separate-export).

> **NOTE — the local modes need no environment variable.**
> `--local-iterate` uses the short shipped capture and `--local-submit` the
> long one. `MLXFAST_CORRECTNESS_GOLDEN_PATH` overrides that choice, and
> `MLXFAST_ENGINE_BIN` overrides the engine binary. `--official` has no
> default golden: it takes the organizer-provisioned hidden golden.

> **WARNING — the local test runs serial unless you give it a depth.**
> `./benchmark.sh` does not read `mtp-head.manifest.json`. Without
> `--mtp-depth`, the local test runs depth 0, also when your tree declares a
> depth. The ranked run uses the declared depth. To test the depth that your
> tree declares, add `--mtp-depth N`, where N is the value that
> `tools/spec-declaration.sh draft-len` prints. When that value is 0, do not
> add the flag.
>
> ```bash
> ./benchmark.sh --local-submit --mtp-depth "$(tools/spec-declaration.sh draft-len)"
> ```

> **WARNING — do not pass `--golden`, `--weights`, or `--score-path` to
> `./benchmark.sh`.**
> The script rejects these flags. Use the environment variables instead.
> `MLXFAST_WEIGHTS_PATH` defaults to `weights`.

### The public captures

The local test checks correctness against a public capture. The two public
captures ship in this repository under
`correctness_prompts/bonsai2-9b-mlx-v1/`. The organizer records them on the
track's box against the pinned target, in one recording, so the short capture
is the long capture truncated. The shipped captures were recorded on M5 box 2
against the pack. Generations A and B are byte-identical.

| Capture | Path | sha256 prefix | Bytes | Expected tokens | Use |
|---|---|---|---|---|---|
| Short | `correctness_prompts/bonsai2-9b-mlx-v1/public-local-iterate.golden.json` | `706f20bd` | 10670 | 256 | `--local-iterate`. The drift tripwire. |
| Long | `correctness_prompts/bonsai2-9b-mlx-v1/public-local-submit.golden.json` | `1eda8bf6` | 21052 | 1024 | `--local-submit`. The long capture. |

Nothing fetches them, and no environment variable is needed to use them. The
hidden goldens are different material: they live in R2 and on the ranked box.
See the note at the top of this file.

> **NOTE — a golden must name the checkpoint it came from.**
> Each `.json` golden must carry a `model_provenance` block. The block names
> the repository and the revision of the pinned model. The loader refuses a
> golden that carries no block. The error message names `model_provenance`.
> The loader does not skip the check when the block is absent. The loader
> accepts only the pack's pin, `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @
> `f4e28a59531731c41b6dfac88cf271bd58f0a287`.

## Repository structure

| Path | What it holds | Status |
|---|---|---|
| `Sources/MLXFastTransform/` | The offline transform that writes `weights/`. | Editable |
| `Sources/MLXFastCLI/` | The trusted CLI, `mlxfast-swift`. | Trusted |
| `Sources/MLXFastCore/` | Shared constants and contracts. | Trusted |
| `Sources/MLXFastTrustedHarness/` | The editable-surface budget, the head declaration reader, transform verification, and the metallib fingerprint. | Trusted |
| `Vendor/mlx-swift/` | The pinned MLX fork, a vendored tree. The listed Metal kernel sources are editable. | Mixed |
| `Vendor/mlx-swift-lm/` | The engine fork, a vendored tree. It holds the model, the runner, and `bench-worker`. The Bonsai 2 and Qwen 3.5 model files are editable. | Mixed |
| `fixtures/` | The track contract, the two checkpoint manifests, the pack's config and tensor inventory, and the Qwen 3.5 9B geometry source. | Trusted |
| `tools/` | Setup, build, lint, and measurement scripts. | Trusted |
| `benchd-bin/` | Where `./tools/fetch-benchd.sh` installs the verified binary. Git ignores it. | Fetched |
| `mtp-head.manifest.json` | The speculative-decoder declaration. It names the decoder and the draft depth. It declares; it carries no weights. | Editable, optional |
| `correctness_prompts/` | The two public captures for the local modes. The hidden goldens live in R2 and on the ranked box, never here. | Trusted |
| `weights/` | The transformed weights the engine loads. | Generated |
| `benchmark.json` | The Yukon track manifest. It lists every editable path. | Trusted |

### Starting a new track from this repository

This repository is the template for the next track. Seed a new repository with a
copy of this tree, then run `tools/new-track.sh` in it. The script stamps the new
identity into the manifest, the contract fixture, the checkpoint file list, the
runner label, the engine pin and the docs. It touches no golden: a track's
goldens live in R2 and on its own box, never in git. It never commits.
`docs/new-track-repo-procedure.md` holds the full procedure and the usage line.

### The engine is a vendored tree

`Vendor/mlx-swift-lm` is a copy of `Layr-Labs/mlx-swift-lm`, as plain files.
That fork holds the three parts of the engine:

* the model, including the packed Bonsai 2 tower and the MTP assistant that
  binds to it;
* the runner, which loads the checkpoint and builds the engine and the one-row
  stepper;
* `bench-worker`, the Engine Protocol v1 server. The benchmarker starts this
  binary. One binary serves every model family, and it selects the runner from
  the checkpoint's own `model_type`.

A plain `git clone` carries the whole engine. This repository has no submodule.

The tree was cut from `Layr-Labs/mlx-swift-lm` `feat/qwen38-flash-next-runner`
at `859f4d9`, with fork `main` `0607011` merged in. The merge, two build fixes
and the Bonsai 2 support files taken from fork `main` `fd0eaac` (pull requests
154 and 155) are commits of THIS repository. Nothing goes upstream. Edit the
tree in place, here. The contract fixture's `mlx_swift_lm_revision` records the
upstream base only.

The fork is a copy and not a submodule for one reason: its Bonsai 2 and Qwen
3.5 model files are an editable surface. An editable path names bytes in this
tree, and a gitlink names a commit.

`Vendor/mlx-swift` is a vendored tree for the same reason: its Metal kernel
sources are editable. The fork asks for `mlx-swift` from the network. SwiftPM
uses this package's local `Vendor/mlx-swift` instead, because a local path
dependency of the root package wins. SwiftPM prints a "conflicting identity"
warning when it does this. The warning is expected.

The copy is cut from `Layr-Labs/mlx-swift` `70052b2`. It holds the two nested
submodules of that repository as plain files: `Source/Cmlx/mlx` at `3fa8f25e`
and `Source/Cmlx/mlx-c` at `02cf6f4d`. To move the pin, copy the whole tree
again from a fresh checkout of the new commit, with its submodules, then
rebuild the metallib.

### The MTP head is staged on its own

This track has one declarable speculative decoder, the MTP head. The pack's
`model.safetensors` carries no head, and its `config.json` declares
`mtp_num_hidden_layers: 0`. The head is the pack's own trained drafter in the
`mtp/` folder of the same repository and revision. It is a standalone
`qwen3_5_mtp` export of 31 tensors, used unmodified.
`fixtures/reference_bonsai2_9b_mtp_head.sha256` pins the two files of that
folder, `config.json` and `model.safetensors`. The paths in that manifest are
relative to `mtp/`.

The head's `config.json` restates the trunk's `text_config`, so it declares
`mtp_num_hidden_layers: 0` too. The loader reads the head depth off the
head's tensors instead: the `layers.<i>.` indices must be `0..<n`, with n from
1 to 4. A non-zero declared depth must agree with the tensors.

| Property | Value |
|---|---|
| Repository | `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @ `f4e28a59531731c41b6dfac88cf271bd58f0a287`, folder `mtp/` |
| Model type | `qwen3_5_mtp` |
| Hidden layers | 1, full attention. Read off the tensors. |
| Tensors | 31, no name prefix, 136,881,152 tensor bytes |
| Quantization | Affine, group size 64, 4 bits |
| Block size | 2 |
| Embeddings | None of its own. It rides the target's embeddings and output head. |
| Permitted draft depths | 1 to 6 |
| Maximum speculative batch | 1. The arm is single-stream. |

The second decoder is the DFlash 2 block drafter, `"decoder": "dflash"`, at
depth 1 to 15 (`block_size - 1`). Its source is Prism's
`Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf` at commit `608e89c3` (sha256
`95149abd`, FFN 12288). `./setup-dflash-drafter.sh` converts it with
`tools/convert-dflash-gguf.py` into `reference_weights/bonsai2-9b-dflash2-drafter`:
the Q8_0 linears become MLX affine 8-bit group-32 linears without loss, and the
norms, the convolution base kernels and the selector codebooks become BF16.
`fixtures/reference_bonsai2_9b_dflash2_drafter.sha256` pins the result. David
sanctioned this one conversion on 2026-10-14.

No submission carries head weights. `./setup-mtp-head.sh` stages the head into
`reference_weights/bonsai2-9b-mtp-head`. It fetches the two pinned files of
`mtp/` only. It uses the same `HF_TOKEN` as `./setup.sh`. `./setup.sh` runs it
after the target.
`tools/resident-up.sh` names the head directory to the worker with
`--drafter`, on both legs. A checkpoint that carries its own `mtp.*` block can
name its own directory.

> **NOTE — the head borrows packed modules.**
> On the Bonsai 2 pack, the target's embedding and output projection are packed
> Hadamard modules. The head calls them as modules, so the forward transform
> applies to the head input and the inverse transform applies to the embedding
> output. Code that reads their raw `.weight` skips both.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 97 entries. The
rule behind the list is simple. Code that **proposes** tokens or computes the
forward pass is editable. Code that **verifies**, **measures**, or **ledgers**
stays trusted.

The editable surface has five groups.

1. The decoder declaration. `mtp-head.manifest.json`. The declaration file
   only. It names the decoder and the draft depth.
2. The offline transform. `Sources/MLXFastTransform/`.
3. The model files. Most are in `Vendor/mlx-swift-lm/`:
   `Libraries/MLXLMCommon/PrismHadamardCheckpoint.swift`,
   `Libraries/MLXLLM/Models/PrismHadamardQwen35.swift`, the nine
   `Libraries/MLXLLM/Models/Qwen35*.swift` tower and MTP files, and the two
   DFlash 2 files `Libraries/MLXLLM/Models/DFlash2Draft.swift` and
   `Libraries/MLXLLM/Models/Qwen35DFlash2Assistant.swift`. The two DFlash 2
   files are inherited from the 27B track and serve the dflash decoder. One more file
   is in the other vendored tree,
   `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift`, which holds the signed
   block transform and the two packed layer types every projection runs.
4. The batching engine and the cache layer: the whole
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` directory
   (the CBv2 engine, the MTP round driver, the paged KV pool and its kernels),
   twelve `Libraries/MLXLMCommon/` cache, attention, RoPE and config files, and
   `Libraries/MLXLMServer/Runtime/ToolStreamHandler.swift`. This is the shape
   the Gemma 4 track opened.
5. The 68 vendored MLX Metal kernel files the forward pass dispatches. These
   are the quantized matmul, the gather-GEMM, SDPA and steel attention,
   RoPE, RMSNorm, softmax, sort, reduce, copy, elementwise, `arg_reduce`, and
   gather indexing.

The rest of the engine fork is trusted: the runner and its registry,
`bench-worker`, the server, and every `Libraries/MLXLMCommon/` file not named
above.

### The decoder declaration

`mtp-head.manifest.json` is the declaration file. It stays editable and
optional. `spec.decoder` names the decoder the submission arms, and this track
declares one:

```json
"spec": { "decoder": "mtp", "enabled": true, "num_speculative_tokens": 3 }
```

An absent `decoder` key reads as `"mtp"`. `"decoder": "dflash"` arms the
DFlash 2 drafter at depth 1 to 15.

The shipped declaration switches the decoder off, so the stock tree runs
serial (depth 0):

```json
"spec": { "decoder": "mtp", "enabled": false, "num_speculative_tokens": 0 }
```

The declaration accepts `"source": "pinned"` only. On this track that means the
pinned MTP head, `fixtures/reference_bonsai2_9b_mtp_head.sha256`, the pack's
own `mtp/` drafter. `"source": "remote"` and `"source": "in_branch"` are
refused by name.

The MTP head's cap is 2 GiB (`max_bytes` = 2147483648). A declaration may lower
the cap. It may not raise it.

An absent declaration selects the pinned MTP head. That is the normal case. A
declaration that is present but broken is a refusal. The runner never falls
back silently.

The size cap is the only gate on a declaration. A declared `sha256` is
optional, and the runner does not verify it against the head bytes.
`docs/participant-contract.md` section 4.3 states that limit plainly.

`tools/resident-up.sh` validates the leg's own declaration through
`tools/spec-declaration.sh`. It decides nothing else: the worker takes the
decoder and the depth from the wire, per request.

The MTP head is the organizer's pinned weights. You may re-quantize it. You may
not replace it, and you may not upload head weights of your own. Custom head
weights are not accepted on this track. A re-quantization happens ON LOAD, in
memory. Nothing on disk changes. The head loader calls `quantize(model:)` while
it binds the head, and the file that holds that call is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.
Change the geometry that call selects. `docs/participant-contract.md` section
4.4 is the authority.

The head only **proposes** tokens. The pinned target model decides every
emitted token. Both legs load the head; only the candidate leg drafts with it.

### The draft depth

> **NOTE — the draft depth is a free lever.**
> The scored stream count is 1 and you may not tune it. The draft depth is
> yours. You declare it in `mtp-head.manifest.json`, which is editable:
>
> ```json
> "spec": { "decoder": "mtp", "enabled": true, "num_speculative_tokens": 3 }
> ```
>
> The MTP head permits a depth from 1 to 6
> (`mtp_head.permitted_draft_depths`). With no `spec` block,
> `enabled: false`, or `0`, the run is serial (depth 0). A depth outside the
> list is refused before the engine starts.
> `tools/spec-declaration.sh describe` prints what your tree declares.
> The ranked run sends the depth to the engine on each request and seals the
> engine's `effective_spec` echo.
>
> A declared depth is scored against the oracle recorded AT THAT DEPTH for each
> live golden (`fixtures/bonsai2_9b_mlx_v1_track.json`
> `live_golden_speculative`, keyed by depth and then by live golden name). The
> runner verifies a draft window in one target forward, and the multi-row
> kernels can round differently from the single-row ones. A speculative run
> therefore need not be token-identical to the serial tape. It must be
> token-identical to its own depth's tape, which the organizer records on the
> pinned runner. Correctness at a depth means matching that tape. On the pack,
> each depth's tape is byte-identical to the serial tape: the stock MTP free
> run is exact at every depth. On the earlier stand-in target it was not.
>
> Every run seals what actually ran: `effective_spec` for the declared depth,
> `effective_mean_draft_len` for the realized draft length.

Try a depth locally with one flag. The local flag names the depth directly, so
you do not have to edit the declaration to try a value. The local test runs
serial unless you give it `--mtp-depth`.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test with no speculation.

```bash
./benchmark.sh --local-iterate --mtp-depth 3
```

This command runs the local test with the MTP head at depth 3. The flag also
exports the head directory to the worker in `BENCH_WORKER_DRAFTER`.

### The byte budget

`benchmark.json` `editableSurfaceByteBudget` caps the enforced editable
surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 7949663 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is absent. The exemption existed to let head weights ride in a
submission outside the source budget. A submission carries no head weights, so
there is nothing to exempt. The two exempt caps stay declared because both
enforcers carry the same numbers as compiled-in fallbacks and this manifest is
what holds them to a reviewed value.

### The target quantization is frozen

The target model's quantization is frozen as shipped. Do not re-quantize a
target weight. Do not re-represent one. Do not change the numerical format of
one. This holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model instead of
optimizing the accepted one.

The MTP head is a narrow exception, and the exception is re-quantization only.
You may re-quantize the head. You may not replace it. The head stays within its
2 GiB declaration cap. The head only proposes tokens, and the pinned target
decides every emitted token.

### What you must not change

- Everything in `Sources/` that `editablePaths` does not list.
- `Package.swift` and `Package.resolved`. The dependency graph is frozen.
- Everything in `Vendor/` that `editablePaths` does not list.
- `fixtures/`, `benchmark.json`, the scripts, the tests, and the
  documents.
- `weights/`, the reference checkpoints, the scores, and the goldens.

Do not hardcode hidden prompts. Do not hardcode hidden token identifiers. Do
not use timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

Do not add a cache keyed on a request's input tokens whose only possible hit is
the harness repeating one identical computation. The benchmark measures
single-pass inference. Input-independent caches stay legal. These are weights,
dequantized tensors, and RoPE or mask tables keyed on shapes and offsets.
Within-request KV reuse also stays legal.

## Local testing vs the ranked run

The local test and the ranked run are different by design. Read this section
before you tune.

The local test runs a **single stream**. It uses a public golden. It prints a
single-stream estimate.

### The ranked box toolchain

The ranked runners are four Apple M5 Max machines with 128 GB of unified
memory. They build with this toolchain.

| Component | Version |
|---|---|
| Xcode | 26.4.1 (build 17E202) |
| Swift | Apple Swift 6.3.1 |
| SDK | macOS 26.4 |
| macOS | 26.6.2 |
| Metal toolchain | `com.apple.dt.toolchain.Metal.32023.883` |

The hosted CI job "swift build + test (hosted macOS, no GPU)" runs on the
GitHub `macos-26` image. That image has Xcode 26.6 (build 17F113) and Apple
Swift 6.3.3. A green hosted CI run does not prove that your code compiles on
the ranked box.

A ranked job builds your candidate in its Setup step, "Setup (toolchain, Metal
kernels, pinned checkpoint)". A compile failure there fails the submission
before any scoring.

Build-check with Xcode 26.4.1 before you submit. Select it with
`DEVELOPER_DIR` or with `sudo xcode-select -s`.

```bash
DEVELOPER_DIR=/Applications/Xcode_26.4.1.app/Contents/Developer \
  swift build --build-tests --force-resolved-versions
```

The time limit of the Swift type checker differs between compiler versions.
When the 6.3.1 compiler reports a type-check timeout, split the long
expression into sub-expressions.

### What each local mode checks

Both local modes run one fused checked-timing pass. The pass teacher-forces the
golden's expected tokens and times the wall clock. It judges correctness from
that same pass. A mismatch is reported as a teacher-forced token mismatch.

A correctness failure does not discard the timing. The benchmarker reruns the
timing phase in a mismatch-tolerant form. It then reports the correctness
failure together with real timing numbers.

| Mode | Decode steps | Expected tokens the golden must hold | Cool gate |
|---|---|---|---|
| `--local-iterate` | 128 | 129 | On, because `./benchmark.sh` always passes `--cool-gate` |
| `--local-submit` | 1023 | 1024 | On |

The gate is on because `./benchmark.sh` arms it. Driving the Swift CLI directly
skips it and times a hot GPU. Use `./benchmark.sh`. See AGENTS.md, "The
cool-down gate".

The two public captures differ in length for this reason. `--local-iterate`
uses the 256-token capture and `--local-submit` the 1024-token capture by
default.

> **NOTE — both local modes check correctness and speed.**
> Neither local mode is a speed-only signal. Both apply the teacher-forced
> check. Neither one runs the ranked gates.

The ranked run also runs one stream at a time, so the two shapes agree. A local
number is still only a smoke signal: your machine is not the ranked box.

> **WARNING — a local score is directional, not predictive.**
> Treat a local score as a smoke signal for speed and correctness. Do not treat
> it as a prediction of the ranked composite. The ranked M5 run is the
> authority.

## Scoring and gates

### The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1.

The decode gain is measured over the decode window only. The seed prefill of
the prompt is not in it. The prefill gain is measured over its own prefill
phase.

### The pair is measured, not stored

A ranked run measures pairs. It measures them on the SAME box, in the SAME
job. The fixture lists the live prompts in `live_goldens`. Pair k measures live
prompt (k - 1) mod N, where N is the number of live prompts, so
`official_pairs` must be a multiple of N. A pair measures TWO legs over one
prompt:

1. The **serial-control leg**. It runs on the organizer's reference tree. That
   tree is a build of this repository at the commit the fixture names in
   `baseline_reference_commit`. This leg uses no speculation. Its tokens are
   checked against the serial tape, never against a per-depth tape.
2. The **candidate leg**. It runs on your tree, at the draft depth you declare.

The score is the ratio of the two measurements. Both numbers come from the same
machine, minutes apart.

**NO FILE HOLDS A BASELINE PAIR.** The scoring constants hold none. The fixture
holds none. The goldens hold none. A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is refused on the ranked path.
`tools/lint-benchmark-manifest.py` keeps the two fields out of the tree.

The organizer stages the reference tree on each ranked box. The ranked job
verifies that tree. It does not fetch it and it does not build it.

`baseline_reference_commit` names `63dec40`, the merge of PR #28. It holds the
pack's runtime code (the pins, the head loader, the FP16 head dtype fix, and the
matmul that reads the F16 scales and biases in place). Re-pin it, and re-stage the reference tree on every box,
whenever a trusted-side code change merges that the control leg must carry.

### Each box has its own calibration

Each ranked box records what its own serial-control leg costs. The record is a
file. `MLXFAST_BASELINE_CALIBRATION` names it.

**THE FILE IS A HEALTH BAND. IT IS NEVER A DENOMINATOR.** The benchmarker
compares the measured control leg against the band. The run stops by name when
the leg falls outside the band. A stale calibration file can stop a run. It can
never move a score.

An operator writes the file on the box:

```bash
tools/calibrate-box.sh "<runner name>" /path/to/baseline-calibration.json
```

The command takes the box GPU lock. It then runs the serial-control leg three
times for each live golden under the full official methodology: the quiescence
gate and the cool gate before each pass, one resident worker for each pass, and
the live goldens the ranked run scores over. It writes one entry for each live
golden: the mean, the coefficient of variation and the band for prefill and for
decode. Decode is the decode window. It writes no file when the coefficient of
variation is more than 1 percent on either axis. A box that cannot repeat
itself has no band. `tools/ranked-box-preflight.sh` accepts calibration file
version 3 only.

The `box` value in the file must equal the runner name. The `reference_commit`
value must equal the fixture's `baseline_reference_commit`.
`tools/ranked-box-preflight.sh` refuses the run when either differs.

### The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 512 |
| Checked decode steps | 128 |
| Golden shape | 512 prompt tokens and 129 expected tokens |
| Streams per window | 1 |
| Timed prompts per pair | 1 (pair k measures live golden (k - 1) mod N) |
| Legs per pair | 2 (serial control, then candidate) |

The two legs run one after the other. Each leg loads the weights once. The
unmeasured warm-up prefill pass stays at 1 pass, and it applies to both legs in
the same way.

The serial-control leg runs entirely inside the reference tree. It uses that
tree's worker, that tree's Metal library and that tree's own transformed
weights. Nothing you change can move it.

### One resident worker per LEG, booted by the benchmarker

A ranked job has two legs on two trees. The weights load ONCE per leg.

The benchmarker starts `bench-worker runtime-worker` once for each phase: the
warmup, the timed prefill, the timed decode and the correctness pass. Each start
used to load the whole checkpoint again. It no longer does: a resident owns the
weights for the leg, and each per-phase worker attaches to it.

**THE RESIDENT BELONGS TO THE LEG, NOT TO THE WINDOW.** The two legs run
different trees with different weights, so one resident cannot serve both. The
benchmarker knows where a leg begins and ends, so the benchmarker boots it. For
each leg it calls that leg's OWN copy of `tools/resident-up.sh`:

```bash
tools/resident-up.sh --boot --spec <serial|mtp> --draft-len <N> --socket-out <file>
tools/resident-up.sh --stop --socket <path>
```

`--boot` loads that tree's own `.build/release/bench-worker`, that tree's own
`weights/` and that tree's own MTP head directory, waits for a healthy hello,
writes the socket path as the first line
of the `--socket-out` file, and exits 0 with the resident still running. A
`<socket>.pid` sidecar and a `<socket>.ready` marker sit beside the socket.
`--stop` ends that resident and removes all three files. A second `--stop` is a
no-op.

`--draft-len` is 0 with `serial` and 1 to 6 with `mtp`. A mismatch is a
refusal.

`--spec` is authoritative over what the leg DOES. It is the label the leg
records, so a reference tree that declares a draft depth still boots a serial
control leg when the benchmarker says serial. The worker takes the decoder and
the depth from the wire, per request, and never from the tree.

`--spec` does NOT choose the drafter. The boot validates the leg's OWN
`mtp-head.manifest.json` through `tools/spec-declaration.sh` and loads the
tree's own MTP head. Nothing crosses between the two trees.

`tools/stage-baseline-workspace.sh` stages the MTP head into the reference
tree from `MLXFAST_MTP_HEAD_REFERENCE_DIR`.

`tools/bonsai2-9b-measure-and-score.sh` boots nothing. It takes the box GPU
lock `/tmp/mtplx-gpu-exclusive.lock` and holds it for the whole measurement,
because a resident holds the whole checkpoint in unified memory whoever booted
it, and the box needs exactly one loader. `tools/resident-up.sh` refuses
to boot when nobody holds that lock, so every per-leg boot happens inside that
window.

| Name | What it is |
|---|---|
| `BENCH_WORKER_RESIDENT_SOCKET` | A leg's resident Unix socket. A per-phase worker attaches to it instead of loading. The measure script never sets it, and REFUSES when the environment does. |
| `RESIDENT_IDENTITY_FILE` | A JSON file that records the resident, its socket, its declared leg and its hello. |

> **WARNING — do not set `BENCH_WORKER_RESIDENT_SOCKET` on a ranked box.**
> It names one already-loaded resident. Both legs would attach to it, so the
> serial-control leg would run on the candidate's weights. The measure script
> and `tools/ranked-box-preflight.sh` both refuse it.

The wrapper form of `tools/resident-up.sh` stays for local, unscored use:

```bash
tools/resident-up.sh --weights <dir> -- <command>
```

It boots one resident, exports `BENCH_WORKER_RESIDENT_SOCKET` to the command,
and halts the resident when the command ends. The ranked path does not use it.

### The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 3 |
| `minPairsPerCohort` | 3 |
| `decodeSpeedupFloor` | 0.95 |
| `decodeSpeedupCeiling` | 5.0 |
| `kvBackend` | `contiguous` |

The scored width is fixed. A width the benchmarker has not certified has no
series tag, and the benchmarker refuses that width rather than run it.

The fixture's `official_pairs` sets how many pairs one ranked run measures.
Each pair scores on its own control leg. The run score is the mean of the
per-pair composites (`official_pair_combine: "mean"`). Every pair stays in the
artifact as measured.

The candidate's slowness gate is the two floors: 0.95 on the prefill gain and
0.95 on the decode gain. The control leg's health band is the box calibration.
The decode band allows 3 percent upward.

`kvBackend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

### Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10% budget**.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require your output to match the serial trajectory token
> for token. A speculative forward verifies a whole draft chain in one pass, and
> that pass diverges from a one-token-at-a-time decode at near-tie argmaxes. The
> gate prices that divergence against the 10% budget. Do not read the gate as
> lossless.

A near-tie argmax can diverge across Apple Silicon generations, even for
correct code. Before you treat a local failure as your own regression, check
whether an unmodified `main` fails at the same token position on your machine.

### Current status

Eight statements are true right now.

1. The target is the Ternary Bonsai 2 9B MLX pack,
   `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @
   `f4e28a59531731c41b6dfac88cf271bd58f0a287`. `target.upstream_model_id`,
   `target.upstream_revision`, the MTP head's pins and the license name it.
   `fixtures/reference_bonsai2_9b_2bit.sha256` pins 22 files of it, and
   `fixtures/reference_bonsai2_9b_mtp_head.sha256` pins the 2 files of its
   `mtp/` folder.
2. The repository is private until prism-ml makes it public. `./setup.sh`
   and `./setup-mtp-head.sh` need `HF_TOKEN`, a Hugging Face token with read
   access to it.
3. `fixtures/bonsai2_9b_mlx_v1_track.json` sets `official_scoring_enabled` to
   `true`. That flag is the single authority on the arm state, and the
   benchmarker enforces it. It was set to `true` in its own change, after the
   ranked box was calibrated for the pack.
4. The golden pins name the pack. The live prompts are the three Bonsai
   coding prompts `pool-9`, `pool-10` and `pool-11` at a 512-token seed.
   `timed_prompt_pool` pins their serial tapes, and `hidden_correctness_golden`
   is the `pool-9` pin. `live_golden_speculative` pins one oracle for each
   live prompt at each depth from `mtp1` to `mtp6`. On the pack, each
   per-depth oracle is byte-identical to its serial tape, also at depths 5
   and 6 with the drafter cap at 6. The objects are in R2 in a new folder,
   `correctness_prompts/bonsai2-9b-mlx-v1/b94a52f3/`, because the bucket is
   append-only.
5. The two public captures were recorded on M5 box 2 against the pack.
   `Tests/MLXFastTests/GoldenTests.swift` pins their digests and sizes.
6. The ranked runner on M5 box 2 advertises the ranked label set
   `[self-hosted, macOS, bonsai2-9b-mlx-v1]`. Box 2 was converged for the reference tree `63dec40` (m5-machine-scripts PR #93) and calibrated on 2026-10-13: three passes, benchd `18498cf`. The decode window is 144.9 to 145.5 tok/s (CV at most 1.28 %), and the prefill is 2,106 to 2,112 tok/s (CV at most 1.34 %). Box 1 was converged for the same reference tree (m5-machine-scripts PR #95) and calibrated on 2026-10-14. Boxes 1 to 4 run this track. The box file `boxes/box-N-bonsai9b.env` in m5-machine-scripts names the engine commit of each box. The direct ranked runs on `168e7e7` passed on box 2 and box 1: depth 0 (runs 38523589111 on box 2 and 38522687260 on box 1) and depth 3 (runs 38522678936 on box 2 and 38523596313 on box 1).
7. Scoring is single-stream and paired. `scored_batch_size` is `1`. The
   candidate leg runs one stream at its declared depth. The serial-control leg
   runs on the organizer's reference tree at the fixture's
   `baseline_reference_commit`. The score is the mean of the per-pair live
   ratios. No golden carries a baseline pair.
8. The bench channel is the bench repository's `main` dist channel.
   `./tools/fetch-benchd.sh` reads that channel's `dist/benchd.manifest.json`,
   verifies the binary against the `sha256` and `bytes` the manifest names, and
   prints the identity it resolved. THE MANIFEST IS THE SOURCE OF TRUTH for
   which `benchd` measures a run, and this document pins nothing: read the
   identity off the script, not off this page.

The engine serves the Qwen 3.5 9B geometry. The runner, the Prism Hadamard
loader and the model read every geometry fact from the checkpoint's
`config.json`. `Sources/MLXFastCore/Constants.swift` and the transform
validator carry the 9B geometry and the pack's block-4096 Hadamard width. The
engine has run serial and MTP depths 1 to 6 against the pack on M5 box 2, in
the golden recordings. `docs/bonsai2-9b-port-notes.md` holds the detail and
the open items.

## Submitting

Use the Yukon CLI for every account operation and every submission operation.

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

This command puts `yukon` on your path.

```bash
yukon login <api-key> --api <url>
```

This command authenticates you.

```bash
yukon clone <benchmark-id-or-name>
```

This command clones the benchmark repository.

```bash
yukon setup
```

This command runs `setupCommand`: `./tools/fetch-benchd.sh`, then
`./setup.sh`.

```bash
yukon run
```

This command runs `benchmarkCommand`, which is
`tools/bonsai2-9b-yukon-run.sh`. Off a ranked box it transforms the weights
when `weights/` is absent, then runs `./benchmark.sh --local-submit` at the
declared MTP depth. The result has `"score": null`. Only the ranked runner
gives a score.

```bash
yukon submit --model "<exact model name>" --note-file submission-note.md
```

This command uploads your editable-path archive.

```bash
yukon submissions
```

This command lists your submissions.

A submission archive replaces the editable paths. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first.
No local run blocks the upload. Run the local test yourself before you submit.

## The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @ `f4e28a59531731c41b6dfac88cf271bd58f0a287`, a 2-bit `prism_hadamard_qwen35` pack of `Qwen/Qwen3.5-9B`. Private until prism-ml makes it public. |
| Target manifest | `fixtures/reference_bonsai2_9b_2bit.sha256` (21 records: one `model.safetensors`, no index, `hadamard.json`) |
| Target config and inventory | `fixtures/bonsai2_9b_config.json` (the pack's `config.json`, verbatim) and `fixtures/bonsai2_9b_tensor_inventory.json` |
| MTP head | The pack's own `mtp/` drafter, pinned by `fixtures/reference_bonsai2_9b_mtp_head.sha256` (2 records, paths relative to `mtp/`) |
| Geometry source | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` `config.json`, checked in as `fixtures/qwen35_9b_base_config.json` |
| Benchmarker | The bench repository's `main` dist channel, resolved at run time (`benchd.manifest.json` is the authority for the bytes; `tools/fetch-benchd.sh` enforces it and logs the resolved `source_commit`/sha256) |
| Track id | `bonsai2-9b-mlx-v1` — the PLATFORM name: leaderboard namespace, runner labels, R2 prefix |
| Engine fork upstream base | `859f4d97e0d1ae9cce0883b044f764283841edfb` (+ fork `main` `0607011` merged; engine-local changes on top) |

### The target model

The geometry is the Qwen 3.5 9B backbone's. The table states the Bonsai 2 9B
pack as `fixtures/bonsai2_9b_config.json` and the tensor inventory record it.
A pack that differs is refused by name. The packing is the Ternary Bonsai 2 27B
pack's, except the Hadamard block: the 9B pack uses 4096, and the 27B pack
uses 1024.

`model.safetensors` holds 1366 tensors in one shard, 3,462,149,260 bytes: 1033
under `language_model.` and 333 under `vision_tower.`. The transform selects
the 1033 `language_model.*` tensors and drops the vision tower.

The pack ships its own `tokenizer.json` (sha256 `5f9e4d49…`, 12,807,982
bytes). Its normalizer, pre-tokenizer, post-processor and decoder are the
same as the stand-in's. It has 26 added tokens, 248044 to 248069. It does not
have the 7 audio and TTS tokens 248070 to 248076 that the stand-in had.
`chat_template.jinja` is byte-identical to the stand-in's.

| Property | Value |
|---|---|
| Architecture | The model type is `prism_hadamard_qwen35`, over a `qwen3_5` backbone. Schema 2. |
| Hidden layers | 32. Every fourth layer is full attention; the other three are gated-delta-net linear attention. |
| Layer counts | 8 full attention, 24 linear attention |
| Attention layers | At indices 3, 7, 11, … 31 |
| Attention | 16 query heads, 4 KV heads, head dimension 256, no bias, output gate on |
| Rotary | Partial (`partial_rotary_factor` 0.25), `rope_theta` 10000000, mRoPE sections 11/11/10 |
| Linear attention | 16 key heads and 32 value heads of dimension 128, conv kernel 4, FP32 state |
| MLP | Dense. A gate and an up projection of width 12288 into a down projection. No experts. |
| Hidden size | 4096 |
| Vocabulary | 248320 |
| Embeddings | Untied |
| Context | 262144 |
| Tokens | eos 248044 |
| Quantization | Affine, group size 128, 2 bits, no mixed precision. FP16 scales and biases. |
| Transform | A signed normalized block Walsh-Hadamard transform of width 4096, folded into all 202 packed modules. `hadamard.json` is the manifest. |
| Language-model tensors | 1033: 9 top-level and 32 per layer |
| MTP head | None in `model.safetensors` (`mtp_num_hidden_layers: 0`). The head is the `mtp/` folder. |
| Vision | The pack ships a tower of 333 tensors. This track serves text only and never builds it. |
| Base model | `Qwen/Qwen3.5-9B` |

Only the 8 full-attention layers carry a KV cache. The 24 linear-attention
layers carry a constant-size recurrent state instead.

Every packed projection applies the forward transform to its input, and the
embedding applies the inverse transform to its output. Both run in FP32. Call
the module; do not read its raw `.weight`.

## Building after an edit

Two build forms matter, because the vendored MLX package builds in JIT mode.

Kernel families with an `mlx-generated/*.cpp` twin compile at runtime from the
C++ source strings inside those files. For these families the twin is the
runtime-effective source. Edit the twin. Keep the readable `.metal` and `.h`
pair in step with it.

RoPE, RMSNorm, the SDPA vector kernel, and `arg_reduce` load ahead of time from
`mlx.metallib`. After you edit one of those sources, run
`tools/build-mlx-metallib.sh`. `./setup.sh` runs that script for you.

`_nax` names are the M5-generation kernel variants. The ranked runner selects
them. Tune the `_nax` twin as well as the plain one.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI into `.build/release`.

```bash
swift build -c release --force-resolved-versions --scratch-path .build-worker \
  --product bench-worker
```

This command builds the scored engine into `.build-worker/release`. The engine
is the fork's generic `bench-worker`, and this package builds it as a
dependency product, so the build reads `Vendor/mlx-swift` and your kernel edits
reach it.

```bash
tools/stage-bench-worker.sh
```

This command copies the engine and its `mlx.metallib` into `.build/release`,
where the benchmarker resolves them.

> **WARNING — always pass `--force-resolved-versions`.**
> The dependency graph is frozen. A bare `swift build` or `swift test` can
> rewrite `Package.resolved`. `./setup.sh` then refuses to run. Restore the
> file with `git checkout -- Package.resolved`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`. It runs repository hygiene checks on `ubuntu-latest`. It runs
`swift build --build-tests` and `swift test` on a hosted `macos-26` runner. It
treats first-party warnings as errors.

CI is advisory. No status check is required. A red run blocks neither a merge
nor a dispatch. `docs/ci-coverage.md` holds the detail.

CI never measures and never scores. CI holds no secret, downloads no weights,
and runs no GPU test. The GPU tests and the checkpoint tests are box-only.
`docs/ci-coverage.md` lists them, and CI fails when that list drifts.

`.github/workflows/benchmark.yml` is the ranked pipeline. It triggers on
`workflow_dispatch` only. It holds no secret. Its ranked job runs on
`[self-hosted, macOS, bonsai2-9b-mlx-v1]`. Before it measures, it verifies
every box-staged asset against the contract's `{sha256, bytes}` pins. It
publishes no score until the runner is registered, the box is staged, and the
benchmarker emits a composite. Each of those gaps gives a non-zero exit and no
artifact.

## Where to get help

| Question | Authority |
|---|---|
| What the track measures, path by path | `benchmark.json` |
| Pins, the timed pool, scoring values | `fixtures/bonsai2_9b_mlx_v1_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering record for this port | `docs/bonsai2-9b-port-notes.md` |
| What CI covers | `docs/ci-coverage.md` |
| Agent and contributor guidance | `AGENTS.md` |

> **NOTE — the order of authority.**
> The ranked M5 run is the authority on any score. The contract fixture
> `fixtures/bonsai2_9b_mlx_v1_track.json` wins over this document. This document
> only explains; it never overrides. If either disagrees with the benchmarker
> about measurement, the benchmarker wins.

## License and attribution

This repository's harness code is licensed per [LICENSE](LICENSE). The
Ternary Bonsai 2 9B pack is Apache-2.0, per the `LICENSE` file of its pinned
revision. This repository distributes no model weights.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) holds the full third-party
attribution.
