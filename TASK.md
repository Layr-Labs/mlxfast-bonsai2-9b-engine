# The task — Bonsai 2 9B MLX

Make the Bonsai 2 9B model run faster on Apple Silicon.

The ranked track is `bonsai2-9b-mlx-v1`. Read [README.md](README.md) for the
setup steps and the repository structure. Read
[`docs/participant-contract.md`](docs/participant-contract.md) for the reasons
behind the rules.

## What you optimize

You optimize the engine. The engine is the MLX runner, the offline transform,
and the vendored MLX Metal kernels that the forward pass dispatches. You also
optimize the speculative-decode arm.

> **NOTE — the target is private until prism-ml makes it public.**
> The target is `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @
> `f4e28a59531731c41b6dfac88cf271bd58f0a287`. You need read access to that
> repository. Set `HF_TOKEN` to a Hugging Face token that has that access
> before you run `./setup.sh`. Do not commit the token or print it.

The target model is the Bonsai 2 9B pack of `Qwen/Qwen3.5-9B`. It is a hybrid
model. The tower is 32 layers. Every fourth layer is full attention, so 8
layers carry a key-value cache and the other 24 are gated-delta-net recurrent
layers. The MLP is dense: a gate and an up projection into a down projection,
with no experts. The embeddings are untied.

The pack uses the Ternary Bonsai 2 27B packing at a wider Hadamard block:
affine 2 bits at group 128, and every one of its 202 linear modules folds a
signed block Walsh-Hadamard transform of width 4096 into its weights. `hadamard.json` declares that transform. The
runtime applies the forward transform to the input of each packed linear and
the inverse transform to the output of the embedding. Read
[`Vendor/mlx-swift-lm/docs/bonsai2.md`](Vendor/mlx-swift-lm/docs/bonsai2.md)
before you change a projection.

The backbone also declares a vision tower. This track serves text only. The
transform drops that namespace and the model never builds the tower.

The track has ONE declarable speculative decoder, the MTP head. It proposes
tokens, and the target model decides every emitted token. The head is staged
beside the target. It is the pack's own `mtp/` folder of the same revision,
pinned as two files of that folder.

| Decoder | Export | Staged by | Depths |
|---|---|---|---|
| MTP head | The pack's own `mtp/` drafter, one 4-bit `qwen3_5_mtp` layer | `./setup-mtp-head.sh` | 1 to 6 |

The MTP head proposes one token at a time. No DFlash 2 drafter exists for the
Qwen 3.5 9B geometry, so this track declares no DFlash 2 decoder.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 97 entries in five
groups.

| Group | Paths |
|---|---|
| The decoder declaration | `mtp-head.manifest.json` (the declaration file only) |
| The offline transform | `Sources/MLXFastTransform/` |
| The model files | `PrismHadamardCheckpoint.swift`, `PrismHadamardQwen35.swift`, the 9 `Qwen35*.swift` tower and MTP files, `DFlash2Draft.swift`, `Qwen35DFlash2Assistant.swift`, and `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift` |
| The batching engine and cache layer | `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` and 12 `MLXLMCommon` cache, attention, RoPE and config files, plus `MLXLMServer/Runtime/ToolStreamHandler.swift` (the Gemma 4 shape) |
| The vendored kernels | The 68 MLX Metal files the forward pass dispatches |

The engine fork `Vendor/mlx-swift-lm` is a vendored tree of plain files. Its
Bonsai 2 and Qwen 3.5 model files, the batching engine and the cache layer are
editable. The runner, `bench-worker` and the server in the same fork stay
trusted.

The rule behind the list is simple. Code that **proposes** tokens or computes
the forward pass is editable. Code that **verifies**, **measures**, or
**ledgers** stays trusted. The two DFlash 2 files are inherited from the 27B
track. This track declares no DFlash 2 decoder, so no run reaches them.

The MTP head is the organizer's pinned weights. You may re-quantize it. You
may not replace it, and you may not upload head weights of your own.

No submission carries a head weight file. Nothing stages one.
`mtp-head.manifest.json` stays editable and optional, and it accepts
`"source": "pinned"` only, which on this track means the pinned MTP head,
`fixtures/reference_bonsai2_9b_mtp_head.sha256`. That pin names the pack's own
`mtp/` drafter.
`"source": "remote"` and `"source": "in_branch"` are refused by name.

The MTP head's declaration cap is 2 GiB. `max_bytes` may lower the cap and may
not raise it. The size cap is the only gate on a declaration. A declared
`sha256` is optional, and the runner does not verify it.

A re-quantization happens ON LOAD, in memory. Nothing on disk changes, and no
artifact travels in a submission.

The head loader calls `quantize(model:)` while it binds the head. That call is
the seam, and the file that holds it is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.
Change the geometry that call selects. `docs/participant-contract.md` section
4.4 is the authority.

> **WARNING — the target quantization is frozen.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. An editable transform does not license the change. The MTP
> head is a narrow exception, and the exception is re-quantization only. You may
> re-quantize the head within its 2 GiB declaration cap. You may not replace it.

> **NOTE — the stream count is locked. The depth is not.**
> The ranked run scores one stream. You may not tune that.
>
> The draft depth is a free lever, declared in `mtp-head.manifest.json`, which
> is editable. The depth is not pinned at 1.
>
> ```json
> "spec": { "decoder": "mtp", "enabled": true, "num_speculative_tokens": 3 }
> ```
>
> An absent `decoder` key reads as `"mtp"`. `"decoder": "dflash"` is refused
> by name. The shipped declaration switches the decoder off, so the stock tree
> runs serial (depth 0).
>
> Select a depth from 1 to 6 (`mtp_head.permitted_draft_depths`). A depth
> outside the list is refused before the engine starts.
>
> Each run seals `effective_spec` and `effective_mean_draft_len`, so the depth
> that ran and the draft length it realized are both visible afterwards.

You may not change anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing code.

## How to run it

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command builds the Swift binaries and downloads the target model. It then
stages the MTP head beside it with `./setup-mtp-head.sh`. Both downloads read
`HF_TOKEN`. A pin that carries the sentinel `BONSAI-MLX-PENDING-ORGANIZER` is
refused by name.

```bash
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
```

This command builds the Swift binaries and downloads nothing.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/bonsai2-9b-target \
  --output weights
```

This command writes the `weights/` tree that the engine loads.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test against the shipped local-iterate capture,
with no speculation. It needs no environment variable: the engine is the one
`./setup.sh` built, and the golden is the shipped capture.

```bash
./benchmark.sh --local-iterate --mtp-depth 3
```

This command runs the same local test with the MTP head at depth 3. The flag
also exports the head directory to the worker in `BENCH_WORKER_DRAFTER`.

> **WARNING — the local test runs serial unless you give it a depth.**
> `./benchmark.sh` does not read `mtp-head.manifest.json`. Without
> `--mtp-depth`, the local test runs depth 0, also when your tree declares a
> depth. The ranked run uses the declared depth. To test the declared depth,
> add `--mtp-depth N`, where N is the value that
> `tools/spec-declaration.sh draft-len` prints.

The two public captures under `correctness_prompts/bonsai2-9b-mlx-v1/` were
recorded on M5 box 2 against the pack. `public-local-iterate.golden.json`
holds 256 expected tokens, and `public-local-submit.golden.json` holds 1024.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1. The decode
gain is measured over the decode window only. The seed prefill is not in it.

The ranked run measures pairs on the same box, in the same job. A pair is a
serial-control leg on the organizer's reference tree, then your candidate leg
at the draft depth you declare, over one live prompt. Each leg times one
stream over a 512-token seed and a 128-step decode window. It runs 3 pairs,
and pair k measures live prompt (k - 1) mod N. The score is the mean of the
per-pair composites. Every pair must clear the 0.95 floor on prefill and on
decode. The ceiling is 5.0. The KV backend is pinned `contiguous`.

The benchmarker applies a per-stream token-tolerance gate with a 10% budget.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require token-for-token equality with the serial
> trajectory. The gate prices divergence against the 10% budget.

## The current state

> **NOTE — official scoring is on.**
> `fixtures/bonsai2_9b_mlx_v1_track.json` sets `official_scoring_enabled` to
> `true`. The live prompts are `pool-9`, `pool-10` and `pool-11`, with one
> oracle for each MTP depth. The organizer recorded them on M5 box 2 against
> the pack and pinned them. On the pack, each per-depth oracle is
> byte-identical to its serial tape. Box 2 (the ranked runner) is converged
> and calibrated for the pack.
> The first ranked runs are pending. The
> stock declaration is serial (depth 0).

> **NOTE — the engine fork is a vendored tree, edited here.**
> `Vendor/mlx-swift-lm` was cut from `Layr-Labs/mlx-swift-lm`
> `feat/qwen38-flash-next-runner` `859f4d9` with fork `main` `0607011` merged in;
> the merge, the Bonsai 2 support files and the build fixes are this
> repository's commits. Nothing goes upstream.
> `docs/bonsai2-9b-port-notes.md` holds the detail.

The repositories stay private until launch.

## Local runs are directional

The local test runs one stream and so does the ranked run, so the two shapes
agree. Your machine is still not the ranked box: its memory budget, its thermal
behavior and its kernel selection all differ. Treat a local score as a smoke
signal, not as a prediction. The ranked M5 run is the authority.

## Authorities

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/bonsai2_9b_mlx_v1_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`, verified against the dist `benchd.manifest.json`) |

Where this document and the contract fixture disagree, the fixture wins. Where
either disagrees with the benchmarker about measurement, the benchmarker wins.
