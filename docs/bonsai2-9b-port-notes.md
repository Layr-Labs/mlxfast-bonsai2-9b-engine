# Bonsai 2 9B MLX — port notes

This document is the engineering record for track `bonsai2-9b-mlx-v1`. It
states what the port changed, what is verified and what is open.
`docs/bonsai2-27b-port-notes.md` is the record of the track this repository
was stamped from.

## 1. The seed

The repository `main` starts with one signed commit, `578f41d`. Its tree is
identical to `Layr-Labs/mlxfast-bonsai2-27b-engine-dev` `main` `08c75e9`.
The history stays in that repository
(`docs/new-track-repo-procedure.md` section 1).

## 2. The layers

The port is five pull requests, one for each layer. Pull request #8 then pins
a stand-in target (section 4.1). Commit `498baf2` replaces the stand-in with
the real pack (section 4.2). Later commits record the pack's goldens and
change the runner for the pack's head (section 4.3).

| Pull request | Layer | What it does |
|---|---|---|
| #1 | Stamp and contract | `tools/new-track.sh` stamps the track id, the manifest name, the fixture name and the runner label. The contract carries the Qwen 3.5 9B geometry and the pending pins. The compiled constants and the transform validator move to the 9B geometry. The golden loader takes an injectable reference pin. |
| #2 | Engine fixes | The fixes of the Nemotron 3.5 30B A3B MLX engine, pull requests #11 to #28, one commit for each source commit. |
| #3 | Vendored model and runner | A test pins the 9B residency. The runner header names the pending pack. |
| #4 | Tools | This track's tool names, the pending pins in the setup scripts, and one decoder. |
| #5 | Docs | The product docs and these notes. |
| #8 | Stand-in pin | The target and the MTP head pin the stand-in. The transform routes `qwen3_5` to its own family. Section 4.1. |
| `498baf2` | Real pack | The target and the MTP head pin `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @ `b94a52f3`. The stand-in family is deleted. Section 4.2. |
| `6a9e491` | Public captures | The two public captures are recorded on M5 box 2 against the pack. The test pins their digests and loses its known-issue wrapper. Section 4.3. |
| `6fdfdc2` | Head dtype | An unquantized MTP head computes in its own weight dtype. Section 4.3. |
| `335d11c` | Serial goldens | The pack's serial tapes are pinned. `baseline_reference_commit` names `6fdfdc2`. Section 4.3. |
| `4fd6a0f` | Loader order | The head loader checks `missingQuantization` before the head depth. Section 4.3. |
| `3720f83` | Per-depth oracles | The pack's oracles for `mtp1` to `mtp7` are pinned. Section 4.3. |
| `6e37ffc` | Stock declaration | `mtp-head.manifest.json` ships serial (depth 0) again. Section 4.3. |

### 2.1 The engine fixes taken in #2

| Nemotron pull request | Fix |
|---|---|
| #11 | `live_goldens` is a list. `live_golden_speculative` is keyed by depth, then by live golden name. Calibration, the preflight and the measure script take one golden flag for each live golden. |
| #12 | The bench-worker phase-close drain settles before it reports `cache_memory`. |
| #13 | `setup.sh` exports the log label to the shard downloaders. |
| #15 | `setup.sh` verifies a read-only reference checkpoint and changes nothing. |
| #16 | The job account boundary: Seatbelt for the build, the CLI and the resident; preflight checks 7a and 7b; `RESIDENT_UP_LOG_DIR`. |
| #17 | Calibration runs three passes. |
| #18 | The score is the mean of the per-pair composites (`official_pair_combine: "mean"`). |
| #22 | The preflight accepts calibration file version 3 only, where decode is the decode window. |
| #23 | `yukon run` off a ranked box runs the local test at the declared depth. |
| #24 | The paged-attention kernel source is cached once per process. |
| #25 | The Metal device is queried once, without the window server. |
| #27 | `tools/resident-up.sh` removes the sandboxed resident's temporary directory. |
| #28 | The decode band allows 3 percent upward. |

Not taken: #13's goldens, seed and re-pin, and #14's arming, because they are
Nemotron's own material; #19, because #1 already ships the stock declaration
serial; #26, because the reference re-pin waits for the final engine. The
score artifact upload in `.github/workflows/benchmark.yml` was already the
same as Nemotron's.

`tools/golden-arming-patch.py` is this track's own tool. It moves to the list
shape: `--live` is repeatable, and each depth carries one oracle pin for each
live golden.

## 3. The pending design

This section records the design before the pack was pinned. Section 4.2
states what changed when the pack was pinned.

The design was built so that the published pack replaces the stand-in pins
and nothing else.

- **Sentinel.** A pin that names bytes nobody has published carries the exact
  string `BONSAI-MLX-PENDING-ORGANIZER`. Since #8, the target and head pins,
  the license and the compiled reference pin name the stand-in (temporary).
  Since the goldens pull request, the hidden correctness golden is a real pin,
  and the fixture carries no sentinel. `./setup.sh` refuses a
  sentinel pin by name. `tools/ranked-box-preflight.sh` refuses any
  `PENDING-ORGANIZER` string.
- **Geometry.** `fixtures/qwen35_9b_base_config.json` is the published
  `config.json` of `Qwen/Qwen3.5-9B` @ `c2022362`, verbatim. The contract's
  `target.geometry_source` pins its sha256 and bytes, and the tests assert the
  contract's geometry against it.
- **Packing.** The pack is assumed to use the Ternary Bonsai 2 27B packing.
  `fixtures/bonsai2_27b_config.json` stays as
  `target.pending_pack.packing_reference`. The
  transform validator derives 1,033 language-model tensors and 202 packed
  modules from the 9B geometry and that packing. The transform tests use a
  synthetic 9B pack config built from the packing reference and the backbone.
- **Golden loader.** The default reference pin is the compiled pin, which is
  the stand-in (temporary). The loader refuses every golden that names another
  checkpoint. The tests pass a synthetic pin, and one test proves that a
  sentinel pin refuses every golden.
- **Two decoders.** The MTP head and the DFlash 2 block drafter (section 4.7).
- **Prompts.** The planned live prompts are the Bonsai coding prompts
  `pool-9`, `pool-10` and `pool-11` at a 512-token seed. Against
  `Qwen/Qwen3.5-9B` @ `c2022362` and the Ternary Bonsai 2 27B pack, the merges,
  the tokenizer pipeline, the 26 shared added tokens (ids 248044–248069,
  including `<|im_start|>` and `<|im_end|>`), and the non-thinking
  chat wrap with `enable_thinking=False` match. The vocabulary is not
  identical: the 27B pack and the EigenLabs 4-bit stand-in add seven audio/TTS
  tokens (ids 248070–248076) that the Qwen snapshot lacks. The 27B chat
  template's thinking-default path injects a reasoning-effort system preamble,
  so callers must pass `enable_thinking=False` for the wrap to match Qwen.
  On the public local-iterate capture (the 512-token seed in
  `public-local-iterate.golden.json`) the three tokenizers give identical
  ids, and the user body re-tokenizes to the capture's ids exactly. A
  template re-wrap with `apply_chat_template` does not give the stored seed
  exactly. Rebuild seeds from the stored ids, not from a template re-wrap.
  `pool-9`, `pool-10` and `pool-11` are R2 tapes and are not checked here.
- **Reference commit.** `baseline_reference_commit` names `6fdfdc2`, the
  real-pack commit that holds the pack's runtime code. Before it, it named
  `3ef01fd`, the merge of #11 (the phase-close cancel in the vendored
  bench-worker), and `90ab29d`, the code commit of #8 (the stand-in (temporary)
  pin). It moves again with any later trusted-side code change.

## 4. The stand-in validation

A laptop run against a stand-in checkpoint proved the engine on the 9B
geometry. The stand-in is `EigenLabs/Qwen3.5-9B-MLX-4bit-mtp` @ `556dcba5`:
the same geometry, at 4 bits, with its own embedded MTP layer. The run used the
repository's local tools. It recorded a serial golden twice, and the two
recordings were byte-identical. It then ran the local test serial and at MTP
depths 1 to 7. Each run passed correctness and matched the serial free run
token for token.

No track golden comes from this laptop run. Its golden stayed on the laptop
and is not in this repository.

### 4.1 The stand-in pin (#8)

David 2026-10-09: "pin it and go, we will replace with a proper bonsai 9b in
later today". Pull request #8 makes the stand-in (temporary) the track pin, so
the track can run the full ranked and Yukon flow before the pack exists. No
weight is modified, mirrored or re-published.

- **Target.** `EigenLabs/Qwen3.5-9B-MLX-4bit-mtp` @ `556dcba5`, `qwen3_5`,
  affine 4-bit at group 64. `fixtures/reference_bonsai2_9b_2bit.sha256` pins
  13 files (not `.gitattributes`). `fixtures/qwen35_9b_standin_config.json` is
  its `config.json`, verbatim, and
  `fixtures/qwen35_9b_standin_tensor_inventory.json` is its tensor inventory
  (1,291 tensors, 3 shards), read off the shard headers.
  `golden_model_type` and the compiled `requiredGoldenModelType` are
  `qwen3_5`. The license is Apache-2.0, from the stand-in's model card.
- **Head.** The stand-in's own embedded layer: 31 `mtp.*` tensors in
  `mtp-00001.safetensors` (`mtp_head.source: "embedded"`).
  `fixtures/reference_bonsai2_9b_mtp_head.sha256` pins 3 files of the same
  revision: `config.json` (it declares `mtplx_mtp`), the index and the shard.
  Every head record is also a target record. The head still stages into its
  own directory, so the tools and the box layout do not change.
  `./setup-mtp-head.sh` sets `MLXFAST_REFERENCE_PINNED_SHARDS_ONLY=1`: the
  index names all three shards, and the head fetches only the shard it pins.
- **Transform.** A new family, `qwen35StandIn`
  (`Sources/MLXFastTransform/Qwen35StandInCheckpointValidation.swift`), takes
  `model_type` `qwen3_5`. It selects the 927 `language_model.` tensors, drops
  `vision_tower.` and `mtp.`, refuses any other namespace, and passes the
  tensors through byte for byte. The Prism Hadamard family stays for the pack.
- **Pack expectations.** The pack's model type, packing reference, quantization,
  Hadamard block and tensor counts moved to `target.pending_pack`. The tests
  that assert them read that block.

To swap in the pack: write the target and head pins from the published tree,
move `target.pending_pack` back into `target`, set `golden_model_type` and
`requiredGoldenModelType` back, and delete the stand-in family, its config and
its inventory. The goldens, the calibration and the reference tree are then
redone for the pack. Section 4.2 records the swap.

### 4.2 The real pack (`498baf2`)

prism-ml published the pack as `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit`. The
repository is private until prism-ml makes it public. Commit `498baf2` pins it
at `b94a52f3ce9047919dce00392a0fc6b8a7968abe`. No weight is modified, mirrored
or re-published.

- **Target.** `prism_hadamard_qwen35`, schema 2, affine 2-bit at group 128,
  FP16 scales and biases, 202 packed modules. One `model.safetensors` holds
  1,366 tensors (1,033 `language_model.` and 333 `vision_tower.`) in
  3,462,149,260 bytes. There is no index. `fixtures/reference_bonsai2_9b_2bit.sha256`
  pins 21 files. It does not pin `.gitattributes`, the `LICENSE` files or the
  `mtp/` folder. `fixtures/bonsai2_9b_config.json` is the pack's `config.json`,
  verbatim, and `fixtures/bonsai2_9b_tensor_inventory.json` is its tensor
  inventory. `target.stand_in` and `target.pending_pack` are gone: the pack's
  fields are in `target`. `golden_model_type` and `requiredGoldenModelType`
  are `prism_hadamard_qwen35`. The license is Apache-2.0, from the pack's
  `LICENSE` file.
- **Hadamard block 4096.** The packing is the Ternary Bonsai 2 27B pack's,
  except the block width. The 27B pack uses 1024. The 9B pack's
  `hadamard.json` and each of its 202 `modules` entries declare 4096. The
  transform validator's block constant is now 4096. Every packed input width of
  the 9B geometry is a multiple of 4096: the hidden size (4096), the MLP
  intermediate size (12288), the attention output input (16 x 256) and the
  linear-attention output input (32 x 128).
- **Head.** The pack ships its own trained drafter in the `mtp/` folder of the
  same revision: a standalone `qwen3_5_mtp` export of 15 FP16 tensors with no
  name prefix, 486,581,248 tensor bytes, unquantized, block size 2.
  `fixtures/reference_bonsai2_9b_mtp_head.sha256` pins its 2 files,
  `config.json` and `model.safetensors`, with paths relative to `mtp/`.
  `mtp_head.source` is `target_repository_folder`. `./setup-mtp-head.sh`
  fetches from the `mtp/` path of the revision. The pinned-shards path
  (`MLXFAST_REFERENCE_PINNED_SHARDS_ONLY`) had no other user and is deleted.
- **Head depth.** The head's `config.json` restates the trunk's
  `text_config`, so it declares `mtp_num_hidden_layers: 0`. The vendored
  loader required 1 to 4 and refused the head. It now accepts 0 to 4 in the
  metadata and reads the depth off the head's tensors: the `layers.<i>.`
  indices must be exactly `0..<n`, with n from 1 to 4. A non-zero declared
  depth must agree with the tensors. The head has one layer.
- **Tokenizer.** The pack's `tokenizer.json` is not the stand-in's (sha256
  `5f9e4d49…`, 12,807,982 bytes). The normalizer, the pre-tokenizer, the
  post-processor and the decoder are the same. It has the 26 added tokens
  248044 to 248069. It does not have the 7 audio and TTS tokens 248070 to
  248076 that the stand-in had. `chat_template.jinja` is byte-identical
  (`a4aee8af…`). `vocab_size` stays 248320.
- **Access.** `./setup.sh` reads `HF_TOKEN` and sends it as a bearer token to
  the huggingface.co source only. It never prints the token.
  `MLXFAST_REFERENCE_AUTH_HEADER` is the general form, and it wins over
  `HF_TOKEN`. `./setup-mtp-head.sh` uses the same path. Participants need read
  access to the repository until prism-ml makes it public.
- **Scoring.** `official_scoring_enabled` is `false`. It was armed for the
  stand-in, after box 2 was calibrated for the stand-in. It is armed again in
  its own change, after box 2 is calibrated for the pack.
- **Goldens.** The golden pins (`timed_prompt_pool`,
  `hidden_correctness_golden`, `live_golden_speculative`) and
  `baseline_reference_commit` do not move in `498baf2`. They still name the
  stand-in era. The goldens are recorded again on the ranked box against the
  pack. The new recordings get new R2 keys, because the bucket is append-only:
  an object at an existing key cannot be replaced. The public captures under
  `correctness_prompts/bonsai2-9b-mlx-v1/` are recorded again too. Until then,
  the golden loader refuses them against the pack's pin. Section 4.3 records
  the new recordings.
- **Deleted.** The stand-in family
  (`Sources/MLXFastTransform/Qwen35StandInCheckpointValidation.swift`), its
  config and tensor inventory fixtures, and the 27B packing reference
  (`fixtures/bonsai2_27b_config.json`).
- **Tests.** The transform tests read the pack's own config instead of a
  synthetic 9B config built from the 27B packing reference. Their packed test
  modules use 4096-wide inputs. The test of the stand-in route is deleted. The
  test that loads the shipped public captures records the loader's refusal as
  a known issue until the captures are recorded again. Commit `6a9e491`
  removes that wrapper.

### 4.3 The pack's goldens and the head dtype (`6a9e491` to `6e37ffc`)

All recordings ran on M5 box 2 against the pack, with
`record-correctness-golden` from benchd `b1a04d4`.

- **Public captures (`6a9e491`).** `public-local-iterate.golden.json`
  (sha256 `62d69f79…`, 10,670 bytes, 256 expected tokens) and
  `public-local-submit.golden.json` (`341f6095…`, 21,052 bytes, 1,024
  expected tokens) were recorded. Generations A and B are byte-identical.
  `Tests/MLXFastTests/GoldenTests.swift` pins both digests and sizes, and the
  known-issue wrapper is gone.
- **Head dtype (`6fdfdc2`).** The pack's `mtp/` head is FP16 and unquantized.
  The Prism trunk hands it FP32 activations. Each head matmul therefore
  promoted to FP32 and copied the head weights on every draft step. A round
  cost about four serial steps, and MTP ran slower than serial at every depth.
  Now an unquantized head computes in its own weight dtype: the activations
  that enter the head are cast to the head's weight dtype. No weight is cast.
  A quantized head does not change. The target verifies every token. The table
  gives benchd `local-iterate` decode-window tok/s on box 2.

  | Prompt | Serial | MTP depth 3 before | MTP depth 3 after |
  |---|---|---|---|
  | `pool-9` | 112.4 | 84.5 | 113.5 |
  | `pool-10` | 112.3 | 101.5 | 155.0 |
  | `pool-11` | 112.7 | 104.3 | 160.0 |

- **Serial tapes (`335d11c`).** `pool-9` (`a1dda9e2…`, 28,730 bytes),
  `pool-10` (`8fec95d9…`, 26,135 bytes) and `pool-11` (`15893728…`, 24,472
  bytes) were recorded with the resident wrapper, stop token 248046 and a
  128-token free run. Generations A and B are byte-identical. A recording on
  the staged reference tree `6fdfdc2` is byte-identical again.
  `hidden_correctness_golden` is the `pool-9` pin. `baseline_reference_commit`
  names `6fdfdc2`.
- **Loader order (`4fd6a0f`).** The `missingQuantization` check reads only
  metadata, so it runs before the head-depth check again. The order does not
  change a valid load.
- **Per-depth oracles (`3720f83`).** The oracles for `mtp1` to `mtp7` were
  recorded with `tools/record-goldens.sh --apply` on the reference tree
  `6fdfdc2`, with one resident for each depth. `benchd validate-golden`
  accepts each oracle. On the pack, each per-depth oracle is byte-identical
  to its serial tape: the MTP free run is exact at every depth. The separate
  keys stay, because the contract keys the oracles by depth.
- **R2 keys.** The bucket is append-only, so the pack's goldens are in a new
  folder, `correctness_prompts/bonsai2-9b-mlx-v1/b94a52f3/`, as
  `<name>.golden.json` and `<name>.mtp<D>.golden.json`. All 24 objects were
  uploaded and read back on 2026-10-13, and each sha256 and byte count matches
  its pin. `tools/record-goldens.sh` now writes each per-depth key in the
  folder of its serial tape.
- **Stock declaration (`6e37ffc`).** The Yukon validation commit `2f8c1a8` set
  `mtp-head.manifest.json` to depth 3. `6e37ffc` ships the stock declaration
  serial (depth 0) again, so a depth-3 submission differs from the stock tree
  by the manifest only.

### 4.4 The MTP round and the F16 constants (`1f45373`)

At depth 3 a round cost about 2.6 serial steps on `pool-10` (22.98 ms for
3.7 committed tokens, serial step 8.89 ms). The measurement used temporary
timers in the engine loop and the Metal command-buffer times on M5 box 2. The
round is GPU-bound. The GPU is idle for 0.4 ms to 0.6 ms of a round, at its
start. The three draft steps take about 3.9 ms, which is the bandwidth time of
the FP16 head and the 2-bit `lm_head`. The 4-row verify forward takes about
18.5 ms, 2.1 serial steps. A layer of the verify takes 0.50 ms, and a layer of
a serial step takes 0.25 ms.

Two costs make the verify expensive:

- **Constant casts.** `quantizedMM` casts the scales and biases to the
  activation dtype before it dispatches. The pack computes in FP32 and stores
  F16 scales and biases on all 202 packed modules. Thus each projection ran two
  cast kernels and then read the widened copies. This cost was in the serial
  step and in the verify.
- **4-row matmuls.** For 2 to 5 rows, MLX uses `qmv_wide`. On the pack's
  shapes it costs 1.7 to 3.2 times the one-row kernel: the dequantize and
  multiply work grows with the rows. Its arithmetic does not change.

`HadamardQuantizedLinear` now does the 2-bit, group-128 matmul of FP32
activations for 1 to 5 rows with two kernels that read the F16 constants in
place: MLX's `qmv_fast` for one row and `qmv_wide` with one tile for 2 to 5
rows. Only the constant loads change. The widening is exact, and the
arithmetic and its order are MLX's. Thus the output is bit-identical to
`quantizedMM`. No weight changes, and no copy is kept.
`Float16ConstantCastTests` checks the bit identity at the pack's widths, and a
change to the reduction order or to the scaling makes it fail. All 24
per-depth oracles (`pool-9`, `pool-10` and `pool-11`, depths 0 to 7) pass.

The table gives benchd `local-iterate` decode-window tok/s on box 2, the mean
of two alternating passes.

| Prompt | Serial before | Serial after | MTP depth 3 before | MTP depth 3 after |
|---|---|---|---|---|
| `pool-9` | 112.5 | 144.9 | 136.0 | 141.5 |
| `pool-10` | 112.6 | 144.9 | 155.7 | 162.6 |
| `pool-11` | 112.3 | 144.9 | 161.5 | 170.8 |

The serial step gains more than the round. Thus the depth-3 decode gain over
serial goes down, from 1.21 to 1.44 before to 0.98 to 1.18 after. The 4-row
verify now costs about 2.5 serial steps. That matmul work is the remaining
large cost of a round.

### 4.5 The MTP verify kernel (`9510d38`)

The 4-row verify matmuls ran a `qmv_wide` replica. On box 2 it cost 2.3 times
the one-row kernel at four rows. The box reads at 587 GB/s, and a register
loop does 7.64 T FMA/s. At four rows a 2-bit matvec is bound by the FP32 FMAs,
not by the weight read. For example, the gate projection reads 14.2 MB in
24 us, and its 4 x 50M FMAs take 26 us at the peak. For a whole verify, the
FMAs alone take 4.15 ms and the weight read takes 3.9 ms. Thus a 4-row verify
cannot cost near one serial step on this GPU.

`HadamardQuantizedLinear` now runs a multi-row kernel for 2 to 5 rows. Each
lane keeps the `qmv_fast` layout. It dequantizes each weight once and does one
FMA per weight per row, and one activation load serves 4 output rows. One row
still runs the `qmv_fast` replica, so the serial step does not change. The
2-to-5-row result is not bit-identical to the one-row kernel, as the
`qmv_wide` result was not. A bit-identical kernel must use the one-row
arithmetic, which costs 1.45 FMA slots per weight per row. The best one
(gate, four rows: 69.0 us) was not much faster than `qmv_wide` (70.7 us).

Microseconds per call at four rows, real pack weights:

| Projection | One row | `qmv_wide` replica | New |
|---|---|---|---|
| gate, up | 31.1 | 70.7 | 56.9 |
| down | 31.1 | 79.4 | 60.3 |
| `in_proj_qkv`, `q_proj` | 21.7 | 51.3 | 40.0 |
| `in_proj_z`, `out_proj`, `o_proj` | 12.3 to 14.7 | 26.3 to 28.2 | 22.6 to 23.7 |
| `k_proj`, `v_proj` | 7.0 to 7.3 | 14.2 | 11.3 |
| `lm_head` | 488.7 | 1440 | 1168 |

All 24 per-depth oracles pass. Three runs of `pool-9` at depths 0 and 3 give
one stream. On `pool-10` at depth 3, the median round goes from 21.6 ms to
19.5 ms. The table gives decode-window tok/s, the mean of two alternating
passes.

| Prompt | Serial before | Serial after | MTP depth 3 before | MTP depth 3 after |
|---|---|---|---|---|
| `pool-9` | 145.0 | 146.2 | 139.8 | 156.4 |
| `pool-10` | 145.0 | 146.6 | 161.5 | 181.2 |
| `pool-11` | 145.2 | 145.2 | 166.5 | 180.3 |

Before section 4.6, the Qwen 3.5 MTP drafter drafted at most four tokens
(`Qwen35InlineMTPAssistant.maximumDraftTokens`). Thus depths 5 to 7 ran as
depth 4. On every pool, depth 4 is slower than depth 3.

### 4.6 The draft-depth cap 6

David set the draft-depth cap of this track to 6 (2026-10-14).

- `Qwen35InlineMTPAssistant.maximumDepth` is 6. `maximumDraftTokens`, the
  draft-chain precondition and the allocation padding derive from it. The
  double-forward control stays at 1.
- The round driver refuses a fixed depth above the drafter cap. It does not
  clamp it. `CBv2MTPConfig.validateFixedDepth(for:)` throws
  `CBv2MTPDepthAboveDrafterCap`, and `RunnerEngineAssembly.makeEngine` calls
  it before the engine exists. Thus `free_decode_begin` fails before any timed
  work. On box 2, `--mtp-depth 7` fails with "the declared MTP depth 7 is
  above the drafter's draft-chain cap of 6", and depth 6 runs at depth 6 (a
  round commits up to 7 tokens).
- The fixture permits depths 1 to 6, and `tools/spec-declaration.sh` refuses a
  declared 7. The `mtp7` pins are gone.
- The multi-row verify kernel serves 2 to 7 rows. At five to seven rows it
  computes two output rows per simdgroup. The reference tree keeps the cap of
  4, so `tools/record-goldens.sh` now records from the checkout it lives in.
  The serial path does not change, so `baseline_reference_commit` stays
  `63dec40`.

Microseconds per call on box 2, real pack weights (two outputs per simdgroup,
one output, and `quantizedMM`):

| Projection | 6 rows: two / one / `quantizedMM` | 7 rows: two / one / `quantizedMM` |
|---|---|---|
| gate, up | 87.5 to 87.7 / 173.4 to 173.8 / 112.1 | 100.3 to 100.8 / 199.6 to 200.9 / 134.4 to 134.6 |
| down | 118.3 / 226.5 / 125.4 | 157.3 / 284.0 / 155.2 |
| `in_proj_qkv`, `q_proj` | 60.6 to 61.5 / 117.9 to 118.5 / 78.6 to 82.3 | 69.2 to 70.2 / 135.4 to 136.4 / 93.4 to 96.7 |
| `in_proj_z`, `out_proj`, `o_proj` | 33.2 to 35.1 / 62.8 to 64.6 / 42.1 to 50.6 | 37.8 to 40.3 / 71.5 to 73.3 / 53.8 to 57.5 |
| `k_proj`, `v_proj` | 14.2 to 14.9 / 18.2 to 20.0 / 20.9 to 22.1 | 15.3 to 15.7 / 22.0 to 22.5 / 20.8 to 23.4 |
| `lm_head` | 2053 / 3684 / 2624 | 2308 / 3788 / 3381 |
| Sum for one verify | 15.6 ms / 29.7 ms / 19.4 ms | 18.5 ms / 34.5 ms / 23.6 ms |

The per-depth oracles were recorded again at the true depths 1 to 6, twice,
with the public captures. Both sets are byte-identical to the pins, so no pin
and no R2 object changed: the free run is exact at every depth. All 21
oracles pass, and three runs of `pool-9` at depths 0 and 6 give one stream.

Decode-window tok/s, the mean of two alternating passes, with acceptance and
tokens per round:

| Prompt | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Depth 5, `quantizedMM` | Depth 6, `quantizedMM` |
|---|---|---|---|---|---|---|
| `pool-9` | 156.8 (0.752, 3.20) | 138.9 (0.657, 3.56) | 139.3 (0.647, 4.13) | 129.1 (0.589, 4.41) | 122.5 | 113.2 |
| `pool-10` | 178.7 (0.912, 3.66) | 165.3 (0.845, 4.27) | 159.3 (0.777, 4.74) | 142.5 (0.680, 4.92) | 146.3 | 129.0 |
| `pool-11` | 181.2 (0.949, 3.77) | 160.3 (0.808, 4.13) | 157.1 (0.777, 4.74) | 148.9 (0.715, 5.12) | 147.1 | 129.7 |

Depth 3 is the fastest on every pool. A round commits more tokens at depths 5
and 6, but the acceptance falls and the 6- and 7-row verify costs more.

### 4.7 The DFlash 2 drafter

The second decoder is the DFlash 2 block drafter. Its source is Prism's
`Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf` at commit `608e89c3` (sha256
`95149abd`, 1,574,233,856 bytes; FFN 12288). The first pin was the FFN-24576
file of commit `c507c837` (sha256 `e94df93f`), which the CUDA track also pins.
The vendored `DFlash2DraftModel` and
`Qwen35DFlash2Assistant` from the 27B seed run it. Only `config.json` and the
weights are specific to this drafter.

- The conversion. `tools/convert-dflash-gguf.py` (Python standard library only)
  writes the Q8_0 linears as MLX affine 8-bit group-32 linears without loss:
  q = byte XOR 0x80, scale = the block's FP16 scale, bias = -128 x scale, the
  mapping of MLX's own GGUF loader (`gguf_quants.cpp`). The scales and biases
  are FP32, so the drafter's matmuls run in FP32. The norms, the convolution
  base kernels and the selector codebooks are BF16. The output is
  1,933,845,738 bytes, sha256 `df74551b`, the same on every run. The
  converter refuses a `config.json` whose geometry differs from the GGUF
  header.
- FP16 overflows. An FP16 drafter gives NaN logits and no accepted token. The
  parameter dtype is BF16.
- The rope. The GGUF declares YaRN (factor 16 over an original context of
  16,384), and llama.cpp applies it, including the attention scale
  1 + 0.1 ln 16 on both rotations. The drafter's `config.json` states it in
  `rope_parameters`, and `DFlash2DraftModel` passes it to `initializeRope`.
  Without it, the accept rate on the public prompt at depth 15 is 0.38; with
  it, 0.87.
- The prefill. The drafter runs at decode only. The target taps its eight
  layers during the prompt forward, and the first block consumes them. The
  scored prefill (benchd's timed `prefill` verb) is 0.99 of serial at depth 15.
- The verify. A depth-D round verifies D + 1 rows. The multi-row kernel
  serves up to 7 rows (depth 6). Depth 7 and above take the stock
  `quantizedMM`. A round costs 27 ms at 4 rows, 40 ms at 7, 46 ms at 8, 52 ms
  at 9 and about 60 ms at 11 to 16 rows on box 2.

Decode-window ratio to serial with the FFN-24576 drafter (one local pass per
cell, box 2), with the accept rate:

| Depth | `pool-9` | `pool-10` | `pool-11` |
|---|---|---|---|
| 3 | 0.831 (0.781) | 0.933 (0.912) | 0.933 (0.912) |
| 5 | 0.862 (0.707) | 0.958 (0.816) | 0.988 (0.858) |
| 6 | 0.917 (0.754) | 0.992 (0.841) | 0.999 (0.841) |
| 7 | 0.888 (0.721) | 0.925 (0.764) | 1.082 (0.924) |
| 8 | 0.814 (0.669) | 0.812 (0.669) | 0.948 (0.809) |
| 10 | 0.692 (0.535) | 0.803 (0.647) | 0.908 (0.747) |
| 15 | 0.709 (0.357) | 0.846 (0.463) | 0.987 (0.538) |

The FFN-12288 drafter, the same measurement:

| Depth | `pool-9` | `pool-10` | `pool-11` |
|---|---|---|---|
| 6 | 0.889 (0.754) | 1.016 (0.795) | 1.059 (0.841) |
| 8 | 0.857 (0.669) | 0.899 (0.711) | 0.997 (0.809) |
| 15 | 0.656 (0.337) | 0.912 (0.498) | 0.975 (0.498) |

The dflash oracles recorded with the FFN-12288 drafter are byte-identical to
the pins.

## 5. Open items

1. Public access. The repository is private. Participants need `HF_TOKEN` with
   read access until prism-ml makes it public. The docs drop that requirement
   when it is public.
2. Closed (section 4.6): the drafter cap is 6, the contract permits 1 to 6,
   and the engine refuses a deeper fixed depth.

### 5.1 Items closed by the stand-in and the pack

The stand-in closed the first recording of the goldens: `pool-9`, `pool-10`
and `pool-11` were recorded on M5 box 2 with the engine's recorder tooling, A
and B byte-identical, uploaded to R2 under
`correctness_prompts/bonsai2-9b-mlx-v1/` and pinned in the fixture. Each
per-depth oracle (`mtp1` to `mtp7`) was recorded at its own depth with
`tools/record-goldens.sh` (benchd `record-correctness-golden --benchmark-spec
mtp:<D>`), A and B byte-identical, under its own R2 key
`<name>.mtp<D>.golden.json`. On the stand-in, the stock MTP verify was not
bit-exact with the serial forward on every prompt (pool-11 diverged at
free-run step 107 at every depth; a forced exact W4G64 verify moved the
divergence to pool-9 at step 12), so serial bytes could not serve as the depth
oracle. The same procedure was applied to the pack (section 4.3). On the pack,
each per-depth oracle equals its serial tape.

The pack closed four earlier items: its name, repository, revision, file pins
and tensor inventory; its MTP head and its pins; whether it carries its own
`mtp.*` block (it does not, and its head is the `mtp/` folder); and its
tokenizer and chat template (section 4.2). The tokenizer check of the live
prompts is closed too: the seeds of `pool-9`, `pool-10` and `pool-11` are the
stored ids, the largest id is 248069, so no seed uses the 7 audio and TTS ids
the pack's tokenizer does not have, and the tapes were recorded on the pack
from those ids.

Later commits closed four more items (section 4.3): the pack's serial tapes
and per-depth oracles, recorded, uploaded to R2 under new keys and pinned; its
public captures, recorded and pinned, with the test's known-issue wrapper
removed; the engine's runs against the pack, serial and MTP depths 1 to 6, in
the oracle recordings; and the re-pin of `baseline_reference_commit` to
`6fdfdc2`.
