# DeepSeek V4 environment variables

This is the V4 runtime control surface for the Linux x86-64 fork. Invalid
numeric values fall back to the documented default; values outside a knob's safe
range are clamped. The `coli` launcher forwards explicit environment settings to
`deepseek_v4`.

Use CLI flags for ordinary operation (`--model`, `--ram`, `--ctx`, `--ngen`).
Environment variables are for fixed profiles and controlled A/B runs.

## Launcher and local API

| Variable | Default | Effect |
|---|---:|---|
| `COLI_MODEL` | unset | Model directory for `c/coli`; equivalent to passing `--model`. |
| `COLI_ENGINE` | automatic | Explicit path to the V4 executable for custom install layouts. |
| `COLI_API_KEY` | unset | Bearer key required by `coli serve` and `coli web` when set. |
| `COLI_MODEL_ID` | launcher default | Model name reported by the HTTP API. |
| `COLI_MAX_QUEUE` | `8` | Maximum requests waiting for the one active V4 context. |
| `COLI_QUEUE_TIMEOUT` | `300` | Seconds a request may wait in that queue. |
| `COLI_ALLOWED_HOSTS` | unset | Comma-separated extra host names accepted by the HTTP host-header guard. |

## Runtime and memory

| Variable | Default | Effect |
|---|---:|---|
| `CTX` | `4096` | Context capacity. On V4 it also sizes two full prefill-state buffers, so larger contexts sharply reduce available expert-cache RAM. |
| `RAM_GB` | automatic | RAM budget used by persistent `coli chat`/`serve`/`web`. The direct binary equivalent is `--memory-gb`. |
| `NGEN` | `1024` in serve mode | Maximum generated tokens for the persistent engine. The direct binary uses `--max-tokens` (default 128). |
| `V4_SCRATCH_MB` | `512` | Planner scratch reserve, clamped to 64–4096 MiB. `128` is the measured starting value for the target profile; it is a reservation, not an allocation limit. |
| `V4_PREFILL_CHUNK` | `0` | Maximum tokens per prefill chunk. `0` keeps the full prompt as one chunk. A positive value bounds S-scaled activation memory; use it for the 256k/1M experiments only after a token A/B. |

## CUDA and placement

Requires a `make -C c deepseek-v4 CUDA=1` build.

| Variable | Default | Effect |
|---|---:|---|
| `V4_VRAM` | `0` | `1` enables the V4 CUDA placement planner. It prioritizes attention KV, movable dense FP8, head, then DSpark; each stage falls back to RAM if it cannot upload. |
| `V4_VRAM_LIMIT_MB` | none | Artificially limits planner-visible free VRAM for rehearsing a smaller card. |
| `V4_VRAM_RESERVE_MB` | automatic | Extra GPU reserve. Without it the planner reserves `free/8`, clamped to 256 MiB–1 GiB. |
| `V4_VRAM_FAIL_AT` | none | Test-build-only failure injection: `kv`, `dense`, `head`, or `dspark`. Requires `make ... TEST_HOOKS=1`; never use in a normal run. |

## KV codec

| Variable | Default | Effect |
|---|---:|---|
| `V4_KV` | `native` | Main attention-KV codec: `f32`, `native`, `turbo4`, `turbo3`, or `turbo2`. `native` reconstructs the model's existing FP8/BF16 representation bit-exactly. Turbo modes are lossy and opt-in. |
| `V4_KV_INDEX` | `native` | Lightning-indexer codec. Same values; `native` is the model's existing FP4 representation and is bit-exact. |

## Expert cache and storage

| Variable | Default | Effect |
|---|---:|---|
| `V4_PIN_SLOTS` | automatic | Absolute number of pinned expert-cache slots per layer. Wins over `V4_PIN_FRACTION` when both are set. |
| `V4_PIN_FRACTION` | automatic | Fraction of available slots pinned per layer. |
| `V4_PIN_RAMP_REQUESTS` | build default (24) | Number of requests over which pins are introduced. `0` disables the ramp. |
| `COLI_V4_AUTOPIN` | on | Seed pins from recorded expert usage when available. |
| `COLI_V4_PREWARM` | off | Preload selected pins at engine start. |
| `COLI_V4_SAVE_USAGE` | off | Persist usage data used by the V4 cache policy. |
| `COLI_V4_DIRECT` | off | Use aligned `O_DIRECT` reads for expert loads when supported. Measure it; DRAM-less NVMe can prefer buffered I/O. |
| `COLI_MODEL_MIRROR` | unset | One or more byte-identical checkpoint copies separated by `,` or `;`. The V4 mirror reader distributes expert reads and falls back to the primary on error. |
| `COLI_DISK_WEIGHTS` | measured | Optional positive drive weights for the mirror reader, e.g. `9,3`. |
| `COLI_V4_EXPERT_PREFETCH` | off | Experimental expert prefetch control; only enable as a measured A/B. |

## DSpark and speculation

All of these preserve target verification, but can reduce throughput when the
drafter has poor acceptance. The launcher intentionally defaults to target-only
decode (`V4_MTP=0`, `V4_DRAFT=0`).

| Variable | Default | Effect |
|---|---:|---|
| `V4_MTP` | `0` | Enable the three-stage DSpark drafter from `mtp.0`–`mtp.2`. |
| `V4_DRAFT` | `0` | Maximum speculative target/draft proposal depth. `3` is the normal DSpark A/B value. |
| `V4_MTP_DRAFT` | `3` | Maximum DSpark proposal depth. |
| `V4_MTP_GB` | `0.45` | DSpark planner reserve in decimal GB, plus its fixed 768 MiB margin. |
| `V4_MTP_MISS` | `96` | Miss threshold used by the DSpark guard. |
| `V4_MTP_MIN` | `3` | Minimum accepted DSpark proposal length. |
| `V4_MTP_CONF` | `0.55` | DSpark confidence threshold. |
| `V4_MTP_NOISE` | disabled | Experimental DSpark noise control. |
| `V4_MTP_LOG` | off | Emit DSpark diagnostics. |
| `V4_NGRAM` | `0` | Prompt-lookup draft depth. |
| `V4_MTP_PARTIAL_KEEP` | off | Keep a DSpark partial acceptance; diagnostic/experimental. |
| `V4_NGRAM_PARTIAL_KEEP` | off | Keep a prompt-lookup partial acceptance; diagnostic/experimental. |
| `COLI_V4_MARKOV_SPEC` | off | Experimental Markov drafter. |
| `COLI_V4_MARKOV_BLOCK` | implementation default | Markov proposal block size. |
| `COLI_V4_MARKOV_KEEP` | off | Keep a partial Markov proposal. |

## CPU placement and diagnostics

| Variable | Default | Effect |
|---|---:|---|
| `V4_OMP_CORES` | `all` physical cores | V4 OpenMP team selection: `all`, `perf`, or an explicit core count. On the i5-13400F, `perf` selects the six P-cores only when the Linux topology exposes them. Benchmark before adopting it. |
| `OMP_NUM_THREADS` | unset | Highest-priority explicit OpenMP team size. Overrides V4's automatic topology choice. |
| `COLI_NO_OMP_TUNE` | off | Disable V4's OpenMP topology tuning. |
| `V4_PREFIX_LOG` | off | Log persistent-prefix reuse. |

The test-only `V4_VRAM_FAIL_AT` and the experimental knobs are not profile
defaults. For the operating profile and why particular values are chosen, see
[deepseek-v4-tuning-32gb.md](deepseek-v4-tuning-32gb.md).
