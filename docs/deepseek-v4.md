# DeepSeek V4 engine

The DeepSeek V4 Flash engine, target path plus the DSpark drafter.

## Scope

- Production code is in `c/deepseek_v4.c`; the experimental public engine and
  session API is in `c/deepseek_v4.h`. The drafter lives in
  `c/deepseek_v4_dspark.inc`.
- Official sharded safetensors checkpoints load through shared `st.h`.
- Standard MXFP4 matrix multiplication uses shared `quant.h`.
- Unified `c/coli` routes `run`, `chat`, `serve`, and `web` to V4. Serving keeps
  the engine and caches warm across requests.
- **DSpark speculative decoding is built in** — three drafter stages read from
  the `mtp.<stage>.` tensors of the same checkpoint, verified against the exact
  target. It is opt-in: `c/coli` starts it disabled (`V4_MTP=0`, `V4_DRAFT=0`),
  and `--no-dspark` switches it off explicitly. Drafting is output-preserving
  within one backend; the acceptance rate is reported per run.
- Build targets are x86-64/aarch64 Linux and Windows/MSYS2. This fork only
  maintains **Linux x86-64**; see `AGENTS.md`.

Destroy every session before destroying its engine.

## Shared migration status

| Checkpoint path | Current implementation | Follow-up |
|---|---|---|
| Safetensors index/range reads | shared `st.h` | done |
| fmt7 standard MXFP4 matmul | shared `quant.h` | done |
| fmt7 resident rows16 expert cache | temporary V4-private layout | **TODO:** migrate after upstream exposes a resident rows16 API |
| fmt8 E4M3 + UE8M0 128x128 scales | shared `st_read_scale_f32` + `quant.h` `matmul_fp8` | done |

Only the rows16 resident-cache layout remains V4-private. Its
`TODO(upstream-fmt7-rows16)` marker names the shared API still needed before
that specialized cache layout can be removed.

## Memory policy

A typical checkpoint has 43 transformer layers, hidden size 4096, and 256
routed experts per sparse layer with top-k 6. Dense weights occupy about
6.27 GiB in total: 5.456 GiB of FP8 weights and expanded scales can move to
VRAM, while 0.810 GiB of BF16/f32/i64 tensors remains on the host. A resident
BF16 output head occupies about 0.99 GiB (1.06 GB). Routed-expert weights are
streamed and cached according to the RAM budget.

The planner reserves workspace and a minimum expert working set, then enables
dense/head residency and grows the expert cache when memory permits. Dense
residency is independent of DSpark and works with `--no-dspark` as well.

With the full drafter enabled, the planner books a DSpark reserve of
`V4_MTP_GB × 1e9 + 768 MiB` — 1.169 GiB at the `0.45` default — *before* it
sizes the expert cache, because the lazy first draft must not borrow its upload
peak from that cache. On the target machine that costs two expert-cache slots.

`--ram GiB` is a planner budget, not an OS-enforced limit. Without it, the
budget is derived from currently available OS memory.

For the 32 GiB target machine, scratch-reserve guidance, the reproducible
benchmark protocol, and the current before/after measurements are maintained in
[DeepSeek V4 tuning for 32 GiB](deepseek-v4-tuning-32gb.md).

## Download

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/DeepSeek-V4-Flash
```

A download can finish with a truncated shard even when the client reports
success. If `st.h` rejects a shard as out of bounds, compare every local shard
size with the Hugging Face repository before treating it as an engine failure.

## Build and use

```bash
cd c
make deepseek-v4
python ./coli run --model /path/to/DeepSeek-V4-Flash --ram 32 \
  "What is the capital of France?"
python ./coli chat --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli serve --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli web --model /path/to/DeepSeek-V4-Flash --ram 32
```

V4 chat uses native model markers. Native serving currently supports greedy
generation and one active KV slot; tools and grammar are rejected. Requests
re-prefill their context, while the process, weights, dense tensors, head, and
expert cache stay warm.

## Validation

The tiny safetensors fixture is generated locally, ignored, and not committed:

```bash
python -m pip install -r tools/requirements-deepseek-v4-tiny.txt
make deepseek-v4-tiny-check
```

This covers loading, teacher forcing, greedy decode, long/repeated sessions,
`--no-dspark` compatibility, and two requests through the persistent
`SUBMIT`/`DATA`/`DONE` protocol. The fixture contains **no `mtp.*` tensors**, so
it cannot exercise any DSpark path — a drafter change that is only green here is
untested.

For a real checkpoint:

```bash
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash \
  MEMORY_GB=32 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20
```

The oracle is target-only. DSpark on/off speed, acceptance and token-identity
evidence are measured separately with `c/tools/bench_v4.py --dspark`; the
current numbers are in [plans/07-head-dspark-vram.md](../plans/07-head-dspark-vram.md).

## Follow-ups

- Add non-greedy sampling and more serving slots.
- Add shared replacements for the two temporary private quant paths above.
- DSpark does not yet pay for itself in throughput — it loses against
  target-only on both the CPU and the CUDA path. That is a property of the
  drafter, not of its placement, and needs its own plan.
