# DeepSeek V4 tuning for 32 GiB

This is the measurement record and operating profile for
DeepSeek-V4-Flash-0731 on the target i5-13400F, 32 GiB DDR4-3200, headless RTX
4070 machine. Estimates are labelled as estimates; blank measurements are not
silently filled from planner arithmetic.

## Current recommendation

Start the V4 engine with:

```bash
V4_SCRATCH_MB=128 CTX=32768 ./deepseek_v4 \
  /path/to/DeepSeek-V4-Flash --prompt-file /path/to/prompt.txt \
  --memory-gb 28 --max-tokens 64
```

- `--memory-gb 28` must remain below the machine's current `MemAvailable` to
  select the explicit-budget branch. `--memory-gb 32` is usually ineffective
  on a 32 GiB host and leaves too little real headroom when it does fit.
  `RAM_GB` is read only by the `SERVE=1` launcher and has no effect in this
  one-shot CLI command.
- `V4_SCRATCH_MB=128` lowers a planner reserve by 384 MiB. It does not constrain
  allocations, so an undersized value makes the plan optimistic rather than
  preventing an out-of-memory condition.
- Set `CTX` to the largest context the workload actually needs. In addition to
  the full attention-cache capacity, the current full-prefill path allocates
  two state buffers totalling `CTX * hc_mult * hidden_size * 4 * 2` bytes. With
  the paper geometry (`hc_mult=4`, `hidden_size=4096`), that is 4 GiB at 32k
  and 16 GiB at 128k. Use 128k only for workloads that need it; the planner now
  accounts for these buffers before assigning the expert cache.

The default remains unchanged: without `V4_SCRATCH_MB`, the reserve is 512 MiB.
Numeric values are clamped to 64–4096 MiB, and malformed values fall back to the
default.

## VRAM tiers (12 GiB RTX 4070, `CUDA=1 V4_VRAM=1`)

With the CUDA build and the card otherwise idle, add `V4_VRAM=1` to the command
above (`CUDA_HOME=/opt/cuda` on Arch/CachyOS, see
[plans/09-arch-cachyos.md](../plans/09-arch-cachyos.md) Commit 3, not yet
built). Measured against the full checkpoint, `--memory-gb 24`,
`CTX=131072`, `V4_KV=native`:

```
vram_tiers free=11.46GiB reserve=1.00GiB kv=vram(0.39GiB) dense=vram(5.46GiB) head=vram(0.99GiB) dspark=ram(0.00GiB) used=6.83GiB
```

Adding `V4_MTP=1 V4_DRAFT=3`:

```
vram_tiers free=11.46GiB reserve=1.00GiB kv=vram(0.39GiB) dense=vram(5.46GiB) head=vram(0.99GiB) dspark=vram(0.32GiB) used=7.15GiB
```

Both fit inside the 11.46 GiB budget with headroom to spare; DSpark on top
still leaves `target_cache=4.28GiB` (8 expert-cache slots) at this context and
RAM budget — well under the 10.70 GiB upper bound derived in
[plans/00-reference.md](../plans/00-reference.md). The planner (Commit 1 of
[plans/08-vram-planner.md](../plans/08-vram-planner.md)) prints this
`vram_tiers` line next to `ram_tiers` on every start; treat a run that reports
`kv=ram` at this budget as a sign something else on the GPU already claimed
memory, not as a planner bug.

`V4_VRAM_LIMIT_MB=<n>` clamps the free VRAM the planner sees, useful for
rehearsing a smaller card. Confirmed against the real checkpoint: at
`V4_VRAM_LIMIT_MB=6144` the (higher-priority) attention-KV mirror still lands
on the device while dense — the single largest tier at 5.46 GiB — falls back
to RAM and head stays resident; degrading in that order, not crashing, is the
point of the priority list in plans/08-vram-planner.md. `V4_VRAM_FAIL_AT=kv|
dense|head|dspark` (`COLI_V4_TEST_HOOKS` builds only) forces a stage's device
upload to fail at runtime; all four were confirmed to reproduce the exact
`V4_VRAM=0` output for the same prompt.

The exclusive-ownership step that would additionally remove the attention-KV
host-side buffer once its device mirror is confirmed (the RAM win noted in
[plans/00-reference.md](../plans/00-reference.md)'s phase table) is not built
yet — the host copy stays allocated alongside the device one either way, so
this profile currently trades VRAM for compute locality, not for RAM headroom
on the KV tier specifically.

## Measurement status

The full checkpoint **is** available on the target machine
(`~/Services/models/colibri/deepseek-v4-flash`), and the Tiny fixture is
regenerable from the pinned CPU-only requirements — phase 06 regenerated it with
`torch==2.13.0+cpu`, `transformers==5.14.1`, `safetensors==0.8.0`. An earlier
version of this section claimed both were missing; that is no longer true and
was never a reason to skip a measurement.

The four-run A/B table below was nevertheless never filled in, because phases 02
and later overtook it. It is still the right shape for a phase-01 re-run:

| Run | `ram_tiers` | Status |
|---|---|---|
| `main`, automatic RAM, 512 MiB scratch | — | not recorded |
| phase branch, automatic RAM, 512 MiB scratch | — | not recorded |
| phase branch, `--memory-gb 28`, 512 MiB scratch | — | not recorded |
| phase branch, `--memory-gb 28 V4_SCRATCH_MB=128` | — | not recorded |

The measurements that **do** exist against the full checkpoint are recorded per
phase, not here: dense residency and the 36 → 46 expert-slot delta in
[plans/06-dense-vram.md](../plans/06-dense-vram.md), head and DSpark placement
with the four-run throughput and acceptance table in
[plans/07-head-dspark-vram.md](../plans/07-head-dspark-vram.md).

Before resident-tier decisions, the expected planner deltas are approximately
+2.6 GiB from replacing the automatic system reserve with the explicit 28 GiB
budget, plus 0.375 GiB from the smaller scratch reserve. These are acceptance
expectations, not measured results. A context change also changes the much
larger full-prefill state reserve, and crossing a dense/head residency threshold
can make the final expert-cache delta non-additive. Expert capacity advances in
whole per-layer slots, so the printed `target_cache` delta is quantized.

The checkpoint geometry should also be recorded with the re-run:

```bash
python3 -c "
import collections,json,sys
c=json.load(open(sys.argv[1]+'/config.json'))
r=c['compress_ratios']
print('layers', len(r), collections.Counter(r))
print('sliding_window', c['sliding_window'], 'head_dim', c['head_dim'])
" /path/to/DeepSeek-V4-Flash
```

Expected: 43 layers split into 2 ratio-0, 21 ratio-4, and 20 ratio-128 layers,
with `sliding_window=128` and `head_dim=512`. The f32 KV estimate derived from
that geometry is about 13.4 KB per context token. With the default
`V4_KV=native` codec since phase 03, the stored KV is smaller by 3.5× (main) and
7.5× (indexer) at bit-identical values.

## Reproducible benchmark

Build the engine, then run each fixed prompt profile once cold and once warm.
Use 32k for the short and medium operating-profile series; the checked-in long
prompt is roughly 64k tokens and therefore uses a deliberate 128k stress run:

```bash
make -C c deepseek-v4
python3 c/tools/bench_v4.py --model /path/to/DeepSeek-V4-Flash \
  --profile short --cache-state cold --ctx 32768
python3 c/tools/bench_v4.py --model /path/to/DeepSeek-V4-Flash \
  --profile short --cache-state warm --ctx 32768
python3 c/tools/bench_v4.py --model /path/to/DeepSeek-V4-Flash \
  --profile medium --cache-state warm --ctx 32768
python3 c/tools/bench_v4.py --model /path/to/DeepSeek-V4-Flash \
  --profile long --cache-state warm --ctx 131072
```

Repeat the medium and long commands with `--cache-state cold`. The profiles
materialize fixed checked-in text at roughly 500, 8k, and 64k tokens; the result
records the actual tokenizer count and the materialized SHA-256. Target-only
mode is the default so decode throughput is comparable with the memory-bandwidth ceiling.
Use `--dspark` for a separate speculative-decoding series; it selects the full
MTP path with `V4_MTP=1 V4_DRAFT=3` and rejects a result with no speculative
attempts instead of labelling it as DSpark. Pass additional V4 knobs as
`--env V4_NAME=value`.

The 32k context is the operating recommendation and therefore the primary
benchmark series. The long-profile result characterises the separate 128k
stress configuration; do not mix its 16 GiB state reserve into the 32k
before/after comparison.

A cold run writes `3` to `/proc/sys/vm/drop_caches`; run `sudo -v` first so the
harness can perform only that privileged write through non-interactive `sudo`.
A warm run first executes an unmeasured short generation with the identical
prompt and environment. The harness disables usage-history writes so warm-up
cannot change the next process's placement policy.

Each successful invocation prints one summary line and appends one JSON object
to `bench/<date>-<revision>.jsonl`. The record includes revision/dirty state,
cache state, prompt hash and token counts, all explicit `V4_*` and `COLI_V4_*`
settings, TTFT, prefill/decode rates, expert counters, disk GB per model-forward
token, RAM tiers, KV capacity, thread count, and DSpark acceptance when enabled.

## Acceptance run

On the target machine, complete this table from the JSONL records and paste the
exact four `ram_tiers` lines above. Compare runs at identical `CTX`; record any
resident-tier transition rather than treating the pre-tier budget delta as an
expert-cache measurement. The default Tiny-fixture tokens must not change. Then
run:

```bash
make -C c test && make -C c check
make -C c deepseek-v4-tiny-check
```

Do not publish a target-only decode result materially above the approximately
13 tok/s DDR4 bandwidth ceiling without first auditing token counting, timing
intervals, cache state, and whether DSpark was enabled.
