# Contributing

This fork is intentionally specific: DeepSeek-V4-Flash-0731 on Linux x86-64,
optimized for the i5-13400F / 32 GiB DDR4-3200 / headless RTX 4070 target.
Read [`AGENTS.md`](AGENTS.md) and
[the reference plan](plans/00-reference.md) before changing code or plans.

Keep one plan per branch and one focused PR per plan. Do not push or merge
without the maintainer's instruction.

Every code change needs:

```sh
make -C c test && make -C c check
make -C c deepseek-v4-tiny-check
```

The Tiny fixture has no `mtp.*` tensors. Changes to DSpark, CUDA placement,
long-context behavior, cache policy, or throughput also require the full
checkpoint and a report with the command, commit, hardware state, cache state,
tokens, RSS/VRAM, and measured result. The final measurement matrix is
[Plan 15](plans/15-final-validation-and-tuning.md).

Model semantics come before speed. Lossy behavior is opt-in, invalid environment
input falls back safely, and each new user-facing `V4_*` or `COLI_V4_*` setting
must be recorded in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
