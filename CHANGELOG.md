# Changelog

## Unreleased

`Piggidragon/colibri` is now a deliberately divergent, Linux-x86-64-only fork
for `deepseek-ai/DeepSeek-V4-Flash-0731` on the documented i5-13400F / 32 GiB
DDR4-3200 / headless RTX 4070 machine.

### Removed

- GLM, Inkling, Kimi, OLMoE, and their formats, converters, fixtures, backends,
  and platform support.
- Metal, Vulkan, HIP, Windows, macOS, and the documentation that represented
  them as supported.

### Changed

- The public entry points, README, environment reference, API guide, and tuning
  record now describe the V4-only engine, its in-checkpoint DSpark drafter,
  CUDA/RAM placement, and V4 validation workflow.
- The phase plans are the authoritative implementation history and performance
  record; [Plan 15](plans/15-final-validation-and-tuning.md) defines the
  outstanding full-checkpoint acceptance matrix.
