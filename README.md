# colibri — DeepSeek V4 Flash on one machine

This is a deliberately narrow fork of [JustVugg/colibri](https://github.com/JustVugg/colibri).
It runs exactly one checkpoint —
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
— on one target system:

| Component | Target |
|---|---|
| CPU | Intel i5-13400F (6 P-cores + 4 E-cores, 16 threads, AVX2) |
| RAM | 32 GiB dual-channel DDR4-3200 |
| GPU | headless RTX 4070, 12 GiB |
| Storage | 1 TB Gen4 NVMe for the checkpoint; optional 512 GB Gen3 SSD mirror |
| OS | CachyOS / Linux x86-64 |

The checkpoint stays in native safetensors form: FP8 dense weights, FP4 routed
experts, and the three built-in DSpark stages at `mtp.0`–`mtp.2`. There is no
conversion step and no separate drafter download.

This is not a general model runner, a multi-platform project, or an upstream
release. GLM, Inkling, Kimi, OLMoE, Metal, Vulkan, HIP, Windows, and macOS were
intentionally removed. The remaining code and documentation optimize for the
machine above without silently changing model precision or router semantics.

## Quick start

Clone the fork, build the CPU engine, and point it at the full checkpoint:

```bash
git clone <your-fork-url> colibri
cd colibri
make -C c deepseek-v4

c/deepseek_v4 ~/Services/models/colibri/deepseek-v4-flash \
  "Explain cache locality in one paragraph." --memory-gb 24 --max-tokens 64
```

The direct binary is the most transparent way to run a one-shot request. It
uses V4's chat template by default; `--raw-prompt` bypasses it and `--thinking`
uses the model's thinking prefix.

For the normal persistent interface, use the included launcher:

```bash
cd c
./coli chat  --model ~/Services/models/colibri/deepseek-v4-flash --ram 24 --ctx 32768
./coli serve --model ~/Services/models/colibri/deepseek-v4-flash --ram 24 --ctx 32768
./coli web   --model ~/Services/models/colibri/deepseek-v4-flash --ram 24 --ctx 32768
```

`chat`, `serve`, and `web` keep the model, dense tensors, head, and expert
cache warm. The HTTP server is OpenAI-compatible, greedy, and currently has one
active V4 context; requests with tools are rejected explicitly.

## RTX 4070 build and profile

CUDA is optional. On the target CachyOS installation the toolkit is under
`/opt/cuda`; its GCC compatibility may require the installed GCC 14:

```bash
make -C c deepseek-v4 CUDA=1 NVCC_CCBIN=g++-14

V4_VRAM=1 V4_SCRATCH_MB=128 CTX=32768 \
  c/deepseek_v4 ~/Services/models/colibri/deepseek-v4-flash \
  "Write a short C function that clamps an integer." \
  --memory-gb 24 --max-tokens 64
```

`V4_VRAM=1` lets the runtime planner place the attention KV, movable FP8 dense
weights, output head, and (if enabled) DSpark on the GPU in priority order. A
failed or insufficient upload falls back to RAM instead of changing output.
The 4070 must be headless while running V4: the planner assumes about 11.7 GiB
of usable VRAM after driver reserve.

For this 32-GiB host, start at `CTX=32768`, `--memory-gb 24`, and
`V4_SCRATCH_MB=128`. A 128k context reserves two full prefill-state buffers
(about 16 GiB on the paper geometry), so it has substantially less room for the
expert cache. Use a larger context only when the workload needs it.

The bit-exact native KV codec is the default. `V4_KV=turbo3` is lossy and remains
an explicit long-context experiment. DSpark and prompt-lookup speculation are
also opt-in through the launcher because their end-to-end payoff still needs the
final full-checkpoint measurements:

```bash
V4_MTP=1 V4_DRAFT=3 ./c/coli chat \
  --model ~/Services/models/colibri/deepseek-v4-flash --ram 24 --ctx 32768
```

See [the target-machine tuning record](docs/deepseek-v4-tuning-32gb.md) for the
measured placement reports and [the environment reference](docs/ENVIRONMENT.md)
for every supported V4 knob.

## Storage and performance expectations

At most six of 256 experts are selected per MoE layer. The cache keeps selected
experts in RAM and streams misses from the checkpoint. A full expert record is
about 13.4 MB; 43 layers therefore move roughly 3.4 GB of expert weights per
generated token before cache hits and DSpark unioning. Dual-channel DDR4-3200
puts an approximate 13 tok/s upper bound on the memory-resident path. This is a
bandwidth bound, not a benchmark claim.

The optional second SSD may hold a byte-identical copy of the checkpoint. The
V4 mirror reader splits read-only expert loads across both drives; one drive is
always the supported default. Use `COLI_MODEL_MIRROR` only after confirming the
single-drive run, and measure `COLI_V4_DIRECT=1` on the actual DRAM-less drives.

## Verify changes

The synthetic Tiny fixture is deterministic and checks the target forward path,
but it contains no `mtp.*` tensors. It cannot validate a DSpark change.

```bash
make -C c test && make -C c check
make -C c deepseek-v4-tiny-check
```

The final hardware acceptance — full checkpoint, CPU/CUDA placement, DSpark,
chunked prefill, and one versus two drives — is specified in
[Plan 15](plans/15-final-validation-and-tuning.md). The implementation roadmap,
hardware budget, and measurements live in [`plans/`](plans/).

## Documentation

- [V4 engine details](docs/deepseek-v4.md)
- [32 GiB + RTX 4070 tuning record](docs/deepseek-v4-tuning-32gb.md)
- [V4 environment variables](docs/ENVIRONMENT.md)
- [V4-only API and web server](docs/api.md)
- [Project plans and reference budget](plans/00-reference.md)

## License

Apache-2.0. The DeepSeek checkpoint has its own Hugging Face license and terms;
read and accept them before downloading or using it.
