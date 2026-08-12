# DeepSeek V4 Flash engine

`c/deepseek_v4.c` is the only inference engine in this fork. It opens the
official `deepseek-ai/DeepSeek-V4-Flash-0731` safetensors checkpoint directly;
the target model and the three DSpark stages are one checkpoint, not two model
directories.

## What is loaded

- 43 V4 layers with a 4096-wide hidden state and 256 routed experts (top-k 6).
- Native FP8 dense weights and FP4 expert weights; no conversion is required.
- DSpark tensors at `mtp.0`, `mtp.1`, and `mtp.2`. The Tiny fixture has no
  `mtp.*` tensors, so only the full checkpoint can validate drafter behavior.
- The default `native` KV codecs store the checkpoint's existing quantized
  representation compactly and reconstruct it bit-exactly. TurboQuant codecs
  remain explicit lossy experiments.

The public runtime supports Linux x86-64 only. It has a dependency-free CPU
path and an optional CUDA path for the headless RTX 4070.

## Build

```bash
make -C c deepseek-v4

# Target CachyOS CUDA build. NVCC_CCBIN is necessary only when CUDA rejects
# the system compiler.
make -C c deepseek-v4 CUDA=1 NVCC_CCBIN=g++-14
```

`ARCH=native` is the default for both `make -C c deepseek-v4` and `make -C c check`
(`check` builds the engine then runs the test suite against it, same ARCH as any
other invocation) — set `ARCH=x86-64-v3` explicitly if you need a binary portable
across different x86-64 hosts, and don't compare timings across different ARCH values.

## Direct use

```bash
c/deepseek_v4 /path/to/deepseek-v4-flash "Hello" \
  --memory-gb 24 --max-tokens 64

c/deepseek_v4 /path/to/deepseek-v4-flash --prompt-file prompt.txt \
  --system "Answer precisely." --thinking --memory-gb 24
```

The direct CLI uses the V4 chat template. `--raw-prompt` bypasses it,
`--stop-sentence` stops at the first sentence terminator, and `--no-dspark`
disables verified drafting for that request.

## Persistent interfaces

`c/coli chat`, `c/coli serve`, and `c/coli web` launch the same V4 engine and
keep the model process warm. `serve` and `web` provide a local OpenAI-compatible
HTTP API; its V4 limits and examples are in [api.md](api.md).

```bash
cd c
./coli chat --model /path/to/deepseek-v4-flash --ram 24 --ctx 32768
./coli serve --model /path/to/deepseek-v4-flash --ram 24 --ctx 32768
```

The V4 server is greedy and serves one active context. It accepts ordinary text
chat/completion requests and rejects unsupported tools explicitly rather than
pretending to execute them.

## Validation

```bash
make -C c test && make -C c check
make -C c deepseek-v4-tiny-check

make -C c deepseek-v4-oracle MODEL=/path/to/deepseek-v4-flash \
  MEMORY_GB=24 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20
```

The first two commands are required on every change. The Tiny fixture verifies
the regular target path, while the full-checkpoint oracle is needed for the
actual model. A DSpark change additionally needs a real-checkpoint run that
records proposals, acceptance, and target-token equality.

For the target hardware profile, placement reports, and benchmark protocol,
see [deepseek-v4-tuning-32gb.md](deepseek-v4-tuning-32gb.md).
