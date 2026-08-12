# V4 checkpoint formats

The only supported input is the official
`deepseek-ai/DeepSeek-V4-Flash-0731` safetensors checkpoint. Do not convert or
repack it for normal use.

The runtime reads its native tensor layouts directly:

| Tensor family | Stored representation |
|---|---|
| Dense projections and shared experts | FP8 E4M3FN with UE8M0 128×128 scales |
| Routed experts | native FP4 rows with their checkpoint scales |
| Output head and selected norms/metadata | BF16, f32, or integer tensors as declared by the checkpoint |
| DSpark | `mtp.0`–`mtp.2` tensors in the same checkpoint |

`packed_rows8` is an AVX2 repack made after loading; it is not an on-disk
format. The `native` KV codec is also not a checkpoint conversion: it stores the
already quantized FP8/BF16 main KV and FP4 indexer representation without the
runtime's former f32 expansion, then reconstructs the same values bit-exactly.

Any alternate format, weight conversion, or lossy KV codec is an experiment and
must have its own validation and opt-in switch. See
[ENVIRONMENT.md](ENVIRONMENT.md) for `V4_KV` choices.
