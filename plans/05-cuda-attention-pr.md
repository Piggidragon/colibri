# Draft PR — Phase 05: CUDA KV + Flash-Sparse-MLA-Attention

**Branch:** `phase-05-cuda-attention`
**Basis:** `main` auf Phase-04-Merge `8b7d346`
**Plan:** [05 — CUDA: KV in VRAM + Flash-Kernel](05-cuda-attention.md)

## Zusammenfassung

Diese PR fügt einen opt-in CUDA-Brückenkopf für DeepSeek-V4 hinzu. Der
quantisierte KV-Cache wird auf ein einzelnes CUDA-Gerät gespiegelt, und die
Sparse-MLA-Attention dekodiert F32, native FP8/BF16 sowie TurboQuant 4/3/2 in
einem block-per-head Online-Softmax-Kernel. Dense-Projektionen und Experten
bleiben CPU-seitig. Bei CUDA-/Kernel-/Upload-Fehlern fällt der Engine-Pfad
einmalig geloggt auf die bestehende CPU-Implementierung zurück.

## Commits

- `25b1d9c` `build: opt-in CUDA for the V4 engine`
- `042e7d2` `feat: V4 CUDA backend — resident quantized KV`
- `2d2a19d` `feat: V4 CUDA flash sparse MLA attention kernel`

## Abnahme

- [x] CPU-Build ohne CUDA bleibt unverändert.
- [x] `make -C c test && make -C c check` grün; 356 Python-Tests.
- [x] Tiny-Fixture und Prefix-Reuse tokenidentisch.
- [x] CUDA 12.0 / GCC 12 / `sm_89`: Backend, Harness und V4-Binary kompilieren/linken.
- [x] Ohne GPU: `V4_VRAM=1` fällt sauber und tokenidentisch auf CPU zurück.
- [ ] CUDA-Harness auf echter RTX 4070: alle fünf Codecs, Heads `{1,8,64}`,
      Selektionen `{1,7,64,2048}`, Sinks, `-1`-Padding, OOB und OOM.
- [ ] Full-Checkpoint: `V4_VRAM=1` gegen CPU tokenidentisch.
- [ ] `nvidia-smi`-Messung des KV-Footprints.
- [ ] CPU-A/B und CUDA-Durchsatz/PCIe-Transfer messen.

## Offener Status

Die PR ist absichtlich noch ein Draft. Die Entwicklungs-VM hat kein NVIDIA-
Gerät; die letzten vier Punkte sind auf der Zielmaschine auszuführen. Bis dahin
bleibt Phase 05 in `AGENTS.md` und der Plantabelle `offen`.

## Testkommandos auf der Zielmaschine

```bash
make -C c v4-cuda-test CUDA=1 CUDA_ARCH=sm_89
make -C c deepseek-v4 CUDA=1 CUDA_ARCH=sm_89
V4_VRAM=1 ./c/deepseek_v4 ...
nvidia-smi
```

Die gemessenen Zahlen und der finale PR-Link gehören nach der GPU-Abnahme in
den `## Ergebnis`-Block von [Plan 05](05-cuda-attention.md).
