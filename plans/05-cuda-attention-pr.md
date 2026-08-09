# PR #7 — Phase 05: CUDA KV + Flash-Sparse-MLA-Attention

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
- [x] `make -C c test && make -C c check` grün; im finalen `check` 358 Python-Tests.
- [x] Tiny-Fixture und Prefix-Reuse tokenidentisch.
- [x] CUDA 12.0 / GCC 12 / `sm_89`: Backend, Harness und V4-Binary kompilieren/linken.
- [x] Ohne GPU: `V4_VRAM=1` fällt sauber und tokenidentisch auf CPU zurück.
- [x] CUDA-Harness auf echter RTX 4070: alle fünf Codecs, Heads `{1,8,64}`,
      Selektionen `{1,7,64,2048}`, Sinks, `-1`-Padding, OOB und OOM.
- [x] Full-Checkpoint: `V4_VRAM=1` gegen CPU tokenidentisch (Teacher-Forcing 1/1,
      Greedy 1/1; kurzer Lauf ebenfalls identischer Text).
- [x] `nvidia-smi`: 160 MiB Prozess-VRAM im Zehn-Token-Orakellauf.
- [x] CPU-Orakel gegen CUDA inklusive PCIe-Transfers gemessen.

## Zielmaschinen-Ergebnis

RTX 4070 (`sm_89`), Treiber 610.57.04, CUDA 13.3. Der native 64-Head-Kernel
erreichte über fünf gemittelte Aufrufe 14.12× bei 640 Zeilen, 14.05× bei 1152
Zeilen und 15.17× bei 7940 Zeilen; Cosine jeweils 1.0000000. Der kurze warme
Full-Checkpoint-Lauf lag bei 0.374 tok/s CPU und 0.379 tok/s CUDA. Wegen leicht
unterschiedlicher Expert-Miss-Zahlen und des winzigen Kontexts ist das kein
belastbarer End-to-end-Speedup; die Kernelmessung isoliert die Phase.

Der Device-KV ist in Phase 05 ein Spiegel mit vollständigem Host-Fallback. Daher
wird hier noch kein RAM frei. Plan 08 übernimmt die exklusive Tier-
Eigentümerschaft und die Rückkopplung in den RAM-Planner. Bei 128k umfasst der
Device-Posten 0.388 GiB (`native`) beziehungsweise 0.133 GiB (`turbo3`); die
früher dokumentierten 0.431/0.165 GiB enthielten den CPU-seitigen Indexer.

## Testkommandos auf der Zielmaschine

```bash
make -C c v4-cuda-test CUDA=1 CUDA_ARCH=sm_89
V4_CUDA_BENCH=1 make -C c v4-cuda-test CUDA=1 CUDA_ARCH=sm_89
make -C c deepseek-v4 CUDA=1 CUDA_ARCH=sm_89
V4_VRAM=1 ./c/deepseek_v4 ...
nvidia-smi
```
