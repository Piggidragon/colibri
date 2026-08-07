# 05 — CUDA: KV in VRAM + Flash-Kernel

Voraussetzung: [00-reference.md](00-reference.md), [02](02-flash-attention.md), [03](03-kv-codec.md)

*Commits:*
1. `build: opt-in CUDA for the V4 engine`
2. `feat: V4 CUDA backend — resident quantized KV`
3. `feat: V4 CUDA flash sparse MLA attention kernel`

## Ziel

Der erste GPU-Brückenkopf für V4. Der KV-Cache zieht nach VRAM, die Attention läuft
als Flash-Kernel mit **Dequantisierung im Register**.

Die harte Voraussetzung ist [03](03-kv-codec.md), nicht [04](04-turboquant.md):
`native` (3.5×/7.5×, bit-exakt) reicht, damit das KV bei den meisten
Kontextlängen in VRAM passt (siehe VRAM-Budget in
[00-reference.md](00-reference.md)), und gibt dem Kernel überhaupt etwas
Kleineres als f32 zu dequantisieren. TurboQuant zahlt zusätzlich ein, sobald es
existiert — nochmal ~3× weniger VRAM *und* ~3× weniger gelesene Bytes im Kernel,
und wird bei 1M der Unterschied zwischen einer Punktlandung und echtem
Spielraum — deshalb ist der Codec-Dispatch unten für alle fünf Formate ausgelegt,
nicht nur `native`. Deshalb ist Attention der richtige erste GPU-Schritt und
nicht die Dense-Matmuls — das KV ist der einzige Teil des Modells, der klein
genug ist, um auch bei großem Kontext vollständig in 12 GB zu passen, und der
Kernel ist in sich abgeschlossen (kein Expert-Streaming beteiligt).

**Umfang, ehrlich abgegrenzt:** das ist *CUDA-Attention für V4*, nicht *CUDA-V4*.
Geroutete Experten bleiben CPU-seitig. Genau die Abgrenzung, die
[c/backend_cuda_ink.h](../c/backend_cuda_ink.h) für Inkling formuliert
(„Deliberately tiny API — expert streaming, attention and everything else stay on
the CPU in this phase"), nur andersherum.

## Build (Commit 1)

`c/Makefile.deepseek-v4` ist heute reines gcc — kein `nvcc`, kein `COLI_CUDA`.

Neu: opt-in `CUDA=1`, **Linux-only**, analog zur Regel für colibri
([c/Makefile:392](../c/Makefile) — `$(error CUDA=1 is supported only on Linux)`):

```make
ifeq ($(CUDA),1)
  ifneq ($(UNAME_S),Linux)
    $(error CUDA=1 for the V4 engine is supported only on Linux)
  endif
  CFLAGS  += -DCOLI_V4_CUDA
  LDFLAGS += -L$(CUDA_HOME)/lib64 -lcudart
  V4_OBJS += backend_cuda_v4.o
endif
```

Objektregel nach dem Muster von `backend_cuda_ink.o` ([c/Makefile:611](../c/Makefile)):

```make
backend_cuda_v4.o: backend_cuda_v4.cu backend_cuda_v4.h
	@command -v "$(NVCC)" >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	"$(NVCC)" $(NVCCFLAGS) -c backend_cuda_v4.cu -o $@
```

Achtung: `Makefile.deepseek-v4` linkt auf Windows `-static`. Der CUDA-Zweig ist
Linux-only, also kollidiert das nicht — aber die `-flto`-Voreinstellung (`LTO ?= 1`)
verträgt sich nicht mit nvcc-Objekten. Im CUDA-Zweig `LTO=0` erzwingen oder das
`.o` von LTO ausnehmen.

**Ohne `CUDA=1` ändert sich am Build nichts.** Der CPU-Pfad bleibt vollständig
erhalten — er ist die Referenz für die Kernel-Tests.

## API (Commit 2)

Neu `c/backend_cuda_v4.{cu,h}`, Oberfläche bewusst klein:

```c
/* Minimal CUDA backend for deepseek_v4.c: the quantised KV cache lives in VRAM
 * and sparse MLA attention runs on-device with dequantisation in registers.
 * Dense projections and routed-expert streaming stay on the CPU in this phase. */

int    v4_cuda_init(int device);              /* 0 = ok */
void   v4_cuda_shutdown(void);
size_t v4_cuda_free_bytes(void);

void  *v4_cuda_kv_alloc(size_t bytes);        /* NULL = OOM */
void   v4_cuda_kv_free(void *base);
int    v4_cuda_kv_write_row(void *base, int slot, const void *row, size_t row_bytes);

/* Flash sparse MLA. window_kv/compressed_kv sind Device-Zeiger; indices sind
 * Host-Arrays (klein, ändern sich pro Token). ctx und q sind Host-f32. */
int v4_cuda_flash_attention(
    float *ctx, const float *q,
    const void *window_kv, int window_size, const int *window_indices,
    const void *compressed_kv, const int *compressed_indices, int compressed_selected,
    const float *sinks, int codec, int heads, int head_dim,
    size_t row_bytes, float scale, int rotated);
```

Intern über die vorhandenen Primitive aus [c/backend_cuda.h](../c/backend_cuda.h):
`coli_cuda_pipe_alloc/free/upload/download/sync`, `coli_cuda_pipe_scratch`,
`coli_cuda_mem_info`. `v4_cuda_init` ruft `coli_cuda_init` mit einem einzelnen
Device — Multi-GPU ist hier ausdrücklich kein Ziel.

**Nicht** wiederverwendbar: `coli_cuda_attention_absorb*`. Die sind auf GLMs
Geometrie gebaut (separates `kv_b` als Absorptionsmatrix, latent/rope getrennt,
kein Top-k, keine Sinks). V4 hat einen einzigen 512-dim-Vektor, der Key und Value
zugleich ist.

## Der Kernel (Commit 3)

Konkrete Dimensionen aus dem Paper (siehe [paper-deepseek-v4.md](paper-deepseek-v4.md)),
sie bestimmen das Kernel-Design:

| | CSA-Layer (21×) | HCA-Layer (20×) |
|---|---|---|
| Zeilen pro Token | `n_win + top-k` = 128 + 512 = **640**, konstant | `n_win +` *alle* komprimierten = 128 + ctx/128 |
| bei 128k | 640 | 1152 |
| bei 1M | 640 | **7940** |

**HCA hat keine Sparse Attention** — jeder Query sieht den kompletten
komprimierten Cache. Deshalb dominieren die HCA-Layer die Lesebandbreite bei
langem Kontext, nicht die CSA-Layer: bei 1M sind es 353 MB/Token bei f32,
100 MB bei `native`, 34 MB bei turbo3. Genau dieser Strom rechtfertigt den Kernel.

Ein Block pro Head, `head_dim=512` Floats im Akkumulator (Shared Memory oder
Register-Tiling), Online-Softmax über die ausgewählten Zeilen:

```
für jede ausgewählte Zeile r:
    lade row_bytes Bytes von r nach Registern/Shared
    dequantisiere gruppenweise (128 Werte je Gruppe)
    score = dot(q_head, dequant) * scale
    warp-reduce score
    if score > running_max: correction = exp(running_max - score);
                            running_sum *= correction; acc *= correction;
                            running_max = score
    p = exp(score - running_max); running_sum += p; p = bf16_round(p)
    acc += p * dequant                        ← derselbe dequantisierte Wert
ctx = bf16_round(acc / running_sum)
```

Der dequantisierte Wert wird **zweimal im selben Registerfenster** benutzt (Score
und Akkumulation) und nie zurückgeschrieben. Das ist der ganze Punkt des
Flash-Umbaus aus Plan 02.

**Sink-Initialisierung** wie auf der CPU: `running_max = sinks[head]`,
`running_sum = 1.0f`.

**Codec-Dispatch** als Template-Parameter, nicht als Laufzeit-`switch` im inneren
Loop — ein Kernel pro Codec, wie es die Referenz mit
`fattn-vec-instance-turbo2_0`/`turbo3_0`/`turbo4_0` macht.

**Rotierter Modus** (`rotated=1`, aus Plan 04): Dequant liefert im WHT-Raum, Q
kommt bereits rotiert vom Host, und die inverse WHT läuft einmal am Ende auf `ctx`.
Die Vorzeichentabellen `turbo_cpu_s1/s2` müssen dafür als `__constant__` auf dem
Device liegen — Muster: `coli_cuda_fp8_set_lut` / `c_e4m3`
([c/backend_cuda.cu:860](../c/backend_cuda.cu)) publiziert die CPU-Tabelle einmal
pro Device, damit sie nicht driften kann. Genauso für Zentroide und Signs:
**eine** Quelle in `c/turbo_quant.h`, per `v4_cuda_publish_tables()` hochgeladen,
Uploads vorher abgelehnt.

## Datenfluss und PCIe

Erste Stufe: Q per H2D hoch (`64 × 512 × 4` = 128 KB), `ctx` per D2H zurück
(ebenfalls 128 KB). Bei 43 Layern **~11 MB pro Token**, über PCIe 4.0 ×16
(~25 GB/s) grob **0.4 ms/Token**.

Akzeptabel als erster Schnitt, und zugleich die Motivation für Plan 06: sobald die
Dense-Projektionen auf dem Gerät rechnen, verlässt Q das Gerät nicht mehr und die
Transfers kollabieren auf den Layer-Input.

Der KV-Write geht ohnehin nur einmal pro Token und Layer hoch (`row_bytes`, bei
turbo3 200 Bytes) — vernachlässigbar.

## Integration

`ColiDeepSeekV4WindowAttentionState` bekommt unter `#ifdef COLI_V4_CUDA` zwei
Device-Zeiger neben den Host-Puffern. Die Codec-Kodierung passiert **auf der CPU**
(sie ist billig, einmal pro Token), hochgeladen wird die fertige Zeile.

Fallback-Kette, die nie hart scheitert:

```
V4_VRAM=1 und v4_cuda_init ok und row_bytes darstellbar
    → Device-KV + Kernel
sonst → Host-KV + CPU-Flash (Plan 02)
```

Ein Fehler im Kernel-Aufruf fällt für diesen Token auf CPU zurück und loggt einmal
— nicht pro Token, sonst ist stderr unbrauchbar.

## Tests

`c/tests/test_v4_attention_cuda.cu`, Regel nach dem Muster von `cuda-test`
([c/Makefile:663](../c/Makefile)), **nicht** in `TEST_BINS` (braucht eine GPU):

- Kernel gegen `coli_v4_flash_attention_ref` auf Zufallsdaten, alle Codecs,
  `heads` ∈ {1, 8, 64}, `head_dim=512`, `topk` ∈ {1, 7, 64, 2048}.
  Toleranz: Cosine `> 0.9999`, relative Abweichung pro Komponente `< 1e-3`.
- Sinks: extreme Werte, wie im CPU-Test.
- `-1`-Padding im Fenster-Index.
- Rotierter vs. unrotierter Modus gegeneinander.
- Tabellen-Publikation: Kernel ohne `v4_cuda_publish_tables` muss **ablehnen**,
  nicht gegen Nullen rechnen (das ist die Lehre aus `g_fp8_lut_ready`,
  [c/backend_cuda.cu:1059](../c/backend_cuda.cu)).
- OOM: `v4_cuda_kv_alloc` mit absurder Größe gibt NULL, kein Absturz.

Dazu ein CPU-seitiger Test, dass die Fallback-Kette greift, wenn
`v4_cuda_init` scheitert (per Test-Hook, Muster `COLI_V4_TEST_HOOKS`).

## Abnahme

- `make -C c deepseek-v4` ohne `CUDA=1` baut unverändert und ist token-identisch.
- `make -C c deepseek-v4 CUDA=1` baut; ohne GPU zur Laufzeit sauberer CPU-Fallback.
- Kernel innerhalb Toleranz gegen CPU-Referenz, alle Codecs.
- `nvidia-smi` zeigt während des Laufs den erwarteten KV-Footprint
  (bei turbo3 und 128k: ~0.35 GiB + Indexer).
- Ein Layer end-to-end mit `V4_VRAM=1` liefert dieselben Tokens wie ohne.

## Risiken

- **`-flto` × nvcc** ist die wahrscheinlichste Build-Überraschung. Im CUDA-Zweig
  abschalten.
- **`Makefile.deepseek-v4` ist eigenständig** und teilt die Toolchain-Erkennung
  von `c/Makefile` nicht. `NVCC`, `CUDA_HOME`, `NVCCFLAGS` müssen dort neu oder per
  Include verfügbar gemacht werden — nicht annehmen, dass sie da sind.
- **Tabellendrift** zwischen CPU und Device ist der klassische stille Fehler:
  falsche Vorzeichen liefern plausibel aussehenden Unsinn. Deshalb eine Quelle und
  ein Ablehnungspfad, kein Default-Nullzustand.
- **0.4 ms/Token PCIe** ist die Untergrenze dieser Stufe. Wenn die Messung
  deutlich schlechter ausfällt, liegt es an fehlendem Pinning der Host-Puffer —
  `coli_cuda_pipe_*` hat dafür bereits Scratch-Slots.
- **12 GB sind knapp.** Der Kernel darf keinen `topk × head_dim`-Puffer anlegen;
  wenn er das doch täte, wäre bei 2048×512×4 = 4 MB pro Head × 64 Heads der Spuk
  sofort vorbei. Genau deshalb Flash.
