# 06 — Dense-Gewichte in VRAM

Voraussetzung: [00-reference.md](00-reference.md), [05-cuda-attention.md](05-cuda-attention.md)

*Commits:*
1. `feat: skip the AVX2 rows8 repack for GPU-resident fp8 tensors`
2. `feat: resident fp8 dense tensors on GPU for V4`
3. `feat: keep the attention query on-device across projections` — nach Messung
   nicht umgesetzt, siehe Ergebnis

## Ergebnis

Implementiert und auf der Zielmaschine abgenommen in PR #8. Der
`CUDA=1 V4_VRAM=1`-Pfad lädt die FP8-Dense-Gewichte samt expandierten Scales
row-major auf die RTX 4070, gibt ihre Hostbuffer frei und dispatcht
Attention-Projektionen, Indexer-Query und Shared Expert auf CUDA. Der Planner
bevorzugt VRAM, fällt bei Platzmangel auf residenten RAM und danach auf Streaming
zurück. Schlägt ein Upload fehl, wird die residente Stufe abgeschaltet und alles
bereits Hochgeladene wieder freigegeben.

### Nachtrag: Review-Befunde zu PR #8

Vier Befunde aus dem Review sind nachgezogen; einer davon ändert das oben
beschriebene Verhalten:

- **Compute-Fehler brechen jetzt ab, statt zu rematerialisieren.** Der Upload
  gibt die Host-FP8-Buffer frei, also hätte der CPU-Rückfall den Tensor pro
  Projektion neu aus dem Checkpoint gelesen — 43 Layer × ~8 Projektionen,
  13–34 MB je Lesevorgang, auf einer DRAM-losen NVMe, ohne jede Meldung. Alle
  realen Fehlerursachen (Scratch-`cudaMalloc`, Launch, Memcpy) bedeuten
  erschöpftes VRAM und heilen nicht bis zum nächsten Aufruf. Der Pfad rastet
  darum nach dem ersten Fehler aus (`coli_v4_dense_cuda_disabled`), meldet eine
  Zeile und schlägt fehl; der Ausweg ist `V4_VRAM=0`. **Damit gilt die
  Abnahmezeile „ebenso bei injiziertem CUDA-Compute-Fehler" unten nicht mehr:**
  unter `COLI_GPU_FAIL_AFTER` bleibt der Lauf nicht mehr tokenidentisch, sondern
  bricht sauber ab. Das ist beabsichtigt — die Alternative war ein stiller Hänger
  auf dem großen Checkpoint.
- **Der VRAM-Fallback gab bereits hochgeladene Layer nicht frei.** Sie blieben
  als tote Device-Allokationen liegen, während `resident_enabled_v2()` sie nie
  wieder erreichte, und ließen danach `v4_cuda_kv_alloc` scheitern — also
  langsamer als `V4_VRAM=0`. `v4_dense_resident_drop_all` räumt jetzt auf.
- **Der Dense-Tier wurde ohne jede Reserve zugelassen.** Siehe
  [Plan 08](08-vram-planner.md): dessen Formel ist vorgezogen.
- **`tests/test_v4_dense_tier` baute sein Objekt im gemeinsamen Verzeichnis**
  und kollidierte unter `make -j` mit dem Sub-Make von `tests/test_deepseek_v4`.
  Eigenes `build/v4-dense-tier/`, Muster `V4_ROWS8_DIR`.

Die Full-Checkpoint-Inventur hat die zentrale Planannahme korrigiert: **6.267 GiB**
sind das gesamte Dense-Inventar, aber nur **5.456 GiB** davon sind FP8-Gewichte
und expandierte Scales und können in dieser Phase wandern. **0.810 GiB**
BF16/f32/i64 bleiben auf dem Host. Bei `--memory-gb 28`, `CTX=4096` wuchs der
Expert-Cache damit von 36 auf 46 Slots pro Layer beziehungsweise von 19.27 auf
24.63 GiB. `nvidia-smi` zeigte 5750 MiB Prozess-VRAM inklusive CUDA-Kontext und
dem kleinen Phase-05-KV.

Abnahme am 2026-08-09, RTX 4070 (`sm_89`), Full Checkpoint, Prompt
`The capital of France is`, DSpark aus:

- Full-Checkpoint-Oracle: Teacher Forcing 10/10, Greedy 1/1. Die frisch mit
  `torch==2.13.0+cpu`, `transformers==5.14.1` und `safetensors==0.8.0`
  regenerierte Tiny-Fixture blieb CPU- und CUDA-seitig vollständig tokenidentisch;
  ebenso bei injiziertem CUDA-Compute-Fehler.
- Der CUDA-Harness prüfte alle Dense-Shapes aus diesem Plan einschließlich eines
  `wo_a`-Rowslices gegen die CPU-Referenz: null Abweichungen außerhalb
  `1e-3 × (|Referenz| + 1e-3)`.
- Bei nahezu gleicher Misszahl (CPU 1442, CUDA 1445) sank TTFT von 36.397 auf
  28.624 s (**−21.4 %**). Zwei Decode-Tokens nach TTFT sanken von 5.268 auf
  3.426 s (**−35.0 %**, 0.380 auf 0.584 tok/s).
- `make -C c test && make -C c check`, der CUDA-Harness und der
  Source-Sync-Test waren grün.

Der optionale dritte Commit wurde verworfen. Durchgängig device-residentes Q
würde rund 11 MB PCIe-Verkehr pro Token sparen, auf PCIe 4 x16 also weniger als
0.4 ms gegenüber dem hier mehrsekündigen Decode. Dafür wären Norm, RoPE und die
Attention-API gemeinsam umzubauen. Der RAM-Gewinn und die gemessene
Beschleunigung hängen nicht davon ab.

## Ziel

**Der größte verschiebbare Einzelposten: 5.456 GiB RAM frei.** Der Expert-Cache
wächst auf der Zielkonfiguration gemessen um zehn Slots pro Layer. Die Dense-
Matmuls laufen auf der GPU; Q bleibt nach der Messentscheidung oben noch nicht
durchgängig auf dem Gerät.

## Ausgangslage

### Was „dense" hier heißt

`v5_dense_inventory` ([c/deepseek_v4.c:981](../c/deepseek_v4.c)) summiert
`stats.total_bytes` über alle Layer — die **komplette Layer-Inventur ohne geroutete
Experten**: Attention-Projektionen, Compressor, Indexer, Router-Gate, Shared Expert,
Norms, HC-Parameter. Laut Checkpoint-Inventur 6.267 GiB insgesamt. Davon sind
5.455 GiB FP8-Weights und 0.001 GiB expandierte Scales; 0.810 GiB sonstige
Tensoren bleiben CPU-seitig.

Die fp8-Tensoren pro Layer, aus `coli_v4_layer_plan`
([c/deepseek_v4.c:355](../c/deepseek_v4.c) ff.), alle mit `.scale` als
`[ceil(rows/128), ceil(cols/128)]` E8M0:

| Tensor | Shape |
|---|---|
| `attn.wkv` | [512, 4096] |
| `attn.wq_a` | [1024, 4096] |
| `attn.wq_b` | [32768, 1024] |
| `attn.wo_a` | [8192, 4096] |
| `attn.wo_b` | [4096, 8192] |
| `ffn.shared_experts.w1/w2/w3` | [2048, 4096] / [4096, 2048] / [2048, 4096] |
| `attn.indexer.wq_b` | [8192, 1024] |

### Wie sie geladen werden

`coli_v4_layer_load` ([c/deepseek_v4.c:512](../c/deepseek_v4.c)) liest jeden Tensor
in `weights->data[i]`. E8M0-Scales werden dabei über `st_read_scale_f32` auf **f32
expandiert** (`resident_bytes = numel * sizeof(float)`, [:527](../c/deepseek_v4.c)) —
das muss in der VRAM-Bilanz mitgezählt werden, ist aber klein (bei `wq_b`: 2048
Scales gegen 33.5 MB Gewichte).

Residenz entscheidet `coli_v4_resident_tier_plan`
([:763](../c/deepseek_v4.c)) — heute binär: passt Dense komplett in den RAM-Plan
oder nicht. `COLI_V4_RESIDENT_MAX_LAYERS = 128`
([c/deepseek_v4_internal.h:628](../c/deepseek_v4_internal.h)) ist reichlich für 43
Layer.

Verbraucht werden sie über `fp8_view` → `ColiTensorView` → `coli_fp8_matvec_ref` /
`coli_fp8_matmul_batch_ref`, die laut Kopfkommentar der `NATIVE_QUANT`-Unit
([:9946](../c/deepseek_v4.c)) durch das **gemeinsame `matmul_fp8` aus `quant.h`**
dispatchen — genau die Implementierung, die `coli_cuda_matmul` mit `fmt=8` auf der
GPU spiegelt.

### `packed_rows8` ist harmloser als es aussieht

Ein früher Verdacht war, `packed_rows8` beschreibe eine feinere Scale-Granularität
(8×128 statt 128×128), die der CUDA-Kernel nicht kann. **Das stimmt nicht.**

`v4_fp8_pack_rows8_inplace` ([c/deepseek_v4.c:488](../c/deepseek_v4.c)) ist eine
reine **Byte-Layout-Transposition innerhalb von 8-Zeilen-Kacheln**, nach dem Laden
in-place angewandt, ausschließlich unter `__AVX2__`:

```c
#ifndef __AVX2__
    return 0;                                   /* no-op ohne AVX2 */
#else
    for (int64_t tile = 0; tile < rows / 8; tile++)
        for (column...) for (lane = 0..7)
            target[column * 8 + lane] = scratch[lane * columns + column];
    return 1;
#endif
```

Es dient dem AVX2-Zielkernel. **Die Scales bleiben unangetastet** — sie werden
getrennt über `st_read_scale_f32` gelesen und behalten ihre 128×128-Granularität
aus `add_fp8` ([:343](../c/deepseek_v4.c)). `block_rows = 8` in der
`ColiTensorView` beschreibt also die *Gewichtskachelung*, nicht die Scale-Auflösung.

**Konsequenz:** Für GPU-Tensoren wird der Repack einfach **übersprungen**. Dann
liegen row-major fp8-Bytes mit 128×128-f32-Scales vor — exakt das Format, das
`coli_cuda_matmul` mit `fmt=8` erwartet
([c/backend_cuda.cu:1066](../c/backend_cuda.cu)). Kein neuer Kernel, keine
Präzisionsfrage, kein Repack-Verlust.

Dasselbe gilt für DSpark (Plan 07), dessen `v4_ds_pack_rows8`
([c/deepseek_v4_dspark.inc:113](../c/deepseek_v4_dspark.inc)) dieselbe
AVX2-Transposition mit demselben Kommentar ist.

## Commit 1 — Repack konditionieren

`coli_v4_layer_load` bekommt die Information, ob dieser Layer aufs Gerät geht:

```c
if (spec->dtype == COLI_ST_F8_E4M3 && spec->rank == 2 && !weights->gpu_resident) {
    int packed = v4_fp8_pack_rows8_inplace(weights->data[i],
                                           spec->shape[0], spec->shape[1]);
    ...
    weights->plan.tensors[i].packed_rows8 = packed > 0;
}
```

`gpu_resident` kommt aus dem Tier-Plan und wird beim Laden gesetzt. Bleibt es 0,
ist das Verhalten byte-identisch zu heute.

Ein Test muss festhalten, dass `packed_rows8 == 0` in diesem Fall auch wirklich
gesetzt bleibt — sonst würde `fp8_view` später `block_rows = 8` melden und der
CPU-Fallback läse falsch.

### `coli_v4_layer_load` gibt es **dreimal**, und das ändert den Zuschnitt

Der Skizze oben fehlt, wo `gpu_resident` überhaupt herkommen kann. Im Amalgam:

| Zeile | Symbol | Repack? | Rolle |
|---|---|---|---|
| [:512](../c/deepseek_v4.c) | `coli_v4_layer_resident_reference_load` (per `#define` umbenannt) | **ja**, bei [:543](../c/deepseek_v4.c) | liest die Tensoren wirklich |
| [:583](../c/deepseek_v4.c) | `coli_v4_layer_load` (Unit `LAYER_RESIDENT`) | nein | exportierter Wrapper, cached pro Layer |
| [:9678](../c/deepseek_v4.c) | `coli_v4_layer_load` (Unit `LAYER`) | **nein** | zweite Übersetzungseinheit, ohne Repack |

Zwei Dinge folgen daraus, die die Skizze oben so nicht hergibt:

1. **Der Repack sitzt nicht in der Funktion, die den Tier-Plan kennt.** Der Wrapper
   bei :583 hat `engine`, ruft den Reference-Loader aber ausdrücklich mit `NULL`
   auf:

   ```c
   if (!resident_enabled_v2(engine))
       return coli_v4_layer_resident_reference_load(
           NULL, weights, effective_config, index, layer, error, error_size);
   ```

   `weights->gpu_resident` abzufragen genügt deshalb nur, wenn das Flag **im
   `weights`-Struct** steht (das durchgereicht wird) — nicht im `engine`. Die
   Skizze oben liest `weights->gpu_resident` und ist damit richtig; wer stattdessen
   `engine->...` nimmt, greift auf `NULL` zu. Das Flag muss also der Aufrufer
   **vor** dem Load in `weights` setzen, an beiden `reference_load`-Aufrufstellen
   ([:594](../c/deepseek_v4.c) und die oben).
2. **Die `LAYER`-Kopie bei :9678 hat gar keinen Repack** — sie ist kein Duplikat,
   sondern eine eigene Variante. Der Source-Sync-Test aus
   [02-flash-attention.md](02-flash-attention.md) lässt `deepseek_v4_layer.c`
   deshalb bewusst aus; Textgleichheit ist hier das falsche Kriterium. Prüfen, ob
   diese Kopie im V4-Zielbuild überhaupt aktiv ist (`V4_TARGET_UNITS` in
   `c/Makefile.deepseek-v4.units`), bevor man sie mitändert.

Siehe die Duplikat-Tabelle in [00-reference.md](00-reference.md).

## Commit 2 — Upload und Dispatch

### Upload

Vor dem ersten Upload muss `coli_cuda_fp8_set_lut` mit `quant.h`s `E4M3_LUT`
gelaufen sein; sonst lehnt `coli_cuda_tensor_upload_g` ab
([c/backend_cuda.cu:1059](../c/backend_cuda.cu)). Das ist Absicht — ein Kernel soll
nie gegen eine Nulltabelle rechnen.

```c
coli_cuda_tensor_upload_g(&device_tensor,
                          weights->data[i],        /* row-major fp8 */
                          scales,                  /* f32, [rows/128][cols/128] */
                          /*fmt*/ 8, I, O, device, /*gs*/ 0);
```

`ColiDeepSeekV4LayerWeights` bekommt unter `#ifdef COLI_V4_CUDA` ein
`ColiCudaTensor *device[COLI_V4_MAX_LAYER_TENSORS]` parallel zu `data[]`.
Freigabe über `coli_cuda_tensor_free` in `coli_v4_layer_free`.

### Matmul-Dispatch

`coli_fp8_matvec_ref` und `coli_fp8_matmul_batch_ref` bekommen einen Vorschaltpfad:
liegt ein Device-Tensor vor und ist CUDA aktiv, dann `coli_cuda_matmul`, sonst wie
bisher. Weil beide ohnehin schon durch `matmul_fp8` dispatchen, ist das ein Zweig
mehr an einer bestehenden Verzweigung — kein neuer Aufrufweg durch die Engine.

Der `ColiTensorView` trägt keinen Device-Zeiger; die Zuordnung läuft über den
Layer-Index. Sauberer als den View aufzublähen, der bewusst „describes bytes it
never owns" ist ([c/tensor.h:11](../c/tensor.h)).

### Planner

`coli_v4_resident_tier_plan` ([:763](../c/deepseek_v4.c)) lernt eine dritte Stufe.
Heute:

```c
if (resident_tiers_fit(available, fixed, dense_bytes, minimum_experts)) {
    plan->dense_resident = 1;
    plan->dense_bytes = dense_bytes;
}
```

Neu: `dense_location ∈ {VRAM, RAM, STREAMED}`, VRAM bevorzugt, RAM als Fallback,
Streaming zuletzt. Die vollständige Stufenlogik über alle Posten kommt in Plan 08 —
hier reicht Dense.

Die `ram_tiers`-Zeile ([:1057](../c/deepseek_v4.c)) muss es ausweisen, sonst ist
nicht sichtbar, was passiert ist:

```
ram_tiers available=28.00GiB dense=vram(host=0.81GiB device=5.46GiB) target_slots=46 ...
```

## Commit 3 — Q bleibt auf dem Gerät (nach Messung verworfen)

Sobald `wq_a`, `wq_b` und `wkv` auf der GPU rechnen, entsteht Q dort. Der
Attention-Kernel aus Plan 05 bekommt eine `_dev`-Variante, die einen Device-Zeiger
statt eines Host-Arrays nimmt — Muster:
`coli_cuda_attention_absorb_batch_dev` ([c/backend_cuda.h](../c/backend_cuda.h)) für GLM.

Damit fällt der 11 MB/Token-PCIe-Verkehr aus Plan 05 auf den Layer-Input zusammen.

Die RMS-Norms und RoPE dazwischen laufen über die vorhandenen
`coli_cuda_pipe_rmsnorm_s` und `coli_cuda_pipe_rope_base`. Was sich nicht lohnt,
bleibt auf der CPU — dann kostet es einen Round-trip, und genau diese Grenze ist
die Messaufgabe dieses Commits.

## Tests

- `c/tests/test_v4_rows8_skip.c` — `coli_v4_layer_load` mit `gpu_resident = 1`
  lässt die Bytes unverändert und `packed_rows8 == 0`; mit 0 verhält es sich
  byte-identisch zu heute. Auf einer Maschine ohne AVX2 sind beide Wege gleich —
  der Test muss das aushalten (`#ifdef __AVX2__` im Erwartungswert).
- `c/tests/test_v4_dense_cuda.cu` — `coli_cuda_matmul` fmt=8 gegen `matmul_fp8`
  auf den Shapes aus der Tabelle oben, mit echten 128×128-Scales. Toleranz wie im
  bestehenden `tests/test_fp8_cuda.cu`.
- `c/tests/test_v4_dense_tier.c` — Planner: genug VRAM → `VRAM`; zu wenig VRAM →
  `RAM`; zu wenig von beidem → `STREAMED`, kein Fehler.
- End-to-end: Tiny-Fixture mit `CUDA=1 V4_VRAM=1` token-identisch zu ohne.

## Abnahme

- `ram_tiers` zeigt `dense=vram`, `target_cache` gemessen **5.36 GiB höher** als vorher.
- `nvidia-smi` zeigt den erwarteten Prozess-Footprint (5750 MiB inklusive CUDA-
  Kontext und KV; davon 5.456 GiB inventarisierte Dense-Tensoren).
- fp8-Matmuls auf GPU innerhalb Toleranz gegen die CPU-Referenz.
- Der mögliche PCIe-Gewinn von Commit 3 ist quantifiziert und gegen den
  gemessenen Decode-Anteil entschieden; siehe Ergebnis.
- Ohne `CUDA=1` unverändertes Verhalten, Bytes und Tokens identisch.

## Risiken

- **Numerik.** `coli_cuda_matmul` akkumuliert nicht notwendig in derselben
  Reihenfolge wie `matmul_fp8`. Logits weichen minimal ab; am Tiny-Fixture auf
  Token-Identität prüfen und, falls sie kippt, als Toleranz dokumentieren statt
  wegzudrücken. Das ist die einzige echte Semantikfrage dieser Phase.
- **5.456 dense + 0.99 head + 0.56 DSpark + 0.30 Workspace = 7.31 GiB
  Fixkosten** lassen auf der headless 12-GB-Karte ~3.39 GiB für den KV — 11.7
  nutzbar minus 1.0 GiB Planner-Reserve aus Plan 08 minus 7.31 (siehe VRAM-Budget in
  [00-reference.md](00-reference.md)). Ohne den Codec aus Plan 03 schließt das
  Budget bei langem Kontext trotzdem nicht. Reihenfolge nicht umdrehen.
- **Der `gpu_resident`-Durchstich** berührt den Ladepfad, den auch der
  CPU-Betrieb nimmt. Ein Fehler dort trifft beide Modi. Deshalb Commit 1 klein und
  separat, mit eigenem Test.
- **Commit 3 kann sich nicht lohnen.** Wenn zu viele Zwischenschritte auf der CPU
  bleiben, ist der Round-trip teurer als der gesparte Transfer. Messen, und wenn es
  nicht trägt, bei Commit 2 stehenbleiben — der RAM-Gewinn hängt nicht daran.
