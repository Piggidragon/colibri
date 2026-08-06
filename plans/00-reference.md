# 00 — Referenzplan: DeepSeek V4 auf 32 GB RAM + RTX 4070

Übergeordnetes Dokument. Hier stehen Zielbild, Hardware-Budget, Codekarte und
Konventionen; die Phasenpläne `01`–`08` setzen das voraus und wiederholen es nicht.

- Branch: `dev/v4-kv-codec`, von `main` (`8f512fc`), Fork `Piggidragon/colibri`
- Experimenteller Branch, gestapelte Commits, direkt darauf arbeiten
- Lizenz: Repo ist Apache-2.0, der TurboQuant-Referenz-Fork MIT → Attribution im
  portierten Header genügt. `reference/` selbst wird **nicht** committet.

## Zielbild

Nicht „KV quantisieren", sondern **DeepSeek-V4-Flash-0731 (284B, ~167 GB) plus den
DSpark-Drafter auf genau dieser Maschine so gut wie möglich fahren**:

| | |
|---|---|
| RAM | 32 GB |
| GPU | RTX 4070, 12 GB |
| Laufwerk A | 1 TB NVMe **Gen4**, DRAM-los (HMB) |
| Laufwerk B | 512 GB SSD **Gen3**, DRAM-los (HMB) |
| OS | CachyOS (Arch-Familie) |

Beide Laufwerke fassen je eine vollständige Modellkopie (167 GB < 512 GB) —
Grundlage für Dual-Streaming, siehe [10-dual-streaming.md](10-dual-streaming.md).

**Checkpoint:** [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir /pfad/DeepSeek-V4-Flash
```

Das ist der **einzige** Download. DSpark liegt im selben Checkpoint unter dem
Präfix `mtp.<stage>.` — colibri sucht die Drafter-Tensoren in
`engine->target_index` ([c/deepseek_v4.c:6325](../c/deepseek_v4.c)), nicht in einem
zweiten Verzeichnis. Ein separates DSpark-Repo gibt es nicht zu laden.

Andere Modelle sind in diesem Fork ausdrücklich kein Ziel; der Rückbau steht in
[11-strip-to-v4.md](11-strip-to-v4.md) und kommt **zuletzt**, weil mehrere Pläne
sich Referenzcode aus den zu löschenden Motoren holen.

Die bindende Größe ist die Expert-Cache-Residenz: ~160 GB geroutete Experten
gegen einen RAM-Cache, der heute bei ~15 GiB landet. Alles, was nicht gecacht ist,
wird pro Token von der Platte gestreamt. Jedes freigemachte GiB RAM ist direkt
mehr resident gehaltene Experten.

Der RTX 4070 liegt dabei ungenutzt daneben: die V4-Engine hat **null**
GPU-Anbindung. `grep -i cuda c/deepseek_v4.c` ist leer, `c/Makefile.deepseek-v4`
ruft nur gcc mit `-fopenmp`. CUDA/Vulkan/Metal hängen ausschließlich an
`c/colibri.c` (GLM-5.2).

Dense-Gewichte (6.27 GiB), BF16-Head (1.06 GiB) und DSpark (~1.25 GiB) belegen
heute RAM, obwohl sie zusammen in 12 GB VRAM passen. **Das ist der Haupthebel.**

## RAM-Bilanz

Abgeleitet aus `build_runtime_plan` ([c/deepseek_v4.c:924](../c/deepseek_v4.c))
und `coli_v4_resource_plan_compute` ([:693](../c/deepseek_v4.c)).

```
MemAvailable                                ~29.0 GiB
 − system reserve   available/8, 512MiB..4GiB  −3.6    Phase 1 (RAM_GB)
 − runtime reserve  2×max_layer + other        −1.4    enthält KV + 512MiB scratch
 − dense resident                              −6.27   Phase 6
 − BF16 head                                   −1.06   Phase 7
 − DSpark reserve (wenn aktiv)                 −1.25   Phase 7
 = Expert-Cache                               ~15.4 GiB → ~27/256 Slots ≈ 11 %
```

Nach allen Phasen: **~29 GiB Expert-Cache ≈ 20 % Residenz**, bei identischer
Semantik.

`per_slot = num_hidden_layers × expert_record_bytes ≈ 43 × 13 MB ≈ 559 MB`, daher
die Slot-Zahlen. `expert_record_bytes` misst w1/w2/w3 + Scales von
`layers.0.ffn.experts.0` ([:892](../c/deepseek_v4.c)).

## VRAM-Budget (4070, ~11.2 GiB nutzbar mit Display)

```
dense fp8          6.27   Phase 6
head bf16          1.06   Phase 7
DSpark             1.25   Phase 7
KV @128k turbo3    0.35   Phase 4/5   (f32 wären 1.18)
Indexer-KV @128k   0.24   Phase 4     (f32; turbo3 → 0.02)
workspace         ~0.30
                  ─────
                   9.47 GiB, Rest ~1.7 GiB
```

Das schließt **nur mit komprimiertem KV**. Codec und VRAM-Residenz sind keine
getrennten Features, sondern Voraussetzung füreinander. Der Planner aus Phase 8
muss pro Stufe einzeln auf RAM zurückfallen können — ein zweiter Monitor frisst
die Reserve.

## Modellgeometrie

Belegt durch das Paper, Abschnitt 4.2.1 — siehe
[paper-deepseek-v4.md](paper-deepseek-v4.md). Keine Schätzungen mehr.

| | |
|---|---|
| `num_hidden_layers` | **43** |
| `hidden_size` | 4096 |
| `num_attention_heads` | 64 |
| `head_dim` (`c`) | **512** = 4 × 128 |
| `qk_rope_head_dim` | 64 (Tail-Dims 448–511) |
| `q_lora_rank` (`d_c`) / `o_lora_rank` (`d_g`) | 1024 / 1024, `o_groups` (`g`) 8 |
| `index_head_dim` (`c^I`) / `index_n_heads` | **128** / 64 |
| **`index_topk`** (Attention-Top-k) | **512** |
| **`sliding_window`** (`n_win`) | **128** |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 6, `moe_intermediate_size` 2048 |
| `num_hash_layers` | 3 |
| `hc_mult` (`n_hc`) / `hc_sinkhorn_iters` (`t_max`) | 4 / 20 |
| `vocab_size` | 129280 |

**Layer-Verteilung** (Paper: erste zwei Layer reines SWA, danach CSA/HCA im
Wechsel; Start bei CSA laut [c/tests/test_deepseek_v4.c:414](../c/tests/test_deepseek_v4.c)):

| Typ | `compress_ratio` | Layer | Anzahl |
|---|---|---|---|
| SWA | 0 | 0, 1 | 2 |
| **CSA** (komprimiert + Indexer) | 4 | 2, 4, … 42 | **21** |
| **HCA** (komprimiert, ohne Indexer) | 128 | 3, 5, … 41 | **20** |

**`head_dim=512` und `index_head_dim=128` sind beide Vielfache von 128.** Deshalb
passt TurboQuant ohne Anpassung der Rotationsgruppe.

**`n_win = 128`, nicht ~2048.** Der Fensterring ist speichermäßig bedeutungslos
(43 × 128 × 512 × 4 B = **10.8 MiB**), bandbreitenmäßig aber in jedem Layer und
jedem Token vollständig präsent.

Das Tiny-Fixture ([c/deepseek_v4_tiny/config.json](../c/deepseek_v4_tiny/config.json))
hat dagegen `head_dim=32`, `sliding_window=8`, `index_head_dim=32` — dort fällt
jeder Turbo-Codec auf f32 zurück (siehe Plan 04).

## Der KV-Cache ist bereits quantisiert — nur f32 gespeichert

**Der wichtigste Fund aus dem Paper**, und er verschiebt die Priorität des ganzen
Vorhabens.

Paper 2.3.4: *„BF16 für die RoPE-Dimensionen, FP8 für die übrigen"* für die
Haupt-KV-Einträge, *„FP4"* für den Lightning Indexer.

colibri **rechnet genau das schon**:

| Strom | Code | Tatsächliche Präzision |
|---|---|---|
| Fenster-KV | [c/deepseek_v4.c:1655](../c/deepseek_v4.c) | fp8-QDQ Dims 0–447 (Block 64, E8M0), bf16 Dims 448–511 |
| CSA/HCA komprimiert | [:2643](../c/deepseek_v4.c) | dasselbe |
| Indexer komprimiert | [:2661](../c/deepseek_v4.c) | Hadamard-Rotation + fp4-QDQ über alle 128 Dims (Block 32) |

```c
output[base + i] = coli_e4m3fn_decode(coli_e4m3fn_encode(normalized)) * scale;
scales[base / block_size] = encoded_scale;        /* E8M0-Byte */
```
([:10096](../c/deepseek_v4.c))

…und legt das Ergebnis dann in `float`-Arrays ab. Der gespeicherte Wert ist exakt
`bf16_round(e4m3_decode(q) · e8m0_decode(s))` — aus `(q, s)` **bit-exakt**
rekonstruierbar.

**Konsequenz:** Es gibt eine 3.5×-Ersparnis auf den Haupt-KV und eine 7.5× auf den
Indexer, die **keine Präzisionsänderung** ist, sondern nur aufhört, quantisierte
Werte in 4× zu großen Containern zu halten. Das ist kein Kompromiss, den man
abwägen müsste — es ist ein Buchhaltungsfehler. Siehe [03-kv-codec.md](03-kv-codec.md).

Nebenbei: dass der Indexer **Hadamard-rotiert und dann auf ein festes Gitter
quantisiert**, ist genau die TurboQuant-Idee — von DeepSeek unabhängig für
denselben Tensor gewählt. Gutes Zeichen für Plan 04.

## Attention-Architektur

V4 ist **absorbierte MLA mit hybridem Cache**:

- **Ein** KV-Vektor pro Position, `head_dim=512`, von allen 64 Heads geteilt.
  `coli_v4_sparse_attention_ref` liest `kv + index * head_dim` als **Key und Value
  zugleich** ([c/deepseek_v4.c:2934](../c/deepseek_v4.c)).
- **Sliding-Window-Ring** `state->kv`, `sliding_window × head_dim` f32, konstant groß.
- **Komprimierter Cache** `state->compressed`, wächst mit `ctx/ratio`.
- **Indexer** wählt per Top-k, welche komprimierten Zeilen überhaupt gesehen werden
  ([:2687](../c/deepseek_v4.c)).
- **Attention-Sinks** pro Head, `attn.attn_sink`, gehen in den Softmax-Nenner.

```
ColiDeepSeekV4WindowAttentionState        c/deepseek_v4.c:1400
  int    window_size, head_dim, layer, ratio
  float *kv                    [window_size × head_dim]   Ring
  float *compressed            [compressed_count × head_dim]
  int    compressed_count, compressed_capacity
  ColiDeepSeekV4CompressorState *compressor
  ColiDeepSeekV4Indexer        *indexer
```

Pro Layer eine Instanz, `session->attention[layer]`
([c/deepseek_v4_internal.h:673](../c/deepseek_v4_internal.h)).

## Die drei Attention-Kopien

**Wichtigste strukturelle Eigenheit des Repos.** `c/deepseek_v4.c` wird von einem
**nicht eingecheckten** `_amalgamate_v4.py` erzeugt (`c/Makefile.deepseek-v4.units`
trägt den Generator-Hinweis) — die committete Datei ist Source of Truth. Der Text
von `deepseek_v4_attention.c` steht **byte-identisch dreimal** darin, jeweils unter
`#define`-Umbenennungen:

| Unit | Zeilen | Struct | `all_kv` | Fenster-Write | Compressor-Write |
|---|---|---|---|---|---|
| `COLI_V4_UNIT_ATTENTION` | 1398–1770 | 1400 | 1676 | 1670 | 1597 |
| `COLI_V4_UNIT_ATTENTION_BATCH` | 1796–2168 | 1798 | 2074 | 2068 | 1995 |
| `COLI_V4_UNIT_ATTENTION_TRANSACTION` | 4570–4942 | 4572 | 4848 | 4842 | 4769 |

Verifiziert identisch:
```bash
sed -n '1398,1770p' c/deepseek_v4.c > /tmp/a1; sed -n '1796,2168p' c/deepseek_v4.c > /tmp/a2
sed -n '4570,4942p' c/deepseek_v4.c > /tmp/a3; diff /tmp/a1 /tmp/a2 && diff /tmp/a1 /tmp/a3
```

Dazu kommt der **Batch-Pfad** `coli_v4_attention_window_batch_ref` ([:2180](../c/deepseek_v4.c)),
der dieselbe Logik pro Item nochmal enthält (`all_kv` bei 2352, Fenster-Write 2328).

**Nichts erzwingt diese Gleichheit heute.** Der Source-Sync-Test aus Plan 02 ist
deshalb Voraussetzung für alles Weitere, nicht Beiwerk.

## TurboQuant — Kern in einem Absatz

Pro 128er-Gruppe: L2-Norm ziehen → normalisieren → WHT-Rotation → jedes Element
auf ein **festes** Zentroid runden → korrigierte Norm `grp_norm/recon_norm` als
fp16 speichern. Die Rotation macht die Verteilung isotrop-gaußförmig, deshalb
reicht ein statisches Codebuch ohne per-Block-Scale/Zero-Point.

Quelle: `reference/llama-cpp-turboquant/ggml/src/ggml-turbo-quant.c`
(arXiv 2504.19874). Blocklayouts in `ggml/src/ggml-common.h:296–354` — die
Byte-Kommentare dort sind veraltet, die `static_assert`s stimmen.

| Codec | Bytes/128 | bpw | Zeile @512 | vs f32 | verlustfrei? |
|---|---|---|---|---|---|
| f32 (heute) | 512 | 32 | 2048 B | 1× | — |
| **native** (fp8+bf16, Paper) | — | 9.1 | **583 B** | **3.5×** | **ja, bit-exakt** |
| turbo4 `{fp16; u8[64]}` | 66 | 4.125 | 264 B | 7.8× | nein |
| turbo3 `{fp16; u8[32]; u8[16]}` | 50 | 3.125 | 200 B | 10.2× | nein |
| turbo2 `{fp16; u8[32]}` | 34 | 2.125 | 136 B | 15.1× | nein |

Indexer-Zeile (128 Dims): f32 512 B → **native fp4 68 B (7.5×, bit-exakt)** → turbo3 50 B.

`native` = `uint8 q[448]` + `uint8 e8m0[7]` + `uint16 bf16_rope[64]` = 583 B.
Referenzqualität turbo aus dem Fork: turbo3 Cosine ≈ 1.0, turbo4 ≈ 0.9956 — gemessen
auf **kontinuierlichen** Daten, nicht auf bereits fp8-quantisierten. Siehe Plan 04.

### KV-Bilanz mit den echten Zahlen

Pro Token, wachsender Anteil (21 CSA-Layer à `1/4` Zeile + `1/4` Indexerzeile,
20 HCA-Layer à `1/128` Zeile):

| Kontext | f32 (heute) | native | turbo3 |
|---|---|---|---|
| 32k | 0.42 GiB | 0.11 GiB | 0.04 GiB |
| 128k | **1.68 GiB** | 0.43 GiB | 0.16 GiB |
| 1M | **13.4 GiB** | 3.4 GiB | 1.3 GiB |

**Gelesene Bytes pro Token** — CSA-Layer lesen konstant `n_win + top-k` = 640
Zeilen, HCA-Layer lesen `n_win +` *alle* komprimierten Einträge, weil HCA keine
Sparse Attention hat:

| Kontext | f32 | native | turbo3 |
|---|---|---|---|
| 128k | 40.6 MB/Token | 11.6 MB | 4.0 MB |
| 1M | **353 MB/Token** | 100 MB | **34 MB** |

Bei ~40 GB/s DDR5 sind 353 MB/Token rund **8.8 s pro Token** allein fürs KV-Lesen
im 1M-Kontext. Die HCA-Layer, nicht die CSA-Layer, dominieren das — und dagegen
helfen Codec (Plan 03/04) und Flash-Kernel (Plan 02/05) gemeinsam.

Drei Dinge, die beim Port zählen:

1. **Zentroide sind auf `N(0, 1/√128)` kalibriert.** Eine 512-Punkt-WHT würde sie um
   `√(128/512)` verfälschen. Bei 128er-Gruppen bleiben, vier pro KV-Zeile.
2. **Die WHT ist orthogonal** (`D(s2)·H/√n·D(s1)`, Produkt orthogonaler Matrizen),
   also `<Rx,Ry> = <x,y>`. Scores sind rotationsinvariant.
   Aber: in V4 ist derselbe Vektor K **und** V, die Value-Akkumulation ist es nicht.
   Weil sie linear ist (`Σpᵢ·R·vᵢ = R·Σpᵢ·vᵢ`), reicht **eine** inverse WHT auf dem
   fertigen Head-Output — O(heads) statt O(topk).
3. **Der QJL-Teil ist toter Code.** Bei `TURBO4_USE_4BIT=1` (Default) ist turbo4 reines
   4-bit-PolarQuant; `turbo_init_rotation`, `turbo_init_qjl`, `matvec` und die zwei
   64 KB-Matrizen laufen nie. Der Port braucht ~200 Zeilen statt 1030.

## CUDA-Ausgangslage

Wiederverwendbar aus [c/backend_cuda.h](../c/backend_cuda.h):

- `coli_cuda_init/shutdown/mem_info/stats`, `coli_cuda_pipe_alloc/free/upload/download/sync`
- `coli_cuda_pipe_scratch`, `coli_cuda_pipe_rmsnorm/rope/silu_mul/add/copy2d`
- **`coli_cuda_matmul` mit `fmt=8`** = fp8-e4m3, ein Byte pro Gewicht, Scales pro
  **128×128**-Block ([c/backend_cuda.cu:1066](../c/backend_cuda.cu)) — genau V4s
  Dense-Format. V4 dispatcht seine fp8-Matmuls ohnehin durch dasselbe shared
  `matmul_fp8` aus `quant.h` ([c/deepseek_v4.c:9946](../c/deepseek_v4.c)).
- `coli_cuda_fp8_set_lut` muss vor jedem fmt=8-Upload aufgerufen sein.

**Nicht** wiederverwendbar: `coli_cuda_attention_absorb*` — auf GLMs Geometrie
zugeschnitten (separates `kv_b`, latent/rope getrennt, kein Top-k, keine Sinks).

**Vorbild für einen modellspezifischen Backend:** `backend_cuda_ink.{cu,h}` für
Inkling, bewusst winzige API, per `INK_CUDA_OBJ` ([c/Makefile:404](../c/Makefile))
und Objektregel ([:611](../c/Makefile)) verdrahtet. Dem folgt `backend_cuda_v4.{cu,h}`.

### `packed_rows8` — kein Hindernis

Naheliegender Verdacht: `packed_rows8`
([c/deepseek_v4_internal.h:153](../c/deepseek_v4_internal.h)) beschreibe eine
feinere Scale-Granularität (8×128 statt 128×128), die `coli_cuda_matmul` nicht kann.
**Stimmt nicht.**

`v4_fp8_pack_rows8_inplace` ([c/deepseek_v4.c:488](../c/deepseek_v4.c)) ist eine
reine **Byte-Layout-Transposition innerhalb von 8-Zeilen-Kacheln**, nach dem Laden
in-place angewandt, ausschließlich unter `__AVX2__`, für den AVX2-Zielkernel. Die
Scales werden getrennt über `st_read_scale_f32` gelesen und behalten ihre
128×128-Granularität aus `add_fp8` ([:343](../c/deepseek_v4.c)). `block_rows = 8`
in der `ColiTensorView` beschreibt die *Gewichtskachelung*, nicht die
Scale-Auflösung.

Für GPU-Tensoren wird der Repack schlicht **übersprungen** — dann liegt row-major
fp8 mit 128×128-f32-Scales vor, exakt das fmt=8-Format. Kein Sonderkernel, kein
Präzisionsverlust. Gilt genauso für DSpark, dessen `v4_ds_pack_rows8`
([c/deepseek_v4_dspark.inc:113](../c/deepseek_v4_dspark.inc)) dieselbe
Transposition mit demselben Kommentar ist.

## Konventionen

**Env-Knöpfe.** `V4_*` für Nutzerknöpfe (`V4_MTP`, `V4_DRAFT`, `V4_NGRAM`),
`COLI_V4_*` für Engine-Interna (`COLI_V4_DIRECT`, `COLI_V4_AUTOPIN`).
Neue Knöpfe dieses Branches:

| Variable | Default | Phase |
|---|---|---|
| `V4_SCRATCH_MB` | 512 | 01 |
| `V4_KV` | `f32` | 03 |
| `V4_KV_INDEX` | `f32` | 03 |
| `V4_KV_ROPE_BF16` | 0 | 04 |
| `V4_KV_ROTATED` | 0 | 04 |
| `V4_FLASH` | 1 | 02 |
| `V4_VRAM` | 0 | 05–08 |

**Default bleibt überall die heutige Semantik.** Die README garantiert, dass die
Default-Policy Modellpräzision und Router-Semantik nicht still ändert.

**Test-Gates.** Eine Make-Regel `tests/test_x$(EXE):` in `c/Makefile` genügt —
`TEST_RULES` ([c/Makefile:360](../c/Makefile)) liest die Regeln per `sed` und macht
daraus automatisch Gates. Keine zentrale Liste, kein Merge-Konflikt.

**Semantik-Grenzen, die jeder Phasenplan respektiert:**

- Der **Indexer** entscheidet per Top-k, *welche* Tokens gesehen werden — das ist
  Router-Semantik. Quantisierung dort ist immer opt-in und getrennt schaltbar.
- Der `head_argmax`-Kommentar ([c/deepseek_v4.c:6774](../c/deepseek_v4.c)) hält
  ausdrücklich fest, dass Akkumulationsreihenfolge und Vokabularordnung erhalten
  bleiben, damit Logits und Tie-Breaking sich nicht ändern. Eine GPU-Reduktion
  ändert das — deshalb Toleranz und opt-in.

## Phasenübersicht

| # | Plan | Liefert | RAM | VRAM |
|---|---|---|---|---|
| 01 | [Messen und RAM-Budget](01-measure-and-ram-budget.md) | Ist-Zahlen, `RAM_GB`, `V4_SCRATCH_MB` | +3.0 GiB | — |
| 02 | [Flash Attention](02-flash-attention.md) | Online-Softmax, `all_kv` weg, Source-Sync-Test | — | — |
| 03 | [KV-Codec](03-kv-codec.md) | **natives fp8+bf16/fp4, bit-exakt**, `context_bytes` folgt | **+1.25 GiB** | — |
| 04 | [TurboQuant](04-turboquant.md) | turbo2/3/4 als verlustbehafteter Tier | +0.27 GiB | — |
| 05 | [CUDA-Attention](05-cuda-attention.md) | `backend_cuda_v4`, Flash-Kernel, KV in VRAM | +0.9 GiB | −0.6 |
| 06 | [Dense in VRAM](06-dense-vram.md) | fp8-Residenz + Matmuls auf GPU | +6.3 GiB | −6.3 |
| 07 | [Head und DSpark in VRAM](07-head-dspark-vram.md) | Head-Matvec + Drafter auf GPU | +2.3 GiB | −2.3 |
| 08 | [VRAM-Planner](08-vram-planner.md) | Stufenplanung, 4070-Profil | — | — |
| 09 | [Arch / CachyOS](09-arch-cachyos.md) | `omp_tune.h`, THP, CUDA-Pfade | — | — |
| 10 | [Dual-Streaming](10-dual-streaming.md) | Mirror-Maschinerie nach V4, gewichtete Stripes | — | — |
| 11 | [Rückbau auf V4](11-strip-to-v4.md) | andere Motoren raus | — | — |

Referenzdokumente ohne Nummer: [paper-deepseek-v4.md](paper-deepseek-v4.md) (das
Paper), [llamacpp-deepseek-v4.md](llamacpp-deepseek-v4.md) (die
llama.cpp-Implementierung und ihre Optimierungen) und
[deepspec-dspark.md](deepspec-dspark.md) (DeepSeeks eigene DSpark-Referenz).

Arbeitsregeln, Branch-/PR-Schema und die Pflege dieser Pläne stehen in
[AGENTS.md](../AGENTS.md).

**Reihenfolge.** 09 trägt die Nummer der Vollständigkeit halber, aber sein erster
Commit gehört **direkt hinter 01**: `deepseek_v4.c` ist die einzige Engine ohne
`omp_tune.h` und läuft damit auf allen logischen Kernen statt auf den physischen.
Der Header misst dafür +2.3× (Zen3, #718). Drei Zeilen, unabhängig vom Rest.

**Wenn die Zeit knapp wird:** `01 → 09.1 → 03 → 06` ist die Reihenfolge mit dem
meisten Ertrag pro Aufwand. 02 ist Voraussetzung für 05, und der Source-Sync-Test
aus 02 sollte in jedem Fall früh kommen.

**03 ist nach dem Paper aufgewertet.** Was dort passiert, ist keine Quantisierung,
sondern das Ende einer Fehlbuchung: 3.5× auf den Haupt-KV und 7.5× auf den Indexer,
**bit-exakt**. Damit steht es in einer Reihe mit 01 und 09.1 — messbarer Gewinn
ohne Semantikfrage. 04 ist danach optional: es holt noch einmal Faktor ~3, aber
verlustbehaftet, und wenn das VRAM-Budget nach 03 schließt, braucht es das nicht.

**10 ist unabhängig von allem anderen** und adressiert die Kostenstelle, die nach
den VRAM-Phasen übrigbleibt: bei ~20 % Expert-Residenz liest jeder Token ~2.6 GB
von der Platte. Es lässt sich jederzeit einschieben.

**11 zuletzt.** Er entfernt nur und braucht als Vorlage, was er löscht.

## Gemeinsame Verifikation

```bash
make -C c test && make -C c check
```

```bash
make -C c deepseek-v4-tiny-check
```

Das Tiny-Fixture braucht torch+transformers CPU-only
([c/tools/requirements-deepseek-v4-tiny.txt](../c/tools/requirements-deepseek-v4-tiny.txt)).
Es muss in **jeder** Phase mit den Defaults token-identisch bleiben — das ist der
eigentliche Regressionsbeweis.

Sobald der Checkpoint da ist (`hf download deepseek-ai/DeepSeek-V4-Flash-0731`,
~167 GB) greift zusätzlich:

```bash
make -C c deepseek-v4-oracle MODEL=/pfad/DeepSeek-V4-Flash MEMORY_GB=28
```

## Was dieser Branch nicht macht

- **Geroutete Experten bleiben auf der CPU.** Streaming und Tiering der 160 GB ist
  ein eigenes Projekt; dieser Branch macht nur mehr RAM dafür frei.
- **Andere Engines bleiben unangetastet.** `colibri.c` (GLM), `inkling.c`,
  `kimi_k3.c`, `olmoe.c` und der gemeinsame `backend_cuda.cu` werden nicht
  angefasst. Der tote Kommentar zu KV8/TQ in
  [c/colibri.c:3567](../c/colibri.c) bleibt stehen.
- **Kein On-Disk-Format ändert sich.** `kv_persist.h` (COLIKV1) gehört zu
  `colibri.c`; `kv_prefix.h` speichert nur Token-IDs und ist von Codec-Änderungen
  unberührt.
