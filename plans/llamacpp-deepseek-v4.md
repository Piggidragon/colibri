# llama.cpp als Referenz für DeepSeek V4

Quelle: `reference/llama-cpp-turboquant`, Snapshot **2026-08-05** (`284ffc731`) —
der TurboQuant-Fork von llama.cpp, der zugleich die V4-Arbeit trägt.

Nicht committen (steht in `.gitignore`).

## Was dort existiert

| Arch | Datei | Zeilen | Was |
|---|---|---|---|
| `LLM_ARCH_DEEPSEEK4` | `src/models/deepseek4.cpp` | 1544 | **Das Modell** — CSA, HCA, mHC, Indexer, MTP |
| `LLM_ARCH_DFLASH` | `src/models/dflash.cpp` | 755 | **Drafter-Arch** für spekulatives Dekodieren |

**`DFLASH` ist nicht DeepSeek-V4-Flash.** Trotz des Namens ist es eine
Drafter-Architektur (`LLM_KV_DFLASH_TARGET_LAYER_IDS`, `..._BLOCK_SIZE`,
`..._MASK_TOKEN_ID`). Das 284B-Modell heißt dort `deepseek4` und deckt Pro und
Flash mit derselben Implementierung ab — der Unterschied sind nur Hyperparameter.

### DSpark ist vorhanden

Als Erweiterung des DFlash-Backbones, nicht als eigene Arch
(`src/models/dflash.cpp:87`):

```
// DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
```

Belege im Baum:

- `LLM_TENSOR_DSPARK_MARKOV_W1/W2`, `LLM_TENSOR_DSPARK_CONF_PROJ`
  (`src/llama-arch.h:651`, `src/llama-arch.cpp:647`)
- `build_dspark_markov_head` (`src/models/dflash.cpp:264`)
- `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK` und der Akzeptanzloop mit
  Anchor-first-Blocklayout (`common/speculative.cpp:933` ff., `common/common.h:176`)
- Konverter: `convert_hf_to_gguf.py --dspark` → `DeepseekV4DSparkModel`
- Commit `07e01ec54 DeepseekV4 MTP + DSpark (#25784)`

`dflash.cpp:23` beschreibt das Backbone als *„DeepSeek-V4 DSpark backbone: stages
are full DSV4 blocks, uniform sliding window (the draft KV ring)"* und verlangt
`sqrtsoftplus`-MoE-Scoring sowie unkomprimierte Attention auf allen Stufen —
dieselben Invarianten, die colibris `deepseek_v4_dspark.inc` prüft.

> Falls du DSpark im *Upstream* nicht gefunden hast: plausibel. PR #25784 kann in
> diesem Fork vorliegen, ohne in `master` gemerged zu sein. Im Fork ist es da.

## Struktur von `deepseek4.cpp`

Die Funktionen bilden 1:1 die Paper-Abschnitte ab und sind der beste
Kreuzvergleich für colibris Implementierung:

| Funktion | Zeile | Paper |
|---|---|---|
| `build_hc_pre` / `build_hc_sinkhorn` / `build_hc_post` / `build_hc_head` | 283–464 | 2.2 mHC |
| `build_overlap_compressed_kv_from_state` | 522 | 2.3.1 CSA-Doppelserie mit Überlappung |
| `build_hca_compressed_kv_from_state` | 464 | 2.3.2 HCA-Kompression |
| `build_lid_top_k` | 606 | 2.3.1 Lightning-Indexer-Top-k |
| `build_top_k_mask` | 703 | Top-k → Attention-Maske |
| `build_csa_lid_attention` | 732 | CSA-Kernattention |
| `build_hca_attention` | 793 | HCA-Kernattention |
| `build_raw_attention` | 848 | SWA-Zweig |
| `dsv4_build_state_snapshot` / `_restore` | 215–262 | 3.5.1 State Cache |

`dsv4_build_state_snapshot`/`_restore` sind die Entsprechung zu colibris
`coli_v4_attention_snapshot_*` — dort ebenfalls, weil der unkomprimierte Rest der
Kompressionsfenster mitgeführt werden muss.

---

## Optimierungsfunde

### 1. `GGML_OP_LIGHTNING_INDEXER` — dedizierter Op mit CUDA-Kernel

`ggml/include/ggml.h:579`, Implementierungen in
`ggml/src/ggml-cuda/lightning-indexer.cu` (588 Zeilen) und dem CPU-Pendant.
Commit `00f5442cc ggml : add GGML_OP_LIGHTNING_INDEXER that implements DeepSeek
V3.2/V4 lightning indexer (#24231)`.

Der CUDA-Kernel nutzt **WMMA/Tensor-Cores** (`lightning_indexer_kernel_wmma`,
Template über `WARPS_PER_BLOCK, K_VECS_PER_BLOCK, N_EMBD, N_HEAD, TYPE_K`), lädt
Gewichte und Q-Kacheln in Shared Memory und verarbeitet 8 Heads × 16 K-Embed pro
inner loop.

**Relevanz:** colibris `coli_v4_indexer_step` ist eine skalare Schleife über alle
`state->count` Kandidaten mit `qsort` am Ende
([c/deepseek_v4.c:2827](../c/deepseek_v4.c)). Bei 1M Kontext sind das 250k
Kandidaten × 64 Heads × 128 Dims **pro Token und pro CSA-Layer**. Das ist neben
den HCA-Layern der zweite große Posten — und der llama.cpp-Kernel ist eine
direkte Vorlage. Gehört zu [05-cuda-attention.md](05-cuda-attention.md).

Zusätzlich: das `qsort` über alle Kandidaten (`descending_score`) ist für ein
Top-512 aus 250k die falsche Datenstruktur — ein partieller Select (Heap oder
`nth_element`-artig) ist O(n) statt O(n log n). Eigener kleiner Gewinn,
unabhängig von CUDA.

### 2. Fusionierte Hyper-Connection-Ops

Commit `0dc74e332 DeepseekV4: Add fused hyper-connection ops (#25585)`.

colibri rechnet mHC in `coli_v4_hc_pre` mit separaten Schritten inklusive
Sinkhorn-Iterationen (`hc_sinkhorn_iters = 20` laut Paper). Bei 43 Layern × 20
Iterationen pro Token ist das messbar. Lohnt eine Messung, bevor man es anfasst.

### 3. Weniger Graph-Splits

Commit `33a75f41c DeepseekV4: reduce graph splits (#25702)`. Für colibri nicht
direkt übertragbar (kein Graph-Executor), aber die zugrundeliegende Beobachtung —
die V4-Attention zerfällt in viele kleine Operationen — gilt auch dort.

### 4. Quantisierter KV-Cache für dsv4 ist gelöst

Commits `024c46ae4 llama: fix quantized kv-cache for dsv4 (#25202)` und
`571d0d540 model: rotate injected K/V cache for DFlash (#25823)`.

Der zweite ist interessant: `build_csa_lid_attention:751` rotiert Q **und** KV mit
derselben Hadamard-Matrix und dreht die Ausgabe hinterher zurück:

```c
if (k_rot) { q = llama_mul_mat_hadamard(ctx0, q, k_rot);
             kv = llama_mul_mat_hadamard(ctx0, kv, k_rot); }
...
out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
if (k_rot) { out = llama_mul_mat_hadamard(ctx0, out, k_rot); }
```

**Das ist exakt der rotierte Score-Raum aus [04-turboquant.md](04-turboquant.md)** —
inklusive der einen Rücktransformation auf der Ausgabe statt pro KV-Zeile.
Unabhängige Bestätigung, dass die Konstruktion für V4 trägt und dass die
Reihenfolge (rotieren → Attention → zurückdrehen) korrekt ist.

### 5. Was llama.cpp **nicht** hat

- **Kein Expert-Streaming von der Platte.** llama.cpp erwartet das Modell im RAM
  oder mmap-baren Speicher. Colibris gesamte Tiering-Maschinerie
  (`expert_store.h`, `tier.h`, Pinning, `COLI_MODEL_MIRROR`) hat dort keine
  Entsprechung. Für ein 284B-Modell auf 32 GB ist genau das der entscheidende
  Teil — hier ist colibri voraus und llama.cpp keine Hilfe.
- **Kein DRAM-loses-SSD-Bewusstsein**, keine Multi-Drive-Verteilung.
- Kein Äquivalent zum nativen fp8+bf16-KV aus [03-kv-codec.md](03-kv-codec.md);
  llama.cpp quantisiert den KV generisch über seine `cache-type-k/v`-Typen.

## Woher Zahlen nehmen, wenn colibri und llama.cpp abweichen

Beide sind Implementierungen desselben Papers. Bei Abweichungen gilt:

1. Das **Paper** ([paper-deepseek-v4.md](paper-deepseek-v4.md)) für die Semantik.
2. Die **offizielle Referenzimplementierung**, auf die das Paper verweist
   (`huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`), für Details,
   die das Paper offenlässt — es sagt selbst, dass es „tiny details" auslässt.
3. `deepseek4.cpp` als dritte Meinung, besonders bei Reihenfolgefragen
   (RoPE-Inverse, Normalisierungsstellen, Sink-Behandlung).

colibris `ref.json`-Oracle bleibt der Maßstab für Regressionen — die Abweichung
gegen llama.cpp ist ein Hinweis, kein Beweis.
