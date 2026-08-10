# 00 — Referenzplan: DeepSeek V4 auf 32 GB RAM + RTX 4070

Übergeordnetes Dokument. Hier stehen Zielbild, Hardware-Budget, Codekarte und
Konventionen; die Phasenpläne `01`–`13` setzen das voraus und wiederholen es nicht.

- Jede Phase bekommt einen eigenen Branch und PR; Planänderungen laufen auf dem
  Branch der zugehörigen Phase mit (Schema und Reihenfolge in
  [AGENTS.md](../AGENTS.md))
- Lizenz: Repo ist Apache-2.0, der TurboQuant-Referenz-Fork MIT → Attribution im
  portierten Header plus vollständiger Lizenztext in `THIRD_PARTY_NOTICES`.
  `reference/` selbst wird **nicht** committet.
- In `reference/` liegen ausschließlich **Lesequellen**, kein Build-Input:
  `DeepSpec`, `llama-cpp-turboquant` und `FlashMLA`
  (siehe [FlashMLA](#flashmla--vendor-referenz-nicht-linkbar)).

## Zielbild

Nicht „KV quantisieren", sondern **DeepSeek-V4-Flash-0731 (284B, ~167 GB) plus den
DSpark-Drafter auf genau dieser Maschine so gut wie möglich fahren**:

| | |
|---|---|
| CPU | Intel **i5-13400F** — 6 P-Cores + 4 E-Cores, 16 Threads, **kein AVX-512**, **keine iGPU** |
| RAM | 32 GB **DDR4-3200**, Dual-Channel — ~45 GB/s, siehe Bandbreitenabschnitt |
| GPU | RTX 4070, 12 GB — **headless im V4-Betrieb**, ~11.7 GiB nutzbar |
| Laufwerk A | 1 TB NVMe **Gen4**, DRAM-los (HMB) — trägt das Modell |
| Laufwerk B | 512 GB SSD **Gen3**, DRAM-los (HMB) — **optional**, zweite Kopie |
| OS | CachyOS (Arch-Familie), **Linux x86-64 only** |

Beide Laufwerke fassen je eine vollständige Modellkopie (167 GB < 512 GB) —
Grundlage für Dual-Streaming, siehe [10-dual-streaming.md](10-dual-streaming.md).
**Laufwerk B ist dabei ein Beschleuniger, keine Voraussetzung:** ohne es läuft
alles unverändert, nur mit ~7 statt ~10 GB/s Lesebandbreite. Kein Plan außer 10
kennt es, und auch 10 hat den Einzellaufwerk-Betrieb als Default und als
Pflicht-Abnahme.

**Checkpoint:** [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
lokal vorhanden unter

```
~/Services/models/colibri/deepseek-v4-flash
```

Das ist das **einzige** Modellverzeichnis. DSpark liegt im selben Checkpoint unter dem
Präfix `mtp.<stage>.` — colibri sucht die Drafter-Tensoren in
`engine->target_index` (`v4_dspark_markov_probe`,
[c/deepseek_v4.c:7571](../c/deepseek_v4.c); Namensbau `v4_ds_name`,
[c/deepseek_v4_dspark.inc:193](../c/deepseek_v4_dspark.inc)), nicht in einem
zweiten Verzeichnis. Ein separates DSpark-Repo gibt es nicht zu laden.

Andere Modelle sind in diesem Fork ausdrücklich kein Ziel; der Rückbau steht in
[11-strip-to-v4.md](11-strip-to-v4.md) und kommt **zuletzt**, weil mehrere Pläne
sich Referenzcode aus den zu löschenden Motoren holen.

Die bindende Größe ist die Expert-Cache-Residenz: **~147 GB** geroutete Experten
(256 Experten × 43 Layer × 13.37 MB) gegen einen RAM-Cache, dessen tatsächliche
Größe stark von `CTX` und den daraus skalierten Prefill-Buffern abhängt. Alles,
was nicht gecacht ist,
wird pro Token von der Platte gestreamt. Jedes freigemachte GiB RAM ist direkt
mehr resident gehaltene Experten.

Phase 05 hat die opt-in CUDA-Anbindung für V4 hergestellt: `CUDA=1 V4_VRAM=1`
spiegelt den Attention-KV und führt Sparse MLA auf der RTX 4070 aus. Phase 06
nutzt denselben Build nun auch für Dense-FP8. Ohne diese beiden Opt-ins bleibt
der CPU-Pfad das unveränderte Defaultverhalten.

Das Dense-Gesamtinventar (6.267 GiB), der BF16-Head (0.99 GiB) und DSpark
(~1.17 GiB) belegen vor den GPU-Phasen RAM. Phase 06 hat am echten Checkpoint
gezeigt, dass davon **5.456 GiB Dense-FP8** verschiebbar sind; 0.810 GiB
BF16/f32/i64 bleiben auf dem Host. Die verschiebbaren Dense-Tensoren, Head und
der residente DSpark-Anteil passen zusammen in 12 GB VRAM. **Das ist der
Haupthebel.**

## RAM-Bilanz

Abgeleitet aus `build_runtime_plan` ([c/deepseek_v4.c:1207](../c/deepseek_v4.c))
und `coli_v4_resource_plan_compute` ([:877](../c/deepseek_v4.c)). Alle Zahlen bei
`CTX=131072` (128k) — derselbe Kontext, den Plan 03/05/08 für ihre Deltas
verwenden, damit Baseline und Phasen-Gewinne vergleichbar bleiben.

Zur KV-Reserve kommen zwei volle Prefill-Zustandsbuffer, die
`coli_v4_session_create` tatsächlich allokiert:

```
session_state = CTX × hc_mult × hidden_size × sizeof(float) × 2
```

Bei der bindenden Paper-Geometrie (`hc_mult=4`, `hidden_size=4096`) sind das
**16.0 GiB bei 128k** beziehungsweise 4.0 GiB bei 32k. Selbst bei einer
Geometrie mit `hc_mult=2`, `hidden_size=2048` wären es bei 128k noch 4.0 GiB.
Eine frühere Fassung des Planners setzte hier konstant 64 Token an und reservierte
nur 8 MiB; die daraus abgeleiteten ~14.4 GiB Expert-Cache waren deshalb falsch.

Für `CTX=131072` besteht `runtime_other` auf der Paper-Geometrie vor DSpark aus:

```
KV f32                         1.68 GiB
zwei Prefill-Zustandsbuffer   16.00 GiB
Scratch (Default / Phase 01)   0.50 / 0.125 GiB
```

Mit aktivem DSpark kommen beim Default `V4_MTP_GB=0.45` exakt
`0.45 × 10⁹ + 768 × 2²⁰ = 1 255 306 368 B` hinzu: **1.255 GB, aber
1.169 GiB**, in dieser Bilanz also rund **1.17 GiB**.

Hinzu kommen die 0.32 GiB Layerreserve unten. Ob Dense und Head resident bleiben,
ist bei 32 GiB damit eine **Tier-Entscheidung**; eine additive Baseline, die alle
drei Posten plus einen großen Expert-Cache gleichzeitig als resident annimmt, ist
nicht zulässig. Plan 01 muss die tatsächliche `ram_tiers`-Entscheidung messen.

**Die `2 × maximum_layer_bytes`-Zeile wird leicht übersehen.**
`coli_v4_resource_plan_compute` addiert sie in die Runtime-Reserve
([:903](../c/deepseek_v4.c)), *zusätzlich* zu `runtime_other`:

```c
if (multiply_u64(inputs->maximum_layer_bytes, 2, &layers_twice) ||
    add_u64(layers_twice, inputs->runtime_other_bytes, &plan->runtime_reserve_bytes))
```

`maximum_layer_bytes` ist der größte Einzel-Layer aus der Dense-Inventur
([:1230](../c/deepseek_v4.c)) — auf dieser Konfiguration Layer 2 (CSA **und**
Hash-Router, also `ffn.gate.tid2eid` statt `.bias`) mit ~0.162 GiB. Zwei davon
sind ~0.32 GiB, die keiner Phase gehören und in keiner Phase verschwinden. Eine
frühere Fassung dieser Bilanz ließ sie ganz weg.

Die Gewinne aus der Phasenübersicht unten summieren sich **nicht** einfach über
Tierwechsel hinweg. Phase 03 senkt den gesamten KV von 1.68 auf 0.43 GiB
(**+1.25 GiB**). Phase 05 spiegelt davon nur den Attention-KV auf das Gerät und
behält den Host-Fallback; der RAM-Gewinn entsteht erst mit exklusiver
Device-Eigentümerschaft in Phase 08 und beträgt 0.388 GiB. Der native
Lightning-Indexer (0.044 GiB bei 128k) bleibt auf der CPU.

Eine Gegenprobe für den theoretischen Endzustand, nachdem Attention-KV, der
verschiebbare Dense-Anteil, Head und DSpark aus dem RAM verschoben sind, zeigt
den verbleibenden Kontextpreis. Der CPU-seitige Indexer-KV und die gemessenen
0.810 GiB nicht-FP8-Dense-Tensoren bleiben dabei abzuziehen:

```
CTX=32k:  28.0 − 0.32 Layer − 4.00 State − 0.125 Scratch − 0.011 Index − 0.810 Dense = 22.73 GiB
CTX=128k: 28.0 − 0.32 Layer − 16.0 State − 0.125 Scratch − 0.044 Index − 0.810 Dense = 10.70 GiB
```

Das sind Obergrenzen vor Slot-Rundung, keine Messwerte. 128k kostet im aktuellen
Full-Prefill-Pfad also rund 12 GiB Expert-Cache gegenüber 32k, auch wenn der KV
später vollständig im VRAM liegt. Eine spätere Chunk-/Streaming-Prefill-
Optimierung könnte diesen Posten ändern; keiner der bestehenden Pläne tut das.

Phase 04 (TurboQuant) zahlt nach exklusiver Attention-KV-Residenz fast vollständig
im VRAM ein (siehe Budget unten, wo es beim 1M-Profil den Ausschlag gibt). Im RAM
bleibt nur die getrennte Indexer-Ersparnis: bei 128k 0.044 GiB `native` gegenüber
0.032 GiB `turbo3`, sofern die semantische Änderung opt-in akzeptiert wird.

`per_slot = num_hidden_layers × expert_record_bytes = 43 × 13.37 MB ≈ 575 MB`
(0.535 GiB), daher die Slot-Zahlen. `expert_record_bytes` summiert die `nbytes`
von w1/w2/w3 **plus** deren `.scale`-Tensoren von `layers.0.ffn.experts.0`
([:892](../c/deepseek_v4.c)) — also 12.58 MB fp4-Gewichte + 0.79 MB E8M0-Scales.

## VRAM-Budget (4070, **headless**)

Die Maschine läuft im V4-Betrieb ohne Display — kein Compositor, keine
Browser-Beschleunigung. `nvidia-smi` meldet für einen 4070 rund 12282 MiB gesamt;
nach Treiberreserve bleiben **~11.7 GiB** nutzbar.

```
nutzbar nach Treiberreserve            11.70 GiB
 − Planner-Reserve (Plan 08)            −1.00   free/8, geklemmt 256 MiB..1 GiB
                                       ──────
 = Budget für die Stufen               10.70 GiB

dense fp8          5.456  Phase 6 — Checkpoint-Inventur: 5.455 GiB Weights
                           + 0.001 GiB expandierte Scales; 0.810 GiB sonstige
                           Dense-Tensoren bleiben im RAM
head bf16          0.99   Phase 7 — 129280 × 4096 × 2 B = 1.059 GB = 0.986 GiB
DSpark             0.32   Phase 7 — **gemessen** 330 MiB für den Backbone beim
                           Lazy-Upload; die 768-MiB-Marge in der RAM-Reserve
                           deckt Head-/Scratch-Bedarf ab, ist kein VRAM-Tensor
workspace         ~0.30
                  ─────
Fixkosten          7.07 GiB   → ~3.63 GiB bleiben für den KV
```

**Vier Korrekturen gegenüber früheren Fassungen, alle nach unten.**

1. Die **Planner-Reserve aus Plan 08** (`free/8`, geklemmt auf 256 MiB…1 GiB)
   fehlte hier ganz. Auf einer 11.7-GiB-Karte greift der obere Clamp, also volle
   1.0 GiB. Sie ist kein Buchhaltungsposten, sondern der Puffer dafür, dass
   `coli_cuda_mem_info` nur eine Momentaufnahme liefert — wer sie hier wegrechnet,
   plant gegen eine Zahl, die der Planner nie vergibt.
2. Der **Head ist 0.99 GiB beziehungsweise 1.06 GB**. 129280 × 4096 × 2 B =
   1.059 **GB**; eine frühere Doku-Fassung trug daran das falsche GiB-Suffix. In
   Millisekunden gerechnet bleibt 1.059 GB / 45 GB/s ≈ 24 ms (siehe unten); nur
   in einer GiB-Bilanz darf man den Dezimalwert nicht ungeprüft addieren. Gilt
   für den Head. Das Dense-Gesamtinventar sind ebenfalls echte 6.267 GiB
   (nachgerechnet über `coli_v4_layer_plan`), aber nicht alles davon ist FP8.
3. Die echte Phase-06-Inventur trennt **5.456 GiB verschiebbare FP8-Weights und
   Scales** von **0.810 GiB BF16/f32/i64**, die auf dem Host bleiben. Die frühere
   Bilanz behandelte fälschlich alle 6.267 GiB als CUDA-Tensoren.
4. Der **DSpark-Posten war mit 0.56 GiB geschätzt; Phase 07 hat 330 MiB
   gemessen** (0.32 GiB). Die Differenz geht direkt in das KV-Budget: ~3.63
   statt ~3.39 GiB. Die Planner-Reserve wächst mit aktivem Drafter entsprechend
   von 1.00 auf 1.32 GiB, siehe [07-head-dspark-vram.md](07-head-dspark-vram.md).

Der **Attention-KV auf dem Gerät** entscheidet damit die erreichbare
Kontextlänge. Die bislang hier geführten Gesamt-KV-Zahlen enthielten fälschlich
auch den Lightning-Indexer, obwohl Phase 05 ihn auf der CPU auswertet und nicht
hochlädt:

| Kontext | KV f32 | KV `native` | KV turbo3 | passt mit |
|---|---|---|---|---|
| 128k | 1.36 | **0.388** | 0.133 | native (~3.01 GiB übrig) |
| 256k | 2.71 | **0.772** | 0.265 | native (~2.62 GiB übrig) |
| 512k | 5.42 | **1.54** | 0.529 | native (~1.85 GiB übrig) |
| 1M | 10.82 | **3.08** | 1.06 | native nominell (~0.31 GiB), turbo3 komfortabel (~2.33 GiB) |

**Damit ist Phase 4 auch für das 1M-Profil nicht rechnerisch zwingend, aber als
Sicherheitsmarge empfohlen.** `native` braucht bei 1M 3.08 GiB und bekommt
nominell rund 3.39 GiB — nur etwa 0.31 GiB Luft, bevor reale Workspace-Spitzen
oder fremde Belegung eingerechnet sind. Bis einschließlich 512k reicht `native`
mit deutlich mehr Luft. Der getrennte native Indexer belegt bei 1M weitere
0.349 GiB RAM, aber kein VRAM.

Vor dem 1M-Profil trotzdem mit `V4_VRAM_LIMIT_MB` (Plan 08) durchspielen, statt
diese Tabelle zu glauben — sie rechnet mit einer nominellen Kartengröße, und der
Workspace-Posten ist die unsicherste Zahl darin.

Codec und VRAM-Residenz sind keine getrennten Features, sondern Voraussetzung
füreinander. Der Planner aus Phase 8 muss trotzdem pro Stufe einzeln auf RAM
zurückfallen können — headless heißt „vorhersehbar", nicht „unbegrenzt", und ein
Fehlschlag beim Upload muss degradieren statt abzustürzen.

## Speicherbandbreite — die Obergrenze, die keiner der Pläne verschiebt

**Dual-Channel DDR4-3200**: 3200 MT/s × 8 B × 2 Kanäle = **51.2 GB/s** theoretisch,
realistisch **~45 GB/s** bei Streaming-Reads. Das ist die härteste Zahl im ganzen
Vorhaben, weil sie unabhängig von Platte, VRAM und Cache-Trefferquote gilt.

### Der Boden für die Dekodierrate

Jeder Token aktiviert 6 geroutete Experten pro Layer, in **allen 43 Layern**.
Ein Expert-Record ist w1/w2/w3 in fp4 plus E8M0-Scales:

```
3 × [2048 × 4096] × 0.5 B  = 12.6 MB   Gewichte
+ Scales (E8M0 je 32 Werte)  ≈  0.8 MB
                              ─────────
                              ~13.4 MB pro Expert

6 Experten × 43 Layer × 13.4 MB       = ~3.4 GB pro Token
```

Diese 3.4 GB müssen **durch den Speicherbus**, auch bei 100 % Cache-Treffer. Bei
~45 GB/s sind das **~77 ms/Token**, also ein Deckel von **~13 tok/s** — mit
perfektem RAM-Cache, ohne jede Platte, ohne jeden Fehltreffer.

**Kein Plan in diesem Baum verschiebt diesen Deckel.** Sie alle arbeiten daran,
sich ihm zu nähern: im theoretischen 32k-Endprofil passen vor Slot-Rundung rund
16–17 % der Experten in den Cache, bei 128k nur rund 8 %. Bei 17 % Residenz kommen
~2.8 GB der 3.4 GB von der Platte: auf einem Laufwerk (~7 GB/s) sind das ~400 ms
und damit gut 2 tok/s, mit dem optionalen zweiten Laufwerk (~10 GB/s nach Plan 10)
~280 ms und damit gut 3 tok/s. Der Weg von dort auf 13 ist das, was hier zu holen
ist. Darüber hinaus ginge nur mit anderer Hardware oder weniger aktivierten
Parametern.

**Was daraus folgt:**

- **DSpark ist wertvoller als es aussieht.** Ein verifizierter Block von *n*
  Draft-Tokens liest die *Vereinigung* ihrer Experten, nicht *n* volle Sätze —
  bei Überlappung deutlich weniger als *n* × 3.4 GB. Spekulatives Dekodieren
  amortisiert genau den Posten, der den Deckel bildet. `V4_MTP=1 V4_DRAFT=3`
  gehört ins Profil, nicht als Kür.
- **THP (Plan 09) wird wichtiger.** Bei knapper Bandbreite kosten TLB-Misses
  anteilig mehr.
- **Jede vermeidbare Kopie kostet messbar.** 3.4 GB/Token einmal zusätzlich
  umzukopieren wären ~77 ms. Der O_DIRECT-Pfad (`COLI_V4_DIRECT`) und die
  Koaleszenz der Expert-Reads sind deshalb keine Feinheiten.

### Der KV-Lesestrom bei langem Kontext

Aus derselben Bandbreite und den Zahlen in [paper-deepseek-v4.md](paper-deepseek-v4.md).
Zeilenzahlen wie in der KV-Bilanz unten (640 CSA-Zeilen konstant; HCA
`n_win + ctx/128`), ausgewertet bei `ctx=131072` bzw. `ctx=1000000`:

| Kontext | Codec | MB/Token | CPU @45 GB/s | GPU @504 GB/s |
|---|---|---|---|---|
| 128k | f32 | 74.7 | 1.7 ms | — |
| 128k | `native` | 21.3 | 0.47 ms | 0.04 ms |
| 1M | f32 | 353 | **7.8 ms** | — |
| 1M | `native` | 100 | **2.2 ms** | 0.20 ms |
| 1M | turbo3 | 34 | 0.76 ms | **0.07 ms** |

(Eine frühere Fassung dieser Tabelle wies die 1M-Zeile in Sekunden statt
Millisekunden aus — Faktor 1000 zu hoch — und berechnete die 128k-Zeile mit
falschen Zeilenzahlen. Beide Fehler sind hier korrigiert.)

Bei 128k ist der KV bandbreitenmäßig belanglos gegen den ~77-ms-Expertendeckel.
**Bei 1M ist er selbst bei f32 nur ~7.8 ms/Token** — spürbar, aber weit unter dem
Expertendeckel, also kein eigenständiges Argument für „untragbar". Die Zeilen, die
der Attention-Kernel liest, begründen Phase 04/05 damit **nicht**; die Treiber
sind das VRAM-Budget oben (bei 1M passt `native` dort nicht mehr), die
O(Kontext)-Staging-Kopie, die Plan 02 beschreibt — bei 128k allein schon
**~1.4 GB/Token** über alle CSA-Layer (67 MB × 21) — und der Indexer-Scan im
nächsten Abschnitt. Das ist die eigentliche Rechtfertigung für Phase 05: nicht die
Lesebandbreite der Zeilen selbst, sondern die Kopie, die der Flash-Umbau aus
Plan 02 bereits auf der CPU beseitigt und die ein naiver CUDA-Kernel sonst
wiederholen würde.

### Der Lightning-Indexer — der Posten, der in keiner Tabelle stand

**Die Tabelle oben zählt nur die Zeilen, die der Attention-Kernel liest.** Bevor
er das tut, muss der Indexer sie auswählen, und dafür bewertet
`coli_v4_indexer_step` in **jedem Token und jedem CSA-Layer alle** bisher
komprimierten Einträge ([:2893](../c/deepseek_v4.c)):

```c
for (int candidate = 0; !result && candidate < state->count; candidate++) {
    const float *key = state->compressed + (size_t)candidate * dimension;
    for (int head = 0; head < heads; head++)          /* 64 */
        for (int i = 0; i < dimension; i++)           /* 128 */
            dot += query[i] * key[i];
}
```

`state->count` wächst mit `ctx/4`. Pro Token über alle 21 CSA-Layer:

| Kontext | `count` | gelesen f32 | gelesen `native` (fp4) | MAC/Token |
|---|---|---|---|---|
| 128k | 32 768 | **352 MB** | 46.8 MB | 5.6 G |
| 1M | 250 000 | **2.69 GB** | 357 MB | 43 G |

Bei ~45 GB/s sind das **7.8 ms** (128k) bzw. **~60 ms** (1M) allein fürs Lesen —
bei 1M in derselben Größenordnung wie der ~77-ms-Expertendeckel und **eine
Größenordnung über** den 353 MB/Token aus der Tabelle davor, die bis hierher als
der dominante KV-Strom galten.

Und die Bandbreite ist vermutlich nicht einmal die Bindung: die Schleife ist
skalar und steht **außerhalb jedes `#pragma omp`** — als einzige der heißen
V4-Schleifen. 5.6 G MAC/Token einfädig sind auch mit Autovektorisierung
dreistellige Millisekunden.

**Was daraus folgt:**

- Der Indexer ist der stärkste Grund für `V4_KV_INDEX=native` (7.5×,
  bit-exakt) — stärker als alles, was Plan 03 sonst anführt.
- Plan 12 Commit 2 ersetzt das `qsort` dahinter (≈4.5 M Vergleiche bei 1M). Das
  ist richtig, aber es ist der **kleinere** Teil: die Bewertungsschleife davor
  kostet ~2 G Operationen, also rund 400× mehr. Beides gehört in denselben PR.
- Der WMMA-Kernel aus llama.cpp (`ggml/src/ggml-cuda/lightning-indexer.cu`, siehe
  [llamacpp-deepseek-v4.md](llamacpp-deepseek-v4.md)) adressiert genau diese
  Schleife.

**Alles hier ist Arithmetik, keine Messung.** Es ist plausibel, dass der Indexer
bei langem Kontext der größte Einzelposten überhaupt ist und die Reihenfolge in
[AGENTS.md](../AGENTS.md) verschieben sollte — aber das entscheidet der Harness
aus [01-measure-and-ram-budget.md](01-measure-and-ram-budget.md), nicht diese
Tabelle. Bis dahin steht die Zahl hier, damit sie niemand ein zweites Mal
übersieht.

### Der Head

1.059 GB BF16 pro Token (0.99 GiB): **~24 ms** auf DDR4-3200, **~2.1 ms** auf dem
4070. Phase 07 spart also ~22 ms/Token — bei 3.3 tok/s sind das ~7 %, bei
13 tok/s ~29 %. Der Nutzen wächst, je weiter die anderen Phasen kommen.

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
| Fenster-KV | `coli_v4_kv_encode_row` [c/deepseek_v4.c:11843](../c/deepseek_v4.c) | fp8-QDQ Dims 0–447 (Block 64, E8M0), bf16 Dims 448–511 |
| CSA/HCA komprimiert | [:3427](../c/deepseek_v4.c) ff. | dasselbe |
| Indexer komprimiert | [:3437](../c/deepseek_v4.c) | Hadamard-Rotation + fp4-QDQ über alle 128 Dims (Block 32) |

Die letzten beiden Zeilen sind **eine** Codestelle, kein Paar: `coli_v4_compressor_step`
verzweigt über `state->rotate_fp4` zwischen fp8/64 und Hadamard+fp4/32 und wählt
mit `quantized` auch, ob der RoPE-Schwanz ausgespart bleibt. Wer nur die eine
Zeile ändert, ändert beide Ströme.

```c
output[base + i] = coli_e4m3fn_decode(coli_e4m3fn_encode(normalized)) * scale;
scales[base / block_size] = encoded_scale;        /* E8M0-Byte */
```
([:11475](../c/deepseek_v4.c))

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
  zugleich** ([c/deepseek_v4.c:3728](../c/deepseek_v4.c)).
- **Sliding-Window-Ring** `state->kv`, `sliding_window × head_dim` f32, konstant groß.
- **Komprimierter Cache** `state->compressed`, wächst mit `ctx/ratio`.
- **Indexer** wählt per Top-k, welche komprimierten Zeilen überhaupt gesehen werden
  (Struct [:3471](../c/deepseek_v4.c), Bewertung und Auswahl in
  `coli_v4_indexer_step` [:3604](../c/deepseek_v4.c) ff.).
- **Attention-Sinks** pro Head, `attn.attn_sink`, gehen in den Softmax-Nenner.

```
ColiDeepSeekV4WindowAttentionState        c/deepseek_v4.c:2010
  int    window_size, head_dim, layer, ratio
  float *kv                    [window_size × head_dim]   Ring
  float *compressed            [compressed_count × head_dim]
  int    compressed_count, compressed_capacity
  ColiDeepSeekV4CompressorState *compressor
  ColiDeepSeekV4Indexer        *indexer
```

Pro Layer eine Instanz, `session->attention[layer]`
([c/deepseek_v4_internal.h:789](../c/deepseek_v4_internal.h)).

## Die duplizierten Units

**Wichtigste strukturelle Eigenheit des Repos.** `c/deepseek_v4.c` wird von einem
**nicht eingecheckten** `_amalgamate_v4.py` erzeugt (`c/Makefile.deepseek-v4.units`
trägt den Generator-Hinweis) — die committete Datei ist Source of Truth. Mehrere
Quelldateien stehen **byte-identisch mehrfach** darin, jeweils unter
`#define`-Umbenennungen.

**Es sind nicht nur die drei Attention-Kopien.** Eine frühere Fassung dieses
Abschnitts nannte nur sie; damit fehlten genau die Funktionen, die Plan 03 und
Plan 12 anfassen wollen. Vollständig:

| Quelle | Kopien | Units (`#ifdef COLI_V4_UNIT_…`) | Anker-Definition |
|---|---|---|---|
| `deepseek_v4_attention.c` | **3** | `ATTENTION`, `ATTENTION_BATCH`, `ATTENTION_TRANSACTION` | `struct ColiDeepSeekV4WindowAttentionState {` |
| `deepseek_v4_compressor.c` | **2** | `COMPRESSOR`, `COMPRESSOR_SNAPSHOT` | `int coli_v4_compressor_step(` |
| `deepseek_v4_indexer.c` | **2** | `INDEXER`, `INDEXER_SNAPSHOT` | `int coli_v4_indexer_step(` |
| `deepseek_v4_layer.c` | **2** | `LAYER_RESIDENT`, `LAYER` | `int coli_v4_layer_plan(` |

**Hier stehen bewusst keine Zeilennummern.** Sie waren in früheren Fassungen mit
`sed -n 'A,Bp' | diff`-Rezepten hinterlegt und nach den Phasen 05–07 sämtlich
falsch — die Datei ist allein dort um rund 1400 Zeilen gewachsen. Die Grenzen
stehen ohnehin in der Datei selbst:

```bash
grep -n 'COLI_V4_UNIT' c/deepseek_v4.c        # alle Unit-Grenzen
grep -n 'struct ColiDeepSeekV4WindowAttentionState {' c/deepseek_v4.c   # die 3 Kopien
```

`LAYER_RESIDENT` ist der einzige Fall, der **kein** exaktes Duplikat ist: es ist
`deepseek_v4_layer.c` **plus** `v4_fp8_pack_rows8_inplace`
([:496](../c/deepseek_v4.c)) und den resident-spezifischen Rows8-Block in
`coli_v4_layer_load`. `coli_v4_layer_plan`, `coli_v4_layer_validate`,
`coli_v4_layer_free` und `coli_v4_layer_data` stehen trotzdem zweimal da.

Dazu kommt der **Batch-Pfad** `coli_v4_attention_window_batch_ref`
([:2959](../c/deepseek_v4.c)), der dieselbe Attention-Logik pro Item nochmal
enthält — und der ist kein Duplikat, sondern eigener Text.

**Prüfen muss man das nicht von Hand.** `c/tests/test_v4_attention_source.py`
vergleicht alle vier Quellen markerbasiert (Attention 3×, Compressor 2×,
Indexer 2×, für Layer die vier gemeinsamen Definitionen mit der bewussten
Rows8-Ausnahme) und hält zusätzlich fest, dass alle Attention-Aufrufer denselben
Zwei-Quellen-Codec-Pfad und denselben CUDA-Einsprung haben.
`c/tests/test_deepseek_v4_dspark_source.py` tut dasselbe für die
Drafter-Invarianten. Beide laufen in `make -C c check`.

**Nichts erzwingt diese Gleichheit heute.** Der Source-Sync-Test aus Plan 02 ist
deshalb Voraussetzung für alles Weitere, nicht Beiwerk — und er muss **alle vier
Quellen** abdecken, nicht nur die Attention.

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
Die Fork-Dokumentation nannte turbo3 Cosine ≈ 1,0 und turbo4 ≈ 0,9956 auf
**kontinuierlichen** Daten. Der fertige CPU-Port aus Phase 04 misst auf denselben
Zufallsvektoren kontinuierlich / nach dem realen FP8+BF16-Vorgitter stattdessen
turbo2 **0,9411 / 0,9411**, turbo3 **0,9833 / 0,9832** und turbo4
**0,9954 / 0,9954**. Diese reproduzierbaren Port-Werte sind die belastbaren
Schranken für **gleich skalierte** NoPE-/RoPE-Hälften; siehe Plan 04. Bei
RoPE-Skalen von 1/4, 1/16 und 16 bleibt der Gesamt-Cosine von turbo3 zwar bei
~0,983, aber der kleinere Anteil fällt bis 0,507 (RoPE bei 1/16) beziehungsweise
0,838 (NoPE bei 16). Der Gesamtwert allein entscheidet deshalb nicht gegen ein
getrenntes RoPE-Layout.

### KV-Bilanz mit den echten Zahlen

Pro Token, wachsender Anteil (21 CSA-Layer à `1/4` Zeile + `1/4` Indexerzeile,
20 HCA-Layer à `1/128` Zeile):

| Kontext | f32 (heute) | native | turbo3 |
|---|---|---|---|
| 32k | 0.42 GiB | 0.11 GiB | 0.04 GiB |
| 128k | **1.68 GiB** | 0.43 GiB | 0.16 GiB |
| 1M | **13.4 GiB** | 3.4 GiB | 1.3 GiB |

**Gelesene Bytes pro Token** — CSA-Layer lesen konstant `n_win + top-k` = 640
Zeilen (21 Layer × 640 × 2048 B = 27.5 MB, unabhängig vom Kontext), HCA-Layer
lesen `n_win +` *alle* komprimierten Einträge, weil HCA keine Sparse Attention hat
(bei 128k: 20 × 1152 × 2048 B = 47.2 MB; bei 1M: 20 × 7940 × 2048 B = 325.2 MB —
diese Zeile rechnet mit `ctx = 1 000 000` für runde Zwischenwerte, die
KV-Bilanz-Tabelle oben mit `ctx = 2^20 = 1 048 576`; der Unterschied ist <5 % und
ändert keine Schlussfolgerung):

| Kontext | f32 | native | turbo3 |
|---|---|---|---|
| 128k | 74.7 MB/Token | 21.3 MB | 7.3 MB |
| 1M | **353 MB/Token** | 100 MB | **34 MB** |

Bei ~45 GB/s DDR4-3200 sind 353 MB/Token rund **7.8 ms pro Token** allein fürs
KV-Lesen im 1M-Kontext — spürbar gegen den ~77-ms-Expertendeckel, aber keine
eigene Untragbarkeitsschwelle (eine frühere Fassung dieser Zahl war um Faktor
1000 zu hoch, siehe Bandbreitenabschnitt oben). Die HCA-Layer, nicht die
CSA-Layer, dominieren das Lesevolumen — und dagegen helfen Codec (Plan 03/04)
und Flash-Kernel (Plan 02/05) gemeinsam, auch wenn keiner davon allein an der
77-ms-Kante hängt.

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
  `matmul_fp8` aus `quant.h` ([c/deepseek_v4.c:11630](../c/deepseek_v4.c)).
- `coli_cuda_fp8_set_lut` muss vor jedem fmt=8-Upload aufgerufen sein.

**Nicht** wiederverwendbar: `coli_cuda_attention_absorb*` — auf GLMs Geometrie
zugeschnitten (separates `kv_b`, latent/rope getrennt, kein Top-k, keine Sinks).

**Vorbild für einen modellspezifischen Backend:** `backend_cuda_ink.{cu,h}` für
Inkling, bewusst winzige API, per `INK_CUDA_OBJ` ([c/Makefile:404](../c/Makefile))
und Objektregel ([:611](../c/Makefile)) verdrahtet. Dem folgt `backend_cuda_v4.{cu,h}`.

### FlashMLA — Vendor-Referenz, nicht linkbar

`reference/FlashMLA` (deepseek-ai/FlashMLA, Stand `15f13e5`, 2026-07-27) enthält
DeepSeeks eigene Sparse-MLA-Kernel — also genau die Attention, die
[Plan 05](05-cuda-attention.md) neu schreibt.

**Warum trotzdem nur Referenz:** Support-Matrix ist **SM90 (Hopper) und SM100
(Datacenter-Blackwell)**, CUDA ≥ 12.8, PyTorch-Extension. Die Kernel leben von
wgmma/TMA/Warp-Specialization; nicht einmal Consumer-Blackwell `sm120` wird
unterstützt, Ada `sm89` erst recht nicht. Auf der 4070 ist davon **keine Zeile
lauffähig oder linkbar**, und die Toolchain (Python/PyTorch/cutlass) widerspricht
dem Zero-Dependency-C-Ziel dieses Forks. Es wird nichts daraus portiert, ohne dass
ein Plan das ausdrücklich als eigenen Commit ausweist.

Was gelesen werden sollte:

- `docs/20250929-hopper-fp8-sparse-deep-dive.md` — das fp8-Sparse-Decoding im
  Detail, inklusive Dequantisierung in Registern. Direkt einschlägig für
  [Plan 05](05-cuda-attention.md) und [Plan 04](04-turboquant.md).
- `csrc/smxx/decode/combine/combine.cu` (230 Zeilen) — der Split-KV-Combine-Kernel,
  der die partiellen Online-Softmax-Zustände mergt. Liegt bewusst unter `smxx`,
  nicht `sm90`: kein Arch-spezifisches Inline-Assembler, also der einzige Teil des
  Repos, der als Algorithmus unverändert auf `sm89` übertragbar wäre.
- `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` und
  `csrc/sm90/prefill/sparse/phase1.cuh` — Aufteilung der KV-Zeilen über die SMs.
  Bei Batch 1 und einem Query-Token ist das die einzige Parallelitätsquelle
  jenseits der 64 Heads; der Head-Outer-Block aus Plan 05 füllt 46 SMs sonst nur
  knapp.
- `README.md`, Abschnitt *Sparse Attention* — Vendor-Konventionen, die die unseren
  bestätigen: ungültige Indizes als `-1` (Decode) bzw. `-1` oder `>= s_kv`
  (Prefill), Rückgabe `(out, max_logits, lse)`, also der exportierte
  Online-Softmax-Zustand aus [Plan 02](02-flash-attention.md).

**Geometrie-Unterschied, der nicht verwischt werden darf:** FlashMLA zielt auf
DSA aus DeepSeek-V3.2-Exp — `head_dim_k=576` = 512 NoPE + 64 RoPE, gepagter Cache,
fp8-KV-Zeile zu 656 Bytes (512 × fp8-e4m3 + 4 × f32-Scale je 128 Werte + 64 ×
bf16 RoPE). V4-Flash hat stattdessen den hybriden Cache aus Fensterring plus
komprimiertem Cache mit `head_dim=512` (siehe [Attention-Architektur](#attention-architektur)),
und unser Codec ist der aus [Plan 03](03-kv-codec.md). Strukturell verwandt,
nicht identisch — Referenz, keine Spezifikation.

### `packed_rows8` — kein Hindernis

Naheliegender Verdacht: `packed_rows8`
([c/deepseek_v4_internal.h:179](../c/deepseek_v4_internal.h)) beschreibe eine
feinere Scale-Granularität (8×128 statt 128×128), die `coli_cuda_matmul` nicht kann.
**Stimmt nicht.**

`v4_fp8_pack_rows8_inplace` ([c/deepseek_v4.c:496](../c/deepseek_v4.c)) ist eine
reine **Byte-Layout-Transposition innerhalb von 8-Zeilen-Kacheln**, nach dem Laden
in-place angewandt, ausschließlich unter `__AVX2__`, für den AVX2-Zielkernel. Die
Scales werden getrennt über `st_read_scale_f32` gelesen und behalten ihre
128×128-Granularität aus `add_fp8` ([:347](../c/deepseek_v4.c)). `block_rows = 8`
in der `ColiTensorView` beschreibt die *Gewichtskachelung*, nicht die
Scale-Auflösung.

Für GPU-Tensoren wird der Repack schlicht **übersprungen** — dann liegt row-major
fp8 mit 128×128-f32-Scales vor, exakt das fmt=8-Format. Kein Sonderkernel, kein
Präzisionsverlust. Gilt genauso für DSpark, dessen `v4_ds_pack_rows8`
([c/deepseek_v4_dspark.inc:130](../c/deepseek_v4_dspark.inc)) dieselbe
Transposition mit demselben Kommentar ist.

## Konventionen

**Env-Knöpfe.** `V4_*` für Nutzerknöpfe (`V4_MTP`, `V4_DRAFT`, `V4_NGRAM`),
`COLI_V4_*` für Engine-Interna (`COLI_V4_DIRECT`, `COLI_V4_AUTOPIN`).
Neue Knöpfe dieses Branches:

| Variable | Default | Phase | Stand |
|---|---|---|---|
| `V4_SCRATCH_MB` | 512 | 01 | gebaut |
| `V4_FLASH` | 1 | 02 | gebaut |
| `V4_KV` | `native` | 03 | gebaut |
| `V4_KV_INDEX` | `native` | 03 | gebaut |
| `V4_VRAM` | 0 | 05–08 | gebaut |
| `V4_VRAM_RESERVE_MB` | `free/8`, geklemmt 256…1024 | 06 (vorgezogen), 08 | gebaut |
| `V4_VRAM_LIMIT_MB` | aus (kappt das gemeldete freie VRAM) | 08 | geplant |
| `V4_VRAM_FAIL_AT` | aus, nur unter `COLI_V4_TEST_HOOKS` | 08 | geplant |
| `V4_OMP_CORES` | `perf`, wenn erkennbar; sonst `all` | 09 | geplant |
| `V4_PIN_SLOTS` | 16 | 12 | geplant |
| `V4_PIN_FRACTION` | aus (Alternative zu `V4_PIN_SLOTS`) | 12 | geplant |
| `V4_PIN_RAMP_REQUESTS` | 24 | 12 | geplant |

Die sechs `gebaut`-Knöpfe stehen in `docs/ENVIRONMENT.md`. Die geerbten
Upstream-Knöpfe (`V4_MTP*`, `V4_DRAFT`, `V4_NGRAM`, `V4_PREFIX_LOG`,
`COLI_V4_*`) stehen dort **nicht** — sie werden vom Code gelesen, sind aber nie
dokumentiert worden. Wer einen davon anfasst, dokumentiert ihn bei der
Gelegenheit nach.

**Diese Tabelle ist die Liste, nicht die Phasenpläne.** Wer einen Knopf einführt
und ihn hier vergisst, hat ihn nur halb eingeführt — die Pflegeregel in
[AGENTS.md](../AGENTS.md) hängt daran, und `docs/ENVIRONMENT.md` wird daraus
gefüllt.

**Default bleibt überall die heutige Semantik — mit einer Ausnahme.** Die README
garantiert, dass die Default-Policy Modellpräzision und Router-Semantik nicht
still ändert. `V4_KV`/`V4_KV_INDEX` sind bewusst die Ausnahme: `native` ist
bit-exakt zu `f32` (siehe [03-kv-codec.md](03-kv-codec.md)), also keine
Präzisionsänderung, nur eine Buchhaltungskorrektur. Alle anderen Knöpfe in dieser
Tabelle halten die Regel ohne Ausnahme.

**Test-Gates.** Eine Make-Regel `tests/test_x$(EXE):` in `c/Makefile` genügt —
`TEST_RULES` ([c/Makefile:359](../c/Makefile)) liest die Regeln per `sed` und macht
daraus automatisch Gates. Keine zentrale Liste, kein Merge-Konflikt.

**Semantik-Grenzen, die jeder Phasenplan respektiert:**

- Der **Indexer** entscheidet per Top-k, *welche* Tokens gesehen werden — das ist
  Router-Semantik. Quantisierung dort ist immer opt-in und getrennt schaltbar.
- Der `head_argmax`-Kommentar ([c/deepseek_v4.c:8097](../c/deepseek_v4.c)) hält
  ausdrücklich fest, dass Akkumulationsreihenfolge und Vokabularordnung erhalten
  bleiben, damit Logits und Tie-Breaking sich nicht ändern. Eine GPU-Reduktion
  ändert das — deshalb Toleranz und opt-in.

## Phasenübersicht

| # | Plan | Liefert | RAM | VRAM |
|---|---|---|---|---|
| 01 | [Messen und RAM-Budget](01-measure-and-ram-budget.md) | Ist-Zahlen, `--memory-gb`/`RAM_GB`, `V4_SCRATCH_MB` | +2.6 GiB Systemreserve +0.375 GiB Scratch, vor Tierwechsel | — |
| 02 | [Flash Attention](02-flash-attention.md) | Online-Softmax, `all_kv` weg, Source-Sync-Test | — | — |
| 03 | [KV-Codec](03-kv-codec.md) | **natives fp8+bf16/fp4, bit-exakt**, `context_bytes` folgt | **+1.25 GiB** | — |
| 04 | [TurboQuant](04-turboquant.md) | turbo2/3/4 als verlustbehafteter Tier | +0.27 GiB vor 08; +0.012 danach³ | −2.0 bei 1M |
| 05 | [CUDA-Attention](05-cuda-attention.md) | `backend_cuda_v4`, Flash-Kernel, Attention-KV-Spiegel | 0 (Host-Shadow) | −0.39 |
| 06 | [Dense in VRAM](06-dense-vram.md) | fp8-Residenz + Matmuls auf GPU | +5.456 GiB | −5.456 |
| 07 | [Head und DSpark in VRAM](07-head-dspark-vram.md) | Head-Matvec + Drafter auf GPU | +2.16 GiB | −1.31² |
| 08 | [VRAM-Planner](08-vram-planner.md) | Stufenplanung, exklusive KV-Eigentümerschaft, 4070-Profil | +0.39 GiB¹ | — |
| 09 | [Arch / CachyOS](09-arch-cachyos.md) | `omp_tune.h`, THP, CUDA-Pfade | — | — |
| 10 | [Dual-Streaming](10-dual-streaming.md) | Mirror-Maschinerie nach V4, gewichtete Stripes | — | — |
| 11 | [Rückbau auf V4](11-strip-to-v4.md) | andere Motoren + Windows raus | — | — |
| 12 | [Expert-Cache-Politik](12-expert-cache-policy.md) | Pin-Deckel, Indexer (Scan + Select), Prefill | — | — |
| 13 | [Frontend V4-only](13-frontend-v4.md) | WebUI, CLI, Serve auf V4 | — | — |
| 14 | [Chunked Prefill + DSpark](14-chunked-prefill-dspark.md) | 256k-Aktivierungsfenster und MTP-Handoff | spart bis zu `CTX/chunk`-fachen State | — |

¹ 03 hat den gesamten KV bereits von 1.68 auf 0.43 GiB (128k) gesenkt. 05 spiegelt
den 0.388-GiB-Attention-Anteil, 08 entfernt erst dessen Host-Shadow; 0.044 GiB
nativer Indexer-KV bleiben im RAM.
² 0.99 GiB Head + **gemessene 0.32 GiB** DSpark-Backbone gehen tatsächlich auf die
GPU (siehe VRAM-Budget oben); die restlichen ~0.85 GiB der 1.17-GiB-DSpark-RAM-Reserve
waren Marge für Head/Scratch, kein eigener VRAM-Posten.
³ Vor 08 komprimiert 04 auch den Host-Shadow. Danach bleibt als RAM-Gewinn nur
der Indexer (0.044 → 0.032 GiB bei 128k); Attention-KV zahlt im VRAM ein. Bei 1M
passt `native` (3.08 GiB) nach der Phase-06-Inventur nominell in das ~3.39-GiB-
Budget, aber mit nur ~0.31 GiB Marge; turbo3 (1.06 GiB) bleibt dort empfohlen.

Referenzdokumente ohne Nummer: [paper-deepseek-v4.md](paper-deepseek-v4.md) (das
Paper), [llamacpp-deepseek-v4.md](llamacpp-deepseek-v4.md) (die
llama.cpp-Implementierung und ihre Optimierungen) und
[deepspec-dspark.md](deepspec-dspark.md) (DeepSeeks eigene DSpark-Referenz).

Arbeitsregeln, Branch-/PR-Schema und die Pflege dieser Pläne stehen in
[AGENTS.md](../AGENTS.md).

**Reihenfolge.** 09 trägt die Nummer der Vollständigkeit halber, aber sein erster
Commit gehört **direkt hinter 01**: `deepseek_v4.c` ruft `coli_omp_tune_threads`
nicht auf (anders als `colibri.c`, `kimi_k3.c`, `olmoe.c`; `inkling.c` fehlt es
ebenfalls, aber das ist nicht Teil dieses Branches) und läuft damit auf allen
logischen Kernen statt auf den physischen. Der Header misst dafür +2.3× (Zen3,
#718). Drei Zeilen, unabhängig vom Rest.

**Wenn die Zeit knapp wird:** `01 → 09.1 → 03 → 06` ist die Reihenfolge mit dem
meisten Ertrag pro Aufwand. 02 ist Voraussetzung für 05, und der Source-Sync-Test
aus 02 sollte in jedem Fall früh kommen.

**03 ist nach dem Paper aufgewertet.** Was dort passiert, ist keine Quantisierung,
sondern das Ende einer Fehlbuchung: 3.5× auf den Haupt-KV und 7.5× auf den Indexer,
**bit-exakt**. Damit steht es in einer Reihe mit 01 und 09.1 — messbarer Gewinn
ohne Semantikfrage. Die 7.5× auf den Indexer wiegen dabei schwerer als die Tabelle
vermuten lässt: sie treffen den Scan aus dem Indexer-Abschnitt oben, also 352 MB
bzw. 2.69 GB pro Token. 04 ist danach optional — außer beim 1M-Profil, wo das
VRAM-Budget ohne turbo3 nicht schließt.

**10 ist unabhängig von allem anderen** und adressiert die Kostenstelle, die nach
den VRAM-Phasen übrigbleibt: selbst beim 32k-Endprofil mit ~17 %
Expert-Residenz liest jeder Token noch ~2.8 GB von der Platte. Es lässt sich
jederzeit einschieben. Sein Commit 3 (HMB, Koaleszenz, Queue-Tiefe) trägt auch
ohne zweites Laufwerk; nur Commit 2 braucht es. **Der Einzellaufwerk-Betrieb
bleibt in 10 Default und Pflicht-Abnahme** — die Zahlen der Baseline in 01 werden
weiterhin ohne Spiegel gemessen, sonst ist der Vergleich über die Phasen hinweg
kaputt.

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

Auf der Zielmaschine ist die gitignorierte `.venv-v4-tiny` mit den dort exakt
gepinnten Versionen eingerichtet. Sie lässt sich ohne CUDA und ohne vollständigen
Checkpoint reproduzieren:

```bash
uv venv .venv-v4-tiny --python /usr/bin/python3
uv pip install --python .venv-v4-tiny/bin/python \
  -r c/tools/requirements-deepseek-v4-tiny.txt
make -C c PYTHON="$PWD/.venv-v4-tiny/bin/python" deepseek-v4-tiny-check
```

Der Generator baut ein deterministisches Drei-Layer-Modell von unter 1 MiB mit
der offiziellen `DeepseekV4ForCausalLM`-Implementierung. Das deckt das lokale
Oracle ab, nicht Durchsatz oder Qualität des vollen Checkpoints.

Gegen den lokalen Checkpoint greift zusätzlich:

```bash
make -C c deepseek-v4-oracle \
  MODEL=~/Services/models/colibri/deepseek-v4-flash MEMORY_GB=28
```

## Was dieser Branch nicht macht

- **Geroutete Experten bleiben auf der CPU.** Streaming und Tiering der ~147 GB ist
  ein eigenes Projekt; dieser Branch macht nur mehr RAM dafür frei.
- **Andere Engines bleiben unangetastet — bis auf eine benannte Ausnahme.**
  `inkling.c`, `kimi_k3.c`, `olmoe.c` und der gemeinsame `backend_cuda.cu` werden
  nicht angefasst. Der tote Kommentar zu KV8/TQ in
  [c/colibri.c:3567](../c/colibri.c) bleibt stehen.
  **Ausnahme:** [10-dual-streaming.md](10-dual-streaming.md) Commit 1 darf die
  Mirror-Maschinerie aus `colibri.c` in einen gemeinsamen `c/mirror.h` ziehen und
  `colibri.c` auf den Header umstellen. Das ist ein Refactor ohne
  Verhaltensänderung, kein GLM-Feature — und Plan 10 nennt die V4-lokale Kopie als
  Rückfall, falls der Schnitt teurer wird als gedacht. Wer diese Ausnahme zieht,
  belegt sie mit `make -C c colibri` plus den GLM-Gates, nicht nur mit den
  V4-Gates.
- **Kein On-Disk-Format ändert sich.** `kv_persist.h` (COLIKV1) gehört zu
  `colibri.c`; `kv_prefix.h` speichert nur Token-IDs und ist von Codec-Änderungen
  unberührt.
