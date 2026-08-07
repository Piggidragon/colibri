# DeepSeek-V4 — Paper-Auszug für die Implementierung

Quelle: *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*,
DeepSeek-AI, arXiv:2606.19348v1 [cs.CL], 26. Apr 2026. Original als `paper.pdf`
im Repo-Wurzelverzeichnis (nicht committen).

Konvertiert mit `pdftotext` aus dem PDF. Formeln überleben die Extraktion nicht
sauber und sind hier in Worten beschrieben — für die exakten Gleichungen ins PDF
schauen. Die Abschnitte 5 (Post-Training, Benchmarks) und 6 sind für dieses
Vorhaben irrelevant und nur zusammengefasst.

Rohextraktionen liegen unter `/tmp/paper_flow.txt` (Lesereihenfolge) und
`/tmp/paper_layout.txt` (Layout, besser für Tabellen).

---

## Abstract, verkürzt

Zwei MoE-Modelle: **DeepSeek-V4-Pro** mit 1.6T Parametern (49B aktiviert) und
**DeepSeek-V4-Flash** mit 284B (13B aktiviert), beide mit 1M-Token-Kontext.

Drei Neuerungen:

1. Hybride Attention aus **Compressed Sparse Attention (CSA)** und **Heavily
   Compressed Attention (HCA)**
2. **Manifold-Constrained Hyper-Connections (mHC)** statt gewöhnlicher Residuals
3. **Muon-Optimizer** fürs Training

Bei 1M Kontext braucht V4-Pro 27 % der Single-Token-Inferenz-FLOPs und **10 % des
KV-Caches** gegenüber DeepSeek-V3.2.

---

## 2.2 Manifold-Constrained Hyper-Connections (mHC)

Erklärt die `hc_*`-Tensoren im Checkpoint (`hc_attn_fn`, `hc_attn_base`,
`hc_attn_scale`, `hc_ffn_*`) und `config.hc_mult` / `config.hc_sinkhorn_iters`.

**Standard-Hyper-Connections** verbreitern den Residual-Stream um Faktor `n_hc`:
statt `R^d` wird der Zustand `R^{n_hc × d}`. Drei lineare Abbildungen pro Layer —
Input-Mapping `A_l ∈ R^{1×n_hc}`, Residual-Transformation `B_l ∈ R^{n_hc×n_hc}`,
Output-Mapping `C_l ∈ R^{n_hc×1}`:

```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)
```

Der eigentliche Layer-Input `A_l X_l` bleibt `d`-dimensional, die innere
Layer-Architektur ändert sich also nicht.

**Die mHC-Neuerung:** `B_l` wird auf die Mannigfaltigkeit der **doppelt
stochastischen Matrizen** (Birkhoff-Polytop) beschränkt — Zeilen- und
Spaltensummen 1, alle Einträge ≥ 0. Damit ist die Spektralnorm `‖B_l‖₂ ≤ 1`, die
Residual-Transformation also nicht-expansiv, was Vorwärts- und Rückwärtspass
numerisch stabilisiert. Die Menge ist multiplikativ abgeschlossen, was tiefe
Stapel stabil hält. `A_l` und `C_l` werden über Sigmoid nicht-negativ und
beschränkt gehalten, um Signalauslöschung zu vermeiden.

Die Projektion auf die Mannigfaltigkeit läuft über **Sinkhorn-Knopp**, im
Checkpoint als `hc_sinkhorn_iters` — im Paper `t_max = 20`.

Parameter werden dynamisch erzeugt: pro Layer aus dem RMS-normalisierten,
geflatteten Eingangszustand, aufgeteilt in eine eingabeabhängige und eine
statische Komponente.

> Bezug zum Code: `coli_v4_hc_pre` ([c/deepseek_v4.c:1159](../c/deepseek_v4.c)),
> aufgerufen aus der Runtime ab [:3081](../c/deepseek_v4.c) ff.

---

## 2.3 Hybride Attention — CSA und HCA

**Das Herzstück für dieses Vorhaben.**

### 2.3.1 Compressed Sparse Attention (CSA)

Zwei Stufen: erst Kompression von je `m` Tokens zu einem KV-Eintrag, dann
DeepSeek Sparse Attention (DSA) — jeder Query-Token beachtet nur `k` der
komprimierten Einträge.

**Komprimierte KV-Einträge.** Aus den Hidden States `H ∈ R^{n×d}` werden **zwei**
KV-Serien `C^a, C^b ∈ R^{n×c}` und zwei zugehörige Kompressionsgewichte
`Z^a, Z^b ∈ R^{n×c}` berechnet (vier trainierbare Matrizen `W^{aKV}, W^{bKV},
W^{aZ}, W^{bZ} ∈ R^{d×c}`).

Je `m` Einträge werden anhand der Gewichte plus gelernter Positionsbiases
`B^a, B^b ∈ R^{m×c}` zu einem Eintrag verdichtet. Der Softmax normalisiert über
**2m** Elemente aus `Z^a` und `Z^b` gemeinsam; das Ergebnis ist eine gewichtete
Summe (Hadamard-Produkt) über beide Serien.

**Überlappung:** Jeder komprimierte Eintrag entsteht aus `2m` KV-Einträgen, aber
die von `C^b` für Eintrag `i` benutzten Indizes überlappen mit denen von `C^a` für
Eintrag `i−1`. Netto komprimiert CSA die Sequenzlänge trotzdem auf `1/m`.

> Bezug zum Code: `coff = ratio == 4 ? 2 : 1` im Layer-Plan
> ([c/deepseek_v4.c:394](../c/deepseek_v4.c)) — die `2` ist genau diese
> Doppelserie. `attn.compressor.ape` hat Shape `[ratio, coff·head_dim]`.

**Lightning Indexer.** Dieselbe Kompressionsoperation erzeugt komprimierte
Indexer-Keys `K^{IComp} ∈ R^{(n/m) × c^I}`. Indexer-Queries entstehen niederrangig
aus demselben latenten Query-Vektor `c^Q_t`, den auch die Hauptattention nutzt.
Der Indexscore zwischen Query `t` und komprimiertem Block `s` ist eine gewichtete
Summe über Indexer-Heads:

```
I_{t,s} = Σ_h  w^I_{t,h} · ReLU( q^I_{t,h} · K^{IComp}_s )
```

Ein Top-k-Selektor behält daraus die `k` besten komprimierten Einträge.

> Der **ReLU** und die gelernten Head-Gewichte `w^I` finden sich 1:1 in
> `coli_v4_indexer_step` ([c/deepseek_v4.c:2900](../c/deepseek_v4.c)):
> `score += fmaxf(dot, 0.0f) * head_weights[head]`.

**Shared-KV MQA.** Die ausgewählten komprimierten Einträge dienen **gleichzeitig
als Key und als Value**. Das ist der Grund, warum in colibri ein einziger
512-dim-Vektor pro Position für alle 64 Heads reicht.

**Grouped Output Projection.** Weil `c·n_h` groß ist, werden die `n_h`
Head-Ausgaben in `g` Gruppen geteilt, jede Gruppe auf `d_g` Dimensionen
projiziert, und erst deren Konkatenation auf `d`. Entspricht `o_groups` und
`o_lora_rank`.

### 2.3.2 Heavily Compressed Attention (HCA)

Wie CSA, aber:

- deutlich größere Kompressionsrate `m' ≫ m`
- **keine** Doppelserie, **keine** Überlappung — nur `C = H·W^{KV}` und `Z = H·W^Z`
- **keine** Sparse Attention, also **kein** Indexer: jeder Query sieht *alle*
  komprimierten Einträge

Shared-KV MQA und Grouped Output Projection wie bei CSA.

### 2.3.3 Weitere Details

**Query- und KV-Normalisierung.** Zusätzliches RMSNorm auf jedem Query-Head und
auf dem einen KV-Head, direkt vor der Core-Attention. Verhindert explodierende
Attention-Logits.

**Partielles RoPE.** RoPE wird auf die **letzten 64 Dimensionen** jedes Query- und
KV-Vektors angewandt. Weil die KV-Einträge zugleich Values sind, tragen die
Core-Attention-Ausgaben sonst *absolute* Positionsinformation. Gegenmaßnahme:
RoPE mit Position **−i** auf die letzten 64 Dimensionen jeder Ausgabe `o_{t,i}`.
Dadurch wird die Positionsinformation wieder relativ.

> Genau die beiden `coli_v4_rope_apply`-Aufrufe in colibri: vorwärts auf `kv`
> ([:1652](../c/deepseek_v4.c)), invers auf `head_output`
> ([:1720](../c/deepseek_v4.c)).

**Sliding-Window-Zweig.** Weil ein Query aus Kausalitätsgründen nur *vorangehende*
komprimierte Blöcke sehen darf, fehlt ihm der eigene Block. Deshalb zusätzlich
`n_win` **unkomprimierte** KV-Einträge der jüngsten `n_win` Tokens, die zusammen
mit den komprimierten in die Core-Attention gehen.

**Attention Sink.** Gelernte Sink-Logits `z'_h` pro Head; `exp(z'_h)` wird dem
Softmax-Nenner hinzugefügt:

```
s_{h,i,j} = exp(z_{h,i,j}) / ( Σ_k exp(z_{h,i,k}) + exp(z'_h) )
```

Damit kann die Summe der Attention-Scores eines Heads kleiner als 1 werden, bis
nahe 0. Entspricht `attn.attn_sink` und dem `denominator = expf(sinks[head] − maximum)`
in `coli_v4_sparse_attention_ref`.

### 2.3.4 Effizienz — **der wichtigste Abschnitt für dieses Vorhaben**

Wörtlich sinngemäß:

> Wir verwenden ein **gemischtes Speicherformat** für KV-Einträge: **BF16 für die
> RoPE-Dimensionen, FP8 für die übrigen**. Das halbiert den KV-Cache gegenüber
> reinem BF16 nahezu.

> Die Attention-Berechnung im Lightning Indexer läuft in **FP4**.

> Gegenüber DeepSeek-V3.2 wird ein **kleineres Attention-Top-k** gewählt.

Und als Vergleichsmaßstab:

> Gegenüber BF16-GQA8 mit Head-Dimension 128 — einer üblichen LLM-Konfiguration —
> sinkt der KV-Cache von DeepSeek-V4 im 1M-Kontext auf etwa **2 %**.

**Konsequenz für colibri:** Das Zielformat des Modells ist fp8 (nope) + bf16
(RoPE). colibri rechnet diese Quantisierung bereits — `coli_fp8_activation_qdq_ref`
auf die ersten 448 Dimensionen und `coli_bf16_round_array` auf die letzten 64
([c/deepseek_v4.c:1655](../c/deepseek_v4.c)) — **speichert das Ergebnis aber als
f32**. Siehe [03-kv-codec.md](03-kv-codec.md).

---

## 3.5 Inference-Framework

### 3.5.1 KV-Cache-Struktur

Der Cache zerfällt in **zwei** Teile (Figure 6):

- **State Cache** — pro Request ein Block fester Größe. Enthält den
  SWA-Abschnitt (die letzten `n_win` Tokens) und die noch nicht
  komprimierbaren Rest-Tokens der CSA/HCA-Zweige.
- **Klassischer KV-Cache** — mehrere Blöcke pro Request. Jeder Block deckt
  `lcm(m, m')` Original-Tokens ab und erzeugt `k1 = lcm(m,m')/m` CSA- und
  `k2 = lcm(m,m')/m'` HCA-komprimierte Tokens.

Bei `m=4`, `m'=128` ist `lcm = 128`, also `k1 = 32` CSA- und `k2 = 1`
HCA-Eintrag pro Block.

**Warum PagedAttention nicht direkt passt:** unterschiedliche Cache-Policies (SWA
hat eigene Eviction) und Alignment-Anforderungen der Attention-Kernel. Lösung:
SWA plus unkomprimierte Rest-Tokens werden wie ein **State-Space-Modell**
behandelt — ein sequenzspezifischer Zustand, der nur von der aktuellen Position
abhängt, aus einem vorab allozierten Pool fester Größe.

### 3.5.2 KV-Cache auf Platte

Für Shared-Prefix-Requests. Komprimierte CSA/HCA-Einträge werden vollständig
gespeichert; beim Treffer werden sie bis zum letzten vollständigen
Kompressionsblock gelesen, der unvollständige Rest wird neu berechnet.

SWA-Einträge sind **etwa 8× voluminöser** als die komprimierten (unkomprimiert
und in *jeder* Schicht vorhanden). Drei Strategien:

- **Full SWA Caching** — alles speichern, keine Redundanz, aber
  schreiblastig und für SSDs ungünstig.
- **Periodic Checkpointing** — alle `p` Tokens die letzten `n_win` sichern,
  Rest neu rechnen. Über `p` einstellbarer Kompromiss.
- **Zero SWA Caching** — nichts speichern. Trick: der SWA-Eintrag eines Tokens
  hängt nur von den `n_win` jüngsten Einträgen der *vorherigen* Schicht ab.
  Mit gecachten CSA/HCA-Einträgen genügt es also, die letzten `n_win · L` Tokens
  neu zu rechnen, um die letzten `n_win` SWA-Einträge eines `L`-Layer-Modells
  wiederherzustellen.

> Für diesen Branch nicht relevant (colibri hat mit `kv_prefix.h` einen anderen
> Ansatz), aber die Zero-SWA-Rechnung ist ein hübsches Argument dafür, dass der
> SWA-Ring beliebig verworfen werden darf.

---

## 4.2.1 Modellkonfiguration

### DeepSeek-V4-Flash — die Zielkonfiguration

| Parameter | Wert |
|---|---|
| Transformer-Layer | **43** |
| Hidden-Dimension `d` | **4096** |
| **Layer 0–1** | **reine Sliding-Window-Attention** |
| **Layer 2–42** | **CSA und HCA im Wechsel** |
| CSA-Kompressionsrate `m` | **4** |
| HCA-Kompressionsrate `m'` | **128** |
| Indexer-Query-Heads `n^I_h` | 64 |
| Indexer-Head-Dimension `c^I` | **128** |
| **Attention-Top-k** | **512** |
| Query-Heads `n_h` | 64 |
| Head-Dimension `c` | **512** |
| Query-Kompressionsdimension `d_c` | 1024 |
| Output-Projektionsgruppen `g` | 8 |
| Zwischenausgabedimension `d_g` | 1024 |
| **Sliding-Window-Größe `n_win`** | **128** |
| MoE | alle Blöcke; Hash-Routing in den ersten **3** |
| Experten | 1 shared + **256** routed, intermediate **2048** |
| Aktivierte Experten | **6** pro Token |
| MTP-Tiefe | 1 |
| mHC `n_hc` / Sinkhorn `t_max` | **4** / **20** |
| Parameter gesamt / aktiviert | **284B** / **13B** |

### DeepSeek-V4-Pro (zum Vergleich, nicht Ziel dieses Branches)

61 Layer, `d`=7168, erste zwei Layer HCA, `n_h`=128, `d_c`=1536, `g`=16,
Top-k **1024**, 384 routed Experts mit intermediate 3072, sonst gleich.
1.6T gesamt / 49B aktiviert.

### Ableitung der Layer-Verteilung

43 Layer, davon 0 und 1 reines SWA. Die restlichen 41 wechseln ab CSA/HCA,
beginnend mit CSA bei Layer 2 — konsistent mit
[c/tests/test_deepseek_v4.c:414](../c/tests/test_deepseek_v4.c)
(`compress_ratios[2] = 4; compress_ratios[3] = 128;`):

| Typ | Layer | Anzahl |
|---|---|---|
| SWA (ratio 0) | 0, 1 | **2** |
| CSA (ratio 4) | 2, 4, 6, … 42 | **21** |
| HCA (ratio 128) | 3, 5, 7, … 41 | **20** |

**Vor dem Bau am echten Checkpoint gegenprüfen** — die Reihenfolge ab Layer 2 ist
aus dem Test abgeleitet, nicht aus dem Paper belegt.

---

## Abgeleitete KV-Bilanz

Aus `head_dim=512`, `index_head_dim=128`, `n_win=128`, Top-k=512 und der
Layer-Verteilung oben.

### Speicher pro Token (wachsender Anteil)

| Layer-Typ | Zeilen/Token | f32 | Paper-Format | turbo3 |
|---|---|---|---|---|
| CSA main (21×) | 1/4 × 512 dims | 512 B | 145.8 B | 50 B |
| CSA indexer (21×) | 1/4 × 128 dims | 128 B | 17 B | 12.5 B |
| HCA (20×) | 1/128 × 512 dims | 16 B | 4.6 B | 1.6 B |
| **Summe** | | **13.4 KiB** | **3.4 KiB** | **1.3 KiB** |

Zeilengrößen: f32 2048 B; Paper-Format 448 fp8 + 7 E8M0-Scales + 64 bf16 = **583 B**;
turbo3 4×50 = **200 B**. Indexer: f32 512 B; fp4 ≈ 68 B; turbo3 = 50 B.

| Kontext | f32 (heute) | Paper-Format | turbo3 |
|---|---|---|---|
| 32k | 0.42 GiB | 0.11 GiB | 0.04 GiB |
| 128k | **1.68 GiB** | **0.43 GiB** | **0.16 GiB** |
| 1M | **13.4 GiB** | 3.4 GiB | 1.3 GiB |

Der **Sliding-Window-Ring ist konstant und winzig**: 43 × 128 × 512 × 4 B =
**10.8 MiB** bei f32. Er ist kein Speicherproblem — wohl aber ein Bandbreitenproblem,
weil er in jedem Layer und jedem Token vollständig gelesen wird.

### Gelesene Bytes pro Token (Bandbreite)

Der Attention-Kernel liest pro Layer `topk` Zeilen:

- **CSA-Layer:** `n_win + top-k` = 128 + 512 = **640 Zeilen**, konstant
  (21 Layer × 640 × 2048 B = 27.5 MB, unabhängig vom Kontext)
- **HCA-Layer:** `n_win +` *alle* komprimierten Einträge = 128 + ctx/128,
  weil HCA keine Sparse Attention hat — **wächst linear mit dem Kontext**
  (bei 128k: 20 × 1152 × 2048 B = 47.2 MB; bei 1M: 20 × 7940 × 2048 B = 325.2 MB)

| Kontext | f32 | Paper-Format | turbo3 |
|---|---|---|---|
| 128k | 74.7 MB/Token | 21.3 MB | 7.3 MB |
| 1M | **353 MB/Token** | 100 MB | **34 MB** |

(Die 128k-Zeile war in einer früheren Fassung mit falschen Zeilenzahlen
berechnet; hier mit der Formel oben korrigiert.)

Bei ~45 GB/s DDR4-3200 sind 353 MB/Token rund **7.8 ms pro Token** allein fürs
KV-Lesen im 1M-Kontext — spürbar, aber unter dem ~77-ms/Token-Bandbreitendeckel
der gerouteten Experten (siehe [00-reference.md](00-reference.md)). Mit turbo3
sind es 0.76 ms, auf dem 4070 bei ~500 GB/s rund 0.07 ms. (Eine frühere Fassung
wies diese Zeiten in Sekunden statt Millisekunden aus — Faktor 1000 zu hoch.)

Das reine KV-Lesen ist damit **kein** eigenständiges Argument für Phase 04/05;
das eigentliche Gewicht liegt beim VRAM-Budget (siehe
[00-reference.md](00-reference.md)) und bei der O(Kontext)-Staging-Kopie, die
[02-flash-attention.md](02-flash-attention.md) beschreibt.

---

## 5–6 Post-Training und Fazit (zusammengefasst)

Nicht implementierungsrelevant.

- **5.1** Post-Training wie DeepSeek-V3.2, aber die gemischte RL-Stufe komplett
  durch **On-Policy Distillation** ersetzt. Domänenspezialisten via GRPO,
  danach Destillation in das Endmodell. Drei Reasoning-Modi (Table 2), „Think
  Max" per System-Prompt-Instruktion (Table 3). Tool-Call-Schema in Table 4,
  Quick-Instruction-Spezialtokens in Table 5.
- **5.3** Benchmarks. Base-Modell-Vergleich in Table 1: V4-Flash-Base schlägt
  V3.2-Base bei 13B statt 37B aktivierten Parametern in den meisten Kategorien
  (MMLU-Pro 68.3 vs 65.5, HumanEval 69.5 vs 62.8, LongBench-V2 44.7 vs 40.2),
  liegt bei BigCodeBench und MATH zurück.
- **5.4** Reale Aufgaben, MRCR-Langkontext (Figure 9), Kosten/Leistung nach
  Reasoning-Aufwand (Figure 10), Schreibqualität (Tables 11/13).
- **6** Fazit, Grenzen, Ausblick.

---

## Was daraus für die Pläne folgt

1. **Das Modell will fp8+bf16 im KV-Cache, colibri speichert f32.** Die
   Quantisierung wird bereits gerechnet, nur das Ergebnis in einem 4× zu großen
   Container abgelegt. → [03-kv-codec.md](03-kv-codec.md)
2. **Der Indexer ist als FP4 vorgesehen**, nicht als f32. Meine frühere
   Empfehlung „Indexer unangetastet lassen" war zu vorsichtig. → [03](03-kv-codec.md)
3. **`n_win` = 128, nicht ~2048.** Der Fensterring ist speichermäßig
   bedeutungslos (10.8 MiB), bandbreitenmäßig aber in jedem Token präsent.
4. **HCA-Layer lesen den kompletten komprimierten Cache.** Sie, nicht die
   CSA-Layer, dominieren die Bandbreite bei langem Kontext. → [02](02-flash-attention.md), [05](05-cuda-attention.md)
5. **Top-k = 512, `n_win` = 128** → CSA-Kernel läuft über exakt 640 Zeilen.
   Konkrete Zahl fürs Kernel-Design. → [05](05-cuda-attention.md)
6. **`lcm(m, m') = 128`** ist die natürliche Blockgröße, falls der Cache je
   blockweise verwaltet wird.
