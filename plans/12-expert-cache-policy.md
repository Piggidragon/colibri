# 12 — Expert-Cache-Politik, Prefill und Indexer-Select

Voraussetzung: [00-reference.md](00-reference.md), [01-measure-and-ram-budget.md](01-measure-and-ram-budget.md)

*Commits:*
1. `perf: lift the compile-time pin ceiling and make it tunable`
2. `perf: partial select for the lightning indexer top-k`
3. `perf: measure and tune the prefill expert batching`

## Warum dieser Plan überhaupt existiert

**Das war die größte Lücke im Planset.** Die Pläne 01–11 machen alle dasselbe:
mehr RAM für den Expert-Cache freimachen und die Lesevorgänge beschleunigen.
Keiner kümmert sich darum, **was im Cache liegt**.

Bei ~20 % Residenz ist das der falsche Schwerpunkt. Ein Treffer kostet null
Bytes von der Platte; ein Fehltreffer kostet ~13.4 MB (ein voller
`expert_record_bytes`-Eintrag, w1/w2/w3 + Scales — siehe
[00-reference.md](00-reference.md)), egal wie schnell das Laufwerk ist. **Die
Trefferquote zu erhöhen schlägt jede Beschleunigung der
Fehltreffer** — und Colibrì hat dafür bereits eine vollständige Maschinerie, die
niemand für dieses Setup eingestellt hat.

## Commit 1 — Die Pin-Obergrenze

`V4HotPolicy` ([c/deepseek_v4.c:5446](../c/deepseek_v4.c)) pinnt Experten nach
gemessener Nutzungshäufigkeit, mit persistierter Historie in
`<model_dir>/.coli_usage`:

| Knopf | Wo | Heute |
|---|---|---|
| `COLI_V4_AUTOPIN` | [:6030](../c/deepseek_v4.c) | an, sofern nicht `0` |
| `COLI_V4_SAVE_USAGE` | [:5546](../c/deepseek_v4.c) | schreibt die Historie zurück |
| `COLI_V4_PREWARM` | [:6048](../c/deepseek_v4.c) | wärmt den Cache aus der Historie |
| `history_seeded` | [:6035](../c/deepseek_v4.c) | erst ab **5000** beobachteten Requests |

**Der Deckel ist eine Compile-Zeit-Konstante** ([:5994](../c/deepseek_v4.c)):

```c
#ifndef COLI_V4_MAX_PIN_SLOTS_PER_LAYER
#define COLI_V4_MAX_PIN_SLOTS_PER_LAYER 4
#endif
    if (maximum_pins > COLI_V4_MAX_PIN_SLOTS_PER_LAYER)
        maximum_pins = COLI_V4_MAX_PIN_SLOTS_PER_LAYER;
```

`Makefile.deepseek-v4:34` hebt ihn auf **16**. Nach den VRAM-Phasen fasst der
Cache aber ~52 Slots pro Layer (29 GiB / 559 MB). **Damit sind höchstens 31 % des
Caches historiengesteuert gepinnt**, der Rest läuft adaptiv über LRU.

Ob 16 richtig ist, weiß niemand — der Wert stammt aus einer Zeit, in der der Cache
kleiner war. Für ein Setup mit stabiler Nutzung (immer dieselbe Person, ähnliche
Prompts) sollte eine höhere Quote gepinnter Experten besser sein; für wechselnde
Lasten ist LRU adaptiver.

**Änderung:** Aus der Compile-Konstante einen Laufzeitknopf machen, Default
unverändert 16, plus die Möglichkeit, ihn relativ zur Slotzahl auszudrücken:

```
V4_PIN_SLOTS=<n>        absolute Obergrenze pro Layer   (Default 16)
V4_PIN_FRACTION=<0..1>  alternativ: Anteil der verfügbaren Slots
```

Clamp wie üblich (Muster `coli_v4_dspark_cache_gb`, [:6288](../c/deepseek_v4.c)).
Dazu `COLI_V4_PIN_RAMP_REQUESTS` (heute 24, `Makefile.deepseek-v4:35`) ebenfalls
zur Laufzeit setzbar.

**Dann messen**, nicht raten: Trefferquote über einen festen Promptsatz bei
`V4_PIN_SLOTS` ∈ {4, 16, 32, 52}. Der Gewinner wird Default für dieses Profil.

Die `.coli_usage`-Historie braucht ≥5000 Requests, um zu greifen. Für ein
persönliches Setup heißt das: **`COLI_V4_SAVE_USAGE=1` von Anfang an setzen** und
die Historie über Wochen wachsen lassen. Gehört ins Tuning-Doc als
Erstinbetriebnahme-Hinweis — sonst wundert man sich, warum Pinning „nichts tut".

## Commit 2 — Der Indexer-Top-k

`coli_v4_indexer_step` ([c/deepseek_v4.c:2827](../c/deepseek_v4.c)) sortiert **alle**
Kandidaten, um die besten `index_topk = 512` zu finden:

```c
IndexScore *scores = malloc((size_t)state->count * sizeof(*scores));
...
qsort(scores, state->count, sizeof(*scores), descending_score);
```

Bei 1M Kontext ist `state->count` = 250 000. Ein voller Sort ist O(n log n) ≈ 4.5M
Vergleiche, wo ein partieller Select O(n) ≈ 250k reicht — pro Token und pro
CSA-Layer, also **21×** pro Token.

**Änderung:** Ein Min-Heap der Größe `topk`, oder ein Quickselect-Durchlauf.
Wichtig ist die **Ordnung des Ergebnisses**: `descending_score` bricht
Gleichstände über den Index (`return a->index - b->index`), und die Auswahl geht
danach in `compressed_indices`. Der Ersatz muss dieselbe Menge und dieselbe
Reihenfolge liefern — sonst ändert sich, welche Tokens die Attention sieht, und
das ist Router-Semantik, nicht Präzision.

Deshalb: der Test vergleicht die **komplette Indexliste** gegen den bisherigen
`qsort`-Pfad, auf Zufallsdaten inklusive vieler Gleichstände. Nicht nur die
Mengengleichheit — die Reihenfolge.

llama.cpp hat für denselben Zweck einen WMMA-CUDA-Kernel
(`ggml/src/ggml-cuda/lightning-indexer.cu`, siehe
[llamacpp-deepseek-v4.md](llamacpp-deepseek-v4.md)). Der Score-Teil gehört
langfristig dorthin; dieser Commit macht nur den CPU-Select richtig, unabhängig
von CUDA.

## Commit 3 — Prefill

**Zweite Lücke im Planset:** alle Pläne optimieren Decode. Bei 128k Kontext ist
aber der Prefill der längere Teil, und er hat ein anderes Kostenprofil — jeder
Token berührt 6 Experten, ein Batch von `B` Tokens berührt bis zu `6B`
verschiedene, aber jeder gelesene Expert bedient potenziell mehrere Tokens.

Vorhanden ist der Batch-Pfad: `coli_v4_block_window_batch_ref`
([:3842](../c/deepseek_v4.c)) und `coli_v4_attention_window_batch_ref`
([:2180](../c/deepseek_v4.c)), aufgerufen aus [:7180](../c/deepseek_v4.c).

Zu messen:

- Wie skaliert die Zeit pro Token mit der Batchgröße? Der Sweet Spot hängt am
  Verhältnis Expert-Wiederverwendung zu Arbeitsspeicherbedarf.
- Deckt sich das mit `COLI_V4_EXPERT_PREFETCH` ([:3251](../c/deepseek_v4.c)) und
  der Worker-Zahl des Lookup-Pipelines?
- Wie verhält sich die Trefferquote im Prefill gegenüber Decode? Wenn Prefill den
  Cache mit Experten flutet, die der Decode nicht braucht, ist das ein Argument
  für getrennte Politiken.

Ergebnis ist primär eine Messung mit Empfehlung im Tuning-Doc. Code nur, wenn die
Messung eine konkrete Änderung nahelegt.

## Weitere Kandidaten, bewusst nicht eingeplant

Damit sie nicht vergessen, aber auch nicht ungeprüft eingebaut werden:

- **Fusionierte mHC-Ops.** llama.cpp hat sie (`0dc74e332 DeepseekV4: Add fused
  hyper-connection ops`). colibri rechnet 20 Sinkhorn-Iterationen pro Layer und
  Token in `coli_v4_hc_pre` ([c/deepseek_v4.c:1159](../c/deepseek_v4.c),
  aufgerufen ab [:3081](../c/deepseek_v4.c) ff.). 43 × 20 pro Token ist
  nicht nichts. **Erst messen, ob es im Profil auftaucht.**
- ~~**AVX-512**~~ — **erledigt, entfällt.** Der i5-13400F (Raptor Lake) hat kein
  AVX-512; Intel hat es auf den Consumer-Hybrid-Chips deaktiviert, und die
  Gracemont-E-Cores hatten es nie. Die vorhandenen AVX2-Pfade sind das Maximum.
  `ARCH=native` bleibt richtig (gibt AVX2+FMA+BMI2), bringt hier aber nichts
  darüber hinaus.
- **io_uring.** Siehe [09-arch-cachyos.md](09-arch-cachyos.md). Nach Plan 10
  messen, dann entscheiden.

## Tests

- `c/tests/test_v4_indexer_select.c` — partieller Select gegen `qsort`:
  identische Indexliste **in identischer Reihenfolge**, für `count` ∈ {1, 512,
  513, 250000}, mit vielen Gleichständen und mit `topk > count`.
- `c/tests/test_v4_pin_env.c` — `V4_PIN_SLOTS` / `V4_PIN_FRACTION` clampen,
  Müll fällt auf Default, beide gleichzeitig gesetzt → dokumentierter Vorrang.
- Trefferquoten-Messung als Skript, nicht als Gate: `c/tools/bench_v4_hitrate.py`,
  fester Promptsatz, gibt Hits/Misses/gelesene GB je Konfiguration aus.

## Abnahme

- Indexer-Select liefert **bitgleiche** Indexlisten und ist bei `count = 250000`
  messbar schneller.
- Die Trefferquoten-Messung liegt für ≥4 Pin-Konfigurationen vor; der Gewinner
  steht als Default im Tuning-Doc, mit den Zahlen.
- Prefill-Messung liegt vor, mit Empfehlung für Batchgröße und Prefetch.
- Tokenfolge unverändert — Cache-Politik und Select-Algorithmus dürfen die
  Ausgabe nicht berühren.

## Risiken

- **Der Select ist Router-Semantik.** Ein Reihenfolgefehler bei Gleichständen
  ändert die Attention-Auswahl und fällt in Stichproben womöglich nicht auf.
  Deshalb der Vergleich der vollständigen Liste, nicht der Menge.
- **Mehr Pins ist nicht automatisch besser.** Gepinnte Slots stehen der LRU nicht
  zur Verfügung; bei wechselnden Lasten kann eine hohe Pin-Quote die Trefferquote
  *senken*. Die Messung muss beide Richtungen zulassen.
- **Die Historie ist personenbezogen.** `.coli_usage` beschreibt, welche Experten
  *deine* Prompts benutzen. Das ist genau der Punkt — aber es heißt auch, dass
  Zahlen aus diesem Plan nicht auf andere übertragbar sind.
