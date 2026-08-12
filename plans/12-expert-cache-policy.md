# 12 — Expert-Cache-Politik, Prefill und der Lightning-Indexer

Voraussetzung: [00-reference.md](00-reference.md), [01-measure-and-ram-budget.md](01-measure-and-ram-budget.md).
Commit 2 trägt erst mit [03-kv-codec.md](03-kv-codec.md) — siehe dort.

*Commits:*
1. `perf: lift the compile-time pin ceiling and make it tunable`
2. `perf: parallel scoring and partial select for the lightning indexer`
3. `perf: measure and tune the prefill expert batching`

## Ergebnis Commit 1

Gebaut: `V4_PIN_SLOTS` (absolut) und `V4_PIN_FRACTION` (Anteil von
`available_pins`) ersetzen die Compile-Konstante
`COLI_V4_MAX_PIN_SLOTS_PER_LAYER` (16 im gebauten Binary) zur Laufzeit;
`V4_PIN_RAMP_REQUESTS` ersetzt ebenso `COLI_V4_PIN_RAMP_REQUESTS` (24). Beide
Funktionen (`coli_v4_pin_slots_ceiling`, `coli_v4_pin_ramp_requests`) leben in
der `RESOURCE_PLAN`-Unit — nicht in `EXPERT_STORE_HOT_ROWS16`, wo sie
aufgerufen werden — aus demselben Grund wie `coli_v4_vram_limit_bytes`: nur
`RESOURCE_PLAN` linkt eigenständig für den Test, der Rest der Hot-Policy-Unit
zieht Compressor/Matmul/RoPE aus anderen Units mit. `V4_PIN_SLOTS` gewinnt,
wenn beide Knöpfe gesetzt sind; Müll-Eingabe fällt auf den Compile-Default
zurück; das Ergebnis wird immer auf `[0, available_pins]` geklemmt.

Getestet in `c/tests/test_v4_pin_env.c` (Default, Clamp nach oben/unten,
Müll, negativ, beide Knöpfe gleichzeitig) und am echten Checkpoint bestätigt:
`v4_hot_policy`-Zeile zeigt korrektes `pin_slots_per_layer` für
`V4_PIN_SLOTS=8`, `V4_PIN_FRACTION=0.5` (→ 12 bei `available_pins=24`) und
`V4_PIN_RAMP_REQUESTS=0`.

**Die Trefferquoten-Messung aus der Abnahme ist nur teilweise gelungen.** Die
im Plan vorgeschlagene Sweep-Matrix (`V4_PIN_SLOTS` ∈ {4, 16, 32, 44} bei 32k,
{4, 8, 16, 22} bei 128k) erwies sich in dieser Sitzung als unpraktikabel: ein
einzelner Lauf mit 16 generierten Tokens bei `CTX=8192` las über 100 GB von
Platte, ohne innerhalb von 280 s fertig zu werden (siehe eigener Abschnitt
unten) — der kalte Expert-Cache dominiert bei jedem Lauf, der groß genug ist,
um die Politik überhaupt zu unterscheiden.

Ausweichend gemessen bei `CTX=4096` mit einem minimalen Zwei-Wort-Prompt (16
Forward-Tokens gesamt, `available_pins=42`, echter Checkpoint,
`COLI_V4_SAVE_USAGE=0`):

| `V4_PIN_SLOTS` | hits | misses | hit_rate | bytes gelesen |
|---|---|---|---|---|
| 2 | 2241 | 1629 | 57.907 % | 21.78 GB |
| 8 | 2240 | 1630 | 57.881 % | 21.79 GB |
| 16 | 2238 | 1632 | 57.829 % | 21.82 GB |
| 33 | 2238 | 1632 | 57.829 % | 21.82 GB |

Die Spanne (57.83–57.91 %) liegt im Rauschen — bei nur 16 Forward-Tokens gibt
es kaum Wiederholung im Working Set, also kaum Gelegenheit, bei der die
Pin-Politik überhaupt greifen könnte. Das ist selbst ein ehrliches Ergebnis:
ein aussagekräftiger Vergleich braucht deutlich mehr Decode-Volumen (oder die
im Plan genannten ≥5000 Requests, ab denen die Autopin-Historie greift), nicht
nur eine längere Sweep-Liste. Die ursprünglich verlangte Messung bei 32k/128k
mit realistischem Prompt bleibt offen.

**Nebenfund, der Platte-I/O-Kosten kalter Läufe:** Ein Probe-Lauf bei
`CTX=8192`, `V4_VRAM=1`, 16 generierte Tokens zeigte über `/proc/PID/io`
`read_bytes` jenseits von 100 GB (> 4× `target_cache=25 GiB`), ohne innerhalb
von 280 s fertig zu werden; die meisten Worker-Threads standen im
D-Zustand (I/O-Wartezustand). Das deckt sich mit der in
[00-reference.md](00-reference.md) dokumentierten niedrigen
Trefferquote im stationären Zustand (~17 % bei 32k, ~8 % bei 128k) — die
meisten Expertenanfragen sind beim kalten Start Fehltreffer, die je einen
Plattenzugriff kosten, und das ist genau der Befund, der diesen Plan
begründet (siehe „Warum dieser Plan überhaupt existiert" unten). Für die
Praxis heißt das: `bench_v4.py`s `medium`/`long`-Profile brauchen auf dieser
Maschine mehrere Minuten pro Konfiguration, nicht die im Plan angenommenen
„paar Minuten".

## Ergebnis Commit 2

Gebaut: die Bewertungsschleife (1) in `coli_v4_indexer_step` (`#pragma omp
parallel for`, mit `reduction(|:decode_failed)` statt eines direkten
Schreibzugriffs auf `result` aus der parallelen Region — letzteres wäre ein
Datenrennen) und der Sort (2) (`qsort` → `coli_v4_indexer_select`, Min-Heap-
basierter Partial-Select, O(count log topk) statt O(count log count)) — in
**beiden** Kopien (`INDEXER`, `INDEXER_SNAPSHOT`), identisch, wie
[AGENTS.md](../AGENTS.md) für die duplizierten Units verlangt.

`coli_v4_indexer_select` liegt als `static inline` in
`deepseek_v4_internal.h` (Typ `ColiV4IndexScore`, geteilt von beiden Kopien
über einen Type-Alias `typedef ColiV4IndexScore IndexScore;`) statt in einer
der beiden Units: die Funktion ist reine Array-Logik ohne V4-Zustand, und eine
unit-lokale Definition hätte in `INDEXER_SNAPSHOT` denselben Rename-Trick
gebraucht wie jedes andere dortige Symbol (`#define coli_v4_indexer_step
snapshot_copy_indexer_step` etc.) — für keinen Gewinn. So bekommt jede
Übersetzungseinheit, die den Header includet, ihre eigene private Kopie, und
der Test braucht kein `deepseek_v4.c` und keinen Unit-Define.

**Getestet:** `c/tests/test_v4_indexer_select.c` vergleicht
`coli_v4_indexer_select` gegen eine Referenz-`qsort` mit demselben
Tie-Break (aufsteigender Index bei Score-Gleichstand) für `count` ∈ {1, 512,
513, 250000}, `topk` ∈ {1, count/2, count, count+37}, und drei
Score-Verteilungen (alle verschieden, viele Gleichstände, alle gleich) —
identische Indexliste in identischer Reihenfolge in jedem Fall, plus die
Randfälle `count=0` und `topk≤0`. Läuft in <0.4 s, keine Abhängigkeit auf
`deepseek_v4.c`.

**Verifiziert gegen den echten Checkpoint:** `make -C c test`, `make -C c
check` und `make -C c deepseek-v4-tiny-check` (Token-identisch) sind grün mit
Commit 1+2 zusammen. Ein manueller Lauf gegen den vollen Checkpoint
(`CTX=4096`, `V4_VRAM=1`, Prompt „Say hi", `--max-tokens 8`) liefert exakt
denselben Text (`Hi! How can I help you today`) wie derselbe Lauf auf dem
Commit-1-only-Stand — die Parallelisierung und der Partial-Select ändern die
Token-Ausgabe nicht.

**Nicht gemessen:** der behauptete Geschwindigkeitsgewinn der
Bewertungsschleife selbst (skaliert sie tatsächlich mit der Teamgröße, oder
ist sie speicherbandbreitengebunden wie in der Abnahme als mögliches Ergebnis
vorgesehen?) — dafür bräuchte es einen Lauf mit großem `state->count`
(langer Kontext), und genau der ist der Lauf, der in dieser Sitzung am
kalten Cache scheiterte (siehe Commit 1 oben). Offen für eine Folge-Messung,
sobald ein Lauf mit warmem Cache oder viel mehr Zeitbudget möglich ist.

## Warum dieser Plan überhaupt existiert

**Das war die größte Lücke im Planset.** Die Pläne 01–11 machen alle dasselbe:
mehr RAM für den Expert-Cache freimachen und die Lesevorgänge beschleunigen.
Keiner kümmert sich darum, **was im Cache liegt**.

Der Plan trägt inzwischen zwei verschiedene Hebel: Commit 1 und 3 sind
Cache-Politik, Commit 2 ist der Indexer und hat damit nichts zu tun — er steht
hier, weil er derselben Sorte Fund entspringt (vorhandene Maschinerie, die nie
gemessen wurde) und dieselbe Messgrundlage aus Plan 01 braucht.

Bei rund 17 % Residenz im 32k-Endprofil und nur rund 8 % bei 128k ist das der
falsche Schwerpunkt. Ein Treffer kostet null
Bytes von der Platte; ein Fehltreffer kostet ~13.4 MB (ein voller
`expert_record_bytes`-Eintrag, w1/w2/w3 + Scales — siehe
[00-reference.md](00-reference.md)), egal wie schnell das Laufwerk ist. **Die
Trefferquote zu erhöhen schlägt jede Beschleunigung der
Fehltreffer** — und Colibrì hat dafür bereits eine vollständige Maschinerie, die
niemand für dieses Setup eingestellt hat.

## Commit 1 — Die Pin-Obergrenze

`V4HotPolicy` ([c/deepseek_v4.c:6688](../c/deepseek_v4.c)) pinnt Experten nach
gemessener Nutzungshäufigkeit, mit persistierter Historie in
`<model_dir>/.coli_usage`:

| Knopf | Wo | Heute |
|---|---|---|
| `COLI_V4_AUTOPIN` | [:6030](../c/deepseek_v4.c) | an, sofern nicht `0` |
| `COLI_V4_SAVE_USAGE` | [:6788](../c/deepseek_v4.c) | schreibt die Historie zurück |
| `COLI_V4_PREWARM` | [:7290](../c/deepseek_v4.c) | wärmt den Cache aus der Historie |
| `history_seeded` | [:7277](../c/deepseek_v4.c) | erst ab **5000** beobachteten Requests |

**Der Deckel ist eine Compile-Zeit-Konstante** ([:5994](../c/deepseek_v4.c)):

```c
#ifndef COLI_V4_MAX_PIN_SLOTS_PER_LAYER
#define COLI_V4_MAX_PIN_SLOTS_PER_LAYER 4
#endif
    if (maximum_pins > COLI_V4_MAX_PIN_SLOTS_PER_LAYER)
        maximum_pins = COLI_V4_MAX_PIN_SLOTS_PER_LAYER;
```

`Makefile.deepseek-v4:44` hebt ihn auf **16** (Linux-Zweig; der MSYS2-Zweig setzt
denselben Wert bei `:34`). Nach den VRAM-Phasen fasst der Cache auf der
korrigierten Bilanz vor Slot-Rundung rund **44 Slots bei 32k**
(23.55 GiB / 0.535 GiB) beziehungsweise nur **22 Slots bei 128k**
(11.55 GiB / 0.535 GiB; siehe [00-reference.md](00-reference.md)). Der
16er-Default pinnt damit rund 36 % beziehungsweise 73 % des Caches. Bei 128k
dominiert LRU also nicht mehr; der sinnvolle Pin-Anteil ist kontextabhängig.

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
Dazu `COLI_V4_PIN_RAMP_REQUESTS` (heute 24, `Makefile.deepseek-v4:45`) ebenfalls
zur Laufzeit setzbar, als `V4_PIN_RAMP_REQUESTS`.

Alle drei Knöpfe gehören in die Tabelle in [00-reference.md](00-reference.md) und
in `docs/ENVIRONMENT.md` — das ist Teil dieses Commits, nicht Nacharbeit.

**Dann messen**, nicht raten: Trefferquote über einen festen Promptsatz bei
`V4_PIN_SLOTS` ∈ {4, 16, 32, 44} bei 32k und {4, 8, 16, 22} bei 128k. Der
Gewinner wird zusammen mit dem gemessenen Kontext als Default für das jeweilige
Profil dokumentiert.

Die `.coli_usage`-Historie braucht ≥5000 Requests, um zu greifen. Für ein
persönliches Setup heißt das: **`COLI_V4_SAVE_USAGE=1` von Anfang an setzen** und
die Historie über Wochen wachsen lassen. Gehört ins Tuning-Doc als
Erstinbetriebnahme-Hinweis — sonst wundert man sich, warum Pinning „nichts tut".

## Commit 2 — Die Indexer-Bewertungsschleife

`coli_v4_indexer_step` steht **zweimal** im Amalgam
([c/deepseek_v4.c:3604](../c/deepseek_v4.c) und [:5409](../c/deepseek_v4.c), Units
`INDEXER` und `INDEXER_SNAPSHOT` — siehe [00-reference.md](00-reference.md)).
Beide Kopien ändern, sonst driften sie.

Die Funktion tut zwei Dinge, und **die Reihenfolge der Kosten ist umgekehrt zur
Intuition**:

```c
for (int candidate = 0; !result && candidate < state->count; candidate++) {   /* (1) */
    const float *key = state->compressed + (size_t)candidate * dimension;
    for (int head = 0; head < heads; head++)          /* 64  */
        for (int i = 0; i < dimension; i++)           /* 128 */
            dot += query[i] * key[i];
}
qsort(scores, state->count, sizeof(*scores), descending_score);              /* (2) */
```

Bei 1M Kontext ist `state->count` = 250 000, pro Token und pro CSA-Layer, also
**21×** pro Token:

| | Arbeit pro Token (1M, alle CSA-Layer) | gelesen |
|---|---|---|
| (1) Bewertung | **43 G MAC** | **2.69 GB** |
| (2) `qsort` | ~4.5 M Vergleiche × 21 ≈ 95 M | — |

**Die Bewertung ist rund 400× teurer als der Sort.** Eine frühere Fassung dieses
Plans führte nur (2) auf und nannte es „den" Gewinn. Beides gehört in diesen
Commit, aber in dieser Reihenfolge.

### (1) zuerst

- **`#pragma omp parallel for` über `candidate`.** Die Schleife ist die einzige
  heiße V4-Schleife ohne OpenMP — die 18 vorhandenen Pragmas
  ([09-arch-cachyos.md](09-arch-cachyos.md)) lassen sie aus. Sie ist rein lesend
  bis auf `scores[candidate]`, also trivial parallelisierbar, und **das ändert
  nichts an der Auswahl**: jeder Eintrag bekommt seinen eigenen Slot.
- **Mit `V4_KV_INDEX=native` (Plan 03) fallen die gelesenen Bytes auf 1/7.5** —
  357 MB statt 2.69 GB. Deshalb ist 03 Voraussetzung, wenn dieser Commit tragen
  soll, und deshalb steht 03 in [AGENTS.md](../AGENTS.md) vor 12.
- Der WMMA-CUDA-Kernel aus llama.cpp
  (`ggml/src/ggml-cuda/lightning-indexer.cu`, siehe
  [llamacpp-deepseek-v4.md](llamacpp-deepseek-v4.md)) adressiert genau diese
  Schleife und ist die Vorlage, sobald [05-cuda-attention.md](05-cuda-attention.md)
  den Build mitbringt. **Nicht** Teil dieses Commits.

### (2) danach

Ein Min-Heap der Größe `topk`, oder ein Quickselect-Durchlauf. Wichtig ist die
**Ordnung des Ergebnisses**: `descending_score` bricht Gleichstände über den Index
(`return a->index - b->index`), und die Auswahl geht danach in
`compressed_indices`. Der Ersatz muss dieselbe Menge und dieselbe Reihenfolge
liefern — sonst ändert sich, welche Tokens die Attention sieht, und das ist
Router-Semantik, nicht Präzision.

Deshalb: der Test vergleicht die **komplette Indexliste** gegen den bisherigen
`qsort`-Pfad, auf Zufallsdaten inklusive vieler Gleichstände. Nicht nur die
Mengengleichheit — die Reihenfolge.

**Auch die Parallelisierung von (1) braucht diesen Test.** Sie ändert die
Auswahl nicht, aber sie ändert die Reihenfolge, in der `scores[]` gefüllt wird,
und ein Fehler dabei sieht aus wie ein Select-Fehler.

## Commit 3 — Prefill

**Zurückgestellt.** Die geforderte Messmatrix braucht wegen des kalten
Expert-Cache mehrere lange, dedizierte Läufe und wird nicht mit einer
Implementierungsphase vermischt. Sie bleibt als eigenständige Messaufgabe in
diesem Plan offen; bis sie mit ausreichend Laufzeit durchgeführt wird, setzt
sie weder eine Batch- noch eine Prefetch- oder Cache-Policy als neuen Default.

Dieser Commit misst und stimmt die bestehende V4-Batchgröße, Prefetch- und
Cache-Politik ab. Er ändert **nicht** die zwei auf `CTX` skalierten
Session-Aktivierungspuffer; deren Chunk-/DSpark-Übergabe gehört in
[Plan 14](14-chunked-prefill-dspark.md).

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
  Token in `coli_v4_hc_pre` ([c/deepseek_v4.c:1760](../c/deepseek_v4.c),
  aufgerufen ab [:4115](../c/deepseek_v4.c) ff.). 43 × 20 pro Token ist
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
  513, 250000}, mit vielen Gleichständen und mit `topk > count`. Derselbe
  Vergleich einmal mit `OMP_NUM_THREADS=1` und einmal mehrfädig, damit die
  Parallelisierung der Bewertungsschleife mit abgedeckt ist.
- `c/tests/test_v4_pin_env.c` — `V4_PIN_SLOTS` / `V4_PIN_FRACTION` clampen,
  Müll fällt auf Default, beide gleichzeitig gesetzt → dokumentierter Vorrang.
- Trefferquoten-Messung als Skript, nicht als Gate: `c/tools/bench_v4_hitrate.py`,
  fester Promptsatz, gibt Hits/Misses/gelesene GB je Konfiguration aus.

## Abnahme

- Indexer-Select liefert **bitgleiche** Indexlisten, ein- und mehrfädig, und ist
  bei `count = 250000` messbar schneller.
- Die Bewertungsschleife skaliert mit der Teamgröße — gemessen, nicht angenommen.
  Wenn sie es nicht tut, ist sie speicherbandbreitengebunden und der Gewinn hängt
  an `V4_KV_INDEX=native` statt an OpenMP; auch das ist ein Ergebnis.
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
