# AGENTS.md

Arbeitsanweisung für Coding-Agents in diesem Fork. Menschen dürfen mitlesen.

## Was dieser Fork ist

Ein Fork von [JustVugg/colibri](https://github.com/JustVugg/colibri) mit **einem**
Ziel: **DeepSeek-V4-Flash-0731** plus den **DSpark**-Drafter auf genau einer
Maschine so gut wie möglich fahren.

| | |
|---|---|
| CPU | Intel **i5-13400F** — 6 P + 4 E Cores, 16 Threads, kein AVX-512 |
| RAM | 32 GB **DDR4-3200** Dual-Channel — ~45 GB/s |
| GPU | RTX 4070, 12 GB — **headless im V4-Betrieb**, ~11.7 GiB nutzbar |
| Laufwerk A | 1 TB NVMe Gen4, DRAM-los (HMB) — trägt das Modell |
| Laufwerk B | 512 GB SSD Gen3, DRAM-los (HMB) — **optional**, zweite Kopie |
| OS | CachyOS (Arch-Familie) |

**Zielplattform ist Linux x86-64.** Windows, macOS, Metal, Vulkan, HIP und die
zugehörigen Shims sind toter Code (Rückbau in [Plan 11](plans/11-strip-to-v4.md)).
Die anderen Motoren des Upstreams (GLM-5.2, Inkling, Kimi K3, OLMoE) ebenso.

**Was bleibt:** `coli chat`, `coli serve`, `coli web` samt `openai_server.py` und
der WebUI — das ist die tägliche Bedienung, siehe [Plan 13](plans/13-frontend-v4.md).

**Optimiere für diese Hardware**, nicht für Allgemeingültigkeit — aber ohne die
Semantik des Modells zu ändern (siehe Regeln).

**Ein** Checkpoint —
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) —
und er **liegt auf dieser Maschine vollständig vor**:

```
~/Services/models/colibri/deepseek-v4-flash    161 GB, 48 Shards, 72 317 Tensoren
```

Das ist das Modellverzeichnis für jeden Test gegen echte Gewichte. Das Binary
nimmt es als erstes Positionsargument:

```bash
c/deepseek_v4 ~/Services/models/colibri/deepseek-v4-flash "prompt" --max-tokens 64
```

Die 4 705 `mtp.0`–`mtp.2`-Tensoren sind darin enthalten, also ist es zugleich
das **einzige** Fixture, mit dem DSpark-Pfade überhaupt laufen: das
Tiny-Fixture aus Regel 4 hat null `mtp.*`-Tensoren. Wer eine Änderung am
Drafter nur gegen Tiny grün bekommt, hat den Drafter nicht getestet.

**DSpark steckt darin.** Die Drafter-Tensoren liegen im selben Checkpoint unter dem
Präfix `mtp.<stage>.` — colibri sucht sie in `engine->target_index`, nicht in einem
zweiten Modellverzeichnis (`v4_dspark_markov_probe`,
[c/deepseek_v4.c:7571](c/deepseek_v4.c); `v4_ds_name`,
[c/deepseek_v4_dspark.inc:193](c/deepseek_v4_dspark.inc)).
Es gibt kein separates DSpark-Repo zu laden; ein solcher Link führt nur in die Irre.

## Die Pläne

Alles Vorhaben steht in [`plans/`](plans/). **Lies vor jeder Implementierung
[`plans/00-reference.md`](plans/00-reference.md)** — dort stehen Hardwarebudget,
Modellgeometrie, Codekarte und die Konventionen, die alle Phasenpläne voraussetzen
und nicht wiederholen.

| # | Plan | Status |
|---|---|---|
| 00 | [Referenz](plans/00-reference.md) | — |
| 01 | [Messen und RAM-Budget](plans/01-measure-and-ram-budget.md) | fertig (#2) |
| 02 | [Flash Attention](plans/02-flash-attention.md) | fertig (#3) |
| 03 | [KV-Codec](plans/03-kv-codec.md) | fertig (#5) |
| 04 | [TurboQuant](plans/04-turboquant.md) | fertig (#6) |
| 05 | [CUDA-Attention](plans/05-cuda-attention.md) | fertig (#7) |
| 06 | [Dense in VRAM](plans/06-dense-vram.md) | fertig (#8) |
| 07 | [Head und DSpark in VRAM](plans/07-head-dspark-vram.md) | fertig (#9) |
| 08 | [VRAM-Planner](plans/08-vram-planner.md) | fertig (#12) |
| 09 | [Arch / CachyOS](plans/09-arch-cachyos.md) | Commit 1 fertig (#11), Rest offen |
| 10 | [Dual-Streaming](plans/10-dual-streaming.md) | Implementierung fertig; Hardwareprofil in Plan 15 |
| 11 | [Rückbau auf V4](plans/11-strip-to-v4.md) | offen, **zuletzt** |
| 12 | [Expert-Cache-Politik](plans/12-expert-cache-policy.md) | Implementierung fertig (#14); Profilabnahme in Plan 15 |
| 13 | [Frontend V4-only](plans/13-frontend-v4.md) | offen, optional |
| 14 | [Chunked Prefill + DSpark](plans/14-chunked-prefill-dspark.md) | Commit 1+2 fertig; End-to-End-Abnahme in Plan 15 |
| 15 | [Abschlussabnahme und Tuning](plans/15-final-validation-and-tuning.md) | offen, **zuletzt** |

Referenzdokumente ohne Nummer:
[Paper](plans/paper-deepseek-v4.md) ·
[llama.cpp](plans/llamacpp-deepseek-v4.md) ·
[DeepSpec/DSpark](plans/deepspec-dspark.md)

### Empfohlene Reihenfolge

Die Nummern sind Kennungen, keine Reihenfolge. So wird gearbeitet:

**Stand:** 01–08 sind fertig, dazu **09 Commit 1** nachgeholt (ohne belegten
Gewinn, siehe dessen Ergebnisblock). **08** ist der VRAM-Tier-Planner samt
32GB+12GB-Doku-Profil, gegen den echten Checkpoint gemessen — siehe dessen
Ergebnisblock; die exklusive KV-Host-Shadow-Entfernung bleibt offen, ist aber
kein Teil dieses Plans mehr. Die verbliebenen Full-Checkpoint- und
Langlaufmessungen aus 09, 10, 12 und 14 liegen gemeinsam in **Plan 15**; als
nächstes wird **10** implementiert. Die Tabelle unten
ist die ursprüngliche Empfehlung und wurde nicht buchstäblich befolgt: 12
wurde bisher übersprungen, dafür lief 02 → 05 → 06 → 07 → 08 am Stück durch,
und 04 kam vor 07 statt danach. Für den Rest gilt sie unverändert.

| Schritt | Plan | Warum hier |
|---|---|---|
| 1 | **01** | Der Benchmark-Harness ist der Maßstab für alles Weitere. Baseline auf `main` aufnehmen, **bevor** irgendetwas geändert wird. Die RAM-Knöpfe fallen nebenbei ab. |
| 2 | **09 Commit 1+1b** | `omp_tune.h` verdrahten (drei Zeilen), **plus** Hybrid-Bewusstsein: die Linux-Zählung sieht 6 P- und 4 E-Cores als „10 gleiche" und lässt die E-Cores in `schedule(static)` das Tempo vorgeben. |
| 3 | **03** | Nativer KV-Codec: 3.5× / 7.5×, bit-exakt. Macht das VRAM-Budget der späteren Phasen erst schließbar. |
| 4 | **12** | Trefferquote schlägt Durchsatz. Bei selbst im 32k-Endprofil nur ~17 % Residenz (128k ~8 %) ist das der größte verbleibende Hebel — und er braucht 01 als Messgrundlage. |
| 5 | **02** | Flash-Attention. Eigener Gewinn, und Voraussetzung für 05. |
| 6 | **05** | CUDA-Attention. Commit 1 zieht die CUDA-Bauinfrastruktur ein, die 06 und 07 brauchen. |
| 7 | **06** | Dense in VRAM — der größte RAM-Einzelposten. |
| 8 | **07** | Head und DSpark in VRAM. |
| 9 | **08** | VRAM-Planner, der 05–07 zu einer Entscheidung zusammenfasst. |
| 10 | **04** | TurboQuant — Pflicht für das 256k-/1M-Langkontextprofil. |
| 11 | **14** | Macht 256k mit TurboQuant und DSpark ohne Voll-Prefill-Buffer möglich; vor jedem 1M-Versuch. |
| 12 | **10** | Dual-Streaming. Unabhängig, kann ab Schritt 2 jederzeit dazwischen. Das zweite Laufwerk ist **optional** — Einzellaufwerk bleibt Default und Pflicht-Abnahme. |
| 13 | **09 Rest** | THP, CUDA-Pfade, Tuning-Doku. |
| 14 | **13** | Frontend. |
| 15 | **11** | Rückbau. |
| 16 | **15** | Finale Full-Checkpoint-Abnahme und ausschließlich messbasiertes Tuning. |

**Harte Abhängigkeiten**, die man nicht umstellen darf:

- **01 vor allem** — ohne Baseline ist keine Behauptung prüfbar
- **02 vor 05** — der CPU-Flash-Kernel ist Vorlage und Testorakel für den CUDA-Kernel
- **03 vor 04** — Turbo hängt am Codec-Interface
- **04 vor 14** — das tägliche 256k-Profil und das 1M-Experiment verwenden Turbo3
- **05 Commit 1 vor 06/07** — dort entsteht der CUDA-Build für V4
- **07 und 08 vor 14** — Chunked Prefill muss den verifizierten DSpark-Decode
  und dessen endgültige VRAM-Planung übernehmen, nicht eine Zwischenform
- **13 vor 11** — sonst ist der Launcher zwischenzeitlich kaputt
- **10 vor 11** — 11 löscht den Mirror-Code, den 10 als Vorlage braucht
- **11 vor 15** — die Abschlussmatrix misst den tatsächlich ausgelieferten V4-only-Stand
- **15 zuletzt**, immer

Bei knapper Zeit: `01 → 09.1 → 03 → 12 → 06` hat den meisten Ertrag pro Aufwand.
Das ist zugleich die Reihenfolge, in der die Gewinne am wenigsten voneinander
abhängen.

**Warum 12 so weit vorne steht:** Ein Cache-Treffer kostet null Bytes von der
Platte, ein Fehltreffer 13.4 MB — unabhängig davon, wie schnell das Laufwerk ist.
Die Trefferquote zu heben schlägt jede Beschleunigung der Fehltreffer. Die
Pläne 01–11 machten alle dasselbe (mehr RAM, schnellere Reads); keiner kümmerte
sich darum, *was* im Cache liegt.

## Branch- und PR-Regeln

**Ein Plan, ein Branch, ein PR.** Keine Sammel-PRs über mehrere Phasen.

```
phase-01-ram-budget
phase-02-flash-attention
phase-03-kv-codec
phase-04-turboquant
phase-05-cuda-attention
phase-06-dense-vram
phase-07-head-dspark-vram
phase-08-vram-planner
phase-09-arch-cachyos
phase-10-dual-streaming
phase-11-strip-to-v4
phase-12-expert-cache
phase-13-frontend
phase-14-chunked-prefill
phase-15-final-validation
```

- Branch von `main`, außer der Plan hängt an einem anderen — dann von dessen
  Branch, und der PR nennt die Abhängigkeit in der ersten Zeile.
- **Es gibt keinen eigenen Doku-Branch.** Änderungen an `plans/` und an dieser
  Datei laufen auf dem Phasenbranch mit, zu dem sie gehören; rein
  übergreifende Doku geht als eigener kleiner PR direkt gegen `main`.
- Die Commits innerhalb eines PRs folgen der Commit-Liste im Plan. Jeder Commit
  baut und ist grün. Kein „fixup später".
- PR-Beschreibung: Link auf den Plan, die Abnahmekriterien als Checkliste, die
  gemessenen Zahlen. Ein PR ohne Messung ist bei einem Performance-Plan nicht fertig.
- Nichts wird gepusht oder gemerged, ohne dass der Mensch es sagt.

## Implementierungsregeln

### 1. Semantik vor Geschwindigkeit

Die Upstream-README garantiert, dass die Default-Policy Modellpräzision und
Router-Semantik **nicht still ändert**. Das gilt hier weiter.

- Jede verlustbehaftete Optimierung ist **opt-in** über eine Env-Variable.
- Ausnahme, wenn sie belegt ist: `V4_KV=native` ist Default, **weil** es
  bit-exakt ist — und der Test dafür ist Teil des PRs.
- Alles, was die Auswahl von Experten oder KV-Einträgen ändert (Router, Indexer,
  Top-k), ist Semantik, nicht Präzision. Eigener Schalter, nie implizit mitgezogen.

### 2. Env-Knöpfe

`V4_*` für Nutzerknöpfe, `COLI_V4_*` für Engine-Interna. Jeder neue Knopf:
Default = heutiges Verhalten, Wertebereich geklemmt, Müll-Eingabe fällt auf
Default zurück, und ein Eintrag in `docs/ENVIRONMENT.md`.

### 3. Die duplizierten Units

`c/deepseek_v4.c` ist ein Amalgam. Der Generator `_amalgamate_v4.py` ist **nicht**
im Repo — die committete Datei ist Source of Truth. **Vier** Quelldateien stehen
mehrfach darin, jeweils unter `#define`-Umbenennungen:

| Quelle | Kopien | Units (`#ifdef COLI_V4_UNIT_…`) |
|---|---|---|
| `deepseek_v4_attention.c` | **3** | `ATTENTION`, `ATTENTION_BATCH`, `ATTENTION_TRANSACTION` |
| `deepseek_v4_compressor.c` | **2** | `COMPRESSOR`, `COMPRESSOR_SNAPSHOT` |
| `deepseek_v4_indexer.c` | **2** | `INDEXER`, `INDEXER_SNAPSHOT` |
| `deepseek_v4_layer.c` | **2** | `LAYER_RESIDENT`, `LAYER` |

Die Units stehen als `#ifdef`-Blöcke in der Datei — `grep -n 'COLI_V4_UNIT' c/deepseek_v4.c`
gibt die aktuellen Grenzen. Zeilennummern hier zu notieren lohnt nicht; sie
verschieben sich bei jedem Commit, der die Datei anfasst.

Jede Änderung muss in **alle** Kopien der betroffenen Quelle, **plus** in den
Batch-Pfad `coli_v4_attention_window_batch_ref`, der die Attention-Logik nochmal
als eigenen Text enthält. Der Compressor/Indexer-Fall ist der, der übersehen wird:
`plans/03-kv-codec.md` und `plans/12-expert-cache-policy.md` fassen beide genau
diese zwei Funktionen an.

**Der Test dafür existiert:** `c/tests/test_v4_attention_source.py` vergleicht die
Kopien aller vier Quellen markerbasiert — Attention 3×, Compressor 2×, Indexer 2×,
und für `deepseek_v4_layer.c` die vier gemeinsamen Definitionen, weil
`LAYER_RESIDENT` bewusst den Rows8-Block dazu hat. Dazu prüft
`c/tests/test_deepseek_v4_dspark_source.py` die Drafter-Invarianten. Beide laufen
in `make -C c check`; wer eine Kopie vergisst, wird dort rot.

### 4. Tests sind Teil des Commits

Eine Make-Regel `tests/test_x$(EXE):` in `c/Makefile` genügt — `TEST_RULES`
([c/Makefile:359](c/Makefile)) leitet die Gates daraus ab. Keine zentrale Liste.

Vor jedem PR:

```bash
make -C c test && make -C c check
```

```bash
make -C c deepseek-v4-tiny-check
```

Das Tiny-Fixture muss mit den Defaults **token-identisch** bleiben. Kippt es, ist
das ein Befund und gehört in den PR-Text — nicht weggedrückt.

Für die erzwungene Neugenerierung braucht der Check die exakt gepinnten
CPU-Pakete aus `c/tools/requirements-deepseek-v4-tiny.txt`. Auf dieser Maschine
ist dafür `.venv-v4-tiny` eingerichtet; sie braucht weder CUDA noch den vollen
Checkpoint. Reproduzierbar neu anlegen und verwenden:

```bash
uv venv .venv-v4-tiny --python /usr/bin/python3
uv pip install --python .venv-v4-tiny/bin/python \
  -r c/tools/requirements-deepseek-v4-tiny.txt
make -C c PYTHON="$PWD/.venv-v4-tiny/bin/python" deepseek-v4-tiny-check
```

Das Fixture ist ein synthetisches, deterministisches Drei-Layer-V4 und kein
Ausschnitt des großen Modells. Es prüft das Transformers-Orakel und die
Tokenidentität; Full-Checkpoint-Qualität und -Durchsatz ersetzt es nicht.

### 5. Messen statt behaupten

Jeder Performance-Plan hat Abnahmekriterien mit Zahlen. Ein PR, der sie nicht
misst, ist nicht fertig. Wenn eine Messung die Erwartung verfehlt, ist das ein
Ergebnis: dokumentieren, nicht schönrechnen.

Fremde Zahlen (die +2.3× aus `omp_tune.h`, die TurboQuant-Cosinuswerte aus dem
llama.cpp-Fork) sind Erwartungen, keine Zusagen. Nie ungeprüft weitergeben.

### 6. Stil

Schreib Code, der wie der umgebende Code liest: Kommentardichte, Namensgebung,
Fehlerkonvention (`set_error`/`plan_error`), Clamp-Muster für Env-Werte. Der
V4-Code kommentiert das *Warum*, nicht das *Was* — halte das durch.

### 7. Das `reference/`-Verzeichnis

`reference/` ist gitignored und enthält fremde Repos zum Nachschlagen:

- `llama-cpp-turboquant` — llama.cpp-Fork mit V4- und TurboQuant-Support
- `DeepSpec` — DeepSeeks Draft-Model-Repo, maßgeblich für DSpark

**Niemals committen.** Auch `paper.pdf` nicht (Fremdcopyright, gitignored).
Bei Widersprüchen gilt: Paper > offizielle Referenzimplementierung > llama.cpp.

## Pflege dieser Datei und der Pläne

**Wenn ein Plan implementiert ist:**

1. Status in der Tabelle oben auf `fertig (#PR)` setzen.
2. Im Plan selbst oben einen Block `## Ergebnis` ergänzen: was tatsächlich gebaut
   wurde, die gemessenen Zahlen, und wo es vom Plan abwich und warum.
3. Wenn dabei eine Annahme aus `plans/00-reference.md` widerlegt wurde: **dort
   korrigieren**, nicht nur im Phasenplan. Die Referenz ist das, was die anderen
   Pläne voraussetzen — eine falsche Zahl dort pflanzt sich fort.
4. Neue Env-Knöpfe in die Tabelle in `plans/00-reference.md` und in
   `docs/ENVIRONMENT.md`.

**Wenn sich beim Implementieren zeigt, dass ein späterer Plan nicht mehr stimmt:**
den späteren Plan im selben PR anpassen. Ein Plan, der der Realität
widerspricht, ist schlimmer als kein Plan.

**Wenn ein Plan sich als überflüssig erweist:** Status `verworfen`, mit Begründung
im Plan. Nicht löschen — die Begründung ist die eigentliche Information.

Diese Datei ist die Landkarte. Wenn sie nicht mehr stimmt, findet der nächste
Agent nichts wieder.

## Bekannte Fallstricke

- **`packed_rows8`** ist kein Checkpoint-Format, sondern ein AVX2-Repack nach dem
  Laden (`v4_fp8_pack_rows8_inplace`, [c/deepseek_v4.c:496](c/deepseek_v4.c)). Die
  Scales bleiben 128×128. Für GPU-Tensoren einfach überspringen.
- **Der Indexer-Cache ist nicht gedeckelt.** Der `capacity > 128`-Deckel gilt nur
  der Erstallokation; `coli_v4_indexer_step` verdoppelt danach unbegrenzt.
- **Der Indexer-Scan ist der größte übersehene Posten.** `coli_v4_indexer_step`
  bewertet pro Token und pro CSA-Layer **alle** `state->count ≈ ctx/4` Einträge
  gegen 64 Heads × 128 Dims — 352 MB/Token bei 128k, 2.69 GB bei 1M, und die
  Schleife steht als einzige der heißen V4-Schleifen **außerhalb jedes
  `#pragma omp`**. Wer die KV-Bandbreite diskutiert, ohne diese Zahl, diskutiert
  den kleineren Posten. Siehe `plans/00-reference.md`.
- **`n_win = 128`**, nicht ~2048. Der Fensterring ist speichermäßig irrelevant,
  bandbreitenmäßig aber in jedem Layer präsent.
- **`CTX` skaliert zwei volle Prefill-Buffer.** `session->state` und
  `session->next` belegen zusammen
  `CTX × hc_mult × hidden_size × 4 × 2` Bytes: auf der Paper-Geometrie
  4 GiB bei 32k und 16 GiB bei 128k. Diese Reserve gehört vor den Expert-Cache;
  ein fester 64-Token-`hidden`-Posten ist eine massive Fehlbuchung.
- **HCA-Layer haben keine Sparse Attention** — sie lesen den kompletten
  komprimierten Cache und dominieren unter den *gelesenen KV-Zeilen* die
  Bandbreite bei langem Kontext (aber nicht gegen den Indexer-Scan oben).
- **Der Head ist 0.99 GiB beziehungsweise 1.06 GB.** 129280 × 4096 × 2 B =
  1.059 **GB**. Für Zeitrechnungen (÷ GB/s) ist der Dezimalwert richtig, in einer
  GiB-Bilanz nicht. Das Dense-Gesamtinventar sind echte 6.267 GiB, davon sind
  aber nur 5.456 GiB FP8-Gewichte und Scales in Phase 06 verschiebbar; 0.810 GiB
  BF16/f32/i64 bleiben im RAM.
- **Die DSpark-Reserve ist 1.17 GiB, nicht 1.25.** Beim Default sind
  `0.45 × 10⁹ + 768 × 2²⁰ = 1 255 306 368 B` genau 1.255 GB, aber 1.169 GiB.
- **~13 tok/s ist der Bandbreiten-Deckel.** Jeder Token schiebt ~3.4 GB
  Expert-Gewichte (6 × 43 × 13.4 MB) durch den Speicherbus, auch bei 100 %
  Cache-Treffer. Auf DDR4-3200 sind das ~77 ms. Kein Plan verschiebt das; sie
  nähern sich ihm nur. Wer eine Messung sieht, die darüber liegt, hat sich
  verrechnet oder etwas anderes gemessen.
- **DFlash ≠ DeepSeek-V4-Flash.** In llama.cpp ist `dflash` eine Drafter-Arch;
  das 284B-Modell heißt dort `deepseek4`.
- **`coli_physical_cores()` ist auf Linux hybrid-blind.** Es zählt eindeutige
  `thread_siblings_list`-Einträge ([c/omp_tune.h:111](c/omp_tune.h)) und liefert
  auf dem i5-13400F **10** — 6 P- plus 4 E-Cores, als wären sie gleich. Der
  macOS-Zweig löst genau das ([c/omp_tune.h:98](c/omp_tune.h), gemessen
  −4.2 % Decode auf M1 Max), der Linux-Zweig nicht. Siehe Plan 09.
- **Kein separater DSpark-Download.** Der Drafter liegt im Hauptcheckpoint unter
  `mtp.<stage>.`. Wer nach einem eigenen DSpark-Repo sucht, sucht falsch.
- **MTP-Tiefe 1, aber drei DSpark-Stufen.** Das Paper nennt
  `num_nextn_predict_layers = 1`; der Drafter fährt trotzdem drei Stufen
  (`V4_DSPARK_STAGES 3`, geprobt werden `mtp.0` bis `mtp.2`). Kein Widerspruch —
  MTP-Tiefe und Drafter-Stufen sind verschiedene Dinge.
