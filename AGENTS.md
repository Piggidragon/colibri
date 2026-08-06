# AGENTS.md

Arbeitsanweisung für Coding-Agents in diesem Fork. Menschen dürfen mitlesen.

## Was dieser Fork ist

Ein Fork von [JustVugg/colibri](https://github.com/JustVugg/colibri) mit **einem**
Ziel: **DeepSeek-V4-Flash-0731** plus den **DSpark**-Drafter auf genau einer
Maschine so gut wie möglich fahren.

| | |
|---|---|
| RAM | 32 GB |
| GPU | RTX 4070, 12 GB VRAM |
| Laufwerk A | 1 TB NVMe Gen4, DRAM-los (HMB) |
| Laufwerk B | 512 GB SSD Gen3, DRAM-los (HMB) |
| OS | CachyOS (Arch-Familie) |

Die anderen Motoren des Upstreams (GLM-5.2, Inkling, Kimi K3, OLMoE) sind hier
toter Code und werden am Ende entfernt. **Optimiere für diese Hardware**, nicht
für Allgemeingültigkeit — aber ohne die Semantik des Modells zu ändern (siehe
Regeln).

**Ein** Checkpoint:
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir /pfad/DeepSeek-V4-Flash
```

**DSpark steckt darin.** Die Drafter-Tensoren liegen im selben Checkpoint unter dem
Präfix `mtp.<stage>.` — colibri sucht sie in `engine->target_index`, nicht in einem
zweiten Modellverzeichnis
([c/deepseek_v4.c:6325](c/deepseek_v4.c), [c/deepseek_v4_dspark.inc:243](c/deepseek_v4_dspark.inc)).
Es gibt kein separates DSpark-Repo zu laden; ein solcher Link führt nur in die Irre.

## Die Pläne

Alles Vorhaben steht in [`plans/`](plans/). **Lies vor jeder Implementierung
[`plans/00-reference.md`](plans/00-reference.md)** — dort stehen Hardwarebudget,
Modellgeometrie, Codekarte und die Konventionen, die alle Phasenpläne voraussetzen
und nicht wiederholen.

| # | Plan | Status |
|---|---|---|
| 00 | [Referenz](plans/00-reference.md) | — |
| 01 | [Messen und RAM-Budget](plans/01-measure-and-ram-budget.md) | offen |
| 02 | [Flash Attention](plans/02-flash-attention.md) | offen |
| 03 | [KV-Codec](plans/03-kv-codec.md) | offen |
| 04 | [TurboQuant](plans/04-turboquant.md) | offen |
| 05 | [CUDA-Attention](plans/05-cuda-attention.md) | offen |
| 06 | [Dense in VRAM](plans/06-dense-vram.md) | offen |
| 07 | [Head und DSpark in VRAM](plans/07-head-dspark-vram.md) | offen |
| 08 | [VRAM-Planner](plans/08-vram-planner.md) | offen |
| 09 | [Arch / CachyOS](plans/09-arch-cachyos.md) | offen |
| 10 | [Dual-Streaming](plans/10-dual-streaming.md) | offen |
| 11 | [Rückbau auf V4](plans/11-strip-to-v4.md) | offen, **zuletzt** |

Referenzdokumente ohne Nummer:
[Paper](plans/paper-deepseek-v4.md) ·
[llama.cpp](plans/llamacpp-deepseek-v4.md) ·
[DeepSpec/DSpark](plans/deepspec-dspark.md)

**Reihenfolge.** Die Nummern sind keine strikte Kette, aber es gibt harte
Abhängigkeiten:

- **09 Commit 1** (`omp_tune.h`) direkt nach 01 — drei Zeilen, größter billiger Gewinn
- **02 vor 05** — der Flash-Kernel ist die Vorlage für den CUDA-Kernel
- **03 vor 04** — Turbo hängt am Codec-Interface
- **10 vor 11** — 11 löscht den Code, den 10 als Vorlage braucht
- **11 zuletzt**, immer

Bei knapper Zeit: `01 → 09.1 → 03 → 06` hat den meisten Ertrag pro Aufwand.

## Branch- und PR-Regeln

**Ein Plan, ein Branch, ein PR.** Keine Sammel-PRs über mehrere Phasen.

```
planning              ← die Pläne selbst und diese Datei
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
```

- Branch von `main`, außer der Plan hängt an einem anderen — dann von dessen
  Branch, und der PR nennt die Abhängigkeit in der ersten Zeile.
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

### 3. Die drei Attention-Kopien

`c/deepseek_v4.c` ist ein Amalgam. Der Text von `deepseek_v4_attention.c` steht
**byte-identisch dreimal** darin (Units `ATTENTION`, `ATTENTION_BATCH`,
`ATTENTION_TRANSACTION`). Der Generator `_amalgamate_v4.py` ist **nicht** im Repo —
die committete Datei ist Source of Truth.

Jede Änderung dort muss in alle drei Kopien, **plus** in den Batch-Pfad
`coli_v4_attention_window_batch_ref`, der dieselbe Logik nochmal enthält.
`plans/02-flash-attention.md` führt einen Test ein, der die Gleichheit erzwingt.
Ist der noch nicht da, prüfe von Hand mit `diff`.

### 4. Tests sind Teil des Commits

Eine Make-Regel `tests/test_x$(EXE):` in `c/Makefile` genügt — `TEST_RULES`
([c/Makefile:360](c/Makefile)) leitet die Gates daraus ab. Keine zentrale Liste.

Vor jedem PR:

```bash
make -C c test && make -C c check
```

```bash
make -C c deepseek-v4-tiny-check
```

Das Tiny-Fixture muss mit den Defaults **token-identisch** bleiben. Kippt es, ist
das ein Befund und gehört in den PR-Text — nicht weggedrückt.

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
  Laden ([c/deepseek_v4.c:488](c/deepseek_v4.c)). Die Scales bleiben 128×128. Für
  GPU-Tensoren einfach überspringen.
- **Der Indexer-Cache ist nicht gedeckelt.** Der `capacity > 128`-Deckel gilt nur
  der Erstallokation; `coli_v4_indexer_step` verdoppelt danach unbegrenzt.
- **`n_win = 128`**, nicht ~2048. Der Fensterring ist speichermäßig irrelevant,
  bandbreitenmäßig aber in jedem Layer präsent.
- **HCA-Layer haben keine Sparse Attention** — sie lesen den kompletten
  komprimierten Cache und dominieren die Bandbreite bei langem Kontext.
- **DFlash ≠ DeepSeek-V4-Flash.** In llama.cpp ist `dflash` eine Drafter-Arch;
  das 284B-Modell heißt dort `deepseek4`.
- **Kein separater DSpark-Download.** Der Drafter liegt im Hauptcheckpoint unter
  `mtp.<stage>.`. Wer nach einem eigenen DSpark-Repo sucht, sucht falsch.
- **MTP-Tiefe 1, aber drei DSpark-Stufen.** Das Paper nennt
  `num_nextn_predict_layers = 1`; der Drafter fährt trotzdem drei Stufen
  (`V4_DSPARK_STAGES 3`, geprobt werden `mtp.0` bis `mtp.2`). Kein Widerspruch —
  MTP-Tiefe und Drafter-Stufen sind verschiedene Dinge.
