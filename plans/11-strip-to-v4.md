# 11 — Rückbau auf DeepSeek V4 Flash

Voraussetzung: **alle vorherigen Pläne**, insbesondere [10-dual-streaming.md](10-dual-streaming.md)

## Ergebnis

Die Löschcommits sind in PR #18. Der Baum enthält nur noch die
Linux-x86-64-V4-Engine, den CUDA-Pfad, den V4-Mirror-Reader und die für
`coli chat`, `coli serve` und `coli web` benötigten Teile. Die anderen
Modellmotoren, GPU-/Plattformbackends, Konverter, Fixtures und ihre Tests
sind entfernt; die transitive V4-Hülle bleibt als
`plans/v4-dependency-closure.txt` geprüft.

**Nicht Teil dieses Rückbaus, weiterhin offen in #18 (siehe PR-Checkliste):**
`c/coli` dispatcht immer noch auf die gelöschten Architekturen (`glm`,
`olmoe`, `inkling`, `kimi_k3`) und die Modul-Docstring/Banner nennen weiter
GLM-5.2 — das ist bewusst [Plan 13](13-frontend-v4.md)s Aufgabe, nicht
dieses Rückbaus (siehe "13 vor 11" oben). Die Statuszeile in AGENTS.md
bleibt deshalb bis 13 gelandet ist ohne "fertig".

Eine Review-Runde der Löschcommits (`/code-review 18`) fand, dass der
Löschvorgang selbst mehrere Regressionen eingeführt hatte, die inzwischen
im selben Branch behoben sind: `c/Makefile`s `test`-Ziel verlor kommentarlos
alle 42 C-Test-Gates (nur noch 4 Python-Module liefen); `deepseek-v4-oracle`
verlor `MEMORY_GB`/`ORACLE_*`; `install`/`uninstall` verloren das
Web-Dashboard bzw. räumten nicht vollständig auf; `flake.nix`,
`docker/Dockerfile.slim` und die Root-`Makefile` riefen gelöschte Ziele auf;
`coli bench`/`coli convert` stürzten ab (ihre Werkzeuge — `eval_glm.py`,
`convert_fp8_to_int4.py` — sind GLM-spezifisch und gelöscht); und
`tools/repack_fp8_passthrough.py` plus zugehörige Tests waren bereits vor
diesem Rückbau tote GLM-fmt=8-Werkzeuge, die jetzt mitentfernt sind. Details
siehe die Commits nach der Review auf diesem Branch.

Die öffentliche Dokumentation wurde auf den einzigen Checkpoint, den DSpark im
selben Checkpoint, die i5-13400F/32-GiB/RTX-4070-Zielmaschine und die verbleibenden
V4-Bedienwege reduziert. Historische Upstream-Release- und Modellseiten wurden
gelöscht statt als scheinbar gültige Anleitung stehen zu bleiben. Die
Full-Checkpoint- und Langlaufmessungen gehören weiterhin bewusst in
[Plan 15](15-final-validation-and-tuning.md), nicht in diesen Rückbau.

Abgenommen auf dem Branch mit `make -C c test && make -C c check` (42
C-Test-Gates plus 335 Python-Tests) sowie der gepinnten Tiny-Neugenerierung und
`make -C c PYTHON="$PWD/.venv-v4-tiny/bin/python" deepseek-v4-tiny-check`:
Target-Orakel und Prefix-Reuse sind token-identisch. Die Tiny-Fixture enthält
keine Drafter-Tensoren; sie ist deshalb kein Ersatz für die in Plan 15 offene
Full-Checkpoint-DSpark-Abnahme.

*Commits:*
1. `chore: audit the V4 dependency closure`
2. `chore: drop the GLM, Inkling, Kimi K3 and OLMoE engines`
3. `chore: drop the backends and helpers V4 does not use`
4. `docs: the fork is a DeepSeek V4 Flash engine`

## Ziel

Nur noch **DeepSeek-V4-Flash-0731** plus den **DSpark**-Drafter. Alles andere ist
in diesem Fork toter Code.

Der Nutzen ist nicht Speicherplatz, sondern Aufmerksamkeit: `c/colibri.c` allein
sind ~9500 Zeilen, die bei jeder Suche, jedem `grep` und jedem Refactor mitlaufen.
Nach dem Rückbau ist der Baum das, was er sein soll — eine V4-Engine.

## Reihenfolge — bitte ernst nehmen

**Dieser Plan kommt zuletzt.** Zwei der vorherigen Pläne holen sich Referenzcode
aus Motoren, die hier gelöscht werden:

| Plan | Braucht aus dem Löschbereich |
|---|---|
| [09](09-arch-cachyos.md) | `omp_tune.h` — von `olmoe.c`/`kimi_k3.c` verdrahtet, als Vorlage |
| [10](10-dual-streaming.md) | **Die komplette Mirror-Maschinerie aus `colibri.c`** |
| [05](05-cuda-attention.md)/[06](06-dense-vram.md) | `backend_cuda.cu` fmt=8, `backend_cuda_ink.*` als Muster |

Wird `colibri.c` vor Plan 10 gelöscht, ist `expert_route`, `mir_pread_striped`,
`mirror_probe_bw` und die Cut-Berechnung weg — ~200 Zeilen sorgfältig
kommentierter, gemessener Code, den niemand gern zweimal schreibt.

## Commit 1 — Die Abhängigkeitshülle bestimmen

**Nicht raten, ermitteln.** Der Baum ist verwoben genug, dass eine Vermutung teuer
wird.

Bekannt ist der direkte Include-Kegel von V4
([c/deepseek_v4_internal.h](../c/deepseek_v4_internal.h)):

```
deepseek_v4.h  tensor.h  expert_store.h  st.h  tok.h  kv_prefix.h
native_quant.h  native_quant_batch.h  native_quant_dual.h  native_quant_fp4_rows16.h
```

plus aus `deepseek_v4.c` selbst: `quant.h`, `json.h`, `compat.h`, `tok_unicode*.h`
und `deepseek_v4_dspark.inc`.

Dazu kommt, was die Pläne **hinzufügen**: `omp_tune.h` (09), `mirror.h` (10),
`v4_kv_codec.h` + `turbo_quant.h` (03/04), `backend_cuda_v4.*` (05) und der
benutzte Teil von `backend_cuda.{cu,h}` (05/06).

Der Audit-Commit erzeugt die Liste mechanisch:

```bash
# transitive Hülle der Header, die die V4-Units tatsächlich ziehen
gcc -MM -I c -DCOLI_V4_UNIT_RUNTIME c/deepseek_v4.c 2>/dev/null | tr ' \\' '\n\n' | grep '\.h$' | sort -u
```

für jede Unit aus `V4_TARGET_UNITS` ([c/Makefile.deepseek-v4.units](../c/Makefile.deepseek-v4.units)),
vereinigt. Ergebnis als `plans/v4-dependency-closure.txt` committen — daraus folgt
die Löschliste, und der Nachweis bleibt im Baum.

## Commit 2 — Die anderen Motoren

| Datei | Zeilen | Anmerkung |
|---|---|---|
| `c/colibri.c` | ~9500 | GLM-5.2. **Erst nach Plan 10.** |
| `c/inkling.c` | | |
| `c/kimi_k3.c` | | |
| `c/olmoe.c` | | |

Mit ihnen: die zugehörigen Make-Ziele (`colibri`, `glm`, `olmoe`, `kimi_k3`), ihre
Tests unter `c/tests/`, ihre Tools unter `c/tools/` (`convert_glm*`,
`convert_olmoe*`, `convert_inkling*`, `download_glm52.py`, `eval_glm.py`,
`k3_*.py`, `make_glm_bench_model.py`, `expert_atlas/`) und ihre Fixtures.

**`c/Makefile` wird dabei zum Hauptrisiko.** Er trägt die Toolchain-Erkennung, die
`TEST_RULES`-Ableitung ([:359](../c/Makefile)) und die CUDA/HIP/Metal/Vulkan-Logik.
`Makefile.deepseek-v4` ist eigenständig, braucht aber Teile davon (siehe Plan 05,
`NVCC`/`CUDA_HOME`). Der Rückbau sollte `c/Makefile` **auf den V4-Pfad eindampfen**,
nicht ersetzen — sonst verliert man die Gate-Ableitung und die
Plattform-Fallunterscheidungen, die dort über Jahre gewachsen sind.

## Commit 2b — Windows

Zielplattform ist **Linux, x86-64**. Alles Windows-Spezifische ist toter Code:

| Weg | Wo |
|---|---|
| `c/build_cuda.bat`, `c/warmup.ps1` | Windows-Build- und Warmup-Skripte |
| `c/tests/audit_win_shims.c` | testet nur die Shims |
| `c/backend_loader.c` | die DLL-Ladepfad-Konstruktion (`LoadLibrary`/`GetProcAddress`) |
| `_WIN32`-Zweige | `compat.h`, `st.h`, `omp_tune.h`, `telemetry.h`, `iobench.c`, `deepseek_v4.c`, `backend_cuda.h` |
| `Makefile`-Zweige | `IS_WIN`, `EXE`, MSYS2-Pfade in `Makefile.deepseek-v4` |

**`backend_loader.c` fällt hier, nicht in Commit 3.** Es ist ausschließlich der
Windows-DLL-Ladepfad; sobald Windows kein Ziel mehr ist, hat es keinen Nutzer.
Eine frühere Fassung dieses Plans führte es zusätzlich in der Kandidatentabelle
von Commit 3 mit „behalten" — das war ein Widerspruch. Wer 2b **nicht** ausführt
(Windows bleibt Ziel), lässt es stehen; dann entfällt aber der ganze Commit.

`compat.h` ist dabei die heikelste Datei: sie bildet `posix_memalign`,
`compat_aligned_free`, `compat_open_direct` und einiges mehr auf Windows ab.
Nach dem Rückbau bleiben die POSIX-Definitionen — aber **`compat_aligned_free`
muss bleiben**, auch wenn es unter POSIX nur `free()` ist. Der Kommentar bei
[c/deepseek_v4.c:6583](../c/deepseek_v4.c) erinnert daran, dass genau diese
Asymmetrie schon einmal einen Bug produziert hat (`63a2c8d`). Ein
Suchen-und-Ersetzen von `compat_aligned_free` → `free` ist die naheliegende und
falsche Vereinfachung.

`Makefile.deepseek-v4` verliert die `IS_WIN`-Verzweigung und den `-static`-Link,
`$(EXE)` wird leer. Auch die `COLI_V4_SUPPORTED`-Gates in `c/Makefile:340` können
auf Linux-x86-64 zusammenschrumpfen — aber erst prüfen, ob `TEST_BINS` daran hängt.

## Commit 3 — Backends und Helfer

Kandidaten, jeweils **erst gegen die Hülle aus Commit 1 prüfen**:

| Datei | Vermutlich | Warum unsicher |
|---|---|---|
| `backend_metal.{h,mm}`, `backend_vulkan.{c,h}`, `shaders/` | löschen | V4 ist Linux/CUDA-only |
| `backend_cuda_ink.{cu,h}` | löschen | Inkling-spezifisch; **erst nach Plan 05**, dient als Muster |
| `backend_cuda.{cu,h}` | **behalten, ausdünnen** | Plan 06 braucht `fmt=8` und die `pipe_*`-Primitive; MLA-Kernel für GLM können weg |
| `backend_loader.c` | — | **schon in Commit 2b weg**, siehe dort. Hier nicht nochmal bewerten |
| `kv_persist.h` | löschen | Nutzt `Model`/`KVState`/`Cfg` aus colibri.c |
| `uring.h` | **behalten** | Nur von colibri.c genutzt, aber der beste Kandidat für V4-Storage (siehe Plan 09) |
| `rans.h`, `fse_coli.h`, `cfse_pack.c`, `iq3_*` | prüfen | Entropie-Tier; V4 nutzt fp4/fp8 nativ |
| `grammar.h`, `schema_gbnf.h`, `sample.h` | prüfen | Hängt am Serve-Pfad von V4 |
| `telemetry.h`, `route_trace.h`, `abl.h`, `tier.h`, `decode_batch.h` | prüfen | Teilweise von V4 genutzt |

**`uring.h` ausdrücklich nicht löschen.** Es ist heute nur von colibri.c benutzt und
sähe nach totem Code aus — aber Plan 09 nennt io_uring als den plausibel größten
Storage-Gewinn nach den VRAM-Phasen, und bei rund 17 % Residenz im 32k-Profil
(rund 8 % bei 128k) ist Storage die dominante Kostenstelle. Löschen hieße, sich
diese Option zu nehmen. Wer es
trotzdem tut, sollte es bewusst tun.

## Was ausdrücklich bleibt

Nicht löschen, auch wenn es nach Beiwerk aussieht:

| Bleibt | Warum |
|---|---|
| `c/coli` | `chat`, `serve`, `web` — die tägliche Bedienung. Wird in [13](13-frontend-v4.md) auf V4 verschlankt, nicht entfernt. |
| `c/openai_server.py` | trägt `coli serve` |
| `web/` | das Dashboard |
| `colibri/cli.py` | pip-Einstiegspunkt |
| `desktop/` | Tauri-Hülle über der WebUI |
| `c/uring.h` | siehe unten |
| `compat.h` (POSIX-Teil) | `compat_aligned_free` ist keine Attrappe |

`desktop/` (Tauri) **bleibt** und erbt die WebUI-Änderungen aus [13](13-frontend-v4.md).
`site/` ist Ermessenssache.

## Commit 4 — Dokumentation

- `README.md` (und die drei Übersetzungen) beschreiben fünf Modellfamilien. Der
  Fork ist eine V4-Engine — das gehört an den Anfang, nicht in eine Fußnote.
- `docs/` verliert `inkling.md`, `kimi_k3.md`, `metal.md`, `vulkan.md`,
  `int4-rans256-g0.md`, `METAL-M5MAX-PERF-REPORT.md`, GLM-Experimente.
- `docs/ENVIRONMENT.md` wird auf die verbleibenden Knöpfe eingedampft — inklusive
  der neuen aus diesem Branch (Tabelle in [00-reference.md](00-reference.md)).
- `CHANGELOG.md`: ein Eintrag, der sagt, dass dieser Fork bewusst divergiert.

## Was der Rückbau kostet

Ehrlich benennen, damit die Entscheidung bewusst bleibt:

- **Kein Upstream-Merge mehr.** Nach dieser Operation ist ein `git pull` von
  `JustVugg/colibri` ein Konfliktfeld. Der Fork wird endgültig eigenständig.
  Wenn du je Upstream-Fixes für V4 übernehmen willst, wird das Handarbeit.
- **Die gelöschten Motoren sind die Vergleichsbasis.** Mehrere Optimierungen
  dieses Branches (`omp_tune.h`, Mirror, `backend_cuda_ink` als API-Muster) kamen
  daher, dass es *andere* Implementierungen zum Abgleich gab. Danach steht V4
  allein.
- **Tests verlieren Abdeckung.** Viele Gates in `c/tests/` prüfen gemeinsame
  Header über die anderen Motoren. `st.h`, `quant.h`, `tok.h` behalten dann nur
  noch V4 als Nutzer — Fehler dort fallen später auf.

Deshalb: **Commit 1 (Audit) lohnt sich auch ohne Commit 2–4.** Wenn der Rückbau
nach dem Audit doch zu teuer aussieht, ist die Abhängigkeitshülle trotzdem ein
nützliches Artefakt.

## Tests

- Nach jedem Löschcommit: `make -C c deepseek-v4` und
  `make -C c deepseek-v4-tiny-check` grün.
- `make -C c test` läuft mit der reduzierten Gate-Menge; die verbleibenden Gates
  müssen alle noch existieren (`TEST_RULES` leitet sie aus `c/Makefile` ab — eine
  gelöschte Regel verschwindet still).
- `c/tests/test_v4_dependency_closure.py` — die Liste aus Commit 1 gegen den
  tatsächlichen Include-Kegel; schlägt an, wenn jemand später einen gelöschten
  Header wieder einführt oder ein neuer ungeprüft dazukommt.
- Tokenfolge am Tiny-Fixture vor und nach dem Rückbau identisch. Der Rückbau darf
  **nichts** am Verhalten ändern; tut er es, war etwas nicht tot.

## Abnahme

- `deepseek_v4` baut, `deepseek-v4-tiny-check` grün, Tokens identisch zu vorher.
- `c/` enthält keinen Verweis mehr auf GLM, Inkling, Kimi, OLMoE
  (`grep -ril "glm\|inkling\|kimi\|olmoe" c/` ist leer bis auf Changelog/History).
- Die Abhängigkeitshülle aus Commit 1 deckt jede verbleibende Datei ab — keine
  Waisen.
- README beschreibt in Satz 1, was der Fork ist.

## Risiken

- **Reihenfolge.** Der größte Fehler wäre, mit diesem Plan anzufangen, weil er
  befriedigend aussieht. Er ist der einzige Plan, der ausschließlich **entfernt**
  und dessen Nutzen indirekt ist.
- **Der Audit kann überraschen.** Die Amalgamation zieht Header über
  `#include`-Ketten, die man beim Lesen nicht sieht — `deepseek_v4_internal.h`
  wird an 15 Stellen erneut inkludiert. Deshalb `gcc -MM` statt Augenmaß.
- **Windows.** `Makefile.deepseek-v4` unterstützt MSYS2, `backend_loader.c` ist der
  Windows-DLL-Pfad. Wenn Windows kein Ziel mehr ist, sagt man das explizit und
  löscht beides — halb entfernt ist schlimmer als beides.
