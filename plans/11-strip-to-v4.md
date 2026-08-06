# 11 — Rückbau auf DeepSeek V4 Flash

Voraussetzung: **alle vorherigen Pläne**, insbesondere [10-dual-streaming.md](10-dual-streaming.md)

*Commits:*
1. `chore: audit the V4 dependency closure`
2. `chore: drop the GLM, Inkling, Kimi K3 and OLMoE engines`
3. `chore: drop the backends and helpers V4 does not use`
4. `docs: the fork is a DeepSeek V4 Flash engine`

## Ziel

Nur noch **DeepSeek-V4-Flash-0731** plus den **DSpark**-Drafter. Alles andere ist
in diesem Fork toter Code.

Der Nutzen ist nicht Speicherplatz, sondern Aufmerksamkeit: `c/colibri.c` allein
sind ~8000 Zeilen, die bei jeder Suche, jedem `grep` und jedem Refactor mitlaufen.
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
| `c/colibri.c` | ~8000 | GLM-5.2. **Erst nach Plan 10.** |
| `c/inkling.c` | | |
| `c/kimi_k3.c` | | |
| `c/olmoe.c` | | |

Mit ihnen: die zugehörigen Make-Ziele (`colibri`, `glm`, `olmoe`, `kimi_k3`), ihre
Tests unter `c/tests/`, ihre Tools unter `c/tools/` (`convert_glm*`,
`convert_olmoe*`, `convert_inkling*`, `download_glm52.py`, `eval_glm.py`,
`k3_*.py`, `make_glm_bench_model.py`, `expert_atlas/`) und ihre Fixtures.

**`c/Makefile` wird dabei zum Hauptrisiko.** Er trägt die Toolchain-Erkennung, die
`TEST_RULES`-Ableitung ([:360](../c/Makefile)) und die CUDA/HIP/Metal/Vulkan-Logik.
`Makefile.deepseek-v4` ist eigenständig, braucht aber Teile davon (siehe Plan 05,
`NVCC`/`CUDA_HOME`). Der Rückbau sollte `c/Makefile` **auf den V4-Pfad eindampfen**,
nicht ersetzen — sonst verliert man die Gate-Ableitung und die
Plattform-Fallunterscheidungen, die dort über Jahre gewachsen sind.

## Commit 3 — Backends und Helfer

Kandidaten, jeweils **erst gegen die Hülle aus Commit 1 prüfen**:

| Datei | Vermutlich | Warum unsicher |
|---|---|---|
| `backend_metal.{h,mm}`, `backend_vulkan.{c,h}`, `shaders/` | löschen | V4 ist Linux/CUDA-only |
| `backend_cuda_ink.{cu,h}` | löschen | Inkling-spezifisch; **erst nach Plan 05**, dient als Muster |
| `backend_cuda.{cu,h}` | **behalten, ausdünnen** | Plan 06 braucht `fmt=8` und die `pipe_*`-Primitive; MLA-Kernel für GLM können weg |
| `backend_loader.c` | behalten | Windows-DLL-Pfad; nur relevant, wenn Windows Ziel bleibt |
| `kv_persist.h` | löschen | Nutzt `Model`/`KVState`/`Cfg` aus colibri.c |
| `uring.h` | **behalten** | Nur von colibri.c genutzt, aber der beste Kandidat für V4-Storage (siehe Plan 09) |
| `rans.h`, `fse_coli.h`, `cfse_pack.c`, `iq3_*` | prüfen | Entropie-Tier; V4 nutzt fp4/fp8 nativ |
| `grammar.h`, `schema_gbnf.h`, `sample.h` | prüfen | Hängt am Serve-Pfad von V4 |
| `telemetry.h`, `route_trace.h`, `abl.h`, `tier.h`, `decode_batch.h` | prüfen | Teilweise von V4 genutzt |

**`uring.h` ausdrücklich nicht löschen.** Es ist heute nur von colibri.c benutzt und
sähe nach totem Code aus — aber Plan 09 nennt io_uring als den plausibel größten
Storage-Gewinn nach den VRAM-Phasen, und bei ~20 % Residenz ist Storage die
dominante Kostenstelle. Löschen hieße, sich diese Option zu nehmen. Wer es
trotzdem tut, sollte es bewusst tun.

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
