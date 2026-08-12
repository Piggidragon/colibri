# 09 — Plattform: Arch / CachyOS

Voraussetzung: [00-reference.md](00-reference.md)

*Commits:*
1. `perf: size the V4 OpenMP team to physical cores` ← **früh ziehen**
2. `perf: back the V4 expert slabs with transparent hugepages`
3. `build: find the CUDA toolkit at Arch's /opt/cuda`
4. `docs: Arch/CachyOS notes in the tuning profile`

## Ergebnis Commit 1 (PR #11)

Commit 1 ist gebaut und grün; der damalige Rest ist unten als Ergebnis
dokumentiert.

`coli_omp_tune_threads("deepseek_v4")` hängt an allen drei Einsprungpunkten des
`GENERATE_STATS`-Units: produktives `main`, `v4_serve_main` (an dem `SERVE=1` am
ersten vorbeigeht) und das Legacy-`main` des First-Token-Tools. Der Include steht
**vor** `#define main coli_v4_first_token_legacy_main` — dahinter würde die
Umbenennung durch den Header laufen. Auf der Zielmaschine meldet er

```
[OMP] deepseek_v4: 10 physical-core threads instead of 16 logical CPUs
```

`OMP_NUM_THREADS` und `COLI_NO_OMP_TUNE` unterdrücken ihn beide, wie der Header
zusichert. Damit ist die Hybrid-Blindheit aus dem Abschnitt unten **gemessen
bestätigt**: 6 P + 4 E ergeben 10, nicht 6.

### Die Messung ist ein Nullergebnis

Gemessen am 2026-08-10, `bench_v4.py --profile short --cache-state warm
--ctx 32768 --ram-gb 24 --scratch-mb 128 --max-tokens 24`, verschränkt
16/10/16/10 gegen Drift:

| Threads | prefill | decode | TTFT |
|---|---:|---:|---:|
| 16 | 0.62 tok/s | 0.14 tok/s | 697.65 s |
| 10 | 0.65 | 0.14 | 663.66 |
| 16 | 0.67 | 0.14 | 646.50 |
| 10 | 0.63 | 0.11 | 689.55 |

Mittel: 16 → prefill 0.645, TTFT 672.1 s; 10 → 0.640 und 676.6 s. Das sind
−0.8 % und +0.7 % — **und die Streuung innerhalb einer Konfiguration ist zehnmal
größer als der Unterschied zwischen ihnen** (TTFT bei 16 Threads 646.50–697.65 s
= 7.9 %). Der Decode-Ausreißer 0.11 gegen dreimal 0.14 ist bei n=2 nicht
interpretierbar; die Daten schließen weder einen kleinen Gewinn noch eine kleine
Regression aus.

**Der Grund steht in derselben Zeile:** `disk=1.69GB/token`,
`expert_hit=50.9%`. Bei 0.14 tok/s dauert ein Token ~7 s, und die gehen für
1.69 GB Expertengewichte von der NVMe drauf. Die Teamgröße wirkt auf Compute,
und Compute ist in diesem Regime nicht der Engpass. Die +2.3× aus `#718` wurden
auf einem Zen3-5950X gemessen — SMT-lastig und rechengebunden, also das
Gegenteil. Sie waren nie eine Zusage, und hier tragen sie nicht.

**Commit 1 wird trotzdem übernommen**, weil er risikoarm ist (rät nie,
respektiert beide Overrides) und V4 mit den drei anderen Motoren gleichzieht —
nicht, weil ein Gewinn belegt wäre. Er ist es nicht.

### Zwei Funde für die Folgecommits

- **`bench_v4.py` maskiert dieses Tuning.** Es setzt `OMP_NUM_THREADS`
  bedingungslos auf `--omp-threads` (Default `os.cpu_count()`), und genau darauf
  steigt `coli_omp_tune_threads` aus. Der Harness aus Plan 01 hätte den neuen
  Pfad nie gemessen; das A/B oben lief deshalb über `--omp-threads 16` gegen
  `--omp-threads 10`, was dasselbe bewirkt. Commit 4 sollte den Harness einen
  „nicht setzen"-Modus bekommen lassen, sonst bleibt die Default-Konfiguration
  unmessbar.
- **`--ram-gb 28` kippt im CPU-Build in den OOM-Killer** (`engine exited with
  -9`, Planner projizierte 27.87 GiB). Das dokumentierte 32-GiB-Profil aus
  `docs/deepseek-v4-tuning-32gb.md` setzt die 5.456 GiB Dense auf der Karte
  voraus; ohne `CUDA=1 V4_VRAM=1` bleiben sie im RAM. Deshalb die 24 GiB oben —
  und deshalb liegen die absoluten Zahlen unter Phase 07 (15 statt 30 Slots).

Die hardwaregebundene Abschlussabnahme (16 / 10 / nur-6-P-Cores, THP-Nachweis
und Durchsatz) liegt in [Plan 15](15-final-validation-and-tuning.md). Die
Tabelle mit den P-Äquivalenten unten bleibt eine Abschätzung.

## Vorbemerkung zur Messbarkeit

Die Codefunde unten sind maschinenunabhängig und verifiziert. Die *Systemwerte*
(THP-Modus, zram, CPU, CUDA-Version) konnte ich **nicht** auf der Zielmaschine
erheben — die Analyse lief in einem Ubuntu-Container mit i5-10310U und ohne CUDA.
Wo unten Annahmen über CachyOS stehen, sind sie als solche markiert und mit dem
Befehl versehen, der sie prüft.

## Commit 1 — `omp_tune.h`, der übersehene Faktor

**Das ist der größte billige Gewinn im ganzen Vorhaben, und er hat nichts mit
Speicher zu tun.**

`c/omp_tune.h` dimensioniert das OpenMP-Team auf **physische** Kerne statt auf
logische. Der Header dokumentiert die Messung:

> `dimensionamento  OMP_NUM_THREADS = core fisici (niente SMT)`
> `#718: +2.3x su Zen3 (5950X, 16C/32T) solo cambiando il numero di thread.`

Verdrahtet ist er in drei Engines:

| Engine | Aufruf |
|---|---|
| `olmoe.c` | [:1105](../c/olmoe.c) `coli_omp_tune_threads("olmoe")` |
| `kimi_k3.c` | [:1750](../c/kimi_k3.c) `coli_omp_tune_threads("kimi_k3")` |
| `colibri.c` | nach dem optionalen Hot-Team-Re-exec |

**`deepseek_v4.c` nicht.** Verifiziert:

```bash
grep -rln "omp_tune.h" c/          # → omp_tune.h, Makefile, colibri.c, kimi_k3.c, olmoe.c, tests/
grep -n "omp_set_num_threads\|OMP_NUM_THREADS" c/deepseek_v4.c   # → leer
```

Dabei hat V4 **18** `#pragma omp parallel for`-Stellen
(`grep -n '#pragma omp' c/deepseek_v4.c`), darunter die heißen: `head_argmax`,
`head_argmax_batch`, `dspark_markov_argmax`, die fp8-Kernel
`coli_fp8_matvec_ref`, `coli_fp8_dual_matvec_ref` und `coli_fp8_matmul_batch_ref`
sowie das Expert-Warmup in `hot_prewarm_history`.

**Eine heiße Schleife fehlt in dieser Liste**, und zwar nicht zufällig: die
Bewertungsschleife in `coli_v4_indexer_step` ([:3694](../c/deepseek_v4.c)) hat
**kein** Pragma, kostet aber bei 128k rund 5.6 G MAC pro Token (Herleitung im
Indexer-Abschnitt von [00-reference.md](00-reference.md)). Sie zu parallelisieren
gehört zu [12-expert-cache-policy.md](12-expert-cache-policy.md) Commit 2, nicht
hierher — aber sie ist der Grund, warum das Team-Sizing aus Commit 1 dort mehr
bringt als anderswo.

Ohne Sizing nimmt libgomp den Default: **alle logischen Kerne inklusive SMT.** Auf
einem Ryzen mit SMT ist das genau die Konfiguration, gegen die #718 gemessen wurde.

### Änderung

Drei Zeilen. In `c/deepseek_v4.c`:

```c
#include <omp.h>        /* omp_set_num_threads/omp_get_max_threads für omp_tune.h */
#include "omp_tune.h"
```

und als erste Anweisung in beiden `main`-Funktionen — der produktiven
([:8512](../c/deepseek_v4.c)) und der Stats-Variante
([:6942](../c/deepseek_v4.c)) — sowie in `v4_serve_main` ([:8450](../c/deepseek_v4.c)),
falls `SERVE=1` daran vorbeigeht:

```c
coli_omp_tune_threads("deepseek_v4");   /* Team auf physische Kerne: siehe omp_tune.h */
```

`c/Makefile.deepseek-v4` bekommt `omp_tune.h` in die Header-Abhängigkeiten,
analog zu `c/Makefile:685` für olmoe.

### Aber: der i5-13400F ist eine Hybrid-CPU

**Und genau dafür ist die Linux-Erkennung blind.**

Der i5-13400F hat **6 P-Cores (Golden Cove, bis 4.6 GHz) + 4 E-Cores (Gracemont,
bis 3.3 GHz)**, 16 logische Threads. Die Linux-Zählung in `coli_physical_cores`
([c/omp_tune.h:111](../c/omp_tune.h)) zählt eindeutige `thread_siblings_list`-Einträge:

```
6 P-Cores × je 2 Siblings  → 6 eindeutige Listen
4 E-Cores × je 1 Sibling   → 4 eindeutige Listen
                             = 10
```

Also ein Team aus **10 Threads über zwei ungleiche Kernklassen**. Und V4 nutzt in
den heißen Schleifen `schedule(static)` — `head_argmax`
([c/deepseek_v4.c:8130](../c/deepseek_v4.c)), die fp8-Matmuls
([:11607](../c/deepseek_v4.c), [:12056](../c/deepseek_v4.c),
[:10451](../c/deepseek_v4.c)). Statische Aufteilung über ungleiche Kerne heißt:
**die E-Cores geben das Tempo vor.**

Der Header kennt das Problem — und löst es nur für macOS
([c/omp_tune.h:98](../c/omp_tune.h)):

> *`hw.perflevel0.logicalcpu` = die PERFORMANCE-Kerne. Auf Apple Silicon zählt
> `hw.physicalcpu` alle, E-Cores eingeschlossen (10 auf einem M1 Max), und mit
> einer Barriere für Matmul gibt der langsamste Thread den Takt an: die E-Cores
> bremsen das Team, statt ihm zu helfen (#707, **−4.2 % Decode**).*

Dieselbe Messung, dieselbe Topologie — nur auf Intel-Hybrid nie umgesetzt, weil es
die zur Entstehungszeit kaum gab.

Grobe Abschätzung mit E ≈ 0.45 × P (Gracemont hat 2×128-bit FMA gegen 2×256-bit
bei Golden Cove, plus niedrigeren Takt):

| Konfiguration | effektive P-Äquivalente |
|---|---|
| `schedule(static)`, 10 Threads | ~4.5 (alle warten auf die E-Cores) |
| nur P-Cores, 6 Threads | ~6.0 |
| `schedule(dynamic/guided)`, 10 Threads | ~7.8 |

**Änderung, als Commit 1b:** Linux-Erkennung um Hybrid-Bewusstsein erweitern. Moderne
Kernel exportieren die Kernklassen direkt:

```
/sys/devices/system/cpu/types/intel_core_*/cpumap    → P-Cores
/sys/devices/system/cpu/types/intel_atom_*/cpumap    → E-Cores
```

Existiert `intel_core_*`, sind das die Performance-Kerne — genau die Entsprechung
zu `hw.perflevel0.logicalcpu`. Fehlt das Verzeichnis (ältere Kernel), bleibt es
bei der heutigen Zählung; die Regel des Headers gilt weiter: **nicht raten, dann
lieber den OpenMP-Default lassen.**

Knopf dafür:

```
V4_OMP_CORES=perf|all|<n>     Default: all; perf ist der zu messende Opt-in
```

**Beides messen**, bevor eines Default wird — die Tabelle oben ist eine
Abschätzung, keine Messung. Der Benchmark-Harness aus
[01](01-measure-and-ram-budget.md) ist genau dafür da. Möglich ist auch, dass
`schedule(guided)` über alle 10 besser abschneidet als 6 P-Cores; dann ist die
richtige Änderung eine andere (Scheduling statt Teamgröße), und das ist ein
Ergebnis, kein Rückschlag.

**Nebeneffekt, der zum I/O-Profil passt:** Bleibt das OpenMP-Team auf den P-Cores,
stehen die E-Cores dem Expert-Loader-Pool zur Verfügung. Der Header warnt
ausdrücklich davor, dass ein leerlaufendes Rechenteam den I/O-Threads die Kerne
klaut (#707, −2.2×) — auf einer Hybrid-CPU lässt sich beides sauber trennen.
`GOMP_CPU_AFFINITY` bzw. `OMP_PLACES` sind die Werkzeuge; auch das gehört gemessen,
nicht angenommen.

### Warum das sichere Teil sicher ist

Der Header trennt bewusst zwei Dinge mit **entgegengesetztem Risikoprofil** und
implementiert nur das ungefährliche:

> `REGOLA SUI FALLIMENTI: se il conteggio dei core fisici non e' determinabile,
> NON si indovina — si lascia il default di OpenMP.`

Kein geratener Wert, kein stiller Fallback auf 1 (der Fehler aus #325).
`omp_set_num_threads` ist Laufzeit-API und wirkt sofort — anders als die
Spin-Wait-Variablen braucht es kein Re-exec.

## Dokumentationshinweis (gehört in Commit 4) — was man ausdrücklich **nicht** setzen soll

CachyOS-Setups neigen zu aggressiven Performance-Env-Variablen. Für **diese**
Arbeitslast ist das kontraproduktiv, und der Header sagt warum:

> `spin-wait  OMP_WAIT_POLICY=active, GOMP_SPINCOUNT, KMP_BLOCKTIME`
> `#707: -2.2x sul decode di un host a bassa residenza (M1 Max 32 GB, ~10% di expert residenti)`
> `Meccanismo: dove il token e' fatto di byte dal disco, una squadra che gira a
> vuoto ruba i core al pool di I/O che sta facendo il lavoro vero.`

**32 GB gegen ~147 GB Experten ist exakt dieses Regime** — ~10 % Residenz nach der
Bilanz in [00](00-reference.md), also praktisch dasselbe Regime wie die zitierte
Messung. Ein wartendes OpenMP-Team klaut hier den I/O-Threads die Kerne.

Also ins Tuning-Doc, als explizite Warnung:

```
# NICHT setzen für DeepSeek V4 auf dieser Maschine:
#   OMP_WAIT_POLICY=active   GOMP_SPINCOUNT=...   KMP_BLOCKTIME=...
```

Nach den Phasen 06/07 (Dense und Head auf der GPU) verschiebt sich das Bild etwas,
weil weniger CPU-Arbeit übrig bleibt — aber disk-bound bleibt es. Wer es messen
will, misst es; die Voreinstellung bleibt „nicht setzen".

## Commit 2 — Transparent Hugepages für die Expert-Slabs

Die Expert-Slabs werden 4-KiB-aligned angefordert
([c/deepseek_v4.c:7103](../c/deepseek_v4.c)):

```c
if (posix_memalign((void **)&slot->slab, 4096, capacity)) {
```

Bei 15–29 GiB Cache mit im Wesentlichen zufälligem Zugriff (Router wählt 6 von 256
Experten pro Layer) ist das TLB-Verhalten relevant: 29 GiB in 4-KiB-Seiten sind
~7.6 M Einträge, in 2-MiB-Seiten ~15 k.

**Annahme, die du prüfen solltest:**

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
```

Steht dort `[madvise]` (Arch-Default, und auch der Default auf meiner Testmaschine),
bekommt der Slab **null** Hugepages, weil niemand `madvise` ruft. Bei `[always]`
passiert es von selbst und dieser Commit ist wirkungslos — dann bitte nichts einbauen.

### Änderung

```c
#ifdef MADV_HUGEPAGE
    /* 2-MiB-Alignment, damit der Kernel überhaupt kollabieren kann; ohne
     * madvise passiert im THP-Modus "madvise" gar nichts. */
    size_t align = capacity >= (2u << 20) ? (2u << 20) : 4096;
    if (posix_memalign((void **)&slot->slab, align, capacity)) { ... }
    if (align > 4096) madvise(slot->slab, capacity, MADV_HUGEPAGE);
#else
    if (posix_memalign((void **)&slot->slab, 4096, capacity)) { ... }
#endif
```

`madvise` ist ein Hinweis — schlägt er fehl, ändert sich nichts. Rückgabewert
bewusst ignorieren, aber einmal beim Start loggen, ob THP aktiv ist, damit der
Effekt zurechenbar bleibt.

**Achtung Freigabe:** der Kommentar bei [:5341](../c/deepseek_v4.c) hält fest, dass
`aligned_slab` unter Windows `_aligned_malloc` bedeutet und mit
`compat_aligned_free` freigegeben werden **muss** — genau der Fehler, den Commit
`63a2c8d` schon einmal beheben musste. Das Alignment zu ändern darf daran nichts
verrücken.

Messbar über `/proc/<pid>/smaps_rollup` (`AnonHugePages`) vor und nach.

## Commit 3 — CUDA-Pfade auf Arch

Betrifft [Plan 05](05-cuda-attention.md).

**`CUDA_HOME ?= /usr/local/cuda`** ([c/Makefile:184](../c/Makefile)) — das ist der
NVIDIA-Runfile-Pfad. Arch' `cuda`-Paket installiert nach **`/opt/cuda`**. `make
... CUDA=1` scheitert dort also mit „nvcc not found", obwohl CUDA installiert ist.

Der neue CUDA-Zweig in `Makefile.deepseek-v4` soll beide finden:

```make
ifeq ($(CUDA_HOME),)
  ifneq ($(wildcard /opt/cuda/bin/nvcc),)      # Arch, CachyOS, Manjaro
    CUDA_HOME := /opt/cuda
  else ifneq ($(wildcard /usr/local/cuda/bin/nvcc),)
    CUDA_HOME := /usr/local/cuda
  endif
endif
```

Ob derselbe Fix nach `c/Makefile:184` gehört, ist eine Frage an den Upstream — er
beträfe GLM mit. Für diesen Branch reicht der V4-Zweig; ein Upstream-PR dafür wäre
klein und nützlich, aber nicht unser Thema.

### Host-Compiler

Arch/CachyOS liefern sehr aktuelle GCC-Versionen; nvcc lehnt nicht unterstützte
Host-Compiler ab. `NVCC_CCBIN` gibt es bereits ([c/Makefile:180](../c/Makefile)),
inklusive Beispiel in Zeile 179. Arch hat `gcc13`/`gcc14` als eigene Pakete genau
dafür:

```bash
make -C c deepseek-v4 CUDA=1 CUDA_HOME=/opt/cuda NVCC_CCBIN=g++-14
```

Gehört ins Tuning-Doc, sonst kostet es jeden Neubau eine Viertelstunde Suchen.

Zu prüfen auf deiner Maschine:

```bash
ls /opt/cuda/bin/nvcc && /opt/cuda/bin/nvcc --version | tail -2 && gcc --version | head -1
```

## Commit 4 — Doku

Ein Abschnitt „Arch / CachyOS" in `docs/deepseek-v4-tuning-32gb.md` (aus
[Plan 01](01-measure-and-ram-budget.md)):

- Build-Zeile mit `CUDA_HOME` und `NVCC_CCBIN`
- die Spin-Wait-Warnung aus Commit 1b
- **zram**: CachyOS aktiviert zram meist per Default. `coli_v4_os_available_memory`
  liest `MemAvailable` aus `/proc/meminfo` ([c/deepseek_v4.c:855](../c/deepseek_v4.c)).
  Unter zram/zswap ist dieser Wert optimistischer, als er für eine Arbeitslast
  taugt, die ihn tatsächlich anfassen will — komprimierbarer Swap hilft einem
  Expert-Cache nicht, er kostet CPU. Das macht das explizite `RAM_GB` aus Plan 01
  **wichtiger**, nicht optional. Prüfen mit `zramctl` und `swapon --show`.
- **`ARCH ?= native`** ist in `c/Makefile.deepseek-v4:40` (Linux-Zweig) bereits
  der Default —
  für einen Build auf der Zielmaschine ist da nichts zu holen. Nur `make check`
  baut mit `PORTABLE_ARCH`; wer Benchmarkzahlen vergleicht, muss wissen, welcher
  von beiden lief.
- **ananicy-cpp** (CachyOS-Default) kann Prozesse umpriorisieren. Für einen
  langlaufenden Inferenzprozess normalerweise harmlos, aber bei
  unerklärlichen Durchsatzschwankungen der erste Verdächtige.

## Nicht in diesem Plan: io_uring

`c/uring.h` existiert und wird **nur von `colibri.c`** benutzt
(`grep -rln "uring.h" c/*.c` → `colibri.c`). V4 liest über `pread`
([:104](../c/deepseek_v4.c), [:6886](../c/deepseek_v4.c)) mit optionalem O_DIRECT
(`COLI_V4_DIRECT`, [:116](../c/deepseek_v4.c)) und `posix_fadvise(WILLNEED)` als
Prefetch ([:198](../c/deepseek_v4.c)).

Bei ~10 % Expert-Residenz ist Storage-I/O **die** dominante Kostenstelle, und ein
aktueller Arch-Kernel bringt ausgereiftes io_uring mit. Batched Submission über
`uring.h` wäre plausibel der nächstgrößte Gewinn nach den VRAM-Phasen.

Aber: das ist ein eigenes Vorhaben in der Größenordnung von Plan 06, es berührt den
Expert-Store-Kern statt der Attention, und der Nutzen hängt an Messungen, die es
noch nicht gibt. **Erst die RAM-Phasen, dann messen, dann entscheiden.** Hier nur
als benannter Kandidat, nicht als Zusage.

## Tests

- `c/tests/test_v4_omp_tune.c` — dass `coli_omp_tune_threads` aus dem V4-Kontext
  aufrufbar ist und bei nicht bestimmbarer Kernzahl den OpenMP-Default lässt.
  `tests/test_omp_tune.c` deckt den Helfer selbst schon ab
  ([c/Makefile:943](../c/Makefile)); hier geht es nur um die Verdrahtung.
- `c/tests/test_v4_slab_align.c` — Slab-Alignment ist 2 MiB ab der Schwelle,
  4 KiB darunter; Freigabe über `compat_aligned_free` bleibt korrekt (der Bug aus
  `63a2c8d` darf nicht zurückkommen).
- Build-Test: `CUDA_HOME`-Autodetektion findet `/opt/cuda`, wenn vorhanden, und
  scheitert sonst mit der bestehenden verständlichen Meldung.

## Ergebnis Rest (ausstehender Teil nach PR #11)

Auf der Zielmaschine am 2026-08-12 geprüft: THP steht auf
`[always] madvise never`. Deshalb wurde Commit 2 bewusst **nicht** eingebaut:
der Kernel versucht Hugepages bereits für alle ausreichend großen anonymen
Slabs, und ein zusätzliches `MADV_HUGEPAGE` hätte keine Wirkung. Der Nachweis
für einen laufenden Prozess bleibt `AnonHugePages` in
`/proc/<pid>/smaps_rollup`; die Hardwaremessung und Durchsatzbewertung gehören
weiterhin zu Plan 15.

Commit 1b ergänzt `coli_v4_omp_tune_threads`: Linux liest bei vorhandener
`intel_core_*`-Topologie die P-Core-CPU-Maske und dedupliziert deren SMT-Siblings.
`V4_OMP_CORES=perf|all|<n>` ist dokumentiert und getestet; `perf` wählt auf dem
i5-13400F sechs P-Core-Worker, während `all` bis zur noch ausstehenden A/B
weiterhin der Default bleibt. `OMP_NUM_THREADS` und `COLI_NO_OMP_TUNE` haben weiterhin
Vorrang. `coli` und der Direkt-Server lassen für V4 nun die OMP-Umgebung frei,
damit der Binary-Pfad nicht mehr vor dem Start maskiert wird; sie setzen auch
keine Spin-Wait-Defaults mehr.

Commit 3 erkennt `/opt/cuda/bin/nvcc` vor `/usr/local/cuda/bin/nvcc` und reicht
einen optionalen `NVCC_CCBIN` als `-ccbin` an nvcc durch. Auf dem Ziel sind CUDA
13.3 und `g++-14` vorhanden; der vollständige CUDA-Build ist die
Hardwareabnahme in Plan 15. Commit 4 ergänzt das CachyOS-Profil einschließlich
zram, THP, ananicy-cpp und des neuen `bench_v4.py`-Automatikmodus.

## Implementierungsabnahme

- `coli_omp_tune_threads` läuft im V4-Start und loggt die Teamgröße. Auf einer
  durchgängig SMT-fähigen CPU ist sie halb so groß wie vorher; **auf dem
  i5-13400F 10 statt 16**, weil nur die 6 P-Cores SMT haben und die 4 E-Cores
  nicht. Wer „halb" als Kriterium nimmt, hält den richtigen Wert für einen Fehler.
- `make -C c deepseek-v4 CUDA=1` baut auf CachyOS ohne manuelles `CUDA_HOME`.

Die Durchsatz- und THP-Hardwaremessung ist Abnahme von Plan 15.

## Risiken

- **+2.3× ist eine fremde Messung** (Zen3, 5950X, GLM). Auf anderer Hardware und
  für V4 kann der Faktor deutlich kleiner sein. Die Änderung ist trotzdem richtig —
  drei Zeilen, dieselbe Behandlung wie in drei anderen Engines, und der Header
  regelt den Fehlerfall sauber. Aber die Zahl nicht als Zusage weitergeben.
- **THP kann Latenz-Spitzen erzeugen.** `defrag` steht auf Arch üblicherweise auf
  `madvise`, was Kompaktierung nur für madvised Bereiche macht — genau unsere. Bei
  fragmentiertem Speicher kann die Allokation dadurch länger dauern. Da sie einmal
  beim Start passiert, ist das akzeptabel; falls doch nicht, ist
  `MADV_HUGEPAGE` ein Einzeiler zum Zurücknehmen.
- **Ich konnte nichts davon auf deiner Maschine messen.** Die drei `bash`-Blöcke
  oben sind die Prüfungen, die vor dem Einbau laufen sollten — insbesondere der
  THP-Modus, weil Commit 2 bei `[always]` wirkungslos wäre.
