# 15 — Abschlussabnahme und profilgeführtes Tuning

Voraussetzung: Alle Implementierungspläne 01–14 einschließlich der verbleibenden
Commits aus 09 und 10 sowie [13-frontend-v4.md](13-frontend-v4.md). Dieser Plan
ist **immer der letzte**: Er führt keine vorab vermuteten Optimierungen ein,
sondern misst den integrierten Stand auf der Zielmaschine und begründet daraus
kleine, getrennte Folge-Commits.

## Zweck und Grenze

Jeder Implementierungsplan behält seine schnellen, deterministischen Gates:
Unit-/Source-Tests, `make -C c test && make -C c check` und, soweit möglich,
das Tiny-Orakel. Sie sind Voraussetzung für jeden Commit und ersetzen keine
V4-Produktabnahme.

Alle teuren, hardwaregebundenen Messungen werden hier gebündelt: Full-Checkpoint
und DSpark, kalte und prozesswarme Caches, 32k/128k/256k/1M, SSD-/HMB-Varianten,
VRAM/RSS sowie Frontend/Serve. Es gibt vorher **keinen** neuen Performance-Default
allein aufgrund einer Erwartung oder eines Teilprofils.

## Commit 1 — reproduzierbare Abschlussmatrix

`test: run the V4 full-checkpoint acceptance matrix`

Auf dem lokalen Checkpoint werden mit festgehaltenem Binary-Commit, Kernel,
CUDA-Treiber, CPU-Affinität, Env und Prompt-Satz mindestens folgende Fälle
ausgeführt:

| Profil | Konfiguration | Pflichtnachweis |
|---|---|---|
| Referenz | Default, ein Laufwerk | Basis für alle Deltas |
| Endprofil 32k | `CUDA=1 V4_VRAM=1 V4_MTP=1 V4_DRAFT=3` | Target-/DSpark-Token-A/B, Cache- und Decode-Metriken |
| Langkontext 128k | native KV, ein Laufwerk | Prefill/Decode, Indexer und Peak RSS/VRAM |
| Langkontext 256k | `V4_KV=turbo3 V4_PREFILL_CHUNK=32768 V4_MTP=1 V4_DRAFT=3 V4_VRAM=1` | Chunk-Grenzen, TTFT, DSpark und Speichergrenzen |
| Experiment 1M | wie 256k, erst nach grünem 256k | 32 Chunks, RSS/VRAM und Langkontext-Decode |

Jeder Fall läuft kalt und in einem persistent geöffneten Prozess warm. Erfasst
werden TTFT, Prefill-/Decode-tok/s, Peak RSS, `ram_tiers`, `vram_tiers`,
`nvidia-smi` vor Open/nach Dense+Head/nach DSpark/Decode, Cache Hits/Misses,
gelesene GB, Indexer- und DSpark-Stufenzeiten. Ein Ergebnis oberhalb des
~13-tok/s-DDR4-Bandbreitendeckels wird als Messfehler untersucht, nicht als Gewinn
übernommen.

Die nach Merge offene Phase-14-Grenzprobe ist Teil dieser Matrix: Ein Prompt über
mindestens eine 64er-Chunkgrenze läuft dreimal (unchunked Target-only, chunked
Target-only, chunked DSpark). Target-Folgen müssen greedy identisch sein; DSpark
muss echte Proposals liefern und exakt gegen dieselbe Target-Folge verifizieren.

## Commit 2 — Speicher- und I/O-Profile

`test: compare cache, prefill, THP and storage profiles`

- Vier oder mehr `V4_PIN_SLOTS`-/`V4_PIN_FRACTION`-Konfigurationen gegen den
  festen Promptsatz: Trefferquote, gelesene GB und Tokenfolge.
- Batchgröße, `COLI_V4_EXPERT_PREFETCH` und Lookup-Worker im Prefill: TTFT,
  Prefill-Durchsatz und Cache-Verdrängung.
- THP-Nachweis (`AnonHugePages`), 6-P-/10-Core-/16-Thread-Vergleich und die
  Arch/CachyOS-Umgebung aus Plan 09.
- Laufwerk A allein: O_DIRECT, Prefetch und HMB-/Queue-Tiefe mit dokumentierten
  Systemwerten. Mit Laufwerk B zusätzlich Spiegelung, automatische und manuelle
  Gewichte, gleichmäßige gegen gewichtete Stripes sowie parallele Bandbreite.

Der Einzellaufwerk-Fall bleibt Pflicht und Referenz. Zwei Laufwerke ersetzen ihn
nicht. Jede Messung enthält die zugrunde liegende Cache-/Thermal- und
Prozesszustandsangabe.

## Commit 3 — nur belegte Optimierungen und Profil-Dokumentation

`perf: apply measured V4 profile refinements`

Aus Commit 1–2 entstehen nur Änderungen, die einen klaren, reproduzierbaren
Gewinn zeigen und die Semantik wahren. Jede Änderung erhält ihren eigenen Commit
mit erneutem Token-A/B und der Vorher-/Nachher-Tabelle. Ohne belegten Gewinn bleibt
der bisherige Default erhalten. Kandidaten sind Pin-/Prefetch-Defaults,
HMB-Queue-/Workerwerte und dokumentierte manuelle Spiegelgewichte; neue
Algorithmen (etwa io_uring oder CUDA-Indexer) brauchen dagegen einen neuen Plan.

Danach aktualisieren `docs/deepseek-v4-tuning-32gb.md`,
`docs/ENVIRONMENT.md`, [00-reference.md](00-reference.md) und die Ergebnisblöcke
der betroffenen Pläne die tatsächlich gemessenen Profile und Abweichungen.

## Abnahme

- Alle schnellen Gates sind auf dem finalen Stand grün.
- Die vollständige Matrix ist mit Rohlogs und zusammengefassten Zahlen
  nachvollziehbar; keine erwartete Zahl wird als gemessen ausgegeben.
- Die Full-Checkpoint-Token-A/Bs decken Default, CUDA/VRAM, Chunking und DSpark
  ab; die 256k-Grenzprobe ist grün, bevor 1M gewertet wird.
- Einzellaufwerk und optionaler Spiegel sind getrennt verglichen.
- Jeder übernommene Profilwert besitzt einen reproduzierten Vorteil; alle anderen
  Optimierungen bleiben aus oder werden als Folgeplan festgehalten.
