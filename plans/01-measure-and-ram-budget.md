# 01 — Messen und RAM-Budget holen

Voraussetzung: [00-reference.md](00-reference.md)

*Commit: `perf: configurable scratch reserve and a 32GB tuning profile for V4`*

## Ziel

Erst die Ist-Zahlen festnageln, dann die zwei RAM-Hebel ziehen, die **ohne
Attention-Umbau** wirken. Erwartung: **+3.0 GiB Expert-Cache**, also ~5 zusätzliche
Expert-Slots pro Layer, bevor eine einzige Zeile Attention-Code angefasst wird.

Dieser Plan ist bewusst der erste: er erzeugt die Messgrundlage, gegen die alle
folgenden Phasen ihren Nutzen belegen.

## Ausgangslage

Der Planner rechnet in `build_runtime_plan` ([c/deepseek_v4.c:924](../c/deepseek_v4.c)):

```c
uint64_t hidden  = 64 * config.hc_mult * config.hidden_size * sizeof(float) * 2;
uint64_t scratch = 512 * MIB;                                    // :961  hartkodiert
uint64_t runtime_other = context_bytes(&config, context) + hidden + scratch;
runtime_other += runtime->dspark_reserve_bytes;                  // :967
```

und in `coli_v4_resource_plan_compute` ([:693](../c/deepseek_v4.c)):

```c
int explicit_process_limit = inputs->user_limit_bytes &&
    inputs->user_limit_bytes < available;                        // :704
uint64_t system = explicit_process_limit ? 0 : available / 8;    // :707
if (!explicit_process_limit && system < 512 * MIB) system = 512 * MIB;
if (system > 4096 * MIB) system = 4096 * MIB;
```

Der bestehende Report ([:1057](../c/deepseek_v4.c)) druckt bereits alles Nötige:

```
ram_tiers available=%.2fGiB dense=%s(%.2fGiB) target_slots=%d
          target_cache=%.2fGiB head=%s projected=%.2fGiB
```

## Hebel 1 — `RAM_GB` explizit setzen (kein Code)

Ein expliziter Prozess-Limit setzt `system_reserve` auf **0**. Automatisch wären es
`available/8`, gedeckelt bei 4 GiB — auf 32 GB also ~3.6 GiB.

`RAM_GB` ist bereits verdrahtet ([:8469](../c/deepseek_v4.c)) →
`open_options.memory_limit_bytes` → `runtime->memory_limit_bytes` ([:6504](../c/deepseek_v4.c))
→ `inputs->user_limit_bytes`.

**Bedingung:** `user_limit_bytes < available`, sonst greift der Zweig nicht. Bei
MemAvailable ~29 GiB ist `RAM_GB=28` richtig, `RAM_GB=32` wäre wirkungslos.

Netto: 28.0 GiB Planner-Budget ohne Systemreserve gegen effektiv ~25.4 GiB
automatisch. **+2.6 GiB.**

## Hebel 2 — `V4_SCRATCH_MB`

Die 512 MiB in `:961` sind eine feste Annahme. Bei Decode mit Batch 1 und
`hc_mult=4` ist der tatsächliche Arbeitsspeicherbedarf ein Bruchteil davon.

```c
static uint64_t v4_scratch_bytes(void) {
    const char *setting = getenv("V4_SCRATCH_MB");
    long mb = setting ? atol(setting) : 512;
    if (mb < 64)  mb = 64;      /* unter 64 MiB wird es unehrlich */
    if (mb > 4096) mb = 4096;
    return (uint64_t)mb * MIB;
}
```

Aufrufstelle `:961`. Der Clamp folgt dem Muster von `coli_v4_dspark_cache_gb`
([:6288](../c/deepseek_v4.c)), das genauso einen Env-Wert gegen Tippfehler absichert.

`V4_SCRATCH_MB=128` gibt **+0.375 GiB**. Konservativ, weil der Wert eine Reserve
ist und keine Allokation — zu klein gewählt heißt nicht OOM, sondern nur ein
optimistischerer Plan.

## Hebel 3 — `CTX` ehrlich wählen (kein Code)

`context_bytes` ([:909](../c/deepseek_v4.c)) reserviert für die **volle** Kontextlänge
im Voraus:

```c
total = num_hidden_layers * sliding_window * head_dim * 4;      /* Fenster, alle Layer */
for (layer...) {
    if (!ratio) continue;
    compressed = ceil(context / ratio);
    total += compressed * head_dim * 4;                          /* komprimiert */
    if (ratio == 4) total += compressed * index_head_dim * 4;    /* Indexer */
}
```

`CTX` default 4096 ([:8456](../c/deepseek_v4.c)), gedeckelt auf
`max_position_embeddings` ([:957](../c/deepseek_v4.c)). Wer 32k braucht, sollte nicht
den Modellmaximalwert stehen lassen.

Das ist zugleich die Formel, an der die Phasen 03/04 ansetzen: sie ist der Ort, an
dem KV-Kompression in Expert-Slots umschlägt.

## Messaufgabe

Vor allen Codeänderungen die tatsächliche `compress_ratios`-Verteilung bestimmen —
sie entscheidet, wieviel die Phasen 03/04 überhaupt bringen können:

```bash
python3 -c "
import json,collections,sys
c=json.load(open(sys.argv[1]+'/config.json'))
r=c['compress_ratios']
print('layers', len(r), collections.Counter(r))
print('sliding_window', c['sliding_window'], 'head_dim', c['head_dim'])
" /pfad/DeepSeek-V4-Flash
```

Daraus die KV-Bytes pro Token:
`Σ_ratio4 (head_dim + index_head_dim)·4/4 + Σ_ratio128 head_dim·4/128`.
Bei geschätzt 14 ratio-4- und 14 ratio-128-Layern sind das ~9.2 KB/Token f32.

Ohne Checkpoint ersatzweise die Ist-Zeile des Planners aufnehmen — sie genügt für
den Vorher/Nachher-Vergleich der Hebel 1 und 2.

## Der Benchmark-Harness — wichtiger als die zwei Hebel

**Ohne ihn ist „hyper-optimiert" nicht überprüfbar.** Jeder folgende Plan hat
Abnahmekriterien mit Zahlen; ohne einen festen Messaufbau sind das Meinungen.

`c/tools/bench_v4.py`, ein Lauf, eine Zeile Ausgabe:

```
bench_v4 rev=<sha> profile=<name> ctx=32768
  prefill 3.2 tok/s | decode 1.41 tok/s | ttft 24.6 s
  expert hit 18.3% | disk 2.41 GB/token | ram_cache 15.4 GiB slots 27
  kv 1.68 GiB | vram —  | omp_threads 16
```

Anforderungen, damit die Zahlen über Wochen vergleichbar bleiben:

- **Fester Promptsatz**, im Repo, nicht generiert. Drei Längen (kurz ~500 Token,
  mittel ~8k, lang ~64k), damit Prefill- und Decode-Regime getrennt sichtbar sind.
- **Feste Seeds**, greedy, keine Sampling-Varianz.
- **Cache-Zustand explizit**: einmal kalt (`echo 3 > /proc/sys/vm/drop_caches`,
  braucht root) und einmal warm. Ein warmer Lauf misst etwas anderes.
- **Ausgabe als JSON-Zeile** nach `bench/<datum>-<sha>.jsonl`, damit sich eine
  Reihe ergibt statt einzelner Screenshots.
- **Immer mit ausgeben, was gesetzt war** — alle `V4_*`-Knöpfe, sonst ist die
  Zeile in zwei Wochen wertlos.

Die Trefferquote ist die wichtigste Zahl darin. Sie kommt aus dem Expert-Store;
falls sie noch nicht exportiert wird, ist das der erste Commit dieses Plans.

**Baseline auf `main` aufnehmen, bevor irgendetwas geändert wird.** Das ist die
Zahl, gegen die alle Pläne antreten.

### Was plausibel ist

Damit eine Messung als richtig oder falsch erkennbar ist, hier die Erwartung aus
dem Bandbreitenabschnitt in [00-reference.md](00-reference.md):

| | Decode |
|---|---|
| Bandbreiten-Deckel (100 % RAM-Treffer, DDR4-3200) | **~13 tok/s** |
| ~20 % Residenz, Dual-Streaming ~10 GB/s | **~3.5 tok/s** |
| ~11 % Residenz (Ausgangslage), ein Laufwerk ~7 GB/s | **~2 tok/s** |

Der Weg von ~2 auf ~3.5 ist das, was die Pläne holen; die 13 sind die Asymptote.
**Eine Messung deutlich über 13 tok/s ist ein Messfehler**, keine Sensation — dann
wurde ein warmer Lauf gemessen, ein zu kurzer Prompt, oder DSpark hat Tokens
akzeptiert, die der Zähler einzeln gezählt hat. Genau dafür gibt der Harness
Konfiguration und Cache-Zustand mit aus.

## Deliverable

`docs/deepseek-v4-tuning-32gb.md` mit:

- der gemessenen `ram_tiers`-Zeile vorher/nachher je Hebel
- der `compress_ratios`-Verteilung und den daraus folgenden KV-Bytes/Token
- der Empfehlung für 32 GB + 12 GB (wächst über die Phasen mit)

Verlinkung aus `docs/deepseek-v4.md` unter „Memory policy", wo `--ram` schon als
Planner-Budget erklärt wird.

## Tests

- `c/tests/test_v4_scratch_env.c` — neue Regel in `c/Makefile`
  (`TEST_RULES` nimmt sie automatisch auf, [c/Makefile:360](../c/Makefile)).
  Prüft: Default 512 MiB, Clamp unten/oben, Müll-Eingabe (`V4_SCRATCH_MB=abc`)
  fällt auf Default. Muster: `test_env_defaults.py` gibt es schon für andere Knöpfe.
- Bestehende Gates: `make -C c test`, `make -C c deepseek-v4-tiny-check`.

## Abnahme

- `ram_tiers`-Zeile zeigt mit `RAM_GB=28 V4_SCRATCH_MB=128` ein um **≥3 GiB**
  höheres `target_cache` als ohne, bei identischem `projected`-Verhalten.
- `target_slots` ist entsprechend gestiegen.
- Tiny-Fixture token-identisch.

## Risiken

- **`RAM_GB` zu hoch gesetzt** deaktiviert die Systemreserve, ohne sie zu ersetzen.
  Bei 32 GB ist 28 der sinnvolle Wert; darüber riskiert man OOM-Killer statt
  Streaming. Im Tuning-Doc explizit machen.
- **`V4_SCRATCH_MB` ist eine Reserve, keine Grenze** — `--ram GiB` ist laut
  `docs/deepseek-v4.md` ausdrücklich „a planner budget, not an OS-enforced limit".
  Zu klein gewählt macht den Plan optimistisch, nicht kaputt; das gehört in die
  Doku, damit niemand es für eine harte Schranke hält.
- Ohne Checkpoint bleibt die `compress_ratios`-Messung offen. Die Hebel 1 und 2
  wirken unabhängig davon.
