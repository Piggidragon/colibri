# 01 — Messen und RAM-Budget holen

Voraussetzung: [00-reference.md](00-reference.md)

*Commit: `perf: configurable scratch reserve and a 32GB tuning profile for V4`*

## Ziel

Erst die Ist-Zahlen festnageln, dann die zwei RAM-Hebel ziehen, die **ohne
Attention-Umbau** wirken. Vor Resident-Tier-Entscheidungen geben sie rund
**3.0 GiB Planner-Budget** frei; wieviel davon als Expert-Cache ankommt, muss der
Harness messen, weil Dense-/Head-Tierwechsel nicht additiv sind.

Dieser Plan ist bewusst der erste: er erzeugt die Messgrundlage, gegen die alle
folgenden Phasen ihren Nutzen belegen.

## Ausgangslage

Der Planner rechnet in `build_runtime_plan` ([c/deepseek_v4.c:924](../c/deepseek_v4.c)):

```c
uint64_t hidden  = context * config.hc_mult * config.hidden_size
                   * sizeof(float) * 2;
uint64_t scratch = coli_v4_scratch_bytes();
uint64_t runtime_other = context_bytes(&config, context) + hidden + scratch;
runtime_other += runtime->dspark_reserve_bytes;                  // :967
```

Die Kontext-Skalierung ist Teil dieses Commits: `coli_v4_session_create`
allokiert `state` und `next` jeweils für `max_prompt_tokens`, und CLI/Serve setzen
das auf den geplanten Kontext. Die frühere 64-Token-Konstante unterschlug bei
128k je nach Geometrie 4 bis 16 GiB und durfte nicht dem Expert-Cache zugeteilt
werden.

und in `coli_v4_resource_plan_compute` ([:693](../c/deepseek_v4.c)):

```c
int explicit_process_limit = inputs->user_limit_bytes &&
    inputs->user_limit_bytes < available;                        // :705
uint64_t system = explicit_process_limit ? 0 : available / 8;    // :713
if (!explicit_process_limit && system < 512 * MIB) system = 512 * MIB;
if (system > 4096 * MIB) system = 4096 * MIB;

multiply_u64(inputs->maximum_layer_bytes, 2, &layers_twice);     // :719
add_u64(layers_twice, inputs->runtime_other_bytes,
        &plan->runtime_reserve_bytes);
```

**Die letzten zwei Zeilen sind die, die man beim Nachrechnen vergisst.** Die
Runtime-Reserve ist `2 × maximum_layer_bytes + runtime_other`, nicht bloß
`runtime_other`. `maximum_layer_bytes` ist der größte Einzel-Layer aus der
Dense-Inventur ([:947](../c/deepseek_v4.c)), hier ~0.162 GiB — also ~0.32 GiB, die
in **keinem** der Hebel unten auftauchen und von keiner späteren Phase verschwinden.
Sie stehen jetzt in der RAM-Bilanz in [00-reference.md](00-reference.md); die
Messaufgabe unten soll sie am echten Planner bestätigen.

Der bestehende Report ([:1057](../c/deepseek_v4.c)) druckt bereits alles Nötige:

```
ram_tiers available=%.2fGiB dense=%s(%.2fGiB) target_slots=%d
          target_cache=%.2fGiB head=%s projected=%.2fGiB
```

## Hebel 1 — Prozessbudget explizit setzen (kein Code)

Ein expliziter Prozess-Limit setzt `system_reserve` auf **0**. Automatisch wären es
`available/8`, gedeckelt bei 4 GiB — auf 32 GB also ~3.6 GiB.

Im `SERVE=1`-Launcher ist `RAM_GB` verdrahtet; der one-shot CLI setzt denselben
`open_options.memory_limit_bytes`-Wert ausschließlich über `--memory-gb`.
Beide Pfade führen über `runtime->memory_limit_bytes` zu
`inputs->user_limit_bytes`.

**Bedingung:** `user_limit_bytes < available`, sonst greift der Zweig nicht. Bei
MemAvailable ~29 GiB sind `--memory-gb 28` beziehungsweise `RAM_GB=28` im
Serve-Pfad richtig; 32 wäre wirkungslos.

Netto: 28.0 GiB Planner-Budget ohne Systemreserve gegen effektiv ~25.4 GiB
automatisch. **+2.6 GiB.**

## Hebel 2 — `V4_SCRATCH_MB`

Die 512 MiB in `:961` sind eine feste Annahme. Bei Decode mit Batch 1 und
`hc_mult=4` ist der tatsächliche Arbeitsspeicherbedarf ein Bruchteil davon.

```c
static uint64_t v4_scratch_bytes(void) {
    const char *setting = getenv("V4_SCRATCH_MB");
    long mb = 512;
    if (setting && setting[0]) {
        char *end = NULL;
        errno = 0;
        long parsed = strtol(setting, &end, 10);
        if (!errno && end != setting && !*end) mb = parsed;
    }
    if (mb < 64)  mb = 64;      /* unter 64 MiB wird es unehrlich */
    if (mb > 4096) mb = 4096;
    return (uint64_t)mb * MIB;
}
```

Aufrufstelle `:961`. Der Clamp folgt dem Muster von `coli_v4_dspark_cache_gb`
([:6288](../c/deepseek_v4.c)). Anders als `atol` lässt die vollständige
`strtol`-Prüfung Müll, Suffixe und Überläufe auf den Default zurückfallen; erst
gültige Zahlen werden geklemmt.

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

`CTX` default 4096 und wird auf `max_position_embeddings` gedeckelt. Es skaliert
nicht nur diese KV-Reserve, sondern auch die beiden oben beschriebenen
Full-Prefill-Zustandsbuffer. Wer 32k braucht, sollte nicht 128k oder den
Modellmaximalwert stehen lassen.

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

**Erwartet wird** (Paper 4.2.1, siehe [00-reference.md](00-reference.md)):
2 Layer mit Ratio 0, **21 mit Ratio 4**, **20 mit Ratio 128** — daraus
**13.4 KB/Token** bei f32. Weicht der Checkpoint davon ab, sind die
KV-Rechnungen in 00, 03 und 04 anzupassen; dann gilt die Pflegeregel aus
[AGENTS.md](../AGENTS.md) (Korrektur in 00, nicht nur hier).

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
| ~17 % Residenz (32k-Endprofil), Dual-Streaming ~10 GB/s | **~3.3 tok/s** |
| ~17 % Residenz (32k-Endprofil), ein Laufwerk ~7 GB/s | **~2.5 tok/s** |
| ~10 % Residenz (Ausgangslage), ein Laufwerk ~7 GB/s | **~2 tok/s** |

**Gemessen wird auf einem Laufwerk.** Das zweite Laufwerk ist optional (siehe
[10-dual-streaming.md](10-dual-streaming.md)), also ist die Einzellaufwerk-Zeile
der Maßstab für alle Phasen — sonst mischt sich der Hardwarebeitrag in jede
Messung. Läuft eine Messreihe mit `COLI_MODEL_MIRROR`, gehört das in dieselbe
Zeile wie die übrigen Knöpfe.

Der Weg von ~2 auf gut 3 ist das, was die Pläne holen; die 13 sind die Asymptote.
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
  (`TEST_RULES` nimmt sie automatisch auf, [c/Makefile:359](../c/Makefile)).
  Prüft: Default 512 MiB, Clamp unten/oben, Müll-Eingabe (`V4_SCRATCH_MB=abc`)
  fällt auf Default; außerdem robustes `CTX`-Parsing und die Formel der beiden
  Session-State-Buffer. Muster: `test_env_defaults.py` gibt es schon für andere Knöpfe.
- Bestehende Gates: `make -C c test`, `make -C c deepseek-v4-tiny-check`.

## Abnahme

- Die Runtime-Reserve enthält exakt die zwei kontextgroßen Session-State-Buffer.
- Bei identischem `CTX` sind die erwarteten Budgetdeltas des expliziten Limits
  und von `V4_SCRATCH_MB=128` sichtbar; Tierwechsel und Slot-Rundung werden als
  Messergebnis dokumentiert, nicht als garantierte **≥3 GiB** Cache-Zunahme.
- Tiny-Fixture token-identisch.

## Risiken

- **Prozessbudget zu hoch gesetzt** deaktiviert die Systemreserve, ohne sie zu
  ersetzen. Bei 32 GB ist 28 der sinnvolle Wert; darüber riskiert man OOM-Killer
  statt Streaming. Im one-shot CLI gilt `--memory-gb`, in Serve `RAM_GB`.
- **`V4_SCRATCH_MB` ist eine Reserve, keine Grenze** — `--ram GiB` ist laut
  `docs/deepseek-v4.md` ausdrücklich „a planner budget, not an OS-enforced limit".
  Zu klein gewählt macht den Plan optimistisch, nicht kaputt; das gehört in die
  Doku, damit niemand es für eine harte Schranke hält.
- Ohne Checkpoint bleibt die `compress_ratios`-Messung offen. Die Hebel 1 und 2
  wirken unabhängig davon.
