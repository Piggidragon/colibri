# 07 — Head und DSpark in VRAM

Voraussetzung: [00-reference.md](00-reference.md), [06-dense-vram.md](06-dense-vram.md)

*Commits:*
1. `feat: BF16 output head on GPU for V4`
2. `feat: DSpark drafter tensors on GPU for V4`

## Messergebnis (PR #9)

Gemessen am 2026-08-10 auf der Zielmaschine (i5-13400F, 32 GiB RAM,
RTX 4070 `sm_89`) gegen den vollständigen Checkpoint unter
`~/Services/models/colibri/deepseek-v4-flash`, mit
`CTX=32768 V4_SCRATCH_MB=128`, `--memory-gb 28`, `OMP_NUM_THREADS=16`, greedy
und dem Rohprompt `Hi`, je 24 erzeugte Tokens. Decode ist `23 / after_first`
aus der `timing`-Zeile, wie in der Vormessung. Der CUDA-Build und der
CUDA-Harness einschließlich `test_v4_head_cuda` sind auf der Karte grün.

| Lauf | Platzierung / Cache | TTFT | Decode | tok/s | DSpark |
|---|---|---:|---:|---:|---|
| CUDA ziel-only | Dense `vram` 5.46 GiB, Head `vram-bf16`, 42 Slots, 22.49 GiB | 9.459 s | 36.531 s | **0.630** | aus |
| CUDA DSpark | wie oben + DSpark `vram-lazy`, 40 Slots, 21.42 GiB | 9.311 s | 46.276 s | 0.497 | 3 Versuche, 9 Vorschläge, 4 akzeptiert (**44.4 %**) |
| CPU ziel-only | Dense `ram` 6.27 GiB, Head resident, 30 Slots, 16.06 GiB | 9.703 s | 45.355 s | 0.507 | aus |
| CPU DSpark | wie oben + DSpark `ram-lazy`, 28 Slots, 14.99 GiB | 12.043 s | 74.719 s | 0.308 | 4 Versuche, 12 Vorschläge, 9 akzeptiert (**75.0 %**) |

**Phase 07 erreicht ihr Ziel.** Head und DSpark-Backbone liegen auf der Karte;
gegenüber dem CPU-Pfad sind das **+6.43 GiB Target-Cache** (42 statt 30 Slots)
und **+24.2 % Decode-Durchsatz** bei Ziel-only. Der DSpark-Backbone belegt beim
Lazy-Upload **330 MiB** VRAM — die Planner-Reserve steigt von 1.00 auf
1.32 GiB — und kostet auf beiden Pfaden identisch 1.07 GiB Target-Cache, also
zwei Slots pro Lage. Die drei gerouteten DSpark-Expert-Caches bleiben
absichtlich im RAM (je 11/256 Slots, 0.15 GB), sie sind nicht Teil des
GPU-Backbones.

**Die Tokenfolgen-Abnahme ist bestanden.** Innerhalb eines Backends ist
Drafting exakt ausgabeerhaltend:

| Vergleich | erste Abweichung |
|---|---|
| CUDA ziel-only vs. CUDA DSpark | keine, identisch |
| CPU ziel-only vs. CPU DSpark | keine, identisch |
| CUDA ziel-only vs. CPU DSpark | Token 14 |
| CPU ziel-only vs. CUDA DSpark | Token 14 |

Die Vormessung meldete an dieser Stelle „erster Unterschied beim 14. Token" und
wertete das als Semantikfehler des Drafters. Die unteren beiden Zeilen zeigen,
woher die Zahl kam: Token 14 ist der Punkt, an dem **CPU und CUDA**
auseinanderlaufen, ganz ohne DSpark. Verglichen wurde also über die
Backend-Achse, nicht über die Drafter-Achse. Gegengeprüft durch Zurücknehmen
des FP8-Aktivierungs-Fixes aus `edb7b09`: auch ohne ihn sind CUDA-ziel-only und
CUDA-DSpark tokenidentisch, bei exakt derselben Akzeptanz (3/9/4, 44.4 %). Der
Fix korrigiert eine reale numerische Abweichung — isoliert gegen
`coli_fp8_matvec_ref` 2.7e-2 auf 9.8e-8 relativer RMS — die auf diesem Prompt
aber nicht sichtbar wird.

**Zwei Abnahmepunkte bleiben offen:**

- **Durchsatz.** DSpark verliert gegen Ziel-only: −21.1 % auf CUDA, −39.3 % auf
  CPU. Das ist kein Phase-07-Regress. Der Verlust tritt auf dem reinen CPU-Pfad
  genauso auf und ist damit eine Eigenschaft des Drafters, nicht der
  Platzierung; Phase 07 macht ihn nur im RAM billiger. Ein Durchsatzgewinn
  durch DSpark bleibt unbelegt und gehört in einen eigenen Plan.
- **Akzeptanzrate.** 44.4 % auf CUDA gegen 75.0 % auf CPU; der Plan verlangte
  „unverändert" als Beleg gleicher Entwürfe. Ursache ist die CPU/CUDA-Divergenz
  des **Targets**: über 45 Teacher-Forcing-Positionen weicht das Top-1-Token auf
  11.1 % der Positionen ab, ganz ohne Drafter. Ein anderer Zielpfad akzeptiert
  andere Vorschläge. `docs/ENVIRONMENT.md` deckt das unter `V4_VRAM` ab
  („equivalent only within the harness tolerances, not bit-identical"), und
  `V4_VRAM` ist opt-in — die Zahl war bis hierher aber ungemessen.

## Ziel

Weitere **2.16 GiB RAM frei** (0.99 Head + ~1.17 DSpark-Reserve) und nebenbei der beste
Rechenzeit-pro-Aufwand-Posten des ganzen Branches: der Head ist ein
bandbreitengebundener Matvec über 1.059 GB pro Token.

Danach ist DSpark auf 32 GB überhaupt erst bezahlbar — es kostet keinen
Expert-Cache mehr.

## Teil 1 — Der Head

### Ausgangslage

`head_argmax` ([c/deepseek_v4.c:8097](../c/deepseek_v4.c)):

```c
const uint16_t *resident = coli_v4_head_cache_data(engine, shard, head->off, resident_bytes);
if (resident) {
    float *scores = malloc(vocab * sizeof(*scores));
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < vocab; row++)
        scores[row] = head_bf16_dot(resident + (size_t)row * d, hidden, d);
    /* linearer Scan für argmax, in Vokabularordnung */
}
```

`vocab = 129280`, `d = 4096`, BF16 → 129280 × 4096 × 2 B = **1.059 GB pro Token
gelesen**. Auf DDR4-3200 mit ~45 GB/s sind das ~24 ms (siehe
[00-reference.md](00-reference.md)); auf dem 4070 mit ~500 GB/s ~2 ms. Faktor ~12,
und gleichzeitig **0.99 GiB** RAM frei.

**GB, nicht GiB.** 1.059 GB sind 0.986 GiB; eine frühere Doku-Fassung trug am
Dezimalwert das falsche GiB-Suffix. Für die Zeitrechnung oben ist 1.059 GB richtig
(÷ GB/s), für die RAM- und VRAM-Bilanzen sind es 0.986 GiB. Wer beide Zahlen
mischt, verrechnet sich um 7 %.

Der Fallback darunter ([:6814](../c/deepseek_v4.c)) streamt in 64-Zeilen-Kacheln,
wenn der Head nicht resident ist — der bleibt unverändert als letzte Stufe.

Geladen wird über `coli_v4_head_cache_load` ([:819](../c/deepseek_v4.c)) in
`engine->head_cache.data`; gelesen über `coli_v4_head_cache_data`
([:848](../c/deepseek_v4.c)), das den Bereich gegen `shard`/`offset`/`length`
validiert.

### Änderung

- `coli_v4_head_cache_load` bekommt eine Device-Variante: statt `malloc` +
  `coli_st_read_at` wird nach `coli_cuda_pipe_alloc` gelesen und hochgeladen
  (in Kacheln, damit kein 1 GiB-Host-Zwischenpuffer nötig ist).
- Neuer Kernel in `backend_cuda_v4.cu`: `v4_cuda_head_argmax(float *best_logit,
  int *best_token, const void *head_dev, const float *hidden_dev, int vocab, int d)`.
  BF16-Dekodierung im Kernel, f32-Akkumulation, Block-weise Reduktion, dann
  ein zweiter Pass für das Argmax.
- `head_argmax` bekommt den Device-Zweig vor dem resident-Zweig; beide Fallbacks
  bleiben erhalten.

**In BF16 belassen.** fp8 würde 0.53 GiB sparen, wäre aber eine stille
Präzisionsänderung an der Stelle, die das Ausgabetoken bestimmt — das fällt unter
die README-Garantie und gehört, wenn überhaupt, in einen eigenen opt-in-Knopf.

### Die Semantikfrage

Der Kommentar in `head_argmax` ist ausdrücklich:

> Each row retains the same scalar accumulation order and the final scan retains
> vocabulary order, so logits/tie-breaking do not change.

Eine GPU-Reduktion **ändert die Akkumulationsreihenfolge**. Bei zwei fast gleichen
Logits kann das Argmax kippen. Das ist keine Kleinigkeit — es ist genau der Fall,
den der Kommentar schützt.

Konsequenzen für die Umsetzung:

- Der Device-Pfad ist **opt-in** (`V4_VRAM=1`), nie Default.
- Für das Argmax bei Gleichstand die Vokabularordnung erzwingen: bei
  `score == maximum` gewinnt der **kleinere** Index, wie es der lineare Scan tut
  (`if (scores[row] > maximum)` — strikt größer, also gewinnt der erste).
- Der Test vergleicht nicht nur den Gewinner, sondern die **Logit-Verteilung**
  gegen die CPU, und meldet die maximale Abweichung.

Damit bleibt der Unterschied auf Rundung beschränkt und ist messbar statt still.

### Kopplung mit DSpark

`v4_ds_...` liest den Head über `coli_v4_head_cache_data`
([c/deepseek_v4_dspark.inc:1206](../c/deepseek_v4_dspark.inc)):

```c
const uint16_t *resident = coli_v4_head_cache_data(...);
uint16_t *raw = resident ? NULL : malloc(...);
```

Wandert der Head aufs Gerät, liefert `coli_v4_head_cache_data` keinen Host-Zeiger
mehr und DSpark fällt auf den Lesepfad zurück — also von resident auf streamend,
was den Drafter langsamer macht als vorher.

**Head und DSpark müssen deshalb gemeinsam wandern.** Deshalb sind es zwei Commits
in einem Plan und nicht zwei Pläne.

## Teil 2 — DSpark

### Ausgangslage

**DSpark ist Teil des Hauptcheckpoints, kein zweites Modell.** Die Drafter-Tensoren
liegen unter `mtp.<stage>.` in
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
und werden aus `engine->target_index` gelesen — `mtp.0.main_proj.weight`,
`mtp.2.markov_head.markov_w1.weight` und so weiter
([c/deepseek_v4.c:7571](../c/deepseek_v4.c),
[c/deepseek_v4_dspark.inc:193](../c/deepseek_v4_dspark.inc)).

Kein zweiter Modellpfad, keine zweite Index-Instanz. Für Plan 10 (Dual-Streaming)
heißt das: die DSpark-Tensoren liegen auf denselben Shards und profitieren
automatisch vom Mirror-Routing.

Nebenbei: das Paper nennt `num_nextn_predict_layers = 1`, der Drafter fährt aber
drei Stufen (`V4_DSPARK_STAGES 3`, geprobt `mtp.0`–`mtp.2`). MTP-Tiefe und
Drafter-Stufen sind verschiedene Dinge — kein Widerspruch, aber eine Stolperstelle.

Reserve: `v4_dspark_full_reserve_bytes` ([c/deepseek_v4.c:7558](../c/deepseek_v4.c)):

```c
double cache = coli_v4_dspark_cache_gb() * 1e9;   /* Default 0.45, clamp 0.15..4.0 */
double total = cache + 768.0 * 1024.0 * 1024.0;   /* ~0.56 GiB resident + Marge */
```

Beim Default sind das `450 000 000 + 805 306 368 = 1 255 306 368 B`, also
**1.255 GB = 1.169 GiB**. Die RAM-Bilanz bucht dafür rund 1.17 GiB, nicht 1.25.

Aktiv nur wenn `V4_MTP` und `V4_DRAFT` gesetzt sind
([:6301](../c/deepseek_v4.c)). Fließt über `runtime->dspark_reserve_bytes` in
`runtime_other` ([:967](../c/deepseek_v4.c)) und damit direkt gegen den
Expert-Cache.

Die residenten Tensoren, aus `g_v4ds_core`
([c/deepseek_v4_dspark.inc:78](../c/deepseek_v4_dspark.inc)):

| Tensor | Typ | Größe |
|---|---|---|
| `markov_w1`, `markov_w2` | BF16 `[129280, 256]` | je 66 MB |
| `main_proj_w` + `main_proj_s` | fp8 + f32-Scales | — |
| `confidence_proj`, `head_fn`, `head_base`, `head_scale`, `norm` | f32 | klein |
| 3 × `V4DSparkStage` Slabs | fp8-Experten-Cache | `V4_MTP_GB`, Default 0.45 GB |

### Änderung

- `markov_w1/w2` sind schlichte BF16-Matrizen → `coli_cuda_pipe_alloc` + Upload,
  Zugriff im Draft-Kernel. 132 MB, trivial.
- `main_proj_w` ist fp8 mit f32-Scales → wie in Plan 06: **Repack überspringen**
  (`v4_ds_pack_rows8`, [:113](../c/deepseek_v4_dspark.inc), ist dieselbe
  AVX2-Transposition), dann `coli_cuda_tensor_upload_g` mit `fmt=8` und
  `coli_cuda_matmul`.
- Die Stage-Slabs sind ein eigener Cache mit `nslots`/`cap_slots`/`slab_bytes`.
  Der ist **nicht** Teil dieser Phase — er bleibt im RAM. Was frei wird, ist der
  residente Teil (~0.56 GiB) plus die 768-MiB-Marge, die nur deshalb so groß ist,
  weil sie Head- und Scratch-Bedarf mit abdeckt.
- `v4_dspark_full_reserve_bytes` muss entsprechend zwischen VRAM- und RAM-Anteil
  trennen, statt eine Pauschale gegen den RAM-Plan zu buchen.

### Erwartung

Konservativ **~1.17 GiB RAM frei**, wenn DSpark aktiv ist. Bei inaktivem DSpark
ändert sich nichts — die Reserve ist dann ohnehin 0.

Zusätzlich wird DSpark schneller: der Drafter läuft dreistufig pro Token
([:1115](../c/deepseek_v4_dspark.inc), `stage_seconds[]`), und die Markov-Matvecs
über 129280 Zeilen sind derselbe bandbreitengebundene Fall wie der Head.

## Tests

- `c/tests/test_v4_head_cuda.cu` — Kernel gegen `head_bf16_dot`-Referenz über
  `vocab ∈ {128, 129280}`, `d = 4096`. Verglichen wird die **Logit-Verteilung**
  (max-abs-Abweichung, Cosine) und der Gewinner. Explizit ein Gleichstandsfall:
  zwei identische Zeilen → kleinerer Index gewinnt.
- `c/tests/test_v4_head_tier.c` — wandert der Head aufs Gerät, muss
  `coli_v4_head_cache_data` konsistent NULL liefern und DSpark denselben Weg gehen;
  ein Test, der die Kopplung festhält, damit sie nicht später auseinanderfällt.
  Geprüft wird die Gegenseitigkeit beider Accessoren über die drei Zustände
  resident / device / streamed plus die Fensterprüfung. Der Test setzt
  `head_cache` direkt und braucht deshalb **kein** CUDA-Gerät; dass ein echter
  Upload den Host-Puffer tatsächlich freigibt, deckt er nicht ab — das zeigt
  erst die `ram_tiers`-Zeile im Messergebnis oben.
- `c/tests/test_v4_dspark_reserve.c` — `dspark_reserve_bytes` steht nach
  `coli_v4_engine_open` auf `V4_MTP_GB × 1e9 + 768 MiB` und bei inaktivem
  DSpark auf 0. **Bewusst modusunabhängig:** der Upload ist transaktional, der
  Host hält den kompletten FP8-Backbone bis zum letzten `V4_DS_UPLOAD`, und die
  Markov-Tensoren bleiben ohnehin resident. Eine kleinere Reserve im
  VRAM-Modus ließe den ersten Lazy-Load seinen Upload-Peak vom Target-Cache
  borgen — die Reserve existiert genau dagegen.
- `c/tests/test_deepseek_v4_dspark_source.py` (existiert) muss grün bleiben — er
  prüft unter anderem, dass die Reserve **vor** dem Expert-Store-Open gesetzt wird
  (`test_drafter_is_lazy_and_budgeted_before_target_cache`). Die Umstellung darf
  diese Reihenfolge nicht brechen.
- End-to-end mit `V4_MTP=1 V4_DRAFT=3`: Akzeptanzrate und Tokenfolge gegen den
  CPU-Lauf.

## Abnahme

- `ram_tiers` zeigt `head=vram`, `target_cache` um ~1 GiB höher.
- Mit `V4_MTP=1 V4_DRAFT=3` ist `target_cache` um weitere ~1.2 GiB höher als vorher
  mit denselben Flags.
- Head-Logits innerhalb dokumentierter Toleranz, Tie-Breaking nach Vokabularordnung.
- DSpark-Akzeptanzrate unverändert (das ist der Beleg, dass der Drafter dieselben
  Entwürfe produziert).
- Zeit pro Token gesunken — der Head war ~24 ms davon.

## Risiken

- **Die Head-Kopplung** ist die subtilste Stelle des ganzen Branches. Wandert der
  Head allein, wird DSpark *langsamer*, ohne dass ein Test anschlägt. Deshalb der
  explizite Kopplungstest.
- **Tie-Breaking.** Wenn am echten Modell Tokens kippen, ist das ein Ergebnis, kein
  Bug — aber es muss gemessen und dokumentiert sein, nicht entdeckt werden.
- **`V4_MTP_GB`-Slabs bleiben im RAM.** Wer erwartet, dass DSpark komplett
  verschwindet, wird enttäuscht; der Expert-Cache des Drafters ist ein eigenes
  Thema.
- **0.99 GiB Upload beim Start** dauert über PCIe ~0.05 s — unkritisch, aber der
  Ladepfad darf dafür keinen 1 GiB-Host-Puffer anlegen. In Kacheln lesen und
  hochladen, wie es der Streaming-Fallback in `head_argmax` schon vormacht.
