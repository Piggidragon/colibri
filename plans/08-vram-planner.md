# 08 — VRAM-Planner und 4070-Profil

Voraussetzung: [00-reference.md](00-reference.md), [06](06-dense-vram.md), [07](07-head-dspark-vram.md)

*Commits:*
1. `feat: VRAM tier planner for V4`
2. `docs: 32GB + 12GB tuning profile for DeepSeek V4`

## Ziel

Die Stufen aus 05–07 sind bis hierher einzeln verdrahtet und entscheiden je für
sich. Dieser Plan macht daraus **eine** Planung mit fester Priorität und
stufenweisem Rückfall, plus die dokumentierte Empfehlung für 32 GB + 12 GB.

Ohne das ist die Konfiguration ein Ratespiel: 12 GB sind knapp genug, dass ein
zweiter Monitor oder ein Browser mit GPU-Beschleunigung die Reserve auffrisst, und
dann muss der Lauf **degradieren, nicht scheitern**.

## Ausgangslage

Der RAM-Planner ist zweistufig und sauber getrennt:

- `coli_v4_resource_plan_compute` ([c/deepseek_v4.c:693](../c/deepseek_v4.c)) —
  Systemreserve, Runtime-Reserve, Mindest-Expert-Cache, Slots.
- `coli_v4_resident_tier_plan` ([:763](../c/deepseek_v4.c)) — entscheidet
  Dense-Residenz gegen das verbleibende Budget, über den Helfer
  `resident_tiers_fit` ([:756](../c/deepseek_v4.c)).
- `coli_v4_expert_store_open_planned` ([:1000](../c/deepseek_v4.c)) — setzt beides
  zusammen, entscheidet zusätzlich die Head-Residenz und druckt `ram_tiers`.

Die Head-Entscheidung dort ist bemerkenswert direkt ([:1033](../c/deepseek_v4.c)):

```c
int requested_head = -1;
int resident_head = safe_payload >= plan.minimum_expert_bytes + head_bytes + 256 * MIB;
if (requested_head == 0) resident_head = 0;
if (requested_head == 1 && !resident_head) { ... }
```

`requested_head` ist fest `-1`, die beiden Zweige sind tot — offensichtlich als
Platzhalter für genau die explizite Stufensteuerung gedacht, die dieser Plan
einzieht.

Auf der GPU-Seite steht `coli_cuda_mem_info(device, &free_bytes, &total_bytes)`
([c/backend_cuda.h](../c/backend_cuda.h)) bereit.

## Commit 1 — Der Planner

### Struktur

`coli_v4_vram_tier_plan` als Spiegel von `coli_v4_resident_tier_plan`, in derselben
`RESOURCE_PLAN`-Unit, mit derselben Fehlerkonvention (`plan_error`):

```c
typedef enum { COLI_V4_TIER_VRAM = 0, COLI_V4_TIER_RAM, COLI_V4_TIER_STREAMED }
    ColiV4TierLocation;

typedef struct {
    uint64_t vram_available_bytes;   /* aus coli_cuda_mem_info, minus Reserve */
    uint64_t dense_bytes;
    uint64_t head_bytes;
    uint64_t dspark_bytes;
    uint64_t kv_bytes;               /* aus context_bytes(), Codec-abhängig */
    uint64_t workspace_bytes;
} ColiV4VramTierInputs;

typedef struct {
    ColiV4TierLocation dense, head, dspark, kv;
    uint64_t vram_used_bytes;
    uint64_t ram_spilled_bytes;      /* was doch im RAM landet */
} ColiV4VramTierPlan;
```

### Priorität

Feste Reihenfolge, jede Stufe einzeln rückfallfähig:

| # | Posten | Warum diese Position |
|---|---|---|
| 1 | **KV** | Ohne KV auf dem Gerät ist der Attention-Kernel aus Plan 05 sinnlos, und es ist der kleinste Posten (0.35 GiB bei turbo3). |
| 2 | **Dense** | Größter RAM-Gewinn pro VRAM-Byte (6.27 GiB für 6.27 GiB) und macht Q device-resident. |
| 3 | **Head** | 1.06 GiB, größter Rechenzeitgewinn, aber unabhängig vom Rest. |
| 4 | **DSpark** | Nur wenn aktiv; muss mit dem Head zusammen wandern (siehe Plan 07). |

**Kopplung erzwingen:** landet der Head im RAM, muss DSpark auch im RAM bleiben —
sonst verliert der Drafter seinen residenten Head. Das ist eine Invariante des
Planners, kein Aufrufervertrag:

```c
if (plan->head != COLI_V4_TIER_VRAM) plan->dspark = plan->head;
```

### VRAM-Reserve

`coli_cuda_mem_info` liefert freien Speicher zum Zeitpunkt des Aufrufs — auf einer
Karte mit Display schwankt der. Analog zur RAM-Systemreserve
([:707](../c/deepseek_v4.c)) eine Reserve abziehen:

```c
uint64_t reserve = free_bytes / 8;
if (reserve < 256 * MIB)  reserve = 256 * MIB;
if (reserve > 1024 * MIB) reserve = 1024 * MIB;
```

Über `V4_VRAM_RESERVE_MB` überschreibbar, mit Clamp — Muster
`coli_v4_dspark_cache_gb` ([:6288](../c/deepseek_v4.c)).

### Rückkopplung in den RAM-Plan

Was auf der GPU liegt, darf im RAM-Plan nicht mehr auftauchen. In
`coli_v4_expert_store_open_planned` ([:1000](../c/deepseek_v4.c)):

```c
uint64_t safe_payload = plan.planner_available_bytes - fixed
                      - (tiers.dense == COLI_V4_TIER_RAM ? dense_bytes : 0);
```

und entsprechend für Head und DSpark. **Das ist die Stelle, an der der ganze
Branch seinen Nutzen realisiert** — ohne sie bleibt der Expert-Cache gleich groß,
egal was auf der GPU liegt.

Dasselbe gilt für `runtime_other` in `build_runtime_plan`
([:961](../c/deepseek_v4.c)): `context_bytes(...)` gehört nur dann hinein, wenn das
KV im RAM liegt.

### Report

Die `ram_tiers`-Zeile ([:1057](../c/deepseek_v4.c)) um eine `vram_tiers`-Zeile
ergänzen — gleiche Form, gleicher Ort, damit beides zusammen im Log steht:

```
ram_tiers  available=28.00GiB dense=vram target_slots=52 target_cache=29.1GiB head=vram projected=27.4GiB
vram_tiers free=11.20GiB reserve=1.00GiB kv=vram(0.59GiB) dense=vram(6.27GiB) head=vram(1.06GiB) dspark=vram(1.25GiB) used=9.17GiB
```

Ohne diesen Report ist nicht nachvollziehbar, warum eine Konfiguration schnell oder
langsam ist. Er ist Teil des Features, nicht Beiwerk.

## Commit 2 — Das Profil

`docs/deepseek-v4-tuning-32gb.md` (aus Plan 01 angelegt) wird zum vollständigen
Profil ausgebaut:

```bash
RAM_GB=28 \
CTX=131072 \
V4_SCRATCH_MB=128 \
V4_KV=turbo3 \
V4_KV_INDEX=turbo3 \
V4_VRAM=1 \
V4_MTP=1 V4_DRAFT=3 \
SNAP=/pfad/DeepSeek-V4-Flash ./deepseek_v4
```

Mit Begründung pro Knopf und einer Tabelle „was passiert, wenn du X weglässt".
Dazu die gemessenen Zahlen: `ram_tiers`/`vram_tiers`, tok/s, Expert-Hit-Rate,
Zeit pro Token aufgeschlüsselt.

Verlinkung aus `docs/deepseek-v4.md` unter „Memory policy".

**`V4_KV_INDEX=turbo3` ist die einzige Empfehlung mit Semantik-Vorbehalt** — sie
quantisiert den Indexer, der die Top-k-Auswahl trifft. Sie gehört nur ins Profil,
wenn die Messung aus Plan 04 zeigt, dass die Tokenfolge stabil bleibt. Sonst
`f32` empfehlen und den Unterschied ausweisen.

## Tests

- `c/tests/test_v4_vram_tier.c` — neue Make-Regel:
  - Alles passt → alle vier Posten `VRAM`.
  - Knapp: KV + Dense passen, Head nicht → `head == RAM` **und** `dspark == RAM`
    (Kopplungsinvariante).
  - Sehr knapp: nur KV passt.
  - Gar kein VRAM (`vram_available_bytes == 0`) → alle `RAM`/`STREAMED`, **kein
    Fehler**. Das ist der Fall „keine GPU im System".
  - Reserve-Clamp: unten, oben, Müll-Env.
  - Überlaufschutz wie im RAM-Planner (`add_u64`/`multiply_u64` benutzen, nicht
    roh rechnen).
- `c/tests/test_v4_tier_feedback.c` — was in VRAM liegt, taucht nicht in
  `safe_payload` auf; `target_cache` steigt entsprechend. Der Test, der den
  eigentlichen Nutzen absichert.
- End-to-end: das Profil oben läuft, `vram_tiers` plausibel, Tokenfolge gegen einen
  CPU-Referenzlauf.

## Abnahme

- Auf dem 4070 zeigt `vram_tiers` alle vier Posten als `vram` bei `used < 10 GiB`.
- `ram_tiers` zeigt `target_cache ≈ 29 GiB` gegen ~15 GiB auf `main`.
- Künstlich verkleinertes VRAM-Budget (`V4_VRAM_RESERVE_MB=8000`) degradiert
  stufenweise statt zu scheitern.
- Ohne GPU im System läuft alles wie auf `main`.
- Das dokumentierte Profil ist reproduzierbar — die Zahlen im Doc stammen aus einem
  echten Lauf, nicht aus dieser Planung.

## Risiken

- **`coli_cuda_mem_info` ist eine Momentaufnahme.** Zwischen Planung und Upload
  kann anderer VRAM-Druck entstehen. Der Upload muss deshalb Fehler tolerieren und
  die betroffene Stufe zur Laufzeit auf RAM zurückstufen — Planung allein genügt
  nicht.
- **Die Kopplungsinvariante Head↔DSpark** ist leicht zu verlieren, wenn später eine
  fünfte Stufe dazukommt. Als Kommentar an die Invariante schreiben, warum sie
  existiert (Plan 07), nicht nur *dass*.
- **Die Rückkopplung in `safe_payload`** ist der Punkt, an dem der Branch nützlich
  wird oder nicht. Wenn `target_cache` nach allen Phasen nicht steigt, ist genau
  hier der Fehler — nicht in den Kerneln.
- **Das Profil altert.** Sobald sich Codec-Defaults oder Reserven ändern, stimmen
  die Zahlen im Doc nicht mehr. Datum und Commit-Hash dazuschreiben.
