# 08 — VRAM-Planner und 4070-Profil

Voraussetzung: [00-reference.md](00-reference.md), [06](06-dense-vram.md), [07](07-head-dspark-vram.md)

*Commits:*
1. `feat: VRAM tier planner for V4`
2. `docs: 32GB + 12GB tuning profile for DeepSeek V4`

## Ziel

Die Stufen aus 05–07 sind bis hierher einzeln verdrahtet und entscheiden je für
sich. Dieser Plan macht daraus **eine** Planung mit fester Priorität und
stufenweisem Rückfall, plus die dokumentierte Empfehlung für 32 GB + 12 GB.

Ohne das ist die Konfiguration ein Ratespiel. Die Maschine läuft headless, das
Budget ist also vorhersehbar (~11.7 GiB) — aber vorhersehbar heißt nicht
unbegrenzt: bei 1M Kontext ist die Reserve knapp, und ein fehlgeschlagener Upload
muss **degradieren, nicht scheitern**.

## Ausgangslage

Der RAM-Planner ist zweistufig und sauber getrennt:

- `coli_v4_resource_plan_compute` ([c/deepseek_v4.c:693](../c/deepseek_v4.c)) —
  Systemreserve, Runtime-Reserve, Mindest-Expert-Cache, Slots.
- `coli_v4_resident_tier_plan` ([:763](../c/deepseek_v4.c)) — entscheidet
  Dense-Residenz gegen das verbleibende Budget, über den Helfer
  `resident_tiers_fit` ([:756](../c/deepseek_v4.c)).
- `coli_v4_expert_store_open_planned` ([:1001](../c/deepseek_v4.c)) — setzt beides
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
    uint64_t kv_bytes;               /* nur Attention-KV; Indexer bleibt CPU */
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
| 1 | **KV** | Ohne KV auf dem Gerät ist der Attention-Kernel aus Plan 05 sinnlos. Der einzige Posten, der mit dem Kontext wächst — 0.388 GiB (`native`, 128k) bis 3.08 GiB (`native`, 1M). Der Lightning-Indexer bleibt CPU-seitig und gehört nicht in diesen VRAM-Posten. |
| 2 | **Dense** | Größter RAM-Gewinn pro VRAM-Byte (6.27 GiB für 6.27 GiB) und macht Q device-resident. |
| 3 | **Head** | 0.99 GiB, größter Rechenzeitgewinn, aber unabhängig vom Rest. |
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
([:713](../c/deepseek_v4.c)) eine Reserve abziehen:

```c
uint64_t reserve = free_bytes / 8;
if (reserve < 256 * MIB)  reserve = 256 * MIB;
if (reserve > 1024 * MIB) reserve = 1024 * MIB;
```

Über `V4_VRAM_RESERVE_MB` überschreibbar, mit Clamp — Muster
`coli_v4_dspark_cache_gb` ([:6288](../c/deepseek_v4.c)).

**Diese Reserve ist Teil des Budgets in [00-reference.md](00-reference.md), nicht
eine Zutat obendrauf.** Auf dem 4070 greift der obere Clamp: `11.7/8 = 1.46` → volle
1.0 GiB, also **10.7 GiB für die vier Stufen**, nicht 11.7. Eine frühere Fassung
des VRAM-Budgets in 00 rechnete ohne sie und kam deshalb auf ~3.5 GiB KV-Spielraum
statt der tatsächlichen ~2.6 — mit der Folge, dass `native` bei 1M dort noch zu
passen schien. Wer diese Reserve ändert, korrigiert das Budget in 00 mit; sonst
planen die beiden Dokumente gegeneinander.

### Rückkopplung in den RAM-Plan

Was auf der GPU liegt, darf im RAM-Plan nicht mehr auftauchen. In
`coli_v4_expert_store_open_planned` ([:1001](../c/deepseek_v4.c)):

```c
uint64_t safe_payload = plan.planner_available_bytes - fixed
                      - (tiers.dense == COLI_V4_TIER_RAM ? dense_bytes : 0);
```

und entsprechend für Head und DSpark. **Das ist die Stelle, an der der ganze
Branch seinen Nutzen realisiert** — ohne sie bleibt der Expert-Cache gleich groß,
egal was auf der GPU liegt.

Dasselbe gilt für `runtime_other` in `build_runtime_plan`
([:961](../c/deepseek_v4.c)): Die bisher gemeinsame `context_bytes(...)`-Zahl muss
in Attention- und Indexer-Anteil getrennt werden. Nur der Attention-Anteil darf
entfallen, wenn der Planner **exklusive** Device-Eigentümerschaft aktiviert und
den Host-Shadow aus Phase 05 entfernt; der Indexer-Anteil bleibt immer im RAM.

### Report

Die `ram_tiers`-Zeile ([:1057](../c/deepseek_v4.c)) um eine `vram_tiers`-Zeile
ergänzen — gleiche Form, gleicher Ort, damit beides zusammen im Log steht:

```
ram_tiers  available=28.00GiB dense=vram target_slots=21 target_cache=~11.2GiB head=vram projected=~11.5GiB
vram_tiers free=11.70GiB reserve=1.00GiB kv=vram(0.39GiB) dense=vram(6.27GiB) head=vram(0.99GiB) dspark=vram(0.56GiB) used=8.51GiB
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
V4_KV=native \
V4_KV_INDEX=native \
V4_VRAM=1 \
V4_MTP=1 V4_DRAFT=3 \
SNAP=/pfad/DeepSeek-V4-Flash ./deepseek_v4
```

Mit Begründung pro Knopf und einer Tabelle „was passiert, wenn du X weglässt".
Dazu die gemessenen Zahlen: `ram_tiers`/`vram_tiers`, tok/s, Expert-Hit-Rate,
Zeit pro Token aufgeschlüsselt.

Verlinkung aus `docs/deepseek-v4.md` unter „Memory policy".

Bei `CTX=131072` reicht `native` — verlustfrei, 0.388 GiB Attention-KV, ~2.19 GiB
übrig. Bis einschließlich 512k bleibt das so (1.54 GiB KV, ~1.04 GiB übrig).
**Erst beim 1M-Profil wird turbo3 Pflicht**: `native` bräuchte dort 3.08 GiB und
bekommt ~2.58 GiB.
Dann gilt der Semantik-Vorbehalt für `V4_KV_INDEX`: turbo auf dem Indexer
quantisiert die Top-k-Auswahl, also Router-Semantik. Ein zweites Profil für 1M
gehört ins Doc, mit den Messungen aus Plan 04 als Beleg, dass die Tokenfolge
stabil bleibt — sonst dort `V4_KV_INDEX=native` lassen (68 B statt 50 B pro Zeile,
das Budget trägt es) und den Unterschied ausweisen.

## Der OOM-Pfad muss geprobt werden, nicht angenommen

Headless heißt: das VRAM-Budget ist **vorhersehbar**, nicht unbegrenzt. Der
Planner rechnet mit einer Momentaufnahme aus `coli_cuda_mem_info`; zwischen
Planung und dem letzten Upload kann trotzdem etwas schiefgehen — ein zweiter
Prozess, ein zu optimistisch geschätzter Workspace, ein Kontext, der länger wird
als geplant.

**Zwei getrennte Pfade, beide müssen degradieren statt abzustürzen:**

1. **Planungszeit** — das Budget reicht nicht. Der Planner stuft vorher zurück.
   Das ist der einfache Fall und der Test dafür ist reine Arithmetik.
2. **Laufzeit** — die Planung sagte VRAM, der Upload scheitert trotzdem
   (`coli_cuda_tensor_upload_g` gibt 0, `v4_cuda_kv_alloc` gibt NULL). Dann muss
   die betroffene Stufe **zur Laufzeit** auf RAM zurückfallen, einmal loggen und
   weiterlaufen.

Der zweite Pfad ist der, der in der Praxis zuschlägt, und der, den man ohne
Absicht nie durchläuft. Deshalb braucht es einen Weg, ihn zu erzwingen:

```
V4_VRAM_LIMIT_MB=<n>    kappt das gemeldete freie VRAM künstlich
V4_VRAM_FAIL_AT=<stufe> lässt den Upload dieser Stufe absichtlich scheitern
                        (kv|dense|head|dspark), nur unter COLI_V4_TEST_HOOKS
```

`V4_VRAM_LIMIT_MB` ist auch ohne Test nützlich: damit lässt sich das Verhalten
auf einer kleineren Karte durchspielen, ohne eine zu besitzen.
`V4_VRAM_FAIL_AT` folgt dem Muster der vorhandenen Fehlerinjektion
(`coli_v4_test_fail_expert_store_open`,
[c/deepseek_v4_internal.h:716](../c/deepseek_v4_internal.h)) und ist in
Produktionsobjekten nicht einkompiliert.

**Abnahme dafür:** Für **jede** der vier Stufen einmal den Laufzeitfehler
injizieren und belegen, dass der Lauf durchläuft und dieselben Tokens liefert wie
ein reiner CPU-Lauf. Nicht nur „stürzt nicht ab" — dieselben Tokens. Ein
Rückfall, der still falsch rechnet, ist schlimmer als ein Absturz.

**Reibung mit 06/07:** Die Pläne 06 und 07 räumen bereits ein, dass eine
GPU-Reduktion die Akkumulationsreihenfolge ändert und dadurch am Tiny-Fixture
ein Token kippen kann (Toleranz statt Bit-Gleichheit). Ein RAM-Fallback, der
*zur Laufzeit* von VRAM auf CPU zurückfällt, wechselt damit unter Umständen
selbst die Akkumulationsreihenfolge — „dieselben Tokens" ist dann nur
erreichbar, wenn der Fallback exakt den CPU-Pfad nimmt (nicht etwa einen
zweiten, CPU-seitigen Reduktionspfad mit eigener Rundung). Das ist umsetzbar,
aber es gehört vor der Implementierung entschieden, nicht erst beim
Abgleich der beiden Abnahmekriterien.

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

- Auf dem 4070 (headless) zeigt `vram_tiers` alle vier Posten als `vram` bei
  `used ≈ 8.51 GiB` für `CTX=131072`/`native`, `reserve = 1.00 GiB`.
- Bei `CTX=131072` liegt `target_cache` nach Slot-Rundung innerhalb der in
  [00-reference.md](00-reference.md) hergeleiteten Obergrenze von 11.51 GiB
  (rund 21–22 Slots). Der Gewinn wird gegen einen echten `main`-Lauf mit
  identischem `CTX` ausgewiesen; die früher genannten 27.6/14.4 GiB beruhten auf
  der inzwischen korrigierten 64-Token-State-Reserve und sind kein Gate mehr.
- Künstlich verkleinertes VRAM-Budget (`V4_VRAM_LIMIT_MB`) degradiert stufenweise
  statt zu scheitern — durchgespielt für 8, 6, 4, 2 und 0 GiB.
- **Laufzeit-OOM je Stufe injiziert** (`V4_VRAM_FAIL_AT=kv|dense|head|dspark`):
  läuft durch und liefert **dieselben Tokens** wie der reine CPU-Lauf.
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
