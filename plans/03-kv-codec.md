# 03 — KV-Codec-Interface und das native Format

Voraussetzung: [00-reference.md](00-reference.md), [paper-deepseek-v4.md](paper-deepseek-v4.md)

Kein hartes Abhängigkeitspaar mit [02-flash-attention.md](02-flash-attention.md) —
[AGENTS.md](../AGENTS.md) führt 03 vor 02 in der empfohlenen Reihenfolge, und die
harte Abhängigkeitsliste dort kennt nur „02 vor 05", nicht „02 vor 03". Das
Interface unten ist trotzdem **für** den Flash-Kernel aus 02 entworfen: landet 03
zuerst, ruft der zweistufige Referenzkernel `coli_v4_kv_dot`/`_accumulate` genauso
auf wie der spätere Flash-Kernel, nur zweimal statt einmal pro Zeile.

*Commits:*
1. `feat: pluggable KV row codec for V4`
2. `feat: store V4 KV entries in their native fp8+bf16 / fp4 form`

## Ziel

Eine Zeilen-Codec-Abstraktion einziehen und mit dem **nativen Format des Modells**
füllen. Das ist die Trageschicht für Plan 04 (TurboQuant) und Plan 05 (CUDA).

**Der Kern dieses Plans hat sich durch das Paper verschoben.** Ursprünglich waren
bf16 und int8 als „risikoarme Formate" gedacht. Beide sind überflüssig: das Modell
hat ein natives Speicherformat, colibri rechnet es bereits, und es ist **besser als
beide** — bei null Präzisionsverlust.

Paper 2.3.4: *„BF16 für die RoPE-Dimensionen, FP8 für die übrigen"*, und *„FP4"*
für den Lightning Indexer.

Der Code tut genau das (`coli_fp8_activation_qdq_ref` auf Dims 0–447 mit Block 64,
`coli_bf16_round_array` auf Dims 448–511, Hadamard + `coli_fp4_activation_qdq_ref`
für den Indexer) — und legt das Ergebnis in `float`-Arrays ab. Der gespeicherte
Wert ist exakt

```
bf16_round( e4m3_decode(q) · e8m0_decode(s) )
```

also aus `(q, s)` **bit-exakt** rekonstruierbar
([c/deepseek_v4.c:10096](../c/deepseek_v4.c)).

| Strom | heute | nativ | Faktor | verlustfrei |
|---|---|---|---|---|
| Fenster-KV, CSA/HCA komprimiert | f32, 2048 B | `u8 q[448]` + `u8 e8m0[7]` + `u16 rope[64]` = **583 B** | 3.51× | **ja** |
| Indexer komprimiert | f32, 512 B | `u4 q[128]` (64 B) + `u8 e8m0[4]` = **68 B** | 7.53× | **ja** |

Das ist kein Kompromiss, den man abwägen müsste, sondern das Ende einer
Fehlbuchung. Deshalb ist `native` der **Default** dieses Plans, nicht `f32`.

**Der Teil, an dem es tatsächlich hängt, ist nicht der Codec, sondern
`context_bytes`.** Wenn die Planner-Buchhaltung weiter f32-Größen reserviert,
erreicht die Ersparnis den Expert-Cache nie. Das ist die eigentliche Arbeit dieses
Plans.

## Ausgangslage

Drei Puffer halten KV-Zeilen, alle `float *`:

| Puffer | Ort | Größe | wächst mit |
|---|---|---|---|
| `state->kv` | [c/deepseek_v4.c:1400](../c/deepseek_v4.c) ff. | `window_size × head_dim` | — (Ring) |
| `state->compressed` | dito | `compressed_count × head_dim` | `ctx / ratio` |
| Indexer `state->compressed` | [:2687](../c/deepseek_v4.c) | `count × index_head_dim` | `ctx / 4` |

**Korrektur zu einer früheren Annahme:** der Indexer-Cache ist **nicht** gedeckelt.
Der `if (state->capacity > 128)` in `coli_v4_indexer_create` betrifft nur die
Erstallokation; `coli_v4_indexer_step` ([:2837](../c/deepseek_v4.c)) verdoppelt
danach unbegrenzt:

```c
if ((position + 1) % 4 == 0 && state->count >= state->capacity) {
    int next_capacity = state->capacity * 2;
    float *grown = realloc(state->compressed, next_capacity * dimension * sizeof(*grown));
```

Er wächst also linear mit `ctx/4 × 128 dims × 4 B` — bei 128k Kontext ~16.8 MB pro
ratio-4-Layer, bei allen **21** solchen Layern ~353 MB. `context_bytes` rechnet
das korrekt mit; es ist echter Verbrauch und ein lohnendes Ziel, aber mit
Semantik-Vorbehalt (unten).

**Und dieser Puffer wird pro Token vollständig gelesen.** `coli_v4_indexer_step`
bewertet jeden Eintrag gegen 64 Heads × 128 Dims, in jedem CSA-Layer, in jedem
Token — 352 MB/Token bei 128k, 2.69 GB bei 1M (Herleitung im Indexer-Abschnitt von
[00-reference.md](00-reference.md)). Die 7.5× aus `V4_KV_INDEX=native` wirken damit
nicht nur auf den Speicherverbrauch, sondern auf den größten Lesestrom, den die
Engine hat. **Das ist der stärkste Einzelgrund für diesen Plan** — stärker als die
3.5× auf den Haupt-KV, die die Tabelle oben prominenter ausweist.

## Änderungen

### Neuer Header `c/v4_kv_codec.h`

Im Stil von [c/kv_prefix.h](../c/kv_prefix.h) und [c/expert_store.h](../c/expert_store.h):
selbsterklärender Kopfkommentar, kleine Oberfläche, keine versteckte Allokation.

```c
typedef enum {
    COLI_V4_KV_F32 = 0,   /* heutiges Verhalten, Rückfallebene */
    COLI_V4_KV_NATIVE,    /* fp8+bf16 (main) bzw. fp4 (indexer) — bit-exakt */
    COLI_V4_KV_TURBO4,    /* Plan 04, verlustbehaftet */
    COLI_V4_KV_TURBO3,
    COLI_V4_KV_TURBO2,
} ColiV4KVCodec;

/* Bytes einer kodierten Zeile. 0 = head_dim für diesen Codec nicht darstellbar. */
size_t coli_v4_kv_row_bytes(ColiV4KVCodec codec, int head_dim);

int coli_v4_kv_encode_row(ColiV4KVCodec, void *dst, const float *src, int head_dim);
int coli_v4_kv_decode_row(ColiV4KVCodec, float *dst, const void *src, int head_dim);

/* Score gegen eine kodierte Zeile, ohne sie zu materialisieren — der Pfad, den
 * der Flash-Kernel aus Plan 02 nimmt. */
float coli_v4_kv_dot(ColiV4KVCodec, const float *query, const void *row, int head_dim);

/* Gewichtete Akkumulation: acc += probability * decode(row). */
void coli_v4_kv_accumulate(ColiV4KVCodec, float *acc, float probability,
                           const void *row, int head_dim);

const char  *coli_v4_kv_codec_name(ColiV4KVCodec);
ColiV4KVCodec coli_v4_kv_codec_from_env(const char *variable, int head_dim,
                                        ColiV4KVCodec fallback);
```

`coli_v4_kv_dot` und `coli_v4_kv_accumulate` sind die einzigen zwei Stellen, an
denen ein Attention-Kernel eine Zeile anfasst — ob zweistufige Referenz (heute)
oder Flash (Plan 02). Für `COLI_V4_KV_F32` sind sie ein direkter Dot bzw. `axpy`
ohne Umweg. Das ist der Grund, warum das Interface so geschnitten ist, unabhängig
davon, welcher der beiden Pläne zuerst landet.

`NATIVE` braucht zwei Varianten, weil Haupt- und Indexerzeilen unterschiedlich
quantisiert sind. Die Unterscheidung läuft über `head_dim` — 512 heißt
Hauptzeile, `index_head_dim` heißt Indexerzeile — oder sauberer über ein
`ColiV4KVStream`-Flag im State.

Zeilengrößen:

| Codec | `head_dim` | Layout | Bytes | vs f32 |
|---|---|---|---|---|
| `F32` | 512 | `float[512]` | 2048 | 1× |
| `NATIVE` | 512 | `u8 q[448]` + `u8 e8m0[7]` + `u16 rope[64]` | **583** | 3.51× |
| `F32` | 128 | `float[128]` | 512 | 1× |
| `NATIVE` | 128 | `u4 q[128]` (64 B) + `u8 e8m0[4]` | **68** | 7.53× |

Dekodierung der Hauptzeile — muss exakt dem Erzeugungspfad entsprechen, sonst ist
die Bit-Gleichheit weg:

```c
/* Dims 0..447: fp8, ein E8M0-Scale je 64er-Block, danach bf16-Rundung */
for (int b = 0; b < 7; b++) {
    float scale = coli_e8m0_decode(row->e8m0[b]);
    for (int i = 0; i < 64; i++)
        dst[b*64 + i] = coli_bf16_round(coli_e4m3fn_decode(row->q[b*64 + i]) * scale);
}
/* Dims 448..511: bereits bf16-gerundet, verlustfrei in uint16 */
for (int i = 0; i < 64; i++) dst[448 + i] = coli_bf16_decode(row->rope[i]);
```

Die Encode-Seite ist ein Umweg, keine Neuquantisierung: `coli_fp8_activation_qdq_ref`
([c/deepseek_v4.c:10076](../c/deepseek_v4.c)) liefert schon **heute** das
dequantisierte `output` und den `scales`-Byte zurück, wirft aber den fp8-Code `q`
selbst weg — der lebt nur als lokale Variable im Loop-Body
([:10096](../c/deepseek_v4.c)). Der Codec braucht ihn trotzdem nicht neu zu
*schätzen*: weil `scale` eine Zweierpotenz ist (E8M0), ist
`coli_e4m3fn_encode(output[i] / scale)` exakt der Code, der `output[i]` erzeugt
hat — keine Rundung, keine zweite Wahl. `output` und `scales` kommen entweder
als zusätzlicher `codes`-Out-Parameter aus der qdq-Funktion, oder werden dort neu
kodiert, wo der Codec sie abgreift; beides ist bit-exakt, ersteres spart die
Rückrechnung. Das ist auch der Grund, warum die Bit-Gleichheit nicht bloß
plausibel, sondern konstruktiv ist.

Für die Indexerzeile analog mit `coli_fp4_activation_qdq_ref`, Block 32, und der
Hadamard-Rotation davor (`coli_hadamard_bf16_ref`) — die bleibt unverändert im
Compressor, der Codec sieht nur das Ergebnis.

### `ColiDeepSeekV4WindowAttentionState` umstellen

In **allen drei Kopien** ([:1400](../c/deepseek_v4.c), [:1798](../c/deepseek_v4.c),
[:4572](../c/deepseek_v4.c)):

```c
struct ColiDeepSeekV4WindowAttentionState {
    int window_size, head_dim, layer, ratio;
    ColiV4KVCodec codec;            /* neu */
    size_t        row_bytes;        /* neu, = coli_v4_kv_row_bytes(codec, head_dim) */
    void   *kv;                     /* war float * */
    void   *compressed;             /* war float * */
    ...
};
```

Betroffene Stellen je Kopie:

| Was | ATTENTION | BATCH | TRANSACTION | Batch-Pfad |
|---|---|---|---|---|
| `create` (calloc `sliding_window × head_dim`) | 1413 | 1811 | 4585 | — |
| `reset` (memset) | 1434 | 1832 | 4606 | — |
| `destroy` (free) | 1445 | 1843 | 4617 | — |
| `prepare_compressed_state` (calloc cap 16) | 1460 | 1858 | 4632 | — |
| `grow_compressed_state` (realloc ×2) | 1481 | 1879 | 4653 | — |
| Compressor-Write | 1597 | 1995 | 4769 | 2247 |
| Fenster-Write | 1670 | 2068 | 4842 | 2328 |

Alle `× head_dim * sizeof(float)` werden `× row_bytes`.

**Compressor-Write braucht einen Zwischenpuffer.** `coli_v4_compressor_step`
schreibt heute direkt in den Cache-Slot:

```c
result = coli_v4_compressor_step(state->compressor,
    state->compressed + (size_t)state->compressed_count * head_dim, ...);
```

Neu: in einen `float scratch[head_dim]` schreiben lassen, dann
`coli_v4_kv_encode_row` in den Slot. Der Scratch gehört zum State, nicht zum
Aufruf — eine Allokation pro Layer, nicht pro Token.

### Compressor und Indexer stehen **doppelt** im Amalgam

**Die Tabelle oben ist nicht vollständig, und die fehlende Hälfte ist die
gefährliche.** Neben den drei Attention-Kopien sind auch Compressor und Indexer
dupliziert (Units `COMPRESSOR`/`COMPRESSOR_SNAPSHOT` und
`INDEXER`/`INDEXER_SNAPSHOT`, byte-identisch unter `#define`-Umbenennung — siehe
die Tabelle in [00-reference.md](00-reference.md)):

| Funktion | Kopie 1 | Kopie 2 | Warum dieser Plan sie anfasst |
|---|---|---|---|
| `coli_v4_compressor_step` | [:2538](../c/deepseek_v4.c) | [:4042](../c/deepseek_v4.c) | Zwischenpuffer statt Direkt-Write |
| Der QDQ-Block darin | [:2643](../c/deepseek_v4.c) | [:4147](../c/deepseek_v4.c) | liefert `(q, scales)` an den Codec |
| `coli_v4_indexer_step` | [:2827](../c/deepseek_v4.c) | [:4390](../c/deepseek_v4.c) | `state->compressed` wird `void *` |
| Wachstums-`realloc` darin | [:2837](../c/deepseek_v4.c) | [:4400](../c/deepseek_v4.c) | `× dimension` wird `× row_bytes` |

Eine frühere Fassung dieses Plans nannte nur die jeweils erste Spalte. Der
Source-Sync-Test aus [02-flash-attention.md](02-flash-attention.md) deckt diese
Quellen nach der dortigen Korrektur mit ab — er fängt aber nur *Drift zwischen*
den Kopien, nicht ein einheitlich falsches Muster in beiden. Die Tabelle
abarbeiten und abhaken.

Der QDQ-Block ist dabei **eine** Stelle für beide Ströme, nicht zwei:
`coli_v4_compressor_step` verzweigt intern über `state->rotate_fp4` zwischen
fp8/Block 64 (Haupt-KV) und Hadamard+fp4/Block 32 (Indexer). Wer dort einen
`codes`-Out-Parameter einzieht, muss beide Zweige bedienen.

### Snapshot

`coli_v4_attention_snapshot_create` / `_restore` ([:4963](../c/deepseek_v4.c)) kopieren
Rohbytes. Nur `sizeof(float)` → `row_bytes`, plus `codec` in den Snapshot
aufnehmen und beim Restore vergleichen (ein Snapshot aus einer anderen
Codec-Konfiguration darf nicht stillschweigend passen):

```c
if (state->codec != snapshot->codec || state->row_bytes != snapshot->row_bytes)
    return -1;
```

### `context_bytes` — die entscheidende Stelle

[c/deepseek_v4.c:909](../c/deepseek_v4.c). Heute:

```c
total  = num_hidden_layers * sliding_window * head_dim * sizeof(float);
total += compressed * head_dim * sizeof(float);
if (ratio == 4) total += compressed * index_head_dim * sizeof(float);
```

Neu, mit den tatsächlich gewählten Codecs:

```c
size_t kv_row    = coli_v4_kv_row_bytes(kv_codec,    config->head_dim);
size_t index_row = coli_v4_kv_row_bytes(index_codec, config->index_head_dim);

total  = (uint64_t)num_hidden_layers * sliding_window * kv_row;
total += compressed * kv_row;
if (ratio == 4) total += compressed * index_row;
```

Die Codec-Wahl muss dafür **vor** `build_runtime_plan` feststehen. Sie hängt nur an
Env und `config`, also in `coli_v4_engine_open` ([:6504](../c/deepseek_v4.c) ff.)
auflösen und in `ColiDeepSeekV4RuntimeOptions` ablegen — dort liegen
`context_tokens`, `memory_limit_bytes` und `dspark_reserve_bytes` schon.

**Ohne diesen Schritt ist der ganze Plan wirkungslos.**

### Indexer-Codec

```
V4_KV        = f32 | native | turbo4 | turbo3 | turbo2      (Default native)
V4_KV_INDEX  = dito                                          (Default native)
```

**Default `native` für beide.** Das ist eine bewusste Abweichung von der sonstigen
Regel dieses Branches („Default bleibt überall die heutige Semantik"), und sie ist
gedeckt: die README garantiert, dass die Default-Policy die Modellpräzision nicht
**still ändert** — `native` ändert sie überhaupt nicht. Die Bit-Gleichheit steht
unten im Abnahmekriterium; solange sie hält, ist das kein Politikwechsel, sondern
eine Fehlerkorrektur. Fällt sie, ist der Default `f32` und der Grund gehört ins
Changelog.

`f32` bleibt als Rückfallebene für die Diagnose und für `head_dim`-Werte, die kein
Codec darstellt.

**Der Indexer bekommt trotzdem einen eigenen Knopf** — nicht wegen `native`, das ist
verlustfrei, sondern für Plan 04. Er entscheidet per Top-k, *welche* Tokens die
Attention überhaupt sieht (`coli_v4_indexer_step` → `descending_score` →
`compressed_indices`). Sobald dort ein verlustbehafteter Codec steht, ändert ein
Quantisierungsfehler die **Auswahl**, nicht die Präzision — das ist Router-Semantik
und muss getrennt schaltbar bleiben, nie implizit mitgezogen. Gehört so in den
Kopfkommentar.

## Tests

- `c/tests/test_v4_kv_codec.c`, neue Make-Regel:
  - **Bit-Gleichheit `native`** — der wichtigste Test des Plans. Aus einem
    Zufallsvektor über `coli_fp8_activation_qdq_ref` + `coli_bf16_round_array` den
    heutigen f32-Zustand erzeugen, daraus `native` kodieren, dekodieren und auf
    `memcmp`-Gleichheit prüfen. Dasselbe für die Indexerzeile über
    `coli_hadamard_bf16_ref` + `coli_fp4_activation_qdq_ref`.
  - `row_bytes` für alle Codecs × `head_dim` ∈ {32, 128, 512, 4096}; 0 wo nicht
    darstellbar.
  - Round-trip `encode → decode`: f32 und `native` **bit-identisch**.
  - `coli_v4_kv_dot` gegen `decode` + manuellen Dot: Abweichung `< 1e-5` relativ.
  - `coli_v4_kv_accumulate` gegen `decode` + manuelles `axpy`.
  - `coli_v4_kv_codec_from_env`: gültige Namen, Müll → Fallback, nicht darstellbares
    `head_dim` → Fallback **mit** Warnung auf stderr.
  - Randfälle: Nullzeile (E8M0-Scale-Untergrenze `1e-4f` im QDQ!), Zeile mit einem
    einzigen Ausreißer, Sättigung bei `±448` (der fp8-Clamp), `±FLT_MAX`.
- `c/tests/test_v4_context_bytes.c` — `context_bytes` skaliert mit dem Codec:
  `native` gibt für `head_dim=512` exakt `583/2048` von f32, für 128 exakt `68/512`.
  Das ist der Test, der die Planner-Kopplung absichert.
- `c/tests/test_v4_attention_source.py` aus Plan 02 muss weiter grün sein.
- `make -C c deepseek-v4-tiny-check` mit `V4_KV=f32` und `native`.

## Abnahme

- **`V4_KV=native` ist bit-identisch zu `V4_KV=f32`** — nicht nur token-identisch.
  Am Tiny-Fixture über einen Vergleich der Attention-Ausgaben, nicht nur der Tokens.
  Das ist das Kriterium, an dem der Default hängt.
- `V4_KV=f32` ist token-identisch zu Plan 02 — der Codec-Umbau allein ändert nichts.
- `ram_tiers` zeigt mit `native` ein messbar höheres `target_cache` als mit `f32`
  bei gleichem `CTX` — der Beweis, dass `context_bytes` mitzieht. Erwartung bei
  128k: KV-Anteil von 1.68 auf 0.43 GiB.
- Snapshot-Round-trip funktioniert je Codec (`coli_v4_attention_snapshot_*`).

## Risiken

- **`void *` verliert Typsicherheit.** Ein vergessenes `× row_bytes` kompiliert und
  liest daneben. Deshalb: `row_bytes` im State speichern statt an jeder Stelle neu
  auszurechnen, und in `create` einmal
  `assert(row_bytes == coli_v4_kv_row_bytes(codec, head_dim))`.
- **Die sieben Umbaustellen × drei Attention-Kopien** sind 21 Änderungen, plus
  Batch-Pfad, **plus** die vier Stellen × zwei Kopien in Compressor und Indexer.
  Insgesamt ~30 Stellen. Der Source-Sync-Test fängt Drift zwischen den Kopien, aber
  nicht ein einheitlich falsches Muster. Beide Tabellen oben abarbeiten und abhaken.
- **Codec-Wahl vor dem Planner** verschiebt Initialisierungsreihenfolge. Wenn
  `coli_v4_engine_open` die Config erst später lädt, muss die Auflösung dorthin
  wandern, wo `config` gültig ist — nicht raten, den Pfad lesen.
- **Die Bit-Gleichheit steht und fällt mit dem Erzeugungspfad.** Sie gilt nur,
  solange der Compressor und `attention_token_impl` weiter `qdq` + `bf16_round` in
  genau dieser Reihenfolge anwenden. Ändert dort jemand etwas, kippt sie
  stillschweigend. Der Test aus dem Abnahmekriterium ist die einzige Absicherung —
  er gehört in die Gates, nicht in ein einmaliges Skript.
- **Der fp8-Pfad hat Sonderfälle.** `coli_fp8_activation_qdq_ref`
  ([:10086](../c/deepseek_v4.c)) klemmt das Blockmaximum auf `≥ 1e-4f` und den
  Exponenten auf `[-127, 127]`, und sättigt Werte bei `±448`. Eine Nullzeile
  bekommt dadurch einen von Null verschiedenen Scale. Der Decoder muss das nicht
  nachbilden — er liest ja den gespeicherten Scale — aber die Tests müssen diese
  Zeilen enthalten, sonst prüft man nur den bequemen Bereich.
- **`head_dim = 512` ist im Codec verdrahtet**, weil `448 = 512 − 64` und die sieben
  64er-Blöcke daraus folgen. Für andere `head_dim` muss `row_bytes` das aus
  `head_dim` und `qk_rope_head_dim` ableiten statt Konstanten zu benutzen — sonst
  bricht das Tiny-Fixture (`head_dim=32`, `rope_dim=16`) unbemerkt.
- **Zwei Streams, ein Enum.** `NATIVE` bedeutet für 512-dim-Zeilen etwas anderes als
  für 128-dim-Zeilen. Die Unterscheidung über `head_dim` zu erschließen ist
  fragil — besser ein explizites Stream-Flag im State, auch wenn es ein Feld mehr ist.
