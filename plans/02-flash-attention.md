# 02 — Flash Attention auf der CPU

Voraussetzung: [00-reference.md](00-reference.md)

*Commits:*
1. `test: assert the three V4 attention copies stay identical`
2. `perf: single-pass online-softmax attention, no all_kv staging`

## Ergebnis

Implementiert wurde ein malloc-freier Online-Softmax-Kernel, der Fenster-Ring
und komprimierten Cache direkt als getrennte Quellen liest. Alle drei
Attention-Units und der eigenständige Batch-Pfad benutzen ihn standardmäßig;
`V4_FLASH=0` wählt weiterhin den zweistufigen Referenzalgorithmus. Auch dieser
Fallback arbeitet nun direkt auf den zwei Quellen, damit `all_kv` samt der
O(Kontext)-Kopie vollständig aus dem Amalgam verschwunden ist.

Der Source-Guardrail zählt und vergleicht die drei Attention-, je zwei
Compressor-/Indexer-Kopien sowie die gemeinsamen Definitionen der beiden
Layer-Units. Der C-Test deckt `topk` 1/7/64/2048, `-1`-Padding, Sink-Extreme,
gleiche Scores, einen um mehr als 80 dominierenden Score, leere Auswahl,
ungültige komprimierte Indizes sowie alle Zufallsfälle mit `V4_FLASH=1` und `0`
ab. Die Zufallsdaten verwenden signierte Queries; verglichen wird mit gemischter
absoluter/relativer BF16-Toleranz plus Cosinus-Grenze. Das Tiny-Orakel ist mit
`V4_FLASH=1` und `0` einschließlich Prefix-Reuse token-identisch. Dabei wurde
ein bereits in Phase 01 angelegter Widerspruch repariert: Der 520-Token-CTX-Test
brauchte eine 768-Token-Fixture, deren `max_position_embeddings` noch 128 war;
die YARN-Originalgrenze bleibt unverändert 128.

Gemessen mit dem reproduzierbaren `--benchmark`-Modus von
`test_v4_flash_attention` auf dem verfügbaren Entwicklungsrechner
(i5-10310U, AVX2), Paper-Geometrie mit 64 Heads × 512 Dimensionen und einem
128k-CSA-Layer (128 Fenster- plus 32.768 komprimierte Zeilen, Top-k 512): Median
aus drei Läufen zu je zehn Wiederholungen **33,397 → 23,548 ms/Layer (1,42×)**.
Über die 21 CSA-Layer entspricht das **701,333 → 494,511 ms pro Token** für
diesen Attention-Anteil. Der Legacy-Arm enthält dabei die reale, zuvor pro Token
ausgeführte 67-MB-Staging-Kopie; die **1,42× messen deren Eliminierung**, nicht
einen isolierten Gewinn des Online-Softmax gegenüber dem zweistufigen Softmax.
Der 167-GB-Checkpoint und die Zielmaschine waren in der Entwicklungsumgebung
nicht vorhanden; das ist daher eine isolierte Staging-/Kernel-Messung, keine
End-to-End-tok/s-Zahl.

Verifikation: `make -C c test`, `make -C c check` und beide Tiny-Modi grün.

## Ziel

Den zweistufigen materialisierten Softmax durch einen Online-Softmax ersetzen und
dabei den `all_kv`-Staging-Puffer loswerden.

Der Grund geht über „ein Durchlauf statt zwei" hinaus: **mit Online-Softmax wird
die Dequantisierung der Phasen 03/04 gratis.** Man liest eine Zeile,
dequantisiert im Register, aktualisiert Max/Summe/Akkumulator und verwirft sie —
der dequantisierte Wert erreicht nie den Speicher. Ohne diesen Umbau müsste der
GPU-Kernel aus Plan 05 einen `topk × 512`-Puffer materialisieren und würde den
halben Bandbreitengewinn der Quantisierung wieder verschenken.

Flash und TurboQuant sind hier dasselbe Feature. Deshalb kommt Flash zuerst.

## Ausgangslage

### Der heutige Kernel

`coli_v4_sparse_attention_ref` ([c/deepseek_v4.c:2934](../c/deepseek_v4.c)):

```c
float *scores = malloc((size_t)topk * sizeof(*scores));      /* pro Aufruf */
for (int head = 0; head < heads; head++) {
    /* Durchlauf 1: alle Scores, Maximum suchen */
    for (int rank = 0; rank < topk; rank++) { ... scores[rank] = score; ... }
    float denominator = expf(sinks[head] - maximum);          /* Sink */
    /* Durchlauf 2: exp, bf16-Runden, Value-Akkumulation */
    for (int rank = 0; rank < topk; rank++) {
        float probability = expf(scores[rank] - maximum);
        denominator += probability;
        probability = coli_bf16_round(probability);           /* TileLang-Cast */
        const float *value = kv + indices[rank] * head_dimension;
        for (int column ...) head_output[column] += probability * value[column];
    }
    for (int column ...) head_output[column] = coli_bf16_round(head_output[column] / denominator);
}
```

Jede KV-Zeile wird zweimal gelesen. `key` und `value` sind **derselbe Zeiger** —
absorbierte MLA.

### Der Staging-Puffer

In `attention_token_impl` ([:1670](../c/deepseek_v4.c) ff.):

```c
float *all_kv = state->compressed_count
    ? malloc((size_t)kv_count * head_dim * sizeof(*all_kv)) : NULL;
...
memcpy(all_kv, state->kv, window_size * head_dim * sizeof(*all_kv));
memcpy(all_kv + window_size * head_dim, state->compressed,
       compressed_count * head_dim * sizeof(*all_kv));
```

`kv_count = window_size + compressed_count`. Das ist eine **O(Kontext)-Allokation
und -Kopie pro Token und pro Layer**, nur damit Fenster und komprimierter Cache in
einem Indexraum liegen. Gelesen werden davon `topk = window_size + selected` Zeilen.

Bei 128k Kontext und ratio 4: `compressed_count` = 32768, also 32768 × 512 × 4 B =
**67 MB memcpy pro Token pro ratio-4-Layer**. Das dominiert alles andere.

## Änderungen

### Schritt 1 — Der Guardrail zuerst

`c/tests/test_v4_attention_source.py`, Muster von
[c/tests/test_deepseek_v4_dspark_source.py](../c/tests/test_deepseek_v4_dspark_source.py):

```python
ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = (ROOT / "deepseek_v4.c").read_text(encoding="utf-8")

# Quelle -> (Anzahl Kopien, Startmarke)
SOURCES = {
    "deepseek_v4_attention.c":  (3, "struct ColiDeepSeekV4WindowAttentionState {"),
    "deepseek_v4_compressor.c": (2, "/* ---- begin include deepseek_v4_compressor.c ---- */"),
    "deepseek_v4_indexer.c":    (2, "/* ---- begin include deepseek_v4_indexer.c ---- */"),
}
END = "/* ---- end include {source} ---- */"
```

Robuster als feste Zeilennummern: die Regionen über Start- und `end include`-Marker
schneiden, dann je Quelle paarweise auf Gleichheit prüfen. Zusätzlich die
Kopienzahl festnageln, damit eine fünfte Kopie nicht unbemerkt durchrutscht.

**Nicht nur die Attention.** Eine frühere Fassung dieses Plans prüfte allein die
drei Attention-Kopien. `deepseek_v4_compressor.c` und `deepseek_v4_indexer.c`
stehen ebenfalls doppelt im Amalgam (Units `*_SNAPSHOT`, siehe die Tabelle in
[00-reference.md](00-reference.md)) — und das sind genau die Funktionen, die
[03-kv-codec.md](03-kv-codec.md) und [12-expert-cache-policy.md](12-expert-cache-policy.md)
anfassen. Ein Test, der sie auslässt, deckt die riskanteren Änderungen nicht ab.

`deepseek_v4_layer.c` bleibt außen vor: die `LAYER_RESIDENT`-Kopie ist ein
Superset (sie trägt zusätzlich `v4_fp8_pack_rows8_inplace`), also ist
Textgleichheit dort das falsche Kriterium. [06-dense-vram.md](06-dense-vram.md)
prüft die beiden `coli_v4_layer_load`-Kopien stattdessen von Hand.

**Dieser Test kommt als eigener Commit vor der Umstellung.** Er muss auf
unverändertem `main` grün sein — sonst stimmt die Annahme nicht.

### Schritt 2 — Online-Softmax

Neue Funktion neben der bestehenden, die als Vergleichsmaßstab erhalten bleibt
(sie wird außerdem von `COLI_V4_UNIT_ATTENTION_CACHE` benutzt, [:8976](../c/deepseek_v4.c)):

```c
/* Online-Softmax über zwei getrennte Zeilenquellen: Fenster-Ring und
 * komprimierter Cache. Ersetzt die Verkettung in einen all_kv-Puffer. */
int coli_v4_flash_attention_ref(
    float *output, const float *queries,
    const float *window_kv, int window_size,
    const float *compressed_kv, int compressed_count,
    const int *window_indices,          /* -1 = leer */
    const int *compressed_indices, int compressed_selected,
    const float *sinks, int heads, int head_dimension, float softmax_scale);
```

Kern pro Head:

```c
float running_max = sinks[head];   /* der Sink ist der initiale Kandidat */
float running_sum = 1.0f;          /* exp(sink - sink) */
memset(acc, 0, head_dimension * sizeof(*acc));

for (jede ausgewählte Zeile) {
    const float *key = row_pointer(...);          /* Fenster ODER komprimiert */
    float score = dot(query, key, head_dimension) * softmax_scale;
    if (score > running_max) {
        float correction = expf(running_max - score);
        running_sum *= correction;
        for (int c = 0; c < head_dimension; c++) acc[c] *= correction;
        running_max = score;
    }
    float probability = expf(score - running_max);
    running_sum += probability;
    probability = coli_bf16_round(probability);
    for (int c = 0; c < head_dimension; c++) acc[c] += probability * key[c];
}
for (int c = 0; c < head_dimension; c++)
    head_output[c] = coli_bf16_round(acc[c] / running_sum);
```

Fünf Punkte, die stimmen müssen:

- **Sink als initialer Max-Kandidat.** Im heutigen Code steht
  `denominator = expf(sinks[head] - maximum)` außerhalb der Schleife. Startet man
  `running_max = sinks[head]`, `running_sum = 1.0f`, ergibt sich derselbe Nenner —
  und das Rescaling zieht ihn korrekt mit.
- **`-1`-Indizes** werden übersprungen, nicht als `-INFINITY`-Score geführt. Die
  heutige Version setzt `scores[rank] = -INFINITY` und überspringt sie im zweiten
  Durchlauf ohnehin.
- **Beide Quellen werden begrenzt.** Fensterindizes müssen kleiner als
  `window_size`, komprimierte Indizes kleiner als `compressed_count` sein;
  andernfalls liefert der Kernel `-1`. Der Batch-Pfad hält dafür den jeweiligen
  kausalen `compressed_count` pro Item fest, statt den Endstand des Batches zu
  verwenden.
- **Die leere Auswahl braucht einen expliziten Guard.** Das ist die eine Stelle,
  an der der Sink-Init eine Verhaltensänderung *verdeckt*: heute läuft
  `coli_v4_sparse_attention_ref` bei ausschließlich `-1`-Indizes in
  `if (!isfinite(maximum)) return -1` ([:2964](../c/deepseek_v4.c)) und meldet
  einen Fehler. Der Online-Softmax hat kein `maximum == -INFINITY` mehr — er
  startet beim Sink — und würde sauber Nullen zurückgeben statt zu scheitern.
  Also mitzählen und selbst abbrechen:

  ```c
  int seen = 0;
  /* ... in der Zeilenschleife: seen++; ... */
  if (!seen) return -1;   /* wie die zweistufige Referenz */
  ```

  Ohne das widerspricht der Kernel dem eigenen Testfall unten („muss sauber
  scheitern"). Eine frühere Fassung dieses Plans hatte genau diesen Widerspruch.
- **Kein `malloc`** mehr im Kernel. Der `acc`-Puffer ist `head_dimension` groß und
  wird vom Aufrufer gestellt.

### Schritt 3 — Umstellung der Aufrufer

`all_kv` und die zwei `memcpy` entfallen; stattdessen werden Fenster- und
komprimierter Zeiger zusammen mit ihren Indexlisten durchgereicht.

| Datei/Unit | Zeilen | Was |
|---|---|---|
| `COLI_V4_UNIT_ATTENTION` | 1670–1705 | `all_kv` raus, Flash-Aufruf |
| `COLI_V4_UNIT_ATTENTION_BATCH` | 2068–2103 | dito |
| `COLI_V4_UNIT_ATTENTION_TRANSACTION` | 4842–4877 | dito |
| Batch-Pfad `coli_v4_attention_window_batch_ref` | 2328–2360 | dito, pro Item |

**Alle vier Stellen, identisch.** Der Test aus Schritt 1 fängt Abweichungen.

Die Index-Berechnung bleibt wie sie ist — sie erzeugt heute schon getrennt
Fenster-Indizes (`i <= position ? i : -1` bzw. Ring-Rotation) und komprimierte
Ordinale (`state->indexer ? compressed_indices[i] : i`).

### Schritt 4 — Schalter

`V4_FLASH=0` schaltet auf die zweistufige Referenz zurück. Default **1** — anders
als bei den Codec-Knöpfen, weil hier keine Präzision geändert, sondern nur die
Summationsreihenfolge verschoben wird, und weil Plan 05 darauf aufbaut.

## Warum das nicht bit-identisch ist

Der Online-Softmax skaliert Akkumulator und Summe nach, wenn das Maximum steigt.
Das `coli_bf16_round(probability)` (der Kommentar in `:2976` nennt TileLangs
BF16-Cast vor dem Value-GEMM) landet dadurch an anderer Stelle im Rechenweg, und
die Additionsreihenfolge in `acc` ändert sich.

**Also Toleranz statt Bit-Gleichheit** — und deshalb bleibt die zweistufige
Referenz im Baum stehen, nicht nur für `V4_FLASH=0`, sondern als Testorakel.

## Tests

- `c/tests/test_v4_attention_source.py` (Schritt 1, eigener Commit).
- `c/tests/test_v4_flash_attention.c` — neue Make-Regel.
  - Flash gegen zweistufige Referenz auf Zufallsdaten: `heads=8`, `head_dim=512`,
    `topk` in {1, 7, 64, 2048}, mit und ohne `-1`-Padding.
  - Sink-Verhalten: extreme Sinks (sehr groß, sehr klein) müssen den Nenner
    korrekt dominieren.
  - Numerische Randfälle: alle Scores gleich; ein Score dominiert um >80;
    `topk=1`; alle Indizes `-1` (muss `-1` liefern wie die zweistufige Referenz —
    siehe den `seen`-Guard oben —, nicht Nullen und nicht NaN).
  - Toleranz: `abs(a-b) <= 1e-3 + 1e-3 * max(abs(a), abs(b))`, Cosine `> 0.9999`.
- Bestehende Gates, insbesondere `make -C c deepseek-v4-tiny-check`.

## Abnahme

- Tiny-Fixture **token-identisch** mit `V4_FLASH=1` und mit `V4_FLASH=0`. Kippt
  ein Token trotzdem (siehe „Token-Identität kann kippen" unten), ist das ein
  dokumentierter Befund, kein bestandenes Kriterium — nicht stillschweigend als
  „token-identisch" abhaken.
- Keine `malloc`-Aufrufe mehr im Attention-Kernel (`grep -c malloc` in der Region).
- `all_kv` kommt in `c/deepseek_v4.c` nicht mehr vor.
- Source-Sync-Test grün.
- Gemessen: Zeit pro Token bei langem Kontext sinkt deutlich — das ist der
  eigentliche Nachweis, dass die O(Kontext)-Kopie weg ist.

## Risiken

- **Vier Aufrufstellen statt einer.** Der Batch-Pfad wird gern vergessen; er hat
  dieselbe Logik pro Item. Der Source-Test deckt nur die identischen Kopien ab,
  **nicht** den Batch-Pfad — der ist eigener Text und muss manuell geprüft werden.
- **Token-Identität kann kippen.** Wenn zwei Logits sehr nah beieinander liegen,
  kann die geänderte Summationsreihenfolge das Argmax drehen. Am Tiny-Fixture
  unwahrscheinlich, aber wenn es passiert: nicht wegdiskutieren, sondern die
  Toleranz und den Fall dokumentieren.
- **Der Sink-Init ist die subtilste Stelle.** Ein Vorzeichenfehler dort fällt in
  Zufallstests kaum auf, verschiebt aber jede Wahrscheinlichkeitsmasse. Der
  Extremwert-Test oben ist dafür da.
