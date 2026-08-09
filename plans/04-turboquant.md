# 04 — TurboQuant (2/3/4 Bit)

Voraussetzung: [00-reference.md](00-reference.md), [03-kv-codec.md](03-kv-codec.md)

*Commits:*
1. `feat: TurboQuant 2/3/4-bit KV codec`
2. ~~`perf: keep V4 attention in the WHT domain, one inverse per head`~~
   (in dieser Phase verworfen; kein `V4_KV_ROTATED`)

## Ergebnis

Implementiert ist der erste Commit als CPU-Referenzpfad: `turbo2`, `turbo3` und
`turbo4` hängen am Codec-Interface aus Phase 03, verwenden exakt 128 Werte pro
WHT-Gruppe und belegen 34/50/66 B je Gruppe. Die gespeicherte fp16-Norm ist um
die Rekonstruktionsenergie korrigiert. Der Decoder dreht jede Gruppe zurück in
den ursprünglichen Raum, sodass die bestehenden Attention- und Indexer-Pfade
unverändert bleiben. Beide Turbo-Schalter sind verlustbehaftet und bleiben
opt-in; insbesondere bleibt `V4_KV_INDEX` getrennt, weil er die Top-k-Auswahl
ändern kann. Nicht durch 128 teilbare Geometrien warnen und fallen auf den vom
Aufrufer vorgegebenen sicheren Codec zurück. Das Tiny-Fixture mit Dimension 32
prüft genau diesen Fallback.

Der deterministische Test über 32 Zeilen à 512 kontinuierliche Zufallswerte und
32 weitere Zeilen auf dem realen FP8+BF16-Vorgitter misst:

| Codec | kontinuierlich | FP8+BF16-Vorgitter | Normverhältnis |
|---|---:|---:|---:|
| turbo2 | 0,9411 | 0,9411 | 1,0000 |
| turbo3 | 0,9833 | 0,9832 | 1,0000 |
| turbo4 | 0,9954 | 0,9954 | 1,0000 |

Die Vorquantisierung verursacht bei **gleicher** Varianz keinen sichtbaren
zusätzlichen Einbruch. Dieser Fall genügt aber nicht für eine Entscheidung über
den RoPE-Schwanz: bei skalierten RoPE-Dimensionen misst turbo3:

| RoPE-Skala | Gesamt | NoPE | RoPE |
|---:|---:|---:|---:|
| 1 | 0,9832 | 0,9832 | 0,9834 |
| 1/4 | 0,9832 | 0,9842 | 0,8850 |
| 1/16 | 0,9832 | 0,9839 | 0,5072 |
| 16 | 0,9836 | 0,8380 | 0,9888 |

Der größere Anteil maskiert den kleineren im Gesamt-Cosine. Der gleichförmige
Codec ist daher nur die derzeit implementierte Variante, keine aus diesen
Zufallsdaten abgeleitete Qualitätsentscheidung gegen `V4_KV_ROPE_BF16`. Eine
solche Entscheidung braucht Statistiken oder Qualitätsmessungen am vollständigen
Checkpoint. Turbo3 liegt außerdem unter dem im Plan grob als „≈ 1,0“ übernommenen
Fork-Wert; 0,9832 ist die belastbare Schranke dieses Ports für den gleich
skalierten Test. Die 128k-Planerrechnung mit Paper-Geometrie ergibt
**1,690 GiB f32 → 0,431 GiB native → 0,165 GiB turbo3**, also 10,24× gegenüber
f32 und 2,61× gegenüber dem vollständigen nativen KV-Kontext einschließlich des
kleineren Indexerformats.

Der zweite Commit (`V4_KV_ROTATED`) wurde verworfen. Er würde
einen zweiten Attention-Kernel einführen, während Phase 05 denselben Spezialpfad
auf CUDA baut; der generische Decode-zurück-in-den-Originalraum-Pfad bleibt das
Testorakel dafür. Phase 05 setzt den rotierten Modus deshalb nicht voraus und
beginnt mit direkter Turbo-Dequantisierung im Originalraum. Entsprechend sind
die beiden nicht implementierten Variablen aus der Referenztabelle entfernt
worden. Ein Full-Checkpoint-Durchsatz- und
Tokenqualitätstest steht aus, weil der 167-GB-Checkpoint in der
Entwicklungsumgebung nicht vorhanden ist. `make test` und `make check` sind
grün. Das vorhandene Tiny-Fixture besteht Oracle und Prefix-Reuse mit Defaults
und mit `V4_KV{,_INDEX}=turbo2|turbo3|turbo4`; alle drei Einstellungen warnen
wie vorgesehen und fallen wegen der 32er-Dimension token-identisch auf `native`
zurück. Das Fixture wurde anschließend mit `torch==2.13.0+cpu`,
`transformers==5.14.1` und `safetensors==0.8.0` aus der lokalen
`.venv-v4-tiny` erzwungen neu erzeugt. Der vollständige
`deepseek-v4-tiny-check` besteht Target-Oracle, Session-/Serve-Protokoll und
Prefix-Reuse token-identisch; der generierte Modellbestand bleibt unter 1 MiB.

Ein 128-Dim-Turbo3-Fall im Flash-Attention-Test läuft dagegen ohne Fallback durch
den zeilenweise kodierten Cache und vergleicht dessen Output bitgleich mit den
vorab dekodierten f32-Zeilen. Der optionale synthetische 128k-Benchmark auf dem
i5-13400F (`-O3 -march=native`) misst in vier Läufen für turbo3/f32 Bereiche von
0,95–1,09× (CSA), 1,01–1,09× (HCA) und 1,10–1,30× (Indexer); die Mediane liegen
bei etwa 1,05×, 1,05× und 1,14×. Das ist der CPU-Preis des generischen
Decode-Pfads in diesem Harness, keine Full-Checkpoint-Tokenrate.

## Ziel

Die drei Turbo-Codecs in das Interface aus Plan 03 einhängen.

**Der Vergleichsmaßstab hat sich durch das Paper verschoben.** Turbo tritt nicht
mehr gegen f32 an, sondern gegen `native` (fp8+bf16, 583 B) — das ist der
verlustfreie Stand nach Plan 03. Die ehrliche Rechnung:

| | Zeile @512 | vs f32 | vs `native` | verlustfrei |
|---|---|---|---|---|
| f32 (vor Plan 03) | 2048 B | 1× | — | — |
| `native` | 583 B | 3.5× | 1× | **ja** |
| turbo4 | 264 B | 7.8× | 2.2× | nein |
| turbo3 | 200 B | 10.2× | **2.9×** | nein |
| turbo2 | 136 B | 15.1× | 4.3× | nein |

Turbo holt also noch einmal einen Faktor ~3 heraus — real, aber nicht mehr die
Größenordnung, die es vor Plan 03 zu sein schien. Das ändert nichts an der
Umsetzung, wohl aber an der Erwartung: **wenn Plan 03 fertig ist und das
VRAM-Budget schließt, ist Plan 04 optional.**

Wo er sich klar lohnt: bei 1M Kontext (3.4 → 1.3 GiB Cache, 100 → 34 MB/Token
Lesebandbreite) und für den Indexer-Cache, dessen `index_head_dim = 128` genau
eine Turbo-Gruppe ist.

## Die zusätzliche Unsicherheit

Nach Plan 03 sind die zu quantisierenden Werte **bereits fp8-quantisiert** —
`bf16_round(e4m3_decode(q) · scale)`, also Werte auf einem groben, ungleichmäßigen
Gitter statt einer kontinuierlichen Verteilung. Die Fork-Referenzwerte (turbo3
Cosine ≈ 1.0, turbo4 ≈ 0.9956) wurden auf **kontinuierlichen** Daten gemessen.

Die WHT-Rotation sollte das Gitter wieder in Richtung Gauß verschmieren — genau
dafür ist sie da, und dass DeepSeek für den Indexer selbst Hadamard-vor-FP4 wählt,
spricht dafür. Aber es ist eine Annahme, keine Messung.

**Deshalb misst der Codec-Test beides:** Round-trip auf kontinuierlichen Daten
(vergleichbar mit dem Fork) und auf fp8-vorquantisierten Daten (der reale Fall).
Fällt Zweiteres deutlich ab, ist das ein Ergebnis und turbo bleibt für die
Hauptzeilen aus.

## Quelle und Umfang

`reference/llama-cpp-turboquant/ggml/src/ggml-turbo-quant.c` (1030 Zeilen, MIT).
Zu portieren sind ~200 davon. Neuer Header `c/turbo_quant.h` mit Attribution im
Kopfkommentar (arXiv 2504.19874, llama.cpp-Fork, MIT → Apache-2.0); der
vollständige MIT-Text liegt in `THIRD_PARTY_NOTICES`.

**`reference/` wird nicht committet.** Vor dem ersten Commit prüfen, dass es in
`.gitignore` steht oder nie gestaget wird.

### Was übernommen wird

| Element | Referenz | Anmerkung |
|---|---|---|
| `turbo_cpu_s1[128]`, `turbo_cpu_s2[128]` | `:203`, `:210` | ±1-Vorzeichentabellen, seed 42 |
| `turbo_cpu_fwht` | `:219` | vorwärts, in-place |
| `turbo_cpu_fwht_inverse` | `:245` | s1/s2 getauscht |
| `CENTROIDS_2BIT[4]` | `:38` | |
| `CENTROIDS_3BIT[8]` | `:41` | |
| `CENTROIDS_4BIT[16]` | `:494` | steht dort **im Funktionskörper**, herausziehen |
| `nearest_centroid_2/3/4bit` | `:163`, `:171`, `:183` | Midpoint-Vergleichsketten |
| Quantisierungsablauf | `:277`–`:340` | Norm → normalisieren → WHT → runden → korrigierte Norm |

### Was ausdrücklich **nicht** übernommen wird

`turbo_init_rotation` (`:69`), `turbo_init_qjl` (`:128`), `matvec` (`:149`),
`turbo_prng_*`, und die beiden statischen `float[128*128]`-Matrizen.

Grund: bei `TURBO4_USE_4BIT=1` — dem Default (`ggml-common.h:312`) — ist turbo4
reines 4-bit-PolarQuant. Der QJL-Zweig ist toter Code. Die Referenz selbst notiert,
dass die 64 KB-Stack-Variante dort mal Stack-Overflows verursacht hat; wir erben
das Problem gar nicht erst.

## Blocklayout

Unverändert zur Referenz, damit die Qualitätszahlen vergleichbar bleiben
(`ggml-common.h:296–354`; die Byte-Kommentare dort sind veraltet, die
`static_assert`s stimmen):

```c
#define COLI_TQ_GROUP 128

typedef struct { uint16_t norm; uint8_t qs[32]; }                  ColiTurbo2Block; /* 34 B */
typedef struct { uint16_t norm; uint8_t qs[32]; uint8_t signs[16]; } ColiTurbo3Block; /* 50 B */
typedef struct { uint16_t norm; uint8_t qs[64]; }                  ColiTurbo4Block; /* 66 B */
```

`norm` ist fp16. Das Repo hat keinen fp16-Helfer in `native_quant.h` (nur bf16) —
also entweder `_Float16` wo verfügbar oder eine kleine `coli_fp16_encode/decode`
im selben Header, getestet gegen Round-trip.

turbo3 packt die unteren 2 Bit des 3-Bit-Index in `qs` (4 pro Byte) und Bit 2 in
`signs` (8 pro Byte) — nicht offensichtlich, aber die CUDA-Kernel aus Plan 05
erwarten genau das.

## Der Kern

Pro 128er-Gruppe:

```c
/* 1 */ float norm_sq = Σ x[j]²;  float grp_norm = sqrtf(norm_sq);
/* 2 */ buf[j] = x[j] / grp_norm;
/* 3 */ turbo_cpu_fwht(buf, 128);
/* 4 */ idx[j] = nearest_centroid_Nbit(buf[j]);  recon_sq += C[idx[j]]²;
/* 5 */ corrected = grp_norm / sqrtf(recon_sq);   blk->norm = fp16(corrected);
```

Schritt 5 ist der Teil, den man leicht übersieht: gespeichert wird **nicht**
`grp_norm`, sondern die um die Rekonstruktionsenergie korrigierte Norm. Ohne das
ist die dequantisierte Zeile systematisch zu kurz.

`inv_sqrt` in `turbo_cpu_fwht` ist `1/√128 = 0.08838834764831845f` für
`group_size=128`.

## Integration in Plan 03

`head_dim = 512` → **vier** Gruppen pro Zeile, vier fp16-Normen:

```c
size_t coli_v4_kv_row_bytes(ColiV4KVCodec codec, int head_dim) {
    if (codec >= COLI_V4_KV_TURBO4) {
        if (head_dim % COLI_TQ_GROUP) return 0;      /* nicht darstellbar */
        int groups = head_dim / COLI_TQ_GROUP;
        switch (codec) {
            case COLI_V4_KV_TURBO4: return groups * sizeof(ColiTurbo4Block);
            case COLI_V4_KV_TURBO3: return groups * sizeof(ColiTurbo3Block);
            case COLI_V4_KV_TURBO2: return groups * sizeof(ColiTurbo2Block);
        }
    }
    ...
}
```

**`head_dim % 128 != 0` gibt 0**, und `coli_v4_kv_codec_from_env` fällt mit einer
Warnung auf f32 zurück. Das Tiny-Fixture hat `head_dim=32`
([c/deepseek_v4_tiny/config.json:15](../c/deepseek_v4_tiny/config.json)) und
`index_head_dim=32` — dort läuft also nie ein Turbo-Codec. Wichtig, weil die
Zentroide auf `N(0, 1/√128)` kalibriert sind: mit 32er-Gruppen wären sie um
`√(128/32) = 2` daneben. Stiller Fallback wäre schlimmer als kein Fallback,
deshalb die Warnung.

Beim echten Modell passt dagegen **beides**: `head_dim=512 = 4×128` und
`index_head_dim=128 = 1×128`.

`coli_v4_kv_decode_row` dequantisiert die vier Gruppen in den zeilengroßen
Scratch, den Attention und Indexer seit dem Review-Fix aus Plan 03 einmal pro
geteilter KV-Zeile verwenden. Die nie produktiv aufgerufenen
`coli_v4_kv_dot`/`_accumulate` wurden in Plan 03 entfernt; TurboQuant bekommt
stattdessen spezialisierte Kernel. Kein Heap und kein kontextgroßer f32-Puffer.

## RoPE-Schwanz

Dims 448–511 sind der RoPE-rotierte Teil ([c/deepseek_v4.c:1652](../c/deepseek_v4.c)):

```c
float *kv_rope = kv + head_dim - rope_dim;      /* rope_dim = 64 */
coli_v4_rope_apply(kv_rope, 1, rope_dim, cosines, sines, 0);
```

Gruppe 3 (Dims 384–511) mischt damit 64 nope- mit 64 rope-Dims, die
unterschiedliche Statistik haben.

**Das Paper trennt hier ausdrücklich** (2.3.4): bf16 für die RoPE-Dimensionen, fp8
für den Rest. Die Trennung ist also nicht meine Idee, sondern die des Modells — und
sie ist nach Plan 03 ohnehin schon im `native`-Codec umgesetzt. Für turbo heißt das:
die Variante mit bf16-RoPE-Tail ist die **erwartungskonforme**, die gleichförmige
die experimentelle.

Problem: `448 = 3.5 × 128` ist kein Vielfaches der Rotationsgruppe. Drei Optionen:

| Variante | Layout | Bytes | Anmerkung |
|---|---|---|---|
| gleichförmig | 4 × turbo3(128) | 200 | mischt nope/rope in Gruppe 3 |
| **bf16-Tail** | 3 × turbo3(128) + 64 × bf16 | **278** | Dims 384–447 fallen unter den Tisch |
| Tail + Rest-fp8 | 3 × turbo3(128) + 64 fp8+1 scale + 64 bf16 | 343 | vollständig, paperkonform |

Die mittlere Zeile stimmt nur, wenn man Dims 384–447 mitquantisiert — 448 teilt
sich nicht in 128er-Gruppen. Sauber ist entweder die erste (alles turbo) oder die
dritte (turbo für 0–383, `native`-fp8 für 384–447, bf16 für 448–511).

Die gleichförmige Variante ist gebaut. Der Test muss mehrere NoPE-/RoPE-
Varianzverhältnisse getrennt ausweisen; sein Gesamt-Cosine allein reicht wegen
der 448:64-Gewichtung nicht. Die gemessene Empfindlichkeit hält die dritte
Variante als möglichen `V4_KV_ROPE_BF16`-Follow-up offen, entschieden wird sie
erst mit Statistiken oder Qualitätsmessungen am vollständigen Checkpoint.

## Verworfener CPU-Follow-up: rotierter Score-Raum

*Nicht implementiert; kein `V4_KV_ROTATED`-Knopf.*

Die WHT ist orthogonal, also `<Rq, Rk> = <q, k>`. Statt jede gelesene Zeile
zurückzudrehen:

1. Q einmal pro Token und Head vorwärts-WHT (`heads × head_dim/128` Transformationen).
2. Ein spezialisierter Turbo-Attention-Pfad arbeitet direkt auf den rotierten
   Zentroidwerten — **keine** inverse WHT pro Zeile. Der generische Decode-once-
   Pfad aus Plan 03 bleibt das Referenzorakel.
3. Der spezialisierte Pfad akkumuliert im rotierten Raum.
4. **Eine** inverse WHT auf `head_output`, *danach* `coli_bf16_round`, *danach* das
   RoPE-Inverse ([:1720](../c/deepseek_v4.c)).

Die Reihenfolge in 4 ist zwingend: `coli_v4_rope_apply(..., 1)` adressiert konkrete
Dimensionen und braucht den Originalraum.

Kosten fallen von O(topk) auf O(heads). Bei `topk` = 2048 und 64 Heads ist das
Faktor 32 weniger WHT-Arbeit.

Korrektheit folgt aus der Linearität: `Σpᵢ·R·vᵢ = R·(Σpᵢ·vᵢ)`. Nicht bit-identisch
zum unrotierten Pfad, weil `coli_bf16_round` in einem anderen Raum landet — also
Toleranz gegen Commit 1.

## Tests

`c/tests/test_v4_turbo_quant.c`, neue Make-Regel:

- **WHT-Orthogonalität**: `‖fwht(x)‖ == ‖x‖` und `<fwht(x), fwht(y)> == <x, y>`
  für Zufalls-`x`, `y`, Toleranz `1e-5` relativ. Das ist die Eigenschaft, auf der
  der rotierte Score-Raum steht.
- **Selbstinversität**: `fwht_inverse(fwht(x)) == x` innerhalb `1e-6`.
- **Round-trip je Codec** auf `N(0,1)`-Vektoren, `head_dim=512`:
  Referenzwerte aus dem Fork — turbo3 Cosine ≈ 1.0, turbo4 ≈ 0.9956. turbo2
  entsprechend niedriger; Ist-Wert messen und als Schranke festschreiben.
- **Normerhaltung**: `‖decode(encode(x))‖ / ‖x‖` ∈ [0.98, 1.02] — fängt einen
  vergessenen Schritt 5.
- **fp16-Round-trip** für die Norm, inkl. sehr kleiner und sehr großer Werte.
- **Packing**: turbo3 `qs`/`signs` bitweise gegen eine unabhängige Referenz-
  implementierung im Test, nicht gegen sich selbst.
- **`head_dim % 128 != 0`** → `row_bytes == 0`, `from_env` fällt zurück und warnt.
- Für den rotierten Pfad: gegen den unrotierten, Cosine `> 0.9999`.

Dazu die Gates aus Plan 03 mit `V4_KV=turbo2|turbo3|turbo4` — am Tiny-Fixture
greift der f32-Fallback, das ist der Punkt: der Test belegt, dass er greift.

## Abnahme

- Alle Round-trip-Zahlen innerhalb der Fork-Referenzwerte.
- `V4_KV=turbo3` am Tiny-Fixture: Warnung auf stderr, f32-Fallback, token-identisch.
- `ram_tiers` zeigt bei simuliertem `head_dim=512` (Unit-Test auf
  `context_bytes`) den 10.2×-Effekt auf den KV-Anteil.
- `reference/` ist nicht im Commit.

## Risiken

- **Zentroide sind an 128 gebunden.** Wer später eine 512-Punkt-WHT will, muss die
  Tabellen um `√(128/512)` skalieren. Als Kommentar an die Tabellen schreiben.
- **Die korrigierte Norm** ist der klassische Portierungsfehler. Der
  Normerhaltungstest ist genau dafür da.
- **turbo3-Packing** ist unintuitiv (2 Bit in `qs`, 1 Bit in `signs`). Ein Fehler
  dort fällt im Cosine-Test möglicherweise **nicht** auf, weil die Zentroide
  symmetrisch sind — deshalb der explizite Bit-Test.
- **turbo2 könnte zu grob sein.** 2.125 bpw ist aggressiv; wenn die Token-Rate am
  echten Modell einbricht, ist das ein Ergebnis, kein Fehler. Messen, dokumentieren,
  nicht als Default anbieten.
- **`_Float16`-Verfügbarkeit** variiert über die unterstützten Targets
  (x86-64 und aarch64, gcc). Wenn nicht verfügbar: eigene Bit-Manipulation, wie
  `coli_e4m3fn_encode` ([c/deepseek_v4.c:9999](../c/deepseek_v4.c)) es für fp8 vormacht.
