# 14 — Chunked Prefill für 256k + DSpark

Voraussetzung: [00-reference.md](00-reference.md), [03-kv-codec.md](03-kv-codec.md),
[04-turboquant.md](04-turboquant.md), [07-head-dspark-vram.md](07-head-dspark-vram.md)
und [08-vram-planner.md](08-vram-planner.md).
[10-dual-streaming.md](10-dual-streaming.md) ist keine Code-Abhängigkeit, aber
Teil der optionalen Maximaldurchsatz-Messung.

*Commits:*

1. `feat: bound V4 prefill activations to configurable chunks`
2. `feat: hand chunked V4 prefill history to DSpark decode`
3. `perf: tune the 256k TurboQuant + DSpark profile`

## Ziel

**256k Kontext ist das tägliche Langkontext-Profil** dieser Maschine:

```bash
CTX=262144 V4_KV=turbo3 V4_PREFILL_CHUNK=32768 \
V4_MTP=1 V4_DRAFT=3 V4_VRAM=1
```

`turbo3` bleibt ein explizites, verlustbehaftetes Opt-in; es ändert weder
Router noch Top-k. Chunking und die DSpark-Übergabe müssen gegen den
unchunked Target-Pfad token-identisch bleiben. DSpark wird erst Bestandteil
dieses Profils, wenn die offene Full-Checkpoint-A/B-Abnahme aus Plan 07 grün ist.

**1M (`CTX=1048576`) ist experimentell**, kein Default und keine
Durchsatz-Zusage. Es benutzt ebenfalls Turbo3 und Chunking, braucht aber eine
eigene Qualitäts-, RAM- und Langkontext-Latenz-Abnahme.

## Warum ein neuer Plan nötig ist

Der V4-Session-Code allokiert heute beim Öffnen unabhängig von der Promptlänge:

```c
session->state = malloc(max_prompt_tokens * hc_mult * hidden_size * sizeof(float));
session->next  = malloc(max_prompt_tokens * hc_mult * hidden_size * sizeof(float));
```

Das sind bei der Checkpoint-Geometrie zusammen **128 KiB pro konfiguriertem
Token**: 4 GiB bei 32k, 16 GiB bei 128k, 32 GiB bei 256k und 128 GiB bei 1M.
MLA und TurboQuant komprimieren persistenten KV, nicht diese Aktivierungen.

Die 64-Token-Schleife in `target_batch` ist nur Micro-Batching eines bereits
vollen `[prompt][hc_mult][hidden]`-Paares. `COLI_PREFILL_CHUNK` in
`c/colibri.c` gehört zum allgemeinen Motor und ist **kein** Schalter für
`deepseek_v4`.

Ein echter V4-Chunk verarbeitet zusammenhängende Prompt-Positionen nacheinander.
Attention-, Compressor- und Indexer-Zustände bleiben über die Grenzen erhalten;
ein 32k-Chunk macht einen 256k-Prompt nicht zu acht getrennten Kontexten. Er
begrenzt nur die temporären Aktivierungen auf 4 GiB.

## Commit 1 — Target-only Chunked Prefill

### Schnittstelle und Speichervertrag

Neuer Nutzerknopf: `V4_PREFILL_CHUNK=0|<tokens>`.

- `0` ist Default und erhält den heutigen Full-Prefill-Pfad bitgenau.
- Gültige Werte werden auf `64..65536` geklemmt und auf ein Vielfaches von 64
  abgerundet; Müll und Überlauf fallen auf `0` zurück.
- Effektiv gilt `min(V4_PREFILL_CHUNK, fresh_prompt_tokens)`; die Größe wird
  im Stats-Report geloggt.
- `CTX` bleibt die KV- und Token-ID-Kapazität. Nur `state` und `next` werden
  für die Chunk-Kapazität statt für `max_prompt_tokens` angelegt.

`coli_v4_session_generate` verarbeitet den nicht wiederverwendeten Prompt-Suffix
in aufeinanderfolgenden Chunks: Embeddings in den kleinen `state`-Puffer laden,
`target_batch` mit absolutem `start` aufrufen, Aktivierungen danach verwerfen.
Die pro Layer fortgeschriebenen KV-/Compressor-/Indexer-Zustände bleiben der
Kontext. Prefix-Reuse darf nicht regressieren; die Logits des finalen Chunks
bleiben die ersten Decode-Logits.

### Tests

- `c/tests/test_v4_prefill_chunk.c`: Parser/Clamp und Allokationsformel. Bei
  `CTX=262144`, Chunk 32768 dürfen `state` und `next` zusammen 4 GiB planen,
  nicht 32 GiB.
- `test_deepseek_v4_tiny.py`: Short-, Long-, Prefix- und Serve-Fälle zusätzlich
  mit Chunk 64; IDs und Länge exakt gegen unchunked.
- Mehrere Chunk-Grenzen innerhalb eines Suffix prüfen lückenlose absolute
  KV-Positionen.

## Commit 2 — DSpark erst nach dem finalen Chunk

Während Prompt-Prefill gibt es keine MTP-Proposals. Nach dem letzten Chunk
beginnt der normale verifizierte Decode. Dafür gilt:

- `v4_mainh_tap` erhält für jeden Token absolute Positionen; sein 128er-Ring
  enthält nach Prefill die letzten Target-Taps unabhängig von Chunkgrenzen.
- DSpark-Stage-KV-Ringe und `main_x_abs` werden pro neuem Prompt genau einmal,
  nicht pro Chunk, zurückgesetzt.
- Prefix-Reuse chunkt nur den neuen Tail und übergibt dieselbe Abschlusshistorie.
- Kann der Übergabevertrag nicht hergestellt werden, muss ein explizites
  `V4_MTP=1 V4_DRAFT>0` klar fehlschlagen statt ungeprüft zu degradieren.

Die offene Tokenabweichung aus Plan 07 ist harte Voraussetzung. Der Test
vergleicht vollständig (a) unchunked Target-only, (b) chunked Target-only und
(c) chunked DSpark mit exakter Target-Verifikation; Chunking darf sie nicht
verdecken.

### Tests

- `c/tests/test_v4_prefill_chunk_dspark.c`: künstliche Taps an mehreren
  Grenzen, Ringpositionen und genau ein Reset pro Prompt.
- `test_deepseek_v4_dspark_source.py`: kein Draft während Prefill, Verifikation
  bleibt vor dem Commit der Ausgabe.
- Tiny-Fixture: Chunk 64, DSpark on/off, korrekte Attempts/Drafted/Accepted und
  nach Plan-07-Fix identische Target-Tokens.

## Commit 3 — 256k-Profil messen und dokumentieren

Der Harness braucht neben One-shot-Kaltläufen einen **persistent geöffneten**
Engine-/Serve-Lauf. Ein zweiter Prozess ist kein warmer Expert-Cache; er kann
nur Pagecache und Usage-Historie wiederverwenden.

Für `CTX=262144`, `V4_KV=turbo3`, `V4_MTP=1`, `V4_DRAFT=3` werden
`V4_PREFILL_CHUNK={4096,8192,16384,32768,65536}` verglichen. Der Gewinner ist
der größte Chunk, der ohne RAM-/CUDA-Druck durchläuft und die beste reale TTFT
liefert. Bei ähnlichen Resultaten gewinnt der größere Chunk, weil er weniger
Expert-Working-Sets erneut lädt.

Pflichtmetriken: Peak RSS, `ram_tiers`, Target-/DSpark-Cache getrennt,
`nvidia-smi` vor Open/nach Dense+Head/nach DSpark/Decode, TTFT, Prefill- und
Decode-tok/s, Hits/Misses, gelesene GB, DSpark-Stufenzeiten sowie vollständige
greedy Token-A/Bs. Einzellaufwerk ist Pflicht; der Plan-10-Mirror ist ein
zusätzlicher Maximaldurchsatzwert, kein Ersatz.

Danach dokumentieren `docs/deepseek-v4-tuning-32gb.md`,
`docs/ENVIRONMENT.md` und `plans/00-reference.md` das echte Profil. Default
bleibt unverändert; Turbo3, DSpark und Chunking sind Opt-ins.

### 1M-Experiment

Erst nach grüner 256k-Abnahme:

```bash
CTX=1048576 V4_KV=turbo3 V4_PREFILL_CHUNK=32768 \
V4_MTP=1 V4_DRAFT=3 V4_VRAM=1
```

Der Planner muss die reale Device-Reserve zulassen. Gemessen werden Peak
RSS/VRAM, 32 Prefill-Chunks, TTFT und Langkontext-Decode. Der Indexer-Scan
wächst bei 1M auf rund 2.69 GB pro Decode-Token; ein erfolgreicher Lauf macht
1M zu einem dokumentierten Experiment, nicht zum Standardprofil.

## Abnahme

- 256k, Chunk 32k, Turbo3 und DSpark laufen auf 32 GiB + RTX 4070 ohne eine
  32-GiB-Aktivierungsallokation.
- Chunked Target-only ist greedy token-identisch zu unchunked Target-only;
  Prefix-Reuse bleibt erhalten.
- Chunked DSpark startet erst nach dem finalen Prefill-Chunk, verifiziert exakt
  gegen Target und liefert dieselbe Tokenfolge.
- Das 256k-Profil enthält kalte und prozesswarme Messungen auf einem Laufwerk,
  plus optionalen Mirror-Lauf, nicht nur extrapolierte tok/s.
- `make -C c test && make -C c check` sowie `make -C c deepseek-v4-tiny-check`
  sind grün.

## Risiken

- Chunkgrenzen können Annahmen über vollständige Prompt-Matrizen in Attention,
  Compressor, Indexer, Prefix-Reuse und MTP-Taps verletzen; absolute Positionen
  und vollständige Token-A/Bs sind wichtiger als TTFT.
- Kleine Chunks können Prefill durch wiederholtes Laden nichtresistenter Experten
  massiv verlangsamen.
- Turbo3 ist absichtlich verlustbehaftet; echte Langkontext-Aufgaben und
  Token-/Qualitätsproben entscheiden, nicht allein eine Cosinuszahl.
- 1M kann in Speicher passen und durch Indexer-Scan oder Prefill-I/O trotzdem
  unbenutzbar sein. Das ist ein Messergebnis, kein Anlass für stille Abstriche.
