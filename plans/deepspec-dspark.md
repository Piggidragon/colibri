# DeepSpec — die DSpark-Referenzimplementierung

Quelle: `reference/DeepSpec` — DeepSeeks eigenes Repo für Training und Evaluation
von Draft-Modellen für spekulatives Dekodieren. Nicht committen (`.gitignore`).

Das ist die **maßgebliche** Quelle für DSpark. `reference/llama-cpp-turboquant`
([llamacpp-deepseek-v4.md](llamacpp-deepseek-v4.md)) ist eine Nachimplementierung;
bei Abweichungen gilt DeepSpec.

## Was drin ist

```
deepspec/modeling/dspark/
  common.py          309   gemeinsame Bausteine
  markov_head.py     319   der semi-autoregressive Markov-Head
  loss.py            334   Trainingsverlust (für uns irrelevant)
  qwen3/  gemma4/          Backbone-spezifische Varianten

deepspec/eval/dspark/
  draft_ops.py       153   ← INFERENZ: Draft-Block-Erzeugung
  confidence_head.py 605   ← Confidence-Head
  evaluator.py       225   ← Akzeptanzschleife
```

**Für colibri zählt `eval/dspark/`, nicht `modeling/`.** Dort steht die
Inferenzzeit-Mechanik, die `c/deepseek_v4_dspark.inc` nachbildet; `modeling/` und
`trainer/` sind Trainingscode.

Vorsicht: die Configs unter `config/dspark/` sind für **Qwen3** und **Gemma4** als
Ziel, nicht für DeepSeek-V4. Die Architekturparameter dort gelten nicht für
`DeepSeek-V4-Flash-DSpark` — die stehen im Checkpoint. Der *Algorithmus* ist
derselbe.

## Was `draft_ops.py` über den Algorithmus verrät

`forward_dspark_draft_block` bestätigt drei Dinge, die colibris Config-Felder
schon andeuten (`dspark_block_size`, `dspark_noise_token_id`, `dspark_markov_rank`
in [c/deepseek_v4.h:31](../c/deepseek_v4.h)):

**Der Draft-Block ist nicht kausal.**

```python
block_hidden = model._forward_backbone(..., is_causal=False)
```

Alle Positionen eines Draft-Blocks sehen einander — das ist das
„semi-autoregressive" aus dem llama.cpp-Kommentar. Ein Block wird in *einem*
Vorwärtspass erzeugt, nicht Token für Token.

**Der Drafter konditioniert auf Noise-Embeddings plus Target-Hidden-States.**

```python
noise_embedding=model.embed_tokens(draft_input_ids),
target_hidden_states=target_hidden_states,
```

Erklärt `dspark_noise_token_id`: die Blockpositionen starten als Noise-Token und
werden durch die Hidden States des Zielmodells konditioniert.

**Der Draft-KV-Ring wird nach jedem Block zurückgeschnitten.**

```python
past_key_values_draft.crop(start)
```

Entspricht dem „uniform sliding window (the draft KV ring)" in
`llama-cpp-turboquant/src/models/dflash.cpp:23`.

## Wofür das nützlich ist

Für [07-head-dspark-vram.md](07-head-dspark-vram.md), wenn der Drafter auf die GPU
wandert: die Akzeptanzrate muss unverändert bleiben, und `evaluator.py` zeigt,
welche Größen dabei überhaupt verglichen werden.

Und als Schiedsrichter, falls colibris Drafter je verdächtige Akzeptanzraten
liefert — `confidence_head.py` und `markov_head.py` sind die Referenz für die
beiden Köpfe, die `deepseek_v4_dspark.inc` implementiert.

**Kein eigener Plan.** DSpark funktioniert in colibri; dieser Branch verschiebt ihn
nur in den VRAM. Sollte sich dabei herausstellen, dass die Implementierung von der
Referenz abweicht, wird das ein eigener Plan — dann mit DeepSpec als Maßstab.
