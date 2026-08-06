# 13 — Frontend auf V4-only: WebUI, CLI-Chat, Server

Voraussetzung: [00-reference.md](00-reference.md), [08-vram-planner.md](08-vram-planner.md)

*Commits:*
1. `feat: V4-only model dispatch in the coli launcher`
2. `feat: surface the RAM/VRAM tier plan in the web dashboard`
3. `chore: drop the multi-engine surface from the web UI`

## Ziel

`coli chat`, `coli serve` und `coli web` bleiben — sie sind die tägliche
Bedienoberfläche. Was verschwindet, ist die Mehr-Modell-Logik dahinter, und was
dazukommt, ist Sichtbarkeit für genau die Stufen, die die Pläne 05–08 einführen.

**Dieser Plan ist optional.** Wenn nach Plan 08 alles läuft, ist er Komfort. Er
wird nützlich, sobald man Konfigurationen vergleichen will, ohne stderr zu lesen.

## Ausgangslage

| Teil | Ort | Umfang |
|---|---|---|
| Launcher | [c/coli](../c/coli) | 1421 Zeilen Python, alle Subkommandos |
| OpenAI-Server | [c/openai_server.py](../c/openai_server.py) | 2861 Zeilen |
| WebUI | [web/](../web) | Vite + React + TS, **23 Dateien** in `src/` |
| Statische Seite | [site/](../site) | Landingpage |
| pip-CLI | [colibri/cli.py](../colibri/cli.py) | dünner Wrapper |
| Desktop | [desktop/](../desktop) | Tauri-Hülle |

Die Mehr-Modell-Logik sitzt im Launcher: `model_arch(model)`
([c/coli:229](../c/coli)), `engine_for(model)` ([:243](../c/coli)),
`env_for_engine(a, arch)` ([:266](../c/coli)), `cuda_binary()` ([:316](../c/coli)).
Sie erkennen die Architektur aus der `config.json` und wählen Binary und
Env-Vorgaben.

In der WebUI erwähnt nur `web/src/App.tsx` überhaupt Modellnamen — die
Modellkopplung ist also klein. 23 Dateien sind überschaubar.

## Commit 1 — Launcher

`model_arch`/`engine_for` schrumpfen auf: erkenne DeepSeek V4, sonst
verständlicher Fehler. `env_for_engine` verliert die Zweige für GLM, Inkling,
Kimi, OLMoE und bekommt stattdessen die Profil-Defaults aus
[08-vram-planner.md](08-vram-planner.md) als Vorbelegung — `RAM_GB`,
`V4_SCRATCH_MB`, `V4_KV`, `V4_VRAM`, jeweils überschreibbar.

`cmd_doctor` ([:740](../c/coli)) und `cmd_plan` ([:722](../c/coli)) sind dabei die
interessanteren Kandidaten: sie melden heute allgemeine Hardwarelage. Nach Plan 08
sollten sie stattdessen **den Stufenplan zeigen** — was landet in VRAM, was in
RAM, wie viele Expert-Slots ergibt das. Das ist dieselbe Information wie die
`ram_tiers`/`vram_tiers`-Zeilen, nur bevor man startet.

**`cmd_tune`** ([:768](../c/coli)) ist der natürliche Ort für die Messungen aus
[12-expert-cache-policy.md](12-expert-cache-policy.md) und
[10-dual-streaming.md](10-dual-streaming.md): Laufwerksbandbreite proben,
Pin-Konfigurationen durchmessen, Empfehlung ausgeben.

## Commit 2 — Dashboard

Die WebUI zeigt laut README des Upstreams Live-Metriken, Hardware-Panel und
Expert-Tiers. Was dort fehlt und nach den Plänen 05–08 existiert:

- **VRAM-Stufen**: welche der vier Posten (KV, Dense, Head, DSpark) liegen auf der
  GPU, welche im RAM, wieviel Reserve bleibt. Entspricht der `vram_tiers`-Zeile.
- **Trefferquote des Expert-Caches** über die Zeit — die Zahl, die nach Plan 12
  am meisten über die Leistung sagt.
- **Laufwerksaufteilung** aus Plan 10: Bytes pro Laufwerk, damit sichtbar ist, ob
  die gewichtete Aufteilung greift.
- **KV-Codec und Kontextlänge**, weil beide das VRAM-Budget bestimmen.

Der Transportweg dafür existiert vermutlich schon (das Dashboard bekommt
Metriken vom Server); zu klären ist, ob `openai_server.py` sie durchreicht oder
ob die Engine sie erst exportieren muss. **Das ist die erste Aufgabe dieses
Commits, vor jeder UI-Arbeit.**

## Commit 3 — Aufräumen

Aus der WebUI die Auswahl mehrerer Modellfamilien entfernen; aus `site/` und
`README`-Bildern das, was andere Modelle zeigt.

**`desktop/` (Tauri):** eigene Entscheidung. Wenn du die Desktop-Hülle nicht
benutzt, kann sie weg — dann aber vollständig, inklusive Build-Schritten. Halb
entfernt ist schlimmer.

## Verhältnis zu Plan 11

[11-strip-to-v4.md](11-strip-to-v4.md) entfernt die **Engines**, dieser Plan die
**Oberfläche** darüber. Sie können in beliebiger Reihenfolge laufen, aber:

- Läuft 13 zuerst, bricht 11 nichts mehr im Launcher.
- Läuft 11 zuerst, ist der Launcher zwischenzeitlich kaputt (er verweist auf
  gelöschte Binaries).

**Empfehlung: 13 vor 11.** Dann ist der Baum nie in einem Zustand, in dem
`coli chat` nicht startet.

## Tests

- `c/tests/test_cli_output.py` existiert und deckt Launcher-Ausgaben ab — er muss
  mitziehen, nicht umgangen werden.
- `c/tests/test_doctor_engine_arch.py` prüft die Architekturerkennung; nach
  Commit 1 testet er den V4-Pfad und den verständlichen Fehler für alles andere.
- WebUI: `npm run build` in `web/` muss durchlaufen. Wenn es dort Tests gibt,
  laufen sie mit.
- Manuell, weil es nicht anders geht: `coli chat`, `coli serve` + eine Anfrage,
  `coli web` + Dashboard im Browser.

## Abnahme

- `coli chat`, `coli serve`, `coli web` starten gegen den V4-Checkpoint.
- `coli plan` zeigt den Stufenplan vor dem Start, konsistent mit dem, was die
  Engine dann tatsächlich meldet.
- Das Dashboard zeigt VRAM-Stufen, Trefferquote und Laufwerksaufteilung.
- Ein nicht unterstütztes Modellverzeichnis gibt eine klare Meldung, keinen
  Traceback.

## Risiken

- **`openai_server.py` ist 2861 Zeilen** und der Teil, der am ehesten stillschweigend
  auf Engine-Verhalten baut. Vor Änderungen prüfen, was er über `coli` und die
  Engine annimmt.
- **Die Metriken könnten noch nicht existieren.** Wenn die Engine Trefferquote und
  Stufenplan nicht exportiert, ist Commit 2 erst ein Engine-Commit und dann ein
  UI-Commit. Das ist in Ordnung — aber es gehört erkannt, bevor jemand an der UI anfängt.
- **Optionalität ernst nehmen.** Wenn die Zeit knapp ist, ist dieser Plan der
  erste, der wartet. Er macht nichts schneller.
