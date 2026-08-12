# 10 — Dual-Streaming über ein optionales zweites DRAM-loses SSD

Voraussetzung: [00-reference.md](00-reference.md)

*Commits:*
1. `feat: port the multi-drive mirror machinery from colibri.c to V4`
2. `perf: bandwidth-weighted stripe chunks for asymmetric drives`

## Das zweite Laufwerk ist optional — und bleibt es

**Ein Laufwerk ist die Grundkonfiguration, nicht der Notfall.** Das zweite
Laufwerk ist ein Beschleuniger, den dieses Setup zufällig danebenliegen hat; der
Plan darf es nicht zur Voraussetzung machen. Verbindlich für beide
Implementierungscommits:

- **Default ist Einzellaufwerk.** Ohne `COLI_MODEL_MIRROR` bzw. `COLI_MODEL_DIRS`
  läuft exakt der heutige Lesepfad — dieselben `pread`s, dieselbe Koaleszenz,
  dieselben Tokens. Das ist die Regel „Default = heutiges Verhalten" aus
  [AGENTS.md](../AGENTS.md), hier ohne Ausnahme.
- **Aus heißt aus, nicht „mit `nsf == 1`".** Ist nur eine Replik registriert, wird
  weder geprobt noch geroutet noch ein zweiter Thread aufgemacht. Kein
  Bandbreiten-Probe beim Start, kein `expert_route`-Aufruf pro Read, kein
  Striper-Setup — der Aus-Pfad kostet **null**, sonst zahlt die Grundkonfiguration
  für ein Feature, das sie nicht benutzt.
- **Der Aus-Pfad ist der Pflicht-Pfad in der Abnahme.** Alle Gates laufen ohne
  zweites Laufwerk grün; die Dual-Gates kommen zusätzlich obendrauf, nicht
  stattdessen. Siehe [Abnahme](#abnahme).
- **HMB-Profiling gilt für beide Konfigurationen.** Es hängt am einzelnen
  Laufwerk, nicht an ihrer Zahl, ist aber eine hardwaregebundene Messaufgabe und
  liegt deshalb in [Plan 15](15-final-validation-and-tuning.md).

Damit ist der Plan auch auf einer Maschine mit nur Laufwerk A vollständig
umsetzbar; es entfallen lediglich die Gewinne aus Commit 2.

## Zielhardware

| | Kapazität | Schnittstelle | DRAM | |
|---|---|---|---|---|
| Laufwerk A | 1 TB | **PCIe Gen4** ×4 | **nein** (HMB) | Modell liegt hier, **Pflicht** |
| Laufwerk B | 512 GB | **PCIe Gen3** ×4 | **nein** (HMB) | zweite Kopie, **optional** |

Das Modell ist ~167 GB. **Beide Laufwerke können eine vollständige Kopie halten**
(167 < 512). Damit ist Spiegelung möglich, nicht nur Aufteilung — und Spiegelung
ist für nur lesende Gewichte die bessere Wahl: jeder Expert ist auf beiden
Laufwerken, also frei routbar und stripebar.

**Warum das überhaupt zählt.** Bei rund 17 % Expert-Residenz im 32k-Endprofil
liest jeder Token 43 Layer × 6 Experten × ~13.4 MB ≈ **3.4 GB** (siehe
Bandbreitenabschnitt in [00-reference.md](00-reference.md)), davon ~83 % von der
Platte, also ~2.8 GB/Token. Auf Laufwerk A allein sind das bei ~7 GB/s
**~0.40 s/Token** — das ist die Zahl der Grundkonfiguration, gegen die Commit 2
antritt. Bei 128k sinkt die Residenz auf rund 8 %, entsprechend ~3.1 GB/Token oder
~0.44 s. Storage ist damit nach den VRAM-Phasen der dominante Posten; beide
Kontexte gehören in die Abschlussmessreihe, und zwar **je einmal mit und ohne
zweites Laufwerk** — sonst ist der Beitrag des Laufwerks nicht von dem der
Implementierung zu trennen. Diese Matrix liegt in
[Plan 15](15-final-validation-and-tuning.md).

## Der Fund: das meiste existiert schon — im falschen Motor

`c/colibri.c` (GLM) hat die komplette Multi-SSD-Maschinerie. `c/deepseek_v4.c`
hat **nichts davon** und liest mit einfachem `pread` auf eine einzige Kopie
([c/deepseek_v4.c:104](../c/deepseek_v4.c), [:6886](../c/deepseek_v4.c)).

Vorhanden in colibri.c:

| Baustein | Ort | Was |
|---|---|---|
| `COLI_MODEL_MIRROR` | [:2004](../c/colibri.c) | Registriert byte-identische Kopien auf anderen Laufwerken |
| `COLI_MODEL_DIRS` | `docs/ENVIRONMENT.md:167` | Alternativ: Shards **aufteilen** statt spiegeln |
| `expert_route(layer, eid)` | [:2023](../c/colibri.c) | Deterministischer Hash → Replik, über 256er-Cut-Tabelle |
| `g_mir_cut[]` | [:2014](../c/colibri.c) | **Gewichtete** kumulative Cuts |
| `COLI_DISK_WEIGHTS` | [:7928](../c/colibri.c) | Explizite Gewichte, Beispiel in der Fehlermeldung: `9,3` |
| `mirror_probe_bw` | [:7941](../c/colibri.c) | **Automatische Bandbreitenmessung** pro Laufwerk |
| `mir_pread_striped` | [:2056](../c/colibri.c) | Ein ≥4 MB O_DIRECT-Read parallel über alle Repliken |
| Per-Laufwerk-Statistik | [:6447](../c/colibri.c) | `MIRROR:`-Zeile mit Bytes und Reads je Drive |

In `st.h` liegt die gemeinsame Basis, die **beide** Motoren nutzen könnten:
`ST_MAX_MIR = 4`, `mfds[][]`/`mdfds[][]`, `st_fd_rep`, `st_direct_fd_rep`,
`st_mirror_add`, `st_prefetch_rep` ([c/st.h:50](../c/st.h) ff.).

Der Designkommentar bei `expert_route` nennt die Randbedingung, die man nicht
verletzen darf:

> *Determinismus ist eine Anforderung, kein Stilmittel: Readahead/WILLNEED und
> der Demand-Pread müssen denselben fd und dieselbe Page-Cache-Seite treffen, und
> im gepufferten Modus darf ein Expert nie zweimal gecacht werden (eine Kopie pro
> Laufwerk).*

## Commit 1 — Mirror-Maschinerie nach V4 portieren

Die Bausteine aus colibri.c in einen gemeinsamen Header ziehen — `c/mirror.h`,
im Stil von `kv_prefix.h`/`omp_tune.h` — und von beiden Motoren benutzen. Damit
ist es kein Copy-Paste und Plan 11 (Rückbau) kann colibri.c löschen, ohne die
Logik mitzunehmen.

**Das ist die einzige Stelle des Branches, die `colibri.c` anfasst.**
[00-reference.md](00-reference.md) sagt unter „Was dieser Branch nicht macht", dass
die anderen Motoren unangetastet bleiben; dieser Commit ist dort als benannte
Ausnahme eingetragen. Sie gilt nur für den Refactor ohne Verhaltensänderung, und
sie kostet einen zusätzlichen Abnahmeschritt: `make -C c colibri` plus die
GLM-Gates müssen grün bleiben, nicht nur die V4-Gates. Wird das zu teuer, greift
der Rückfall aus den Risiken unten (V4-lokale Kopie mit Verweis) — dann entfällt
die Ausnahme und der Eintrag in 00 auch.

Inhalt: `expert_route`, die Cut-Berechnung, `mirror_probe_bw`, `mir_pread_striped`
und die Statistikzähler. Alles hängt nur an `shards *` aus `st.h`, nicht an
`Model`/`Cfg` — der Schnitt ist sauber.

Anzubinden in V4:

- `coli_st_index_open` ([c/st.h](../c/st.h)) ruft nach dem Öffnen
  `st_mirror_add` für jeden Eintrag aus `COLI_MODEL_MIRROR`. Ist die Variable
  ungesetzt oder zeigt sie auf ein Verzeichnis ohne passende Shards, bleibt es bei
  einer Replik — **ohne Fehler und ohne Probe**, nur mit einer Zeile im Log.
- Der Expert-Lesepfad (`v4_pread_full_try`, [:5639](../c/deepseek_v4.c), und der
  Slab-Pfad um [:5861](../c/deepseek_v4.c)) routet über `expert_route(layer, eid)`
  und versucht `mir_pread_striped` vor dem Einzel-Read.
- `coli_st_prefetch_at` ([:188](../c/deepseek_v4.c)) muss `posix_fadvise` auf
  **denselben** fd absetzen, den der spätere Demand-Read nimmt — sonst wärmt man
  die falsche Page-Cache-Seite. Das ist genau der Determinismus-Punkt oben.

**Der Aus-Pfad wird explizit gebaut, nicht angenommen.** Ein einziges
`if (nrep < 2)` ganz oben in den Lesepfaden springt auf den heutigen Code, bevor
Routing oder Striper überhaupt betrachtet werden — ein vorhersagbar sprungfreier
Zweig, kein Durchlaufen der Mirror-Logik mit `nsf == 1`. Dasselbe beim Start:
`mirror_probe_bw` läuft nur, wenn mindestens zwei Repliken registriert sind, sonst
kostet die Grundkonfiguration Sekunden Startzeit für eine Messung, aus der nichts
folgt. Auch `COLI_DISK_WEIGHTS` wird dann nicht gelesen.

**Reihenfolge:** Dieser Commit muss **vor** Plan 11 liegen. Wird colibri.c vorher
gelöscht, ist die Referenzimplementierung weg.

## Commit 2 — Gewichtete Stripe-Chunks

Hier liegt der eigentliche Beitrag für **asymmetrische** Laufwerke — und **nur
hier** wird das zweite Laufwerk gebraucht. Ohne zweites Laufwerk ist dieser Commit
ein Test und tote Codepfade; das ist in Ordnung und ändert nichts an der
Reihenfolge, aber es ist der Grund, warum die Abnahme zweigeteilt ist.

`mir_pread_striped` teilt heute **gleichmäßig** ([:2064](../c/colibri.c)):

```c
int64_t chunk = ((len + nsf-1)/nsf + 4095) & ~4095LL;
```

Bei Gen4 (~7 GB/s) und Gen3 (~3.5 GB/s) bedeutet das: beide bekommen 6.7 MB des
13.4-MB-Experten, der Gen4-Thread ist nach ~1.0 ms fertig und **wartet ~1.0 ms**
auf den Gen3-Thread. Effektiver Durchsatz 7 GB/s statt der möglichen 10.5 GB/s —
der komplette Gewinn des zweiten Laufwerks verpufft.

Richtig ist eine Aufteilung **proportional zur gemessenen Bandbreite**. Die
Gewichte gibt es schon: `mirror_probe_bw` liefert GB/s pro Replik, das
Routing-Cut-Table benutzt sie bereits. Der Striper muss dieselben Gewichte
verwenden:

```c
/* Chunk-Grenzen proportional zu wt[], auf 4K aufgerundet; der letzte
 * Chunk absorbiert die Rundung. Bei 7.0 / 3.5 GB/s ergibt das 2:1
 * statt 1:1 und die Threads werden gleichzeitig fertig. */
int64_t bound[MIR_REPS+1];
double W = 0; for (int i = 0; i < nsf; i++) W += wt[srep[i]];
double cum = 0; bound[0] = 0;
for (int i = 0; i < nsf; i++) {
    cum += wt[srep[i]];
    bound[i+1] = ((int64_t)(len * cum / W) + 4095) & ~4095LL;
    if (bound[i+1] > len) bound[i+1] = len;
}
```

Erwartung: von 7 auf ~10 GB/s aggregiert, also beim 32k-Profil **~0.28 s statt
~0.40 s pro Token** und bei 128k **~0.31 s statt ~0.44 s** (jeweils rund 30 %
weniger Streaming-Zeit).

Die Einschränkung `len >= 4 MB` bleibt richtig — bei kleineren Reads dominiert die
Latenz und zwei Threads kosten mehr als sie bringen. Ein Expert-Record von
~13.4 MB liegt komfortabel darüber.

Bei einer Replik ist `nsf == 1`, `W == wt[0]` und die Schleife liefert
`bound = {0, len}` — der Striper degeneriert korrekt zum Einzel-Read. Das ist die
Rückfallebene für den Fall, dass der Aus-Pfad aus Commit 1 einmal nicht greift,
**nicht** der Aus-Pfad selbst: der liegt davor und spart auch den Aufruf.

## Abschlussprofil: DRAM-lose SSDs

Die folgenden HMB-/Queue-Experimente gehören in
[Plan 15](15-final-validation-and-tuning.md). Sie sind kein dritter
Implementierungscommit und setzen keinen Default, bevor die vollständige Matrix
einen reproduzierbaren Gewinn zeigt.

**Diese Profilierung hängt nicht am zweiten Laufwerk.** Sie gilt pro Laufwerk und
trägt in der Grundkonfiguration genauso wie in der gespiegelten — wer nur
Laufwerk A hat, erhält hier trotzdem einen vollständigen Messfall.

Die Laufwerke haben **keinen eigenen DRAM** und nutzen HMB (Host Memory Buffer),
also geliehenen Host-RAM für die FTL-Mapping-Tabelle — typisch 64 MB.

Zwei Konsequenzen:

**Große sequentielle Reads sind der gute Fall.** Ein Expert-Record ist
`w1+w2+w3` plus Scales, zusammengefasst gelesen: ~13.4 MB am Stück. Das ist genau
das Zugriffsmuster, bei dem DRAM-lose Laufwerke kaum schlechter sind als solche
mit DRAM — die FTL muss wenige Mapping-Einträge nachladen. **Der bestehende
Coalescing-Pfad ist damit wichtiger als auf DRAM-Laufwerken**, nicht weniger.
Vor Änderungen an der Lesekoaleszenz also prüfen, dass sie erhalten bleibt.

**Tiefe Queues mit vielen kleinen Reads sind der schlechte Fall.** HMB-Laufwerke
brechen bei hoher Queue-Tiefe mit verstreuten Zugriffen deutlich ein, weil
FTL-Misses über PCIe nachgeladen werden. Der persistente Lookup-Pipeline-Pfad
(`COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER`, drei Worker,
[c/deepseek_v4.c:4289](../c/deepseek_v4.c)) sollte deshalb **messbar** eingestellt
werden statt auf gut Glück:

- `COLI_V4_EXPERT_PREFETCH` ([:3251](../c/deepseek_v4.c)) und die Workerzahl gegen
  die tatsächliche Lesezeit pro Token messen
- HMB-Größe prüfen: `nvme id-ctrl /dev/nvme0 | grep -i hmb` bzw.
  `cat /sys/class/nvme/nvme0/device/hmb` je nach Treiber
- Kernel-Queue-Tiefe: `cat /sys/block/nvme0n1/queue/nr_requests`,
  Scheduler `none` ist für NVMe richtig und auf Arch Default

**Auf `/sys/block/*/queue/read_ahead_kb` achten**, wenn gepuffert gelesen wird:
Readahead über einen O_DIRECT-Pfad ist wirkungslos, über den gepufferten
verdoppelt er bei bereits koaleszierten 12-MB-Reads nur die Arbeit.

Ergebnis ist primär eine Messung mit dokumentierten Einstellungen im Tuning-Doc,
nicht notwendig Code. Nur ein belegter Gewinn darf daraus einen Profilwert
ableiten.

## Die drei Konfigurationen

| Konfiguration | Wie | Erwartung 32k | Wann |
|---|---|---|---|
| **Einzellaufwerk** (Default) | nichts setzen | ~7 GB/s, ~0.40 s/Token | immer lauffähig, Grundkonfiguration |
| **Spiegelung** | `COLI_MODEL_MIRROR` | ~10 GB/s, ~0.28 s/Token | zweites Laufwerk mit ≥167 GB frei |
| **Aufteilung** | `COLI_MODEL_DIRS` | ~7 GB/s, Kapazität summiert | zweites Laufwerk, aber zu klein für eine volle Kopie |

Die Aufteilung legt jeden Shard genau einmal ab: die Kapazität summiert sich, aber
ein Expert liegt nur auf *einem* Laufwerk — kein Striping, kein
Routing-Freiheitsgrad, also kein Bandbreitengewinn. Sie ist die Antwort auf zu
wenig Platz, nicht auf zu wenig Durchsatz.

**Für dieses Setup ist Spiegelung klar besser** (167 GB × 2 = 334 GB von 1.5 TB —
der Platz reicht), weil beide Freiheitsgrade erhalten bleiben. Im Tuning-Doc als
bewusste Entscheidung festhalten, nicht als Zufall — zusammen mit der Zahl für die
Grundkonfiguration, damit ablesbar bleibt, was das zweite Laufwerk wirklich bringt.

## Tests

**Kein Test setzt ein zweites physisches Laufwerk voraus.** Zwei Verzeichnisse auf
demselben Dateisystem reichen für alles außer der Bandbreitenmessung; die Gates
laufen damit auch in CI und auf einer Maschine mit einer SSD.

- `c/tests/test_v4_mirror.c` — neue Make-Regel:
  - `expert_route` ist deterministisch und verteilt über viele `(layer, eid)`
    proportional zu den Cuts (χ²-artige Grobprüfung, keine Exaktheit).
  - Gewichtete Chunk-Grenzen: bei `wt = {7, 3.5}` und `len = 13.4 MB` sind die
    Grenzen 4K-aligned, lückenlos, summieren auf `len`, Verhältnis ≈ 2:1.
  - Entartete Fälle: eine Replik, `wt` gleich, `len` knapp über/unter 4 MB,
    `len` kleiner als `nsf × 4096`.
  - Partieller Mirror: Shard nur auf der Primärkopie → Fallback ohne Striping.
- `c/tests/test_v4_mirror_off.c` — **der Aus-Pfad als eigenes Gate**: ohne
  `COLI_MODEL_MIRROR` bleibt der Mirror-Zähler bei einer Replik, der
  Bandbreiten-Probe läuft **nicht** (Zähler oder Log-Marke prüfen) und der
  Lesepfad liefert byteidentische Puffer wie vor Commit 1. Der Test ist die
  Zusicherung, dass die Grundkonfiguration nichts für das Feature bezahlt.
- `c/tests/test_v4_mirror_fallback.c` — ein Lesefehler auf der Replik fällt auf
  die Primärkopie zurück und loggt **einmal**, nicht pro Read. Dazu: ein
  `COLI_MODEL_MIRROR`, das auf ein nicht existierendes oder shard-freies
  Verzeichnis zeigt, startet normal auf einer Replik statt abzubrechen.
- End-to-end mit zwei Verzeichnissen auf demselben Dateisystem (Kopie des
  Tiny-Fixtures): identische Tokens mit und ohne `COLI_MODEL_MIRROR`.

## Implementierungsabnahme

**Pflicht — ohne zweites Laufwerk, auf jeder Maschine nachvollziehbar:**

- `make -C c test && make -C c check` und `deepseek-v4-tiny-check` grün, Tokens
  identisch zum Stand vor dem Branch.
- `test_v4_mirror_off` belegt: eine Replik, kein Probe, kein Verhaltensdelta.

**Zusätzlich, wenn ein zweites Laufwerk steckt:**

- `MIRROR:`-Statistikzeile zeigt beide Laufwerke mit Bytes und Reads.
- Tokenfolge unverändert, mit und ohne Spiegel identisch. Storage-Routing darf nie
  Semantik berühren.

Die Start-/Decodezeit-, HMB- und physische Zwei-Laufwerk-Messmatrix ist Abnahme
von Plan 15. Fehlt Laufwerk B, bleibt dieser Plan mit den semantischen Gates
abnahmefähig; der optionale Hardwarevergleich wird dort nachgeholt.

Der PR ist mit dem Pflichtteil abnahmefähig. Fehlt das zweite Laufwerk, wird das
in der PR-Beschreibung vermerkt und der Dual-Teil nachgereicht — er wird nicht
weggelassen und nicht geschätzt.

## Risiken

- **Der Aus-Pfad verrottet still.** Ein Feature, das der Entwickler immer an hat,
  wird ungetestet — und dann kostet die Grundkonfiguration irgendwann doch einen
  Probe beim Start oder einen Routing-Aufruf pro Read. Dagegen steht
  `test_v4_mirror_off` als Gate; die Messreihen in [01](01-measure-and-ram-budget.md)
  laufen weiterhin **ohne** Spiegel, damit die Baseline vergleichbar bleibt.
- **Zwei Kopien = doppelte Schreiblast beim Anlegen** (167 GB je Laufwerk). Bei
  DRAM-losen QLC/TLC-Laufwerken ohne SLC-Cache-Reserve kann das lange dauern und
  den Cache füllen. Einmalig, aber im Tuning-Doc erwähnen — und ein Grund, die
  Spiegelung erst anzulegen, wenn Commit 2 tatsächlich gemessen werden soll.
- **Der Probe misst Momentanbandbreite.** Ein Laufwerk im thermischen Throttling
  oder mit vollem SLC-Cache misst niedriger, als es im Betrieb liefert. Deshalb
  bleibt `COLI_DISK_WEIGHTS` als manueller Override — für dieses Setup ist `2,1`
  eine vernünftige feste Vorgabe.
- **Der Probe misst die Laufwerke einzeln, benutzt werden sie gleichzeitig.**
  `mirror_probe_bw` läuft pro Replik nacheinander ([c/colibri.c:7941](../c/colibri.c)).
  Auf dem i5-13400F hängen die beiden Laufwerke fast sicher an verschiedenen
  Anbindungen: 16 PCIe-5.0- plus 4 PCIe-4.0-Lanes kommen von der CPU (die x16
  gehen an die GPU, die x4 an einen M.2-Steckplatz), weitere M.2-Plätze hängen am
  Chipsatz und teilen sich dessen DMI-Uplink mit USB, SATA und Netzwerk.
  DMI 4.0 ×8 (B760/H770/Z790) reicht mit ~15.8 GB/s für ein Gen3-Laufwerk locker,
  DMI ×4 (H610) mit ~7.9 GB/s ebenfalls — aber die Summe zweier einzeln gemessener
  Werte kann den **gleichzeitigen** Durchsatz überschätzen, und dann sind die
  Gewichte falsch.
  **Deshalb zusätzlich parallel messen** (beide Repliken gleichzeitig lesen) und
  die Gewichte daraus ableiten. Weicht die parallele Summe deutlich von der
  sequentiellen ab, teilen sich die Laufwerke einen Flaschenhals — das gehört ins
  Tuning-Doc, weil es auch die Erwartung an das Striping ändert.
  Zum Nachsehen, welches Laufwerk woran hängt:
  `lspci -tv | grep -i nvme` und `ls -l /sys/block/nvme*n1/device`.
- **Page-Cache-Doppelung** im gepufferten Modus, falls das deterministische
  Routing bricht. `DIRECT=1` umgeht das; die V4-Entsprechung ist
  `COLI_V4_DIRECT` ([:116](../c/deepseek_v4.c)). Im Profil setzen.
- **Der gemeinsame Header ist ein Refactor an laufendem Code.** colibri.c ist
  ~9500 Zeilen und der GLM-Pfad ist nicht Ziel dieses Branches — wenn der Schnitt
  zu teuer wird, ist eine V4-lokale Kopie mit Verweis auf das Original die
  ehrlichere Zwischenlösung, solange Plan 11 sie erbt.
