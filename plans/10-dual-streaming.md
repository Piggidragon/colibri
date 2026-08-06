# 10 — Dual-Streaming über zwei DRAM-lose SSDs

Voraussetzung: [00-reference.md](00-reference.md)

*Commits:*
1. `feat: port the multi-drive mirror machinery from colibri.c to V4`
2. `perf: bandwidth-weighted stripe chunks for asymmetric drives`
3. `perf: HMB-aware queue depth for DRAM-less SSDs`

## Zielhardware

| | Kapazität | Schnittstelle | DRAM |
|---|---|---|---|
| Laufwerk A | 1 TB | **PCIe Gen4** ×4 | **nein** (HMB) |
| Laufwerk B | 512 GB | **PCIe Gen3** ×4 | **nein** (HMB) |

Das Modell ist ~167 GB. **Beide Laufwerke können eine vollständige Kopie halten**
(167 < 512). Damit ist Spiegelung möglich, nicht nur Aufteilung — und Spiegelung
ist für nur lesende Gewichte die bessere Wahl: jeder Expert ist auf beiden
Laufwerken, also frei routbar und stripebar.

**Warum das überhaupt zählt.** Bei ~20 % Expert-Residenz nach den VRAM-Phasen
liest jeder Token 43 Layer × 6 Experten × ~12.6 MB ≈ **3.25 GB**, davon ~80 % von
der Platte, also ~2.6 GB/Token. Auf Laufwerk A allein sind das bei ~7 GB/s
**0.37 s/Token**. Storage ist damit nach den VRAM-Phasen der dominante Posten.

## Der Fund: das meiste existiert schon — im falschen Motor

`c/colibri.c` (GLM) hat die komplette Multi-SSD-Maschinerie. `c/deepseek_v4.c`
hat **nichts davon** und liest mit einfachem `pread` auf eine einzige Kopie
([c/deepseek_v4.c:100](../c/deepseek_v4.c), [:5644](../c/deepseek_v4.c)).

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

Inhalt: `expert_route`, die Cut-Berechnung, `mirror_probe_bw`, `mir_pread_striped`
und die Statistikzähler. Alles hängt nur an `shards *` aus `st.h`, nicht an
`Model`/`Cfg` — der Schnitt ist sauber.

Anzubinden in V4:

- `coli_st_index_open` ([:27](../c/deepseek_v4.c)) ruft nach dem Öffnen
  `st_mirror_add` für jeden Eintrag aus `COLI_MODEL_MIRROR`.
- Der Expert-Lesepfad (`v4_pread_full_try`, [:5639](../c/deepseek_v4.c), und der
  Slab-Pfad um [:5861](../c/deepseek_v4.c)) routet über `expert_route(layer, eid)`
  und versucht `mir_pread_striped` vor dem Einzel-Read.
- `coli_st_prefetch_at` ([:188](../c/deepseek_v4.c)) muss `posix_fadvise` auf
  **denselben** fd absetzen, den der spätere Demand-Read nimmt — sonst wärmt man
  die falsche Page-Cache-Seite. Das ist genau der Determinismus-Punkt oben.

**Reihenfolge:** Dieser Commit muss **vor** Plan 11 liegen. Wird colibri.c vorher
gelöscht, ist die Referenzimplementierung weg.

## Commit 2 — Gewichtete Stripe-Chunks

Hier liegt der eigentliche Beitrag für **asymmetrische** Laufwerke.

`mir_pread_striped` teilt heute **gleichmäßig** ([:2064](../c/colibri.c)):

```c
int64_t chunk = ((len + nsf-1)/nsf + 4095) & ~4095LL;
```

Bei Gen4 (~7 GB/s) und Gen3 (~3.5 GB/s) bedeutet das: beide bekommen 6.3 MB des
12.6-MB-Experten, der Gen4-Thread ist nach 0.9 ms fertig und **wartet 0.9 ms** auf
den Gen3-Thread. Effektiver Durchsatz 7 GB/s statt der möglichen 10.5 GB/s — der
komplette Gewinn des zweiten Laufwerks verpufft.

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

Erwartung: von 7 auf ~10 GB/s aggregiert, also **~0.26 s statt 0.37 s pro Token**
bei 2.6 GB Nachladung.

Die Einschränkung `len >= 4 MB` bleibt richtig — bei kleineren Reads dominiert die
Latenz und zwei Threads kosten mehr als sie bringen. Ein Expert-Record von
~12.6 MB liegt komfortabel darüber.

## Commit 3 — DRAM-lose SSDs

Beide Laufwerke haben **keinen eigenen DRAM** und nutzen HMB (Host Memory Buffer),
also geliehenen Host-RAM für die FTL-Mapping-Tabelle — typisch 64 MB.

Zwei Konsequenzen:

**Große sequentielle Reads sind der gute Fall.** Ein Expert-Record ist
`w1+w2+w3` plus Scales, zusammengefasst gelesen: ~12.6 MB am Stück. Das ist genau
das Zugriffsmuster, bei dem DRAM-lose Laufwerke kaum schlechter sind als solche
mit DRAM — die FTL muss wenige Mapping-Einträge nachladen. **Der bestehende
Coalescing-Pfad ist damit wichtiger als auf DRAM-Laufwerken**, nicht weniger.
Vor Änderungen an der Lesekoaleszenz also prüfen, dass sie erhalten bleibt.

**Tiefe Queues mit vielen kleinen Reads sind der schlechte Fall.** HMB-Laufwerke
brechen bei hoher Queue-Tiefe mit verstreuten Zugriffen deutlich ein, weil
FTL-Misses über PCIe nachgeladen werden. Der persistente Lookup-Pipeline-Pfad
(`COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER`, drei Worker,
[c/deepseek_v4.c:3005](../c/deepseek_v4.c)) sollte deshalb **messbar** eingestellt
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

Ergebnis dieses Commits ist primär eine **Messung mit dokumentierten Einstellungen**
im Tuning-Doc, nicht notwendig Code. Wenn die Messung eine feste Workerzahl oder
Queue-Tiefe nahelegt, wird daraus ein Default für dieses Profil.

## Alternative: `COLI_MODEL_DIRS` statt Spiegelung

Wenn der Platz knapp wird (167 GB × 2 = 334 GB von 1.5 TB — er wird nicht knapp),
wäre `COLI_MODEL_DIRS` die Alternative: Shards **aufteilen**, jeder liegt genau
einmal. Dann summiert sich die Kapazität, aber ein Expert ist nur auf *einem*
Laufwerk — kein Striping, kein Routing-Freiheitsgrad.

**Für dieses Setup ist Spiegelung klar besser**, weil der Platz reicht und beide
Freiheitsgrade (Routing und Striping) erhalten bleiben. Im Tuning-Doc als
bewusste Entscheidung festhalten, nicht als Zufall.

## Tests

- `c/tests/test_v4_mirror.c` — neue Make-Regel:
  - `expert_route` ist deterministisch und verteilt über viele `(layer, eid)`
    proportional zu den Cuts (χ²-artige Grobprüfung, keine Exaktheit).
  - Gewichtete Chunk-Grenzen: bei `wt = {7, 3.5}` und `len = 12.6 MB` sind die
    Grenzen 4K-aligned, lückenlos, summieren auf `len`, Verhältnis ≈ 2:1.
  - Entartete Fälle: eine Replik, `wt` gleich, `len` knapp über/unter 4 MB,
    `len` kleiner als `nsf × 4096`.
  - Partieller Mirror: Shard nur auf der Primärkopie → Fallback ohne Striping.
- `c/tests/test_v4_mirror_fallback.c` — ein Lesefehler auf der Replik fällt auf
  die Primärkopie zurück und loggt **einmal**, nicht pro Read.
- End-to-end mit zwei Verzeichnissen auf demselben Dateisystem (Kopie des
  Tiny-Fixtures): identische Tokens mit und ohne `COLI_MODEL_MIRROR`.

## Abnahme

- `MIRROR:`-Statistikzeile zeigt beide Laufwerke mit Bytes und Reads.
- Der Probe meldet ~7 und ~3.5 GB/s; die Cuts stehen ≈ 2:1.
- Gemessene Lesezeit pro Token sinkt gegenüber Einzellaufwerk um **≥40 %**
  (Erwartung ~7 → ~10 GB/s aggregiert).
- Gewichtete Stripes schlagen gleichmäßige messbar — beide Varianten gegeneinander
  messen, nicht nur die neue.
- Tokenfolge unverändert. Storage-Routing darf nie Semantik berühren.

## Risiken

- **Zwei Kopien = doppelte Schreiblast beim Anlegen** (167 GB je Laufwerk). Bei
  DRAM-losen QLC/TLC-Laufwerken ohne SLC-Cache-Reserve kann das lange dauern und
  den Cache füllen. Einmalig, aber im Tuning-Doc erwähnen.
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
  `COLI_V4_DIRECT` ([:112](../c/deepseek_v4.c)). Im Profil setzen.
- **Der gemeinsame Header ist ein Refactor an laufendem Code.** colibri.c ist
  ~8000 Zeilen und der GLM-Pfad ist nicht Ziel dieses Branches — wenn der Schnitt
  zu teuer wird, ist eine V4-lokale Kopie mit Verweis auf das Original die
  ehrlichere Zwischenlösung, solange Plan 11 sie erbt.
