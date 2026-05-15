# Oppsummering — torsdag 14. mai 2026
**Gruppe 4.5 — det vi har gjort i kveld**

> Denne oppsummeringen handler **kun om kveldens jobb med peer-review-leveransen**. Ingenting i selve forskningsoppgaven er endret i dag. Hvis dere vil vite hva oppgaven generelt går ut på, se `002_meetings/2026-04-27_oppsummering_til_gruppa.md`.

I kveld jobbet vi med **peer-review-delen av faget** — altså vurderingen vi skal levere på en annen gruppes rapport. Vi har også laget en streng *selvevaluering* av vår egen oppgave, fordi det ser ut til at vi ikke kommer til å få noen tilbakemelding tilbake fra den andre gruppa.

---

## Bakgrunn — hvorfor vi gjorde dette

Hver gruppe i LOG650 skal lese en annen gruppes rapport og levere en kort, skriftlig peer-review tilbake. Det er en obligatorisk del av faget. Vi har ventet lenge på å få oppgaven fra **Gruppe 03 (Bergens Beste)**, og endelig fikk vi tak i den i dag (jeg måtte selv hente den fra læreren fordi gruppa glemte å sende den).

Tenk dere at peer-review er som **å lese gjennom en venns kladd til eksamensbesvarelse og si hva som funker og hva som ikke funker**. Vi skal være ærlige, men også konstruktive — det vi peker på skal være noe de faktisk kan fikse.

I kveld var målet:
1. Lage selve peer-reviewen av G03's rapport (det som skal leveres)
2. Lage en intern selvevaluering av vår egen rapport — en slags "hva ville en streng sensor sagt"

---

## Del A — Peer-review av Gruppe 03's rapport

### Steg 1 — Vi leste G03's oppgave grundig

G03 sin rapport heter *"Optimalisering av bakkestøtte-ressurser ved Bergen Lufthavn Flesland"*. Den handler om hvordan flyplassen bruker gates og busser i rushtiden, og bruker simulering (Python + SimPy) til å teste forskjellige scenarioer.

Min ærlige vurdering etter å ha lest den: **det tekniske er greit, men det akademiske rammeverket rundt er svakt**. Et eksempel for å forklare:

- De har **ingen ordentlig litteraturgjennomgang** — bare et kort teoretisk rammeverk på halvannen side. Det betyr at de aldri viser leseren *hva andre forskere har gjort før dem* eller *hvorfor deres egen studie er nødvendig*.
- De har **ingen kildehenvisninger inne i selve teksten** — bare fem referanser bak i rapporten, hvor fire er lærebøker og null er fagartikler.
- **Valideringstabellen deres** har "Ikke beregnet" i nesten alle felt. De har skrevet at de skal validere, men ikke gjort det.
- **Ingen figurer i hele rapporten** — selv om de har masse tallmateriale som kunne vært visualisert.

Til sammenligning har vår egen rapport ekte litteraturgjennomgang, APA-7-siteringer overalt, fem figurer, og en validert pipeline. Forskjellen er stor.

### Steg 2 — Vi skrev første utkast (versjon 1)

Vi laget en peer-review som dekket alle de syv vurderingsområdene fra veiledningen:

| Område | Vår vurdering av G03 |
|---|---|
| Innledning | Sterk |
| Litteraturgjennomgang | **Svak** (mangler reell gjennomgang) |
| Metode | Sterk (men validering er ikke gjort) |
| Analyse og resultater | God (men ingen figurer) |
| Diskusjon | God (men kobles ikke til forskningsspørsmålene) |
| Konklusjon | God (svarer ikke punkt for punkt) |
| Skriveflyt og formelt | God (mangler APA 7) |

Første utkast var greit, men det var et problem: **språket var for akademisk**. Hvis G03 sliter med faguttrykk i utgangspunktet, ville en review full av "etterprøvbarhet", "operasjonaliserbare" og "artikulert forskningshull" vært vanskelig å lese. Det hjelper ingen.

### Steg 3 — Vi forenklet språket og laget en penere forside (versjon 2)

I versjon 2 byttet vi ut alt akademisk fagspråk med dagligspråk. *"Det som fungerer bra"* og *"Det dere bør forbedre"* erstattet *"Styrker"* og *"Forbedringspunkter"*. Setningene ble kortere. Punktlister ble brukt mye mer.

Vi laget også en **profesjonell forside** med blå gradient-banner, info-kort, og estetisk layout — slik at hele leveransen ser seriøs og gjennomarbeidet ut.

### Steg 4 — Vi utvidet til 4 sider med konkrete eksempler (versjon 3, endelig versjon)

Vurderingsmalen sier at peer-reviewen skal være på 2–4 sider. Vi utvidet til 4 sider med:

- **En oversiktstabell helt øverst** — leseren ser med ett blikk hva som er sterkt og svakt før de leser detaljene.
- **Forslag til søkeord** for litteratur (AGAP, DES airport operations, apron bus operations osv.) — slik at G03 ikke bare får kritikk, men også vet hvor de kan begynne å lete.
- **Konkrete figurforslag** med "hva den viser" og "hvorfor den hjelper" — slik at de kan lage figurene direkte fra simuleringsdataene.
- **"Spørsmål dere kan tenke over"** — fire spørsmål av typen sensor sannsynligvis vil stille på muntlig.

Den endelige versjonen heter **`Peer to Peer view av Gruppe 03.pdf`** og ligger i `Peer to peer/`-mappen.

---

## Del B — Selvevaluering av vår egen rapport

Siden vi sannsynligvis ikke får noen peer-review tilbake fra G03 (eller hvis vi gjør, blir den trolig på et annet faglig nivå), valgte vi å **lage en streng selvevaluering**. Tanken er: *hva ville en sensor som leter etter feil ha sagt?*

Vi brukte **akkurat samme strenge briller** som vi brukte på G03's rapport. Resultatet — selv om vår oppgave er solid — er at det fortsatt er reelle ting å forbedre.

### De viktigste tingene en sensor kan plukke på

| Problem | Hva det betyr | Hvor alvorlig |
|---|---|---|
| **Sigma-problemet** | Vi bruker tall fra 2025 til å regne ut usikkerhet, og bruker så samme tall til å vurdere økonomien for 2025. Litt som å lese fasiten først og så ta prøven. | Viktig — én linje kode å fikse |
| **Lite data** | 24 måneder treningsdata med en SARIMA-modell som har 6 parametere. På grensen. | Vi nevner det, men ikke nok |
| **Ingen statistisk test** | Vi sier SARIMA er 15,8 % bedre enn naiv strategi, men har ikke testet om forskjellen er ekte eller flaks. | Bør legge til Diebold-Mariano-test |
| **Etikk er ikke nevnt** | Vi navngir bedriften OG daglig leder ved fullt navn, men sier ingenting om samtykke. | Må fikses før innlevering |
| **Figurene vises kanskje ikke** | Vi peker til bilder med relative stier som kan bli tomme firkanter i PDF-en. | Test eksporten! |
| **Lengden** | Vi er på 60+ sider. Sjekk om det er innenfor formelle krav. | Ukjent — må sjekkes |

### Det som er sterkt med vår oppgave

For å være tydelig: dette er **ikke** en knusende kritikk. Selvevalueringen sier også at:

- Vi har en ekte litteraturgjennomgang med ordentlige primærkilder (Box, Hyndman, Petruzzi & Dada, Lee et al., Forrester, Pinedo, Christopher) som brukes aktivt i argumentasjonen.
- APA 7 er konsekvent gjennom hele teksten.
- Den matematiske formuleringen er presis.
- Residualdiagnostikken (ADF, Ljung-Box, Shapiro-Wilk) er metodisk forbilledlig.
- Konklusjonen svarer eksplisitt på FS1, FS2, FS3 og FS4 hver for seg — noe G03 *ikke* gjør.
- Implementeringsplanen i tre faser er sjelden velgjennomført på bachelornivå.

Selvevalueringens samlede konklusjon: *"Med tre dagers målrettet arbeid på de syv prioriterte punktene kan dette løftes fra en solid bacheloroppgave til en eksepsjonell."*

---

## Del C — Vi ryddet i mappestrukturen

I løpet av kvelden lagde vi flere versjoner av peer-reviewen mens vi jobbet. For å holde oversikten ryddig ordnet vi det slik til slutt:

```
Peer to peer/
├── Peer to Peer view av Gruppe 03.pdf      ← den endelige versjonen
├── Peer to Peer view av Gruppe 03.md       ← markdown-original
├── Hovedrapport_Flesland_G03.md            (G03 sin oppgave)
├── veiledning peer-review LOG650 (2).pdf   (malen fra fag-emnet)
├── peer to peer forslag/
│   ├── (alle gamle utkast — v1, v2, html-filer, konverteringsskript)
└── Selv review/
    ├── Selvevaluering_Gruppe_4.5.pdf       ← vår streng selvkritikk
    └── Selvevaluering_Gruppe_4.5.md
```

**Tanken bak:** Den endelige peer-reviewen ligger lett synlig i hovedmappa. Alt arbeidsmateriale (utkast, mellomfiler) er gjemt i `peer to peer forslag/` slik at de ikke roter til oversikten. Selvevalueringen vår ligger i sin egen mappe `Selv review/` slik at det er tydelig at det er en intern leveranse, ikke noe som leveres inn.

---

## Hva betyr dette for sluttresultatet?

**Selve forskningsoppgaven vår har ikke endret seg i kveld.** Modellene, tallene og konklusjonene er nøyaktig de samme.

Det vi har laget i kveld er:

1. **En ferdig peer-review** vi kan levere inn på G03's oppgave (oppfyller den obligatoriske leveransen).
2. **En intern arbeidsliste** med syv konkrete punkter vi bør fikse i vår egen oppgave før innlevering.

Den siste er minst like viktig som den første. Selvevalueringen er essensielt en *sjekkliste vi selv kan bruke* for å gjøre oppgaven enda bedre. Hvis vi får tid, bør vi prioritere disse syv punktene i denne rekkefølgen:

1. **Fiks sigma-problemet** (én linje kode)
2. **Skriv inn et avsnitt om etiske hensyn** (innhent samtykke fra Marit)
3. **Test PDF-eksporten** og bekreft at figurene faktisk vises
4. **Legg til en Diebold-Mariano-test** for SARIMA vs naiv
5. **Avrund hovedfunnet** til "ca. 700 000 NOK" eller "+14 %" i sammendrag og konklusjon
6. **Reduser lengden** ved å kondensere mappingtabellen og flytte vedlegg
7. **Skriv eksplisitt forskningshull** i kapittel 2.2

Punktene 1–3 er kritiske og bør prioriteres uansett.

---

## Filer som er endret eller laget i dag

| Fil | Hva som skjedde |
|---|---|
| `Peer to peer/Peer to Peer view av Gruppe 03.pdf` | **Ny — endelig peer-review (skal leveres)** |
| `Peer to peer/Peer to Peer view av Gruppe 03.md` | Markdown-original av samme |
| `Peer to peer/Selv review/Selvevaluering_Gruppe_4.5.pdf` | **Ny — vår strenge selvevaluering** |
| `Peer to peer/Selv review/Selvevaluering_Gruppe_4.5.md` | Markdown-original av samme |
| `Peer to peer/peer to peer forslag/` | Ny mappe med alle arbeidsutkast (v1, v2, html, konverteringsskript) |
| `002_meetings/2026-05-14_oppsummering_til_gruppa.md` | Denne oppsummeringen |

---

## Hvordan åpne dagens leveranse

For dere som vil se det:

1. **Peer-reviewen vi skal levere på G03:** `Peer to peer/Peer to Peer view av Gruppe 03.pdf`
2. **Vår egen selvevaluering (intern):** `Peer to peer/Selv review/Selvevaluering_Gruppe_4.5.pdf`
3. **Denne oppsummeringen som PDF:** `002_meetings/2026-05-14_oppsummering_til_gruppa.pdf` (kommer)

---

## Del D — Vi oppgraderte designet på alle møtereferater

Helt på slutten av kvelden gjorde vi en siste ting: **alle møtereferater fra hele prosjektet ble omformet til samme profesjonelle business-stil**.

### Hvorfor

Tidligere hadde møtereferatene et enkelt "ren markdown"-design — funksjonelt, men ikke spesielt presentabelt. Når dette er dokumenter som blir liggende i prosjektmappa og potensielt blir lest av sensor eller veileder, fortjener de en form som matcher kvaliteten på selve oppgaven.

### Hva vi gjorde

1. **Hentet ned offisielle logoer** for både Høgskolen i Molde og Skoringen.
2. **Bygget en ny PDF-mal** (`000_templates/md_to_pdf_business.py`) med:
    - Logo-letterhead øverst (HiMolde til venstre, Skoringen til høyre, atskilt av en rød/marineblå fargestripe)
    - Tittelblokk med eyebrow, hovedtittel, undertittel og en metadata-rad (Dokument-ID, Dato, Gruppe, Casebedrift)
    - Profesjonell fargepalett (marineblå #1e3a5f og HiMolde-rød #c8102e)
    - Tabeller med blå header og alternerende radstriper
    - Sidefot med automatisk "Side X av Y"
3. **Konverterte alle fem møtereferater** til samme design:

| Dato | Fil | Tittel |
|---|---|---|
| Søn 26. apr | `2026-04-26_sondag.pdf` | Møtereferat — Oppstart |
| Man 27. apr | `2026-04-27_oppsummering_til_gruppa.pdf` | Hva vi har gjort så langt |
| Ons 29. apr | `2026-04-29_oppsummering_til_gruppa.pdf` | Oppsummering — Rapportstruktur |
| Tor 30. apr | `2026-04-30_oppsummering_til_gruppa.pdf` | Oppsummering — Litteratur og kilder |
| Tor 14. mai | `2026-05-14_oppsummering_til_gruppa.pdf` | Møtereferat — Peer-review-leveranse |

### Hva det betyr fremover

Når vi lager nye møtereferater, er det bare å skrive en vanlig markdown-fil og kjøre:

```
python 000_templates/md_to_pdf_business.py <fil.md> --title "..." --date-str "..."
```

Da får vi automatisk samme design, samme logoer og samme oppsett — uten å måtte tenke på styling. Logoene ligger i `000_templates/logos/` og er embedded direkte i PDF-en (base64), så filene er fullstendig selvstendige.

---

## Trenger dere noe?

Hvis noe i selvevalueringen er uklart, eller dere vil diskutere prioriteringen før vi setter i gang med fiksene, ta gjerne en titt på PDF-en og kom med innspill. Det er ingen panikk — vi har fortsatt tid før innlevering, men det er greit å vite hva som ligger foran oss.

Spør gjerne hvis noe er uklart. God natt — vi tar kvelden her.

— Gustavo
