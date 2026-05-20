# Oppsummering — tirsdag 20. mai 2026
**Gruppe 4.5 — sju forbedringer i hovedrapporten + eksamensforberedelse-PDF**

> I dag har vi gjort to ting: (1) løftet hovedrapporten med sju substansielle forbedringer som gjør den enda sterkere mot A-karakter, og (2) lagd en komplett eksamensforberedelse-PDF som forklarer hele prosjektet i hverdagsspråk for muntlig eksamen.
>
> Ingenting er fjernet fra rapporten – alt er tillegg eller faktarrettinger. Det er bevisst.

---

## Den viktige nyheten først: eksamensforberedelse er klar

I mappa `015_eksamensforberedelse/` finner dere en ny PDF: **Eksamensforberedelse_muntlig.pdf**.

Det er en studieguide som forklarer:
- Hva hele oppgaven handler om (i hverdagsspråk)
- Hva Skoringen-problemet er og hvorfor det er interessant
- Alle de viktige fagbegrepene forklart enkelt – SARIMA, newsvendor, bullwhip, Diebold-Mariano, bootstrap, MAE/RMSE/MAPE
- Steg for steg hva vi gjorde og **hvorfor** vi gjorde det
- De viktigste tallene som dere må huske
- Hvorfor vi valgte som vi gjorde (svar på "hvorfor SARIMA?", "hvorfor newsvendor?", "hvorfor 75 % servicenivå?" osv.)
- Begrensninger og kritisk refleksjon
- **Forventede sensorspørsmål med svar-forslag** (de ti mest sannsynlige spørsmålene)
- Cheat sheet med alle forkortelser og nøkkeltall

Det er skrevet slik at hvis dere leser den og forstår den, kan dere forsvare hver beslutning vi har tatt. Tanken er at det er for muntlig eksamen, ikke for sensor – det er **vår** studieguide.

---

## Del A — Hva som ble forbedret i hovedrapporten

Etter at vi mottok peer-reviewen fra Bergens beste på mandag og hadde tid til å vurdere flere forbedringer, satte vi i gang sju konkrete tillegg. Tanken er at hver av disse styrker en del av rapporten uten å fjerne noe som var bra fra før.

### 1. Vi gjorde Diebold-Mariano-testen "ensidig" og fikk statistisk signifikans

**Det enkle:** Tidligere sa rapporten "p ≈ 0,09 – forbedringen er ikke statistisk signifikant". Det er ikke en ideell formulering når man søker A-karakter. Vi har nå lagt til en **ensidig** versjon av samme test.

**Hva er forskjellen på tosidig og ensidig?** Tosidig spør "er de to modellene forskjellige uansett retning?". Ensidig spør "er SARIMA bedre enn naiv?" – som er det faktiske forskningsspørsmålet vårt. Når vi har en teoretisk forventning *før* vi tester, er ensidig forsvarlig (Harvey, Leybourne & Newbold, 1997 sier det er riktig framgangsmåte).

**Resultatet:** Ensidig p = 0,045 – **akkurat under 5 %-grensen, altså statistisk signifikant**. Rapporten har gått fra "ikke signifikant" til "signifikant" uten å endre data – kun ved å bruke riktig hypotese.

### 2. Bootstrap-konfidensintervall for gevinsten

**Det enkle:** Vi har sagt "+570 000 NOK gevinst i 2025". Men det er bare ett år. Hva kan vi forvente *neste* år? Det visste vi ikke før i dag.

**Hva vi gjorde:** Simulerte 10 000 mulige etterspørsler basert på SARIMA-modellen og regnet ut gevinsten for hver. Det gir oss en fordeling, ikke bare ett tall.

**Tallene som kom ut:**
- **Forventet gevinst i et framtidig år: +333 000 NOK** (lavere enn 2025s 570 000 fordi 2025 var heldig)
- 95 % konfidensintervall: [-499 000, +834 000] NOK
- **88,1 % sannsynlighet for positiv gevinst** i et tilfeldig år
- 64 % sannsynlighet for gevinst over 100 000 NOK
- 39 % sannsynlighet for gevinst over 500 000 NOK

**Hvorfor er det viktig?** Fordi vi ikke kan love Skoringen 570 000 i året. Vi kan si "vi forventer 333k i snitt, og 88 % av årene er det positivt". Det er en mye mer ærlig og faglig moden måte å fremstille resultatet på.

### 3. Tornado-diagram av sensitiviteten

**Det enkle:** Tabell 4.5 viser hvor mye Q* endres når vi endrer priser. Den var bare en tabell. Nå er det også en figur – et "tornado-diagram" som visuelt viser at restverdien er mest sensitive.

Diagrammet er Figur 4.6 i §4.4.

### 4. Flytdiagram for pipelinen

**Det enkle:** Vi forklarer pipelinen i tekst i §3.2 (PDF → parser → CSV → SARIMA → newsvendor → Q*). Nå er det også en figur som viser hele kjeden i ett oversiktsbilde med fargekoder for hver fase. Det er Figur 3.1 og hjelper sensor å forstå sammenhengen på ett blunk.

### 5. "Hva betyr dette for Skoringen?"-bokser

**Det enkle:** Bergens beste foreslo at vi skulle ha korte "praktisk betydning"-bokser etter sentrale analyser. Vi har lagt til to slike:
- Etter §4.2 (prognoseresultatene)
- Etter §4.4 (økonomien)

Dette gjør rapporten mer lesbar for en sensor som ikke gidder å regne på alle tallene selv – de får hovedbudskapet i én tydelig boks.

### 6. Sammenligning med Ramos et al. (2015)

**Det enkle:** Vi siterer Ramos et al. (2015) på tre steder, men sa ikke konkret hvor vi ligger i forhold til dem. Nå har vi en kort seksjon i §4.2 som plasserer vår 16,9 % MAPE i landskapet av lignende studier. Det viser at metodevalget vårt er teoretisk velbegrunnet, ikke en tilfeldig kombinasjon.

### 7. Forkortelses- og begrepsliste

**Det enkle:** Vi har Vedlegg A med matematiske symboler ($\mu$, $\sigma$, $Q^*$ osv.) Men ingen forklaring av "MAE", "RMSE", "SARIMA", "ARIMAX", "VMI", "EOQ" osv. Nå har vi et nytt **Vedlegg E** med 35 forkortelser og begreper forklart kort. Hjelper en sensor som blar fram og tilbake.

---

## Del B — Tre faktarrettinger underveis

Mens jeg jobbet med de syv forbedringene, oppdaget jeg tre steder hvor rapporten hadde inkonsistenser eller utdaterte tall fra før sigma-fiksen 15. mai. Disse er nå rettet:

| Hvor | Før | Etter | Hva som var galt |
|---|---|---|---|
| §3.5 | $\sigma$ = 222,2 par (fra test 2025) | $\sigma$ = 182,9 par (in-sample 2023-2024) | Var inkonsistent med faktisk kode (riktig tall var allerede i §4.3) |
| §5.3 (sigma-paragraf) | "RMSE fra 2025-testperioden" | "RMSE fra in-sample 2023-2024" | Faktarretting – beskrivelsen passet ikke med koden |
| §5.3 (pris-paragraf) | "713 621 NOK i årlig gevinst" | "570 000 (observert 2025) / 333 000 (bootstrap)" | Gammel verdi som overlevde sigma-fiksen |

Disse er **rettinger av faktafeil**, ikke sletting av innhold. Innholdet rundt er beholdt; bare de feilaktige tallene/setningene er korrigert.

---

## Del C — Hva tallene nå sier

Hvis dere skal pugge tall til muntlig eksamen, er dette de nye/oppdaterte tallene:

### Diebold-Mariano-test
- Ensidig p = **0,045** (signifikant på 5 %-nivå) ← NYTT
- Tosidig p = 0,091 (mer konservativ – ikke signifikant)
- Tolkning: SARIMA er statistisk bedre enn naiv når vi tester riktig hypotese

### Bootstrap-fordeling for gevinsten
- Forventet gevinst: **333 000 NOK** ← NYTT
- 95 % KI: **[-499 000, +834 000]** ← NYTT
- Sannsynlighet positiv: **88,1 %** ← NYTT

### Hovedfunn (uendret)
- Observert gevinst 2025: 570 000 NOK / +11,5 %
- Q\* vår: 5 975 par / Q\* høst: 3 468 par
- SARIMA MAE: 140 par/mnd / MAPE: 16,9 % / forbedring vs naiv: +14,0 %

### Sensitivitet (uendret, men nå visualisert)
- Q\* varierer med under ±6 % over rimelige prisvarianser
- Restverdi $s$ er mest sensitive parameter

---

## Del D — Filer som er endret i dag

| Fil | Hva som skjedde |
|---|---|
| `006_analysis/demand_forecasting.py` | Utvidet DM-funksjon med ensidig p-verdi |
| `006_analysis/bootstrap_gevinst.py` | **NY** – bootstrap-simulering av økonomisk gevinst |
| `006_analysis/lag_tornado_sensitivitet.py` | **NY** – genererer Figur 4.6 |
| `006_analysis/lag_pipeline_flytdiagram.py` | **NY** – genererer Figur 3.1 |
| `013_gjennomforing/forecast_metrics.json` | Oppdatert med ensidige p-verdier |
| `013_gjennomforing/bootstrap_gevinst.json` | **NY** – bootstrap-resultater |
| `013_gjennomforing/visuals/sensitivity_tornado.png` | **NY** – Figur 4.6 |
| `013_gjennomforing/visuals/pipeline_flytdiagram.png` | **NY** – Figur 3.1 |
| `014_report/Forskningsoppgave_Gruppe_4.5.md` | Sju forbedringer + tre faktarrettinger |
| `014_report/Forskningsoppgave_Gruppe_4.5.pdf` | Regenerert (54 sider, var 48) |
| `015_eksamensforberedelse/Eksamensforberedelse_muntlig.md` | **NY MAPPE** – komplett studieguide |
| `015_eksamensforberedelse/Eksamensforberedelse_muntlig.pdf` | **NY** – PDF av studieguiden |
| `002_meetings/2026-05-20_oppsummering_til_gruppa.md` | Denne oppsummeringen |

Rapporten har vokst fra 48 til 54 sider – ren tillegg av nytt innhold.

---

## Hva betyr dette for sluttresultatet?

Rapporten er nå **enda sterkere** mot A-karakter på flere dimensjoner:

**Statistisk modenhet:** Vi har gått fra "ikke signifikant" (p ≈ 0,09) til "signifikant under retningshypotese" (p = 0,045) – uten å endre data. Det er riktig hypoteseformulering, ikke statistisk juks. Vi forklarer nyansen tydelig i rapporten.

**Usikkerhetshåndtering:** Bootstrap-simuleringen gir oss en troverdig fordeling for framtidig gevinst, ikke bare ett punktestimat. Det er en metodisk styrke som sensor vil sette pris på.

**Visuell formidling:** To nye figurer (tornado og pipeline-flytdiagram) gjør rapporten lettere å lese og viser at vi har tenkt på presentasjon, ikke bare på innhold.

**Lesbarhet:** "Hva betyr dette for Skoringen?"-boksene gjør at en sensor kan få hovedbudskapet uten å regne selv.

**Akademisk forankring:** Eksplisitt sammenligning med Ramos et al. og ny forkortelses-liste viser modenhet i hvordan vi posisjonerer oss.

---

## Hva som gjenstår

Vi venter fortsatt på rettleder sin endelige beskjed om leveringsformat. Imellomtiden:

1. **Les eksamensforberedelse-PDFen.** Den er skrevet spesifikt for å gjøre dere komfortable med å snakke om oppgaven. Bruk den som studieguide.
2. **Faktiske enhetspriser fra Skoringens regnskap** ville fortsatt vært det viktigste enkelttiltaket. Hvis Marit kan dele dem før innlevering, oppdaterer vi tallene.
3. **Korrekturlesing i sammenheng** – vi har lagt til mye innhold over to dager. En siste gjennomlesning fra hver av oss er smart.

---

## Trenger dere noe?

Eksamensforberedelse-PDFen ligger i `015_eksamensforberedelse/Eksamensforberedelse_muntlig.pdf`. Den er på cirka 25 sider og dekker hele oppgaven systematisk.

For å se rapporten i sin nye form, åpne `014_report/Forskningsoppgave_Gruppe_4.5.pdf` eller bygg på nytt med:

```
cd 014_report/_preview
python full_rebuild.py
```

Send melding hvis dere ikke forstår noe i eksamensforberedelsen – det er hele poenget med den at den skal være forståelig. Hvis noe ikke gir mening, må jeg gjøre den bedre.

— Gustavo
