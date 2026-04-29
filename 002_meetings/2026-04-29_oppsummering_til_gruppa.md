# LOG650 — Hva vi har gjort så langt
**Gruppe 4.5 — forskningsoppgave om Skoringen Råholt**
**Dato:** 29. april 2026

---

## Hva oppgaven handler om — i én setning
Skoringen Råholt får bare bestille sko **to ganger i året** fra leverandøren. Vi finner ut hvor mye de bør bestille hver gang, slik at de hverken går tom eller sitter igjen med usolgte sko.

---

## Hovedidé (forklart enkelt)

- **Steg 1 — Spå framtiden:** Vi ser på hva som er solgt før, og bruker en matematisk modell (SARIMA) til å gjette hvor mye som blir solgt neste sesong.
- **Steg 2 — Bestem mengden:** Vi bruker en formel som heter "Newsvendor". Den finner balansen mellom å bestille for lite (tapt salg) og for mye (sko som blir liggende).
- **Steg 3 — Sjekk pengene:** Vi regner ut hvor mye butikken sparer eller tjener på å bruke vår metode i stedet for det de gjør i dag.

---

## Resultatene våre (på 2025-data)

| Hva | Tall |
|---|---|
| Treff på prognosen | **15,8 % bedre** enn dagens metode |
| Anbefalt vårbestilling 2025 | 6 030 par sko |
| Anbefalt høstbestilling 2025 | 3 568 par sko |
| Estimert ekstra fortjeneste pr år | **+713 621 NOK (+14,4 %)** |

---

## Sånn er prosjektet bygget opp på GitHub

Repoet ligger på `github.com/LOG650/G04-gruppe-4.5`. Mappestrukturen er lagt opp slik at navnet forteller hva som er inni:

| Mappe | Hva ligger der |
|---|---|
| `000_templates/` | Maler fra skolen (proposal, APA, prosjektstyringsplan) |
| `001_info/` | Kontaktinfo, bedriftsinfo om Skoringen, beslutningslogg |
| `002_meetings/` | Møtereferater + mal til nye referater |
| `003_referanser/` | Litteraturliste **og** hele LOG650-pensumet (33 seksjoner) |
| `004_data/` | PDF-rapporter fra butikken + vaskede CSV-filer |
| `006_analysis/` | Python-kode (PDF-leser, SARIMA, newsvendor) |
| `011_proposal/` | Prosjektforslaget |
| `012_plan/` | Prosjektplan, risiko, interessenter, Gantt |
| `013_gjennomforing/` | Brukermanual, valideringsrapport, alle figurer |
| `014_report/` | **Selve forskningsoppgaven** (`Forskningsoppgave_Gruppe_4.5.md`) |

---

## Pensumet er aktivt brukt — ikke bare nevnt

Vi har lastet ned hele LOG650-pensumet og lagt det i `003_referanser/Kompendium/`. Hver gang oppgaven bruker en metode, viser vi hvilken pensumseksjon det er hentet fra.

**Eksempler:**
- SARIMA-prognosen → pensum **Ch01 §3** (trend og sesong)
- Newsvendor-formelen → pensum **Ch05 §5**
- Bullwhip-effekten → pensum **Ch05 §3**
- EOQ-kritikken → pensum **Ch10 §4**

Til sammen brukes **22 av 33 pensumseksjoner aktivt** i oppgaven. En oversiktstabell ligger i §2.5 i hovedrapporten.

---

## Vil dere lese rapporten?

1. **Som vanlig tekst:** åpne `014_report/Forskningsoppgave_Gruppe_4.5.md` i en hvilken som helst tekstleser (eller GitHub viser den direkte).
2. **Som pen, ferdig-formatert side med matematikk og figurer:** dobbeltklikk `014_report/Forskningsoppgave_Gruppe_4.5.html` — den åpnes i nettleseren og ser ut som en ferdig oppgave.
3. **Som PDF, klar til å sendes/skrives ut:** `014_report/peer_to_peer_2026-04-29/Forskningsoppgave_Gruppe_4.5.pdf`.

---

## Trenger dere hjelp til å forstå noe?

Det viktigste å forstå er **hvorfor** vi gjør tingene, ikke nødvendigvis matematikken bak. Hovedlinja er:

1. Butikken bestiller blindt → de bommer ofte
2. Vi gir dem en bedre prognose → de bommer mindre
3. Mindre bom = mer penger og mindre sløsing

Spør meg gjerne hvis det er noe konkret som er uklart — vi gjør oss kjent med dette sammen før innleveringen.

— Gustavo
