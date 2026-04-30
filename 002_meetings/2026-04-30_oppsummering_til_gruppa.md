# Oppsummering — torsdag 30. april 2026
**Gruppe 4.5 — det vi har gjort med oppgaven i dag**

> Denne oppsummeringen handler **kun om det vi har endret i dag**. Hvis dere vil vite hva oppgaven generelt går ut på, se `002_meetings/2026-04-27_oppsummering_til_gruppa.md`. Den forklarer prosjektet steg for steg.

Dagens jobb handlet om **hvilke bøker og artikler vi viser til** i rapporten — altså referansene. Ingenting i selve resultatene eller metodene har endret seg. Det er språket og kildehenvisningene som er strammet opp.

---

## Bakgrunn — hvorfor vi gjorde dette

Foreleseren sa noe viktig i april: *Pensumkompendiet (de 33 kapitlene fra LOG650-faget) skal vi bruke som **arbeidsverktøy** når vi skriver — for å slå opp, finne struktur, og lære språket. Men når vi viser til en kilde i selve oppgaven, skal vi peke til de **etablerte fagbøkene** som hele verdens forskere bruker, ikke til vårt egne kompendium.*

Tenk dere at dere skriver om Norge i en bok. Da sier dere ikke *"se forelesningsnotatene mine"* — dere sier *"se Store norske leksikon"*. Det er samme prinsipp.

Vi måtte derfor finne de "ekte" bøkene bak hver metode vi bruker, og endre referansene fra *"Pensumets kapittel X §Y"* til *"Forfatter (Årstall)"*.

---

## Steg 1 — Vi fant fem "standardverk" som dekker alle metodene våre

Vi gikk gjennom oppgaven og listet hvilke metoder vi bruker. Så fant vi den boka eller artikkelen som er **selve standardreferansen** for hver metode. Dette er fem stykker:

| Hva metoden handler om | Kilden vi nå viser til |
|---|---|
| Planlegging og bestillingsstørrelse (når og hvor mye) | **Pinedo (2016)** — *Scheduling: Theory, algorithms, and systems* |
| Hvordan dele lite hylleplass mellom mange varer | **Hartmann & Briskorn (2010)** — om "ressursbegrenset planlegging" |
| Risikoanalyse med tilfeldige simuleringer (Monte Carlo) | **Vose (2008)** — *Risk analysis: A quantitative guide* |
| Hvordan lage feilmarginer ut fra dataene selv | **Efron & Tibshirani (1993)** — *An introduction to the bootstrap* |
| Hvordan kombinere ulike metoder smart | **Puchinger & Raidl (2005)** — om hybride løsningsmetoder |

**Hvorfor akkurat disse fem:** De dekker tematisk alt vi gjør i oppgaven. Det er bedre å lene seg på fem solide, anerkjente kilder enn å spre seg tynt over mange.

---

## Steg 2 — Vi byttet ut "Pensumets Ch10 §4" med "Pinedo (2016)" i selve teksten

I rapporten sto det tidligere setninger som dette:

> *"Pensumets Ch10 §4 (kvantumsrabatt-EOQ) gir det formelle rammeverket for å forstå..."*

Nå står det:

> *"Pinedo (2016) behandler slik sjelden, satsvis bestilling som et klassisk lot-sizing-problem..."*

**Hva endringen betyr:** Vi gir samme forklaring, men nå nevner vi forskeren som faktisk skrev fagboka. Kompendiet nevnes bare som *"jf. kompendiets Ch10 §4 som pedagogisk oppslag"* — altså som et sted hvor leseren kan slå opp og se metoden vist enklere.

Vi gjorde dette flere steder i kapittel 2 (om bullwhip-effekten, om EOQ, om RCPSP) og kapittel 4 (om hvordan vi sjekker at SARIMA-modellen vår er pålitelig).

---

## Steg 3 — Vi ryddet i referanselisten bak i rapporten

I kapittel 7 (Referanser) gjorde vi to ting:

1. **Lagt til de fem nye kildene** alfabetisk i listen — Efron & Tibshirani, Hartmann & Briskorn, Pinedo, Puchinger & Raidl, og Vose.
2. **Skrevet en kort forklaring** øverst i pensum-delen om at kompendiet brukes som arbeidsverktøy — ikke som primærkilde — og at sitatene går til de etablerte verkene.

---

## Steg 4 — Vi oppdaterte to tilhørende dokumenter

**Filen `003_referanser/AKADEMISK_LITTERATUR.md`** (hvor vi fører liste over all litteratur):
- La til en helt ny seksjon med de fem standardverkene øverst
- Flyttet kompendium-listen ned til en ny seksjon kalt *"Kompendiet som arbeidsredskap"*
- Skrev tydelig hvorfor: *"sitater i rapporten går til de etablerte verkene"*

**Filen `011_proposal/proposal.md`** (selve forskningsforslaget):
- Oppdaterte avsnittet om litteraturforankring slik at det stemmer overens med endringen
- Nå nevnes både kompendiet (som arbeidsredskap) og de fem primærkildene

---

## Hva betyr alt dette for sluttresultatet?

**Ingenting har endret seg i tallene, modellene eller konklusjonene.** SARIMA-prognosen er den samme. Newsvendor-bestillingen er den samme. Resultatene er de samme.

Det som har endret seg, er **hvor seriøst rapporten ser ut for sensor**. Når sensor blar i referanselisten, ser de nå navn som Pinedo, Vose og Efron — bøker som er pensum på operasjonsanalyse-master rundt om i verden. Det signaliserer at vi har faglig tyngde og vet hvor metodene kommer fra opprinnelig.

---

## Filer som er endret i dag

| Fil | Hva som skjedde |
|---|---|
| `014_report/Forskningsoppgave_Gruppe_4.5.md` | 7 steder i teksten + referanselisten |
| `003_referanser/AKADEMISK_LITTERATUR.md` | Ny seksjon med 5 primærkilder, omdøpt seksjonsnummer |
| `011_proposal/proposal.md` | Ett avsnitt oppdatert |

---

## Trenger dere noe?

Hvis dere vil se selve referanselisten, åpne PDF-en og bla til siste kapittel ("Referanser"). Der ser dere alle bøkene og artiklene oppgaven viser til — nå med de fem nye standardverkene.

Spør gjerne hvis noe er uklart.

— Gustavo
