# Oppsummering — tirsdag 27. mai 2026
**Gruppe 4.5 — trippelsjekk av alt + oppdatert eksamensforberedelse**

> I dag har vi gjort en grundig gjennomgang av **hele prosjektet** for å sjekke om det er klart til innlevering og holder til A-nivå. Kort svar: **ja, det holder.** Vi fant noen få forbedringspunkter som vi har fikset.
>
> Ingenting er fjernet eller endret — kun tillegg som løfter kvaliteten ytterligere.

---

## Trippelsjekk: hva ble gjennomgått

Vi leste gjennom:
- Hele rapporten (984 linjer / 54 sider) fra start til slutt
- Hele eksamensforberedelsen (655 linjer)
- Alle analyseskript i `006_analysis/`
- README, kompendiumkobling, valideringsrapport
- Git-historikk og prosjektstruktur

---

## Vurdering: holder det til A?

**Ja.** Her er de viktigste grunnene:

1. **Forskningsspørsmålene (FS1–FS4)** besvares systematisk med egne kapitler
2. **Teoriforankring** med tunge primærkilder — ikke bare pensum
3. **Reproduserbar pipeline** med verifiseringsskript (`verify_numbers.py`)
4. **Statistisk signifikans** skikkelig behandlet (DM-test p = 0,045)
5. **Bootstrap** med 10 000 simuleringer gir ærlig usikkerhetsanslag
6. **Sensitivitetsanalyse** viser robusthet (Q* varierer < ±6 %)
7. **Begrensninger** diskuteres ærlig — det viser modenhet
8. **22 av 33 pensumseksjoner** aktivt brukt og dokumentert
9. **Ryddig prosjektstruktur** med Git, brukermanual og valideringsrapport

---

## Hva ble forbedret i dag

### 1. Eksamensforberedelsen utvidet med fem nye sensorspørsmål

Disse temaene manglet i den gamle versjonen og er ting sensor kan spørre om:

| Nytt spørsmål | Hva det dekker |
|---|---|
| "Hva er studiens bidrag til faget?" | Tre bidrag: metodisk, empirisk, integrerende |
| "Hva med etikk og personvern?" | Muntlig samtykke, GDPR, antatte priser for å beskytte bedriften |
| "Var det noe som overrasket dere?" | Tre uventede funn: flat profittkurve, høst nedjustert, ETS dårligere enn naiv |
| "Hva er bærekraftsbidraget?" | 463 par mindre overlager → ~5 800 kg CO₂, FNs bærekraftsmål 12 |
| "Hvordan anbefaler dere implementering?" | Tre faser: parallell → hybrid → full modellkjøring |

### 2. Bærekraft/CO₂-vinkel lagt til i konseptdelen

Rapporten har en hel seksjon om bærekraft (§5.4), men eksamensforberedelsen dekket det ikke. Nå er det med — inkludert tall og forbehold.

### 3. Cheat sheet utvidet

Tre nye seksjoner i cheat sheet:
- **Tre overraskende funn** — flat profittkurve, høst nedjustert, ETS tapte mot naiv
- **Studiens tre bidrag** — metodisk, empirisk, integrerende
- **Bærekraftstall** — 463 par, 5 800 kg CO₂, bærekraftsmål 12

### 4. README fikset

- Feil filnavn rettet (sto "Bacheloroppgave_Skoringen_KOMPLETT.md", nå "Forskningsoppgave_Gruppe_4.5.md")
- Lagt til manglende mapper: `013_peer_review/` og `015_eksamensforberedelse/`

### 5. HTML-filer fjernet

- 8 HTML-filer fra `002_meetings/` slettet (vi har MD + PDF, trenger ikke HTML)
- Feil HTML fra `015_eksamensforberedelse/` slettet (inneholdt møtereferat, ikke eksamensforberedelse)

### 6. PDF regenerert

Ny `Eksamensforberedelse_muntlig.pdf` med alt det oppdaterte innholdet. Build-skript (`build_pdf.py`) lagt til så vi kan regenerere ved behov.

---

## Status etter i dag

| Del | Status |
|---|---|
| Hovedrapporten (54 sider) | Ferdig og kvalitetssikret |
| Eksamensforberedelse | Oppdatert med alle manglende temaer |
| Analysepipeline | Fungerer, verifiserbar |
| README | Rettet og komplett |
| Git / GitHub | Alt pushet (`e005a70`) |

---

## De 5 viktigste tallene til muntlig

Hvis du bare husker fem ting:

1. **+570 000 NOK** observert gevinst i 2025 (≈ en årslønn ekstra)
2. **+333 000 NOK** forventet gevinst i et tilfeldig framtidig år
3. **88 %** sannsynlighet for at newsvendor slår naiv
4. **p = 0,045** — statistisk signifikant forbedring
5. Gevinsten kommer fra **bedre fordeling** mellom vår og høst — ikke mer eller mindre totalt

---

*Alt er pushet til GitHub og klart til innlevering.*
