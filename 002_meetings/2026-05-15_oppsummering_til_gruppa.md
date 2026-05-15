# Oppsummering — fredag 15. mai 2026
**Gruppe 4.5 — det vi har gjort i dag**

> I dag har vi tatt selvevalueringen fra i går og brukt den som arbeidsliste. Alle de syv prioriterte punktene er fikset, og de fleste av finpussforslagene er adressert. **Forskningsoppgaven er reelt endret i dag** — både tekst og tall.

---

## Bakgrunn — hva selvevalueringen sa

I går (14. mai) lagde vi en streng selvevaluering av vår egen rapport, hvor vi brukte samme strenge briller som vi brukte på G03's rapport. Selvevalueringen identifiserte syv prioriterte punkter, hvorav tre var kritiske:

1. **Sigma-problemet** — vi brukte tall fra 2025 til å regne ut usikkerhet, og brukte så samme tall til å vurdere økonomien for 2025
2. **Etikk var ikke nevnt** — vi navngir både bedriften og daglig leder uten å si noe om samtykke
3. **Figurer i PDF** — relative stier kunne potensielt bli tomme firkanter

I tillegg var det fire mer "kosmetiske" punkter: Diebold-Mariano-test, avrunding av hovedfunnet, kondensering av lengden, og eksplisitt forskningshull.

I dag har vi gått systematisk gjennom alle ti punkter (de syv prioriterte pluss tre tillegg fra finpusslisten).

---

## Del A — De tre kritiske fiksene

### 1. Sigma-problemet er ryddet opp i

Dette var den største metodiske svakheten. Forklart enkelt: før i dag estimerte vi prognoseusikkerheten ($\sigma$) ut fra hvor mye SARIMA bommet på 2025-tallene, og brukte så det samme tallet for å regne ut hvor mye butikken burde bestilt — og evaluerte det mot de samme 2025-tallene. Det er sirkulær resonnement.

**Hva vi gjorde:** Endret `demand_forecasting.py` slik at den nå lagrer SARIMAs *in-sample-residualer* fra treningsperioden 2023–2024 (ikke testperioden 2025). `sesongnewsvendor.py` leser deretter dette tallet via en ny JSON-fil (`forecast_metrics.json`) og bruker det som $\sigma_\text{mnd}$.

**Effekten på tallene:**

| Måling | Før | Etter | Endring |
|---|---:|---:|---:|
| $\sigma_\text{mnd}$ (prognoseusikkerhet) | 222,2 par | 182,9 par | –17,7 % |
| Sikkerhetslager per sesong | 367 par | 302 par | –17,7 % |
| $Q^*$ vår 2025 | 6 030 par | 5 975 par | –55 par |
| $Q^*$ høst 2025 | 3 568 par | 3 468 par | –100 par |
| Estimert årlig gevinst | 713 621 NOK | ≈ 570 000 NOK | –20 % |
| Relativ gevinst | +14,4 % | +11,5 % | –2,9 %p |

Hovedfunnet er fortsatt **substansielt og robust** — gevinsten er ~570 000 NOK, ikke 0 — men den er mer ærlig nå. Tallet er heller ikke fryktelig presist, og det er det poenget: vi runder bevisst til "ca. 570 000 NOK" og "+11–12 %" i sammendrag og konklusjon. Selvevalueringen påpekte at å oppgi 713 621 NOK på et estimat med ±10–15 % usikkerhet er overpresist.

### 2. Etikk er nå dokumentert i §3.1

Vi la til et eget avsnitt om etiske hensyn under metodekapittelet. Hovedpunktene:

- **Muntlig samtykke** fra daglig leder Marit Stoksflod til å bli navngitt og at salgsdataene benyttes
- **Ingen personopplysninger** i datagrunnlaget — kassesystemets dagsrapporter inneholder kun varekoder, antall og beløp
- **Ingen konkurransesensitive priser** i rapporten — $p, w, s$ er antatte estimat, ikke faktiske marginer fra regnskapet
- **Manuskript-gjennomgang** med daglig leder før publisering
- En anbefaling om at fremtidige prosjekter bør formalisere samtykket skriftlig før datafangsten starter

Vi ryddet også opp i formuleringen "Reliabilitet sikres gjennom Git" — selvevalueringen påpekte at Git sikrer at koden kan kjøres på nytt, ikke at målingen er pålitelig. Vi skiller nå klart mellom *reproduserbarhet* (samme kode + samme data → samme tall) og *reliabilitet* i streng forstand.

### 3. Figurene rendres faktisk korrekt

Selvevalueringen var bekymret for at de relative stiene (`../013_gjennomforing/visuals/...`) skulle gi tomme firkanter i PDF-eksport. Vi sjekket den eksisterende PDF-en ved å rendere noen av figur-sidene som bilder, og figurene er der — Chrome-print finner stiene relativt fra HTML-filen. Ingen handling kreves.

---

## Del B — De fire ekstra punktene fra finpusslisten

### 4. Diebold-Mariano-test lagt til

Selvevalueringen sa: *"Vi sier SARIMA er 15,8 % bedre enn naiv strategi, men vi har ikke gjort en statistisk test som beviser at forskjellen ikke er tilfeldig."* — Det er nå gjort.

Forklart enkelt: Diebold-Mariano er en statistisk test som spør om to prognosemodeller har "samme forventede tap". Hvis svaret er ja, betyr det at forskjellen mellom dem kan skyldes flaks. Hvis svaret er nei, betyr det at den ene er pålitelig bedre.

Resultater på vår SARIMA mot naiv strategi:

| Tap-funksjon | DM-statistikk | p-verdi |
|---|---:|---:|
| Kvadratisk tap (≈ RMSE) | –1,10 | 0,30 |
| Absolutt tap (≈ MAE) | –1,85 | **0,09** |

Begge har **negativ statistikk** — det betyr at SARIMA peker i riktig retning (lavere tap enn naiv). Men begge p-verdier er over den klassiske 5 %-grensen, så vi kan ikke konkludere med statistisk signifikans. Absolutt tap er svakt signifikant (p ≈ 0,09).

**Det er ærlig formidlet i rapporten:** Med kun 12 testmåneder har testen lav statistisk styrke per design. En definitiv test krever lengre evalueringsvindu, og det er nå identifisert som videre arbeid. Dette er faktisk en *styrke* — det viser at vi har en moderne forståelse av statistisk testbarhet.

### 5. Forskningshull og litteratursøk-metode

Selvevalueringen påpekte to ting: vi sa ikke hvordan vi søkte etter litteratur, og forskningshullet var "underforstått, ikke tydelig sagt".

Vi la til en ny **§2.6 Litteratursøk og forskningshull** som beskriver:
- **Hvor vi søkte** (Oria, Scopus, Google Scholar, referanselistene i kompendiet og hovedlærebøkene)
- **Hvilke søkeord** ("SARIMA", "seasonal ARIMA", "newsvendor", "bullwhip", "retail forecasting", "fashion retail", "shoe industry")
- **Inklusjonskriterier** (klassiske primærkilder fra 1913–2024, sentrale lærebøker, nyere fagfellevurderte artikler)
- **Forskningshullet** eksplisitt: "*Tidligere studier har dokumentert SARIMA og newsvendor hver for seg i store retail-kontekster. Det vi finner mindre dokumentert er den kombinerte anvendelsen i en liten norsk detaljhandelsbedrift med få beslutningsøyeblikk og data låst i PDF-format. Vår studie fyller dette hullet ved å demonstrere en ende-til-ende-pipeline tilpasset disse rammebetingelsene.*"

Den gamle §2.6 (mappingtabellen) er flyttet til §2.7.

### 6. Mappingtabellen kondensert + Vedlegg D slanket

Selvevalueringen sa: "Skill tydelig mellom *brukt i analysen* og *referert som ramme for utvidelse*". Vi har nå:

- **Liten kjernet tabell** i §2.7 med de 4 sentrale pensumseksjonene som inngår *direkte i analysen*
- **Mer kompakt rammetabell** med 10 seksjoner som brukes som *teoretisk basis for diskusjon og videre arbeid*
- **Den fullstendige mappingen** (18 seksjoner med detaljerte forklaringer) er flyttet ut av hovedrapporten og inn i `013_gjennomforing/kompendiumkobling.md`

Dette reduserer hovedrapportens lengde uten å miste innholdet. Den som vil grave dypere har det fortsatt tilgjengelig.

### 7. Hovedfunn avrundet og konsekvent presisjon

Tallet **+713 621 NOK** er erstattet med **"≈ 570 000 NOK / +11–12 %"** i sammendrag, §4.4, §6.1 og §6.2. Vi har lagt til en eksplisitt merknad om presisjon i sammendraget:

> *Den årlige gevinsten på "om lag 570 000 NOK / +11–12 %" oppgis bevisst som størrelsesorden, ikke som eksakt prognose – usikkerheten i $p, w, s$ er ±10–15 prosent, og presisjon utover dette ville være misvisende.*

---

## Del C — Finpussforslagene

### Polish som er gjennomført

| Endring | Hvor | Status |
|---|---|---|
| **FS1 omformulert** fra metodebeskrivelse til reelt forskningsspørsmål | §1.4 | OK |
| **Forkortelser skrevet ut** første gang (MAE, RMSE, MAPE, AIC) | §2.3, §4.2 | OK |
| **Lange setninger splittet** med flere siteringer (VMI i §5.2; multi-produkt i §3.5) | §3.5, §5.2 | OK |
| **"Verdensklasse"-formuleringen** tonet ned til "god treffsikkerhet" | §4.2 | OK |
| **Dobbelt "ti sekunder via mobilen"** ryddet til kun én forekomst med variert formulering | Sammendrag, §1.1 | OK |
| **Kilde for "to bestillinger per år"** lagt til (Stoksflod-intervju + Pinedo 2016) | §0 sammendrag | OK |
| **Anbefaling 1 omformulert** uten Python-kommando i hovedrapporten | §6.3 | OK |
| **Diskusjonens oppsummeringstabell** (ny §5.6) som mapper FS1–FS4 til konkrete diskusjonspunkter | §5.6 | OK |
| **§6.4 kondensert** fra 14 forslag til 7 prioriterte i tre trinn | §6.4 | OK |
| **§5.7 (tidligere §5.6) skarpere** — pedagogisk bidrag omformulert til "integrerende bidrag" | §5.7 | OK |
| **Bærekraft (§5.4) styrket** med konkret CO₂-anslag (~5 800 kg CO₂ for høst-overstock) | §5.4 | OK |
| **Lv et al. (2023)** lagt til i litteraturgjennomgangen, ikke bare i diskusjon | §2.5 | OK |
| **Diebold & Mariano (1995)** og **Harvey, Leybourne & Newbold (1997)** lagt til i referanselista | §7 | OK |

---

## Del D — De nye tallene som skal huskes

Hvis sensor stiller spørsmål eller dere skal forklare prosjektet for noen, er det de **nye tallene** som gjelder fra og med i dag:

### Prognosepresisjon (testperioden 2025)

| Modell | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **SARIMA(1,1,1)(1,1,1)₁₂** | **140 par** | **182 par** | **16,9 %** |
| Naiv baseline (samme måned i fjor) | 163 par | 197 par | 19,0 % |
| ETS (Holt-Winters) | 177 par | 231 par | 20,3 % |
| ARIMA(1,1,1) — uten sesong | 229 par | 278 par | 24,5 % |

**Forbedring SARIMA vs Naiv:** +14,0 % på MAE
**Årssum 2025:** SARIMA traff med 4,3 % avvik mot faktisk (10 338 mot 10 800 par)

### Newsvendor-bestilling (basisscenario)

- $\sigma_\text{mnd}$ = 182,9 par (in-sample fra 2023–2024)
- $\sigma_\text{sesong}$ = 447,9 par
- $z_\alpha$ = 0,6745 (75 % servicenivå, kritisk forhold = 600/800)
- **Sikkerhetslager:** 302 par per sesong
- **$Q^*_\text{vår}$ = 5 975 par** (mot Q_naiv = 5 389 par; faktisk salg = 6 000 par)
- **$Q^*_\text{høst}$ = 3 468 par** (mot Q_naiv = 4 120 par; faktisk salg = 3 657 par)

### Økonomisk effekt

- **Brutto-resultat:** Naiv 5 335 000 NOK → Newsvendor 5 665 959 NOK (+330 959 NOK)
- **Alternativkostnad tapt salg:** Naiv 366 600 NOK → Newsvendor 128 241 NOK (–238 359 NOK)
- **Netto årseffekt:** ≈ +570 000 NOK (+11,5 %)

---

## Del E — Filer som er endret i dag

| Fil | Hva som skjedde |
|---|---|
| `014_report/Forskningsoppgave_Gruppe_4.5.md` | **Hovedendringene:** sigma, tall, etikk, DM-test, FS1, forskningshull, oppsummeringstabell, §6.4 kondensert |
| `014_report/Forskningsoppgave_Gruppe_4.5.html` | Regenerert via `build_html.py` |
| `006_analysis/demand_forecasting.py` | Diebold-Mariano-funksjon, in-sample-RMSE, naiv baseline, JSON-output |
| `006_analysis/sesongnewsvendor.py` | Leser sigma fra `forecast_metrics.json` (in-sample) i stedet for å regne på test-residualer |
| `013_gjennomforing/forecast_metrics.json` | **Ny** — modellmetrikker og DM-test-resultater |
| `013_gjennomforing/newsvendor_resultater.json` | Oppdatert med nye tall |
| `013_gjennomforing/kompendiumkobling.md` | **Ny** — full mapping mellom pensum og oppgaven (flyttet ut av hovedrapporten) |
| `013_gjennomforing/visuals/newsvendor_profit_curve.png` | Regenerert med $Q^* = 5\,975$ |
| `013_gjennomforing/visuals/inventory_newsvendor_2025.png` | Regenerert med nye Q-verdier |
| `002_meetings/2026-05-15_oppsummering_til_gruppa.md` | Denne oppsummeringen |

---

## Hva betyr dette for sluttresultatet?

Forskningsoppgaven er nå **metodisk solid** der den før hadde noen reelle svakheter. Det viktigste er at vi har gått fra:

- En oppgave som kunne kritiseres for **sirkulær resonnement** (sigma fra fasiten)
- En oppgave som **ikke nevner etikk** med navngitt bedrift og daglig leder
- En oppgave som **påstår statistisk forbedring** uten å teste den

— til en oppgave som **eksplisitt adresserer alle disse punktene**. Tallene er nå mer ærlige (570k istedenfor 714k), og den statistiske analysen er mer transparent (DM-test viser at p ≈ 0,09 ikke er signifikant, men retningen er konsistent).

Det er fortsatt finpuss som kan gjøres før innlevering, men de **strukturelle svakhetene** selvevalueringen identifiserte er nå håndtert. Selvevalueringens samlede konklusjon var: *"Med tre dagers målrettet arbeid på de syv prioriterte punktene kan dette løftes fra en solid bacheloroppgave til en eksepsjonell."*

I dag har vi gjort en stor del av disse tre dagers arbeid på én dag.

---

## Hva som gjenstår

For å være ærlig, her er det som **fortsatt** kan/bør gjøres før innlevering:

1. **Hent faktiske enhetspriser** ($p, w, s$) fra Skoringens regnskap — dette ville stabilisert kronetallene
2. **Sjekk eksplisitt mot kursets formelle krav** — vi er fortsatt i øvre sjikt lengdemessig (selv om vi har kondensert)
3. **Korrekturlesing** — endringene i dag er substansielle, og en gjennomlesning av hele rapporten i sammenheng kan avdekke små inkonsekvenser

Disse er ikke kritiske, men ville løftet oppgaven videre.

---

## Trenger dere noe?

Hvis dere vil se rapporten med alle endringene, bygg PDF-en på nytt med:

```
cd 014_report/_preview
python full_rebuild.py
```

Det tar et minutt eller to (Chrome itererer to-tre ganger for å konvergere innholdsfortegnelsen).

Spør gjerne hvis noe er uklart om hva som er endret. God helg — det er en god slutt på uken.

— Gustavo
