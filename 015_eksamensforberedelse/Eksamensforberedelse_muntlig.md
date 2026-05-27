# Eksamensforberedelse — muntlig eksamen LOG650
**Gruppe 4.5 — Sesongbestilling under prognoseusikkerhet hos Skoringen Råholt**

> Dette dokumentet er en komplett gjennomgang av hele bacheloroppgaven vår – fra problemstilling til ferdig analyse – skrevet som studieguide for muntlig eksamen. Det forklarer **hva** vi har gjort og **hvorfor**, slik at vi kan svare på sensors spørsmål med innsikt og ikke bare ramse opp tall.
>
> Tanken er at hvis du leser dette og forstår det, kan du forsvare hver beslutning vi har tatt i prosjektet. Det er skrevet i hverdagsspråk – ingen forhåndskunnskap om statistikk eller optimering forventes.

---

## Innhold

- **Del 1.** Helhetsbildet — hva handler prosjektet om?
- **Del 2.** Bedriften og problemet — Skoringen Råholt
- **Del 3.** Verktøykassa — alle begreper forklart enkelt
- **Del 4.** Steg for steg — hva vi gjorde, og hvorfor
- **Del 5.** Resultatene — tallene som teller
- **Del 6.** Begrunnelser — hvorfor vi valgte som vi gjorde
- **Del 7.** Begrensninger og kritisk refleksjon
- **Del 8.** Forventede sensorspørsmål + svar-forslag
- **Del 9.** Cheat sheet — alt på én side

---

# Del 1 — Helhetsbildet

## Hva handler hele oppgaven om?

På én setning: **Vi har bygget et beslutningsstøttesystem som hjelper en lokal skobutikk å bestille riktig mengde sko to ganger i året.**

Skoringen Råholt på Eidsvoll må bestemme to ganger i året hvor mange par sko de skal kjøpe inn. De har gjort dette etter erfaring og magefølelse i mange år. Vi har testet om vi kan gjøre det bedre med matematikk og data.

Svaret er: **ja, men ikke fordi de skal kjøpe mer eller mindre totalt – de skal fordele bedre mellom vår og høst.**

## Hvorfor er dette interessant?

Tre grunner gjør dette til et godt forskningsprosjekt:

1. **Det er et reelt problem hos en reell bedrift.** Skoringen Råholt har gitt oss tilgang til tre års salgsdata. Vi jobber ikke med kunstige eksempler.
2. **Det er et matematisk velformulert problem.** Skobransjen har en spesiell egenskap: man bestiller bare to ganger i året. Det gjør problemet til en klassisk *newsvendor-situasjon* (avisselger-problemet) – noe det finnes mye teori om.
3. **Det er en metodisk utfordring.** Skoringens data var "låst" i over 1 000 PDF-rapporter. En del av jobben var å bygge en automatisert pipeline som låser opp dataene. Det er et reelt problem mange små bedrifter har.

## De fire fasene i prosjektet

Bacheloroppgaven har gått gjennom fire faser:

| Fase | Hva vi gjorde | Hvor det er dokumentert |
|---|---|---|
| **Fase 1 – Proposal** | Definerte problemstilling og forskningsspørsmål, fikk dem godkjent | `011_proposal/` |
| **Fase 2 – Prosjektplan** | Risikoanalyse, interessentkartlegging, framdriftsplan, Gantt | `012_plan/` |
| **Fase 3 – Gjennomføring** | All faktisk analyse: data, modeller, beregninger, peer review | `013_gjennomforing/` + `013_peer_review/` |
| **Fase 4 – Rapport** | Skriving av hovedrapporten (54 sider) | `014_report/` |

Vi har også lagt til en mappe `015_eksamensforberedelse/` (denne her) som ikke er en del av leveransen, men gruppens egen studieguide.

## Hva endret seg underveis?

Vi gikk inn i prosjektet med en idé om at "Just-In-Time" (JIT) – hyppige, små bestillinger – kunne være løsningen. Etter intervju med daglig leder Marit Stoksflod oppdaget vi at bestillingsfrekvensen ikke er et valg butikken har: leverandørene tilbyr bare to bestillinger i året. Da reformulerte vi problemet til "bestille riktig mengde ved hver av de to bestillingene", som er *newsvendor*-problemet. Dette er en viktig læring i seg selv: man må forstå rammebetingelsene før man velger metode.

---

# Del 2 — Bedriften og problemet

## Hvem er Skoringen Råholt?

- **Lokalt**: Eidsvoll kommune, like nord for Oslo.
- **Konkurrenter**: Jessheim Storsenter (et stort kjøpesenter noen kilometer unna) og e-handel.
- **Selger**: Sko (alle typer – herre, dame, barn, sport, sandaler, støvler).
- **Lager**: Cirka 3 000 par i butikkens egen kapasitet. Når det blir for trangt, leier de eksternt lager.
- **Daglig leder**: Marit Stoksflod – vår kontakt.
- **Datamengde**: 36 månedsobservasjoner (2023–2025), totalt 29 619 par solgt over tre år.

## Skobransjens spesielle utfordringer

Vi løfter tre faktorer i rapporten som gjør sesongbestilling i sko *vanskeligere* enn i mange andre bransjer:

1. **Størrelsesfordeling**. En skomodell finnes i 10–15 størrelser. Hvis en kunde leter etter størrelse 39 og du bare har 42 igjen, er det null tilgjengelighet uavhengig av totallager.
2. **Sesongstruktur**. Norge har fire distinkte årstider, og halvparten av sortimentet må byttes ut to ganger i året (vår: sandaler/joggesko; høst: boots/vinterstøvler).
3. **Bestillingsregime**. Leverandørene gir kun to bestillingsvinduer per år. Dette er en **bransjebetingelse**, ikke en butikkbeslutning. Den henger sammen med lange ledetider fra produksjon i Asia/Sør-Europa og volumrabatter (Pinedo, 2016).

## Hvorfor er sesongbestilling så vanskelig?

Med kun **to beslutningsøyeblikk per år** (februar og august) blir hver bestilling et stort økonomisk veddemål med 6 måneders horisont:

- **For mye**: Kapital bundet i overlager, må selges med 50–70 % rabatt ved sesongslutt.
- **For lite**: Tomme hyller akkurat når trafikken er høyest, tapte kunder, varig svekket rykte.

Det er ikke noe mellomalternativ – du kan ikke "kjøpe litt til" hvis salget tar av.

## Problemstillingen og forskningsspørsmålene

**Hovedproblemstilling:**

> *"Hvordan kan SARIMA-baserte etterspørselsprognoser kombinert med newsvendor-logikk forbedre Skoringen Råholts sesongbestillinger sammenlignet med dagens praksis basert på fjorårssalg?"*

Vi brutte det ned i fire forskningsspørsmål (FS):

- **FS1 (Datafangst)**: Kan vi låse opp dataene fra PDF-rapportene og bygge et reproduserbart datasett?
- **FS2 (Statistisk)**: Hvilken prognosemodell – naiv, ETS, ARIMA, SARIMA – er best?
- **FS3 (Beslutningsteoretisk)**: Hva er optimal sesongbestilling Q\* etter newsvendor-modellen?
- **FS4 (Økonomisk)**: Hva er den årlige effekten av å bytte fra "samme som i fjor" til SARIMA-newsvendor?

Disse fire spørsmålene strukturerer hele rapporten – kapitlene 4.1, 4.2, 4.3 og 4.4 svarer på hver sin.

---

# Del 3 — Verktøykassa: alle begreper forklart enkelt

Dette er den viktigste delen for muntlig eksamen. Hvis du forstår denne, kan du svare på de fleste spørsmål om "hvorfor brukte dere X?".

## EOQ — den klassiske modellen vi *ikke* kunne bruke

**EOQ (Economic Order Quantity)** er en formel fra 1913 av Harris. Den sier at den optimale bestillingsmengden er:

$$ Q^* = \sqrt{\frac{2DS}{H}} $$

der D er årlig etterspørsel, S er bestillingskostnad, H er lagerholdskostnad. Du har sannsynligvis sett denne i pensum.

**Problem for vår case**: EOQ forutsetter at:
- Etterspørselen er konstant (vi har sesong)
- Du kan bestille når du vil (vi kan bare to ganger i året)
- Det er ingen restverdi for usolgte enheter (sesongsko har lavere verdi etter sesongen)

Alle disse forutsetningene er brutt hos Skoringen. EOQ gir feil svar i vårt case.

**Hvis sensor spør "hvorfor ikke EOQ?":** "Fordi EOQ forutsetter konstant etterspørsel og fleksibel bestillingsfrekvens. Vi har verken det. Vi har stokastisk sesongetterspørsel og kun to bestillingsvinduer per år, som er klassisk newsvendor-situasjon."

## Newsvendor-modellen — kjernen i oppgaven

**Tankeeksperiment**: Tenk deg en avisselger som hver morgen skal bestemme hvor mange aviser han kjøper inn for å selge i løpet av dagen.
- Bestiller han for få → mister salg
- Bestiller han for mange → sitter igjen med aviser som er verdiløse etter dagens slutt

Newsvendor-modellen sier: bestill det antallet $Q^*$ som balanserer disse to kostnadene. Det er en **engangs**-beslutning under usikker etterspørsel.

**Formelen:**

$$ Q^* = \mu + z_\alpha \cdot \sigma $$

der:
- $\mu$ = forventet etterspørsel (vi får den fra SARIMA-prognosen)
- $\sigma$ = standardavvik for etterspørselen (også fra SARIMA)
- $z_\alpha$ = en faktor som bestemmer servicenivået (mer om dette under)

**Kritisk forhold** (CR) er det optimale servicenivået:

$$ \text{CR} = \frac{p - w}{p - s} = \frac{C_u}{C_u + C_o} $$

der:
- $C_u = p - w$ er underbestillingskostnaden (tapt margin per ikke-solgt par)
- $C_o = w - s$ er overbestillingskostnaden (tap per usolgt par)

For oss: $p = 1200$, $w = 600$, $s = 400$, så CR = 600/800 = **0,75** = 75 % servicenivå. Det betyr at vi designer bestillingen slik at vi dekker etterspørselen i 75 % av tilfellene.

**Hvis sensor spør "hvorfor 75 %?":** "Fordi det er det matematisk optimale gitt våre antatte priser. Vi bruker formelen CR = (p−w)/(p−s) som veier kostnaden ved understocking mot kostnaden ved overstocking. Butikkeier kan velge å gå høyere (f.eks. 90 %) for merkevarebygging, men 75 % er det formelle optimum."

## SARIMA — vår prognosemodell

**Hva er SARIMA?**

SARIMA er en utvidelse av ARIMA. La oss bygge opp begrepet:

- **AR** (AutoRegressive): "Det som skjer i dag, henger sammen med det som skjedde i går." Vi prediker $Y_t$ fra $Y_{t-1}, Y_{t-2}, ...$
- **MA** (Moving Average): "Det som skjer i dag, henger sammen med feilene jeg gjorde i gårsdagens prognose."
- **I** (Integrated/Differensiering): Mange tidsserier vokser over tid. Hvis du analyserer veksten (forskjellen fra måned til måned) i stedet for nivået, blir serien mer stabil.

ARIMA kombinerer disse tre: $(p, d, q)$ der p er AR-orden, d er differensieringsorden, q er MA-orden.

**Hvor kommer "S" (Seasonal) inn?**

Sko har sesongmønster: høyt i april/mai og september, lavt i januar. SARIMA legger til en sesongkomponent som "ser tilbake 12 måneder" i tillegg til "ser tilbake 1 måned". Notasjonen blir:

$$ \text{SARIMA}(p,d,q) \times (P,D,Q)_s $$

der (P, D, Q) er sesongversjonene og s=12 for månedsdata.

**Vår modell**: SARIMA(1,1,1)(1,1,1)$_{12}$ – valgt av automatisk grid-søk basert på lavest AIC.

**Hvis sensor spør "forklar SARIMA på 30 sekunder":** "SARIMA er en tidsseriemodell som kombinerer to ting: korttidsavhengighet (ARIMA-delen, hva som skjedde forrige måned påvirker denne) og sesongavhengighet (sesongdelen, hva som skjedde samme måned i fjor påvirker denne). For sko, der både trend og sesong er viktig, er det den naturlige modellen."

## Bullwhip-effekten — relevant for hvordan vi snakker med leverandøren

**Tankeeksperiment**: Forestill deg at det er et stort fotballarrangement på TV. Folk kjøper litt mer pizza enn vanlig. Pizzaen-butikken legger inn en ekstra ordre til grossisten. Grossisten ser en plutselig økning og bestiller ekstra fra leverandøren. Leverandøren ser to nivåer av "ekstra" og bestiller ekstra fra produsenten...

Resultatet: produsenten ser **mye større svingninger** enn det som faktisk skjer ute i butikken. Det er bullwhip-effekten (Lee, Padmanabhan & Whang, 1997).

**Hvorfor er det relevant for Skoringen?** Når de bestiller to gigantbestillinger i året, ser leverandøren et meget volatilt signal. Hvis Skoringen delte sin **rullerende månedsprognose** med leverandøren, kunne leverandøren planlegge bedre. Dette er anbefaling 3 i §6.3.

## Diebold-Mariano-testen — er forskjellen ekte eller tilfeldighet?

**Problem**: Vi har målt at SARIMA har MAE 140 par og naiv har MAE 163 par. Det er en forbedring på 14 prosent. Men er det fordi SARIMA *virkelig* er bedre, eller har vi bare vært heldige med akkurat 2025?

**DM-testen** sammenligner forventet tap mellom to prognosemodeller. Den gir oss en p-verdi:
- p < 0,05 → vi kan være rimelig sikre på at forskjellen er ekte
- p > 0,05 → forskjellen kan skyldes tilfeldighet

**Ensidig vs tosidig test**:
- **Tosidig**: "Er de forskjellige?" (svarer på begge retninger – kunne være SARIMA bedre eller verre)
- **Ensidig**: "Er SARIMA bedre enn naiv?" (svarer kun på vår faktiske hypotese)

Vi har en teoretisk forventning *før* vi testet om at SARIMA skal være bedre (det finnes 50 års litteratur som sier det). Da er ensidig test forsvarlig (Harvey, Leybourne & Newbold, 1997).

**Våre tall**:
- Tosidig p ≈ 0,09 (akkurat over 5 % – ikke signifikant)
- Ensidig p ≈ 0,045 (akkurat under 5 % – **signifikant**)

**Hvis sensor spør "p = 0,09 og 0,045 – hva betyr det egentlig?":** "Det betyr at sannsynligheten for å se en så stor forbedring i SARIMAs favør hvis modellene egentlig var like presise, er 4,5 %. Det er lavt nok til at vi forkaster nullhypotesen om at modellene er like. Men vi anerkjenner at testen er på grensa, og at lengre testperiode hadde gitt enda sterkere konklusjon."

## Bootstrap — simuler tusen alternative virkeligheter

**Tankeeksperiment**: Vi har målt at gevinsten i 2025 var 570 000 NOK. Men 2025 er bare ett år. Hva kan vi forvente i et tilfeldig framtidig år?

**Bootstrap-prinsippet**: Vi simulerer 10 000 mulige etterspørsler basert på SARIMA-modellen (mu og sigma), og regner ut gevinsten for hver. Da får vi en *fordeling* av gevinster, ikke bare ett tall.

**Våre tall**:
- Forventet gevinst: 333 000 NOK (gjennomsnitt over 10 000 simuleringer)
- 95 % konfidensintervall: [-499 000, +834 000] NOK
- 88,1 % sannsynlighet for positiv gevinst

**Hvorfor er forventet (333k) lavere enn observert (570k)?** Fordi 2025 var en heldig realisasjon for newsvendor-strategien. Det er en viktig nyansering: vi kan *ikke* love butikken 570 000 i året framover. Vi kan love at i gjennomsnitt vil de tjene 333 000 mer, men noen år blir det mindre og noen år mer.

**Hvis sensor spør "kan dere garantere gevinsten på 570 000 NOK?":** "Nei. 570 000 var det observerte i 2025. Vår bootstrap-simulering viser at over framtidige år forventer vi 333 000 NOK i snitt, med 95 % konfidensintervall som inkluderer både negative og positive utfall. Modellen er forventet bedre – med 88 % sannsynlighet – men ikke garantert hvert år."

## MAE, RMSE og MAPE — tre måter å måle prognosefeil

Disse tre handler om samme grunnting: hvor mye bommer prognosen?

| Mål | Formel | Hva det måler | Når brukes det |
|---|---|---|---|
| **MAE** | gjennomsnittlig $|Y - \hat{Y}|$ | Gjennomsnittlig avvik i samme enhet som data (par) | Lett å forklare |
| **RMSE** | $\sqrt{\text{gj.snitt}(Y - \hat{Y})^2}$ | Straffer store avvik hardere | Bruker vi som proxy for $\sigma$ |
| **MAPE** | gj.snitt $|Y - \hat{Y}|/Y \cdot 100\%$ | Prosent-avvik | Lett å sammenligne på tvers av problem |

**Våre tall for SARIMA**:
- MAE = 140 par/mnd (vi bommer i snitt 140 par per måned)
- RMSE = 182 par
- MAPE = 16,9 % (vi bommer i snitt 16,9 % per måned)

**Forskjellen RMSE og MAE**: hvis du bommer mye ett enkelt år, øker det RMSE mer enn MAE. RMSE er strengere overfor store enkeltavvik.

## AIC — hvordan vi valgte modell

**AIC (Akaike Information Criterion)** er et tall som balanserer to ting:
- Hvor godt modellen passer dataene (lavere er bedre)
- Hvor kompleks modellen er (færre parametre er bedre)

Vi prøvde mange kombinasjoner av (p,d,q,P,D,Q) og valgte den med lavest AIC. Det er en standard metode for å unngå *overfitting* (modellen er så kompleks at den memoriserer dataene i stedet for å lære fra dem).

## ADF, Ljung-Box og Shapiro-Wilk — diagnose-testene

Disse tre brukte vi for å sjekke at SARIMA-modellen vår er teknisk god:

- **ADF (Augmented Dickey-Fuller)**: "Er serien stasjonær?" – dvs. har den konstant forventning og varians over tid. Vår serie ble stasjonær etter d=1, D=1 differensiering (p < 0,01).
- **Ljung-Box**: "Er det noe gjenværende mønster i feilene?" – Hvis ja, har vi en dårlig modell. Hos oss p > 0,05 → modellen er ok.
- **Shapiro-Wilk**: "Er feilene normalfordelt?" – Vi trenger det for å bruke newsvendor med normalfordeling-antakelse. Hos oss er det oppfylt.

## Bærekraft og CO₂ — hvorfor er det relevant?

Bedre prognoser har en direkte bærekraftseffekt: redusert overstock betyr færre sko som må selges som ukurans eller kastes.

**Vårt grove anslag (§5.4):**
- Newsvendor reduserer overlager med 463 par i høstsesongen 2025
- Industristandard: ~12,5 kg CO₂-ekvivalenter per par sko (Quantis, 2018)
- Spart fotavtrykk: ~5 800 kg CO₂ per år (omtrent som én flyreise Oslo–New York tur/retur)

**Viktig forbehold:** Reell sparing forutsetter at leverandøren faktisk *produserer* mindre når Skoringen bestiller mindre. Hvis leverandøren bare selger overskuddet til en annen butikk, flyttes overlageret oppover i kjeden uten miljøgevinst.

**Kobling til FNs bærekraftsmål 12 – Ansvarlig forbruk og produksjon.**

**Hvis sensor spør "hva er bærekraftsbidraget?":** "Bedre prognose reduserer overstock, som betyr færre sko produsert uten å nå en kunde. Vårt anslag er 5 800 kg CO₂ spart per år fra denne ene butikken. Det er beskjedent i absolutt størrelse, men illustrerer prinsippet om at logistikkanalyse kan ha både økonomiske og miljømessige gevinster samtidig."

## Servicenivå — hvorfor 75 % og ikke 95 %?

**75 %** er det matematisk optimale gitt våre priser (CR = 600/800). Det betyr: vi designer bestillingen slik at i 3 av 4 sesonger har vi nok lager. I 1 av 4 sesonger blir vi tomme.

**Hvorfor ikke høyere?** Fordi å bestille mer enn nødvendig er dyrt. Hver ekstra par koster 600 NOK å kjøpe inn, og hvis det ikke selges må det rabattsalges for 400 NOK – netto tap 200 NOK per usolgt par.

**Hvorfor ikke lavere?** Fordi hvert tapt salg koster 600 NOK (margin = p − w = 1200 − 600).

Modellen veier disse to mot hverandre og finner 75 % som det matematisk korrekte balansepunktet.

**Butikkeier kan velge høyere** (f.eks. 90 %) hvis hen vil bygge merkevarekapital som "stedet hvor du finner det du trenger". Det er en strategisk, ikke en matematisk, beslutning. Vi anbefaler at hen tar den bevisst.

---

# Del 4 — Steg for steg: hva vi gjorde, og hvorfor

## Steg 1: Datafangst (PDF → CSV)

**Hva vi gjorde:** Skoringen ga oss cirka 1 100 daglige Z-rapporter (dagsoppgjør) i PDF-format. Hver rapport hadde linjer som:

```
123456 (varekode)  Antall: 2  Pris: 599  Total: 1198
```

PDF er et "tegne-format" – det vet ikke hva "varekode" eller "pris" er, det vet bare hvor på siden tegnene er. Vi bygde en Python-pipeline med `pdfplumber` som:

1. Åpner hver PDF
2. Identifiserer hvor på siden ulike felter er ($x$-koordinater)
3. Sjekker hver linje mot regex `^\d{6}` (linjen må starte med et 6-sifret varenummer)
4. Aggregerer til dagsdata, deretter månedsdata
5. Sjekker kontrollsum mot "Total salg"-feltet i bunnen

**Hvorfor er dette viktig?** Det er den reelle praktiske flaskehalsen for små bedrifter. De har data – men dataene er "låst". Vår pipeline er reproduserbar og kan brukes av andre butikker.

**Hvis sensor spør "var det vanskelig?":** "Ja og nei. Selve regex'en er triviell. Det vanskelige var å finne riktige $x$-koordinater for hver kolonne, og å håndtere kantsituasjoner som returer (negative beløp), tomlinjer, og kontrollere kontrollsum. Den ferdige pipelinen er på ca. 200 linjer Python og kan kjøres på alle Skoringens framtidige rapporter."

## Steg 2: Datavasking

**Hva vi gjorde:**
1. **Returer**: Inkludert som negative salgsbeløp i nettoaggregeringen. Det gir oss "netto-etterspørsel" som er det newsvendor bruker.
2. **Uteliggere**: Identifisert med Z-score > 3 (mer enn 3 standardavvik fra månedsgjennomsnittet). Manuelt inspisert og korrigert hvis det var registreringsfeil.
3. **Frekvenskonvertering**: Daglige data aggregert til måned via `pandas.resample('MS')`. Dag-til-dag-støy er ikke relevant for sesongbestilling.

**Resultat**: 36 månedsobservasjoner (jan 2023 – des 2025). Det er "knapt nok" data for SARIMA – men det er det vi har, og det er det vi må jobbe med.

**Hvorfor månedsdata og ikke ukentlig?** To grunner: (1) det matcher Skoringens egne månedsrapporter, og (2) det gir akkurat nok observasjoner ($n = 36$) til at SARIMA kan estimeres meningsfullt mens vi reduserer støy.

## Steg 3: Beskrivende analyse

**Hva vi gjorde:** Plottet salget over tid, så på sesongmønsteret per måned, beregnet deskriptiv statistikk.

**Hovedfunn:**
- Total vekst fra 9 041 (2023) → 9 778 (2024) → 10 800 (2025) – ca. 9 % i året
- To klare topper: april/mai (vårtopp) og august/september (høsttopp)
- Forhold høyeste/laveste måned = 2,2x (sterk sesong)
- Sesongmønsteret er **stabilt** på tvers av år (viktig for SARIMA-egnethet)

**Hvorfor er stabiliteten viktig?** Fordi SARIMA forutsetter at sesongkomponenten gjentar seg konsistent. Hvis sesongmønsteret endret seg fundamentalt fra år til år, hadde SARIMA ikke vært en god modell.

## Steg 4: Modellvalg og estimering

**Hva vi gjorde:** Vi sammenlignet fire modeller:

| Modell | MAE | RMSE | MAPE | Forbedring vs naiv |
|---|---:|---:|---:|---:|
| **SARIMA(1,1,1)(1,1,1)₁₂** | **140** | **182** | **16,9 %** | **+14,0 %** |
| Naiv (samme måned i fjor) | 163 | 197 | 19,0 % | – |
| ETS (Holt-Winters) | 177 | 231 | 20,3 % | –8,8 % |
| ARIMA(1,1,1) uten sesong | 229 | 278 | 24,5 % | –40,5 % |

**Hovedinnsikt fra denne tabellen:**
- ARIMA uten sesong er den dårligste – det beviser at sesongleddet bærer informasjon.
- ETS er overraskende dårligere enn naiv – fordi ETS i sin additive form ikke håndterer vekst i sesongtoppene.
- SARIMA vinner på alle tre mål.

**Hvorfor sammenligne mot naiv?** Naiv = "bestill samme som i fjor" = realistisk approksimasjon av hva butikken gjør i dag. Forbedring mot naiv er den relevante målestokken, ikke forbedring mot en teoretisk perfekt modell.

## Steg 5: Residualdiagnostikk

**Hva vi gjorde:** Sjekket at SARIMA-modellen er teknisk god ved tre tester:
- **ADF** på residualene (etter differensiering) → stasjonær (p < 0,01)
- **Ljung-Box** → ingen gjenværende autokorrelasjon (p > 0,05)
- **Shapiro-Wilk** → tilnærmet normalfordelt

**Hvorfor er dette viktig?** Fordi vi senere skal bruke modellens $\sigma$ i newsvendor-formelen. Hvis modellen var dårlig spesifisert, ville $\sigma$ være feil og hele bestillingsanbefalingen kollapse.

## Steg 6: Diebold-Mariano-test

**Hva vi gjorde:** Testet om SARIMA-forbedringen er statistisk signifikant.

**Resultater:**

| Tap-funksjon | DM-stat | Tosidig p | Ensidig p |
|---|---:|---:|---:|
| Kvadratisk (≈RMSE) | –1,10 | 0,295 | 0,147 |
| Absolutt (≈MAE) | –1,85 | 0,091 | **0,045** |

**Konklusjon:** Med ensidig test (som passer vår teoretiske forhåndsforventning) er forbedringen signifikant på 5 %-nivå (p = 0,045).

**Hvorfor brukte vi Harvey-Leybourne-Newbold-korreksjon?** Fordi vi har kort testserie ($n = 12$). HLN-korreksjonen tar hensyn til at standard DM-test er for liberal for små utvalg.

## Steg 7: Newsvendor-beregning

**Hva vi gjorde:** Brukte SARIMA-prognosen ($\mu$ og $\sigma$) til å regne ut optimal bestilling med newsvendor-formelen.

**Resultater:**

| Sesong 2025 | $\mu_{\text{SARIMA}}$ | $\sigma$ | $Q^*$ | $Q_{\text{naiv}}$ | Faktisk |
|---|---:|---:|---:|---:|---:|
| Vår (mar–aug) | 5 673 | 448 | **5 975** | 5 389 | 6 000 |
| Høst (sep–feb) | 3 166 | 448 | **3 468** | 4 120 | 3 657 |

**Tolkning:**
- Vår: newsvendor anbefaler 586 par mer enn naiv. Naiv undervurderer den voksende trenden.
- Høst: newsvendor anbefaler 652 par mindre enn naiv. Naiv overvurderer fordi den repeterer 2024s engangshopp i september.

**Hva er sikkerhetslager?** Det er den ekstra delen $z_\alpha \cdot \sigma = 0,67 \cdot 448 = 302$ par. Det er det vi legger på toppen av forventet etterspørsel for å absorbere prognoseusikkerheten.

## Steg 8: Økonomisk evaluering

**Hva vi gjorde:** Regnet ut realisert profitt under begge strategier, gitt faktisk 2025-salg.

**Resultater:**

| Komponent | Naiv | Newsvendor | Differanse |
|---|---:|---:|---:|
| Bruttoresultat | 5 335 000 | 5 665 959 | +330 959 |
| Tapt salg-kost | 366 600 | 128 241 | –238 359 |
| Netto årseffekt | 4 968 400 | 5 537 718 | **+569 318** |

**Avrundet til kommunikasjon**: **≈ +570 000 NOK / +11,5 %** i nettoresultat.

**Viktig**: Den totale bestillingen er nesten lik (9 443 vs 9 509 par). Gevinsten kommer **ikke** fra å bestille mer, men fra **bedre fordeling** mellom vår og høst.

## Steg 9: Sensitivitetsanalyse

**Hva vi gjorde:** Sjekket hvor mye $Q^*$ endres hvis vi varierer de antatte prisene ($p, w, s$).

**Resultater (Tabell 4.5 og Figur 4.6, tornado-diagram):**

| Endring | Q* vår | % av basis |
|---|---:|---:|
| Lav margin (p=1000) | 5 866 | –1,8 % |
| **Basis (p=1200, s=400)** | **5 975** | **0 %** |
| Høy margin (p=1500) | 6 080 | +1,8 % |
| Lav restverdi (s=200) | 5 786 | –3,2 % |
| Høy restverdi (s=550) | 6 312 | +5,6 % |

**Konklusjon:** Q\* varierer med under ±6 % over rimelige scenarioer. Konklusjonen om at newsvendor slår naiv beholdes i alle scenarioer.

**Hva er den mest sensitive parameteren?** Restverdien $s$. Logisk: lav $s$ øker overstockingskostnaden $C_o = w - s$ og trekker $Q^*$ ned mot $\mu$.

## Steg 10: Bootstrap-usikkerhet

**Hva vi gjorde:** Simulerte 10 000 framtidige etterspørselsrealisasjoner og beregnet gevinsten under hver.

**Resultater:**

| Mål | Verdi |
|---|---:|
| Forventet gevinst | +333 000 NOK |
| 95 % konfidensintervall | [–499 000, +834 000] NOK |
| Sannsynlighet for positiv gevinst | 88,1 % |
| Sannsynlighet for > 100k NOK | 64,0 % |
| Sannsynlighet for > 500k NOK | 38,9 % |

**Tolkning:** I 88 % av framtidige år forventer vi at newsvendor er bedre enn naiv. I 12 % av årene kan naiv tilfeldigvis være bedre. Forventet gevinst er substansielt positiv, men ikke garantert hvert år.

---

# Del 5 — Resultatene: tallene som teller

Hvis du skal pugge 10 tall til muntlig, er det disse:

### Prognosepresisjon (FS2)
1. **SARIMA MAE = 140 par/mnd**
2. **SARIMA MAPE = 16,9 %**
3. **Forbedring vs naiv: +14,0 %**
4. **Diebold-Mariano ensidig p = 0,045** (signifikant)
5. **Årssum 2025: 4,3 % avvik mot faktisk salg**

### Bestillingsanbefalinger (FS3)
6. **Q\* vår 2025 = 5 975 par** (Q_naiv = 5 389)
7. **Q\* høst 2025 = 3 468 par** (Q_naiv = 4 120)
8. **Servicenivå = 75 %** (CR = 0,75)
9. **Sikkerhetslager = 302 par per sesong**

### Økonomi (FS4)
10. **Observert gevinst 2025 = +570 000 NOK / +11,5 %**
11. **Forventet gevinst (bootstrap) = +333 000 NOK**
12. **95 % KI = [–499k, +834k]**, 88 % sannsynlighet for positiv gevinst

### Robusthet
13. **Sensitivitet Q\* < ±6 %** under prisparameter-variasjoner
14. **Rangering newsvendor > naiv beholdes** i alle testede scenarioer

---

# Del 6 — Begrunnelser: hvorfor vi valgte som vi gjorde

Dette er nesten viktigere enn tallene. Sensor vil spørre **"hvorfor?"**. Her er svarene:

## Hvorfor SARIMA og ikke ARIMA?
ARIMA uten sesongledd ga MAE 229 par – verre enn naiv baseline (163). Sesongleddet bærer informasjon i et sterkt sesongavhengig datasett. Det er ikke bare en teoretisk forventning – vi har empirisk bevis.

## Hvorfor newsvendor og ikke EOQ?
EOQ forutsetter konstant etterspørsel og fleksibel bestillingsfrekvens. Vi har stokastisk sesongetterspørsel og kun to bestillingsvinduer per år. Det er den klassiske newsvendor-situasjonen.

## Hvorfor månedsdata og ikke ukentlige eller daglige?
- Måneder matcher Skoringens egne rapporter
- Daglige data har for mye støy fra dag-til-dag-variasjon
- Ukentlige ville gitt 156 observasjoner, men sesongstrukturen ($s = 52$) blir vanskelig å estimere med kun 3 års data
- Måneder gir oss 36 observasjoner og $s = 12$, som er forsvarlig

## Hvorfor antatte priser, ikke faktiske?
Skoringen kunne ikke gi oss sensitive pris- og margindata. Vi brukte antatte verdier basert på bransjeerfaring ($p = 1200$, $w = 600$, $s = 400$) og gjorde sensitivitetsanalyse for å vise at konklusjonen er robust. **Etisk valg**: vi avslører ikke butikkens konkurransesensitive priser.

## Hvorfor aggregert SKU?
- Skoringens datasystem rapporterte ikke konsekvent på SKU-nivå
- SKU-modell ville krevd 10–100x mer data for å estimere stabilt
- Aggregert SKU er en forsvarlig forenkling for **strategisk** bestillingsanbefaling. Det fanger ikke størrelsesallokering, men det er en kjent begrensning som er identifisert som videre arbeid.

## Hvorfor 75 % servicenivå?
Det er det matematisk optimale gitt våre priser (CR = $(p-w)/(p-s)$ = 0,75). Butikkeier kan velge høyere strategisk, men 75 % er det "newsvendor-riktige".

## Hvorfor in-sample residualer for $\sigma$?
Hvis vi brukte test-residualer (2025), ville vi bruke samme data både til å estimere usikkerheten og til å evaluere økonomien. Det er sirkulær resonnement. In-sample (2023-2024) gir et uavhengig sigma-estimat.

## Hvorfor ensidig DM-test?
Vi har en *a priori* teoretisk forventning om retningen (SARIMA skal være bedre enn naiv i sesongavhengig data, basert på 50 år med litteratur). Da er ensidig test forsvarlig og riktigere (Harvey, Leybourne & Newbold, 1997).

## Hvorfor bootstrap-simulering?
Punktestimatet på 570 000 NOK gjelder bare 2025. For å si noe om hva som er **forventet** i framtiden, må vi simulere mange framtidige realisasjoner. Bootstrap gir oss en distribusjon, ikke bare ett tall.

## Hvorfor kvantitativ casestudie?
Vi har én bedrift (ikke statistisk utvalg), men vi bruker kvantitative metoder (statistikk, optimering). Det er en kvantitativ casestudie. Designet er deduktivt: vi tar etablert teori (SARIMA, newsvendor) og tester den i et konkret case.

---

# Del 7 — Begrensninger og kritisk refleksjon

Dette må du være ærlig om i muntlig eksamen. Hvis du later som om oppgaven ikke har svakheter, virker du naiv.

## Begrensning 1: Aggregeringsnivå
Vi modellerer alle sko som én SKU. I virkeligheten avhenger tilgjengelighet av størrelse: en kunde med størrelse 39 har null tilgjengelighet hvis butikken bare har 42 igjen. Vår modell ser ikke det.

**Hvordan ville en utvidelse se ut?** Multi-produkt $(Q,R)$-modell med delt kapasitet (Silver et al., 2016). Krever vesentlig mer data (SKU-nivå) og mer kompleks modell.

## Begrensning 2: Få beslutningsøyeblikk
Med kun 12 testmåneder har Diebold-Mariano-testen lav statistisk styrke. Ensidig p = 0,045 er signifikant, men marginalt. Lengre testperiode (24+ måneder) vil gi tryggere konklusjon.

## Begrensning 3: Antatte priser
$p, w, s$ er estimat. Sensitivitetsanalysen viser at konklusjonen er robust, men kronetallene har ±10–15 % usikkerhet.

## Begrensning 4: Ingen eksogene faktorer
Vi har ikke modellert vær, kampanjer, lokale begivenheter. Lv et al. (2023) viser at vær alene kan redusere prognosefeil 10–20 % i klesdetaljhandelen. ARIMAX-utvidelse er anbefalt som videre arbeid.

## Begrensning 5: Treningsperiode 24 måneder
Det er nær minste forsvarlige datavolum for SARIMA(1,1,1)(1,1,1)$_{12}$. Hvis det skjer et strukturelt skift i markedet (ny konkurrent, pandemi, motesvingning), vil modellen feile inntil den får trent på den nye virkeligheten.

## Begrensning 6: Ekstern validitet
Funnene gjelder strengt tatt bare for Skoringen Råholt 2023–2025. Å generalisere til andre butikker eller bransjer krever replikasjon.

---

# Del 8 — Forventede sensorspørsmål + svar-forslag

Disse er konkrete spørsmål du sannsynligvis får. Forslag til svar er korte – ikke pugg ordrett, men forstå strukturen.

## Q: "Hva er hovedfunnet i oppgaven?"
**A**: "Vi har vist at en kombinasjon av SARIMA-prognose og newsvendor-modell kan forbedre Skoringen Råholts årlige nettoresultat med om lag 570 000 NOK eller 11,5 prosent under 2025-data. Forventningen over framtidige år, basert på bootstrap-simulering, er noe lavere – 333 000 NOK – men med 88 prosent sannsynlighet for positiv gevinst. Det viktige er at gevinsten ikke kommer fra å bestille mer eller mindre totalt, men fra bedre fordeling mellom vår og høst."

## Q: "Hvorfor er dette viktig for Skoringen?"
**A**: "Skoringen tar to store økonomiske veddemål i året med 6 måneders horisont. Å treffe riktig er hele lønnsomhetsdriveren – for mye gir overlager med rabattsalg, for lite gir tomme hyller og tapte kunder. Modellen vi har bygget gir dem et statistisk forankret beslutningsgrunnlag i stedet for ren erfaringsbasert magefølelse."

## Q: "Hva er den største svakheten i oppgaven?"
**A**: "Aggregeringen til én SKU. Vi modellerer alt skosalg som én vare, men i virkeligheten er størrelsesfordelingen kritisk. En kunde med størrelse 39 bryr seg ikke om at butikken har 100 sko i 42. Multi-SKU-modell med delt kapasitet er anbefalt som viktigste videre arbeid."

## Q: "Hvis dere skulle gjort oppgaven på nytt – hva ville dere endret?"
**A**: "Tre ting. (1) Få faktiske enhetspriser fra Skoringens regnskap fra start, så vi kunne unngått sensitivitetsanalysen. (2) Inkludere værvariabler via ARIMAX – Lv et al. viser at det kan redusere prognosefeilen 10–20 prosent. (3) Lengre testperiode – 12 måneder gir lav statistisk testpotens."

## Q: "Hva er forskjellen på MAE og RMSE?"
**A**: "MAE er gjennomsnittlig absolutt avvik. RMSE er kvadratrota av gjennomsnittlig kvadrert avvik. Forskjellen er at RMSE straffer store enkeltavvik hardere enn MAE. Hvis du bommer 50 par på 9 måneder og 500 par på 3 måned, har du samme MAE, men ulik RMSE."

## Q: "Hvorfor bruker dere ensidig Diebold-Mariano?"
**A**: "Fordi vi har en teoretisk forventning *før* vi testet om at SARIMA skal være bedre enn naiv i et sesongavhengig datasett. Det er begrunnet i 50 år med litteratur. Når retningen er forhåndsantatt, er ensidig test både forsvarlig og statistisk korrekt (Harvey, Leybourne & Newbold, 1997). Den tosidige varianten svarer på et annet spørsmål – 'er de forskjellige uansett retning?' – som ikke er vår faktiske hypotese."

## Q: "Hva betyr p = 0,045 i praksis?"
**A**: "Det betyr at hvis SARIMA og naiv egentlig var like presise modeller, ville vi sett en så stor forbedring i SARIMAs favør kun 4,5 prosent av gangene ved tilfeldig variasjon. Det er lavt nok til at vi forkaster hypotesen om at modellene er like. Vi anerkjenner at testen er på grensa, og at lengre testperiode hadde gitt sterkere konklusjon."

## Q: "Hva sier bootstrap-resultatet egentlig?"
**A**: "Bootstrap viser fordelingen av gevinster vi kan forvente over framtidige år. Forventet gevinst er 333 000 NOK. Det er lavere enn det observerte 570 000 i 2025 fordi 2025 var en gunstig realisasjon. 95 prosent konfidensintervall inkluderer både negative og positive verdier, men sannsynligheten for positiv gevinst er 88 prosent. Newsvendor er forventet bedre, men ikke garantert hvert år."

## Q: "Hvorfor brukte dere PDF-pipelinen?"
**A**: "Fordi Skoringens salgsdata var låst i over 1 000 PDF-rapporter. Standard import-verktøy som Excel kan ikke lese tabelldata fra PDF. Vi bygde en koordinatbasert pipeline med pdfplumber som identifiserer hvor på siden hver kolonne er, og henter ut tallene. Det er en metodisk bidrag – den samme pipelinen kan brukes av andre butikker med samme problem."

## Q: "Hva er bullwhip-effekten og hvorfor er den relevant?"
**A**: "Bullwhip-effekten er at små svingninger i sluttkundens etterspørsel forsterkes oppover i forsyningskjeden. Lee et al. (1997) kvantifiserer det til 2–4 ganger forsterkning per ledd. For Skoringen er det relevant fordi de plasserer to gigantbestillinger i året – det er et meget volatilt signal til leverandøren. Hvis Skoringen delte sin månedsprognose med leverandøren, kunne leverandøren planlegge bedre, og på sikt forhandle frem bedre priser eller leveringsbetingelser. Det er anbefaling 3 i konklusjonen."

## Q: "Kunne dere ikke brukt maskinlæring? Det er moderne."
**A**: "Vi vurderte LightGBM/gradient boosting, men forkastet det av to grunner. (1) Vi har 24 treningsobservasjoner – ML-modeller blir overfit på så lite data. (2) ML-modeller sin styrke er å håndtere mange korrelerte features og ikke-lineære interaksjoner, og det får ikke utfoldet seg uten større datavolum og rikere feature-sett. Ved utvidelse til SKU/dag-nivå med 10 000+ observasjoner ville ML være forsvarlig, og vi har anbefalt det som videre arbeid."

## Q: "Hvorfor 75 % servicenivå og ikke 95 %?"
**A**: "75 prosent er det matematisk optimale gitt våre antatte priser. Formelen er kritisk forhold = (p-w)/(p-s) = 600/800 = 0,75. Hvis butikken vil ha 95 % – for å bygge merkevarekapital eller redusere risiko for utsolgt – kan de øke. Men det krever større sikkerhetslager og dermed mer kapitalbinding. Det er en strategisk beslutning butikkeier bør ta bevisst."

## Q: "Hvorfor antatte priser? Det undergraver hele analysen!"
**A**: "Vi var åpne om det. To grunner: Skoringen kunne ikke gi sensitive marginstall, og pris-strukturen er konkurransesensitiv informasjon. Vi gjorde sensitivitetsanalyse for å vise at *rangeringen* mellom newsvendor og naiv er robust – konklusjonen kollapser ikke uansett hvilke rimelige verdier vi velger. Men ja, de absolutte kronetallene har ±10–15 prosent usikkerhet, og det er den første anbefalingen for videre arbeid."

## Q: "Hva er studiens bidrag til faget?"
**A**: "Tre ting. (1) Et *metodisk* bidrag: pipelinen som konverterer ustrukturerte PDF-rapporter til strukturert tidsserie er reproduserbar og kan brukes av andre butikker med samme problem. (2) Et *empirisk* bidrag: vi gir et konkret datapunkt for hva man kan forvente i prognoseforbedring og økonomisk effekt i en liten norsk skobutikk. (3) Et *integrerende* bidrag: vi viser hvordan SARIMA, newsvendor og bullwhip-teori kan kobles til én sammenhengende beslutningsstøttemodell – en hybrid løsningsstrategi i Puchinger og Raidls (2005) forstand."

## Q: "Hva med etikk og personvern – hvordan håndterte dere det?"
**A**: "Daglig leder Marit Stoksflod ga muntlig samtykke til at butikken, hennes navn og salgsdataene brukes. Dataene inneholder ingen personopplysninger – kassesystemets Z-rapporter har varekoder og beløp, ikke kundeidentifikatorer – så GDPR er ikke direkte berørt. Prisparameterne er antatte estimat, ikke faktiske tall fra regnskapet, nettopp for å unngå å avsløre konkurransesensitiv informasjon. Vi anerkjenner at samtykket ideelt sett burde vært skriftlig fra start."

## Q: "Var det noe som overrasket dere i resultatene?"
**A**: "Tre ting. (1) Den *flate profittkurven* nær Q* – newsvendor-løsningen er langt mer robust enn forventet for små feil i parameterne. Selv ±300 par avvik fra optimum har marginal effekt på profitt. (2) Høstsesongen ble *nedjustert*, ikke oppjustert – vi trodde modellen primært ville vise at butikken bestilte for lite, men den viste at naiv strategi *overestimerte* høsten pga. et engangshopp i september 2024. (3) ETS var *dårligere enn naiv* baseline – det var uventet fordi ETS normalt er konkurransedyktig med SARIMA."

## Q: "Hva er bærekraftsbidraget?"
**A**: "Redusert overstock betyr færre sko som produseres uten å nå en kunde. Newsvendor reduserer overlager med 463 par i høst 2025, tilsvarende ca. 5 800 kg CO₂ spart. Det er beskjedent i absolutt størrelse – omtrent som én flyreise Oslo–New York – men det illustrerer prinsippet om at logistikkanalyse kan gi både økonomiske og miljømessige gevinster. Kobling til FNs bærekraftsmål 12."

## Q: "Hvordan anbefaler dere at butikken implementerer dette i praksis?"
**A**: "Gradvis i tre faser. Fase 1 (høst 2026): parallellkjøring der modellen genererer en anbefaling, men daglig leder tar beslutningen. Avvik logges. Fase 2 (2027): hybridkjøring der modellen er primær input, men leder kan justere ±10 %. Fase 3 (fra 2028): full modellkjøring med menneskelig overstyring kun ved unike hendelser. Poenget er å bygge tillit gradvis – ikke erstatte erfaring over natten."

## Q: "Hva er konklusjonen i en setning?"
**A**: "SARIMA-prognose kombinert med newsvendor-modell gir Skoringen Råholt et statistisk forankret beslutningsgrunnlag for sesongbestilling, med en forventet årlig effekt på +333 000 NOK (88 % sannsynlighet for positiv gevinst), drevet av bedre fordeling – ikke større volum – mellom vår og høst."

---

# Del 9 — Cheat sheet

## Alle viktige forkortelser

| Forkortelse | Forklaring |
|---|---|
| AIC | Modellvalgskriterium (lavere = bedre) |
| ARIMA | AutoRegressive Integrated Moving Average – tidsseriemodell uten sesong |
| ARIMAX | ARIMA med eksogene variabler (vær, kampanjer) |
| CR | Critical Ratio = (p-w)/(p-s) – optimalt servicenivå |
| DM | Diebold-Mariano-test |
| ETS | Exponential Smoothing (Holt-Winters er en variant) |
| EOQ | Economic Order Quantity – klassisk Harris-formel (ikke brukt) |
| HLN | Harvey-Leybourne-Newbold – små-utvalg-korreksjon for DM |
| KI | Konfidensintervall |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| RMSE | Root Mean Squared Error |
| SARIMA | Seasonal ARIMA – vår hovedmodell |
| SKU | Stock Keeping Unit – én størrelse av én modell |
| VMI | Vendor Managed Inventory |
| Z-rapport | Daglig avslutningsrapport fra kasse |

## Alle nøkkeltall

| Mål | Verdi |
|---|---:|
| SARIMA MAE | 140 par/mnd |
| SARIMA MAPE | 16,9 % |
| Forbedring vs naiv | +14,0 % |
| DM ensidig p | 0,045 |
| $\sigma_{\text{mnd}}$ | 182,9 par |
| Sikkerhetslager | 302 par/sesong |
| $z_\alpha$ | 0,6745 (75 % CR) |
| Q\* vår 2025 | 5 975 par |
| Q\* høst 2025 | 3 468 par |
| Faktisk salg vår 2025 | 6 000 par |
| Faktisk salg høst 2025 | 3 657 par |
| Observert gevinst 2025 | +570 000 NOK (+11,5 %) |
| Bootstrap forventet | +333 000 NOK |
| 95 % KI | [–499k, +834k] |
| P(positiv gevinst) | 88,1 % |

## Newsvendor på 30 sekunder

$$ Q^* = \mu + z_\alpha \cdot \sigma $$

- $\mu$ = forventet etterspørsel (fra SARIMA)
- $\sigma$ = standardavvik (fra SARIMA)
- $z_\alpha = \Phi^{-1}(\text{CR})$ der CR = $(p-w)/(p-s)$ = 600/800 = 0,75

For oss: $Q^* = \mu + 0,67 \cdot \sigma$

## SARIMA på 30 sekunder

$\text{SARIMA}(p,d,q)\times(P,D,Q)_s$ – kombinerer:
- AR ($p$): "ser tilbake $p$ måneder"
- I ($d$): "modellerer endringer, ikke nivåer"
- MA ($q$): "lærer av forrige prognosefeil"
- Sesong-versjoner ($P, D, Q$) med periode $s = 12$ for månedsdata

Vår: SARIMA(1,1,1)(1,1,1)$_{12}$

## De fire forskningsspørsmålene

- **FS1**: Datafangst – kan vi låse opp PDF-dataene? **Ja, pipeline ble bygd.**
- **FS2**: Prognose – hvilken modell er best? **SARIMA.**
- **FS3**: Bestilling – hva er optimalt $Q^*$? **5 975 vår / 3 468 høst.**
- **FS4**: Økonomi – hva er gevinsten? **+570k observert, +333k forventet.**

## Tre overraskende funn

1. **Flat profittkurve** – ±300 par fra Q* gir marginal profittendring (robust)
2. **Høst ble nedjustert** – modellen sa "bestill mindre", ikke "bestill mer"
3. **ETS tapte mot naiv** – uvanlig, men skyldes additiv form vs voksende sesong

## Studiens tre bidrag

1. **Metodisk**: reproduserbar PDF→CSV-pipeline for små bedrifter
2. **Empirisk**: +14 % MAE-forbedring og +11,5 % nettoresultat i norsk skobutikk
3. **Integrerende**: SARIMA + newsvendor + bullwhip i én hybrid beslutningsstøttemodell

## Bærekraft

- 463 par mindre overlager → ~5 800 kg CO₂ spart/år
- FNs bærekraftsmål 12 – Ansvarlig forbruk og produksjon
- Forutsetter at leverandøren faktisk produserer mindre

---

## Avsluttende råd til muntlig

1. **Forklar med eksempler, ikke formler.** Hvis du sier "newsvendor balanserer over- og understockingskostnad" virker du faglig. Hvis du sier "tenk på avisselgeren som må kjøpe inn aviser om morgenen" virker du forståelig. Begge er riktige, men det andre er bedre eksamensteknikk.

2. **Vær ærlig om svakheter.** Hvis du later som om oppgaven er perfekt, mister du tillit. Sensor *vet* at oppgaven har begrensninger. Det at *dere* vet hva de er, viser modenhet.

3. **Knytt tallene til praksis.** Ikke bare si "+570 000 NOK". Si "+570 000 NOK – det er omtrent en hel årslønn ekstra for butikken". Da viser du at du forstår størrelsen.

4. **Bruk "vi" og "vår analyse"**, ikke "modellen" eller "rapporten". Det er deres oppgave. Eie den.

5. **Hvis du er usikker på et spørsmål**, si "Det er et interessant spørsmål. La meg tenke et øyeblikk." Det er bedre enn å gjette.

Lykke til. Dere har gjort en grundig og solid jobb.

— Gustavo
