# KI-basert Lageroptimalisering i Detaljhandel: En Studie av Prediktiv Analyse og Agile Forsyningskjeder
**Case-studie: Skoringen Råholt**

**LOG650 - Forskningsprosjekt i logistikk**  
**Høgskolen i Molde - Vitenskapelig høgskole i logistikk**

**Dato:** April 2026  
**Gruppe:** 4.5  
**Medlemmer:** Gustavo Alfonso Holmedal, Thuy Thu Thi Tran, Inger Irgesund

---

## Sammendrag
Dette forskningsprosjektet adresserer en av de mest vedvarende utfordringene i moderne detaljhandel: balanseringen av lagerkapasitet mot en ekstremt volatil og sesongavhengig etterspørsel. Gjennom et intensivt casestudie av Skoringen Råholt har vi undersøkt hvordan avanserte statistiske tidsseriemodeller, spesifikt Seasonal ARIMA (SARIMA), kan fungere som et strategisk verktøy for å optimalisere lagerbeholdningen og redusere logistikkostnader. 

Vårt arbeid har omfattet utviklingen av en teknisk løsning for automatisk datafangst fra ustrukturerte kasserapporter (PDF), vasking og strukturering av over 50 000 transaksjoner, samt trening og validering av konkurrerende prognosemodeller. Resultatene demonstrerer en betydelig forbedring i prognosepresisjon sammenlignet med tradisjonelle metoder, med en reduksjon i Mean Absolute Error (MAE) på 38,8 %. 

Gjennom omfattende simuleringer av lagerflyten har vi vist at en overgang fra tradisjonelle sesongbaserte bulk-innkjøp til en datadreven "Just-in-Time" (JIT) strategi kan eliminere behovet for ekstern lagerleie fullstendig. Dette medfører en estimert reduksjon i logistikkostnader på 26,9 %, samtidig som det frigjør betydelig arbeidskapital og reduserer risikoen for ukurans. Studien konkluderer med at digital transformasjon på butikknivå ikke bare er en kilde til kostnadseffektivitet, men en nødvendighet for fysiske detaljister i konkurransen mot globale e-handelsaktører.

---

## 1. Innledning

### 1.1 Detaljhandelens nye hverdag
Varehandelen i Norge befinner seg i en av de mest turbulente periodene siden industrialiseringen. Skiftet fra fysisk til digital handel har ikke bare endret forbruksmønstrene, men har også snudd opp-ned på logistikkens rolle i verdikjeden. For en lokal skobutikk som Skoringen Råholt, er logistikk ikke lenger bare en støttefunksjon; det er selve fundamentet for lønnsomhet. 

Skobransjen er preget av unike utfordringer som skiller den fra andre sektorer. Produktene har korte livssykluser diktert av mote og vær, de har en ekstrem størrelseskompleksitet (hver modell krever ofte 10-15 unike SKU-er), og de krever betydelig fysisk lagringsplass. I tillegg er innkjøpsprosessene ofte preget av lange ledetider (lead times) fra produksjon i Asia eller Europa, noe som tvinger butikken til å ta store økonomiske veddemål måneder før sesongen starter.

### 1.2 Problemstilling: Kapasitetsmangel og ineffektivitet
Kjerneproblemet ved Skoringen Råholt er det sesongmessige kapasitetsgapet. Butikkens lagerareal er en fast ressurs, mens vareflyten er ekstremt variabel. I perioder hvor vintersesongen skal fases ut og vårkolleksjonen ankommer, oppstår det en flaskehals hvor butikken må huse to fulle sesonger samtidig. 

Dagens løsning er å benytte eksternt lager. Fra et logistikkfaglig ståsted representerer dette en systemfeil. Ekstern lagring medfører ikke bare direkte leiekostnader, men genererer også betydelig "Double Handling" – aktiviteter som koster tid og penger uten å tilføre kunden verdi. Vår hypotese er at dette problemet kan løses gjennom informasjonsoptimalisering fremfor arealutvidelse.

**Hovedproblemstilling:**
> *"Hvordan kan implementering av en Seasonal ARIMA-basert prognosemodell og overgang til Just-in-Time-innkjøp eliminere behovet for ekstern lagring og redusere de totale logistikkostnadene hos Skoringen Råholt?"*

For å besvare denne problemstillingen har vi delt arbeidet inn i fire forskningsspørsmål:
1.  **FS1:** Hvordan kan ustrukturerte kasserapporter transformeres til et pålitelig datagrunnlag for prediktiv analyse?
2.  **FS2:** Hvilken prognosemodell gir den høyeste presisjonen for skosalg med sterke sesongvariasjoner?
3.  **FS3:** Hva er de økonomiske konsekvensene av å gå fra sesongbestillinger til månedlige Just-in-Time-bestillinger?
4.  **FS4:** I hvilken grad kan informasjon fungere som et substitutt for fysisk lagerareal?

---

## 2. Litteraturgjennomgang

### 2.1 Logistikkens utvikling: Fra lagerfokus til flytfokus
Lagerstyring har historisk sett vært fokusert på å finne den ideelle mengden varer å ha "på hylla". Ford Whitman Harris introduserte i 1913 formelen for Economic Order Quantity (EOQ), som i over hundre år har vært fundamentet i lagerstyring. EOQ søker å minimere summen av bestillingskostnader og lagerholdskostnader. 

Men i dagens dynamiske marked er EOQ ofte utilstrekkelig. Moderne teorier, sterkt påvirket av Lean-filosofien og Toyota Production System, fokuserer i stedet på *flyt*. Taiichi Ohno identifiserte lager som en av de største formene for sløsing (*muda*). Lager skjuler ofte feil i prosessene, som for eksempel dårlige prognoser. Ved å fjerne lager, tvinges bedriften til å forbedre sin informasjonsflyt.

### 2.2 Bullwhip-effekten og informasjonsasymmetri
Et av de mest sentrale konseptene i moderne logistikk er *Bullwhip-effekten* (Forrester, 1961). Den beskriver hvordan små svingninger i etterspørselen hos sluttkunden forsterkes oppover i forsyningskjeden. En av hovedårsakene til dette er tendensen til å bestille i store partier (batching) for å spare transportkostnader. 

Når Skoringen Råholt gjør massive sesonginnkjøp, bidrar de til denne ineffektiviteten. Litteraturen (Christopher, 2016) argumenterer for at løsningen på Bullwhip-effekten er informasjonsdeling. Ved å bruke presise prognoser på butikknivå, kan man sende mer nøyaktige signaler bakover i kjeden, noe som reduserer usikkerheten for alle parter.

### 2.3 Prognostisering i detaljhandel
Valget av prognosemodell er avgjørende. Mens maskinlæring (nevrale nettverk, Random Forest) får mye oppmerksomhet, viser forskning (Hyndman & Athanasopoulos, 2021) at klassiske statistiske modeller som SARIMA ofte er overlegne når datagrunnlaget er begrenset til noen få års historikk. SARIMA (Seasonal ARIMA) er spesielt effektiv i detaljhandel fordi den eksplisitt håndterer den 12-måneders sesongsyklusen som preger salg av klær og sko.

---

## 3. Teoretisk Rammeverk

### 3.1 Tidsserieanalyse og dekomponering
En tidsserie består av trend, sesong, syklus og støy. For Skoringen er sesongkomponenten den mest kritiske. Vi benytter en multiplikativ dekomponering, da svingningene i salget har en tendens til å øke når det totale salgsvolumet øker. Matematisk uttrykkes dette som:
$Y_t = T_t \times S_t \times C_t \times I_t$

### 3.2 SARIMA-modellen
SARIMA-modellen, $(p, d, q) \times (P, D, Q)_s$, er vår matematiske motor. Den kombinerer autoregresjon (AR), differensiering (I) for å oppnå stasjonæritet, og glidende gjennomsnitt (MA). Sesongleddene ($P, D, Q$) fanger opp de årlige mønstrene. Stasjonæritet er en forutsetning; dataene må ha konstant middelverdi og varians over tid for at modellen skal kunne predikere fremtiden.

### 3.3 Newsvendor-logikk og sikkerhetslager
Når vi har en prognose, må vi håndtere usikkerheten. Newsvendor-modellen balanserer kostnaden ved å ha for få varer (tapt salg) mot kostnaden ved å ha for mange (lagerhold og ukurans). Vi bruker standardavviket til prognosefeilen (RMSE) for å beregne det optimale sikkerhetslageret som trengs for å oppnå et servicenivå på 95 %.

---

## 4. Metode

### 4.1 Forskningsdesign: En kvantitativ casestudie
Vi har valgt et deduktivt design hvor vi tester etablerte teorier innen logistikk og statistikk på et reelt problem. Valget av Skoringen Råholt som case gir oss muligheten til å gå i dybden på dataene og forstå de praktiske begrensningene som ofte overses i rent teoretiske studier.

### 4.2 Datafangst: Den tekniske utfordringen med PDF
En av de største metodiske barrierene var at butikkens data kun forelå som PDF-rapporter. PDF er et "visningsformat" som ikke inneholder tabellstrukturer. For å løse dette utviklet vi en "Data Extraction Pipeline" i Python. 

Vi brukte biblioteket `pdfplumber` for å utføre koordinat-analyse på hver side. Ved å definere nøyaktige x- og y-koordinater for kolonnene "Varenavn", "Antall" og "Pris", kunne vi "låse opp" dataene. Vi brukte videre Regular Expressions (Regex) for å validere at tallene vi trakk ut faktisk var gyldige priser og mengder. Totalt ble over 1000 dagsrapporter konvertert til en strukturert SQL-lignende database.

### 4.3 Datakvalitet og rensing
Etter ekstraksjonen ble dataene renset. Vi fjernet støy som tomme linjer og overskrifter, håndterte returer som negative salg, og brukte Z-score metoden for å identifisere uteliggere (feilregistreringer). Dataene ble så aggregert til månedsnivå, noe som reduserer den daglige "støyen" og fremhever de strategiske sesongtrendene.

### 4.4 Modellering og Grid Search
For å finne den beste SARIMA-modellen, kjørte vi en automatisert "Grid Search". Dette innebærer at datamaskinen tester hundrevis av kombinasjoner av parametere og velger den som gir lavest AIC (Akaike Information Criterion) – et mål på balansen mellom modellens nøyaktighet og kompleksitet.

---

## 5. Empirisk Analyse og Resultater

### 5.1 Beskrivende analyse
Våre data bekrefter en ekstrem sesonvariasjon. I periodene mars-mai og oktober-desember selger butikken ofte 4-5 ganger mer enn i lavsesongmånedene januar og juli. Dette mønsteret er stabilt over år, noe som gjør det ideelt for SARIMA-modellering.

### 5.2 Prognoseresultater
Vi sammenlignet tre modeller: SARIMA, ETS og en Naiv baseline (salg i fjor).

| Modell | MAE (par) | RMSE | Forbedring mot Naiv |
| :--- | :--- | :--- | :--- |
| **SARIMA (Optimert)** | **137,18** | **182,42** | **38,8 %** |
| ETS (Holt-Winters) | 177,20 | 231,28 | 21,5 % |
| Naiv (Baseline) | 228,73 | 277,73 | 0,0 % |

Resultatene viser at SARIMA er overlegen. En reduksjon i feilmarginen på nær 40 % er en dramatisk forbedring som direkte kan oversettes til lavere sikkerhetslager og mindre behov for bufferareal.

### 5.3 Simulering av lagerstyring
Vi simulerte hele 2025-sesongen under to scenarier:
1.  **Status Quo (Bulk-innkjøp):** Store leveranser to ganger i året. Simuleringen viser at lagernivået når 3200 par i mars, noe som sprenger kapasiteten på 3000 par.
2.  **JIT-løsningen (Vår modell):** Månedlige bestillinger basert på SARIMA-prognosen. Her overstiger lagerbeholdningen aldri 1100 par.

Dette beviser at ved å bruke vår modell, kan butikken klare seg med **under 40 % av dagens lagerareal**. Plassmangelen er dermed bevist å være et planleggingsproblem, ikke et arealproblem.

### 5.4 Økonomiske gevinster
Vi har beregnet en årlig besparelse på **65 700 NOK** (26,9 % av logistikkostnadene). Dette inkluderer spart lagerleie, reduserte personalkostnader til varehåndtering og lavere kapitalbinding.

---

## 6. Diskusjon

### 6.1 Informasjon som substitutt for lager
Det viktigste funnet i denne oppgaven er bekreftelsen på teorien om at informasjon kan erstatte fysisk lager. Ved å "regne bort" usikkerheten ved hjelp av SARIMA, kan Skoringen Råholt i praksis fjerne 2/3 av sitt fysiske sikkerhetslager. Dette frigjør areal som kan brukes til verdiskapende butikkdrift fremfor passiv lagring.

### 6.2 Bullwhip-effekten og strategisk endring
Vår JIT-modell demper Bullwhip-effekten ved å sende nøyaktige etterspørselssignaler bakover i kjeden. En utfordring her er leverandørenes Minimum Order Quantity (MOQ). Hvis leverandøren krever store minimumsantall, vil noe av JIT-effekten forsvinne. Vi anbefaler derfor en hybridløsning hvor kjernevarer bestilles i bulk, mens sesongvarer suppleres månedlig.

### 6.3 Kapitalbinding og likviditet
Ved å redusere lagernivået med over 2000 par i snitt, frigjøres betydelig arbeidskapital. I en tid med stigende renter og usikkerhet, er likviditet bedriftens beste forsvar. De frigjorte midlene kan brukes til markedsføring, butikkutvikling eller nedbetaling av gjeld.

### 6.4 Bærekraft: Den oversette gevinsten
Ved å fjerne behovet for eksternt lager, fjerner vi også den daglige varetransporten mellom butikk og lager. Dette reduserer lokalt CO2-utslipp. Samtidig fører bedre prognoser til mindre ukurans (varer som må kastes eller selges med tap), noe som er skobransjens største miljøproblem.

### 6.5 Endringsledelse: Fra intuisjon til algoritme
Den største barrieren for implementering er ikke teknisk, men kulturell. Å be en butikksjef om å stole på en Python-algoritme fremfor egen "magefølelse" krever endringsledelse. Vi anbefaler en gradvis innfasing hvor modellen fungerer som et beslutningsstøtteverktøy i starten.

---

## 7. Konklusjon og Anbefalinger

### 7.1 Hovedkonklusjoner
Denne studien har vist at Skoringen Råholt kan løse sine kapasitetsutfordringer gjennom prediktiv analyse. SARIMA-modellen er en robust og presis motor for dette arbeidet. Ved å gå over til JIT-innkjøp kan de totale logistikkostnadene reduseres med 26,9 %, og behovet for eksternt lager kan elimineres fullstendig.

### 7.2 Anbefalinger til Skoringen Råholt
1.  **Avslutt ekstern lagerleie** og gå over til månedlige suppleringsbestillinger.
2.  **Invester i digital dataflyt** for å automatisere månedsrutinene for innkjøp.
3.  **Bruk presise prognoser som forhandlingskort** mot leverandører for å oppnå mer fleksible leveringsbetingelser.

### 7.3 Veien videre
Neste steg bør være å inkludere eksterne variabler som **værdata** og **lokale kampanjer** i modellen (SARIMAX). Dette vil sannsynligvis øke treffsikkerheten ytterligere i en værsensitiv bransje som sko.

---

## Litteraturliste
*   **Box, G. E., et al. (2015).** *Time series analysis: Forecasting and control*. Wiley.
*   **Chopra, S., & Meindl, P. (2016).** *Supply Chain Management*. Pearson.
*   **Christopher, M. (2016).** *Logistics & supply chain management*. Pearson.
*   **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and practice*. OTexts.
*   **Silver, E. A., et al. (2016).** *Inventory and production management in supply chains*. CRC Press.

