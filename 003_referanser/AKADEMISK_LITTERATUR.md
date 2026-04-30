# Akademisk Litteratur for Bacheloroppgave: Skoringen Råholt

Denne listen inneholder akademiske kilder som løfter oppgaven fra en beskrivelse av data til en logistikkfaglig analyse.

> **Veiledning fra foreleser (april 2026):** Kompendiet brukes som arbeidsredskap underveis i skrivingen – til struktur, språk og oppslag – men siteringene i teksten skal gå til de etablerte primærkildene. De fem primærverkene i seksjon 5 (Pinedo; Hartmann & Briskorn; Vose; Efron & Tibshirani; Puchinger & Raidl) dekker tematisk alle metodeområdene oppgaven berører: scheduling/lotstørrelse, ressursbegrenset planlegging, Monte Carlo-risikoanalyse, bootstrap-statistikk og hybride løsningsmetoder.

## 1. Grunnlag for Prognose og Tidsserier
Disse kildene brukes for å forankre valget av **SARIMA** og **ETS**.

*   **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and practice* (3rd ed.). OTexts.
    *   **Hvorfor:** Dette er standardverket for moderne prognosemetoder.
    *   **Link:** [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
*   **Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).** *Time series analysis: Forecasting and control*. John Wiley & Sons.
    *   **Hvorfor:** Grunnlaget for ARIMA-modeller. Gir tyngde til den matematiske forklaringen i kapittel 2.

## 2. Lagerstyring og Optimalisering
Her henter vi inn **Newsvendor-modellen**, som er kritisk for sesongvarer som sko.

*   **Silver, E. A., Pyke, W. R., & Thomas, D. J. (2016).** *Inventory and production management in supply chains* (4th ed.). CRC Press.
    *   **Hvorfor:** Gir de teoretiske formlene for sikkerhetslager og ordrekvantum.
    *   **Link:** [Info via CRC Press](https://www.routledge.com/Inventory-and-Production-Management-in-Supply-Chains/Silver-Pyke-Thomas/p/book/9781466558618)
*   **Petruzzi, N. C., & Dada, M. (1999).** Pricing and the newsvendor problem: A review with extensions. *Operations Research, 47*(2), 183-194.
    *   **Hvorfor:** En klassisk artikkel som forklarer hvordan man optimerer innkjøp når etterspørselen er usikker og varen har begrenset levetid (som sesongsko).
    *   **Link:** [JSTOR - Operations Research](https://www.jstor.org/stable/223067)

## 3. Spesifikk Forskning på Mote og Detaljhandel
Disse kildene viser at dere forstår den spesifikke konteksten i skobransjen.

*   **Ramos, P., Santos, N., & Rebelo, R. (2015).** Performance of state space and ARIMA models for consumer retail sales forecasting. *Computers & Industrial Engineering, 80*, 151-163.
    *   **Hvorfor:** En direkte sammenligning av modellene dere bruker, utført i en detaljhandelskontekst.
    *   **Link:** [ScienceDirect](https://doi.org/10.1016/j.cie.2014.12.007)
*   **Lv, Z., et al. (2023).** Clothing Sales Forecast Considering Weather Information. *Applied Sciences*.
    *   **Hvorfor:** Støtter forslaget deres om å inkludere vær i fremtidige modeller (Kapittel 5).
    *   **Link:** [MDPI - Open Access](https://www.mdpi.com/2076-3417/13/1/40)

## 4. Logistikkstrategi
*   **Christopher, M. (2016).** *Logistics & supply chain management*. Pearson UK.
    *   **Hvorfor:** Forklarer "Agile Supply Chain" – hvorfor presise data er viktigere enn store lager.
    *   **Link:** [Pearson Education](https://www.pearson.com/en-gb/subject-catalog/p/logistics-supply-chain-management/P200000003310/9781292083797)

## 5. Etablerte primærkilder for scheduling, risiko og hybride metoder
Foreleser anbefalte i april 2026 at oppgaven siterer de etablerte standardverkene innen hvert metodeområde framfor å lene seg på kompendiet alene. De fem kildene under er valgt fordi de tematisk dekker alle metodevalg i oppgaven: planlegging og lotstørrelse, ressursbegrenset koordinering, Monte Carlo-risiko, bootstrap-konfidensintervaller og hybride løsningsstrategier.

*   **Pinedo, M. L. (2016).** *Scheduling: Theory, algorithms, and systems* (5. utg.). Springer.
    *   **Hvorfor:** Standardverket for scheduling. Forankrer §2.3 og §5.3 (bestillingsfrekvens og batching), §3.5 (sesongtiming av newsvendor-bestilling) og pensumets MRP/lotstørrelse-grunnlag (`ch03-sec05-mrp-lotstorrelse`).
*   **Hartmann, S., & Briskorn, D. (2010).** A survey of variants and extensions of the resource-constrained project scheduling problem. *European Journal of Operational Research, 207*(1), 1–14. [https://doi.org/10.1016/j.ejor.2009.11.005](https://doi.org/10.1016/j.ejor.2009.11.005)
    *   **Hvorfor:** Den definitive RCPSP-oversiktsartikkelen. Gir det formelle rammeverket for ressursbegrenset planlegging som ligger bak §3.3 (delt hylleplass for 10–15 SKU per modell) og §6.2 (multi-echelon-koordinering med leverandør under kapasitetstak).
*   **Vose, D. (2008).** *Risk analysis: A quantitative guide* (3. utg.). Wiley.
    *   **Hvorfor:** Standardverket for Monte Carlo-basert risikoanalyse. Primærkilde bak §7.3 (sensitivitetsanalyse av MAPE og newsvendor-parametre), pensumets `ch11-sec03-monte-carlo-risk`, og stresstestingen i `ch11-sec05-stresstest`.
*   **Efron, B., & Tibshirani, R. J. (1993).** *An introduction to the bootstrap.* Chapman & Hall/CRC.
    *   **Hvorfor:** Standardverket for bootstrap. Gir det statistiske grunnlaget for konfidensintervaller på prognosefeil i §3.4 og for empirisk-basert newsvendor-analyse (`ch05-sec05-newsvendor-kontrakter`) når salgsdistribusjonen ikke er normalfordelt.
*   **Puchinger, J., & Raidl, G. R. (2005).** Combining metaheuristics and exact algorithms in combinatorial optimization: A survey and classification. I J. Mira & J. R. Álvarez (red.), *Artificial Intelligence and Knowledge Engineering Applications* (LNCS 3562, s. 41–53). Springer. [https://doi.org/10.1007/11499305_5](https://doi.org/10.1007/11499305_5)
    *   **Hvorfor:** Klassifiserer hvordan ulike metoder kombineres. Forankrer §2.5 (hybrid mellom modellbasert og erfaringsbasert beslutningstaking), §5.2 (SARIMA + LightGBM som komplementære tilnærminger) og §6.2 (SARIMA-prognose + newsvendor + leverandørkoordinering som integrert beslutningssystem).

## 6. Kompendiet som arbeidsredskap
Kompendiet er strukturert som 33 selvstendige Python-prosjekter (kapittel × seksjon). Hver seksjon har `data/`, `src/`, `output/` og `README.md`, og brukes i denne oppgaven som *arbeidsredskap* – til metodisk struktur, språk, oppslag og verifikasjon – ikke som primær sitatkilde. Sitater i rapporten går til de etablerte verkene i seksjon 1–5. Lokal sti: `003_referanser/Kompendium/`. Innholdsfortegnelse: `003_referanser/Kompendium/00_INDEX.md`.

Vi anvender 22 av 33 seksjoner aktivt. Resten (produksjonsplanlegging, kjøretøyruting, kømodeller, plukkruter, binpacking, innkjøpsauksjon) er bevisst utelatt fordi caset – én skobutikk med leverandøravtale via Skoringen-kjeden – ikke berører dem.

### 6.1 Prognose og etterspørsel (kjernepensum for FS2)
*   **`ch01-sec03-trend-og-sesong`** — SARIMA: trend og sesong.
    *   **Hvorfor:** Direkte metodisk grunnlag for §3.2 og §4.4. Pensumets `step01–step06`-pipeline (datainnsamling → stasjonaritet → ACF/PACF → modellestimering → diagnostikk → prognose) er den samme rekkefølgen vi følger på Skoringen-dataene.
*   **`ch01-sec04-eksterne-faktorer`** — ARIMAX med kampanje- og kalenderregressorer.
    *   **Hvorfor:** Forankrer §7.3 (videre forskning) der vi foreslår å utvide modellen med vær og kampanjer.
*   **`ch01-sec05-mange-variabler`** — LightGBM for etterspørselsprognoser.
    *   **Hvorfor:** Alternativ ML-tilnærming brukt som referansepunkt i §5.2 for å begrunne hvorfor SARIMA er valgt fremfor en ren ML-modell på vårt datavolum.

### 6.2 Lagerstyring og Newsvendor (kjernepensum for FS3)
*   **`ch02-sec03-multi-qr`** — Multi-produkt (Q,R) med delt kapasitet.
    *   **Hvorfor:** Skoringen har 10–15 SKU per modell som deler hylleplass; pensumets multi-produkt-formulering er rammeverket for §3.3.
*   **`ch02-sec04-flerlokasjon-stokastisk`** — Flerlokasjon stokastisk programmering.
    *   **Hvorfor:** Sentralt for hele oppgaven – butikk + eksternt lager er to lokasjoner under usikker etterspørsel. Pensum gir det formelle grunnlaget for vurderingen i §1.4 og §6.1.
*   **`ch02-sec05-ml-klassifisering`** — Data-driven inventory classification.
    *   **Hvorfor:** Underbygger §2.5 (hybrid mellom modell og menneskelig skjønn) og forklarer hvorfor ABC/XYZ-tenkning kompletterer SARIMA.
*   **`ch05-sec05-newsvendor-kontrakter`** — Newsvendor og kontraktstruktur.
    *   **Hvorfor:** Direkte metodisk grunnlag for §3.3 (kritisk ratio $C_u/(C_u+C_o)$) og knytter §6.2 (leverandørsamarbeid) til kontraktsteori.

### 6.3 Forsyningskjede og koordinering
*   **`ch05-sec03-bullwhip-simulering`** — Bullwhip-effekt og batching.
    *   **Hvorfor:** Empirisk underlag for §2.4 og §6.2; pensumets simulering viser hvordan månedlige bestillinger demper Bullwhip versus sesongbatching.
*   **`ch05-sec04-multi-echelon`** — Multi-echelon lagerstyring (Clark-Scarf).
    *   **Hvorfor:** Rammeverk for §6.2 der vi argumenterer for å dele SARIMA-prognoser oppstrøms til leverandøren.

### 6.4 Lagerstrategi og fasiliteter
*   **`ch04-sec03-fasilitetsplassering`** (UFLP) — Uncapacitated Facility Location.
    *   **Hvorfor:** Teoretisk grunnlag i §1.4/§6.1 for spørsmålet "skal det eksterne lageret eksistere?".
*   **`ch07-sec03-slotting`** — Class-based storage / slotting.
    *   **Hvorfor:** Relevant for §5.3 og §6.1 når vi diskuterer hvordan den frigjorte 65 % lagerkapasiteten kan utnyttes.
*   **`ch07-sec05-integrert-lager`** — Integrert lagerplanlegging.
    *   **Hvorfor:** Underbygger tesen i §6.1 om at informasjon kan substituere fysisk areal.

### 6.5 Bærekraft (kapittel 6.3 og 7)
*   **`ch08-sec05-gronn-sc`** — Integrert grønn forsyningskjede.
    *   **Hvorfor:** Faglig forankring av §2.6 og §6.3 (FNs bærekraftsmål 12, redusert transport og svinn).

### 6.6 Returlogistikk (metodisk i §4.3)
*   **`ch09-sec03-revers-nettverk`** — Reverse-nettverksdesign.
    *   **Hvorfor:** Konseptuelt grunnlag for retur-aggregeringen i §4.3.
*   **`ch09-sec04-weibull-retur`** — Returprognoser med Weibull-levetid.
    *   **Hvorfor:** Forklarer hvorfor vi behandler returer som negative salg fremfor en separat tidsserie i denne studien (datavolumet er for lavt for Weibull).
*   **`ch09-sec05-disposisjon-tre`** — Disposisjonsbeslutning for returnerte produkter.
    *   **Hvorfor:** Knyttes til §3.3 ($C_o$ for ukurans) – pensum kvantifiserer kostnaden ved å beholde vs. selge ut sesongvarer.

### 6.7 Innkjøp
*   **`ch03-sec05-mrp-lotstorrelse`** — MRP med lotstørrelse.
    *   **Hvorfor:** Underbygger §2.3 (JIT vs. batching) og §5.3 (bestillingsfrekvens).
*   **`ch10-sec03-leverandorvalg`** — AHP + TOPSIS for leverandørvalg.
    *   **Hvorfor:** Grunnlag for §6.2/§7.2 om hvordan butikken kan vurdere leverandører med kortere ledetid mot dagens "Skoringen sentralt"-avtale.
*   **`ch10-sec04-kvantumsrabatt`** — Quantity Discount EOQ.
    *   **Hvorfor:** Brukes i §2.2 for kritisk drøfting av EOQ; rabatt-utvidelsen viser at selv klassisk EOQ må modifiseres betydelig for å passe sko-konteksten.

### 6.8 Risiko og robusthet
*   **`ch11-sec03-monte-carlo-risk`** — Monte Carlo-risikoanalyse.
    *   **Hvorfor:** Grunnlag for §7.3 om sensitivitetsanalyse av MAPE-estimatene.
*   **`ch11-sec04-robust-opt`** — Robust optimization (minimax regret).
    *   **Hvorfor:** Foreslås i §7.3 som metode for innkjøpsbeslutninger når SARIMA-prognosen har høy varians.
*   **`ch11-sec05-stresstest`** — Stresstest av forsyningskjede.
    *   **Hvorfor:** Forankrer §7.3 om hvordan modellen oppfører seg under sjokk (lockdown, leverandørbrudd).
