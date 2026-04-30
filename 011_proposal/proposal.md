# Prosjektforslag – LOG650 Gruppe 4.5

## Gruppemedlemmer
- Gustavo Alfonso Holmedal (prosjektleder)
- Thuy Thu Thi Tran
- Inger Irgesund

## Casebedrift
Skoringen Råholt – fysisk skobutikk i Eidsvoll kommune, en del av Skoringen-kjeden. Daglig leder Marit Stoksflod har gjort tilgjengelig reelle salgs- og lagerdata. Det benyttes ingen syntetiske eller hypotetiske data.

## Bransjekontekst og bestillingsregime
Skobransjen er preget av et stramt sesongregime. Skoringen plasserer **to bestillinger pr år** (vår og høst) hos sine leverandører. Dette er en bransjebetingelse, ikke en valgfri praksis, og er drevet av leverandørenes produksjonssykluser og kvantumsrabattstrukturer (jf. Ch10 §4 i pensum). Optimaliseringen ligger derfor ikke i å øke bestillingsfrekvensen, men i å treffe **riktig mengde** ved hver av de to bestillingene.

## Problemstilling
Hovedproblemstilling:

> *"Hvordan kan SARIMA-baserte etterspørselsprognoser kombinert med newsvendor-logikk forbedre Skoringen Råholts sesongbestillinger sammenlignet med dagens praksis basert på fjorårssalg?"*

Forskningsspørsmål:
- **FS1 (Teknisk):** Hvordan transformere ustrukturerte daglige salgsrapporter (PDF) til et reproduserbart datagrunnlag for kvantitativ analyse?
- **FS2 (Statistisk):** Hvilken prognosemodell – ARIMA, SARIMA eller ETS – gir best treffsikkerhet på Skoringens månedssalg?
- **FS3 (Beslutningsteoretisk):** Hva er den optimale sesongbestillingsmengden $Q^*$ etter newsvendor-modellen, og hvordan er løsningen avhengig av servicenivå og prisparametere?
- **FS4 (Økonomisk):** Hvilken estimert årlig effekt på bruttoresultat og tapt salg har en overgang fra naiv bestilling til newsvendor-bestilling?

## Datagrunnlag
- Daglige salgsrapporter (PDF) fra kassesystemet, ca. 1 100 filer for 2023–2025.
- Månedsrapporter (PDF) for 2023, 2024 og 2025.
- Lagerinformasjon og praktiske rammebetingelser oppgitt av butikkeier.
- Pipeline i Python (`pdfplumber`) konverterer rådata til strukturerte CSV-filer.

## Metode
Studien er en kvantitativ casestudie med deduktiv tilnærming. Vi anvender etablert teori fra pensum (SARIMA, newsvendor, bullwhip) på empiri fra én bedrift.

- **Prognose:** SARIMA $(p,d,q)\times(P,D,Q)_{12}$ estimert med automatisert AIC-basert grid-søk i `statsmodels`. Sammenlignes mot ARIMA, ETS og naiv baseline.
- **Bestilling:** Newsvendor-modell (Ch05 §5): $Q^* = \mu + z_\alpha \cdot \sigma$ der $\mu$ er sesongprognose og $\sigma$ er prognoseens RMSE skalert til sesonglengde.
- **Validering:** Out-of-sample test på 2025-data; ADF-test for stasjonæritet; Ljung-Box for residualanalyse.

## Beslutningsvariabler
- Sesongbestillingsmengde for vår (mar–aug) og høst (sep–feb).
- Servicenivå (gjennom valg av kritisk forhold).

## Mål
Hovedmål: Levere en kvantitativ beslutningsmodell som reduserer både tapt salg og overlager hos Skoringen Råholt, basert på reelle data og pensumforankret metodikk.

Forventet utkomme: Estimert reduksjon i alternativkostnader knyttet til tapt salg og overlager, samt et reproduserbart verktøy for fremtidige sesongbestillinger.

## Avgrensninger
Studien begrenses til:
- Produktkategorien sko (én aggregert SKU).
- Én butikk (Skoringen Råholt).
- Kvantitativ etterspørselsanalyse på månedsnivå.

Studien modellerer ikke:
- Størrelsesfordeling pr modell.
- Leverandørspesifikke leveringstider og valutarisiko.
- Eksterne faktorer som vær og kampanjer (henvises til videre arbeid med ARIMAX, Ch01 §4).
- Detaljerte transportkostnader eller intern logistikk.

## Forutsetninger
- Tilgang til historiske salgs- og lagerdata fra Skoringen Råholt.
- Antatte enhetspriser ($p, w, s$) brukes som estimat og er gjenstand for sensitivitetsanalyse.
- Tidsrammen er begrenset til vårsemesteret 2026; endelig levering 31. mai 2026.
- Ingen direkte økonomiske kostnader; ressursbruk måles i arbeidstimer.

## Pensum og litteraturforankring
Etter veiledning fra foreleser (april 2026) brukes LOG650-kompendiet *Kvantitative metoder i logistikk* (Rekdal, 2026) som arbeidsredskap – særlig Ch01 §3 (SARIMA), Ch05 §5 (Newsvendor) og Ch05 §3 (Bullwhip) – mens metodevalg siteres til etablerte primærverk: Pinedo (2016, scheduling), Hartmann & Briskorn (2010, RCPSP), Vose (2008, Monte Carlo-risiko), Efron & Tibshirani (1993, bootstrap) og Puchinger & Raidl (2005, hybride metoder). Full litteraturliste i hovedrapportens kapittel 7 og i `003_referanser/AKADEMISK_LITTERATUR.md`.
