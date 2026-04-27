# Prosjektlogg: Utvikling og Gjennomføring

Denne loggen beskriver de viktigste milepælene og tekniske utfordringene i gjennomføringsfasen av prosjektet.

| Dato | Aktivitet | Utfall / Løsning |
| :--- | :--- | :--- |
| **05.03.2026** | Datainnsamling hos Skoringen Råholt | Mottok ca. 1100 PDF-filer med dagsalgsrapporter. |
| **10.03.2026** | Initial testing av PDF-parsing | **Utfordring:** Tabellene i PDF-en hadde ulik struktur fra år til år. **Løsning:** Implementerte fleksibel kolonne-deteksjon i Python. |
| **15.03.2026** | Datavasking og rensing | Identifiserte og fjernet feilregistreringer. Aggregerte data til månedlige tidsserier. |
| **20.03.2026** | Modellvalg og trening | Testet ARIMA, SARIMA og ETS. SARIMA ga best resultat på grunn av sterke sesongvariasjoner. |
| **25.03.2026** | Implementering av Newsvendor-logikk | Utviklet skript for å beregne optimalt sikkerhetslager basert på modellens usikkerhet (RMSE). |
| **01.04.2026** | Backtesting av prognosemodellene | SARIMA gir lavest MAE (137 par/mnd, 15,8 % bedre enn naiv baseline). |
| **10.04.2026** | Visualisering og rapportering | Genererte endelige grafer for lagerprofil, sesongtrender og prognosenøyaktighet. |
| **15.04.2026** | Ferdigstilling av dokumentasjon | Skrev teknisk dokumentasjon og brukermanual for bedriften. |
| **27.04.2026** | Reformulering av optimaliseringsmodell | Etter avklaring med butikkeier: bestillingsregimet er bundet til 2 sesongbestillinger pr år. JIT-narrativet ble erstattet av newsvendor-formel for sesongbestilling, jf. pensumets Ch05 §5. Estimert årlig effekt: +713 621 NOK (+14,4 % nettoresultat). |

## Viktige læringspunkter
1. **Datakvalitet:** Vi undervurderte tiden det tok å vaske ustrukturerte PDF-data. Dette var den største flaskehalsen i prosjektet.
2. **Modellstabilitet:** SARIMA-modellen er sensitiv for parametervalg. Bruk av AIC-basert grid-søk var kritisk for å finne en robust modell.
3. **Bransjekontekst slår teori:** Vi startet med et "Just-in-Time"-narrativ basert på generell logistikkteori, men butikkens bransjebetingelse om to bestillinger pr år tvang oss til å reformulere problemet som en sesongnewsvendor. Pensumets Ch05 §5 ga den nødvendige teoretiske rammen for den nye formuleringen.
4. **Praktisk relevans:** Modellen må kunne kjøres med ett kommando i forkant av hver sesongbestilling. Dette førte til utviklingen av `run_full_pipeline.py` og `sesongnewsvendor.py` som kan kjøres etter hverandre.
