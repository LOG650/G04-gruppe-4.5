# Prosjektlogg: Utvikling og Gjennomføring

Denne loggen beskriver de viktigste milepælene og tekniske utfordringene i gjennomføringsfasen av prosjektet.

| Dato | Aktivitet | Utfall / Løsning |
| :--- | :--- | :--- |
| **05.03.2026** | Datainnsamling hos Skoringen Råholt | Mottok ca. 1100 PDF-filer med dagsalgsrapporter. |
| **10.03.2026** | Initial testing av PDF-parsing | **Utfordring:** Tabellene i PDF-en hadde ulik struktur fra år til år. **Løsning:** Implementerte fleksibel kolonne-deteksjon i Python. |
| **15.03.2026** | Datavasking og rensing | Identifiserte og fjernet feilregistreringer. Aggregerte data til månedlige tidsserier. |
| **20.03.2026** | Modellvalg og trening | Testet ARIMA, SARIMA og ETS. SARIMA ga best resultat på grunn av sterke sesongvariasjoner. |
| **25.03.2026** | Implementering av Newsvendor-logikk | Utviklet skript for å beregne optimalt sikkerhetslager basert på modellens usikkerhet (RMSE). |
| **01.04.2026** | Simulering av 2025-sesongen | Gjennomførte "backtesting" av modellen. Resultatene viste en potensiell kostnadsbesparelse på 26,9 %. |
| **10.04.2026** | Visualisering og rapportering | Genererte endelige grafer for lagersammenligning og prognosenøyaktighet. |
| **15.04.2026** | Ferdigstilling av dokumentasjon | Skrev teknisk dokumentasjon og brukermanual for bedriften. |

## Viktige læringspunkter
1.  **Datakvalitet:** Vi undervurderte tiden det tok å vaske ustrukturerte PDF-data. Dette var den største flaskehalsen i prosjektet.
2.  **Modellstabilitet:** SARIMA-modellen er svært sensitiv for parametervalg. Bruk av Grid Search var kritisk for å finne de optimale (p, d, q) verdiene.
3.  **Praktisk relevans:** Gjennom dialog med daglig leder forstod vi at modellen må være enkel å bruke i en travel hverdag, noe som førte til utviklingen av `run_full_pipeline.py`.
