# Valideringsrapport: Statistisk integritet og modelltesting

Dette dokumentet samler de statistiske testene som er utført for å sikre at SARIMA-modellen og den påfølgende newsvendor-bestillingen er valide og pålitelige. Tallene er beregnet direkte fra `004_data/skoringen_monthly_clean.csv` og `004_data/forecast_results.csv` av skriptet `006_analysis/verify_numbers.py`.

## 1. Stasjonæritet (Augmented Dickey-Fuller)
ARIMA-baserte modeller forutsetter at tidsserien er stasjonær. ADF-testen ble kjørt på rådataene og på den sesongmessig differensierte serien.

| Serie | p-verdi | Konklusjon |
|---|---:|---|
| Rådata (`Antall_par`) | 0,82 | Ikke stasjonær |
| Etter sesongmessig differensiering ($d=1, D=1$) | 0,002 | Stasjonær |

Differensieringen er dermed nødvendig og tilstrekkelig før modellestimering.

## 2. Restanalyse (Ljung-Box og normalfordeling)
Etter at SARIMA(1,1,1)(1,1,1)$_{12}$ er estimert på 2023–2024-data, undersøkes residualene fra in-sample-fittet:

- **Normalfordeling:** Histogrammet er tilnærmet normalfordelt rundt null. Skewness og kurtosis er innenfor akseptable grenser ($|skew|<1$, $|excess\ kurt|<3$).
- **Autokorrelasjon (Ljung-Box):** p-verdi $> 0{,}05$ for typiske lag-valg (12, 24). Vi forkaster ikke nullhypotesen om at residualene er hvit støy.

Konklusjon: Modellen er tilstrekkelig spesifisert. Residualenes RMSE kan dermed brukes som proxy for prognoseusikkerheten $\sigma$ i newsvendor-formelen.

## 3. Out-of-sample test (2025)
Modellen ble trent på 2023–2024 og predikerte hele 2025 uten tilgang til fasit. Resultatene fra `verify_numbers.py`:

| Modell | MAE (par) | RMSE | MAPE |
|---|---:|---:|---:|
| **SARIMA(1,1,1)(1,1,1)$_{12}$** | **137,18** | 222,22 | 17,35 % |
| Naiv (samme måned i fjor) | 162,83 | 197,33 | 18,96 % |
| ETS (Holt-Winters, additiv) | 177,20 | 231,28 | 20,28 % |
| ARIMA(1,1,1) – uten sesong | 228,73 | 277,73 | 24,46 % |

På årsbasis: faktisk salg 2025 var 10 800 par, SARIMA-prognosen 10 530 par – avvik på 2,5 %.

## 4. Newsvendor-bestilling
Med antatte enhetspriser $p = 1\,200$, $w = 600$, $s = 400$ NOK/par:
- Kritisk forhold: $\text{CR} = (p-w)/(p-s) = 0{,}75$
- $z_\alpha = \Phi^{-1}(0{,}75) = 0{,}6745$
- Sesongstandardavvik: $\sigma_i = 222{,}2 \cdot \sqrt{6} = 544{,}3$ par
- Sikkerhetslager pr sesong: $z_\alpha \cdot \sigma_i = 367$ par

| Sesong 2025 | $\mu_{\text{SARIMA}}$ | $Q^*$ | $Q_{\text{naiv}}$ | Faktisk |
|---|---:|---:|---:|---:|
| Vår (mar–aug) | 5 663 | 6 030 | 5 389 | 6 000 |
| Høst (sep–feb) | 3 201 | 3 568 | 4 120 | 3 657 (sep–des, 4 mnd registrert) |

## 5. Sensitivitetsanalyse
$Q^*$ for vårsesongen 2025 ble testet mot variasjoner i prisparametrene:

| Scenario | $p$ | $w$ | $s$ | CR | $z_\alpha$ | $Q^*$ vår |
|---|---:|---:|---:|---:|---:|---:|
| Lav margin | 1 000 | 600 | 400 | 0,667 | 0,43 | 5 897 |
| Basisscenario | 1 200 | 600 | 400 | 0,750 | 0,67 | 6 030 |
| Høy margin | 1 500 | 600 | 400 | 0,818 | 0,91 | 6 157 |
| Lav restverdi | 1 200 | 600 | 200 | 0,600 | 0,25 | 5 801 |
| Høy restverdi | 1 200 | 600 | 550 | 0,923 | 1,43 | 6 439 |

Rangeringen newsvendor > naiv beholdes i alle scenarier, og $Q^*$ varierer under ±10 % over rimelige scenarioer. Mest sensitiv er restverdien $s$.

## 6. Begrensninger
- Prognoseusikkerheten ($\sigma_{\text{mnd}} = 222{,}2$) er estimert fra ett års out-of-sample-data. Dette kan være en optimistisk eller pessimistisk anslag avhengig av om 2025 var et "rolig" år.
- Newsvendor antar normalfordelt etterspørsel. Ved kraftig skjevhet bør empiriske kvantiler brukes i stedet for $z_\alpha$.
- Enhetsprisene er estimat. Reell kalkyle krever transaksjonsdata fra Skoringens regnskap.
