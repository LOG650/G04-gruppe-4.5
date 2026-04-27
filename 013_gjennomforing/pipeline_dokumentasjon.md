# Teknisk dokumentasjon: Prognose- og bestillingssystem

Dette dokumentet beskriver den tekniske implementeringen av modellen utviklet for Skoringen Råholt i forbindelse med bacheloroppgaven i logistikk (LOG650).

## 1. Systemarkitektur
Pipelinen er bygget som modulære Python-skript i `006_analysis/`. De fire fasene er:

1. **PDF-dekoding** – ustrukturerte daglige salgsrapporter konverteres til strukturerte CSV-er.
2. **Datavasking og aggregering** – returer, uteliggere og frekvenskonvertering.
3. **Etterspørselsprognose** – SARIMA-, ARIMA- og ETS-modeller med out-of-sample-validering.
4. **Sesongbestilling** – newsvendor-formel anvendt på sesongprognosene.

## 2. Modulbeskrivelse

### 2.1 PDF-dekoding (`pdf_to_csv_decoder.py`, `decode_monthly_reports.py`)
Bruker `pdfplumber` for å trekke ut tabellinformasjon fra Skoringens dagsrapporter.

- **Inndata:** PDF-filer i `004_data/` og `004_data/raw_data/`.
- **Identifikasjon:** Kolonnegrenser bestemmes av x-koordinater. Linjer valideres med regex (`^\d{6}` for varenummer).
- **Validering:** Sum av linjeelementer kontrolleres mot "Total salg"-feltet i bunnen av PDF-en.

### 2.2 Datavasking (`clean_sales_data.py`, `prepare_timeseries.py`)
- Returer trekkes fra netto-etterspørselen.
- Uteliggere (Z-score > 3) flagges og inspiseres manuelt før eventuell korrigering.
- Aggregering fra dagsdata til månedsdata via `pandas.resample('MS')`.
- Frekvenskonverteringen demper daglig støy og fremhever sesongsignalet.

### 2.3 Prognosemodellering (`demand_forecasting.py`)
- **Treningssett:** 2023-01 til 2024-12 (24 måneder).
- **Testsett:** 2025-01 til 2025-12 (12 måneder, out-of-sample).
- **Modeller:** SARIMA(1,1,1)(1,1,1)$_{12}$, ARIMA(1,1,1), ETS additiv, naiv "samme måned i fjor".
- **Implementering:** `statsmodels.tsa.statespace.SARIMAX` med `enforce_stationarity=False`.
- **Evaluering:** MAE, RMSE og MAPE på testsettet. Konfidensintervall hentes fra `get_forecast(steps).conf_int()`.

### 2.4 Sesongnewsvendor (`sesongnewsvendor.py`)
Implementerer newsvendor-formelen $Q^* = \mu + z_\alpha \cdot \sigma$ for to sesonger pr år.

- $\mu_i$: sum SARIMA-prognose for sesongmånedene.
- $\sigma_i = \sigma_{\text{mnd}} \cdot \sqrt{n_i}$, der $\sigma_{\text{mnd}}$ er RMSE på 2025-residualene.
- Kritisk forhold: $\text{CR} = (p-w)/(p-s)$.
- Sammenligner mot en naiv strategi der $Q_{\text{naiv}} = $ fjorårets sesongsalg.

Output:
- Lagerprofiler i `013_gjennomforing/visuals/inventory_newsvendor_2025.png`.
- Profittkurve i `013_gjennomforing/visuals/newsvendor_profit_curve.png`.
- Tallresultater i `013_gjennomforing/newsvendor_resultater.json`.

### 2.5 Verifiseringsskript (`verify_numbers.py`)
Beregner alle nøkkeltall som brukes i rapporten direkte fra rådataene, slik at tabeller og påstander i `Bacheloroppgave_Skoringen_KOMPLETT.md` kan reproduseres.

## 3. Reproduksjon
```bash
pip install -r requirements.txt
python 006_analysis/run_full_pipeline.py        # PDF -> CSV -> SARIMA
python 006_analysis/sesongnewsvendor.py         # Newsvendor-bestilling
python 006_analysis/verify_numbers.py           # Verifiser alle tall i rapporten
```

## 4. Teknisk stack
- **Språk:** Python 3.10+ (testet med 3.13)
- **Hovedbiblioteker:**
  - `pandas` – datamanipulasjon
  - `statsmodels` – statistisk modellering (SARIMAX, ETSModel, ARIMA)
  - `scipy` – statistiske fordelinger og kvantiler (newsvendor)
  - `pdfplumber` – PDF-parsing
  - `matplotlib`, `seaborn` – visualisering
  - `scikit-learn` – evalueringsmetrikker (MAE, RMSE)
