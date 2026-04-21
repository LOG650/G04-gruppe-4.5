# Teknisk Dokumentasjon: Prognose- og Lagerstyringssystem

Dette dokumentet beskriver den tekniske implementeringen av systemet utviklet for Skoringen Råholt i forbindelse med bacheloroppgaven i logistikk (LOG650).

## 1. Systemarkitektur
Systemet er bygget som en modulær Python-pipeline som består av fire hovedfaser:
1.  **Data Extraction:** Konvertering av ustrukturerte PDF-dagsrapporter til CSV.
2.  **Preprocessing:** Rensking, validering og aggregering av salgsdata.
3.  **Forecasting:** Trening og evaluering av SARIMA- og ETS-modeller.
4.  **Optimization:** Simulering av lagernivåer basert på Newsvendor-logikk.

## 2. Modulbeskrivelse

### 2.1 PDF-Dekoding (`pdf_to_csv_decoder.py`)
Bruker `pdfplumber` for å identifisere tabellstrukturer i Skoringens dagsrapporter. 
- **Inndata:** PDF-filer fra `004 data/raw_data/`.
- **Logikk:** Identifiserer kolonner basert på horisontale koordinater. Håndterer linjeskift i varenavn og fjerner tomme rader.
- **Validering:** Sjekker at summen av linje-elementer stemmer med "Total salg" i bunnen av PDF-en.

### 2.2 Datarensing (`clean_sales_data.py` & `prepare_timeseries.py`)
Transformerer rådata til en stasjonær tidsserie.
- Fjerner interne overføringer og feilregistreringer.
- Aggregerer daglige salg til månedlige intervaller ved hjelp av `pandas.resample()`.
- Utfører sesongmessig dekomponering for å skille ut trend og sesongkomponenter.

### 2.3 Prognosemotor (`demand_forecasting.py`)
Kjernen i systemet som benytter `statsmodels` for statistisk analyse.
- **SARIMA:** Implementert med automatisert parameter-søk (Grid Search).
- **Validering:** Benytter "Time Series Cross-Validation" for å sikre at modellen ikke overfittes til historiske data.

### 2.4 Lagersimulering (`inventory_optimization.py`)
Oversetter prognoser til praktisk lagerstyring.
- Beregner optimalt sikkerhetslager ($SS$) ved formelen: $SS = Z \times \sigma_{forecast} \times \sqrt{L}$.
- Beregner re-bestillingspunkt og simulerer daglige lagernivåer gjennom et helt år.

## 3. Teknisk Stack
- **Språk:** Python 3.10+
- **Hovedbiblioteker:** 
  - `pandas` (datamanipulasjon)
  - `statsmodels` (statistisk modellering)
  - `pdfplumber` (PDF-parsing)
  - `matplotlib` / `seaborn` (visualisering)
