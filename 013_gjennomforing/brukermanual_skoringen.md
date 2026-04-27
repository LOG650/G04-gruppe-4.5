# Brukermanual: Sesongbestillingsverktøy for Skoringen Råholt

Dette er en kort veiledning for daglig leder om hvordan verktøyet brukes til å forberede de to årlige sesongbestillingene (vår og høst).

## 1. Klargjøring av data
Hver måned legger butikken inn de nye salgsrapportene:
1. Lagre dagsrapportene fra kassesystemet som PDF.
2. Plasser filene i mappen `004_data/raw_data/`.
3. Sørg for at filnavnet inneholder måned og år (f.eks. `Dagsalgsrapport_Januar_2026.pdf`).

## 2. Generere oppdatert prognose og bestillingsanbefaling
Før hver sesongbestilling, kjør pipelinen:
```bash
python 006_analysis/run_full_pipeline.py
python 006_analysis/sesongnewsvendor.py
```
Skriptene oppdaterer SARIMA-modellen med de nyeste salgstallene og beregner ny anbefalt $Q^*$ for kommende sesong.

## 3. Tolkning av resultatene
Resultatfiler etter kjøring:
- `013_gjennomforing/visuals/demand_forecast_comparison.png` – forventet månedssalg sammenlignet med modellens prognose.
- `013_gjennomforing/visuals/inventory_newsvendor_2025.png` – simulert lagerprofil for to bestillingsstrategier.
- `013_gjennomforing/visuals/newsvendor_profit_curve.png` – forventet profitt som funksjon av bestilt mengde.
- `013_gjennomforing/newsvendor_resultater.json` – nøkkeltall (anbefalt $Q^*$ pr sesong, sikkerhetslager, kritisk forhold).

Den primære anbefalingen er $Q^*$-verdien for kommende sesong i JSON-filen.

## 4. Bestillingsstrategi
- **Frekvens:** Beholdes på to bestillinger pr år (vår + høst), i tråd med leverandørenes regime.
- **Mengde:** Bruk $Q^*$ fra newsvendor-modellen som utgangspunkt. Modellen inkluderer allerede et sikkerhetslager basert på prognoseusikkerheten – det er derfor ikke nødvendig å legge til "litt ekstra" manuelt.
- **Servicenivå:** Verktøyet er konfigurert til 75 % servicenivå (kritisk forhold). Ved ønske om høyere servicenivå (f.eks. 90 %), juster parameterne $p$, $w$, $s$ i `sesongnewsvendor.py` eller endre $z_\alpha$ direkte.
- **Eksterne faktorer:** Modellen baserer seg på historikk og fanger ikke opp ekstreme værvarsler eller planlagte kampanjer. Ved slike kjente avvik kan modellens anbefaling justeres manuelt med ±10 %.

## 5. Loggføring
Etter hver sesong: noter avviket mellom $Q^*$, faktisk solgt og evt. justeringer som ble gjort. Disse loggene er det viktigste grunnlaget for å forbedre modellen i kommende sesonger.
