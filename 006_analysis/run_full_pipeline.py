import os
import subprocess
import pandas as pd
import datetime

def run_pipeline():
    print("=== SKORINGEN RÅHOLT KI-OPTIMALISERING PIPELINE ===")
    
    # 1. Dekoding av månedsrapporter
    print("\n[1/4] Dekoder månedsrapporter...")
    subprocess.run(["python", "006_analysis/decode_monthly_reports.py"], check=True)
    
    # 2. Tidsseriekonvertering (valgfritt hvis månedsrapportene er nok)
    # Men vi kjører den for å ha dagsdata tilgjengelig hvis trengs
    # print("\n[2/4] Klargjør tidsserier...")
    # subprocess.run(["python", "006_analysis/prepare_timeseries.py"], check=True)
    
    # 3. Prognosetrening og prediksjon
    print("\n[2/4] Trener prognosemodeller (SARIMA/ETS/ARIMA)...")
    subprocess.run(["python", "006_analysis/demand_forecasting.py"], check=True)
    
    # 4. Optimalisering og resultatberegning
    print("\n[3/4] Beregner optimal lagerstyring...")
    subprocess.run(["python", "006_analysis/inventory_optimization.py"], check=True)
    
    # 5. Generer bestillingsanbefaling for NESTE måned
    print("\n[4/4] Genererer bestillingsanbefaling...")
    
    # Last inn de nyeste prognosene
    forecast_results = pd.read_csv("004_data/forecast_results.csv", index_col=0)
    # Her antar vi at 'neste måned' er den første måneden etter de faktiske dataene i forecast_results
    # Men i vår simulering er 2025 allerede i forecast_results.
    # La oss hente den nyeste prognosen for 'fremtiden'
    
    # For demonstrasjon, hent anbefaling for den kommende måneden (f.eks. Januar 2026 hvis 2025 er testdata)
    # I dette eksempelet bruker vi siste rad i forecast_results som eksempel på anbefaling
    last_row = forecast_results.iloc[-1]
    last_date = pd.to_datetime(forecast_results.index[-1])
    next_month = last_date + pd.DateOffset(months=1)
    
    rec_order = last_row['SARIMA'] # Enkel anbefaling
    
    print("-" * 40)
    print(f"ANBEFALING FOR {next_month.strftime('%B %Y')}:")
    print(f"Forventet etterspørsel: {rec_order:.0f} par")
    print(f"Anbefalt bestillingsmengde: {rec_order + 100:.0f} par (inkl. 100 par buffer)")
    print("-" * 40)
    
    print("\nPipeline fullført! Alle resultater og visualiseringer er oppdatert i '013_gjennomforing/visuals/'.")

if __name__ == "__main__":
    run_pipeline()
