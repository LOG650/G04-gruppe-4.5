import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_optimization():
    # Last inn prognoser og historiske data (vi trenger 2024 data for Naiv strategi)
    monthly_df = pd.read_csv("004 data/skoringen_monthly_clean.csv")
    monthly_df['Dato'] = pd.to_datetime(monthly_df['År'].astype(str) + '-' + monthly_df['Måned'].astype(str) + '-01')
    monthly_df = monthly_df.set_index('Dato').sort_index()
    
    forecast_df = pd.read_csv("004 data/forecast_results.csv", index_col=0)
    forecast_df.index = pd.to_datetime(forecast_df.index)
    
    # Parametere
    HOLDING_COST = 6 # NOK per par per måned
    SHORTAGE_COST = 300 # Tap per par (tapt salg)
    ORDER_SETUP_COST = 5000 # Fast kostnad per bestilling (frakt, adm)
    INITIAL_INVENTORY = 500
    MAX_CAPACITY = 3000
    
    actual_2025 = forecast_df['Faktisk']
    sarima_forecast = forecast_df['SARIMA']
    
    # Finn salg fra 12 måneder tilbake for hver måned i 2025
    def get_naive_order(date):
        prev_year_date = date - pd.DateOffset(years=1)
        if prev_year_date in monthly_df.index:
            return monthly_df.loc[prev_year_date, 'Antall_par']
        return monthly_df['Antall_par'].mean() # Backup

    history = {'Naive': [], 'Optimized': []}
    costs = {'Naive': 0, 'Optimized': 0}
    
    # --- SIMULERING NAIV ---
    inv = INITIAL_INVENTORY
    for date, actual in actual_2025.items():
        order = get_naive_order(date)
        inv += order
        if inv > MAX_CAPACITY: inv = MAX_CAPACITY
        
        sold = min(inv, actual)
        shortage = max(0, actual - inv)
        inv -= sold
        
        costs['Naive'] += (inv * HOLDING_COST) + (shortage * SHORTAGE_COST) + (ORDER_SETUP_COST if order > 0 else 0)
        history['Naive'].append(inv)

    # --- SIMULERING OPTIMALISERT (SARIMA + Just-in-time) ---
    inv = INITIAL_INVENTORY
    for date, actual in actual_2025.items():
        # Vi bestiller basert på SARIMA-prognosen
        # Vi prøver å lande på et sluttsalgslager på 100 par (buffer)
        target_end_inv = 100 
        order = max(0, sarima_forecast.loc[date] + target_end_inv - inv)
        
        inv += order
        if inv > MAX_CAPACITY: inv = MAX_CAPACITY
        
        sold = min(inv, actual)
        shortage = max(0, actual - inv)
        inv -= sold
        
        costs['Optimized'] += (inv * HOLDING_COST) + (shortage * SHORTAGE_COST) + (ORDER_SETUP_COST if order > 0 else 0)
        history['Optimized'].append(inv)

    print(f"Total Kostnad (Naiv - fjorårets salg): {costs['Naive']:,.0f} NOK")
    print(f"Total Kostnad (KI-Optimert - SARIMA): {costs['Optimized']:,.0f} NOK")
    savings = costs['Naive'] - costs['Optimized']
    print(f"Besparelse: {savings:,.0f} NOK ({savings/costs['Naive']*100:.1f}%)")

    # Visualisering
    plt.figure(figsize=(12, 6))
    plt.plot(actual_2025.index, history['Naive'], label='Lager (Naiv: Fjorår)', color='red', linestyle='--')
    plt.plot(actual_2025.index, history['Optimized'], label='Lager (Optimert: SARIMA)', color='green')
    plt.title('Sammenligning av Lagerstrategier (2025)')
    plt.ylabel('Antall par på lager')
    plt.legend()
    plt.grid(True)
    plt.savefig("013 fase 3 - gjennomføring/visuals/inventory_comparison_v2.png")
    
if __name__ == "__main__":
    run_optimization()
