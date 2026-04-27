"""
Utvidet lagersimulering: Sammenligner faktisk bulk-praksis (to store
sesongbestillinger pr. år) mot Just-in-Time (månedlige SARIMA-baserte
bestillinger). Genererer figur og rapporterer nøkkeltall.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = "004_data"
OUT_DIR = "013_gjennomforing/visuals"
os.makedirs(OUT_DIR, exist_ok=True)

monthly = pd.read_csv(f"{DATA_DIR}/skoringen_monthly_clean.csv")
monthly['Dato'] = pd.to_datetime(monthly['År'].astype(str) + '-' + monthly['Måned'].astype(str) + '-01')
monthly = monthly.set_index('Dato').sort_index()

forecast = pd.read_csv(f"{DATA_DIR}/forecast_results.csv", index_col=0)
forecast.index = pd.to_datetime(forecast.index)

faktisk = forecast['Faktisk']
sarima = forecast['SARIMA']

HOLDING_COST = 6
SHORTAGE_COST = 300
ORDER_SETUP_COST = 5000
INITIAL_INVENTORY = 500
MAX_CAPACITY = 3000

def simulate_bulk(actual_series, forecast_series):
    """
    Bulk-strategi: to store bestillinger pr. år.
    Februar: bestiller hele behovet for vårsesongen (mar-aug) basert på prognose.
    August: bestiller hele behovet for høst/vinter (sep-feb).
    """
    inv = INITIAL_INVENTORY
    history = []
    cost = 0
    capacity_hits = 0
    n_orders = 0
    for date, actual in actual_series.items():
        order = 0
        if date.month == 2:
            spring = [date + pd.DateOffset(months=k) for k in range(1, 7)]
            order = sum(forecast_series.loc[d] for d in spring if d in forecast_series.index)
            n_orders += 1
        elif date.month == 8:
            fall = [date + pd.DateOffset(months=k) for k in range(1, 7) if (date + pd.DateOffset(months=k)) in forecast_series.index]
            order = sum(forecast_series.loc[d] for d in fall)
            n_orders += 1
        inv += order
        if inv > MAX_CAPACITY:
            capacity_hits += 1
        sold = min(inv, actual)
        shortage = max(0, actual - inv)
        inv -= sold
        cost += inv*HOLDING_COST + shortage*SHORTAGE_COST + (ORDER_SETUP_COST if order > 0 else 0)
        history.append(inv)
    return history, cost, capacity_hits, n_orders

def simulate_jit(actual_series, forecast_series):
    inv = INITIAL_INVENTORY
    history = []
    cost = 0
    capacity_hits = 0
    n_orders = 0
    for date, actual in actual_series.items():
        target = forecast_series.loc[date] + 100
        order = max(0, target - inv)
        if order > 0:
            n_orders += 1
        inv += order
        if inv > MAX_CAPACITY:
            capacity_hits += 1
        sold = min(inv, actual)
        shortage = max(0, actual - inv)
        inv -= sold
        cost += inv*HOLDING_COST + shortage*SHORTAGE_COST + (ORDER_SETUP_COST if order > 0 else 0)
        history.append(inv)
    return history, cost, capacity_hits, n_orders

bulk_hist, bulk_cost, bulk_hits, bulk_n = simulate_bulk(faktisk, sarima)
jit_hist, jit_cost, jit_hits, jit_n = simulate_jit(faktisk, sarima)

print("=== BULK (sesongbestillinger feb + aug) ===")
print(f"  Antall bestillinger:    {bulk_n}")
print(f"  Maks lager:             {max(bulk_hist):.0f} par")
print(f"  Måneder med kap.brudd:  {bulk_hits}")
print(f"  Total kostnad:          {bulk_cost:,.0f} NOK")

print("\n=== JIT (månedlige SARIMA-bestillinger) ===")
print(f"  Antall bestillinger:    {jit_n}")
print(f"  Maks lager:             {max(jit_hist):.0f} par")
print(f"  Måneder med kap.brudd:  {jit_hits}")
print(f"  Total kostnad:          {jit_cost:,.0f} NOK")

print(f"\n=== SAMMENLIGNING ===")
print(f"  Reduksjon i maks lager: {(1 - max(jit_hist)/max(bulk_hist))*100:.1f}%")
print(f"  Kostnadsbesparelse:     {bulk_cost - jit_cost:,.0f} NOK ({(bulk_cost-jit_cost)/bulk_cost*100:.1f}%)")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.plot(faktisk.index, bulk_hist, label='Bulk-strategi (feb + aug)', color='#c0392b', linewidth=2.4, marker='s')
ax.plot(faktisk.index, jit_hist,  label='JIT-strategi (månedlig)',   color='#27ae60', linewidth=2.4, marker='o')
ax.axhline(MAX_CAPACITY, color='black', linestyle='--', linewidth=1.4, label=f'Lagerkapasitet ({MAX_CAPACITY} par)')
ax.set_title('Lagernivå 2025: Bulk-bestillinger vs Just-in-Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Måned')
ax.set_ylabel('Antall par på lager')
ax.legend(loc='upper right')
plt.tight_layout()
out_path = f"{OUT_DIR}/inventory_bulk_vs_jit.png"
plt.savefig(out_path, dpi=300)
print(f"\nFigur lagret: {out_path}")
