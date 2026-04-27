"""
Verifiseringsskript: Beregner alle nøkkeltall som brukes i rapporten,
slik at vi kan oppdatere teksten med korrekte tall.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = "004_data"

monthly = pd.read_csv(f"{DATA_DIR}/skoringen_monthly_clean.csv")
monthly['Dato'] = pd.to_datetime(monthly['År'].astype(str) + '-' + monthly['Måned'].astype(str) + '-01')
monthly = monthly.set_index('Dato').sort_index()

forecast = pd.read_csv(f"{DATA_DIR}/forecast_results.csv", index_col=0)
forecast.index = pd.to_datetime(forecast.index)

print("=== DATAGRUNNLAG ===")
print(f"Antall måneder totalt:        {len(monthly)}")
print(f"Periode:                       {monthly.index.min().date()} – {monthly.index.max().date()}")
print(f"Totalt antall par solgt:       {int(monthly['Antall_par'].sum()):,}")
print(f"Totalt antall par 2023:        {int(monthly.loc['2023','Antall_par'].sum()):,}")
print(f"Totalt antall par 2024:        {int(monthly.loc['2024','Antall_par'].sum()):,}")
print(f"Totalt antall par 2025:        {int(monthly.loc['2025','Antall_par'].sum()):,}")

print("\n=== PROGNOSEPRESISJON 2025 (test-vinduet) ===")
faktisk = forecast['Faktisk']
for col in ['SARIMA', 'ETS', 'ARIMA']:
    pred = forecast[col]
    mae = mean_absolute_error(faktisk, pred)
    rmse = np.sqrt(mean_squared_error(faktisk, pred))
    mape = np.mean(np.abs((faktisk - pred) / faktisk)) * 100
    print(f"{col:7s}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:5.2f}%")

# Naiv baseline: salg samme måned i fjor
naiv_pred = []
for date in faktisk.index:
    prev = date - pd.DateOffset(years=1)
    naiv_pred.append(monthly.loc[prev, 'Antall_par'] if prev in monthly.index else monthly['Antall_par'].mean())
naiv_pred = pd.Series(naiv_pred, index=faktisk.index)
mae_n = mean_absolute_error(faktisk, naiv_pred)
rmse_n = np.sqrt(mean_squared_error(faktisk, naiv_pred))
mape_n = np.mean(np.abs((faktisk - naiv_pred) / faktisk)) * 100
print(f"{'Naiv':7s}  MAE={mae_n:7.2f}  RMSE={rmse_n:7.2f}  MAPE={mape_n:5.2f}%")

# Forbedring SARIMA vs Naiv
forbedring_mae = (mae_n - mean_absolute_error(faktisk, forecast['SARIMA'])) / mae_n * 100
print(f"\nSARIMA forbedrer MAE med {forbedring_mae:.1f}% mot Naiv baseline")

print("\n=== ÅRSPROGNOSE 2025 (sum) ===")
print(f"Faktisk sum 2025:    {int(faktisk.sum()):,} par")
print(f"SARIMA sum 2025:     {int(forecast['SARIMA'].sum()):,} par")
print(f"Avvik:               {abs(faktisk.sum() - forecast['SARIMA'].sum()) / faktisk.sum() * 100:.1f}%")

print("\n=== LAGERSIMULERING 2025 ===")
HOLDING_COST = 6
SHORTAGE_COST = 300
ORDER_SETUP_COST = 5000
INITIAL_INVENTORY = 500
MAX_CAPACITY = 3000

actual_2025 = faktisk
sarima_forecast = forecast['SARIMA']

def simulate(orders_strategy):
    inv = INITIAL_INVENTORY
    history = []
    cost = 0
    capacity_hits = 0
    for date, actual in actual_2025.items():
        order = orders_strategy(date, inv)
        inv += order
        if inv > MAX_CAPACITY:
            capacity_hits += 1
            inv = MAX_CAPACITY
        sold = min(inv, actual)
        shortage = max(0, actual - inv)
        inv -= sold
        cost += inv*HOLDING_COST + shortage*SHORTAGE_COST + (ORDER_SETUP_COST if order > 0 else 0)
        history.append(inv)
    return history, cost, capacity_hits

# Naiv: bestiller fjorårets salg samme måned
def naiv_order(date, inv):
    prev = date - pd.DateOffset(years=1)
    return monthly.loc[prev, 'Antall_par'] if prev in monthly.index else monthly['Antall_par'].mean()

# JIT: bestiller (SARIMA-prognose + buffer) - dagens lager
def jit_order(date, inv):
    target = sarima_forecast.loc[date] + 100
    return max(0, target - inv)

hist_n, cost_n, hits_n = simulate(naiv_order)
hist_o, cost_o, hits_o = simulate(jit_order)

print(f"Naiv  – Maks lager: {max(hist_n):.0f} par   Kostnad: {cost_n:,.0f} NOK   Kap.brudd: {hits_n} mnd")
print(f"JIT   – Maks lager: {max(hist_o):.0f} par   Kostnad: {cost_o:,.0f} NOK   Kap.brudd: {hits_o} mnd")
print(f"Besparelse: {cost_n - cost_o:,.0f} NOK ({(cost_n-cost_o)/cost_n*100:.1f}%)")

print("\n=== SESONGMØNSTER (gjennomsnitt 2023-2025) ===")
mnd_avg = monthly.groupby(monthly.index.month)['Antall_par'].mean().round(0)
navn = ['Jan','Feb','Mar','Apr','Mai','Jun','Jul','Aug','Sep','Okt','Nov','Des']
for i, m in enumerate(mnd_avg, start=1):
    print(f"  {navn[i-1]}: {int(m):4d} par")
print(f"\nHøyeste mnd: {navn[mnd_avg.idxmax()-1]} ({int(mnd_avg.max())} par)")
print(f"Laveste mnd: {navn[mnd_avg.idxmin()-1]} ({int(mnd_avg.min())} par)")
print(f"Forhold høy/lav: {mnd_avg.max()/mnd_avg.min():.2f}x")
