import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano-test for like prognosepresisjon mellom modell 1 og 2.

    H0: E[d_t] = 0 (modellene er like presise),  d_t = |e1_t|^power - |e2_t|^power.
    Returnerer (DM-statistikk, tosidig p-verdi). Bruker Harvey-Leybourne-Newbold
    små-utvalg-korreksjon (Harvey, Leybourne & Newbold, 1997).
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    mean_d = float(np.mean(d))
    # Newey-West-type lang-løps-varians med h-1 lag
    gamma0 = float(np.mean((d - mean_d) ** 2))
    var_d = gamma0
    for k in range(1, h):
        gamma_k = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        var_d += 2 * gamma_k
    if var_d <= 0:
        return float("nan"), float("nan")
    dm = mean_d / np.sqrt(var_d / n)
    # Harvey-Leybourne-Newbold-korreksjon
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * dm
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(hln), df=n - 1))
    return float(hln), float(p_value)


def run_forecasting():
    file_path = "004_data/skoringen_monthly_clean.csv"
    df = pd.read_csv(file_path)
    
    # Lag en dato-indeks
    df['Dato'] = pd.to_datetime(df['År'].astype(str) + '-' + df['Måned'].astype(str) + '-01')
    df = df.set_index('Dato')
    df = df.sort_index()
    
    # Vi fokuserer på 'Antall_par' som er den viktigste variabelen for lagerstyring
    series = df['Antall_par']
    
    # Del i trening (første 24 mnd) og test (siste 12 mnd)
    train = series.iloc[:24]
    test = series.iloc[24:]
    
    print(f"Trener på data fra {train.index.min()} til {train.index.max()}")
    print(f"Tester på data fra {test.index.min()} til {test.index.max()}")
    
    results = {}

    # 1. ETS Modell (Error, Trend, Seasonal)
    try:
        # Vi antar additiv sesongvariasjon siden vi har få dataår
        ets_model = ETSModel(train, error='add', trend='add', seasonal='add', seasonal_periods=12)
        ets_fit = ets_model.fit()
        ets_pred = ets_fit.forecast(len(test))
        results['ETS'] = ets_pred
    except Exception as e:
        print(f"ETS feilet: {e}")

    # 2. ARIMA Modell (p,d,q)
    try:
        # Enkel ARIMA (1,1,1) som utgangspunkt
        arima_model = ARIMA(train, order=(1,1,1))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(len(test))
        results['ARIMA'] = arima_pred
    except Exception as e:
        print(f"ARIMA feilet: {e}")

    # 3. SARIMA Modell (p,d,q)(P,D,Q,s)
    try:
        # Vi tester et utvalg av parametere for å finne den beste modellen
        # I en full analyse ville vi brukt auto_arima, men her viser vi valget manuelt
        best_mae = float('inf')
        best_order = (1,1,1)
        best_seasonal = (1,1,1,12)
        
        # Vi bruker en robust SARIMA-konfigurasjon som er kjent for å fungere på retail-data
        sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), 
                              enforce_stationarity=False, 
                              enforce_invertibility=False)
        sarima_fit = sarima_model.fit(disp=False)
        sarima_pred = sarima_fit.get_forecast(steps=len(test))
        sarima_mean = sarima_pred.predicted_mean
        sarima_conf = sarima_pred.conf_int()

        results['SARIMA'] = sarima_mean
        results['SARIMA_CONF'] = sarima_conf  # konfidensintervall for visualisering

        # In-sample residualer fra treningsperioden (2023-2024). Disse, og IKKE
        # test-residualene, brukes som sigma_mnd i sesongnewsvendor – ellers
        # ville vi brukt 2025-fasiten til å vurdere økonomien for 2025.
        # Hopp over de første 13 punktene som er påvirket av d=1,D=1-init.
        insample_resid = sarima_fit.resid.iloc[13:]
        sarima_insample_rmse = float(np.sqrt(np.mean(insample_resid ** 2)))
        results['SARIMA_INSAMPLE_RMSE'] = sarima_insample_rmse
    except Exception as e:
        print(f"SARIMA feilet: {e}")
        sarima_insample_rmse = None

    # Evaluering og Sikkerhetslager-beregning
    print("\nModell Evaluering (MAE - Mean Absolute Error):")
    metrics = {}
    for name, pred in results.items():
        if '_CONF' in name or 'INSAMPLE' in name:
            continue
        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
        print(f"{name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        metrics[name] = {"MAE": float(mae), "RMSE": float(rmse)}

    # Naiv baseline: salg samme måned i fjor (representerer dagens praksis)
    naiv_pred = []
    for d in test.index:
        prev = d - pd.DateOffset(years=1)
        if prev in train.index:
            naiv_pred.append(train.loc[prev])
        else:
            naiv_pred.append(train.mean())
    naiv_pred = pd.Series(naiv_pred, index=test.index)
    mae_naiv = mean_absolute_error(test, naiv_pred)
    rmse_naiv = np.sqrt(mean_squared_error(test, naiv_pred))
    print(f"Naiv:   MAE = {mae_naiv:.2f}, RMSE = {rmse_naiv:.2f}")
    metrics['Naiv'] = {"MAE": float(mae_naiv), "RMSE": float(rmse_naiv)}

    # Diebold-Mariano-test: SARIMA vs Naiv. Vi bruker h=1 (klassisk 1-stegs-DM)
    # fordi testhorisonten er kort (n=12). En multi-step-variant med h=12 ville
    # krevd en lang-løps-varians basert på 11 autokovarianser fra kun 12 obs,
    # som ikke gir meningsfullt varianseestimat. Power=2 (kvadratisk tap).
    dm_stat, dm_p = (float("nan"), float("nan"))
    if 'SARIMA' in results:
        e_sarima = (test.values - results['SARIMA'].values).astype(float)
        e_naiv = (test.values - naiv_pred.values).astype(float)
        dm_stat, dm_p = diebold_mariano(e_sarima, e_naiv, h=1, power=2)
        print(f"\nDiebold-Mariano (SARIMA vs Naiv, h=1, kvadratisk tap):")
        print(f"  DM-stat = {dm_stat:.3f},  tosidig p-verdi = {dm_p:.3f}")
        print("  (negativ DM-stat => SARIMA har lavere forventet tap enn Naiv)")

        # Tilleggstest med absolutt tap (mer robust for kort serie)
        dm_stat_abs, dm_p_abs = diebold_mariano(e_sarima, e_naiv, h=1, power=1)
        print(f"Diebold-Mariano (SARIMA vs Naiv, h=1, absolutt tap):")
        print(f"  DM-stat = {dm_stat_abs:.3f},  tosidig p-verdi = {dm_p_abs:.3f}")

    # Sikkerhetslager-tall: bruk in-sample-RMSE fra treningsperioden, ikke
    # test-RMSE. Dette unngår sirkulariteten "estimere usikkerhet på 2025 og
    # deretter vurdere økonomien på 2025" som selvevalueringen påpekte.
    if 'SARIMA_INSAMPLE_RMSE' in results:
        sigma_train = results['SARIMA_INSAMPLE_RMSE']
        print(f"\nIn-sample RMSE (2023-2024-residualer, brukt som sigma_mnd): {sigma_train:.1f} par")
        print(f"Anbefalt sikkerhetslager ved 95 % servicenivå: {1.65 * sigma_train:.0f} par")
        metrics['SARIMA_INSAMPLE_RMSE'] = sigma_train

    # Lagre metrikker til JSON så rapport og sesongnewsvendor leser konsistente tall
    Path("013_gjennomforing").mkdir(exist_ok=True)
    out_metrics = {
        "modeller": metrics,
        "diebold_mariano_sarima_vs_naiv": {
            "h": 1,
            "tap_kvadratisk": {"dm_stat": dm_stat, "p_verdi_tosidig": dm_p},
            "tap_absolutt": {
                "dm_stat": dm_stat_abs if 'SARIMA' in results else float("nan"),
                "p_verdi_tosidig": dm_p_abs if 'SARIMA' in results else float("nan"),
            },
        },
    }
    with open("013_gjennomforing/forecast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)
    print("Metrikker lagret: 013_gjennomforing/forecast_metrics.json")

    # Visualisering
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Trening (Historisk)', color='black')
    plt.plot(test.index, test, label='Faktisk (Test)', color='blue', marker='o')
    
    colors = {'ETS': 'red', 'ARIMA': 'green', 'SARIMA': 'orange'}
    for name, pred in results.items():
        if name not in colors:
            continue  # hopp over konfidensintervaller og hjelpestrukturer
        plt.plot(test.index, pred, label=f'Prognose {name}', color=colors[name], linestyle='--')
        
    plt.title('Etterspørselsprognose for Skoringen Råholt (Antall par sko)')
    plt.xlabel('Dato')
    plt.ylabel('Antall par solgt')
    plt.legend()
    plt.grid(True)
    
    # Lagre grafen
    output_img = "013_gjennomforing/visuals/demand_forecast_comparison.png"
    plt.savefig(output_img)
    print(f"\nGraf lagret i: {output_img}")
    
    # Lagre resultatene til CSV for senere bruk i optimalisering
    test_results = pd.DataFrame({'Faktisk': test})
    for name, pred in results.items():
        if name not in colors:
            continue
        test_results[name] = pred

    test_results.to_csv("004_data/forecast_results.csv")
    print("Detaljerte resultater lagret i 004_data/forecast_results.csv")

if __name__ == "__main__":
    run_forecasting()
