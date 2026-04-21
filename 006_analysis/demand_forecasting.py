import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_forecasting():
    file_path = "004 data/skoringen_monthly_clean.csv"
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
        results['SARIMA_CONF'] = sarima_conf # Lagre konfidensintervall for visualisering
    except Exception as e:
        print(f"SARIMA feilet: {e}")

    # Evaluering og Sikkerhetslager-beregning
    print("\nModell Evaluering (MAE - Mean Absolute Error):")
    for name, pred in results.items():
        if '_CONF' in name: continue
        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
        print(f"{name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        
        # Beregn anbefalt sikkerhetslager (Z * RMSE) for 95% servicenivå
        if name == 'SARIMA':
            safety_stock = 1.65 * rmse
            print(f"\nAnbefalt sikkerhetslager basert på SARIMA-usikkerhet: {safety_stock:.0f} par")

    # Visualisering
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Trening (Historisk)', color='black')
    plt.plot(test.index, test, label='Faktisk (Test)', color='blue', marker='o')
    
    colors = {'ETS': 'red', 'ARIMA': 'green', 'SARIMA': 'orange'}
    for name, pred in results.items():
        plt.plot(test.index, pred, label=f'Prognose {name}', color=colors[name], linestyle='--')
        
    plt.title('Etterspørselsprognose for Skoringen Råholt (Antall par sko)')
    plt.xlabel('Dato')
    plt.ylabel('Antall par solgt')
    plt.legend()
    plt.grid(True)
    
    # Lagre grafen
    output_img = "013 fase 3 - gjennomføring/visuals/demand_forecast_comparison.png"
    plt.savefig(output_img)
    print(f"\nGraf lagret i: {output_img}")
    
    # Lagre resultatene til CSV for senere bruk i optimalisering
    test_results = pd.DataFrame({'Faktisk': test})
    for name, pred in results.items():
        test_results[name] = pred
    
    test_results.to_csv("004 data/forecast_results.csv")
    print("Detaljerte resultater lagret i 004 data/forecast_results.csv")

if __name__ == "__main__":
    run_forecasting()
