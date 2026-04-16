import pandas as pd
import numpy as np

def prepare_timeseries():
    input_path = "004 data/skoringen_salgsdata_clean.csv"
    output_path = "004 data/skoringen_sales_timeseries.csv"
    
    df = pd.read_csv(input_path)
    
    # 1. Konverter Måned til nummer
    month_map = {
        'Januar': 1, 'Februar': 2, 'Mars': 3, 'April': 4,
        'Mai': 5, 'Juni': 6, 'Juli': 7, 'August': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }
    df['Måned_Nr'] = df['Måned'].map(month_map)
    
    # 2. Forsøk å lage en korrekt dato-kolonne
    # Vi må være forsiktige her fordi 'Dag' kan inneholde feilverdier fra PDF-en
    def create_date(row):
        try:
            # Sjekk om dag er gyldig (1-31)
            day = int(row['Dag'])
            if 1 <= day <= 31:
                return pd.Timestamp(year=int(row['År']), month=int(row['Måned_Nr']), day=day)
            return pd.NaT
        except:
            return pd.NaT

    df['Dato'] = df.apply(create_date, axis=1)
    
    # Fjern rader uten gyldig dato
    df = df.dropna(subset=['Dato'])
    
    # 3. Rens data for outliers
    # En liten skobutikk selger neppe 1000+ par på en vanlig dag (basert på snittet i dataene)
    # Vi setter en konservativ grense på 200 par per dag og 100.000 i omsetning
    df = df[df['Antall_par'] < 200]
    df = df[df['Omsetning_total'] < 100000]
    df = df[df['Antall_par'] > 0]
    df = df[df['Omsetning_total'] > 0]
    
    # Sorter etter dato
    df = df.sort_values('Dato')
    
    # 4. Aggreger til ukesnivå for å få mer stabile trender for ETS/ARIMA
    # Vi summerer salg og tar snitt av dekningsgrad/priser
    weekly_df = df.resample('W', on='Dato').agg({
        'Antall_par': 'sum',
        'Omsetning_sko': 'sum',
        'Omsetning_total': 'sum',
        'Dekningsgrad_prosent': 'mean',
        'Snittpris_sko': 'mean'
    }).reset_index()
    
    # Lagre tidsserien
    weekly_df.to_csv(output_path, index=False)
    print(f"Tidsserie klargjort og lagret i: {output_path}")
    print(f"Antall uker i datasettet: {len(weekly_df)}")
    print("\nDe første ukene:")
    print(weekly_df.head())
    print("\nDe siste ukene:")
    print(weekly_df.tail())

if __name__ == "__main__":
    prepare_timeseries()
