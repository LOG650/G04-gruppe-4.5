import pandas as pd
import os
import re

def clean_sales_data():
    file_path = "004_data/skoringen_salgsdata_total.csv"
    output_path = "004_data/skoringen_salgsdata_clean.csv"
    
    if not os.path.exists(file_path):
        print("Fant ikke rådata-filen.")
        return

    df = pd.read_csv(file_path)
    
    # 1. Fjern rader som ikke er faktiske salgsdata (f.eks. overskrifter fra PDF-en)
    # Vi beholder bare rader der 'Dato' er et tall og 'Verdi_1' ikke inneholder tekst som 'Side'
    df = df[pd.to_numeric(df['Dato'], errors='coerce').notnull()]
    df = df[~df['Verdi_1'].astype(str).str.contains('Side', na=False)]
    
    # 2. Trekk ut Måned og År fra filnavnet
    def extract_month_year(filename):
        filename = filename.lower()
        # Finn år (to siffer på slutten av navnet, f.eks. '23', '24', '25')
        year_match = re.search(r'(\d{2})\.pdf$', filename)
        year = "20" + year_match.group(1) if year_match else "Ukjent"
        
        # Finn måned
        months = {
            'jan': 'Januar', 'feb': 'Februar', 'mar': 'Mars', 'apr': 'April',
            'mai': 'Mai', 'jun': 'Juni', 'jul': 'Juli', 'aug': 'August',
            'sep': 'September', 'okt': 'Oktober', 'nov': 'November', 'des': 'Desember'
        }
        month = "Ukjent"
        for key, val in months.items():
            if key in filename:
                month = val
                break
        return month, year

    df[['Måned', 'År']] = df['Filnavn'].apply(lambda x: pd.Series(extract_month_year(x)))
    
    # 3. Gi kolonnene riktige navn basert på analysen
    # Vi beholder alle kolonner, men gir dem navn som gir mening
    column_mapping = {
        'Dato': 'Dag',
        'Verdi_1': 'Salg_per_kunde',
        'Verdi_2': 'Dekningsgrad_prosent',
        'Verdi_3': 'Snittpris_totalt',
        'Verdi_4': 'Snittpris_sko',
        'Verdi_5': 'Omsetning_sko',
        'Verdi_6': 'Omsetning_tilbehor',
        'Verdi_7': 'Omsetning_total',
        'Verdi_8': 'Antall_par'
    }
    df = df.rename(columns=column_mapping)
    
    # 4. Konverter tall-strenger til faktiske tall (komma -> punktum)
    # Vi gjør dette VELDIG forsiktig for å ikke endre verdier
    cols_to_fix = ['Dekningsgrad_prosent', 'Omsetning_sko', 'Omsetning_tilbehor', 'Omsetning_total', 'Antall_par', 'Salg_per_kunde', 'Snittpris_totalt', 'Snittpris_sko']
    for col in cols_to_fix:
        if col in df.columns:
            # Fjern eventuelle mellomrom og bytt komma med punktum
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            # Konverter til tall (float)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Organiser kolonnene på en ryddig måte
    final_columns = ['År', 'Måned', 'Dag', 'Antall_par', 'Omsetning_sko', 'Omsetning_tilbehor', 'Omsetning_total', 'Dekningsgrad_prosent', 'Snittpris_sko', 'Filnavn']
    df = df[final_columns]
    
    # Lagre den rensede filen
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Rensing fullført! Renset fil lagret i: {output_path}")
    print(f"Antall rader i renset data: {len(df)}")
    print("\nDe første radene i de rensede dataene:")
    print(df.head())

if __name__ == "__main__":
    clean_sales_data()
