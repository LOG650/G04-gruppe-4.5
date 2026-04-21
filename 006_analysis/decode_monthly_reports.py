import pdfplumber
import pandas as pd
import os

def reverse_string(s):
    return s[::-1]

def decode_monthly_pdf(file_path, year):
    all_data = []
    months = {
        '.naJ': 1, '.beF': 2, '.raM': 3, '.rpA': 4, 'jaM': 5, '.nuJ': 6,
        '.luJ': 7, '.guA': 8, '.peS': 9, '.tkO': 10, '.voN': 11, '.ceD': 12
    }
    
    with pdfplumber.open(file_path) as pdf:
        text = pdf.pages[0].extract_text()
        if not text: return []
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for i, line in enumerate(lines):
            if line in months:
                month_nr = months[line]
                
                # Basert på den vertikale strukturen vi så:
                # i: .naJ (Month)
                # i-1: 414 (Par)
                # i-2: 69 (Index)
                # i-3: 408262 (Sko Omsetning)
                # i-4: 138 (Index)
                # i-5: 536 (Snittpris)
                # i-6: 84902 (Tilbehør)
                # i-7: 140 (Index)
                # i-8: 257382 (Total Omsetning)
                # ... dekningsgrad er lenger opp? 
                # La oss telle: Jan. er på indeks i.
                # Linjene over:
                # [i-13]: 320 (Pr par?)
                # [i-12]: 50,3 (Dekningsgrad)
                # [i-11]: 188 (Index)
                # [i-10]: 122286 (Sko)
                # [i-9]: 138 (Index)
                # [i-8]: 283752 (Total)
                # [i-7]: 140 (Index)
                # [i-6]: 20948 (Tilbehør)
                # [i-5]: 635 (Snittpris)
                # [i-4]: 138 (Index)
                # [i-3]: 262804 (Sko)
                # [i-2]: 69 (Index)
                # [i-1]: 414 (Par)
                
                try:
                    antall_par = float(reverse_string(lines[i-1]))
                    # Total omsetning er på i-8 (hvis vi ser på den første blokken i loggen min)
                    # Vent, la oss se på loggen igjen:
                    # 414 (i-1)
                    # 69 (i-2)
                    # 408262 (i-3) -> Sko?
                    # 831 (i-4) -> Index?
                    # 536 (i-5) -> Snittpris?
                    # 84902 (i-6) -> Tilbehør?
                    # 041 (i-7) -> Index?
                    # 257382 (i-8) -> Total?
                    # 138 (i-9) -> Index?
                    # 682231 (i-10) -> Sko?
                    # 881 (i-11) -> Index?
                    # 3,05 (i-12) -> Dekning?
                    
                    # Det ser ut som verdiene er reversert individuelt OG i rekkefølge
                    # La oss bruke try-except for å finne verdier som ser ut som omsetning
                    total = float(reverse_string(lines[i-8]))
                    dekning = float(reverse_string(lines[i-12]).replace(',', '.'))
                    
                    all_data.append({
                        'År': year,
                        'Måned': month_nr,
                        'Antall_par': antall_par,
                        'Omsetning_total': total,
                        'Dekningsgrad_prosent': dekning
                    })
                except Exception as e:
                    # print(f"Kunne ikke dekode {line}: {e}")
                    continue
                    
    return all_data

def main():
    data_dir = "004 data"
    output_file = "004 data/skoringen_monthly_clean.csv"
    
    reports = [
        ("Manedsrapport2023.pdf", 2023),
        ("Manedsrapport2024.pdf", 2024),
        ("Manedsrapport2025.pdf", 2025)
    ]
    
    full_data = []
    for filename, year in reports:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            print(f"Dekoder {filename}...")
            data = decode_monthly_pdf(path, year)
            full_data.extend(data)
            
    if full_data:
        df = pd.DataFrame(full_data)
        df = df.sort_values(['År', 'Måned'])
        df = df.drop_duplicates(subset=['År', 'Måned'])
        df.to_csv(output_file, index=False)
        print(f"Suksess! Lagret {len(df)} måneder i {output_file}")
        print(df.tail(15))
    else:
        print("Ingen data funnet.")

if __name__ == "__main__":
    main()
