import pdfplumber
import pandas as pd
import os
import re

def reverse_string(s):
    return s[::-1]

def is_date_marker(s):
    # Sjekker om strengen er en dato (1-31)
    try:
        val = int(reverse_string(s))
        return 1 <= val <= 31
    except:
        return False

def extract_data_from_pdf(file_path):
    all_rows = []
    file_name = os.path.basename(file_path)
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            current_day_data = []
            
            # Vi leter etter tall-sekvenser. 
            # I denne PDF-en kommer datoen først, så 8 verdier som tilhører den datoen.
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Snu teksten for å få riktige tall
                fixed_val = reverse_string(line)
                
                # Vi ser etter rader som starter med en dato
                # Logikken her er: Hvis vi finner en dato, starter vi en ny rad.
                # Men siden dataene kommer i vertikale blokker, må vi være smarte.
                
                # Siden PDF-strukturen er vertikal (Dato -> Verdi 1 -> Verdi 2...), 
                # samler vi opp verdier til vi har en full rad.
                current_day_data.append(fixed_val)
                
            # Etter å ha samlet alle verdier på siden, må vi gruppere dem.
            # Basert på analysen er det ca 9 verdier per dato-rad.
            # Vi leter etter dager (1-31) og samler verdiene som følger.
            
            i = 0
            while i < len(current_day_data):
                val = current_day_data[i]
                # Sjekk om dette er starten på en dato-rad
                if val.isdigit() and 1 <= int(val) <= 31:
                    row = [file_name]
                    row.append(val) # Dato
                    # Hent de neste 8 verdiene (Salg, Par, Snittpris, etc.)
                    for j in range(1, 9):
                        if i + j < len(current_day_data):
                            row.append(current_day_data[i+j])
                        else:
                            row.append("")
                    all_rows.append(row)
                    i += 9 # Hopp til neste mulige dato
                else:
                    i += 1
    
    return all_rows

def main():
    data_dir = "004_data"
    output_file = "004_data/skoringen_salgsdata_total.csv"
    
    all_extracted_data = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.pdf') and not f.startswith('Manedsrapport') and not f.startswith('~$')]
    
    print(f"Fant {len(files)} dagsalgsrapporter. Starter dekoding...")

    for file in files:
        file_path = os.path.join(data_dir, file)
        print(f"Prosesserer {file}...")
        try:
            rows = extract_data_from_pdf(file_path)
            all_extracted_data.extend(rows)
        except Exception as e:
            print(f"Feil ved lesing av {file}: {e}")

    if all_extracted_data:
        # Kolonnenavn basert på det vi ser i PDF-en
        columns = ['Filnavn', 'Dato', 'Verdi_1', 'Verdi_2', 'Verdi_3', 'Verdi_4', 'Verdi_5', 'Verdi_6', 'Verdi_7', 'Verdi_8']
        df = pd.DataFrame(all_extracted_data, columns=columns[:len(all_extracted_data[0])])
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSuksess! Data er lagret i: {output_file}")
        print(f"Totalt antall dager ekstrahert: {len(df)}")
        print("\nDe første 5 radene:")
        print(df.head())
    else:
        print("Fant ingen dagsalgsdata i PDF-ene.")

if __name__ == "__main__":
    main()
