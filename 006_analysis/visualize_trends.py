import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Konfigurer stilen for alle diagrammer
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def create_visualizations():
    data_path = "004 data/skoringen_salgsdata_clean.csv"
    output_dir = "013 fase 3 - gjennomføring/visuals"
    
    # Lag mappen hvis den ikke finnes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(data_path)
    
    # 1. Forberedelse av data: Sorter kronologisk
    month_order = ['Januar', 'Februar', 'Mars', 'April', 'Mai', 'Juni', 
                   'Juli', 'August', 'September', 'Oktober', 'November', 'Desember']
    df['Måned'] = pd.Categorical(df['Måned'], categories=month_order, ordered=True)
    df = df.sort_values(['År', 'Måned', 'Dag'])
    
    # Lag en dato-kolonne for tidsserien
    # Vi må mappe månedsnavn til tall
    month_map = {m: i+1 for i, m in enumerate(month_order)}
    # Vi bruker errors='coerce' for å håndtere ugyldige datoer som kan ha sneket seg med i PDF-uthentingen
    df['Date_Index'] = pd.to_datetime(df.assign(month=df['Måned'].map(month_map), 
                                               year=df['År'], 
                                               day=df['Dag'])[['year', 'month', 'day']], errors='coerce')
    
    # Fjern rader der datoen ble ugyldig
    df = df.dropna(subset=['Date_Index'])

    # --- DIAGRAM 1: Månedlig totalomsetning over tid ---
    monthly_sales = df.groupby(['År', 'Måned'], observed=True)['Omsetning_total'].sum().reset_index()
    monthly_sales['Måned_År'] = monthly_sales['Måned'].astype(str) + " " + monthly_sales['År'].astype(str)
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=monthly_sales, x=range(len(monthly_sales)), y='Omsetning_total', marker='o', linewidth=2.5, color='#2c3e50')
    plt.xticks(range(len(monthly_sales)), monthly_sales['Måned_År'], rotation=45, ha='right')
    plt.title('Månedlig Totalomsetning (2023 - 2026)', fontsize=16, fontweight='bold')
    plt.ylabel('Omsetning i NOK')
    plt.xlabel('Tid (Måned/År)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_omsetning_over_tid.png", dpi=300)
    plt.close()

    # --- DIAGRAM 2: Sesongvariasjoner (Sammenligning av år) ---
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=monthly_sales, x='Måned', y='Omsetning_total', hue='År', marker='o', palette='viridis', linewidth=2)
    plt.title('Sesongtrender: Sammenligning av årsomsetning', fontsize=16, fontweight='bold')
    plt.ylabel('Omsetning i NOK')
    plt.xlabel('Måned')
    plt.legend(title='År')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_sesongtrender_sammenligning.png", dpi=300)
    plt.close()

    # --- DIAGRAM 3: Antall par solgt per måned ---
    monthly_pairs = df.groupby(['År', 'Måned'], observed=True)['Antall_par'].sum().reset_index()
    plt.figure(figsize=(14, 7))
    sns.barplot(data=monthly_pairs, x='Måned', y='Antall_par', hue='År', palette='coolwarm')
    plt.title('Salgsvolum: Antall par solgt per måned', fontsize=16, fontweight='bold')
    plt.ylabel('Antall solgte par')
    plt.xlabel('Måned')
    plt.legend(title='År')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_salgsvolum_par.png", dpi=300)
    plt.close()

    print(f"Visualiseringer er ferdig generert og lagret i: {output_dir}")
    print("Filer generert:")
    print("1. 01_omsetning_over_tid.png - Viser total utvikling.")
    print("2. 02_sesongtrender_sammenligning.png - Viser mønstre på tvers av år.")
    print("3. 03_salgsvolum_par.png - Viser volumutvikling.")

if __name__ == "__main__":
    create_visualizations()
