# Risikoanalyse: Lageroptimalisering Skoringen Råholt

Dette dokumentet identifiserer potensielle risikoer i prosjektgjennomføringen og beskriver tiltak for å minimere disse.

## 1. Risikomatrise

| Risiko | Sannsynlighet (1-5) | Konsekvens (1-5) | Risikoverdi (S x K) | Tiltak |
| :--- | :---: | :---: | :---: | :--- |
| **Mangelfull datakvalitet** | 4 | 5 | 20 | Utvikle robuste vaske-algoritmer og manuell kontroll av utvalgte data. |
| **Overfitting av modellen** | 3 | 4 | 12 | Bruke krysvalidering og "out-of-sample" testing (2025-data). |
| **Motstand mot endring hos ansatte** | 3 | 3 | 9 | Utvikle en brukervennlig manual og involvere daglig leder tidlig. |
| **Tekniske begrensninger i PDF-parsing** | 4 | 4 | 16 | Teste flere Python-biblioteker (`pdfplumber`, `tabula`) tidlig i fasen. |
| **Feilaktige teoretiske antagelser** | 2 | 4 | 8 | Kontinuerlig litteraturgjennomgang og veiledning fra fagmiljøet. |

## 2. Tiltaksplan for kritiske risikoer (Verdi > 15)

### 2.1 Datakvalitet
Siden vi baserer oss på PDF-rapporter, er risikoen for "Garbage In, Garbage Out" stor. 
- **Tiltak:** Vi har implementert en kontrollsum-validering i Python-scriptet som sjekker hver rad mot totalen i PDF-en.

### 2.2 Tekniske begrensninger (PDF-parsing)
PDF-filer er beryktet for å være ustrukturerte.
- **Tiltak:** Vi har bygget en modulær parser som enkelt kan justeres dersom Skoringen endrer formatet på sine rapporter.
