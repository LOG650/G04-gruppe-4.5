# Bacheloroppgave: Lageroptimalisering hos Skoringen Råholt
**LOG650 - Forskningsprosjekt i logistikk**

Dette depotet inneholder det komplette forskningsprosjektet utført av Gruppe 4.5. Prosjektet kombinerer logistikkteori, dataanalyse og automatisering for å løse reelle kapasitetsutfordringer i detaljhandelen.

## 📁 Prosjektstruktur

For å navigere i prosjektet, følg denne logiske rekkefølgen:

- **`000_templates/`**: Offisielle krav og maler for oppgaven.
- **`003_referanser/`**: Akademisk litteratur og kildeliste (APA 7).
- **`004_data/`**: Datagrunnlag. Inneholder `raw_data` (PDF-rapporter) og vaskede CSV-filer.
- **`006_analysis/`**: Python-kildekoden for pipeline, prognoser og simulering.
- **`011_proposal/`**: Det opprinnelige prosjektforslaget.
- **`012_plan/`**: Prosjektstyring, risikoanalyse og interessentkartlegging.
- **`013_gjennomforing/`**: Teknisk dokumentasjon, valideringsrapporter og visualiseringer.
- **`014_report/`**: Den endelige bacheloroppgaven i Markdown-format.

## 🚀 Teknisk Løsning
Systemet automatiserer prosessen fra ustrukturerte salgsrapporter til prediktive innkjøpsbeslutninger:
1.  **Parsing:** PDF-data trekkes ut via `pdfplumber`.
2.  **Modellering:** Bruker SARIMA for å fange opp sesongvariasjoner.
3.  **Simulering:** Newsvendor-logikk brukes for å beregne optimalt sikkerhetslager og bevise kostnadsbesparelser.

## 🛠 Installasjon og Bruk
For å reprodusere analysen, installer nødvendige biblioteker:
```bash
pip install -r requirements.txt
```
Kjør hele pipelinen med:
```bash
python 006_analysis/run_full_pipeline.py
```

---
*Utviklet av: Gustavo Alfonso Holmedal, Thuy Thu Thi Tran, Inger Irgesund (April 2026)*
