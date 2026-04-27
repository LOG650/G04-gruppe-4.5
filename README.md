# Bacheloroppgave: Sesongbestilling under prognoseusikkerhet hos Skoringen Råholt
**LOG650 - Forskningsprosjekt i logistikk**

Dette depotet inneholder det komplette forskningsprosjektet utført av Gruppe 4.5. Prosjektet kombinerer SARIMA-baserte etterspørselsprognoser med newsvendor-modellen for å forbedre Skoringen Råholts to årlige sesongbestillinger.

## 📁 Prosjektstruktur

| Mappe | Innhold |
|---|---|
| `000_templates/` | Offisielle krav og maler for oppgaven (proposal-mal, prosjektstyringsplan, APA 7) |
| `001_info/` | Kontaktinformasjon, bedriftsinfo og beslutningslogg |
| `002_meetings/` | Møtereferater (faste søndag- og onsdag-møter) + mal |
| `003_referanser/` | Akademisk litteratur og pensumkompendium |
| `003_referanser/Kompendium/` | LOG650-pensumet med innholdsfortegnelse `00_INDEX.md` |
| `004_data/` | Datagrunnlag: PDF-rapporter, `raw_data/` og vaskede CSV-filer |
| `006_analysis/` | Python-kildekode for pipeline, prognoser og simulering |
| `011_proposal/` | Prosjektforslaget |
| `012_plan/` | Prosjektstyring, risikoanalyse og interessentkartlegging |
| `013_gjennomforing/` | Teknisk dokumentasjon, valideringsrapport, visualiseringer og brukermanual |
| `014_report/` | Bacheloroppgaven (`Bacheloroppgave_Skoringen_KOMPLETT.md`) |

## 🚀 Teknisk Løsning
Systemet er en pipeline fra ustrukturerte salgsrapporter til kvantitative bestillingsanbefalinger:
1. **Parsing:** PDF-data trekkes ut via `pdfplumber`.
2. **Modellering:** SARIMA-modell fanger trend og sesong (pensum Ch01 §3).
3. **Bestilling:** Newsvendor-formel beregner optimal sesongbestilling $Q^* = \mu + z_\alpha \cdot \sigma$ (pensum Ch05 §5).

## 🛠 Installasjon og bruk
Installer nødvendige biblioteker:
```bash
pip install -r requirements.txt
```
Kjør hele pipelinen:
```bash
python 006_analysis/run_full_pipeline.py     # PDF -> CSV -> SARIMA-prognose
python 006_analysis/sesongnewsvendor.py      # Newsvendor-bestilling pr sesong
python 006_analysis/verify_numbers.py        # Verifiser nøkkeltall i rapporten
```

## 📜 Konvensjoner
- Alle mappenavn bruker `snake_case` med understrek (`nnn_lowercase`).
- Norske bokstaver (`æ`, `ø`, `å`) brukes i innhold, men unngås i fil- og mappenavn for å unngå Windows/Git-rare.
- Datoformat i filnavn: `YYYY-MM-DD`.

---
*Utviklet av: Gustavo Alfonso Holmedal, Thuy Thu Thi Tran, Inger Irgesund (April 2026)*
