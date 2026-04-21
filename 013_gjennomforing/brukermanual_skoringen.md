# Brukermanual: Prognoseverktøy for Skoringen Råholt

Dette er en enkel veiledning for daglig leder om hvordan verktøyet skal brukes i den daglige driften for å optimalisere innkjøp og unngå eksternt lager.

## 1. Klargjøring av data
Hver måned må de nye dagsrapportene legges inn i systemet:
1.  Lagre dagsrapportene fra kassesystemet som PDF.
2.  Legg filene i mappen `004 data/raw_data/`.
3.  Sørg for at filnavnet inneholder måned og år (f.eks. `Dagsalgsrapport_Januar_2026.pdf`).

## 2. Kjøring av prognose
For å generere nye prognoser og innkjøpsanbefalinger:
1.  Åpne terminalen (eller Python-miljøet).
2.  Kjør kommandoen: `python run_full_pipeline.py`.
3.  Systemet vil nå automatisk lese de nye filene, oppdatere modellen og generere nye grafer.

## 3. Tolking av resultatene
Etter kjøring vil mappen `013 fase 3 - gjennomføring/visuals/` inneholde oppdaterte grafer:
- **`demand_forecast_comparison.png`**: Viser forventet salg for de neste 3-6 månedene. Bruk den blå linjen som utgangspunkt for innkjøp.
- **`inventory_optimization_simulation.png`**: Viser anbefalt lagernivå. Hvis den grønne linjen nærmer seg butikkens tak, bør du vurdere å utsette en leveranse.

## 4. Innkjøpsstrategi (Anbefaling)
- **Frekvens:** Vi anbefaler å gå over til månedlige bestillinger der det er mulig.
- **Sikkerhetsmargin:** Modellen legger automatisk inn en sikkerhetsbuffer. Du trenger ikke å "bestille litt ekstra" for sikkerhets skyld, da dette allerede er ivaretatt av matematikken.
- **Væravvik:** Husk at modellen baserer seg på historikk. Ved ekstreme værvarsler (f.eks. tidlig snøfall), bør du justere opp innkjøpet av vintersko med ca. 10-15 % utover modellens anbefaling.
