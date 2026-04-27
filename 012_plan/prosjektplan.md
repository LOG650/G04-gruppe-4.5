# Prosjektplan – LOG650 Gruppe 4.5
**Prosjektnavn:** Sesongbestilling under prognoseusikkerhet hos Skoringen Råholt
**Utarbeidet:** Februar 2026 (siste revisjon april 2026)
**Leveringsfrist:** 31. mai 2026

## 1. Prosjektmål og Omfang
- **Hovedmål:** Utvikle en kvantitativ modell som kombinerer SARIMA-etterspørselsprognose og newsvendor-logikk for å forbedre Skoringen Råholts sesongbestillinger, gitt bransjebetingelsen om to bestillinger pr år (vår + høst).
- **Deltakere:** 
    - Gustavo Alfonso Holmedal (Prosjektleder)
    - Inger Irgesund
    - Thuy Thu Thi Tran
- **Kunde:** Skoringen Råholt (v/ Marit Stoksflod)

## 2. Milepæler og Fremdriftsplan
*Basert på prosjektstyringsplanens kritiske linje:*
- [ ] **Milepæl 1:** Godkjent prosjektplan (Fase 2)
- [ ] **Milepæl 2:** Ferdigstilt datainnsamling fra Skoringen Råholt
- [ ] **Milepæl 3:** Ferdigstilt analyse av data og modellutvikling
- [ ] **Milepæl 4:** Første utkast av rapport ferdig
- [ ] **Milepæl 5:** Endelig rapport ferdigstilt (Frist: Mai 2026)

## 3. Organisering og Kommunikasjon
- **Møtefrekvens:** Ukentlige interne saksstatusmøter (Teams).
- **Samhandling:** Bruk av VS Code med Gemini CLI, Python for analyse, og Microsoft Project for fremdriftsstyring.
- **Veiledning:** Månedlige prosjektgjennomganger med veilederne Bård Inge Pettersen og Per Kristian Rekdal.
- **Kvalitetssikring:** Bruk av uformelle Peer-to-Peer vurderinger (maks 1 time per review) og brukerreviews med butikkeier.

## 4. Risikoanalyse
| Risiko | Sannsynlighet | Konsekvens | Tiltak |
| :--- | :---: | :---: | :--- |
| Manglende/ufullstendig datagrunnlag | Lav | Høy | Tidlig dialog og avklaring med butikkeier |
| Datakvalitet som gjør analyse vanskelig | Middels | Middels | Sette av tid til grundig datarensing og klargjøring |
| Tekniske utfordringer med verktøy | Lav | Middels | Bruk av kjente verktøy (Excel/Python) og Gemini CLI |
| Begrenset tid til analyse/rapport | Middels | Høy | Streng oppfølging av kritisk linje i Gantt-planen |

## 5. Ressursbehov
- **Data:** Historiske salgs- og lagerdata fra Skoringen Råholt.
- **Programvare:** Microsoft Excel, Python, VS Code, Microsoft Project.
- **Ekspertise:** Veiledning fra Høgskolen i Molde og praktisk innsikt fra butikkens drift.

## 6. Metodikk og Tekniske Rammer
- **Analysemodeller:** Testing og sammenligning av ETS, ARIMA og SARIMA for etterspørselsprognoser (pensumets Ch01 §3). Newsvendor-modell (Ch05 §5) for sesongbestilling.
- **KPI-fokus:** Tapt salg (alternativkostnad), overlager og estimert nettoresultat. Servicenivå styres via det kritiske forholdet $(p-w)/(p-s)$.
- **Avgrensninger:** Prosjektet fokuserer kun på produktkategorien "sko" som én aggregert SKU og tar ikke høyde for størrelsesfordeling, eksterne faktorer (vær, kampanjer) eller leveringstid. Bestillingsfrekvensen er gitt som to pr år (bransjebetingelse).
