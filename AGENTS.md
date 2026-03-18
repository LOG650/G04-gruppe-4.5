# Prosjektspesifikke arbeidsregler

## Språk og tegnsett

- Bruk norsk i rapporttekst, statusfiler og planfiler.
- Behold norske bokstaver `æ`, `ø` og `å` i både kode, Markdown og genererte tabeller når filene skal leses av mennesker.
- Lagre tekstfiler som ren `UTF-8` uten BOM hvis ikke noe annet er nødvendig.
- Hvis PowerShell viser feil tegn, anta ikke at filen er ødelagt før filinnholdet er kontrollert direkte.
- Vær ekstra oppmerksom på `status.md`, `rapport.md` og genererte `.md`-tabeller, siden disse lett blir stygge ved feil encoding.

## Rapportskriving

- Skriv innhold fortløpende i rapporten underveis i prosjektet, ikke vent til alt analysearbeid er ferdig.
- Skill tydelig mellom:
  - `Casebeskrivelse`: beskriver bedriften, situasjonen og historiske fakta.
  - `Metode og data`: beskriver metodevalg, datagrunnlag, datakvalitet og datasplitt.
  - `Analyse/Resultat`: brukes først når faktisk prognoseanalyse og modellvurdering er gjort.
- Beskrivende figurer for historisk salg skal inn i casekapitlet, ikke i analysekapitlet.
- Datatabeller som dokumenterer datasettet skal inn i datakapitlet.

## Figurer i rapporten

- Bruk HTML for bilder i `rapport.md`, ikke vanlig Markdown-bildeformat, når bredde og sentrering skal styres.
- Standard for rapportfigurer i dette prosjektet:
  - sentrert figur
  - `width="80%"`
  - kort figurtekst under figuren
- Figurtekst skal være:
  - sentrert
  - liten skrift
  - kursiv
- Foretrukket mønster:

```html
<div align="center">
  <img src="..." alt="..." width="80%">
  <p align="center"><small><i>Figur X Kort figurtekst.</i></small></p>
</div>
```

- Figurteksten skal være kort og nøktern, ikke en hel forklaring.

## Tabeller i rapporten

- Tabeller kan limes inn direkte som Markdown-tabeller når de er små og lesbare.
- Tabeller skal ha en kort introdujonssetning i brødteksten før de settes inn.
- Bruk tabellnummer i teksten, for eksempel `Tabell 4.1`.

## Kode og analyse

- Analysearbeid i `006 analysis` organiseres etter aktiviteter i prosjektplanen.
- Ett felles `uv`-prosjekt brukes for hele `006 analysis`.
- Skript, figurer og resultatfiler skal ligge i samme aktivitetsmappe.
- Filnavn i analyseartefakter skal være korte, ryddige og prefikset med `fig_` og `tab_`.

## Praktiske preferanser i dette repoet

- Når noe er fullført i prosjektet, oppdater både planfiler og `status.md`.
- Nye arbeidssteg bør legges inn i planen før aktiviteten lukkes.
- Når noe bare er en antagelse, skriv det eksplisitt som antagelse og ikke som verifisert fakta.
