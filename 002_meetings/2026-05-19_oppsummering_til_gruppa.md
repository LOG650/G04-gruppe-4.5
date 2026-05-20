# Oppsummering — mandag 19. mai 2026
**Gruppe 4.5 — peer-review-leveranse og forbedringer fra mottatt review**

> I dag har vi to ting på agendaen: (1) lagt inn både peer-reviewen vi skrev om Gruppe 03 og peer-reviewen vi mottok fra **Bergens beste** i mappa rettlederen ba om, og (2) brukt tilbakemeldingene fra Bergens beste til å gjøre **substansielle forbedringer** i hovedrapporten. Alt er pushet til GitHub.

---

## Bakgrunn — hva som skjedde

I helga trodde vi vi ikke kom til å få noen peer-review fra den andre gruppa, så vi hadde laget en selvevaluering som backup. I dag dukket reviewen fra **Bergens beste** plutselig opp — datert 16. mai. Det er en grundig, ryddig review som plukker opp flere virkelig viktige punkter, blant annet en intern inkonsistens i tallene våre som vi ikke hadde sett selv.

Rettleder har bedt om at peer-reviewene legges i en mappe i GitHub-repoet. Vi tolket den som **`013_peer_review/`** for å følge repo-konvensjonen `nnn_lowercase_snake_case`.

---

## Del A — Peer-review-leveransen

### Ny mappe: `013_peer_review/`

Innholdet er nå pushet:

| Fil | Innhold |
|---|---|
| `Peer_review_G03_av_Gruppe45.md` | Reviewen vi skrev om Gruppe 03 (kildeformat) |
| `Peer_review_G03_av_Gruppe45.html` | Samme review, stylet HTML (v3 — siste versjon) |
| `Peer_review_G03_av_Gruppe45.pdf` | Samme review, PDF (regenerert fra v3-HTML) |
| `Peer_review_LOG650_Gruppe_4_5.pdf` | Reviewen Bergens beste skrev om oss |

PDF-en fra v3-HTML ble generert via samme Edge-headless-metode som `md_to_pdf.py` bruker, slik at stilen er identisk med tidligere versjoner.

---

## Del B — Hva Bergens beste sa om oss

Reviewen er totalt sett **veldig positiv** og kaller oppgaven "en sterk forskningsoppgave med høy faglig relevans, tydelig praktisk forankring og en ambisiøs kvantitativ analyse". De tre konkrete anbefalingene de løfter frem er:

1. **Lag en samletabell over alle sentrale antakelser** (pris, restverdi, servicenivå, RMSE, aggregeringsnivå)
2. **Stram inn teori- og innledningsdelene** så hovedargumentet kommer raskere frem
3. **Tydeliggjør skillet mellom faktiske data, modellberegninger og scenarioestimater** i resultat- og konklusjonskapitlene

I tillegg flere mindre punkter per kapittel:
- Tydeligere forskningshull i §2.6 (de mente "kunne vært tydeligere")
- Diskuter forventede vs uventede funn i §5
- Trinnvis implementeringsanbefaling i konklusjonen
- Korte "Hva betyr dette for Skoringen?"-bokser

---

## Del C — Hva vi har gjort med tilbakemeldingene

Vi har gjort fire substansielle endringer i hovedrapporten i dag. Fokus var på A-karakter-løft, ikke på å fjerne ting.

### 1. Ny §3.7 — Samletabell over sentrale antakelser

Dette var Bergens bestes anbefaling **#1**, og det er en av deres viktigste innspill. Vi har lagt til en 14-rads tabell mellom §3.6 og kapittel 4 som samler alle:

- **Parameterantakelser** (p, w, s, servicenivå)
- **Modellforutsetninger** (sigma via RMSE, uavhengige månedsfeil, normalfordeling)
- **Designvalg** (én aggregert SKU, treningsperiode, testperiode, sesongdefinisjon, returbehandling, eksogene variabler)
- **Fakta** (lagerkapasitet 3 000 par fra intervju)

Hver rad har: verdi/form, type, sensitivitet for hovedkonklusjonen, og peker til hvor antakelsen drøftes videre. Etterfulgt av tre korte observasjoner: prisparametere er estimat, aggregering er den mest vidtrekkende modellforutsetningen, og sigma-estimering kan oppgraderes med bootstrap.

Dette gjør det vesentlig enklere for sensor å vurdere hvor robuste resultatene er.

### 2. Ny §5.8 — Forventede og uventede funn

Bergens beste etterlyste eksplisitt diskusjon av hva som overrasket. Vi har lagt til en helt ny seksjon i kapittel 5 som starter med fire forventede funn (sesongleddet bærer informasjon, SARIMA slår naiv, restverdien er mest sensitive parameter, eksternt lager kan ikke elimineres ved bedre prognose) og deretter tre **uventede funn**:

- **Den flate profittkurven nær Q*** — newsvendor er mer robust enn forventet for små feil i $\mu, \sigma$
- **Høstsesongen ble *nedjustert*, ikke oppjustert** — vi gikk inn med en intuisjon om at modellen ville si "bestill mer", men SARIMA fant at fjorårets september-hopp var et engangstilfelle
- **ETS var dårligere enn naiv baseline** — uventet, fordi Holt-Winters normalt er konkurransedyktig med SARIMA

Disse tre observasjonene er ikke korrigeringer av kjent teori, men illustrasjoner av at empirisk evaluering kan gi resultater som strider mot intuisjonen man hadde før analysen.

### 3. §6.3 reformulert som trinnvis implementering

Bergens beste foreslo trinnvis innføring (modellen først som støtteverktøy parallelt med dagens praksis, så justering etter hvert som faktiske priser blir tilgjengelige). Vi har dette allerede i §5.5, men det var ikke synlig nok i konklusjonen. Vi har nå reformulert intro til §6.3 til:

> "Anbefalingene bør gjennomføres som en *trinnvis implementering* (parallellkjøring → hybridkjøring → full modellkjøring), slik vi har beskrevet i §5.5. Tanken er at modellen først skal *supplere* dagens praksis – ikke erstatte den over natten."

Anbefaling 1 er omskrevet med eksplisitt referanse til parallellkjøring som fase 1.

### 4. Intern inkonsistens fikset: 713 621 NOK → 570 000 NOK

Dette er den mest interessante delen. Bergens beste sa i §2.4: *"newsvendor-strategien gir en estimert forbedring i nettoresultat på 713 621 kroner, tilsvarende 14,4 prosent"* — men i sammendraget vårt og i §6.1 sier vi **570 000 NOK / +11,5 %**. De plukket opp tallet vi *brukte å bruke*, ikke det canonical tallet.

Det var en gammel verdi som hadde overlevd i §5.1 og §5.7 etter at sigma-fiksen 15. mai endret hovedfunnet. Vi har nå:

| Hvor | Før | Etter |
|---|---:|---:|
| §5.1 (informasjon som beslutningsstøtte) | 713 621 NOK / +14,4 % | ≈ 570 000 NOK / +11,5 % |
| §5.1 (dekomponering) | +566 423 NOK vår / +147 198 NOK høst | om lag +700 000 NOK vår / om lag −130 000 NOK høst |
| §5.7 (empirisk bidrag) | 15,8 % MAE-reduksjon / +14,4 % nettoresultat | 14,0 % MAE-reduksjon / +11,5 % nettoresultat |

Nå er alle tallene konsistente med Tabell 4.4 og `newsvendor_resultater.json`.

---

## Del D — Hva vi *ikke* gjorde (bevisst)

Bergens beste foreslo også:

- **Stram inn teori- og innledningsdelene** — vi vurderte dette, men brukeren var tydelig på at vi **ikke** skal fjerne noe som ikke direkte forbedrer oppgaven. Teorikapittelet er en av de identifiserte styrkene, så vi lar det stå.
- **Korte "Hva betyr dette for Skoringen?"-bokser underveis** — dette er et lesbarhetsforslag som kunne lagt til 10–15 nye bokser i rapporten. Vi anså at dette ville gjøre rapporten lengre uten å heve den faglige kvaliteten, og lot det være.

Dette er bevisste valg — vi fulgte de tre **konkrete** anbefalingene Bergens beste prioriterte, men ikke alle deres mindre forslag.

---

## Del E — Filer som er endret i dag

| Fil | Hva som skjedde |
|---|---|
| `013_peer_review/` (ny mappe) | 4 filer: vår review (md, html, pdf) + mottatt review (pdf) |
| `014_report/Forskningsoppgave_Gruppe_4.5.md` | Ny §3.7 (antakelsestabell), ny §5.8 (forventede/uventede funn), §6.3 reformulert, §5.1 og §5.7 fikset for inkonsistens, TOC oppdatert |
| `014_report/Forskningsoppgave_Gruppe_4.5.html` | Regenerert via `build_html.py` |
| `014_report/Forskningsoppgave_Gruppe_4.5.pdf` | Regenerert via `_preview/full_rebuild.py` (Chrome iterert to ganger til TOC konvergerte) |
| `014_report/peer_to_peer_2026-04-30/Forskningsoppgave_Gruppe_4.5.pdf` | Samme PDF kopiert hit (snapshot-mappa) |
| `002_meetings/2026-05-19_oppsummering_til_gruppa.md` | Denne oppsummeringen |

Sidetallet i rapporten er nå **48 sider** (fra 47 før).

---

## Del F — Tallene som gjelder fra og med i dag

Ingen av kjernetallene er endret — kun internt konsistente nå.

### Prognosepresisjon (uendret)
- SARIMA: MAE 140 par, RMSE 182, MAPE 16,9 %
- +14,0 % forbedring vs naiv baseline
- 4,3 % avvik på årssum 2025

### Newsvendor (uendret)
- $Q^*_\text{vår}$ = 5 975 par
- $Q^*_\text{høst}$ = 3 468 par
- Sikkerhetslager 302 par per sesong

### Økonomisk effekt (nå konsistent overalt)
- **≈ 570 000 NOK netto årseffekt (+11,5 %)**
- Bruttoresultat: +331 000 NOK
- Alternativkostnad tapt salg: −238 000 NOK

---

## Hva betyr dette for sluttresultatet?

Rapporten er nå **mer ærlig og mer leservennlig** der den før hadde to reelle svakheter:

1. Den hadde en **intern inkonsistens** i de økonomiske tallene som en oppmerksom sensor ville ha fanget opp
2. Den manglet et **samlet overblikk** over hvilke antakelser som ligger til grunn — leseren måtte spores opp gjennom hele metodekapittelet

Begge er nå håndtert. I tillegg har vi en **eksplisitt refleksjon om uventede funn** som styrker diskusjonskapittelet — det er ikke et must-have, men det viser modenhet i analysen.

Bergens beste sin samlede vurdering var: *"en sterk forskningsoppgave med høy faglig relevans, tydelig praktisk forankring og en ambisiøs kvantitativ analyse"*. Med endringene i dag mener jeg vi har styrket oppgaven på de tre konkrete områdene de prioriterte, uten å miste noe av det de roste.

---

## Hva som gjenstår

Rettleder skrev: *"Vi skal komme tilbake til akkurat korleis hovudrapporten skal leverast."* Så vi venter på det endelige leveringsformatet. Imellomtiden:

1. **Faktiske enhetspriser fra Skoringens regnskap** (fortsatt det viktigste enkelttiltaket for å stabilisere kronetallene)
2. **Korrekturlesing av rapporten i sammenheng** — vi har lagt til en del nytt innhold; verdt en gjennomlesning
3. **Diskusjonen rundt om Bergens beste sine andre forslag** (innledning-/teori-stramming, lesebokser) bør tas opp i gruppa hvis dere mener noen av dem er verdt å vurdere

---

## Trenger dere noe?

For å se rapporten i sin nye form, bygg PDF-en på nytt:

```
cd 014_report/_preview
python full_rebuild.py
```

Den ligger også ferdigbygget i `014_report/Forskningsoppgave_Gruppe_4.5.pdf` etter dagens commit. Hvis dere vil se selve peer-reviewen vi mottok, ligger den i `013_peer_review/Peer_review_LOG650_Gruppe_4_5.pdf`.

Spør gjerne hvis noe er uklart.

— Gustavo
