<div class="cover">
  <div class="cover-band">
    <div class="cover-eyebrow">LOG650 &middot; Bacheloroppgave i Logistikk &middot; Våren 2026</div>
    <div class="cover-title">Peer-review</div>
    <div class="cover-subtitle">Tilbakemelding på prosjektrapport</div>
  </div>

  <div class="cover-card">
    <div class="cover-row">
      <div class="cover-label">Rapport som vurderes</div>
      <div class="cover-value"><em>Optimalisering av bakkestøtte-ressurser ved Bergen Lufthavn Flesland</em><br><span class="cover-sub">En kvantitativ simuleringsstudie av gate-utnyttelse og busstransport</span></div>
    </div>
    <div class="cover-divider"></div>
    <div class="cover-grid">
      <div>
        <div class="cover-label">Vurdert gruppe</div>
        <div class="cover-value-big">G03 &middot; Bergens Beste</div>
      </div>
      <div>
        <div class="cover-label">Vurderende gruppe</div>
        <div class="cover-value-big">Gruppe 4.5</div>
      </div>
      <div>
        <div class="cover-label">Dato</div>
        <div class="cover-value-big">14. mai 2026</div>
      </div>
      <div>
        <div class="cover-label">Sider</div>
        <div class="cover-value-big">3</div>
      </div>
    </div>
  </div>

  <div class="cover-footer">Tilbakemeldingen er ment som konstruktiv støtte og bygger på vurderingskriteriene i veiledningen for peer-review i LOG650.</div>
</div>

<div class="page-break"></div>

# 1. Helhetsinntrykk

Rapporten er godt strukturert, har klare forskningsspørsmål og en simulering som faktisk fungerer og gir resultater. Den matematiske beskrivelsen i kapittel 3.4 er en sterk del. Dere viser også at dere er ærlige om svakheter ved en deterministisk modell.

De viktigste tingene som trekker rapporten ned:

- **Kapittel 2 er ikke en litteraturgjennomgang** — det er bare et kort teoretisk rammeverk uten tidligere forskning og uten henvisninger til andre studier.
- **Valideringstabellen er nesten tom** — det står "Ikke beregnet" i de fleste feltene.
- **Ingen kildehenvisninger i selve teksten** — bare en kort referanseliste på fem kilder helt til slutt.
- **Ingen figurer i hele rapporten** — selv om dere har mye tallmateriale som kunne vært visualisert.
- **Konklusjonen svarer ikke direkte på de fire forskningsspørsmålene** dere stiller i innledningen.

Under går vi gjennom hvert vurderingsområde med konkrete forbedringsforslag.

---

# 2. Områdevis vurdering

## 2.1 Innledning

**Det som fungerer bra:**

- Bakgrunnen forklarer raskt hvorfor gate-kapasitet og busstransport henger sammen.
- De fire forskningsspørsmålene i 1.3 er konkrete og lar seg faktisk svare på med simulering.
- Avgrensningen i 1.4 er ryddig og forklarer godt hva dere har valgt å holde utenfor.

**Det dere bør forbedre:**

- Dere skriver at Flesland er "Norges nest største flyplass" og at trafikken har økt — men uten kilde eller tall. Legg inn passasjertall fra Avinor.
- Det står ikke tydelig nok hvorfor studien er viktig. Hvem har nytte av svarene? Avinor? Flyselskapene? Skriv det rett ut.
- Avslutt innledningen med en kort setning om hvordan resten av rapporten er bygget opp.

## 2.2 Litteraturgjennomgang og teoretisk forankring

**Dette er rapportens svakeste område.** Kapittel 2 heter "Teoretisk Rammeverk", men det er ikke det samme som en litteraturgjennomgang. Veiledningen ber om tre ting som mangler:

- **Tidligere forskning på området.** Det finnes mange studier på lufthavn­simulering og gate-allokering, men ingen av dem er nevnt.
- **Forskningshullet.** Hvorfor er akkurat *deres* studie nødvendig? Hva har ikke andre allerede gjort?
- **Sammenligning av metoder.** Hvorfor DES og ikke optimering eller agentbasert simulering?

**Andre konkrete problemer:**

- Referanselisten har bare fem kilder. Fire av dem er lærebøker, ingen er fagartikler.
- Det finnes **ingen kildehenvisninger inne i selve teksten**. Påstander som "DES er spesielt sterkt på å modellere kø-situasjoner" må ha en kilde.

**Vårt forslag:** Bygg ut kapittel 2 med 5–8 fagfellevurderte artikler om lufthavn­simulering, og skriv eksplisitt: *"Vår studie skiller seg fra tidligere arbeid ved at..."*. Ta i bruk APA 7 i hele teksten, ikke bare i referanselisten.

## 2.3 Metode

**Det som fungerer bra:**

- Beskrivelsen av forskningsdesign og bruk av simulering som "digitalt laboratorium" er ryddig forklart.
- Datavasken (3.2) er detaljert og gir leseren tillit til at data­grunnlaget er solid.
- **Den matematiske formuleringen i 3.4 er en stor styrke** — mange bachelor­oppgaver hopper over dette.
- Dere skiller riktig mellom verifisering og validering i 3.5.

**Det dere bør forbedre:**

- **Valideringen er bare delvis gjennomført.** Tabellen i 3.5 har "Ikke beregnet" i nesten alle felt. Enten må dere fylle inn faktiske tall fra Avinor-loggene, eller forklare ærlig hvorfor det ikke lot seg gjøre.
- **Reliabilitet er ikke nevnt.** Skriv en setning eller to om at deterministisk simulering gir samme svar ved samme input, og hva som endrer seg når dere senere tar inn tilfeldighet.
- **Etiske hensyn er ikke nevnt.** Selv om det er enkelt (ingen personidentifiserende data), bør dere si det.
- **Kapittel 3.6 er forvirrende.** Dere kaller scenarioene "planlagte" — men flere av dem er allerede gjennomført og ligger i resultattabellen i 4.2. Slå dem sammen, eller flytt 3.6 til kapittel 6 om videre forskning.

## 2.4 Analyse og resultater

**Det som fungerer bra:**

- Den deskriptive statistikken i 4.1 (141 fly, fordeling D/I/S, fly­størrelse) gir leseren et godt bilde før selve simuleringen.
- Tabell 1 oppsummerer ti scenarioer på en oversiktlig måte.
- Refleksjonen i 4.4 om at "lav gjennomsnittlig ventetid kan skjule sårbarhet" er innsiktsfull.

**Det dere bør forbedre:**

- **Ingen figurer.** Dere har mye fint tallmateriale som ville fungert som diagrammer:
    - Hvor mange gates er opptatt time for time gjennom døgnet?
    - Histogram over ventetider.
    - Stolpediagram som sammenligner de ti scenarioene.
- **Maks ventetid er 15,0 minutter i åtte av ti scenarioer.** Dette ser ut som en innebygd grense i koden, ikke et naturlig resultat. Sjekk om det er en timeout eller en busstur-runde, og forklar det.
- **Trafikkvekst-scenarioene gir lite mening uten kontekst.** "+30 % vekst" gir bare 8 ekstra fly totalt. Forklar hvor mange fly som faktisk er i peak (kl. 15:00–17:30) før og etter veksten.
- **Det stokastiske scenariet** står som "snitt" uten antall replikasjoner eller konfidensintervall — selv om dere har formelen for 95 % KI på linje 228. Vis n, standardavvik og KI.
- **Kun én testdag (17. juni 2026).** Forklar hvorfor akkurat denne dagen er valgt — er det den travleste i året? Et gjennomsnitt av topp-10 dager?
- **Et viktig funn er ikke kommentert:** Ventetiden er identisk (0,71 min) ved 2 og 3 sjåfører. Det betyr at to sjåfører trolig er nok — si det rett ut.
- Det er noen rare linjebrudd og halve avsnitt rundt linje 296–299. Rydd opp.

## 2.5 Diskusjon

**Det som fungerer bra:**

- Diskusjonen i 5.1 trekker frem at "lave gjennomsnittsverdier kan skjule sårbarhet" — et godt poeng.
- Koblingen til Littles Lov i 5.2 er et naturlig faglig anker.
- De praktiske anbefalingene i 5.3 er konkrete.

**Det dere bør forbedre:**

- **Funnene knyttes ikke tydelig til de fire forskningsspørsmålene.** Veiledningen krever dette. Lag gjerne fire underseksjoner eller fire bullet-punkter som svarer ett-til-ett på FS1, FS2, FS3 og FS4.
- **Ingen sammenligning med andre studier.** Hvordan står Flesland-tallene seg mot tilsvarende studier av andre lufthavner? (Dette krever at litteraturgjennomgangen i 2.2 først bygges ut.)
- **APOC** dukker opp i 5.3 uten å være forklart tidligere. Skriv ut hva forkortelsen betyr.
- Implikasjoner for **teori og policy** mangler — alt er rettet mot drift hos Avinor. Skriv et avsnitt om hva studien bidrar med metodisk.

## 2.6 Konklusjon

**Det som fungerer bra:**

- Begrensningene i 6.2 er ærlig formulert.
- Forslagene til videre forskning i 6.3 er relevante.

**Det dere bør forbedre:**

- **Konklusjonen svarer ikke på forskningsspørsmålene.** Dere nevner at "to sjåfører er kritisk" og at "20 % vekst kan håndteres" — men FS3 og FS4 dukker aldri opp eksplisitt. Bruk fire bullet-punkter, ett per FS.
- **Studiens bidrag til teori er ikke nevnt.** Selv om bidraget er praktisk, kan dere si: *"Modellen viser hvordan koblede ressurser (gate × buss × sjåfør) kan modelleres i ett enhetlig DES-rammeverk."*
- **Punkt 3 i 6.3** snakker om "tomgangskjøring ved gate-konflikter". Mener dere fly i hold-mønster, eller fly på bakken som venter? Vær mer presis.

## 2.7 Skriveflyt og formelle aspekter

**Det som fungerer bra:**

- Språket er stort sett klart og fagrettet.
- Tabellene er ryddige og leservennlige.
- Bruken av kursiv for engelske fagord (turnaround, flex-gates) er konsekvent.

**Det dere bør forbedre:**

- **APA 7 er ikke fulgt.** Ingen kildehenvisninger i løpende tekst, og bare 5 referanser totalt. Dette er en av rapportens største svakheter formelt sett.
- **Ingen figurer i hele rapporten.** Veiledningen vurderer eksplisitt figurer og visuelt støtte­materiale.
- **Forkortelser:** DES og FCFS er forklart, men APOC er ikke. D/I/S kunne vært definert tydeligere første gang.
- **Vedleggsreferansene** (linje 407–408) peker til Python-filer i et `04_src`-katalog. Avklar om disse faktisk leveres med rapporten.
- **Originalitet:** Bruken av faktisk Avinor-data og koblingen mellom gate, buss og sjåfør er praksisnær og relevant. Dette kan løftes tydeligere frem hvis litteraturgjennomgangen bygges ut.

---

# 3. Hvis dere har lite tid — gjør dette først

Vi anbefaler å prioritere i denne rekkefølgen:

1. **Bygg ut kapittel 2** til en ekte litteraturgjennomgang med fagartikler og et tydelig forskningshull. Innfør APA 7-siteringer i hele teksten.
2. **Fyll inn valideringstabellen** med faktiske tall — eller forklar ærlig hvorfor de ikke kunne beregnes.
3. **Legg til 3–5 figurer** (gateutnyttelse over tid, ventetidsfordeling, scenariosammenligning).
4. **Strukturer diskusjon og konklusjon** rundt de fire forskningsspørsmålene.
5. **Forklar/sjekk** at maks ventetid på 15 minutter ikke er en innebygd grense, og tallfest hva "+20 %/+30 % vekst i peak" faktisk betyr i antall fly.

---

<div class="closing">
Reviewen er gjennomført ut fra kriteriene i Tabell 1 i veiledningen for peer-review i LOG650. Tilbakemeldingen er ment som konstruktiv hjelp til videre arbeid og reflekterer Gruppe 4.5 sin samlede vurdering. Lykke til videre &mdash; det er et godt fundament å bygge på.
</div>
