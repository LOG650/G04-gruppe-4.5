# Kompendiumkobling – fullstendig tabell

Dette dokumentet er teknisk vedlegg til *Forskningsoppgave Gruppe 4.5 – Sesongbestilling under prognoseusikkerhet*. Det utfyller §2.7 i hovedrapporten med detaljert mapping mellom pensumkompendiet, primærkilder og oppgavens kapitler.

## Pensumseksjoner brukt som ramme for diskusjon og videre arbeid

Tabellen under viser pensumseksjoner som ikke inngår direkte i hovedanalysen, men som gir det teoretiske grunnlaget for diskusjonen av begrensninger (§5.3) og forslag til videre arbeid (§6.4).

| Kompendium-seksjon (arbeidsredskap) | Hvor brukt | Funksjon | Sitert primærkilde |
|---|---|---|---|
| Ch01 §4 (eksterne-faktorer / ARIMAX) | §5.3, §6.4 | Eksogene regressorer (vær, kampanjer) som videre modell | Box et al. (2015); Hyndman & Athanasopoulos (2021); Lv et al. (2023) |
| Ch01 §5 (mange-variabler / LightGBM) | §3.4, §6.4 | ML-alternativ; vurdert som hybrid komplement til SARIMA | Puchinger & Raidl (2005) |
| Ch02 §3 (multi-produkt Q,R) | §3.5, §5.3 | Ramme for SKU/størrelsesutvidelse av newsvendor | Silver et al. (2016); Hartmann & Briskorn (2010) |
| Ch02 §4 (flerlokasjon stokastisk) | §1.4, §5.3, §6.4 | Butikk + eksternt lager som to lokasjoner under usikkerhet | Silver et al. (2016); Hartmann & Briskorn (2010) |
| Ch02 §5 (ML-klassifisering) | §6.4 | ABC/XYZ-klassifisering som forarbeid til SKU-modell | Silver et al. (2016) |
| Ch03 §5 (MRP-lotstørrelse) | §2.1, §5.5 | JIT vs. batching innenfor to-bestillingsregimet | Pinedo (2016) |
| Ch04 §3 (fasilitetsplassering UFLP) | §1.4, §5.1 | Teoretisk grunnlag for "trenger det eksterne lageret eksistere?" | Klassisk fasilitetslokaliseringsteori (lærebokstoff) |
| Ch05 §4 (multi-echelon Clark-Scarf) | §5.2 | Koordinering med leverandør gjennom delte prognoser | Silver et al. (2016); Hartmann & Briskorn (2010) |
| Ch07 §3 (slotting / class-based storage) | §5.1 | Hvordan utnytte frigjort kapasitet i butikklager | Pinedo (2016); Slack & Brandon-Jones (2019) |
| Ch07 §5 (integrert lagerplanlegging) | §5.1 | Informasjon som beslutningsstøtte i lagerstyring | Christopher (2016); Slack & Brandon-Jones (2019) |
| Ch08 §5 (grønn forsyningskjede) | §5.4 | Bærekraftsmål 12 og redusert overproduksjon | Christopher (2016) |
| Ch09 §3 (revers-nettverk) | §3.3 | Konseptuelt grunnlag for retur-aggregering | Silver et al. (2016) |
| Ch09 §4 (Weibull-retur) | §3.3 | Begrunnelse for å behandle returer som negative salg fremfor egen tidsserie | Silver et al. (2016) |
| Ch09 §5 (disposisjon-tre) | §2.4, §6.4 | Kvantifisering av $C_o$ (ukurans) i sesongvarer | Petruzzi & Dada (1999) |
| Ch10 §3 (leverandørvalg AHP+TOPSIS) | §5.5, §6.4 | Rammeverk for å vurdere alternative leverandører med kortere ledetid | Puchinger & Raidl (2005) |
| Ch11 §3 (Monte Carlo-risk) | §4.4 (Tabell 4.5), §6.4 | Sensitivitetsanalyse av prisparametere | Vose (2008); Efron & Tibshirani (1993) |
| Ch11 §4 (robust optimization) | §6.4 | Minimax regret-strategi når SARIMA-prognose har høy varians | Vose (2008) |
| Ch11 §5 (stresstest) | §6.4 | Modellens robusthet under sjokk (lockdown, leverandørbrudd) | Vose (2008) |

## Pensumkompendiets struktur

LOG650-pensumet (`003_referanser/Kompendium/`) er strukturert som 33 selvstendige Python-prosjekter, ett per kapittelseksjon. Hvert prosjekt følger samme struktur:

```
chXX-secYY-<emne>/
├── README.md                  # Kort beskrivelse
├── pyproject.toml             # Avhengigheter (uv-styrt)
├── data/                      # Datasett (genereres typisk i step01)
├── output/                    # Figurer og JSON-resultater
└── src/
    ├── step01_datainnsamling.py
    ├── step02_<analyse>.py
    ├── ...
    └── stepNN_anbefaling.py
```

Vår egen `006_analysis/`-pipeline følger samme prinsipper: hver fase er et selvstendig skript, dataflyten skjer via filer (CSV/JSON) på disk, og resultatene kan inspiseres uavhengig av om alle stegene er kjørt. For studenter eller forskere som ønsker å utvide oppgaven, er det dermed en direkte sammenheng mellom vår kode og pensumkodens organisering, hvilket gjør det enklere å adoptere f.eks. ARIMAX-implementasjonen fra Ch01 §4 eller multi-echelon-modellen fra Ch05 §4.
