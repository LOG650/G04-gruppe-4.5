# Beslutningslogg

Logg over faglige og praktiske beslutninger som påvirker prosjektets retning. Hver oppføring skal kunne stå alene som dokumentasjon av *hva* som ble besluttet, *hvorfor*, og *hvem* som var involvert.

## Mal for nye oppføringer

```markdown
### YYYY-MM-DD – [Kort tittel på beslutningen]
**Beslutning:** [Hva som ble besluttet]
**Begrunnelse:** [Hvorfor – hvilke alternativer ble vurdert]
**Involverte:** [Hvem var med på beslutningen]
**Konsekvens:** [Hva endrer dette i prosjektet]
```

---

## Logg

### 2026-04-27 – Reformulering av optimaliseringsmodell
**Beslutning:** Den opprinnelige Just-in-Time-modellen (månedlige bestillinger) er erstattet med en sesongnewsvendor-modell (to bestillinger pr år, optimalisert mengde).
**Begrunnelse:** Etter avklaring med Marit Stoksflod ble det klart at Skoringen er bundet til to leverandørbestillinger pr år (vår + høst). JIT-narrativet var dermed urealistisk. Pensumets Ch05 §5 (Newsvendor) gir et bedre rammeverk for problemet.
**Involverte:** Gruppen + Marit Stoksflod (kunde).
**Konsekvens:** Hovedrapporten, proposal, brukermanual og pipeline-dokumentasjon er omskrevet. Estimert årlig effekt: +713 621 NOK (+14,4 % nettoresultat) under antatte enhetspriser.

### 2026-04-27 – Valg av Newsvendor-parametere som estimat
**Beslutning:** Bruke antatte enhetspriser ($p=1\,200$, $w=600$, $s=400$ NOK/par) i Newsvendor-modellen, eksplisitt markert som estimat.
**Begrunnelse:** Faktiske priser ikke tilgjengelig på dette tidspunktet. Sensitivitetsanalyse i kapittel 4.4.1 viser at konklusjonen er robust over rimelige variasjoner.
**Involverte:** Gruppen.
**Konsekvens:** Tabeller og figurer i rapporten merkes med "estimat". Ekte priser bør hentes fra Skoringens regnskap før endelig levering.

### 2026-04-27 – Mappestruktur konsolidert
**Beslutning:** All mappenavngivning bruker `nnn_lowercase` (snake_case). Mapper med mellomrom omdøpt; tomme/duplikate mapper slettet.
**Begrunnelse:** Inkonsistens skapte feil i Python-stier og forvirrende navigasjon. AGENTS.md krever konsistens.
**Involverte:** Gruppen.
**Konsekvens:** README oppdatert. Python-skript har fått korrigerte stier.
