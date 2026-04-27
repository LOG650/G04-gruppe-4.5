# 002_meetings – Møtereferater

Gruppen møtes fast to ganger pr uke:

- **Søndag** – ukeplanlegging
- **Onsdag** – statusmøte / midtveissjekk

## Konvensjon for filnavn

Hver møterapport lagres som én fil i denne mappen, med filnavnet:

```
YYYY-MM-DD_<dag>.md
```

Eksempler:
- `2026-04-26_sondag.md`
- `2026-04-29_onsdag.md`

Bruk **uten norske bokstaver i filnavn** (sondag, ikke søndag) for å unngå Windows/Git-rare. Innholdet i fila kan selvfølgelig bruke æ, ø, å.

## Mal

Bruk `_mal_motereferat.md` som utgangspunkt. Kopier den, bytt navn til riktig dato, og fyll inn.

## Vedlegg

Hvis et møte trenger vedlegg (skjermbilder, eksterne PDF-er, store tabeller), legg dem i en undermappe med samme navn som møtefilen, men uten `.md`-endelsen:

```
2026-04-29_onsdag.md            # selve referatet
2026-04-29_onsdag/              # vedlegg
  ├── skjermbilde.png
  └── tilbud.pdf
```

For enkle møter trenger man ikke vedleggsmappen.
