"""Bootstrap-konfidensintervall for økonomisk gevinst (newsvendor vs naiv).

Metodisk grunnlag:
- Vi har SARIMA-prognose (mu) og prognoseusikkerhet (sigma) for hver sesong.
- Bestillingene Q_news = mu + z*sigma og Q_naiv = fjorårssalg er deterministiske
  gitt prognosen og fjorårssalget.
- Den realiserte etterspørselen i en framtidig sesong er en stokastisk variabel
  D ~ Normal(mu, sigma) (Box et al., 2015; Efron & Tibshirani, 1993).
- Gevinsten naivvendor − naiv er en funksjon av D, så den arver fordelingen
  fra D.
- Vi simulerer N realisasjoner D_i for begge sesonger, beregner gevinst per
  realisasjon, og rapporterer 2.5-/97.5-persentilen som 95 % konfidensintervall.

Skriptet bruker ikke faktisk salg for 2025 – det estimerer hva *fordelingen*
av gevinster vil se ut framover, gitt SARIMA-prognosen og prognoseusikkerheten.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Reproduserbar tilfeldighet
rng = np.random.default_rng(seed=20260519)

# -- Last parametere fra hovedanalysen ------------------------------------
results_path = Path("013_gjennomforing/newsvendor_resultater.json")
with open(results_path, encoding="utf-8") as f:
    base = json.load(f)

P = base["parametre"]["p"]
W = base["parametre"]["w"]
S = base["parametre"]["s"]
SIGMA_MND = base["parametre"]["rmse_pr_mnd"]

# Hent sesongdata
seasons = base["sesonger"]


def realisert_okonomi(q: float, d: float, p: float, w: float, s: float) -> float:
    """Realisert nettoresultat for en sesongbestilling Q gitt etterspørsel D.

    Nettoresultat = bruttoresultat − alternativkostnad ved tapt salg.
    """
    solgt = min(q, d)
    overskudd = max(0.0, q - d)
    mangel = max(0.0, d - q)
    brutto = p * solgt + s * overskudd - w * q
    tapt = (p - w) * mangel
    return brutto - tapt


def simuler_gevinst(n_iter: int = 10000) -> dict:
    """Bootstrap gevinstfordelingen via parametrisk sampling fra SARIMA-fordelingen."""
    gevinster = np.zeros(n_iter)
    netto_naiv = np.zeros(n_iter)
    netto_news = np.zeros(n_iter)

    for i in range(n_iter):
        total_naiv = 0.0
        total_news = 0.0
        for s_data in seasons:
            mu = s_data["mu_sarima"]
            sigma = s_data["sigma_sesong"]
            q_naiv = s_data["Q_naiv"]
            q_news = s_data["Q_newsvendor"]
            # Trekk en realisert etterspørsel for denne sesongen
            d = rng.normal(loc=mu, scale=sigma)
            d = max(0.0, d)  # etterspørsel kan ikke være negativ
            n_naiv = realisert_okonomi(q_naiv, d, P, W, S)
            n_news = realisert_okonomi(q_news, d, P, W, S)
            total_naiv += n_naiv
            total_news += n_news
        netto_naiv[i] = total_naiv
        netto_news[i] = total_news
        gevinster[i] = total_news - total_naiv

    return {
        "n_iter": n_iter,
        "mean_gevinst": float(np.mean(gevinster)),
        "median_gevinst": float(np.median(gevinster)),
        "std_gevinst": float(np.std(gevinster, ddof=1)),
        "ki_95": (float(np.percentile(gevinster, 2.5)), float(np.percentile(gevinster, 97.5))),
        "ki_90": (float(np.percentile(gevinster, 5.0)), float(np.percentile(gevinster, 95.0))),
        "andel_positiv_gevinst": float(np.mean(gevinster > 0)),
        "andel_gevinst_over_100k": float(np.mean(gevinster > 100_000)),
        "andel_gevinst_over_500k": float(np.mean(gevinster > 500_000)),
        "samples": gevinster.tolist()[:1000],  # lagre 1000 for evt. plotting
    }


if __name__ == "__main__":
    print("Bootstrap-konfidensintervall for årlig gevinst (newsvendor vs naiv)")
    print("=" * 70)
    print(f"Iterasjoner: 10 000")
    print(f"Seed: 20260519 (reproduserbart)")
    print(f"Parametere: p={P}, w={W}, s={S}, sigma_mnd={SIGMA_MND:.1f}")
    print()

    out = simuler_gevinst(n_iter=10_000)

    print(f"Forventet gevinst (gjennomsnitt over 10 000 iter): {out['mean_gevinst']:>+12,.0f} NOK")
    print(f"Median gevinst:                                    {out['median_gevinst']:>+12,.0f} NOK")
    print(f"Standardavvik på gevinsten:                        {out['std_gevinst']:>12,.0f} NOK")
    print()
    print(f"95 % konfidensintervall: [{out['ki_95'][0]:>+12,.0f}, {out['ki_95'][1]:>+12,.0f}] NOK")
    print(f"90 % konfidensintervall: [{out['ki_90'][0]:>+12,.0f}, {out['ki_90'][1]:>+12,.0f}] NOK")
    print()
    print(f"Sannsynlighet for positiv gevinst:    {out['andel_positiv_gevinst']*100:>5.1f} %")
    print(f"Sannsynlighet for gevinst > 100k NOK: {out['andel_gevinst_over_100k']*100:>5.1f} %")
    print(f"Sannsynlighet for gevinst > 500k NOK: {out['andel_gevinst_over_500k']*100:>5.1f} %")

    # Lagre til JSON
    out_path = Path("013_gjennomforing/bootstrap_gevinst.json")
    # Ikke lagre samples i fil (for stor)
    out_clean = {k: v for k, v in out.items() if k != "samples"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_clean, f, indent=2, ensure_ascii=False)
    print(f"\nResultater lagret: {out_path}")
