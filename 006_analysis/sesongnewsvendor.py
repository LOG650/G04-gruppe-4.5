"""
Sesongnewsvendor for Skoringen Råholt.

Skobransjen plasserer 2 bestillinger pr år (vår + høst). Optimaliseringen
ligger i hvor mye som bestilles. Dette skriptet implementerer pensumets
newsvendor-formel (Ch05 §5):

    Q* = mu + z_alpha * sigma

der mu er SARIMA-prognose for sesongen, sigma er prognoseusikkerheten,
og z_alpha bestemmes av kritisk forhold (p - w) / (p - s).

Sammenligning:
  - Naiv strategi:  Q_naiv = fjorårets sesongsalg
  - Newsvendor:     Q*     = SARIMA-prognose + sikkerhetslager
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA_DIR = Path("004_data")
OUT_DIR = Path("013_gjennomforing/visuals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Økonomiske parametere (estimat for skobransjen) -----------------------
P_RETAIL = 1200      # Utsalgspris pr par (estimert snitt)
W_WHOLESALE = 600    # Innkjøp fra leverandør (estimert snitt, ca 50 % av p)
S_SALVAGE = 400      # Restverdi etter sesong (rabattsalg ~ 33 % av p)
HOLDING_COST = 6     # NOK pr par pr måned i lager

# Kritisk forhold (Ch05 §5 newsvendor)
CR = (P_RETAIL - W_WHOLESALE) / (P_RETAIL - S_SALVAGE)
Z_ALPHA = stats.norm.ppf(CR)

# -- Last data -------------------------------------------------------------
monthly = pd.read_csv(DATA_DIR / "skoringen_monthly_clean.csv")
monthly["Dato"] = pd.to_datetime(
    monthly["År"].astype(str) + "-" + monthly["Måned"].astype(str) + "-01"
)
monthly = monthly.set_index("Dato").sort_index()

forecast = pd.read_csv(DATA_DIR / "forecast_results.csv", index_col=0)
forecast.index = pd.to_datetime(forecast.index)

# Prognoseusikkerhet (RMSE pr måned, estimert fra SARIMA-feilen i 2025)
residualer = forecast["Faktisk"] - forecast["SARIMA"]
sigma_pr_mnd = float(np.sqrt(np.mean(residualer ** 2)))


def sesong_prognose(year: int, season: str, source: str = "SARIMA") -> float:
    """Sum prognose for vår (mar-aug) eller høst (sep-feb)."""
    if season == "vår":
        months = [(year, m) for m in range(3, 9)]
    else:
        months = [(year, m) for m in range(9, 13)] + [(year + 1, m) for m in range(1, 3)]
    total = 0.0
    for y, m in months:
        d = pd.Timestamp(year=y, month=m, day=1)
        if d in forecast.index:
            total += forecast.loc[d, source]
        elif d in monthly.index:
            total += monthly.loc[d, "Antall_par"]
    return total


def sesong_faktisk(year: int, season: str) -> float:
    """Sum faktisk salg for sesongen (måneder vi har data for)."""
    if season == "vår":
        months = [(year, m) for m in range(3, 9)]
    else:
        months = [(year, m) for m in range(9, 13)] + [(year + 1, m) for m in range(1, 3)]
    total = 0.0
    for y, m in months:
        d = pd.Timestamp(year=y, month=m, day=1)
        if d in monthly.index:
            total += monthly.loc[d, "Antall_par"]
    return total


def newsvendor_q(mu: float, n_months: int) -> tuple[float, float]:
    """Returner (Q*, sikkerhetslager) for en sesong med n_months måneder."""
    sigma = sigma_pr_mnd * np.sqrt(n_months)
    safety = Z_ALPHA * sigma
    q_star = mu + safety
    return q_star, safety


def naiv_q(year: int, season: str) -> float:
    """Naiv strategi: bestiller fjorårets sesongsalg."""
    return sesong_faktisk(year - 1, season)


def kostnad(q: float, faktisk: float) -> dict:
    """Beregn realisert økonomi for én sesongbestilling."""
    solgt = min(q, faktisk)
    overskudd = max(0.0, q - faktisk)
    mangel = max(0.0, faktisk - q)

    omsetning = P_RETAIL * solgt + S_SALVAGE * overskudd
    innkjop = W_WHOLESALE * q
    bruttoresultat = omsetning - innkjop
    tapt_salg = (P_RETAIL - W_WHOLESALE) * mangel  # alternativkostnad

    return {
        "Q_bestilt": q,
        "Faktisk_salg": faktisk,
        "Solgt": solgt,
        "Overskudd": overskudd,
        "Mangel": mangel,
        "Omsetning": omsetning,
        "Innkjop": innkjop,
        "Bruttoresultat": bruttoresultat,
        "Tapt_salg_alt_kost": tapt_salg,
        "Netto_etter_alt_kost": bruttoresultat - tapt_salg,
    }


# -- Hovedanalyse ----------------------------------------------------------
print("=" * 70)
print("SESONGNEWSVENDOR – SKORINGEN RÅHOLT")
print("=" * 70)
print(f"\nØkonomiske parametere (estimat):")
print(f"  p (utsalgspris)        = {P_RETAIL} NOK/par")
print(f"  w (innkjøp leverandør) = {W_WHOLESALE} NOK/par")
print(f"  s (restverdi)          = {S_SALVAGE} NOK/par")
print(f"  Kritisk forhold        = (p-w)/(p-s) = {CR:.4f}")
print(f"  z_alpha                = {Z_ALPHA:.4f}  (servicenivå {CR*100:.1f} %)")
print(f"\nPrognoseusikkerhet (SARIMA på 2025):")
print(f"  RMSE pr måned          = {sigma_pr_mnd:.1f} par")

resultater = []
for year, season, n_mnd in [(2025, "vår", 6), (2025, "høst", 6)]:
    print(f"\n--- {season.upper()}SESONGEN {year} ({n_mnd} mnd) ---")

    mu_sarima = sesong_prognose(year, season)
    faktisk = sesong_faktisk(year, season)
    q_news, safety = newsvendor_q(mu_sarima, n_mnd)
    q_n = naiv_q(year, season)

    print(f"  SARIMA-prognose (mu)    : {mu_sarima:7.0f} par")
    print(f"  Sesong-sigma            : {sigma_pr_mnd*np.sqrt(n_mnd):7.1f} par")
    print(f"  Sikkerhetslager (z*sig) : {safety:7.1f} par")
    print(f"  Q*_newsvendor           : {q_news:7.0f} par")
    print(f"  Q_naiv (fjorårssalg)    : {q_n:7.0f} par")
    print(f"  Faktisk salg sesong     : {faktisk:7.0f} par")

    res_n = kostnad(q_n, faktisk)
    res_q = kostnad(q_news, faktisk)
    print(f"  Naiv strategi:")
    print(f"    Mangel                : {res_n['Mangel']:7.0f} par   "
          f"Tapt salg-kost: {res_n['Tapt_salg_alt_kost']:>10,.0f} NOK")
    print(f"    Overskudd             : {res_n['Overskudd']:7.0f} par   "
          f"Bruttoresultat: {res_n['Bruttoresultat']:>10,.0f} NOK")
    print(f"  Newsvendor-strategi:")
    print(f"    Mangel                : {res_q['Mangel']:7.0f} par   "
          f"Tapt salg-kost: {res_q['Tapt_salg_alt_kost']:>10,.0f} NOK")
    print(f"    Overskudd             : {res_q['Overskudd']:7.0f} par   "
          f"Bruttoresultat: {res_q['Bruttoresultat']:>10,.0f} NOK")

    resultater.append({
        "år": year, "sesong": season, "antall_mnd": n_mnd,
        "mu_sarima": mu_sarima, "sigma_sesong": sigma_pr_mnd*np.sqrt(n_mnd),
        "Q_naiv": q_n, "Q_newsvendor": q_news, "faktisk": faktisk,
        "naiv": res_n, "newsvendor": res_q,
    })

# -- Aggregert årseffekt ---------------------------------------------------
sum_naiv_brutto = sum(r["naiv"]["Bruttoresultat"] for r in resultater)
sum_news_brutto = sum(r["newsvendor"]["Bruttoresultat"] for r in resultater)
sum_naiv_alt = sum(r["naiv"]["Tapt_salg_alt_kost"] for r in resultater)
sum_news_alt = sum(r["newsvendor"]["Tapt_salg_alt_kost"] for r in resultater)
sum_naiv_netto = sum_naiv_brutto - sum_naiv_alt
sum_news_netto = sum_news_brutto - sum_news_alt

print("\n" + "=" * 70)
print("ÅRSEFFEKT 2025 (vår + høst)")
print("=" * 70)
print(f"  Bruttoresultat naiv       : {sum_naiv_brutto:>12,.0f} NOK")
print(f"  Bruttoresultat newsvendor : {sum_news_brutto:>12,.0f} NOK")
print(f"  Differanse brutto         : {sum_news_brutto - sum_naiv_brutto:>+12,.0f} NOK")
print()
print(f"  Tapt salg-kost naiv       : {sum_naiv_alt:>12,.0f} NOK")
print(f"  Tapt salg-kost newsvendor : {sum_news_alt:>12,.0f} NOK")
print(f"  Reduksjon i tapt salg     : {sum_naiv_alt - sum_news_alt:>+12,.0f} NOK")
print()
print(f"  Netto etter alt-kost naiv      : {sum_naiv_netto:>12,.0f} NOK")
print(f"  Netto etter alt-kost newsvendor: {sum_news_netto:>12,.0f} NOK")
gevinst = sum_news_netto - sum_naiv_netto
print(f"  TOTAL GEVINST                  : {gevinst:>+12,.0f} NOK")
print(f"  Relativ gevinst                : {gevinst/abs(sum_naiv_netto)*100:+.1f} %")

# -- Lagernivåkurver: hvordan ser sesongbestilling ut over tid -------------
def lagerprofil(q_vaar: float, q_host: float, faktisk_serie: pd.Series) -> list[float]:
    """Simuler lagernivå månedlig gitt 2 bestillinger pr år."""
    inv = 200  # startlager
    history = []
    for date, sale in faktisk_serie.items():
        # Bestilling ankommer i begynnelsen av mar (vår) og sep (høst)
        if date.month == 3:
            inv += q_vaar
        elif date.month == 9:
            inv += q_host
        sold = min(inv, sale)
        inv -= sold
        history.append(inv)
    return history


faktisk_2025 = forecast["Faktisk"]
# Trekk ut 2025-bestillinger
res_2025_v = next(r for r in resultater if r["sesong"] == "vår")
res_2025_h = next(r for r in resultater if r["sesong"] == "høst")

inv_naiv = lagerprofil(res_2025_v["Q_naiv"], res_2025_h["Q_naiv"], faktisk_2025)
inv_news = lagerprofil(res_2025_v["Q_newsvendor"], res_2025_h["Q_newsvendor"], faktisk_2025)

# -- Figur 1: Lagerprofil -------------------------------------------------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.plot(faktisk_2025.index, inv_naiv, label="Naiv (fjorårssalg)",
        color="#c0392b", marker="s", linewidth=2.4)
ax.plot(faktisk_2025.index, inv_news, label="Newsvendor (SARIMA + sikkerhetslager)",
        color="#27ae60", marker="o", linewidth=2.4)
ax.axhline(3000, color="black", linestyle="--", linewidth=1.4,
           label="Lagerkapasitet (3 000 par)")
ax.set_title("Lagernivå 2025 ved 2 sesongbestillinger pr år",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Måned")
ax.set_ylabel("Antall par på lager")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "inventory_newsvendor_2025.png", dpi=300)
plt.close()
print(f"\nFigur lagret: {OUT_DIR / 'inventory_newsvendor_2025.png'}")


# -- Figur 2: Newsvendor profittkurve for vårsesongen ---------------------
mu_v = res_2025_v["mu_sarima"]
sig_v = sigma_pr_mnd * np.sqrt(6)
Q_range = np.linspace(mu_v - 3 * sig_v, mu_v + 3 * sig_v, 400)


def E_profitt(Q: float, mu: float, sigma: float) -> float:
    z = (Q - mu) / sigma
    L = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))  # standard tap
    E_solgt = mu - sigma * L
    E_overskudd = Q - E_solgt
    return P_RETAIL * E_solgt + S_SALVAGE * E_overskudd - W_WHOLESALE * Q


profitt = np.array([E_profitt(q, mu_v, sig_v) for q in Q_range])
q_star = mu_v + Z_ALPHA * sig_v

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(Q_range, profitt / 1000, color="#1F6587", linewidth=2.2,
        label=r"$E[\Pi(Q)]$ (vårsesong 2025)")
ax.axvline(q_star, color="#961D1C", linestyle="--", linewidth=1.7,
           label=f"$Q^* = {q_star:.0f}$")
ax.axvline(mu_v, color="#307453", linestyle=":", linewidth=1.4,
           label=fr"$\mu_{{SARIMA}} = {mu_v:.0f}$")
ax.set_xlabel(r"$Q$  (bestilt antall par)")
ax.set_ylabel(r"$E[\Pi(Q)]$  (tusen NOK)")
ax.set_title("Newsvendor: forventet profitt for vårsesongen 2025",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower center")
plt.tight_layout()
plt.savefig(OUT_DIR / "newsvendor_profit_curve.png", dpi=300)
plt.close()
print(f"Figur lagret: {OUT_DIR / 'newsvendor_profit_curve.png'}")


# -- Lagre tall til JSON ---------------------------------------------------
out = {
    "parametre": {
        "p": P_RETAIL, "w": W_WHOLESALE, "s": S_SALVAGE,
        "kritisk_forhold": round(CR, 4), "z_alpha": round(Z_ALPHA, 4),
        "rmse_pr_mnd": round(sigma_pr_mnd, 2),
    },
    "sesonger": [
        {
            "år": r["år"], "sesong": r["sesong"],
            "mu_sarima": round(r["mu_sarima"], 0),
            "sigma_sesong": round(r["sigma_sesong"], 1),
            "Q_naiv": round(r["Q_naiv"], 0),
            "Q_newsvendor": round(r["Q_newsvendor"], 0),
            "faktisk": round(r["faktisk"], 0),
            "naiv_brutto": round(r["naiv"]["Bruttoresultat"], 0),
            "naiv_tapt_salg": round(r["naiv"]["Tapt_salg_alt_kost"], 0),
            "naiv_overskudd": round(r["naiv"]["Overskudd"], 0),
            "naiv_mangel": round(r["naiv"]["Mangel"], 0),
            "news_brutto": round(r["newsvendor"]["Bruttoresultat"], 0),
            "news_tapt_salg": round(r["newsvendor"]["Tapt_salg_alt_kost"], 0),
            "news_overskudd": round(r["newsvendor"]["Overskudd"], 0),
            "news_mangel": round(r["newsvendor"]["Mangel"], 0),
        }
        for r in resultater
    ],
    "år_sum": {
        "naiv_brutto": round(sum_naiv_brutto, 0),
        "newsvendor_brutto": round(sum_news_brutto, 0),
        "naiv_tapt_salg": round(sum_naiv_alt, 0),
        "newsvendor_tapt_salg": round(sum_news_alt, 0),
        "naiv_netto": round(sum_naiv_netto, 0),
        "newsvendor_netto": round(sum_news_netto, 0),
        "gevinst": round(gevinst, 0),
        "gevinst_pst": round(gevinst / abs(sum_naiv_netto) * 100, 2),
        "maks_lager_naiv": round(max(inv_naiv), 0),
        "maks_lager_newsvendor": round(max(inv_news), 0),
    }
}

with open(OUT_DIR.parent / "newsvendor_resultater.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nResultater lagret: {OUT_DIR.parent / 'newsvendor_resultater.json'}")
