"""Tornado-diagram for sensitivitet av Q* (vår 2025) for endringer i p, w, s.

Visualiserer Tabell 4.5 i rapporten. Hver parameter får én rad. Stolpen
strekker seg fra "lavt-scenario" til "høyt-scenario" verdi for Q*, sentrert
rundt basisscenarioet (5 975 par).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("013_gjennomforing/visuals")
OUT.mkdir(parents=True, exist_ok=True)

# Sensitivitet fra Tabell 4.5: (parameter, lavt-Q, høyt-Q, lav-label, høy-label)
basis_Q = 5975
scenarios = [
    ("Restverdi s", 5786, 6312, "s = 200", "s = 550"),
    ("Margin p",    5866, 6080, "p = 1 000", "p = 1 500"),
]

# Sorter etter total spennvidde (mest sensitiv øverst i tornado)
scenarios = sorted(scenarios, key=lambda x: x[2] - x[1], reverse=True)

fig, ax = plt.subplots(figsize=(9.5, 3.6))

y_positions = np.arange(len(scenarios))
bar_height = 0.55

colors_low = "#c0392b"   # rødlig for "lav-scenario"
colors_high = "#27ae60"  # grønnlig for "høy-scenario"

for i, (name, low_q, high_q, low_lab, high_lab) in enumerate(scenarios):
    # Stolpen til venstre for basis (lavt-scenario)
    ax.barh(i, basis_Q - low_q, left=low_q, height=bar_height,
            color=colors_low, alpha=0.85, edgecolor="white")
    # Stolpen til høyre for basis (høyt-scenario)
    ax.barh(i, high_q - basis_Q, left=basis_Q, height=bar_height,
            color=colors_high, alpha=0.85, edgecolor="white")
    # Tekstetiketter
    ax.text(low_q - 10, i, low_lab, ha="right", va="center", fontsize=9, color="#444")
    ax.text(high_q + 10, i, high_lab, ha="left", va="center", fontsize=9, color="#444")
    # Verdier på stolpene
    pct_low = (low_q - basis_Q) / basis_Q * 100
    pct_high = (high_q - basis_Q) / basis_Q * 100
    ax.text((low_q + basis_Q) / 2, i, f"{pct_low:+.1f}%",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text((high_q + basis_Q) / 2, i, f"{pct_high:+.1f}%",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

# Vertikal linje for basisscenarioet
ax.axvline(basis_Q, color="black", linewidth=1.6, linestyle="-")
ax.text(basis_Q, len(scenarios) - 0.2, f"Basis Q* = {basis_Q}",
        ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_yticks(y_positions)
ax.set_yticklabels([s[0] for s in scenarios], fontsize=10)
ax.set_xlabel("Optimal bestilling $Q^*$ (par sko, vår 2025)", fontsize=10)
ax.set_title("Sensitivitet av $Q^*$ for endringer i newsvendor-parametere",
             fontsize=12, fontweight="bold")
ax.set_xlim(basis_Q - 350, basis_Q + 500)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
# Fjern y-akse-streken for renere look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "sensitivity_tornado.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Lagret: {OUT / 'sensitivity_tornado.png'}")
