"""Flytdiagram for pipelinen: PDF-rådata → strukturert data → SARIMA → newsvendor → Q*.

Bygges som matplotlib-figur slik at den passer rapportens visuelle stil.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("013_gjennomforing/visuals")
OUT.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, label, sub=None, color="#e9f1f8", border="#1F6587"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=color, edgecolor=border, linewidth=1.6,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2 + (0.07 if sub else 0),
            label, ha="center", va="center", fontsize=10, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.18,
                sub, ha="center", va="center", fontsize=8, color="#555",
                style="italic")


def draw_arrow(ax, x1, y1, x2, y2, label=None):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=18, color="#1F6587", linewidth=1.8,
    )
    ax.add_patch(arrow)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.10,
                label, ha="center", va="bottom", fontsize=8, color="#444",
                style="italic")


fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.set_aspect("equal")
ax.axis("off")

# Rad 1: Datafangst
draw_box(ax, 0.1, 3.4, 2.1, 1.0,
         "PDF-rådata", sub="~1 100 Z-rapporter",
         color="#fdebd0", border="#a04000")
draw_box(ax, 2.6, 3.4, 2.1, 1.0,
         "PDF-parsing", sub="pdfplumber, koord.",
         color="#fdebd0", border="#a04000")
draw_box(ax, 5.1, 3.4, 2.1, 1.0,
         "Regex-validering", sub="^\\d{6}, returer",
         color="#fdebd0", border="#a04000")
draw_box(ax, 7.6, 3.4, 2.1, 1.0,
         "Aggregering", sub="dag → måned",
         color="#fdebd0", border="#a04000")
draw_box(ax, 10.1, 3.4, 1.7, 1.0,
         "CSV", sub="36 mnd-obs.",
         color="#d5f5e3", border="#196f3d")

draw_arrow(ax, 2.2, 3.9, 2.6, 3.9)
draw_arrow(ax, 4.7, 3.9, 5.1, 3.9)
draw_arrow(ax, 7.2, 3.9, 7.6, 3.9)
draw_arrow(ax, 9.7, 3.9, 10.1, 3.9)

# Pil ned (CSV → SARIMA)
draw_arrow(ax, 10.95, 3.4, 10.95, 2.6, label="FS1 → FS2")

# Rad 2: Modellering
draw_box(ax, 8.5, 1.6, 3.3, 1.0,
         "SARIMA-modell", sub="(1,1,1)(1,1,1)$_{12}$",
         color="#e8daef", border="#5b2c6f")
draw_box(ax, 4.5, 1.6, 3.3, 1.0,
         "Newsvendor-formel", sub=r"$Q^* = \mu + z_\alpha\,\sigma$",
         color="#e8daef", border="#5b2c6f")
draw_arrow(ax, 8.5, 2.1, 7.8, 2.1, label=r"$\mu, \sigma$")

# Pil ned (newsvendor → Q*)
draw_arrow(ax, 6.15, 1.6, 6.15, 0.9, label="FS3 → FS4")

# Rad 3: Output
draw_box(ax, 3.4, 0.05, 5.5, 0.85,
         "Bestillingsanbefaling $Q^*$ + økonomisk evaluering",
         sub="vår 2025: 5 975 par   |   høst 2025: 3 468 par",
         color="#d6eaf8", border="#1F6587")

# Tilbake-arrow for sigma-feedback (in-sample 2023-2024)
draw_arrow(ax, 10.0, 2.6, 10.0, 3.4)
ax.text(10.3, 3.0, "in-sample\nres. 2023-2024",
        ha="left", va="center", fontsize=7, color="#5b2c6f", style="italic")

# Tittel
ax.text(6, 4.8, "Pipeline: fra ustrukturert PDF til bestillingsanbefaling",
        ha="center", va="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "pipeline_flytdiagram.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Lagret: {OUT / 'pipeline_flytdiagram.png'}")
