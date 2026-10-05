"""Schematic of the abstracted Tongping Aviation Town layout used by the simulator
: five boarding spots, their demand groups and the moving hotspot."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures", "fig_study_area.png")

SPOTS = [  # name, x, y, group
    ("S1 Town entrance", 1.6, 1.3, "morning"),
    ("S2 Museum", 3.3, 1.9, "morning"),
    ("S3 Ridge overlook A", 5.9, 4.3, "afternoon"),
    ("S4 Ridge overlook B", 7.6, 3.4, "afternoon"),
    ("S5 Valley viewpoint", 6.4, 1.4, "afternoon"),
]
COL = {"morning": "#2471a3", "afternoon": "#ca6f1e"}


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 9.4)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    # terrain: ridge band and river
    ax.fill_between([4.6, 9.3], [3.2, 2.4], [5.3, 5.3], color="#d5e8d4", zorder=0)
    ax.text(8.6, 5.0, "ridge", fontsize=8, color="#5b7f5b", ha="right")
    ax.plot([4.2, 5.4, 6.4, 7.5, 9.2], [0.2, 0.9, 1.0, 0.7, 1.2], color="#7fb3d5", lw=6,
            solid_capstyle="round", zorder=0)
    ax.text(8.9, 0.75, "river valley", fontsize=8, color="#2e86c1", ha="right")
    # town and airport
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 3.6, 2.3, boxstyle="round,pad=0.05",
                                fc="#f2f3f4", ec="#aab7b8", zorder=0))
    ax.text(0.45, 2.4, "Tongping Aviation Town", fontsize=8, color="#566573")
    ax.add_patch(FancyBboxPatch((0.4, 3.6), 2.6, 0.9, boxstyle="round,pad=0.05",
                                fc="#eaecee", ec="#808b96", zorder=0))
    ax.text(1.7, 4.05, "Tongping general-aviation\nairport", fontsize=7.5, ha="center",
            va="center", color="#34495e")
    # spots
    for name, x, y, g in SPOTS:
        ax.scatter([x], [y], s=260, color=COL[g], edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(x, y + 0.36, name, fontsize=8, ha="center", va="bottom", zorder=4)
    # moving hotspot
    arr = FancyArrowPatch((3.6, 2.2), (5.6, 3.9), connectionstyle="arc3,rad=-0.25",
                          arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#7f8c8d",
                          linestyle="--", zorder=2)
    ax.add_patch(arr)
    ax.text(4.0, 3.35, "demand moves\nmorning $\\rightarrow$ afternoon", fontsize=7.5,
            color="#566573", ha="center")
    # legend
    for k, (lab, g) in enumerate((("morning-peak group (peak at 30% of the day)", "morning"),
                                  ("afternoon-peak group (peak at 70% of the day)", "afternoon"))):
        ax.scatter([0.55], [0.0 - 0.0 + 5.15 - 0.32 * k], s=60, color=COL[g])
        ax.text(0.75, 5.15 - 0.32 * k, lab, fontsize=7.5, va="center")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
