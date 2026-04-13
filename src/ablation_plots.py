"""Plots for the causal-ablation experiment.

  - ablation_families.png  Per-family: ablated accuracy vs size-matched
                           random baseline. Highlights the load-bearing
                           family with a red star.
  - ablation_per_class.png Per-class accuracy before vs after ablating
                           each family. Tells us which modulations
                           become confused by which ablation.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def families_bar(data: dict, out: Path):
    fams = sorted(data["classical_families"].keys(),
                  key=lambda k: -data["classical_families"][k]["n_features"])
    n = [data["classical_families"][f]["n_features"] for f in fams]
    acc = [data["classical_families"][f]["ablated_accuracy"] for f in fams]
    rand_m = [data["classical_families"][f]["random_mean"] for f in fams]
    rand_s = [data["classical_families"][f]["random_std"] for f in fams]
    exceeds = [data["classical_families"][f]["exceeds_random"] for f in fams]

    xs = np.arange(len(fams))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    b_abl = ax.bar(xs - w / 2, acc, w, color="#d62728",
                   label="ablated", edgecolor="#222")
    b_rand = ax.bar(xs + w / 2, rand_m, w, yerr=rand_s, capsize=3,
                    color="#888", label=f"random baseline (n={20})",
                    edgecolor="#222")
    ax.axhline(data["baseline_accuracy"], color="#2ca02c", lw=1.2, ls="--",
               alpha=0.7, label="no-ablation baseline")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f}\n(n={k})" for f, k in zip(fams, n)],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("accuracy after ablation")
    ax.set_title("Causal ablation by classical-feature family — "
                 "★ = hurts more than matched random ablation")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower left", fontsize=9)

    for i, star in enumerate(exceeds):
        if star:
            ax.text(xs[i] - w / 2, acc[i] - 0.08, "★",
                    ha="center", va="center", fontsize=22,
                    color="#d62728", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def per_class_grid(data: dict, out: Path):
    """Per-class accuracy for each family ablation — rows=families, cols=classes."""
    fams = sorted(data["classical_families"].keys(),
                  key=lambda k: -data["classical_families"][k]["n_features"])
    classes = data["classes"]

    # Baseline per-class accuracy (from confusion_matrix_baseline)
    cm = np.array(data["confusion_matrix_baseline"])
    base_pc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    rows = []
    labels = []
    for f in fams:
        pc = data["classical_families"][f]["per_class_accuracy"]
        rows.append([pc[c] for c in classes])
        n = data["classical_families"][f]["n_features"]
        star = "★" if data["classical_families"][f]["exceeds_random"] else " "
        labels.append(f"{f} (n={n}) {star}")

    mat = np.array(rows)
    # Delta vs baseline — clearer visualization
    delta = mat - base_pc[None, :]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(delta, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(fams)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            d = delta[i, j]
            txt = f"{d:+.2f}"
            color = "white" if abs(d) > 0.45 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Per-class accuracy delta vs baseline — "
                 "which modulations each ablation confuses")
    fig.colorbar(im, ax=ax, label="Δ accuracy vs baseline")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def main():
    data = json.loads(Path("results/ablation.json").read_text())
    families_bar(data, Path("results/ablation_families.png"))
    per_class_grid(data, Path("results/ablation_per_class.png"))


if __name__ == "__main__":
    main()
