"""Aggregate per-seed runs from run_multi_seed into median/IQR summaries
and produce the headline multi-seed plots.

Reads results/multi_seed.json and writes:
  - results/multi_interpretability.png
        SAE vs PCA mean max-|r| per seed, plus median bars
  - results/multi_family_class.png
        median (over seeds) per-family per-class accuracy delta,
        with a per-cell count of how many seeds saw that (family, class)
        pair as load-bearing
  - results/multi_summary.json
        derived numeric summaries that feed the report

The "robust" result we're looking for: which (family, class) pairs
consistently show large negative deltas across seeds. That tells us
which causal structure is seed-independent.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Minimum per-class accuracy drop (vs baseline) that counts as "this
# family broke this class in this seed".
DROP_THRESHOLD = 0.25


def main():
    data = json.loads(Path("results/multi_seed.json").read_text())
    runs = data["runs"]
    n_seeds = len(runs)

    # 1. SAE vs PCA interpretability across seeds
    sae_mean_r = []
    pca_mean_r = []
    sae_strong = []
    pca_strong = []
    for r in runs:
        a = r["analysis"]
        sae_mean_r.append(a["sae_stats"]["mean_max_r"])
        pca_mean_r.append(a["pca_stats"]["mean_max_r"])
        sae_strong.append(a["sae_stats"]["fraction_strong"])
        pca_strong.append(a["pca_stats"]["fraction_strong"])
    sae_mean_r = np.array(sae_mean_r)
    pca_mean_r = np.array(pca_mean_r)
    sae_strong = np.array(sae_strong)
    pca_strong = np.array(pca_strong)

    print("per-seed mean max |r| (SAE vs PCA):")
    for r, sm, pm, ss, ps in zip(runs, sae_mean_r, pca_mean_r,
                                  sae_strong, pca_strong):
        print(f"  seed {r['seed']}:  SAE {sm:.3f}  PCA {pm:.3f}  "
              f"strong_frac SAE {ss:.3f}  PCA {ps:.3f}")
    print(f"median SAE mean_max_r: {np.median(sae_mean_r):.3f}")
    print(f"median PCA mean_max_r: {np.median(pca_mean_r):.3f}")
    print(f"ratio: {np.median(sae_mean_r) / max(np.median(pca_mean_r), 1e-12):.2f}x")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(n_seeds)
    w = 0.38
    ax.bar(xs - w / 2, sae_mean_r, w, color="#2ca02c",
           label=f"SAE (median {np.median(sae_mean_r):.2f})",
           edgecolor="#222")
    ax.bar(xs + w / 2, pca_mean_r, w, color="#888",
           label=f"PCA (median {np.median(pca_mean_r):.2f})",
           edgecolor="#222")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {r['seed']}" for r in runs])
    ax.set_ylabel("mean max |r| with classical features")
    ax.set_title("SAE vs PCA interpretability across 5 seeds")
    ax.axhline(np.median(sae_mean_r), ls="--", color="#2ca02c", alpha=0.4)
    ax.axhline(np.median(pca_mean_r), ls="--", color="#888", alpha=0.4)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/multi_interpretability.png", dpi=150)
    print("saved → results/multi_interpretability.png")

    # 2. Per-(family, class) median delta and seed-count matrix
    all_families = set()
    all_classes = None
    for r in runs:
        fams = r["ablation"]["classical_families"]
        all_families.update(fams.keys())
        all_classes = r["ablation"]["classes"]

    # Sort families by which shows up in the most seeds (and within that,
    # by median total effect size)
    family_seeds = {f: 0 for f in all_families}
    for r in runs:
        for f in r["ablation"]["classical_families"]:
            family_seeds[f] += 1

    families = sorted(all_families,
                      key=lambda f: (-family_seeds[f], f))

    classes = all_classes
    n_fam = len(families)
    n_cls = len(classes)

    # For each (family, class): list of deltas across seeds
    deltas = np.full((n_fam, n_cls, n_seeds), np.nan)
    for s_idx, r in enumerate(runs):
        fams = r["ablation"]["classical_families"]
        base_cm = np.array(r["ablation"]["confusion_matrix_baseline"])
        base_pc = base_cm.diagonal() / np.maximum(base_cm.sum(axis=1), 1)
        for fi, fam in enumerate(families):
            if fam not in fams:
                continue
            pc = fams[fam]["per_class_accuracy"]
            for ci, c in enumerate(classes):
                deltas[fi, ci, s_idx] = pc[c] - base_pc[ci]

    median_delta = np.nanmedian(deltas, axis=2)
    # count: how many seeds saw this family break this class (drop >= threshold)
    broken_count = (deltas <= -DROP_THRESHOLD).sum(axis=2)
    family_present = (~np.isnan(deltas)).sum(axis=2)

    # Plot median delta heatmap with counts overlaid
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(median_delta, aspect="auto", cmap="RdBu",
                   vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n_cls))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_fam))
    ax.set_yticklabels([f"{f}  (in {family_seeds[f]}/{n_seeds} seeds)"
                         for f in families], fontsize=9)
    for i in range(n_fam):
        for j in range(n_cls):
            d = median_delta[i, j]
            if np.isnan(d):
                continue
            bc = broken_count[i, j]
            fp = family_present[i, j]
            if abs(d) > 0.10:
                color = "white" if abs(d) > 0.45 else "black"
                ax.text(j, i, f"{d:+.2f}\n{bc}/{fp}", ha="center", va="center",
                        fontsize=6, color=color)
    ax.set_title(
        f"Per-(family, class) median accuracy delta across {n_seeds} seeds\n"
        f"Each cell: median Δ (top) and seed-count 'broken' "
        f"(≥{int(100*DROP_THRESHOLD)}% drop, bottom)"
    )
    fig.colorbar(im, ax=ax, label="median Δ accuracy vs baseline")
    fig.tight_layout()
    fig.savefig("results/multi_family_class.png", dpi=150)
    print("saved → results/multi_family_class.png")

    # 3. Derived summary: which (family, class) pairs are robustly load-bearing?
    robust_pairs = []
    for fi, fam in enumerate(families):
        for ci, c in enumerate(classes):
            seen = family_present[fi, ci]
            broke = broken_count[fi, ci]
            if seen >= 3 and broke >= seen * 0.6:
                robust_pairs.append({
                    "family": fam,
                    "class": c,
                    "seen_in": int(seen),
                    "broke_in": int(broke),
                    "median_delta": float(median_delta[fi, ci]),
                })
    robust_pairs.sort(key=lambda r: (r["median_delta"], -r["broke_in"]))

    print(f"\nrobustly load-bearing (family, class) pairs "
          f"({len(robust_pairs)} found; ≥60% of seeds broke ≥25%):")
    for r in robust_pairs:
        print(f"  {r['family']:15s} → {r['class']:8s}  "
              f"median Δ {r['median_delta']:+.2f}  "
              f"broke {r['broke_in']}/{r['seen_in']} seeds")

    summary = {
        "n_seeds": n_seeds,
        "sae_mean_r_per_seed": sae_mean_r.tolist(),
        "pca_mean_r_per_seed": pca_mean_r.tolist(),
        "sae_strong_per_seed": sae_strong.tolist(),
        "pca_strong_per_seed": pca_strong.tolist(),
        "sae_median_mean_r": float(np.median(sae_mean_r)),
        "pca_median_mean_r": float(np.median(pca_mean_r)),
        "ratio_median": float(np.median(sae_mean_r) /
                               max(np.median(pca_mean_r), 1e-12)),
        "families": families,
        "classes": classes,
        "family_seeds": family_seeds,
        "median_delta_matrix": median_delta.tolist(),
        "broken_count_matrix": broken_count.tolist(),
        "robust_pairs": robust_pairs,
    }
    Path("results/multi_summary.json").write_text(json.dumps(summary, indent=2))
    print("saved → results/multi_summary.json")


if __name__ == "__main__":
    main()
