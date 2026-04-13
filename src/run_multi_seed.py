"""Run the full sae-rf-classifier pipeline across multiple seeds.

For each seed:
  1. Retrain the classifier from scratch with that seed.
  2. Collect activations + train SAE.
  3. Run SAE-vs-classical correlation analysis.
  4. Run per-family causal ablation.

Each seed's outputs land in results/seed_{s}/*.

Then aggregate all seeds into a single combined JSON and emit the
median/IQR plots via aggregate_multi_seed.py. Non-essential large
files (activations.npz) are not preserved across seeds — only the
reproducible metadata.

Run:
    python -m src.run_multi_seed --seeds 0,1,2,3,4
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/multi_seed.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    py = sys.executable
    all_runs = []

    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        base = Path(f"results/seed_{seed}")
        base.mkdir(parents=True, exist_ok=True)

        classifier_ckpt = base / "classifier.pt"
        sae_ckpt = base / "sae.pt"
        activations = base / "activations.npz"
        analysis = base / "sae_vs_classical.json"
        ablation = base / "ablation.json"

        run([
            py, "-m", "src.train",
            "--seed", str(seed),
            "--out", str(classifier_ckpt),
        ])
        run([
            py, "-m", "src.collect_and_train_sae",
            "--classifier", str(classifier_ckpt),
            "--seed", str(seed),
            "--out-sae", str(sae_ckpt),
            "--out-activations", str(activations),
        ])
        run([
            py, "-m", "src.analyze_sae_vs_classical",
            "--activations", str(activations),
            "--out", str(analysis),
        ])
        run([
            py, "-m", "src.ablation",
            "--classifier", str(classifier_ckpt),
            "--sae", str(sae_ckpt),
            "--activations", str(activations),
            "--analysis", str(analysis),
            "--out", str(ablation),
        ])

        all_runs.append({
            "seed": seed,
            "analysis": json.loads(analysis.read_text()),
            "ablation": json.loads(ablation.read_text()),
        })

        # Activations are large and regenerable. Delete them once we're done
        # with this seed so the multi-seed run doesn't leave ~45 MB laying
        # around for a 5-seed sweep.
        if activations.exists():
            activations.unlink()

    combined = {
        "seeds": [r["seed"] for r in all_runs],
        "runs": all_runs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(combined, indent=2))
    print(f"\nsaved combined → {out}")


if __name__ == "__main__":
    main()
