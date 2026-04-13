"""Train the modulation classifier on synthetic IQ data."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .classifier import ModClassifier
from .synth_data import generate, CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--out", default="results/classifier.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    X, y = generate(n_per_class=512, snr_db=args.snr)
    n_tr = int(0.8 * len(X))
    perm = torch.randperm(len(X))
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    train_ds = TensorDataset(X[tr_idx], y[tr_idx])
    test_ds = TensorDataset(X[te_idx], y[te_idx])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch)

    model = ModClassifier(n_classes=len(CLASSES)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(args.device), yb.to(args.device)
                correct += (model(xb).argmax(-1) == yb).sum().item()
                total += len(yb)
        print(f"epoch {epoch}  test_acc {correct/total:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
