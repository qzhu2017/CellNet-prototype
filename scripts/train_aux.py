#!/usr/bin/env python3
"""Train and evaluate the auxiliary space-group + density-distribution model.

  python scripts/train_aux.py                       # train on datasets/joint-z12, evaluate on its test split
  python scripts/train_aux.py --eval-only --checkpoint outputs/aux_sg_density/best.pt

Space-group head: p(Hall, Z′ | molecule) over all (Hall, Z′) classes. The target for a
molecule is uniform over every (Hall, Z′) it is known in (training rows), so polymorphs in
different groups are all "right". Density head: 3-Gaussian mixture p(log ρ | molecule,
Hall, Z′), trained by negative log-likelihood.

Evaluation (test split, molecules never seen in training):
  space group : top-k coverage (any known form of the molecule in the top k) vs the
                frequency baseline (always guess the most common classes)
  density     : error of the mixture mean, NLL, and calibration (fraction of test crystals
                inside the central 50 / 68 / 95 % predictive intervals)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellnet.aux_model import (  # noqa: E402
    SpaceGroupDensityNet, mixture_cdf, mixture_mean_std, mixture_nll,
)
from cellnet.precomputed import load_precomputed_csv  # noqa: E402
from cellnet.splits import split_by_smiles_group  # noqa: E402

D = ROOT / "datasets/joint-z12"


class Vocab:
    def __init__(self, samples):
        pairs = sorted({(s.hall_number, float(s.zprime)) for s in samples})
        self.classes = pairs
        self.cls_idx = {p: i for i, p in enumerate(pairs)}
        self.halls = sorted({p[0] for p in pairs})
        self.hall_idx = {h: i for i, h in enumerate(self.halls)}
        self.zprimes = sorted({p[1] for p in pairs})
        self.zp_idx = {z: i for i, z in enumerate(self.zprimes)}

    def to_json(self):
        return {"classes": [[h, z] for h, z in self.classes], "halls": self.halls, "zprimes": self.zprimes}

    @classmethod
    def from_json(cls, d):
        v = cls.__new__(cls)
        v.classes = [(int(h), float(z)) for h, z in d["classes"]]
        v.cls_idx = {p: i for i, p in enumerate(v.classes)}
        v.halls = [int(h) for h in d["halls"]]
        v.hall_idx = {h: i for i, h in enumerate(v.halls)}
        v.zprimes = [float(z) for z in d["zprimes"]]
        v.zp_idx = {z: i for i, z in enumerate(v.zprimes)}
        return v


class AuxDataset(Dataset):
    """One item per crystal; the space-group target is that molecule's set of known forms."""

    def __init__(self, samples, vocab: Vocab, ld_mean: float, ld_std: float):
        self.samples = samples
        self.vocab = vocab
        forms = defaultdict(set)
        for s in samples:
            forms[s.smiles].add(vocab.cls_idx[(s.hall_number, float(s.zprime))])
        self.forms = forms
        rows_per_mol = Counter(s.smiles for s in samples)
        self.w = {smi: 1.0 / n for smi, n in rows_per_mol.items()}  # each molecule counts once
        self.ld_mean, self.ld_std = ld_mean, ld_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        g = s.graph
        tgt = torch.zeros(1, len(self.vocab.classes))
        f = sorted(self.forms[s.smiles])
        tgt[0, f] = 1.0 / len(f)
        return Data(
            x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
            hall=torch.tensor([self.vocab.hall_idx[s.hall_number]]),
            zprime=torch.tensor([self.vocab.zp_idx[float(s.zprime)]]),
            sg_target=tgt,
            sg_weight=torch.tensor([self.w[s.smiles]], dtype=torch.float32),
            ld=torch.tensor([(s.log_density - self.ld_mean) / self.ld_std], dtype=torch.float32),
        )


def losses(model, batch):
    sg_logits, (logit, mu, ls) = model(batch, batch.hall, batch.zprime)
    ce = -(batch.sg_target * torch.log_softmax(sg_logits, -1)).sum(-1)
    sg = (ce * batch.sg_weight).sum() / batch.sg_weight.sum()
    dn = mixture_nll(logit, mu, ls, batch.ld).mean()
    return sg, dn


@torch.no_grad()
def evaluate(model, samples, vocab, ld_mean, ld_std, device, train_class_freq, batch_size=512):
    """Space-group coverage per molecule and density calibration per crystal."""
    model.eval()
    ds = AuxDataset(samples, vocab, ld_mean, ld_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs_by_mol, ld_rows = {}, []
    for k, batch in enumerate(loader):
        batch = batch.to(device)
        sg_logits, (logit, mu, ls) = model(batch, batch.hall, batch.zprime)
        p = torch.softmax(sg_logits, -1).cpu().numpy()
        mean, std = mixture_mean_std(logit, mu, ls)
        cdf = mixture_cdf(logit, mu, ls, batch.ld)
        nll = mixture_nll(logit, mu, ls, batch.ld)
        base = k * batch_size
        for j in range(batch.num_graphs):
            s = samples[base + j]
            probs_by_mol.setdefault(s.smiles, p[j])
            ld_rows.append((float(batch.ld[j]), float(mean[j]), float(std[j]), float(cdf[j]), float(nll[j])))

    # --- space group
    forms = defaultdict(set)
    for s in samples:
        forms[s.smiles].add(vocab.cls_idx.get((s.hall_number, float(s.zprime)), -1))
    prior_order = np.argsort(-train_class_freq)
    res = {}
    for k in (1, 3, 5, 10):
        model_hit = np.mean([bool(set(np.argsort(-probs_by_mol[m])[:k]) & f) for m, f in forms.items()])
        base_hit = np.mean([bool(set(prior_order[:k]) & f) for m, f in forms.items()])
        res[f"sg_top{k}"] = float(model_hit)
        res[f"sg_top{k}_baseline"] = float(base_hit)
    # adaptive set: smallest set reaching 90 % predicted probability
    sizes, hits = [], []
    for m, f in forms.items():
        order = np.argsort(-probs_by_mol[m])
        n = int(np.searchsorted(np.cumsum(probs_by_mol[m][order]), 0.9) + 1)
        sizes.append(n)
        hits.append(bool(set(order[:n]) & f))
    res["sg_p90_set_size_median"] = float(np.median(sizes))
    res["sg_p90_set_coverage"] = float(np.mean(hits))
    res["n_test_molecules"] = len(forms)

    # --- density (normalised log units -> % in density)
    a = np.array(ld_rows)
    y, mean, std, cdf, nll = a.T
    res["density_mae_pct"] = float(100 * (math.exp(np.median(np.abs(y - mean)) * ld_std) - 1))
    res["density_sigma_pct_median"] = float(100 * (math.exp(np.median(std) * ld_std) - 1))
    res["density_nll"] = float(nll.mean())
    for c in (0.5, 0.68, 0.95):
        lo, hi = (1 - c) / 2, 1 - (1 - c) / 2
        res[f"density_cover_{int(c * 100)}"] = float(np.mean((cdf > lo) & (cdf < hi)))
    res["n_test_crystals"] = len(samples)
    return res, probs_by_mol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", default=str(D / "train_precomputed.csv"))
    ap.add_argument("--test-csv", default=str(D / "test_precomputed.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs/aux_sg_density"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--n-mix", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: limit rows")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if args.device == "auto" else args.device))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_all = load_precomputed_csv(args.train_csv, max_samples=args.max_rows, verbose=False)
    test = load_precomputed_csv(args.test_csv, max_samples=args.max_rows, verbose=False)
    print(f"loaded {len(train_all):,} train / {len(test):,} test crystals in {time.time() - t0:.0f} s", flush=True)

    if args.eval_only:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        vocab = Vocab.from_json(ck["vocab"])
    else:
        vocab = Vocab(train_all + test)  # label vocabulary only (no targets), as in train.py
    tr_idx, va_idx, _ = split_by_smiles_group(train_all, val_frac=0.1, test_frac=0.0, seed=args.seed)
    train = [train_all[i] for i in tr_idx]
    val = [train_all[i] for i in va_idx]
    ld = np.array([s.log_density for s in train])
    ld_mean, ld_std = float(ld.mean()), float(ld.std())
    freq = np.zeros(len(vocab.classes))
    for s in train:
        freq[vocab.cls_idx[(s.hall_number, float(s.zprime))]] += 1

    g0 = train[0].graph
    model = SpaceGroupDensityNet(
        node_dim=g0.x.shape[1], edge_dim=g0.edge_attr.shape[1], n_classes=len(vocab.classes),
        n_halls=len(vocab.halls), n_zprimes=len(vocab.zprimes), n_mix=args.n_mix,
    ).to(dev)

    if args.eval_only:
        model.load_state_dict(ck["model"])
        ld_mean, ld_std = ck["ld_mean"], ck["ld_std"]
    else:
        print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}  classes {len(vocab.classes)}  "
              f"device {dev}", flush=True)
        tl = DataLoader(AuxDataset(train, vocab, ld_mean, ld_std), batch_size=args.batch_size, shuffle=True)
        vl = DataLoader(AuxDataset(val, vocab, ld_mean, ld_std), batch_size=512, shuffle=False)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best, bad, hist = 1e9, 0, []
        for ep in range(1, args.epochs + 1):
            model.train()
            tot = [0.0, 0.0, 0]
            for batch in tl:
                batch = batch.to(dev)
                sg, dn = losses(model, batch)
                loss = sg + dn
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot[0] += sg.item(); tot[1] += dn.item(); tot[2] += 1
            sched.step()
            model.eval()
            v = [0.0, 0.0, 0]
            with torch.no_grad():
                for batch in vl:
                    batch = batch.to(dev)
                    sg, dn = losses(model, batch)
                    v[0] += sg.item(); v[1] += dn.item(); v[2] += 1
            vsg, vdn = v[0] / v[2], v[1] / v[2]
            hist.append({"epoch": ep, "train_sg": tot[0] / tot[2], "train_density_nll": tot[1] / tot[2],
                         "val_sg": vsg, "val_density_nll": vdn})
            flag = ""
            if vsg + vdn < best:
                best, bad, flag = vsg + vdn, 0, " *"
                torch.save({"model": model.state_dict(), "vocab": vocab.to_json(), "ld_mean": ld_mean,
                            "ld_std": ld_std, "n_mix": args.n_mix, "epoch": ep,
                            "node_dim": g0.x.shape[1], "edge_dim": g0.edge_attr.shape[1]}, out / "best.pt")
            else:
                bad += 1
            print(f"epoch {ep:3d}  train sg {tot[0] / tot[2]:.3f} dens {tot[1] / tot[2]:.3f} | "
                  f"val sg {vsg:.3f} dens {vdn:.3f}{flag}  ({time.time() - t0:.0f} s)", flush=True)
            if bad >= args.patience:
                print("early stop", flush=True)
                break
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"best epoch {ck['epoch']}", flush=True)

    res, _ = evaluate(model, test, vocab, ld_mean, ld_std, dev, freq)
    (out / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print("TEST (molecules never seen in training):")
    for k in (1, 3, 5, 10):
        print(f"  space group top-{k:<2d}: model {100 * res[f'sg_top{k}']:.1f}%   "
              f"frequency baseline {100 * res[f'sg_top{k}_baseline']:.1f}%")
    print(f"  90%-probability set: median {res['sg_p90_set_size_median']:.0f} classes, "
          f"covers {100 * res['sg_p90_set_coverage']:.1f}% of molecules")
    print(f"  density: error of mixture mean {res['density_mae_pct']:.2f}%, median predicted spread "
          f"{res['density_sigma_pct_median']:.2f}%, NLL {res['density_nll']:.3f}")
    print(f"  density calibration: inside 50/68/95% intervals = {100 * res['density_cover_50']:.1f} / "
          f"{100 * res['density_cover_68']:.1f} / {100 * res['density_cover_95']:.1f}%")


if __name__ == "__main__":
    main()
