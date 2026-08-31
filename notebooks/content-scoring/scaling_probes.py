"""Three questions tonight left open, in order of how much they change the write-up.

1. Do the momentum features earn their place? They were added after the comparison was
   run, and a single seed suggested they change nothing. Three seeds with the group in
   and out settles it.
2. What does a fine-tuned encoder do with the *full* corpus? Every transformer number so
   far is on a handicapped subsample, and this model gains ~0.016 going 300k -> 710k.
3. Which learning rate and schedule does that encoder actually want? Nothing has been
   tuned for it -- 2e-5 for three epochs was a guess that happened to work.
"""

import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from transformers import AutoModel, AutoTokenizer

import content_scoring.features as F
from ai_residency_common import load_checkpoint, set_seed
from config import Config
from content_scoring.data import build_splits, iterate_batches
from content_scoring.evaluate import predict
from content_scoring.model import build_model

DEV = torch.device("mps")
MINI = "sentence-transformers/all-MiniLM-L6-v2"


def report(tag, p, yte):
    lab = (yte >= 100).astype(int)
    print(
        f"  {tag:<52} rho {spearmanr(p, yte).statistic:.4f}  ROC {roc_auc_score(lab, p):.3f}  "
        f"PR {average_precision_score(lab, p):.3f}",
        flush=True,
    )


def own_model(sp, mat, cfg, seed, drop_groups=()):
    """Train the service's own architecture, optionally zeroing whole feature groups."""
    set_seed(seed)
    m = build_model(mat, sp.spec.num_domains, cfg).to(DEV)
    keep = np.ones(len(F.COLUMN_ORDER), dtype=bool)
    for g in drop_groups:
        keep[F.GROUP_SLICES[g]] = False
    mask = torch.tensor(keep, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(
        [p for p in m.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    lossf, best, bstate = nn.MSELoss(), -9, None
    yv = sp.frames["val"]["score"].to_numpy(float)
    for _ in range(20):
        m.train()
        for tk, mk, nu, do, ta in iterate_batches(
            sp.tensors["train"], 1024, shuffle=True, device=DEV
        ):
            opt.zero_grad()
            lossf(m(tk, mk, nu * mask, do), ta).backward()
            opt.step()
        s = spearmanr(predict(m, sp.tensors["val"], DEV, "log_mse"), yv).statistic
        if s > best:
            best, bstate = s, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bstate)
    return predict(m, sp.tensors["test"], DEV, "log_mse")


class Enc(nn.Module):
    def __init__(self, enc, width, feat_dim, hidden=256):
        super().__init__()
        self.encoder, self.head = (
            enc,
            nn.Sequential(nn.Linear(width + feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)),
        )

    def forward(self, ids, mask, feats):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head(torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), feats], 1)).squeeze(-1)


def encoder_run(sp, seed, lr, epochs, n_train=None, batch=128):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MINI)
    model = Enc(
        AutoModel.from_pretrained(MINI, dtype=torch.float32), 384, sp.tensors["train"][2].shape[1]
    ).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    titles = {s: sp.frames[s]["title"].tolist() for s in ("train", "val", "test")}
    feats = {s: sp.tensors[s][2].numpy() for s in ("train", "val", "test")}
    targets = sp.tensors["train"][4].numpy()
    idx = (
        np.arange(len(titles["train"]))
        if n_train is None
        else np.random.default_rng(0).permutation(len(titles["train"]))[:n_train]
    )

    def infer(split, limit=None):
        model.eval()
        rows = titles[split][:limit]
        block = feats[split][: len(rows)]
        out = np.zeros(len(rows))
        with torch.no_grad():
            for i in range(0, len(rows), 256):
                b = tok(
                    rows[i : i + 256],
                    padding=True,
                    truncation=True,
                    max_length=48,
                    return_tensors="pt",
                ).to(DEV)
                f = torch.tensor(block[i : i + 256], dtype=torch.float32, device=DEV)
                out[i : i + 256] = model(b["input_ids"], b["attention_mask"], f).cpu().numpy()
        return np.expm1(np.clip(out, -20, 20))

    yv = sp.frames["val"]["score"].to_numpy(float)[:20000]
    best, bstate = -9, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        order = np.random.default_rng(ep + seed).permutation(idx)
        for j in range(0, len(order), batch):
            sel = order[j : j + batch]
            b = tok(
                [titles["train"][k] for k in sel],
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors="pt",
            ).to(DEV)
            f = torch.tensor(feats["train"][sel], dtype=torch.float32, device=DEV)
            y = torch.tensor(targets[sel], dtype=torch.float32, device=DEV)
            opt.zero_grad()
            lossf(model(b["input_ids"], b["attention_mask"], f), y).backward()
            opt.step()
        s = spearmanr(infer("val", 20000), yv).statistic
        print(
            f"      epoch {ep + 1}/{epochs} val rho {s:.4f} ({time.time() - t0:.0f}s)", flush=True
        )
        if s > best:
            best, bstate = s, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(bstate)
    return infer("test")


if __name__ == "__main__":
    stage = sys.argv[1]
    c = Config(objective="log_mse", fusion="late")
    ck = load_checkpoint(c.embeddings_path)
    sp = build_splits(c, ck["extra"]["word_to_index"])
    mat = ck["model_state_dict"]["in_embed.weight"]
    yte = sp.frames["test"]["score"].to_numpy(float)

    if stage == "momentum":
        print("\n### Stage 1 -- do the momentum features earn their place?", flush=True)
        for label, drop in (("with momentum", ()), ("without momentum", ("momentum",))):
            for seed in (42, 43, 44):
                report(f"{label}, seed {seed}", own_model(sp, mat, c, seed, drop), yte)
    elif stage == "fulldata":
        print("\n### Stage 2 -- fine-tuned encoder on the full 710,722 rows", flush=True)
        for seed in (42, 43, 44):
            print(f"    seed {seed}", flush=True)
            report(
                f"MiniLM fine-tuned + metadata, full data, seed {seed}",
                encoder_run(sp, seed, lr=2e-5, epochs=3),
                yte,
            )
    elif stage == "lr":
        print("\n### Stage 3 -- what schedule does the encoder want?", flush=True)
        for lr in (1e-5, 3e-5, 5e-5):
            print(f"    lr={lr}", flush=True)
            report(
                f"MiniLM lr={lr}, 3 epochs, 300k",
                encoder_run(sp, 42, lr=lr, epochs=3, n_train=300_000),
                yte,
            )
    print(
        "\n  reference: my late fusion 710k rho 0.3659 ROC 0.678 PR 0.090 | "
        "MiniLM 300k rho 0.3700 ROC 0.710 PR 0.099",
        flush=True,
    )
