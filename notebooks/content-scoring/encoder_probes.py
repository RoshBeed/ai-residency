"""Can a stronger pretrained encoder break the ceiling?

Four probes, all with the same target (log1p score), the same optimiser and the same
head, so the only thing that varies is the text representation:

  1. MiniLM frozen              -- the baseline from the last probe
  2. MiniLM fine-tuned          -- isolates "does adapting the encoder to this task help"
  3. Qwen2.5-0.5B + LoRA        -- a model with real world knowledge, adapted cheaply
  4. each of the above + metadata

Reported against our own numbers: word2vec text-only 0.2708, late fusion 0.3680.
"""

import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ai_residency_common import load_checkpoint, set_seed
from config import Config
from content_scoring.data import build_splits

DEV = torch.device("mps")
c = Config(objective="log_mse")
w2i = load_checkpoint(c.embeddings_path)["extra"]["word_to_index"]
sp = build_splits(c, w2i)
FEATS = {s: sp.tensors[s][2].numpy() for s in ("train", "val", "test")}
Y = {s: sp.tensors[s][4].numpy() for s in ("train", "val", "test")}
RAW = {s: sp.frames[s]["score"].to_numpy(float) for s in ("train", "val", "test")}
TITLES = {s: sp.frames[s]["title"].tolist() for s in ("train", "val", "test")}


class Scorer(nn.Module):
    """Encoder -> mean pool -> optional metadata concat -> MLP -> log1p(score)."""

    def __init__(self, encoder, width, use_feats, feat_dim, hidden=256):
        super().__init__()
        self.encoder, self.use_feats = encoder, use_feats
        self.head = nn.Sequential(
            nn.Linear(width + (feat_dim if use_feats else 0), hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ids, mask, feats):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        if self.use_feats:
            pooled = torch.cat([pooled, feats], dim=1)
        return self.head(pooled).squeeze(-1)


def evaluate(model, tok, split, batch=256, limit=None):
    model.eval()
    titles = TITLES[split][:limit]
    # Sliced to the same limit as the titles. Without this the final partial batch pairs
    # 32 pooled rows with 256 feature rows and the concat blows up -- and only on a split
    # whose length is not a multiple of the batch, which is why it survived probe one.
    feats = FEATS[split][: len(titles)]
    preds = np.zeros(len(titles), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(titles), batch):
            b = tok(
                titles[i : i + batch],
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors="pt",
            ).to(DEV)
            f = torch.tensor(feats[i : i + batch], dtype=torch.float32, device=DEV)
            preds[i : i + batch] = model(b["input_ids"], b["attention_mask"], f).cpu().numpy()
    actual = RAW[split][: len(titles)]
    p = np.expm1(np.clip(preds, -20, 20))
    lab = (actual >= 100).astype(int)
    return (spearmanr(p, actual).statistic, roc_auc_score(lab, p), average_precision_score(lab, p))


def probe(name, model_id, *, lora, frozen, use_feats, n_train, epochs, lr, batch):
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
    width = enc.config.hidden_size
    if lora:
        targets = ["q_proj", "v_proj"] if "qwen" in model_id.lower() else ["query", "value"]
        enc = get_peft_model(
            enc, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets)
        )
    if frozen:
        for p in enc.parameters():
            p.requires_grad = False
    model = Scorer(enc, width, use_feats, FEATS["train"].shape[1]).to(DEV)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.MSELoss()
    idx = np.random.default_rng(0).permutation(len(TITLES["train"]))[:n_train]
    print(
        f"\n=== {name}  ({total / 1e6:.0f}M total, "
        f"{trainable / 1e6:.2f}M trainable, n_train={len(idx):,})",
        flush=True,
    )
    best, best_state = -9, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        order = np.random.default_rng(ep).permutation(idx)
        for j in range(0, len(order), batch):
            sel = order[j : j + batch]
            b = tok(
                [TITLES["train"][k] for k in sel],
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors="pt",
            ).to(DEV)
            f = torch.tensor(FEATS["train"][sel], dtype=torch.float32, device=DEV)
            y = torch.tensor(Y["train"][sel], dtype=torch.float32, device=DEV)
            opt.zero_grad()
            lossf(model(b["input_ids"], b["attention_mask"], f), y).backward()
            opt.step()
        v = evaluate(model, tok, "val", limit=20000)[0]
        print(f"  epoch {ep + 1}/{epochs}  val rho {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if v > best:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    rho, auc, pr = evaluate(model, tok, "test")
    print(f"  -> TEST rho {rho:.4f}  viral ROC-AUC {auc:.3f}  PR-AUC {pr:.3f}", flush=True)
    return rho


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    mini = "sentence-transformers/all-MiniLM-L6-v2"
    qwen = "Qwen/Qwen2.5-0.5B"
    if which in ("all", "mini"):
        probe(
            "MiniLM fine-tuned + metadata",
            mini,
            lora=False,
            frozen=False,
            use_feats=True,
            n_train=300_000,
            epochs=3,
            lr=2e-5,
            batch=128,
        )
    if which in ("all", "qwen"):
        probe(
            "Qwen2.5-0.5B LoRA, title only",
            qwen,
            lora=True,
            frozen=False,
            use_feats=False,
            n_train=120_000,
            epochs=2,
            lr=1e-4,
            batch=32,
        )
        probe(
            "Qwen2.5-0.5B LoRA + metadata",
            qwen,
            lora=True,
            frozen=False,
            use_feats=True,
            n_train=120_000,
            epochs=2,
            lr=1e-4,
            batch=32,
        )
    print(
        "\n  reference: our word2vec text-only 0.2708 | our late fusion 0.3680 "
        "(viral ROC-AUC 0.678, PR-AUC 0.084)"
    )
    print("  reference: MiniLM FROZEN text-only 0.2889 | MiniLM FROZEN + metadata 0.3638")
