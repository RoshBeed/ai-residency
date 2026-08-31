"""The second half of the overnight queue.

Stage 4 asks the question the whole evening kept circling: if the goal is finding the
posts that reach the front page, is a regression on score even the right objective? A
classifier trained directly on `score >= 100` optimises the thing we actually measure,
and cannot be satisfied by predicting the middle.

Stage 5 is the clean size comparison. Every Qwen number so far is on 120k rows against
MiniLM's 300k, so "a 495M model ties a 23M one" is not yet a fair statement.
"""

import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ai_residency_common import load_checkpoint, set_seed
from config import Config
from content_scoring.data import build_splits, iterate_batches
from content_scoring.evaluate import predict
from content_scoring.model import build_model

DEV = torch.device("mps")
VIRAL = 100


def report(tag: str, scores: np.ndarray, actual: np.ndarray) -> None:
    """One line per experiment, always with both the ranking and the tail metric."""
    labels = (actual >= VIRAL).astype(int)
    rho = spearmanr(scores, actual).statistic
    print(
        f"  {tag:<54} rho {rho:.4f}  ROC {roc_auc_score(labels, scores):.3f}  "
        f"PR {average_precision_score(labels, scores):.3f}",
        flush=True,
    )


def train_own(sp, matrix, config, seed, objective):
    """The service's own architecture under one of three objectives.

    `viral` is the new one: a binary head on `score >= 100`, selected on validation
    PR-AUC rather than Spearman. It gives up the ability to predict a score at all in
    exchange for optimising exactly the decision the ranking is used for.
    """
    set_seed(seed)
    model = build_model(matrix, sp.spec.num_domains, config).to(DEV)
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = {
        "log_mse": nn.MSELoss(),
        "poisson": nn.PoissonNLLLoss(log_input=True, full=False),
        "viral": nn.BCEWithLogitsLoss(),
    }[objective]
    val_actual = sp.frames["val"]["score"].to_numpy(float)
    val_labels = (val_actual >= VIRAL).astype(int)
    best, best_state = -9.0, None

    for _ in range(20):
        model.train()
        for tokens, mask, numeric, domain, log_target in iterate_batches(
            sp.tensors["train"], 1024, shuffle=True, device=DEV
        ):
            raw = torch.expm1(log_target)
            target = {
                "log_mse": log_target,
                "poisson": raw,
                "viral": (raw >= VIRAL).float(),
            }[objective]
            optimiser.zero_grad()
            loss_fn(model(tokens, mask, numeric, domain), target).backward()
            optimiser.step()

        scores = predict(
            model, sp.tensors["val"], DEV, "log_mse" if objective != "poisson" else "poisson"
        )
        # Selected on what each objective is for: PR-AUC for the classifier, rank
        # correlation for the regressors. Selecting all three on Spearman would quietly
        # hobble the classifier on the metric it exists to win.
        current = (
            average_precision_score(val_labels, scores)
            if objective == "viral"
            else spearmanr(scores, val_actual).statistic
        )
        if current > best:
            best = current
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return predict(
        model, sp.tensors["test"], DEV, "log_mse" if objective != "poisson" else "poisson"
    )


class QwenScorer(nn.Module):
    """A LoRA-adapted decoder with a regression head over mean-pooled hidden states."""

    def __init__(self, encoder, width, feature_dim, hidden=256):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(width + feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, ids, mask, feats):
        states = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        pooled = (states * weights).sum(1) / weights.sum(1).clamp(min=1)
        return self.head(torch.cat([pooled, feats], dim=1)).squeeze(-1)


def train_qwen(sp, seed, n_train, epochs=2, lr=1e-4, batch=32, rank=16):
    from peft import LoraConfig, get_peft_model

    set_seed(seed)
    name = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(name, dtype=torch.float32)
    base = get_peft_model(
        base,
        LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]
        ),
    )
    titles = {s: sp.frames[s]["title"].tolist() for s in ("train", "val", "test")}
    feats = {s: sp.tensors[s][2].numpy() for s in ("train", "val", "test")}
    model = QwenScorer(base, base.config.hidden_size, feats["train"].shape[1]).to(DEV)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.MSELoss()
    targets = sp.tensors["train"][4].numpy()
    index = np.random.default_rng(0).permutation(len(titles["train"]))[:n_train]

    def infer(split, limit=None):
        model.eval()
        rows = titles[split][:limit]
        block = feats[split][: len(rows)]
        out = np.zeros(len(rows))
        with torch.no_grad():
            for i in range(0, len(rows), 128):
                b = tok(
                    rows[i : i + 128],
                    padding=True,
                    truncation=True,
                    max_length=48,
                    return_tensors="pt",
                ).to(DEV)
                f = torch.tensor(block[i : i + 128], dtype=torch.float32, device=DEV)
                out[i : i + 128] = model(b["input_ids"], b["attention_mask"], f).cpu().numpy()
        return np.expm1(np.clip(out, -20, 20))

    val_actual = sp.frames["val"]["score"].to_numpy(float)[:20000]
    best, best_state = -9.0, None
    for epoch in range(epochs):
        model.train()
        started = time.time()
        order = np.random.default_rng(epoch + seed).permutation(index)
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
            optimiser.zero_grad()
            loss_fn(model(b["input_ids"], b["attention_mask"], f), y).backward()
            optimiser.step()
        current = spearmanr(infer("val", 20000), val_actual).statistic
        print(
            f"      epoch {epoch + 1}/{epochs} val rho {current:.4f}"
            f" ({time.time() - started:.0f}s)",
            flush=True,
        )
        if current > best:
            best = current
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return infer("test")


if __name__ == "__main__":
    stage = sys.argv[1]
    config = Config(objective="log_mse", fusion="late")
    checkpoint = load_checkpoint(config.embeddings_path)
    splits = build_splits(config, checkpoint["extra"]["word_to_index"])
    matrix = checkpoint["model_state_dict"]["in_embed.weight"]
    actual = splits.frames["test"]["score"].to_numpy(float)

    if stage == "tail":
        print("\n### Stage 4 -- is regression even the right objective for the tail?", flush=True)
        for objective in ("log_mse", "poisson", "viral"):
            for seed in (42, 43, 44):
                report(
                    f"{objective} objective, seed {seed}",
                    train_own(splits, matrix, config, seed, objective),
                    actual,
                )
    elif stage == "qwen300":
        print("\n### Stage 5 -- Qwen at the same 300k rows MiniLM used", flush=True)
        scores = train_qwen(splits, 42, 300_000)
        report("Qwen-0.5B LoRA + metadata, 300k rows", scores, actual)

    print(
        "\n  reference: my late fusion 710k rho 0.3659 ROC 0.678 PR 0.090 | "
        "MiniLM 300k rho 0.3700 ROC 0.710 PR 0.099 | Qwen 120k rho 0.3731 ROC 0.716 PR 0.103",
        flush=True,
    )
