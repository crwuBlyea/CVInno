# ============================================================
# plot_utils.py — shared plotting for the frozen-feature baselines.
# Produces comparable figures for probe.py (ViT) and probe_cnn.py (CNN).
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

def plot_all(probs, ys, va, name, tag, outdir):
    """
    probs : (N,) P(AI) on validation
    ys    : (N,) labels (0 human / 1 AI)
    va    : validation DataFrame with .p (probs) and .src columns
    name  : human-readable model name, e.g. "frozen dinov2-base"
    tag   : filename prefix, e.g. "baseline_vit"
    outdir: Path to results/figures/
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. score distributions ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(probs[ys == 0], bins=bins, alpha=0.6, label="human", color="#3b7", edgecolor="none")
    ax.hist(probs[ys == 1], bins=bins, alpha=0.6, label="AI",    color="#c55", edgecolor="none")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("P(AI)"); ax.set_ylabel("count")
    ax.set_title(f"Validation score distributions — {name}")
    ax.legend(); fig.tight_layout()
    fig.savefig(outdir / f"{tag}_scores.png", dpi=150)
    plt.close(fig)

    # ---------- 2. reliability diagram ----------
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs > 0.5).astype(int) == ys).astype(float)
    edges = np.linspace(0.5, 1.0, 11)
    mids, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        mids.append(conf[m].mean()); accs.append(correct[m].mean()); counts.append(m.sum())
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(mids, accs, "o-", color="#357")
    ax.plot([0.5, 1], [0.5, 1], "k:", lw=1, label="perfect calibration")
    for x, y, n in zip(mids, accs, counts):
        ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, 6),
                    fontsize=7, color="#888")
    ax.set_xlabel("stated confidence"); ax.set_ylabel("actual accuracy")
    ax.set_title(f"Reliability — {name}")
    ax.legend(); fig.tight_layout()
    fig.savefig(outdir / f"{tag}_reliability.png", dpi=150)
    plt.close(fig)

    # ---------- 3. ROC curve ----------
    fpr, tpr, _ = roc_curve(ys, probs)
    auc = roc_auc_score(ys, probs)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color="#357", label=f"val AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("recall (TPR)")
    ax.set_title(f"ROC — {name}")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(outdir / f"{tag}_roc.png", dpi=150)
    plt.close(fig)

    # ---------- 4. per-source metrics ----------
    rows = []
    for s in va.src.unique():
        sub = va[va.src == s]
        if (sub.label == 1).any():
            rows.append((s, "AI recall", (sub[sub.label == 1].p > .5).mean()))
        if (sub.label == 0).any():
            rows.append((s, "human fpr", (sub[sub.label == 0].p > .5).mean()))
    if rows:
        labels = [f"{s}\n{m}" for s, m, _ in rows]
        vals = [v for _, _, v in rows]
        cols = ["#c55" if m == "AI recall" else "#3b7" for _, m, _ in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(rows)), vals, color=cols)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_ylabel("rate"); ax.set_ylim(0, 1)
        ax.set_title(f"Per-source — {name}")
        fig.tight_layout()
        fig.savefig(outdir / f"{tag}_per_source.png", dpi=150)
        plt.close(fig)

    print(f"figures saved -> {outdir} ({tag}_*.png)")
    return [outdir / f"{tag}_{k}.png" for k in
            ["scores", "reliability", "roc", "per_source"]]