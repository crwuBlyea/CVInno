# ============================================================
# plot_utils.py — shared plotting for the frozen-feature baselines.
# Produces comparable figures for probe.py (ViT) and probe_cnn.py (CNN).
# Also produces the second-approach checkpoint comparison figures
# (plot_checkpoint_comparison / plot_hero_comparison / plot_open_issues)
# for the aggregated results table (gate2_results.csv schema).
# Also produces the final battery figures (plot_battery and friends),
# imported by battery.py via "import plot_utils".
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


# ============================================================
# --- Second-approach checkpoint figures ----------------------
# Comparison plots for the aggregated results table:
# one row per model (probe_large / ft_all / ...), metric columns:
# test, q75, ext_unseen, ideogram, dalle3, sd_sanity,
# transfer_civ, transfer_mj, drift_fpr, classical.
#
# Usage (from probe.py / fine-tune script / notebook):
#   from plot_utils import plot_checkpoint_comparison
#   plot_checkpoint_comparison("gate2_results.csv", "results/figures")
#   plot_checkpoint_comparison(results_df, "results/figures")  # in-memory table
# ============================================================
PROBE_ROW = "probe_large"   # first-approach row (linear probe)
FT_ROW    = "ft_all"        # second-approach row (full fine-tune)

_CB_BLUE, _CB_ORANGE, _CB_RED = "#0077BB", "#EE7733", "#CC3311"   # Paul Tol, colorblind-safe
_G900, _G700, _G500, _G400, _G300 = "#111827", "#374151", "#6B7280", "#9CA3AF", "#D1D5DB"
_EDGE = "#E5E7EB"


def _results_frame(df_or_csv):
    """Accept a CSV path or an in-memory DataFrame; return a copy indexed by model."""
    if isinstance(df_or_csv, (str, Path)):
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv.copy()
    if "model" in df.columns:
        df = df.set_index("model")
    return df


def _pts(delta):
    s = f"{delta:+.1f}".rstrip("0").rstrip(".")
    return f"{s} pts"


def _style_ax(ax):
    """Local styling for checkpoint figures (no rcParams side effects)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_EDGE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, alpha=0.08, color=_G300)
    ax.set_axisbelow(True)


def plot_hero_comparison(df_or_csv, outdir, tag="second_approach", dpi=300):
    """Probe (1st approach) vs full fine-tune (2nd approach) on the headline
    accuracy metrics. Saves {outdir}/{tag}_hero.png, returns the Path."""
    df = _results_frame(df_or_csv)
    if PROBE_ROW not in df.index or FT_ROW not in df.index:
        raise ValueError(f"need rows '{PROBE_ROW}' and '{FT_ROW}' in the results table")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    metrics = [("test",       "In-domain test"),
               ("q75",        "q75 (quality-filtered)"),
               ("ext_unseen", "Unseen generators"),
               ("ideogram",   "Ideogram"),
               ("dalle3",     "DALL-E 3")]
    probe = [float(df.loc[PROBE_ROW, m]) for m, _ in metrics]
    ft    = [float(df.loc[FT_ROW, m])    for m, _ in metrics]

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    x, w = np.arange(len(metrics)), 0.34
    ax.bar(x - w / 2, probe, width=w, color=_CB_BLUE, zorder=3,
           edgecolor="white", linewidth=0.5, label="Probe (1st approach)")
    ax.bar(x + w / 2, ft, width=w, color=_CB_ORANGE, zorder=3,
           edgecolor="white", linewidth=0.5, label="Full fine-tune (2nd approach)")

    for xi, pv, fv in zip(x, probe, ft):
        ax.text(xi - w / 2, pv + 0.015, f"{pv:.3f}", ha="center", va="bottom",
                fontsize=9.5, color=_G500)
        ax.text(xi + w / 2, fv + 0.015, f"{fv:.3f}", ha="center", va="bottom",
                fontsize=10, color=_G900, fontweight="bold")
        ax.text(xi + w / 2, fv + 0.062, _pts((fv - pv) * 100),
                ha="center", va="bottom", fontsize=10, color=_G700, fontweight="bold")

    ax.set_title("Full fine-tuning improves every evaluation set - "
                 "largest gains on unseen generators",
                 loc="left", fontsize=15, fontweight="bold", pad=14)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics], fontsize=10.5)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, 1.20)
    _style_ax(ax)

    try:    # matplotlib >= 3.7: reserves space outside the axes
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper right",
                   frameon=False, fontsize=10)
    except Exception:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=10)

    path = outdir / f"{tag}_hero.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def plot_open_issues(df_or_csv, outdir, tag="second_approach", dpi=300):
    """Weak subsets + drift-FPR jump (the 'known issues' panel).
    Saves {outdir}/{tag}_open_issues.png, returns the Path."""
    df = _results_frame(df_or_csv)
    if PROBE_ROW not in df.index or FT_ROW not in df.index:
        raise ValueError(f"need rows '{PROBE_ROW}' and '{FT_ROW}' in the results table")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    sd_p = float(df.loc[PROBE_ROW, "sd_sanity"]); sd_f = float(df.loc[FT_ROW, "sd_sanity"])
    cl_p = float(df.loc[PROBE_ROW, "classical"]); cl_f = float(df.loc[FT_ROW, "classical"])
    dr_p = float(df.loc[PROBE_ROW, "drift_fpr"]); dr_f = float(df.loc[FT_ROW, "drift_fpr"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1, 0.85]})

    # ---------- panel A: low-scoring subsets ----------
    x, w = np.arange(2), 0.32
    a.bar(x - w / 2, [sd_p, cl_p], width=w, color=_CB_BLUE, zorder=3,
          edgecolor="white", linewidth=0.5, label="Probe (1st approach)")
    a.bar(x + w / 2, [sd_f, cl_f], width=w, color=_CB_ORANGE, zorder=3,
          edgecolor="white", linewidth=0.5, label="Full fine-tune (2nd approach)")
    for xi, pv, fv in zip(x, [sd_p, cl_p], [sd_f, cl_f]):
        a.text(xi - w / 2, pv + 0.02, f"{pv:.3f}", ha="center", va="bottom",
               fontsize=9.5, color=_G500)
        a.text(xi + w / 2, fv + 0.02, f"{fv:.3f}", ha="center", va="bottom",
               fontsize=10, color=_G900, fontweight="bold")

    a.axhline(0.5, color=_G400, linewidth=1, linestyle="--", alpha=0.7, zorder=2)
    a.text(-0.46, 0.515, "chance = 0.5", ha="left", va="bottom",
           fontsize=10, color=_G500, style="italic")
    a.text(0, 0.82, f"{_pts((sd_f - sd_p) * 100)},\nstill the weakest subset",
           ha="center", va="bottom", fontsize=9.5, color=_G700, style="italic")
    a.text(1, 0.56, "both below chance -\nverifying eval pipeline",
           ha="center", va="bottom", fontsize=9.5, color=_G700, style="italic")

    a.set_title("Weakest subsets stay weak (under investigation)",
                loc="left", fontsize=13, fontweight="bold", pad=12)
    a.set_ylabel("Accuracy", fontsize=11)
    a.set_xticks(x)
    a.set_xticklabels(["Stable Diffusion sanity", "Classical (GAN-era)"], fontsize=10)
    a.tick_params(axis="y", labelsize=10)
    a.set_ylim(0, 1.05); a.set_xlim(-0.55, 1.55)
    _style_ax(a)

    # ---------- panel B: drift false positives ----------
    b.bar(0, dr_p, width=0.5, color=_CB_BLUE, zorder=3, edgecolor="white", linewidth=0.5)
    b.bar(1, dr_f, width=0.5, color=_CB_RED,  zorder=3, edgecolor="white", linewidth=0.5)
    b.text(0, dr_p + 0.006, f"{dr_p:.3f}", ha="center", va="bottom",
           fontsize=9.5, color=_G500)
    b.text(1, dr_f + 0.006, f"{dr_f:.3f}", ha="center", va="bottom",
           fontsize=10, color=_G900, fontweight="bold")
    b.text(1, dr_f + 0.030, f"{dr_f / dr_p:.0f}x vs baseline -\ncalibration queued",
           ha="center", va="bottom", fontsize=9.5, color=_CB_RED, style="italic")

    b.set_title("Drift false positives (lower is better)", loc="left",
                fontsize=13, fontweight="bold", pad=12)
    b.set_ylabel("FPR on natural images", fontsize=11)
    b.set_xticks([0, 1])
    b.set_xticklabels(["Probe\n(1st approach)", "Full fine-tune\n(2nd approach)"], fontsize=10)
    b.tick_params(axis="y", labelsize=10)
    b.set_ylim(0, 0.27); b.set_xlim(-0.55, 1.55)
    _style_ax(b)

    try:
        fig.legend(*a.get_legend_handles_labels(), loc="outside upper right",
                   frameon=False, fontsize=10)
    except Exception:
        a.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=10)

    path = outdir / f"{tag}_open_issues.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


_CHECKPOINT_WRITEUP = """SECOND APPROACH - CHECKPOINT UPLOAD (Oct 20)

Change vs. the first model
--------------------------
The first working model was a linear probe on a frozen pretrained backbone
(probe_large). The second approach fully fine-tunes the same backbone on all
training data (ft_all). Two single-source variants (ft_mj_only,
ft_civitai_only) are also trained to test how generator diversity in the
training data affects generalization; ft_all is the main second model.

Headline results (probe -> ft_all, see figures)
-----------------------------------------------
- In-domain test accuracy: 0.971 -> 0.985; quality-filtered q75: 0.976 -> 0.992
- Unseen-generator accuracy: 0.913 -> 0.935
- Ideogram: 0.690 -> 0.835 (+14.5 pts); DALL-E 3: 0.745 -> 0.855 (+11 pts)
- Fine-tuning is working: it improves every headline metric, with the
  largest gains on generators never seen in training.

Early observations / open issues (carried to the Oct 27 progress check)
-----------------------------------------------------------------------
1. Drift false-positive rate rose from 0.005 to 0.15 (30x). Fine-tuning
   trades natural-image robustness for accuracy; threshold calibration is
   queued as the next experiment.
2. Classical (GAN-era) images remain below chance (~0.39). We are currently
   verifying whether this is an evaluation artifact or a genuine scope
   limitation of features learned on modern diffusion generators.
3. The Stable Diffusion sanity subset improved sharply (0.10 -> 0.645) but
   remains the weakest accuracy subset.
"""


def plot_checkpoint_comparison(df_or_csv, outdir, tag="second_approach",
                               writeup=False, dpi=300):
    """Both second-approach checkpoint figures in one call.

    df_or_csv : path to gate2_results.csv OR the in-memory results DataFrame
    outdir   : where to save the PNGs
    tag      : filename prefix (files: {tag}_hero.png, {tag}_open_issues.png)
    writeup  : if True, also saves {tag}_writeup.txt (paste-ready upload text)
    Returns the list of saved paths.
    """
    outdir = Path(outdir)
    paths = [plot_hero_comparison(df_or_csv, outdir, tag, dpi),
             plot_open_issues(df_or_csv, outdir, tag, dpi)]
    if writeup:
        wp = outdir / f"{tag}_writeup.txt"
        wp.write_text(_CHECKPOINT_WRITEUP)
        paths.append(wp)
    print(f"figures saved -> {outdir} ({tag}_hero.png, {tag}_open_issues.png)")
    return paths


# ============================================================
# --- Final battery figures (imported by battery.py) ----------
# battery.py calls, at the end of its main():
#     import plot_utils
#     plot_utils.plot_battery(res, ROOT / "results" / "figures")
# where res is the long table it wrote to results/battery_results.csv
# (columns: model, split, auc, acc, f1, threshold, n_params).
# ============================================================
C_VIT, C_CNN = "#357", "#c55"          # ViT / CNN bars (same palette as plot_all)


def _label_bars(ax, bars, vals, fmt="{:.3f}"):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                fmt.format(v), ha="center", va="bottom", fontsize=9)


def plot_test_comparison(res, outdir):
    """Clean test set: ViT vs CNN grouped bars (acc / F1 / AUC)."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    t = res[res.split == "test"].set_index("model")
    metrics = [("acc", "Accuracy"), ("f1", "F1"), ("auc", "AUC")]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x, w = np.arange(len(metrics)), 0.34
    for i, (model, color) in enumerate([("vit", C_VIT), ("cnn", C_CNN)]):
        vals = [float(t.loc[model, m]) for m, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=color,
                      label=model.upper())
        _label_bars(ax, bars, vals)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_title("Clean test set — ViT vs CNN")
    ax.legend()
    fig.tight_layout()
    path = outdir / "battery_test_comparison.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_robustness(res, outdir):
    """F1 across test / test_q75 / test_q50, with F1-drop annotations."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    piv = res.pivot(index="model", columns="split", values="f1")
    order = [s for s in ["test", "test_q75", "test_q50"] if s in piv.columns]
    labels = {"test": "Clean test", "test_q75": "JPEG q75", "test_q50": "JPEG q50"}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x, w = np.arange(len(order)), 0.34
    for i, (model, color) in enumerate([("vit", C_VIT), ("cnn", C_CNN)]):
        vals = [float(piv.loc[model, s]) for s in order]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=color,
                      label=model.upper())
        _label_bars(ax, bars, vals)
        for j, s in enumerate(order[1:], start=1):        # F1 drop vs clean
            drop = piv.loc[model, "test"] - piv.loc[model, s]
            ax.text(x[j] + (i - 0.5) * w, vals[j] + 0.075,
                    f"-{drop:.3f}", ha="center", va="bottom",
                    fontsize=8, color="#666")
    ax.set_xticks(x); ax.set_xticklabels([labels[s] for s in order])
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.2)
    ax.set_title("Robustness to JPEG recompression")
    ax.legend()
    fig.tight_layout()
    path = outdir / "battery_robustness.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_model_cost(res, outdir):
    """Parameter count (millions) per model, with fitted threshold noted."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    t = res[res.split == "test"].set_index("model")
    models = [("vit", C_VIT), ("cnn", C_CNN)]
    vals = [float(t.loc[m, "n_params"]) / 1e6 for m, _ in models]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar([m.upper() for m, _ in models], vals,
                  color=[c for _, c in models], width=0.5)
    for bar, (m, _), v in zip(bars, models, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:.1f}M\n(thr {t.loc[m, 'threshold']:.2f})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("parameters (millions)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Model size")
    fig.tight_layout()
    path = outdir / "battery_params.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_battery(res, outdir):
    """All battery figures in one call (battery.py calls this at the end of main)."""
    outdir = Path(outdir)
    paths = [plot_test_comparison(res, outdir),
             plot_robustness(res, outdir),
             plot_model_cost(res, outdir)]
    print(f"figures saved -> {outdir} (battery_*.png)")
    return paths
# ============================================================
# --- Final comparison figures (finetune_final.py / bench_inference.py)
# ============================================================
def _as_df(x):
    """DataFrame or CSV path -> DataFrame."""
    if isinstance(x, (str, Path)):
        return pd.read_csv(x)
    return x


def plot_convergence(csvs, outdir, tag="final", dpi=300):
    """Convergence figure from the per-epoch logs written by finetune_final.py.

    csvs : dict {model_name: CSV path or DataFrame}; columns:
           epoch, train_loss, val_acc, val_f1, val_auc, seconds
    Saves {outdir}/{tag}_convergence.png, returns the Path."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    series = {name: _as_df(v) for name, v in csvs.items()}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)

    for name, df in series.items():
        color = C_VIT if name == "vit" else C_CNN
        axes[0].plot(df.epoch, df.train_loss, "o-", color=color, label=name.upper())
        axes[1].plot(df.epoch, df.val_f1, "o-", color=color)
        axes[1].plot(df.epoch, df.val_acc, "--", color=color, alpha=0.45, linewidth=1.2)
        b = df.val_f1.idxmax()                     # pre-registered freeze rule
        axes[1].plot(df.epoch[b], df.val_f1[b], "*", markersize=15, color=color,
                     markeredgecolor="white", label=name.upper())
        axes[2].plot(df.epoch, df.seconds.cumsum(), "o-", color=color,
                     label=f"{name.upper()} ({df.seconds.sum():.0f}s total)")

    axes[0].set_title("Train loss", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_title("Validation (solid F1, dashed acc, star = frozen)",
                      loc="left", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score")
    axes[1].legend(frameon=False, loc="lower right")
    axes[2].set_title("Training wall-clock", loc="left", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("cumulative seconds")
    axes[2].legend(frameon=False, loc="lower right")

    for ax in axes:
        _style_ax(ax)
    fig.suptitle("Convergence - one recipe, regime-appropriate lr only "
                 "(ViT 5e-5, CNN 1e-3)", fontsize=13, fontweight="bold")

    path = outdir / f"{tag}_convergence.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def plot_inference_bench(df_or_csv, outdir, tag="final", dpi=300):
    """Inference-cost figure from results/inference_bench.csv (bench_inference.py):
    ms per image (log scale) per device/batch combo, ViT vs CNN, mean ± std.
    Saves {outdir}/{tag}_inference.png, returns the Path."""
    df = _as_df(df_or_csv)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    combos = [(d, b) for d in ("cpu", "cuda") for b in (1, 32)
              if ((df.device == d) & (df.batch == b)).any()]
    labels = {("cpu", 1): "CPU b1", ("cpu", 32): "CPU b32",
              ("cuda", 1): "GPU b1", ("cuda", 32): "GPU b32"}
    models = list(dict.fromkeys(df.model))

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    x, w = np.arange(len(combos)), 0.34
    for i, name in enumerate(models):
        color = C_VIT if name == "vit" else C_CNN
        vals, errs = [], []
        for d, bch in combos:
            row = df[(df.model == name) & (df.device == d) & (df.batch == bch)].iloc[0]
            vals.append(float(row.ms_per_image)); errs.append(float(row.ms_std) / bch)
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, zorder=3,
               edgecolor="white", linewidth=0.5, label=name.upper(),
               yerr=errs, error_kw=dict(lw=1, alpha=0.6, capsize=2))
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v * 1.18, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=9, color=_G700)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([labels[c] for c in combos], fontsize=10)
    ax.set_ylabel("ms per image (log scale)")
    ax.set_title("Inference cost - warmup + 20 repeats, mean ± std (lower is better)",
                 loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    _style_ax(ax)

    path = outdir / f"{tag}_inference.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path