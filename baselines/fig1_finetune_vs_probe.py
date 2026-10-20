# Generates a clean bar comparison of the final ViT vs CNN models
# (validation metrics + wall-clock)

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Data from convergence_*.csv (best / final epoch)
# ------------------------------------------------------------------
val_f1  = [0.9444, 0.9029]
val_auc = [0.9846, 0.9687]
val_acc = [0.9443, 0.8988]
train_sec = [3008, 715]          # total wall-clock

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
C_VIT, C_CNN = "#357", "#c55"
_G700, _G300 = "#374151", "#D1D5DB"
_EDGE = "#E5E7EB"

def _style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_EDGE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, alpha=0.08, color=_G300)
    ax.set_axisbelow(True)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)

# Left panel – accuracy metrics
metrics = ["Val F1", "Val AUC", "Val Acc"]
vals_vit = [val_f1[0], val_auc[0], val_acc[0]]
vals_cnn = [val_f1[1], val_auc[1], val_acc[1]]

x, w = np.arange(len(metrics)), 0.34
axes[0].bar(x - w/2, vals_vit, width=w, color=C_VIT, zorder=3,
            edgecolor="white", linewidth=0.5, label="ViT")
axes[0].bar(x + w/2, vals_cnn, width=w, color=C_CNN, zorder=3,
            edgecolor="white", linewidth=0.5, label="CNN")

for xi, v in zip(x - w/2, vals_vit):
    axes[0].text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=9, color=_G700)
for xi, v in zip(x + w/2, vals_cnn):
    axes[0].text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=9, color=_G700)

axes[0].set_title("Validation metrics (best epoch)", loc="left",
                  fontsize=13, fontweight="bold", pad=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, fontsize=10.5)
axes[0].set_ylim(0.80, 1.08)
axes[0].set_ylabel("Score")
_style_ax(axes[0])
axes[0].legend(frameon=False, loc="upper right")

# Right panel – training cost
bars = axes[1].bar([0, 1], train_sec, color=[C_VIT, C_CNN], width=0.55,
                   edgecolor="white", linewidth=0.5, zorder=3)

axes[1].text(0, train_sec[0] + 80, "3008 s\n(~50 min)", ha="center", va="bottom",
             fontsize=10, color=_G700)
axes[1].text(1, train_sec[1] + 80, "715 s\n(~12 min)", ha="center", va="bottom",
             fontsize=10, color=_G700)

axes[1].set_title("Training wall-clock (5 epochs)", loc="left",
                  fontsize=13, fontweight="bold", pad=10)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["ViT", "CNN"], fontsize=10.5)
axes[1].set_ylabel("Seconds")
axes[1].set_ylim(0, 3800)
_style_ax(axes[1])

fig.suptitle("Final comparison — one recipe, regime-appropriate LR only\n"
             "(ViT 5e-5 · CNN 1e-3)", fontsize=14, fontweight="bold")

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
outdir = Path("results/figures")
outdir.mkdir(parents=True, exist_ok=True)
path = outdir / "fig_vit_vs_cnn.png"
fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"saved → {path}")