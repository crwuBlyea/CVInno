# ============================================================
# probe.py — baseline 1: logistic regression on frozen DINOv2-base
# CLS-token embeddings (768-d). Validates the full pipeline and sets
# the floor for the fine-tuning comparison.
# ============================================================
import numpy as np, pandas as pd, torch, joblib
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoModel
from plot_utils import plot_all

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "meta.csv"
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEV}")

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EVAL_TFM = Compose([Resize(256, antialias=True), CenterCrop(224),
                    ToTensor(), Normalize(MEAN, STD)])

df = pd.read_csv(META)
tr, va = df[df.split == "train"], df[df.split == "val"]
print(f"train {len(tr)} | val {len(va)}")

model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()

def embed(paths, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            xs = torch.stack([EVAL_TFM(Image.open(ROOT / p).convert("RGB"))
                              for p in paths[i:i+bs]]).to(DEV)
            out.append(model(pixel_values=xs).last_hidden_state[:, 0].cpu())
            print(f"  {i:5d}/{len(paths)}", end="\r", flush=True)
    print()
    return torch.cat(out).numpy()

print("embedding train..."); X_tr = embed(tr.path.values)
print("embedding val...");   X_va = embed(va.path.values)

clf = LogisticRegression(max_iter=2000, C=10).fit(X_tr, tr.label.values)
probs = clf.predict_proba(X_va)[:, 1]
y = va.label.values

auc = roc_auc_score(y, probs)
acc = accuracy_score(y, probs > .5)
f1  = f1_score(y, probs > .5)

print(f"\nBASELINE 1 — linear probe on frozen dinov2-base:")
print(f"  val AUC: {auc:.4f}   accuracy: {acc:.4f}   F1: {f1:.4f}")

va = va.copy(); va["p"] = probs
for s in va.src.unique():
    sub = va[va.src == s]
    if (sub.label == 1).any():
        print(f"  {s:8s}  AI recall: {(sub[sub.label==1].p > .5).mean():.3f}")
    if (sub.label == 0).any():
        print(f"  {s:8s}  human fpr: {(sub[sub.label==0].p > .5).mean():.3f}")

(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "baseline_metrics.md", "a") as f:
    f.write(f"\n## Baseline 1 — frozen dinov2-base + logistic regression\n")
    f.write(f"val AUC {auc:.4f} | accuracy {acc:.4f} | F1 {f1:.4f} "
            f"| device {DEV} | n_train {len(tr)} | n_val {len(va)}\n")
figs = plot_all(probs, y, va,
                name="frozen dinov2-base + logistic regression",
                tag="baseline_vit",
                outdir=ROOT / "results" / "figures")
np.save(ROOT / "results" / "X_va_vit.npy", X_va)
joblib.dump(clf, ROOT / "results" / "probe_vit.joblib")
print("\nDONE — results appended to results/baseline_metrics.md")