# ============================================================
# battery.py — the ONE-TIME final evaluation on the test set
# (and its q75/q50 corrupted copies). This script is run once,
# after both models are trained and frozen.
# Writes: results/battery_results.csv + results/robustness_summary.csv
#         + figures via the shared plot_utils module:
#           results/figures/battery_*.png
# ============================================================
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))               # repo root  -> plot_utils.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))  # training/  -> train.py
from train import ArtDS, EVAL_TFM, logits_of, evaluate, DEV, ROOT

import plot_utils   # shared plotting module (battery figures live here now)

META = ROOT / "data" / "meta.csv"

def load_vit():
    from transformers import AutoModelForImageClassification
    return AutoModelForImageClassification.from_pretrained(
        str(ROOT / "training" / "runs" / "vit")).to(DEV).eval()

def load_cnn():
    import timm
    m = timm.create_model("resnet34", num_classes=2).to(DEV)
    m.load_state_dict(torch.load(ROOT / "training" / "runs" / "cnn" / "model.pt",
                                 map_location=DEV))
    return m.eval()

def main():
    df = pd.read_csv(META)
    val   = df[df.split == "val"]
    vit, cnn = load_vit(), load_cnn()

    # ---- fit decision threshold on VALIDATION (never test) ----
    _, vit_val_probs, _ = evaluate(vit, val)
    _, cnn_val_probs, _ = evaluate(cnn, val)
    y_val = val.label.values
    candidates = np.linspace(0.05, 0.95, 91)
    thr_vit = max(candidates, key=lambda t: f1_score(y_val, (vit_val_probs > t).astype(int)))
    thr_cnn = max(candidates, key=lambda t: f1_score(y_val, (cnn_val_probs > t).astype(int)))

    ROWS = []
    for name, m, thr in [("vit", vit, thr_vit), ("cnn", cnn, thr_cnn)]:
        for split in ["test", "test_q75", "test_q50"]:
            frame = df[df.split == split]
            metrics, probs, ys = evaluate(m, frame)
            pred = probs > thr
            ROWS.append(dict(
                model=name, split=split,
                auc=metrics["auc"],
                acc=accuracy_score(ys, pred),
                f1=f1_score(ys, pred),
                threshold=thr,
                n_params=sum(p.numel() for p in m.parameters()),
            ))
            print(f"{name:4s} {split:9s}: acc {ROWS[-1]['acc']:.4f}  "
                  f"f1 {ROWS[-1]['f1']:.4f}  auc {ROWS[-1]['auc']:.4f}")
        del m
    if DEV == "cuda":
        torch.cuda.empty_cache()

    res = pd.DataFrame(ROWS)
    res.to_csv(ROOT / "results" / "battery_results.csv", index=False)

    piv = res.pivot(index="model", columns="split", values="f1")
    piv["q75_drop"] = piv["test"] - piv["test_q75"]
    piv["q50_drop"] = piv["test"] - piv["test_q50"]
    print("\n", piv.round(4))
    piv.to_csv(ROOT / "results" / "robustness_summary.csv")

    plot_utils.plot_battery(res, ROOT / "results" / "figures")

if __name__ == "__main__":
    main()
