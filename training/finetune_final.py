# ============================================================
# finetune_final.py — trains the two FINAL comparison models
# (remaining-experiments plan, item 1).
#   ViT : DINOv2-base  (transformers)   lr 5e-5
#   CNN : ResNet-34    (timm)           lr 1e-3
# One recipe: identical optimizer / schedule / epochs / batch /
# augmentation / seed — the learning rate is the ONLY difference.
#
# Per-epoch convergence log (plan requirement):
#   train loss, val accuracy/F1/AUC, wall-clock seconds per epoch
#   -> results/convergence_vit.csv, results/convergence_cnn.csv
#   (rewritten after every epoch — crash-safe on Kaggle)
#
# Pre-registered selection rule: each model is frozen at its best
# val-F1 epoch (the test set is never touched here):
#   ViT -> training/runs/vit          (save_pretrained; battery.load_vit reads this)
#   CNN -> training/runs/cnn/model.pt (state_dict;    battery.load_cnn reads this)
#
# Run:  python scripts/finetune_final.py     (GPU, ~1-2 h)
# ============================================================
import random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, \
    ToTensor, Normalize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))               # repo root -> plot_utils.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))  # training/ -> train.py
from train import ArtDS, evaluate, DEV, ROOT
from tqdm import tqdm
from baselines import plot_utils   # convergence figure at the end

# ---------------- one shared recipe (identical for both models) ------------
SEED, BATCH, EPOCHS, WEIGHT_DECAY = 0, 32, 5, 0.01
LR = {"vit": 5e-5, "cnn": 1e-3}              # the ONLY difference (regime-appropriate)

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TRAIN_TFM = Compose([
    RandomResizedCrop(224, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

RESULTS = ROOT / "results"
RUNS    = ROOT / "training" / "runs"


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_vit():
    """DINOv2-base + fresh 2-class head (transformers)."""
    from transformers import AutoModelForImageClassification
    return AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-base", num_labels=2)


def build_cnn():
    """ResNet-34, ImageNet init, fresh 2-class head (timm)."""
    import timm
    return timm.create_model("resnet34", pretrained=True, num_classes=2)


def _logits(out):
    """HF models return an object with .logits; timm models return a tensor."""
    return out.logits if hasattr(out, "logits") else out


def run_training(name, model, val_frame, loader):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))

    log, best_f1 = [], -1.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        bar = tqdm(loader, desc=f"[{name}] epoch {epoch}/{EPOCHS}",
                   unit="batch", leave=False)
        for x, y in bar:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(_logits(model(x)), y)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
            bar.set_postfix(loss=f"{tot / max(n, 1):.4f}")
        train_loss = tot / max(n, 1)

        # per-epoch validation via the SAME evaluate() the battery uses
        model.eval()
        _, probs, ys = evaluate(model, val_frame)
        val_acc = accuracy_score(ys, probs > 0.5)
        val_f1  = f1_score(ys, probs > 0.5)
        val_auc = roc_auc_score(ys, probs)
        secs = time.time() - t0

        log.append(dict(epoch=epoch, train_loss=train_loss, val_acc=val_acc,
                        val_f1=val_f1, val_auc=val_auc, seconds=secs))
        print(f"[{name}] epoch {epoch}/{EPOCHS}  loss {train_loss:.4f}  "
              f"val acc {val_acc:.4f}  f1 {val_f1:.4f}  auc {val_auc:.4f}  "
              f"({secs:.0f}s)")

        # crash-safe incremental log
        RESULTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log).to_csv(RESULTS / f"convergence_{name}.csv", index=False)

        # pre-registered rule: freeze the best val-F1 epoch
        if val_f1 > best_f1:
            best_f1 = val_f1
            if name == "vit":
                (RUNS / "vit").mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(RUNS / "vit"))
            else:
                (RUNS / "cnn").mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), RUNS / "cnn" / "model.pt")

    print(f"[{name}] frozen at best val F1 = {best_f1:.4f}")
    return pd.DataFrame(log)


def main():
    set_seed()
    df = pd.read_csv(ROOT / "data" / "meta.csv")
    val_frame = df[df.split == "val"]
    train_frame = df[df.split == "train"]
    train_ds = ArtDS(train_frame, TRAIN_TFM) 
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        num_workers=2, pin_memory=(DEV == "cuda"))

    logs = {}
    for name, builder in [("vit", build_vit), ("cnn", build_cnn)]:
        print(DEV)
        print(f"\n=== fine-tuning {name} (lr {LR[name]:g}) ===")
        logs[name] = run_training(name, builder(), val_frame, loader)
        del builder
        if DEV == "cuda":
            torch.cuda.empty_cache()

    plot_utils.plot_convergence(logs, RESULTS / "figures")


if __name__ == "__main__":
    main()
