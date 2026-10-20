# ============================================================
# train.py — checkpoint 2: full fine-tuning of both backbones
# Usage:
#   python training/train.py --model vit
#   python training/train.py --model cnn
# One recipe; only the learning rate differs (regime-appropriate).
# Writes: training/runs/<model>/ + results/convergence_<model>.csv
# ============================================================
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torchvision.transforms import (Compose, Normalize, RandomResizedCrop,
                                    RandomHorizontalFlip, Resize, CenterCrop,
                                    ToTensor)

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "meta.csv"
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

TRAIN_TFM = Compose([RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
                     RandomHorizontalFlip(), ToTensor(), Normalize(MEAN, STD)])
EVAL_TFM  = Compose([Resize(256, antialias=True), CenterCrop(224),
                     ToTensor(), Normalize(MEAN, STD)])

class ArtDS(Dataset):
    def __init__(self, df, tfm): self.df, self.tfm = df.reset_index(drop=True), tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tfm(Image.open(ROOT / r.path).convert("RGB")), int(r.label)

def logits_of(m, x):
    out = m(x)
    return out.logits if hasattr(out, "logits") else out

def build_model(kind):
    if kind == "vit":
        from transformers import AutoModelForImageClassification
        m = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base", num_labels=2, ignore_mismatched_sizes=True)
        lr = 5e-5          # fine-tuning regime: small LR protects pretrained features
    else:
        import timm
        m = timm.create_model("resnet34", pretrained=True, num_classes=2)
        lr = 1e-3          # CNN fine-tune regime: ImageNet-pretrained nets tolerate more
    return m.to(DEV), lr, sum(p.numel() for p in m.parameters())

def evaluate(m, frame):
    m.eval()
    loader = DataLoader(ArtDS(frame, EVAL_TFM), batch_size=128, num_workers=4)
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(logits_of(m, x.to(DEV)), 1)[:, 1].cpu())
            ys.append(y)
    probs, ys = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    return dict(auc=roc_auc_score(ys, probs),
                acc=accuracy_score(ys, probs > .5),
                f1=f1_score(ys, probs > .5)), probs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vit", "cnn"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--patience", type=int, default=7)
    a = ap.parse_args()

    df = pd.read_csv(META)
    tr, va = df[df.split == "train"], df[df.split == "val"]

    model, lr, n_params = build_model(a.model)
    print(f"model={a.model}  params={n_params/1e6:.1f}M  lr={lr}  device={DEV}")

    loader = DataLoader(ArtDS(tr, TRAIN_TFM), batch_size=a.bs, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=a.epochs,
                                                steps_per_epoch=len(loader))

    outdir = ROOT / "training" / "runs" / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    log = []
    best, patience = 0.0, 0
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                loss = F.cross_entropy(logits_of(model, x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        dt = time.time() - t0

        m, probs, ys = evaluate(model, va)
        log.append(dict(model=a.model, epoch=ep, train_loss=tot/nb,
                        val_auc=m["auc"], val_acc=m["acc"], val_f1=m["f1"],
                        epoch_seconds=dt, params=n_params))
        print(f"ep {ep:02d}: loss {tot/nb:.4f}  auc {m['auc']:.4f}  "
              f"acc {m['acc']:.4f}  f1 {m['f1']:.4f}  [{dt:.0f}s]", flush=True)

        if m["auc"] > best:
            best, patience = m["auc"], 0
            if a.model == "vit":
                model.save_pretrained(outdir)
            else:
                torch.save(model.state_dict(), outdir / "model.pt")
        else:
            patience += 1
            if patience >= a.patience:
                print("early stop"); break

    pd.DataFrame(log).to_csv(ROOT / "results" / f"convergence_{a.model}.csv", index=False)
    print(f"\nDONE — best val AUC {best:.4f} -> {outdir}")

if __name__ == "__main__":
    main()