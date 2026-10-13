# ============================================================
# preprocess.py — data loading, hygiene, preprocessing, splits,
# corruption sets. Writes data/meta.csv + data/proc/
# Expects: data/raw/{ai/, human/, metadata.csv}
# ============================================================
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "data"

META_CSV = RAW / "metadata.csv"
NAME_COL, LABEL_COL = "filename", "label"     # ← adjust to your CSV columns

# ---------- 1. load + normalize labels ----------
raw = pd.read_csv(META_CSV)

def to_label(v):
    s = str(v).strip().lower()
    if s in {"ai", "fake", "generated", "1"}:  return 1
    if s in {"human", "real", "0"}:            return 0
    return None

raw["label_id"] = raw[LABEL_COL].map(to_label)
assert raw.label_id.isna().sum() == 0, \
    f"unmapped label values: {raw[LABEL_COL].unique()}"

# ---------- 2. resolve image paths ----------
files = [p for p in RAW.rglob("*")
         if p.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"}]
by_name, by_stem = {}, {}
for p in files:
    by_name.setdefault(p.name, p)
    by_stem.setdefault(p.stem, p)

def resolve(name):
    n = str(name).strip()
    for c in (RAW / n, by_name.get(Path(n).name), by_stem.get(Path(n).stem)):
        if c and Path(c).is_file():
            return c
    return None

raw["src_path"] = raw[NAME_COL].map(resolve)
missing = raw.src_path.isna().sum()
assert missing == 0, f"{missing} CSV rows have no matching image"
print(f"{len(raw)} labeled images resolved")

# ---------- 3. dedup (content hash) ----------
def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

raw["md5"] = raw.src_path.map(md5)
n_dupe = raw.duplicated("md5", keep="first").sum()
conflicts = (raw[raw.duplicated("md5", keep=False)]
             .groupby("md5").label_id.nunique() > 1).sum()
assert conflicts == 0, f"{conflicts} cross-class duplicate images — remove manually"
raw = raw.drop_duplicates("md5").reset_index(drop=True)
print(f"dedup: removed {n_dupe} duplicates -> {len(raw)} images")

# ---------- 4. preprocess: center-square, downscale-only ----------
rows = []
for i, r in enumerate(raw.itertuples()):
    img = Image.open(r.src_path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    if side > 1024:
        img = img.resize((1024, 1024), Image.LANCZOS)
    folder = "ai" if r.label_id else "human"
    out = OUT / "proc" / folder / f"{i:05d}_{Path(r.src_path).stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    rows.append(dict(path=str(out.relative_to(ROOT)), label=r.label_id, src=folder))
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(raw)}")

df = pd.DataFrame(rows)

# ---------- 5. stratified split (label x source) ----------
df["stratum"] = df.label.astype(str) + "_" + df.src
tr, test = train_test_split(df, test_size=0.10, stratify=df.stratum, random_state=42)
tr, val  = train_test_split(tr, test_size=0.111, stratify=tr.stratum, random_state=42)
df["split"] = "train"
df.loc[val.index, "split"] = "val"
df.loc[test.index, "split"] = "test"

# ---------- 6. corruption sets: JPEG q75 / q50 copies of TEST ----------
for q in (75, 50):
    add = []
    for r in df[df.split == "test"].itertuples():
        p = ROOT / r.path
        qout = OUT / f"proc_q{q}" / p.parent.name / p.name
        qout.parent.mkdir(parents=True, exist_ok=True)
        Image.open(p).convert("RGB").save(qout, "JPEG", quality=q)
        add.append(dict(path=str(qout.relative_to(ROOT)), label=r.label,
                        src=r.src, split=f"test_q{q}"))
    df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)

df.to_csv(OUT / "meta.csv", index=False)
print(df.split.value_counts())
print(df.groupby(["split", "label"]).size())
print(f"\nDONE -> {OUT / 'meta.csv'}")

(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "split_summary.txt", "w") as f:
    f.write(df.split.value_counts().to_string() + "\n\n")
    f.write(df.groupby(["split", "label"]).size().to_string() + "\n")