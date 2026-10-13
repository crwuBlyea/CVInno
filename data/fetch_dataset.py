# ============================================================
# fetch_dataset.py — download the project dataset from Kaggle
# into data/raw/ (ai/, human/, metadata.csv)
# Public dataset: no Kaggle account or credentials needed.
# ============================================================
import shutil
from pathlib import Path

SLUG = "lidercrewlidercrew/aidetect"    
RAW = Path(__file__).resolve().parent / "raw"

def find_dir(cache, name):
    direct = cache / name
    if direct.is_dir():
        return direct
    hits = [p for p in cache.rglob(name) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"'{name}' folder not found in {cache}")
    return hits[0]

def find_file(cache, name):
    direct = cache / name
    if direct.is_file():
        return direct
    hits = list(cache.rglob(name))
    if not hits:
        raise FileNotFoundError(f"'{name}' not found in {cache}")
    return hits[0]

def main():
    import kagglehub
    print(f"downloading '{SLUG}' from Kaggle (first run: several GB)...")
    cache = Path(kagglehub.dataset_download(SLUG))
    print(f"cached at: {cache}")

    RAW.mkdir(parents=True, exist_ok=True)
    ai_dir    = find_dir(cache, "ai")
    human_dir = find_dir(cache, "human")
    meta_csv  = find_file(cache, "metadata.csv")

    shutil.copytree(ai_dir,    RAW / "ai",    dirs_exist_ok=True)
    shutil.copytree(human_dir, RAW / "human", dirs_exist_ok=True)
    shutil.copy(meta_csv,      RAW / "metadata.csv")

    n = len(list((RAW / "ai").glob("*"))) + len(list((RAW / "human").glob("*")))
    print(f"\ndataset ready: {n} images + metadata.csv -> {RAW}")
    print("next: python data/preprocess.py")

if __name__ == "__main__":
    main()