
---

## Quick start (reproduction)

```bash
pip install -r requirements.txt

# 1. Download & preprocess
python data/fetch_dataset.py
python data/preprocess.py

# 2. Baselines (Checkpoint 1)
python baselines/probe.py
python baselines/probe_cnn.py

# 3. Full fine-tuning (Checkpoint 2)
python training/finetune_final.py
