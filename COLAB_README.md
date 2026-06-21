# Running the LGN record experiment on Google Colab

The local 8 GB GPU thrashes on the high-capacity configs (om8 runs at <100 MB free).
A Colab **T4 (16 GB)** runs them at batch 16 + best-hard with headroom — no thrashing,
~3-4x faster, and you can push past what fits locally.

## Setup (paste cells into a new Colab notebook; Runtime -> change runtime type -> GPU)

**1. Pick GPU + upload `lgn_colab.zip`** (8.7 MB, in `LGN_Nano/`):
```python
from google.colab import files
up = files.upload()          # choose lgn_colab.zip
!unzip -q lgn_colab.zip -d /content/lgn
!pip -q install datasets
```

**2. Sanity check the GPU + import:**
```python
import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory//1024**3, "GB")
%cd /content/lgn/LGN_Nano
```

## Run the record config — now at batch 16 (what wouldn't fit locally with best-hard)
```python
!python run.py scale --strategy greedy --heatmap results/report/hybrid_all_heat/heatmap.json \
  --hybrid_all --hybrid_ln2 copy_trainable --learn_pool --checkpoint results/baseline.pt \
  --n_bits 8 --grad_checkpoint --batch_size 16 --cage \
  --lut_k 4 --out_gate_mult 4 --k 16 --out_gate_mult_layers 0:8 11:8 10:8 9:8 \
  --imitation_steps 200 --finetune_steps 3000 --anneal_in_finetune \
  --ft_keep_best_hard --results_dir results/report/lgn_record
```
Final lines print `transformer` vs `LGN (hard)` accuracy. This is the clean test:
**om8_k16 + best-hard at batch 16** vs the local record **48.18%**.

## If you have more memory (T4 16 GB or Colab Pro A100/L4) — push past the local ceiling
These OOM on 8 GB but should fit on 16 GB+:
```python
# (a) om8 EVERYWHERE (not just sensitive) + best-hard
!python run.py scale --strategy greedy --heatmap results/report/hybrid_all_heat/heatmap.json \
  --hybrid_all --hybrid_ln2 copy_trainable --learn_pool --checkpoint results/baseline.pt \
  --n_bits 8 --grad_checkpoint --batch_size 16 --cage --lut_k 4 --out_gate_mult 8 --k 16 \
  --imitation_steps 200 --finetune_steps 3000 --anneal_in_finetune --ft_keep_best_hard \
  --results_dir results/report/lgn_om8_all

# (b) om16 on the sensitive layers (the L0 output-resolution curve had not saturated)
!python run.py scale --strategy greedy --heatmap results/report/hybrid_all_heat/heatmap.json \
  --hybrid_all --hybrid_ln2 copy_trainable --learn_pool --checkpoint results/baseline.pt \
  --n_bits 8 --grad_checkpoint --batch_size 16 --cage --lut_k 4 --out_gate_mult 4 --k 16 \
  --out_gate_mult_layers 0:16 11:16 10:16 9:16 8:8 7:8 \
  --imitation_steps 200 --finetune_steps 3000 --anneal_in_finetune --ft_keep_best_hard \
  --results_dir results/report/lgn_om16_sens
```

## Notes / gotchas
- **Save results before the session dies.** Colab disconnects after idle/12 h. After a run:
  `from google.colab import files; files.download('results/report/lgn_record/metrics.json')`
  (or mount Drive and write `--results_dir` under `/content/drive/MyDrive/...`).
- A run is ~1-3 h on a T4 depending on config; A100 is much faster. Keep the tab active.
- Reference numbers (local, this 8 GB box): transformer 54.82; pure-LGN record **48.18** (om8_k16);
  keepL0 (kept 1 MLP) 44.16; frozen-gates control 27.43.
- `--ft_keep_best_hard` evaluates the hard model every 500 steps and keeps the best — it's the
  +0.8 pp training lever validated on the smaller config; batch 16 lets it run without OOM.
