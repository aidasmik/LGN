"""Pure-LGN focus pipeline (no kept FFN). Runs sequentially so nothing races for the GPU:
  1) training-dynamics L0 screen  (learning-vs-imitating + can tuning close the soft-hard gap)
  2) pure-LGN combos              (LUT arity x output gates x adaptive capacity x CAGE, all 12 LGN)
  3) master report
"""
import os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
print('[focus] 1/3 training-dynamics screen', flush=True)
subprocess.run([PY, '-u', 'experiments/screen_train_l0.py'])
print('[focus] 2/3 pure-LGN combos', flush=True)
subprocess.run([PY, '-u', 'experiments/run_combos_lgn.py'])
print('[focus] 3/3 report', flush=True)
subprocess.run([PY, 'experiments/report_ffn.py'])
print('[focus] done', flush=True)
