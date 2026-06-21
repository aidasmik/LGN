"""Follow-up: wider recurrent state at L0 (state width is the main lever)."""
import os, subprocess, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CK = 'results/baseline.pt'; HM = 'results/aggressive/heatmap.json'; OUT = 'results/report'
BASE = ['--strategy', 'greedy', '--imitation_steps', '200', '--anneal_in_finetune',
        '--finetune_steps', '3000', '--learn_pool', '--heatmap', HM, '--checkpoint', CK]
CONFIGS = [
    ('rec_L0_w3072', ['--recurrent', '--recurrent_layers', '0', '--recurrent_state_width', '3072']),
    ('rec_L0_w4096', ['--recurrent', '--recurrent_layers', '0', '--recurrent_state_width', '4096']),
]
os.makedirs(OUT, exist_ok=True)
for name, extra in CONFIGS:
    out = f'{OUT}/{name}'
    if os.path.exists(f'{out}/metrics.json'):
        print(f'[skip] {name}', flush=True); continue
    print(f'\n[scale] {name}', flush=True)
    r = subprocess.run([sys.executable, 'run.py', 'scale'] + BASE + extra + ['--results_dir', out])
    print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)
print('Wide recurrent experiments finished.', flush=True)
