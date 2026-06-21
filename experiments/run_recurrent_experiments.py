"""Recurrent (RDDLGN-inspired stateful) LGN scaling experiments.

Ordered cheapest/most-informative first: L0-only variants (greedy puts L0 last, so the
recurrent layer is only in the forward during the final layer's training -> ~aggressive
cost). The multi-layer and all-layer configs are heavier and run last.

Each config -> results/report/<name>/metrics.json (skip-if-done).
"""
import os, subprocess, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CK = 'results/baseline.pt'
HM = 'results/aggressive/heatmap.json'
OUT = 'results/report'

BASE = ['--strategy', 'greedy', '--imitation_steps', '200', '--anneal_in_finetune',
        '--finetune_steps', '3000', '--learn_pool', '--heatmap', HM, '--checkpoint', CK]

R = '--recurrent'
CONFIGS = [
    # L0-only, main config + state-width sweep
    ('rec_L0_w1024',        [R, '--recurrent_layers', '0', '--recurrent_state_width', '1024']),
    ('rec_L0_w512',         [R, '--recurrent_layers', '0', '--recurrent_state_width', '512']),
    ('rec_L0_w2048',        [R, '--recurrent_layers', '0', '--recurrent_state_width', '2048']),
    # L0-only, state-init sweep
    ('rec_L0_learned',      [R, '--recurrent_layers', '0', '--recurrent_state_width', '1024', '--recurrent_state_init', 'learned']),
    ('rec_L0_residual',     [R, '--recurrent_layers', '0', '--recurrent_state_width', '1024', '--recurrent_state_init', 'residual']),
    # L0-only, depth sweep
    ('rec_L0_depth2',       [R, '--recurrent_layers', '0', '--recurrent_state_width', '1024', '--recurrent_depth', '2']),
    # multi-layer (heavier)
    ('rec_L0_L11',          [R, '--recurrent_layers', '0', '11', '--recurrent_state_width', '1024']),
    ('rec_L0_L10_L11',      [R, '--recurrent_layers', '0', '10', '11', '--recurrent_state_width', '1024']),
    # all layers recurrent (slowest — runs last)
    ('rec_all',             [R, '--recurrent_state_width', '1024']),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        out = f'{OUT}/{name}'
        if os.path.exists(f'{out}/metrics.json'):
            print(f'[skip] {name}'); continue
        print(f'\n{"="*60}\n[scale] {name}\n{"="*60}', flush=True)
        cmd = [sys.executable, 'run.py', 'scale'] + BASE + extra + ['--results_dir', out]
        print(' '.join(cmd), flush=True)
        r = subprocess.run(cmd)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED ({r.returncode})', flush=True)
    print('\nAll recurrent experiments finished.', flush=True)


if __name__ == '__main__':
    main()
