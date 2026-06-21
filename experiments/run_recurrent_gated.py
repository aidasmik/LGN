"""Gated (flip-flop/latch-inspired) recurrent LGN vs vanilla recurrent.

Goal: isolate the GATING effect from state width. We already have vanilla recurrent
at L0 for w1024/w2048, so we run the *gated* variant at the SAME widths and compare:
  (1) gated@w1024 vs vanilla@w1024  -> pure retention/keep-gate effect
  (2) does gated saturate later than vanilla as width grows (w1024 -> w2048)?

Waits for the still-running `rec_all` job to finish (its metrics.json) before starting,
so the two don't contend for the GPU. Skip-if-done per config.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CK = 'results/baseline.pt'
HM = 'results/aggressive/heatmap.json'
OUT = 'results/report'
REC_ALL_DONE = f'{OUT}/rec_all/metrics.json'

BASE = ['--strategy', 'greedy', '--imitation_steps', '200', '--anneal_in_finetune',
        '--finetune_steps', '3000', '--learn_pool', '--heatmap', HM, '--checkpoint', CK]

G = ['--recurrent', '--recurrent_gated']
CONFIGS = [
    ('rec_gated_L0_w1024', G + ['--recurrent_layers', '0', '--recurrent_state_width', '1024']),
    ('rec_gated_L0_w2048', G + ['--recurrent_layers', '0', '--recurrent_state_width', '2048']),
]


def wait_for_rec_all():
    if os.path.exists(REC_ALL_DONE):
        print('[gated] rec_all already done -> starting immediately.', flush=True)
        return
    print('[gated] waiting for rec_all to finish (polling its metrics.json)...', flush=True)
    while not os.path.exists(REC_ALL_DONE):
        time.sleep(60)
    print('[gated] rec_all finished -> GPU free, starting gated runs.', flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    # rec_all was intentionally stopped (too slow) -> start gated immediately unless told to wait.
    if os.environ.get('GATED_WAIT_REC_ALL') == '1':
        wait_for_rec_all()
    for name, extra in CONFIGS:
        out = f'{OUT}/{name}'
        if os.path.exists(f'{out}/metrics.json'):
            print(f'[skip] {name}', flush=True); continue
        print(f'\n{"="*60}\n[scale] {name}\n{"="*60}', flush=True)
        cmd = [sys.executable, '-u', 'run.py', 'scale'] + BASE + extra + ['--results_dir', out]
        print(' '.join(cmd), flush=True)
        r = subprocess.run(cmd)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED ({r.returncode})', flush=True)
    print('\nGated recurrent experiments finished.', flush=True)


if __name__ == '__main__':
    main()
