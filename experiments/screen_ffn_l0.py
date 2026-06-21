"""Small controlled screen for LGN-as-FFN quality, isolated on the BOTTLENECK layer L0
(attention kept frozen everywhere; only L0's FFN -> LGN, measured independently).

L0 is where the signal is (+0.43 hard degradation in the main run) so improvements show
clearly and cheaply. Each config = one heatmap on --layers 0 -> L0 hard/soft degradation.
Vary ONE thing from a fixed base (copy_trainable ln_2, n_bits=8, depth=1, no calib/STE).

Run: python experiments/screen_ffn_l0.py   (writes results/report/screen_l0/<name>/)
"""
import json, os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report/screen_l0'
BASE = ['--hybrid_all', '--learn_pool', '--checkpoint', 'results/baseline.pt',
        '--imitation_steps', '200', '--finetune_steps', '1500', '--anneal_in_finetune',
        '--layers', '0']
T = ['--hybrid_ln2', 'copy_trainable']     # the fixed base ln_2 mode for non-ln2 screens

CONFIGS = [
    # P1.1 ln_2 handling
    ('ln2_fresh',        ['--hybrid_ln2', 'fresh']),
    ('base_copytrain',   T),                                  # base: copy_trainable, n_bits8, depth1
    ('ln2_frozen',       ['--hybrid_ln2', 'copy_frozen']),
    # P1.2 encoding bits
    ('nbits1',           T + ['--n_bits', '1']),
    ('nbits4',           T + ['--n_bits', '4']),
    ('nbits16',          T + ['--n_bits', '16']),
    # P1.3 calibration
    ('calib',            T + ['--learn_binary_calibration']),
    # P3 capacity (depth)
    ('depth2',           T + ['--depth', '2']),
    ('depth3',           T + ['--depth', '3']),
    # P5 hard-forward training
    ('ste',              T + ['--ste']),
    ('cage',             T + ['--cage']),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/heatmap.json'):
            print(f'[skip] {name}', flush=True); continue
        cmd = [sys.executable, '-u', 'run.py', 'heatmap'] + BASE + extra + ['--results_dir', d]
        print(f'\n$ {name}: {" ".join(extra)}', flush=True)
        r = subprocess.run(cmd)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)
    summarize()


def summarize():
    print(f'\n{"="*58}\nL0 FFN-replacement screen (lower degradation = better)\n{"="*58}')
    print(f'{"config":18} | {"hard_deg":>9} | {"soft_deg":>9} | {"gap":>7}')
    rows = []
    for name, _ in CONFIGS:
        p = f'{OUT}/{name}/heatmap.json'
        if not os.path.exists(p):
            continue
        r = json.load(open(p))[0]
        gap = r['hard_degradation'] - r['soft_degradation']
        rows.append((name, r['hard_degradation'], r['soft_degradation'], gap))
        print(f'{name:18} | {r["hard_degradation"]:>9.4f} | {r["soft_degradation"]:>9.4f} | {gap:>7.4f}')
    if rows:
        best = min(rows, key=lambda x: x[1])
        print(f'\n  best (lowest hard degradation): {best[0]}  ({best[1]:+.4f})')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summarize()
    else:
        main()
