"""Batch 2 of the L0 FFN screen: the NEW signed real->binary encoding (sign + pos/neg
magnitude thermometers). Waits for batch 1 (screen_ffn_l0.py) to finish so the two don't
contend for the GPU, then measures signed encoding at n_bits 4 and 8 on L0, and reprints
the combined screen table.
"""
import json, os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report/screen_l0'
BASE = ['--hybrid_all', '--learn_pool', '--checkpoint', 'results/baseline.pt',
        '--imitation_steps', '200', '--finetune_steps', '1500', '--anneal_in_finetune',
        '--layers', '0', '--hybrid_ln2', 'copy_trainable']
SIGNED = [
    ('signed4', ['--signed_encoding', '--n_bits', '4']),
    ('signed8', ['--signed_encoding', '--n_bits', '8']),
]
# all configs to show in the final combined table (batch1 names + signed)
ALL = ['ln2_fresh', 'base_copytrain', 'ln2_frozen', 'nbits1', 'nbits4', 'nbits16',
       'calib', 'depth2', 'depth3', 'ste', 'cage', 'signed4', 'signed8']


def main():
    # wait for batch 1's last config (cage) so the GPU is free
    while not os.path.exists(f'{OUT}/cage/heatmap.json'):
        print('[signed] waiting for batch-1 screens to finish...', flush=True)
        time.sleep(60)
    for name, extra in SIGNED:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/heatmap.json'):
            print(f'[skip] {name}', flush=True); continue
        cmd = [sys.executable, '-u', 'run.py', 'heatmap'] + BASE + extra + ['--results_dir', d]
        print(f'\n$ {name}: {" ".join(extra)}', flush=True)
        r = subprocess.run(cmd)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)
    summarize()


def summarize():
    print(f'\n{"="*58}\nL0 FFN-replacement screen (lower hard_deg = better)\n{"="*58}')
    print(f'{"config":18} | {"hard_deg":>9} | {"soft_deg":>9} | {"gap":>7}')
    rows = []
    for name in ALL:
        p = f'{OUT}/{name}/heatmap.json'
        if not os.path.exists(p):
            continue
        r = json.load(open(p))[0]
        gap = r['hard_degradation'] - r['soft_degradation']
        rows.append((name, r['hard_degradation']))
        print(f'{name:18} | {r["hard_degradation"]:>9.4f} | {r["soft_degradation"]:>9.4f} | {gap:>7.4f}')
    if rows:
        best = min(rows, key=lambda x: x[1])
        print(f'\n  best (lowest hard degradation): {best[0]}  ({best[1]:+.4f})')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summarize()
    else:
        main()
