"""Quick TRAINING-dynamics screen on L0 (the bottleneck), canonical 2-input gate.
Answers cheaply: is the LGN limited by learning (soft) or by discretization (hard gap),
and does tuning the training close the gap? Each config = one L0 heatmap (~2 min); we read
BOTH soft_degradation and hard_degradation, and the gap between them.

Questions tested:
  * imitate vs learn-itself : imitation_steps {0, 200(base), 800}
  * commitment / hard gap   : ent_gate {0.02(base), 0.05, 0.1}; temp_end {0.05, 0.1(base), 0.2}
  * imitation target        : imit_loss {mse(base), kl}; ft_imit_weight {0, 0.5}
  * hard-forward            : ste, cage

Waits for the GPU (overnight batch) to free first. -> results/report/screen_l0_train/<name>/.
"""
import json, os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report/screen_l0_train'
BASE = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
        'results/baseline.pt', '--n_bits', '8', '--imitation_steps', '200',
        '--finetune_steps', '1500', '--anneal_in_finetune', '--layers', '0', '--batch_size', '8']

CONFIGS = [
    ('base',        []),
    ('imit0',       ['--imitation_steps', '0']),
    ('imit800',     ['--imitation_steps', '800']),
    ('entgate05',   ['--ent_gate', '0.05', '--ft_ent_gate', '0.025']),
    ('entgate10',   ['--ent_gate', '0.10', '--ft_ent_gate', '0.05']),
    ('tempend05',   ['--temp_end', '0.05']),
    ('tempend20',   ['--temp_end', '0.20']),
    ('imitkl',      ['--imit_loss', 'kl']),
    ('ftimit05',    ['--ft_imit_weight', '0.5']),
    ('ste',         ['--ste']),
    ('cage',        ['--cage']),
]


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 9999


def main():
    # Run when the GPU is genuinely idle (called in sequence by the recovery orchestrator after
    # the batch finishes). Require it free for two consecutive checks so we don't catch a gap.
    while True:
        if gpu_free() >= 5000:
            time.sleep(20)
            if gpu_free() >= 5000:
                break
        print(f'[train-screen] GPU busy (free {gpu_free()} MiB) - waiting...', flush=True)
        time.sleep(60)
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/heatmap.json'):
            print(f'[skip] {name}', flush=True); continue
        cmd = [sys.executable, '-u', 'run.py', 'heatmap'] + BASE + extra + ['--results_dir', d]
        print(f'\n$ {name}: {" ".join(extra)}', flush=True)
        subprocess.run(cmd)
    summarize()


def summarize():
    print(f'\n{"="*60}\nL0 training-dynamics screen (2-input gate)\n{"="*60}')
    print(f'{"config":12} | {"soft_deg":>9} | {"hard_deg":>9} | {"gap":>7}')
    rows = []
    for name, _ in CONFIGS:
        p = f'{OUT}/{name}/heatmap.json'
        if not os.path.exists(p):
            continue
        r = json.load(open(p))[0]
        gap = r['hard_degradation'] - r['soft_degradation']
        rows.append((name, r['soft_degradation'], r['hard_degradation'], gap))
        print(f'{name:12} | {r["soft_degradation"]:>9.4f} | {r["hard_degradation"]:>9.4f} | {gap:>7.4f}')
    if rows:
        bh = min(rows, key=lambda x: x[2]); bg = min(rows, key=lambda x: x[3])
        print(f'\n  best hard_deg: {bh[0]} ({bh[2]:+.4f}) | smallest gap: {bg[0]} ({bg[3]:+.4f})')


if __name__ == '__main__':
    summarize() if (len(sys.argv) > 1 and sys.argv[1] == 'summary') else main()
