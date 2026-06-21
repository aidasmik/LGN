"""Decisive pure-LGN runs (all 12 layers logic, no FFN kept). Take the winning combo
(LUT4 x out_gate_mult4 x CAGE = 43.40% at batch 8) and:
  1) lgn_best_b32      : run it CLEAN at batch 32 -> fair vs keepL0 (44.16, also batch 32).
                         Does pure LGN beat keep-an-MLP?
  2) lgn_best_max_b32  : + functional init (MLP-guided candidate connections) + weighted_pool.
                         Do the new LGN-only ideas push higher on top?
Both batch 32 (grad_checkpoint) so they're mutually comparable and comparable to keepL0.
Waits for the GPU to free. -> results/report/<name>/.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
# batch 16 (not 32): batch32 + LUT4 + om4 + CAGE + checkpoint is ~6-8h/run and OOM-tight.
# Mild confound vs keepL0 (batch32) -- but if these hit >=44 at a batch disadvantage, it's a
# clear pure-LGN win. (Winner was 43.40 at batch 8; batch16 should be a bit higher.)
BASE = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
        'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint', '--batch_size', '16',
        '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune',
        '--lut_k', '4', '--out_gate_mult', '4', '--cage']

CONFIGS = [
    ('lgn_best_b16',     []),
    ('lgn_best_max_b16', ['--mlp_guided_init', '--weighted_pool']),
]


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 0


def main():
    while True:
        if gpu_free() >= 5000:
            time.sleep(20)
            if gpu_free() >= 5000:
                break
        print(f'[best] GPU busy (free {gpu_free()}) - waiting...', flush=True); time.sleep(120)
    for name, extra in CONFIGS:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/metrics.json'):
            print(f'[skip] {name}', flush=True); continue
        cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
            BASE + extra + ['--results_dir', d]
        print(f'\n[best] {name}: {" ".join(extra)}', flush=True)
        for attempt in (1, 2):                              # auto-retry a transient GPU glitch
            r = subprocess.run(cmd)
            if r.returncode == 0 and os.path.exists(f'{d}/metrics.json'):
                print(f'[done] {name}', flush=True); break
            print(f'[retry] {name} rc={r.returncode}', flush=True); time.sleep(60)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])
    print('\nBest-config runs finished.', flush=True)


if __name__ == '__main__':
    main()
