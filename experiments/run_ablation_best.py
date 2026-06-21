"""Matched honesty control for the 46.09% winner: IDENTICAL architecture and training
(LUT4 x out_gate_mult4 x CAGE, batch 16) but --freeze_logic -> gates stay at RANDOM init,
only plumbing (ln_2, pool affine) trains. identity_logic can't do this (shape-incompatible
with out_gate_mult>1). The 46.09-minus-this gap = accuracy genuinely earned by LEARNED logic.
Waits for the GPU to free (the max variant run). -> results/report/lgn_best_frozen_control/.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
CMD = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT,
       '--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
       'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint', '--batch_size', '16',
       '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune',
       '--lut_k', '4', '--out_gate_mult', '4', '--cage',
       '--freeze_logic', '--results_dir', f'{OUT}/lgn_best_frozen_control']


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 0


def main():
    if os.path.exists(f'{OUT}/lgn_best_frozen_control/metrics.json'):
        print('[skip] control already done'); return
    while True:
        if gpu_free() >= 5000:
            time.sleep(20)
            if gpu_free() >= 5000:
                break
        print(f'[ablation] GPU busy (free {gpu_free()}) - waiting...', flush=True)
        time.sleep(120)
    for attempt in (1, 2):
        r = subprocess.run(CMD)
        if r.returncode == 0 and os.path.exists(f'{OUT}/lgn_best_frozen_control/metrics.json'):
            print('[done] frozen control', flush=True); break
        print(f'[retry] rc={r.returncode}', flush=True); time.sleep(60)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])


if __name__ == '__main__':
    main()
