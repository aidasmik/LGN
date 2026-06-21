"""Next two pure-LGN pushes past 46.09% (both config-only, proven unsaturated levers):
  1) lgn_lut6_om4_cage : LUT6 beat LUT4 in every prior comparison; the winner uses LUT4.
  2) lgn_best_om8_k16  : winner + out_gate_mult8 on the sensitive layers (L0 curve was not
                         saturated at om4) + k=16 candidate pool (big L0 gain, never run full).
Both batch 16 + grad_checkpoint (same as the 46.09 winner -> directly comparable).
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
BASE = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT,
        '--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
        'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint', '--batch_size', '16',
        '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune', '--cage']

CONFIGS = [
    # om8_k16 first (LUT4-based, fits comfortably). LUT6+om4 at batch 16 thrashed (~7.5s/step
    # at 122MiB free) -> re-queued LAST at batch 8 where it has headroom.
    ('lgn_best_om8_k16',  ['--lut_k', '4', '--out_gate_mult', '4', '--k', '16',
                           '--out_gate_mult_layers', '0:8', '11:8', '10:8', '9:8']),
    ('lgn_lut6_om4_cage', ['--lut_k', '6', '--out_gate_mult', '4', '--batch_size', '8']),
]


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 0


def main():
    while gpu_free() < 5000:
        print(f'[next2] GPU busy (free {gpu_free()}) - waiting...', flush=True); time.sleep(120)
    for name, extra in CONFIGS:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/metrics.json'):
            print(f'[skip] {name}', flush=True); continue
        print(f'\n[next2] {name}: {" ".join(extra)}', flush=True)
        for attempt in (1, 2):
            r = subprocess.run(BASE + extra + ['--results_dir', d])
            if r.returncode == 0 and os.path.exists(f'{d}/metrics.json'):
                print(f'[done] {name}', flush=True); break
            print(f'[retry] {name} rc={r.returncode}', flush=True); time.sleep(60)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])
    print('\nnext2 finished.', flush=True)


if __name__ == '__main__':
    main()
