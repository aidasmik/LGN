"""Recovery: the overnight batch's last two configs (lut4_cage, keepL0_lut4) crashed on a
transient GPU/driver glitch (exit 0xC0000142) before creating output. Re-run them, then run
the training-dynamics L0 screen, then the master report. retries each failed run up to 2x.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
HYB = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
       'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint',
       '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune',
       '--batch_size', '16']

FAILED = [
    ('hybrid_all_lut4_cage', ['--lut_k', '4', '--cage']),
    ('hybrid_all_keepL0_lut4', ['--lut_k', '4', '--protected_layers', '0']),
]


def scale(name, extra, tries=2):
    d = f'{OUT}/{name}'
    if os.path.exists(f'{d}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        HYB + extra + ['--results_dir', d]
    for attempt in range(1, tries + 1):
        print(f'\n[recovery] {name} (attempt {attempt}): {" ".join(extra)}', flush=True)
        r = subprocess.run(cmd)
        if r.returncode == 0 and os.path.exists(f'{d}/metrics.json'):
            print(f'[done] {name}', flush=True); return
        print(f'[retry] {name} failed (rc={r.returncode}); waiting 60s', flush=True)
        time.sleep(60)
    print(f'!!! {name} STILL FAILING after {tries} attempts', flush=True)


def main():
    for name, extra in FAILED:
        scale(name, extra)
    print('\n[recovery] running training-dynamics screen...', flush=True)
    subprocess.run([sys.executable, '-u', 'experiments/screen_train_l0.py'])
    print('\n[recovery] master report:', flush=True)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])
    print('\nRecovery finished.', flush=True)


if __name__ == '__main__':
    main()
