"""Per-layer adaptive precision: base n_bits=8 everywhere, but n_bits=16 only on the
quantization-sensitive layers (L0, L9, L10, L11). Tests whether we can recover most of
the blanket-n_bits16 gain (35.35 -> 38.24%) while spending the extra bits on just 4 of 12
layers (much cheaper). Reuses the existing sensitivity heatmap for greedy order.
"""
import os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
COMMON = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool',
          '--checkpoint', 'results/baseline.pt']


def run(name, extra):
    if os.path.exists(f'{OUT}/{name}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        COMMON + ['--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune'] + \
        extra + ['--results_dir', f'{OUT}/{name}']
    print(f'\n$ {name}: {" ".join(extra)}', flush=True)
    r = subprocess.run(cmd)
    print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)


def main():
    # adaptive: cheap base (8) + precision (16) on the 4 sensitive layers
    run('hybrid_all_adaptive', ['--n_bits', '8', '--precision_layers', '0', '9', '10', '11',
                                '--high_n_bits', '16'])
    print('\nAdaptive-precision run finished.', flush=True)


if __name__ == '__main__':
    main()
