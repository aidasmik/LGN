"""Follow-up from the L0 screen: input PRECISION (n_bits) is the dominant lever.
  1. n_bits=32 on L0  -> does precision saturate, or keep halving degradation?
  2. full all-layer scaling at n_bits=16 -> does the headline accuracy beat the
     n_bits=8 main run (35.35%)? (reuses the existing sensitivity heatmap for greedy order)
Each -> results/report/<name>/ (skip-if-done).
"""
import os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
COMMON = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool',
          '--checkpoint', 'results/baseline.pt']


def run(cmd, name):
    print(f'\n$ {name}: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd)
    print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)


def main():
    # 1. precision saturation check on the bottleneck layer
    if not os.path.exists(f'{OUT}/screen_l0/nbits32/heatmap.json'):
        run([sys.executable, '-u', 'run.py', 'heatmap'] + COMMON +
            ['--n_bits', '32', '--imitation_steps', '200', '--finetune_steps', '1500',
             '--anneal_in_finetune', '--layers', '0', '--results_dir', f'{OUT}/screen_l0/nbits32'],
            'nbits32_L0')

    # 2. full all-layer scaling at n_bits=16 (the headline test of the precision lever)
    if not os.path.exists(f'{OUT}/hybrid_all_nbits16/metrics.json'):
        run([sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] +
            COMMON + ['--n_bits', '16', '--imitation_steps', '200', '--finetune_steps', '3000',
                      '--anneal_in_finetune', '--results_dir', f'{OUT}/hybrid_all_nbits16'],
            'hybrid_all_nbits16')

    print('\nPrecision follow-up finished.', flush=True)


if __name__ == '__main__':
    main()
