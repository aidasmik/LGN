"""Clean full-model LUT runs using gradient checkpointing (4.6x less memory) so they train at
full batch_size instead of the confounded batch-8 workaround.
  lut4_ckpt: batch 32 -> directly comparable to out_mult2 (38.34%) / n_bits16 (38.24%), but at
             1x gates vs their 2x. (n_bits8 baseline = 35.35%.)
  lut6_ckpt: batch 16 (LUT6 ~7GB/layer; checkpoint caps to one layer at a time) -> the
             FPGA-native-unit headline.
-> results/report/<name>/  (skip-if-done). Reuses the sensitivity heatmap for greedy order.
"""
import os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
HYB = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool',
       '--checkpoint', 'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint',
       '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune']


def scale(name, extra):
    d = f'{OUT}/{name}'
    if os.path.exists(f'{d}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        HYB + extra + ['--results_dir', d]
    print(f'\n$ {name}: {" ".join(extra)}', flush=True)
    print('[done]' if subprocess.run(cmd).returncode == 0 else f'!!! {name} FAILED', flush=True)


def main():
    scale('hybrid_all_lut4_ckpt', ['--lut_k', '4', '--batch_size', '32'])
    scale('hybrid_all_lut6_ckpt', ['--lut_k', '6', '--batch_size', '16'])
    print('\nCheckpointed LUT runs finished.', flush=True)


if __name__ == '__main__':
    main()
