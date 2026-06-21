"""Full all-layer confirmation that OUTPUT resolution (not input n_bits) is the LGN-as-FFN
lever. Keep input cheap (n_bits=8) and widen only the readout:
  out_mult2 -> output res 16 (compare vs n_bits16 full = 38.24%)
  out_mult4 -> output res 32 (compare vs n_bits32; push toward transformer)
Reuses the existing sensitivity heatmap for greedy order. -> results/report/<name>/.
"""
import os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
COMMON = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool',
          '--checkpoint', 'results/baseline.pt', '--n_bits', '8',
          '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune']


def run(name, extra):
    if os.path.exists(f'{OUT}/{name}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        COMMON + extra + ['--results_dir', f'{OUT}/{name}']
    print(f'\n$ {name}: {" ".join(extra)}', flush=True)
    r = subprocess.run(cmd)
    print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)


def main():
    run('hybrid_all_outmult2', ['--out_gate_mult', '2'])
    run('hybrid_all_outmult4', ['--out_gate_mult', '4'])
    print('\nOutput-resolution full runs finished.', flush=True)


if __name__ == '__main__':
    main()
