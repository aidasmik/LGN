"""Cheap L0-only Conv1D variant screen."""
import json
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'results/new_lgn_inputs/conv_l0_variants'
BASE = [
    sys.executable, '-u', 'run.py', 'heatmap',
    '--hybrid_all', '--hybrid_ln2', 'copy_trainable',
    '--learn_pool', '--checkpoint', 'results/baseline.pt',
    '--n_bits', '8', '--grad_checkpoint', '--batch_size', '8',
    '--imitation_steps', '50', '--finetune_steps', '250',
    '--anneal_in_finetune', '--cage',
    '--lut_k', '4', '--out_gate_mult', '4', '--k', '16',
    '--out_gate_mult_layers', '0:8', '11:8', '10:8', '9:8',
    '--eval_iters', '5', '--layers', '0',
]

CONFIGS = [
    ('pre_s1_c128', [
        '--pre_conv1d', '--pre_conv1d_channels', '128',
        '--pre_conv1d_kernel', '3', '--pre_conv1d_stride', '1',
    ]),
    ('post_s1_c128', [
        '--post_conv1d', '--post_conv1d_channels', '128',
        '--post_conv1d_kernel', '3', '--post_conv1d_stride', '1',
    ]),
    ('post_s2_c2', [
        '--post_conv1d', '--post_conv1d_channels', '2',
        '--post_conv1d_kernel', '3', '--post_conv1d_stride', '2',
    ]),
    ('prepost_s1_c128', [
        '--pre_conv1d', '--pre_conv1d_channels', '128',
        '--pre_conv1d_kernel', '3', '--pre_conv1d_stride', '1',
        '--post_conv1d', '--post_conv1d_channels', '128',
        '--post_conv1d_kernel', '3', '--post_conv1d_stride', '1',
    ]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        d = os.path.join(OUT, name)
        hp = os.path.join(d, 'heatmap.json')
        if os.path.exists(hp):
            print(f'[skip] {name}', flush=True)
            continue
        print('\n' + '=' * 80, flush=True)
        print(f'[conv-l0] {name}', flush=True)
        subprocess.run(BASE + extra + ['--results_dir', d])
    rows = []
    for name, _ in CONFIGS:
        hp = os.path.join(OUT, name, 'heatmap.json')
        if os.path.exists(hp):
            r = json.load(open(hp))[0]
            rows.append((r['hard_degradation'], name, r))
    print('\nCONV L0 SUMMARY hard_degradation (lower is better)', flush=True)
    for d, name, r in sorted(rows):
        print(f'{name:18} hard={d:+.4f} soft={r["soft_degradation"]:+.4f}', flush=True)


if __name__ == '__main__':
    main()
