"""Screen the new LGN input/interconnect ideas.

Runs short L0/L10/L11 heatmaps on the current strong hybrid-all FFN-replacement
recipe. These are screens, not final record runs; full greedy scaling should only
follow if the screens beat the reference and the frozen-logic controls show that
the gates, not the float plumbing, earned the gain.
"""
import json
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'results/new_lgn_inputs'
BASE = [
    sys.executable, '-u', 'run.py', 'heatmap',
    '--hybrid_all', '--hybrid_ln2', 'copy_trainable',
    '--learn_pool', '--checkpoint', 'results/baseline.pt',
    '--n_bits', '8', '--grad_checkpoint', '--batch_size', '8',
    '--imitation_steps', '50', '--finetune_steps', '250',
    '--anneal_in_finetune', '--cage',
    '--lut_k', '4', '--out_gate_mult', '4', '--k', '16',
    '--out_gate_mult_layers', '0:8', '11:8', '10:8', '9:8',
    '--eval_iters', '5', '--layers', '0', '10', '11',
]

CONV = [
    '--pre_conv1d', '--pre_conv1d_channels', '128',
    '--pre_conv1d_kernel', '3', '--pre_conv1d_stride', '2',
    '--post_conv1d', '--post_conv1d_channels', '128',
    '--post_conv1d_kernel', '3', '--post_conv1d_stride', '2',
]

CONFIGS = [
    ('reference', []),
    ('lloydmax', ['--binary_encoder', 'lloydmax', '--lloyd_ema', '0.95']),
    ('topk', ['--interconnect', 'topk_block_sparse', '--topk_sparse_k', '8']),
    ('conv_prepost_s2', CONV),
    ('conv_prepost_s2_frozen', CONV + ['--freeze_logic']),
    ('combo_conv_lloyd_topk', CONV + [
        '--binary_encoder', 'lloydmax', '--lloyd_ema', '0.95',
        '--interconnect', 'topk_block_sparse', '--topk_sparse_k', '8',
    ]),
    ('combo_conv_lloyd_topk_frozen', CONV + [
        '--binary_encoder', 'lloydmax', '--lloyd_ema', '0.95',
        '--interconnect', 'topk_block_sparse', '--topk_sparse_k', '8',
        '--freeze_logic',
    ]),
]


def run_one(name, extra):
    d = os.path.join(OUT, name)
    hp = os.path.join(d, 'heatmap.json')
    if os.path.exists(hp):
        print(f'[skip] {name}', flush=True)
        return
    os.makedirs(d, exist_ok=True)
    cmd = BASE + extra + ['--results_dir', d]
    print('\n' + '=' * 80, flush=True)
    print(f'[screen] {name}', flush=True)
    print('$ ' + ' '.join(cmd), flush=True)
    r = subprocess.run(cmd)
    print(f'[screen] {name} rc={r.returncode}', flush=True)


def summarize():
    rows = []
    for name, _ in CONFIGS:
        hp = os.path.join(OUT, name, 'heatmap.json')
        if not os.path.exists(hp):
            continue
        data = json.load(open(hp))
        avg = sum(r['hard_degradation'] for r in data) / len(data)
        worst = max(r['hard_degradation'] for r in data)
        rows.append((avg, worst, name, data))
    print('\nSUMMARY hard_degradation on screened layers (lower is better)', flush=True)
    for avg, worst, name, data in sorted(rows):
        per = ' '.join(f"L{r['layer_idx']}={r['hard_degradation']:+.4f}" for r in data)
        print(f'{name:30} avg={avg:+.4f} worst={worst:+.4f}  {per}', flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        run_one(name, extra)
        summarize()
    print('\nnew_lgn_inputs screens finished.', flush=True)


if __name__ == '__main__':
    main()
