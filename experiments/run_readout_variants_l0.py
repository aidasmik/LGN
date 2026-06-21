"""Cheap L0-only screen for the three readout ideas:
  #2 pool_curve      learned per-channel nonlinear count->value readout
  #4 residual_scale  per-channel learned alpha on the LGN contribution
  #5 ensemble        N parallel gate banks, averaged (candidate-lottery variance reduction)
Same BASE as run_conv_variants_l0.py so results compare to results/new_lgn_inputs/reference
(L0 hard_degradation = 0.0737). Lower hard_degradation is better.
"""
import json
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'results/new_lgn_inputs/readout_l0'
REF = 'results/new_lgn_inputs/reference/heatmap.json'   # hard_deg 0.0737
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
    ('pool_curve',        ['--pool_curve']),
    ('residual_scale',    ['--residual_scale']),
    ('curve_res',         ['--pool_curve', '--residual_scale']),
    ('ensemble2',         ['--ensemble', '2']),
    ('ensemble3',         ['--ensemble', '3']),
    ('ens2_curve_res',    ['--ensemble', '2', '--pool_curve', '--residual_scale']),
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
        print(f'[readout-l0] {name}', flush=True)
        subprocess.run(BASE + extra + ['--results_dir', d])
    ref = json.load(open(REF))[0]['hard_degradation'] if os.path.exists(REF) else None
    rows = []
    for name, _ in CONFIGS:
        hp = os.path.join(OUT, name, 'heatmap.json')
        if os.path.exists(hp):
            r = json.load(open(hp))[0]
            rows.append((r['hard_degradation'], name, r))
    print('\nREADOUT L0 SUMMARY  hard_degradation (lower=better)  ref=%.4f' %
          (ref if ref is not None else float('nan')), flush=True)
    for d, name, r in sorted(rows):
        delta = (' delta %+.4f' % (d - ref)) if ref is not None else ''
        print('%-16s hard=%+.4f soft=%+.4f%s' % (name, d, r['soft_degradation'], delta), flush=True)


if __name__ == '__main__':
    main()
