"""Honesty control + noise check for the ensemble (#5) L0 winner.
  ens2_frozen : ensemble2 with --freeze_logic  -> LGN_contrib = frozen_hard - real_hard
  ref_s7      : reference config, different seed -> noise floor of hard_degradation
  ens2_s7     : ensemble2, different seed        -> is the -0.013 edge stable across seeds?
real refs: reference=0.0737, ensemble2=0.0609 (seed 1337).
"""
import json
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/new_lgn_inputs/readout_l0'
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
    ('ens2_frozen', ['--ensemble', '2', '--freeze_logic']),
    ('ref_s7',      ['--seed', '7']),
    ('ens2_s7',     ['--ensemble', '2', '--seed', '7']),
]
for name, extra in CONFIGS:
    d = os.path.join(OUT, name)
    if os.path.exists(os.path.join(d, 'heatmap.json')):
        print('[skip]', name, flush=True); continue
    print('\n' + '=' * 70 + '\n[control]', name, flush=True)
    subprocess.run(BASE + extra + ['--results_dir', d])

print('\nCONTROL SUMMARY (hard_degradation, lower=better)', flush=True)
for n in ['ref_s7', 'ens2_s7', 'ens2_frozen']:
    p = os.path.join(OUT, n, 'heatmap.json')
    if os.path.exists(p):
        r = json.load(open(p))[0]
        print('%-14s hard=%+.4f soft=%+.4f' % (n, r['hard_degradation'], r['soft_degradation']), flush=True)
