"""Test whether a more expressive gate PRIMITIVE beats more 2-input gates at equal gate count.
The capacity finding said degradation is bound by gate count; this asks if a K-input LUT gate
(any function of K inputs = 1 FPGA LUT-K) does more per gate than the 2-input gate.

  Phase 1 (cheap): L0 screen. lut3 / lut4 at 1x gate count vs the 2-input base (0.4346) and
                   out_mult2 (0.1995, 2x 2-input gates). Same n_bits=8 input everywhere.
  Phase 2: full all-layer lut4 scaling (the headline vs n_bits8 baseline 35.35%).
  Phase 3: lut4 identity control (dead gates) -> honesty check.

-> results/report/...  (skip-if-done). Reuses the sensitivity heatmap for greedy order.
"""
import json, os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
HYB = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool',
       '--checkpoint', 'results/baseline.pt', '--n_bits', '8']


def heat(name, extra):
    d = f'{OUT}/screen_l0/{name}'
    if os.path.exists(f'{d}/heatmap.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'heatmap'] + HYB + \
        ['--imitation_steps', '200', '--finetune_steps', '1500', '--anneal_in_finetune',
         '--layers', '0'] + extra + ['--results_dir', d]
    print(f'\n$ L0 {name}: {" ".join(extra)}', flush=True)
    print('[done]' if subprocess.run(cmd).returncode == 0 else f'!!! {name} FAILED', flush=True)


def scale(name, extra):
    d = f'{OUT}/{name}'
    if os.path.exists(f'{d}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        HYB + ['--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune'] + \
        extra + ['--results_dir', d]
    print(f'\n$ scale {name}: {" ".join(extra)}', flush=True)
    print('[done]' if subprocess.run(cmd).returncode == 0 else f'!!! {name} FAILED', flush=True)


def l0_summary():
    def deg(n):
        p = f'{OUT}/screen_l0/{n}/heatmap.json'
        return json.load(open(p))[0]['hard_degradation'] if os.path.exists(p) else None
    print(f'\n{"="*60}\nL0: gate PRIMITIVE vs more gates (lower = better; all n_bits8)\n{"="*60}')
    rows = [('2-input base', '1x gates', deg('base_copytrain')),
            ('lut3 (LUT3)',  '1x gates', deg('lut3')),
            ('lut4 (LUT4)',  '1x gates', deg('lut4')),
            ('out_mult2',    '2x gates', deg('outmult2')),
            ('out_mult4',    '4x gates', deg('outmult4'))]
    for nm, cost, d in rows:
        print(f'{nm:16} | {cost:9} | {("%.4f"%d) if d is not None else "-":>9}')


def main():
    heat('lut3', ['--lut_k', '3'])
    heat('lut4', ['--lut_k', '4'])
    l0_summary()
    scale('hybrid_all_lut4', ['--lut_k', '4'])
    scale('hybrid_all_lut4_identity', ['--lut_k', '4', '--identity_logic', '--imitation_steps', '0'])
    print('\nLUT experiment finished.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'l0':
        l0_summary()
    else:
        main()
