"""Disentangle INPUT vs OUTPUT resolution on the bottleneck layer L0.

In aggressive mode group_size == n_bits, so raising n_bits raised BOTH input precision and
output (sum_pool) resolution at once. out_gate_mult widens only the readout, so:
    n_bits8 + out_gate_mult M  ->  input res 8, output res 8*M
Compare against the input sweep (n_bits 8/16/32 = input&output 8/16/32):
  - if out_gate_mult2 (in8/out16) ~ n_bits16 (in16/out16)  -> OUTPUT resolution was the lever
  - if out_gate_mult2 ~ n_bits8   (in8/out8)               -> INPUT resolution was the lever
Each config = one L0 heatmap. Writes results/report/screen_l0/<name>/.
"""
import json, os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report/screen_l0'
BASE = ['--hybrid_all', '--learn_pool', '--checkpoint', 'results/baseline.pt',
        '--imitation_steps', '200', '--finetune_steps', '1500', '--anneal_in_finetune',
        '--layers', '0', '--hybrid_ln2', 'copy_trainable', '--n_bits', '8']
CONFIGS = [('outmult2', ['--out_gate_mult', '2']),
           ('outmult4', ['--out_gate_mult', '4']),
           ('outmult8', ['--out_gate_mult', '8'])]


def main():
    for name, extra in CONFIGS:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/heatmap.json'):
            print(f'[skip] {name}', flush=True); continue
        cmd = [sys.executable, '-u', 'run.py', 'heatmap'] + BASE + extra + ['--results_dir', d]
        print(f'\n$ {name}: {" ".join(extra)}', flush=True)
        r = subprocess.run(cmd)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)
    summarize()


def summarize():
    def deg(name):
        p = f'{OUT}/{name}/heatmap.json'
        return json.load(open(p))[0]['hard_degradation'] if os.path.exists(p) else None
    print(f'\n{"="*64}\nL0: INPUT (n_bits) vs OUTPUT (out_gate_mult) resolution\n{"="*64}')
    print(f'{"config":16} | {"input res":>9} | {"output res":>10} | {"hard_deg":>9}')
    rows = [('nbits8 (base)',   8,  8, deg('base_copytrain')),
            ('outmult2',        8, 16, deg('outmult2')),
            ('outmult4',        8, 32, deg('outmult4')),
            ('outmult8',        8, 64, deg('outmult8')),
            ('nbits16',        16, 16, deg('nbits16')),
            ('nbits32',        32, 32, deg('nbits32'))]
    for nm, ir, orr, d in rows:
        print(f'{nm:16} | {ir:>9} | {orr:>10} | {("%.4f"%d) if d is not None else "-":>9}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summarize()
    else:
        main()
