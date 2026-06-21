"""Does gate arity keep paying? LUT5/LUT6 L0 screen (the FPGA-native LUT6 is the key one),
plus memory-safe full LUT4/LUT6 runs.

L0 screen: batch 32 (single layer fits) -> clean apples-to-apples arity comparison vs
base/lut3/lut4/out_mult. Full runs: --batch_size 8 (the memory fix; LUT-K at 12 layers is
memory-heavy) -> absolute headline (note: smaller batch than the n_bits8/out_mult baselines,
so treat the full LUT number as confirmation, with the L0 screen as the precise comparison).
"""
import json, os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
HYB = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool',
       '--checkpoint', 'results/baseline.pt', '--n_bits', '8']


def heat(name, extra):
    # LUT6 (2^6 table) is memory-heavy even single-layer -> batch 16 for safe margin during
    # finetune. Minor confound vs the batch-32 base/lut3/lut4 screens, but far below the
    # ~0.1 arity steps we're resolving, so the trend stays clear.
    d = f'{OUT}/screen_l0/{name}'
    if os.path.exists(f'{d}/heatmap.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'heatmap'] + HYB + \
        ['--imitation_steps', '200', '--finetune_steps', '1500', '--anneal_in_finetune',
         '--layers', '0', '--batch_size', '16'] + extra + ['--results_dir', d]
    print(f'\n$ L0 {name} (batch 16)', flush=True)
    print('[done]' if subprocess.run(cmd).returncode == 0 else f'!!! {name} FAILED', flush=True)


def scale(name, extra):
    d = f'{OUT}/{name}'
    if os.path.exists(f'{d}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        HYB + ['--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune',
               '--batch_size', '8'] + extra + ['--results_dir', d]
    print(f'\n$ scale {name} (batch 8)', flush=True)
    print('[done]' if subprocess.run(cmd).returncode == 0 else f'!!! {name} FAILED', flush=True)


def l0_summary():
    def deg(n):
        p = f'{OUT}/screen_l0/{n}/heatmap.json'
        return json.load(open(p))[0]['hard_degradation'] if os.path.exists(p) else None
    print(f'\n{"="*62}\nL0: gate arity vs gate count (lower = better; all n_bits8)\n{"="*62}')
    for nm, cost, key in [('2-input base', '1x', 'base_copytrain'), ('LUT3', '1x', 'lut3'),
                          ('LUT4', '1x', 'lut4'), ('LUT5', '1x', 'lut5'), ('LUT6', '1x', 'lut6'),
                          ('2-input out_mult2', '2x', 'outmult2'),
                          ('2-input out_mult4', '4x', 'outmult4')]:
        d = deg(key)
        print(f'{nm:20} | {cost:3} | {("%.4f"%d) if d is not None else "-":>9}')


def main():
    heat('lut5', ['--lut_k', '5'])
    heat('lut6', ['--lut_k', '6'])
    l0_summary()
    # full LUT4 at batch 8 (memory-safe re-run of the headline). Full LUT6 is skipped: at
    # ~7GB/layer it cannot fit 12 layers even at batch 8 without gradient checkpointing.
    scale('hybrid_all_lut4_b8', ['--lut_k', '4'])
    print('\nLUT6 experiment finished.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'l0':
        l0_summary()
    else:
        main()
