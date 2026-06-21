"""Master overnight batch for the LGN-as-FFN phase. Waits for the GPU to free (the running
lut6_ckpt to finish), then runs everything in priority order with skip-if-done, and prints the
master report at the end. Safe to re-run.

Memory policy (documented for comparability):
  * L0 combo screens: single layer, batch 8 + grad_checkpoint -> all mutually comparable at b8.
  * Full runs: batch 16 + grad_checkpoint (LUT6/wide readouts are memory-heavy). Minor confound
    vs the batch-32 baselines (LUT4 b8=36.58 vs b32=36.28 showed batch effect ~0.3pp).

Priority order (per the limited-compute plan): L0 combos -> adaptive_A -> lut4+outmult2 ->
adaptive_B -> selective LUT6 -> joint polish (LM, KL) -> LUT4+CAGE (close the soft-hard gap).
"""
import json, os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
CK = 'results/baseline.pt'
HYB = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint', CK,
       '--n_bits', '8', '--grad_checkpoint']
FT = ['--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune']


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 9999


def wait_gpu(min_free=5000):
    while gpu_free() < min_free:
        print(f'[overnight] GPU busy (free {gpu_free()} MiB) - waiting for current run...', flush=True)
        time.sleep(120)
    print(f'[overnight] GPU free ({gpu_free()} MiB) - starting batch.', flush=True)


def run(name, kind, extra):
    d = f'{OUT}/{name}'
    done = f'{d}/heatmap.json' if kind == 'heat' else f'{d}/metrics.json'
    if os.path.exists(done):
        print(f'[skip] {name}', flush=True); return
    if kind == 'heat':
        cmd = [sys.executable, '-u', 'run.py', 'heatmap'] + HYB + FT[:4] + \
            ['--finetune_steps', '1500', '--layers', '0', '--batch_size', '8'] + extra + \
            ['--results_dir', d]
    else:
        cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
            HYB + FT + ['--batch_size', '16'] + extra + ['--results_dir', d]
    print(f'\n{"="*60}\n[overnight] {name}: {" ".join(extra)}\n{"="*60}', flush=True)
    r = subprocess.run(cmd)
    print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED ({r.returncode})', flush=True)


# ---- L0 combo screens: are LUT arity and out_gate_mult ADDITIVE? (Task C) ----
L0 = [
    ('screen_l0_combo/base',     'heat', []),
    ('screen_l0_combo/outmult4', 'heat', ['--out_gate_mult', '4']),
    ('screen_l0_combo/lut4',     'heat', ['--lut_k', '4']),
    ('screen_l0_combo/lut6',     'heat', ['--lut_k', '6']),
    ('screen_l0_combo/lut4_om2', 'heat', ['--lut_k', '4', '--out_gate_mult', '2']),
    ('screen_l0_combo/lut4_om4', 'heat', ['--lut_k', '4', '--out_gate_mult', '4']),
    ('screen_l0_combo/lut6_om2', 'heat', ['--lut_k', '6', '--out_gate_mult', '2']),
    ('screen_l0_combo/lut6_om4', 'heat', ['--lut_k', '6', '--out_gate_mult', '4']),
    # NEW (no-new-code) ideas, screened cheaply on L0 first:
    ('screen_l0_combo/lut4_weighted', 'heat', ['--lut_k', '4', '--weighted_pool']),  # weighted x LUT (free)
    ('screen_l0_combo/lut4_k16',      'heat', ['--lut_k', '4', '--k', '16']),        # larger candidate pool
    ('screen_l0_combo/lut4_ste',      'heat', ['--lut_k', '4', '--ste']),            # soft-hard gap
]
# ---- full runs (Tasks B, C, D) ----
FULL = [
    ('adaptive_outmult_A', 'scale', ['--out_gate_mult_layers', '0:8', '11:4', '10:4', '9:4', '8:2', '7:2']),
    # NEW idea #7: keep L0 as the real transformer FFN (protected), LGN the rest at outmult4.
    ('hybrid_all_keepL0_outmult4', 'scale', ['--out_gate_mult', '4', '--protected_layers', '0']),
    ('hybrid_all_lut4_outmult2', 'scale', ['--lut_k', '4', '--out_gate_mult', '2']),
    # NEW idea #3: weighted readout x LUT gates (free; readout failed on 2-input, retry on LUT).
    ('hybrid_all_lut4_weighted', 'scale', ['--lut_k', '4', '--weighted_pool']),
    ('adaptive_outmult_B', 'scale', ['--out_gate_mult_layers', '0:8', '11:8', '10:8', '9:8', '8:4', '7:4']),
    # NEW idea #2: STE hard-forward to close LUT's large soft-hard gap (complements CAGE).
    ('hybrid_all_lut4_ste', 'scale', ['--lut_k', '4', '--ste']),
    ('hybrid_all_lut6_sel', 'scale', ['--lut_k', '4', '--lut_k_layers', '0:6', '11:6', '10:6', '9:6']),
    ('hybrid_all_outmult4_polish_lm', 'scale', ['--out_gate_mult', '4', '--joint_polish_steps', '500']),
    ('hybrid_all_outmult4_polish_kl', 'scale', ['--out_gate_mult', '4', '--joint_polish_steps', '500',
                                                '--joint_polish_kl_weight', '0.5']),
    ('hybrid_all_lut4_cage', 'scale', ['--lut_k', '4', '--cage']),
    # NEW idea #7b: keep L0 as real MLP AND give the rest LUT4 (combine the two best levers).
    ('hybrid_all_keepL0_lut4', 'scale', ['--lut_k', '4', '--protected_layers', '0']),
]


def main():
    wait_gpu()
    for name, kind, extra in L0:
        run(name, kind, extra)
    for name, kind, extra in FULL:
        run(name, kind, extra)
    print('\n[overnight] all runs attempted. Final report:', flush=True)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])
    # combo-additivity summary
    print('\n=== L0 combo: are LUT and out_gate_mult additive? ===', flush=True)
    for n in ['base', 'outmult4', 'lut4', 'lut6', 'lut4_om2', 'lut4_om4', 'lut6_om2', 'lut6_om4']:
        p = f'{OUT}/screen_l0_combo/{n}/heatmap.json'
        if os.path.exists(p):
            print(f'  {n:12} {json.load(open(p))[0]["hard_degradation"]:+.4f}', flush=True)
    print('\nOvernight batch finished.', flush=True)


if __name__ == '__main__':
    main()
