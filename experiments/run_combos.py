"""Combine proven techniques to beat keepL0_outmult4 (44.16%). Two strategies:
  A) keep more real MLPs:  keep {L0,L11} / keep {L0,L9,L10,L11}, rest outmult4.
  B) best per-token LGN:   keepL0 + adaptive capacity + CAGE; keepL0 + LUT-sensitive + outmult2 + CAGE.
Waits for the GPU (recovery + training screen) to free, then runs. -> results/report/<name>/.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
HYB = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
       'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint',
       '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune',
       '--batch_size', '16']

CONFIGS = [
    # A: keep more real MLPs (rest = outmult4)
    ('keep_L0_L11_outmult4',   ['--out_gate_mult', '4', '--protected_layers', '0', '11']),
    ('keep_top4_outmult4',     ['--out_gate_mult', '4', '--protected_layers', '0', '9', '10', '11']),
    # B: best per-token LGN, keep only L0
    ('keepL0_adaptive_cage',   ['--protected_layers', '0', '--out_gate_mult', '2', '--cage',
                                '--out_gate_mult_layers', '9:8', '10:8', '11:8', '8:4', '7:4']),
    ('keepL0_lut_om_cage',     ['--protected_layers', '0', '--lut_k', '4', '--out_gate_mult', '2',
                                '--cage', '--lut_k_layers', '9:6', '10:6', '11:6']),
]


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 0


def main():
    # wait until the GPU is idle for two consecutive checks (recovery+train screen done)
    while True:
        if gpu_free() >= 5000:
            time.sleep(20)
            if gpu_free() >= 5000:
                break
        print(f'[combos] GPU busy (free {gpu_free()}) - waiting...', flush=True); time.sleep(120)
    for name, extra in CONFIGS:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/metrics.json'):
            print(f'[skip] {name}', flush=True); continue
        cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
            HYB + extra + ['--results_dir', d]
        print(f'\n[combos] {name}: {" ".join(extra)}', flush=True)
        r = subprocess.run(cmd)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED ({r.returncode})', flush=True)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])
    print('\nCombos finished.', flush=True)


if __name__ == '__main__':
    main()
