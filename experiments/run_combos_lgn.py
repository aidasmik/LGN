"""Beat the best PURE-LGN config (out_gate_mult4 = 41.77%, all 12 layers LGN) by stacking the
proven LGN levers -- NO kept FFN anywhere, every layer is logic:
  * LUT arity (more expressive gate)         [+]
  * out_gate_mult / adaptive capacity        [+]  (per-layer, heavier on sensitive L0/9/10/11)
  * CAGE hard-forward (closes soft-hard gap)  [+]  (gave LUT +2.8pp alone -- under-exploited)

Ordered most-promising first. Waits for the GPU (recovery/train screen) to free. Heavy LUT
combos run at batch 8 (grad_checkpoint), 2-input combos at batch 16 -- batch noted for
comparability vs the batch-32 outmult4 baseline. -> results/report/<name>/.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
# Training recipe from the L0 training-dynamics screen: more IMITATION is the biggest lever
# (imit800 -0.12 vs imit200; imit0 catastrophic), + sharper temp_end (0.05, smallest gap).
# These apply to every combo on top of the capacity/CAGE levers.
BASE = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
        'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint',
        '--imitation_steps', '800', '--temp_end', '0.05',
        '--finetune_steps', '3000', '--anneal_in_finetune']

CONFIGS = [
    # name, batch, extra   (all 12 layers LGN; no --protected_layers)
    # CONTROLLED: best gate config (outmult4) + ONLY the new training recipe (imit800+temp05),
    # no CAGE -> isolates how much more imitation lifts the 41.77% headline.
    ('lgn_outmult4_imit800', '16', ['--out_gate_mult', '4']),
    ('lgn_lut4_om4_cage', '8',  ['--lut_k', '4', '--out_gate_mult', '4', '--cage']),
    ('lgn_adaptive_max_cage', '8', ['--lut_k', '4', '--cage',
                                    '--lut_k_layers', '0:6', '11:6', '10:6', '9:6',
                                    '--out_gate_mult_layers', '0:2', '11:2', '10:2', '9:2']),
    ('lgn_adaptiveB_cage', '16', ['--cage',
                                  '--out_gate_mult_layers', '0:8', '11:8', '10:8', '9:8', '8:4', '7:4']),
    ('lgn_outmult4_cage', '16', ['--out_gate_mult', '4', '--cage']),
]


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 0


def main():
    while True:
        if gpu_free() >= 5000:
            time.sleep(20)
            if gpu_free() >= 5000:
                break
        print(f'[lgn-combos] GPU busy (free {gpu_free()}) - waiting...', flush=True); time.sleep(120)
    for name, batch, extra in CONFIGS:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/metrics.json'):
            print(f'[skip] {name}', flush=True); continue
        cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
            BASE + ['--batch_size', batch] + extra + ['--results_dir', d]
        print(f'\n[lgn-combos] {name} (batch {batch}): {" ".join(extra)}', flush=True)
        r = subprocess.run(cmd)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED ({r.returncode})', flush=True)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])
    print('\nPure-LGN combos finished.', flush=True)


if __name__ == '__main__':
    main()
