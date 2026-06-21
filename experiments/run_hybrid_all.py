"""All-attention hybrid: keep FROZEN pretrained attention in EVERY layer, replace only
the FFN/MLP with LGN. Answers: how close to the transformer can we get, and which FFNs
are most sensitive?

Phases (controlled core first):
  0. heatmap  -> per-FFN sensitivity (each layer independently: attention kept, FFN->LGN)
  A. main     -> greedy cumulative scaling, all 12 FFNs eventually LGN  (the headline number)
  ctrl. identity-LGN control -> same pipeline but gates are pass-through. If this is
        ~as good as A, the ln_2/pool plumbing (not the logic) is doing the work => fake LGN.

Primary ln_2 mode = copy_trainable (faithful trained pre-MLP signal + cheap recalibration).
Each phase -> results/report/<name>/  (skip-if-done). aggressive LGN defaults, n_bits=8, depth=1.
"""
import os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CK = 'results/baseline.pt'
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'

COMMON = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint', CK]


def run(cmd):
    print('\n$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f'!!! FAILED ({r.returncode}): {" ".join(cmd)}', flush=True)
    return r.returncode == 0


def main():
    os.makedirs(OUT, exist_ok=True)

    # Phase 0: per-FFN sensitivity heatmap (each layer measured independently).
    if not os.path.exists(HEAT):
        run([sys.executable, '-u', 'run.py', 'heatmap'] + COMMON +
            ['--imitation_steps', '200', '--finetune_steps', '1500', '--anneal_in_finetune',
             '--results_dir', f'{OUT}/hybrid_all_heat'])
    else:
        print('[skip] heatmap done', flush=True)

    # Phase A: main cumulative scaling (greedy, easy-first by the heatmap).
    if not os.path.exists(f'{OUT}/hybrid_all_main/metrics.json'):
        run([sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] +
            COMMON + ['--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune',
                      '--results_dir', f'{OUT}/hybrid_all_main'])
    else:
        print('[skip] main done', flush=True)

    # Control: identity-LGN (gates pass-through). Detects whether the LGN actually does work.
    if not os.path.exists(f'{OUT}/hybrid_all_identity/metrics.json'):
        run([sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] +
            COMMON + ['--identity_logic', '--imitation_steps', '0', '--finetune_steps', '1500',
                      '--anneal_in_finetune', '--results_dir', f'{OUT}/hybrid_all_identity'])
    else:
        print('[skip] identity control done', flush=True)

    print('\nAll-attention hybrid CORE finished.', flush=True)


if __name__ == '__main__':
    main()
