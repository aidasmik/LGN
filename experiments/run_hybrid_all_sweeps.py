"""Broad sweeps for the all-attention hybrid experiment. Run AFTER run_hybrid_all.py
(needs its sensitivity heatmap). Each config -> results/report/<name>/ (skip-if-done).

  B. Encoding   : n_bits in {1 (threshold), 4, 8(main)}
  C. Capacity   : depth in {2, 3}  (width_mult is a no-op under no_in_proj=aggressive,
                   so capacity here = logic depth / encoding bits, not a Linear width)
  D. Sensitivity: keep the top-K most-sensitive FFNs as full transformer blocks
                   (--protected_layers from the heatmap), rest hybrid LGN. K in {2,4}.
  E. Calibration: main + --learn_binary_calibration ; ln_2 ablations copy_frozen / fresh
"""
import json, os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CK = 'results/baseline.pt'
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
COMMON = ['--hybrid_all', '--learn_pool', '--checkpoint', CK]
FT = ['--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune']


def run(name, extra):
    out = f'{OUT}/{name}'
    if os.path.exists(f'{out}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        COMMON + FT + extra + ['--results_dir', out]
    print(f'\n$ {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd)
    print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)


def most_sensitive(k):
    """Top-k FFN layer indices by hard_degradation (largest = most sensitive)."""
    rows = json.load(open(HEAT))
    rows = sorted(rows, key=lambda r: r['hard_degradation'], reverse=True)
    return [r['layer_idx'] for r in rows[:k]]


def main():
    while not os.path.exists(HEAT):
        print('[sweeps] waiting for sensitivity heatmap...', flush=True); time.sleep(60)

    # B. Encoding sweep (default ln2=copy_trainable for all)
    LN2 = ['--hybrid_ln2', 'copy_trainable']
    run('hybrid_all_nbits1', LN2 + ['--no-binary_io'])          # plain threshold (n_bits ignored)
    run('hybrid_all_nbits4', LN2 + ['--n_bits', '4'])
    # n_bits8 == hybrid_all_main (already run)

    # C. Capacity sweep
    run('hybrid_all_depth2', LN2 + ['--depth', '2'])
    run('hybrid_all_depth3', LN2 + ['--depth', '3'])

    # D. Sensitivity: keep top-K most-sensitive FFNs as full transformer blocks
    for k in (2, 4):
        prot = most_sensitive(k)
        run(f'hybrid_all_keep{k}_transformer', LN2 + ['--protected_layers'] + [str(i) for i in prot])

    # E. Calibration + ln_2 ablations
    run('hybrid_all_calib',       LN2 + ['--learn_binary_calibration'])
    run('hybrid_all_ln2_frozen',  ['--hybrid_ln2', 'copy_frozen'])
    run('hybrid_all_ln2_fresh',   ['--hybrid_ln2', 'fresh'])

    print('\nAll-attention hybrid SWEEPS finished.', flush=True)


if __name__ == '__main__':
    main()
