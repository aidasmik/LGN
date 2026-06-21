"""A-C experiment queue on top of the 46.09% winner (LUT4 x om4 x CAGE, batch16, pure LGN):

  A (structure, screened on L0 first -- full runs only if the screen looks promising):
    residual depth-2 DAG wiring   (depth hurt before; residual should fix it)
    plain depth-2 control          (isolates the residual effect)
    gated LUT pairs                (2x LUT cost -> 2K-input functions)
  B (training, always run):
    winner + finetune 6000 + per-layer anneal + best-hard checkpoint selection
  C (variance, always run):
    winner at seeds 7 and 42       (candidate-lottery spread; best-of-N harvest)

Waits for the in-flight next2 queue (om8_k16 + lut6) to fully finish before taking the GPU.
"""
import json, os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
WINNER = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
          'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint', '--cage',
          '--lut_k', '4', '--out_gate_mult', '4']
FT = ['--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune']


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 0


def wait_for_next2():
    """Don't steal the GPU between next2's two configs: wait for both sentinels, or for the
    GPU to be idle for 10 consecutive minutes (covers a crashed/finished next2)."""
    idle = 0
    while True:
        done = (os.path.exists(f'{OUT}/lgn_best_om8_k16/metrics.json') and
                os.path.exists(f'{OUT}/lgn_lut6_om4_cage/metrics.json'))
        if done:
            break
        idle = idle + 1 if gpu_free() >= 5000 else 0
        if idle >= 5:
            print('[abc] GPU idle 10 min - assuming next2 finished/crashed.', flush=True)
            break
        print(f'[abc] waiting for next2 (free {gpu_free()} MiB, idle {idle}/5)...', flush=True)
        time.sleep(120)


def heat(name, extra):
    d = f'{OUT}/abc_l0/{name}'
    if os.path.exists(f'{d}/heatmap.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'heatmap'] + WINNER + \
        ['--imitation_steps', '200', '--finetune_steps', '1500', '--anneal_in_finetune',
         '--layers', '0', '--batch_size', '8'] + extra + ['--results_dir', d]
    print(f'\n[abc-L0] {name}: {" ".join(extra)}', flush=True)
    subprocess.run(cmd)


def l0_deg(name):
    p = f'{OUT}/abc_l0/{name}/heatmap.json'
    return json.load(open(p))[0]['hard_degradation'] if os.path.exists(p) else None


def scale(name, extra, batch='16'):
    d = f'{OUT}/{name}'
    if os.path.exists(f'{d}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        WINNER + FT + ['--batch_size', batch] + extra + ['--results_dir', d]
    print(f'\n[abc] {name} (batch {batch}): {" ".join(extra)}', flush=True)
    for attempt in (1, 2):
        r = subprocess.run(cmd)
        if r.returncode == 0 and os.path.exists(f'{d}/metrics.json'):
            print(f'[done] {name}', flush=True); return
        print(f'[retry] {name} rc={r.returncode}', flush=True); time.sleep(60)


def main():
    wait_for_next2()
    # --- A: L0 screens (winner-config L0 reference from combo screen: lut4_om4 = 0.0519) ---
    heat('residual',     ['--depth', '2', '--logic_residual'])
    heat('depth2_plain', ['--depth', '2'])
    heat('gated',        ['--gated_lut'])
    for n in ('residual', 'depth2_plain', 'gated'):
        print(f'[abc-L0] {n}: {l0_deg(n)}', flush=True)

    # --- B: training run (always) ---
    scale('lgn_best_train', ['--finetune_steps', '6000', '--per_layer_anneal',
                             '--ft_keep_best_hard'])

    # --- A full runs, gated on the screens (reference 0.0519; loose threshold 0.06) ---
    if (l0_deg('residual') or 9) < 0.06:
        scale('lgn_best_residual', ['--depth', '2', '--logic_residual'])
    else:
        print('[abc] residual screen weak -> skipping full run', flush=True)
    if (l0_deg('gated') or 9) < 0.06:
        scale('lgn_best_gated', ['--gated_lut'], batch='8')   # 2x LUT memory
    else:
        print('[abc] gated screen weak -> skipping full run', flush=True)

    # --- C: seed spread (always) ---
    scale('lgn_best_s7',  ['--seed', '7'])
    scale('lgn_best_s42', ['--seed', '42'])

    subprocess.run([sys.executable, 'experiments/report_ffn.py'])
    print('\nA-C queue finished.', flush=True)


if __name__ == '__main__':
    main()
