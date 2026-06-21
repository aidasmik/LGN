"""Learned multi-level aggregation (weighted_pool) experiment. The sum_pool readout collapses
g bits to g+1 levels; a per-channel learned weight per bit gives up to 2^g levels at ZERO extra
gate cost (block-diagonal, bit_width params). Tests whether that cheap readout matches the
gate-expensive out_gate_mult at the same output resolution -- and includes the identity-LGN
control: if identity gates + weighted_pool is ~as good, the readout (not the logic) is the FFN.

Waits for the GPU to free (out_mult4 finishing or crashing) before starting, so it doesn't OOM.
Reuses the existing sensitivity heatmap for greedy order. -> results/report/<name>/.
"""
import json, os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
COMMON = ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--checkpoint', 'results/baseline.pt',
          '--n_bits', '8', '--imitation_steps', '200', '--finetune_steps', '3000',
          '--anneal_in_finetune']


def gpu_free_mib():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().split('\n')[0])
    except Exception:
        return 9999  # if we can't query, don't block


def wait_for_gpu(min_free=5000):
    while gpu_free_mib() < min_free:
        print(f'[weighted] GPU busy (free {gpu_free_mib()} MiB) - waiting for out_mult4 to free it...',
              flush=True)
        time.sleep(120)
    print(f'[weighted] GPU free ({gpu_free_mib()} MiB) - starting.', flush=True)


def run(name, extra):
    if os.path.exists(f'{OUT}/{name}/metrics.json'):
        print(f'[skip] {name}', flush=True); return
    cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
        COMMON + extra + ['--results_dir', f'{OUT}/{name}']
    print(f'\n$ {name}: {" ".join(extra)}', flush=True)
    r = subprocess.run(cmd)
    print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED', flush=True)


def main():
    wait_for_gpu()
    # weighted readout: 2^group_size output levels at NO extra gate cost (group_size = n_bits = 8)
    run('hybrid_all_weighted', ['--learn_pool', '--weighted_pool'])
    # honesty control: dead (identity) gates + the same weighted readout
    run('hybrid_all_weighted_identity', ['--learn_pool', '--weighted_pool', '--identity_logic',
                                         '--imitation_steps', '0'])
    print('\nWeighted-pool experiment finished.', flush=True)


if __name__ == '__main__':
    main()
