"""Two follow-ups after Colab ran out:

  A) PURE LGN with token_shift INSTEAD of attention, using the now-OPTIMIZED FFN gates.
     This drops the frozen float attention entirely -> a fully-Boolean / FPGA-efficient network
     again, but with LUT4 + out_gate_mult + CAGE + k16 (the gates that took the FFN side from
     27% to ~48%). Reference: OLD token_shift K=2 with plain 2-input gates = 36.22%.

  B) Cheaper local test of the record idea: om8_k16 + best-hard at batch 8 (batch-16 OOMs the
     8GB box). Batch-8 has a known ~2-3pp penalty vs batch 16, so this is a floor, not the
     clean number Colab would give.

Waits for the GPU (seed-7 finishing) to free. -> results/report/<name>/.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'

CONFIGS = [
    # A: pure LGN (NO --hybrid_all), token_shift for cross-token, optimized gates
    ('tshift2_opt', ['--token_shift', '2', '--lut_k', '4', '--out_gate_mult', '2', '--cage',
                     '--k', '16', '--learn_pool', '--grad_checkpoint', '--batch_size', '8',
                     '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune']),
    # B: cheaper record test (om8_k16 + best-hard, batch 8) -- this one IS hybrid (keeps attention)
    ('lgn_record', ['--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool',
                    '--n_bits', '8', '--grad_checkpoint', '--batch_size', '8', '--cage',
                    '--lut_k', '4', '--out_gate_mult', '4', '--k', '16',
                    '--out_gate_mult_layers', '0:8', '11:8', '10:8', '9:8',
                    '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune',
                    '--ft_keep_best_hard']),
]
COMMON = ['--checkpoint', 'results/baseline.pt']


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
        print(f'[tshift] GPU busy (free {gpu_free()}) - waiting...', flush=True); time.sleep(120)
    for name, extra in CONFIGS:
        d = f'{OUT}/{name}'
        if os.path.exists(f'{d}/metrics.json'):
            print(f'[skip] {name}', flush=True); continue
        cmd = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT] + \
            COMMON + extra + ['--results_dir', d]
        print(f'\n[tshift] {name}: {" ".join(extra)}', flush=True)
        for attempt in (1, 2):
            r = subprocess.run(cmd)
            if r.returncode == 0 and os.path.exists(f'{d}/metrics.json'):
                print(f'[done] {name}', flush=True); break
            print(f'[retry] {name} rc={r.returncode}', flush=True); time.sleep(60)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])
    print('\ntshift + record finished.', flush=True)


if __name__ == '__main__':
    main()
