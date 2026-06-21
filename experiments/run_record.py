"""Record attempt: the 48.18% config (om8_k16) + the validated best-hard training recipe.
Must run at batch 8 (batch 16 + the hard-model eval that best-hard needs would OOM on this
8GB GPU). So this is really a test: can best-hard offset the batch-8 downgrade? If it lands
>= 48.18 it's a new record; if not, 48.18 (batch 16) stands as the GPU-practical ceiling.
Waits for the A-C battery (last config lgn_best_s42) to finish. -> results/report/lgn_record/.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
HEAT = f'{OUT}/hybrid_all_heat/heatmap.json'
CMD = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy', '--heatmap', HEAT,
       '--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool', '--checkpoint',
       'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint', '--batch_size', '8',
       '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune', '--cage',
       '--lut_k', '4', '--out_gate_mult', '4', '--k', '16',
       '--out_gate_mult_layers', '0:8', '11:8', '10:8', '9:8',
       '--ft_keep_best_hard', '--results_dir', f'{OUT}/lgn_record']


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().split('\n')[0])
    except Exception:
        return 0


def main():
    if os.path.exists(f'{OUT}/lgn_record/metrics.json'):
        print('[skip] record already done'); return
    # wait for A-C battery to finish (its last config) or the GPU to go idle 10 min
    idle = 0
    while not os.path.exists(f'{OUT}/lgn_best_s42/metrics.json'):
        idle = idle + 1 if gpu_free() >= 5000 else 0
        if idle >= 5:
            print('[record] GPU idle 10 min - proceeding.', flush=True); break
        print(f'[record] waiting for A-C battery (free {gpu_free()} MiB)...', flush=True)
        time.sleep(120)
    for attempt in (1, 2):
        r = subprocess.run(CMD)
        if r.returncode == 0 and os.path.exists(f'{OUT}/lgn_record/metrics.json'):
            print('[done] record', flush=True); break
        print(f'[retry] rc={r.returncode}', flush=True); time.sleep(60)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])


if __name__ == '__main__':
    main()
