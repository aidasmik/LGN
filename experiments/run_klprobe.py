"""Cheap verify-before-commit: does KL distillation to the transformer's logits help?
om4 base (fits at batch 16, ~40min) + joint-polish with KL, vs om4's 46.09. If it lifts,
commit to om8 + KL on Colab; if not, skip. Waits for the GPU. -> results/report/lgn_om4_kl/.
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
CMD = [sys.executable, '-u', 'run.py', 'scale', '--strategy', 'greedy',
       '--heatmap', f'{OUT}/hybrid_all_heat/heatmap.json',
       '--hybrid_all', '--hybrid_ln2', 'copy_trainable', '--learn_pool',
       '--checkpoint', 'results/baseline.pt', '--n_bits', '8', '--grad_checkpoint',
       '--batch_size', '16', '--cage', '--lut_k', '4', '--out_gate_mult', '4',
       '--imitation_steps', '200', '--finetune_steps', '3000', '--anneal_in_finetune',
       '--joint_polish_steps', '800', '--joint_polish_kl_weight', '0.5',
       '--results_dir', f'{OUT}/lgn_om4_kl']


def gpu_free():
    try:
        return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                   '--format=csv,noheader,nounits'], capture_output=True, text=True,
                   timeout=30).stdout.strip().split('\n')[0])
    except Exception:
        return 0


def main():
    if os.path.exists(f'{OUT}/lgn_om4_kl/metrics.json'):
        print('[skip] done'); return
    while gpu_free() < 5000:
        print(f'[klprobe] GPU busy (free {gpu_free()}) - waiting...', flush=True); time.sleep(120)
    for attempt in (1, 2):
        r = subprocess.run(CMD)
        if r.returncode == 0 and os.path.exists(f'{OUT}/lgn_om4_kl/metrics.json'):
            print('[done] klprobe', flush=True); break
        print(f'[retry] rc={r.returncode}', flush=True); time.sleep(60)
    subprocess.run([sys.executable, 'experiments/report_ffn.py'])


if __name__ == '__main__':
    main()
