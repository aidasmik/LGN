"""Control baseline: keep the full trained attention in every layer but DELETE the MLP
sublayer entirely (x = x + attn(ln_1(x)), no FFN at all). This is the honest upper anchor
for 'how much does the FFN matter?': any LGN-FFN result must sit between this (FFN removed)
and the full transformer (FFN intact).

Writes results/report/no_mlp/metrics.json. Run: python experiments/baseline_no_mlp.py
"""
import json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
import torch
import torch.nn as nn
from lgn import ExperimentConfig, make_gpt
from pipeline import WikiText2, estimate_metrics


class _Zero(nn.Module):
    """Drop-in for Block.mlp: contributes nothing (FFN removed)."""
    def forward(self, x):
        return torch.zeros_like(x)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)
    cfg = ExperimentConfig()
    data = WikiText2(cfg.data, device)
    model, _ = make_gpt(cfg.model, cfg.data, device)
    ckpt = 'results/baseline.pt'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    base_m = estimate_metrics(model, data, cfg.train.eval_iters, cfg.train.batch_size)
    for blk in model.transformer.h:
        blk.mlp = _Zero()                       # attention-only: remove every FFN
    nomlp_m = estimate_metrics(model, data, cfg.train.eval_iters, cfg.train.batch_size)

    print(f'\n{"":16} | {"loss":>8} | {"ppl":>8} | {"acc %":>7}')
    print(f'{"transformer":16} | {base_m["loss"]:>8.4f} | {base_m["perplexity"]:>8.3f} | {base_m["accuracy"]*100:>7.2f}')
    print(f'{"attention-only":16} | {nomlp_m["loss"]:>8.4f} | {nomlp_m["perplexity"]:>8.3f} | {nomlp_m["accuracy"]*100:>7.2f}')
    print(f'  -> the FFN is worth {(base_m["accuracy"]-nomlp_m["accuracy"])*100:+.2f} pp of accuracy')

    os.makedirs('results/report/no_mlp', exist_ok=True)
    with open('results/report/no_mlp/metrics.json', 'w') as f:
        json.dump({'transformer': base_m, 'attention_only_no_mlp': nomlp_m}, f, indent=2)


if __name__ == '__main__':
    main()
