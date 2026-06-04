"""Smoke / correctness tests for the RDDLGN-inspired recurrent LGN layer (CPU-only)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lgn import (ExperimentConfig, ModelConfig, DataConfig, make_gpt,
                 RecurrentLogicGateGPTLayer, HardRecurrentLogicGateGPTLayer,
                 GatedRecurrentLogicGateGPTLayer, HardGatedRecurrentLogicGateGPTLayer)


def _gpt_cfg():
    from model import GPTConfig
    return GPTConfig(block_size=8, vocab_size=32, n_layer=2, n_head=4, n_embd=16, dropout=0.0)


def _make_layer(state_width=64, depth=1, state_init='zero', seed=0, cls=RecurrentLogicGateGPTLayer):
    torch.manual_seed(seed)
    g = _gpt_cfg()
    return cls(
        g, layer_idx=0, binary_io=True, n_bits=4, no_in_proj=True, sum_pool=True,
        learn_pool=True, state_width=state_width, recurrent_depth=depth, state_init=state_init)


def _make_gated(state_width=64, depth=1, state_init='zero', seed=0):
    return _make_layer(state_width, depth, state_init, seed, cls=GatedRecurrentLogicGateGPTLayer)


def test_recurrent_is_causal():
    layer = _make_layer().eval()
    B, T, C = 2, 8, 16
    x = torch.randn(B, T, C)
    t = 4
    y1 = layer(x)
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn(B, T - (t + 1), C)   # perturb the FUTURE only
    y2 = layer(x2)
    assert torch.allclose(y1[:, :t + 1], y2[:, :t + 1], atol=1e-6), "output depends on future tokens!"


def test_soft_hard_equivalence():
    for init in ('zero', 'learned', 'residual'):
        layer = _make_layer(state_init=init, seed=1).eval()
        if init == 'learned':
            with torch.no_grad():
                layer.initial_state.normal_()   # non-trivial learned state
        x = torch.randn(2, 8, 16)
        y_soft_hard = layer(x, hard=True)
        hard = HardRecurrentLogicGateGPTLayer(layer).eval()
        y_hard = hard(x)
        assert torch.allclose(y_soft_hard, y_hard, atol=1e-6), f"soft(hard=True) != hard for init={init}"


def test_gradient_flow():
    layer = _make_layer().train()
    x = torch.randn(2, 8, 16, requires_grad=False)
    out = layer(x)
    loss = out.pow(2).mean()
    loss.backward()
    g = layer.logic[0]
    for name in ('conn_logits_a', 'conn_logits_b', 'gate_logits'):
        p = getattr(g, name)
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient on {name}"


def test_depth2_and_state_width():
    layer = _make_layer(state_width=32, depth=2, seed=2).eval()
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape == x.shape
    hard = HardRecurrentLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Gated (flip-flop / latch-inspired) recurrent LGN
# ---------------------------------------------------------------------------

def test_gated_is_causal():
    layer = _make_gated().eval()
    B, T, C = 2, 8, 16
    x = torch.randn(B, T, C)
    t = 4
    y1 = layer(x)
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn(B, T - (t + 1), C)
    y2 = layer(x2)
    assert torch.allclose(y1[:, :t + 1], y2[:, :t + 1], atol=1e-6), "gated output depends on future!"


def test_gated_soft_hard_equivalence():
    for init in ('zero', 'learned', 'residual'):
        for depth in (1, 2):
            layer = _make_gated(state_width=32, depth=depth, state_init=init, seed=3).eval()
            if init == 'learned':
                with torch.no_grad():
                    layer.initial_state.normal_()
            x = torch.randn(2, 8, 16)
            y_soft_hard = layer(x, hard=True)
            hard = HardGatedRecurrentLogicGateGPTLayer(layer).eval()
            y_hard = hard(x)
            assert torch.allclose(y_soft_hard, y_hard, atol=1e-6), \
                f"gated soft(hard=True) != hard for init={init}, depth={depth}"


def test_gated_gradient_flow():
    """Both the candidate (self.logic) AND keep (self.keep_logic) stacks must get gradients."""
    layer = _make_gated(depth=2).train()
    x = torch.randn(2, 8, 16)
    layer(x).pow(2).mean().backward()
    for stack_name in ('logic', 'keep_logic'):
        stack = getattr(layer, stack_name)
        for li, g in enumerate(stack):
            for name in ('conn_logits_a', 'conn_logits_b', 'gate_logits'):
                p = getattr(g, name)
                assert p.grad is not None and p.grad.abs().sum() > 0, \
                    f"no gradient on {stack_name}[{li}].{name}"


def test_gated_pipeline_construction():
    """recurrent + recurrent_gated -> GatedRecurrentLogicGateGPTLayer; hard conversion -> Hard variant."""
    from pipeline import _build_logic_layer, make_hard_model

    cfg = ExperimentConfig()
    cfg.model = ModelConfig(n_layer=2, n_head=4, n_embd=16, dropout=0.0)
    cfg.data = DataConfig(block_size=8, vocab_size=32)
    cfg.logic.learn_pool = True
    cfg.logic.recurrent = True
    cfg.logic.recurrent_gated = True
    cfg.logic.recurrent_state_width = 32

    base, gpt_cfg = make_gpt(cfg.model, cfg.data, 'cpu')
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert isinstance(layer, GatedRecurrentLogicGateGPTLayer)
    base.replace_layer(0, layer)
    hard = make_hard_model(base, [0], 'cpu')
    assert isinstance(hard.transformer.h[0], HardGatedRecurrentLogicGateGPTLayer)
    # vanilla path (gated off) must still give the plain recurrent layer
    cfg.logic.recurrent_gated = False
    base2, gpt_cfg2 = make_gpt(cfg.model, cfg.data, 'cpu')
    layer2 = _build_logic_layer(base2, 0, gpt_cfg2, cfg.logic)
    assert isinstance(layer2, RecurrentLogicGateGPTLayer)
    assert not isinstance(layer2, GatedRecurrentLogicGateGPTLayer)


def test_pipeline_heatmap_and_scale_recurrent():
    from pipeline import WikiText2, run_heatmap, run_scaling

    class TinyData:
        def __init__(self):
            self.block_size = 8
            self.device = 'cpu'
            g = torch.Generator().manual_seed(0)
            self.train = torch.randint(0, 32, (4000,), generator=g)
            self.val = torch.randint(0, 32, (800,), generator=g)
        def get_batch(self, split='train', batch_size=8, generator=None):
            src = self.train if split == 'train' else self.val
            ix = torch.randint(len(src) - self.block_size - 1, (batch_size,), generator=generator)
            x = torch.stack([src[i:i + self.block_size] for i in ix])
            y = torch.stack([src[i + 1:i + 1 + self.block_size] for i in ix])
            return x, y
        def fixed_val_batches(self, eval_iters=2, batch_size=8):
            gg = torch.Generator().manual_seed(7)
            return [self.get_batch('val', batch_size, gg) for _ in range(eval_iters)]

    cfg = ExperimentConfig()
    cfg.model = ModelConfig(n_layer=2, n_head=4, n_embd=16, dropout=0.0)
    cfg.data = DataConfig(block_size=8, vocab_size=32)
    cfg.logic.learn_pool = True
    cfg.logic.recurrent = True
    cfg.logic.recurrent_state_width = 32
    cfg.train.imitation_steps = 2
    cfg.train.finetune_steps = 2
    cfg.train.eval_iters = 2
    cfg.train.batch_size = 4
    cfg.train.anneal_in_finetune = True

    base, gpt_cfg = make_gpt(cfg.model, cfg.data, 'cpu')
    data = TinyData()
    run_heatmap(base, gpt_cfg, data, cfg, save_path=None, layers=[0])
    run_scaling(base, gpt_cfg, data, cfg, strategy='uniform', heatmap_results=None, save_path=None)


if __name__ == '__main__':
    test_recurrent_is_causal(); print('causal OK')
    test_soft_hard_equivalence(); print('soft-hard OK')
    test_gradient_flow(); print('grad OK')
    test_depth2_and_state_width(); print('depth2/state_width OK')
    test_gated_is_causal(); print('gated causal OK')
    test_gated_soft_hard_equivalence(); print('gated soft-hard OK')
    test_gated_gradient_flow(); print('gated grad OK')
    test_gated_pipeline_construction(); print('gated pipeline OK')
    test_pipeline_heatmap_and_scale_recurrent(); print('pipeline OK')
    print('ALL RECURRENT TESTS PASSED')
