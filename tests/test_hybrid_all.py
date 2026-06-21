"""Tests for the all-attention hybrid experiment: every layer keeps frozen pretrained
attention and replaces only the FFN/MLP with LGN. Covers --hybrid_all expansion,
ln_2 handling (fresh / copy_trainable / copy_frozen), binary calibration, frozen-grad
guarantees, and soft<->hard equivalence. CPU-only."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lgn import (ExperimentConfig, ModelConfig, DataConfig, make_gpt,
                 LogicGateGPTLayer, HardLogicGateGPTLayer,
                 HybridLogicGateGPTLayer, HardHybridLogicGateGPTLayer,
                 LearnedLogicLayer, HardLogicLayer, LearnedLUTLayer, HardLUTLayer)
from pipeline import _build_logic_layer, _enable_lgn_grads, make_hard_model


def _cfg():
    cfg = ExperimentConfig()
    cfg.model = ModelConfig(n_layer=2, n_head=4, n_embd=16, dropout=0.0)
    cfg.data = DataConfig(block_size=8, vocab_size=32)
    cfg.logic.learn_pool = True
    return cfg


def _base():
    cfg = _cfg()
    base, gpt_cfg = make_gpt(cfg.model, cfg.data, 'cpu')
    return base, gpt_cfg, cfg


# --------------------------------------------------------------------------- #
# hybrid_all construction + frozen attention
# --------------------------------------------------------------------------- #

def test_hybrid_all_builds_hybrid_everywhere():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0, 1]          # what --hybrid_all expands to for n_layer=2
    for idx in (0, 1):
        layer = _build_logic_layer(base, idx, gpt_cfg, cfg.logic)
        assert isinstance(layer, HybridLogicGateGPTLayer), f"L{idx} not hybrid"


def test_frozen_attention_never_requires_grad():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0, 1]
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    _enable_lgn_grads(layer)                   # pipeline would call this
    for name, p in layer.named_parameters():
        if name.startswith('attn.') or name.startswith('ln_1.'):
            assert not p.requires_grad, f"frozen attention param {name} requires grad!"
    # logic params MUST be trainable
    assert any(p.requires_grad and 'logic' in n for n, p in layer.named_parameters())


# --------------------------------------------------------------------------- #
# ln_2 modes
# --------------------------------------------------------------------------- #

def test_ln2_fresh_is_default_and_not_copied():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    assert cfg.logic.hybrid_ln2 == 'fresh'     # legacy default preserved
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    orig = base.transformer.h[0].ln_2.weight
    # fresh LayerNorm starts at weight=1 -> differs from a trained ln_2 (unless trivially equal)
    assert layer.ln2_mode == 'fresh'


def test_ln2_copy_trainable_matches_trained_and_trains():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.hybrid_ln2 = 'copy_trainable'
    # make the trained ln_2 non-trivial so a copy is detectable
    with torch.no_grad():
        base.transformer.h[0].ln_2.weight.normal_()
        base.transformer.h[0].ln_2.bias.normal_()
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert torch.allclose(layer.ln_2.weight, base.transformer.h[0].ln_2.weight)
    assert torch.allclose(layer.ln_2.bias, base.transformer.h[0].ln_2.bias)
    _enable_lgn_grads(layer)
    assert all(p.requires_grad for p in layer.ln_2.parameters()), "copy_trainable ln_2 must train"


def test_ln2_copy_frozen_matches_trained_and_is_frozen():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.hybrid_ln2 = 'copy_frozen'
    with torch.no_grad():
        base.transformer.h[0].ln_2.weight.normal_()
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert torch.allclose(layer.ln_2.weight, base.transformer.h[0].ln_2.weight)
    _enable_lgn_grads(layer)
    assert not any(p.requires_grad for p in layer.ln_2.parameters()), "copy_frozen ln_2 must NOT train"


# --------------------------------------------------------------------------- #
# binary calibration
# --------------------------------------------------------------------------- #

def test_calibration_builds_params_and_default_off():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    plain = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert not plain.learn_binary_calibration
    assert not hasattr(plain, 'cal_scale')
    cfg.logic.learn_binary_calibration = True
    cal = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert cal.learn_binary_calibration and hasattr(cal, 'cal_scale') and hasattr(cal, 'cal_shift')


def test_calibration_gradients_flow():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.learn_binary_calibration = True
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    _enable_lgn_grads(layer)
    x = torch.randn(2, 8, 16)
    layer(x).pow(2).mean().backward()
    assert layer.cal_scale.grad is not None and layer.cal_scale.grad.abs().sum() > 0
    assert layer.cal_shift.grad is not None


# --------------------------------------------------------------------------- #
# soft <-> hard equivalence (the honesty guarantee) across the new options
# --------------------------------------------------------------------------- #

def test_hybrid_soft_hard_equiv_all_ln2_modes_with_calibration():
    for ln2 in ('fresh', 'copy_trainable', 'copy_frozen'):
        for cal in (False, True):
            base, gpt_cfg, cfg = _base()
            cfg.logic.hybrid_layers = [0]
            cfg.logic.hybrid_ln2 = ln2
            cfg.logic.learn_binary_calibration = cal
            with torch.no_grad():
                base.transformer.h[0].ln_2.weight.normal_()
            layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
            if cal:
                with torch.no_grad():
                    layer.cal_scale.normal_(mean=1.0, std=0.3); layer.cal_shift.normal_(std=0.3)
            hard = HardHybridLogicGateGPTLayer(layer).eval()
            x = torch.randn(2, 8, 16)
            assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6), \
                f"hybrid soft(hard)!=hard for ln2={ln2}, cal={cal}"


def test_plain_logic_soft_hard_equiv_with_calibration():
    base, gpt_cfg, cfg = _base()
    cfg.logic.learn_binary_calibration = True
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()  # not hybrid -> LogicGateGPTLayer
    assert isinstance(layer, LogicGateGPTLayer)
    with torch.no_grad():
        layer.cal_scale.normal_(mean=1.0, std=0.3); layer.cal_shift.normal_(std=0.3)
    hard = HardLogicGateGPTLayer(layer).eval()
    x = torch.randn(2, 8, 16)
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)


# --------------------------------------------------------------------------- #
# signed real->binary encoding
# --------------------------------------------------------------------------- #

def test_signed_encoding_bit_budget():
    from lgn import _signed_thermometer_ste
    h = torch.randn(3, 7)
    enc = _signed_thermometer_ste(h, n_bits=4, training=False)
    assert enc.shape == (3, 7 * (2 * 4 + 1)), "signed encoding must emit 2*n_bits+1 bits/scalar"
    assert set(enc.unique().tolist()) <= {0.0, 1.0}, "encoding must be binary"


def test_signed_encoding_group_size_and_equiv():
    for cls_hybrid in (True, False):
        base, gpt_cfg, cfg = _base()
        cfg.logic.signed_encoding = True
        cfg.logic.n_bits = 4
        if cls_hybrid:
            cfg.logic.hybrid_layers = [0]
            cfg.logic.hybrid_ln2 = 'copy_trainable'
        layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
        assert layer.group_size == 2 * 4 + 1                # aggressive: bits/scalar per channel
        x = torch.randn(2, 8, 16)
        hard = (HardHybridLogicGateGPTLayer(layer) if cls_hybrid
                else HardLogicGateGPTLayer(layer)).eval()
        assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6), \
            f"signed soft(hard)!=hard (hybrid={cls_hybrid})"


def test_signed_encoding_gradients():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.signed_encoding = True
    cfg.logic.n_bits = 4
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    _enable_lgn_grads(layer)
    x = torch.randn(2, 8, 16)
    layer(x).pow(2).mean().backward()
    g = layer.logic[0]
    assert g.gate_logits.grad is not None and g.gate_logits.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# Conv1D adapters + LloydMax encoder + TopK interconnect
# --------------------------------------------------------------------------- #

def test_conv1d_adapters_shape_equiv_and_gradients():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.pre_conv1d = True
    cfg.logic.pre_conv1d_channels = 32
    cfg.logic.pre_conv1d_kernel = 3
    cfg.logic.pre_conv1d_stride = 2
    cfg.logic.post_conv1d = True
    cfg.logic.post_conv1d_channels = 16
    cfg.logic.post_conv1d_kernel = 3
    cfg.logic.post_conv1d_stride = 2
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape == x.shape
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)
    layer.train(); _enable_lgn_grads(layer)
    layer(x).pow(2).mean().backward()
    assert layer.pre_conv1d.conv.weight.grad is not None
    assert layer.post_conv1d.conv.weight.grad is not None


def test_lloydmax_encoder_updates_stats_and_hard_equiv():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.binary_encoder = 'lloydmax'
    cfg.logic.n_bits = 4
    cfg.logic.lloyd_ema = 0.5
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert hasattr(layer, 'lloyd_base_thresholds')
    old_mean = layer.lloyd_mean.clone()
    x = torch.randn(2, 8, 16)
    layer.train()
    layer(x)
    assert not torch.allclose(old_mean, layer.lloyd_mean), "LloydMax stats should update in training"
    layer.eval()
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)


def test_topk_block_sparse_logic_and_lut_equiv_grads():
    x = (torch.rand(5, 16) > 0.5).float()
    g = LearnedLogicLayer(16, 16, interconnect='topk_block_sparse', topk_sparse_k=4).eval()
    hard = HardLogicLayer(g).eval()
    assert torch.allclose(g(x, hard=True), hard(x), atol=1e-6)
    g.train()
    g(x).pow(2).mean().backward()
    assert g.topk_a.c_sparse.grad is not None and g.topk_a.c_sparse.grad.abs().sum() > 0
    assert g.gate_logits.grad is not None and g.gate_logits.grad.abs().sum() > 0

    lut = LearnedLUTLayer(16, 16, lut_k=4, interconnect='topk_block_sparse',
                          topk_sparse_k=4).eval()
    hard_lut = HardLUTLayer(lut).eval()
    assert torch.allclose(lut(x, hard=True), hard_lut(x), atol=1e-6)
    lut.train()
    lut(x).pow(2).mean().backward()
    assert lut.topk.c_sparse.grad is not None and lut.topk.c_sparse.grad.abs().sum() > 0
    assert lut.lut_logits.grad is not None and lut.lut_logits.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# #2 nonlinear readout curve, #4 residual gate scaling, #5 within-layer ensemble
# --------------------------------------------------------------------------- #

def _readout_cfg():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.binary_io = True
    cfg.logic.sum_pool = True
    cfg.logic.no_in_proj = True
    cfg.logic.n_bits = 4
    cfg.logic.out_gate_mult = 2
    return base, gpt_cfg, cfg


def test_pool_curve_inits_linear_and_hard_equiv():
    base, gpt_cfg, cfg = _readout_cfg()
    cfg.logic.pool_curve = True
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert hasattr(layer, 'pool_curve')
    # init must be linear (== fixed sum_pool centering) so default behaviour is preserved
    g = layer.group_size
    expect = (torch.arange(g + 1).float() - g / 2) / (g / 2)
    assert torch.allclose(layer.pool_curve[0], expect, atol=1e-6)
    layer.eval()
    x = torch.randn(2, 8, 16)
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-5)
    layer.train(); _enable_lgn_grads(layer)
    layer(x).pow(2).mean().backward()
    assert layer.pool_curve.grad is not None and layer.pool_curve.grad.abs().sum() > 0


def test_residual_scale_alpha_and_hard_equiv():
    base, gpt_cfg, cfg = _readout_cfg()
    cfg.logic.residual_scale = True
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert hasattr(layer, 'pool_alpha')
    assert torch.allclose(layer.pool_alpha, torch.ones_like(layer.pool_alpha))
    layer.eval()
    x = torch.randn(2, 8, 16)
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-5)
    layer.train(); _enable_lgn_grads(layer)
    layer(x).pow(2).mean().backward()
    assert layer.pool_alpha.grad is not None and layer.pool_alpha.grad.abs().sum() > 0


def test_ensemble_banks_average_and_hard_equiv():
    base, gpt_cfg, cfg = _readout_cfg()
    cfg.logic.ensemble = 4
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    assert layer.ensemble_banks is not None and len(layer.ensemble_banks) == 4
    # banks must be distinct (different seeds -> different candidate wiring)
    a = layer.ensemble_banks[0]
    b = layer.ensemble_banks[1]
    assert not torch.allclose(a.gate_logits, b.gate_logits)
    layer.eval()
    x = torch.randn(2, 8, 16)
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert hard.ensemble_banks is not None and len(hard.ensemble_banks) == 4
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-5)
    layer.train(); _enable_lgn_grads(layer)
    layer(x).pow(2).mean().backward()
    grads = [bk.gate_logits.grad for bk in layer.ensemble_banks]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


def test_all_three_readout_ideas_combined_hard_equiv():
    base, gpt_cfg, cfg = _readout_cfg()
    cfg.logic.pool_curve = True
    cfg.logic.residual_scale = True
    cfg.logic.ensemble = 3
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
    x = torch.randn(2, 8, 16)
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-5)


# --------------------------------------------------------------------------- #
# output-side resolution (out_gate_mult)
# --------------------------------------------------------------------------- #

def test_out_gate_mult_widens_readout_and_equiv():
    for hybrid in (True, False):
        base, gpt_cfg, cfg = _base()
        cfg.logic.n_bits = 8
        cfg.logic.out_gate_mult = 4
        if hybrid:
            cfg.logic.hybrid_layers = [0]
        layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
        assert layer.group_size == 8 * 4, "out_gate_mult must multiply the readout group size"
        assert layer.logic[-1].out_dim == 16 * 8 * 4, "final logic layer must widen by out_gate_mult"
        x = torch.randn(2, 8, 16)
        hard = (HardHybridLogicGateGPTLayer(layer) if hybrid
                else HardLogicGateGPTLayer(layer)).eval()
        assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)


def test_out_gate_mult_incompatible_with_identity():
    base, gpt_cfg, cfg = _base()
    cfg.logic.identity_logic = True
    cfg.logic.out_gate_mult = 2
    try:
        _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
        assert False, "expected ValueError for identity_logic + out_gate_mult>1"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# weighted pool (learned multi-level aggregation readout)
# --------------------------------------------------------------------------- #

def test_weighted_pool_init_is_noop():
    """At init (uniform 2/g weights, -1 bias) weighted_pool must equal fixed sum_pool centering."""
    base, gpt_cfg, cfg = _base()
    cfg.logic.learn_pool = False
    fixed = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
    cfg.logic.weighted_pool = True
    weighted = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
    # copy the logic gates so only the pooling differs
    weighted.logic.load_state_dict(fixed.logic.state_dict())
    weighted.norm.load_state_dict(fixed.norm.state_dict())
    x = torch.randn(2, 8, 16)
    assert torch.allclose(fixed(x), weighted(x), atol=1e-6), "weighted_pool init must be a no-op"


def test_weighted_pool_equiv_and_grads():
    for hybrid in (True, False):
        base, gpt_cfg, cfg = _base()
        cfg.logic.weighted_pool = True
        if hybrid:
            cfg.logic.hybrid_layers = [0]
        layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
        assert layer.pool_w.shape == (16, layer.group_size)
        with torch.no_grad():                       # make weights non-trivial
            layer.pool_w.normal_(); layer.pool_b.normal_()
        x = torch.randn(2, 8, 16)
        hard = (HardHybridLogicGateGPTLayer(layer) if hybrid
                else HardLogicGateGPTLayer(layer)).eval()
        assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)
        layer.train(); _enable_lgn_grads(layer)
        layer(x).pow(2).mean().backward()
        assert layer.pool_w.grad is not None and layer.pool_w.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# K-input LUT gate
# --------------------------------------------------------------------------- #

def test_multilinear_reproduces_boolean_function():
    from lgn import _multilinear, _corner_table
    corners = _corner_table(2)                      # (4,2): (0,0),(1,0),(0,1),(1,1)
    # AND = 1 only at corner (1,1) = index 3
    T = torch.tensor([[[0., 0., 0., 1.]]])          # (1,1,4)
    for (x0, x1), want in [((1, 1), 1), ((1, 0), 0), ((0, 1), 0), ((0, 0), 0)]:
        a = torch.tensor([[[float(x0), float(x1)]]])   # (N=1,M=1,K=2)
        out = float(_multilinear(a, corners, T))
        assert abs(out - want) < 1e-6, f"AND({x0},{x1}) = {out}, want {want}"


def test_lut_gate_builds_and_equiv():
    for hybrid in (True, False):
        base, gpt_cfg, cfg = _base()
        cfg.logic.lut_k = 4
        if hybrid:
            cfg.logic.hybrid_layers = [0]
        layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
        from lgn import LearnedLUTLayer
        g = layer.logic[0]
        assert isinstance(g, LearnedLUTLayer) and g.lut_k == 4
        assert g.lut_logits.shape[-1] == 2 ** 4 and g.cand.shape[1] == 4
        x = torch.randn(2, 8, 16)
        hard = (HardHybridLogicGateGPTLayer(layer) if hybrid
                else HardLogicGateGPTLayer(layer)).eval()
        assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6), \
            f"LUT soft(hard)!=hard (hybrid={hybrid})"


def test_lut_gate_gradients():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.lut_k = 4
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    _enable_lgn_grads(layer)
    x = torch.randn(2, 8, 16)
    layer(x).pow(2).mean().backward()
    g = layer.logic[0]
    assert g.conn_logits.grad.abs().sum() > 0, "no gradient on LUT connections"
    assert g.lut_logits.grad.abs().sum() > 0, "no gradient on LUT truth table"


def test_make_hard_model_hybrid_all():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0, 1]
    cfg.logic.hybrid_ln2 = 'copy_trainable'
    cfg.logic.learn_binary_calibration = True
    for idx in (0, 1):
        base.replace_layer(idx, _build_logic_layer(base, idx, gpt_cfg, cfg.logic))
    hard = make_hard_model(base, [0, 1], 'cpu')
    for idx in (0, 1):
        assert isinstance(hard.transformer.h[idx], HardHybridLogicGateGPTLayer)


# --------------------------------------------------------------------------- #
# per-layer capacity (out_gate_mult_layers / lut_k_layers) + LUT x outmult combo
# --------------------------------------------------------------------------- #

def test_per_layer_capacity_overrides():
    from lgn import LearnedLUTLayer, LearnedLogicLayer
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0, 1]
    cfg.logic.out_gate_mult = 1
    cfg.logic.out_gate_mult_layers = {0: 4}
    cfg.logic.lut_k_layers = {0: 6}
    L0 = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    L1 = _build_logic_layer(base, 1, gpt_cfg, cfg.logic)
    assert L0.out_gate_mult == 4 and L0.lut_k == 6 and isinstance(L0.logic[-1], LearnedLUTLayer)
    assert L1.out_gate_mult == 1 and L1.lut_k == 0 and isinstance(L1.logic[-1], LearnedLogicLayer)


def test_lut_plus_outmult_combo_equiv_and_grads():
    """LUT-K and out_gate_mult together: shapes, soft<->hard equivalence, gradients on both."""
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.lut_k = 4
    cfg.logic.out_gate_mult = 2
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
    g = layer.logic[-1]
    from lgn import LearnedLUTLayer
    assert isinstance(g, LearnedLUTLayer)
    assert layer.group_size == 8 * 2                       # n_bits8 * out_gate_mult2
    assert g.out_dim == 16 * 8 * 2                         # eff_C * n_bits * out_gate_mult
    x = torch.randn(2, 8, 16)
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6), "LUT+outmult soft(hard)!=hard"
    layer.train(); _enable_lgn_grads(layer)
    layer(x).pow(2).mean().backward()
    assert g.conn_logits.grad.abs().sum() > 0 and g.lut_logits.grad.abs().sum() > 0


def test_per_layer_hard_conversion_mixed_model():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0, 1]
    cfg.logic.lut_k_layers = {0: 4}                        # L0 LUT, L1 2-input
    cfg.logic.out_gate_mult_layers = {1: 2}               # L1 wider readout
    for i in (0, 1):
        base.replace_layer(i, _build_logic_layer(base, i, gpt_cfg, cfg.logic))
    hard = make_hard_model(base, [0, 1], 'cpu')
    x = torch.randint(0, 32, (2, 8))
    logits, loss = hard(x, x)                              # targets -> full-sequence logits
    assert logits.shape == (2, 8, 32) and torch.isfinite(loss)  # forward works end-to-end


def test_mlp_guided_init_biases_candidates():
    """Functional init: candidate connections should concentrate on the MLP's important inputs."""
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.lut_k = 4
    with torch.no_grad():                                   # only channel 3 matters to the MLP
        base.transformer.h[0].mlp.c_fc.weight.zero_()
        base.transformer.h[0].mlp.c_fc.weight[:, 3] = 5.0
    cfg.logic.mlp_guided_init = True
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    cand = layer.logic[0].cand                              # n_bits=8 -> channel 3 = bits 24..31
    frac = ((cand >= 24) & (cand < 32)).float().mean().item()
    assert frac > 0.8, f"mlp_guided_init should concentrate candidates on channel 3 (got {frac:.2f})"
    # and a random-init layer should NOT
    cfg.logic.mlp_guided_init = False
    rnd = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).logic[0].cand
    assert ((rnd >= 24) & (rnd < 32)).float().mean().item() < 0.3


def test_freeze_logic_control():
    """Honesty control: logic params frozen at random init, plumbing trainable. Must work
    at shapes where identity_logic can't (out_gate_mult>1, lut_k)."""
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.lut_k = 4
    cfg.logic.out_gate_mult = 4
    cfg.logic.freeze_logic = True
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic)
    _enable_lgn_grads(layer)
    assert not any(p.requires_grad for n, p in layer.named_parameters() if n.startswith('logic.')), \
        "freeze_logic: logic params must not train"
    assert any(p.requires_grad for n, p in layer.named_parameters() if n.startswith('ln_2.')), \
        "freeze_logic: plumbing must still train"
    # forward + hard conversion still work
    x = torch.randn(2, 8, 16)
    hard = HardHybridLogicGateGPTLayer(layer.eval()).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)


# --------------------------------------------------------------------------- #
# A1 residual depth + A2 gated LUT
# --------------------------------------------------------------------------- #

def test_residual_depth_wiring_and_equiv():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.lut_k = 4
    cfg.logic.depth = 2
    cfg.logic.logic_residual = True
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
    bw = 16 * 8                                            # eff_C * n_bits
    assert layer.logic[1].in_dim == 2 * bw, "residual layer must read [bits, prev_out]"
    x = torch.randn(2, 8, 16)
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)
    layer.train(); _enable_lgn_grads(layer)
    layer(x).pow(2).mean().backward()
    assert all(l.lut_logits.grad.abs().sum() > 0 for l in layer.logic), "grads must reach both layers"


def test_gated_lut_pairs_equiv_and_cost():
    base, gpt_cfg, cfg = _base()
    cfg.logic.hybrid_layers = [0]
    cfg.logic.lut_k = 4
    cfg.logic.gated_lut = True
    layer = _build_logic_layer(base, 0, gpt_cfg, cfg.logic).eval()
    g = layer.logic[0]
    assert g.lut_logits.shape[0] == 2 * g.out_dim, "gated = two LUT banks (honest 2x cost)"
    x = torch.randn(2, 8, 16)
    hard = HardHybridLogicGateGPTLayer(layer).eval()
    assert torch.allclose(layer(x, hard=True), hard(x), atol=1e-6)
    # hard outputs stay binary (AND of bits)
    out_bits = hard.logic[0](( torch.rand(4, g.in_dim) > 0.5).float())
    assert set(out_bits.unique().tolist()) <= {0.0, 1.0}


def test_malformed_layer_spec_errors():
    import run
    for bad in (['0'], ['x:4'], ['0:0'], ['-1:4'], ['0:4:2']):
        try:
            run._parse_layer_spec(bad, 'test')
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
    assert run._parse_layer_spec(['0:8', '11:4'], 'test') == {0: 8, 11: 4}


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); print(f'{name} OK')
    print('ALL HYBRID-ALL TESTS PASSED')
