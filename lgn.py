"""
Core definitions: config, logic gate layers, nanoGPT wrappers.
"""

import copy
import os
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LogicConfig:
    # Width of the trained Linear path (only used when no_in_proj=False; no-op in aggressive).
    width_mult: int = 2
    depth: int = 1
    k: int = 4
    activation: str = 'sigmoid'
    conn_init_scale: float = 0.02
    gate_init_scale: float = 0.02
    hybrid_layers: list = field(default_factory=list)  # layers that keep frozen attention
    # How the pre-MLP norm (ln_2) is handled in hybrid layers:
    #   'fresh'          -> new LayerNorm trained from scratch (legacy default, preserves old results)
    #   'copy_trainable' -> copy trained ln_2, allow fine-tuning (faithful input + cheap recalibration)
    #   'copy_frozen'    -> copy trained ln_2 and freeze (strict "only the MLP function changed")
    hybrid_ln2: str = 'fresh'
    identity_logic: bool = False                        # ablation: LGN body = pass-through
    # Per-channel binary calibration: replace the squashing activation with a learned
    # sigmoid(scale*x + bias) per feature channel before thermometer encoding. Init to
    # identity-vs-sigmoid (scale=1, bias=0). Tiny (2 params/channel); a dense map is a
    # SEPARATE ablation, not this. Helps map ~Gaussian post-norm activations into bits.
    learn_binary_calibration: bool = False
    # Signed real->binary encoding: sign bit + positive-magnitude thermometer + negative-
    # magnitude thermometer (2*n_bits+1 bits/scalar). Keeps sign+magnitude of zero-centered
    # post-norm activations instead of squashing to [0,1]. Mutually exclusive with calibration.
    signed_encoding: bool = False

    # AGGRESSIVE SETUP (the honest default — no trained float transform around the gates)
    binary_io: bool = True
    n_bits: int = 8
    # Per-layer adaptive precision: layers in `precision_layers` use `high_n_bits` instead of
    # `n_bits`. Input-quantization degradation scales ~1/n_bits, but only the sensitive layers
    # (L0/L9/L10/L11) are quantization-limited — so spend bits there, stay cheap in the middle.
    precision_layers: list = field(default_factory=list)
    high_n_bits: int = 16
    # Output-side resolution: widen the final logic layer by out_gate_mult so each output
    # channel sums out_gate_mult x more gates (a finer sum_pool readout). Decouples output
    # resolution from input n_bits (in aggressive mode group_size == n_bits, conflating them).
    out_gate_mult: int = 1
    # Per-layer override of out_gate_mult, e.g. {0: 8, 11: 4}. Unlisted layers use the global.
    out_gate_mult_layers: dict = field(default_factory=dict)
    # Gate primitive arity: 0/2 = the 2-input gate (16-function LUT2); >=3 = a K-input LUT gate
    # (learned 2^K truth table via multilinear extension; hard-snaps to one FPGA LUT-K). Tests
    # whether a more expressive primitive beats more 2-input gates at equal gate count.
    lut_k: int = 0
    # Per-layer override of lut_k, e.g. {0: 6, 11: 6}. Unlisted layers use the global.
    lut_k_layers: dict = field(default_factory=dict)
    # Functional init (hybrid layers): seed the first logic layer's candidate connections from the
    # trained MLP's input importance (||W1[:,c]||) instead of a blind random lottery.
    mlp_guided_init: bool = False
    # Honesty control: freeze the logic parameters (connections + gate/LUT tables) at their
    # RANDOM init; only the plumbing (ln_2, pool affine, calibration) trains. Shape-compatible
    # with any out_gate_mult/lut_k, unlike identity_logic. The trained-vs-frozen gap = the
    # accuracy genuinely earned by LEARNED logic.
    freeze_logic: bool = False
    # A1: residual/DAG wiring for depth>=2 -- deeper gate layers draw candidates from BOTH the
    # previous layer's outputs AND the original input bits (wires are free in hardware). Fixes
    # the error-compounding that made naive depth hurt; gives FFN-like 2-layer composition.
    logic_residual: bool = False
    # A2: gated LUT pairs -- each output = LUT_a(x) AND LUT_b(x) (soft: product). A 2K-input
    # function class at 2x LUT cost; mimics the FFN's multiplicative gating. Requires lut_k>=2.
    gated_lut: bool = False
    # Gradient checkpointing on the logic stack: recompute the (memory-heavy, esp. LUT-K)
    # logic forward during backward instead of storing it. ~2x logic compute, big memory cut,
    # so large gates run at full batch_size instead of a noisy reduced batch.
    grad_checkpoint: bool = False
    # Learned multi-level aggregation: replace the equal-weight bit COUNT with a per-channel
    # learned weight per bit-in-group: out[c] = sum_i w[c,i]*bit[c,i] + b[c]. Same gates, but
    # up to 2^group_size output levels instead of group_size+1. Block-diagonal (own group only,
    # no channel mixing) -> bit_width params, NOT a dense out_proj. Init == learn_pool (no-op).
    weighted_pool: bool = False
    sum_pool: bool = True
    no_in_proj: bool = True
    # Learnable per-channel affine on the sum_pool output (cheap residual-stat matching).
    learn_pool: bool = False
    # Fixed causal token shift: each position sees [x[t-K]..x[t]] — a local cross-token
    # receptive field for the pointwise LGN. K=0 disables. (The single mechanism, with
    # hybrid/selective, that actually raises accuracy.)
    token_shift: int = 0
    # Optional causal Conv1D adapters around the LGN body. Off by default because trainable
    # float convs can become "fake LGN" plumbing unless checked with freeze_logic controls.
    pre_conv1d: bool = False
    pre_conv1d_channels: int = 0       # 0 = keep current token_shift width
    pre_conv1d_kernel: int = 3
    pre_conv1d_stride: int = 1
    pre_conv1d_groups: int = 1
    post_conv1d: bool = False
    post_conv1d_channels: int = 0      # 0 = n_embd; != n_embd gets a 1x1 return projection
    post_conv1d_kernel: int = 3
    post_conv1d_stride: int = 1
    post_conv1d_groups: int = 1
    # Real->bits encoder. 'activation' is the historical sigmoid/thermometer path.
    # 'lloydmax' thresholds raw pre-LGN activations with Gaussian Lloyd-Max thresholds
    # whose mean/std are tracked from the actual activation distribution.
    binary_encoder: str = 'activation'
    lloyd_ema: float = 0.99
    lloyd_min_std: float = 1e-3
    # Optional sparse trainable interconnect for gate input selection. 'random' is the
    # historical candidate-softmax lottery; 'topk_block_sparse' ports the block-topk idea.
    interconnect: str = 'random'
    topk_sparse_k: int = 8
    topk_sparse_scale: float = 1.0
    # #2 learned per-channel nonlinear readout: map the group bit-count through a per-channel
    # learned curve (interp knots), init linear so default == sum_pool. Requires sum_pool.
    pool_curve: bool = False
    # #4 residual gate scaling: per-channel learned alpha on the LGN contribution (x+alpha*LGN).
    residual_scale: bool = False
    # #5 within-layer ensembling: N independent depth-1 gate banks, averaged (variance
    # reduction over the candidate lottery). ensemble=1 == current. Requires depth==1.
    ensemble: int = 1
    # RDDLGN-inspired recurrent/stateful LGN (alternative cross-token mechanism to token_shift).
    # state_t = Logic([token_bits_t, state_{t-1}]). Causal. NOT full RDDLGN encoder/decoder.
    recurrent: bool = False
    recurrent_layers: list = field(default_factory=list)  # empty = all replaced layers; else only these idx
    recurrent_state_width: int = None                     # None = token bit width; must divide n_embd
    recurrent_depth: int = 1
    recurrent_state_init: str = "zero"                    # 'zero' | 'learned' | 'residual'
    # Opt-in flip-flop/latch-inspired gated update (requires recurrent=True):
    #   candidate = LogicCandidate([token_bits, state]); keep = LogicKeep([token_bits, state])
    #   state = keep*state + (1-keep)*candidate  (keep is itself a learned LOGIC stack).
    recurrent_gated: bool = False

@dataclass
class TrainConfig:
    baseline_steps: int = 5_000
    baseline_lr: float = 1e-3
    batch_size: int = 32
    eval_iters: int = 30
    log_every: int = 500
    imitation_steps: int = 1_000
    imitation_lr: float = 2e-3
    temp_start: float = 2.0
    temp_end: float = 0.1
    ent_conn: float = 0.001
    ent_gate: float = 0.02
    finetune_steps: int = 1_000
    finetune_lr: float = 2e-3
    ft_ent_conn: float = 0.0005
    ft_ent_gate: float = 0.01
    per_layer_anneal: bool = False  # scale imitation steps by layer difficulty
    ft_log_sharpness: bool = True   # print per-layer sharpness
    ft_eval_hard: bool = False      # evaluate hard-snapped model
    # B4: keep the BEST-hard checkpoint seen during fine-tune (eval every 500 steps) instead
    # of the final step. The deployed model is the hard one, so select by hard validation.
    ft_keep_best_hard: bool = False
    imit_loss: str = 'mse'
    ste: bool = False               # straight-through estimator (forward hard, backward soft)
    # CAGE — Align Forward Adapt Backward (2026, arxiv 2603.14157).
    # Implies STE (hard forward) and adapts the BACKWARD-pass softmax temperature τ_b
    # based on an EMA of average commitment confidence. Closes the discretization gap
    # by construction (forward = inference). Schedule: τ_b in [tau_min, tau_max] linearly
    # interpolated by EMA confidence c_ema (1/K-1.0 -> tau_max-tau_min).
    cage: bool = False
    cage_tau_max: float = 3.0
    cage_tau_min: float = 0.5
    cage_ema:     float = 0.99
    # Direct (from-scratch) training: anneal temperature DURING fine-tune on LM loss
    # instead of during imitation. Lets the LGN learn its own solution, not imitate MLP.
    anneal_in_finetune: bool = False
    # Curriculum: decaying MSE-to-MLP term blended into fine-tune (weight*1.0 -> 0).
    # 0 = pure LM loss. Single-layer fine-tune only (heatmap).
    ft_imit_weight: float = 0.0
    # NOTE: 'freeze_unreplaced' was removed — the base model is ALWAYS frozen by
    # _make_logic_model / _add_logic_layer (only LGN layer params get requires_grad=True),
    # so the flag was redundant. All degradation numbers are already 'pure LGN' measurements.
    # Joint polish: after sequential scaling, fine-tune ALL LGN layers together to
    # coordinate them (fixes greedy myopia). 0 = disabled.
    joint_polish_steps: int = 0
    # System-level distillation during joint polish: KL of student logits to the
    # original transformer's logits. Global coordination signal (vs per-layer MLP). 0 = LM only.
    joint_polish_kl_weight: float = 0.0

@dataclass
class DataConfig:
    train_chars: int = 5_000_000
    val_chars: int = 500_000
    block_size: int = 64
    vocab_size: int = 256

@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    results_dir: str = "results"
    seed: int = 1337

# ---------------------------------------------------------------------------
# nanoGPT wrappers
# ---------------------------------------------------------------------------

NANOGPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'nanogpt_layer_lab', 'nanoGPT')
)
if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)

from model import GPT, GPTConfig  # noqa: E402


def _patch_replace_layer():
    if hasattr(GPT, 'replace_layer'):
        return
    def replace_layer(self, idx, layer):
        if not 0 <= idx < len(self.transformer.h):
            raise IndexError(f"idx {idx} out of range")
        self.transformer.h[idx] = layer
    GPT.replace_layer = replace_layer

_patch_replace_layer()


def apply_token_shift(normed, taps):
    """Channel-aligned causal token shift.

    normed: (B, T, C). taps: list of positive ints, e.g. [1,2] for token_shift K=2.
    Returns (B, T, C*(len(taps)+1)) where contiguous blocks of (len(taps)+1) channels
    are ONE channel's time history [t, t-tap0, ...]. First tap positions zeroed (causal).
    """
    if not taps:
        return normed
    B, T, C = normed.shape
    parts = [normed]
    for tap in taps:
        shifted = torch.roll(normed, shifts=tap, dims=1)
        shifted[:, :tap] = 0
        parts.append(shifted)
    return torch.stack(parts, dim=-1).reshape(B, T, C * (len(taps) + 1))


class CausalConv1D(nn.Module):
    """Causal Conv1D over token time for (B,T,C) tensors.

    Stride > 1 is restored to the original length by repeat-interleaving each causal
    sample forward. Position t therefore only depends on positions <= t.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, name='conv1d'):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"{name}: kernel_size must be >= 1 (got {kernel_size}).")
        if stride < 1:
            raise ValueError(f"{name}: stride must be >= 1 (got {stride}).")
        if groups < 1:
            raise ValueError(f"{name}: groups must be >= 1 (got {groups}).")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                f"{name}: in/out channels ({in_channels}->{out_channels}) must be divisible "
                f"by groups={groups}.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, groups=groups)

    def forward(self, x):
        B, T, C = x.shape
        if C != self.in_channels:
            raise ValueError(f"CausalConv1D expected {self.in_channels} channels, got {C}.")
        y = x.transpose(1, 2)                              # (B,C,T)
        y = F.pad(y, (self.kernel_size - 1, 0))            # causal left padding
        y = self.conv(y).transpose(1, 2)                   # (B,T',Cout)
        if self.stride > 1:
            y = y.repeat_interleave(self.stride, dim=1)
        if y.size(1) < T:
            pad = y[:, -1:].expand(B, T - y.size(1), self.out_channels)
            y = torch.cat([y, pad], dim=1)
        return y[:, :T]


def _lloyd_max_standard_thresholds(n_bits, iters=80, tol=1e-10):
    """Gaussian Lloyd-Max thresholds for n_bits threshold outputs.

    Uses n_bins=n_bits+1, so the returned threshold vector has length n_bits and
    preserves the existing "n_bits outputs per scalar" budget.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1 for LloydMax encoding (got {n_bits}).")
    n_bins = n_bits + 1
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    q = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
    edges = normal.icdf(q)
    edges[0], edges[-1] = -float('inf'), float('inf')
    levels = torch.zeros(n_bins, dtype=torch.float64)
    root2 = 2.0 ** 0.5
    inv_sqrt_2pi = 1.0 / ((2.0 * torch.pi) ** 0.5)

    def pdf(z):
        return torch.where(torch.isinf(z), torch.zeros_like(z),
                           inv_sqrt_2pi * torch.exp(-0.5 * z * z))

    def cdf(z):
        return 0.5 * (1.0 + torch.erf(z / root2))

    for _ in range(iters):
        old = edges.clone()
        for i in range(n_bins):
            a, b = edges[i], edges[i + 1]
            mass = cdf(b) - cdf(a)
            levels[i] = (pdf(a) - pdf(b)) / mass.clamp_min(1e-30)
        edges[1:-1] = 0.5 * (levels[:-1] + levels[1:])
        if torch.max(torch.abs(edges[1:-1] - old[1:-1])) < tol:
            break
    return edges[1:-1].to(torch.float32)


class TopKBlockSparseInterconnect(nn.Module):
    """Block-sparse top-k interconnect from x:(N,in) to y:(N,out).

    The input is randomly reordered, split into blocks of `topk`, and each output in a
    block learns a soft/hard selection among that block's inputs. This is a structured
    alternative to the per-output random candidate lottery.
    """
    def __init__(self, layer_inputs, layer_outputs, topk, name='', init='dirac',
                 sparsity_scale=1.0, seed=None):
        super().__init__()
        if topk < 1:
            raise ValueError(f"{name}: topk must be >= 1 (got {topk}).")
        if layer_inputs % topk != 0:
            raise ValueError(f"{name}: layer_inputs={layer_inputs} must be divisible by topk={topk}.")
        self.layer_inputs = layer_inputs
        self.layer_outputs = layer_outputs
        self.topk = topk
        self.n_blocks = layer_inputs // topk
        if layer_outputs % self.n_blocks != 0:
            raise ValueError(
                f"{name}: layer_outputs={layer_outputs} must be divisible by n_blocks={self.n_blocks}.")
        self.outputs_per_block = layer_outputs // self.n_blocks
        self.sparsity_scale = float(sparsity_scale)
        self.binarized = False
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        self.register_buffer('reorder', torch.randperm(layer_inputs, generator=g))
        self.c_sparse = nn.Parameter(torch.zeros(
            self.n_blocks, self.topk, self.outputs_per_block, dtype=torch.float32))
        init = init.lower()
        if init == 'uniform':
            nn.init.uniform_(self.c_sparse, a=0.0, b=1.0)
        elif init in ('dirac', 'unique'):
            with torch.no_grad():
                nn.init.normal_(self.c_sparse, mean=0.0, std=0.01)
                idx = torch.randint(self.topk, (self.n_blocks, self.outputs_per_block), generator=g)
                # Keep the Dirac bias trainable: the source script used 10.0, but with the
                # default sparsity scale that saturates softmax enough to kill gradients.
                self.c_sparse[torch.arange(self.n_blocks).unsqueeze(1),
                              idx, torch.arange(self.outputs_per_block).unsqueeze(0)] = 1.0
        else:
            nn.init.normal_(self.c_sparse, mean=0.0, std=1.0)

    def _weights(self, hard=False, ste=False):
        soft = F.softmax(self.c_sparse * self.sparsity_scale, dim=1)
        if ste:
            hard_w = F.one_hot(self.c_sparse.argmax(dim=1), self.topk).to(dtype=self.c_sparse.dtype)
            hard_w = hard_w.permute(0, 2, 1).contiguous()
            return soft + (hard_w - soft).detach()
        if hard or self.binarized:
            hard_w = F.one_hot(self.c_sparse.argmax(dim=1), self.topk).to(dtype=self.c_sparse.dtype)
            return hard_w.permute(0, 2, 1).contiguous()
        return soft

    def forward(self, x, hard=False, ste=False):
        x = x[:, self.reorder]
        x = x.view(-1, self.n_blocks, self.topk)
        if hard or self.binarized:
            idx = self.c_sparse.argmax(dim=1).unsqueeze(0).expand(x.shape[0], -1, -1)
            return torch.gather(x, dim=2, index=idx).reshape(x.shape[0], self.layer_outputs)
        conn = self._weights(hard=hard, ste=ste).to(device=x.device, dtype=x.dtype)
        return torch.einsum("bnk,nko->bno", x, conn).reshape(x.shape[0], self.layer_outputs)

    @torch.no_grad()
    def commitment(self):
        return float(F.softmax(self.c_sparse * self.sparsity_scale, dim=1).max(dim=1).values.mean())

    def entropy_loss(self, weight=0.001):
        p = F.softmax(self.c_sparse * self.sparsity_scale, dim=1)
        return weight * (-(p * (p + 1e-8).log()).sum(dim=1).mean())

    def binarize(self):
        self.binarized = True


class HardTopKBlockSparseInterconnect(nn.Module):
    def __init__(self, soft: TopKBlockSparseInterconnect):
        super().__init__()
        self.layer_outputs = soft.layer_outputs
        self.n_blocks = soft.n_blocks
        self.topk = soft.topk
        self.register_buffer('reorder', soft.reorder.detach().clone())
        self.register_buffer('indices', soft.c_sparse.detach().argmax(dim=1).clone())

    def forward(self, x):
        x = x[:, self.reorder].view(-1, self.n_blocks, self.topk)
        idx = self.indices.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.gather(x, dim=2, index=idx).reshape(x.shape[0], self.layer_outputs)


def make_gpt(model_cfg, data_cfg, device='cuda'):
    cfg = GPTConfig(
        block_size=data_cfg.block_size,
        vocab_size=data_cfg.vocab_size,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        dropout=model_cfg.dropout,
    )
    return GPT(cfg).to(device), cfg

# ---------------------------------------------------------------------------
# Logic gate matrix  (16 gates expressed over [1, A, B, A*B])
# ---------------------------------------------------------------------------

LOGIC_GATE_MATRIX = torch.tensor([
    [0,  0,  0,  0],  # False
    [0,  0,  0,  1],  # AND
    [0,  1,  0, -1],  # A AND NOT B
    [0,  1,  0,  0],  # A
    [0,  0,  1, -1],  # NOT A AND B
    [0,  0,  1,  0],  # B
    [0,  1,  1, -2],  # XOR
    [0,  1,  1, -1],  # OR
    [1, -1, -1,  1],  # NOR
    [1, -1, -1,  2],  # XNOR
    [1,  0, -1,  0],  # NOT B
    [1,  0, -1,  1],  # A OR NOT B
    [1, -1,  0,  0],  # NOT A
    [1, -1,  0,  1],  # NOT A OR B
    [1,  0,  0, -1],  # NAND
    [1,  0,  0,  0],  # True
], dtype=torch.float32)


def diff_logic_gates(a, b):
    basis = torch.stack([torch.ones_like(a), a, b, a * b], dim=-1)
    return basis @ LOGIC_GATE_MATRIX.to(device=a.device, dtype=a.dtype).T  # (..., 16)


def annealed_temperature(step, total, start=2.0, end=0.1):
    return start * (end / start) ** (step / max(total - 1, 1))


def _apply_activation(h, name):
    """Pointwise activation dispatch shared by all logic gate blocks."""
    if name == 'sigmoid':     return torch.sigmoid(h)
    if name == 'tanh':        return torch.tanh(h)
    if name == 'relu':        return F.relu(h)
    if name == 'hardsigmoid': return F.hardsigmoid(h)
    return h  # 'none'


def _binarize_ste(h, threshold=0.5):
    """Threshold to {0, 1}. Forward: hard binary, backward: identity gradient."""
    binary = (h > threshold).to(dtype=h.dtype)
    return h + (binary - h).detach()


def _thermometer_ste(h, n_bits, training):
    """Thermometer encoding: each scalar in [0,1] becomes n_bits binary features.
    bit_i = (h > (i+1)/(n_bits+1)). Output shape: (..., D) -> (..., D*n_bits).

    Forward: hard binary thermometer.
    Backward: TRUE identity STE. Each bit contributes gradient 1/n_bits w.r.t. h,
    summed across bits = 1. Previously used a clamped-ramp surrogate where total
    gradient scaled with h (vanishing near 0, exploding near 1) - that broke STE
    semantics and starved low-magnitude inputs of learning signal."""
    *prefix, D = h.shape
    levels = torch.linspace(1.0 / (n_bits + 1), n_bits / (n_bits + 1), n_bits,
                            device=h.device, dtype=h.dtype)
    expanded = h.unsqueeze(-1).expand(*prefix, D, n_bits)
    hard = (expanded > levels).to(dtype=h.dtype)
    if training:
        # Identity STE: backward grad = d(out_total)/dh = 1. Each of n_bits outputs
        # contributes h/n_bits in the soft path, so summed gradient w.r.t. h = 1.
        soft = expanded / n_bits
        out = soft + (hard - soft).detach()
    else:
        out = hard
    return out.reshape(*prefix, D * n_bits)


def signed_bits_per_scalar(n_bits):
    """Bit budget of the signed encoding: 1 sign bit + n_bits pos-mag + n_bits neg-mag."""
    return 2 * n_bits + 1


def _to_bits(layer, h, training):
    """Shared real->binary encoding for every logic block. Signed mode bypasses the [0,1]
    squash; otherwise squash (activation or learned calibration) then thermometer/threshold."""
    if getattr(layer, 'signed_encoding', False):
        return _signed_thermometer_ste(h, layer.n_bits, training)
    if getattr(layer, 'binary_encoder', 'activation') == 'lloydmax':
        return _lloydmax_thermometer_ste(layer, h, training)
    h = layer._squash(h)
    if not layer.binary_io:
        return h
    if layer.n_bits > 1:
        return _thermometer_ste(h, layer.n_bits, training)
    return _binarize_ste(h) if training else (h > 0.5).to(dtype=h.dtype)


def _chain_gates(layer, h, call):
    """Run the gate sublayers. With logic_residual (A1), deeper layers read BOTH the original
    input bits and the previous layer's output (DAG wiring -- wires are free in hardware),
    fixing the error-compounding that made naive depth hurt."""
    residual = getattr(layer, 'logic_residual', False)
    bits0 = h
    out = call(layer.logic[0], h)
    for l in layer.logic[1:]:
        out = call(l, torch.cat([bits0, out], dim=-1) if residual else out)
    return out


def _run_logic_stack(layer, h, hard, ste):
    """Run the logic sublayers, optionally gradient-checkpointed (recompute in backward to
    cap memory for large LUT-K gates). Checkpoint only in training (no grad needed at eval).
    #5 ensemble: when ensemble_banks is set, run N independent depth-1 gate banks on the
    same input bits and average their outputs (variance reduction over the candidate lottery)."""
    banks = getattr(layer, 'ensemble_banks', None)
    if banks is not None:
        if getattr(layer, 'grad_checkpoint', False) and layer.training:
            outs = [_ckpt.checkpoint(lambda x, bb=b: bb(x, hard=hard, ste=ste), h,
                                     use_reentrant=False) for b in banks]
        else:
            outs = [b(h, hard=hard, ste=ste) for b in banks]
        return sum(outs) / len(outs)
    call = lambda l, x: l(x, hard=hard, ste=ste)
    if getattr(layer, 'grad_checkpoint', False) and layer.training:
        return _ckpt.checkpoint(lambda x: _chain_gates(layer, x, call), h, use_reentrant=False)
    return _chain_gates(layer, h, call)


def _run_logic_stack_hard(layer, h):
    """Hard-model mirror of _run_logic_stack (same DAG wiring, discrete sublayers)."""
    banks = getattr(layer, 'ensemble_banks', None)
    if banks is not None:
        outs = [b(h) for b in banks]
        return sum(outs) / len(outs)
    return _chain_gates(layer, h, lambda l, x: l(x))


def _apply_alpha(layer, out):
    """#4 residual gate scaling: per-channel learned alpha on the LGN contribution, so the
    circuit can supply a scaled correction (x + alpha*LGN) rather than a full reconstruction.
    Init alpha=1 -> identical to current; mirrored as a frozen buffer in the hard model."""
    if getattr(layer, 'residual_scale', False):
        return out * layer.pool_alpha
    return out


def _count_curve(layer, grp):
    """#2 learned per-channel nonlinear readout: map the group bit-count through a per-channel
    learned curve (linear interp between integer-count knots). Honest (per-channel, no
    cross-channel mixing); strictly generalizes learn_pool's affine count->value. Init linear
    so it starts == fixed sum_pool. Hard counts are integer -> exact knot (soft==hard)."""
    count = grp.sum(dim=-1)                                  # (N, C) in [0, g]
    g = layer.group_size
    lo = count.detach().floor().clamp(0, g - 1)
    frac = (count - lo).clamp(0, 1)
    lo_i = lo.long()
    N, C = count.shape
    curve = layer.pool_curve.unsqueeze(0).expand(N, C, g + 1)   # (N, C, g+1)
    v0 = torch.gather(curve, 2, lo_i.unsqueeze(-1)).squeeze(-1)
    v1 = torch.gather(curve, 2, (lo_i + 1).unsqueeze(-1)).squeeze(-1)
    return v0 * (1 - frac) + v1 * frac


def _init_readout_extras(layer, residual_scale, pool_curve):
    """Create #2 (pool_curve) and #4 (residual_scale) readout params on a soft layer."""
    layer.residual_scale = residual_scale
    if residual_scale:
        layer.pool_alpha = nn.Parameter(torch.ones(layer.C))
    layer.pool_curve_enabled = pool_curve
    if pool_curve:
        assert getattr(layer, 'sum_pool', False), "pool_curve requires sum_pool"
        g = layer.group_size
        base = (torch.arange(g + 1, dtype=torch.float32) - g / 2) / (g / 2)
        layer.pool_curve = nn.Parameter(base.unsqueeze(0).repeat(layer.C, 1))


def _copy_readout_extras(hard, soft):
    """Mirror #2/#4 readout params onto a hard layer as frozen buffers."""
    hard.residual_scale = getattr(soft, 'residual_scale', False)
    if hard.residual_scale:
        hard.register_buffer('pool_alpha', soft.pool_alpha.detach().clone())
    hard.pool_curve_enabled = getattr(soft, 'pool_curve_enabled', False)
    if hard.pool_curve_enabled:
        hard.register_buffer('pool_curve', soft.pool_curve.detach().clone())


def _pool(layer, h, B, T):
    """Shared readout: bits -> (B,T,C). pool_curve = per-channel learned nonlinear count->value
    curve (#2); weighted_pool = per-channel learned weight per bit; learn_pool = per-channel
    affine on the bit count; else fixed centering. residual_scale (#4) applies a per-channel
    alpha to the result. All are block-diagonal except out_proj (non-sum_pool dense baseline)."""
    if not layer.sum_pool:
        return _apply_alpha(layer, layer.out_proj(h).view(B, T, layer.C))
    grp = h.view(B * T, layer.C, layer.group_size)
    if getattr(layer, 'pool_curve_enabled', False):
        normed = _count_curve(layer, grp)
    elif getattr(layer, 'weighted_pool', False):
        normed = (grp * layer.pool_w).sum(dim=-1) + layer.pool_b
    elif layer.learn_pool:
        normed = grp.sum(dim=-1) * layer.pool_scale + layer.pool_shift
    else:
        normed = (grp.sum(dim=-1) - layer.group_size / 2) / (layer.group_size / 2)
    return _apply_alpha(layer, normed.view(B, T, layer.C))


def _signed_thermometer_ste(h, n_bits, training):
    """Signed encoding for ~Gaussian post-LayerNorm activations: a sign bit plus a
    positive-magnitude thermometer plus a negative-magnitude thermometer. Each scalar
    -> (2*n_bits + 1) bits. Unlike the plain path it does NOT pre-squash to [0,1], so it
    keeps both sign and magnitude resolution (zero-centered activations were wasting half
    the thermometer range under sigmoid). Forward hard, backward identity STE.
    bit layout per scalar: [sign, pos_1..pos_n, neg_1..neg_n]."""
    *prefix, D = h.shape
    levels = torch.linspace(1.0 / (n_bits + 1), n_bits / (n_bits + 1), n_bits,
                            device=h.device, dtype=h.dtype)
    # magnitudes squashed into [0,1): 2*sigmoid(relu(.)) - 1 maps [0,inf) -> [0,1)
    pos_p = (2.0 * torch.sigmoid(torch.relu(h)) - 1.0).unsqueeze(-1)   # (..., D, 1)
    neg_p = (2.0 * torch.sigmoid(torch.relu(-h)) - 1.0).unsqueeze(-1)
    sign = (h > 0).to(dtype=h.dtype).unsqueeze(-1)                     # (..., D, 1)
    pos_hard = (pos_p > levels).to(dtype=h.dtype)                      # (..., D, n_bits)
    neg_hard = (neg_p > levels).to(dtype=h.dtype)
    hard = torch.cat([sign, pos_hard, neg_hard], dim=-1)              # (..., D, 2n+1)
    if training:
        sign_soft = torch.sigmoid(h).unsqueeze(-1)                    # smooth sign surrogate
        pos_soft = (pos_p / n_bits).expand(*prefix, D, n_bits)        # identity-STE magnitude
        neg_soft = (neg_p / n_bits).expand(*prefix, D, n_bits)
        soft = torch.cat([sign_soft, pos_soft, neg_soft], dim=-1)
        out = soft + (hard - soft).detach()
    else:
        out = hard
    return out.reshape(*prefix, D * (2 * n_bits + 1))


def _lloydmax_thermometer_ste(layer, h, training):
    """Gaussian Lloyd-Max threshold encoding with activation-distribution EMA stats."""
    if not layer.binary_io:
        return h
    if training and getattr(layer, 'lloyd_update_stats', True):
        with torch.no_grad():
            batch_mean = h.detach().mean(dim=0)
            batch_std = h.detach().std(dim=0, unbiased=False).clamp_min(layer.lloyd_min_std)
            beta = layer.lloyd_ema
            layer.lloyd_mean.mul_(beta).add_(batch_mean, alpha=1.0 - beta)
            layer.lloyd_std.mul_(beta).add_(batch_std, alpha=1.0 - beta)
    thresholds = (layer.lloyd_mean.unsqueeze(-1) +
                  layer.lloyd_std.clamp_min(layer.lloyd_min_std).unsqueeze(-1) *
                  layer.lloyd_base_thresholds.to(device=h.device, dtype=h.dtype).unsqueeze(0))
    expanded = h.unsqueeze(-1)
    hard = (expanded > thresholds.unsqueeze(0)).to(dtype=h.dtype)
    if training:
        soft = expanded / layer.n_bits
        out = soft + (hard - soft).detach()
    else:
        out = hard
    return out.reshape(h.shape[0], h.shape[1] * layer.n_bits)

# ---------------------------------------------------------------------------
# Soft learnable logic layer
# ---------------------------------------------------------------------------

def _sample_cand(in_dim, shape, generator, cand_dist=None):
    """Sample candidate input indices. Uniform random by default; if cand_dist (an importance
    distribution over inputs, e.g. from the MLP's weights -- functional init) is given, sample
    preferentially from important inputs instead of the blind random lottery."""
    n = 1
    for s in shape:
        n *= s
    if cand_dist is not None:
        # CPU generator -> sample on CPU regardless of the model's device (the buffer moves
        # with the module later). Fixes cuda-tensor + cpu-generator multinomial mismatch.
        d = cand_dist.detach().cpu().to(torch.float).clamp_min(0) + 1e-6
        idx = torch.multinomial(d, n, replacement=True, generator=generator)
        return idx.reshape(*shape)
    return torch.randint(0, in_dim, shape, generator=generator)


class LearnedLogicLayer(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, seed=None, temperature=1.0,
                 conn_init_scale=0.02, gate_init_scale=0.02, identity=False, cand_dist=None,
                 interconnect='random', topk_sparse_k=8, topk_sparse_scale=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.interconnect = interconnect
        self.temperature = float(temperature)
        # CAGE (Align Forward Adapt Backward, 2026): independent backward-pass temperature.
        # None = use self.temperature for backward (vanilla STE). When set, softmax(logits/τ_b)
        # is used in the STE backward path while forward stays hard argmax.
        self.backward_temp = None
        self.identity = identity
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        if interconnect == 'topk_block_sparse':
            self.topk_a = TopKBlockSparseInterconnect(
                in_dim, out_dim, topk_sparse_k, name='logic_a',
                sparsity_scale=topk_sparse_scale, seed=None if seed is None else seed + 17)
            self.topk_b = TopKBlockSparseInterconnect(
                in_dim, out_dim, topk_sparse_k, name='logic_b',
                sparsity_scale=topk_sparse_scale, seed=None if seed is None else seed + 31)
        elif interconnect == 'random':
            self.register_buffer('cand_a', _sample_cand(in_dim, (out_dim, k), g, cand_dist))
            self.register_buffer('cand_b', _sample_cand(in_dim, (out_dim, k), g, cand_dist))
            self.conn_logits_a = nn.Parameter(torch.randn(out_dim, k) * conn_init_scale)
            self.conn_logits_b = nn.Parameter(torch.randn(out_dim, k) * conn_init_scale)
        else:
            raise ValueError(f"unknown interconnect '{interconnect}'")
        self.gate_logits = nn.Parameter(torch.randn(out_dim, 16) * gate_init_scale)

    def set_temperature(self, t):
        self.temperature = float(t)

    def set_backward_temp(self, t):
        """CAGE: set independent backward STE temperature (None = use self.temperature)."""
        self.backward_temp = None if t is None else float(t)

    def _sm(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)

    def _sm_back(self, logits):
        """Backward-pass softmax: uses backward_temp if set (CAGE), else self.temperature."""
        t = self.backward_temp if self.backward_temp is not None else self.temperature
        return F.softmax(logits / t, dim=-1)

    @torch.no_grad()
    def commitment(self):
        """CAGE: average max-softmax across this layer's logits, used to update τ_b.
        Higher = more committed to a single choice."""
        gate_c = float(self._sm(self.gate_logits).max(dim=-1).values.mean())
        if self.interconnect == 'topk_block_sparse':
            conn_c = 0.5 * (self.topk_a.commitment() + self.topk_b.commitment())
        else:
            conn_c = 0.5 * (float(self._sm(self.conn_logits_a).max(dim=-1).values.mean()) +
                            float(self._sm(self.conn_logits_b).max(dim=-1).values.mean()))
        return 0.5 * (gate_c + conn_c)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        def ent(logits):
            p = self._sm(logits)
            return -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        if self.interconnect == 'topk_block_sparse':
            conn_term = self.topk_a.entropy_loss(conn_w) + self.topk_b.entropy_loss(conn_w)
        else:
            conn_term = conn_w * (ent(self.conn_logits_a) + ent(self.conn_logits_b))
        return conn_term + gate_w * ent(self.gate_logits)

    @torch.no_grad()
    def sharpness(self):
        if self.interconnect == 'topk_block_sparse':
            conn_a = self.topk_a.commitment()
            conn_b = self.topk_b.commitment()
        else:
            conn_a = float(self._sm(self.conn_logits_a).max(dim=-1).values.mean())
            conn_b = float(self._sm(self.conn_logits_b).max(dim=-1).values.mean())
        return {
            'conn_a': conn_a,
            'conn_b': conn_b,
            'gate':   float(self._sm(self.gate_logits).max(dim=-1).values.mean()),
        }

    def _select(self, x, cand, logits, hard, ste=False):
        gathered = x[:, cand]
        if ste:
            # STE (vanilla or CAGE if backward_temp set): forward=hard, backward=soft.
            soft = self._sm_back(logits)
            hard_w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
            w = soft + (hard_w - soft).detach()
        elif hard:
            w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
        else:
            w = self._sm(logits)
        return (gathered * w).sum(dim=-1)

    def forward(self, x, hard=False, ste=False):
        if self.identity:
            return x  # ablation: bypass all logic computation
        if self.interconnect == 'topk_block_sparse':
            a = self.topk_a(x, hard=hard, ste=ste)
            b = self.topk_b(x, hard=hard, ste=ste)
        else:
            a = self._select(x, self.cand_a, self.conn_logits_a, hard, ste=ste)
            b = self._select(x, self.cand_b, self.conn_logits_b, hard, ste=ste)
        gates = diff_logic_gates(a, b)
        if ste:
            soft = self._sm_back(self.gate_logits)
            hard_gp = F.one_hot(self.gate_logits.argmax(dim=-1), 16).to(dtype=x.dtype)
            gp = soft + (hard_gp - soft).detach()
        elif hard:
            gp = F.one_hot(self.gate_logits.argmax(dim=-1), 16).to(dtype=x.dtype)
        else:
            gp = self._sm(self.gate_logits)
        return (gates * gp).sum(dim=-1)

# ---------------------------------------------------------------------------
# GPT block wrapper: norm -> sigmoid(proj) -> logic stack -> proj + residual
# ---------------------------------------------------------------------------

class LogicGateGPTLayer(nn.Module):
    def __init__(self, gpt_cfg, layer_idx, logic_width=None, depth=1, k=4, seed=1000,
                 activation='sigmoid', conn_init_scale=0.02, gate_init_scale=0.02,
                 identity_logic=False, binary_io=False, n_bits=1, sum_pool=False,
                 no_in_proj=False, learn_pool=False, token_shift=0,
                 learn_binary_calibration=False, signed_encoding=False, out_gate_mult=1,
                 weighted_pool=False, lut_k=0, grad_checkpoint=False,
                 logic_residual=False, gated_lut=False,
                 pre_conv1d=False, pre_conv1d_channels=0, pre_conv1d_kernel=3,
                 pre_conv1d_stride=1, pre_conv1d_groups=1,
                 post_conv1d=False, post_conv1d_channels=0, post_conv1d_kernel=3,
                 post_conv1d_stride=1, post_conv1d_groups=1,
                 binary_encoder='activation', lloyd_ema=0.99, lloyd_min_std=1e-3,
                 interconnect='random', topk_sparse_k=8, topk_sparse_scale=1.0,
                 residual_scale=False, pool_curve=False, ensemble=1):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.logic_residual = logic_residual and depth > 1
        C = gpt_cfg.n_embd
        self.C = C
        self.layer_idx = layer_idx
        self.logic_width = logic_width or C * 4
        self.activation = activation
        self.binary_io = binary_io
        self.n_bits    = n_bits if binary_io else 1
        self.sum_pool  = sum_pool
        self.learn_pool = learn_pool
        self.token_shift = token_shift
        self.signed_encoding = signed_encoding
        # bits emitted per input scalar (signed encoding triples-ish the budget)
        self._bps = signed_bits_per_scalar(self.n_bits) if signed_encoding else self.n_bits
        # Causal token shift: each position sees [x[t-K]..x[t]]. eff_C = (K+1)*C.
        self._taps = list(range(1, token_shift + 1)) if token_shift > 0 else []
        eff_C = C * (len(self._taps) + 1)
        self.eff_C = eff_C
        self.pre_conv1d_enabled = pre_conv1d
        logic_in_C = eff_C
        if pre_conv1d:
            logic_in_C = pre_conv1d_channels or eff_C
            self.pre_conv1d = CausalConv1D(
                eff_C, logic_in_C, kernel_size=pre_conv1d_kernel,
                stride=pre_conv1d_stride, groups=pre_conv1d_groups, name='pre_conv1d')
        self.logic_in_C = logic_in_C
        self.post_conv1d_enabled = post_conv1d
        if post_conv1d:
            post_C = post_conv1d_channels or C
            self.post_conv1d = CausalConv1D(
                C, post_C, kernel_size=post_conv1d_kernel,
                stride=post_conv1d_stride, groups=post_conv1d_groups, name='post_conv1d')
            if post_C != C:
                self.post_conv1d_return = nn.Conv1d(post_C, C, kernel_size=1)
        self.no_in_proj = no_in_proj
        if no_in_proj:
            assert binary_io, "no_in_proj requires binary_io"
            bit_width = logic_in_C * self._bps
        else:
            bit_width = self.logic_width * self._bps
        # Output-side resolution: widen the FINAL logic layer by out_gate_mult so each output
        # channel sums more gates -> a finer readout (group_size*mult levels). Decouples output
        # resolution from input n_bits (which in aggressive mode also sets group_size).
        if identity_logic and out_gate_mult != 1:
            raise ValueError("identity_logic is incompatible with out_gate_mult>1 (non-square logic).")
        self.out_gate_mult = out_gate_mult
        out_width = bit_width * out_gate_mult
        if sum_pool:
            assert binary_io, "sum_pool requires binary_io"
            assert out_width % C == 0, (
                f"sum_pool requires out_width ({out_width}) divisible by n_embd ({C}).")
            self.group_size = out_width // C
            self.weighted_pool = weighted_pool
            if weighted_pool:
                # per-channel, per-bit weight; init uniform (2/g) + (-1) bias == learn_pool/fixed.
                self.pool_w = nn.Parameter(torch.full((C, self.group_size), 2.0 / self.group_size))
                self.pool_b = nn.Parameter(torch.full((C,), -1.0))
            elif learn_pool:
                # Init to match fixed centering: (pooled - g/2)/(g/2) = pooled*(2/g) - 1
                self.pool_scale = nn.Parameter(torch.full((C,), 2.0 / self.group_size))
                self.pool_shift = nn.Parameter(torch.full((C,), -1.0))
        else:
            self.weighted_pool = weighted_pool
        self.norm = nn.LayerNorm(C)
        if not no_in_proj:
            self.in_proj = nn.Linear(logic_in_C, self.logic_width)
        # depth-1 intermediate layers stay square (bit_width); the last widens to out_width.
        # lut_k>=2 swaps the 2-input gate for a more expressive K-input LUT gate.
        # A1: residual wiring -> deeper layers read [input_bits, prev_out] (2x in_dim).
        self.lut_k = lut_k
        def _in(i):
            return bit_width if i == 0 else (2 * bit_width if self.logic_residual else bit_width)
        self.logic = nn.ModuleList([
            _make_gate_layer(lut_k, _in(i), out_width if i == depth - 1 else bit_width,
                             k=k, seed=seed + layer_idx * 100 + i, gated=gated_lut,
                             conn_init_scale=conn_init_scale, gate_init_scale=gate_init_scale,
                             identity=identity_logic, interconnect=interconnect,
                             topk_sparse_k=topk_sparse_k, topk_sparse_scale=topk_sparse_scale)
            for i in range(depth)
        ])
        self.ensemble = ensemble
        if ensemble > 1:
            assert depth == 1, "ensemble requires depth==1"
            self.logic = nn.ModuleList([
                _make_gate_layer(lut_k, bit_width, out_width, k=k,
                                 seed=seed + layer_idx * 100 + 7000 * j, gated=gated_lut,
                                 conn_init_scale=conn_init_scale, gate_init_scale=gate_init_scale,
                                 identity=identity_logic, interconnect=interconnect,
                                 topk_sparse_k=topk_sparse_k, topk_sparse_scale=topk_sparse_scale)
                for j in range(ensemble)])
            self.ensemble_banks = self.logic
        else:
            self.ensemble_banks = None
        _init_readout_extras(self, residual_scale, pool_curve)
        if not sum_pool:
            self.out_proj = nn.Linear(out_width, C)
        # Per-channel binary calibration on the pre-thermometer features (width = bit_width//n_bits).
        self.learn_binary_calibration = learn_binary_calibration
        if learn_binary_calibration:
            feat_w = logic_in_C if no_in_proj else self.logic_width
            self.cal_scale = nn.Parameter(torch.ones(feat_w))
            self.cal_shift = nn.Parameter(torch.zeros(feat_w))
        self.binary_encoder = binary_encoder
        self.lloyd_ema = lloyd_ema
        self.lloyd_min_std = lloyd_min_std
        self.lloyd_update_stats = True
        if binary_encoder == 'lloydmax':
            if not binary_io:
                raise ValueError("binary_encoder='lloydmax' requires binary_io=True.")
            if signed_encoding or learn_binary_calibration:
                raise ValueError("binary_encoder='lloydmax' is mutually exclusive with "
                                 "signed_encoding and learn_binary_calibration.")
            feat_w = logic_in_C if no_in_proj else self.logic_width
            self.register_buffer('lloyd_base_thresholds',
                                 _lloyd_max_standard_thresholds(self.n_bits))
            self.register_buffer('lloyd_mean', torch.zeros(feat_w))
            self.register_buffer('lloyd_std', torch.ones(feat_w))
        elif binary_encoder != 'activation':
            raise ValueError(f"unknown binary_encoder '{binary_encoder}'")
        self.dropout = nn.Dropout(gpt_cfg.dropout)
        self.use_ste = False  # STE toggle (set during fine-tune / CAGE)

    def _squash(self, h):
        """Map features into [0,1] before binarization. Calibration mode replaces the
        plain activation with a learned per-channel sigmoid(scale*h + bias)."""
        if self.learn_binary_calibration:
            return torch.sigmoid(h * self.cal_scale + self.cal_shift)
        return _apply_activation(h, self.activation)

    def set_temperature(self, t):
        for l in self.logic: l.set_temperature(t)

    def set_backward_temp(self, t):
        """CAGE: propagate independent backward STE temperature to all sublayers."""
        for l in self.logic: l.set_backward_temp(t)

    @torch.no_grad()
    def commitment(self):
        """CAGE: average commitment confidence across this block's sublayers."""
        if not self.logic:
            return 1.0
        return sum(l.commitment() for l in self.logic) / len(self.logic)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _aggregate(self, h, B, T):
        """(B*T, bit_width) → (B, T, C). sum_pool: fixed group-sum (+ optional learn_pool
        affine); else trained out_proj Linear."""
        return _pool(self, h, B, T)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.logic_in_C)
        return self.in_proj(normed_btx.reshape(B * T, self.logic_in_C))

    def _apply_pre_conv(self, normed):
        return self.pre_conv1d(normed) if self.pre_conv1d_enabled else normed

    def _apply_post_conv(self, y):
        if not self.post_conv1d_enabled:
            return y
        y = self.post_conv1d(y)
        if hasattr(self, 'post_conv1d_return'):
            y = self.post_conv1d_return(y.transpose(1, 2)).transpose(1, 2)
        return y

    def forward(self, x, hard=False):
        B, T, C = x.shape
        normed = self._apply_pre_conv(apply_token_shift(self.norm(x), self._taps))
        h = self._apply_in_proj(normed, B, T)
        h = _to_bits(self, h, self.training)
        ste = self.use_ste and not hard
        h = _run_logic_stack(self, h, hard, ste)
        return x + self.dropout(self._apply_post_conv(self._aggregate(h, B, T)))

# ---------------------------------------------------------------------------
# Hard (fully discrete) versions
# ---------------------------------------------------------------------------

class HardLogicLayer(nn.Module):
    def __init__(self, soft: LearnedLogicLayer):
        super().__init__()
        self.identity = getattr(soft, 'identity', False)
        self.interconnect = getattr(soft, 'interconnect', 'random')
        with torch.no_grad():
            if self.interconnect == 'topk_block_sparse':
                self.topk_a = HardTopKBlockSparseInterconnect(soft.topk_a)
                self.topk_b = HardTopKBlockSparseInterconnect(soft.topk_b)
            else:
                choice_a = soft.conn_logits_a.argmax(dim=-1)
                choice_b = soft.conn_logits_b.argmax(dim=-1)
                idx_a = soft.cand_a.gather(1, choice_a.unsqueeze(1)).squeeze(1)
                idx_b = soft.cand_b.gather(1, choice_b.unsqueeze(1)).squeeze(1)
                self.register_buffer('idx_a', idx_a.clone())
                self.register_buffer('idx_b', idx_b.clone())
            self.register_buffer('coeffs', LOGIC_GATE_MATRIX[soft.gate_logits.argmax(dim=-1).cpu()].clone())

    def forward(self, x):
        if self.identity:
            return x
        if self.interconnect == 'topk_block_sparse':
            a, b = self.topk_a(x), self.topk_b(x)
        else:
            a, b = x[:, self.idx_a], x[:, self.idx_b]
        c = self.coeffs.to(device=x.device, dtype=x.dtype)
        return c[:, 0] + c[:, 1]*a + c[:, 2]*b + c[:, 3]*a*b


# ---------------------------------------------------------------------------
# K-input LUT gate (LUT-K): a more expressive primitive than the 2-input gate.
# Each output selects K inputs and applies a learned 2^K-entry truth table via the
# multilinear extension (out = sum_corners T[c] * prod_i [a_i if c_i else 1-a_i]),
# the K-input generalization of the 2-input [1,A,B,AB] polynomial. Hard-snaps to a
# discrete LUT-K = exactly one FPGA LUT. lut_k=2 reproduces the 2-input gate's expressivity.
# ---------------------------------------------------------------------------

def _corner_table(lut_k, device=None, dtype=torch.float32):
    return torch.tensor([[(i >> b) & 1 for b in range(lut_k)] for i in range(2 ** lut_k)],
                        dtype=dtype, device=device)


def _multilinear(a, corners, T):
    """Multilinear extension of a truth table. a: (N, M, K) inputs in [0,1]; corners: (2^K, K);
    T: (..., M, 2^K) -> (N, M). Builds the per-corner product incrementally (peak (N,M,2^K),
    not (N,M,2^K,K)) so the 12-layer model fits in memory."""
    weights = None                                       # (N, M, 2^K)
    for s in range(corners.shape[1]):                    # over the K input slots
        cs = corners[:, s]                               # (2^K,)
        a_s = a[..., s:s + 1]                            # (N, M, 1)
        factor = cs * a_s + (1 - cs) * (1 - a_s)         # (N, M, 2^K)
        weights = factor if weights is None else weights * factor
    return (weights * T).sum(dim=-1)                     # (N, M)


class LearnedLUTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, lut_k=4, k=4, seed=None, temperature=1.0,
                 conn_init_scale=0.02, gate_init_scale=0.02, identity=False, cand_dist=None,
                 gated=False, interconnect='random', topk_sparse_k=8, topk_sparse_scale=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lut_k = lut_k
        self.k = k
        self.interconnect = interconnect
        self.temperature = float(temperature)
        self.backward_temp = None
        self.identity = identity
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        # Gated pairs (A2): internally double the LUT bank; output = lut_a * lut_b (hard: AND).
        # A 2K-input function class at an honest 2x LUT cost.
        self.gated = gated
        M = out_dim * (2 if gated else 1)
        self.M = M
        # K input slots, each selected via softmax over k candidate connections (random or MLP-guided)
        if interconnect == 'topk_block_sparse':
            self.topk = TopKBlockSparseInterconnect(
                in_dim, M * lut_k, topk_sparse_k, name='lut_topk',
                sparsity_scale=topk_sparse_scale, seed=None if seed is None else seed + 43)
        elif interconnect == 'random':
            self.register_buffer('cand', _sample_cand(in_dim, (M, lut_k, k), g, cand_dist))
            self.conn_logits = nn.Parameter(torch.randn(M, lut_k, k) * conn_init_scale)
        else:
            raise ValueError(f"unknown interconnect '{interconnect}'")
        # 2^K learnable truth-table logits per output (sigmoid -> [0,1], snap -> {0,1})
        self.lut_logits = nn.Parameter(torch.randn(M, 2 ** lut_k) * gate_init_scale)
        self.register_buffer('corners', _corner_table(lut_k))

    def set_temperature(self, t): self.temperature = float(t)
    def set_backward_temp(self, t): self.backward_temp = None if t is None else float(t)
    def _sm(self, logits): return F.softmax(logits / self.temperature, dim=-1)
    def _sm_back(self, logits):
        t = self.backward_temp if self.backward_temp is not None else self.temperature
        return F.softmax(logits / t, dim=-1)

    @torch.no_grad()
    def commitment(self):
        if self.interconnect == 'topk_block_sparse':
            conn_c = self.topk.commitment()
        else:
            conn_c = float(self._sm(self.conn_logits).max(dim=-1).values.mean())
        p = torch.sigmoid(self.lut_logits)
        lut_c = float(torch.maximum(p, 1 - p).mean())
        return 0.5 * (conn_c + lut_c)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        if self.interconnect == 'topk_block_sparse':
            conn_term = self.topk.entropy_loss(conn_w)
        else:
            p = self._sm(self.conn_logits)
            conn_ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
            conn_term = conn_w * conn_ent
        q = torch.sigmoid(self.lut_logits)
        lut_ent = -(q * (q + 1e-8).log() + (1 - q) * ((1 - q) + 1e-8).log()).mean()
        return conn_term + gate_w * lut_ent

    @torch.no_grad()
    def sharpness(self):
        conn = (self.topk.commitment() if self.interconnect == 'topk_block_sparse'
                else float(self._sm(self.conn_logits).max(dim=-1).values.mean()))
        p = torch.sigmoid(self.lut_logits)
        gate = float(torch.maximum(p, 1 - p).mean())
        return {'conn_a': conn, 'conn_b': conn, 'gate': gate}

    def _select(self, x, slot, hard, ste):
        cand, logits = self.cand[:, slot], self.conn_logits[:, slot]
        gathered = x[:, cand]                              # (N, out_dim, k)
        if ste:
            soft = self._sm_back(logits)
            hard_w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
            w = soft + (hard_w - soft).detach()
        elif hard:
            w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
        else:
            w = self._sm(logits)
        return (gathered * w).sum(dim=-1)                  # (N, out_dim)

    def forward(self, x, hard=False, ste=False):
        if self.identity:
            return x
        if self.interconnect == 'topk_block_sparse':
            a = self.topk(x, hard=hard, ste=ste).view(x.shape[0], self.M, self.lut_k)
        else:
            a = torch.stack([self._select(x, s, hard, ste) for s in range(self.lut_k)], dim=-1)  # (N,M,K)
        if ste:
            t = self.backward_temp if self.backward_temp is not None else self.temperature
            soft = torch.sigmoid(self.lut_logits / t)
            hard_T = (self.lut_logits > 0).to(dtype=x.dtype)
            T = soft + (hard_T - soft).detach()
        elif hard:
            T = (self.lut_logits > 0).to(dtype=x.dtype)
        else:
            T = torch.sigmoid(self.lut_logits / self.temperature)
        y = _multilinear(a, self.corners, T.unsqueeze(0))
        if self.gated:                                     # (N, 2*out) -> AND/product of halves
            half = y.shape[-1] // 2
            y = y[:, :half] * y[:, half:]
        return y


class HardLUTLayer(nn.Module):
    def __init__(self, soft: LearnedLUTLayer):
        super().__init__()
        self.identity = getattr(soft, 'identity', False)
        self.lut_k = soft.lut_k
        self.gated = getattr(soft, 'gated', False)
        self.interconnect = getattr(soft, 'interconnect', 'random')
        self.M = getattr(soft, 'M', soft.out_dim * (2 if self.gated else 1))
        with torch.no_grad():
            if self.interconnect == 'topk_block_sparse':
                self.topk = HardTopKBlockSparseInterconnect(soft.topk)
            else:
                choice = soft.conn_logits.argmax(dim=-1)               # (out_dim, lut_k)
                idx = soft.cand.gather(2, choice.unsqueeze(-1)).squeeze(-1)
                self.register_buffer('idx', idx.clone())
            self.register_buffer('lut', (soft.lut_logits > 0).to(torch.float32))
            self.register_buffer('corners', soft.corners.detach().clone())

    def forward(self, x):
        if self.identity:
            return x
        if self.interconnect == 'topk_block_sparse':
            a = self.topk(x).view(x.shape[0], self.M, self.lut_k)
        else:
            a = x[:, self.idx]                                      # (N, M, lut_k)
        T = self.lut.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        y = _multilinear(a, self.corners.to(dtype=x.dtype), T)
        if self.gated:                                              # AND of the two LUT banks
            half = y.shape[-1] // 2
            y = y[:, :half] * y[:, half:]
        return y


def _make_gate_layer(lut_k, in_d, out_d, gated=False, **kw):
    """Build a gate sublayer: K-input LUT gate when lut_k>=2, else the 2-input gate."""
    if lut_k and lut_k >= 2:
        return LearnedLUTLayer(in_d, out_d, lut_k=lut_k, gated=gated, **kw)
    if gated:
        raise ValueError("gated_lut requires lut_k>=2 (it pairs two LUT banks).")
    return LearnedLogicLayer(in_d, out_d, **kw)


def _make_hard_gate_layer(soft):
    """Hard mirror of a gate sublayer (LUT or 2-input)."""
    return HardLUTLayer(soft) if isinstance(soft, LearnedLUTLayer) else HardLogicLayer(soft)


class HardLogicGateGPTLayer(nn.Module):
    def __init__(self, soft: LogicGateGPTLayer):
        super().__init__()
        self.layer_idx = soft.layer_idx
        self.activation = soft.activation
        self.binary_io  = soft.binary_io
        self.n_bits     = soft.n_bits
        self.sum_pool   = soft.sum_pool
        self.learn_pool = soft.learn_pool
        self.no_in_proj = soft.no_in_proj
        self.C          = soft.C
        self.eff_C      = soft.eff_C
        self.logic_in_C = soft.logic_in_C
        self.token_shift = soft.token_shift
        self._taps      = soft._taps
        self.pre_conv1d_enabled = getattr(soft, 'pre_conv1d_enabled', False)
        if self.pre_conv1d_enabled:
            self.pre_conv1d = copy.deepcopy(soft.pre_conv1d)
        self.post_conv1d_enabled = getattr(soft, 'post_conv1d_enabled', False)
        if self.post_conv1d_enabled:
            self.post_conv1d = copy.deepcopy(soft.post_conv1d)
            if hasattr(soft, 'post_conv1d_return'):
                self.post_conv1d_return = copy.deepcopy(soft.post_conv1d_return)
        self.group_size = getattr(soft, 'group_size', None)
        self.weighted_pool = getattr(soft, 'weighted_pool', False)
        if self.sum_pool and self.weighted_pool:
            self.register_buffer('pool_w', soft.pool_w.detach().clone())
            self.register_buffer('pool_b', soft.pool_b.detach().clone())
        elif self.sum_pool and self.learn_pool:
            self.register_buffer('pool_scale', soft.pool_scale.detach().clone())
            self.register_buffer('pool_shift', soft.pool_shift.detach().clone())
        self.norm     = copy.deepcopy(soft.norm)
        if not self.no_in_proj:
            self.in_proj = copy.deepcopy(soft.in_proj)
        if not self.sum_pool:
            self.out_proj = copy.deepcopy(soft.out_proj)
        self.dropout  = copy.deepcopy(soft.dropout)
        self.learn_binary_calibration = getattr(soft, 'learn_binary_calibration', False)
        self.signed_encoding = getattr(soft, 'signed_encoding', False)
        if self.learn_binary_calibration:
            self.register_buffer('cal_scale', soft.cal_scale.detach().clone())
            self.register_buffer('cal_shift', soft.cal_shift.detach().clone())
        self.binary_encoder = getattr(soft, 'binary_encoder', 'activation')
        self.lloyd_ema = getattr(soft, 'lloyd_ema', 0.99)
        self.lloyd_min_std = getattr(soft, 'lloyd_min_std', 1e-3)
        self.lloyd_update_stats = False
        if self.binary_encoder == 'lloydmax':
            self.register_buffer('lloyd_base_thresholds', soft.lloyd_base_thresholds.detach().clone())
            self.register_buffer('lloyd_mean', soft.lloyd_mean.detach().clone())
            self.register_buffer('lloyd_std', soft.lloyd_std.detach().clone())
        self.logic_residual = getattr(soft, 'logic_residual', False)
        self.logic    = nn.ModuleList([_make_hard_gate_layer(l) for l in soft.logic])
        self.ensemble_banks = self.logic if getattr(soft, 'ensemble_banks', None) is not None else None
        _copy_readout_extras(self, soft)
        for p in self.parameters():
            p.requires_grad = False

    def _squash(self, h):
        if self.learn_binary_calibration:
            return torch.sigmoid(h * self.cal_scale + self.cal_shift)
        return _apply_activation(h, self.activation)

    def _aggregate(self, h, B, T):
        return _pool(self, h, B, T)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.logic_in_C)
        return self.in_proj(normed_btx.reshape(B * T, self.logic_in_C))

    def _apply_pre_conv(self, normed):
        return self.pre_conv1d(normed) if self.pre_conv1d_enabled else normed

    def _apply_post_conv(self, y):
        if not self.post_conv1d_enabled:
            return y
        y = self.post_conv1d(y)
        if hasattr(self, 'post_conv1d_return'):
            y = self.post_conv1d_return(y.transpose(1, 2)).transpose(1, 2)
        return y

    def forward(self, x):
        B, T, C = x.shape
        normed = self._apply_pre_conv(apply_token_shift(self.norm(x), self._taps))
        h = self._apply_in_proj(normed, B, T)
        h = _to_bits(self, h, training=False)
        h = _run_logic_stack_hard(self, h)
        return x + self.dropout(self._apply_post_conv(self._aggregate(h, B, T)))


# ---------------------------------------------------------------------------
# Recurrent / stateful LGN layer (RDDLGN-inspired, NOT full encoder-decoder RDDLGN)
# ---------------------------------------------------------------------------

class RecurrentLogicGateGPTLayer(nn.Module):
    """Stateful logic block. Per token, a logic stack updates a hidden state from the
    current token's bits and the previous state:

        state_t = Logic([token_bits_t, state_{t-1}])
        out_t   = group_sum(state_t)            # aggregate state_width -> C

    Causal by construction: out_t depends only on tokens <= t. This is the RDDLGN idea
    (give the gates internal state so logic can mix across the sequence) applied as a
    drop-in GPT-block replacement — NOT a full RDDLGN encoder/decoder rewrite.
    """
    def __init__(self, gpt_cfg, layer_idx, logic_width=None, depth=1, k=4, seed=1000,
                 activation='sigmoid', conn_init_scale=0.02, gate_init_scale=0.02,
                 identity_logic=False, binary_io=True, n_bits=8, sum_pool=True,
                 no_in_proj=True, learn_pool=False,
                 state_width=None, recurrent_depth=1, state_init='zero'):
        super().__init__()
        C = gpt_cfg.n_embd
        self.C = C
        self.layer_idx = layer_idx
        self.logic_width = logic_width or C * 4
        self.activation = activation
        # Guardrails: fail early & clearly on configs the recurrent layer cannot honour.
        if not binary_io:
            raise ValueError("recurrent LGN requires binary_io=True (logic operates on bits).")
        if not sum_pool:
            raise ValueError("recurrent LGN currently only supports sum_pool aggregation; "
                             "pass sum_pool=True (group-sum of the state).")
        if state_init not in ('zero', 'learned', 'residual'):
            raise ValueError(f"unknown recurrent_state_init '{state_init}' "
                             "(expected 'zero' | 'learned' | 'residual').")
        self.binary_io = True
        self.sum_pool = True
        self.n_bits = n_bits
        self.no_in_proj = no_in_proj
        self.learn_pool = learn_pool
        self.state_init = state_init
        # token feature width fed into the logic stack (before binarization)
        token_feat = C if no_in_proj else self.logic_width
        self.token_bw = token_feat * n_bits
        # hidden state width (default = token bit width); must divide C for group-sum
        self.state_width = state_width if state_width is not None else self.token_bw
        if self.state_width % C != 0:
            raise ValueError(
                f"recurrent_state_width ({self.state_width}) must be divisible by n_embd ({C}).")
        self.group_size = self.state_width // C
        self.recurrent_depth = recurrent_depth
        self.norm = nn.LayerNorm(C)
        if not no_in_proj:
            self.in_proj = nn.Linear(C, self.logic_width)
        # logic stack: first layer reads [token_bits, state], the rest read state only
        layers = []
        for i in range(recurrent_depth):
            in_dim = (self.token_bw + self.state_width) if i == 0 else self.state_width
            layers.append(LearnedLogicLayer(
                in_dim, self.state_width, k=k, seed=seed + layer_idx * 100 + 50 + i,
                conn_init_scale=conn_init_scale, gate_init_scale=gate_init_scale,
                identity=identity_logic))
        self.logic = nn.ModuleList(layers)
        if learn_pool:
            self.pool_scale = nn.Parameter(torch.full((C,), 2.0 / self.group_size))
            self.pool_shift = nn.Parameter(torch.full((C,), -1.0))
        if state_init == 'learned':
            self.initial_state = nn.Parameter(torch.zeros(self.state_width))
        self.dropout = nn.Dropout(gpt_cfg.dropout)
        self.use_ste = False

    # --- API shared with LogicGateGPTLayer ---
    def set_temperature(self, t):
        for l in self.logic: l.set_temperature(t)

    def set_backward_temp(self, t):
        for l in self.logic: l.set_backward_temp(t)

    @torch.no_grad()
    def commitment(self):
        if not self.logic:
            return 1.0
        return sum(l.commitment() for l in self.logic) / len(self.logic)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    # --- forward machinery ---
    def _token_bits(self, x, training):
        h = self.norm(x)
        if not self.no_in_proj:
            h = self.in_proj(h)
        h = _apply_activation(h, self.activation)
        if self.n_bits > 1:
            return _thermometer_ste(h, self.n_bits, training)
        return _binarize_ste(h) if training else (h > 0.5).to(dtype=h.dtype)

    def _init_state(self, B, token_bits, device, dtype):
        if self.state_init == 'zero':
            return torch.zeros(B, self.state_width, device=device, dtype=dtype)
        if self.state_init == 'learned':
            return self.initial_state.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        if self.state_init == 'residual':
            first = token_bits[:, 0]                                  # (B, token_bw)
            if self.token_bw >= self.state_width:
                return first[:, :self.state_width]
            pad = torch.zeros(B, self.state_width - self.token_bw, device=device, dtype=dtype)
            return torch.cat([first, pad], dim=-1)
        raise ValueError(f"unknown state_init '{self.state_init}'")

    def _aggregate_state(self, state, B):
        grp = state.view(B, self.C, self.group_size)
        if self.learn_pool:
            return grp.sum(dim=-1) * self.pool_scale + self.pool_shift
        return (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)   # (B, C)

    def _run(self, x, hard, ste, training):
        B, T, C = x.shape
        token_bits = self._token_bits(x, training)                   # (B, T, token_bw)
        state = self._init_state(B, token_bits, x.device, x.dtype)
        outs = []
        for t in range(T):                                           # causal: only past/current
            z = torch.cat([token_bits[:, t], state], dim=-1)
            s = self.logic[0](z, hard=hard, ste=ste)
            for l in self.logic[1:]:
                s = l(s, hard=hard, ste=ste)
            state = s
            outs.append(self._aggregate_state(state, B))
        return torch.stack(outs, dim=1)                              # (B, T, C)

    def forward(self, x, hard=False):
        ste = self.use_ste and not hard
        return x + self.dropout(self._run(x, hard, ste, self.training))


class HardRecurrentLogicGateGPTLayer(nn.Module):
    """Hard-snapped mirror of RecurrentLogicGateGPTLayer (discrete gates)."""
    def __init__(self, soft: RecurrentLogicGateGPTLayer):
        super().__init__()
        self.layer_idx  = soft.layer_idx
        self.activation = soft.activation
        self.n_bits     = soft.n_bits
        self.no_in_proj = soft.no_in_proj
        self.learn_pool = soft.learn_pool
        self.state_init = soft.state_init
        self.C          = soft.C
        self.token_bw   = soft.token_bw
        self.state_width = soft.state_width
        self.group_size = soft.group_size
        self.norm = copy.deepcopy(soft.norm)
        if not self.no_in_proj:
            self.in_proj = copy.deepcopy(soft.in_proj)
        if self.learn_pool:
            self.register_buffer('pool_scale', soft.pool_scale.detach().clone())
            self.register_buffer('pool_shift', soft.pool_shift.detach().clone())
        if self.state_init == 'learned':
            self.register_buffer('initial_state', soft.initial_state.detach().clone())
        self.dropout = copy.deepcopy(soft.dropout)
        self.logic = nn.ModuleList([HardLogicLayer(l) for l in soft.logic])
        for p in self.parameters():
            p.requires_grad = False

    def _token_bits(self, x):
        h = self.norm(x)
        if not self.no_in_proj:
            h = self.in_proj(h)
        h = _apply_activation(h, self.activation)
        if self.n_bits > 1:
            return _thermometer_ste(h, self.n_bits, training=False)
        return (h > 0.5).to(dtype=h.dtype)

    def _init_state(self, B, token_bits, device, dtype):
        if self.state_init == 'zero':
            return torch.zeros(B, self.state_width, device=device, dtype=dtype)
        if self.state_init == 'learned':
            return self.initial_state.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        if self.state_init == 'residual':
            first = token_bits[:, 0]
            if self.token_bw >= self.state_width:
                return first[:, :self.state_width]
            pad = torch.zeros(B, self.state_width - self.token_bw, device=device, dtype=dtype)
            return torch.cat([first, pad], dim=-1)
        raise ValueError(f"unknown state_init '{self.state_init}'")

    def _aggregate_state(self, state, B):
        grp = state.view(B, self.C, self.group_size)
        if self.learn_pool:
            return grp.sum(dim=-1) * self.pool_scale + self.pool_shift
        return (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)

    def forward(self, x):
        B, T, C = x.shape
        token_bits = self._token_bits(x)
        state = self._init_state(B, token_bits, x.device, x.dtype)
        outs = []
        for t in range(T):
            z = torch.cat([token_bits[:, t], state], dim=-1)
            s = self.logic[0](z)
            for l in self.logic[1:]:
                s = l(s)
            state = s
            outs.append(self._aggregate_state(state, B))
        return x + self.dropout(torch.stack(outs, dim=1))


# ---------------------------------------------------------------------------
# Gated recurrent LGN: flip-flop / latch-style state retention (opt-in).
#   candidate_t = LogicCandidate([token_bits_t, state_{t-1}])
#   keep_t      = LogicKeep([token_bits_t, state_{t-1}])
#   state_t     = keep_t * state_{t-1} + (1 - keep_t) * candidate_t   (soft)
#   state_t     = where(keep_t, state_{t-1}, candidate_t)             (hard)
# The keep gate is itself a learned LOGIC stack (not a sigmoid/dense gate). This is an
# extension inspired by flip-flop/latch state retention; the original RDDLGN paper does
# NOT claim a GRU-style keep gate.
# ---------------------------------------------------------------------------

class GatedRecurrentLogicGateGPTLayer(RecurrentLogicGateGPTLayer):
    """Recurrent LGN with a logic-based keep/overwrite gate (opt-in extension)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gated = True
        # `self.logic` (from the parent) is the CANDIDATE stack. Build a same-shape KEEP stack.
        keep = []
        for i, cand_layer in enumerate(self.logic):
            keep.append(LearnedLogicLayer(
                cand_layer.in_dim, cand_layer.out_dim, k=cand_layer.k,
                seed=1234 + self.layer_idx * 100 + 70 + i))
        self.keep_logic = nn.ModuleList(keep)

    # --- API: iterate BOTH candidate (self.logic) and keep stacks ---
    def _all(self):
        return list(self.logic) + list(self.keep_logic)

    def set_temperature(self, t):
        for l in self._all(): l.set_temperature(t)

    def set_backward_temp(self, t):
        for l in self._all(): l.set_backward_temp(t)

    @torch.no_grad()
    def commitment(self):
        ls = self._all()
        return sum(l.commitment() for l in ls) / len(ls)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self._all())

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self._all()]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _run(self, x, hard, ste, training):
        B, T, C = x.shape
        token_bits = self._token_bits(x, training)
        state = self._init_state(B, token_bits, x.device, x.dtype)
        outs = []
        for t in range(T):
            z = torch.cat([token_bits[:, t], state], dim=-1)
            cand = self.logic[0](z, hard=hard, ste=ste)
            for l in self.logic[1:]:
                cand = l(cand, hard=hard, ste=ste)
            keep = self.keep_logic[0](z, hard=hard, ste=ste)
            for l in self.keep_logic[1:]:
                keep = l(keep, hard=hard, ste=ste)
            if hard:
                state = torch.where(keep > 0.5, state, cand)
            else:
                state = keep * state + (1.0 - keep) * cand   # keep in {0,1} under STE
            outs.append(self._aggregate_state(state, B))
        return torch.stack(outs, dim=1)


class HardGatedRecurrentLogicGateGPTLayer(HardRecurrentLogicGateGPTLayer):
    """Hard-snapped mirror of GatedRecurrentLogicGateGPTLayer."""
    def __init__(self, soft: GatedRecurrentLogicGateGPTLayer):
        super().__init__(soft)                                   # sets up norm/in_proj/pool/init + candidate self.logic
        self.keep_logic = nn.ModuleList([HardLogicLayer(l) for l in soft.keep_logic])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        B, T, C = x.shape
        token_bits = self._token_bits(x)
        state = self._init_state(B, token_bits, x.device, x.dtype)
        outs = []
        for t in range(T):
            z = torch.cat([token_bits[:, t], state], dim=-1)
            cand = self.logic[0](z)
            for l in self.logic[1:]:
                cand = l(cand)
            keep = self.keep_logic[0](z)
            for l in self.keep_logic[1:]:
                keep = l(keep)
            state = torch.where(keep > 0.5, state, cand)
            outs.append(self._aggregate_state(state, B))
        return x + self.dropout(torch.stack(outs, dim=1))


# ---------------------------------------------------------------------------
# Hybrid layer: keep original (trained) attention sublayer, replace MLP only
# ---------------------------------------------------------------------------

class HybridLogicGateGPTLayer(nn.Module):
    """Drop-in replacement for a nanoGPT Block where the attention sublayer is
    copied FROZEN from the trained baseline and the MLP sublayer is replaced
    by a learnable logic circuit. Now supports ALL aggressive flags so the MLP
    side can be truly aggressive (binary_io + no_in_proj + sum_pool + ...)."""

    def __init__(self, gpt_cfg, layer_idx, original_block,
                 logic_width=None, depth=1, k=4, seed=1000,
                 activation='sigmoid', conn_init_scale=0.02, gate_init_scale=0.02,
                 identity_logic=False, binary_io=False, n_bits=1, sum_pool=False,
                 no_in_proj=False, learn_pool=False, token_shift=0,
                 ln2_mode='fresh', learn_binary_calibration=False, signed_encoding=False,
                 out_gate_mult=1, weighted_pool=False, lut_k=0, grad_checkpoint=False,
                 mlp_guided_init=False, logic_residual=False, gated_lut=False,
                 pre_conv1d=False, pre_conv1d_channels=0, pre_conv1d_kernel=3,
                 pre_conv1d_stride=1, pre_conv1d_groups=1,
                 post_conv1d=False, post_conv1d_channels=0, post_conv1d_kernel=3,
                 post_conv1d_stride=1, post_conv1d_groups=1,
                 binary_encoder='activation', lloyd_ema=0.99, lloyd_min_std=1e-3,
                 interconnect='random', topk_sparse_k=8, topk_sparse_scale=1.0,
                 residual_scale=False, pool_curve=False, ensemble=1):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.logic_residual = logic_residual and depth > 1
        C = gpt_cfg.n_embd
        self.C = C
        self.layer_idx = layer_idx
        self.logic_width = logic_width or C * 4
        self.activation = activation
        self.binary_io = binary_io
        self.n_bits    = n_bits if binary_io else 1
        self.sum_pool  = sum_pool
        self.learn_pool = learn_pool
        self.token_shift = token_shift
        self.signed_encoding = signed_encoding
        self._bps = signed_bits_per_scalar(self.n_bits) if signed_encoding else self.n_bits
        self._taps = list(range(1, token_shift + 1)) if token_shift > 0 else []
        eff_C = C * (len(self._taps) + 1)
        self.eff_C = eff_C
        self.pre_conv1d_enabled = pre_conv1d
        logic_in_C = eff_C
        if pre_conv1d:
            logic_in_C = pre_conv1d_channels or eff_C
            self.pre_conv1d = CausalConv1D(
                eff_C, logic_in_C, kernel_size=pre_conv1d_kernel,
                stride=pre_conv1d_stride, groups=pre_conv1d_groups, name='pre_conv1d')
        self.logic_in_C = logic_in_C
        self.post_conv1d_enabled = post_conv1d
        if post_conv1d:
            post_C = post_conv1d_channels or C
            self.post_conv1d = CausalConv1D(
                C, post_C, kernel_size=post_conv1d_kernel,
                stride=post_conv1d_stride, groups=post_conv1d_groups, name='post_conv1d')
            if post_C != C:
                self.post_conv1d_return = nn.Conv1d(post_C, C, kernel_size=1)
        self.no_in_proj = no_in_proj
        if no_in_proj:
            assert binary_io, "no_in_proj requires binary_io"
            bit_width = logic_in_C * self._bps
        else:
            bit_width = self.logic_width * self._bps
        if identity_logic and out_gate_mult != 1:
            raise ValueError("identity_logic is incompatible with out_gate_mult>1 (non-square logic).")
        self.out_gate_mult = out_gate_mult
        out_width = bit_width * out_gate_mult
        if sum_pool:
            assert binary_io, "sum_pool requires binary_io"
            assert out_width % C == 0, (
                f"sum_pool requires out_width ({out_width}) divisible by n_embd ({C}).")
            self.group_size = out_width // C
            self.weighted_pool = weighted_pool
            if weighted_pool:
                self.pool_w = nn.Parameter(torch.full((C, self.group_size), 2.0 / self.group_size))
                self.pool_b = nn.Parameter(torch.full((C,), -1.0))
            elif learn_pool:
                self.pool_scale = nn.Parameter(torch.full((C,), 2.0 / self.group_size))
                self.pool_shift = nn.Parameter(torch.full((C,), -1.0))
        else:
            self.weighted_pool = weighted_pool

        # FROZEN: attention sublayer copied verbatim from the trained baseline.
        self.ln_1 = copy.deepcopy(original_block.ln_1)
        self.attn = copy.deepcopy(original_block.attn)
        for p in self.ln_1.parameters(): p.requires_grad = False
        for p in self.attn.parameters(): p.requires_grad = False
        self.ln_1.eval(); self.attn.eval()

        # Pre-MLP norm (ln_2): how faithfully we keep the trained pre-FFN signal.
        #   fresh          -> new LayerNorm (legacy default)
        #   copy_trainable -> copy trained ln_2, fine-tune it (faithful + recalibration)
        #   copy_frozen    -> copy trained ln_2, freeze (only the MLP FUNCTION changes)
        if ln2_mode not in ('fresh', 'copy_trainable', 'copy_frozen'):
            raise ValueError(f"unknown hybrid_ln2 mode '{ln2_mode}' "
                             "(expected 'fresh' | 'copy_trainable' | 'copy_frozen').")
        self.ln2_mode = ln2_mode
        if ln2_mode == 'fresh':
            self.ln_2 = nn.LayerNorm(C)
        else:
            self.ln_2 = copy.deepcopy(original_block.ln_2)
            if ln2_mode == 'copy_frozen':
                for p in self.ln_2.parameters(): p.requires_grad = False

        # TRAINABLE: LGN MLP replacement (same shape as LogicGateGPTLayer)
        if not no_in_proj:
            self.in_proj = nn.Linear(logic_in_C, self.logic_width)
        self.lut_k = lut_k
        # Functional init: bias the (otherwise random) candidate connections of the FIRST logic
        # layer toward the input channels the trained MLP weights most heavily (||W1[:,c]||),
        # attacking the dead-candidate lottery. Only layer 0 reads the input bits.
        cand_dist = None
        if mlp_guided_init and hasattr(original_block, 'mlp') and hasattr(original_block.mlp, 'c_fc'):
            imp = original_block.mlp.c_fc.weight.detach().norm(dim=0)        # [C] per-channel
            reps = (logic_in_C + C - 1) // C
            imp = imp.repeat(reps)[:logic_in_C].repeat_interleave(self._bps)
            cand_dist = imp
        # A1: with residual wiring, deeper layers read [input_bits, prev_out] -> 2x in_dim.
        def _in(i):
            return bit_width if i == 0 else (2 * bit_width if self.logic_residual else bit_width)
        self.logic = nn.ModuleList([
            _make_gate_layer(lut_k, _in(i), out_width if i == depth - 1 else bit_width,
                             k=k, seed=seed + layer_idx * 100 + i, gated=gated_lut,
                             conn_init_scale=conn_init_scale, gate_init_scale=gate_init_scale,
                             identity=identity_logic, cand_dist=cand_dist if i == 0 else None,
                             interconnect=interconnect, topk_sparse_k=topk_sparse_k,
                             topk_sparse_scale=topk_sparse_scale)
            for i in range(depth)
        ])
        self.ensemble = ensemble
        if ensemble > 1:
            assert depth == 1, "ensemble requires depth==1"
            self.logic = nn.ModuleList([
                _make_gate_layer(lut_k, bit_width, out_width, k=k,
                                 seed=seed + layer_idx * 100 + 7000 * j, gated=gated_lut,
                                 conn_init_scale=conn_init_scale, gate_init_scale=gate_init_scale,
                                 identity=identity_logic, cand_dist=cand_dist,
                                 interconnect=interconnect, topk_sparse_k=topk_sparse_k,
                                 topk_sparse_scale=topk_sparse_scale)
                for j in range(ensemble)])
            self.ensemble_banks = self.logic
        else:
            self.ensemble_banks = None
        _init_readout_extras(self, residual_scale, pool_curve)
        if not sum_pool:
            self.out_proj = nn.Linear(out_width, C)
        self.learn_binary_calibration = learn_binary_calibration
        if learn_binary_calibration:
            feat_w = logic_in_C if no_in_proj else self.logic_width
            self.cal_scale = nn.Parameter(torch.ones(feat_w))
            self.cal_shift = nn.Parameter(torch.zeros(feat_w))
        self.binary_encoder = binary_encoder
        self.lloyd_ema = lloyd_ema
        self.lloyd_min_std = lloyd_min_std
        self.lloyd_update_stats = True
        if binary_encoder == 'lloydmax':
            if not binary_io:
                raise ValueError("binary_encoder='lloydmax' requires binary_io=True.")
            if signed_encoding or learn_binary_calibration:
                raise ValueError("binary_encoder='lloydmax' is mutually exclusive with "
                                 "signed_encoding and learn_binary_calibration.")
            feat_w = logic_in_C if no_in_proj else self.logic_width
            self.register_buffer('lloyd_base_thresholds',
                                 _lloyd_max_standard_thresholds(self.n_bits))
            self.register_buffer('lloyd_mean', torch.zeros(feat_w))
            self.register_buffer('lloyd_std', torch.ones(feat_w))
        elif binary_encoder != 'activation':
            raise ValueError(f"unknown binary_encoder '{binary_encoder}'")
        self.dropout = nn.Dropout(gpt_cfg.dropout)
        self.use_ste = False

    def _squash(self, h):
        if self.learn_binary_calibration:
            return torch.sigmoid(h * self.cal_scale + self.cal_shift)
        return _apply_activation(h, self.activation)

    def train(self, mode=True):
        super().train(mode)
        self.ln_1.eval(); self.attn.eval()
        return self

    def set_temperature(self, t):
        for l in self.logic: l.set_temperature(t)

    def set_backward_temp(self, t):
        """CAGE: propagate independent backward STE temperature to all sublayers."""
        for l in self.logic: l.set_backward_temp(t)

    @torch.no_grad()
    def commitment(self):
        if not self.logic:
            return 1.0
        return sum(l.commitment() for l in self.logic) / len(self.logic)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _aggregate(self, h, B, T):
        return _pool(self, h, B, T)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.logic_in_C)
        return self.in_proj(normed_btx.reshape(B * T, self.logic_in_C))

    def _apply_pre_conv(self, normed):
        return self.pre_conv1d(normed) if self.pre_conv1d_enabled else normed

    def _apply_post_conv(self, y):
        if not self.post_conv1d_enabled:
            return y
        y = self.post_conv1d(y)
        if hasattr(self, 'post_conv1d_return'):
            y = self.post_conv1d_return(y.transpose(1, 2)).transpose(1, 2)
        return y

    def forward(self, x, hard=False):
        x = x + self.attn(self.ln_1(x))          # FROZEN attention
        B, T, C = x.shape
        normed = self._apply_pre_conv(apply_token_shift(self.ln_2(x), self._taps))
        h = self._apply_in_proj(normed, B, T)
        h = _to_bits(self, h, self.training)
        ste = self.use_ste and not hard
        h = _run_logic_stack(self, h, hard, ste)
        return x + self.dropout(self._apply_post_conv(self._aggregate(h, B, T)))


class HardHybridLogicGateGPTLayer(nn.Module):
    """Hard-snapped HybridLogicGateGPTLayer. Attention stays continuous and identical
    to the original; only the LGN MLP is discretised. Supports all aggressive flags."""

    def __init__(self, soft: HybridLogicGateGPTLayer):
        super().__init__()
        self.layer_idx  = soft.layer_idx
        self.activation = soft.activation
        self.binary_io  = soft.binary_io
        self.n_bits     = soft.n_bits
        self.sum_pool   = soft.sum_pool
        self.learn_pool = soft.learn_pool
        self.no_in_proj = soft.no_in_proj
        self.C          = soft.C
        self.eff_C      = soft.eff_C
        self.logic_in_C = soft.logic_in_C
        self.token_shift = soft.token_shift
        self._taps      = soft._taps
        self.pre_conv1d_enabled = getattr(soft, 'pre_conv1d_enabled', False)
        if self.pre_conv1d_enabled:
            self.pre_conv1d = copy.deepcopy(soft.pre_conv1d)
        self.post_conv1d_enabled = getattr(soft, 'post_conv1d_enabled', False)
        if self.post_conv1d_enabled:
            self.post_conv1d = copy.deepcopy(soft.post_conv1d)
            if hasattr(soft, 'post_conv1d_return'):
                self.post_conv1d_return = copy.deepcopy(soft.post_conv1d_return)
        self.group_size = getattr(soft, 'group_size', None)
        self.weighted_pool = getattr(soft, 'weighted_pool', False)
        if self.sum_pool and self.weighted_pool:
            self.register_buffer('pool_w', soft.pool_w.detach().clone())
            self.register_buffer('pool_b', soft.pool_b.detach().clone())
        elif self.sum_pool and self.learn_pool:
            self.register_buffer('pool_scale', soft.pool_scale.detach().clone())
            self.register_buffer('pool_shift', soft.pool_shift.detach().clone())
        self.ln_1 = copy.deepcopy(soft.ln_1)
        self.attn = copy.deepcopy(soft.attn)
        self.ln_2 = copy.deepcopy(soft.ln_2)
        if not self.no_in_proj:
            self.in_proj = copy.deepcopy(soft.in_proj)
        if not self.sum_pool:
            self.out_proj = copy.deepcopy(soft.out_proj)
        self.dropout  = copy.deepcopy(soft.dropout)
        self.ln2_mode = getattr(soft, 'ln2_mode', 'fresh')
        self.learn_binary_calibration = getattr(soft, 'learn_binary_calibration', False)
        self.signed_encoding = getattr(soft, 'signed_encoding', False)
        if self.learn_binary_calibration:
            self.register_buffer('cal_scale', soft.cal_scale.detach().clone())
            self.register_buffer('cal_shift', soft.cal_shift.detach().clone())
        self.binary_encoder = getattr(soft, 'binary_encoder', 'activation')
        self.lloyd_ema = getattr(soft, 'lloyd_ema', 0.99)
        self.lloyd_min_std = getattr(soft, 'lloyd_min_std', 1e-3)
        self.lloyd_update_stats = False
        if self.binary_encoder == 'lloydmax':
            self.register_buffer('lloyd_base_thresholds', soft.lloyd_base_thresholds.detach().clone())
            self.register_buffer('lloyd_mean', soft.lloyd_mean.detach().clone())
            self.register_buffer('lloyd_std', soft.lloyd_std.detach().clone())
        self.logic_residual = getattr(soft, 'logic_residual', False)
        self.logic    = nn.ModuleList([_make_hard_gate_layer(l) for l in soft.logic])
        self.ensemble_banks = self.logic if getattr(soft, 'ensemble_banks', None) is not None else None
        _copy_readout_extras(self, soft)
        for p in self.parameters():
            p.requires_grad = False

    def _squash(self, h):
        if self.learn_binary_calibration:
            return torch.sigmoid(h * self.cal_scale + self.cal_shift)
        return _apply_activation(h, self.activation)

    def _aggregate(self, h, B, T):
        return _pool(self, h, B, T)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.logic_in_C)
        return self.in_proj(normed_btx.reshape(B * T, self.logic_in_C))

    def _apply_pre_conv(self, normed):
        return self.pre_conv1d(normed) if self.pre_conv1d_enabled else normed

    def _apply_post_conv(self, y):
        if not self.post_conv1d_enabled:
            return y
        y = self.post_conv1d(y)
        if hasattr(self, 'post_conv1d_return'):
            y = self.post_conv1d_return(y.transpose(1, 2)).transpose(1, 2)
        return y

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        B, T, C = x.shape
        normed = self._apply_pre_conv(apply_token_shift(self.ln_2(x), self._taps))
        h = self._apply_in_proj(normed, B, T)
        h = _to_bits(self, h, training=False)
        h = _run_logic_stack_hard(self, h)
        return x + self.dropout(self._apply_post_conv(self._aggregate(h, B, T)))
