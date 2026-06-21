import argparse
import json
import os
import torch

from lgn import ExperimentConfig, make_gpt
from pipeline import WikiText2, train_baseline, estimate_loss, run_heatmap, run_scaling


def _common_args(p):
    # model
    p.add_argument('--n_embd',          type=int,   default=128)
    p.add_argument('--n_layer',         type=int,   default=12)
    p.add_argument('--n_head',          type=int,   default=4)
    p.add_argument('--dropout',         type=float, default=0.0)
    # logic architecture
    p.add_argument('--width_mult',      type=int,   default=2)
    p.add_argument('--depth',           type=int,   default=1)
    p.add_argument('--k',               type=int,   default=4)
    p.add_argument('--activation',      type=str,   default='sigmoid',
                   choices=['sigmoid', 'tanh', 'relu', 'hardsigmoid', 'none'])
    p.add_argument('--conn_init_scale', type=float, default=0.02)
    p.add_argument('--gate_init_scale', type=float, default=0.02)
    p.add_argument('--hybrid_layers',   type=int,   nargs='*', default=[],
                   help='layer indices that keep original attention; logic replaces MLP only')
    p.add_argument('--hybrid_all',      action='store_true',
                   help='shortcut: keep frozen pretrained attention in EVERY layer, replace only the '
                        'FFN/MLP with LGN (expands to --hybrid_layers 0..n_layer-1).')
    p.add_argument('--hybrid_ln2',      type=str, default='fresh',
                   choices=['fresh', 'copy_trainable', 'copy_frozen'],
                   help="hybrid pre-MLP norm: fresh (legacy) | copy_trainable (copy trained ln_2, "
                        "tune it) | copy_frozen (copy + freeze; only the MLP function changes).")
    p.add_argument('--learn_binary_calibration', action='store_true',
                   help='per-channel learned sigmoid(scale*x+bias) before thermometer encoding '
                        '(maps ~Gaussian post-norm activations into bits; init = plain sigmoid).')
    p.add_argument('--precision_layers', type=int, nargs='*', default=[],
                   help='layer indices that use --high_n_bits instead of --n_bits '
                        '(spend precision only on quantization-sensitive layers, e.g. 0 9 10 11).')
    p.add_argument('--high_n_bits', type=int, default=16,
                   help='n_bits for layers in --precision_layers (default 16).')
    p.add_argument('--out_gate_mult', type=int, default=1,
                   help='widen the final logic layer by this factor -> finer sum_pool readout '
                        '(more output levels). Decouples OUTPUT resolution from input n_bits.')
    p.add_argument('--out_gate_mult_layers', type=str, nargs='*', default=[],
                   help='per-layer out_gate_mult as LAYER:VALUE pairs, e.g. 0:8 11:4 10:4. '
                        'Unlisted layers use the global --out_gate_mult.')
    p.add_argument('--lut_k_layers', type=str, nargs='*', default=[],
                   help='per-layer lut_k as LAYER:VALUE pairs, e.g. 0:6 11:6 8:4. '
                        'Unlisted layers use the global --lut_k.')
    p.add_argument('--lut_k', type=int, default=0,
                   help='gate primitive arity: 0/2 = 2-input gate (LUT2); >=3 = K-input LUT gate '
                        '(learned 2^K truth table, hard-snaps to one FPGA LUT-K). More expressive '
                        'per gate; tests whether the primitive beats more 2-input gates at equal count.')
    p.add_argument('--logic_residual', action='store_true',
                   help='A1: DAG/residual depth -- gate layers past the first read BOTH the '
                        'input bits and the previous layer output (needs --depth >= 2).')
    p.add_argument('--gated_lut', action='store_true',
                   help='A2: gated LUT pairs -- each output = LUT_a AND LUT_b (2x LUT cost, '
                        '2K-input function class). Requires --lut_k >= 2.')
    p.add_argument('--ft_keep_best_hard', action='store_true',
                   help='B4: select by hard validation -- keep the best-hard checkpoint seen '
                        'during fine-tune (evaluated every 500 steps) instead of the final step.')
    p.add_argument('--freeze_logic', action='store_true',
                   help='HONESTY CONTROL: freeze logic params at random init; train only the '
                        'plumbing (ln_2, pool). Shape-compatible with any out_gate_mult/lut_k.')
    p.add_argument('--mlp_guided_init', action='store_true',
                   help='functional init: seed the first logic layer\'s candidate connections from '
                        'the trained MLP\'s input importance (||W1[:,c]||) instead of random.')
    p.add_argument('--grad_checkpoint', action='store_true',
                   help='gradient-checkpoint the logic stack (recompute in backward) to cut memory '
                        'for large LUT-K gates -> run full batch_size instead of a reduced one.')
    p.add_argument('--weighted_pool', action='store_true',
                   help='learned per-channel per-bit readout weights (up to 2^group_size output '
                        'levels at NO extra gate cost; block-diagonal, not a dense out_proj). '
                        'Init == learn_pool, so it can only help.')
    p.add_argument('--signed_encoding', action='store_true',
                   help='signed real->binary encoding: sign bit + pos-magnitude thermometer + '
                        'neg-magnitude thermometer (2*n_bits+1 bits/scalar). Keeps sign+magnitude '
                        'of zero-centered post-norm activations. Mutually exclusive with calibration.')
    p.add_argument('--identity_logic',  action='store_true',
                   help='ablation: LearnedLogicLayer returns input as output')

    # ===================================================================
    # AGGRESSIVE SETUP IS THE DEFAULT (no trained Linear around the LGN).
    # binary_io + no_in_proj + sum_pool are ON by default — this is the
    # setup where the logic gates actually do the work.
    #   * --classic            -> original Linear-sandwich setup (all OFF)
    #   * --no-binary_io / --no-sum_pool / --no-no_in_proj -> toggle one
    # ===================================================================
    p.add_argument('--classic',         action='store_true',
                   help='ORIGINAL Linear-sandwich setup (trained in_proj + out_proj around LGN). '
                        'By DEFAULT the aggressive setup is used (no Linear; LGN does the work).')
    p.add_argument('--binary_io',       action=argparse.BooleanOptionalAction, default=True,
                   help='binarize LGN inputs to {0,1} via STE (default: ON)')
    p.add_argument('--n_bits',          type=int, default=8,
                   help='bits per scalar in binarization (1 = plain threshold, >1 = thermometer; aggressive uses 8)')
    p.add_argument('--sum_pool',        action=argparse.BooleanOptionalAction, default=True,
                   help='replace out_proj with fixed group-sum aggregation (default: ON)')
    p.add_argument('--no_in_proj',      action=argparse.BooleanOptionalAction, default=True,
                   help='remove the trained Linear before LGN; LGN reads the embedding directly (default: ON)')
    # ===================================================================

    p.add_argument('--learn_pool',      action='store_true',
                   help='learnable per-channel affine on sum_pool output (cheap residual-stat matching)')
    p.add_argument('--token_shift',     type=int, default=0,
                   help='Fixed causal token shift K: each position sees [x[t-K]..x[t]] (cross-token via local context). The one mechanism (with hybrid/selective) that raises accuracy.')
    p.add_argument('--pre_conv1d', action='store_true',
                   help='add causal Conv1D after norm/token_shift and before LGN binarization')
    p.add_argument('--pre_conv1d_channels', type=int, default=0,
                   help='pre-LGN Conv1D output channels (0 = token_shift width)')
    p.add_argument('--pre_conv1d_kernel', type=int, default=3)
    p.add_argument('--pre_conv1d_stride', type=int, default=1,
                   help='pre-LGN Conv1D temporal stride; output is causally restored to original T')
    p.add_argument('--pre_conv1d_groups', type=int, default=1)
    p.add_argument('--post_conv1d', action='store_true',
                   help='add causal Conv1D after LGN aggregation and before residual')
    p.add_argument('--post_conv1d_channels', type=int, default=0,
                   help='post-LGN Conv1D channels (0 = n_embd; != n_embd uses 1x1 return projection)')
    p.add_argument('--post_conv1d_kernel', type=int, default=3)
    p.add_argument('--post_conv1d_stride', type=int, default=1,
                   help='post-LGN Conv1D temporal stride; output is causally restored to original T')
    p.add_argument('--post_conv1d_groups', type=int, default=1)
    p.add_argument('--binary_encoder', type=str, default='activation',
                   choices=['activation', 'lloydmax'],
                   help='real->binary encoder: historical activation thermometer or LloydMax thresholds')
    p.add_argument('--lloyd_ema', type=float, default=0.99,
                   help='EMA decay for LloydMax activation mean/std')
    p.add_argument('--lloyd_min_std', type=float, default=1e-3,
                   help='minimum std for LloydMax activation thresholds')
    p.add_argument('--interconnect', type=str, default='random',
                   choices=['random', 'topk_block_sparse'],
                   help='gate input interconnect: random candidate lottery or block-sparse top-k')
    p.add_argument('--topk_sparse_k', type=int, default=8,
                   help='block size for --interconnect topk_block_sparse')
    p.add_argument('--topk_sparse_scale', type=float, default=1.0,
                   help='softmax scale for block-sparse top-k interconnect')
    p.add_argument('--pool_curve', action='store_true',
                   help='#2 learned per-channel nonlinear readout (count->value curve); needs sum_pool')
    p.add_argument('--residual_scale', action='store_true',
                   help='#4 per-channel learned alpha on the LGN contribution (x + alpha*LGN)')
    p.add_argument('--ensemble', type=int, default=1,
                   help='#5 within-layer gate banks averaged (variance reduction); needs depth==1')
    # RDDLGN-inspired recurrent/stateful LGN (alternative cross-token mechanism)
    p.add_argument('--recurrent',       action='store_true',
                   help='use a recurrent/stateful LGN layer (state_t = Logic([token_bits_t, state_{t-1}])). Causal; NOT full RDDLGN encoder/decoder.')
    p.add_argument('--recurrent_layers', type=int, nargs='*', default=[],
                   help='layer indices to make recurrent (empty = all replaced layers).')
    p.add_argument('--recurrent_state_width', type=int, default=None,
                   help='hidden state width (default = token bit width); must divide n_embd.')
    p.add_argument('--recurrent_depth', type=int, default=1,
                   help='number of logic sublayers in the recurrent update.')
    p.add_argument('--recurrent_state_init', type=str, default='zero',
                   choices=['zero', 'learned', 'residual'],
                   help='initial hidden state: zero | learned param | residual (from first token bits).')
    p.add_argument('--recurrent_gated', action='store_true',
                   help='flip-flop/latch-inspired gated update (requires --recurrent): '
                        'state = keep*state + (1-keep)*candidate, where keep is a learned LOGIC gate.')
    # training
    p.add_argument('--baseline_steps',  type=int,   default=5_000)
    p.add_argument('--imitation_steps', type=int,   default=1_000)
    p.add_argument('--finetune_steps',  type=int,   default=1_000)
    p.add_argument('--eval_iters',      type=int,   default=30,
                   help='val batches used in estimate_loss (lower = faster, noisier)')
    p.add_argument('--batch_size',      type=int,   default=32,
                   help='training batch size (lower it for memory-heavy gates, e.g. LUT-K).')
    p.add_argument('--per_layer_anneal', action='store_true',
                   help='scale imitation steps by layer difficulty')
    p.add_argument('--ft_log_sharpness', action='store_true', default=True,
                   help='print per-layer sharpness during fine-tuning')
    p.add_argument('--ft_eval_hard',    action='store_true',
                   help='evaluate hard-snapped model periodically during fine-tuning')
    p.add_argument('--imit_loss',       type=str, default='mse', choices=['mse', 'kl'],
                   help='imitation loss: mse (match activations) or kl (match output distribution)')
    p.add_argument('--ste',             action='store_true',
                   help='straight-through estimator during fine-tuning (forward=hard, backward=soft)')
    # CAGE — Align Forward Adapt Backward (arxiv 2603.14157, 2026)
    p.add_argument('--cage',            action='store_true',
                   help='CAGE: hard forward (argmax) + adaptive backward temperature based on commitment confidence. Closes the discretization gap by construction.')
    p.add_argument('--cage_tau_max',    type=float, default=3.0,
                   help='CAGE: max backward temperature (early training, exploratory). Default 3.0.')
    p.add_argument('--cage_tau_min',    type=float, default=0.5,
                   help='CAGE: min backward temperature (late training, sharp). Default 0.5.')
    p.add_argument('--cage_ema',        type=float, default=0.99,
                   help='CAGE: EMA decay for commitment confidence (higher = slower adaptation). Default 0.99.')
    p.add_argument('--anneal_in_finetune', action='store_true',
                   help='direct training: anneal temperature during fine-tune on LM loss instead of imitation')
    p.add_argument('--ft_imit_weight',  type=float, default=0.0,
                   help='curriculum: decaying MSE-to-MLP weight blended into fine-tune (0 = pure LM)')
    # Training dynamics knobs (defaults match TrainConfig, so omitting them = current behavior).
    p.add_argument('--temp_start',      type=float, default=2.0,
                   help='annealing temperature start (softer = more exploration). Default 2.0.')
    p.add_argument('--temp_end',        type=float, default=0.1,
                   help='annealing temperature end (sharper = smaller soft-hard gap). Default 0.1.')
    p.add_argument('--ent_conn',        type=float, default=0.001,
                   help='imitation: connection-softmax entropy weight (commitment pressure).')
    p.add_argument('--ent_gate',        type=float, default=0.02,
                   help='imitation: gate/LUT entropy weight (raise to force commitment / shrink hard gap).')
    p.add_argument('--ft_ent_conn',     type=float, default=0.0005,
                   help='fine-tune: connection entropy weight.')
    p.add_argument('--ft_ent_gate',     type=float, default=0.01,
                   help='fine-tune: gate/LUT entropy weight.')
    p.add_argument('--layers',          type=int, nargs='*', default=None,
                   help='restrict heatmap to these layer indices (default: all)')
    p.add_argument('--seed',            type=int, default=1337,
                   help='random seed (for variance / repeatability experiments)')
    # NOTE: --freeze_unreplaced removed (was a no-op: the base is ALWAYS frozen by
    # _make_logic_model / _add_logic_layer; only LGN layer params get requires_grad=True).
    p.add_argument('--joint_polish_steps', type=int, default=0,
                   help='scaling: final joint fine-tune of ALL LGN layers together (0 = off)')
    p.add_argument('--joint_polish_kl_weight', type=float, default=0.0,
                   help='joint polish: system-level KL distillation to original transformer logits (0 = LM only)')
    # misc
    p.add_argument('--results_dir',     type=str,   default='results')
    p.add_argument('--checkpoint',      type=str,   default=None)


def _parse_layer_spec(items, flag):
    """Parse ['0:8','11:4'] -> {0:8, 11:4} with clear errors for malformed specs."""
    out = {}
    for it in items or []:
        if it.count(':') != 1:
            raise ValueError(f"--{flag}: '{it}' must be LAYER:VALUE (one colon).")
        ks, vs = it.split(':')
        try:
            k, v = int(ks), int(vs)
        except ValueError:
            raise ValueError(f"--{flag}: '{it}' must be int:int (e.g. 0:8).")
        if k < 0:
            raise ValueError(f"--{flag}: layer index '{k}' must be >= 0.")
        if v < 1:
            raise ValueError(f"--{flag}: value for layer {k} must be >= 1 (got {v}).")
        out[k] = v
    return out


def _build_cfg(args):
    cfg = ExperimentConfig()
    # model
    cfg.model.n_embd    = args.n_embd
    cfg.model.n_layer   = args.n_layer
    cfg.model.n_head    = args.n_head
    cfg.model.dropout   = args.dropout
    # logic architecture
    cfg.logic.width_mult      = args.width_mult
    cfg.logic.depth           = args.depth
    cfg.logic.k               = args.k
    cfg.logic.activation      = args.activation
    cfg.logic.conn_init_scale = args.conn_init_scale
    cfg.logic.gate_init_scale = args.gate_init_scale
    # --hybrid_all expands to all layers. Reject mixing with explicit --hybrid_layers (ambiguous).
    if args.hybrid_all:
        if args.hybrid_layers:
            raise ValueError("pass either --hybrid_all OR --hybrid_layers, not both.")
        cfg.logic.hybrid_layers = list(range(args.n_layer))
    else:
        cfg.logic.hybrid_layers = args.hybrid_layers
    cfg.logic.hybrid_ln2      = args.hybrid_ln2
    cfg.logic.learn_binary_calibration = args.learn_binary_calibration
    cfg.logic.signed_encoding = args.signed_encoding
    cfg.logic.precision_layers = args.precision_layers
    cfg.logic.high_n_bits = args.high_n_bits
    cfg.logic.out_gate_mult = args.out_gate_mult
    cfg.logic.out_gate_mult_layers = _parse_layer_spec(args.out_gate_mult_layers, 'out_gate_mult_layers')
    cfg.logic.lut_k_layers = _parse_layer_spec(args.lut_k_layers, 'lut_k_layers')
    cfg.logic.weighted_pool = args.weighted_pool
    cfg.logic.lut_k = args.lut_k
    cfg.logic.mlp_guided_init = args.mlp_guided_init
    cfg.logic.freeze_logic = args.freeze_logic
    cfg.logic.logic_residual = args.logic_residual
    cfg.logic.gated_lut = args.gated_lut
    cfg.train.ft_keep_best_hard = args.ft_keep_best_hard
    if args.logic_residual and args.depth < 2:
        raise ValueError("--logic_residual needs --depth >= 2 (it wires deeper gate layers).")
    if args.gated_lut and args.lut_k < 2:
        raise ValueError("--gated_lut requires --lut_k >= 2.")
    cfg.logic.grad_checkpoint = args.grad_checkpoint
    if args.signed_encoding and args.learn_binary_calibration:
        raise ValueError("--signed_encoding and --learn_binary_calibration are mutually exclusive "
                         "(signed encoding does its own real->binary mapping).")
    if args.signed_encoding and not args.binary_io and not args.classic:
        raise ValueError("--signed_encoding requires binary_io (it emits bits). Drop --no-binary_io.")
    cfg.logic.identity_logic  = args.identity_logic
    # Aggressive setup is the default; --classic flips back to the Linear-sandwich setup.
    if args.classic:
        cfg.logic.binary_io  = False
        cfg.logic.no_in_proj = False
        cfg.logic.sum_pool   = False
        cfg.logic.n_bits     = 1
    else:
        cfg.logic.binary_io  = args.binary_io
        cfg.logic.no_in_proj = args.no_in_proj
        cfg.logic.sum_pool   = args.sum_pool
        cfg.logic.n_bits     = args.n_bits
    cfg.logic.learn_pool      = args.learn_pool
    # width_mult only sizes the trained in_proj Linear (no_in_proj=False). Under the aggressive
    # default (no_in_proj=True) the gate count is eff_C*bits, so width_mult is a NO-OP there.
    if cfg.logic.no_in_proj and args.width_mult != 2:
        print(f"[warn] --width_mult {args.width_mult} is a NO-OP under no_in_proj (aggressive); "
              f"gate count = eff_C*bits. Use --depth / --n_bits for real capacity, or --no-no_in_proj.")
    cfg.logic.token_shift     = args.token_shift
    cfg.logic.pre_conv1d = args.pre_conv1d
    cfg.logic.pre_conv1d_channels = args.pre_conv1d_channels
    cfg.logic.pre_conv1d_kernel = args.pre_conv1d_kernel
    cfg.logic.pre_conv1d_stride = args.pre_conv1d_stride
    cfg.logic.pre_conv1d_groups = args.pre_conv1d_groups
    cfg.logic.post_conv1d = args.post_conv1d
    cfg.logic.post_conv1d_channels = args.post_conv1d_channels
    cfg.logic.post_conv1d_kernel = args.post_conv1d_kernel
    cfg.logic.post_conv1d_stride = args.post_conv1d_stride
    cfg.logic.post_conv1d_groups = args.post_conv1d_groups
    cfg.logic.binary_encoder = args.binary_encoder
    cfg.logic.lloyd_ema = args.lloyd_ema
    cfg.logic.lloyd_min_std = args.lloyd_min_std
    cfg.logic.interconnect = args.interconnect
    cfg.logic.topk_sparse_k = args.topk_sparse_k
    cfg.logic.topk_sparse_scale = args.topk_sparse_scale
    cfg.logic.pool_curve = args.pool_curve
    cfg.logic.residual_scale = args.residual_scale
    cfg.logic.ensemble = args.ensemble
    if args.pool_curve and not cfg.logic.sum_pool:
        raise ValueError("--pool_curve requires sum_pool (it is a learned readout over the bit count).")
    if args.ensemble < 1:
        raise ValueError("--ensemble must be >= 1.")
    if args.ensemble > 1 and args.depth > 1:
        raise ValueError("--ensemble requires depth==1 (banks are parallel, not chained).")
    if args.binary_encoder == 'lloydmax':
        if args.signed_encoding or args.learn_binary_calibration:
            raise ValueError("--binary_encoder lloydmax is mutually exclusive with "
                             "--signed_encoding and --learn_binary_calibration.")
        if not cfg.logic.binary_io:
            raise ValueError("--binary_encoder lloydmax requires binary_io=True.")
    if args.interconnect == 'topk_block_sparse' and args.topk_sparse_k < 1:
        raise ValueError("--topk_sparse_k must be >= 1.")
    cfg.logic.recurrent             = args.recurrent
    cfg.logic.recurrent_layers      = args.recurrent_layers
    cfg.logic.recurrent_state_width = args.recurrent_state_width
    cfg.logic.recurrent_depth       = args.recurrent_depth
    cfg.logic.recurrent_state_init  = args.recurrent_state_init
    cfg.logic.recurrent_gated       = args.recurrent_gated
    if args.recurrent_gated and not args.recurrent:
        raise ValueError("--recurrent_gated requires --recurrent (the gated update is a "
                         "variant of the recurrent layer). Pass --recurrent --recurrent_gated.")
    # training
    cfg.train.baseline_steps   = args.baseline_steps
    cfg.train.imitation_steps  = args.imitation_steps
    cfg.train.finetune_steps   = args.finetune_steps
    cfg.train.eval_iters       = args.eval_iters
    cfg.train.batch_size       = args.batch_size
    cfg.train.per_layer_anneal = args.per_layer_anneal
    cfg.train.ft_log_sharpness = args.ft_log_sharpness
    cfg.train.ft_eval_hard     = args.ft_eval_hard
    cfg.train.imit_loss        = args.imit_loss
    cfg.train.ste              = args.ste
    cfg.train.cage             = args.cage
    cfg.train.cage_tau_max     = args.cage_tau_max
    cfg.train.cage_tau_min     = args.cage_tau_min
    cfg.train.cage_ema         = args.cage_ema
    cfg.train.anneal_in_finetune = args.anneal_in_finetune
    cfg.train.ft_imit_weight   = args.ft_imit_weight
    cfg.train.temp_start       = args.temp_start
    cfg.train.temp_end         = args.temp_end
    cfg.train.ent_conn         = args.ent_conn
    cfg.train.ent_gate         = args.ent_gate
    cfg.train.ft_ent_conn      = args.ft_ent_conn
    cfg.train.ft_ent_gate      = args.ft_ent_gate
    cfg.train.joint_polish_steps = args.joint_polish_steps
    cfg.train.joint_polish_kl_weight = args.joint_polish_kl_weight
    cfg.results_dir = args.results_dir
    os.makedirs(cfg.results_dir, exist_ok=True)
    return cfg


def _load_or_train(cfg, model, data, args):
    ckpt = args.checkpoint or os.path.join(cfg.results_dir, 'baseline.pt')
    if os.path.exists(ckpt):
        print(f'Loading baseline from {ckpt}')
        device = next(model.parameters()).device
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)  # older torch without weights_only
        # Validate architecture compatibility BEFORE load_state_dict, so a mismatch
        # (changed n_layer / n_embd / block_size) fails early with a clear message.
        want = model.state_dict()
        bad = [k for k in want if k in state and tuple(state[k].shape) != tuple(want[k].shape)]
        missing = [k for k in want if k not in state]
        if bad or missing:
            raise RuntimeError(
                f"Checkpoint '{ckpt}' is incompatible with the current model config "
                f"(n_layer={cfg.model.n_layer}, n_embd={cfg.model.n_embd}, "
                f"block_size={cfg.data.block_size}).\n"
                f"  {len(bad)} shape-mismatched tensors (e.g. {bad[:3]})\n"
                f"  {len(missing)} missing keys (e.g. {missing[:3]})")
        model.load_state_dict(state)
    else:
        train_baseline(model, data, cfg.train)
        torch.save(model.state_dict(), os.path.join(cfg.results_dir, 'baseline.pt'))


def cmd_heatmap(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    cfg = _build_cfg(args)
    data = WikiText2(cfg.data, device)
    model, gpt_cfg = make_gpt(cfg.model, cfg.data, device)
    print(f'Model: {cfg.model.n_layer}L x {cfg.model.n_embd}d  ({sum(p.numel() for p in model.parameters()):,} params)')
    _load_or_train(cfg, model, data, args)
    save_path = os.path.join(cfg.results_dir, 'heatmap.json')
    run_heatmap(model, gpt_cfg, data, cfg, save_path=save_path, layers=args.layers)
    print(f'\nSaved -> {save_path}')


def cmd_scale(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    cfg = _build_cfg(args)
    data = WikiText2(cfg.data, device)
    model, gpt_cfg = make_gpt(cfg.model, cfg.data, device)
    _load_or_train(cfg, model, data, args)
    heatmap_results = None
    if args.strategy == 'greedy':
        with open(args.heatmap) as f:
            heatmap_results = json.load(f)
    save_path = os.path.join(cfg.results_dir, f'scale_{args.strategy}.json')
    run_scaling(model, gpt_cfg, data, cfg,
                strategy=args.strategy, heatmap_results=heatmap_results, save_path=save_path,
                protected_layers=args.protected_layers)
    print(f'\nSaved -> {save_path}')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_h = sub.add_parser('heatmap')
    _common_args(p_h)

    p_s = sub.add_parser('scale')
    _common_args(p_s)
    p_s.add_argument('--strategy',          type=str, default='greedy',
                     choices=['greedy', 'uniform'],
                     help='greedy=easy-first by per-layer difficulty (heatmap); uniform=every n//8th layer.')
    p_s.add_argument('--heatmap',           type=str, default='results/heatmap.json')
    p_s.add_argument('--protected_layers',  type=int, nargs='*', default=[],
                     help='layer indices to never replace (e.g. --protected_layers 0 11)')

    args = parser.parse_args()
    {'heatmap': cmd_heatmap, 'scale': cmd_scale}[args.cmd](args)


if __name__ == '__main__':
    main()
