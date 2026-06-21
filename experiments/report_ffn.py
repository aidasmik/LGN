"""Task F reporting: one master table for the LGN-as-FFN phase + four plots.
Columns: config | accuracy | loss | perplexity | hard_val | soft_val | soft-hard gap |
relative gate cost | notes.  Plots -> results/figs/ffn/.

Run: python experiments/report_ffn.py
"""
import json, os, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
FIG = 'results/figs/ffn'

# (dir, label, relative gate cost vs base 2-input @1x, notes). cost None = N/A.
CONFIGS = [
    ('__transformer__',          'transformer (FFN intact)',  None, 'ceiling'),
    ('no_mlp',                   'attention-only (no FFN)',   0.0,  'floor; FFN removed'),
    ('hybrid_all_identity',      'identity-LGN (dead gates)', 0.0,  'plumbing only'),
    ('hybrid_all_main',          'base LGN-FFN (n_bits8)',    1.0,  '2-input, 1x gates'),
    ('hybrid_all_weighted',      'weighted_pool',             1.0,  'smart readout; no gain'),
    ('hybrid_all_nbits16',       'n_bits16',                  2.0,  'in16/out16'),
    ('hybrid_all_outmult2',      'out_gate_mult2',            2.0,  'in8/out16'),
    ('hybrid_all_outmult4',      'out_gate_mult4',            4.0,  'in8/out32'),
    ('hybrid_all_adaptive',      'adaptive n_bits (16@L0/9/10/11)', 1.33, 'per-layer precision'),
    ('hybrid_all_lut4_ckpt',     'LUT4 (full, batch32)',      1.0,  '1x LUT4 gates'),
    ('hybrid_all_lut4_b8',       'LUT4 (batch8)',             1.0,  'batch-8 confound'),
    ('hybrid_all_lut6_ckpt',     'LUT6 (full, batch16)',      1.0,  '1x LUT6 gates'),
    ('adaptive_outmult_A',       'adaptive_outmult_A',        None, 'L0=8 L9-11=4 L7-8=2'),
    ('adaptive_outmult_B',       'adaptive_outmult_B',        None, 'L0=8 L9-11=8 L7-8=4'),
    ('hybrid_all_lut4_outmult2', 'LUT4 + out_gate_mult2',     2.0,  'combo'),
    ('hybrid_all_lut4_weighted', 'LUT4 + weighted_pool',      1.0,  'readout x LUT (free)'),
    ('hybrid_all_lut4_ste',      'LUT4 + STE',                1.0,  'close soft-hard gap'),
    ('hybrid_all_lut4_cage',     'LUT4 + CAGE',               1.0,  'close soft-hard gap'),
    ('hybrid_all_lut6_sel',      'LUT6 sensitive / LUT4 easy',None, 'per-layer lut_k'),
    ('hybrid_all_keepL0_outmult4', 'keep L0 MLP + rest outmult4', None, 'L0 stays transformer FFN'),
    ('hybrid_all_keepL0_lut4',   'keep L0 MLP + rest LUT4',    None, 'L0 stays transformer FFN'),
    ('lgn_best_b16',             'LUT4+om4+CAGE (batch16)',    4.0,  'pure-LGN best'),
    ('lgn_best_frozen_control',  'same arch, FROZEN random gates', 4.0, 'honesty control for best'),
    ('lgn_lut6_om4_cage',        'LUT6+om4+CAGE',              4.0,  'arity push'),
    ('lgn_best_om8_k16',         'winner + om8-sens + k16',    None, 'capacity/conn push (RECORD b16)'),
    ('lgn_record',               'om8_k16 + best-hard (b8)',   None, 'record attempt'),
    ('tshift2_opt',              'PURE LGN: token_shift + opt gates', None, 'no attention (vs old tshift 36.22)'),
    ('lgn_om4_kl',               'om4 + KL-distill polish',    4.0,  'KL probe vs om4 46.09'),
    ('lgn_best_train',           'winner + 6k ft + best-hard', 4.0,  'B: training'),
    ('lgn_best_residual',        'winner + residual depth2',   8.0,  'A1: DAG depth'),
    ('lgn_best_gated',           'winner + gated LUT pairs',   8.0,  'A2: gated (batch8)'),
    ('lgn_best_s7',              'winner seed 7',              4.0,  'C: seed spread'),
    ('lgn_best_s42',             'winner seed 42',             4.0,  'C: seed spread'),
    ('lgn_best_max_b16',         '+ functional init + weighted', 4.0, 'pure-LGN max'),
    ('lgn_outmult4_imit800',     'outmult4 + imit800 (training)', 4.0, 'isolate imitation win'),
    ('lgn_lut4_om4_cage',        'LUT4 + outmult4 + CAGE',     4.0,  'pure-LGN combo'),
    ('lgn_adaptive_max_cage',    'adaptive LUT6/outmult + CAGE', None, 'pure-LGN, per-layer max'),
    ('lgn_adaptiveB_cage',       'adaptive outmult + CAGE',    None, 'pure-LGN'),
    ('lgn_outmult4_cage',        'outmult4 + CAGE',            4.0,  'pure-LGN'),
    ('hybrid_all_outmult4_polish_lm', 'outmult4 + joint polish (LM)', 4.0, 'Task D'),
    ('hybrid_all_outmult4_polish_kl', 'outmult4 + joint polish (KL)', 4.0, 'Task D'),
]


def _metrics(d):
    p = f'{OUT}/{d}/metrics.json'
    return json.load(open(p)) if os.path.exists(p) else None


def _scale_final(d):
    p = f'{OUT}/{d}/scale_greedy.json'
    if not os.path.exists(p):
        return None, None
    rows = json.load(open(p))
    return rows[-1].get('soft_val'), rows[-1].get('hard_val')


def table():
    print(f'\n{"config":34} | {"acc%":>6} | {"loss":>6} | {"ppl":>6} | '
          f'{"hard_v":>7} | {"soft_v":>7} | {"gap":>6} | {"gates":>6} | notes')
    print('-' * 130)
    tf = None
    rows_for_plot = []
    for d, label, cost, notes in CONFIGS:
        if d == '__transformer__':
            m = _metrics('hybrid_all_main')           # any run carries the transformer metrics
            if m:
                tf = m['transformer']
                print(f'{label:34} | {tf["accuracy"]*100:>6.2f} | {tf["loss"]:>6.3f} | '
                      f'{tf["perplexity"]:>6.2f} | {"-":>7} | {"-":>7} | {"-":>6} | {"-":>6} | {notes}')
            continue
        m = _metrics(d)
        if not m:
            continue
        # the non-transformer block: 'lgn_hard' for scaling runs, 'attention_only_no_mlp' for no_mlp
        g = m.get('lgn_hard') or m.get('attention_only_no_mlp')
        if not g:
            continue
        sv, hv = _scale_final(d)
        gap = (hv - sv) if (sv is not None and hv is not None) else None
        print(f'{label:34} | {g["accuracy"]*100:>6.2f} | {g["loss"]:>6.3f} | {g["perplexity"]:>6.2f} | '
              f'{("%.4f"%hv) if hv else "-":>7} | {("%.4f"%sv) if sv else "-":>7} | '
              f'{("%+.4f"%gap) if gap is not None else "-":>6} | '
              f'{("%.2fx"%cost) if cost is not None else "-":>6} | {notes}')
        if cost is not None:
            rows_for_plot.append((label, cost, g['accuracy'] * 100, gap))
    return tf, rows_for_plot


def plots(tf, rows):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except Exception as e:
        print(f'(matplotlib unavailable: {e})'); return
    os.makedirs(FIG, exist_ok=True)

    # 1. accuracy vs relative gate cost
    if rows:
        plt.figure(figsize=(8, 5))
        for label, cost, acc, _ in rows:
            plt.scatter(cost, acc); plt.annotate(label, (cost, acc), fontsize=7,
                                                 xytext=(4, 2), textcoords='offset points')
        if tf:
            plt.axhline(tf['accuracy'] * 100, ls='--', color='gray', label='transformer')
        plt.xlabel('relative gate cost (x base)'); plt.ylabel('hard accuracy %')
        plt.title('LGN-FFN: accuracy vs gate cost'); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{FIG}/accuracy_vs_gatecost.png', dpi=120); plt.close()
        print(f'  saved {FIG}/accuracy_vs_gatecost.png')

    # 2. L0 degradation by method
    sl = 'results/report/screen_l0'
    if os.path.isdir(sl):
        items = []
        for nm in sorted(os.listdir(sl)):
            p = f'{sl}/{nm}/heatmap.json'
            if os.path.exists(p):
                items.append((nm, json.load(open(p))[0]['hard_degradation']))
        if items:
            items.sort(key=lambda x: x[1])
            plt.figure(figsize=(10, 5))
            plt.bar([x[0] for x in items], [x[1] for x in items], color='#c0504d')
            plt.ylabel('L0 hard degradation'); plt.xticks(rotation=60, ha='right', fontsize=7)
            plt.title('L0 FFN-replacement degradation by method'); plt.tight_layout()
            plt.savefig(f'{FIG}/l0_degradation.png', dpi=120); plt.close()
            print(f'  saved {FIG}/l0_degradation.png')

    # 3. cumulative scaling for the best COMPLETE config (has metrics.json; highest accuracy)
    best, best_acc = None, -1
    for d, *_ in CONFIGS:
        if d == '__transformer__':
            continue
        m = _metrics(d)
        if m and m.get('lgn_hard') and m['lgn_hard']['accuracy'] > best_acc \
                and os.path.exists(f'{OUT}/{d}/scale_greedy.json'):
            best, best_acc = d, m['lgn_hard']['accuracy']
    if best:
        sc = json.load(open(f'{OUT}/{best}/scale_greedy.json'))
        plt.figure(figsize=(8, 5))
        plt.plot([r['n_replaced'] for r in sc], [r['hard_degradation'] for r in sc], 'o-')
        plt.xlabel('# FFNs -> LGN'); plt.ylabel('hard val-loss degradation')
        plt.title(f'Cumulative scaling (best complete: {best}, {best_acc*100:.1f}%)'); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{FIG}/cumulative_best.png', dpi=120); plt.close()
        print(f'  saved {FIG}/cumulative_best.png  (best={best})')

    # 4. soft-hard gap by config
    gaps = [(l, gp) for l, c, a, gp in rows if gp is not None]
    if gaps:
        plt.figure(figsize=(9, 5))
        plt.bar([x[0] for x in gaps], [x[1] for x in gaps], color='#4f81bd')
        plt.ylabel('hard_val - soft_val'); plt.xticks(rotation=60, ha='right', fontsize=7)
        plt.title('Soft-hard gap by config'); plt.tight_layout()
        plt.savefig(f'{FIG}/soft_hard_gap.png', dpi=120); plt.close()
        print(f'  saved {FIG}/soft_hard_gap.png')


if __name__ == '__main__':
    tf, rows = table()
    print('\n=== plots ===')
    plots(tf, rows)
