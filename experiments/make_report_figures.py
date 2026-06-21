"""Generate all figures for the LGN-as-FFN report. Reads real numbers from results/**/metrics.json
and results/new_lgn_inputs/**/heatmap.json so the figures stay reproducible. Saves PNGs to
results/report/figures/.

Metric: next-byte top-1 accuracy on the fixed (seed-1234) byte-level WikiText-2 val set; LGN
always the HARD (discretized) model. Screens use L0 hard_degradation (loss increase, lower=better)."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = 'results/figs/report'   # under results/figs/ so figures are tracked by git
os.makedirs(FIG, exist_ok=True)

# consistent palette
C_TF   = '#2c3e50'   # transformer ceiling
C_GOOD = '#27ae60'   # real lever / win
C_LGN  = '#2980b9'   # LGN result
C_NEU  = '#95a5a6'   # neutral / no gain
C_BAD  = '#c0392b'   # worse
C_REF  = '#7f8c8d'   # reference line


def lgn_acc(path):
    m = json.load(open(path))
    g = m.get('lgn_hard') or m.get('attention_only_no_mlp')
    return g['accuracy'] * 100, g['loss']


def tf_acc(path):
    return json.load(open(path))['transformer']['accuracy'] * 100


def hdeg(name, sub=''):
    p = os.path.join('results/new_lgn_inputs', sub, name, 'heatmap.json')
    return json.load(open(p))[0]['hard_degradation']


R = 'results/report'

# --------------------------------------------------------------------------- #
# Figure 1 — headline: where LGN-as-FFN lands vs the transformer ceiling
# --------------------------------------------------------------------------- #
def fig_headline():
    tf = tf_acc(f'{R}/lgn_best_om8_k16/metrics.json')                  # 54.87
    att = lgn_acc(f'{R}/lgn_best_om8_k16/metrics.json')[0]             # 48.18
    pure_tf = tf_acc(f'{R}/tshift2_opt/metrics.json')                  # 54.67
    pure = lgn_acc(f'{R}/tshift2_opt/metrics.json')[0]                 # 43.54
    ident = lgn_acc(f'{R}/hybrid_all_identity/metrics.json')[0]        # 26.46
    nomlp = lgn_acc(f'{R}/no_mlp/metrics.json')[0]                     # 5.47

    labels = ['Transformer\n(FFN intact)', 'Attention-LGN\n(best, om8+k16)',
              'Pure LGN\n(token-shift, no attn)', 'Identity gates\n(plumbing only)',
              'Attention only\n(no FFN)']
    vals = [tf, att, pure, ident, nomlp]
    cols = [C_TF, C_LGN, C_LGN, C_NEU, C_NEU]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=cols, edgecolor='white', width=0.66)
    ax.axhline(tf, color=C_TF, ls='--', lw=1, alpha=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f'{v:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('Next-byte top-1 accuracy (%)')
    ax.set_title('LGN as an FFN replacement', fontsize=13)
    ax.set_ylim(0, 60)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig1_headline.png', dpi=150); plt.close(fig)
    print('fig1_headline.png', [round(v, 2) for v in vals])


# --------------------------------------------------------------------------- #
# Figure 2 — what moves the needle: single-technique ablations (attention-LGN)
# --------------------------------------------------------------------------- #
def fig_levers():
    base = lgn_acc(f'{R}/hybrid_all_main/metrics.json')[0]   # 35.35
    best = lgn_acc(f'{R}/lgn_best_om8_k16/metrics.json')[0]  # 48.18
    tf = tf_acc(f'{R}/lgn_best_om8_k16/metrics.json')
    items = [  # (label, dir, is_real_lever)
        ('out_gate_mult4 (output capacity)', 'hybrid_all_outmult4', True),
        ('out_gate_mult2',                   'hybrid_all_outmult2', True),
        ('LUT4 + CAGE',                      'hybrid_all_lut4_cage', True),
        ('LUT4 + STE',                       'hybrid_all_lut4_ste', True),
        ('LUT6 arity',                       'hybrid_all_lut6_ckpt', True),
        ('LUT4 alone',                       'hybrid_all_lut4_ckpt', True),
        ('n_bits16 (input precision)',       'hybrid_all_nbits16', False),
        ('weighted_pool readout',            'hybrid_all_weighted', False),
    ]
    rows = [(lbl, lgn_acc(f'{R}/{d}/metrics.json')[0], real) for lbl, d, real in items]
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [C_GOOD if r[2] and r[1] > base + 0.5 else C_NEU for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(labels, vals, color=cols, edgecolor='white')
    ax.axvline(base, color='#34495e', ls='--', lw=1.2)
    ax.text(base, len(rows) - 0.3, f' base {base:.1f}%', color='#34495e', fontsize=9, va='top')
    ax.axvline(best, color=C_LGN, ls='-', lw=1.5)
    ax.text(best, len(rows) - 0.3, f' best combined {best:.1f}%', color=C_LGN, fontsize=9, va='top')
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f'{v:.1f}', va='center', fontsize=9)
    from matplotlib.patches import Patch
    ax.set_xlabel('Hard accuracy (%) — single technique added to base LGN-FFN')
    ax.set_title('Which levers move LGN-as-FFN', fontsize=13)
    ax.set_xlim(base - 3, best + 2)
    ax.legend(handles=[Patch(color=C_GOOD, label='real capacity/training lever'),
                       Patch(color=C_NEU, label='no real gain')],
              loc='lower right', frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig2_levers.png', dpi=150); plt.close(fig)
    print('fig2_levers.png', [(l, round(v, 2)) for l, v, _ in rows])


# --------------------------------------------------------------------------- #
# Figure 3 — honesty control: do the learned gates do real work?
# --------------------------------------------------------------------------- #
def fig_honesty():
    best = lgn_acc(f'{R}/lgn_best_om8_k16/metrics.json')[0]
    ident = lgn_acc(f'{R}/hybrid_all_identity/metrics.json')[0]
    nomlp = lgn_acc(f'{R}/no_mlp/metrics.json')[0]
    tf = tf_acc(f'{R}/lgn_best_om8_k16/metrics.json')
    labels = ['Attention only\n(no FFN block)', 'Identity gates\n(attn+readout+residual,\ndead logic)',
              'Learned LGN\n(real gates)']
    vals = [nomlp, ident, best]
    cols = [C_NEU, C_NEU, C_GOOD]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, vals, color=cols, edgecolor='white', width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f'{v:.1f}%', ha='center', fontweight='bold')
    ax.axhline(tf, color=C_TF, ls='--', lw=1, alpha=0.5)
    ax.text(0, tf + 0.4, f'transformer ceiling {tf:.1f}%', color=C_TF, fontsize=9)
    ax.set_ylabel('Hard accuracy (%)')
    ax.set_title('Honesty control', fontsize=13)
    ax.set_ylim(0, 60)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig3_honesty.png', dpi=150); plt.close(fig)
    print('fig3_honesty.png', dict(nomlp=round(nomlp, 2), identity=round(ident, 2),
          learned=round(best, 2), contrib=round(best - ident, 2)))


# --------------------------------------------------------------------------- #
# Figure 4 — the ideas that did NOT work (L0 screen, lower=better)
# --------------------------------------------------------------------------- #
def fig_screen():
    ref = hdeg('reference')
    rows = [
        ('residual_scale (#4)',      hdeg('residual_scale', 'readout_l0')),
        ('ensemble2 (#5, noise)',    hdeg('ensemble2', 'readout_l0')),
        ('conv post s1',             hdeg('post_s1_c128', 'conv_l0_variants')),
        ('LloydMax encoder',         hdeg('lloydmax')),
        ('conv pre s1',              hdeg('pre_s1_c128', 'conv_l0_variants')),
        ('conv pre+post s1',         hdeg('prepost_s1_c128', 'conv_l0_variants')),
        ('pool_curve (#2)',          hdeg('pool_curve', 'readout_l0')),
        ('TopK block-sparse',        hdeg('topk')),
        ('conv post s2 (ch=stride)', hdeg('post_s2_c2', 'conv_l0_variants')),
    ]
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [C_NEU if v < ref + 0.02 else C_BAD for v in vals]   # ensemble's seed-1337 dip is noise (fig5)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(labels, vals, color=cols, edgecolor='white')
    ax.axvline(ref, color=C_REF, ls='--', lw=1.5)
    ax.text(ref, len(rows) - 0.2, f' reference {ref:.3f}', color=C_REF, fontsize=9, va='top')
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8.5)
    ax.set_xlabel('L0 hard_degradation (val-loss increase vs transformer FFN, lower = better)')
    ax.set_title('Screened ideas vs reference', fontsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig4_screen.png', dpi=150); plt.close(fig)
    print('fig4_screen.png  ref=%.4f' % ref, [(l, round(v, 4)) for l, v in rows])


# --------------------------------------------------------------------------- #
# Figure 5 — why we don't trust the ensemble "win": the edge flips with the seed
# --------------------------------------------------------------------------- #
def fig_ensemble_noise():
    ref1 = hdeg('reference'); ref7 = hdeg('ref_s7', 'readout_l0')
    e1 = hdeg('ensemble2', 'readout_l0'); e7 = hdeg('ens2_s7', 'readout_l0')
    import numpy as np
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    b1 = ax.bar(x - w / 2, [ref1, ref7], w, label='reference', color=C_REF)
    b2 = ax.bar(x + w / 2, [e1, e7], w, label='ensemble2', color=C_LGN)
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                    f'{b.get_height():.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(['seed 1337', 'seed 7'])
    ax.set_ylabel('L0 hard_degradation (lower=better)')
    d1, d7 = e1 - ref1, e7 - ref7
    ax.set_title('Ensemble win is within the noise floor', fontsize=13)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(ref1, ref7, e1, e7) * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig5_ensemble_noise.png', dpi=150); plt.close(fig)
    print('fig5_ensemble_noise.png', dict(edge_s1337=round(d1, 4), edge_s7=round(d7, 4)))


# --------------------------------------------------------------------------- #
# Figure 6 — per-layer replacement difficulty (each FFN replaced alone)
# --------------------------------------------------------------------------- #
def fig_per_layer():
    d = json.load(open(f'{R}/hybrid_all_heat/heatmap.json'))
    d.sort(key=lambda r: r['layer_idx'])
    idx = [r['layer_idx'] for r in d]
    deg = [r['hard_degradation'] for r in d]
    cols = [C_BAD if v > 0.02 else (C_GOOD if v < -0.005 else C_NEU) for v in deg]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([f'L{i}' for i in idx], deg, color=cols, edgecolor='white')
    ax.axhline(0, color='#34495e', lw=1)
    for b, v in zip(bars, deg):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.012 if v >= 0 else -0.012),
                f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8.5)
    ax.set_ylabel('hard_degradation when this layer alone is LGN (lower = easier)')
    ax.set_title('Per-layer replacement difficulty', fontsize=13)
    ax.set_ylim(min(deg) - 0.05, max(deg) + 0.07)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_GOOD, label='easier than FFN (replacing helps)'),
                       Patch(color=C_NEU, label='≈ free'),
                       Patch(color=C_BAD, label='harder (needs capacity)')],
              loc='upper center', frameon=False, fontsize=9, ncol=3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig6_per_layer.png', dpi=150); plt.close(fig)
    print('fig6_per_layer.png', [(f'L{i}', round(v, 3)) for i, v in zip(idx, deg)])


if __name__ == '__main__':
    fig_headline()
    fig_levers()
    fig_honesty()
    fig_screen()
    fig_ensemble_noise()
    fig_per_layer()
    print('\nAll figures written to', FIG)
