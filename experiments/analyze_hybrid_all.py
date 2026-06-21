"""Summarize the all-attention hybrid experiment: FFN sensitivity ranking, the main
transformer-vs-LGN metric table, the cumulative degradation curve, and a fake-LGN check
(identity control). Reads results/report/*; writes figures to results/figs/hybrid_all/.

Usage: python experiments/analyze_hybrid_all.py
"""
import json, os, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = 'results/report'
FIG = 'results/figs/hybrid_all'


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fmt_pct(x):
    return f'{x*100:.2f}' if x is not None else '-'


def sensitivity():
    heat = load(f'{OUT}/hybrid_all_heat/heatmap.json')
    if not heat:
        print('(no heatmap yet)'); return None
    rows = sorted(heat, key=lambda r: r['hard_degradation'], reverse=True)
    print('\n=== Per-FFN sensitivity (attention kept, only that FFN -> LGN) ===')
    print(f'{"rank":>4} {"layer":>5} {"hard_deg":>9} {"soft_deg":>9}')
    for i, r in enumerate(rows):
        print(f'{i+1:>4} {r["layer_idx"]:>5} {r["hard_degradation"]:>9.4f} {r.get("soft_degradation",0):>9.4f}')
    print(f'  most sensitive: {[r["layer_idx"] for r in rows[:4]]}  '
          f'least: {[r["layer_idx"] for r in rows[-4:]]}')
    return rows


def metrics_table():
    print('\n=== transformer vs hard LGN (FFN replaced, attention kept) ===')
    print(f'{"config":28} | {"loss":>7} | {"ppl":>7} | {"acc %":>7} | {"n_lgn":>5}')
    names = ['hybrid_all_main', 'hybrid_all_identity', 'hybrid_all_calib',
             'hybrid_all_ln2_frozen', 'hybrid_all_ln2_fresh',
             'hybrid_all_nbits1', 'hybrid_all_nbits4',
             'hybrid_all_depth2', 'hybrid_all_depth3',
             'hybrid_all_keep2_transformer', 'hybrid_all_keep4_transformer']
    shown_tf = False
    for nm in names:
        m = load(f'{OUT}/{nm}/metrics.json')
        if not m:
            continue
        if not shown_tf:
            t = m['transformer']
            print(f'{"transformer (baseline)":28} | {t["loss"]:>7.4f} | {t["perplexity"]:>7.3f} | '
                  f'{fmt_pct(t["accuracy"]):>7} | {"-":>5}')
            shown_tf = True
        g = m['lgn_hard']
        print(f'{nm:28} | {g["loss"]:>7.4f} | {g["perplexity"]:>7.3f} | '
              f'{fmt_pct(g["accuracy"]):>7} | {m.get("n_lgn_layers","-"):>5}')

    # fake-LGN check
    main = load(f'{OUT}/hybrid_all_main/metrics.json')
    ident = load(f'{OUT}/hybrid_all_identity/metrics.json')
    if main and ident:
        d = (main['lgn_hard']['accuracy'] - ident['lgn_hard']['accuracy']) * 100
        print(f'\n  FAKE-LGN CHECK: main {fmt_pct(main["lgn_hard"]["accuracy"])}% vs '
              f'identity-LGN {fmt_pct(ident["lgn_hard"]["accuracy"])}%  (gap {d:+.2f} pp)')
        print('  -> small gap means ln_2/pool plumbing (not the gates) is doing the work.')


def figures():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'(matplotlib unavailable: {e})'); return
    os.makedirs(FIG, exist_ok=True)

    heat = load(f'{OUT}/hybrid_all_heat/heatmap.json')
    if heat:
        rows = sorted(heat, key=lambda r: r['layer_idx'])
        idx = [r['layer_idx'] for r in rows]
        deg = [r['hard_degradation'] for r in rows]
        plt.figure(figsize=(8, 4))
        plt.bar(idx, deg, color='#c0504d')
        plt.xlabel('layer'); plt.ylabel('hard val-loss degradation')
        plt.title('Per-FFN sensitivity (attention kept, FFN->LGN)')
        plt.xticks(idx); plt.tight_layout()
        plt.savefig(f'{FIG}/ffn_sensitivity.png', dpi=120); plt.close()
        print(f'  saved {FIG}/ffn_sensitivity.png')

    sc = load(f'{OUT}/hybrid_all_main/scale_greedy.json')
    if sc:
        n = [r['n_replaced'] for r in sc]
        hd = [r['hard_degradation'] for r in sc]
        plt.figure(figsize=(8, 4))
        plt.plot(n, hd, 'o-', color='#4f81bd')
        plt.xlabel('# FFNs replaced by LGN'); plt.ylabel('hard val-loss degradation')
        plt.title('Cumulative all-attention hybrid scaling')
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f'{FIG}/cumulative_scaling.png', dpi=120); plt.close()
        print(f'  saved {FIG}/cumulative_scaling.png')


if __name__ == '__main__':
    sensitivity()
    metrics_table()
    print('\n=== figures ===')
    figures()
