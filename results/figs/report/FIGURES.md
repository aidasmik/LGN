# Report figures — LGN as an FFN replacement

All figures regenerate from real result files via `python experiments/make_report_figures.py`.
Metric: next-byte top-1 accuracy on the fixed (seed-1234) byte-level WikiText-2 validation set;
the LGN is always reported as the **hard** (discretized) model. Screens use **L0 hard_degradation**
(increase in val loss vs the transformer FFN; lower = better).

| # | file | caption |
|---|------|---------|
| 1 | `fig1_headline.png` | **Where LGN-as-FFN lands.** Transformer 54.9% → Attention-LGN (best, om8+k16) **48.2% = 88% of transformer** → Pure LGN (token-shift, no attention) **43.5% = 80% of transformer**. Controls: identity/dead gates 26.5%, attention-only (no FFN) 5.5%. |
| 2 | `fig2_levers.png` | **Which levers actually move the metric** (single technique added to the base LGN-FFN, 35.4%). Real levers (green): output capacity `out_gate_mult` (+6.4pp at 4×), LUT arity, CAGE/STE. No real gain (grey): input precision `n_bits16`, `weighted_pool`. Best *combined* config reaches 48.2%. |
| 3 | `fig3_honesty.png` | **Honesty control.** Attention-only 5.5% → identity gates (attention + readout + residual, dead logic) 26.5% → learned LGN 48.2%. The learned logic contributes **+21.7pp** over the plumbing — the gates do real work. |
| 4 | `fig4_screen.png` | **Ideas that did not work** (L0 screen vs optimized reference 0.074). Conv1D (neutral at best, badly worse with stride), LloydMax encoder (neutral), TopK block-sparse (worse), nonlinear readout `pool_curve` #2 (worse), `residual_scale` #4 (neutral), `ensemble` #5 (noise — see fig 5). None reliably beats the reference. |
| 5 | `fig5_ensemble_noise.png` | **Why the ensemble "win" is rejected.** ensemble2's edge is −0.013 at seed 1337 but **+0.009 at seed 7** — the sign flips, so the apparent gain is within the screen's noise floor. (Methodological point: every candidate gain is seed-checked.) |
| 6 | `fig6_per_layer.png` | **Per-layer replacement difficulty** (each FFN replaced alone, others intact). L0 is by far the hardest (+0.43); middle layers L1–L6 are near-redundant (negative — replacing them slightly *helps*); late layers climb back up (L11 +0.23). Motivates putting extra gate capacity on L0/L9/L10/L11. |

## Source runs
- Transformer / Attention-LGN best: `results/report/lgn_best_om8_k16/metrics.json` (54.87 / 48.18)
- Pure LGN token-shift: `results/report/tshift2_opt/metrics.json` (43.54, tf 54.67)
- Controls: `results/report/hybrid_all_identity/metrics.json` (26.46), `results/report/no_mlp/metrics.json` (5.46)
- Levers: `results/report/hybrid_all_*` metrics.json
- Per-layer: `results/report/hybrid_all_heat/heatmap.json`
- Screens: `results/new_lgn_inputs/**/heatmap.json` (reference, conv_l0_variants, lloydmax, topk, readout_l0)
