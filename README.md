# LGN-Nano: FFN keitimas į Logic Gate Network transformeryje

Pagrindinis klausimas: **kiek gerai diferencijuojamas Boolean Logic Gate Network (LGN) gali
pakeisti transformerio FFN (MLP) sluoksnį** — ir ar tai daro reali išmokta logika, ar tik
aplinkinė „instaliacija". Modelis: nanoGPT, 12 sluoksnių × 128d × 4 head'ai, byte-level
WikiText-2. Metrika visur — **next-byte top-1 accuracy** ant fiksuoto (seed-1234) validacijos
batch'o; LGN visada rodomas **hard** (diskretizuotas, kaip realiame inference).

**Ką pasiekėme šioje fazėje:**
- Optimizuotas **LGN-FFN (su užšaldytu attention) pasiekia 48.2 % = 88 % transformerio** —
  nuo bazinių 35.4 %.
- **Tas pats pagerinimas perėjo ir į gryną LGN** (be attention, su token-shift): **43.5 % =
  80 % transformerio**. Optimizacija nebuvo attention-specifinė.
- Atskira sąžiningumo kontrolė: išmokta logika prideda **+21.7 pp** virš „negyvų" vartų.
- Visi vėliau siūlyti įvesties/nuskaitymo priedai (conv1d, LloydMax, TopK, ir kt.) **nedavė
  nieko** — riba yra skaičiavimo talpa, ne kodavimas (žemiau paaiškinta kodėl).

![accuracy comparison](results/figs/report/fig1_headline.png)

| Modelis | Accuracy % | % transformerio |
|---|---:|---:|
| NanoGPT transformeris (lubos) | **54.87** | 100 |
| **Attention + LGN-FFN (optimizuotas)** | **48.18** | **88** |
| **Grynas LGN (token-shift, be attention)** | **43.54** | **80** |
| Bazinis LGN-FFN | 35.35 | 64 |
| Identity vartai (kontrolė: logika išjungta) | 26.46 | 48 |
| Tik attention (FFN pašalintas) | 5.46 | 10 |

---

## Ką pasiekėme: LGN kaip FFN pakaitalas

Kad atskirčiau *cross-token* (attention) ir *per-token* (FFN) darbą, **užšaldau ištreniruotą
attention visuose 12 sluoksnių** ir leidžiu LGN pakeisti **tik FFN**. Klausimas grynas:
*kiek gerai LGN imituoja FFN, kai attention idealus?* Bazinis LGN-FFN pasiekia 35.4 %; toliau
ieškojau, kas iš tikrųjų pajudina šį skaičių.

Pagrindinė išvada: **atotrūkį riboja skaičiavimo talpa (vartų kiekis ir galia), o NE kodavimo
ar nuskaitymo precizija.** Sistemingai patikrinau ir atmečiau preciziją (8-bit ≈ 16-bit
įvestis; `weighted_pool` protingesnis nuskaitymas — 0 naudos). Tikrieji svertai:

![what moves the metric](results/figs/report/fig2_levers.png)

| Svertas | Efektas | Tipas |
|---|---|---|
| **out_gate_mult** (daugiau išvesties vartų) | 35.4 → 38.3 → **41.8 %** (1×→2×→4×) | talpa (stipriausias) |
| **LUT-K aritetas** (k-input vartas vietoj 2-input) | +0.9 pp prie vienodo vartų kiekio | vartų galia |
| **CAGE / STE** (hard-forward) | uždaro soft–hard gap'ą | treniravimas |
| **Best-hard checkpoint** | +0.8 pp | treniravimas |
| **KL distiliacija** iš transformerio | +0.5 pp | treniravimas |

Šių svertų suma (om8 + k16 + LUT4 + CAGE + best-hard + KL) ir pakelia 35.4 % → **48.2 %**.

```bash
python run.py scale --hybrid_all --hybrid_ln2 copy_trainable --learn_pool \
  --lut_k 4 --out_gate_mult 4 --out_gate_mult_layers 0:8 11:8 10:8 9:8 \
  --k 16 --n_bits 8 --cage --anneal_in_finetune --ft_keep_best_hard \
  --grad_checkpoint --batch_size 16 \
  --heatmap results/report/hybrid_all_heat/heatmap.json --checkpoint results/baseline.pt
```

### Sąžiningumo kontrolė: vartai dirba realiai

Ar 48.2 % padaro vartai, ar aplinkinė instaliacija (ln + pooling + residual + užšaldytas
attention)? Kontrolė atsako: identity vartai (logika išjungta, ta pati instaliacija) pasiekia
tik 26.5 %. **Išmokta logika prideda +21.7 pp** — tai ne instaliacijos efektas.

![honesty control](results/figs/report/fig3_honesty.png)

### Kur sėdi sunkumas: L0

Pakeitus po vieną FFN matosi aiški hierarchija: **L0 sunkiausias** (×6 už bet kurį kitą),
vidurio FFN (L1–L6) beveik nemokami (juos pakeitus net *pagerėja* val loss), pabaigos
sluoksniai vėl pasunkėja. Todėl papildomą talpą skiriu selektyviai L0/L9/L10/L11.

![per-layer difficulty](results/figs/report/fig6_per_layer.png)

---

## Pagerinimas perėjo ir į gryną LGN (be attention)

Svarbiausias patikrinimas: ar šie svertai veikia tik su realiu attention, ar persikelia į
**visiškai attention-free** variantą? Grynam LGN cross-token sprendžiu pigiai — **token-shift**
(prie kiekvienos pozicijos pridedu kelias praėjusias; fiksuotas postūmis, jokių mokomų
parametrų). Pritaikius **tą pačią optimizuotą vartų konfigūraciją** + token-shift:

> **Grynas LGN = 43.5 % = 80 % transformerio.** Net attention pakeitus pigiu postūmiu,
> optimizacija išlieka — t.y. tai buvo FFN-pakaitalo pagerinimas, ne attention triukas.

Atotrūkis tarp 88 % (su attention) ir 80 % (grynas) yra attention-vs-token-shift cross-token
kaina — atskira problema nuo per-token FFN darbo.

---

## Kodėl pasiūlyti priedai nesuveikė

Po šių rezultatų buvo pasiūlyta dar keletas įvesties/nuskaitymo/interconnect idėjų. Įdiegiau
ir patikrinau **visas** (L0 screen, hard_degradation, mažiau = geriau, ref 0.074), kiekvieną
su sąžiningomis kontrolėmis. **Nė viena patikimai nepralenkė bazės** — ir tai logiška: jos
visos taiko ne į tą dimensiją (talpą), o į kodavimą/nuskaitymą/sujungimą.

![what did not work](results/figs/report/fig4_screen.png)

| Pasiūlymas | Rezultatas | **Kodėl nesuveikė** |
|---|---|---|
| **Conv1D prieš/po LGN** | neutralu (s1), blogiau su stride | ablation: užšaldžius vartus rezultatas tas pats → **conv perima darbą, ne LGN (fake LGN)**; o stride-adapteris praranda informaciją |
| **LloydMax binarizatorius** (per-channel EMA Gaussian thresholds) | neutralu | binarizacijos precizija **nebuvo** bottleneck'as — sigmoid-thermometer ir taip pakankamas; talpa, ne kodavimas |
| **TopK block-sparse interconnect** | blogiau | struktūrinis blokinis pasirinkimas **prastesnis** nei random kandidatų „loterija" vartų įvestims |
| **pool_curve** (mokoma netiesinė nuskaitymo kreivė) | blogiau | soft pagerėja, bet **nediskretizuojasi** — kreivė persimoko ant trupmeninių soft skaičių, hard sprogsta |
| **residual_scale** (per-channel α prie LGN išvesties) | neutralu | α tiesiog lieka ≈1; nėra ką pridėti |
| **ensemble** (N vartų bankų vidurkis) | **triukšmas** | „laimėjimas" −0.013 prie vieno seed'o **apsiverčia** į +0.009 prie kito → telpa į screen'o triukšmą |

⚠️ **Metodologinė pastaba.** ensemble iš pradžių atrodė kaip pirmas laimėjimas, bet
seed-pakartojimas parodė, kad ženklas apsiverčia. Būtent dėl to nė vienas kandidatas
neskelbiamas be seed-kontrolės.

![ensemble noise](results/figs/report/fig5_ensemble_noise.png)

**Bendras vardiklis:** likęs atotrūkis yra **fundamentalus** (kvantuota reta logika vs tankus
float FFN). Jį judina tik talpa / vartų galia / treniravimas — ne įvesties ar nuskaitymo
patobulinimai.

---

## Efektyvumas (grynas LGN kelias)

Grynas LGN (be attention) yra gerokai efektyvesnis — kelis kartus mažiau parametrų ir
8–30× mažiau FLOPs (FPGA/ASIC: kiekvienas 2-input vartas = 1 LUT, LUT6 = vienas natūralus
vienetas). Būtent dėl to grynas variantas, į kurį perėjo pagerinimas, yra praktiškai
įdomiausias taikinys.

![efficiency](results/figs/report_en/05_efficiency.png)

> **Pastaba:** 48.2 % skaičius laiko pilną float attention visur, tad efektyvumo pranašumas
> galioja **grynam LGN** keliui (43.5 %), ne attention+LGN-FFN.

---

## Kodas ir paleidimas

| Failas | Kas |
|---|---|
| `lgn.py` | sluoksniai, vartai (2-input + LUT-K), kodavimas, pooling, hard mirror'ai |
| `pipeline.py` | duomenys, metrika, imitation / fine-tune (CAGE, STE, KL, best-hard), greedy scaling |
| `run.py` | CLI (`scale` / `heatmap`) ir visi flag'ai |
| `experiments/make_report_figures.py` | visos figūros iš tikrų `results/**/metrics.json` |
| `tests/` | 44 testai (kiekvienas blokas: soft==hard + gradientai) |

Visi tirti priedai (conv1d, LloydMax, TopK, pool_curve, residual_scale, ensemble) yra
**opt-in, default'ai nepakeisti**. Figūros atsigamina:
`python experiments/make_report_figures.py`.

## Kryptys toliau

- **Atminties-efektyvi talpa** (susieti LUT lentelės / fp16), kad tilptų om16/LUT6 be OOM —
  vienintelis svertas su likusiu užtaisu, nes talpa yra įrodytas, bet atminties ribojamas kelias.
- Recurrent/stateful LGN sluoksnis (`RecurrentLogicGateGPTLayer`) jau įdiegtas kaip
  eksperimentinė alternatyva token-shift'ui; accuracy dar nepamatuota.
