# LGN-Nano: Logic Gate Networks transformerio sluoksniuose

Tikrinu, kiek nanoGPT transformerio sluoksnių galima pakeisti į Boolean **Learned Logic
Gate Networks (LGN)** — ir, svarbiausia, ar iš to lieka realaus loginio darbo, ar tik
aplinkinių Linear sluoksnių kompensacija. Modelis: nanoGPT, 12 sluoksnių × 128d × 4 head'ai,
byte-level WikiText-2. Metrika visur — **next-byte top-1 accuracy** ant fiksuoto (seed-1234)
validacijos batch'o; LGN visada rodomas **hard** (diskretizuotas, kaip realiame inference).

Tiriu dvi kryptis atskirai:

1. **Grynas LGN** (be attention; cross-token sprendžiamas token-shift'u) — maksimalus
   efektyvumas, taikinys FPGA/ASIC.
2. **Attention + LGN-FFN** (užšaldau ištreniruotą attention, LGN keičia tik FFN/MLP) — švarus
   klausimas, *kiek gerai LGN imituoja FFN, kai attention idealus.*

Trumpa esmė po viso optimizavimo: geriausias **attention + LGN-FFN pasiekia 48.2 % = 88 %
transformerio**, geriausias **grynas LGN (token-shift) — 43.5 % = 80 % transformerio**.
Atotrūkis iki transformerio realus, bet jo NEpramuša jokie kodavimo / nuskaitymo / įvesties
patobulinimai — riba yra **skaičiavimo talpa (vartų kiekis)**, ne precizija.

![accuracy comparison](results/figs/report/fig1_headline.png)

| Modelis | Accuracy % | % transformerio | Ką keičia |
|---|---:|---:|---|
| NanoGPT transformeris | **54.87** | 100 | 12 sluoksnių baseline (lubos) |
| Attention + LGN-FFN (geriausias) | **48.18** | **88** | užšaldytas attention visur, optimizuoti FFN vartai |
| Grynas LGN (token-shift) | **43.54** | **80** | jokio attention; token-shift + tie patys vartai |
| out_gate_mult4 (vienas svertas) | 41.77 | 76 | tik išvesties vartų kiekis 4× |
| LUT4 + CAGE | 39.07 | 71 | galingesnis vartas + hard-forward |
| Bazinis LGN-FFN | 35.35 | 64 | 2-input vartai, 1× |
| Identity vartai (kontrolė) | 26.46 | 48 | tik „instaliacija" (ln + pooling + residual) |
| Tik attention (FFN pašalintas) | 5.46 | 10 | grindys — FFN vertas +43 pp |

---

## L0 cross-token bottleneck

Pagrindinis bottleneck, dėl kurio krenta visas tikslumas, yra **L0 (pirmas sluoksnis)
cross-token apribojimas**. Problema ta, kad **kiekvienas tokenas pats iš savęs daug
nereiškia** — informacija ateina iš konteksto, kokie tokenai buvo prieš jį. Transformeryje
attention leidžia referencuoti buvusius tokenus, o grynas LGN kiekvieną poziciją apdoroja
atskirai (pointwise), todėl praranda daug tikslumo.

Kai užšaldau attention visuose 12 sluoksnių ir keičiu tik FFN, matosi aiški **sluoksnių
hierarchija**: L0 sunkiausias (×6 už bet kurį kitą), vidurio FFN (L1–L6) beveik nemokami
(juos pakeitus net *pagerėja* val loss), o pabaigos sluoksniai vėl pasunkėja (L11 antras
sunkiausias).

![per-layer difficulty](results/figs/report/fig6_per_layer.png)

Būtent dėl to papildomą vartų talpą skiriu selektyviai L0/L9/L10/L11, o ne tolygiai.

---

## Cross-token sprendimai (grynam LGN)

### Hybrid L0
Transformerio blokas turi dvi dalis — **MLP** ir **attention**. L0 sluoksniui **nukopijuoju
attention dalį iš jau ištreniruoto transformerio** (palieku užšaldytą), o LGN naudoju tik
vietoj MLP. Taip LGN gauna nebe raw embedding'ą, o **attention jau apdorotą srautą**.

### TokenShift
Prieš paduodant signalą į LGN, prie kiekvienos pozicijos **pridedu keletą praėjusių
pozicijų** — LGN mato `[x[t], x[t-1], ..., x[t-K]]`. Tai duoda vartams lokalų cross-token
langą **pigiai ir sąžiningai** (tik fiksuotas pozicijų postūmis, jokių mokomų parametrų —
skirtingai nei conv/linear, kur papildomas sluoksnis pats išmoktų dalį darbo).

Šie du metodai sprendžia **tą pačią** problemą: jų kombinacija beveik nieko neprideda, todėl
TokenShift neblogai imituoja attention. Su optimizuotais vartais (žemiau) grynas LGN +
token-shift pasiekia **43.5 %** — net attention pakeitus pigiu postūmiu lieka **80 %
transformerio**.

---

## LGN kaip FFN pakaitalas: kur tikrosios lubos?

Kad atskirčiau *cross-token* (attention) ir *per-token* (FFN) darbą, atlikau švarų
eksperimentą: **užšaldau ištreniruotą attention VISUOSE 12 sluoksnių** ir leidžiu LGN pakeisti
**tik FFN**. Klausimas grynas: *kiek gerai LGN gali imituoti FFN, kai attention idealus?*

![honesty control](results/figs/report/fig3_honesty.png)

**Pirma išvada — attention nebuvo vienintelė problema.** Net su idealiu attention bazinis
LGN-FFN pasiekia tik 35 %. Kontrolė patvirtina, kad vartai dirba realiai (ne „instaliacija"):
išmokta logika prideda **+21.7 pp** virš identity-LGN (negyvi vartai, ta pati instaliacija).

### Kas yra tikrasis svertas — NE precizija, o vartų KIEKIS

Hipotezė buvo, kad bottleneck'as — binarizacijos precizija. Sistemingai patikrinau ir
**atmečiau** ją:

- **Įvesties precizija nesvarbi.** `out_mult2` (8-bit įvestis) ≈ `n_bits16` (16-bit) — perpus
  mažiau įvesties bitų, tas pats rezultatas.
- **Nuskaitymo (readout) precizija nesvarbi.** `weighted_pool` (mokomi per-bitiniai svoriai)
  nieko nedavė (35.31 % ≈ 35.35 % bazė).
- **Vartų kiekis — DUODA.** Daugiau išvesties vartų kanalui: 35.4 → 38.3 → **41.8 %**
  (vartai 1× → 2× → 4×). Tai pats stipriausias vienas svertas.

![what moves the metric](results/figs/report/fig2_levers.png)

Taigi LGN-FFN atotrūkį riboja **skaičiavimo talpa**, ne kodavimo/nuskaitymo precizija.

### Efektyvumo svertas: vartų ARITETAS (k-input LUT)

Jei riba — vartų kiekis, klausimas: *ar galingesnis primityvas padaro daugiau vienam vartui?*
Įdiegiau **k-input LUT vartą** (LUT-K: mokoma 2^K-įrašų lentelė per multitiesį išplėtimą;
hard-snap'inasi į vieną FPGA LUT-K). Ant L0 (sunkiausio), vienodas vartų kiekis: LUT4 ≈
2-input su 2× vartų; LUT6 ≈ 2-input su ~2.7× vartų.

⚠️ **Sąžininga korekcija — L0 efektas nepersikelia į visą modelį.** Visuose 12 sluoksnių LUT4
duoda tik **+0.9 pp** virš to paties vartų kiekio 2-input bazės, o vartų *padvigubinimas*
(out_mult2) — +3.0 pp. Priežastis: tik keli sunkūs sluoksniai gauna naudos iš galingesnio
primityvo; vidurio FFN ir taip lengvi, tad vidurkis atskiedžiamas. Aritetas — realus, bet
**kuklus** svertas, svarbus ten, kur sluoksnis sunkus, ne visur.

### Treniravimo svertai (be papildomų vartų)

- **CAGE / STE** (hard-forward): forward daromas kietas (argmax, kaip inference), gap'as iš
  principo dingsta; gradientas minkštas su adaptyvia temperatūra. Uždaro soft–hard gap'ą.
- **Best-hard checkpoint** (renku ne geriausią soft, o geriausią *hard*): ~+0.8 pp.
- **KL distiliacija** iš transformerio logitų: ~+0.5 pp.

Šių svertų suma (om8 + k16 + LUT4 + CAGE + best-hard + KL) ir pakelia bazinį 35.4 % iki
**48.2 %**.

Paleidimas (visi sluoksniai hibridiniai, geriausia konfigūracija):
```bash
python run.py scale --hybrid_all --hybrid_ln2 copy_trainable --learn_pool \
  --lut_k 4 --out_gate_mult 4 --out_gate_mult_layers 0:8 11:8 10:8 9:8 \
  --k 16 --n_bits 8 --cage --anneal_in_finetune --ft_keep_best_hard \
  --grad_checkpoint --batch_size 16 \
  --heatmap results/report/hybrid_all_heat/heatmap.json --checkpoint results/baseline.pt
```

**Sąžiningumo pastaba:** šie skaičiai laiko **pilną float attention** visuose 12 sluoksnių,
tad efektyvumo pranašumas (žemiau) galioja **tik grynam LGN** keliui — šios fazės tikslas buvo
*suprasti* LGN-kaip-FFN ribą, ne pasiekti efektyvumo rekordą.

---

## Selective LGN

Kiek galima palikti transformer sluoksnių, paaukojant efektyvumą už tikslumą — tolydi kreivė
tarp gryno LGN ir transformerio.

![selective curve](results/figs/report_en/07_selective_curve.png)

---

## Konfigūracijos, kurios nieko reikšmingo nedavė

Didžioji dalis idėjų **nepasiteisino** — ir tai pati naudingiausia dalis, nes atskiria
tikruosius svertus nuo iliuzijų. L0 screen'as (hard_degradation, mažiau = geriau, ref 0.074),
visi po sąžiningomis kontrolėmis:

![what did not work](results/figs/report/fig4_screen.png)

| Bandymas | Rezultatas | Kodėl |
|---|---|---|
| Depth + random interconnect | pablogėjo | hard-snap klaidos kaupiasi |
| weighted_pool / signed encoding | 0 | protingesnis nuskaitymas talpos nepakeičia |
| n_bits16 (įvesties precizija) | 0 | ne tas svertas |
| **Conv1D prieš/po LGN** | neutralu (s1) arba blogiau (stride) | + ablation: užšaldžius vartus rezultatas tas pats → **conv perima darbą (fake LGN)** |
| **LloydMax binarizatorius** (per-channel EMA Gaussian thresholds) | neutralu | sigmoid-thermometer ir taip pakankamas |
| **TopK block-sparse interconnect** | blogiau | random kandidatų „loterija" ir taip geresnė |
| **pool_curve** (mokoma per-channel netiesinė nuskaitymo kreivė) | blogiau | soft pagerėja, bet nediskretizuojasi (soft–hard sprogimas) |
| **residual_scale** (per-channel α) | neutralu | α tiesiog lieka ≈1 |
| **ensemble** (N vartų bankų vidurkis) | **triukšmas** | atrodė +0.013, bet ženklas apsiverčia su kitu seed'u |

⚠️ **Metodologinė pastaba.** ensemble iš pradžių atrodė kaip pirmas laimėjimas (−0.013 prie
seed 1337), bet seed-pakartojimas parodė **+0.009 prie seed 7** — ženklas apsiverčia, t.y.
„laimėjimas" telpa į screen'o triukšmo ribą. Būtent tokius mirage'us ir gaudo seed-kontrolė;
nė vienas kandidatas neskelbiamas be jos.

![ensemble noise](results/figs/report/fig5_ensemble_noise.png)

**Bendra išvada:** vienintelės kryptys, kurios juda metriką — **talpa (out_gate_mult),
vartų galia (LUT-K) ir treniravimas (CAGE + best-hard + KL)**. Kiekvienas įvesties /
nuskaitymo / interconnect priedas yra neutralus arba blogesnis; likęs atotrūkis yra
**fundamentalus** (kvantuota reta logika vs tankus float FFN).

---

## Efektyvumas (grynas LGN kelias)

Grynas LGN (be attention) yra gerokai efektyvesnis — kelis kartus mažiau parametrų ir
8–30× mažiau FLOPs. FLOPs/token — teorinis aritmetinių operacijų kiekis vienam tokenui per
12 blokų: transformer bloke skaičiuoju attention + MLP matricų daugybas, LGN bloke Linear
sluoksnių nėra — lieka tik vartai (≈5 operacijos) ir sum_pool.

![efficiency](results/figs/report_en/05_efficiency.png)

| Config | Total params | FLOPs/token | FLOPs vs transformer |
|---|---:|---:|---:|
| Transformer | 2.45 M | 2.56 M | 1.0× |
| Aggressive (grynas) | 0.37 M | 0.086 M | **29.7× mažiau** |
| Hybrid L0 | 0.44 M | 0.168 M | 15.2× mažiau |
| Token shift K=2 | 0.96 M | 0.258 M | 9.9× mažiau |

Realių hardware skaičių nelyginau, nes ant GPU LGN visada veikia prasčiau už transformerį —
GPU optimizuotas tankiai matricų daugybai. Tikrasis pranašumas realizuojamas **FPGA/ASIC**,
kur kiekvienas 2-input vartas = 1 LUT, o LUT6 = vienas natūralus FPGA vienetas.

---

## Literatūra

- **„Mind the Gap" (NeurIPS 2025)** — soft–hard gap'as per Gumbel noise + STE. Image
  rezultatai geri, mano byte-LM setup'e nepasiteisino.
- **„Light DLGN" (2025)** — vartų reparametrizacija (IWP): 4× mažiau parametrų. Labiau tinka
  image, čia −5 pp.
- **[Recurrent DDLGN (2025)](https://arxiv.org/abs/2508.06097)** — sprendžia pagrindinę
  problemą (cross-token) per **stateful vartus (flip-flops, latches)**. Galimai pakeistų
  attention pačiame LGN lygmenyje; reikalauja didelio pertvarkymo.
- **[CAGE „Align Forward, Adapt Backward" (2026)](https://arxiv.org/abs/2603.14157)** —
  forward kietas (argmax), gradientas minkštas. Įdiegiau; gap'ą sumažino ~perpus.

---

## Recurrent / stateful LGN (RDDLGN-inspired, eksperimentinis)

Kaip pirmą žingsnį RDDLGN kryptimi pridėjau **recurrent/stateful LGN sluoksnį** —
alternatyvą TokenShift'ui. Vietoj fiksuoto kaimynų lango, kiekvienam tokenui logikos stack'as
atnaujina **paslėptą būseną**:

```
state_t = Logic([token_bits_t, state_{t-1}])
out_t   = group_sum(state_t)
```

Causal ir leidžia logikai maišyti informaciją per seką per būseną. **Tai NĖRA pilnas RDDLGN
encoder–decoder** — tik stateful mechanizmas, drop-in GPT-bloko pakaitalas
(`RecurrentLogicGateGPTLayer`), suderinamas su esamu pipeline'u.

```bash
python run.py scale --recurrent --recurrent_layers 0 \
  --recurrent_state_width 1024 --recurrent_depth 1 --recurrent_state_init zero \
  --learn_pool --heatmap results/aggressive/heatmap.json --checkpoint results/baseline.pt
```

### Gated (flip-flop / latch-inspired) būsenos atnaujinimas

Vanilla recurrent kiekviename žingsnyje **visą būseną perrašo** — nėra mechanizmo *išlaikyti*
bitą per ilgesnę seką. Pridėjau (opt-in) **gated** atnaujinimą: be kandidato dar mokomas
atskiras **keep** loginis stack'as, kuris sprendžia, laikyti ar perrašyti:

```
candidate_t = LogicCandidate([token_bits_t, state_{t-1}])
keep_t      = LogicKeep([token_bits_t, state_{t-1}])
state_t     = where(keep_t, state_{t-1}, candidate_t)   # hard
```

Svarbu: **keep vartas pats yra mokomas LOGIKOS stack'as** (`LearnedLogicLayer`), ne
sigmoid/dense gate'as — visas mechanizmas lieka Boolean ir hard-snap'inamas. Tai
flip-flop/latch-**inspired** plėtinys, ne teiginys, kad RDDLGN paper'is naudojo GRU-stiliaus
keep vartą.

```bash
python run.py scale --recurrent --recurrent_gated --recurrent_layers 0 \
  --recurrent_state_width 1024 --recurrent_state_init zero \
  --learn_pool --heatmap results/aggressive/heatmap.json --checkpoint results/baseline.pt
```

---

## Kodo struktūra

| Failas | Kas |
|---|---|
| `lgn.py` | visi sluoksniai, vartai (2-input + LUT-K), kodavimas, pooling, hard mirror'ai |
| `pipeline.py` | duomenys, imitation / fine-tune (CAGE, STE, KL, best-hard), greedy scaling |
| `run.py` | CLI (`scale` / `heatmap`) ir visi flag'ai |
| `experiments/` | eksperimentų orkestratoriai + `make_report_figures.py` (figūros iš tikrų metrikų) |
| `tests/` | 44 testai (kiekvienas naujas blokas: soft==hard + gradientai) |

Figūros atsigamina iš tikrų `results/**/metrics.json` per
`python experiments/make_report_figures.py`. Visi nauji blokai (conv1d, LloydMax, TopK,
pool_curve, residual_scale, ensemble) yra **opt-in, default'ai nepakeisti**.

## Kryptys toliau

- Recurrent sluoksnio accuracy dar nepamatuota — sekantis žingsnis (state-width / depth / init
  sweep'ai; ar pralenkia TokenShift'ą).
- Atminties-efektyvi talpa (susieti LUT lentelės / fp16), kad tilptų om16/LUT6 be OOM — tai
  vienintelis svertas su likusiu užtaisu, nes talpa yra įrodytas, bet atminties ribojamas
  kelias.
