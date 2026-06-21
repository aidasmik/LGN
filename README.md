# LGN-Nano: FFN keitimas į Logic Gate Network transformeryje

Tikrinu, kiek gerai diferencijuojamas Boolean **Logic Gate Network (LGN)** gali pakeisti
transformerio FFN (MLP) sluoksnį — ir, svarbiausia, ar tą darbą atlieka reali išmokta logika,
ar tik aplinkinė „instaliacija" aplink ją. Modelis: nanoGPT, 12 sluoksnių × 128d × 4 head'ai,
byte-level WikiText-2. Metrika visur — **next-byte top-1 accuracy** ant fiksuoto (seed-1234)
validacijos batch'o; LGN visada rodau **hard** (diskretizuotą, lygiai kaip realiame inference).

Trumpa esmė šios fazės: pradėjau nuo bazinio LGN-FFN ties 35.4 % ir, sistemingai ieškodamas
kas iš tikrųjų pajudina skaičių, **užšaldytu attention pasiekiau 48.2 % = 88 % transformerio**.
Įdomiausia, kad **tas pats pagerinimas perėjo ir į gryną LGN** (visai be attention, su
token-shift) — **43.5 % = 80 % transformerio**. T.y. tai buvo realus FFN-pakaitalo
pagerinimas, ne attention triukas. Visi vėliau siūlyti įvesties/nuskaitymo priedai (conv1d,
LloydMax, TopK ir kt.) nedavė nieko, ir žemiau paaiškinu kodėl.

![accuracy comparison](results/figs/report/fig1_headline.png)

| Modelis | Accuracy % | % transformerio |
|---|---:|---:|
| NanoGPT transformeris (lubos) | **54.87** | 100 |
| **Attention + LGN-FFN (optimizuotas)** | **48.18** | **88** |
| **Grynas LGN (token-shift, be attention)** | **43.54** | **80** |
| Bazinis LGN-FFN | 35.35 | 64 |
| Identity vartai (logika išjungta) | 26.46 | 48 |
| Tik attention (FFN pašalintas) | 5.46 | 10 |

---

## Ką pasiekėme: LGN kaip FFN pakaitalas

Kad atskirčiau *cross-token* (attention) ir *per-token* (FFN) darbą, padariau švarų
eksperimentą: užšaldau ištreniruotą attention visuose 12 sluoksnių ir leidžiu LGN pakeisti
**tik FFN**. Klausimas grynas — *kiek gerai LGN imituoja FFN, kai attention idealus?* Bazinis
variantas duoda 35.4 %, ir nuo čia ieškojau, kas šį skaičių realiai kelia.

Pirmas dalykas, kurį teko atmesti — preciziją. Galvojau, kad bottleneck'as yra binarizacijos
tikslumas, bet ne: 8-bit įvestis ≈ 16-bit, o protingesnis nuskaitymas (`weighted_pool`) nedavė
nieko. **Atotrūkį riboja skaičiavimo talpa — vartų kiekis ir galia — o ne kodavimas.** Štai
kas tikrai veikia:

![what moves the metric](results/figs/report/fig2_levers.png)

| Svertas | Efektas | Tipas |
|---|---|---|
| **out_gate_mult** (daugiau išvesties vartų) | 35.4 → 38.3 → **41.8 %** (1×→2×→4×) | talpa (stipriausias) |
| **LUT-K aritetas** (k-input vartas vietoj 2-input) | +0.9 pp prie vienodo vartų kiekio | vartų galia |
| **CAGE / STE** (hard-forward) | uždaro soft–hard gap'ą | treniravimas |
| **Best-hard checkpoint** | +0.8 pp | treniravimas |
| **KL distiliacija** iš transformerio | +0.5 pp | treniravimas |

Sudėjus šiuos svertus (out_gate_mult 8 + k16 + LUT4 + CAGE + best-hard + KL) bazinis 35.4 %
pakyla iki **48.2 %**.

### Ablation testas: ar vartai išvis ką nors daro?

Geriausias būdas savęs nepergudrauti — pažiūrėti, ką pasiekia ta pati „instaliacija" be
logikos. Įjungiu identity vartus (ln + pooling + residual + užšaldytas attention išlieka, bet
logika nieko neskaičiuoja) ir gaunu tik 26.5 %. Vadinasi **išmokta logika prideda +21.7 pp** —
tai ne pooling'o ar residual'o nuopelnas, vartai tikrai dirba.

![ablation test](results/figs/report/fig3_honesty.png)

### Kur sėdi sunkumas: L0

Pakeitus po vieną FFN matosi aiški hierarchija. **L0 sunkiausias** — maždaug ×6 už bet kurį
kitą; vidurio FFN (L1–L6) beveik nemokami (juos pakeitus val loss net *pagerėja*), o pabaigos
sluoksniai vėl pasunkėja. Todėl papildomą vartų talpą skiriu ne tolygiai, o būtent
L0/L9/L10/L11.

![per-layer difficulty](results/figs/report/fig6_per_layer.png)

---

## Pagerinimas perėjo ir į gryną LGN (be attention)

Svarbiausias man patikrinimas buvo, ar šie svertai veikia tik su realiu attention, ar
persikelia ir į **visiškai attention-free** variantą. Grynam LGN cross-token sprendžiu pigiai —
**token-shift**: prie kiekvienos pozicijos pridedu kelias praėjusias (fiksuotas postūmis,
jokių mokomų parametrų). Pritaikęs tą pačią optimizuotą vartų konfigūraciją + token-shift,
gaunu:

> **Grynas LGN = 43.5 % = 80 % transformerio.** Net attention pakeitus pigiu postūmiu,
> optimizacija išlieka — taigi pagerinau būtent FFN-pakaitalą, ne attention.

Atotrūkis tarp 88 % (su attention) ir 80 % (grynas) yra attention-vs-token-shift cross-token
kaina — atskira problema nuo per-token FFN darbo.

---

## Kodėl pasiūlyti priedai nesuveikė

Po šių rezultatų buvo pasiūlyta dar keletas įvesties/nuskaitymo/interconnect idėjų. Įdiegiau ir
patikrinau jas visas (L0 screen, hard_degradation, mažiau = geriau, ref 0.074), kiekvieną su
ablation kontrolėmis. **Nė viena patikimai nepralenkė bazės** — ir tai logiška: jos visos taiko
ne į tą dimensiją (talpą), o į kodavimą, nuskaitymą ar sujungimą.

![what did not work](results/figs/report/fig4_screen.png)

| Pasiūlymas | Rezultatas | **Kodėl nesuveikė** |
|---|---|---|
| **Conv1D prieš/po LGN** | neutralu (s1), blogiau su stride | ablation parodė: užšaldžius vartus rezultatas tas pats → **conv perima darbą, ne LGN (fake LGN)**; o stride-adapteris dar ir praranda informaciją |
| **LloydMax binarizatorius** (per-channel EMA Gaussian thresholds) | neutralu | binarizacijos precizija **nebuvo** bottleneck'as — sigmoid-thermometer ir taip pakankamas; problema talpoje, ne kodavime |
| **TopK block-sparse interconnect** | blogiau | struktūrinis blokinis pasirinkimas **prastesnis** nei paprasta random kandidatų „loterija" vartų įvestims |
| **pool_curve** (mokoma netiesinė nuskaitymo kreivė) | blogiau | soft pagerėja, bet **nediskretizuojasi** — kreivė persimoko ant trupmeninių soft skaičių, o hard sprogsta |
| **residual_scale** (per-channel α prie LGN išvesties) | neutralu | α tiesiog lieka ≈1; nėra ko pridėti |
| **ensemble** (N vartų bankų vidurkis) | **triukšmas** | „laimėjimas" −0.013 prie vieno seed'o **apsiverčia** į +0.009 prie kito → telpa į screen'o triukšmą |

⚠️ **Metodologinė pastaba.** ensemble iš pradžių atrodė kaip pirmas tikras laimėjimas, bet
seed-pakartojimas parodė, kad ženklas apsiverčia. Būtent dėl to nė vieno kandidato neskelbiu be
seed-kontrolės — kitaip būčiau pranešęs nesamą pagerinimą.

Bendras vardiklis paprastas: likęs atotrūkis yra **fundamentalus** (kvantuota reta logika vs
tankus float FFN). Jį judina tik talpa, vartų galia ir treniravimas — ne įvesties ar nuskaitymo
patobulinimai.

---

## Efektyvumas (grynas LGN kelias)

Grynas LGN (be attention) yra gerokai efektyvesnis — kelis kartus mažiau parametrų ir 8–30×
mažiau FLOPs. Tikrasis pranašumas realizuojamas FPGA/ASIC, kur kiekvienas 2-input vartas = 1
LUT, o LUT6 = vienas natūralus vienetas. Būtent dėl to grynas variantas, į kurį perėjo
pagerinimas, man įdomiausias praktiškai.

![efficiency](results/figs/report_en/05_efficiency.png)

> 48.2 % skaičius laiko pilną float attention visur, tad efektyvumo pranašumas galioja **grynam
> LGN** keliui (43.5 %), ne attention+LGN-FFN.

---

## Galutinis palyginimas

Sudėjus viską į vieną vietą, susidaro trys aiškūs taškai tikslumo/efektyvumo kreivėje:

| Modelis | Accuracy % | % transformerio | Params | FLOPs vs transf. | Kada rinktis |
|---|---:|---:|---:|---:|---|
| Transformeris | 54.87 | 100 | 2.45 M | 1× | kai svarbu tik tikslumas |
| Attention + LGN-FFN | 48.18 | 88 | ≈ pilnas attention | ~1× | geriausias per-token tikslumas; bet attention dominuoja, efektyvumo naudos nėra |
| **Grynas LGN (token-shift)** | 43.54 | 80 | ~0.96 M | **~10× mažiau** | geriausias tikslumo/efektyvumo balansas (FPGA/ASIC taikinys) |

Išvada: **LGN realiai gali atlikti per-token FFN darbą** (ablation patvirtina +21.7 pp), o jį
optimizavęs pasiekiau 88 % transformerio su attention ir 80 % visai be jo. Likęs atotrūkis
nėra kodavimo ar nuskaitymo reikalas — jis fundamentalus, ir vienintelis svertas su likusiu
užtaisu yra **talpa** (atminties-efektyvūs LUT, kad tilptų didesni vartai be OOM). Tai ir yra
kita kryptis.
