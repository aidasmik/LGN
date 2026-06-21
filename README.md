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

## Kaip optimizavau patį LGN

Kad atskirčiau *cross-token* (attention) ir *per-token* (FFN) darbą, padariau švarų
eksperimentą: užšaldau ištreniruotą attention visuose 12 sluoksnių ir leidžiu LGN pakeisti
**tik FFN**. Klausimas grynas — *kiek gerai LGN imituoja FFN, kai attention idealus?* Bazinis
variantas duoda 35.4 %, ir nuo čia ieškojau, kas šį skaičių realiai kelia.

### Kaip LGN keičia FFN

Kad būtų aišku, ką optimizuoju — LGN gauna sluoksnio aktyvaciją (128 kanalų float vektorių),
ją binarizuoja, paleidžia per loginių vartų stack'ą ir nuskaito atgal į 128 float kanalus:

1. **Binarizacija (įvestis).** Kiekvienas kanalas suspaudžiamas (sigmoid) ir paverčiamas į
   `n_bits` bitų termometru → 128 × n_bits įvesties bitų. Gradientui — STE (straight-through).
2. **Vartai.** Kiekvienas išvesties vartas išsirenka iš `k` kandidatinių įvesties laidų
   (softmax) ir vieną iš 16 Boolean funkcijų (irgi softmax). Soft treniruojant = svertinis
   mišinys; hard inference = argmax (viena laidų pora + viena funkcija). 2-input vartas:
   `g(A,B)=c0+c1A+c2B+c3AB`.
3. **Nuskaitymas (readout).** Išvesties vartai grupuojami po kanalą, `sum_pool` suskaičiuoja
   vienetukus grupėje → vienas skaičius kanalui. Būtent čia ir slypi pagrindinis talpos svertas.

### Diagnozė: talpa, ne precizija

Pirmas dalykas, kurį teko atmesti — preciziją. Galvojau, kad bottleneck'as yra binarizacijos
tikslumas, bet ne: 8-bit įvestis ≈ 16-bit (perpus mažiau įvesties bitų, tas pats rezultatas),
o protingesnis nuskaitymas (`weighted_pool`, iki 2^g lygių vietoj g+1) nedavė nieko. Išvada:
**atotrūkį riboja skaičiavimo talpa — vartų kiekis ir galia — o ne kaip tiksliai užkoduoju ar
perskaitau.** Visa likusi optimizacija nuo to ir atsispiria.

### Talpos svertai (stipriausi)

![what moves the metric](results/figs/report/fig2_levers.png)

| Svertas | Ką daro | Efektas |
|---|---|---|
| **out_gate_mult** | daugiau išvesties vartų kanalui → didesnė readout rezoliucija (sum_pool grupė platesnė) | 35.4 → 38.3 → **41.8 %** (1×→2×→4×) |
| **LUT-K aritetas** | 2-input vartą (silpniausias primityvas) keičia k-input LUT: mokoma 2^K-įrašų lentelė per multitiesį išplėtimą, hard'e = vienas FPGA LUT-K | +0.9 pp prie *vienodo* vartų kiekio |
| **Selektyvi talpa** | papildomus vartus deda tik į sunkius sluoksnius (L0/L9/L10/L11), ne tolygiai | talpa eina ten, kur reikia |

### Treniravimo svertai (be papildomų vartų)

| Svertas | Ką daro | Efektas |
|---|---|---|
| **CAGE / STE** | forward kietas (argmax, lygiai kaip inference), backward minkštas su adaptyvia temperatūra → soft–hard gap'as principe dingsta | uždaro gap'ą |
| **Best-hard checkpoint** | renka geriausią pagal *hard* val (tai, kas svarbu inference), ne pagal soft | +0.8 pp |
| **KL distiliacija** | mokosi atkartoti viso transformerio išvesties skirstinį, ne tik teisingą byte'ą | +0.5 pp |

### Kaip viskas suvedama (pipeline)

1. **Imitation** — kiekvienam sluoksniui pirma treniruoju LGN, kad MSE prasme atkartotų
   originalaus FFN išvestį (šiltas startas).
2. **Fine-tune** — tada derinu su LM loss (+ KL), CAGE/STE ir best-hard atranka; temperatūrą
   per fine-tune anneal'inu soft → hard.
3. **Greedy scaling** — keičiu sluoksnius po vieną, **lengviausią pirma** (pagal sunkumo
   heatmap'ą), kad likęs tinklas spėtų prisitaikyti prie kiekvieno naujo LGN sluoksnio.

Sudėjus visus svertus (out_gate_mult 8 + k16 + LUT4 + CAGE + best-hard + KL) bazinis 35.4 %
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
