## Modelių apžvalga

![bendras palyginimas](results/figs/report/fig1_headline.png)

| Modelis | Accuracy % | % transf. | Ką keičia |
|---|---:|---:|---|
| NanoGPT transformeris | **54.87** | 100 | 12 sluoksnių baseline |
| **Attention + LGN-FFN (optimized)** | **48.18** | 88 | užšaldytas attention visur, FFN i LGN, optimizuoti gates |
| **Pure LGN (token shift)** | **43.54** | 80 | no attention; token-shift + tie patys optimizuoti gates |
| Baseline LGN-FFN | 35.35 | 64 | FFN i LGN be optimizacijų |
| Identity gates | 26.46 | 48 | logika išjungta (kontrolė) |
| Tik attention (be FFN) | 5.46 | 10 | grindys |

## LGN optimizacija

Trumpai, kaip LGN keičia FFN: gaunama kanalo aktyvacija, ji binarizuojama (kiekvienas iš 128 kanalų suspaudžiamas sigmoid'u ir paverčiamas `n_bits` thermometer bitais), paleidžiama per learned gates stack'ą, o output gates per kanalą sugrupuojami ir su sum_pool nuskaitomi atgal į float. Klausimas - kur šitame kelyje sėdi bottleneck.

Pirma atmesta **precision**. Prielaida buvo, kad riboja binarizacijos tikslumas, bet ne: 8-bit įvestis duoda tą patį kaip 16-bit, o weighted_pool (mokami per-bitiniai readout svoriai) nepridėjo nieko. Riba - gryna **capacity** (kiek gates ir kokie galingi), ne kodavimas.

![kas labiausiai padeda](results/figs/report/fig2_levers.png)

Daugiausiai padeda **gate count output'e**. sum_pool nuskaito grupę gates į vieną skaičių, tad daugiau gates grupėje reiškia daugiau galimų lygių kanalui. Vien tai duoda 35.4 -> 38.3 -> 41.8% (1x -> 2x -> 4x). Capacity dedama netolygiai: globaliai 4x, o sunkiausiems sluoksniams (L0, L9, L10, L11) 8x.

Kitas dalykas - **gate arity (LUT-K)**. 2-input gate yra silpniausias įmanomas vartas (16 funkcijų iš dviejų bitų). Vietoj jo naudojamas k-input LUT gate: mokoma 2^K-įrašų truth table, soft treniruojant įvertinama per multilinear extension, hard'e snap'inasi į vieną FPGA LUT-K. Ant L0 LUT4 prilygsta maždaug 2x daugiau 2-input gates, LUT6 maždaug 2.7x; visame modelyje efektas kuklesnis (apie +0.9 pp), nes nauda susikoncentruoja sunkiuose sluoksniuose.

Pakeitus po vieną FFN aiškiai matosi, kurie sluoksniai sunkesni - L0 sunkiausias (apie 6x už bet kurį kitą), vidurio FFN (L1-L6) beveik nemokami (juos pakeitus val loss net pagerėja), o pabaiga (ypač L11) vėl sunkesnė.

![sluoksnių sunkumas](results/figs/report/fig6_per_layer.png)

Likę pagerinimai - **training** pusėje, nedidinantys gate count. Diskretizacija sukuria soft-hard gap'ą (treniruojama soft, inference hard). Jį tvarko STE (forward diskretus, backward gradientas pro jį tarsi tolydus) ir CAGE (forward kietas, argmax kaip inference; backward minkštas su adaptyvia temperatūra, gap'as sumažėja maždaug perpus). Dar pasirenkamas geriausias checkpoint pagal hard rezultatą, ne soft (apie +0.8 pp), ir LGN mokomas atkartoti visą transformerio output distribution, ne tik teisingą byte'ą (KL distillation, apie +0.5 pp). Connections irgi svarbu - kiekvienas gate renkasi iš k kandidatinių input laidų (k=16).

Viskas suvedama nuosekliai: pirma kiekvienas LGN sluoksnis warm-startinamas imituojant originalų FFN (MSE), tada visas fine-tune'inamas kartu, o sluoksniai keičiami po vieną - lengviausias pirma, kad likęs tinklas spėtų prisitaikyti. Sudėjus visus šiuos pagerinimus (out_gate_mult 8 + k16 + LUT4 + CAGE + best-hard + KL), 35.4% pakilo iki 48.2%.

## Ablation testas


![ablation](results/figs/report/fig3_honesty.png)

## Pure LGN su token shift

e attention cross-token sprendžiamas pigiai - token-shift'u (prie kiekvienos pozicijos pridedamos kelios praėjusios, jokių mokomų parametrų). Su token-shift vietoj attention tas pats optimizuotas LGN pasiekia 43.54%, arba 80% transformerio. Tai rodo, kad pagerintas pats FFN-pakaitalas, o ne pasinaudota attention'u.

## Palyginimas su ankstesniais aproachais

Ankstesnėje fazėje LGN kaip FFN buvo treniruojamas **be jokio attention** (grynas variantas), o cross-token sprendžiamas token-shift'u. Dabartinė gate-optimizacija tą patį grynąjį kelią aiškiai pajudino:

| Grynas LGN (be attention) | Acc % (anksčiau) | Acc % (po optimizacijos) |
|---|---:|---:|
| be cross-token (aggressive) | 27.22 | - |
| token-shift K=2 | 36.22 | **43.54** |

Tiesioginis palyginimas - tas pats token-shift pure LGN: **36.22 -> 43.54 (+7.3 pp)** vien iš gate-optimizacijos, nieko nepridėjus prie cross-token dalies. Tas pats optimizuotas grynas LGN (43.54, be attention) net pralenkia ankstesnius variantus, kurie laikydavo dalį transformerio - Hybrid L0 (33.5, paliktas tik L0 attention) ir Selective (39.01, palikti 4 pilni transformer sluoksniai).

Attention + LGN-FFN (48.18) yra atskiras isolation eksperimentas: jame attention užšaldomas ne tik L0, o visuose 12 sluoksnių, kad švariai išmatuotų patį FFN-i-LGN darbą. Būtent ten rastos gate'ų idėjos ir persikėlė atgal į grynąjį (be attention) variantą.

## Kitų pasiūlymų testavimas

Lygiagrečiai išbandyti ir kiti pasiūlymai - Conv1D, LloydMax binarizer ir TopK block-sparse interconnect. Pilnas run'as (visi 12 sluoksnių, greedy scaling) trunka 12+ valandų, tad testuota efektyviai ir su griežtomis kontrolėmis: paleidžiama tik ant L0 (sunkiausio sluoksnio, kur efektas turėtų būti ryškiausias, nors L0 linkęs gain'us perdėti); kiekvienam daroma ablation control - gates užšaldomi, kad matytųsi, ar improvement'ą duoda LGN ar pats priedas; o jei kažkas atrodo geriau, kartojama su kitu seed, kad realus pagerinimas atsiskirtų nuo triukšmo.

![patikrinti priedai](results/figs/report/fig4_screen.png)

Nė vienas priedas patikimai nepralenkė baseline'o - dėl tos pačios priežasties: visi taiko ne į modelio capacity, o į encoding, readout ar interconnect.

| Bandymas | Rezultatas | Kodėl |
|---|---|---|
| Conv1D (prieš/po LGN) | neutralu, blogiau su stride | ablation: convolution perima darbą, ne LGN (fake LGN) |
| LloydMax binarizer | nieko | binarizacijos precision ir taip nebuvo problema |
| TopK block-sparse interconnect | blogiau | prastesnis nei paprastas random gate candidate parinkimas |
| Gate ensemble | triukšmas | tariamas gain'as apsivertė vos pakeitus seed |

Seed-check vertė geriausiai matosi su gate ensemble: iš pradžių atrodė kaip (-0.013), bet su kitu seed ženklas apsivertė (+0.009)

![seed-check](results/figs/report/fig5_ensemble_noise.png)

## Efektyvumas

Grynas LGN (be attention) gerokai efektyvesnis - pure LGN variantas pasiekia 80% transformerio acc su maždaug 2.5x mažiau params (2.45 M -> apie 0.96 M) ir maždaug 10x mažiau FLOPs. Su attention variantu acc aukštesnis (88%), bet efficiency naudos nėra.
![efektyvumas](results/figs/report_en/05_efficiency.png)

## Apibendrinimas

Liko maždaug 20% gap'as tarp transformerio ir LGN, kurio paprastais architektūros priedais užpildyti nepavyko - jis atrodo fundamentalus (kvantuota sparse logika vs dense float FFN). Kita vertus, 20% skirtumas su maždaug 2.5x mažiau parametrų nėra blogas rezultatas.
