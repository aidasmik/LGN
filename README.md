# LGN-Nano: Logic Gate Networks transformerio sluoksniuose

Projektas tiria, kiek transformerio (nanoGPT) galima pakeisti diferencijuojamais **Logic Gate Networks (LGN)** — tinklais iš mokomų Boolean vartų (AND, XOR ir t.t.) vietoj float matricų daugybų. Motyvacija — efektyvumas: LGN natūraliai mapinasi į hardware (vienas vartas ≈ 1 FPGA LUT), tad jei jie sugeba atlikti transformerio darbą, gaunama didelė nauda inference greičio ir energijos prasme. Setup'as visur tas pats — nanoGPT (12 sluoksnių × 128d × 4 heads), byte-level WikiText-2. Metrika — next-byte top-1 accuracy ant fiksuoto val batch'o; LGN visada matuojamas kaip **hard** (diskretus) modelis, kaip realiame inference.

Šis etapas sutelktas į vieną kryptį — LGN kaip **FFN replacement**, paliekant attention. Tikslas izoliuoti per-token darbą: attention užšaldomas visuose 12 sluoksnių, o į LGN keičiamas tik FFN. Taip galima grynai įvertinti, kiek gerai LGN atlieka FFN darbą, kai attention idealus. Baseline buvo tik 35.4% acc (transformerio 54.87%), o po optimizacijos pasiekta 48.18%, arba 88% transformerio. Svarbiausia — tas pats pagerinimas išsilaikė ir variante be attention (token shift): 43.54%, arba 80% transformerio.

## Modelių apžvalga

![bendras palyginimas](results/figs/report/fig1_headline.png)

| Modelis | Accuracy % | % transf. | Ką keičia |
|---|---:|---:|---|
| NanoGPT transformeris | **54.87** | 100 | 12 sluoksnių baseline (lubos) |
| **Attention + LGN-FFN (optimized)** | **48.18** | 88 | užšaldytas attention visur, FFN→LGN, optimizuoti gates |
| **Pure LGN (token shift)** | **43.54** | 80 | jokio attention; token-shift + tie patys optimizuoti gates |
| Baseline LGN-FFN | 35.35 | 64 | FFN→LGN be optimizacijų |
| Identity gates | 26.46 | 48 | logika išjungta (kontrolė) |
| Tik attention (be FFN) | 5.46 | 10 | grindys |

## LGN optimizacija

Trumpai, kaip LGN keičia FFN: gaunama kanalo aktyvacija, ji binarizuojama (kiekvienas iš 128 kanalų suspaudžiamas sigmoid'u ir paverčiamas `n_bits` thermometer bitais), paleidžiama per learned gates stack'ą, o output gates per kanalą sugrupuojami ir su sum_pool nuskaitomi atgal į float (suskaičiuojami vienetukai grupėje). Klausimas — kur šitame kelyje sėdi bottleneck.

Pirma atmesta **precision**. Prielaida buvo, kad riboja binarizacijos tikslumas, bet ne: 8-bit įvestis duoda tą patį kaip 16-bit (perpus mažiau bitų — tas pats acc), o weighted_pool (mokami per-bitiniai readout svoriai) nepridėjo nieko. Riba — gryna **capacity** (kiek gates ir kokie galingi), ne kodavimas.

![svertai](results/figs/report/fig2_levers.png)

Stipriausias svertas — **gate count output'e**. sum_pool nuskaito grupę gates į vieną skaičių, tad daugiau gates grupėje reiškia daugiau galimų lygių kanalui (didesnė readout rezoliucija). Vien tai duoda 35.4 → 38.3 → 41.8% (1× → 2× → 4×). Capacity dedama netolygiai: globaliai 4×, o sunkiausiems sluoksniams (L0, L9, L10, L11) — 8×.

Antras svertas — **gate arity (LUT-K)**. 2-input gate yra silpniausias primityvas (16 funkcijų iš dviejų bitų). Vietoj jo naudojamas k-input LUT gate: mokoma 2^K-įrašų truth table, soft treniruojant įvertinama per multilinear extension, hard'e snap'inasi į vieną FPGA LUT-K. Ant L0 LUT4 ≈ 2× daugiau 2-input gates, LUT6 ≈ ~2.7×; visame modelyje efektas kuklesnis (~+0.9 pp), nes nauda susikoncentruoja sunkiuose sluoksniuose.

Kodėl būtent L0 ir paskutiniai: pakeitus po vieną FFN matosi aiški hierarchija — L0 sunkiausias (~×6 už bet kurį kitą), vidurio FFN (L1–L6) beveik nemokami (juos pakeitus val loss net pagerėja), o pabaiga (ypač L11) vėl sunkesnė.

![sluoksnių sunkumas](results/figs/report/fig6_per_layer.png)

Likę svertai — **training**, nedidinantys gate count. Diskretizacija sukuria soft–hard gap'ą (treniruojama soft, inference hard). Jį tvarko STE (forward diskretus, backward gradientas pro jį tarsi tolydus) ir CAGE (forward kietas, argmax kaip inference; backward minkštas su adaptyvia temperatūra — gap'as sumažėja maždaug perpus). Dar pasirenkamas geriausias checkpoint pagal hard rezultatą, ne soft (~+0.8 pp), ir LGN mokomas atkartoti visą transformerio output distribution, ne tik teisingą byte'ą (KL distillation, ~+0.5 pp). Connections irgi svarbu — kiekvienas gate renkasi iš k kandidatinių input laidų (k=16).

Viskas suvedama nuosekliai: pirma kiekvienas LGN sluoksnis warm-startinamas imituojant originalų FFN (MSE), tada visas fine-tune'inamas kartu, o sluoksniai keičiami po vieną — lengviausias pirma, kad likęs tinklas spėtų prisitaikyti. Sudėjus visus svertus (out_gate_mult 8 + k16 + LUT4 + CAGE + best-hard + KL), 35.4% pakilo iki 48.2%.

## Ablation testas

Su tiek aplinkinių komponentų (ln, pooling, residual, užšaldytas attention) lengva apsigauti, kad pagerinimą duoda jie, o ne pati logika. Todėl visa instaliacija paliekama, bet gates išjungiami (identity) — toks variantas pasiekia tik 26.5%, vadinasi LGN realiai prideda **+21.7 pp**. Darbą atlieka gates, ne pooling ar residual.

![ablation](results/figs/report/fig3_honesty.png)

## Pure LGN su token shift

Svarbu patikrinti, ar svertai veikia tik su attention, ar ir be jo. Be attention cross-token sprendžiamas pigiai — token-shift'u (prie kiekvienos pozicijos pridedamos kelios praėjusios, jokių mokomų parametrų). Su token-shift vietoj attention tas pats optimizuotas LGN pasiekia 43.54%, arba 80% transformerio. Tai rodo, kad pagerintas pats FFN-pakaitalas, o ne pasinaudota attention'u.

## Palyginimas su ankstesniais aproachais

Ankstesnėje fazėje LGN kaip FFN buvo treniruojamas **be jokio attention** (grynas variantas), o cross-token sprendžiamas token-shift'u. Dabartinė gate-optimizacija tą patį grynąjį kelią aiškiai pajudino:

| Grynas LGN (be attention) | Acc % (anksčiau) | Acc % (po optimizacijos) |
|---|---:|---:|
| be cross-token (aggressive) | 27.22 | — |
| token-shift K=2 | 36.22 | **43.54** |

Tiesioginis palyginimas — tas pats token-shift pure LGN: **36.22 → 43.54 (+7.3 pp)** vien iš gate-optimizacijos, nieko nepridėjus prie cross-token dalies. Tas pats optimizuotas grynas LGN (43.54, be attention) net pralenkia ankstesnius variantus, kurie laikydavo dalį transformerio — Hybrid L0 (33.5, paliktas tik L0 attention) ir Selective (39.01, palikti 4 pilni transformer sluoksniai).

Attention + LGN-FFN (48.18) yra atskiras isolation eksperimentas: jame attention užšaldomas ne tik L0, o visuose 12 sluoksnių, kad švariai išmatuotų patį FFN→LGN darbą. Būtent ten rastos gate'ų idėjos ir persikėlė atgal į grynąjį (be attention) variantą.

## Kitų pasiūlymų testavimas

Lygiagrečiai išbandyti ir kiti pasiūlymai — Conv1D, LloydMax binarizer ir TopK block-sparse interconnect. Pilnas run'as (visi 12 sluoksnių, greedy scaling) trunka 12+ valandų, tad testuota efektyviai ir su griežtomis kontrolėmis: paleidžiama tik ant L0 (sunkiausio sluoksnio, kur efektas turėtų būti ryškiausias, nors L0 linkęs gain'us perdėti); kiekvienam daroma ablation control — gates užšaldomi, kad matytųsi, ar improvement'ą duoda LGN ar pats priedas; o jei kažkas atrodo geriau, kartojama su kitu seed, kad realus pagerinimas atsiskirtų nuo triukšmo.

![patikrinti priedai](results/figs/report/fig4_screen.png)

Nė vienas priedas patikimai nepralenkė baseline'o — dėl tos pačios priežasties: visi taiko ne į modelio capacity, o į encoding, readout ar interconnect, t.y. ne į tą dimensiją, kuri yra bottleneck.

| Bandymas | Rezultatas | Kodėl |
|---|---|---|
| Conv1D (prieš/po LGN) | neutralu, blogiau su stride | ablation: convolution perima darbą, ne LGN (fake LGN) |
| LloydMax binarizer | nieko | binarizacijos precision ir taip nebuvo problema |
| TopK block-sparse interconnect | blogiau | prastesnis nei paprastas random gate candidate parinkimas |
| Gate ensemble | triukšmas | tariamas gain'as apsivertė vos pakeitus seed |

Seed-check vertė geriausiai matosi su gate ensemble: iš pradžių atrodė kaip realus pagerinimas (−0.013), bet su kitu seed ženklas apsivertė (+0.009). Be seed-kontrolės tai būtų pasirodę kaip realus rezultatas.

![seed-check](results/figs/report/fig5_ensemble_noise.png)

## Efektyvumas

Grynas LGN (be attention) gerokai efektyvesnis — pure LGN variantas pasiekia 80% transformerio acc su ~2.5× mažiau params (2.45 M → ~0.96 M) ir ~10× mažiau FLOPs. Su attention variantu acc aukštesnis (88%), bet efficiency naudos nėra (attention dominuoja compute). Realių hardware skaičių nėra — ant GPU LGN visada lėtesnis, nes GPU optimizuotas tankiai matricų daugybai; pranašumas realizuojamas FPGA/ASIC, kur vartas = LUT.

![efektyvumas](results/figs/report_en/05_efficiency.png)

## Apibendrinimas

Liko ~20% gap'as tarp transformerio ir LGN, kurio paprastais architektūros priedais užpildyti nepavyko — jis atrodo fundamentalus (kvantuota sparse logika vs dense float FFN). Kita vertus, 20% skirtumas su ~2.5× mažiau parametrų nėra blogas rezultatas. Didžiausias šuolis greičiausiai būtų RDDLGN kryptyje (stateful gates vietoj attention), bet tai jau didelis architektūros pertvarkymas.
