# LGN-Nano: Logic Gate Networks transformerio sluoksniuose

Tiriu, kiek transformerio (nanoGPT) galima pakeisti diferencijuojamais **Logic Gate Networks (LGN)** — tinklais iš mokomų Boolean vartų (AND, XOR ir t.t.) vietoj float matricų daugybų. Motyvacija — efektyvumas: LGN natūraliai mapinasi į hardware (kiekvienas vartas ≈ 1 FPGA LUT), tad jei jie sugeba atlikti transformerio darbą, tai duotų didelę naudą inference greičio ir energijos prasme. Visur naudoju tą patį setup'ą — nanoGPT (12 sluoksnių × 128d × 4 heads), byte-level WikiText-2, o metrika visur next-byte top-1 accuracy ant fiksuoto val batch'o; LGN visada matuoju kaip **hard** (diskretų) modelį, lygiai kaip realiame inference.

Šiame etape sutelkiau dėmesį į vieną kryptį — optimizuoti LGN kaip **FFN replacement**, paliekant attention. Idėja izoliuoti per-token darbą: užšaldau jau ištreniruotą attention visuose 12 sluoksnių ir keičiu tik FFN į LGN, tad galiu klausti grynai, kiek gerai LGN atlieka FFN darbą, kai attention idealus. Baseline'as buvo tik 35.4% acc (transformerio 54.87%), bet galiausiai pavyko pasiekti 48.18%, arba 88% transformerio — ir, kas svarbiausia, tas pats pagerinimas išsilaikė net perėjus į variantą be attention (token shift), 43.54%, arba 80% transformerio.

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

## Kaip optimizavau LGN

Trumpai, kaip LGN keičia FFN: gauna kanalo aktyvaciją, ją binarizuoja (kiekvienas iš 128 kanalų suspaudžiamas sigmoid'u ir paverčiamas `n_bits` thermometer bitais), paleidžia per learned gates stack'ą, ir output gates per kanalą sugrupuoja bei nuskaito atgal į float su sum_pool (suskaičiuoja vienetukus grupėje). Visa optimizacija — apie tai, kur šitame kelyje sėdi tikrasis bottleneck.

Pirmas dalykas, kurį teko atmesti — **precision**. Galvojau, kad riboja binarizacijos tikslumas, bet ne: 8-bit įvestis duoda praktiškai tą patį kaip 16-bit (perpus mažiau bitų — tas pats acc), o weighted_pool (mokami per-bitiniai readout svoriai, iki 2^g lygių vietoj g+1) nepridėjo nieko. Vadinasi riba yra gryna **capacity** — kiek gates ir kokie jie galingi — ne kodavimas.

![svertai](results/figs/report/fig2_levers.png)

Stipriausias svertas — **gate count output'e**. sum_pool nuskaito grupę gate output'ų į vieną skaičių, tad kuo daugiau gates grupėje, tuo daugiau galimų lygių vienam kanalui (didesnė readout rezoliucija). Vien padidinus output gates gaunu 35.4 → 38.3 → 41.8% (1× → 2× → 4×). Capacity dedu netolygiai: globaliai 4×, o sunkiausiems sluoksniams (L0, L9, L10, L11) — 8×, nes likę sluoksniai tos talpos tiesiog nepanaudoja.

Antras svertas — **gate arity (LUT-K)**. 2-input gate yra silpniausias primityvas (tik 16 Boolean funkcijų iš dviejų bitų). Pakeičiau jį k-input LUT gate: vietoj fiksuoto rinkinio mokoma 2^K-įrašų truth table, soft treniruojant įvertinama per multilinear extension (tolydi K bitų interpoliacija), o hard'e snap'inasi į vieną diskrečią lentelę = vienas FPGA LUT-K. Ant sunkaus L0 LUT4 ≈ 2× daugiau 2-input gates, LUT6 ≈ ~2.7×; visame modelyje efektas kuklesnis (~+0.9 pp), nes nauda susikoncentruoja sunkiuose sluoksniuose.

Kodėl būtent L0 ir paskutiniai: pakeitęs po vieną FFN ir pamatavęs degradaciją, matau aiškią hierarchiją — **L0 sunkiausias** (apie ×6 už bet kurį kitą), vidurio FFN (L1–L6) beveik nemokami (juos pakeitus val loss net pagerėja), o pabaiga (ypač L11) vėl pasunkėja.

![sluoksnių sunkumas](results/figs/report/fig6_per_layer.png)

Likę svertai — **training**, nedidinantys gate count. Diskretizacija sukuria soft–hard gap'ą (treniruoju soft, inference hard), tad naudoju STE (straight-through estimator: forward diskretus, backward gradientą leidžiu pro jį tarsi tolydus) ir CAGE (forward kietas, argmax kaip inference, o backward minkštas su adaptyvia temperatūra — gap'ą sumažina maždaug perpus). Dar renkuosi geriausią checkpoint'ą pagal hard rezultatą, ne soft (~+0.8 pp), ir mokau LGN atkartoti ne tik teisingą byte'ą, bet ir visą transformerio output distribution (KL distillation, ~+0.5 pp). Connections irgi svarbu — kiekvienas gate renkasi iš k kandidatinių input laidų (k=16), kad nestrigtų prie blogo input'o.

Viskas suvedama nuosekliai: pirma kiekvieną LGN sluoksnį warm-startinu imituodamas originalaus FFN output'ą (MSE), tada fine-tune'inu viską kartu, o sluoksnius keičiu po vieną — lengviausią pirma, kad likęs tinklas spėtų prisitaikyti. Sudėjus visus svertus (out_gate_mult 8 + k16 + LUT4 + CAGE + best-hard + KL), 35.4% pakilo iki 48.2%.

## Ablation testas

Su tiek aplinkinių komponentų (ln, pooling, residual, užšaldytas attention) lengva apsigauti, kad pagerinimą duoda jie, o ne pati logika. Todėl palikau visą instaliaciją, bet išjungiau pačius gates (identity) — toks variantas pasiekia tik 26.5%, vadinasi išmoktas LGN realiai prideda **+21.7 pp**. Gates tikrai dirba, ne pooling ar residual.

![ablation](results/figs/report/fig3_honesty.png)

## Pure LGN su token shift

Svarbiausias patikrinimas — ar svertai veikia tik su realiu attention, ar persikelia ir be jo. Cross-token tada sprendžiu pigiai token-shift'u (prie kiekvienos pozicijos pridedu kelias praėjusias, jokių mokomų parametrų). Įdėjus tą patį optimizuotą LGN su token-shift vietoj attention, acc nukrito tik iki 43.54%, arba 80% transformerio. Tai, kad optimizacija išsilaikė ir be attention, rodo, jog pagerinau būtent FFN-pakaitalą, o ne pasinaudojau attention'u.

## Palyginimas su ankstesniais aproachais

Ankstesnėje fazėje LGN kaip FFN treniravau **be jokio attention** (grynas variantas), o cross-token spręsdavau token-shift'u. Dabartinė gate-optimizacija tą patį grynąjį kelią aiškiai pajudino:

| Grynas LGN (be attention) | Acc % (anksčiau) | Acc % (po optimizacijos) |
|---|---:|---:|
| be cross-token (aggressive) | 27.22 | — |
| token-shift K=2 | 36.22 | **43.54** |

Tiesioginis palyginimas — tas pats token-shift pure LGN: **36.22 → 43.54 (+7.3 pp)** vien iš gate-optimizacijos, nieko nepridėjus prie cross-token dalies. Tas pats optimizuotas grynas LGN (43.54, visai be attention) net pralenkia ankstesnius variantus, kurie dar laikydavo dalį transformerio — Hybrid L0 (33.5, paliktas tik L0 attention) ir Selective (39.01, palikti 4 pilni transformer sluoksniai).

O attention + LGN-FFN setup'as (48.18) yra atskiras isolation eksperimentas: jame attention užšaldau ne tik L0, o visuose 12 sluoksnių, kad švariai išmatuočiau patį FFN→LGN darbą. Būtent ten suoptimizuotos gate'ų idėjos ir persikėlė atgal į grynąjį (be attention) variantą.

## Kaip testavau kitus pasiūlymus

Lygiagrečiai išbandžiau ir kitus pasiūlymus — Conv1D, LloydMax binarizer ir TopK block-sparse interconnect. Pilnas run'as (visi 12 sluoksnių, greedy scaling) trunka 12+ valandų, tad testavau efektyviai ir su griežtomis kontrolėmis, kad nieko nepaskelbčiau per anksti: paleidžiu tik ant L0 (sunkiausio sluoksnio, kur efektas turėtų būti ryškiausias, nors L0 linkęs gain'us perdėti); kiekvienam darau ablation control — užšaldau pačius gates ir žiūriu, ar improvement'ą duoda LGN ar pats priedas; o jei kažkas atrodo geriau, pakartoju su kitu seed, kad atskirčiau realų pagerinimą nuo triukšmo.

![patikrinti priedai](results/figs/report/fig4_screen.png)

Nė vienas priedas patikimai nepralenkė baseline'o — ir, mano supratimu, dėl tos pačios priežasties: visi jie taiko ne į modelio capacity, o į encoding, readout ar interconnect, t.y. ne į tą dimensiją, kuri yra bottleneck.

| Bandymas | Rezultatas | Kodėl |
|---|---|---|
| Conv1D (prieš/po LGN) | neutralu, blogiau su stride | ablation: convolution perima darbą, ne LGN (fake LGN) |
| LloydMax binarizer | nieko | binarizacijos precision ir taip nebuvo problema |
| TopK block-sparse interconnect | blogiau | prastesnis nei paprastas random gate candidate parinkimas |
| Gate ensemble | triukšmas | tariamas gain'as apsivertė vos pakeitus seed |

Seed-check vertė geriausiai matosi su gate ensemble: iš pradžių atrodė kaip realus pagerinimas (−0.013), bet su kitu seed ženklas apsivertė (+0.009) — t.y. telpa į screen'o triukšmą. Būtent dėl tokių mirage'ų nieko nelaikau pagerinimu be seed-kontrolės.

![seed-check](results/figs/report/fig5_ensemble_noise.png)

## Efektyvumas

Grynas LGN (be attention) yra gerokai efektyvesnis — pure LGN variantas pasiekia 80% transformerio acc su ~2.5× mažiau params (2.45 M → ~0.96 M) ir ~10× mažiau FLOPs. Su attention variantu acc aukštesnis (88%), bet ten efficiency naudos nėra (attention dominuoja compute). Realių hardware skaičių nelyginau, nes ant GPU LGN visada veikia prasčiau — pranašumas realizuojamas FPGA/ASIC, kur vartas = LUT.

![efektyvumas](results/figs/report_en/05_efficiency.png)

## Kur esame

Vis dar liko ~20% gap'as tarp transformerio ir LGN, kurio jau nelabai turiu idėjų kaip kompensuoti — visi paprasti architektūros priedai patikrinti ir neduoda, o likęs gap'as atrodo fundamentalus (kvantuota sparse logika vs dense float FFN). Kita vertus, manau, kad 20% skirtumas su ~2.5× mažiau parametrų nėra blogas rezultatas. Didžiausias šuolis greičiausiai būtų RDDLGN kryptyje (stateful gates vietoj attention), bet tai jau didelis architektūros pertvarkymas.
