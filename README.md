# LGN-Nano: FFN keitimas į Logic Gate Network transformeryje

Išbandžiau aproachą — optimizuoti LGN kaip FFN replacement, paliekant attention sluoksnį. Šis aproachas davė labai gerų rezultatų, kurie perėjo net ir į variantą be palikto attention, naudojant token shift.

Idėja tokia: užšaldęs jau ištreniruotą attention visuose 12 sluoksnių, keičiau tik FFN į LGN. Baseline'as buvo tik 35.4% acc (transformerio 54.87%), bet galiausiai pavyko pasiekti 48.18%, 88% transformerio.

![bendras palyginimas](results/figs/report/fig1_headline.png)

Daugiausiai davė didesnis gate count kiekvienam output kanalui (tai tiesiog padidina, kiek info LGN gali sukrauti į vieną kanalą) ir galingesni gates, vietoj paprastų 2-input gates naudoju k-input LUT gates (LUT, lookup table, maža išmokstama tiesos lentelė, mapinanti K įvesties bitus tiesiai į output), kurie vienu vartu išmoksta sudėtingesnę funkciją. Capacity nedėjau tolygiai, o daugiau ten, kur sunkiausia (L0 ir paskutiniai sluoksniai), nes vidurinius LGN pakeičia beveik be nuostolio.

![sluoksnių sunkumas](results/figs/report/fig6_per_layer.png)

Prie to pridėjau keletą training patobulinimų, kurie nedidina gate count. Pirma — STE (straight-through estimator): forward einu diskrečiu keliu, o gradientą backward leidžiu pro jį tarsi būtų tolydus. Antra — CAGE: forward daromas kietas (argmax, lygiai kaip realiame inference), o backward skaičiuojamas minkštai su adaptyvia temperatūra, tad soft–hard gap praktiškai dingsta. Dar renkuosi geriausią checkpoint'ą pagal hard rezultatą, ir mokau LGN atkartoti ne tik teisingą atsakymą, bet ir visą transformerio output distribution (KL distillation). Pradedu nuo to, kad LGN warm-startina imituodamas originalų FFN, paskui fine-tune'inu viską kartu, o sluoksnius keičiu po vieną — lengviausią pirma, kad likęs tinklas spėtų prisitaikyti. Taip acc pakilo nuo 35.4% iki 48.2%.

![svertai](results/figs/report/fig2_levers.png)

Kad vėl nebūtų, jog kažkuri kita bloko dalis (pooling, residual ar pats attention) atlieka visą darbą, padariau ablation testą - išjungiau pačią logiką ir palikau tik likusią bloko dalį. Toks variantas duoda tik 26.5%, vadinasi išmoktas LGN realiai prideda +21.7 pp, t.y. gates tikrai veikia.

![ablation](results/figs/report/fig3_honesty.png)

Labai geras rezultatas gavosi ir tada, kai tą patį naujai optimizuotą LGN įdėjau į anksčiau naudotus sluoksnius su token shift vietoj attention. Net visai be attention acc nukrito tik iki 43.54%, arba 80% transformerio.

Lygiagrečiai išbandžiau ir kitus pasiūlymus — Conv1D, LloydMax binarizer ir TopK block-sparse interconnect. Testavau juos efektyviai, nes jau ir kiti testai trukdavo 12+ valandų: paleidžiu tik ant L0 (sunkiausio sluoksnio, kur efektas turėtų būti ryškiausias), kiekvienam dar darau ablation control, užšaldau pačius gates, kad pamatyčiau, ar pagerinimą duoda LGN ar pats priedas; o jei kažkas atrodo geriau, pakartoju su kitu seed, kad įsitikinčiau, jog tai ne atsitiktinumas.

Nė vienas patikimai nepralenkė baseline'o, ir, mano supratimu, dėl tos pačios priežasties, visi jie taiko ne į modelio talpą, o į encoding, readout ar interconnect.

![patikrinti priedai](results/figs/report/fig4_screen.png)

Trumpai: Conv1D iš pradžių atrodė labai gerai, bet ablation parodė, kad tuomet darbą atlieka pati convolution, o ne LGN; LloydMax nieko nedavė, nes binarizacijos precision ir taip nebuvo problema; TopK interconnect pasirodė net prastesnis nei paprastas random gate candidate parinkimas.

## Rezultatai ir kur esame

Trumpai apibendrinu. Pavyko parodyti, kad LGN realiai gali atlikti per-token (FFN) darbą — ne tik kompensuojamas aplinkinių sluoksnių — ir tą darbą gerokai suoptimizuoti. Galutiniai skaičiai:

| Modelis | Acc % | % transf. | Params | FLOPs vs transf. |
|---|---|---|---|---|
| NanoGPT transformeris (ceiling) | 54.87 | 100 | 2.45 M | 1× |
| Attention + LGN-FFN (optimized) | 48.18 | 88 | ≈ full attention | ~1× |
| Pure LGN (token shift, be attention) | 43.54 | 80 | ~0.96 M | ~10× fewer |
| Baseline LGN-FFN | 35.35 | 64 | — | — |
| Identity gates (control) | 26.46 | 48 | — | — |

![efektyvumas](results/figs/report_en/05_efficiency.png)

Performance prasme įdomiausias pure LGN variantas: 80% transformerio acc su ~2.5× mažiau params ir ~10× mažiau FLOPs. Su attention variantu acc aukštesnis (88%), bet ten efficiency naudos nėra.

Vis dar liko 20% gap'as tarp transformerio ir LGN, kurio jau nelabai turiu idėjų kaip galima būtų kompensuoti. Kita vertus, manau, kad 20% skirtumas, su ~2.5× mažiau parametrų, nėra blogas rezultatas.
