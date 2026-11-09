# Protocol — tien leerbare exponentoffsets per MX-groep

## Onderzoeksvraag en gecontroleerde wijziging

Kan een ruimere keuze uit gehele exponentoffsets 0–9 een bruikbaar gezamenlijk E4M3/E2M3/E2M1-model leren? De eerdere binaire proef liet alleen offsets 5 en 6 toe. De geslaagde tienkeuzepilot heeft de technische werking en looptijd gecontroleerd. Deze nieuwe run behoudt exact hetzelfde recept, begint opnieuw en traint 100 epochs. Gewichten en schaalhulpwaarden leren gedurende de volledige training; er is geen bevriezing na 50.

De directe basis is het eerder geleverde pakket `mx_triple_learned_scale10_pilot5_v1`, voortbouwend op de binaire schaalproef. De zes MX-lagen gebruiken groepen van 32 langs de afgevlakte invoer/kernelas binnen ieder outputkanaal, met nul-padding van de staart. Er zijn 4.080 groepen. Eén FP32-theta per groep geeft 137.034 parameters: 132.954 oorspronkelijke parameters plus 4.080 schaalparameters.

Theta begint exact op 6, zonder RNG-trekking. De oorspronkelijke modelgewichten en RNG-toestand blijven daardoor dezelfde als in de binaire proef. De E4-bronschaal wordt volgens de bestaande maximumregel opnieuw bepaald; zij is geen zelfstandige leerparameter. Onafhankelijke schalen per exponentbreedte zijn geen onderdeel van dit experiment.

## Forward en harde keuze

Voor iedere groep wordt theta binnen [0,9] afgerond naar het dichtstbijzijnde gehele getal. Een halve waarde kiest de hogere offset. Drempels zijn 0,5;1,5;…;8,5. De implementatie bepaalt eerst de onderliggende gehele waarde en vergelijkt de fractie met 0,5; zo kan FP32-optelling van 0,5 een waarde vlak onder een drempel niet onbedoeld naar boven afronden.

| Offset | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Schaalfactor | ×1 | ×2 | ×4 | ×8 | ×16 | ×32 | ×64 | ×128 | ×256 | ×512 |

Per laag/minibatch wordt één E4M3-bron gecodeerd uit de oorspronkelijke voorbewerking `G(W)=tanh(W)/max(abs(tanh(W)))*detach(mean(abs(W)))`, inclusief de bestaande nulguard. E4M3 gebruikt dezelfde nearest/ties-to-even-elementafronding en E8M0-bronschaal.

| Gewichtsmodus | Elementbewerking | Effectieve schaal |
|---|---|---|
| E4M3 | oorspronkelijke broncode `s e3 e2 e1 e0 m2 m1 m0` | oorspronkelijke E8M0-schaal |
| E2M3 | behoud `s e3 e2 m2 m1 m0` | bronexponentcode plus gekozen offset 0–9 |
| E2M1 | E2M3-code als uint8 `>>2` | exact dezelfde E2-schaal |

Er wordt geen directe E2-encoder gebruikt. De twee lagere modi delen één harde keuze per groep. Alle drie passes gebruiken dezelfde master- en thetaversie; broncodes, keuzes en afgeleide codes blijven tijdens de passes onveranderd. De cache is leeg vóór Adam. Schaaloptelling gebruikt een breder integerdatatype. E8M0-code255 of hoger veroorzaakt FAIL, ook bij nulblokken; er is geen schaalclipping of uint8-wrap. Offset0 is toegestaan en behoudt voor die groep de bronschaal.

## Backward en optimizer

De bestaande identity-STE naar G(W) blijft gelijk. De mastergradient wordt niet met de gekozen schaalfactor vermenigvuldigd; de afgeleiden van G(W) blijven aanwezig.

De discrete keuzefunctie gebruikt de benadering `d(offset)/d(theta)≈1`. Voor de werkelijk gebruikte gedecodeerde lagere gewichten q geldt:

```text
grad_theta[b] = ln(2) * sum_real_elements_in_block(grad_q * q)
```

De E2M3- en E2M1-bijdragen worden op dezelfde theta gesommeerd. E4M3 heeft geen directe theta-gradient. Padding draagt niet bij; een volledig nulblok heeft nul schaalgradient. Teacheroutputs zijn detached. Er wordt geen LSQ-normalisatie, deling door groepsgrootte of exploratieruis toegevoegd. Deze surrogate-gradient is geen exacte afgeleide van de discrete keuze.

Er zijn twee Adam-parametergroepen, gewichten en theta, met dezelfde oorspronkelijke defaults, lr 0,001 en weight_decay 0. Na drie backwards volgt één Adam-update. Een post-step-hook projecteert theta vervolgens naar[0,9]. De gewichten krijgen geen extra update. De scheduler gebruikt dezelfde gamma 0,1 en milestones 50/75/90 voor beide groepen.

## Data, training en checkpoints

SVHN train-only: 73.257 trainbeelden, 26.032 officiële testbeelden; geen extra-split en geen aparte validatiesplit. Model `svhnq`, expand 2, bestaande 40×40-transforms, seed 20261009, batch 128, workers 0, drop_last=False. Activaties A32; masters, gradients en Adam-momenten FP32. De oorspronkelijke CUDA-backend wordt behouden.

De trainingsvolgorde is E4M3→E2M3→E2M1. E4M3 gebruikt hard CE. E2M3 gebruikt soft CE met de detached E4M3-softmax; E2M1 gebruikt de detached E2M3-softmax. Temperatuur 1; gesommeerde gradients zonder deling door drie. Na iedere epoch worden alle modi met hard CE op de volledige testset geëvalueerd, met bevroren evaluatiestaat. De drie bestaande BN-banken leren alleen tijdens hun eigen trainingsmodus mee.

De executable loopt standaard tot 100 volledige epochs: 573 updates per epoch,57.300 updates en 171.900 trainingspasses in totaal. De leersnelheid voor gewichten én theta is 0,001 in epochs 1–50, 0,0001 in 51–75, 0,00001 in 76–90 en0,000001 in 91–100. De dalingen volgen na 50/75/90; er wordt geen groep bevroren. Er is geen warmstart uit de pilot of een andere proef. Een expliciete stopgrens 1..100 of Ctrl+C kan de run op een volledige epochgrens onderbreken; het rapport blijft onder 100 onvolledig.

Alleen checkpoints van afgeronde epochs binnen dit nieuwe tienkeuze-honderd-epochcontract kunnen worden hervat; de training zelf mag nog niet voltooid zijn. Zij bevatten model, theta, beide Adam-groepen, scheduler, RNG en geschiedenis. Hervatten maakt een nieuwe sessiemap. De E4M3-selectie is het eerste strikte maximum van de testaccuracy. Beide lagere scores komen uit datzelfde checkpoint. Epoch100 wordt afzonderlijk naast het beste E4M3-checkpoint gerapporteerd zodra alle honderd epochs bevestigd zijn. Bij een onderbreking staat de laatste afgeronde epoch uitdrukkelijk als onvolledig eindpunt vermeld. Alle volledige checkpoints blijven behouden, inclusief50 met model, theta, Adam, scheduler, RNG en geschiedenis; epoch 50_checkpoint.json verwijst naar het beschikbare bestand en de hash. Een eventuele latere vriesproef kan daaruit de tak51–100 starten zonder de huidige tak met verder leren opnieuw uit te voeren.

## Diagnostiek en notebook

Per groep wordt theta, offset en factor opgeslagen op epoch 0 en elke voltooide epoch. Per laag/epoch worden alle tien aantallen en fracties, theta-min/max/gemiddelde/populatiestandaardafwijking, continue updates, schaalgradientnormen en projecties geregistreerd.

Een verandering van offset telt als één wissel; een sprong 5→7 heeft afstand 2. Totaal wissels, som van sprongafstanden, grootste sprong, unieke wisselende groepen en netto verschil tussen epochgrenzen zijn afzonderlijke metingen. Gekozen offset 0/9 wordt apart geteld van theta exact op 0/9. De notebook controleert deze gegevens tegen history en de volledige blokstaat.

De werkelijke binaire honderd-epochtraining met schaalleren tot 100 en de vaste ×32/×64-referenties worden meegeleverd voor de hoofdvergelijking. Volledige curves, epoch 100 en het beste gezamenlijke E4M3-checkpoint worden getoond. Bij een kortere huidige run worden referenties eerst beperkt tot dezelfde gemeten epochrange; hun beste checkpoint wordt daarbinnen gekozen. De uitgevoerde tienkeuzepilot wordt afzonderlijk op epoch 5 vergeleken en telt niet als volledige referentie. De notebook controleert de gepinde bronbestanden, wordt zonder outputs geleverd en leest alleen JSON/CSV. De volgende onafhankelijke schaalproef en een eventuele freeze50-tak vallen buiten dit pakket.

Lage accuracy, constante voorspellingen, nul maar eindige gradients of geen harde wissels zijn mogelijke onderzoeksuitkomsten, geen automatische technische FAIL. Eén run per variant en de oorspronkelijke CUDA-backend onderbouwen geen significantieclaim. De officiële testset wordt ook voor checkpointselectie gebruikt.

## Compacte exports en kosten

De flexibele bron bevat E4M3-codes, E8M0-schalen, noodzakelijke FP32-uitzonderingen/BN-staat en één vierbit-keuze per groep. De codes 0–9 staan rechtstreeks voor de exponentoffset; 10–15 zijn ongeldig. De groepen worden row-major afgevlakt. Keuze 2j staat in de lage nibble, keuze 2j+1 in de hoge nibble van een byte. Bij een oneven aantal groepen moet de ongebruikte hoge nibble nul zijn. De schema-identiteit verschilt van de oude éénbit-export.

Voor dit model vragen de 4.080 keuzes 2.040 bytes; de oude binaire keuzes vroegen 510 bytes. Het verschil is 1.530 bytes. In de bronexport ontbreken FP32-theta en masters van de zes MX-lagen. Lagere exports worden uitsluitend afgeleid uit de opgeslagen en opnieuw ingeladen bron plus keuzes. Hun gewichten worden exact vergeleken met de live route op het trainingsdevice; dit is geen nieuwe volledige testsetinferentie.

De analytische extra trainingstoestand van theta, gradient en twee FP32-Adam-momenten blijft 4.080×16=65.280 bytes, oftewel 63,75 KiB. Dit sluit tijdelijke buffers, activaties, optimizerstaptellers en allocatoroverhead uit. Gepackte bytes zijn geen `.pt`-bestandsgrootte of totale VRAM-meting. Een losse lagere uitvoerprecisie gebruikt alleen zijn effectieve E8M0-schaal; de keuze-index hoeft daar niet opnieuw te worden opgeslagen.

## Bronnen

De eerdere projectcode en gevalideerde codecs vormen de basis. De logaritmische schaalgradient is de eerder gekozen eigen surrogate voor exponenttruncatie; het is niet de LSQ-gradientformule. De representatieschalen blijven E8M0 volgens [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/). Het algemene idee van leren van quantisatieparameters sluit aan bij [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153).
