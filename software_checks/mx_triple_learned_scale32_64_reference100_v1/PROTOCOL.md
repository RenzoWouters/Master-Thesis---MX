# Protocol — honderd epochs met geleerde binaire schaalcorrectie per MX-blok

## Onderzoeksvraag

Kan een geleerde keuze tussen de reeds onderzochte correcties ×32 en ×64, afzonderlijk per MX-blok, een bruikbaar gezamenlijk E4M3/E2M3/E2M1-model opleveren? De afgeronde vijf-epochpilot bevestigt techniek en daadwerkelijk leren van de hulpwaarden. Deze verse honderd-epochrun onderzoekt het verdere trainingsverloop en de accuracies tegenover vaste ×32 en ×64. Het zoekdomein blijft beperkt tot die twee schaalopties; er wordt geen globale schaaloptimaliteit verondersteld.

## Forward

De zes MX-lagen en de bestaande groepering blijven gelijk: langs de afgevlakte invoer/kernelas binnen ieder uitvoerkanaal, met 32 elementen per blok en nul-padding voor een onvolledig blok. Er zijn 4.080 blokken. Eén FP32-parameter theta per blok geeft 4.080 extra parameters boven op de oorspronkelijke 132.954; het totaal is 137.034.

Theta begint op 6 zonder RNG-trekking. De oorspronkelijke netwerkgewichten en RNG-toestand blijven bij het toevoegen van theta gelijk. Omdat theta deel uitmaakt van de nieuwe modelstaat, zijn `initial_model_sha256` en `initial_base_model_sha256` verschillende begrippen: de tweede sluit alleen de zes theta-tensors uit en is geschikt voor vergelijking met de eerdere referenties.

Voor ieder blok geldt: theta < 5,5 kiest offset 5; theta >= 5,5 kiest offset 6. De daadwerkelijke forward gebruikt dus uitsluitend een factor 32 of 64. Tussenliggende theta-waarden zijn alleen trainingstoestand.

Per laag en minibatch wordt eenmaal de bestaande E4M3-bron uit de oorspronkelijke voorbewerking G(W) gecodeerd. Deze voorbewerking is `tanh(W)/max(abs(tanh(W)))*detach(mean(abs(W)))`, met de bestaande nulguard. De E4M3-encoder, nearest/ties-to-even, blokindeling en bronschaalbepaling blijven ongewijzigd.

| Modus | Elementbewerking | Schaal |
|---|---|---|
| E4M3 | oorspronkelijke broncode `s e3 e2 e1 e0 m2 m1 m0` | oorspronkelijke E8M0-bronschaal |
| E2M3 | behoud `s e3 e2 m2 m1 m0` | bronschaalcode plus harde blokoffset 5 of 6 |
| E2M1 | E2M3-code als uint8 `>> 2` | exact dezelfde lagere schaal als E2M3 |

De schaaloptelling gebruikt een breder integer-type; overflow naar E8M0-code 255 of hoger veroorzaakt FAIL, ook bij nulblokken. Er is geen clipping of uint8-wrap van de representatieschaal. Deze guard staat los van de bewuste projectie van theta na Adam.

De broncodes, harde keuzes en afgeleide codes zijn detached en blijven binnen de drie passes onveranderd. Codes, masters en theta-versies worden bewaakt. Er wordt geen directe E2-encoder gebruikt. De cache is leeg vóór de optimizerupdate; de voor backward noodzakelijke gedecodeerde gewichten blijven in de autograd-context beschikbaar.

## Backward

De bestaande identity-STE naar G(W) blijft intact voor alle drie modi. Er komt geen factor 32 of 64 op de mastergradient. De afgeleiden van G(W) blijven aanwezig.

De discrete offsetfunctie heeft vrijwel overal nulgradient. De nieuwe experimentele afspraak is `d(offset)/d(theta) ≈ 1`. Voor de werkelijk gedecodeerde lagere gewichten q resulteert dit in:

```text
grad_theta[b] = ln(2) * sum_i(grad_q[b,i] * q[b,i])
```

De som omvat uitsluitend echte elementen van het blok; padding levert geen bijdrage. De E2M3- en E2M1-bijdragen worden opgeteld op dezelfde theta. E4M3 heeft geen directe theta-gradient. De teacheroutputs blijven detached, zodat geen extra gradient via de teacher terugloopt.

Dit is de gekozen logaritmische STE voor deze offsetproef. Het is geen exacte afgeleide van de harde keuze en niet de LSQ-gradientformule. Er wordt geen aanvullende LSQ-normalisatie of deling door blokgrootte toegepast. Een volledig nulblok heeft volgens deze regel nul schaalgradient. Een richting uit de benadering garandeert geen lagere echte loss na een discrete omschakeling.

## Optimizer en trainingsduur

De oorspronkelijke Adam-defaults worden behouden. Zijn oorspronkelijke parametergroep bevat nu alleen de netwerkgewichten; een tweede groep bevat de zes theta-tensors. Beide starten op lr=0,001, weight_decay=0. De bestaande scheduler past dezelfde factor 0,1 toe na epochs 50, 75 en 90. Na iedere ene gezamenlijke Adam-update projecteert een post-step-hook theta naar [5,6]. Deze projectie is geen extra optimizerstap en verandert de gewichten niet.

De training gebruikt SVHN train-only: 73.257 trainbeelden en 26.032 testbeelden, zonder extra-split of aparte validatiesplit. Het model blijft `svhnq`, expand=2, met bestaande 40×40-transforms, seed 20261009, batch128 en drop_last=False.

De drie passes volgen E4M3→E2M3→E2M1. E4M3 gebruikt hard CE; E2M3 leert met soft CE van de detached E4M3-output; E2M1 leert van de detached E2M3-output. Temperatuur1, gesommeerde gradients, één update per minibatch. Alle modi worden na iedere epoch met hard CE op de volledige officiële testset geëvalueerd. A32, FP32-master/gradients/Adam en de drie bestaande BN-banken blijven behouden.

De standaardrun voert honderd epochs uit. Dat zijn 573 trainminibatches per epoch, 57.300 updates en 171.900 trainingspasses. `--stop-after-epoch` accepteert een gehele eindepoch van 1 tot en met 100; een gedeeltelijke run blijft onvoltooid gemarkeerd. De uitvoer gebruikt een nieuwe resultatenmap; er wordt geen bestaand checkpoint als warmstart geladen.

## Leersnelheid van beide parametergroepen

| Trainings-epochs | Gewichten-lr | Theta-lr |
|---|---:|---:|
| 1–50 | 0,001 | 0,001 |
| 51–75 | 0,0001 | 0,0001 |
| 76–90 | 0,00001 | 0,00001 |
| 91–100 | 0,000001 | 0,000001 |

De grenzen slaan op de gebruikte lr tijdens de epoch. De schedulerstap na epoch50 maakt dus de lagere lr gereed voor epoch51. Beide groepen en hun Adam-staat worden samen hersteld bij hervatten. Er is geen nieuwe warm-up, andere schaalgradient of gewijzigde drempel ingevoerd.

## Diagnostiek en checkpointselectie

Dezelfde selectieregel blijft gelden: het eerste strikte maximum van E4M3-testaccuracy. De notebook rapporteert alle modi uit dat gezamenlijke checkpoint en uit epoch100. Een onvoltooide run toont herkenbaar de laatste voltooide epoch. Alleen een run met alle honderd volledige epochs krijgt de volledige completion-status; de nieuwe echte honderd-epochtraining is bij aflevering nog niet uitgevoerd.

Per laag en epoch worden theta-min/max/gemiddelde/populatiestandaardafwijking, veranderde hulpwaarden, update-norm, aantallen ×32/×64, netto keuzewijzigingen, alle omschakelingen tijdens updates, unieke omschakelende blokken, projectietellingen en schaalgradientnormen opgeslagen. `scale_block_state.csv` bevat epoch0 en iedere volledig afgewerkte epoch. Bij hervatten wordt de volledige geschiedenis behouden.

Geen harde omschakelingen kan betekenen dat theta wel beweegt maar nog geen drempel bereikt. Lage accuracy, constante voorspellingen, nul maar eindige gradients of nul keuzewijzigingen zijn diagnostische resultaten; ze veroorzaken op zichzelf geen FAIL.

De meegeleverde ×32- en ×64-referenties zijn echte volledige runs. Bij honderd voltooide epochs worden alle honderd epochs vergeleken. Bij een gedeeltelijke nieuwe run beperkt de notebook hun curves en E4M3-selecties tot dezelfde reeds gemeten eerste epochs. De officiële testset wordt zoals eerder voor checkpointselectie gebruikt; geselecteerde beste scores zijn geen onafhankelijke testschatting. Eén run per methode en de oorspronkelijke niet-deterministische CUDA-backend onderbouwen geen significantieclaim.

## Exports en hervatten

Hervatbare checkpoints bevatten netwerkstaat, theta, beide Adam-groepen, scheduler, RNG en epochgeschiedenis. Alleen onvoltooide checkpoints van dit nieuwe contract worden geaccepteerd.

De compacte E4M3-bronexport bevat de oorspronkelijke elementcodes en E8M0-schalen, noodzakelijke FP32-uitzonderingen en BN-staat, en per blok één gepakt bit voor de lagere schaalkeuze. Bit0 betekent offset5 en bit1 offset6. De bits zijn row-major en little-endian binnen iedere byte; ongebruikte paddingbits moeten nul zijn. Dit geeft 510 bytes keuzedata. Er zitten geen FP32-theta-tensors of masters van de zes MX-lagen in deze export.

E2M3 en E2M1 worden uitsluitend uit de opgeslagen en opnieuw ingeladen bron plus keuzebits afgeleid. Gewichten worden op het trainingsdevice exact met de live route vergeleken. Deze exportcontrole is geen extra volledige testsetinferentie. Bevroren reconstructie heeft geen FP32-theta of encoder nodig.

De analytische extra trainingstoestand voor theta, gradient en twee FP32-Adam-momenten bedraagt 4.080×16 = 65.280 bytes (63,75 KiB). Dit omvat geen tijdelijke tensors, optimizerstaptellers of framework/allocatoroverhead. Tijd, geheugengebruik en modelaccuracy worden afzonderlijk gemeten; de FP32-convolutie-emulator meet geen native MX-hardwaredoorvoer of energie.

## Bronnen en afbakening

De bestaande projectcode en eerder gevalideerde codecs vormen de implementatiebasis. Het algemene idee van een expliciete benaderde gradient voor een quantisatieparameter sluit aan bij [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153), maar de gebruikte gradient hierboven is voor onze eigen binaire exponentoffset afgeleid. De representatieschalen volgen [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/). De E4M3-bronschaal zelf is in deze proef niet trainbaar.
