# Protocol — drie precisies uit één E4M3-bron tijdens training

## Onderzoeksvraag

Kan gezamenlijke training zich aanpassen aan dezelfde naïeve exponent- en
mantissabewerking die eerder post-hoc is geëvalueerd? De directe drieprecisie-
referentie en alle eerdere resultaten blijven behouden. Deze nieuwe proef
begint vanaf dezelfde verse seed 20261009 en loopt vijf volledige epochs.

De pilot controleert uitvoerbaarheid, leergedrag en tijd. Een volledige proef
met honderd epochs wordt daarna afzonderlijk beoordeeld. Een vijf-epochscore
is niet rechtstreeks het trainingseffect tegenover een honderd-epochcheckpoint.

## Gewichtsbron en bitbewerkingen

De bestaande zes MX-convoluties zijn `layers.3`, `.6`, `.10`, `.13`, `.16` en
`.20`. De eerste convolutie, laatste lineaire laag en overige modelstaat
behouden het eerdere beleid. De voorbewerking is exact dezelfde G(W):

`tanh(W) / max(abs(tanh(W))) * mean(abs(W)).detach()`

De bestaande guard voor volledig nulgewicht blijft aanwezig. W is de gedeelde
FP32-updatemaster. E4M3 kwantiseert G(W) met de bestaande exact-binary-v2-codec:
nearest rounding, ties-to-even, eindige saturatie, lokale subnormals en signed
zero. De E4M3-NaN-codes 127/255 worden geweigerd.

| Modus | Broncode | Doelcode | Schaal |
|---|---|---|---|
| E4M3 | FP32 G(W) | `s e3 e2 e1 e0 m2 m1 m0` via RNE | E4M3-schaalregel |
| E2M3 | E4M3-code | `s e3 e2 m2 m1 m0` | Kopie E4M3-schaal |
| E2M1 | Afgeleide E2M3-code | `s e3 e2 m2` | Kopie dezelfde E4M3-schaal |

De gebruikte exponentbewerking is:

```python
sign = u8 >> 7
exponent4 = (u8 >> 3) & 15
mantissa3 = u8 & 7
u6 = (sign << 5) | ((exponent4 >> 2) << 3) | mantissa3
u4 = u6 >> 2
```

De doelcode wordt gewoon als E2M3 respectievelijk E2M1 met bias 1 gedecodeerd.
Er is geen compensatie voor bronbias 7. Geen lagere element- of tensorencoder
wordt in deze afleiding aangeroepen. De pure bitfuncties komen byte-identiek
uit de uitgevoerde post-hocproef via `posthoc_truncation.py`.

## Schaalbeleid en gedeelde bron

Blokgrootte 32, binnen ieder outputkanaal over afgevlakte Cin*Kh*Kw, positieve
nulpadding aan de staart. De E8M0-schaalexponent voor een niet-nulblok is
`floor(log2(amax)) - 8`, begrensd op [-127,127]. Een nulblok krijgt schaal 1.

E4M3 wordt eenmaal per MX-laag en per minibatch gecodeerd. De twee lagere
codeverzamelingen worden uit exact die bron gemaakt. Alle drie modi gebruiken
byte-identieke schaalcodes binnen die broncyclus. De schalen kunnen door
veranderde mastergewichten tussen minibatches wijzigen. Er is geen apart
leerbare schaalparameter of extra schaal-/biascorrectie.

De broncache bevat uitsluitend losgekoppelde integercodes en schaalbytes.
Elke pass bouwt een eigen G(W)-autogradgraph. De cache wordt na de derde
forward geleegd, vóór de laatste backward en de Adam-update. Een gewijzigde
master, broncode, schaal, passvolgorde of geneste broncyclus wordt geweigerd.
Diagnostiek buiten de minibatchpasses en checkpoint-export hebben hun eigen
encoderingen en worden niet als trainingspasses geteld.

## Gradientbenadering

De forward gebruikt exact de gedecodeerde MX-waarde, ook voor zeer kleine
afgeleide waarden. Een eigen autogradfunctie voorkomt rekenkundige annulering
door een uitdrukking als `w + (q-w).detach()`.

De backward is een identity straight-through estimator ten opzichte van G(W)
voor elk van de drie routes. De oorspronkelijke afgeleide van G(W) blijft
behouden. Er is geen vermenigvuldiging van de gradient met 1/4 of 1/64 en geen
gradient door bitcodes of schaalkeuze. Dit is dezelfde benadering als in de
eerdere mantissatraining; het is geen exacte afgeleide van een bitbewerking.

## Training en BatchNorm

De oorspronkelijke Any-Precision-forwardfunctie blijft inhoudelijk gelijk,
met uitsluitend de reeds bestaande vervanging van drie selectorparen. De
broncache en diagnostiek worden er via hooks omheen geplaatst.

Per minibatch:

1. Gradients nulzetten.
2. E4M3-forward, harde CE met labels, backward.
3. E2M3-forward, soft CE met `softmax(E4M3_logits.detach())`, backward.
4. E2M1-forward, soft CE met `softmax(E2M3_logits.detach())`, backward.
5. Eén Adam-update op de gesommeerde gradients.

Temperatuur 1; geen deling door drie, extra FP32-teacher, extra harde studentloss,
gradientclip of aanpassing van de optimizer. De opeenvolgende dropout-
RNG-trekkingen blijven zoals in de bronlus. Evaluatie gebruikt volgorde 4/6/8
en harde CE voor iedere modus. De drie modi zien dezelfde minibatchbeelden.

BN-banken: E4M3 sleutel 8, E2M3 sleutel 32, E2M1 sleutel 4. Deze sleutels zijn
opslag-ID's; activaties zijn overal A32. De actieve BN-parameters en running
statistics worden tijdens training gewoon geleerd, ieder eenmaal per minibatch.
Tijdens test zijn ze bevroren. Banken 1/2 blijven ongebruikt, ongewijzigd en
zonder Adamstaat. Er is geen afzonderlijke herkalibratiefase.

Het model is CNN-8/svhnq expand 2 met 132.954 parameters inclusief de vijf
bewaarde BN-banken per BN-laag. Alle masterparameters, gradients en Adam-
momenten blijven FP32. De bestaande cuDNNbenchmark-/TF32-standaarden blijven
behouden; geen AMP of torch.compile. Het is numerieke MX-emulatie in FP32-
tensoroperaties, zonder claim over native MX-snelheid of energie.

## Data, schema en selectie

SVHN train-only: 73.257 trainbeelden en 26.032 officiële testbeelden; geen
extra of nieuwe validatiesplit. De oorspronkelijke Resize(40)/ToTensor/
Normalize(mean=std=0,5), batch 128, workers 0, pin_memory=True en drop_last=False
blijven gelijk. Train shuffelt, test niet. Nieuwe initialisatie, seed 20261009,
Adam lr 0,001, weight_decay 0. Het honderd-epochschema daalt na 50/75/90 met 0,1.
De vijf pilotepochs blijven allemaal op lr 0,001.

Er zijn 573 trainminibatches per epoch. De volledige pilot heeft 2.865 Adam-
updates en 8.595 trainingspasses. Iedere testepoch bevat 204 minibatches met
alle drie modi. Elke actieve BN-bank heeft na de pilot 2.865 updates.

Best is het eerste strikte maximum van E4M3-testaccuracy binnen deze pilot.
Alle drie modi worden op datzelfde checkpoint gerapporteerd. Last is epoch 5
bij voltooiing; een eerdere gecontroleerde stop wordt expliciet gemeld.
De notebook toont beide groepen en vermeldt dat epoch 100 nog niet bestaat.
Bestselectie op de officiële testset is geen onafhankelijke schatting na
selectie. Dit blijft één seed en een korte technische/empirische pilot.

## Metingen en falen

Per epoch blijven de drie train- en testmetrics, correcte aantallen, LR,
gradient-/Adam-/BN-controles, broncycli, gewichtsveranderingen, sampledekking,
timing en checkpointidentiteit bewaard. Daarnaast bevatten history en CSV
de tien voorspelde klasseaantallen per testmodus en de echte labelverdeling.
Dit observeert eventuele instorting naar één vaste klasse zonder de training
te veranderen of op een accuracydrempel te stoppen.

Lage accuracy, geen accuracyverbetering, een eindige nulgradient of geen
gewijzigde MX-masterwaarden is geen automatische technische FAIL. Verkeerde
stappen, ontbrekende gradients, verkeerde dtypes, niet-eindige toestand,
verkeerde broncodes of schaalbytes en onvolledige data zijn wel fouten.

## Checkpoints, compacte exports en behoud

Elke epoch krijgt een volledig FP32-trainingscheckpoint met optimizer,
scheduler, RNG, geschiedenis en exact contract. Resume is alleen toegestaan
binnen deze pilot en blijft begrensd op epoch 5. Een eerder direct of
tweemoduscheckpoint is geen geldig invoercheckpoint.

Best en last krijgen elk een E4M3-bronexport met codes/schalen, alle benodigde
BN-banken en FP32-uitzonderingsstaat, zonder de zes MX-mastertensors. Die bron
wordt opgeslagen, gehasht en herladen. E2M3 en E2M1 worden uitsluitend daaruit
afgeleid, opgeslagen en herladen. Hun gewichten worden exact vergeleken met
de live trainingsroute op hetzelfde device; er wordt geen volledige extra
testsetpass vanuit de export uitgevoerd. De korte softwarecontrole vergelijkt
wel exportlogits op synthetische beelden.

Alle uitvoer komt in nieuwe mappen; oude resultaten en datasetbestanden worden
behouden. De notebook is een lege uitvoerbare template. Softwarevalidaties
gebruiken uitsluitend synthetische input en zijn geen onderzoeksresultaten.
