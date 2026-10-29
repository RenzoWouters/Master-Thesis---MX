# Protocol: gezamenlijke route-B-pilot van vijf epochs

## Onderzoeksvraag

Kan hetzelfde drieprecisienetwerk tijdens training beter leren functioneren
met naïeve exponenttruncatie wanneer de lagere gedeelde schalen vanaf het begin
exact64 maal de actuele E4M3-bronschaal zijn?

Dit vervolgt de post-hocevaluatie van route B. Het vergelijkt een nieuwe training
vanaf de oorspronkelijke initialisatie met de eerdere directe en naïeve
truncatieproeven. Een eerdere A-checkpoint wordt niet gebruikt als warmstart.
Route C was een afrondingscontrole en wordt niet als trainingroute ingevoegd.

## Exacte forwardbewerking

Per MX-laag en minibatch wordt de huidige FP32-master W met de bestaande G(W)
voorbewerkt. Alleen E4M3 wordt rechtstreeks gecodeerd; dezelfde integercodes
worden door de drie precisiepasses gedeeld. Broncode c heeft de indeling
`[s e3 e2 e1 e0 m2 m1 m0]`, met bestaande E4M3-bias 7. De lagere codes zijn:

```python
sign = c >> 7
exponent = (c >> 3) & 15
mantissa = c & 7
b6 = (sign << 5) | ((exponent >> 2) << 3) | mantissa
b4 = b6 >> 2
```

De E2-formaten gebruiken hun bestaande bias 1. Er wordt geen andere mapping
of aanvullende elementafronding toegepast. De mantissafunctie is dezelfde
ongewijzigde bitshift als in de eerdere proeven.

Voor de E8M0-broncode k geldt `S = 2**(k-127)`. E4M3 behoudt k. Beide lagere
modi gebruiken k+6, dus `64*S`. De optelling gebeurt in int32 vóór terugconversie
naar uint8. Alle blokken volgen dit beleid, ook nulblokken en paddingblokken.
Als k+6 groter is dan 254, stopt de proef expliciet; er is geen clipping of
wraparound. De oorspronkelijke E4-codec met zijn bestaande schaalkeuze blijft
ongewijzigd. Broncode 255 en de E4-NaN-codes blijven ongeldig. Decodering naar
de FP32-inferentiegewichten moet eindig zijn.

De scalaire factor 64 is vast, maar S zelf kan na een optimizerupdate veranderen.
Er bestaan geen afzonderlijke geleerde schaalparameters. E2M1 behoudt de
E2M3-schaal; de schaal wordt niet nogmaals met64 vermenigvuldigd.

## Backward en gedeelde bron

De bestaande straight-through estimator wordt toegepast op de volledige
codebewerking. Voor elke modus geldt in backward een identity tegenover G(W):
de upstreamgradient wordt onveranderd doorgegeven, zonder extra factor 64.
De oorspronkelijke afgeleide van G(W) naar W blijft behouden. Er lopen geen
gradients door integercodes of schaalbytes.

De implementatie decodeert de exacte forwardwaarde in een eigen autogradfunctie.
Hij gebruikt geen floating-point aftrekconstructie die grote waarden kan laten
wegvallen. De cache bevat uitsluitend detached codes en schalen. Iedere pass
bouwt zijn eigen G(W)-autogradgraph. Er is één E4-encoding per laag per minibatch,
geen E2-tensorencoder en geen tweede kwantisatie uit een afzonderlijke bron.
Na de derde forward is de cache leeg, vóór de derde backward en Adam-update.

De schaalwijziging en identity-STE-behandeling zijn afzonderlijk getest. Een
factor 64 buiten de STE plaatsen en daardoor de gradient mee vermenigvuldigen
zou een extra trainingswijziging zijn en hoort niet bij deze proef.

## Ongewijzigde trainingsopzet

| Onderdeel | Instelling |
|---|---|
| Netwerk | oorspronkelijke svhnq/CNN-8, expand 2, 132.954 parameters |
| MX-lagen | dezelfde zes convoluties |
| Gewichtsmodi | E4M3, E2M3, E2M1 |
| Blokindeling | 32 waarden, binnen outputkanaal; dezelfde padding |
| Activaties | A32 in alle drie modi |
| Master / gradients / Adam-momenten | FP32 |
| Data | SVHN train 73.257 en test 26.032; geen extra of validatiesplit |
| Initialisatie | verse seed 20261009, oorspronkelijke modelconstructie |
| Minibatch | 128; shuffle train; volledige dekking, geen drop_last |
| Optimizer | Adam, lr 0,001, weight decay 0 |
| LR-schema | oorspronkelijk 100-epochschema, dalingen na 50/75/90 |
| Duur | vaste stop na 5 volledige epochs |
| Trainpasses | E4M3 → E2M3 → E2M1 |
| Trainloss | hard CE voor E4; detached E4-softtargets voor E2M3; detached E2M3-softtargets voor E2M1 |
| Update | gradients optellen, geen deling door 3; één Adam-stap per minibatch |
| Evaluatie | volgorde E2M1 → E2M3 → E4M3; harde CE voor alle modi |
| BN-banken | sleutel 8/32/4 voor E4M3/E2M3/E2M1; normaal leren in train, bevroren in eval |
| Backend | oorspronkelijke cuDNNbenchmark=True; geen nieuwe deterministische trainingoverride |

De vijf epochs omvatten 573 trainminibatches per epoch, dus 2.865 optimizerupdates
en 8.595 trainingspasses. De test bevat 204 minibatches per epoch. Het checkcommando
mag voor zijn synthetische deterministische vergelijking een proceslokale
cuBLAS-instelling zetten; het run/resumecommando verandert die instelling niet.

## Checkpoints, selectie en exports

Iedere volledige epoch wordt als afzonderlijk FP32-trainingscheckpoint bewaard,
met optimizer, scheduler, RNG, voortgang en contract. Een stop op een epochgrens
kan worden hervat naar maximaal epoch 5, uitsluitend binnen hetzelfde nieuwe
route-B-contract. Oude A-/directe checkpoints worden niet geaccepteerd.

Voor de presentatie worden twee groepen vastgelegd:

1. De laatste voltooide epoch; bij een volledige pilot epoch 5.
2. Het eerste strikte maximum van E4M3-testaccuracy binnen de voltooide pilot.

In beide groepen worden alle drie de modi uit hetzelfde checkpoint gebruikt.
Er wordt geen apart beste checkpoint per lagere precisie gekozen. De officiële
testset wordt hiermee ook voor selectie gebruikt; dit is de bestaande werkwijze,
geen onafhankelijke held-out modelschatting. Epoch100 is nog niet gemeten.

Per gekozen checkpoint wordt eerst de E4-bronexport opgeslagen en gecontroleerd
herladen. Daarna ontstaan E2M3 en E2M1 uitsluitend uit die broncodes, bronhash,
bron-schaalcode +6 en gedeelde overige modelstaat. De zes FP32-MX-masters ontbreken
uit de compacte exports. De aparte volledige checkpoints blijven beschikbaar.
De gedecodeerde exportgewichten worden op het trainingdevice exact vergeleken
met dezelfde live route; dit is geen extra volledige testset-evaluatie.

## Rapportage en interpretatie

Het rapport onderscheidt de E4-schaaloffset 0 van de lagere offset 6. Het valideert
alle broncycli, zes E4-encoderingen per minibatch, beide lagere +6-relaties,
mantissaschaalbehoud, lege caches vóór updates, detached teachers, FP32-toestand,
BN-tellers, voorbeelddekking en de geverifieerde geselecteerde checkpointgegevens.

Accuracy en klasseverdeling zijn uitkomsten. Een lage accuracy, één voorspelde
klasse, nul eindige gradients of een epoch zonder MX-masterwijzigingen wordt
gerapporteerd en niet stilzwijgend met andere instellingen gerepareerd. NaN/Inf,
ongeldige schaalcodes, ontbrekende actieve gradients of onjuiste toestand zijn
technische fouten. Er is geen minimumaccuracy voor een technische PASS.

De notebook geeft learning curves, correcte losslabels, beide checkpointgroepen,
klassentellingen, gewichts-/code-/schaalwijzigingen, tijd en geheugen. Hij gebruikt
de round-trip CSV-parser uit de eerdere correctie. Hij bevat geen modeluitvoering
en is zonder opgeslagen output meegeleverd.

De vijf-epochresultaten worden naast vergelijkbare eerdere vijf-epochresultaten
beoordeeld. Ze bewijzen geen eindconvergentie of algemene superioriteit. De vaste
factor 64 is geen geleerd optimum en herstelt geen verloren exponentbits. De
numerieke emulator draait met FP32-convoluties; er wordt geen native MX-snelheid
of energievoordeel uit deze looptijd afgeleid.
