# Van eenmodus naar twee MX-modi, zonder oude code te overschrijven

## 1. De drie noodzakelijke wijzigingen in de trainingsfunctie

`dual_support.adapt_forward_text` leest de werkelijke forwardfunctie uit de
meegeleverde auteursbron. Op precies drie plaatsen (test, teacher, student)
worden deze gekoppelde selectors vervangen:

```python
model.apply(lambda m: setattr(m, 'wbit', bw))
model.apply(lambda m: setattr(m, 'abit', bw))
```

wordt:

```python
select_mode(model, bw)
```

De interne bitlijst wordt `4,6`. Een AST-test verwijdert alleen deze selectors
uit oud en nieuw en controleert exacte gelijkheid van alle resterende code.
Loss, backwardvolgorde, softmax(detach), gradientaccumulatie, dropout en Adam-
stap zijn dus geen nieuwe implementatie. De volledige leesbare functie staat
in `provenance/dual_forward.py.txt`, de drie wijzigingen in
`forward_selectors.patch`. De bron-`train.py` zelf wordt niet overschreven.

Conceptueel:

```python
optimizer.zero_grad()
select_mode(model, 6)             # E2M3, A32, eigen hoge BN-bank
high_logits = model(images)
hard_ce(high_logits, labels).backward()
teacher = softmax(high_logits.detach(), dim=1)
select_mode(model, 4)             # E2M1, nog steeds A32, eigen lage BN-bank
low_logits = model(images)
original_soft_ce(low_logits, teacher).backward()
optimizer.step()                 # som van beide gradients, NIET gedeeld door 2
```

## 2. A32 staat los van de gewichtsmodus

`dual_adapter.DualMXQuantizer` vervangt de quantize_fn van dezelfde zes lagen.
De actuele FP32-master blijft de parameter; eerst dezelfde G(W)-voorbewerking,
dan directe quantisatie naar de gekozen modus. A32-activaties veranderen niet.

`WeightModeBatchNorm` behoudt de oorspronkelijke bn_dict en dus dezelfde
initialisatie/parameternamen. Gewichtsmodus6 kiest bankkey32; modus4 kiest
bankkey4. Dit zijn opslaglabels. **Alle abit-attributen blijven 32.**
De overige drie banken worden bewaard maar niet bijgewerkt. Zo behouden we
modelstaatstructuur en initialisatie zonder een ongedocumenteerde BN-pruning.

## 3. Bestaande E2M3-code behouden, E2M1 toevoegen

`mx_format.py` en `mx_adapter.py` blijven byte-identiek aan de geslaagde
voorgaande implementatie. `mx_e2m1.py` is een afzonderlijke controleerbare fork
met een mantissabit, codes0..15, max6 en twee vierbitcodes per byte. De diff
staat in `codec_e2m1.patch`. Schaalregel, blokindeling en grensbeleid blijven
hetzelfde. Alle zestien codes en afrondgrenzen zijn onafhankelijk getest.

De keuze is nog gewone quantisatie. Bij scale1 bijvoorbeeld:
G=1.375 -> E2M3=1.375; directe E2M1=1.5. Een codebitshift van E2M3 naar E2M1
zou hier 1.0 geven. De test controleert expliciet dat deze proef de directe
uitkomst gebruikt, niet onbedoeld truncatie.

## 4. Uitbreiding van registratie en eindrapport

`DualMeterRecorder` gebruikt dezelfde upstream AverageMeter-berekening, maar
nu zes meters voor twee modi in plaats van drie voor een modus. De lage
trainingsloss krijgt het correcte label recursive_soft_cross_entropy; beide
testlosses zijn hard_cross_entropy. Geen extra diagnostische losses in de
optimizer. Non-finite metrics/gradients stoppen de run; geen stil skippen.

Niet-mutating hooks controleren de volgorde6->4, het aantal forwards, een
losgekoppelde teacher en een optimizerupdate per minibatch. De bestaande
hervat-, data-, logging- en checkpointkern blijft hergebruikt. Iedere actieve
BN-bank moet evenveel batches hebben als er optimizerstappen zijn geweest;
ongebruikte banken hebben geen gradient of Adamstaat.

Per epoch worden de wijzigingen van gedeelde FP32-masters en de twee sets
codes bijgehouden. Masterupdatecijfers worden voor beide modusrijnen herhaald;
ze beschrijven dezelfde tensors en mogen niet dubbel geteld worden.

## 5. Geen verborgen truncatieklaar exportresultaat

Een trainingcheckpoint bevat de gedeelde FP32-trainingsstaat. Voor best/last
maken we aparte direct afgeronde E2M3- en E2M1-exports met hun juiste BN-bank.
Ze bevatten geen zes FP32-masters, maar wel de uitzonderingslagen/BN-staat.
De verschillende exports zijn NIET door truncatie uit een bestand verkregen.
Een latere truncatieproef moet de lage codes werkelijk uit de hoge bron halen.

## 6. Reproduceerbaarheid en oude vergelijkingen

Zelfde architectuur, seed, data en initialisatiebeleid. Twee passes gebruiken
wel meer dropout-RNG-trekkingen dan een pass. Daarom wordt de werkelijke
samplevolgorde gelogd, niet aangenomen dat alle latere epochs exact dezelfde
volgorde hebben als oude eenmodusproeven. Binnen de nieuwe runner is het
hervatten op CPU gecontroleerd; cuDNN benchmark blijft niet-deterministisch.

Een pilotcheckpoint wordt niet als warmstart gebruikt. Ruwe data, oude code
en alle eerdere resultaten blijven ongewijzigd. README bevat de commando's.
