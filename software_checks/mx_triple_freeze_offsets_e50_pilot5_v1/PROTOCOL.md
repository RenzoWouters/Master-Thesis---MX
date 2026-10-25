# Protocol — vaste offset vanaf epoch 51

## Vraag en vergelijking

Helpt het in epochs 51–55 om de na epoch 50 geleerde extra exponentoffset per MX-blok vast te zetten, terwijl de gewichten verder leren?

Nieuwe variant: vijf nieuwe epochs met vaste offset. Historische controle: de oorspronkelijke, reeds gemeten epochs 51–55 van de 100-epochtraining met geleerde ×32/×64-keuze. Er wordt geen dynamische controle opnieuw getraind.

## Vertrekpunt

Oorspronkelijke run: `20261025_150724_515132_joint_run`, learned-scale reference100.

- Epoch: 50, volledig afgerond na evaluatie en scheduler.
- Adamupdates: 28.650.
- Gewichts- en theta-LR in het vertrekcheckpoint: beide 0,0001 voor epoch 51.
- Epoch-50-model-SHA256: `14035d75b62ba2f6dee356cf30ef03cd14b65a715ae0aa11923b29b301fd9008`.
- Oorspronkelijk contract-SHA256: `fc9d8c652e8977052cd89ed10905c9e5a77155b34c14a4412b1d47b0b9a92cc6`.
- 4.080 blokken: 1.654 met offset 5 en 2.426 met offset 6.

Checkpointbytes worden bij uitvoering tegen hun eigen checksum gecontroleerd. Bovendien worden het volledige oudercontract, de modelhash, de oorspronkelijke checkpointgeschiedenis en de theta-staat tegen de meegeleverde, uit de werkelijke feedbackbundel afgeleide referentie gecontroleerd. De bron voor die referentie is gedocumenteerd in `parent_reference/REFERENCE.json`.

## Interventie en training

Herstel eerst model, BatchNorm, beide Adamgroepen, alle momenten, scheduler en RNG onder het oorspronkelijke contract. Stel daarna de LR van uitsluitend de theta-groep op nul. Zet ook de bijhorende `initial_lr`, scheduler-`base_lrs` en `_last_lr` op nul. Gewichten, theta, Adam-momenten, de gewichts-LR en RNG veranderen bij deze ingreep niet.

Behoud `requires_grad=True` en de oorspronkelijke trainingsfunctie zodat quantisatie, backward en controles gelijk blijven. Theta-gradients en theta-Adam-momenten blijven berekend. Vanaf epoch 51 veranderen de theta-parameters en harde offsetkeuzes zelf niet meer. Iedere Adamstap wordt hierop gecontroleerd.

Behoud het oorspronkelijke tijdschema: epochs 51–55, 573 minibatches per epoch, gewichts-LR 0,0001; schedulerpositie blijft absoluut. Aan het einde van epoch 55 zijn er in totaal 31.515 updates, waarvan 2.865 nieuwe en 8.595 nieuwe trainingspasses.

SVHN: train-only 73.257, test 26.032, batch 128, workers 0, geen extra/validatieset. CNN-8/svhnq expand 2; A32; FP32-masters, gradients en Adam. Drie originele recursieve distillatiepasses en één Adamupdate per minibatch. E4M3-codes en E8M0-bronschaal volgen actuele gewichten; E2M3 gebruikt de twee hoogste exponentbits plus dezelfde teken-/mantissabits en de vaste extra offset; E2M1 volgt door E2M3-codes twee bits naar rechts te schuiven, met dezelfde lagere schaal.

De historische sample-order-hash voor train en test moet iedere nieuwe epoch overeenkomen. De CUDA-backend blijft de oorspronkelijke `cudnn.benchmark=True`-instelling; bitexacte reproduceerbaarheid wordt niet geclaimd.

## Uitkomsten

Primair: testaccuracy per precisie op epoch 55 en verschil met historische epoch 55. Secundair: eerste strikt hoogste E4M3-testaccuracy binnen epochs 51–55, met alle drie precisies uit datzelfde checkpoint; dezelfde selectie afzonderlijk toegepast op de historische controle.

Diagnostiek: gewichtsupdates, loss, nul theta-/keuzewijzigingen, bronencoderingen, per-block ×32/×64-verdeling, LR en momenten, datasetdekking en samplevolgorde, epochduren. Een lage accuracy is een meetuitkomst en maakt de technische controle op zichzelf niet ongeldig.

De nieuwe resultaten worden in een nieuwe map opgeslagen. De ouderrun, diens checkpoint, bestaande notebooks en historische resultaten worden gelezen en behouden.
