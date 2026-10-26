# Protocol: bevroren offsets na epoch 50, eindpunt epoch 100

## Doel

Onderzoeken of de positieve korte voortzetting met vaste, eerder geleerde ×32/×64-blokkeuzes behouden blijft tot het oorspronkelijke eindpunt epoch 100. De bestaande dynamische 100-epochtraining blijft de historische controle.

## Herkomst en vertrek

De bron voor dit pakket is de feitelijke bevroren pilot in `20261025_174221_578057_freeze_run`. Die vertrok van de oorspronkelijke geleerde training na epoch 50 en heeft epochs 51–55 correct afgewerkt met exact vaste theta en offsetkeuzes.

- Te laden parent: epoch 55, volgende epoch 56, 31.515 Adamupdates.
- Model-SHA256: `89af4fe6473463abdab2b41a8985b2ac1a9ad927c730f8cb1b03aa4bcd9c10db`.
- Checkpoint-SHA256: `293097deafb2c5b673a44633aca94d08fe4ba619b0a2b34838df45c291290568`.
- Pilotcontract-SHA256: `f01fba81fd9de1378cb3b19f9dc2d27e0875d81eebc4bba149ed3a41d0b12cfe`.
- Vastgezette keuzes: 1.654 blokken offset5 en 2.426 blokken offset6; dezelfde theta-waarden als na epoch 50.

De volledige checkpointgeschiedenis wordt tegen de feitelijke feedback gecontroleerd. Alleen de drie timingvelden die pas na het schrijven van checkpoint 55 bekend werden, ontbreken in die binaire snapshot; hun werkelijke waarden uit de feedback worden voor de vervolgrapportage behouden.

Het oorspronkelijke dynamische `history.json` wordt op de laptop gelezen en moet SHA256 `f82164837a70e9cea1e7a52932d7d2f193371de2f0bd7fda87618cbc29867c5a` hebben. Het wordt ongewijzigd in de nieuwe resultaten opgenomen. Daarmee zijn samplevolgorde, gedeelde eerste 50 epochs, historische metrics en eerdere checkpointidentiteiten beschikbaar zonder een nieuwe controletraining.

## Voortzetting

Model, BatchNorm, beide Adamgroepen en momenten, scheduler en RNG worden strikt tegen het pilotcontract hersteld. De oorspronkelijke omgeving, dataset en broncode moeten overeenkomen. De optimizer en scheduler worden na dat herstel niet aangepast: de theta-LR was al nul. Alleen de looptijdgrens en het nieuwe afstammingscontract worden vastgelegd.

Uitvoering: uitsluitend 56–100, met behoud van de absolute epoch-, Adam- en BN-tellers. De vijf bevroren piloteepochs blijven ongewijzigd in de voortzettingsgeschiedenis.

Gewichts-LR: 0,0001 in56–75, 0,00001 in76–90, 0,000001 in91–100. Theta-LR blijft nul. De scheduler stapt na training en evaluatie, zoals in de oorspronkelijke code.

Dataprotocol, SVHN train-only/test, CNN-8/svhnq expand2, batch128, workers0, A32 en FP32-masters/gradients/momenten blijven gelijk. E4M3-bronencodering, exponent-MSB-truncatie naar E2M3, mantissashift naar E2M1, recursieve detached teachers en één Adamupdate per minibatch worden byte-identiek uit de eerdere runtime hergebruikt.

De E4M3-bronschaal blijft dynamisch. Beide lagere precisies gebruiken hun huidige bronshared exponent plus de vaste offset na epoch 50. Alle theta-waarden en harde keuzes worden iedere update gecontroleerd. Theta-gradients en Adam-momenten blijven als diagnostische uitvoering berekend; deze proef optimaliseert die trainingskosten niet.

## Uitkomsten en selectie

Primair: drie testaccuracies op epoch 100 en verschillen met historische dynamische epoch 100.

Secundair: eerste strikt hoogste E4M3-testaccuracy binnen51–100, met alle drie precisies uit hetzelfde checkpoint. De historische dynamische selectie gebruikt hetzelfde venster.

Daarnaast: gebruikelijke globale selectie over1–100. Voor het bevroren traject bestaat dit uit de gedeelde originele1–50, de feitelijke bevroren51–55 en de nieuwe bevroren56–100. Een globale selectie vóór de freeze wordt gelabeld als `shared_before_freeze`; de export bevat dan de oorspronkelijke theta van dat checkpoint. De bevroren keuzes van epoch50 worden niet achteraf op dat oudere model toegepast.

Alle geselecteerde checkpoints worden tegen het contract van hun eigen stadium gecontroleerd. Nieuwe bestanden krijgen eigen namen en worden niet over bestaande resultaten geschreven.

## Interpretatie

Dit is één voortzetting met een historische controle, dezelfde opgeslagen RNG-toestand en gecontroleerde samplevolgorde. De oorspronkelijke CUDA-backend is niet bitexact reproduceerbaar. Official-test-selectie is geen onafhankelijke validatieschatting; dit protocol bepaalt evenmin het optimale moment om offsets vast te zetten.

Voltooiing vereist alle nieuwe epochs56–100. Gedeeltelijke uitvoeringen blijven expliciet onvolledig en hervatbaar.
