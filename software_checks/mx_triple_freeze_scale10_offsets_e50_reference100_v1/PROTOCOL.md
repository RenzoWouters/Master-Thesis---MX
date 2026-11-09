# Protocol: bevroren offsets na epoch 50, eindpunt epoch 100

## Doel

Onderzoeken of de positieve korte voortzetting met vaste, eerder geleerde blokkeuzes uit tien factoren (×1, ×2, ×4, ×8, ×16, ×32, ×64, ×128, ×256 en ×512) behouden blijft tot het oorspronkelijke eindpunt epoch 100. De bestaande dynamische 100-epochtraining blijft de historische controle.

## Herkomst en vertrek

De bron voor dit pakket is de feitelijke bevroren pilot in `20261109_183845_084991_freeze_run`. Die vertrok van de oorspronkelijke geleerde training na epoch 50 en heeft epochs 51–55 correct afgewerkt met exact vaste theta en offsetkeuzes.

- Te laden parent: epoch 55, volgende epoch 56, 31.515 Adamupdates.
- Model-SHA256: `8e7960d8ab54a1762ab358b87b54283076a048e36d0e7a04122dd7a2923bd287`.
- Checkpoint-SHA256: `212cd65e11ceaef1d862ec77027a2deaea0b66aa6a3992e40ae029d2c2b659b9`.
- Pilotcontract-SHA256: `4e8a7f7fd77611ffb2fef8ee4776f0c029beaa272d6cecee9f4d95a9af94bc85`.
- Vastgezette keuzes: 5 blokken offset4, 1.191 offset5, 2.745 offset6 en 139 offset7; de andere zes offsets nul; dezelfde theta-waarden als na epoch 50.

De volledige checkpointgeschiedenis wordt tegen de feitelijke feedback gecontroleerd. Alleen de drie timingvelden die pas na het schrijven van checkpoint 55 bekend werden, ontbreken in die binaire snapshot; hun werkelijke waarden uit de feedback worden voor de vervolgrapportage behouden.

Het oorspronkelijke dynamische `history.json` wordt op de laptop gelezen en moet SHA256 `f644f5ce9426924c772e9e7d0208e4365e00065edc2fbcb17e89cf660f7b4be9` hebben. Het wordt ongewijzigd in de nieuwe resultaten opgenomen. Daarmee zijn samplevolgorde, gedeelde eerste 50 epochs, historische metrics en eerdere checkpointidentiteiten beschikbaar zonder een nieuwe controletraining.

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
