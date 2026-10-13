# Vergelijking met de GitHub-implementatie

Bron is het door de gebruiker aangeleverde Any-Precision-DNNs-master(1).zip,
niet een stilzwijgend vervangen actuele repositoryversie. Het archief is byte-identiek.
De bestanden source_notes/upstream_README.md en upstream_train_cifar10.sh zijn
ongewijzigde kopieen uit dat archief. Hun hash staat in het pakketmanifest.

| Onderdeel | Auteurscode | Deze run |
|---|---|---|
| Model/quantizer/loss | resnet20q, expand5, ReLU/tanh32, recursieve distillatie | Ongewijzigde originele modules |
| Trainingsschema | train_cifar10.sh:200; LR0.001;100/150/180; Adam;WD0 | Hetzelfde uitvoerbare recept |
| Batch en data | train.py:128; datasets/cifar10.py:50.000 train/10.000 test | Dezelfde data en batch, geen holdout |
| Preprocessing | datasets/data.py: crop, flip, normalisatie | Zelfde functie, moderne torchvision |
| Trainingstap | W32-labels; W8->4->2->1 recursieve targets; som gradients; Adam | Zelfde bewerkingen, hergebruik van gecontroleerde train_batch |
| Evaluatie/selectie | testset heet val; elke epoch; model_best gekozen op hoogste-precisiescore | Bewust expliciete eindtest van epoch200; geen best-on-test-selectie |
| Checkpoint/hervatten | model/optimizer/epoch; niet alle RNGs | Toegevoegde geteste volledige epochgrens-hervatting |
| Runtime | Historische pakketten; cudnn.benchmark=True, pin_memory=True | Huidige geteste omgeving; deterministisch, pin_memory=False |

## Juiste vergelijkingsrij

Resnet-20-Any (recursive), niet dedicated, hard of soft:

| bits gewichten/activaties |1|2|4|8|32|
|---|---:|---:|---:|---:|---:|
| README top1 procent |91.89|93.90|93.86|93.75|94.11|

De tabel staat in de oorspronkelijke README. Dezelfde waarden werden bij de
voorbereiding ook op de officiele repository gezien:
https://github.com/SHI-Labs/Any-Precision-DNNs

De README koppelt de cijfers niet aan een door ons vastgesteld volledig log met
seed, epochbudget en checkpointselectie. Dat het script200 vermeldt is een goede
reden om dit recept te gebruiken, maar geen bewijs van de herkomst van iedere score.
Dit verschil wordt niet weggewerkt door af te stemmen op de testset.

## Afbakening tegenover de paper

De paper vermeldt400 epochs met150/250/350 (PDF-pagina5). De README benoemt zelf
wijzigingen voor ReLU en tanh op32bit. In deze run reconstrueren we de paper niet:
we behouden de geleverde GitHub-variant en het bijgeleverde CIFAR-recept. De generieke
README-opmerking lr0.1->0.5 wordt niet op CIFAR10 toegepast; het concrete CIFAR-script
specificeert0.001. Er wordt niets aan het netwerk gewijzigd om scores af te dwingen.

Voor latere MX-experimenten is de eigen onder hetzelfde protocol getrainde
Any-Precision-referentie de directe baseline. De auteurscijfers zijn externe context.
