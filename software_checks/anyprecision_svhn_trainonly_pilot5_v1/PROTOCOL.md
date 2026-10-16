# Vastgelegd train-only-protocol

## Bronnen en expliciete afwijking

Auteursbron: *Any-Precision Deep Neural Networks*, aangeleverde PDF, p. 5,
Implementation Details, en p. 6 (gedrukte p. 10768), tabel 1, CNN-8 / SVHN / Ours.
De paper gebruikt train + extra, 100 epochs, Adam zonder weight decay, lr 0.001,
LR-mijlpalen 50/75/90 met factor 0.1, vijf precisies 1/2/4/8/32 en recursieve KD.
De historische seed en volledige checkpointprovenance zijn niet vastgesteld.

Uitvoerbare bron: het eerder aangeleverde Any-Precision-DNNs-master(3).zip,
SHA256 7e36c5f039a78e3174421de7ff097066f95dde2837ea02f11913f3be0ba7f73c.
Geen nieuw opgehaalde GitHub-code of stilzwijgende implementatiewissel.

Onderzoekswijziging ten opzichte van de vorige vijf-epochpilot: gebruik de reeds
bestaande split 'train' van de ONGEWIJZIGDE SVHNMIX-klasse in plaats van 'train_extra'.
Geen eigen datasetklasse, geen willekeurige subset en geen nieuwe validatiesplit.
Alleen train_32x32.mat en test_32x32.mat worden geverifieerd en ingelezen. Extra wordt
niet gebruikt, niet vereist en niet aangeraakt. Een aanwezig extra-bestand blijft staan.

## Behouden instellingen

svhnq factory met expand=2, zeven convoluties en een lineaire eindlaag; eerste/laatste
laag blijven zoals in de bron. Dropout(0.5), vijf BatchNorm-banken en de bestaande
kwantisator blijven behouden, inclusief tanh/normalisatie bij W32. Geen MX of truncatie.
Resize(40), ToTensor, Normalize(mean=std=0.5). Batch 128, workers=0, pin_memory=True,
shuffle voor train, niet voor test; drop_last=False. Geen nieuwe augmentatie.

De originele functie forward() uit train.py wordt zonder AST-wijziging ingeladen.
Training: 32 -> 8 -> 4 -> 2 -> 1, hard-label loss op de hoogste precisie, daarna
recursieve soft-target loss met detached teacher; vijf backwardpasses, eenmaal Adam
per minibatch. Evaluatie: 1 -> 2 -> 4 -> 8 -> 32, no_grad, model.eval; geen BN-update.
Scheduler.step na training en evaluatie. De milestone bij 50 wordt pas na epoch 50
verwerkt: epochs 1..5 gebruiken allemaal lr=0.001.

## Pilot en trainingsbudget

100 epochs blijft het geplande schema; deze uitvoering stopt uiterlijk na epoch 5.
73.257 trainbeelden leveren 573 minibatches en updates per epoch; vijf epochs geven
2.865 updates. Honderd train-only-epochs geven 57.300 updates, niet 472.200 zoals
train+extra. Het gewijzigde updatebudget volgt uit de gekozen datasplit en wordt niet
verstopt door de learning rate op vijf epochs te herschalen.

Run start vanaf een nieuwe initialisatie met dezelfde lokale seed 20261009 als de
vorige pilot. Geen warmstart uit een model dat extra heeft gezien. Hervatten is alleen
binnen dit identieke train-only-contract toegestaan. De CLI staat geen stop >5 toe.

## Checkpointselectie en rapportage

De broncode gebruikt val als alias voor de officiele testset. Dit blijft behouden.
Na iedere epoch: volledige 26.032 testbeelden bij de vijf precisies. Selectie op de
strikt hoogste W32/A32-top1 volgens de oorspronkelijke gewogen AverageMeter; bij een
gelijke score blijft de eerste epoch behouden. Niet afzonderlijk per bit de beste
score samenvoegen. Alle precisies horen bij hetzelfde geselecteerde checkpoint.
Beste en laatste checkpoint zijn afzonderlijke rapportrollen en fysieke kopieen,
naast de onveranderlijke individuele epochcheckpoints.

Deze testscore is niet onafhankelijk van checkpointselectie. Er is geen nieuwe
validatiesplit ingevoerd. Een later MX-ontwikkelprotocol is een aparte beslissing.
Paperresultaten zijn alleen context: andere hoeveelheid data EN andere updatebudgetten.
Ook na honderd train-only-epochs is dit geen volledige paperreproductie.

## Uitvoering versus onderzoek

resume_core.py en smoke_support.py zijn byte-identiek overgenomen. De epochloop,
loss, optimizer/scheduler, metrics, timingmethode en RNG-beleid zijn behouden.
Zie TRAINONLY_CHANGE_MANIFEST.json en trainonly.patch voor verschillen met de vorige
pilot. De reeds bestaande utils.py-compatibiliteitspatch staat afzonderlijk in
compatibility.patch. Er is geen nieuwe compatibiliteitspatch aan de auteurscode.

cuDNN benchmark=True, workers=0 en pin_memory=True blijven. Geen AMP, torch.compile
of TF32-override. Alle backendinstellingen en softwareversies worden vastgelegd.
Hervatstaat bevat model, optimizer, scheduler en toevalstoestanden; er is geen
algemene belofte van bitexact dezelfde CUDA-resultaten. Logging heeft tijdsoverhead.
De notebook leest uitsluitend opgeslagen gegevens. Er zijn geen echte train-only-
onderzoeksresultaten in dit pakket. De laptoprun levert die pas op.
