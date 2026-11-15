# Lokale controle van het afleverpakket

Deze controles zijn uitgevoerd op **Python 3.12, PyTorch 2.10.0+cpu en
torchvision 0.25.0+cpu**, met twee CPU-threads. Er is geen CUDA-device gebruikt
en er zijn geen SVHN-beelden getraind of geëvalueerd. De synthetische fixtures
zijn technische tests en leveren geen onderzoeksaccuracy op.

De normale gebruikerscheck controleert aanvullend de bestaande Python
3.13-/PyTorch 2.10.0-/torchvision 0.25.0-omgeving en CUDA. Het pakket installeert
of upgradet niets. De volgende stap is die check op de laptop, gevolgd door de
afgesproken vijf echte SVHN-epochs.

## Numerieke codecs

`new_codecs_cpu_report.json`: beide nieuwe codecs slagen tegen een onafhankelijke
scalaroracle. Alle 248 eindige E5M2-codes en alle 64 E3M2-codes, signed zero,
afrondingsmidpoints, saturatie, ongeldige codes, 255 eindige E8M0-schalen,
FP32-machtsgrenzen en extremen, padding, packing en identity-STE zijn gecontroleerd.
De bestaande E4M3-, E2M3- en E2M1-codecs zijn byte-identiek aan de aangeleverde
directe referentie; hun hashes staan in `REUSE_MANIFEST.json`.

## Gezamenlijke trainingsstap

`joint_cpu_report.json`: twee opeenvolgende stappen van de echte CNN-8 komen
exact overeen met een onafhankelijk uitgeschreven vijf-formaatreferentie op
CPU. Dit betreft logits, teacherdoelen, losses, somgradiënten, modelstaat,
BatchNorm, Adamstaat en RNG/dropoutverbruik. Elke minibatch heeft vijf forwards
en backwards, één `zero_grad` en één optimizerupdate. E2M1 krijgt het gemiddelde
van de losgekoppelde 6-bitkansen. De oorspronkelijke initialisatietensors zijn
na de BN-hernoeming exact behouden. Alle vijf compacte exports zijn teruggelezen
en numeriek met de actieve modellen vergeleken.

## Runner en fouten

`runner_cpu_report.json`: de echte runner is met kleine, expliciet synthetische
datasets tot epoch 5 uitgevoerd. Een aaneengesloten run en stoppen na epoch 2
met hervatting komen op CPU exact overeen in model-, optimizer-, scheduler-,
RNG-staat en alle metrics. Eerdere ruwe epochbestanden worden gecontroleerd en
naar de nieuwe runmap gekopieerd. Verkeerde protocollen en een voltooid
epoch-5-checkpoint worden als hervatbron geweigerd.

Een geïnjecteerde checkpointschrijffout op epoch 5 bewaart de raw metingen voor
foutanalyse, maar houdt het aantal voltooide epochs en de aggregate metrics op
epoch 4. Deze fout wordt niet als geslaagde proef gemeld.

`cli_guard_review.json`: CPU-training, een interne CPU-testvlag bij training,
epoch 6 en hervatten tijdens `check` worden afgewezen. De laatste terminalregel
is dan `GLOBAAL RESULTAAT: FAIL`. De uitgevoerde volledige CPU-check eindigt met
`GLOBAAL RESULTAAT: PASS`. Dit zijn technische controles, geen claim over
SVHN-accuracy of CUDA-compatibiliteit zonder de laptopcheck.

De testcode voor de runner staat in `test_runner.py`. Ze gebruikt alleen kleine
synthetische invoer en maakt nieuwe tijdelijke resultaatmappen. Een vaste seed
bewijst geen bitidentieke CUDA-hervatting; de productiebackend blijft zoals in
de eerdere proef.

## Notebook

`notebook_validation_report.json`: complete vijf epochs, een onvoltooide run,
check zonder metrics en metingen zonder bevestigd checkpoint worden correct
weergegeven. Dubbele rijen en accuracy die niet klopt met de tellingen worden
geweigerd. Een primaire epoch-5-tabel vereist vijf geldige testregels én
`report.completed_epochs == 5`. Curves hebben afzonderlijke panelen per formaat
en uniforme stippen met percentages. Er zijn geen ingevulde synthetische
resultaten in de meegeleverde notebooktemplate.

## Reikwijdte

De controle toont aan dat het pakket de afgesproken proef uitvoert op de lokaal
geteste CPU-paden. De gebruikerscheck moet de CUDA-paden nog bevestigen. De
vijf-epochtraining meet daarna het leerverloop en de looptijd. Ze bewijst geen
eindaccuracy na honderd epochs, statistische superioriteit, bitexacte
GPU-herhaalbaarheid of hardwarewinst.
