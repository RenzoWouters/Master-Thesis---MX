# Uitgevoerde controles — gezamenlijke directe referentie100

Deze map documenteert controles van de ontvangen GPU-pilot en van het nieuwe
honderd-epochpakket. De lokale CPU-fixtures zijn uitsluitend softwarecontroles;
hun accuracies en tijden zijn geen SVHN-resultaten of voorspellingen voor de
laptop. Er is hier geen echte SVHN-training of lange GPU-run gestart.

## 1. Ontvangen vijf-epochrun

`pilot_feedback_audit.json`: PASS op 62 onafhankelijke consistentiecontroles.
De bron is `feedback_bundle(20261023-125617).zip`, SHA256
`d8ad773e4b2298ac6c0885d6fdc4f04cc3c141f864196340ef83a97c4814ecfc`.

Gecontroleerd: 45 packagehashes, upstreambron en beperkte compatibiliteitspatch,
de drie aangepaste forwardselectorparen, dataconfiguratie, alle 30 metricrijen,
90 gewichtupdaterijen, zes geselecteerde rijen, selectie, tijden en exportscope.
De vijf epochs rapporteren 2.865 Adam-updates en 8.595 trainingspasses, correcte
BN-tellers, volledige datadekking, eindige gradients en FP32-Adamstaat.
Best en last zijn beide epoch 5.

De feedbackbundel bevat rapporten, geen checkpointtensors, datasetarrays of
minibatchlogs. BN-, gradient-, dekking- en exportcontroles zijn runtimeasserties
van de hashgecontroleerde code; de onderliggende tensors zijn hier niet opnieuw
geladen. Exportvalidatie in de run betreft gedecodeerde gewichten op het
trainingsdevice, geen extra volledige testpass vanuit het exportbestand.

## 2. Onafhankelijke review van de wijzigingen

`independent_review.json`: PASS. Veertien kernbestanden zijn byte-identiek aan
de geslaagde pilot. De AST van de trainingslus is gelijk, afgezien van een
foutmelding. Gecontroleerd zijn onder andere eindpunt100, hervatcontract,
LR-overgangen, E4M3-checkpointselectie, gemeten tijd en de twee notebookpanelen.
Een oude pilottekst in timingmetadata is vervangen door tekst passend bij de
volledige of gedeeltelijke run, uitsluitend in de nieuwe runner.

## 3. Echte runner met twee synthetische CPU-beelden

`reference100_validation_report.json`: PASS, circa 39,41 seconden voor de
uiteindelijke gerichte test. Testomgeving: Linux, Python 3.12, PyTorch 2.10.0+cpu,
torchvision 0.25.0+cpu. De gebruikersrun gebruikt zijn bestaande Windows/CUDA-
omgeving; de lokale test is geen reproductie van dat device.

`test_reference100.py` voert de echte CLI-main, het model, de gezamenlijke lus,
Adam, scheduler, checkpoints, exports en feedbackgeneratie uit. Alleen de
runtime/datacontext wordt vervangen door twee kunstmatige beelden en de
upstream CUDA-aanroep wordt binnen de test naar CPU geleid. De productie-CLI
krijgt geen CPU- of synthetische bypass.

De uiteindelijke test bevat 102 synthetische optimizerupdates:
100 doorlopende mini-epochs, één hervatte epoch 99→100 en één gecontroleerde
vroegtijdige stop. De productie-instellingen voor seed, formaten, data-aantallen,
batchgrootte en volledig schema worden apart geverifieerd; alleen de fixture
gebruikt twee beelden en batch twee.

Gecontroleerd:

- 100 volledige mini-epochs, 300 trainingspasses en 600 metricrijen;
- de daadwerkelijke LR-grenzen 50/51, 75/76 en 90/91;
- per minibatch drie forwards, twee losgekoppelde teachers en één Adam-update;
- A32, FP32-masters/momenten, de drie actieve BN-banken en eindige gradients;
- de juiste hard-/soft-losslabels en teacherketen;
- één gedeeld bestcheckpoint geselecteerd op het eerste E4M3-maximum;
- zes directe best/last-exports met exact herdecodeerde gewichten;
- resume 99→100 met exact dezelfde model-, Adam-, scheduler- en RNG-staat op CPU;
- een vroege stop zonder onterechte PASS of gemeten honderd-epochduur;
- weigering van pilotcontract en reeds voltooid epoch100-checkpoint vóór updates;
- CLI-grenzen, lege notebookkopie, feedbackbundel en ongewijzigde invoer;
- onderscheid tussen gemeten volledige duur en projectie bij een gedeeltelijke run.

De reeds op de laptop geslaagde numerieke kern is niet opnieuw volledig getest
op iedere codecwaarde. De nieuwe controles richten zich op het gewijzigde
honderd-epochgedrag. Bitexacte CPU-hervatting garandeert geen bitexacte
CUDA-hervatting onder het behouden cuDNN-beleid.

## 4. Notebook en aansluiting op de runner

`notebook_validation_report.json`: PASS voor gegenereerde complete100- en
STOPPED5-rapporten. Alle negen codecellen uitgevoerd; vijf figuurtypen visueel
gecontroleerd. Zeven opzettelijke fouten worden herkend, waaronder gemengde
bestepochs, ontbrekende modus, verkeerde teacher/losslabels, onterechte
voltooiingsclaim en een onjuiste gemeten totaalduur. Opslaan maakt nieuwe
figuurmappen en laat de invoer ongewijzigd.

`runner_notebook_integration.json`: PASS bij de werkelijke uitvoerbestanden
van de bovengenoemde synthetische runner. Continuous100, resume99→100 en
STOPPED1 geven geen analysefouten. De eerste twee bevestigen epoch100; de
vroeg gestopte run niet. Alle negen cellen van de door de runner gekopieerde
notebook zijn eveneens uitgevoerd met figuuropslag uitgeschakeld.

Er is geen uitgevoerde notebook opgeslagen of meegeleverd. Synthetische
figuren, datasets en checkpoints staan buiten het distributiepakket.

## 5. Samenstelling en reproduceerbaarheid

`ASSEMBLY_AUDIT.json` controleert het behouden pilotpakket, de 14 identieke
kernbestanden, de geteste runtimehashes, één lege actieve notebook en de
syntax van de actieve Python-bestanden. `PACKAGE_SHA256.json` wordt na het
opnemen van alle documentatie en validatierapporten opgesteld en volledig
gecontroleerd. De hashes van de geteste runtimecode blijven daarbij gelijk;
de uiteindelijke manifest bevat daarnaast documentatie en rapporten.

De lokale rapporten noemen hun fixturepaden voor herleidbaarheid. Die paden
zijn geen vereiste voor de Windows-run. Voor een nieuwe lokale test kan een
nieuwe lege uitvoermap worden meegegeven; alle testuitvoer blijft gescheiden
van onderzoeksexperimenten.
