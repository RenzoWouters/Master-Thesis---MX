# Ontwikkelvalidatie: wat wel en niet is uitgevoerd

Datum: 9 oktober 2026. Omgeving: Linux/CPU, Python3.13.5, PyTorch2.10.0+cpu,
torchvision0.25.0+cpu, NumPy2.3.5, Pillow12.3.0. Dit is NIET de Windows-laptop.

## Geslaagde controles

1. `test_parity_and_protocol.py`: twee opeenvolgende minibatches vergeleken met
   de daadwerkelijke `forward`-functie uit de aangeleverde `train.py`. Alleen de
   CUDA-aanroepen van die functie worden voor deze lokale CPU-test in het geheugen
   naar `.to('cpu')` vertaald. Modeltoestand, optimizertoestand en trainingslosses
   waren na beide stappen exact gelijk aan die van de nieuwe `train_batch`.
   Ook de stratificatie, disjuncte pools, geneste selecties en batchgroottegewogen
   aggregatie zijn gecontroleerd. Zie parity_protocol_report.json.
2. `test_offline_validation.py`: twee afzonderlijke processen met kunstmatige
   CIFAR-vormige bestanden; elk 20 train/20 validatiebeelden, 1 epoch, batch2,
   ongewijzigd resnet20q expand5 en alle vijf precisies. De oorspronkelijke CIFAR10-
   decoder en oorspronkelijke transformatiefunctie werden gebruikt. Alleen de
   checksummetadata van de datasetklasse werden IN DE TESTHARNESS IN HET GEHEUGEN
   vervangen door checksums van de kunstmatige bestanden. Deze omzeiling staat
   NIET in run_pilot.py en wordt bij normaal uitvoeren NIET aangeroepen.
   Beide processen kregen exact dezelfde modeltoestand, metrics, samplevolgorde,
   geaugmenteerde minibatchhashes en splitindices. Checkpointchecks slaagden;
   nul opgevangen waarschuwingen. Zie offline_repeatability_report.json.
3. `test_downloader.py`: gesimuleerde HTTP-response met een klein testarchief;
   extractie, geverifieerd hergebruik en weigering om corrupte bestaande bestanden
   te overschrijven slaagden. Zie downloader_test_report.json. Geen echt netwerk.

## Niet uitgevoerd / niet vastgesteld

- De echte CIFAR-10-download is in deze omgeving geprobeerd en mislukte wegens
  `Temporary failure in name resolution`. Een downloadtoolpoging slaagde evenmin.
  Zie real_download_attempt_failed.log. Er werd hier geen echte CIFAR-10-training
  uitgevoerd. De echte download wordt pas op de laptop getest.
- De volledige pilot (5.000/1.000, drie epochs, batch32) is hier NIET uitgevoerd.
- De nieuwe pilotrunner is hier NIET op Windows/CUDA uitgevoerd.
- Geen uitspraken over werkelijke CIFAR-10-accuracy, convergentie of hardwarewinst.
- De scores in de offline logs zijn uitsluitend van kunstmatige testbeelden.

## Tijdens ontwikkeling gecorrigeerde testharnessfout

De eerste aparte pariteitstest vergat CUBLAS_WORKSPACE_CONFIG in te stellen
voordat de bestaande configure-helper werd gebruikt. Dit gaf een KeyError.
De TESTHARNESS is gecorrigeerd; de productie-pilotrunner stelde deze variabele
reeds correct in en hoefde daarvoor niet te worden gewijzigd.
Zowel de oorspronkelijke foutlog als de geslaagde herhaling zijn bewaard.

Deze controles vervangen niet de nog uit te voeren pilot op de gebruikerslaptop.
