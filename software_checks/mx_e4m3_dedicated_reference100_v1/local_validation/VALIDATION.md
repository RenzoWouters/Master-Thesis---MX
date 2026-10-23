# Uitgevoerde validatie

## Werkelijke pilot op de laptop

De echte SVHN-pilot van vijf epochs is geanalyseerd uit de aangeleverde
`feedback_bundle(20261022-164547).zip`, SHA256
`f0401e655e91ed447885291aecaf1d3390fd3078aaaa07e7d4d5484ae43a8391`.
De run is `20261022_184112_866638_mx_run`; het rapport vermeldt PASS.
De vijf epochs bevatten 2.865 optimizerupdates, volledige train-/testdekking,
eindige gradients en updates van alle zes MX-gewichtslagen. Best en last
zijn epoch 5: 24.246/26.032 correcte testvoorspellingen (93,1392%).
De rapporten melden ongewijzigde bron- en datasetbestanden.

Alle 21 aangeleverde bestanden zijn byte-identiek behouden in
`pilot5_analysis/raw/`. De aparte pilotnotebook bevat uitsluitend deze echte
metingen. De acht codecellen zijn uitgevoerd; vier figuren zijn ingebed en
als PNG opgeslagen en visueel gecontroleerd. Zie
`pilot5_analysis/ANALYSIS_AUDIT.json` voor hashes en de analysecontrole.
Trainingscheckpoints en de dataset zaten niet in de feedbackbundel: hier is
geen nieuwe SVHN-evaluatie of herhash van de checkpointbestanden uitgevoerd.

## Nieuwe honderd-epochrunner — korte synthetische CPU-controle

`reference100_validation_report.json` rapporteert PASS. De echte runner,
upstream modelcode, Adam, trainingslus, checkpoint- en exportcode zijn gebruikt
met twee synthetische afbeeldingen op CPU. Dit zijn 102 zeer kleine
optimizerupdates in ongeveer 30 seconden, geen SVHN-training van honderd epochs
en geen GPU-prestatiemeting.

Gecontroleerd:

- Het standaard-eindpunt is 100; PASS als voltooide referentie ontstaat pas
  na epoch 100.
- Alle vier LR-fasen zijn daadwerkelijk doorlopen: epochs 1–50 met 0,001,
  51–75 met 0,0001, 76–90 met 0,00001 en 91–100 met 0,000001.
- Best volgt het eerste strikte maximum; last is bij voltooiing epoch 100.
- Beide MX-exports reconstrueren exact de gecontroleerde gewichten.
- De gemeten honderd-epochduur verschijnt uitsluitend na voltooiing.
- Hervatten van synthetische epoch 99 tot 100 gaf op CPU dezelfde model-,
  optimizer-, scheduler-, RNG- en metricstaat als doorlopend uitvoeren.
- Een stop na een volledige epoch wordt als STOPPED gerapporteerd.
- Een pilotcheckpoint, een al voltooid honderd-epochcheckpoint en ongeldige
  CLI-keuzes worden geweigerd vóór nieuwe optimizerupdates.
- Bronbestanden, synthetische invoer en pakketbestanden bleven tijdens de
  controle ongewijzigd.

De verwachte FAIL-statussen bij geweigerde checkpoints in het JSON-rapport
zijn geslaagde negatieve controles. De productie-CLI heeft geen CPU- of
synthetische trainingsmodus; de test past de context uitsluitend binnen het
testproces aan. Er is geen SVHN-data geladen of gedownload.

Testomgeving: Linux, Python 3.12.14, PyTorch 2.10.0+cpu,
torchvision 0.25.0+cpu, NumPy 2.3.5 en SciPy 1.17.0.
De laptopomgeving is niet gewijzigd. Deze CPU-controle garandeert geen
bitexacte hervatting op CUDA; het bestaande cuDNN-beleid blijft behouden.

## Honderd-epochnotebook — controles zonder training

`notebook_validation_report.json` rapporteert PASS voor twee expliciet
synthetische rapportfixtures: volledig 100 en gestopt op 37.
Alle codecellen zijn uitgevoerd en de belangrijkste figuren zijn visueel
gecontroleerd. De labels maken de synthetische herkomst zichtbaar.

De controle verifieert de lege aangeleverde template, selectie bij een
gelijke beste score, ontbrekende epoch-100-balk bij STOPPED, gemeten versus
geschatte tijd, de gerapporteerde TF32-instelling en ongewijzigde invoer.
Expliciet figuren opslaan maakt een nieuwe map met herkomstinformatie.
Daarnaast zijn de echte rapportbestanden van de bovenstaande synthetische
runnercontrole door de notebookhelper gelezen: de volledige run en de vroege
stop gaven geen analysefouten en de juiste voltooiingsstatus.

## Grenzen en hergebruik

De codec, MX-adapter, ondersteuning voor het model, data, smoke-controle en
hervatten zijn byte-identiek aan de op de laptop geslaagde pilot-v2; de
hashes staan in `REUSE_MANIFEST.json`. De wijziging betreft het eindpunt,
experimentidentiteit en de eindrapportage/notebook. De oorspronkelijke
trainingslus is behouden.

Er zijn nog geen echte honderd-epochaccuracies of looptijden met dit nieuwe
pakket. De schatting van circa 42 minuten komt uitsluitend uit de echte
vijf-epochpilot. Alle synthetische controles zijn validatie van programmagedrag
en mogen niet als onderzoeksresultaten worden gebruikt.
