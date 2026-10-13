# Protocol: Any-Precision GitHub200 referentie v1

## Status en doel

Zelf getrainde referentie volgens het uitvoerbare CIFAR-10-trainingsrecept in de
gebruikers-ZIP, met checkpoints en gescheiden eindtestevaluatie. Geen MX,
geen training met truncatie vanuit opgeslagen codes, geen hardwarewinstmeting.
De README-resultaten zijn externe auteursresultaten, geen acceptatiedrempel.

## Vastgelegde instellingen

- Model: ongewijzigde resnet20q; expand=5; 6.786.330 parameters inclusief BatchNorm.
- Alle 50.000 officiele trainingsbeelden in de oorspronkelijke loader-volgorde.
- Geen onafhankelijke validatiesplit; geen testbeelden tijdens training.
- Batch 128; shuffle=True; drop_last=False; 391 minibatches per epoch (laatste 80).
- 200 epochs: 78.200 gezamenlijke optimizerstappen bij volledige voltooiing.
- Ieder minibatch: W32/A32 hard-label cross entropy; daarna 8,4,2,1 met
  recursieve soft-target cross entropy; gradients opgeteld; een Adam-stap.
- Adam via originele optimizerfabriek; LR0.001; weight_decay=0.
- MultiStepLR via originele schedulerfabriek, gamma0.1, milestones100/150/180.
- Begin epochs1-100:0.001; 101-150:0.0001;151-180:0.00001;181-200:0.000001.
  Een epochnummer duidt een volledige doorgang aan; sched.step volgt na de epoch.
- Originele preprocessing: RandomHorizontalFlip, RandomCrop(32,padding4),
  ToTensor en de originele CIFAR-normalisatie; test zonder willekeurige augmentatie.
- FP32-mastergewichten. Lagere precisies bij elke forward opnieuw uit deze gewichten.
  De 32-bitroute houdt de ReLU/tanh-transformatie van de code, geen gewone FP32-ResNet.

## Uitvoerkeuzes

Nieuw run_reference.py; de oorspronkelijke train.py-launcher wordt niet gestart.
Voor de trainingsstap wordt ongewijzigd run_pilot.train_batch hergebruikt.
Voor herstel worden ongewijzigd resume_core en de gecontroleerde runtime-initialisatie
hergebruikt. Alleen dezelfde drie upstream-compatibiliteitsaanpassingen als eerder.

Seed20261009; doorlopende Python/NumPy/CPU/CUDA/loader-RNGs; workers0;
pin_memory=False; deterministische algoritmen; AMP, TF32 en torch.compile uit.
De moderne PyTorch/torchvision-omgeving is niet numeriek identiek bewezen aan
Python3.7/PyTorch1.1.0/torchvision0.2.1 uit de oorspronkelijke README.
Geen automatische fallback naar een ander apparaat of trainingsbudget.

## Sessies, checkpoints, gegevensbescherming

Eerste sessie standaard vijf ECHTE epochs; geplande duur blijft200. request-stop
pauzeert aan een complete epochgrens. Iedere uitvoering heeft een unieke map.
Nieuwe training start vanaf seed-initialisatie, niet vanaf pilot/check-checkpoints.

Checkpoint epoch0 bewaart de initialisatie. Daarna iedere complete epoch een
nieuw .pt-bestand plus .pt.sha256. Bevat model/BatchNorm, Adam-staat, scheduler,
RNGs, tellers, historie, bron-/config-/omgevingscontract. Schrijven via een nieuw
pending-bestand, publiceren en checksum schrijven, dan inhoud volledig teruglezen.
Nooit bestaande checkpoints overschrijven; geen stroomuitvalduurzaamheid geclaimd.
Bij ontbreken van het complete paar geldt de laatste EERDERE complete epoch.

Herstellen gebeurt na het aanmaken van alle objecten, RNG als laatste. De volgende
trainingsstap zet train() en de juiste precisies. Configuratie, dataset-/bronhashes en
runtime-omgeving moeten overeenkomen. Oudere 400-epochcheckpoints zijn niet geldig
voor dit nieuwe protocol. Ze blijven ongewijzigd bewaard.

## Resultaten en evaluatie

Per epoch: trainingsloss (eigen objective), hard-label CE en top1/top5 voor alle
vijf precisies, voorbeelden, learning rate, model/hash, inputvolgordehashes,
eindige gradients/states en BatchNorm-controles. Trainingsmetrics zijn online,
met veranderende gewichten/augmentatie; geen vergelijkbare testaccuracy.

Testbeleid: vooraf gekozen endpoint200, een checkpoint, alle10.000 testbeelden,
alle vijf precisies, origineel eval()/no_grad-pad. Geen aparte beste epoch per precisie,
geen modelselectie op testaccuracy. Dit verschilt van het oorspronkelijke train.py
(dat elke epoch de testset als val gebruikt en op W32/A32 de beste checkpoint kiest).
De herkomst van de gepubliceerde README-checkpointselectie is niet vastgesteld.

De evaluate-opdracht maakt naast test_metrics.csv een github_comparison.csv/json.
Referentie: README-rij Resnet-20-Any(recursive), waarden [91.89,93.90,93.86,93.75,94.11]
voor [1,2,4,8,32]. Verschil = eigen accuracy - auteursaccuracy, in procentpunten.
De README en targets zijn bevroren in het pakket, met controlegetallen, niet live
opgehaald tijdens training. Geen classificatie PASS/FAIL op grond van scoreverschil.
Een scoreverschil is niet op zichzelf bewijs voor een bug. Historieseeds, numerieke
omgeving, endpointselectie en runprovenance zijn niet allemaal bekend.

Wanneer rekentijd tot een vroeger eindpunt leidt, wordt de reden VOOR testinzage
vastgelegd en het resultaat benoemd als een afgebroken/ingekorte uitvoering van dit
schema. Dat wordt niet voorgesteld als een volledig afgewerkte200-epochreferentie.

## Tijd en opslag

Trainingstijd omvat data-invoer, vijf forward/backwardpaden, Adam, metrics, inputhashing,
diagnose en batchlogging. Checks en checkpoint+readback apart; CUDA synchronisatie
bij tijdgrenzen. Geen zuivere GPU-kernbenchmark. Eerste epoch van elke sessie apart;
resterende schatting gebaseerd op overige afgeronde epochs en actuele config200.
Geen eindtesttijd, wachttijd tussen sessies of garanties voor latere thermische drift.

Er worden201 checkpoints gemaakt als alle epochs worden voltooid (incl.epoch0).
Opslagprojectie is een indicatie op basis van de actuele bestandsgrootte; geschiedenis
groeit. Bestaande data en eerdere experimentele resultaten worden niet gewijzigd.
