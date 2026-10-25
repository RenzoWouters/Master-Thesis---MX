# Validatie van het volledige ×32-pakket

**Eindstatus: PASS.** De echte vijf-epochpilot is geanalyseerd en de gewijzigde honderd-epochrunner en notebook zijn lokaal gecontroleerd. Er is hier geen nieuwe volledige SVHN-training of GPU-benchmark uitgevoerd.

## Echte pilot als vertrekpunt

De aangeleverde feedback van `20261024_192400_420212_joint_run` bevat vijf volledige SVHN-epochs, 2.865 optimizerupdates en 8.595 trainingspasses. Alle 66 pakketbestanden komen overeen met de afgeleverde pilot. Initialisatie, dataset, broncode, omgeving en backend stemmen overeen met de meegeleverde ×64-referentie. De numerieke exportaudits in de rapporten zijn consistent met de vaste schaaloffset +5 en mantissatruncatie met behoud van schaal.

De grote model- en exporttensors zitten niet in de feedback. De controle van de echte pilot berust daarom op de rapporten, ruwe tabellen, hashes en gelogde CUDA-controles; er is hier geen nieuwe inferentie op die echte checkpoints uitgevoerd. Zie `../pilot_analysis/independent_audit.json` en `../pilot_analysis/RESULTATEN.md`.

De bestaande pilotnotebook is in geheugen op de echte feedback uitgevoerd: alle elf codecellen en acht figuren werken, de echte ×64-vergelijking is actief en alle 33 bronbestanden bleven ongewijzigd. Er is geen uitgevoerde notebook opgeslagen of opnieuw afgeleverd.

## Numerieke kern

Achttien codec-, adapter-, ondersteunings- en bronbestanden zijn byte-identiek aan de geslaagde ×32-pilot. De bestaande uitvoerige kernvalidatie is aan deze hashes gebonden en bewaard onder `../provenance`. Alleen experimentidentiteit, eindpunt en rapportagebeleid veranderen. Het vaste schaalverschil blijft +5 (×32), activaties blijven A32 en masters/gradients/Adam-momenten blijven FP32.

## Runner en hervatting

`test_training_integration.py` heeft de echte runner uitgevoerd op drie expliciet synthetische CPU-beelden. Het volledige geval omvatte honderd mini-epochs, 200 optimizerupdates, 600 trainingspasses en 600 metricrijen. Inclusief hervattings- en stopgevallen waren er 210 synthetische updates. Dit zijn technische controles, geen SVHN-accuracyresultaten.

De vier LR-fasen en de werkelijke overgangen na epochs 50, 75 en 90 zijn gecontroleerd. Hervatten vanaf 99 naar 100 en vanaf 49/74/89 over een LR-overgang herstelde model, Adam, scheduler en RNG exact in deze CPU-controle. Dit geeft geen aanvullende bitexacte CUDA-garantie voor de oorspronkelijke backend.

Vroege stop blijft onvolledig. Verkeerde pilot- of ×64-contracts en reeds voltooide checkpoints worden voor updates geweigerd. Broncodegebruik, drie passes per update, identity-STE, FP32-staten, BN-banken, bronexports en reconstructie uitsluitend uit E4M3-codes zijn gecontroleerd. Ook de oorspronkelijke `main/check`-ingang slaagde. De productiecode is aan de geteste snapshot gebonden in `independent_runner_review.json`.

## Notebook

De leverbare notebook heeft elf lege codecellen en acht figuurtypen. De controles omvatten zeven positieve gevallen en negentien negatieve rapportgevallen. Epoch 100 en het gezamenlijke E4M3-bestcheckpoint worden afzonderlijk getoond. De ×64-vergelijking gebruikt dezelfde trainingsduur en selectieregel; bij onvolledige runs wordt alleen de gezamenlijk gemeten epochrange gebruikt.

Alle cellen zijn daarnaast in geheugen uitgevoerd op echte uitvoer van de synthetische runner: honderd epochs met E4M3-best op epoch 13 en een run gestopt na epoch 1. De ruwe invoer bleef ongewijzigd. Synthetische training wordt niet als echte SVHN-vergelijking gepresenteerd. De checkpointfiguren en vergelijkingslegenda zijn visueel gecontroleerd. De definitieve helper-, notebook- en testhashes staan in `notebook_validation_report.json` en `runner_notebook_integration.json`.

## Eindassemblage

`final_assembly_validation.json` legt vast dat de actuele bestanden met de finale testhashes overeenkomen, de oude pakketten en aangeleverde feedback ongewijzigd zijn, de ruwe ×64-referentie behouden is en de leverbare notebook geen outputs bevat. `../PACKAGE_SHA256.json` dekt alle geleverde bestanden behalve het manifest zelf. De ZIP bevat geen caches, modelcheckpoints of nieuwe trainingsresultaten.

De resterende omgevingscontrole is de gewone `check`-opdracht op de bestaande Windows/CUDA-trainomgeving, vóór `run`. Deze opdracht verandert het trainingsprotocol niet.
