# Lokale softwarevalidatie — tien schaalkeuzes, freeze na epoch 50

Deze rapporten betreffen technische correctheid. Er is geen echte SVHN-training uitgevoerd, geen dataset gedownload en geen CUDA-accuracy of GPU-snelheid gemeten. Alle lokale trainingsbeelden zijn synthetische CPU-tensors. De productie-CLI biedt geen synthetische of CPU-trainingsoptie.

## Uitgevoerde controles

- `parent_fixture_report.json`: 55 werkelijk uitgevoerde synthetische epochs met drie beelden en twee minibatches per epoch. De beginstaat bevat alle tien offsets, beide grenzen en exacte/naastliggende FP32-waarden rond alle halve afrondingsdrempels. Deze fixture levert volledige Adam-, BN-, scheduler- en RNG-staat bij epoch 50.
- `training_integration_report.json`: productievoortzetting 51–55, stoppen na 52, exact hervatten tot 55 en gecontroleerd stoppen. Model, Adam, scheduler en RNG zijn op CPU exact gelijk bij ononderbroken en gesplitste voortzetting. Dezelfde oorspronkelijke beeldvolgorde blijft behouden. Theta en alle harde keuzes blijven exact vast; gewichten en BN trainen verder. Alle tien histogramcategorieën en nulwaarden voor sprong-/wisseldiagnostiek zijn gecontroleerd. Compacte exports worden onafhankelijk uit vierbitnibbles gelezen en alleen uit broncodes gereconstrueerd; 4.080 groepen gebruiken 2.040 selectorbytes.
- `check_entry_validation_report.json` en `freeze_checks.json`: de productiecontrole met oorspronkelijke codec-/gradienttests, strikt parentherstel en 15 freezecontroles. Ook een theta-afwijking binnen dezelfde harde bin wordt geweigerd. De E4M3-bronschaal blijft dynamisch.
- `notebook_validation_report.json` en `notebook_layout_validation.json`: afzonderlijke controle van analyse, complete/partiële/hervatte runs, fout-/checkuitvoer en figuurlayout. De geleverde notebook heeft geen uitgevoerde cellen of opgeslagen figuren.

## Gevonden en opgeloste rapportagefout

De eerste integratie vond dat een vooraf klaargezette notebook bij export opnieuw met exclusieve bestandscreatie werd geschreven. Dat gaf `FileExistsError`. Alleen `write_notebook` is aangepast: een bestaande notebook en helper worden geaccepteerd als hun inhoud exact overeenkomt; afwijkende bestanden worden niet overschreven.

De oorspronkelijke numerieke integratieresultaten zijn behouden. AST- en SHA256-controles bewijzen dat alle andere runnerfuncties en de numerieke kern ongewijzigd zijn. Daarna zijn de opgeslagen toestanden/exports opnieuw gelezen en is alleen de eerder falende productie-ingang opnieuw uitgevoerd: één synthetische epoch, twee updates. Die export slaagde. Ook een ontbrekend broncheckpoint levert nu aantoonbaar een FAIL-feedbackbundel met een lege notebook, zonder training.

`validation_attempts.json` vermeldt daarnaast twee lokale testfixturecorrecties en de installatie van het ontbrekende `pip` via `ensurepip` in de tijdelijke CPU-testomgeving. Dit zijn geen fouten in onderzoeksdata. Oude binary-freeze-rapporten zijn niet als validatie van dit pakket hergebruikt.

## Reproduceren

Gebruik een CPU-PyTorch-omgeving met de bijbehorende dependencies. De scripts `build_parent_fixture.py`, `test_training_integration.py` en `test_check_entry.py` accepteren expliciete nieuwe uitvoermappen. `test_reporting_tail.py` documenteert de beperkte hercontrole na de notebookcorrectie en leest de bewaarde synthetische integratie-uitvoer; het verandert geen onderzoeksbestanden.

De werkelijke Windows/CUDA-omgeving en het echte checkpoint 50 worden pas door `run_train.py check` op de laptop gecontroleerd. De CPU-gelijkheid is geen garantie op bitexacte CUDA-hervatting.
