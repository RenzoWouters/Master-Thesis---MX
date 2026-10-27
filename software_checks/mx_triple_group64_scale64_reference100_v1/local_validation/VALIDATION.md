# B64 ×64 full100 — validatie

De nieuwe proef behoudt alle 19 numerieke modules, runtimeonderdelen en productiezelftests van de daadwerkelijk geslaagde B64-pilot. Alleen de runnerlevenscyclus en de analyse worden uitgebreid van vijf naar honderd epochs. De voorbereiding bevat geen echte SVHN-training en geen CUDA-benchmark.

## Echte pilot als basis

De gebruikersbundel van 27 oktober 2026 is gecontroleerd op 397 consistentievoorwaarden. Vijf epochs, 2.865 updates en 8.595 trainingspasses zijn volledig geregistreerd. Pakket, initialisatie, datasets, broncode, A32/FP32, schaalroute, exportgeometrie en beide referenties zijn consistent. De bestaande pilotnotebook blijft bruikbaar. De audit en ongewijzigde bronbundel staan in `pilot_analysis`.

## Volledige runner op kleine synthetische data

`training_integration_report.json` bevat de uitvoering van de werkelijke hoofdloop met drie synthetische CPU-beelden:

- Honderd kleine epochs en alle vier werkelijke LR-fasen.
- Hervatten 99→100 en over de grenzen 49→50, 74→75 en 89→90; model, Adam, scheduler en RNG exact binnen de CPU-test.
- Stop na een volledige epoch met correcte gedeeltelijke status.
- Afwijzen van pilot-, B16-, B32-, incompatibele en reeds voltooide checkpoints vóór updates.
- Onafhankelijke reconstructie van alle zes geselecteerde exports uitsluitend uit opgeslagen E4M3-broncodes, met schaaloffset +6.
- Controle van bitmapping, identity-STE, werkelijke laagvormen en B64-kostengeometrie.
- Kopiëren van alle drie referentiemappen en de onuitgevoerde notebook met het correcte resultatenpad.

De gerapporteerde actieve groepsgrootte moet overeenkomen met de configuratie en de echte modelgeometrie: **64**. De kosten worden in de test expliciet aan de echte laagvormen getoetst. Een overgenomen B16-fixturewaarde is tijdens de voorbereiding door de notebookcontrole onderschept en gecorrigeerd; de definitieve rapporten zijn afkomstig van de daaropvolgende echte heruitvoering van de synthetische test. De productiecode hoefde daarvoor niet te wijzigen.

Deze controles omvatten in totaal 210 kleine optimizerupdates over de verschillende testgevallen. De synthetische accuracies zijn geen onderzoeksresultaten en worden niet met de echte historische accuracies vergeleken.

## Werkelijke opstarttak

`startup_regression_report.json` controleert `create_context(check_only=False)` met de echte bestands- en datasetvalidatie. Alleen externe gegevensinhoud/lader en CPU-apparaatkeuze zijn expliciete testfixtures.

De juiste bestanden blijven `train_32x32.mat` en `test_32x32.mat`. Virtuele datasets hebben de oorspronkelijke aantallen 73.257/26.032. De echte modelconstructie, Adam, scheduler, dataloaders, transformaties en eerste train-/testbatches worden uitgevoerd. Het extra-bestand wordt niet gehasht of geladen. Er worden nul trainingsepochs en nul optimizerupdates in deze opstarttest uitgevoerd.

## Notebook

De definitieve notebook bevat 24 cellen, waarvan 14 codecellen, met lege outputs en `execution_count=None`. Ze wordt gecontroleerd op de volledige en gedeeltelijke synthetische runneroutputs, lege data en gerichte ongeldige gevallen. De ingelezen resultaten blijven ongewijzigd.

Een selectiefixture maakt het E4M3-maximum, een gelijke latere score en de eigen maxima van de lagere modi verschillend. Alle drie precisies blijven bij één E4M3-geselecteerde epoch. Beide volledige historische referenties worden gevalideerd op hun eigen globale E4M3-maximum: B32 epoch 42 en B16 epoch 53. Gedeeltelijke runs begrenzen de vergelijking tot hetzelfde beschikbare bereik en krijgen geen epoch-100-score.

Onjuiste kosten, verkeerd contract, gemengde checkpointselectie en gewijzigde referentiegegevens worden geweigerd. Synthetische resultaten kunnen geen echte onderzoeksvergelijking opleveren. De grafieken worden visueel gecontroleerd. Alleen de onuitgevoerde notebook wordt geleverd.

## Onafhankelijke review en eindassemblage

`runner_implementation_audit.json` bindt de werkelijk geteste code aan haar hashes. `independent_delta_audit.json` controleert de numerieke gelijkheid met de geslaagde B64-pilot en de runnerstructuur met de bestaande B16-full100-runner. De volledige `run_epochs`-functie blijft gelijk aan de werkende full100-lus.

De eindassemblage controleert `PROTOCOL.json` tegenover `run_train.config()`, de gelijkheid van de 19 behouden modules, de definitieve notebookhashes, alle historische bronbestanden en de ZIP-inhoud. Oude pilotrapporten staan uitsluitend als provenance. Privésnapshots van de runtimevalidatie kunnen eerdere document- of notebookhashes bevatten; de finale code en notebook worden afzonderlijk aan hun definitieve validatierapport gebonden.

De ongewijzigde volledige codecsuite wordt bij deze stagewijziging niet opnieuw lokaal uitgevoerd. De gerichte full100-test herhaalt wel bit/schaal/STE- en exportcontroles. De gebruiker voert `check` op de Windows/CUDA-omgeving uit. CPU-exactheid en een vaste seed zijn geen garantie voor bitexacte CUDA-herhaling of hervatting.
