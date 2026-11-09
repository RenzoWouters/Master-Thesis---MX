# Validatie van de 100-epochrunner

Alle hieronder genoemde uitvoeringen gebruikten uitsluitend drie synthetische CPU-beelden. Er is geen SVHN- of CUDA-training uitgevoerd en de synthetische accuracies/tijden zijn geen onderzoeksresultaten.

## Uitgevoerde controles

De echte `main`-lus is uitgevoerd tot epoch 100 (200 kleine optimizerupdates), gevolgd door hervatten van checkpoint 50 tot 100 (100 nieuwe updates) en één partiële epoch (2 updates). Model inclusief theta, beide Adamgroepen, scheduler en RNG waren exact gelijk tussen doorlopende en hervatte CPU-uitvoering. Alle LR-overgangen na 50/75/90, het volledige onbevroren checkpoint 50, gedeelde checkpointselectie, exportbits en contract-/CLI-weigeringen zijn gecontroleerd.

## Twee onderscheiden validatiebevindingen

1. De oorspronkelijke integratieharness bereikte alle functionele controles, maar faalde vervolgens in zijn laatste boekhoudcontrole: één oude aanroep van `s.verify_package()` controleerde de nog niet afgewerkte pakketmanifestatie. De herbruikbare harness is gecorrigeerd. De bestaande outputs zijn onafhankelijk opnieuw ingelezen; hiervoor is de training niet herhaald. `training_integration_report.json` vermeldt deze harnessfout expliciet.
2. Een latere notebookcontrole vond een afgebroken `history.json` in de oorspronkelijke synthetische hervatmap; ook de bijbehorende feedback-ZIP bevatte die afgebroken versie. De oorzaak kon niet worden vastgesteld. De oorspronkelijke bestanden blijven behouden. De volledige trace is hersteld vanuit het geverifieerde checkpoint 100 en de onveranderde laatste tijd-sidecar, en de productie-export is tweemaal opnieuw uitgevoerd in aparte mappen, zonder training. Beide nieuwe exports waren volledig.

De definitieve runner leest aan het einde van `export_results` de volledige opgeslagen geschiedenis terug, vergelijkt deze exact met de bedoelde trace en legt bytes, SHA256 en epochaantal vast. Een afgebroken, ontbrekend of inhoudelijk onvolledig JSON-bestand geeft nu een duidelijke fout voordat de export als geslaagd wordt gemeld. De definitieve export en zijn feedback-ZIP zijn gecontroleerd; aparte opzettelijk beschadigde testkopieën zijn afgewezen.

`runner_validation_binding.json` is het samenvattende eindrapport. `history_export_guard_validation.json` bevat de concrete bestands- en AST-controles. De AST-vergelijking toont dat tussen de numeriek geteste runner en de definitieve runner uitsluitend de extra exportcontrole, de bijbehorende hulpfunctie en een hashlib-import veranderden. Alle overige runtimebestanden zijn bytegelijk.

`test_training_integration.py` is de herbruikbare volledige CPU-harness. `verify_existing_training_integration.py` registreert de historische readback op de toenmalige runtime. `history_export_guard_test_executed.py` bewaart de werkelijk uitgevoerde lokale exportregressie met de paden naar de bewaarde synthetische fixtures; deze fixturebestanden zelf worden niet in het pakket opgenomen. De definitieve pakketmanifestatie wordt bij pakketassemblage afzonderlijk gecontroleerd.
