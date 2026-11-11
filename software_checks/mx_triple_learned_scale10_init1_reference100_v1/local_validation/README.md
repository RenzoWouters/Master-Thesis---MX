# Validatie van de init×1-training van honderd epochs

Deze map documenteert softwarecontroles op kleine synthetische CPU-fixtures. Zij bevat geen nieuwe SVHN-accuracies en geen gemeten GPU-trainingstijd.

## Uitgevoerde controles

- `check_entry_report.json`: de echte `check`-ingang met CPU-routering, codec- en schaalgradientzelftests, beginwaarde theta0, beginweights-/BN-vergelijking met het onafhankelijke oorspronkelijke CPU-model, bronlocatieguards en vroege notebookcreatie. PASS.
- `training_integration_report.json`: de echte `main`, drie-pass-trainingslus, Adam, scheduler, checkpoints en exports. Honderd mini-epochs op drie beelden (twee minibatches per epoch), een gedeeltelijke run en hervatting vanaf epoch 50. De hervatte eindtoestand is exact gelijk in gewichten, theta, Adam, scheduler en RNG. Alle LR-overgangen op 50/75/90 en actieve schaalupdates na 50 zijn gecontroleerd. Onafhankelijke bit-/schaaldecode van de opgeslagen exports slaagt. Het pilotcontract en ongeldige CLI-eindpunten worden afgewezen. PASS.
- `reference_contract_report.json`: de echte historische referentie bevat alle honderd epochs; de vergelijking produceert 300 rijen. Bij een expliciete vijf-epochreferentie blijven de overige 95 epochs als ontbrekend gemarkeerd. Afwijkende basisgewichten en datavolgorde worden afgewezen. Dit is uitsluitend een schemacontrole met kopieën van bestaande traces; geen nieuwe onderzoeksdata. PASS.
- `plot_qa_report.json`: 23 controles op de echte referenties, plots, checkpointselectie, asbereiken en lege/gedeeltelijke uitvoer. De relevante figuren zijn visueel beoordeeld. PASS.
- `notebook_cells_qa_report.json`: alle 17 codecellen uitgevoerd op de complete100-, partial2-, resume50_to100- en lege check-fixtures. Alle vier geslaagd; de geleverde notebook zelf is onuitgevoerd. Alleen de bestaande pandas-deprecation over `float_precision='round_trip'` werd gemeld; parsing en consistentiecontrole slagen.
- `independent_code_review.json`: onafhankelijke read-onlybeoordeling van de 100-epochplanning, numerieke invarianten, bronidentiteit en veilige hervatting. PASS.

Alle zestien hergebruikte Pythonbestanden, inclusief de numerieke adapter, codecs en trainingsondersteuning, zijn byte-identiek aan de daadwerkelijk uitgevoerde init×1-pilot. `REUSE_MANIFEST.json` legt dit vast. De nieuwe runnerhash is vastgelegd in de rapporten; wijzigingen in documentatie of de notebook veranderen de geteste numerieke runtime niet.

## Reproduceren

De harnassen verwachten het pakketpad en een nieuwe uitvoermap:

```bash
python local_validation/test_init1_check.py /volledig/pakketpad /nieuwe/checkmap
python local_validation/test_init1_full100_integration.py /volledig/pakketpad /nieuwe/integratiemap
python local_validation/test_reference_contract.py /volledig/pakketpad /nieuwe/referentiecontrole
```

De tests zijn uitgevoerd met Linux, Python 3.12 en PyTorch 2.10.0+cpu. De productierunner blijft de bestaande Windows Python 3.13/CUDA-omgeving vereisen. Er zijn geen afhankelijkheden van de gebruiker gewijzigd en er is geen SVHN gedownload.

## CPU-initialisatie tegenover de historische gebruikers-GPU

De oorspronkelijke auteursarchitectuur heeft op de lokale CPU een andere initiële byteshash dan de historische Windows/cu128-run, ook zonder de MX-adapter. De precieze platformoorzaak is niet vastgesteld. Daarom berekent het CPU-checkharnas een eigen oorspronkelijke CPU-oracle in een apart interpreterproces. Alleen binnen de harnassen wordt een expliciete synthetische CPU-referentie gebruikt. De historische bestanden blijven ongewijzigd en de productierunner blijft streng toetsen tegen de oorspronkelijke gebruikersreferentie. Er is geen productie-CLI-optie om die controle uit te zetten.

Synthetische scores worden niet vergeleken met historische onderzoeksaccuracies. Een CPU-controle bewijst geen bitexacte CUDA-hervatting; de oorspronkelijke cuDNN-instellingen blijven behouden. Voer de echte `check` op de gebruikers-GPU uit voordat de SVHN-training wordt gestart.
