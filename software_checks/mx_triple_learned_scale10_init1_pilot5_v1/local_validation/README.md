# Lokale validatie van de init×1-pilot

Deze controles betreffen softwarecorrectheid. Ze bevatten geen nieuwe SVHN-resultaten en geen CUDA-meting.

## Geslaagde controles

- `check_entry_report.json`: de echte `check`-ingang, met uitsluitend CPU-routering in de testharnas; codec-, schaalgradient-, grens-, initialisatie- en bronlocatiecontroles geslaagd.
- `training_integration_report.json`: de echte trainingslus op drie synthetische beelden, vijf kleine epochs; stoppen na epoch 2 en hervatten tot 5 leverde exact dezelfde gewichten, Adam-status, scheduler en RNG op. De lagere precisies zijn uitsluitend uit de E4-bronbits afgeleid. Opgeslagen exports zijn onafhankelijk uitgelezen en gecontroleerd. De CLI weigert epoch 6 en 100.
- `initialization_oracle_review.json`: het oorspronkelijke auteursmodel, de oude theta6-adapter en de nieuwe theta0-adapter hebben op dezelfde CPU exact dezelfde FP32-basisgewichten en BN-toestand.
- `notebook_qa_report.json`: 29 controles op inhoud, schaalkeuzes, asbereiken, grensgedrag en presentatie; negen figuren visueel beoordeeld.
- `cpu_notebook_qa_report.json`: alle dertien notebookcodecellen uitgevoerd op een volledige, gedeeltelijke en hervatte synthetische run. De meegeleverde notebook zelf is onuitgevoerd.

## CPU-oracle versus historische GPU-run

De CPU-initialisatiehash verschilt van de historische Windows/cu128-hash, ook bij het ongewijzigde oorspronkelijke auteursmodel. De precieze onderliggende platformoorzaak is niet vastgesteld. Daarom gebruiken de CPU-testharnassen uitsluitend in hun eigen proces een aparte, onafhankelijk berekende CPU-referentie. De oorspronkelijke onderzoeksbestanden blijven ongewijzigd. De productierunner blijft streng controleren op de werkelijke historische Windows/CUDA-basisgewichten en BN-toestand; er is geen CLI-optie om deze controle te omzeilen.

Synthetische scores worden niet met historische SVHN-accuracies vergeleken. Alle synthetische testuitvoer is expliciet gemarkeerd.

## Reproduceren van de twee CPU-harnassen

Deze scripts zijn ontwikkelaarscontroles in een afzonderlijke CPU-omgeving, geen opdrachten voor de normale SVHN-run. Zij verwachten twee argumenten: het pakketpad en een nog niet bestaande uitvoermap. Voorbeeld:

```bash
python local_validation/test_init1_check.py /volledig/pakketpad /nieuwe/controlemap
python local_validation/test_init1_integration.py /volledig/pakketpad /nieuwe/integratiemap
```

Uitgevoerd met Linux, Python 3.12 en PyTorch 2.10.0+cpu. De gebruiker voert daarnaast de echte `check` uit in de bestaande Windows `.venv-train`, zoals beschreven in de hoofd-README.

## Waarschuwingen en afbakening

Tijdens notebookvalidatie meldde de lokale pandas-versie een deprecation van het overgenomen `float_precision='round_trip'`. Het inlezen en alle consistentiecontroles slaagden. Eerdere ontwikkelcontroles ontdekten onjuiste aannames in de testharnassen over absolute uitvoerpaden en gekopieerde schaalbytes bij uitsluitend factor ×1; die harnassen zijn gecorrigeerd. De productiecode hoefde daarvoor niet te worden gewijzigd.

Een CPU-test kan geen snelheid, accuracy of bitexacte hervatting op de gebruikers-GPU aantonen. De oorspronkelijke cuDNN-instellingen blijven behouden.
