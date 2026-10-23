# Codewijzigingen tegenover de vijf-epochpilot

De achttien kern- en bronbestanden uit de werkende pilot blijven byte-identiek.
Dit omvat alle codecs, adapters, de oorspronkelijke trainingslus met broncache,
STE, de pure post-hocbitfuncties, de corezelftests, runtime, dataset- en
resume-infrastructuur. `REUSE_MANIFEST.json` legt dit vast.

## Trainingsrunner

`run_train.py` krijgt een afzonderlijk honderd-epochcontract, uitvoerpad en
pakketnaam. De configuratie verschilt alleen in de schema- en doelbeschrijving,
het eindpunt 100 in plaats van 5 en de rapportage voor epoch 100. De bestaande
LR-mijlpalen 50/75/90 waren al in de pilot geconfigureerd en blijven gelijk.

De runner begint opnieuw vanaf initialisatie. Resume is uitsluitend geldig
binnen dit nieuwe contract. Een gecontroleerde stop vóór epoch 100 blijft
hervatbaar en krijgt geen onterechte voltooiingsstatus. De gemeten tijd voor
honderd epochs wordt alleen gevuld als alle honderd epochs aanwezig zijn;
bij een kortere geschiedenis blijft de totale honderd-epochtijd een projectie.

Er verandert niets aan de gewichtsroutes, schaalkeuze, bitbewerking, gradients,
losses, teacherketen, BN-beleid, dataset, optimizer, seed of backendinstellingen.
De compacte bron- en afgeleide exports en voorspellingstellingen blijven gelijk.
De exacte runnerdiff staat in `provenance/pilot5_to_reference100_run_train.py.patch`.

## Notebook en documentatie

De nieuwe lege notebook en helper herkennen het honderd-epochcontract en tonen
accuracyfiguren voor epoch 100 en het door E4M3 gekozen beste checkpoint. Vroeg
stoppen en check-only worden expliciet onderscheiden van een voltooide run.
Alle zes figuren blijven aanwezig, inclusief de voorspelde klassendiversiteit.
De beschrijvende notebookmetadata benoemen nu expliciet exponent- en mantissa-
truncatie en het eindpunt 100; de echte training volgt de runnerconfiguratie.

`pilot_analysis` bevat de ongewijzigde bronmetingen van de uitgevoerde pilot,
de relevante directe referentie, de onafhankelijke audit en de duiding.
Deze onderzoeksresultaten zijn gescheiden van synthetische softwarevalidaties.
Er wordt één lege honderd-epochnotebook geleverd, zonder uitgevoerde kopie.
