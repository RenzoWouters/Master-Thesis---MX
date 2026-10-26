# Validatie van de voortzetting vanaf epoch 55

De productiecode en de resultaatnotebook zijn gecontroleerd met kleine, expliciet gemarkeerde synthetische CPU-proeven. Er is hier geen SVHN-training uitgevoerd en er zijn geen nieuwe onderzoeksaccuracies of CUDA-tijden gemeten.

## Trainingsstaat en voortzetting

Een synthetische oorspronkelijke training liep werkelijk van epoch 1 tot 100. De ongewijzigde eerdere pilotcode vertrok vervolgens van checkpoint 50 en voerde epochs 51–55 uit met vaste offsets. Vanuit dat echte synthetische checkpoint 55 is de nieuwe runner getest.

De volledige voortzetting 56–100 slaagt. Model, BatchNorm, Adam-momenten, scheduler en RNG worden exact hersteld. De vijf bestaande bevroren epochs blijven behouden; alle 4.080 theta-waarden en harde offsetkeuzes blijven na iedere update gelijk aan epoch 50. Gewichten en actieve BatchNorm-banken leren verder. De gewichts-LR daalt na epochs 75 en 90; de theta-LR blijft nul.

Afzonderlijke hervatproeven vanaf epochs 74 en 89 geven op CPU exact dezelfde model-, optimizer-, scheduler- en RNG-toestand als ononderbroken voortzetten. Dit is een softwarecontrole en geen garantie op bitexacte CUDA-uitvoering.

## Selecties en exports

De proef selecteerde drie verschillende herkomsten: een globaal beste checkpoint vóór de freeze, een beste checkpoint uit de vijf-epochpilot en het nieuwe eindcheckpoint 100. Alle exports zijn gecontroleerd tegen het contract van hun eigen fase. De compacte exports zijn onafhankelijk uit broncodes, bronschalen en verpakte blokkeuzes gereconstrueerd, zonder de gewichtencoders opnieuw te gebruiken. Een checkpoint van vóór de freeze behoudt zijn eigen theta.

De rapportage onderscheidt nieuwe epochs 56–100, de volledige bevroren fase 51–100 en de absolute teller 1–100. De notebook controleert ook de geërfde geschiedenis en de geselecteerde herkomst.

## Invoer, opdrachtinterface en behoud van bestanden

Elf corrupte toestand- of referentiegevallen en vijf onveilige uitvoerlocaties worden afgewezen. De echte `main(check)` is in een afzonderlijk CPU-proces uitgevoerd, inclusief de geërfde codec- en gradienttests en het herstel van checkpoint 55. De echte `main(run)` is met standaarddoel 100 en een gecontroleerde stop na 56 getest. Negen ongeldige opdrachtvarianten worden afgewezen. De tests substitueren uitsluitend CPU-routing, kleine synthetische datasets en expliciete fixture-identiteiten; deze route is geen publieke trainingsoptie.

De 127 bestanden uit het eerdere pilotmanifest en de volledige feitelijke feedback zijn ongewijzigd in het pakket opgenomen en opnieuw op hashes gecontroleerd. De onafhankelijke broncodecontrole staat in `CODE_REVIEW.md` en `code_review.json`.

## Notebook en distributie

De notebook is uitgevoerd op de volledige en gedeeltelijke synthetische uitvoer. Tabellen, selectievensters, figuren en integriteitscontroles worden beschreven in `notebook_validation_report.json`. De meegeleverde notebook is leeg: alle cellen hebben geen uitvoer of uitvoernummer. De runner schrijft na de echte training een eigen lege kopie met het juiste resultatenpad.

`package_assembly_validation.json` legt de laatste bron-, manifest-, protocol- en notebookcontroles vast. `PACKAGE_SHA256.json` in de hoofdmap bevat de hashes van de complete distributie. De echte laptopcontrole blijft `run_train.py check`; de daadwerkelijke CUDA/SVHN-voortzetting begint pas met `run_train.py run`.
