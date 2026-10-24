# Softwarevalidatie van de vijf-epochpilot met schaal ×64

## Uitkomst en reikwijdte

De lokale softwarecontroles zijn geslaagd. Er is uitsluitend met kleine
synthetische CPU-fixtures gewerkt. Er is hier geen SVHN-dataset geladen of
gedownload, geen echte SVHN-training uitgevoerd en geen GPU-accuracy of
GPU-looptijd gemeten. De echte proef start op de laptop met de commando's in
`README.md`: eerst `check`, daarna bij PASS `run`.

De productiecode is vóór de definitieve controles bevroren. De rapporten
bevatten de SHA256-hashes van de daadwerkelijk geteste bestanden. De laatste
pakketcontrole verifieert deze hashes, het hergebruik en de lege notebook.
`PACKAGE_SHA256.json` omvat daarnaast de meegeleverde documentatie,
herkomstbestanden en validatierapporten.

## Uitgevoerde controles

| Onderdeel | Bewijsbestand | Uitkomst |
|---|---|---|
| Bitbewerking, schaal, STE, handmatige updates, bronexports | `core_training_validation.json` | PASS |
| Werkelijke `main/check`-route op CPU | `check_entry_validation_report.json` | PASS |
| Werkelijke `main/run/resume`-route met synthetische data | `training_integration_report.json` | PASS |
| Notebook op vijf geconstrueerde rapportsituaties | `notebook_validation_report.json` | PASS |
| Notebook op werkelijk geschreven synthetische runneruitvoer | `runner_notebook_integration.json` | PASS |
| Onafhankelijke controle van runner en proefcontract | `independent_runner_review.json` | PASS |
| Eerdere pakketten ongewijzigd | `prior_packages_unchanged.json` | PASS |
| Laatste samenstellingscontrole | `final_assembly_report.json` | PASS |

### Codes, schaaloffset en gradients

Alle 254 eindige E4M3-codes zijn vergeleken met een onafhankelijke bitstring-
en scalaire schaalberekening. De nieuwe lagere forwards zijn exact gelijk
aan de gearchiveerde post-hocfuncties voor B6 en B4. Alle 249 geldige
bron-schaalcodes die met 6 verhoogd kunnen worden zijn gecontroleerd,
inclusief nulblokken en padding. De zes broncodes 249 tot en met 254 worden
bij deze bewerking expliciet geweigerd. Ook niet-eindige gedecodeerde
FP32-gewichten worden geweigerd.

Voor alle drie modi is de exacte forwardwaarde en de identity STE over de
volledige route gecontroleerd. Er komt geen extra factor 64 in backward;
de oorspronkelijke afgeleide van G(W) blijft behouden. Twee volledige
updates met drie precisiepasses zijn vergeleken met een onafhankelijke
handmatige trainingslus. Alle 346 vergeleken tensors waren exact gelijk,
zowel met het oorspronkelijke backendbeleid als in de deterministische
controle. Daarbij zijn Adam, de drie BN-banken, detached teachers, FP32,
broncache en RNG-toestand meegenomen.

De opgeslagen E4M3-bron is herladen en gebruikt om alle drie modi opnieuw
op te bouwen, terwijl toegang tot G(W), MX-masters en encoders was geblokkeerd.
Alle drie logitvergelijkingen waren exact gelijk. Oude contractschema's en
onjuiste schaalmetadata zijn als negatieve controles afgewezen.

### Volledige runner, opslaan en hervatten

De integratietest gebruikt het echte model, de echte trainingslus en de echte
opslag-/exportroutes. Uitsluitend in de testharness worden CUDA-aanroepen naar
CPU omgeleid en worden drie synthetische beelden met batchgrootte 2 aangeboden.
Dit test ook de laatste onvolledige minibatch. De test gebruikt LR-mijlpalen
2/3/4 om het hervatten over LR-overgangen te controleren. De productiecode
behoudt ongewijzigd batchgrootte 128 en mijlpalen 50/75/90.

In totaal zijn 18 synthetische Adam-updates uitgevoerd: 10 in een doorlopende
vijf-epochrun, 6 bij hervatten vanaf epoch 2 en 2 bij een stop na epoch 1.
Het eindmodel, de optimizer, scheduler en RNG waren voor doorlopend versus
hervat op deze CPU-fixture exact gelijk. Dit geeft geen bitexacte
CUDA-hervatgarantie; het oorspronkelijke cuDNN-beleid blijft actief.

Tijdens de werkelijke run en resume waren zowel lagere tensorencoders als
lagere elementencoders geblokkeerd. Per laag en minibatch is exact één
E4M3-bron gebruikt. De twee lagere schalen zijn broncode +6, en de
mantissastap kopieert de E2M3-schaal. Gecontroleerd zijn onder meer de drie
trainpasses, één Adam-stap, voorbeelddekking, BN-tellers, klassehistogrammen,
selectie door E4M3 en de zes exports voor het gemeenschappelijke beste en
laatste checkpoint.

De echte check-entry is eveneens uitgevoerd met alleen de CPU-runtimeadapter.
Deze eindigde met PASS. Verwachte afwijzingen van een oud checkpointcontract,
een al voltooide pilot en ongeldige CLI-opties zijn afzonderlijk getest.
De productie-CLI accepteert geen synthetische modus en kan niet naar
100 epochs worden omgezet met `--stop-after-epoch 100`.

### Notebook

De meegeleverde notebook bevat tien lege codecellen. Alle cellen zijn in het
geheugen uitgevoerd op geconstrueerde situaties voor vijf complete epochs,
een eerdere stop, één voorspelde klasse, grote eindige CE-waarden en een
check zonder training. Negentien onjuiste metadata-/consistentiesituaties
zijn gecontroleerd. De CSV-parser bewaart decimale waarden met
`float_precision='round_trip'`.

Daarna zijn dezelfde cellen uitgevoerd op de werkelijk geschreven
synthetische runnerresultaten van de doorlopende run, vroege stop en resume.
Er zijn zes figuren aangemaakt, waaronder de accuracycurves en de gezamenlijke
checkpointvergelijking. De respectievelijk 76, 45 en 60 invoerbestanden bleven
byte voor byte gelijk. Model- en checkpointimports bleven daarbij geblokkeerd.
Alle zes figuren zijn visueel gecontroleerd; de accuracyfiguren zijn ook bij
de eindcontrole bekeken. De previews gebruiken expliciet synthetische data
en zijn geen onderzoeksresultaten.

De notebook maakt duidelijk dat epoch 100 nog niet gemeten is. Hij vergelijkt
binnen de pilot epoch 5 met E4M3-best, telkens met de drie precisies uit
hetzelfde checkpoint. `SAVE_FIGURES=False` is de standaard. Er wordt geen
uitgevoerde notebook meegeleverd.

## Hergebruik en praktische beperking

De vijftien bestanden in `REUSE_MANIFEST.json`, waaronder de broncode-archive,
codecs, gedeelde trainingmechanica en resumeondersteuning, zijn byte voor
byte gelijk aan het eerdere naïeve vijf-epochpakket. De eerdere pilot,
100-epochtraining, post-hocevaluatie en notebookcorrectie zijn ongewijzigd
gebleven. De nieuwe training schrijft in een eigen resultatenroot.

Een technische PASS is geen garantie op hoge accuracy. Ook een lage accuracy,
één voorspelde klasse of nul eindige gradients blijft een geldige te
bespreken meetuitkomst. Deze controles onderbouwen dat de afgesproken proef
wordt uitgevoerd en gerapporteerd; alleen de nieuwe echte training kan haar
leergedrag op SVHN laten zien.
