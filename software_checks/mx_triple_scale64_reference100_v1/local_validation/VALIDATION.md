# Softwarevalidatie van het honderd-epochpakket

## Uitkomst en onderscheid met de echte training

De softwarecontroles van dit pakket zijn **PASS**. Ze gebruiken kleine
synthetische CPU-fixtures. Er is hier geen echte honderd-epochtraining op
SVHN of CUDA uitgevoerd. De echte vijf-epochpilot van de gebruiker staat
afzonderlijk onder `pilot_analysis`, inclusief de oorspronkelijke meettabellen.

De 42 runtime-, documentatie- en herkomstbestanden zijn vóór de definitieve
controles bevroren. De bewijsrapporten bevatten de hashes van de geteste
versies. Het definitieve pakketmanifest omvat daarnaast analysebestanden,
validatierapporten en testharnassen.

| Controle | Bewijsbestand | Uitkomst |
|---|---|---|
| Werkelijke check-entry met CPU-adapter | `check_entry_validation_report.json` | PASS |
| Werkelijke honderd-epochloop op drie synthetische beelden | `training_integration_report.json` | PASS |
| Stop, resume en werkelijke LR-grenzen 50/75/90 | `training_integration_report.json` | PASS |
| Notebook op vijf rapportsituaties en negentien negatieve gevallen | `notebook_validation_report.json` | PASS |
| Notebook op werkelijk opgeslagen synthetische runneruitvoer | `runner_notebook_integration.json` | PASS |
| Onafhankelijke beoordeling van runner en proefcontract | `independent_runner_review.json` | PASS |
| Eerdere pakketten ongewijzigd | `prior_packages_unchanged.json` | PASS |
| Laatste samenstellingscontrole | `final_assembly_report.json` | PASS |

## Ongewijzigde numerieke core

De achttien bestanden in `REUSE_MANIFEST.json` zijn byte-identiek aan het
geslaagde scale64-pilotpakket. De eerdere uitvoerige code-/schaal-/STE-controle
is opgenomen als `provenance/scale64_pilot5_core_training_validation.json`.
Zijn hashes zijn tegen de huidige bestanden gecontroleerd.

Die controle omvat alle 254 eindige E4M3-codes, exacte overeenkomst met
post-hoc B6/B4, geldige schaaloffsets en expliciete overflowafwijzing,
de identity STE en behoud van G(W)-afgeleiden. Twee driepassupdates zijn
vergeleken met een onafhankelijke handmatige lus; alle 346 vergeleken tensors
waren exact gelijk. Het nieuwe volledige pakket verandert deze core niet.

De nieuwe integratie controleert bovendien onafhankelijk de bitbewerking,
broncode +6, mantissaschaal en identity STE zonder extra backwardfactor64.
Lagere tensor- en elementencoders waren tijdens de volledige run en resumes
geblokkeerd. De werkelijke check-entry voert zijn eigen synthetische
codec- en trainingscontroles eveneens uit en eindigde met PASS.

## Volledige runner en hervatten

De definitieve integratie gebruikt het echte model, `main`, de echte
trainingslus, checkpointopslag en exports. Alleen in het testproces worden
CUDA-aanroepen naar CPU omgeleid en de data vervangen door drie synthetische
beelden met batchgrootte 2. Dit test ook de laatste onvolledige minibatch.
De honderd epochs en LR-mijlpalen 50/75/90 blijven in deze controle werkelijk
behouden. De productieconfig behoudt de volledige SVHN-omvang en batch128.

De definitieve integratie bevat **210 kleine Adam-updates**:

- 200 updates voor honderd synthetische epochs;
- 2 updates voor hervatten van epoch99 naar100;
- 2 updates voor een gecontroleerde stop na epoch1;
- 6 updates voor de afzonderlijke resumes 49→50, 74→75 en 89→90.

Deze definitieve controle duurde ongeveer 44 seconden op CPU. Model,
optimizer, scheduler en RNG waren bij de geteste hervatgrenzen exact gelijk
aan de bijbehorende doorlopende toestand. Dit bewijst geen bitexacte
CUDA-herhaling; het oorspronkelijke cuDNNbenchmark-beleid blijft actief.

Gecontroleerd zijn verder: één E4M3-bron per laag/minibatch, drie passes en
één Adam-update, detached teachers, eindige FP32-toestand, BN-tellers,
broncache, klassehistogrammen, voorbeelddekking, E4M3-selectie en correcte
voltooiingsstatus uitsluitend na epoch100. Pilot-, oude naïeve en reeds
voltooide checkpoints worden afgewezen vóór herstel of nieuwe updates.

De twee geselecteerde bronexports en vier lagere exports zijn gecontroleerd
op codes, schalen, bronhashes en het ontbreken van de zes FP32-MX-masters.
Reconstructie uitsluitend uit opgeslagen E4M3-broncodes levert exact gelijke
logits, terwijl G(W) en encoders verboden zijn.

## Notebook en figuren

De notebook is getest op complete100, vroege stop, constante klasse,
grote eindige CE-waarden en check zonder training. Alle tien codecellen
en zes figuren zijn verwerkt; negentien onjuiste consistentiegevallen
worden afgewezen. Alle zes figuren zijn visueel gecontroleerd. De
accuracycurves en checkpointfiguur zijn ook bij de eindcontrole bekeken.
Deze previews zijn expliciet synthetisch en zijn geen onderzoeksresultaten.

Vervolgens is de notebook uitgevoerd op werkelijk opgeslagen synthetische
uitvoer van continuous100, stop-na1 en resume99→100. Alle tien cellen en
zes figuren werkten; de respectievelijk 836, 45 en 44 invoerbestanden bleven
byte voor byte gelijk. Model- en checkpointimports waren geblokkeerd.

Epoch100 wordt uitsluitend getoond als het voltooide rapport en de gegevens
dat ondersteunen. De tweede checkpointgroep volgt de E4M3-selectie, met
alle drie modi uit datzelfde checkpoint. Bij een onderbreking blijft
epoch100 expliciet nog niet gemeten. De CSV-parser gebruikt `round_trip`.
De meegeleverde notebook heeft geen outputs en `SAVE_FIGURES=False`.

## Auditgrenzen

De echte vijf-epochfeedbackbundel bevat rapporten en metingen, maar geen
checkpointtensors. Haar analyse valideert de metadata, aantallen en
gerapporteerde livecontroles; zij draait geen nieuwe inferentie op die echte
gewichten. De synthetische softwarevalidatie van dit volledige pakket
valideert wél zijn opslag- en reconstructieroutes met lokale testtensors.

De softwarevalidatie voorspelt geen accuracy na honderd echte epochs.
De Windows/CUDA-check en de echte run worden door de gebruiker uitgevoerd
met de commando's in `README.md`.
