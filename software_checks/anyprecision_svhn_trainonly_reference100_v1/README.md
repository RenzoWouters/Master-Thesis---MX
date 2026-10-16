# CNN-8 / SVHN train-only: volledige referentie van 100 epochs

Nieuwe run vanaf initialisatie; geen extra, geen MX, geen pilot-warmstart.
Dit pakket bouwt op de geslaagde vijf-epochversie voort. Het verandert geen
trainingshyperparameters, alleen het eindpunt en de rapportage voor honderd epochs.

## 1. Plaatsen

Pak de inhoud uit in een NIEUWE map:

    C:\Users\renzo\master-thesis-mx\software_checks\anyprecision_svhn_trainonly_reference100_v1

run_svhn.py moet rechtstreeks in die map staan. De rootbestanden behouden hun
inhoud voor de hashcontroles. Bewerk niet de oude experimentmappen of pakketten.

## 2. Controle zonder training

Open PowerShell in VS Code:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_trainonly_reference100_v1\run_svhn.py" check
```

Ga alleen verder na EINDSTATUS: PASS. Deze controle downloadt niets en doet
geen optimizerstappen; de vormcontrole gebruikt geen echte trainingsbeelden.
Gebruik je bestaande .venv-train; geen herinstallatie of upgrade nodig.
Bij FAIL eerst het rapport bekijken; verander niet blind softwareversies.
De check en de training zijn aparte processen, dus de check gebruikt geen
trainings-RNG uit de latere run.

## 3. Volledige run starten

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_trainonly_reference100_v1\run_svhn.py" run --stop-after-epoch 100
```

Alle honderd epochs lopen automatisch achter elkaar. Geen --checkpoint en geen
resume voor deze nieuwe referentie. De CLI weigert een checkpoint bij run.
Honderd is het enige toegestane eindpunt in deze productie-CLI; voor vijf epochs
blijft het eerdere pilotpakket bestaan. De learning rate daalt NIET eerder.

Verwachte startcontroles:

    REFERENTIE: 100 train-only epochs; geen extra, geen MX, geen pilot-warmstart.
    DATA PASS: train=73257, test=26032; extra NIET gebruikt; ...
    MINIBATCHES per epoch: train=573; test=204; batch=128, drop_last=False.
    NIEUWE initialisatie. Geen ResNet- of proefcheckpoint ingeladen.
    EPOCH 1/100 (origineel schema: 100); LR=0.001

De bestaande train_32x32.mat en test_32x32.mat worden gelezen uit data\svhn.
Er is daarom geen --download nodig. Het extra-bestand mag daar blijven staan,
maar wordt niet geopend of gewijzigd. Bestaande corrupte databestanden worden
geweigerd, niet overschreven. Alleen als train/test ontbreken kan --download
worden toegevoegd; de oude auteurs-URL en veilige integriteitscontrole blijven.

Laat de laptop aangesloten en voorkom slaapstand tijdens de run. Houd dezelfde
uitvoeringsomstandigheden als in de proef. Er is geen automatische wijziging
van Windows-energie-instellingen. De eerdere proef suggereerde circa 57 minuten;
werkelijke eindtijd en accuracies worden in deze run gemeten.

## 4. Tijdens de training

De bovenste terminalregel geeft de unieke resultatenmap onder:

    C:\Users\renzo\master-thesis-mx\reference_runs\anyprecision_svhn_trainonly100

De submap eindigt op _run. De _check-map hoort alleen bij de voorafgaande controle.
De originele UPSTREAM epoch 0 is de eerste epoch; onze namen tellen vanaf 1.

LR tijdens epochs 1-50: 0.001; 51-75: 0.0001; 76-90: 0.00001; 91-100: 0.000001.
Elke epoch bevat alle trainbeelden eenmaal en een volledige evaluatie op test
bij alle vijf precisies. Na honderd epochs zijn er 57300 optimizerstappen.
Er is geen vroegtijdige stop op basis van een testscore.

## 5. Einde en notebook

Een geslaagde volledige run eindigt met EINDSTATUS: PASS, completed_epoch=100,
full_training_completed=true en optimizer_steps=57300 in run_report.json.
Controleer waarschuwingen; PASS betekent technische voltooiing, geen bepaalde
vereiste eindaccuracy of bewijs van paperreproductie.

Open SVHN_trainonly_reference100_resultaten.ipynb UIT DE RESULTATENMAP en kies
.venv-train als kernel; Run All. Het exacte runpad staat al ingevuld. Het notebook
leest opgeslagen JSON/CSV en begint geen training. De template in de pakketmap
staat niet op jouw runpad en is niet bedoeld als eerste notebook om uit te voeren.

Belangrijkste output:

- checkpoints\epoch_00001.pt tot epoch_00100.pt met .sha256-bestanden;
- best_checkpoint.pt en last_checkpoint.pt, aparte rollen en checksums;
- epoch_metrics.csv: alle epochs, train/test en vijf precisies;
- selected_metrics.csv: vijf scores van hetzelfde beste checkpoint en vijf van last;
- completion_summary.json en learning_rate_schedule.csv: voltooiing en LR-controles;
- epoch_times.csv, timing_estimate.json: werkelijk gemeten tijd en aparte projectie;
- history.json, dataset_manifest.json, source_manifest.json, fullrun_change_manifest.json;
- pip_freeze.txt, pip_check.txt, run.log, eventuele warnings.log;
- feedback_bundle.zip: kleine rapportenbundel, geen gewichten, dataset of minibatchlogs.

Het beste checkpoint is het eerste strikte maximum op W32/A32-testaccuracy.
Niet iedere precisie kiest haar eigen epoch. Last is na voltooiing epoch 100.
Ook als beide rollen naar dezelfde epoch wijzen blijven ze zichtbaar.
Alle ruwe resultaten lokaal bewaren; deel feedback_bundle.zip uit de _run-map.

## 6. Alleen bij een onverwachte onderbreking

Een enkele Ctrl+C vraagt een gecontroleerde stop NA de lopende epoch en opslag.
Een tweede Ctrl+C stopt direct; gedeeltelijke epochs worden niet hervat.
Sluit niet expres de terminal om te stoppen. Een STOP_AFTER_EPOCH-bestand in de
actieve runmap heeft hetzelfde epochgrensgedrag.

Hervatten blijft een noodvoorziening voor checkpoints van DIT honderd-epochpakket.
Geen checkpoint van de vijf-epochproef gebruiken en geen warmstart op bestaande
train+extra-resultaten. Houd de hele eerdere runmap en het pakket intact: de
historie verwijst naar die eerdere checkpoints/sidecars. De hervatte uitvoering
schrijft naar een nieuwe _resume-map, zonder eerdere data te overschrijven.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_trainonly_reference100_v1\run_svhn.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_EEN_CHECKPOINT_UIT_DEZE_100EPOCHRUN\epoch_00050.pt"
```

Het pad hierboven is uitsluitend een voorbeeld voor noodgebruik. De normale run
heeft dit niet nodig. Herstel vereist dezelfde omgeving, bron, data en instellingen.
Een goede RNG-restauratie is geen belofte van bitexacte CUDA-uitvoering met het
behouden cudnn.benchmark=True-beleid.

## 7. Wetenschappelijke status

Train-only is de afgesproken ontwikkelvariant van het auteursrecept, niet dezelfde
data als in de paper. Zelfde aantal epochs betekent minder optimizerupdates.
Paperwaarden worden daarom als context getoond, niet als gelijkbudget-succesgrens.
De testset is betrokken bij checkpointselectie, dus de beste testscore is niet
onafhankelijk van die selectie. Er is nog geen MX, opgeslagen-code-truncatie of
continual-learning-experiment. Zie PROTOCOL.md en COMPATIBILITY.md.

## 8. Lokale validatie

Zie local_validation/VALIDATION.md en de JSON-testresultaten. Die zijn synthetische
CPU-softwaretests, geen echte SVHN-training of GPU-tijdmeting. Ze testen ook honderd
mini-epochs en de LR-grenzen. Herhalen vraagt extra notebook-testafhankelijkheden;
voor de gewone productie-CLI zijn die tests NIET nodig.

    python local_validation/test_svhn.py PAD_NAAR_EEN_NIEUWE_LEGE_TESTMAP
