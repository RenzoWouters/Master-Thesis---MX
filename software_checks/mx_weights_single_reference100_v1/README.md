# Volledige eenprecisietraining: controle en MX, elk 100 epochs

Twee APARTE modellen, ieder opnieuw vanaf dezelfde vastgelegde initialisatie.
Start de controle en MX-run apart en NA ELKAAR, niet tegelijkertijd op dezelfde GPU.
Elke opdracht doorloopt alle 100 epochs automatisch. Geen pilot-warmstart.

## Plaatsen

Pak de inhoud uit in:
`C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_single_reference100_v1`
`run_train.py` staat rechtstreeks in die map. Vervang geen oude experimentmap.

## Controleren

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_single_reference100_v1\run_train.py" check
```

Pas doorgaan na `EINDSTATUS: PASS`. Deze check doet enkele synthetische updates
in beide modi; geen SVHN-training of download. Gebruik de bestaande werkende
`.venv-train`. Geen installaties/upgrades nodig; pandas/matplotlib zijn voor de
notebook en zijn niet nodig in de trainingslus.

## 1. Controle starten (een volledige ononderbroken training)

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_single_reference100_v1\run_train.py" run --mode control --stop-after-epoch 100
```

Dit traint W32/A32 met de oorspronkelijke tanh/normalisatie, zonder MX.
Wacht totdat deze opdracht klaar is. Bewaar het vermelde resultatenpad.

## 2. Daarna MX starten (een andere volledige ononderbroken training)

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_single_reference100_v1\run_train.py" run --mode mx --stop-after-epoch 100
```

Dit traint E2M3/A32, met FP32-mastergewichten en FP32-Adamstaat. Het gebruikt
GEEN gewichten van de controlerun of de pilot. Dezelfde seed start opnieuw.

Beide modi: volledige SVHN-train (73257), test (26032), batch 128, Adam lr .001,
weight decay 0, hard-label cross-entropy, een forward/backward/update per batch.
Er is geen distillatie, truncatie of activatie-/gradientkwantisatie. De zes MX-lagen,
E8M0-schaal en groepen van 32 blijven gelijk aan de geslaagde pilot.

De bestanden in `data\svhn` worden hergebruikt en niet gewijzigd. Het extra-bestand
wordt niet gebruikt, geopend of gehasht. Alleen als train/test ontbreken kan
uitdrukkelijk `--download` worden toegevoegd; er is geen automatische download.

LR tijdens epochs 1-50: .001; 51-75: .0001; 76-90: .00001; 91-100: .000001.
573 trainingsbatches per epoch; 57300 optimizerupdates per volledige run.
De bestaande auteursfunctie telt UPSTREAM-epochs vanaf 0, onze rapporten vanaf 1.

## Resultaten en vergelijking

Nieuwe mappen:
`mx_runs\mx_weights_single_reference100\<tijdstip>_control_run`
`mx_runs\mx_weights_single_reference100\<tijdstip>_mx_run`

Iedere run bewaart 100 afzonderlijke checkpoints en checksums, volledige hervatstaat,
`best_checkpoint.pt`, `last_checkpoint.pt`, JSON/CSV, `training_summary.json`,
`feedback_bundle.zip` en `MX_single_reference100_resultaten.ipynb`.
In de MX-run komen daarnaast `mx_source_best.pt` en `mx_source_last.pt`.
Die MX-bronbestanden bevatten geen FP32-masters van de zes doellagen; ze zijn geen
hervatcheckpoint. De overige lagen en BN-staat blijven aanwezig.

Beste: eerste strikte maximum van de actieve modus op de officiele testset.
Laatste: epoch 100. Testselectie is niet onafhankelijk; beide rollen blijven apart.
De hoofdvergelijking is ook expliciet epoch 100 tegenover epoch 100.

Open de notebook uit een RUNMAP (niet de template of een checkmap), kies .venv-train,
en voer Run All uit. Voor de vergelijking open je de MX-notebook en stel je in:
`CONTROL_DIR = Path(r"<volledig bestaand pad naar de control_run>")`.
De notebook controleert initialisatie, alle train/test-volgordehashes, configuratie,
data en omgeving. Zij leest alleen opgeslagen gegevens; geen modelevaluatie/training.
Upload de twee feedbackbundels met herkenbare namen (of voeg ze samen in een nieuwe ZIP).

Voltooiing: `status=PASS`, `completed_epoch=100`, `optimizer_steps=57300`,
`full_training_completed=true`. Na 100 epochs is de WERKELIJK gemeten epochtijd
leidend, niet de extrapolatie. Een synthetische CPU-test is geen GPU-tijdmeting.
De pilots raamden ongeveer 24 minuten voor control en 40 minuten voor MX;
thermische belasting en achtergrondwerk kunnen dit veranderen.

## Noodstop en hervatten

Gepland: ieder model in een keer trainen. Checkpoints blijven als noodvoorziening.
Een Ctrl+C vraagt stoppen na de lopende epoch; een tweede onderbreekt direct.
`STOP_AFTER_EPOCH` in de actieve runmap vraagt dezelfde veilige epochgrensstop.
Alleen volledig afgeronde epochs hebben een hervatbaar checkpoint.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_single_reference100_v1\run_train.py" resume --mode mx --checkpoint "C:\...\checkpoints\epoch_00050.pt" --stop-after-epoch 100
```

Gebruik een werkelijk bestaand checkpoint van DIT pakket en DEZELFDE modus.
Pas `--mode control` toe voor control. Behoud alle eerdere sessiemappen: bij
hervatten kan de selectie naar een eerder checkpoint verwijzen. Pilot-, oudere
Any-Precision- en andere-moduscheckpoints worden via hun contract geweigerd.
Hervatten schrijft naar een nieuwe map en overschrijft de bron niet.
Geen mid-epoch-herstel of garantie van bitexact identieke CUDA-uitvoering.

## Reproduceerbaarheid

`TRAINING_CHANGE_MANIFEST.json` en `fullrun.patch` tonen de overgang van pilot naar
100 epochs. De volledige epochlus is AST-identiek; single_support, MX-codec, STE,
architectuur, optimizer- en hervatcode zijn ongewijzigd. Oude audits staan expliciet
onder `provenance/pilot5`, en beschrijven niet de limiet van dit nieuwe pakket.
Nieuwe softwaretests en beperkingen staan in `local_validation/VALIDATION.md`.
Dit is een eigen eenmodus/train-only-onderzoeksvariant, geen exacte paperreproductie.
