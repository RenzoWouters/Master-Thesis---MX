# Vaste schaal ×32 — gezamenlijke referentietraining van 100 epochs

De vijf-epochpilot is technisch geslaagd. Dit pakket voert dezelfde E4M3 → E2M3 → E2M1-route uit gedurende honderd epochs. De twee lagere modi gebruiken de E4M3-bronschaal ×32, dus exponentoffset +5. Activaties blijven A32; de updatemaster, gradients en Adam-momenten blijven FP32. Er zijn geen geleerde schalen.

De volledige proef begint opnieuw vanaf dezelfde oorspronkelijke initialisatie en seed, zoals de eerdere ×64-referentie. De pilot blijft een afzonderlijk bewaard experiment. Bekijk `pilot_analysis/RESULTATEN.md` voor de gecontroleerde pilotresultaten.

## Installeren en starten

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`, zodat het script hier staat:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_scale32_reference100_v1\run_train.py
```

Gebruik de bestaande `.venv-train`; het pakket installeert of vernieuwt geen dependencies.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale32_reference100_v1\run_train.py" check
```

Start na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale32_reference100_v1\run_train.py" run
```

De bestaande SVHN train- en testbestanden onder `data\svhn` worden hergebruikt. Een afwijkende locatie kan met `--data-root "PAD_NAAR_SVHN"`; `--download` is alleen nodig als de dataset ontbreekt. SVHN-extra wordt niet gebruikt.

Elke opdracht krijgt een nieuwe map onder `mx_runs\mx_triple_scale32_reference100`. De geschatte duur op basis van de echte ×32-pilot is ongeveer **1 uur 45 minuten** voor honderd epochs, afhankelijk van laptopomstandigheden. Opstart, dataload en eindexport komen erbij. Dit is een projectie, geen reeds gemeten volledige ×32-looptijd.

## Stoppen en hervatten

Eén `Ctrl+C` vraagt een stop na de huidige volledige epoch. Een tweede `Ctrl+C` stopt direct; alleen het laatste volledig geschreven checkpoint is hervatbaar. Ook een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap vraagt een stop na een volledige epoch.

Hervat met een checkpoint uit **deze honderd-epochrun**:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale32_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Optimizer, scheduler, RNG en volledige eerdere epochgeschiedenis worden hersteld. Er wordt een nieuwe sessiemap gemaakt; eerdere checkpoints blijven behouden. De originele CUDA-backend biedt geen bitexacte hervatgarantie. Checkpoints van de ×32-pilot of ×64-training worden niet als warmstart geaccepteerd.

## Notebook en resultaten

Open na afloop `MX_triple_scale32_reference100_resultaten.ipynb` in de resultatenmap. Het resultatenpad is door de runner ingevuld. De notebook heeft lege outputs en start geen training.

De hoofdfiguren tonen alle drie precisies op **epoch 100** en op het **gezamenlijke checkpoint gekozen door de hoogste E4M3-testaccuracy**. E2M3 en E2M1 worden steeds uit datzelfde gekozen checkpoint gerapporteerd. Daarnaast zijn er leer- en losscurves, voorspellingstellingen, timing, GPU-geheugen en vergelijkingsfiguren met de volledige bestaande ×64-referentie. Die referentie is al meegeleverd en wordt naast de notebook gekopieerd.

Een onderbroken run blijft duidelijk onvolledig; epoch 100 wordt dan niet als gemeten weergegeven. Bij tussentijdse vergelijkingen worden de histories beperkt tot de gedeelde afgewerkte epochrange.

Alle epochcheckpoints, best/last-checkpoints en compacte MX-exports blijven lokaal bewaard. Deel na afloop de `feedback_bundle.zip` uit de resultatenmap van de training. Die bundel bevat de rapporten en notebook, zonder grote checkpoints of datasetbestanden.

## Vergelijkbaarheid

De numerieke kern is byte-identiek aan de geslaagde ×32-pilot. Het eindpunt wordt 100 en het bestaande LR-schema daalt na epochs 50, 75 en 90. Een volledige run telt 57.300 optimizerupdates en 171.900 trainingspasses.

De ×32-pilot lag op epoch 5 onder ×64 voor alle drie modi. Deze vervolgproef bepaalt of dat verschil na volledige training blijft bestaan. Zij veronderstelt geen accuracyverbetering. De officiële testset wordt zoals eerder gebruikt voor E4M3-checkpointselectie; de geselecteerde beste score is geen onafhankelijke testschatting.

De exacte instellingen, wijzigingen en validaties staan in `PROTOCOL.md`, `CODEWIJZIGINGEN.md` en `local_validation/VALIDATION.md`.
