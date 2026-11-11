# Onafhankelijke E2-schaal — volledige training van 100 epochs

Dit pakket vervolgt de geslaagde vijfepochpilot. De numerieke quantisatie, absolute schaalparameterisatie en gradients blijven byte voor byte gelijk aan de daadwerkelijk uitgevoerde pilotcode. De volledige training begint **opnieuw vanaf nul**. Het pilotcheckpoint wordt niet als startpunt gebruikt.

Per B32-groep heeft E4M3 de bestaande dynamische bronschaal. E2M3 en E2M1 delen één eigen, absoluut geleerde E8M0-schaal. Die begint eenmaal op 64 × de aanvankelijke E4-schaal en leert daarna zelfstandig. Lagere elementcodes blijven uit de E4-broncodes getrunceerd. A32, FP32-masters, drie passes en één Adam-update per minibatch blijven behouden.

## Uitvoeren

Plaats de uitgepakte map in:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_independent_e2_scale_reference100_v1`

Voer eerst uit met de bestaande trainingsomgeving:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_reference100_v1\run_train.py" check
```

Na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_reference100_v1\run_train.py" run
```

`check` gebruikt synthetische gegevens en voert geen SVHN-training uit. `run` traint 100 volledige epochs met train-only SVHN. De schaalparameters blijven tot en met epoch 100 leerbaar; zij worden niet op epoch 50 bevroren. Beide Adam-parametergroepen volgen de oorspronkelijke LR-overgangen op 50/75/90. De start-LR is 0,001.

De bestaande dataset wordt gecontroleerd en hergebruikt. Er worden geen pakketten geïnstalleerd of geüpgraded. Downloaden gebeurt alleen met expliciet `--download`. Eerdere resultaten en ruwe data blijven behouden. Elke opdracht krijgt een nieuwe map onder:

`mx_runs\mx_triple_independent_e2_scale_reference100`

Op basis van de werkelijke pilot is circa **1 uur 52 minuten** geraamd voor een nieuwe volledige training. Dit is een extrapolatie, geen gemeten 100-epochtijd of garantie. Laptopbelasting, temperatuur en overige overhead kunnen de tijd veranderen.

## Checkpoints en hervatten

Alle voltooide epochs blijven opgeslagen, inclusief epoch 50. Een afzonderlijke epoch-50-verwijzing maakt een eventueel later freeze-experiment mogelijk; deze run past geen freeze toe.

Eén keer Ctrl+C vraagt stoppen na de lopende epoch. Een tweede Ctrl+C stopt direct; alleen de laatste volledig opgeslagen epoch is hervatbaar. Hervatten herstelt gewichten, absolute schaalparameters, Adam-momenten, scheduler en RNG:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00050.pt"
```

Alleen een checkpoint uit exact dit 100-epochpakket en dezelfde vastgelegde omgeving is hervatbaar. De oorspronkelijke cuDNN-instellingen blijven behouden; bitexacte CUDA-hervatting is niet gegarandeerd. Een gedeeltelijke run wordt niet als voltooide 100-epochtraining aangeduid.

## Notebook en feedback

`MX_triple_independent_e2_scale_reference100_resultaten.ipynb` en `notebook_support.py` staan in het pakket en worden naar de resultatenmap gekopieerd. Open de notebook vanuit die map en voer alle cellen uit. In de pakketkopie kan `RUN_DIR` expliciet worden ingevuld. Er wordt geen uitgevoerde notebook meegeleverd.

De notebook toont de volledige accuracyontwikkeling, het gemeenschappelijke epoch-100-eindpunt en de drie modi op het checkpoint dat op beste E4-accuracy is gekozen. Vergelijkingen gebruiken de echte volledige vaste ×64- en tienkeuzetrainingen, telkens binnen hetzelfde beschikbare epochvenster. Een gedeeltelijke run krijgt zijn werkelijke laatste epoch, zonder verzonnen epoch-100-score.

De grafieken van absolute E2-exponenten en geobserveerde E2/E4-verhoudingen blijven afzonderlijk. Ook bronveranderingen, continue schaalupdates, harde exponentwissels, gradientdiagnostiek, opslag en looptijd worden getoond. Accuracyassen zoomen in en omvatten alle weergegeven waarden. De notebook leest alleen CSV/JSON en start geen training.

Deel na afloop `feedback_bundle.zip` uit de nieuwe resultatenmap. Checkpoints en dataset blijven lokaal. `PILOT_REVIEW.md` bevat de beoordeling van de werkelijke vijfepochproef; `PROTOCOL.md` beschrijft de volledige proef en `local_validation/` de softwarecontrole.
