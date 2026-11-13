# Onafhankelijke E2-schaal vast na epoch 50 — vervolg tot epoch 100

Dit pakket herstelt het **oorspronkelijke checkpoint van epoch 50** uit de onafhankelijke-E2-referentierun en voert **epochs 51–100** uit: vijftig nieuwe epochs. De eerdere freeze-pilot wordt niet hervat; haar vijf epochs worden opnieuw uitgevoerd.

De absolute E2-schalen blijven vanaf epoch 51 vast. De gewichten en de drie actieve BatchNorm-banken leren verder. De E4-bronschaal wordt dynamisch uit de actuele gewichten berekend. E2M3 en E2M1 delen de bevroren E2-schaal per blok.

## Uitpakken

Pak de pakketmap uit onder:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_independent_e2_scale_freeze50_reference100_v1`

Het standaard startcheckpoint is:

`C:\Users\renzo\master-thesis-mx\mx_runs\mx_triple_independent_e2_scale_reference100\20261111_172102_961497_joint_run\checkpoints\epoch_00050.pt`

**Behoud de volledige oorspronkelijke resultatenmap met haar checkpoints.** De export van het globaal beste E4-checkpoint kan een eerdere epoch nodig hebben, bijvoorbeeld epoch 34. Bewaar ook alle checkpoints van de nieuwe run.

Gebruik de bestaande `.venv-train` en SVHN-bestanden. Het pakket bevat geen dataset of trainingscheckpoints en voert geen automatische upgrades uit.

## Eerst controleren

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_freeze50_reference100_v1\run_train.py" check
```

Dit controleert het parentcheckpoint, de softwareomgeving, het herstel en de freeze met een begrensde synthetische proef. Er start geen SVHN-training.

Is de parentmap verplaatst, voeg dan aan zowel `check` als `run` de optie `--parent-checkpoint` met het werkelijke volledige pad naar hetzelfde `epoch_00050.pt` toe. De runner controleert de vaste SHA-256.

## Training tot epoch 100 starten

Na `EINDSTATUS: PASS` van de controle:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_freeze50_reference100_v1\run_train.py" run
```

`run` herstelt epoch 50 en start daarna op epoch 51. De standaard eindepoch is 100. Iedere opdracht schrijft naar een nieuwe map onder:

`mx_runs\mx_triple_independent_e2_scale_freeze50_reference100`

De bestaande doorleercontrole is al tot epoch 100 uitgevoerd. Zij wordt niet opnieuw getraind.

Op basis van de freeze-pilot is **circa 1 uur en 40 minuten** een praktische raming voor de vijftig nieuwe epochs, exclusief opstarten en exports. Dit is een extrapolatie, geen gemeten full100-looptijd.

## Herstel en learning-rate

Het volledige model, beide Adam-groepen, de scheduler en Python/NumPy/Torch CPU/CUDA-RNG worden hersteld. Alle 4.080 absolute E2-logschalen `phi`, hun afgeronde schaalbytes en hun Adam-momenten en staptellers blijven daarna exact behouden.

| Nieuwe epochs | Gewichts-learning-rate | E2-schaal-learning-rate |
|---|---:|---:|
| 51–75 | 0,0001 | 0 |
| 76–90 | 0,00001 | 0 |
| 91–100 | 0,000001 | 0 |

De oorspronkelijke scheduler wordt voortgezet, met dalingen na de voltooide epochs 75 en 90. A32, FP32-masters, B32, drie precisies, recursive distillation, drie passes en één Adam-stap per minibatch blijven behouden.

Een vaste absolute E2-schaal betekent niet dat de lagere elementcodes of de verhouding E2/E4 vaststaan: de elementcodes volgen de actuele E4-broncodes via dezelfde truncatie.

## Onderbreken en hervatten

Eén Ctrl+C vraagt stoppen na de lopende epoch. Gebruik daarna alleen een volledig opgeslagen checkpoint uit **dit nieuwe pakket**:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_freeze50_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00075.pt"
```

De oorspronkelijke parentmap moet ook nog beschikbaar zijn. Geef zo nodig opnieuw `--parent-checkpoint` op. Het checkpoint van de oude vijfepochpilot hoort niet bij het contract van dit pakket.

Voor een bewuste tussentijdse stop kan `--stop-after-epoch 75` worden gebruikt. De rapportage vermeldt dan dat de volledige run nog niet voltooid is. Een nieuwe `resume` zonder die optie gaat verder tot epoch 100.

Het bestaande cuDNN-beleid garandeert geen bitexacte GPU-herhaling; de nieuwe uitvoering van epochs 51–55 kan licht afwijken van de eerdere pilot.

## Notebook en feedback

Open vanuit de **nieuwe resultatenmap**:

`MX_triple_independent_e2_scale_freeze50_reference100_resultaten.ipynb`

Voer alle cellen uit. De notebook leest CSV/JSON en start geen training. De meegeleverde notebook is onuitgevoerd en bevat nog geen nieuwe full100-resultaten.

De notebook toont volledige curves, detail vanaf epoch 50, vergelijkingen op dezelfde eindepoch, het gedeelde beste E4-checkpoint, de beste E4-selectie binnen de nieuwe vertakking, schaaldiagnostiek en de tijd van de nieuwe epochs. Accuracyvergelijkingen gebruiken varianten op de as, een apart paneel per precisie, gelijke cirkels en percentages bij de punten.

Deel na afloop `feedback_bundle.zip` uit de resultatenmap. Checkpoints en dataset blijven lokaal. Ook bij een fout bevat de feedbackbundel het rapport en de relevante code.

## Onderzoeksstatus

Dit is een verkennende vervolgproef na de geslaagde pilot, met één ouderrun en één vertakking. De officiële testset wordt ook voor checkpointselectie gebruikt. Een verschil in deze run bewijst geen algemene superioriteit of optimale freezegrens.

De numerieke gewichts-MX-emulatie gebruikt FP32-kernels. De verlenging verandert de numerieke adapter niet en bewijst geen hardware-, energie- of latencywinst. Zie `PROTOCOL.md` en `VALIDATION.md` voor de proefopzet en de daadwerkelijk uitgevoerde controles.
