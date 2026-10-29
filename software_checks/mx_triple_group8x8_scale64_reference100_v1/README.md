# 8×8-groepen met vaste ×64 — volledige training van 100 epochs

De echte vijf-epochpilot is geslaagd. Dit pakket voert dezelfde **8×8 / E4M3 → E2M3 → E2M1 / vaste ×64**-route uit tot epoch 100. Activaties blijven A32; mastergewichten, gradiënten en Adam-momenten blijven FP32. De numerieke code is byte-identiek aan de geslaagde pilot.

## Pilotresultaten

| Groepsindeling, steeds ×64 | E4M3 op epoch 5 | E2M3 op epoch 5 | E2M1 op epoch 5 |
|---|---:|---:|---:|
| 1×32 | 93,009% | 91,299% | 90,892% |
| 1×64 | 92,617% | 90,973% | 91,295% |
| **8×8** | **93,139%** | **91,676%** | **91,503%** |

Voor alle drie runs is epoch 5 tevens het unieke beste E4M3-checkpoint binnen de eerste vijf epochs. Deze vroege resultaten zijn voldoende aanleiding om dezelfde configuratie volledig te trainen; ze bewijzen geen betere eindaccuracy. De brongegevens en analyse staan onder `pilot_analysis`.

## Plaatsen en uitvoeren

Pak de map `mx_triple_group8x8_scale64_reference100_v1` uit onder `C:\Users\renzo\master-thesis-mx\software_checks`. Gebruik de bestaande trainingsomgeving:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group8x8_scale64_reference100_v1\run_train.py" check
```

Na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group8x8_scale64_reference100_v1\run_train.py" run
```

De volledige run start **vanaf dezelfde oorspronkelijke initialisatie**, met seed 20261009. De pilot wordt behouden als afzonderlijk experiment. Dit pakket laadt zijn vijf-epochcheckpoint niet als warmstart. De oorspronkelijke CUDA-backend garandeert geen bitexacte herhaling van de eerste vijf epochs.

De eindepoch staat vast op **100**. Adam start met LR 0,001; de oorspronkelijke dalingen na 50/75/90 blijven gelijk. De volledige run bevat 57.300 Adamupdates en 171.900 trainingspasses.

De pilot duurde in totaal 4 min 47 s. Zijn mediane normale epoch bedroeg 54,42 s. De projectie voor een verse volledige run is **ongeveer 1 uur 31 minuten** bij vergelijkbare laptopcondities; de werkelijke duur wordt tijdens de nieuwe run gemeten.

De bestaande `train_32x32.mat` en `test_32x32.mat` worden hergebruikt, met de oorspronkelijke resize naar 40×40. Alleen de train- en testsplit worden gebruikt. Geen dependency-upgrades. Voor een andere datamap: `--data-root "PAD_NAAR_SVHN"`. Downloads zijn uitsluitend toegestaan als je expliciet `--download` toevoegt.

## Resultaten en notebook

Iedere opdracht maakt een nieuwe map onder `mx_runs\mx_triple_group8x8_scale64_reference100`. Bestaande code, pilot en eerdere resultaten blijven behouden.

Open na de run **`MX_triple_group8x8_scale64_reference100_resultaten.ipynb` in de nieuwe resultatenmap** en kies Run All. De runner vult het pad in. De notebook wordt onuitgevoerd meegeleverd; standaard `SAVE_FIGURES=False`.

De notebook vergelijkt 8×8 met de bestaande volledige 1×32 ×64- en 1×64 ×64-trainingen en toont:

- Het volledige accuracyverloop voor alle drie precisies.
- Een grafiek voor **epoch 100**.
- Een grafiek voor het **gezamenlijke checkpoint gekozen op de hoogste E4M3-accuracy**. De lagere precisies komen uit datzelfde checkpoint per training; er worden geen losse maxima per precisie gecombineerd.
- Schaalbytes, padding en bitgepakte code- en schaalopslag per laag en voor het gehele MX-gedeelte.
- Gemeten looptijd, geregistreerd CUDA-geheugen, quantisatiediagnostiek, voorspelde klassen en controles van de opgeslagen bron-/doelexports.

Bij een onderbroken run worden het laatste voltooide checkpoint en de beste score binnen het beschikbare bereik getoond. Referenties worden voor zulke vergelijkingen op hetzelfde epochenbereik begrensd; ontbrekende epoch-100-resultaten worden niet ingevuld.

Deel daarna **`feedback_bundle.zip` uit de resultatenmap van `run`**. De grote checkpoints blijven lokaal.

## Ongewijzigde numerieke route en kosten

Elke groep bestaat uit 8 outputkanalen × 8 posities in de afgevlakte Cin×Kh×Kw-dimensie, dus 64 gewichten per E8M0-bronschaal. De gewichtsmatrix wordt op beide assen aangevuld tot veelvouden van acht. E4M3 wordt eenmaal per laag en minibatch uit G(W) gecodeerd.

Exponenttruncatie behoudt de hoogste twee exponentbits. De lagere schaalcode is broncode +6, dus schaal ×64. E2M1 ontstaat met `E2M3_code >> 2`, met behoud van de E2M3-schaal. De bestaande identity-STE en recursieve driepassentraining met één Adamupdate blijven gelijk; geen geleerde schaal of directe lagere quantizer.

| Indeling | Schaalbytes | Paddingposities | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---|---:|---:|---:|---:|---:|
| 1×32 | 4.080 | 1.088 | 134.640 B | 102.000 B | 69.360 B |
| 1×64 | 2.160 | 8.768 | 140.400 B | 105.840 B | 71.280 B |
| **8×8** | **2.024** | **64** | **131.560 B** | **99.176 B** | **66.792 B** |

Dit betreft de 129.472 gewichten in zes MX-lagen, inclusief schalen en padding. De kolommen zijn alternatieve exports; de E4M3-bron volstaat om de lagere precisies af te leiden. FP32-uitzonderingen/BN, optimizer, containerheaders en tijdelijke tensors vallen buiten de tabel. De exports behouden alle vijf BN-banken; de eerdere transferproef telde alleen operationele banken.

De training gebruikt FP32-kernels met MX-gewichtemulatie. Looptijd en CUDA-geheugen betekenen geen native-MX-hardware-, energie- of transfermeting.

## Stoppen en hervatten

Eén keer Ctrl+C of een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap stopt na de lopende volledige epoch. Een tweede Ctrl+C stopt direct.

Hervat uitsluitend een onvoltooide run uit ditzelfde volledige 8×8-pakket:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group8x8_scale64_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Model, Adam, scheduler, RNG en voorgeschiedenis worden hersteld. Geen bitexacte CUDA-hervatgarantie. Het contract weigert pilotcheckpoints, oude rijgroepen, afwijkende schalen en reeds voltooide 100-epochcheckpoints.

## Validatie en interpretatie

`local_validation/VALIDATION.md` beschrijft de lokale CPU-controles met kleine synthetische fixtures, waaronder alle LR-grenzen en stoppen/hervatten. Er is tijdens de voorbereiding geen nieuwe echte SVHN-training gestart. `check` controleert vervolgens de bestaande Windows/CUDA-omgeving.

Lage accuracy is een geldig meetresultaat en geen automatische technische FAIL. De proef gebruikt één seed; de officiële testset wordt ook voor checkpointselectie gebruikt. De oorspronkelijke pilotimplementatie en validatie blijven bewaard onder `provenance/group8x8_pilot5_*` en `provenance/inherited_group8x8_pilot5_validation`.
