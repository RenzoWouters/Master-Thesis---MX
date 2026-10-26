# Vaste offsets na epoch 50 — voortzetten van epoch 55 tot 100

Dit pakket zet de geslaagde bevroren vijf-epochproef voort vanaf het bestaande checkpoint **na epoch 55**. Het voert uitsluitend **epochs 56–100** uit. De offsetkeuzes blijven die van epoch 50; de gewichten en actieve BatchNorm-banken blijven verder leren. De bestaande dynamische 100-epochtraining is de historische controle.

## 1. Uitpakken

Pak de zip uit in:

```text
C:\Users\renzo\master-thesis-mx\software_checks
```

Het nieuwe startbestand staat dan hier:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_freeze_offsets_e50_reference100_v1\run_train.py
```

Behoud de hele uitgepakte map. De vorige proefcode en de oorspronkelijke trainingsruntime zijn onveranderd meegeleverd onder `pilot_runtime`; het nieuwe startbestand bepaalt het nieuwe traject en de rapportage.

## 2. Eerst controleren

Gebruik dezelfde `.venv-train` als bij de vorige run:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_freeze_offsets_e50_reference100_v1\run_train.py" check
```

De check voert codec-/gradientcontroles uit en verifieert het echte bevroren checkpoint na epoch 55. Ook controleert hij de historische gegevens en de eerdere checkpoints die nodig kunnen zijn voor de beste-checkpointselectie. Hiervoor wordt geen SVHN-training uitgevoerd.

De volgende bestaande locaties worden automatisch gebruikt:

```text
mx_runs\mx_triple_freeze_offsets_e50_pilot5\20261025_174221_578057_freeze_run\checkpoints\epoch_00055.pt

mx_runs\mx_triple_learned_scale32_64_reference100\20261025_150724_515132_joint_run
```

Bewaar beide volledige resultatenmappen met checkpoints en `.sha256`-bestanden. De tweede map levert ook de oorspronkelijke `history.json`, die tegen de vastgelegde hash wordt gecontroleerd. Daaruit komen de historische samplevolgordes voor epochs 56–100 en de gedeelde geschiedenis van vóór de freeze. Deze bestanden worden gelezen en in de nieuwe resultaten gearchiveerd; ze worden niet gewijzigd.

Als dezelfde bestanden verplaatst zijn, kun je de nieuwe locaties opgeven:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_freeze_offsets_e50_reference100_v1\run_train.py" check --parent-checkpoint "PAD\NAAR\epoch_00055.pt" --dynamic-run "PAD\NAAR\OORSPRONKELIJKE_100_EPOCH_RESULTATENMAP"
```

Gebruik bij `run` en `resume` dan dezelfde twee opties. De inhoud moet nog overeenkomen met de vastgelegde werkelijke runs. Bij ontbrekende of afwijkende invoer stopt de code met een concreet foutbericht. Er worden geen afhankelijkheden bijgewerkt en geen datasets gedownload.

## 3. Voortzetten tot epoch 100

Na `EINDSTATUS: PASS` van de check:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_freeze_offsets_e50_reference100_v1\run_train.py" run
```

Dit herstelt model, BatchNorm, beide Adamgroepen, momenten, scheduler en RNG uit checkpoint 55. De bevroren offsetgroep heeft daar al LR nul; de voortzetting verandert de optimizerstaat niet.

| Nieuwe epochs | Gewichts-LR | Theta-LR |
|---|---:|---:|
| 56–75 | 0,0001 | 0 |
| 76–90 | 0,00001 | 0 |
| 91–100 | 0,000001 | 0 |

De scheduler wordt telkens na training en evaluatie van een epoch bijgewerkt, zoals voordien. De oorspronkelijke LR-overgangen na 75 en 90 blijven dus behouden.

Er volgen 45 nieuwe epochs, 25.785 nieuwe Adamupdates en 77.355 nieuwe trainingspasses. Inclusief de overgenomen bevroren epochs 51–55 omvat de fase na freeze 50 epochs en 28.650 updates. De absolute teller over het volledige traject eindigt op 57.300 updates.

Reken op ongeveer **45–60 minuten** onder vergelijkbare laptopomstandigheden. De nieuwste vijf epochs geven een tempo van circa één minuut per epoch; de eerdere volledige run bevatte ook een tragere periode. De werkelijke nieuwe duur wordt afzonderlijk opgeslagen van de al gemeten vijf piloteepochs.

## 4. Wat blijft vast en wat verandert

- Alle 4.080 theta-waarden blijven exact gelijk aan de waarden na epoch 50: 1.654 blokken op ×32 en 2.426 op ×64.
- Een controle na iedere optimizerupdate verifieert zowel de FP32-theta als de harde offsetkeuze.
- De FP32-mastergewichten en actieve BatchNorm-banken leren verder.
- De E4M3-bronschaal en codes worden uit de actuele gewichten bepaald en kunnen veranderen.
- E2M3 behoudt dezelfde exponenttruncatie en vaste extra blokoffset. E2M1 volgt met dezelfde mantissatruncatie en behoudt exact dezelfde schaal als E2M3.
- Activaties blijven A32; masters, gradients en Adam-momenten blijven FP32.
- De drie oorspronkelijke recursieve distillatiepasses en één Adamupdate per minibatch blijven gelijk.
- Theta-gradients en theta-Adam-momenten worden nog berekend bij theta-LR nul. Deze proef meet het accuracyeffect; zij claimt geen geoptimaliseerde kostenbesparing.
- De samplevolgordes voor train en test worden iedere nieuwe epoch met de opgeslagen dynamische training vergeleken. Bitexacte CUDA-reproduceerbaarheid wordt niet geclaimd.

## 5. Resultaten en notebook

Iedere opdracht maakt een nieuwe resultatenmap onder:

```text
mx_runs\mx_triple_freeze_offsets_e50_reference100
```

Na de training staat daarin de lege, uitvoerbare notebook met het juiste runpad:

```text
MX_triple_freeze_offsets_e50_reference100_resultaten.ipynb
```

De notebook toont onder meer:

1. Accuracycurves voor E4M3, E2M3 en E2M1, met de reeds uitgevoerde epochs 51–55 en de nieuwe epochs 56–100.
2. Een accuracy-staafdiagram voor **epoch 100**, de primaire vergelijking met de bestaande dynamische training.
3. Een accuracy-staafdiagram op het **E4M3-beste checkpoint binnen epochs 51–100**, met alle drie precisies uit hetzelfde geselecteerde checkpoint.
4. De gebruikelijke **globale E4M3-selectie binnen epochs 1–100**, opnieuw met alle drie precisies op hetzelfde checkpoint. Als het beste checkpoint vóór de freeze ligt, wordt dat nadrukkelijk vermeld. Dat gedeelde oude model wordt niet als een resultaat na freeze voorgesteld.
5. Volledige accuracycurves over 1–100, verschillen met de dynamische referentie, offsetverdeling en nul wisselingen in de bevroren fase, verliescurves en gemeten tijden.

De eerste 50 epochs van de twee trajecten zijn dezelfde. Daarna gebruikt de bevroren curve uitsluitend de werkelijk uitgevoerde bevroren epochs. Bij een gedeeltelijke run gebruikt de notebook het gezamenlijk beschikbare vergelijkingsvenster en toont hij nog geen eindresultaat voor epoch 100.

De resultaten bevatten `history.json`, CSV-metrics, `freeze_summary.json`, `lineage.json`, de historische referenties en checkpoints per nieuwe epoch. De drie selecties hebben afzonderlijke checkpointkopieën en compacte MX-exports met achtervoegsels `last`, `best_postfreeze` en `best_global`. Een geselecteerd checkpoint kan uit de gedeelde eerste 50 epochs, de bevroren pilot51–55 of de nieuwe voortzetting56–100 komen; ieder wordt tegen het juiste oorspronkelijke contract gecontroleerd.

**Deel na afloop `feedback_bundle.zip` uit de nieuwe resultatenmap.** De bundel bevat de analysegegevens en herkomst, zonder grote binaire trainingscheckpoints.

## 6. Stoppen en hervatten

Eén keer **Ctrl+C** vraagt een gecontroleerde stop na de huidige volledige epoch. De runner slaat die epoch op en maakt een gedeeltelijke rapportage. Een tweede Ctrl+C stopt meteen; alleen volledig geschreven checkpoints zijn dan bruikbaar.

Je kunt ook vooraf een stopgrens kiezen, bijvoorbeeld:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_freeze_offsets_e50_reference100_v1\run_train.py" run --stop-after-epoch 75
```

Hervatten gebeurt met een checkpoint uit **deze nieuwe voortzetting**:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_freeze_offsets_e50_reference100_v1\run_train.py" resume --checkpoint ".\mx_runs\mx_triple_freeze_offsets_e50_reference100\JOUW_RESULTATENMAP\checkpoints\epoch_00075.pt"
```

Bewaar de volledige eerdere resultatenmap, inclusief `epoch_summaries`, checksums en de oorspronkelijke twee invoermappen. Hervatten maakt een nieuwe resultatenmap en neemt de eerder voltooide geschiedenis mee. De bevroren theta, Adam-momenten, scheduler en RNG worden hersteld; er wordt geen LR-schema herstart.

Een gecontroleerde stop vóór epoch 100 krijgt `STOPPED_AT_EPOCH_BOUNDARY`. Alleen een volledig afgeronde voortzetting tot 100 krijgt `PASS` als trainingseindstatus.

## 7. Technische validatie

De verslagen in `local_validation` betreffen kleine synthetische CPU-fixtures. Ze controleren de daadwerkelijke trainingslus, beide LR-overgangen, hervatten, de drie herkomsten van geselecteerde checkpoints, exacte compacte exports en de notebook. Ze zijn geen echte SVHN-accuracy of CUDA-tijdmeting. De echte laptopcontrole is bovenstaande `check`; de echte voortzetting begint bij `run`.
