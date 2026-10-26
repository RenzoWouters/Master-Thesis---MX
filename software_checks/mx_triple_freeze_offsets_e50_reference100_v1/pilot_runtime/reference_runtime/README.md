# Geleerde schaalkeuze ×32/×64 — volledige training van honderd epochs

Deze proef verlengt het gevalideerde vijf-epochprotocol tot honderd epochs. De numerieke route, schaalgradient, optimizerinstellingen en initialisatie blijven gelijk. De echte pilot is voltooid en technisch gecontroleerd; de volledige nieuwe SVHN-training wordt met dit pakket op de laptop uitgevoerd.

## Uitpakken en controleren

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. Het script staat dan onder `software_checks\mx_triple_learned_scale32_64_reference100_v1\run_train.py`.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale32_64_reference100_v1\run_train.py" check
```

`check` controleert de codecs, schaalgradient, optimizer/projectie en exports met synthetische tensors en modelpasses op jouw omgeving. Het start geen SVHN-training en installeert of wijzigt geen pakketten.

## De volledige training starten

Start na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale32_64_reference100_v1\run_train.py" run
```

Iedere opdracht krijgt een nieuwe sessiemap onder `mx_runs\mx_triple_learned_scale32_64_reference100`. De dataset onder `data\svhn` wordt hergebruikt. Voor een andere locatie kan `--data-root "PAD"` worden toegevoegd; `--download` is alleen nodig wanneer de dataset ontbreekt.

Dit is een verse honderd-epochrun vanaf seed 20261009, met alle 4.080 hulpwaarden op 6 (×64). De vijf-epochpilot blijft een afzonderlijke proef. Zijn checkpoint wordt niet als warmstart ingeladen. Bij gelijkblijvende omstandigheden voorspelt de gemeten pilot ongeveer **1 uur en 50 minuten** voor honderd epochs, plus voorbereiding en eindrapportage. Dat is een schatting, geen reeds gemeten volledige looptijd.

## Instellingen

- E4M3 als gedeelde bron; naïeve exponentbittruncatie naar E2M3; mantissatruncatie naar E2M1.
- Harde keuze per blok: theta < 5,5 kiest ×32; theta >= 5,5 kiest ×64. Beide lagere modi delen die schaal.
- De E4M3-bronschaalbepaling blijft ongewijzigd. A32; FP32-mastergewichten, gradients, hulpwaarden en Adam-momenten.
- Dezelfde log-exponent-STE: `grad_theta = ln(2) * sum_real_block(grad_q * q)`. E2M3 en E2M1 tellen op; geen directe E4M3-schaalgradient of aanvullende LSQ-normalisatie.
- Drie passes en één gezamenlijke Adam-update per minibatch. Beide parametergroepen starten op lr=0,001. Projectie van theta naar [5,6] na iedere update.
- LR-dalingen na epochs 50, 75 en 90 gelden voor beide groepen. Een voltooide run bevat 57.300 updates en 171.900 trainingspasses.

## Stoppen en hervatten

Eén `Ctrl+C` vraagt een stop na de lopende volledige epoch. Een tweede `Ctrl+C` stopt onmiddellijk. Een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultaatmap vraagt eveneens een stop op de epochgrens. Alleen volledig opgeslagen checkpoints kunnen worden hervat.

Een vooraf gekozen gedeeltelijke run is bijvoorbeeld:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale32_64_reference100_v1\run_train.py" run --stop-after-epoch 25
```

Hervat een onvoltooid checkpoint van dit nieuwe pakket tot epoch100:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale32_64_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Hervatten bewaart model, theta, beide Adam-groepen, scheduler, RNG en de volledige meetgeschiedenis in een nieuwe sessiemap. Checkpoints van andere experimenten worden geweigerd. Het oorspronkelijke CUDA-backendbeleid garandeert geen bitexacte hervatting.

## Notebook en feedback

De resultaatmap bevat `MX_triple_learned_scale32_64_reference100_resultaten.ipynb`, met het juiste pad ingevuld en lege outputs. Voer de notebook zelf uit. Ze toont de accuracies van alle drie modi op epoch100 en op het gezamenlijke checkpoint dat gekozen is door de hoogste E4M3-accuracy. Een gedeeltelijke run toont herkenbaar zijn laatste voltooide epoch.

De twee vaste honderd-epochreferenties ×32 en ×64 zijn meegeleverd. Bij een onvoltooide nieuwe run worden vergelijkingen beperkt tot de gezamenlijk beschikbare eerste epochs; de beste E4M3-epoch wordt per training binnen diezelfde periode gekozen. De echte geleerde vijf-epochpilot blijft als afzonderlijke referentie bewaard.

De notebook toont ook schaalkeuzes per laag, theta-verdelingen, omschakelingen, gradients, projecties, losses, voorspellingen, looptijd en GPU-geheugen. Lage accuracy of weinig omschakelingen blijven geldige onderzoeksresultaten en veroorzaken op zichzelf geen technische FAIL.

Deel na afloop `feedback_bundle.zip` uit de trainingsresultaatmap. Alle grote hervatbare checkpoints en compacte exports blijven op de laptop. De bronexport bevat 510 bytes aan gepakte harde schaalkeuzes; hij bevat geen FP32-theta of mastergewichten van de zes MX-lagen.

Details: `PROTOCOL.md`, `CODEWIJZIGINGEN.md`, `PILOT_BEOORDELING.md` en `local_validation/VALIDATION.md`. Eerdere pakketten en resultaten worden behouden.
