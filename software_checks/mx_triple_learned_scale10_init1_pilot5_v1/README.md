# SVHN CNN-8 — tien schaalkeuzes, initialisatie ×1, vijf epochs

Deze proef onderzoekt of de eerdere start bij ×64 de geleerde schaalverdeling heeft beïnvloed. Alle 4.080 schaalparameters beginnen nu op `theta=0`, dus factor ×1. Daarna blijven **alle tien factoren ×1, ×2, ×4, ×8, ×16, ×32, ×64, ×128, ×256 en ×512 leerbaar**. Het is een nieuwe training vanaf dezelfde oorspronkelijke FP32-gewichten en BN-toestand, met dezelfde seed en samplevolgorde als de historische vijf-epochproef die bij ×64 begon.

×1 betekent bij de start **S_E2 = S_E4**. De absolute E4-bronschaal blijft per groep uit de actuele gewichten bepaald; zij wordt niet op de numerieke waarde één gezet. Dit is nog geen proef met onafhankelijke schalen per exponentformaat.

## Starten in PowerShell

Plaats de map `mx_triple_learned_scale10_init1_pilot5_v1` onder `software_checks` van het bestaande project. De nieuwe map staat naast de eerdere pakketten.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_init1_pilot5_v1\run_train.py" check

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_init1_pilot5_v1\run_train.py" run
```

Voer `run` uit nadat `check` PASS geeft. De check voert codec-, gradient-, optimizer-, initialisatie- en hervatcontroles uit op synthetische data. Zij traint niet op SVHN. `run` gebruikt de bestaande SVHN `train`- en `test`-bestanden en stopt uiterlijk na epoch 5. Elke opdracht maakt een nieuwe resultatenmap onder `mx_runs/mx_triple_learned_scale10_init1_pilot5`.

Een andere bestaande datasetmap kan met `--data-root` worden opgegeven. Alleen een expliciete `--download` staat toe dat ontbrekende data worden opgehaald. Bestaande ruwe data en oude resultaten worden niet overschreven.

## Wat blijft gelijk?

- CNN-8/SVHN, batch 128, seed 20261009, train-only zonder `extra`.
- E4M3 → E2M3 → E2M1; hoogste exponentbits behouden en twee lage mantissabits afkappen.
- Groepen van 32 gewichten, E8M0-schalen, A32-activaties en FP32-mastergewichten.
- Per minibatch dezelfde drie passes, dezelfde recursieve distillatielosses en één Adam-update.
- De bestaande schaalgradient, projectie naar [0,9], Adam-learningrate 0,001 voor gewichten én schaalparameters.
- De oorspronkelijke LR-mijlpalen 50/75/90. Ze worden niet naar vijf epochs samengedrukt.

De enige bedoelde numerieke wijziging is `THETA_INITIAL = 6.0` → `0.0`. De runner krijgt daarnaast een eigen experimentidentiteit, een harde grens van vijf epochs, vergelijkingsrapporten en aangepaste notebook. Het pakket kan geen training tot epoch 50 of 100 starten. De schalen worden tijdens deze vijf epochs niet bevroren.

## Resultaten en notebook

In iedere resultatenmap staat `MX_triple_learned_scale10_init1_pilot5_resultaten.ipynb`, samen met `notebook_support.py`. Open die notebook vanuit de resultatenmap en voer haar zelf uit. Zij vergelijkt uitsluitend de beschikbare epochs 1–5 met de originele init×64-pilot; eindresultaten van epoch 100 worden hier niet als referentie gebruikt.

De rapportage bewaart onder meer:

- Accuracy en loss van de drie precisies per epoch; epoch 5 en het binnen deze pilot op E4M3 gekozen beste checkpoint.
- De verdeling over alle tien factoren, aantallen en aandelen, veranderingen per epoch en per laag.
- De continue theta-waarden en de beweging vanaf de ondergrens. Die zijn relevant als nog weinig groepen de eerste drempel `theta=0.5` passeren.
- `initialization_comparison.csv` en `initialization_reference_report.json`: vergelijking met init×64 en controle op dezelfde beginweights/BN en dezelfde train-/testvolgorde.
- Checkpoints, RNG-/optimizer-/schedulerstatus, code-identiteit, datasetcontroles en timing.

De vergelijking gebruikt de ongewijzigde onderzoeksdata uit `init64_reference`. De exacte oorspronkelijke run is `20261104_185824_003629_joint_run`; de bronchecksums staan in `REFERENCE.json`. De bewaarde bestanden onder `provenance` documenteren de oorspronkelijke runtime; zij worden niet uitgevoerd.

De feedbackbundel bevat de notebook, rapporten, tabellen en relevante code. SVHN en de grote checkpoints blijven lokaal. Deel na afloop de nieuwe `feedback_bundle.zip`.

## Onderbreken en hervatten binnen deze pilot

Voor een geplande stop na twee epochs:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_init1_pilot5_v1\run_train.py" run --stop-after-epoch 2
```

De status is dan `STOPPED_AT_EPOCH_BOUNDARY`. Hervat met het in die run opgeslagen checkpoint:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_init1_pilot5_v1\run_train.py" resume --checkpoint "<volledig pad naar epoch_00002.pt>" --stop-after-epoch 5
```

Gebruik de werkelijke checkpointnaam uit de resultatenmap. Alleen checkpoints van dit pakket en dit experiment worden aanvaard. `run` start altijd opnieuw; `resume` bewaart de bestaande geschiedenis en schrijft nieuwe resultaten in een nieuwe map. Een oude init×64-run kan niet als init×1-run worden hervat.

## Interpretatie

Deze korte proef meet de eerste leerbeweging vanaf ×1. Een lage accuracy of het uitblijven van een harde factorswitch is een geldig onderzoeksresultaat, geen reden om tijdens de proef de learningrate, beginfactor of schaalgradient aan te passen. De grenscontrole test expliciet dat een theta op nul met een geschikte gradient omhoog kan bewegen; een update onder nul wordt door de bestaande projectie op nul gezet.

Vijf epochs tonen niet of twee initialisaties uiteindelijk naar hetzelfde resultaat convergeren. De historische vergelijking gebruikt één seed. De oorspronkelijke CUDA-backend is behouden; een gelijke seed en samplevolgorde betekenen daarom niet dat een GPU-herhaling bitexact is. Het beste checkpoint wordt zoals eerder geselecteerd op de officiële testset; dat is geen onafhankelijke validatieschatting.

## Uitgevoerde softwarecontroles

De CPU-zelftests en een begrensde synthetische proef van de echte trainingslus zijn geslaagd, inclusief exacte hervatting en controle van de opgeslagen MX-exports. De notebook is gecontroleerd op volledige, gedeeltelijke en hervatte uitvoer. Er is hier geen SVHN-training of CUDA-training uitgevoerd. De rapporten, reproduceerbare CPU-testharnassen en de afbakening van deze controles staan in `local_validation/README.md`. Voer daarom eerst de meegeleverde `check` op je eigen GPU uit.
