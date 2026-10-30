# MobileNetV2 / CIFAR-10 — vijf volledige epochs

Deze pilot onderzoekt of de besproken MobileNetV2-reconstructie met alle vijf Any-Precision-modi begint te leren en gedurende enkele volledige epochs hetzelfde tempo aanhoudt. Elke epoch gebruikt **alle 50.000 trainingsbeelden** en evalueert **alle 10.000 testbeelden**. De pilot stopt uiterlijk na epoch 5. Een latere referentietraining van 400 epochs begint opnieuw vanaf nul.

De eerdere tijdproef geeft een raming van ongeveer **tien minuten voor vijf epochs**, plus opstart en gegevenscontrole. De nieuwe meting in deze pilot bepaalt de werkelijke duur.

## Uitpakken

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. Daarna moet dit bestand bestaan:

```text
software_checks\anyprecision_cifar10_mobilenetv2_k3_pilot5_v1\run_train.py
```

Gebruik de bestaande `.venv-train`. Het pakket installeert geen nieuwe Python-pakketten en verandert de eerdere code of resultaten niet.

## 1. Eerst controleren

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_cifar10_mobilenetv2_k3_pilot5_v1\run_train.py" check
```

De controle omvat CPU-zelftests, een begrensde synthetische GPU-proef en een controle van checkpointopslag en hervatten. De GPU-proef doet één opwarmupdate en twee meetupdates met batch 128, en controleert ook de kleinere eindbatches van 80 trainingsbeelden en 16 testbeelden. Er is hiervoor geen CIFAR-10 nodig. De GPU-worker heeft een harde limiet van **240 seconden**. De voorafgaande pakketcontrole en CPU-zelftests hebben afzonderlijke limieten van **45 en 180 seconden**.

## 2. Na PASS: vijf epochs vanaf nul

De volledige dataset uit de geslaagde tijdproef staat in `data\cifar10_mobilenet_complete`. Gebruik die map expliciet:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_cifar10_mobilenetv2_k3_pilot5_v1\run_train.py" run --data-root ".\data\cifar10_mobilenet_complete"
```

Dit gebruikt nieuwe modelgewichten en een nieuwe Adamoptimizer. De gewichten uit de tijdproef worden niet overgenomen. Elke trainingsbatch wordt in de volgorde **W32/A32 → W8/A8 → W4/A4 → W2/A2 → W1/A1** uitgevoerd, met vijf backwards en één gezamenlijke Adam-update. De batchgrootte blijft 128; de laatste kleinere batches worden ook verwerkt.

Na elke voltooide epoch worden de metrics en een checkpoint opgeslagen. Het rapport vergelijkt de laatste voltooide epoch met de epoch die de hoogste testaccuracy bij W32/A32 had. De accuracies van de andere modi komen steeds uit **datzelfde gekozen checkpoint**. Bij een exacte W32-tie wordt de vroegste epoch behouden.

Het werkproces heeft een harde limiet van **1.800 seconden (30 minuten)**. Er verschijnen batchmeldingen en elke vijftien seconden een melding van de procesbewaking. Bij Ctrl+C of een timeout stopt alleen de procesboom van deze proef.

### Alleen als de dataset nog ontbreekt

De runner zoekt standaard eerst in `data\cifar10_mobilenet_complete`. Een expliciete andere map kan via `--data-root`. Alleen met `--download` mag de officiële dataset worden opgehaald. Een bestaande onvolledige of corrupte dataset wordt niet overschreven; gebruik dan een nieuwe aparte datamap.

## Pauzeren en hervatten binnen deze pilot

Voor een geplande pauze kun je bij het starten bijvoorbeeld `--stop-after-epoch 2` toevoegen. De runner bewaart dan twee volledige epochs. Hervat daarna met de checkpoint van de laatste voltooide epoch:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_cifar10_mobilenetv2_k3_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_DE_CHECKPOINT.pt" --data-root ".\data\cifar10_mobilenet_complete"
```

Neem het exacte checkpointpad over uit het rapport of de resultatenmap. Een hervatting maakt een **nieuwe resultatenmap**, behoudt de eerdere geschiedenis en gaat door tot maximaal epoch 5. Ze herstelt model, Adam, BatchNorm, globale epochteller, learning-rateschema en de opgeslagen toevalsgeneratoren. Er is geen bitexacte CUDA-garantie bij het behouden `cudnn.benchmark=True`-beleid.

Bij een onverwachte onderbreking hervat je vanaf de laatste volledig opgeslagen epoch; werk uit een onvoltooide epoch wordt opnieuw uitgevoerd. De diagnostische checkpoint van de eerdere tijdproef wordt niet geaccepteerd. Deze pilot biedt geen opdracht om door te trainen tot epoch 400.

## Resultaten, notebook en feedback

Elke opdracht schrijft naar een nieuwe map onder:

```text
mx_runs\anyprecision_cifar10_mobilenetv2_k3_pilot5\
```

Deel de **`feedback_bundle.zip`** uit die map. Checkpoints en CIFAR-10 blijven lokaal. Reken op ongeveer 35 MB per epochcheckpoint, dus ongeveer 175 MB voor vijf checkpoints; de werkelijke grootte staat in het rapport.

Het meegeleverde **`MobileNetV2_CIFAR10_pilot5.ipynb`** is onuitgevoerd. Open het met de gebruikelijke Python-kernel met pandas en matplotlib. Het toont:

- het volledige accuracyverloop van de vijf modi over de pilot;
- training en test per modus, en de losses met hun verschillende betekenis;
- de laatste voltooide epoch en het checkpoint gekozen op W32/A32;
- looptijd en GPU-geheugen per epoch, plus een raming voor langere trainingen.

Het notebook kiest standaard het nieuwste rapport, ook wanneer dat FAIL of een onderbroken proef is. Vul `RUN_DIR` in om één specifieke run of een uitgepakte feedbackbundel te bekijken. Er worden alleen werkelijk gerapporteerde epochs getoond.

Dit blijft een **gedocumenteerde MobileNetV2-reconstructie**. Vijf epochs zijn bruikbaar voor de eerste leerprogressie, maar leveren nog geen vergelijking met de eindaccuracies uit de paper. Zie `PROTOCOL.md` voor de modelkeuzes en meetdefinities.
