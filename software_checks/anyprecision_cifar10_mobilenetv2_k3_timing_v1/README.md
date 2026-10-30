# MobileNetV2 / CIFAR-10 — korte geheugen- en tijdproef

Dit pakket onderzoekt of de voorgestelde MobileNetV2-variant praktisch op je laptop kan trainen. Het meet een volledige Any-Precision-update over **W/A 32, 8, 4, 2 en 1**, inclusief backward en één Adam-update. Het start geen training van vijf of 400 epochs.

## Uitpakken

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. Daarna moet dit bestand bestaan:

```text
software_checks\anyprecision_cifar10_mobilenetv2_k3_timing_v1\run_check.py
```

Gebruik dezelfde `.venv-train` als voor de eerdere proeven. Het pakket installeert of wijzigt geen Python-pakketten.

## 1. Eerst controleren

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_cifar10_mobilenetv2_k3_timing_v1\run_check.py" check
```

Deze controle voert CPU-zelftests en een begrensde GPU-proef met synthetische beelden van 32×32 uit. De GPU-proef gebruikt **batch 128 en alle vijf precisies**, met één opwarmupdate en drie meetupdates. Er wordt hiervoor geen CIFAR-10 gedownload of getraind.

Een wachtmelding verschijnt elke vijftien seconden. Die bevestigt dat de bewaking actief is; alleen de stapmeldingen bevestigen voortgang van de GPU-proef. De **GPU-worker** heeft een harde tijdslimiet van **180 seconden**. De voorafgaande pakketcontrole en CPU-zelftests hebben afzonderlijke limieten van 45 en 120 seconden; de totale opdracht kan dus langer dan drie minuten duren. Een timeout wordt als timeout gerapporteerd en niet automatisch als een geheugenfout beschouwd.

## 2. Alleen na PASS: meten met echte CIFAR-10-data

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_cifar10_mobilenetv2_k3_timing_v1\run_check.py" run
```

De runner zoekt de bijbehorende geslaagde controle. De dataproef begint vervolgens met **nieuwe willekeurige modelgewichten** en doet:

- 100 trainingsbatches van 128 beelden: 12.800 verschillende voorbeelden uit de officiële trainingsset;
- de eerste tien batches als opwarming voor de tijdstatistiek;
- evaluatie van 1.000 testbeelden, gelijk verdeeld over de tien klassen, op alle vijf precisies;
- een lokale diagnostische checkpoint en een raming voor vijf, 100 en 400 epochs.

Het pakket verwacht bestaande CIFAR-10-data, normaal in `data\cifar10\cifar-10-batches-py`. Met `--data-root` kun je een andere bestaande bovenliggende datamap aanwijzen. Voeg alleen wanneer de officiële data nog ontbreken `--download` toe:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_cifar10_mobilenetv2_k3_timing_v1\run_check.py" run --download
```

De proef met echte data heeft een tijdslimiet van **600 seconden voor het bewaakte werkproces**. Bij Ctrl+C of een timeout stopt de bewaking alleen de processen die voor deze proef zijn gestart. Een toekomstige volledige training begint opnieuw vanaf nul; deze korte diagnose is geen startcheckpoint voor een vervolgtraining.

## Resultaten en notebook

Elke opdracht maakt een nieuwe resultatenmap onder:

```text
mx_runs\anyprecision_cifar10_mobilenetv2_k3_timing\
```

Deel na afloop **`feedback_bundle.zip`** uit de nieuwe resultatenmap. De grote checkpoint en CIFAR-10 blijven lokaal. Bestaande resultaten en bronbestanden worden niet overschreven.

Open **`MobileNetV2_CIFAR10_geheugen_tijd.ipynb`** met een Python-kernel waarin pandas en matplotlib beschikbaar zijn. Het notebook is onuitgevoerd meegeleverd. Standaard zoekt het de meest recente resultaatmap van deze proef. Je kunt in de eerste instellingen ook `RUN_DIR` invullen, bijvoorbeeld om een oudere run of een uitgepakte feedbackbundel te bekijken.

Het notebook toont stapduur, GPU-geheugen, de gemeten componenten van de looptijd en de geprojecteerde trainingsduur. De vroege loss en accuracy zijn uitsluitend diagnostisch: **100 batches zijn geen volledige epoch en geen vergelijking met de eindresultaten uit de paper**.

## Wat wordt precies getest?

- MobileNetV2, breedtefactor 1; eerste convolutie stride 1 voor CIFAR-10.
- De centrale 3×3-convoluties gebruiken **acht ingangskanalen per groep**, oftewel `groups = c / 8` (`K=3`). Dit is geen MX-groepsgrootte.
- De Any-Precision-quantisatieoperatoren en de recursieve distillatieloss komen uit de meegeleverde oorspronkelijke broncode.
- FP32-mastergewichten en FP32-rekenopslag; geen AMP, geen TF32, geen MX. De lagere W/A-modi worden met fake quantization gesimuleerd.
- Een vooraf ingestelde allocatorlimiet van maximaal 85% van fysiek GPU-geheugen en 90% van aanvankelijk vrij GPU-geheugen. Dit is geen directe meting van Windows' gedeeld GPU-geheugen.
- Adam, learning rate 0,001, geen weight decay. De latere paperplanning is 400 epochs met learning-rateverlagingen op 150, 250 en 350; die lange training is niet onderdeel van dit pakket.

Dit is een **gedocumenteerde reconstructie**, geen bevestigde exacte MobileNetV2-implementatie van de auteurs. Zie `PROTOCOL.md` voor de bekende paperinstellingen, eigen keuzes en grenzen van de tijdschatting.
