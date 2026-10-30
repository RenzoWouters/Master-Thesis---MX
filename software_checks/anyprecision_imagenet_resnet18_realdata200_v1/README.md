# ResNet-18 / ImageNet: 200 trainingsbatches en korte validatie

Deze proef controleert de praktische uitvoering met echte ImageNet-beelden: dataladen, training, geheugen, een checkpoint en validatie. Ze start een nieuw model en stopt na **200 trainingsbatches**. De vijf precisies worden bij iedere batch gezamenlijk getraind.

## Uitvoeren

Pak de map `anyprecision_imagenet_resnet18_realdata200_v1` uit onder:

`C:\Users\renzo\master-thesis-mx\software_checks`

De voorbereide ImageNet-map moet deze indeling hebben:

| Pad | Inhoud |
|---|---|
| `train\n########\...` | Trainingsbeelden per officiële WordNet-klasse |
| `val\n########\...` | Validatiebeelden per dezelfde klasse |

Het pakket controleert de volledige ILSVRC2012-dataset: 1.000 klassen, 1.281.167 trainingsbeelden en 50.000 validatiebeelden. Het laadt tijdens de proef slechts de hieronder beschreven selectie. Als de dataset ontbreekt of de indeling niet klopt, volgt een duidelijke fout vóór GPU-training. De proef downloadt, verplaatst of herlabelt geen data.

Met de standaardlocatie `C:\Users\renzo\master-thesis-mx\data\imagenet`:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_imagenet_resnet18_realdata200_v1\run_pilot.py" run
```

Staat ImageNet elders, geef de map die `train` en `val` bevat:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_imagenet_resnet18_realdata200_v1\run_pilot.py" run --imagenet-root "D:\datasets\imagenet"
```

Alleen de dataset controleren kan met `data` in plaats van `run`. Alleen de CPU-zelftests uitvoeren kan met `selftest`.

## Opzet

| Onderdeel | Instelling |
|---|---|
| Model | Dezelfde gecontroleerde ResNet-18-reconstructie met oorspronkelijke Any-Precision-quantizers |
| Precisies | W/A 32, 8, 4, 2, 1 |
| Training | 200 batches × 256 = **51.200 beelden**, willekeurige volgorde zonder herhaling |
| BatchNorm | Vier opeenvolgende groepen van 64; alleen de buffers van de eerste groep blijven per precisie behouden |
| Validatie | **5.000 beelden**, vijf per klasse; batches van 64; alle vijf precisies |
| Optimizer | SGD; LR 0,3; momentum 0,9; weight decay 0,00001 |
| Rekenen | IEEE FP32; `cudnn.benchmark=True`; geen AMP, MX of truncatie |
| Dataladen | Standaard vier workers; te wijzigen met `--workers` |

De eerdere synthetische meting gaf ongeveer 2,095 seconden per volledige update. Dat komt voor 200 updates rekenkundig neer op ongeveer zeven minuten, vóór extra kosten voor echte data, validatie en opslag. De werkelijke duur wordt in deze proef gemeten.

Het afzonderlijke proefproces heeft een **automatische tijdslimiet van twintig minuten**, inclusief initialisatie, dataladen, training, checkpointopslag en validatie. De voorafgaande CPU-tests en datasetinventarisatie komen daar nog bij. Elke vijftien seconden kan een wachtmelding verschijnen; die melding bevestigt dat de bewaking actief is, niet dat een GPU-update voltooid is.

## Resultaten

Elke uitvoering krijgt een eigen map onder:

`mx_runs\anyprecision_imagenet_resnet18_realdata200\DATUM_TIJD_run`

Daar staan het rapport, de logs, een lokale checkpoint na de 200 trainingsupdates, `feedback_bundle.zip` en de **onuitgevoerde** notebook `ResNet18_ImageNet_200batches.ipynb`.

De notebook vergelijkt de volledige staptijd met de rekentijd, toont het verloop over de proef en de geheugenmetingen, en geeft een voorzichtige tijdsprojectie. De vroege train- en validatieaccuracies zijn diagnostiek; ze zijn geen eindresultaat of reproductie van de paperaccuracy.

**Deel `feedback_bundle.zip`.** De dataset en checkpointtensors blijven lokaal. Een latere volledige training begint opnieuw vanaf nul; het checkpoint van deze gedeeltelijke epoch dient alleen om de proef vast te leggen. Het pakket start geen vervolgepoch en bevat geen hervatcommando voor volledige training.

| Status | Betekenis |
|---|---|
| PASS | De gevraagde uitvoering is volledig voltooid en gecontroleerd. |
| OOM | Een geheugenallocatie is mislukt. |
| TIMEOUT | Het proefproces is bij de tijdslimiet gestopt; dit bewijst geen OOM. |
| FAIL | Een andere controle of uitvoering is mislukt; zie het rapport. |

De exacte herkomst, keuzes en meetgrenzen staan in `PROTOCOL.md`.
