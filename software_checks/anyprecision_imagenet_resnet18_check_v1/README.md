# ResNet-18 / ImageNet: korte geheugen- en snelheidscontrole

Deze proef onderzoekt of de afgesproken ResNet-18-uitvoering praktisch haalbaar is op jouw GPU. Het pakket is zelfstandig en gebruikt synthetische beelden; **ImageNet is niet nodig**.

## Uitvoeren

Pak de map `anyprecision_imagenet_resnet18_check_v1` uit onder:

`C:\Users\renzo\master-thesis-mx\software_checks`

Voer daarna uit:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_imagenet_resnet18_check_v1\run_check.py" check
```

De controle voert eerst CPU-zelftests uit. Daarna volgt **één opwarmupdate en twee gemeten updates** op de GPU. Elke update verwerkt alle vijf precisies en één globale batch van 256 beelden, verdeeld in vier groepen van 64. De GPU-proef stopt uiterlijk na **300 seconden**, afgezien van een korte afsluitfase. De CPU-controles en het opstarten komen daar nog bij.

## Vastgelegde opzet

| Onderdeel | Instelling |
|---|---|
| Model | ImageNet-ResNet-18 uit de gebruikte `trunc_models`-tak van TruncQuant, gecombineerd met de originele Any-Precision-quantizers |
| Precisies | W/A 32, 8, 4, 2 en 1; eerste laag, classifier en projectieconvoluties blijven FP32 |
| Batch | Globaal 256; vier opeenvolgende groepen van 64; BatchNorm-beleid zoals vier DataParallel-replica's |
| Rekenen | IEEE FP32; `cudnn.benchmark=True` |
| Loss | Hard cross-entropy voor 32 bit; recursieve knowledge distillation voor lagere precisies |
| Optimizer | Eén SGD-update per globale batch; LR 0,3; momentum 0,9; weight decay 0,00001 |
| Input | Synthetisch, 224 × 224 pixels |

Dit is een **gedocumenteerde reconstructie**. De exacte ResNet-18-code en volledige instellingen achter het oorspronkelijke Any-Precision-paperresultaat zijn niet aangetroffen in de geraadpleegde bronnen. De batch, verdeling in groepen van 64 en weight decay zijn expliciete startkeuzes. De architectuurbron en alle keuzes staan in `PROTOCOL.md` en in het runrapport.

De proef gebruikt gewone afrondende quantisatie. Er wordt geen MX, exponent- of mantissatruncatie, AMP of activation checkpointing toegevoegd.

## Resultaten bekijken

Elke uitvoering maakt een nieuwe map onder:

`mx_runs\anyprecision_imagenet_resnet18_check\DATUM_TIJD_check`

Daar staan het rapport, de logs, `feedback_bundle.zip` en de **onuitgevoerde** notebook `ResNet18_geheugen_snelheid_check.ipynb`. De notebook toont:

- Beide gemeten staptijden en de vastgelegde PyTorch-geheugenpieken.
- De eerdere ResNet-50-controles als hardwarecontext. Model en learning rate verschillen, dus dit is geen vergelijking waarbij maar één instelling verandert.
- Een voorzichtige rekentijdprojectie, alleen als beide ResNet-18-metingen succesvol zijn voltooid. Dataladen, validatie en checkpointopslag vallen buiten deze projectie.

**Deel de feedbackbundel.** Een PASS betekent dat de korte synthetische uitvoering is geslaagd. We beoordelen daarna de snelheid en geheugenruimte voordat een ImageNet-pilot wordt voorbereid.

| Status | Betekenis |
|---|---|
| PASS | De controles en volledige synthetische GPU-proef zijn geslaagd. |
| OOM | De GPU-proef rapporteerde een geheugenallocatiefout. |
| TIMEOUT | De GPU-proef is wegens de tijdslimiet gestopt; dit bewijst geen OOM. |
| FAIL | Een andere controle of uitvoering is mislukt; zie rapport en log. |

Dit pakket bevat geen commando voor een ImageNet-training. Iedere uitvoering krijgt een eigen resultatenmap; eerdere resultaten worden niet overschreven.

Alleen de CPU-zelftests uitvoeren kan met dezelfde opdracht en `selftest` in plaats van `check`. Een CPU-PASS is geen GPU-meting.
