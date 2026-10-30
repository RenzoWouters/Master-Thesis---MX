# ResNet-50: korte controle van de cuDNN-keuze

Deze diagnose onderzoekt of een andere selectie van convolutionele GPU-kernels het geheugenverbruik en de staptijd verbetert. De twee eerdere controles zijn technisch geslaagd, maar hun gemeten updates duren ongeveer 84–112 seconden. Tijdens de controle is het toegewezen GPU-geheugen bijna vol en is een kleine hoeveelheid gedeeld geheugen zichtbaar. Dat ondersteunt onderzoek naar geheugendruk; het bewijst niet dat alle vertraging daardoor ontstaat. **15,6 GB in de screenshot is het maximum van de grafiek, niet de gebruikte hoeveelheid gedeeld geheugen.**

## Uitvoeren

Pak de map `anyprecision_resnet50_backend_check_v1` uit onder:

`C:\Users\renzo\master-thesis-mx\software_checks`

Laat de bestaande map `anyprecision_imagenet_resnet50_pilot2_v1_1` ernaast staan. De diagnose gebruikt en controleert die bestaande code; ze wijzigt die map niet.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_resnet50_backend_check_v1\run_check.py" check
```

**ImageNet is niet nodig.** Er worden één opwarmupdate en twee gemeten updates op synthetische beelden uitgevoerd. De GPU-proef heeft een tijdslimiet van 300 seconden. De eerdere controles worden niet opnieuw uitgevoerd; hun echte meetgegevens zitten in `baseline_results.json`.

## Wat vergelijken we?

| Instelling | Eerdere controles | Nieuwe diagnose |
|---|---|---|
| `cudnn.benchmark` | `True` | **`False`** |
| Model en quantizers | Auteursmodel `resnet50q` | Hetzelfde |
| Precisies | W/A 32, 8, 4, 2, 1 | Hetzelfde |
| Globale batch / BN-groepen | 256 / vier groepen van 64 | Hetzelfde |
| Loss en optimizer | Recursieve losses; één SGD-update | Hetzelfde |
| Rekenen | IEEE FP32; geen AMP | Hetzelfde |
| Data | Synthetische input 224×224 | Hetzelfde |

Met `benchmark=True` kan cuDNN meerdere algoritmen uitproberen om een snelle kernel te kiezen. Met `False` verandert die selectieroute. De gekozen kernel en zijn werkgeheugen kunnen daardoor verschillen. **Een lager geheugenverbruik of hogere snelheid is niet gegarandeerd.** Ook bitexact gelijke CUDA-uitkomsten worden niet gegarandeerd. Er wordt geen kleinere BN-batch, activation checkpointing of andere precisie ingevoerd.

De extra controles en voortgang tijdens de opwarming horen niet bij de twee gemeten staptijden. De metingen omvatten forward, backward en SGD voor alle vijf precisies; dataladen, JPEG-decodering, validatie en checkpoints vallen erbuiten. De opwarming start nog zonder de SGD-momentumstaat die tijdens de eerste update wordt aangemaakt. Daarom vergelijken we de twee gemeten updates tussen de proeven.

## Resultaten

Elke uitvoering maakt een nieuwe map onder:

`mx_runs\anyprecision_resnet50_backend_check\DATUM_TIJD_check`

Daar staan `run_report.json`, logs, `feedback_bundle.zip` en de **onuitgevoerde** notebook `Resultaten_backend_check.ipynb`. De notebook krijgt automatisch het pad van deze resultatenmap. Ze vergelijkt de twee bestaande controles met de nieuwe diagnose, met grafieken van beide gemeten staptijden en de PyTorch-geheugenpieken.

- **PASS:** de nieuwe synthetische uitvoering is voltooid; beoordeel daarnaast tijd en geheugen.
- **OOM:** de proef liep tegen een geheugenfout aan.
- **TIMEOUT:** de proef voltooide niet binnen de tijdslimiet; dit bewijst geen OOM.
- **FAIL:** een andere controle of uitvoering is mislukt; de fout staat in het rapport.

Onvoltooide proeven krijgen geen verzonnen nulmetingen in de grafieken. PyTorch `allocated` en `reserved` beschrijven tensors en de geheugenallocator; het zijn geen directe metingen van fysieke VRAM-residentie of Windows-gedeeld geheugen.

**Deel de feedbackbundel. Deze diagnose start geen ImageNet-training en is geen GPU-checkbewijs voor de bestaande pilot.** Ook na PASS beoordelen we eerst de tijd en het geheugen. Bestaande resultaten en checkpoints blijven behouden.

## Bronnen

- [PyTorch 2.10: cuDNN benchmarking](https://docs.pytorch.org/docs/2.10/notes/randomness.html#cuda-convolution-benchmarking)
- [PyTorch 2.10: geheugenbeheer](https://docs.pytorch.org/docs/2.10/notes/cuda.html#memory-management)
- [NVIDIA: gebruik van systeemgeheugen bij GPU-geheugendruk](https://nvidia.custhelp.com/app/answers/detail/a_id/5490)
- [Microsoft: toegewezen en gedeeld GPU-geheugen in Taakbeheer](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)
