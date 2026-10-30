# ImageNet / ResNet-50 — technische controle en pilot van twee epochs

Dit pakket start de nieuwe **Any-Precision GitHub-referentie**. Het gebruikt het volledige `resnet50q`-model van de auteurs en vijf gewichts-/activatiemodi: **W32/A32, W8/A8, W4/A4, W2/A2 en W1/A1**. Het bevat uitsluitend technische controles en een pilot van **twee volledige epochs**. De latere volledige training van 80 epochs wordt een afzonderlijke, verse run.

## 1. Plaatsen en eerst de GPU-controle uitvoeren

Pak de map `anyprecision_imagenet_resnet50_pilot2_v1` uit onder:

`C:\Users\renzo\master-thesis-mx\software_checks`

Gebruik de bestaande trainingsomgeving; dit pakket installeert of wijzigt geen dependencies:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_imagenet_resnet50_pilot2_v1\run_train.py" check
```

**Voor `check` is ImageNet nog niet nodig.** De controle gebruikt synthetische FP32-inputs van 224×224. Ze controleert de CPU-zelftests en probeert in twee afzonderlijke GPU-processen:

| Controle | Volledige trainingsstap | Betekenis |
|---|---|---|
| `native256` | 256 beelden tegelijk, alle vijf precisies, backward en SGD | Geheugendiagnose; gewone BatchNorm over 256 beelden |
| `serial_dp4` | Vier groepen van 64, alle vijf precisies, één SGD-update | Gekozen pilotuitvoering; BatchNorm volgens de verdeling over vier GPU's |

Elke GPU-proef doet één opwarmupdate en twee gemeten updates. De warmstartgewichten van deze controles worden weggegooid. Gemeten worden de volledige stapduur en piekgeheugen, niet alleen één forwardpass. De stapduur omvat geen ImageNet-invoer, JPEG-decodering, validatie of checkpointschrijven en is daarom nog geen betrouwbare totale trainingstijd.

Een `OOM` bij **native256** mag voorkomen. Als de gekozen **serial_dp4**-uitvoering wel slaagt, kan de eindstatus nog steeds **PASS** zijn. Kijk naar de laatste regel **`EINDSTATUS`**. Wanneer ook groepen van 64 niet passen, wordt de eindstatus `BLOCKED_GPU_MEMORY`; `run` blijft dan geblokkeerd. Er wordt niet automatisch een kleinere referentiegroep, AMP of een ander model gekozen.

De controle gebruikt GPU 0 en de bestaande PyTorch 2.10.x / torchvision 0.25.x-omgeving. De kleine lokale CPU-tests bewijzen geen beschikbare CUDA-capaciteit; juist deze opdracht meet dat op de laptop.

## 2. Echte ImageNet-data

Voor de pilot is de volledige **ImageNet-1k / ILSVRC2012**-dataset nodig: 1.281.167 trainbeelden en 50.000 validatiebeelden, verdeeld over 1.000 klassen. Dit pakket bevat de dataset niet en downloadt niets.

`--imagenet-root` wijst naar de map die **train** en **val** bevat. Beide splits moeten dezelfde WNID-klasmappen hebben, bijvoorbeeld `train\n01440764\...JPEG` en `val\n01440764\...JPEG`. Iedere validatieklasse bevat 50 beelden. Een map met 50.000 losse validatie-JPEG's is nog niet geschikt; daarvoor zijn de officiële validatielabels nodig. Het pakket raadt geen labels en verplaatst geen bestanden.

De officiële datasetinformatie staat op [ImageNet](https://image-net.org/download.php). Voor de aantallen: [officiële downloadtabel](https://ftp.image-net.org/download) en [ILSVRC-dataoverzicht](https://image-net.org/challenges/LSVRC/2013/2013-downloads.php).

Controleer een al voorbereide dataset eventueel afzonderlijk:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_imagenet_resnet50_pilot2_v1\run_train.py" data --imagenet-root "PAD_NAAR_IMAGENET"
```

Vervang `PAD_NAAR_IMAGENET` door de echte map. De inventaris is alleen-lezen. Ze controleert klassen, aantallen, bestandsnamen/groottes en enkele afbeeldingsdecodes. Dit is geen volledige inhouds- of labelauthenticiteitscontrole. Als ImageNet nog niet beschikbaar is, voer dan eerst alleen `check` uit en deel daarvan de feedbackbundel.

## 3. Pilot starten na PASS en met voorbereide data

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_imagenet_resnet50_pilot2_v1\run_train.py" run --imagenet-root "PAD_NAAR_IMAGENET"
```

Zonder `--imagenet-root` wordt `data\imagenet` onder de projectmap gebruikt. De runner zoekt de nieuwste geslaagde `check` van exact dezelfde pakketcode, softwareversies en GPU. Een andere controle kan expliciet met `--check-report "PAD_NAAR_RUN_REPORT.JSON"` worden aangewezen.

| Instelling | Waarde |
|---|---|
| Model | Ongewijzigd auteursmodel `resnet50q` |
| Precisies | W/A 1, 2, 4, 8 en 32 |
| Globale trainbatch | **256** beelden per optimizerupdate |
| Uitvoering | Eén GPU; vier opeenvolgende groepen van normaal **64** |
| Optimizer | SGD; momentum **0,9**; weight decay **1e-5** |
| Learning rate | **0,5**; factor 0,1 na epochs **45 / 60 / 70** |
| Pilot | **2** volledige train- en validatie-epochs |
| Volledig referentierecept | **80** epochs; geen verkorte LR-planning voor de pilot |
| Loss | Hard CE op 32 bit; recursieve soft CE op 8 → 4 → 2 → 1 bit |
| Gewichtsupdate | Eén update na alle precisies én alle groepen; geen deling door vijf |
| Rekenen | FP32-emulatie, IEEE FP32; geen AMP/TF32/`torch.compile` |
| Initialisatie | Nieuwe gewichten; onze vastgelegde seed **20261009** |
| DataLoader | 4 workers; `drop_last=False`; validatiebatch 64 |
| Checkpointkeuze | Hoogste validatie-top-1 van W32/A32; alle modi uit dezelfde epoch |

**De vier groepen zijn geen gewone onbeperkte gradientaccumulatie.** De runner behoudt alleen de BatchNorm-running-statistieken van de eerste groep per precisie, zoals bij de eerste replica van `nn.DataParallel`. De andere groepen gebruiken hun eigen batchstatistieken voor forward/backward. De som van hun gewogen gradients bepaalt samen met de andere precisies één update. Zie `PROTOCOL.md` voor de precieze uitvoering en controles.

De laatste trainbatch bevat 143 beelden en wordt verdeeld als 36/36/36/35. Alle trainbeelden worden dus gebruikt. Een volledige epoch bevat 5.005 optimizerupdates en 25.025 logische precisiepasses. De seriële uitvoering doet daarbij 100.100 fysieke forward/backwardpasses. Er zijn geen MX-gewichten in deze eerste referentieproef.

De pilot bewaart een checkpoint na elke volledige epoch en stopt automatisch na epoch 2. Een lage maar eindige vroege accuracy is een meetresultaat en geen technische FAIL. De 80-epoch-run kan met dit pakket niet automatisch worden gestart.

## 4. Resultaten, notebook en feedback

Elke opdracht maakt een nieuwe map onder:

`mx_runs\anyprecision_imagenet_resnet50_pilot2\DATUM_TIJD_OPDRACHT`

Bestaande resultaten en broncode blijven behouden. In de nieuwe map staan onder andere:

- `run_report.json`, het consolelog en de bron-/pakketidentiteit.
- Bij `check`: de twee GPU-geheugenrapporten en CPU-zelftestresultaten.
- Bij training: `metrics.csv`, data-inventaris, checkpointselectie en epochduur.
- `checkpoints\epoch_001.pt` en `epoch_002.pt` na voltooide epochs; inclusief SGD, scheduler en RNG.
- **`AnyPrecision_ResNet50_ImageNet_pilot2_resultaten.ipynb`**, onuitgevoerd, met ingevulde resultatenmap.
- **`feedback_bundle.zip`** om terug te sturen. De dataset en grote checkpoints zitten hier niet in.

Open de notebook uit de resultatenmap en kies **Run All**. De notebook toont geheugen/stapduur, de volledige beschikbare top-1- en top-5-curves, de laatste epoch en het gezamenlijke checkpoint dat op W32/A32 is gekozen. Na twee epochs volgt een tijdprojectie voor een verse 80-epoch-run. Er worden geen epoch-80-resultaten ingevuld. Grafieken schrijven standaard geen nieuwe bestanden; optionele exports krijgen een eigen nieuwe map.

De GitHub-referentie is de gepubliceerde rij **Resnet-50-Any (recursive)**:

| Modus | GitHub top-1 |
|---|---:|
| W32/A32 | 74,63% |
| W8/A8 | 74,07% |
| W4/A4 | 73,84% |
| W2/A2 | 71,66% |
| W1/A1 | 58,77% |

Dit zijn eindresultaten van de gepubliceerde referentie, geen doelwaarden na twee epochs. De README stelt niet vast van welke exacte epoch deze tabelwaarden afkomstig zijn. De GitHub-instellingen en resultaten verschillen van de paper; onder andere de GitHub-start-LR is 0,5 tegenover 0,3 in de paper.

## 5. Stoppen en hervatten van de pilot

Eenmaal **Ctrl+C** vraagt stoppen na de lopende epoch, inclusief validatie en checkpoint. Een tweede Ctrl+C stopt direct. Een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap werkt eveneens na de epoch.

Alleen een onvoltooide pilot met checkpoint na **epoch 1** kan worden hervat:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_imagenet_resnet50_pilot2_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_epoch_001.pt" --imagenet-root "PAD_NAAR_IMAGENET"
```

Hervatten maakt opnieuw een resultatenmap en bewaart de oude run. Gebruik hetzelfde aantal workers; standaard 4. Model, optimizer, scheduler, RNG en sampler worden hersteld. De huidige cuDNN-backend biedt geen bitexacte hervatgarantie. Bewaar het checkpoint bij zijn metadata-JSON zodat ook de oorspronkelijke tijdmeting beschikbaar blijft. Een onderbroken deel van een epoch is niet hervatbaar.

De latere volledige 80-epoch-training begint **opnieuw vanaf epoch 0**. Pilotcheckpoints worden daarvoor niet als warmstart gebruikt.

## 6. Wat hier vooraf is gecontroleerd

De voorbereiding omvat onafhankelijke kleine CPU-vergelijkingen van de seriële uitvoering met vier expliciete modelkopieën, een volledige vijfmodus-update van het echte auteursmodel op kleine synthetische input, checkpoint-/RNG- en LR-controles, datafixtures en notebookcontroles. De samenvatting en rapporten staan in `local_validation`.

De auteursbron-ZIP is ongewijzigd en wordt per opdracht gecontroleerd op SHA-256. De echte Windows/CUDA-run en de volledige ImageNet-data zijn hier niet beschikbaar; daarom is `check` op de laptop de eerste noodzakelijke stap. Een CPU-PASS wordt niet als GPU-PASS gebruikt.

Bronnen: [auteursrepository](https://github.com/SHI-Labs/Any-Precision-DNNs), [ImageNet-trainscript](https://github.com/SHI-Labs/Any-Precision-DNNs/blob/master/train_imagenet.sh), [DataParallel](https://docs.pytorch.org/docs/2.10/generated/torch.nn.DataParallel.html) en [BatchNorm2d](https://docs.pytorch.org/docs/2.10/generated/torch.nn.BatchNorm2d.html).
