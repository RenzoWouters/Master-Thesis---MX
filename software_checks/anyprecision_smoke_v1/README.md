# Any-Precision: technische uitvoertest v1

**Dit is geen referentietraining en geen nieuw MX-experiment.** Dit pakket test
het door jou aangeleverde Any-Precision-model met enkele synthetische minibatches.
Er worden geen pakketten geinstalleerd, geen bestanden gedownload en geen bestaande
experimenten of oorspronkelijke codebestanden gewijzigd.

## Bronnen en herkomst

Het bestand `upstream/Any-Precision-DNNs-master.zip` is byte voor byte gelijk aan
jouw bijlage `Any-Precision-DNNs-master(1).zip`. De bestandsnaam in het pakket is
alleen vereenvoudigd. SHA-256:

```
7e36c5f039a78e3174421de7ff097066f95dde2837ea02f11913f3be0ba7f73c
```

Het archief bevat de oorspronkelijke `README.md`, `LICENSE.md`, modellen en code.
Bron: SHI-Labs / Any-Precision-DNNs; Haichao Yu et al., *Any-Precision Deep Neural
Networks*. De oorspronkelijke licentietekst en auteursverwijzingen blijven behouden.
De extra runner en documentatie zijn voor deze technische controle toegevoegd;
ze zijn geen door de oorspronkelijke auteurs gepubliceerde trainingsresultaten.

Relevante onderdelen uit het aangeleverde archief:

| Bestand | Gebruikte onderdelen |
|---|---|
| `models/resnet_quan.py:72-103,198-200` | Originele `resnet20q`, inclusief `expand=5` |
| `models/quan_ops.py:11-28,67-98,102-134` | Per-precisie-BatchNorm, quantizers, STE en convolutionele uitvoer |
| `models/losses.py:4-14` | Originele soft cross-entropy voor distillatie |
| `optimizer.py:4-9` | Originele optimizerfabriek |
| `train.py:180-212` | Trainingsvolgorde die de aparte runner volgt |
| `train_cifar10.sh:16-25` | Model, precisies, Adam, learning rate en weight decay |
| `utils.py:28-43` | Accuracyfunctie, met een gedocumenteerde compatibiliteitsfix |

## Bestanden in dit pakket

- `run_smoke.py`: de nieuwe, afzonderlijke technische test.
- `upstream/Any-Precision-DNNs-master.zip`: ongewijzigd oorspronkelijk archief.
- `SOURCE_MANIFEST.json`: controlegetallen voor het archief en alle bronbestanden.
- `CHANGELOG.md`: exacte scheiding tussen compatibiliteitsfixes en testinstellingen.
- `local_validation/`: controle-uitvoer van deze runner in de Linux-CPU-omgeving
  van de assistent; dit zijn nadrukkelijk geen resultaten van jouw laptop.

## Uitvoeren op jouw laptop

Pak dit pakket uit naar:

```
C:\Users\renzo\master-thesis-mx\software_checks\anyprecision_smoke_v1
```

Voer vanuit de projectmap uit:

```powershell
& ".\.venv-train\Scripts\python.exe" -B ".\software_checks\anyprecision_smoke_v1\run_smoke.py" `
    --device cuda `
    --output-root ".\technical_checks\anyprecision_smoke"
```

Deze runner is gericht op PyTorch 2.10.x en gebruikt de al geinstalleerde NumPy.
`tensorboardX`, `gpustat` en `scipy` zijn voor deze test niet nodig.
Er is geen automatische terugval van CUDA naar CPU. Een expliciete CPU-test is
mogelijk met `--device cpu`, maar vergelijk diens floating-pointwaarden niet met
de GPU-run alsof ze bit voor bit identiek moeten zijn.

## Vastgelegde testinstellingen

Model: de oorspronkelijke `resnet20q` met 10 outputs en verbredingsfactor 5.
Dit model heeft met de vijf BatchNorm-banken 6.786.330 parameters, hier lokaal
vastgesteld door `sum(p.numel() for p in model.parameters())`.

De precisies zijn `1,2,4,8,32`. Zowel `wbit` als `abit` worden op de geselecteerde
waarde gezet, zoals in de oorspronkelijke code. De eerste convolutie, de
skipconvoluties en de classificatielaag zijn in dit model gewone floating-pointlagen;
dat blijft zo. Deze test maakt het netwerk niet volledig integer of bit-packed.

Elke herhaling bestaat uit twee minibatches van twee synthetische invoertensors
met vorm `3 x 32 x 32`, getrokken uit N(0,1). Labels zijn onafhankelijke willekeurige
gehele getallen van 0 tot en met 9. Modelseed: 20261009; dataseed: 20261010.
Deze invoer is dus geen beeldclassificatiedataset en heeft geen leerbare
semantische relatie met de labels.

Per minibatch:

1. W32/A32: supervised cross-entropy met de synthetische labels en backward.
2. W8/A8, W4/A4, W2/A2, W1/A1: achtereenvolgens soft cross-entropy met de
   losgekoppelde voorspellingen van de direct voorgaande hogere precisie.
3. Een Adam-optimizerstap nadat alle vijf gradientbijdragen zijn opgeteld.

Adam wordt via de oorspronkelijke `get_optimizer_config()` aangemaakt, met
learning rate 0.001 en weight decay 0.0 uit het CIFAR-10-shellscript. Er is voor
deze twee stappen geen learning-rateplanning of epochtraining.

Er zijn twee volledige herhalingen, elk opnieuw geinitialiseerd met dezelfde seed.
De tweede is een technische herhaalbaarheidstest, geen tweede onafhankelijke
statistische trainingsreplicatie. Losses, model- en optimizertoestand en evaluatie-
uitvoer moeten binnen deze uitvoering exact overeenkomen.

Daarna wordt het checkpoint van de eerste herhaling van schijf geladen in een
nieuw model en een nieuwe optimizer. Modeltoestand, optimizertoestand en
voorspellingen worden vergeleken. Een extra trainingsstap na laden, schedulerherstel,
data-cursorherstel en volledige hervatting van training worden nog niet getest.

## Numerieke uitvoerinstellingen

De runner vraagt deterministische algoritmen met `warn_only=False`, zet
`cudnn.benchmark=False` en `cudnn.deterministic=True`, gebruikt de nieuwe
`fp32_precision='ieee'`-instellingen voor matmul en convoluties, en zet
`CUBLAS_WORKSPACE_CONFIG=:4096:8` voor dit proces voordat CUDA wordt geinitialiseerd.
Er is geen AMP of `torch.compile`. CPU-threadaantal: 2.

Dit verandert geen quantizer of netwerkarchitectuur, maar legt de numerieke
uitvoering van de technische controle vast. Exacte gelijkheid is geen garantie
tussen verschillende computers, devices, PyTorch-versies of afzonderlijke processen.
Een niet-deterministische operatie of andere fout wordt niet stilzwijgend omzeild:
het rapport krijgt `FAIL` en bevat de traceback.

Geverifieerde API-documentatie:
- https://docs.pytorch.org/docs/2.10/notes/randomness.html
- https://docs.pytorch.org/docs/2.10/notes/cuda.html
- https://docs.pytorch.org/docs/2.10/generated/torch.load.html

## Wat een PASS aantoont

De daadwerkelijk geimporteerde model-, quantizer-, loss- en optimizermodules werken
voor deze kleine invoer en configuratie. Alle precisies kunnen forward/backward,
de parameters krijgen eindige gradients en een gecontroleerd convolutioneel
mastergewicht verandert bij elke optimizerstap. De BatchNorm-tellers lopen zoals
de broncode voorschrijft. Evaluatie wijzigt de opgeslagen toestand niet. De korte
run is binnen het proces herhaalbaar en het checkpoint behoudt de gecontroleerde
toestand en voorspellingen bij laden.

De aparte accuracyfunctie krijgt tevens een known-answer-unit-test; accuracy op
de willekeurige modeldata wordt bewust niet als onderzoeksresultaat gerapporteerd.

## Wat een PASS niet aantoont

Het oorspronkelijke `train.py` wordt niet integraal gestart. Zijn datasetimports,
TensorBoard, GPU-autoselectie, CIFAR-10-download en augmentaties worden niet getest.
De runner gebruikt een gewone `TensorDataset` met `num_workers=0`; dit vervangt
geen controle van de oorspronkelijke CIFAR-10-loader.

Dit is geen bruikbaar getraind model, geen validatie van accuracy of convergentie,
geen paperreproductie, geen continual learning en geen hardwarebenchmark. De korte
looptijd en de eventueel gerapporteerde CUDA-geheugenpiek zijn alleen diagnosegegevens.

De oorspronkelijke quantizer berekent lage precisies nog steeds opnieuw uit
floating-pointgewichten. Het checkpoint bevat FP32-mastergewichten en optimizer-
toestand, geen opgeslagen compact 8-bit- of MX-bronmodel. De 32-bitroute behoudt
de `tanh`/normalisatie/herschaling uit de broncode; zij is geen zuivere standaard-
FP32-ResNet-baseline. MX, mantissatruncatie en TruncQuant worden nog niet toegevoegd.

## Uitvoer en foutafhandeling

Elke uitvoering maakt een unieke map:

```
technical_checks/anyprecision_smoke/<tijdstip>/
```

Met daarin:

- `smoke_report.json`, `summary.txt`, `run.log`: controles, eventuele fouten en waarschuwingen.
- `pip_freeze.txt`, `runner_used.py`, `source_manifest.json`: omgeving en exacte gebruikte code.
- `source_workcopy/`, `compatibility.patch`: vers uitgepakte code en de twee zichtbare patches.
- `synthetic_inputs.pt`: exact gebruikte synthetische invoer en labels.
- `smoke_checkpoint.pt`: tijdelijke technische model- en optimizertoestand, circa 78 MiB.

Deel alleen `smoke_report.json` en eventueel `run.log`; het checkpoint hoeft niet
geupload te worden. Rapporten van oude runs worden nooit overschreven. Een fout
voor het uitvoeren van het netwerk krijgt ook een rapport, voor zover de uitvoermap
kon worden aangemaakt. De exitcode is 0 bij PASS en 1 bij FAIL.

**Volgende stap na controle:** een kleine datagerichte CIFAR-10-pilot voorbereiden,
met expliciete datasplits, compatibiliteitswijzigingen, seeds en afzonderlijke
benaming ten opzichte van een echte referentietraining.
