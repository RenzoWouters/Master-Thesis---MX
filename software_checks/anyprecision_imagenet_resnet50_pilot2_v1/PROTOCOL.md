# Protocol en afwijkingen van de oorspronkelijke uitvoeromgeving

## Doel

Een uitvoerbare GitHub-referentie voor ImageNet / ResNet-50 voorbereiden met dezelfde globale batch, de oorspronkelijke quantisatie en recursieve any-precision-loss. De eerste echte run is beperkt tot twee epochs om geheugen, technische werking en tijd te beoordelen. De training van 80 epochs blijft een latere, verse proef.

Dit is de configuratie van de gepubliceerde GitHub-code. Het is geen claim dat de historische seed, hardware of ieder detail van het oorspronkelijke paperexperiment is teruggevonden.

## Bronnen en integriteit

De meegeleverde `upstream/Any-Precision-DNNs-master.zip` komt uit de door de gebruiker aangeleverde projectbron. SHA-256:

`7e36c5f039a78e3174421de7ff097066f95dde2837ea02f11913f3be0ba7f73c`

`SOURCE_MANIFEST.json` controleert ook alle 19 bronbestanden. Per opdracht wordt een nieuwe werkkopie gemaakt. Geen upstreambestand wordt gepatcht. De runner importeert het auteursmodel, de quantizers, de soft loss en de optimizerfactory rechtstreeks onder een eigen modulenaam. De originele trainer wordt als bron bewaard; de nieuwe runner implementeert dezelfde lossvolgorde en voert extra controles/logging uit.

Het auteursmodel is een pre-activation ResNet-50 met bottleneckverdeling `[3,4,6,3]`, 1.000 outputs en vijf BN-banken per schakelbare BN-module. Het gecontroleerde totaal is **25.769.512 parameters**, inclusief de extra precisieafhankelijke BN-parameters. De FP32-stand gebruikt ook de gewichtsnormalisatie van de auteurscode; dit is geen willekeurig torchvision-FP32-model.

## Eén minibatch op één GPU

De originele configuratie gebruikt `batch_size=256` en vier GPU's. De trainer gebruikt `nn.DataParallel`. Elke replica krijgt normaal 64 beelden. De BatchNorm-berekening is lokaal; alleen de in-place bufferupdates van replica 0 blijven in het hoofdmodel behouden.

De seriële uitvoering doet daarom per globale minibatch:

1. Zet de gekozen precisie, in volgorde **32, 8, 4, 2, 1**.
2. Leg de bufferstaat vast voordat die precisie wordt uitgevoerd.
3. Verdeel de input met `torch.chunk(4)`; verwerk de vier stukken met dezelfde huidige mastergewichten.
4. Bereken hard CE voor 32 bit. Iedere lagere precisie gebruikt de losgekoppelde softmaxvoorspelling van de direct hogere precisie op dezelfde voorbeelden.
5. Vermenigvuldig de mean loss van een stuk met `n_stuk / N_globaal` en doe backward. Er wordt geen gradient gewist tussen stukken of precisies.
6. Herstel de oorspronkelijke bufferstaat vóór replica 1, 2 en 3. Dit gebeurt na de vorige backward, zodat autograd geen tussentijdse bufferwijziging krijgt.
7. Behoud na deze precisie uitsluitend de bufferstaat die door replica 0 werd geproduceerd.
8. Ga verder met de volgende precisie. Na alle vijf precisies volgt precies **één** SGD-update.

Voor een bitmodus b is de totale bijdrage daardoor de som van de lokale gemiddelde losses gewogen met n/N; dit is de gemiddelde loss over dezelfde globale batch. De vijf bitmodi worden opgeteld, zoals bij de auteurs. Er wordt niet door vijf gedeeld.

Gewone viermaal-accumulatie zou running-statistieken viermaal aanpassen. Dat gedrag gebruikt deze pilot niet. Een native batch 256 zou BatchNorm over 256 beelden uitvoeren. Daarom wordt native256 uitsluitend als geheugendiagnose gemeten.

De laatste batch van de volledige trainingsset heeft **143** voorbeelden; `torch.chunk(4)` geeft **36/36/36/35**. De gewichten n/N houden die laatste update correct. De datasets worden niet ingekort en `drop_last=False` blijft behouden.

Het model bevat geen dropout of andere stochastische forwardlaag die een extra replicatie-RNG-beleid nodig maakt. CPU-tests vergelijken de numerieke semantiek met een onafhankelijke oracle met expliciete kopieën en globale losses. Dat is geen bitexacte vier-GPU-CUDA-test: kernels en optelvolgorde kunnen op echte hardware afrondingsverschillen geven.

## Vast trainingsrecept

SGD met LR 0,5, momentum 0,9 en weight decay 1e-5. MultiStepLR verlaagt met factor 0,1 na 45, 60 en 70 voltooide epochs. De twee pilotepochs gebruiken dus allebei LR 0,5. Het pakket laat geen ander epochaantal, andere batch, andere precisielijst of AMP toe.

Trainaugmentatie: PIL-RGB, `RandomResizedCrop(224)` met bilineaire interpolatie, horizontale flip, ToTensor en ImageNet-normalisatie. Validatie: Resize 256, CenterCrop 224, dezelfde normalisatie. Trainbeelden worden geschud. Alle 50.000 validatiebeelden worden in alle vijf modi geëvalueerd met bevroren BN en hard CE.

De validatiebatch is **64** om extra piekgeheugen te vermijden. In evaluatiemodus koppelt BN geen voorbeelden meer via batchstatistieken. De validatie gebruikt dus dezelfde opgeslagen statistieken en alle voorbeelden; de berekeningsvolgorde kan minimale floating-pointverschillen geven.

De eigen seed 20261009 is expliciet vastgelegd. De oorspronkelijke auteursseed is niet bekend. Vier DataLoader-workers zijn gekozen als praktische instelling voor de laptop, tegenover de parserdefault 16 in de oude trainer. Bij hervatten moet het aantal workers gelijk blijven. Een afzonderlijke training-samplergenerator wordt opgeslagen. Deze implementatie claimt geen historische of bitexacte match van de augmentation-RNG.

De berekeningen blijven FP32. PyTorch 2.10's nieuwe `fp32_precision="ieee"`-instellingen voorkomen automatische TF32-versnelling; er wordt geen oude en nieuwe precision-API gemengd. `cudnn.benchmark=True` blijft behouden. De oude omgeving (Python 3.7 / Torch 1.1 / torchvision 0.2.1) wordt niet teruggeïnstalleerd op de moderne laptop.

## Controle, fouten en toestemming om te trainen

De GPU-geheugenproeven draaien in verse processen. De parent maakt tijdens zijn CPU-tests geen CUDA-model aan. Een fout in native256 vervuilt zo niet de CUDA-staat van de seriële proef. Elke proef omvat de model-/inputallocatie, één volledige opwarmupdate, twee gemeten updates en controles op eindige waarden. Piek `allocated` en `reserved` zijn PyTorch-geheugentellers; dit is geen hardware-energie- of geheugentransfermeting.

De trainingsopdracht vereist een PASS-rapport van dezelfde pakketcode, relevante packageversies, CUDA-build en GPU-eigenschappen. De gekozen route moet slagen. Een geheugen-OOM van native256 is een toegelaten diagnose; een andere fout wordt eerst onderzocht. Een OOM van serial_dp4 blokkeert de pilot. Er is geen automatische kleinere trainbatch, kleiner BN-groepje, AMP of CPU-fallback.

Bij iedere echte train- en validatiebatch worden eindige losses en geldige accuracy-tellers gecontroleerd. Vóór ieder checkpoint worden model-, BN- en optimizerstaat op eindige FP32-waarden gecontroleerd. Lage maar eindige accuracy is geen softwarefout.

## Checkpoints en rapportage

Checkpoints worden pas na een volledige train- plus validatie-epoch geschreven en bevatten model, SGD-momentum, scheduler, Python/NumPy/Torch/CUDA-RNG, samplerstaat en metriekhistorie. Metadata gebruikt veilige primitieve waarden en tensors, zodat `torch.load(..., weights_only=True)` volstaat. De runner weigert checkpoints van andere experimenten, andere pakketcode, andere data-inventaris of een al voltooide pilot.

Het beste checkpoint wordt geselecteerd op de hoogste W32/A32-top-1 op de ImageNet-validatieset, met strikt `>` voor een verbetering. Alle andere precisies worden op diezelfde epoch gerapporteerd. De laatste epoch is een afzonderlijke vergelijking. De gepubliceerde GitHub-tabel bevat geen verifieerbaar epochnummer; die waarden krijgen dus geen verzonnen label “epoch 80”.

Na twee volledige epochs is de tijdprojectie voor een nieuwe run: de gemeten duur van epoch 1 plus 79 maal de gemeten duur van epoch 2. Trainen, validatie en checkpointschrijven tellen mee; data-inventaris en aanvankelijke technische checks niet. Temperatuur, andere processen, opslag en dataladen kunnen de echte duur veranderen. Een losse synthetische GPU-stap wordt niet als volledige ImageNet-tijdmeting gepresenteerd.

De feedbackbundel sluit datasets en model-/optimizercheckpoints uit. Ze bevat de kleine rapporten, logs, notebook, runnerbron en gecontroleerde upstreamwerkkopie. Alle bestaande runs blijven behouden.

## Primaire bronnen

- [Any-Precision README en resultaten](https://github.com/SHI-Labs/Any-Precision-DNNs)
- [ImageNet-launchscript](https://github.com/SHI-Labs/Any-Precision-DNNs/blob/master/train_imagenet.sh)
- [ImageNet-trainingslus](https://github.com/SHI-Labs/Any-Precision-DNNs/blob/master/train_imagenet.py)
- [PyTorch 2.10 DataParallel](https://docs.pytorch.org/docs/2.10/generated/torch.nn.DataParallel.html)
- [PyTorch 2.10 BatchNorm2d](https://docs.pytorch.org/docs/2.10/generated/torch.nn.BatchNorm2d.html)
- [PyTorch scatterimplementatie](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/cuda/comm.cpp)
- [Paper](https://arxiv.org/abs/1911.07346)
