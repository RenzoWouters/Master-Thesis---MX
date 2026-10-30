# Protocol — MobileNetV2-K3 op CIFAR-10

## Doel en afbakening

De vraag is of deze MobileNetV2-variant op de beschikbare laptop bruikbaar is voor herhaalde Any-Precision- en latere MX-proeven. Deze eerste meting controleert modelgedrag, volledige trainingsupdates, GPU-geheugen en geschatte looptijd. Zij bepaalt nog niet het uiteindelijke accuracyverschil tussen precisies.

De opdrachten `check` en `run` zijn begrensde diagnoses. Er is geen opdracht voor vijf of 400 volledige epochs. Alle uitvoer komt in een nieuwe map; de bestaande onderzoekscode en resultaten blijven intact.

## Herkomst en vastgelegde keuzes

| Onderdeel | Keuze | Status |
|---|---|---|
| Dataset | Officiële CIFAR-10, 50.000 trainbeelden en 10.000 testbeelden | Dataset uit de Any-Precision-proef |
| Architectuur | MobileNetV2, breedtefactor 1, gebruikelijke inverted residual-stages | Expliciete reconstructiekeuze |
| Invoer | 32×32; eerste convolutie stride 1; overige standaard strides | Expliciete CIFAR-aanpassing |
| Groepen | `K=3`; elke centrale 3×3-laag heeft `groups = hidden_channels / 8` | Gebaseerd op MoBiNet; niet bevestigd als auteurskeuze van Any-Precision |
| Precisies | W/A 1, 2, 4, 8 en 32 | Any-Precision-opzet |
| Updategewichten | Eén gedeelde FP32-master | Any-Precision-opzet |
| Batch | 128 op één GPU, zonder microbatch-accumulatie | Algemene GitHub-default; niet afzonderlijk voor MobileNetV2 bevestigd |
| Optimizer | Adam, learning rate 0,001, weight decay 0 | Expliciet in de Any-Precision-paper |
| Lang trainingsrecept | 400 epochs; learning rate ×0,1 op 150, 250 en 350 | Expliciet in de paper; alleen een referentie voor de raming |
| Dropout | 0,2 voor de classifier | Standaard MobileNetV2-keuze, hier expliciet behouden |
| Rekenen | FP32, geen autocast/AMP of TF32 | Vast beleid voor deze proef |
| cuDNN | Benchmark ingeschakeld | Zoals in de bestaande GitHubreferentie; geen bitexacte CUDA-garantie |
| Seed | 20261030 | Vastgelegde proefkeuze; geen garantie op bitexacte CUDA-herhaling |

De originele Any-Precision-repository bevat geen MobileNetV2-modeldefinitie. De bijgevoegde MoBiNet-paper geeft de regel `groups = c / 2**K`, maar behandelt een ander netwerk en bevestigt niet welke K voor Any-Precision op CIFAR-10 gebruikt werd. Deze proef maakt die ontbrekende keuze expliciet; zij claimt geen exacte reproductie van de gepubliceerde MobileNetV2-accuracy.

## Model en quantisatiepunten

De 3×3-groepsconvoluties vervangen uitsluitend de centrale depthwise-convoluties van de inverted residual-blokken. De 1×1-expansie en lineaire projectie blijven in dezelfde volgorde staan. De lineaire projectie en de residual-optelling krijgen geen extra clamp of activatie: de lineaire bottleneck van MobileNetV2 blijft behouden.

De eerste convolutie, bijbehorende eerste activatie en uiteindelijke lineaire classifier blijven floating point. Interne niet-lineaire activatiepunten gebruiken ReLU bij de 32-bitmodus en de bestaande Any-Precision-clamp naar [0,1] met afrondende quantisatie voor de lagere modi. De laatste 1×1-convolutie vóór pooling behoort tot de gequantiseerde interne lagen. Alle BatchNorm-posities krijgen vijf afzonderlijke banken.

W/A-labels benoemen de gekozen bitbreedte op deze quantisatiepunten. De uitgangen van de lineaire projecties en residual-optellingen worden ook in de lagere modi niet naar een laagbitrooster afgerond. De volgende 1×1-expansie ontvangt daardoor een floating-point-invoer. Dit is een expliciete reconstructiekeuze om de lineaire bottleneck te behouden. Daarnaast gebruikt alle fake quantization FP32-tensoropslag; deze proef meet geen werkelijk gepakte laagbitberekening.

De oorspronkelijke `quan_ops.py` en `losses.py` zijn ongewijzigd in `upstream/` opgenomen. Ook hun hoogste gewichtsmodus blijft behouden: de GitHuboperator gebruikt bij 32 bits nog de `tanh`/normalisatiestap met de gemiddelde absolute mastergewichtswaarde als schaal. De W32-modus is daarom de **GitHub-achtige hoogste precisie**, niet een apart gewoon MobileNetV2-model zonder die gewichtsoperator. De herkomst en hashes staan in `SOURCE_MANIFEST.json`.

## Eén trainingsstap

Een volledige trainingsstap gebruikt dezelfde batch voor de volgorde 32 → 8 → 4 → 2 → 1. De 32-bitpass krijgt de harde cross-entropy met de labels. Elke lagere precisie krijgt zachte doelen van de onmiddellijk hogere precisie. Die doelen zijn losgekoppeld van de gradientgraph.

Na iedere pass wordt backward uitgevoerd. De gradients worden over de vijf passes **opgeteld**, zonder door vijf te delen. Na alle vijf passes volgt precies één Adam-update. De vijf BatchNorm-banken worden elk bij hun eigen pass bijgewerkt. Een gemeten stapduur bevat dus al alle vijf precisies; de raming vermenigvuldigt die duur niet opnieuw met vijf.

## Opdrachten

### `check`

De controle verifieert de code op de CPU en voert daarna een bewaakte synthetische GPU-diagnose uit. Die bevat één opwarmupdate en drie gemeten volledige updates met batch 128. Alleen de GPU-worker heeft de tijdslimiet van 180 seconden. De voorafgaande pakketcontrole en CPU-zelftests hebben afzonderlijke limieten van 45 en 120 seconden. De synthetische controle bepaalt of alle vijf precisies, gradients, optimizer en BatchNorm-banken uitvoerbaar zijn en hoe hoog de gemeten GPU-geheugenpiek is.

Synthetische tijden meten geen echte datalader, augmentatie of dataset-I/O. Een daarvan afgeleide projectie is uitsluitend een indicatie voor de rekencomponent en mag niet als totale CIFAR-10-trainingsduur worden gelezen.

### `run`

De dataproef vereist een bijbehorende geslaagde controle van dit pakket. Zij start vanuit een nieuwe, reproduceerbaar geïnitialiseerde master en optimizer; de synthetische oefenupdates worden niet overgenomen.

De brondata worden op integriteit gecontroleerd. De trainingsdata hebben de GitHubaugmentatie: willekeurig horizontaal spiegelen, crop 32 met padding 4, tensorconversie en normalisatie met gemiddelden `(125.3,123.0,113.9)/255` en standaardafwijkingen `(63.0,62.1,66.7)/255`. Testbeelden krijgen alleen tensorconversie en dezelfde normalisatie.

Uit een vastgelegde willekeurige permutatie van de volledige trainingsset worden 100 batches van 128 gekozen, zonder teruglegging. Dezelfde geselecteerde batch voedt alle vijf precisies. De eerste tien updates blijven in het meetbestand zichtbaar, maar tellen niet mee voor de schatting van de stabiele stapduur.

De validatiediagnose gebruikt 1.000 voorbeelden uit de officiële testset, 100 per klasse. Er worden geen trainingsvoorbeelden aan deze evaluatie toegevoegd. Alle vijf precisies worden in evaluatiemodus gemeten. De eerste validatiebatch en de kleinere laatste batch worden niet gebruikt om de duur van een volledige batch te schatten.

De diagnostische checkpoint blijft lokaal. Een eventuele latere volledige training begint opnieuw vanaf nul. Dit pakket claimt geen bitexacte hervatting en biedt geen opdracht om vanuit de diagnosecheckpoint verder te trainen.

## Geheugen, tijd en procesbewaking

De runner maakt onderscheid tussen werkelijk gealloceerd PyTorch-geheugen, gereserveerd allocatorgeheugen en het fysieke GPU-geheugen. Gereserveerd geheugen omvat gealloceerd geheugen: die twee waarden mogen niet worden opgeteld. Geheugen buiten PyTorch, bijvoorbeeld driver- of desktopgebruik, valt niet volledig onder de allocatortellers.

Vóór modelallocatie wordt de PyTorch-allocatorlimiet ingesteld op `min(0.85 * totaal, 0.90 * aanvankelijk_vrij)` bytes. Zo wordt niet geprobeerd de volledige fysieke capaciteit voor deze proef te gebruiken. Een OOM met deze limiet betekent dat de gekozen route niet binnen dit vastgelegde geheugenbudget paste. De metingen tonen geen directe Windows-residentie of hoeveelheid gedeeld GPU-geheugen en garanderen niet dat andere processen hun geheugenverbruik constant houden. De batch wordt bij een OOM niet stilzwijgend verkleind of in microbatches opgesplitst.

Timing onderscheidt datalevertijd, host-naar-GPU-overdracht en de berekening. De CUDA-berekening wordt gesynchroniseerd gemeten, inclusief de vijf forwards, backwards en Adam. De stapwandtijd bevat bovendien de beperkte meet- en administratieve overhead. Notebook en rapport tonen opwarmstappen afzonderlijk.

Elke vijftien seconden meldt het bovenliggende proces dat het nog wacht. Dat is geen bewijs van GPU-voortgang. Een timeout, CUDA-OOM en gewone Pythonfout blijven verschillende eindstatussen. Bij timeout of Ctrl+C wordt uitsluitend de door deze opdracht gestarte procesboom beëindigd, inclusief eventuele dataladerprocessen. Windows gebruikt hiervoor een Job Object; het rapport vermeldt of de beëindiging gecontroleerd kon worden.

## Raming van de volledige looptijd

Bij batch 128 bevat een volledige epoch 391 trainingsbatches en 79 validatiebatches. De schatting gebruikt de stabiele gemeten wandtijd van volle batches. De kleinere laatste batches worden voor de extrapolatie conservatief als volle batches gerekend. Ook de gemeten duur van één checkpointopslag per epoch wordt meegenomen. Die opslagmeting bevat serialisatie en bestandsverplaatsing, maar niet de daaropvolgende SHA256-controle van deze diagnostische checkpoint.

De projecties voor vijf, 100 en 400 epochs bestaan uit training, evaluatie op de volledige 10.000 testbeelden en checkpointopslag. Zij zijn **ramingen, geen uitgevoerde trainingen**. Een laag/hoog scenario op basis van de gemeten stappercentielen is geen statistisch betrouwbaarheidsinterval.

De raming is afhankelijk van deze batch, vijf precisies en hetzelfde backendbeleid. Zij garandeert geen langdurig constant GPU-tempo: temperatuur, vermogensinstelling, andere programma's en datalevering kunnen veranderen. Eenmalige installatie, eventuele datasetdownload/-extractie en procesopstart worden niet als terugkerende epochkosten meegerekend. De korte meetfase is geen volledige studie van laptopthermiek.

## Interpretatie van loss en accuracy

De getoonde vroege metrics dienen om foutieve of niet-eindige berekeningen te herkennen. Na 100 batches heeft het model nog geen volledige epoch doorlopen. De loss van de hoogste precisie gebruikt bovendien harde labels, terwijl de lagere precisies distillatiedoelen gebruiken; de losswaarden zijn daarom geen gelijksoortige kwaliteitsrangschikking.

De evaluatie van 1.000 beelden is een diagnostische steekproef. Zij mag niet worden vergeleken met de gepubliceerde eindaccuracy van een volledig getraind model of met de bestaande SVHN- en CIFAR-referenties. Deze proef beantwoordt primair of verdere training praktisch uitvoerbaar lijkt.

## Bronnen

- Any-Precision-paper: <https://arxiv.org/abs/1911.07346>, in het bijzonder Implementation Details en Algorithm 1.
- Oorspronkelijke Any-Precision-code: <https://github.com/SHI-Labs/Any-Precision-DNNs>; exacte aangeleverde archief- en bestandshashes staan in `SOURCE_MANIFEST.json`.
- MoBiNet-preprint: <https://arxiv.org/abs/1907.12629v2>, sectie 3.3 en tabellen 3–4.
- Standaard MobileNetV2-structuur: <https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/mobilenetv2.py>.

