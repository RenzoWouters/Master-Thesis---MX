# Protocol: ResNet-18 met 200 echte ImageNet-trainingsbatches

## Doel

Deze proef onderzoekt of de op de laptop geslaagde synthetische ResNet-18-uitvoering ook praktisch werkt met dataladen, augmentatie, echte beelden, een checkpoint en validatie. De training begint met nieuwe willekeurige gewichten en is begrensd op 200 globale updates. De resultaten dienen voor een besluit over een volgende pilot; er wordt geen volledig getraind netwerk of paperaccuracy verwacht.

## Vast model en recept

Het model en de engine zijn overgenomen uit de gecontroleerde ResNet-18-proef. De architectuur komt uit `TruncQuant/trunc_models/trunc_resnet_quan.py`: vier stages `[2, 2, 2, 2]` met BasicBlocks en kanaalbreedtes 64, 128, 256, 512. De zestien branchconvoluties en bijbehorende activaties gebruiken de oorspronkelijke Any-Precision-quantizers. Stem, projectieconvoluties en classifier blijven FP32. De zestien switchable-BN-modules hebben ieder vijf precisiebanken; de stem en drie projecties behouden hun gedeelde BN.

Deze combinatie is een **gedocumenteerde reconstructie**. De exacte ResNet-18-code en alle runinstellingen achter de oorspronkelijke Any-Precision-paperresultaten zijn niet aangetroffen in de geraadpleegde bronnen. De bronbestanden blijven ongewijzigd en worden op hashes gecontroleerd. De hoogste modus behoudt ook de `tanh`-gewichtentransformatie van de auteurscode; W32/A32 betekent hier niet automatisch een ongewijzigde torchvision-FP32-ResNet-18.

| Instelling | Waarde | Herkomst |
|---|---|---|
| Precisies en volgorde | W/A 32 → 8 → 4 → 2 → 1 | Oorspronkelijke Any-Precision-aanpak |
| Globale batch | 256 | Expliciete startkeuze, aansluitend bij het beschikbare ResNet-50-script |
| Lokale BN-groepen | 64 + 64 + 64 + 64 | Dezelfde gecontroleerde serial-DP4-uitvoering |
| Loss hoogste precisie | Hard cross-entropy | Oorspronkelijke training |
| Loss lagere precisies | Recursieve distillatie met detached hogere voorspellingen | Oorspronkelijke training |
| SGD | LR 0,3; momentum 0,9; weight decay 0,00001 | Paper-LR, algemene auteursoptimizer; weight decay overgenomen als startkeuze |
| Seed | 20261009 | Onze vastgelegde startkeuze |
| Rekenen | IEEE FP32; AMP uit | Dezelfde gecontroleerde uitvoering |
| Backend | `cudnn.benchmark=True`, `cudnn.deterministic=False` | Dezelfde gecontroleerde uitvoering; geen bitexacte CUDA-garantie |
| MX / truncatie / activation checkpointing | Uit | Any-Precision-referentiefase |
| Trainingsupdates | Precies 200 | Begrenzing van deze haalbaarheidsproef |

Per precisie volgen vier forwards en backwards, waarbij iedere groepsloss wordt gewogen met `n/N`. De vijf precisielosses worden opgeteld, zonder een extra deling door vijf. Na de twintig forward/backward-paren volgt één SGD-update.

Alle groepen beginnen per precisie met dezelfde BN-buffers. Alleen de bijgewerkte buffers van de eerste groep worden behouden, zoals bij vier DataParallel-replica's. Per globale update stijgt iedere precisiebank één keer; de vier gedeelde BN-modules stijgen vijf keer. Validatie gebruikt bevroren BN en verricht geen optimizer-updates.

## Dataset en selectie

De invoer is een vooraf klaargemaakte, volledige ImageNet-1k / ILSVRC2012-dataset. Train en val moeten dezelfde 1.000 gesorteerde WordNet-klasmappen bevatten. Verwacht worden 1.281.167 trainingsbeelden en 50.000 validatiebeelden, met vijftig validatiebeelden per klasse. De controle inventariseert de bestanden en decodeert enkele voorbeelden. Ze bevestigt structuur, aantallen en een pad/grootte-fingerprint; ze controleert niet onafhankelijk alle semantische labels of alle bestandsinhoud.

Voor training worden 51.200 beelden zonder herhaling gekozen in een seed-vaste willekeurige volgorde. Die selectie vormt de eerste 200 batches van een geshuffelde epoch. Het betreft dus een gedeeltelijke epoch, geen kleine nieuwe dataset waarmee volledige ImageNet-resultaten kunnen worden gerapporteerd.

Voor de korte validatie worden seed-vast vijf beelden per klasse gekozen: 5.000 beelden totaal. Zo omvat de validatie alle klassen en wordt de bias van alleen de eerste ImageFolder-klassen vermeden. Validatiebatch 64 verdeelt onafhankelijke evaluatiebeelden; zij verandert de trainingsbatch of het BN-trainingsbeleid niet. De laatste validatiebatch bevat acht beelden.

De traintransformatie volgt het oorspronkelijke recept: PIL RGB, `RandomResizedCrop(224)`, bilineaire interpolatie met antialiasing, horizontale flip, tensorconversie en ImageNet-normalisatie. Validatie gebruikt resize naar korte zijde 256, center crop 224 en dezelfde normalisatie. Dataladerkeuze en validatieselectie gebruiken aparte randomgeneratoren, zodat ze de willekeurige modelinitialisatie niet stil wijzigen. Het aantal dataloader-workers wordt in het rapport vastgelegd.

De dataset wordt niet gedownload, uitgepakt, verplaatst of herlabeld. Dezelfde brondata en eerdere resultaten blijven behouden. Een ontbrekende of onvolledige dataset blokkeert de echte proef vóór GPU-training.

## Tijdmeting

De 200 echte updates behoren allemaal tot de proef, ook de eerste updates waarin GPU-kernels en dataladers opwarmen. De notebook laat daarom zowel alle afzonderlijke staptijden als gemiddelden over opeenvolgende groepen van twintig zien.

De meetvelden onderscheiden wachten op een batch, overdracht naar de GPU en de volledige vijf-precisie-update met forward, backward en SGD. De totale staptijd bevat de gemeten datawachttijd, overdracht en update. Er wordt ook een totale fase- en opdrachttijd bijgehouden voor kosten die buiten deze intervallen vallen. De rekentijd kan naast de eerdere synthetische staptijd worden gezet; dat is een vergelijking van meetomstandigheden, geen nieuw gecontroleerd experiment waarin slechts één factor verandert.

Geheugenqueries, controle van BN en voortgangsrapportage horen niet bij de zuivere compute-meting. Door prefetching kan dataverwerking gedeeltelijk met GPU-werk overlappen. De gemeten datawachttijd is daarom de zichtbare wachttijd van de trainingslus, niet de totale CPU-tijd voor decodering en augmentatie.

De eerdere synthetische controle leverde twee staptijden van 2,09946 en 2,08959 seconden op. De mediaan was 2,09453 seconden per globale update; de hoogste vastgelegde gereserveerde PyTorch-geheugenpiek, inclusief opwarming, was 3,07031 GiB. Deze gegevens zijn historische hardwarecontext uit de meegestuurde eerdere gebruikersrun.

## Geheugen

`allocated` telt actief PyTorch-tensorgeheugen; `reserved` telt het door de PyTorch-allocator beheerde geheugen. Geen van beide is een directe meting van fysieke VRAM-residentie of van Windows-gedeeld geheugen. De notebook toont vastgelegde standen en pieken met hun meetbereik.

Een TIMEOUT kan de laatste update afbreken voordat een definitieve piek is opgeslagen. Een gedeeltelijk rapport bevat alleen de al vastgelegde waarnemingen. Ontbrekende staptijden of geheugenwaarden worden niet als nul of als geslaagde meting weergegeven.

## Checkpoint en stoppen

Na de 200 trainingsupdates wordt de modeltoestand, optimizerstaat en RNG-informatie lokaal vastgelegd. Het checkpoint wordt expliciet als diagnose na een gedeeltelijke epoch aangeduid. Het is geen startpunt voor de latere volledige referentietraining: die begint opnieuw vanaf nul.

Het bewaakte kindproces omvat initialisatie, dataladen, de 200 updates, checkpointopslag en validatie. De vaste limiet is 1.200 seconden. Bij TIMEOUT of interrupt wordt de procesboom van deze proef opgeruimd, inclusief de bijbehorende dataloader-workers. De voorafgaande datasetinventarisatie en CPU-zelftests vallen buiten die limiet. Wachtmeldingen komen om de vijftien seconden; zij zijn geen bewijs van voltooide GPU-vooruitgang.

Elke uitvoering schrijft naar een unieke nieuwe resultatenmap. Het feedbackarchief bevat rapporten, meetgegevens en de gebruikte code; datasetbeelden en modeltensors blijven buiten dit archief. Er volgt geen automatische extra epoch of volledige training.

## Interpretatie van accuracy en tijdprojecties

Trainaccuracy tijdens 200 updates en accuracy op 5.000 validatiebeelden beschrijven een zeer vroeg trainingsstadium. De validatie is geen volledige ImageNet-valscore en ondersteunt geen vergelijking met de eindaccuracies uit de paper. De verschillende trainlosses hebben bovendien een ander doel: hard CE voor W32/A32, distillatie voor de overige precisies.

Alleen een volledig geslaagde run kan een projectie voor een langere training tonen. De notebook gebruikt de updates 21–200 om de sterkste opstartinvloed apart te houden. Met een globale batch van 256 zijn er `ceil(1.281.167 / 256) = 5.005` updates per volledige epoch. De korte validatietijd kan rekenkundig worden opgeschaald van 5.000 naar 50.000 beelden, maar vaste kosten, laatste batches en cachegedrag schalen niet exact lineair.

De scenario's zijn rekenkundige extrapolaties, geen betrouwbaarheidsinterval of gemeten epochduur. Langdurige temperatuur, I/O, validatie en checkpointkosten blijven relevant. Een latere volledige training zou het paperschema van 80 epochs volgen, met LR-dalingen na 45, 60 en 70 epochs. Dat schema wordt tijdens deze 200 updates niet doorlopen.

## Bronnen

- [Any-Precision-paper](https://arxiv.org/html/1911.07346v2).
- [Oorspronkelijke Any-Precision-code](https://github.com/SHI-Labs/Any-Precision-DNNs).
- [TruncQuant-code en gebruikte architectuurtak](https://github.com/a2jinhee/TruncQuant).
- [PyTorch 2.10 — DataParallel](https://docs.pytorch.org/docs/2.10/generated/torch.nn.DataParallel.html).
- [PyTorch 2.10 — DataLoader](https://docs.pytorch.org/docs/2.10/data.html).
- [PyTorch 2.10 — CUDA-geheugenbeheer](https://docs.pytorch.org/docs/2.10/notes/cuda.html#memory-management).
- [ImageNet-downloadinformatie](https://image-net.org/download.php).
