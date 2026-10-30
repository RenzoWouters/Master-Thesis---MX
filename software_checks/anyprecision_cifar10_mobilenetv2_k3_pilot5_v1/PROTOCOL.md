# Protocol — MobileNetV2-K3, CIFAR-10, vijf epochs

## Onderzoeksvraag

Deze pilot volgt op de geslaagde korte geheugen- en tijdproef. Hij meet of de vijf Any-Precision-modi tijdens enkele volledige epochs leren, of de berekeningen eindig blijven en of de gemeten snelheid aanhoudt. De pilot bepaalt nog niet het uiteindelijke verschil tussen de precisies.

De opdrachten in dit pakket kunnen tot maximaal epoch 5 trainen. Alle runs starten in een nieuwe resultatenmap. `run` begint met nieuwe gewichten; `resume` kan alleen een voltooide epoch van dezelfde pilot voortzetten. De geplande latere referentie van 400 epochs begint opnieuw vanaf nul.

## Vastgelegde instellingen en herkomst

| Onderdeel | Instelling | Herkomst |
|---|---|---|
| Data | CIFAR-10, 50.000 train / 10.000 test, 32×32 RGB | Any-Precision-werklast |
| Model | MobileNetV2, breedtefactor 1, gebruikelijke inverted residual-stages | Gedocumenteerde reconstructie |
| CIFAR-aanpassing | Eerste convolutie stride 1; overige standaard strides | Expliciete eigen keuze |
| Groepsconvoluties | `K=3`, `groups = hidden_channels / 8` in de 17 centrale 3×3-lagen | Gebaseerd op de MoBiNet-regel; niet bevestigd als Any-Precision-auteurskeuze |
| Precisies | W/A 32, 8, 4, 2 en 1; één gedeelde FP32-master | Any-Precision-opzet |
| Batch | 128, laatste kleinere batch behouden | Algemene GitHub-default; niet afzonderlijk bevestigd voor MobileNetV2 |
| Optimizer | Adam; lr 0,001; betas (0,9, 0,999); epsilon 1e-8; weight decay 0 | Paperrecept en expliciete optimizerinstellingen |
| Lang schema | 400 epochs, lr ×0,1 na 150, 250 en 350 voltooide epochs | Paperrecept; deze pilot voert alleen de eerste vijf epochs uit |
| Dropout | 0,2 vóór de classifier | Standaard MobileNetV2-keuze |
| Rekenen | FP32, geen AMP of TF32 | Zelfde beleid als de geslaagde tijdproef |
| Backend | `cudnn.benchmark=True`, niet-deterministisch | Bestaand GitHubbeleid; geen bitexacte CUDA-belofte |
| Seed / loader | Seed 20261030; nul dataloaderworkers | Expliciete pilotinstellingen |

De learning rate blijft in deze vijf epochs 0,001. Het schema wordt niet samengedrukt naar vijf epochs. Bij hervatten telt de globale epoch verder; de planning begint niet opnieuw. Voor het langere schema betekent een grens van 150 dat epoch 151 de verlaagde learning rate gebruikt.

De oorspronkelijke Any-Precision-repository bevat geen MobileNetV2-modeldefinitie. Ook de keuze van K voor hun CIFAR-10-resultaten is niet bevestigd. De bijgevoegde MoBiNet-paper beschrijft `groups = c / 2**K` voor een ander netwerk. Daarom is dit een controleerbare reconstructie en geen claim van exacte reproductie van de gepubliceerde accuracies.

## Model en quantisatie

De centrale 3×3-depthwise-convoluties van MobileNetV2 zijn vervangen door groepsconvoluties met acht ingangskanalen per groep. De 1×1-expansies, lineaire projecties en residualverbindingen blijven behouden. K is een convolutiekeuze en is niet de groepsgrootte van MX.

De eerste convolutie met zijn ReLU en de uiteindelijke lineaire classifier blijven floating point. Op interne niet-lineaire activatiepunten gebruikt de hoogste modus ReLU. De lagere modi gebruiken de bestaande Any-Precision-clamp naar [0,1] en afrondende quantisatie. De laatste 1×1-convolutie vóór pooling wordt wel gequantiseerd.

Lineaire projectie-uitgangen en residualoptellingen krijgen geen extra activatiequantisatie. Daardoor kan de volgende 1×1-expansie een floating-point-invoer ontvangen. Dit is een expliciete reconstructiekeuze om de lineaire bottleneck te behouden. De W/A-labels beschrijven de geplaatste quantisatoren; zij betekenen niet dat iedere tussentensor op een laagbitrooster ligt. Alle fake quantization gebruikt FP32-tensoropslag.

Alle 52 BatchNorm-posities hebben vijf aparte banken. De oorspronkelijke `upstream/quan_ops.py` en `upstream/losses.py` zijn bytegelijk overgenomen uit de aangeleverde code. Hun W32-gewichtsmodus behoudt de `tanh`-/normalisatiestap en de gemiddelde absolute mastergewichtswaarde als schaal. W32 is dus de hoogste GitHub-achtige modus; het is geen afzonderlijke conventionele MobileNetV2 zonder deze gewichtsoperator.

## Trainings- en evaluatieprocedure

Elke epoch doorloopt alle 50.000 trainingsvoorbeelden één keer in een nieuwe, met de eigen generator bepaalde willekeurige volgorde. Batchgrootte 128 geeft 391 updates, met 80 beelden in de laatste batch. De trainingsaugmentatie volgt de GitHubvolgorde: horizontaal spiegelen, crop 32 met padding 4, tensorconversie en normalisatie. Gemiddelden zijn `(125.3,123.0,113.9)/255`; standaardafwijkingen `(63.0,62.1,66.7)/255`.

Eén update voert dezelfde batch uit op 32 → 8 → 4 → 2 → 1 bits. W32 krijgt harde cross-entropy met de labels. Elke lagere modus krijgt zachte doelen van de onmiddellijk hogere modus, losgekoppeld van de gradientgraph. Na iedere pass volgt backward; de vijf gradients worden opgeteld zonder door vijf te delen. Daarna volgt één Adam-update.

Na iedere epoch worden alle 10.000 officiële testbeelden geëvalueerd, zonder augmentatie en zonder gewichts- of BN-updates. Batchgrootte 128 geeft 79 batches, met 16 beelden in de laatste batch. Alle vijf modi worden op dezelfde beelden gemeten. De term `validation` in de rapportvelden verwijst naar deze officiële testset; er is geen aparte validatiesplit gemaakt.

De brondata worden vóór het laden gecontroleerd tegen de officiële checksums, inclusief `batches.meta`. De loader zoekt eerst de complete datasetmap van de geslaagde tijdproef. Bestaande data worden niet overschreven. Een ontbrekende dataset wordt alleen met expliciet `--download` opgehaald.

## Checkpoints en hervatting

Elke complete epoch krijgt een eigen checkpoint met modelstatus, Adamstatus, alle BN-buffers, globale epoch/update-tellers, learning-rateplanning, Python-/NumPy-/PyTorch-RNG-status en beide DataLoader-generatorstatussen. De checkpoint wordt eerst als tijdelijk bestand geschreven en pas daarna als complete checkpoint gepubliceerd. Reeds bestaande checkpoints worden niet overschreven.

Hervatten gebeurt uitsluitend op een epochgrens, met dezelfde code/configuratie, passend runtimebeleid en dezelfde gecontroleerde datasetinhoud. De nieuwe resultatenmap bevat de eerdere epochgeschiedenis plus de nieuwe epochs. Een afgebroken epoch wordt vanaf de laatst voltooide checkpoint opnieuw uitgevoerd. Checkpoints van de eerdere timingproef worden niet als pilotcheckpoint geaccepteerd.

Het herstellen van RNG- en optimizerstatus voorkomt een opzettelijke herstart van training. Bij het gehandhaafde niet-deterministische CUDA-beleid is bitexacte gelijkheid tussen een onderbroken en een ononderbroken GPU-run niet gegarandeerd. Het rapport vermeldt de uitgevoerde hervatcontroles en eventuele afwijkingen.

Deze checkpoints zijn bedoeld voor hervatten binnen de vijf-epochpilot. Een latere 400-epochreferentie begint vanaf nul, zoals afgesproken.

## Checkpointselectie en presentatie

Het laatste voltooide checkpoint wordt naast het checkpoint met de hoogste W32/A32-top-1 getoond. Bij een exacte tie wordt de vroegste epoch gekozen. De andere precisies worden beoordeeld op datzelfde gekozen checkpoint; iedere precisie kiest dus niet zijn eigen beste epoch.

Tijdens een volledige pilot is de laatste epoch 5. Bij een bewuste pauze of onderbreking vermeldt het notebook de werkelijk laatste voltooide epoch. Onvoltooide epochs worden niet aangevuld of geïnterpoleerd. De accuracylijnen tonen de ruwe metingen zonder smoothing.

Selectie op de officiële testset volgt de hier afgesproken rapporteringsconventie. De waarde van het beste checkpoint is daardoor een geselecteerde testscore en geen onafhankelijke nieuwe test. De score van de vooraf vastgelegde laatste epoch blijft daarom steeds zichtbaar.

## Tijd, geheugen en procesbewaking

De pakketcontrole en CPU-zelftests hebben afzonderlijke limieten van 45 en 180 seconden. De GPU-check heeft een limiet van 240 seconden en omvat één opwarmupdate plus twee meetupdates met batch 128, een trainingsupdate met batch 80, evaluatie met batch 16 en de hervatdiagnose. De pilotworker heeft een limiet van 1.800 seconden; dit is een bovengrens, geen voorspelde duur. De ouder meldt iedere vijftien seconden dat de bewaking actief is. De batchmeldingen tonen de feitelijke voortgang. Bij een timeout of Ctrl+C beëindigt de bewaking alleen de eigen procesboom. Windows gebruikt daarvoor een Job Object.

De PyTorch-allocatorlimiet is `min(0.85 * totaal, 0.90 * aanvankelijk_vrij)` bytes. Gealloceerd geheugen is onderdeel van gereserveerd geheugen; deze waarden worden niet opgeteld. Het notebook toont de piektellers. Die beschrijven PyTorch-allocaties en geven geen directe meting van Windows-residentie of gedeeld GPU-geheugen.

De gemeten epochtijden bevatten de training op alle vijf precisies, evaluatie op de volledige testset en checkpointopslag. Eenmalige datasetdownload, controles en procesopstart worden apart van de terugkerende epochkosten behandeld. Langere looptijden worden uit complete gemeten epochs geraamd. Als meerdere epochs beschikbaar zijn, laat de standaardraming de eerste epoch buiten het gemiddelde wegens de eerste opwarming. De gebruikte epochs staan bij de raming.

De raming voor 50, 100 of 400 epochs is geen uitgevoerde training en geen garantie voor langdurig identieke snelheid. Temperatuur, energiebeleid, opslag en andere processen kunnen de looptijd veranderen. De vijf precisies zitten al in één trainingsupdate; er wordt geen extra factor vijf toegepast.

## Interpretatie

Deze pilot kan laten zien of alle modi beginnen te leren en of numerieke problemen optreden. Hij kan niet vaststellen dat de modellen geconvergeerd zijn of dat hun eindaccuracy de paper volgt. Een achterstand van lage precisies kan in deze fase mede een verschil in leersnelheid zijn.

Trainingsloss van W32 gebruikt harde labels; de lagere modi gebruiken distillatiedoelen. Die losswaarden zijn daarom niet onderling als gelijksoortige kwaliteitsmaat te rangschikken. Tijdens evaluatie gebruiken alle modi gewone cross-entropy met labels. De notebook toont beide fasen apart.

## Bronnen

- Any-Precision-paper: <https://arxiv.org/abs/1911.07346>, Implementation Details en Algorithm 1.
- Aangeleverde Any-Precision-code: <https://github.com/SHI-Labs/Any-Precision-DNNs>; exacte hashes in `SOURCE_MANIFEST.json`.
- MoBiNet-preprint: <https://arxiv.org/abs/1907.12629v2>, sectie 3.3 en tabellen 3–4.
- MobileNetV2-structuur: <https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/mobilenetv2.py>.
