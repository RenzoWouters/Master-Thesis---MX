# Protocol: ResNet-18, haalbaarheid op één GPU

## Doel en reikwijdte

Deze controle meet de uitvoering van een gedocumenteerde Any-Precision-ResNet-18 op synthetische ImageNet-vormige input. Ze beantwoordt twee praktische vragen: hoeveel tijdelijk GPU-geheugen vraagt de gekozen uitvoering, en hoeveel tijd kost één volledige globale update?

Er worden geen ImageNet-beelden geladen, geen epochs uitgevoerd en geen datasetaccuracies gemeten. De twee eerdere succesvolle ResNet-50-controles en de onvoltooide backendcontrole zijn historische context. Zij worden niet opnieuw gestart.

## Architectuur en herkomst

De architectuurbasis is `trunc_models/trunc_resnet_quan.py` uit het meegeleverde officiële TruncQuant-archief. De ImageNet-runner van die repository importeert deze `trunc_models`-tak. De andere `models`-tak is geen uitwisselbare bron voor deze proef.

Het gebruikte model heeft:

- Vier stages met `[2, 2, 2, 2]` BasicBlocks en kanaalbreedtes 64, 128, 256 en 512.
- Een ImageNet-stem met 7×7-convolutie, stride 2, BatchNorm, ReLU en maxpool.
- Post-activation-residuële blokken met twee 3×3-convoluties.
- Afzonderlijke BatchNorm-banken per precisie in de hoofdbranches.
- Gedeelde BatchNorm in de stem en projectiebranches, zoals in deze concrete architectuurbron.
- Een FP32-stemconvolutie, FP32-projectieconvoluties en een FP32-classifier; quantisatie in de zestien convoluties van de BasicBlocks.

De quantizers komen uit de oorspronkelijke Any-Precision-code. Het gedrag wordt op CPU vergeleken met de gewone afrondende route (`trunc=False`) van de genoemde TruncQuant-architectuur. Ook de hoogste precisie behoudt de gewichtentransformatie van de auteurscode; “W32/A32” betekent hier dus niet automatisch een ongewijzigde torchvision-FP32-ResNet-18.

Deze bronkeuze bewijst geen identiteit met de ResNet-18-implementatie achter het oorspronkelijke Any-Precision-paperresultaat; die implementatie is niet aangetroffen in de geraadpleegde bronnen. De proef en een eventuele latere training worden als reconstructie benoemd. Bestaande bronarchieven blijven ongewijzigd; het rapport legt herkomst en hashes vast.

## Vastgelegde instellingen

| Onderdeel | Waarde | Herkomst / betekenis |
|---|---|---|
| W/A-precisies | 1, 2, 4, 8, 32 | Any-Precision-paper |
| Uitvoervolgorde | 32 → 8 → 4 → 2 → 1 | Recursieve distillatie |
| Globale batch | 256 | Startkeuze aansluitend bij het beschikbare ResNet-50-script; niet bevestigd voor de originele ResNet-18-run |
| Lokale BN-groepen | 64, 64, 64, 64 | Startkeuze passend bij die vier-replica-uitvoering |
| SGD learning rate | 0,3 | Any-Precision-paper, ImageNet any-precision |
| SGD momentum | 0,9 | Beschikbare algemene auteurscode |
| Weight decay | 0,00001 | Startkeuze aansluitend bij het beschikbare ResNet-50-script |
| Seed | 20261009 | Onze reproduceerbare startkeuze |
| Rekentensors | FP32, IEEE | Software-emulatie; lage bitbreedtes zijn geen gepakte rekentensors |
| AMP / checkpointing | Uit / uit | Geen extra geheugenmaatregel in deze proef |
| cuDNN benchmark | True | Zoals de oorspronkelijke uitgevoerde ResNet-50-controles |
| cuDNN deterministic | False | Geen bitexacte CUDA-garantie |
| GPU-input | 256 × 3 × 224 × 224 | Synthetische beelden en labels |
| Updates | 1 opwarming + 2 gemeten | Geen volledige training |
| Tijdslimiet GPU-proces | 300 s | Inclusief GPU-initialisatie, opwarming en metingen |

Een eventuele latere volledige training zou het paperschema van 80 epochs gebruiken, met de learning rate telkens ×0,1 na 45, 60 en 70 epochs. Die training is geen onderdeel van dit pakket. De pilot en de volledige training zouden afzonderlijke runs zijn; de volledige training zou opnieuw vanaf epoch 0 beginnen.

## Trainingsstap en BatchNorm

De bestaande gecontroleerde Any-Precision-engine verwerkt elke precisie op vier opeenvolgende groepen. Per groep volgt direct backward; autograd-graphs worden niet tussen groepen of precisies bewaard. Iedere groepsloss krijgt gewicht `n / N`; de vijf precisielosses worden opgeteld zonder een extra deling door vijf.

De 32-bitpass gebruikt hard cross-entropy. Elke volgende precisie leert van detached voorspellingen van de onmiddellijk hogere precisie. Na alle twintig forward/backward-paren volgt precies één optimizer-update.

Voor iedere precisie beginnen de vier groepen met dezelfde BN-buffers. Alleen de bijgewerkte buffers van de eerste groep blijven behouden, overeenkomstig de buffersemantiek van vier DataParallel-replica's. Daardoor worden precisiespecifieke BN-banken één keer per globale update bijgewerkt. Gedeelde BN-buffers volgen één behouden update per precisie, dus vijf keer per globale update. Het rapport en de CPU-zelftests bewaken deze architectuurspecifieke situatie.

## Wat de staptijd meet

De twee gemeten tijden omvatten de volledige vijf-precisie-update: forward, backward en SGD. De input staat al op de GPU. De grenssynchronisatie hoort bij de meting; extra validatie, voortgangsbestanden en geheugenqueries worden niet binnen de gemeten engine-aanroep ingevoegd.

De opwarming bevat uitgebreidere controles en voortgang. De SGD-momentumstaat wordt tijdens de eerste optimizer-update aangemaakt. Daarom is de opwarmduur geen gelijkwaardige derde timingmeting.

Dataladen, JPEG-decodering, augmentatie, echte datasetvalidatie en het opslaan van trainingscheckpoints vallen buiten de gemeten staptijd. Twee synthetische updates laten geen betrouwbare uitspraak toe over langdurige thermische stabiliteit, complete epochduur of uiteindelijke accuracy.

## Geheugen en onvoltooide proeven

`allocated` is de teller voor actief tensor-geheugen binnen PyTorch. `reserved` is de ruimte die de PyTorch-allocator beheert. Beide tellers zijn geen directe meting van fysieke VRAM-residentie of Windows-gedeeld geheugen. De gerapporteerde GPU-capaciteit in een grafiek dient als context.

Het rapport bewaart geheugenstanden en pieken bij voltooide meetgrenzen. Een update die bij de tijdslimiet wordt afgebroken heeft geen voltooide stapduur. De laatste vastgelegde piek kan in dat geval kleiner zijn dan de onbekende piek van de afgebroken update. De notebook markeert zulke gegevens als gedeeltelijk en vult niets aan met nulwaarden of schattingen.

De ouder bewaakt een afzonderlijk GPU-proces. TIMEOUT en OOM zijn verschillende uitkomsten. Een wachtmelding toont dat de ouder nog wacht; ze bewijst geen GPU-voortgang. De totale opdracht kan langer duren dan 300 seconden door voorafgaande CPU-controles, opstart en afsluiting.

## Vergelijking met ResNet-50 en tijdprojectie

`baseline_results.json` bevat echte eerdere gebruikersmetingen: twee voltooide ResNet-50-controles met `benchmark=True` en één gedeeltelijke controle met `benchmark=False`. Zij geven aan hoe dezelfde laptop eerder reageerde. ResNet-18 gebruikt een andere architectuur en LR 0,3 tegenover LR 0,5 in de eerdere controles. De vergelijking is daarom geen experiment waarin uitsluitend één modelparameter verandert en geeft geen geïsoleerde oorzaak voor snelheidsverschillen.

Bij een volledige PASS geeft de notebook drie rekenkundige scenario's: de snelste, de middelste en de langzaamste van de twee gemeten staptijden. Met 1.281.167 trainingsbeelden en globale batch 256 zijn er `ceil(1.281.167 / 256) = 5.005` updates per epoch. Deze scenario's zijn geen betrouwbaarheidsinterval en geen gemeten epochduur; ze sluiten de overige trainingskosten uit.

## Beslissing na de controle

Een PASS betekent dat de korte synthetische uitvoering en de voorafgaande controles zijn geslaagd. Er volgt geen automatische vrijgave of start van een ImageNet-pilot. Eerst beoordelen we de afzonderlijke staptijden, de geheugenruimte en eventuele aanwijzingen voor onstabiele metingen. Een verdere pilot moet ook een volledige evaluatie en het dataladen op de laptop meenemen.

## Bronnen

- [Any-Precision-paper, implementatiedetails en tabel 1](https://arxiv.org/html/1911.07346v2).
- [Oorspronkelijke Any-Precision-code](https://github.com/SHI-Labs/Any-Precision-DNNs).
- [TruncQuant-code](https://github.com/a2jinhee/TruncQuant).
- [PyTorch 2.10 — DataParallel](https://docs.pytorch.org/docs/2.10/generated/torch.nn.DataParallel.html).
- [PyTorch 2.10 — CUDA-geheugenbeheer](https://docs.pytorch.org/docs/2.10/notes/cuda.html#memory-management).
