# Protocol — gezamenlijke ×32-referentie, honderd epochs

## Doel en vertrekpunt

De echte ×32-pilot met vijf epochs is technisch geslaagd. Dit experiment verlengt dezelfde trainingsmethode naar honderd epochs om een vergelijking met de bestaande volledige ×64-training mogelijk te maken. De nieuwe volledige run start vanaf de oorspronkelijke seed 20261009-initialisatie. De pilot wordt niet als warmstart gebruikt.

De numerieke codec-, adapter- en trainingskern is byte-identiek aan de pilot. De configuratieverschillen tegenover de pilot zijn uitsluitend de experimentidentiteit en omschrijving, `pilot_stop_epoch=5` vervangen door `reference_end_epoch=100`, en rapportage van epoch100.

## Numerieke route

De zes MX-convoluties zijn `layers.3`, `layers.6`, `layers.10`, `layers.13`, `layers.16`, `layers.20`. Elke uitvoerchannel wordt langs de afgevlakte invoer/kernelas in blokken van 32 gegroepeerd; een onvolledig blok krijgt +nul-padding.

De oorspronkelijke FP32-voorbewerking blijft `G(W) = tanh(W) / max(abs(tanh(W))) * detach(mean(abs(W)))`, inclusief de bestaande guard voor een volledig nulgewichtstensor.

Per laag en minibatch wordt eenmaal een actuele E4M3-bron uit G(W) gecodeerd. De E4M3-encoder gebruikt nearest/ties-to-even. Voor een niet-nulblok is de E8M0-bronschaalexponent `floor(log2(amax))-8`, met de bestaande schaalbereikregel; nulblokken hebben exponent 0.

De lagere representaties ontstaan uitsluitend door de afgesproken bitbewerking:

| Stap | Elementcode | Schaal |
|---|---|---|
| E4M3 | `s e3 e2 e1 e0 m2 m1 m0` | Actuele E4M3-bronschaal |
| E2M3 | `s e3 e2 m2 m1 m0` | E8M0-broncode +5, dus ×32 |
| E2M1 | E2M3-code als uint8 `>>2` | Exact dezelfde schaal als E2M3 |

De schaaloptelling gebeurt in een breder integer-type. Broncode 0…249 kan met +5 naar een eindige E8M0-code worden omgezet; broncode 250…254 wordt afgewezen, ook bij nulblokken. NaN-code 255 is verboden. Er is geen clipping of uint8-wrap bij de lagere schaalcorrectie. De E4M3-bronschaal blijft behouden en ×32 wordt bij de mantissastap niet nogmaals toegepast.

De bron en lagere codes zijn detached integerdata en worden gedurende één minibatch gedeeld tussen drie passes. Elke pass bouwt een eigen G(W)-autogradgraph. De volledige codeketen gebruikt identity-STE naar G(W), zonder factor 32 in backward. De oorspronkelijke G(W)-afgeleiden blijven aanwezig. Er is geen gradient door codes of schalen en er zijn geen trainbare schaalparameters.

Activaties blijven A32. Masters, gradients en Adam-momenten zijn FP32. De emulator gebruikt FP32 `F.conv2d`; codecintermediaten zijn FP64. Dit experiment meet geen native MX-kernel, hardwaredoorvoer of energie.

## Training

| Instelling | Waarde |
|---|---|
| Model | Any-Precision `svhnq`, expand=2, 132.954 parameters |
| Dataset | SVHN train: 73.257; test: 26.032; geen extra |
| Transforms | Bestaande 40×40-invoer uit de aangeleverde code |
| Initialisatie | Seed20261009,nieuwe run; geen getraind checkpoint |
| Batchgrootte | 128, drop_last=False |
| Optimizer | Adam, lr=0.001, weight_decay=0 |
| Schema | 100 epochs; LR-factor 0.1 na 50/75/90 |
| Trainvolgorde | E4M3→E2M3→E2M1 |
| Loss | Hard CE voor E4M3; soft CE van detached E4M3 naar E2M3 en detached E2M3 naar E2M1; T=1 |
| Update | Som van drie gradients, één Adam-update per minibatch |
| Evaluatie | Alle modi na iedere epoch; hard CE; geen optimizerupdates |
| BatchNorm | Modusbanken 8/32/4 leren in training; evaluatie bevroren |
| Backend | Bestaande CUDA/cuDNN-defaults, cudnn.benchmark=True |

Learning rates: epochs 1–50: 0.001; 51–75: 0.0001; 76–90: 0.00001; 91–100: 0.000001. Iedere epoch bevat 573 train- en 204 testminibatches. Een volledige run bevat 57.300 optimizerupdates en 171.900 trainingsforwards. Er is geen extra FP32-teacherforward.

De configuratie en numerieke bron komen uit de eerder gevalideerde code. De originele recursive-distillation-forward, dropoutvolgorde, lossfuncties en optimizersequencing blijven behouden.

## Selectie en grafieken

Alle volledige epochs worden bewaard. Het beste gezamenlijke checkpoint is het eerste strikte maximum van de E4M3-testaccuracy binnen de afgewerkte training. Alle drie modi worden op datzelfde checkpoint gerapporteerd. De tweede vaste rapportage is epoch100.

De notebook toont beide checkpointkeuzes met accuracybalken voor alle drie precisies, de volledige leercurves en een vergelijking met de bestaande ×64-referentie. Die referentie is de echte run `20261024_151210_665535_joint_run`, met globaal E4M3-bestcheckpoint 42 en eindpunt 100. De zes meegestuurde ruwe referentiebestanden worden met hashes gecontroleerd. De ×32- en ×64-checkpoints worden per training op E4M3 gekozen; er worden geen eigen beste lagere precisiecheckpoints gecombineerd.

Bij een onderbroken run bestaat nog geen bevestigde epoch-100-score. Tussentijdse curves en beste-checkpointvergelijkingen gebruiken alleen de gedeelde reeds gemeten epochrange. De notebook maakt duidelijk onderscheid tussen voltooide en onvoltooide referenties.

De officiële testset wordt volgens het historische protocol voor checkpointselectie gebruikt. Het geselecteerde beste resultaat is daardoor geen onafhankelijke testschatting. Eén seed geeft geen onzekerheidsinterval over initialisaties. Gelijke seed en initiële modelhash geven geen bitexacte CUDA-garantie.

## Checkpoints, hervatten en exports

Elk volledig epochcheckpoint bewaart FP32-modelstaat, Adam, scheduler, RNG en contract. Ctrl+C of `STOP_AFTER_EPOCH` kan een stop na een volledige epoch vragen. Hervatten herstelt eerdere epochgeschiedenis en trainingsstaat in een nieuwe resultatenmap. Alleen onvoltooide checkpoints van dit volledige ×32-contract worden geaccepteerd.

Best/last-kopieën en compacte bron-/afgeleide exports blijven beschikbaar. De compacte E4M3-export bevat codes, E8M0-schalen en noodzakelijke FP32-uitzonderingen/BN-staat, zonder de zes MX-mastergewichten. E2M3 en E2M1 worden uitsluitend uit die opgeslagen en opnieuw ingeladen E4M3-bron afgeleid. Gedecodeerde exportgewichten worden op het trainingsdevice exact met de live route vergeleken; deze controle is geen nieuwe volledige testsetinferentie.

Nieuwe resultaten staan uitsluitend onder `mx_runs/mx_triple_scale32_reference100`. Eerdere pakketten, resultaten en ruwe data worden behouden. De feedbackbundel bevat rapporten, tabellen, de onuitgevoerde notebook, helper en ×64-referentie; grote `.pt`-checkpoints blijven op de laptop.

## Technische en inhoudelijke beoordeling

Onjuiste contracts, niet-eindige waarden, structurele fouten of corrupte exports veroorzaken FAIL. Lage accuracy, constante voorspellingen of exact nul maar eindige gradients zijn diagnostische resultaten en geen automatische technische fout.

De pilot wijst voorlopig in het voordeel van ×64. De honderd-epochproef onderzoekt of dat verschil na de LR-dalingen en verdere training blijft bestaan. De looptijd van circa 1 uur 45 is een projectie uit de pilot; zij wordt na deze run vervangen door werkelijk gemeten tijden.
