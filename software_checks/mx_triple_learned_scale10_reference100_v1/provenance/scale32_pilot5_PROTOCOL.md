# Protocol — vaste schaal ×32, gezamenlijke pilot van vijf epochs

## Onderzoeksvraag

Hoe verloopt gezamenlijke training met E4M3-brongewichten, exponent-MSB-truncatie naar E2M3 en daaropvolgende mantissatruncatie naar E2M1 wanneer de vaste lagere schaalcorrectie ×32 is in plaats van ×64?

Dit is een nieuwe training vanaf dezelfde oorspronkelijke modelinitialisatie. Er wordt geen eerder getraind checkpoint gebruikt als warmstart. Het pakket voert vijf epochs uit en start geen 100-epochtraining.

## Numeriek contract

De bestaande FP32-voorbewerking blijft behouden:

`G(W) = tanh(W) / max(abs(tanh(W))) * detach(mean(abs(W)))`.

Voor een volledig nulgewichtstensor geldt de bestaande nulguard. De zes MX-convoluties zijn `layers.3`, `layers.6`, `layers.10`, `layers.13`, `layers.16`, `layers.20`. Groepen bevatten 32 waarden binnen dezelfde uitvoerchannel; een onvolledig laatste blok krijgt +nul-padding.

Per laag en minibatch wordt de actuele E4M3-bron één keer uit G(W) berekend, met de bestaande RNE-encoder en E8M0-blokschaal. Voor een niet-nulblok is de bronschaalexponent `floor(log2(amax))-8`, met de bestaande E8M0-bereikregel. Nulblokken hebben bronschaalexponent 0.

De codeketen is:

1. E4M3-bron: `[s e3 e2 e1 e0 m2 m1 m0]`.
2. E2M3: `[s e3 e2 m2 m1 m0]`, dus de hoogste twee exponentbits blijven behouden. Er is geen extra elementafronding of biascorrectie.
3. E2M1: de E2M3-code als unsigned integer twee bits naar rechts verschuiven. De twee laagste mantissabits verdwijnen.

Voor beide lagere modi is de E8M0-schaalcode `source_scale_code + 5`. De toevoeging gebeurt na promotie naar een bredere integer. Broncode 0 t/m 249 is toegestaan; broncodes 250 t/m 254 worden afgewezen omdat de afgeleide schaal niet eindig in E8M0 past. Code 255 blijft NaN en is verboden. De regel geldt ook voor nulblokken; er is geen clipping of uint8-wrap. De E4M3-bronschaal wordt niet aangepast. E2M1 kopieert exact de E2M3-schaal; ×32 wordt geen tweede keer toegepast.

De elementcodes en schalen vormen detached integerdata in een cache gedurende de drie precisiepasses. Elke pass maakt zijn eigen G(W)-autogradgraph. De volledige quantisatieroute gebruikt identity-STE ten opzichte van G(W), zonder factor 32 in backward; de originele afgeleiden van G(W) blijven aanwezig. De cache is leeg vóór iedere optimizerupdate.

Activaties blijven A32. Alle updateparameters, gradients en Adam-momenten zijn FP32. Convoluties worden in de emulator als FP32 `F.conv2d` uitgevoerd; dit pakket meet geen native low-bit throughput of energie.

## Trainingsprotocol

| Instelling | Waarde |
|---|---|
| Model | Any-Precision `svhnq`, expand=2, 132.954 parameters |
| Dataset | SVHN train: 73.257; officiële test: 26.032; geen extra |
| Invoer | Bestaande 40×40-transforms uit de aangeleverde code |
| Seed | 20261009 |
| Batchgrootte | 128, drop_last=False |
| Optimizer | Adam, lr=0,001, weight_decay=0 |
| Geconfigureerde LR-dalingen | Na epochs 50, 75, 90; gamma=0,1 |
| Werkelijke stop | Uiterlijk epoch 5; LR blijft 0,001 |
| Passvolgorde training | E4M3, E2M3, E2M1 |
| Verlies | Hard CE voor E4M3; soft CE E4M3→E2M3 en E2M3→E2M1; teachers detached, T=1 |
| Updates | Som van drie gradients; één Adam-update per minibatch |
| Evaluatie | Alle modi na elke epoch; hard CE; geen optimizerupdate |
| BatchNorm | Bestaande drie modusbanken; leren in training, bevroren in evaluatie |
| Backend | Originele CUDA/cuDNN-instellingen, benchmark=True |

De opbouw van de bron-forward wordt uitsluitend voor de modusselectie aangepast; de verliesfuncties, dropoutvolgorde en optimizersequencing blijven die van de aangeleverde implementatie.

Met 573 trainingsminibatches per epoch telt een volledige pilot 2.865 optimizerupdates en 8.595 trainingsforwards. Evaluatie gebruikt 204 minibatches per epoch en drie modi per minibatch.

## Checkpoints en vergelijking

Elke volledige epoch krijgt een immutable FP32-resumecheckpoint met model, optimizer, scheduler, RNG en contract. De runner selecteert de **eerste strikte maximale E4M3-testaccuracy** binnen de afgewerkte pilotepochs. E2M3 en E2M1 worden uit datzelfde checkpoint gerapporteerd. Daarnaast worden alle modi op de laatste pilotepoch gerapporteerd.

Bij vijf volledige epochs staan dus twee overzichtsfiguren centraal: epoch 5 en het gezamenlijke E4M3-beste checkpoint binnen epochs 1–5. Als dat dezelfde epoch is, vermeldt de notebook dit; er worden geen andere lagere precisiecheckpoints geselecteerd.

De vaste ×64-referentie komt uit de voltooide run `20261024_151210_665535_joint_run`. De meegeleverde bestanden zijn byte-identieke kopieën van diens feedback en hebben SHA256-controles. Alleen de eerste vijf epochs worden voor de pilotvergelijking gebruikt, inclusief een nieuwe selectie binnen dat bereik. Het globale beste E4M3-checkpoint op epoch 42 hoort niet bij die vergelijking. Bij een onderbroken ×32-run wordt geen ontbrekende epoch 5 verzonnen.

Beide experimenten gebruiken dezelfde seed en oorspronkelijke initialisatie. Dit maakt historische CUDA-runs niet bitexact. Eén seed en vijf epochs ondersteunen een verkennende vergelijking; een verschil in vroege accuracy bewijst geen optimum na 100 epochs. De officiële testset wordt zoals eerder voor checkpointselectie gebruikt; de geselecteerde scores zijn daardoor geen onafhankelijke testschatting.

## Exports en bestandsbehoud

Per geselecteerd checkpoint blijft één E4M3-bronexport beschikbaar met elementcodes, schalen en de benodigde FP32-uitzonderingen en BN-staat. De zes FP32-MX-mastergewichten zitten niet in deze compacte bronexport. De E2M3- en E2M1-exports worden alleen uit de opgeslagen en opnieuw ingeladen E4M3-bron afgeleid. Exacte codes/schalen en gedecodeerde gewichten worden op het trainingsdevice tegen de live route gecontroleerd. De aparte FP32-resumecheckpoints blijven behouden.

Alle nieuwe resultaten staan onder `mx_runs/mx_triple_scale32_pilot5` in unieke mappen. Eerdere pakketten, data en resultaten worden niet overschreven. De feedbackbundel bevat rapporten, meettabellen, de lege notebook, analysehulp en de ×64-referentietabellen, maar geen grote modelcheckpoints of dataset.

## Technische status versus onderzoeksuitkomst

Structurele fouten, niet-eindige waarden, gewijzigde contracts of corrupte exports geven FAIL. Een lage accuracy, constante voorspelde klasse, of een exact nul maar eindige gradient/update is een diagnostische uitkomst en wordt niet automatisch afgekeurd.

De technische controle `check` gebruikt synthetische invoer en downloadt geen dataset. Lokale validatie van het pakket gebruikt een afzonderlijke CPU-harness; de productie-CLI behoudt de bestaande Python 3.13/PyTorch 2.10.0/torchvision 0.25.0/CUDA-omgeving. Alleen de gebruiker voert de echte SVHN-pilot op de laptop uit.
