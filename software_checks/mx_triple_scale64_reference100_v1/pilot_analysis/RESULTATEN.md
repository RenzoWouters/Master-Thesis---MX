# Resultaten van de echte vijf-epochpilot met schaal ×64

De gezamenlijke route-B-training is technisch geslaagd en leert in alle drie precisies. Na vijf epochs liggen de lagere precisies ongeveer tien procentpunten hoger dan in de eerdere naïeve truncatiepilot met behoud van de E4M3-bronschaal. De directe drieprecisiepilot blijft op dit meetpunt nog iets hoger.

Deze analyse betreft de echte SVHN/CUDA-runs van de gebruiker. Er is voor deze analyse geen training uitgevoerd. De volledige honderd-epochrun met schaal ×64 is nog niet uitgevoerd.

## Accuracy op de twee geselecteerde checkpoints

Voor alle drie de pilots is epoch 5 ook het checkpoint met de hoogste E4M3-testaccuracy binnen epochs 1–5. De cijfers voor **epoch 5** en **E4M3-best** zijn daardoor hier identiek. Binnen elke run komen alle drie precisies uit hetzelfde checkpoint.

| Precisie | Direct uit bron | Naïeve truncatie, schaal S | Truncatie, schaal 64S | 64S − naïef | 64S − direct |
|---|---:|---:|---:|---:|---:|
| E4M3 | 93,11% | 82,08% | 92,85% | +10,78 pp | -0,25 pp |
| E2M3 | 93,07% | 81,50% | 91,86% | +10,36 pp | -1,21 pp |
| E2M1 | 92,80% | 81,86% | 91,66% | +9,80 pp | -1,13 pp |

De tabel is afgerond op twee decimalen. `comparison.json` en `comparison.csv` bewaren zowel de gerapporteerde accuracy als de uit `correct1 / n` berekende accuracy. De JSON/CSV bevatten ook de exacte breuken en de verschillen in procentpunten. De kleine floating-pointafwijking tussen de twee accuracyberekeningen verandert deze afgeronde cijfers niet.

## Verloop en technische controles

| Epoch | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| 1 | 87,87% | 86,44% | 86,39% |
| 2 | 90,74% | 88,79% | 88,49% |
| 3 | 92,06% | 90,35% | 90,22% |
| 4 | 92,44% | 91,52% | 91,37% |
| 5 | 92,85% | 91,86% | 91,66% |

De testaccuracy neemt in deze vijf epochs voor alle drie modi toe. Iedere modus voorspelt in iedere epoch alle tien klassen. De testloss op epoch 5 is E4M3 0,246666, E2M3 0,283670, E2M1 0,295374. E2M1 heeft wel een kleine stijging in test-CE van 0,292849 op epoch 4 naar 0,295374 op epoch 5, terwijl zijn accuracy nog stijgt. De verliezen zijn dus niet allemaal strikt dalend.

De run bevat vijf volledige epochs, 2.865 Adam-updates en 8.595 trainingspasses. De aangeleverde geschiedenis meldt voor iedere epoch volledige train-/testdekking, eindige gradients, FP32-Adamtoestand, correcte BN-tellers en één E4M3-bron per MX-laag en minibatch. De aantallen bronafleidingen, schaaloffsets +6 en behouden E2M3-schalen bij mantissatruncatie zijn onderling consistent. Alle zes MX-masters veranderden in iedere epoch.

De exportmetadata meldt exact gelijke gedecodeerde gewichten tegenover de live trainingsroute, met de lagere modi afgeleid uit de opgeslagen E4M3-bron. De feedbackbundel bevat de checkpointtensors zelf niet. Deze analyse controleert daarom de metadata, aantallen en consistentie van de gerapporteerde livecontroles; zij decodeert de gewichten niet opnieuw en voert geen modelinferentie uit.

De twee geregistreerde waarschuwingen betreffen verouderde `device`-argumenten bij PyTorch `pin_memory()` en `is_pinned()`. Er is geen technische fout gemeld.

## Wat de vergelijking ondersteunt

De drie pilots hebben exact dezelfde initiële modelhash, datasetbestanden, transforms, softwareomgeving, backendinstellingen en gemeenschappelijke trainingsinstellingen. De gedeelde initiële modelhash is `a8fcd20c40a204ba24cabfa36258f9501d55b9d4d4bdfb8ba742b8c827af127f`. De vergelijking gebruikt telkens hetzelfde aantal epochs en dezelfde officiële testset. De broncode en numerieke route van de schaal×64-pilot komen overeen met het voorbereide pilotpakket; alle vijftig gerapporteerde pakkethashes zijn gecontroleerd.

De resultaten ondersteunen dat de vaste schaalfactor ×64 tijdens gezamenlijke training een veel bruikbaardere route oplevert dan de eerdere naïeve route met dezelfde bronschaal. De resterende achterstand ten opzichte van directe kwantisatie is ongeveer 0,25 procentpunt voor E4M3, 1,21 voor E2M3 en 1,13 voor E2M1. Vijf epochs bepalen nog niet hoe groot die achterstand na honderd epochs zal zijn.

Ook E4M3 verbetert sterk ten opzichte van de naïeve truncatiepilot, terwijl de E4M3-kwantisator zelf gelijk blijft. Dat is consistent met de gedeelde optimalisatie: andere lagere forwards veranderen de gezamenlijke trainingsdynamiek en daarmee de geleerde mastergewichten en overige toestand. Dit experiment is geen afzonderlijke causale meting van dat effect. Het aandeel van masteraanpassing, BN-parameters, BN-statistieken en recursieve distillatie is niet van elkaar geïsoleerd.

Dit blijft één seed en een pilot van vijf epochs. De officiële testset is ook gebruikt voor checkpointselectie. De factor 64 herstelt de verwijderde exponentbits niet, en er is geen bewijs van algemene superioriteit of van een native MX-snelheids-/energievoordeel.

## Tijd en volgende proef

- Gemeten volledige sessie: **296,90 seconden**, ongeveer 4 minuten en 57 seconden.
- Som van de gemeten epochs, inclusief evaluatie en checkpointopslag: **289,74 seconden**.
- Projectie voor een verse honderd-epochrun: **95,94 minuten**, ongeveer 1 uur en 36 minuten.
- Projecties op basis van de geobserveerde snelle en trage epochs: **88,75–104,71 minuten**. Dit is geen betrouwbaarheidsinterval.

De tijdprojectie omvat training, evaluatie, diagnostiek en checkpointschrijven. Zij is geen gemeten honderd-epochtijd en geen voorspelling van accuracy. Langdurige warmteontwikkeling en achtergrondbelasting kunnen de werkelijke duur beïnvloeden.

De logische volgende meting is een verse honderd-epochtraining met exact dezelfde schaal×64-route, dezelfde A32/FP32-opzet en het oorspronkelijke LR-schema 50/75/90. Daarna kunnen epoch 100 en het door E4M3 gekozen beste checkpoint worden vergeleken met de overeenkomstige volledige eerdere trainingen.

## Herkomst en controleerbaarheid

- `raw/`: geselecteerde ongewijzigde bestanden uit `feedback_bundle(20261024-125844).zip`.
- `naive_reference_raw/`: ongewijzigde referentiebestanden uit `feedback_bundle(20261023-162654).zip`.
- `direct_reference_raw/`: ongewijzigde referentiebestanden uit `feedback_bundle(20261023-125617).zip`.
- `comparison.json`: volledige getallen, deltas, vergelijkbaarheidsvoorwaarden en SHA256-herkomst.
- `comparison.csv`: dezelfde drie precisievergelijkingen in tabelvorm.
- `independent_audit.json`: uitgevoerde metadata-/telcontroles, hun reikwijdte en bronhashes.

Alle achttien gekopieerde bronbestanden zijn byte voor byte vergeleken met de oorspronkelijke upload-ZIPs. De originele uploads en eerdere experimentpakketten zijn ongewijzigd gebleven.
