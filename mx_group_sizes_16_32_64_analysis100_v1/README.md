# Groepsgroottes 16, 32 en 64 — vergelijking na 100 epochs

Datum: 27 oktober 2026. Alle drie de trainingen gebruiken vaste schaalfactor ×64 voor de lagere precisies, A32, FP32-mastergewichten en dezelfde exponent- en mantissatruncatieroute.

## Conclusie

De nieuwe B64-training is technisch geslaagd. Groepgrootte 32 blijft op basis van deze runs de beste uitgangspositie: de hoogste accuracies op epoch 100 en de kleinste berekende gewichtsopslag bij alle drie precisies. Zowel B16 als B64 scoort in alle laatste 25 epochs bij iedere precisie lager dan B32. Dit ondersteunt een praktische keuze voor B32 binnen deze configuratie; één run per groepgrootte bewijst geen algemeen optimum.

## Epoch 100

| Groepgrootte | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| 16 | 94,7910% | 94,5452% | 94,5721% |
| 32 | 94,9293% | 94,8986% | 94,7910% |
| 64 | 94,6911% | 94,5452% | 94,5106% |

B64 − B32: −0,2382 / −0,3534 / −0,2804 procentpunt. Dat zijn 62 / 92 / 73 minder correcte voorspellingen op 26.032 testbeelden, in de volgorde E4M3 / E2M3 / E2M1.

## Checkpoint gekozen op hoogste E4M3-accuracy

Per training bepaalt uitsluitend de eerste strikte hoogste E4M3-testaccuracy het checkpoint. Alle drie modi worden op datzelfde checkpoint gerapporteerd.

| Groepgrootte | Gekozen epoch | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|---:|
| 16 | 53 | 94,9408% | 94,4299% | 94,3992% |
| 32 | 42 | 95,0868% | 93,8422% | 93,9229% |
| 64 | 53 | 94,8294% | 94,5144% | 94,4261% |

B64 is bij deze selectieregel beter voor de twee lagere precisies dan B32. Dat is een echte uitkomst voor deze checkpoints, maar geen algemene voorsprong van B64: B32 wordt op epoch 42 gekozen, vóór de eerste LR-daling, terwijl B16 en B64 epoch 53 selecteren. Op hetzelfde epoch 53 behaalt B32 94,9485 / 94,5874 / 94,5721%, telkens hoger dan B16 en B64. De keuze van het trainingsmoment beïnvloedt de vergelijking dus sterk. De oorspronkelijke selectieregel is behouden.

## Verloop aan het einde van de training

Gemiddelde accuracies over epochs 91–100:

| Groepgrootte | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| 16 | 94,8417% | 94,5813% | 94,5417% |
| 32 | 94,9197% | 94,8479% | 94,7419% |
| 64 | 94,7384% | 94,5479% | 94,5148% |

De gemiddelden beschrijven het verloop binnen één training. De epochs zijn geen onafhankelijke herhalingen en leveren geen betrouwbaarheidsinterval over seeds. Bij B16 én B64 liggen alle drie modi in elk epoch 76–100 onder B32.

De lagere-moduswinst uit de afzonderlijke B64-pilot op epoch 5 houdt in deze volledige training geen stand. De full100-run startte opnieuw uit de oorspronkelijke initialisatie; hij hervatte de pilot niet. De eerste vijf epochs zijn daarom een nieuwe uitvoering. De bestaande CUDA-instellingen geven geen bitexacte herhaalgarantie.

## Kosten

De tabel betreft gepackte gewichtscodes plus één 8-bit shared scale per groep voor de zes MX-lagen met 129.472 echte gewichten. Groepen blijven binnen een outputkanaal; onvolledige rijstaarten worden opgevuld.

| Groepgrootte | Schaalbytes | Paddingplaatsen | E4M3 bytes | E2M3 bytes | E2M1 bytes |
|---|---:|---:|---:|---:|---:|
| 16 | 8.096 | 64 | 137.632 | 105.248 | 72.864 |
| 32 | 4.080 | 1.088 | 134.640 | 102.000 | 69.360 |
| 64 | 2.160 | 8.768 | 140.400 | 105.840 | 71.280 |

B64 bespaart 1.920 schaalbytes tegenover B32, maar heeft 7.680 extra paddingplaatsen. Netto is de gewichtsopslag daardoor 4,28% / 3,76% / 2,77% groter. B16 heeft minder padding, maar meer schaalbytes; ook daar is de totale opslag hoger dan bij B32.

De drie precisies zijn alternatieve exports en worden niet opgeteld. Dit is een analytische gepackte layout, geen bestandsgrootte of gemeten GPU-gebruik. FP32-uitzonderingen, BatchNorm, optimizerstatus, containers en werkbuffers zijn niet meegeteld. Eén E4M3-bron kan de lagere codes afleiden. De vaste offset vraagt geen geleerde parameter of per-groep-keuzebits.

## Technische controle en duur

De B64-run voltooide 100 epochs, 57.300 Adam-updates en 171.900 trainingspasses. De metriekregistraties, checkpointsselectie, LR-fasen, datasetdekking, broncache, schaalrelaties, BN-banken en FP32-status zijn consistent. De uitgevoerde packagehashes stemmen overeen met het eerder geleverde full100-pakket. De zes exporttabellen passen exact bij de onafhankelijk berekende B64-geometrie.

De volledige B64-run duurde 5.943,630 seconden, ongeveer 1 uur 39 minuten en 4 seconden. De som van epochmetingen was 5.935,449 seconden. Vergelijkbare sommen: B16 5.874,069 seconden en B32 6.032,919 seconden. B64 is in deze uitvoering ongeveer 1,62% sneller dan B32, maar dit zijn afzonderlijke laptopruns in een FP32-emulator. Er volgt geen gecontroleerd native-MX-snelheids- of energievoordeel uit.

De E4M3-bronschalen blijven op de actuele gewichten gebaseerd. Alleen de relatieve offset voor de lagere precisies is vast +6; dit is geen bevriezing van alle schalen.

## Gebruik van de bestaande notebook

De bestaande `MX_triple_group64_scale64_reference100_resultaten.ipynb` bevat reeds de volledige B16/B32/B64-vergelijking, de twee hoofdcheckpointkeuzes, curves en kosten. Er wordt geen nieuwe of uitgevoerde notebook geleverd. De aanvullende PNG-figuren zijn rechtstreeks gebaseerd op dezelfde resultaten. `group_sizes_checkpoints_detail.png` gebruikt punten en een duidelijk ingezoomde accuracy-as om de kleine verschillen zichtbaar te maken.

## Vervolgvoorstel

B32 behouden voor het verdere schaalonderzoek. Een concrete volgende verkennende proef is een vaste ×48-factor, eerst vijf epochs, met dezelfde A32/FP32-training en dezelfde bittruncatie. Daarmee wordt één tussenwaarde onderzocht voordat een nieuwe methode voor leren uit meerdere schaalkeuzes wordt toegevoegd. ×48 = 1,5 × 2^5 kan niet uitsluitend met de huidige gehele E8M0-exponentoffset worden uitgedrukt; die proef vereist expliciet een aanvullende vaste factor 1,5 voor de lagere modi. De implementatie- en hardwarekosten daarvan horen bij die nieuwe proef. Hier is daarvoor geen code aangepast of training gestart.

## Bronnen en grenzen

- Nieuwe bron: `feedback_bundle(20261027-180420).zip`, SHA256 `ead1f28e75810021579bd303f6a695c17b76bd8389b3881538b9be6a08b198f0`.
- B64-run: `20261027_172434_760515_joint_run`.
- B16- en B32-referenties zijn de ongewijzigde volledige runs die in de feedbackbundel zijn opgenomen.
- De officiële testset wordt ook gebruikt voor checkpointselectie. De geselecteerde score is daardoor geen onafhankelijke testschatting.
- De feedback bevat geen binaire checkpoints, exportbestanden of individuele voorspellingen. De analyse controleert de geregistreerde resultaten en berekent kosten onafhankelijk; zij voert geen nieuwe modelevaluatie uit.
- Alle oorspronkelijke bestanden en resultaten zijn bytegewijs behouden.

De CSV-tabellen behouden de volledige numerieke precisie. De JSON-audits leggen de controles en bronhashes vast. De map `figures` bevat presentatiefiguren; `sources` bevat de oorspronkelijke nieuwe feedbackbundel.
