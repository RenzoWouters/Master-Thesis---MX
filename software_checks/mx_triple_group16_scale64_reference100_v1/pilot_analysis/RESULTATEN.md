# Analyse van de echte B16 ×64-pilot

Bron: `feedback_bundle(20261026-175140).zip`, SHA256 `f61de5d8cdbb3c46e6962c194c7aec07f0d9dacc18ce0e8d9f25d197258229c2`.

## Technische uitkomst

PASS: vijf volledige epochs, 2.865 Adamupdates, 8.595 trainingspasses. A32/FP32, groepsgrootte 16, vaste offset +6, gedeelde E4M3-bron en mantissaschaal zijn correct gecontroleerd. Modelinitialisatie, upstreambron, datasetbestanden en transformaties stemmen overeen met de B32-referentie. De aangeleverde pakkethashes passen exact bij pilotv2.

De audit controleerde 196 consistentievoorwaarden. Het feedbackbestand bevat geen checkpoint-/exportbinaries. Exportcontrole berust hier op de geregistreerde exacte CUDA-vergelijkingen van de runner en onafhankelijke herberekening van de zes laagvormen; er is geen nieuwe modelevaluatie uitgevoerd.

## Accuracy op epoch 5

| Precisie | B32 ×64 | B16 ×64 | Verschil B16−B32 | Extra correcte voorspellingen |
|---|---:|---:|---:|---:|
| E4M3 | 93,009% | 92,932% | −0,077 procentpunt | −20 |
| E2M3 | 91,299% | 92,060% | +0,761 procentpunt | +198 |
| E2M1 | 90,892% | 92,167% | +1,275 procentpunt | +332 |

Beide runs selecteren binnen epochs 1–5 ook epoch 5 als het gezamenlijke E4M3-beste checkpoint. Eindpunt en geselecteerd beste checkpoint vallen hier dus samen.

Het voordeel voor de lagere precisies verschijnt vooral in epoch 5. E2M1 bij B16 lag in epochs 1–4 nog onder B32; E2M3 lag erboven in epochs 3 en 5. Dit is een veelbelovend eindpunt, geen bewijs van een over de hele training betere leercurve of hogere eindaccuracy. De volledige 100-epochproef is nodig om te zien of het voordeel standhoudt. Het betreft één seed met de oorspronkelijke niet-bitexacte CUDA-backend.

## Looptijd

B16: 260,367s voor de vijf epochs inclusief training/evaluatie/checkpoints; 267,414s voor het hele proces. B32: 303,728s voor dezelfde eerste vijf epochs. Dat is in deze metingen 14,28% korter. Het verschil is beschrijvend en geen gecontroleerde hardware-efficiëntiemeting.

B16 heeft een mediane epochtijd van 51,739s in epochs 2–5. De projectie voor een nieuwe 100-epochrun is ongeveer 86,26minuten, oftewel 1 uur 26 minuten, exclusief opstart/eindexport en bij vergelijkbare laptopomstandigheden. Een langetermijnmeting van deze B16-run bestaat nog niet.

## Kosten

| B | Schalen | Padding | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---:|---:|---:|---:|---:|---:|
|16|8.096|64|137.632 bytes|105.248 bytes|72.864 bytes|
|32|4.080|1.088|134.640 bytes|102.000 bytes|69.360 bytes|
|64|2.160|8.768|140.400 bytes|105.840 bytes|71.280 bytes|

Voor de zes MX-lagen kost de B16-bron 2.992 bytes meer dan B32 (+2,22%). De lagere afzonderlijke exports nemen respectievelijk 3,18% en 5,05% toe. Meer schalen worden gedeeltelijk gecompenseerd door minder padding. B64 heeft door padding een grotere bronrepresentatie dan B32. B64-accuracy is niet gemeten.

Dit zijn afzonderlijke bitgepakte representaties van 129.472 MX-gewichten plus padding en schalen. Ze worden niet opgeteld tot drie residentmodellen en zijn geen checkpoint-/GPU-geheugencijfers. FP32-uitzonderingen, BN-toestand en containers zijn uitgesloten.

## Vervolg

De pilot geeft geen technische reden om te stoppen. De volgende proef is een verse B16 ×64-training van 100 epochs met exact dezelfde numerieke kern. De vergelijking gebruikt epoch 100 en per training het eerste strikte E4M3-maximum, met alle drie modi uit hetzelfde geselecteerde checkpoint. De bestaande B32 ×64-run wordt hergebruikt.

De oorspronkelijke pilotnotebook leest deze echte bundel correct: metrics en checkpointselectie zijn veilig, de referentie is compatibel en er zijn geen analysefouten. Er is geen nieuwe uitgevoerde pilotnotebook nodig.
