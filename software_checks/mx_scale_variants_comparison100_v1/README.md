# Vier bestaande MX-schaalvarianten vergelijken

Dit pakket vergelijkt de reeds uitgevoerde 100-epochresultaten van vaste ×32, vaste ×64, geleerde ×32/×64-blokkeuzes en dezelfde geleerde variant met vaste offsets na epoch 50. Alle benodigde metric- en selectiebestanden zijn inbegrepen. De notebook leest uitsluitend deze resultaten; zij start geen training, laadt geen checkpoints en downloadt geen data.

## Gebruik

Pak de ZIP uit onder:

```text
C:\Users\renzo\master-thesis-mx\software_checks
```

Open daarna:

```text
software_checks\mx_scale_variants_comparison100_v1\MX_schaalvarianten_vergelijking100.ipynb
```

Gebruik dezelfde Python-kernel als bij de eerdere resultaatnotebooks. De analyse gebruikt pandas, NumPy en Matplotlib. Voer de cellen van boven naar beneden uit. De meegeleverde notebook is leeg en bevat geen opgeslagen celuitvoer.

Behoud de hele uitgepakte map: naast de notebook staan de helper, de brongegevens in `data` en het bronmanifest. Het pakket kan zijn locatie vinden vanuit de pakketmap of vanuit de projectmap met `software_checks`. Bij een andere indeling kan `BASE_DIR` in de parametercel worden ingevuld. Met `SAVE_FIGURES = True` kunnen de figuren ook worden opgeslagen; standaard blijft dit uit.

## Welke vier varianten?

| Variant | Extra offset voor E2M3 en E2M1 | Verloop |
|---|---|---|
| Vast ×32 | Overal +5 | Alle 100 epochs vast |
| Vast ×64 | Overal +6 | Alle 100 epochs vast |
| Geleerd tot 100 | Per MX-blok +5 of +6 | Keuzes leren tijdens alle 100 epochs |
| Geleerd tot 50, daarna vast | Geleerde +5/+6-keuzes na epoch 50 | Gemeenschappelijke geleerde epochs 1–50; vaste keuzes in 51–100 |

In alle gevallen wordt E4M3 uit de actuele gewichten bepaald. E2M3 volgt via dezelfde exponenttruncatie en E2M1 via dezelfde mantissatruncatie, met behoud van de E2M3-schaal. Het vastzetten betreft de extra offset, terwijl de E4M3-bronschaal dynamisch blijft.

De vier varianten gebruiken SVHN train-only met 73.257 trainingsbeelden en 26.032 testbeelden, CNN-8/svhnq expand2, A32, FP32-mastergewichten, dezelfde oorspronkelijke basisinitialisatie, seed, datatransformaties, batchgrootte, Adam, LR-mijlpalen en drie recursieve distillatiepasses. De geleerde varianten voegen 4.080 theta-parameters toe. De bevroren variant is een vertakking van de geleerde run, geen vierde onafhankelijke training vanaf epoch 1.

## Selectieregels en figuren

De twee hoofdvergelijkingen blijven de reeds gebruikte regels:

1. **Epoch 100:** dezelfde vooraf vastgelegde eindepoch voor alle varianten.
2. **Globaal beste E4M3, epochs 1–100:** per variant de eerste epoch met strikt maximale E4M3-testaccuracy; alle drie precisies worden op diezelfde epoch weergegeven.

Aanvullend wordt voor alle vier varianten dezelfde E4M3-selectie binnen **51–100** toegepast. Dit venster maakt de fase na de freeze afzonderlijk vergelijkbaar. Het vervangt de globale selectie niet.

Iedere selectie heeft een tabel en een accuracyfiguur. Daarnaast toont de notebook de volledige accuracycurves van alle vier varianten, met een apart paneel per precisie. Voor de bevroren variant bestaat de curve uit de werkelijk gedeelde geleerde epochs 1–50 en de werkelijk uitgevoerde bevroren epochs 51–100.

## Resultaten op epoch 100

| Variant | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| Vast ×32 | 94,7065% | 94,8525% | 94,9140% |
| Vast ×64 | 94,9293% | 94,8986% | 94,7910% |
| Geleerd tot 100 | 94,9332% | 94,1764% | 93,9498% |
| Geleerd tot 50, daarna vast | 94,7757% | 94,7142% | 94,5260% |

Vastzetten na 50 verbetert de lagere precisies tegenover blijven doorleren. Vast ×64 scoort op epoch 100 echter bij alle drie precisies hoger dan de bevroren variant: respectievelijk 0,1537, 0,1844 en 0,2651 procentpunt, oftewel 40, 48 en 69 extra juiste voorspellingen.

De volledig geleerde variant scoort bij E4M3 slechts één testbeeld hoger dan vast ×64, met tegelijk 188 en 219 minder juiste voorspellingen bij E2M3 en E2M1. Dit levert in deze proef geen overtuigende gezamenlijke meerwaarde op.

Vast ×32 en vast ×64 vormen een afweging. Op epoch 100 is ×64 0,2228 procentpunt hoger bij E4M3 en 0,0461 bij E2M3; ×32 is 0,1229 procentpunt hoger bij E2M1.

## Globale E4M3-selectie

| Variant | Epoch | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|---:|
| Vast ×32 | 55 | 94,8448% | 95,0061% | 94,7757% |
| Vast ×64 | 42 | 95,0868% | 93,8422% | 93,9229% |
| Geleerd tot 100 | 34 | 95,0138% | 93,1930% | 93,5925% |
| Geleerd tot 50, daarna vast | 34 | 95,0138% | 93,1930% | 93,5925% |

De twee geleerde varianten selecteren hier exact hetzelfde checkpoint 34 uit hun gemeenschappelijke voorgeschiedenis. Die gelijke resultaten zijn geen afzonderlijke meting van het effect van de freeze.

De aanvullende selectie binnen 51–100 kiest respectievelijk epochs 55, 61, 51 en 51. Ook bij dit gemeenschappelijke venster scoort vast ×64 in alle drie precisies hoger dan beide geleerde varianten. Het ontbreken van een voordeel tegenover ×64 komt dus niet uitsluitend door de globale selectie vóór de freeze.

## Interpretatie en kosten

De huidige methode om per blok tussen ×32 en ×64 te leren levert in deze vergelijking nog geen accuracyvoordeel tegenover de eenvoudige vaste factoren. De freeze-proef blijft informatief: zij toont dat het vastzetten van de keuzes de gekozen leerprocedure verbetert tegenover steeds blijven doorleren. Dit zegt niet dat alle methoden voor geleerde schalen ongunstig zijn of dat ×32/×64 theoretisch optimaal zijn.

De geleerde keuze vraagt één keuzebit per MX-blok: voor deze 4.080 blokken is dat 510 bytes in de compacte representatie. Een uniforme vaste factor vraagt geen afzonderlijke keuze per blok. Tijdens de training gebruikt de leerroute daarnaast theta, gradients en optimizerstaat. In de uitgevoerde freeze-run worden theta-gradients en Adam-momenten nog berekend bij theta-LR nul; een geoptimaliseerde besparing is niet gemeten.

## Herkomst en verificatie

De bestanden in `data` zijn ongewijzigde kopieën van de oorspronkelijke resultaten. De vaste-factorreferenties waren al opgenomen in het eerder geleverde freeze100-pakket. De dynamische en bevroren resultaten komen uit de werkelijke feedback van de voltooide freeze100-run. `SOURCE_MANIFEST.json` legt de oorspronkelijke runnamen, bronarchieven en bestandshashes vast.

De notebook controleert de hashes, de volledige epochreeks, de drie precisies, de overeenstemming tussen accuracies en juiste-aantallen, de geselecteerde checkpoints en de gedeelde voorgeschiedenis. De rapporten in `local_validation` bevatten de onafhankelijke bronvergelijking en controle van de notebook op deze werkelijke gegevens.

Deze vergelijking gebruikt één seed en één workload. De oorspronkelijke CUDA-backend garandeert geen bitexacte herhaling. Checkpointselectie gebeurt op de officiële testset en is geen onafhankelijke validatieschatting. Voor ×32 zijn alle metrics en instellingen beschikbaar, maar is de volledige oorspronkelijke `history.json` niet opgenomen in de bewaarde referentie; de per-epoch sample-orderhashes van die run zijn hier dus niet opnieuw onafhankelijk gecontroleerd.

Trainingsduur is geen hoofdvergelijking in dit pakket. Een voortzetting van 45 epochs mag niet met een volledige 100-epochrun worden vergeleken, en variaties in laptopbelasting kunnen tijden beïnvloeden. Er is geen nieuwe training of checkpoint-forward nodig om de getoonde vergelijking te reproduceren.
