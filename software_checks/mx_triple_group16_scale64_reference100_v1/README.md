# Groepsgrootte 16 met vaste ×64 — volledige training van 100 epochs

De vijfepochpilot is technisch geslaagd. Dit pakket voert dezelfde numerieke route uit gedurende **100 epochs**: E4M3 als gedeelde bron, exponenttruncatie naar E2M3 met vaste ×64, en mantissatruncatie naar E2M1 met behoud van de E2M3-schaal. Activaties blijven A32; mastergewichten, gradients en Adam-momenten zijn FP32.

De volledige proef begint opnieuw vanaf de oorspronkelijke initialisatie en seed, zoals bij onze eerdere volledige proeven. De vijfepochpilot blijft een afzonderlijk bewaard experiment. De bestaande volledige B32 ×64-run vormt de referentie; die wordt niet opnieuw getraind.

## Plaatsen en uitvoeren

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. Het script staat daarna hier:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_group16_scale64_reference100_v1\run_train.py
```

Gebruik dezelfde bestaande `.venv-train`; het pakket installeert of vernieuwt geen dependencies. Voer eerst de technische controle uit:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group16_scale64_reference100_v1\run_train.py" check
```

Start na `EINDSTATUS: PASS` de volledige training:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group16_scale64_reference100_v1\run_train.py" run
```

De bestaande bestanden `train_32x32.mat` en `test_32x32.mat` onder `data\svhn` worden hergebruikt. De bestandsnamen verwijzen naar beeldafmetingen; ze veranderen niet door de schaalfactor of groepsgrootte. Een afwijkende locatie kan via `--data-root "PAD_NAAR_SVHN"`; alleen met expliciet `--download` worden ontbrekende bestanden gedownload. SVHN-extra wordt niet gebruikt.

Iedere opdracht maakt een nieuwe map onder `mx_runs\mx_triple_group16_scale64_reference100`. De pilot mat ongeveer 51,7 seconden per stabiele epoch, inclusief training, evaluatie en checkpoint. De projectie voor de nieuwe 100 epochs is **ongeveer 1 uur 26 minuten**, exclusief opstart en eindexport; reken praktisch op ongeveer anderhalf uur bij vergelijkbare laptopomstandigheden. Dit is een schatting, geen reeds gemeten volledige B16-looptijd.

## Stoppen en hervatten

Eén keer `Ctrl+C` vraagt een stop na de huidige volledige epoch. Een tweede keer stopt direct. Een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap vraagt eveneens een stop na de lopende epoch.

Hervat met een checkpoint uit **deze volledige B16-run**:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group16_scale64_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Optimizer, scheduler, RNG en de eerdere epochgeschiedenis worden hersteld. De hervatte sessie krijgt een nieuwe map; bestaande checkpoints blijven behouden. De oorspronkelijke CUDA-backend biedt geen bitexacte hervatgarantie. Checkpoints van de vijfepochpilot of een andere groepsgrootte worden niet als warmstart ingeladen.

## Notebook: epoch 100 én het E4M3-beste checkpoint

Open na afloop `MX_triple_group16_scale64_reference100_resultaten.ipynb` in de resultatenmap. Het resultatenpad wordt door de runner ingevuld. De notebook is onuitgevoerd, heeft lege outputs en leest alleen resultaten. `SAVE_FIGURES=False` is de standaard.

De hoofdfiguren vergelijken **B16 ×64 met B32 ×64**:

- Accuracycurves van de drie precisies.
- Accuracies van alle drie modi op **epoch 100**.
- Accuracies van alle drie modi uit het **gezamenlijke checkpoint geselecteerd door de hoogste E4M3-accuracy**, per training. E2M3/E2M1 worden niet uit afzonderlijke eigen beste epochs gekozen.
- Verliescurves, voorspellingstellingen, quantisatiediagnostiek, looptijd en gemeten GPU-geheugen.
- De expliciete kostenanalyse hieronder, met tabellen per laag en grafieken voor schalen, padding en totale opslag.

Een onderbroken run wordt duidelijk als onvolledig weergegeven; epoch 100 wordt dan niet als gemeten ingevuld. Tussentijdse vergelijkingen worden beperkt tot het gemeenschappelijk afgewerkte epochbereik. De bestaande volledige B32-referentie en de bewaarde B16-pilotgegevens worden bij de resultaten gekopieerd.

De runner bewaart alle epochcheckpoints, best/last-checkpoints en compacte MX-exports. Deel na afloop de `feedback_bundle.zip` uit de resultatenmap van de training. Grote checkpoints en datasetbestanden blijven lokaal en zitten niet in die bundel.

## Kosten: schalen, padding en representatiegrootte

Zonder padding is de overhead van één 8-bit gedeelde schaal `8/B` bits per gewicht. Onze werkelijke indeling groepeert per outputkanaal over de afgevlakte `Cin*Kh*Kw`-rij en vult de staart van iedere rij aan. De resultaten zijn daarom:

| B | Groepen / schalen | Paddingposities | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---:|---:|---:|---:|---:|---:|
| 16 | 8.096 | 64 | 137.632 bytes | 105.248 bytes | 72.864 bytes |
| 32 | 4.080 | 1.088 | 134.640 bytes | 102.000 bytes | 69.360 bytes |
| 64 | 2.160 | 8.768 | 140.400 bytes | 105.840 bytes | 71.280 bytes |

De zes MX-lagen bevatten samen 129.472 echte gewichten. Vergeleken met B32 kost de E4M3-bron bij B16 **2.992 bytes extra (+2,22%)**; voor een afzonderlijke E2M3-export is dit +3.248 bytes (+3,18%) en voor E2M1 +3.504 bytes (+5,05%). De drie exportgroottes worden niet opgeteld alsof drie volledige modellen tegelijk opgeslagen moeten worden: de bron volstaat om de lagere codes af te leiden.

Bij B64 dalen de schaalbytes maar stijgt de padding. Bijvoorbeeld: de grootste laag heeft 800 gewichten per outputkanaal, die bij B64 tot 832 posities worden aangevuld. B64 heeft daardoor in deze indeling meer bronopslag dan B32. De B64-kolom is voorlopig alleen een kostenberekening; er is geen B64-accuracy gemeten.

Dit zijn logisch bitgepakte representaties van de zes MX-lagen, geen volledige checkpointgroottes of gemeten GPU-geheugen. FP32-uitzonderingen, BN-toestand, containers en tijdelijke emulatietensors zijn uitgesloten. Elke actuele export wordt tegen de berekende geometrie gecontroleerd. De bestanden `group_size_costs.json`, `group_size_costs_summary.csv` en `group_size_costs_layers.csv` bevatten de reproduceerbare berekeningen.

## Waarom doorgaan na de pilot?

Op epoch 5 haalde B16 E4M3/E2M3/E2M1 respectievelijk 92,93%, 92,06% en 92,17%. Tegenover B32 op dezelfde epoch zijn de verschillen −0,08, +0,76 en +1,28 procentpunt. Het voordeel voor de lagere precisies verschijnt vooral in de vijfde epoch; het was niet in elke vroege epoch aanwezig. De volledige training onderzoekt of het voordeel standhoudt. Ze veronderstelt geen betere eindaccuracy.

Alle onderzoeksinstellingen en de numerieke kern zijn gelijk aan de geslaagde B16-pilot. Het bestaande LR-schema daalt na epochs 50, 75 en 90. Een volledige run telt 57.300 optimizerupdates en 171.900 trainingspasses. De officiële testset wordt zoals eerder gebruikt voor E4M3-checkpointselectie; de geselecteerde beste score is geen onafhankelijke testschatting.

Zie `pilot_analysis/RESULTATEN.md` voor de pilotanalyse en `PROTOCOL.md`, `CODEWIJZIGINGEN.md` en `local_validation/VALIDATION.md` voor de exacte keuzes en controles.
