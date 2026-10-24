# Honderd epochs — gezamenlijke MX-training met vaste schaal ×64

Dit pakket vervolgt de geslaagde vijf-epochpilot met dezelfde route B:
E4M3 als bron, naïeve exponenttruncatie naar E2M3 met schaal ×64 en daarna
mantissatruncatie naar E2M1. Het start een verse training van honderd epochs
met dezelfde oorspronkelijke initialisatie, seed en trainingsinstellingen.

De echte pilot eindigde met E4M3 **92,85495%**, E2M3 **91,85618%** en
E2M1 **91,66411%** testaccuracy. Epoch 5 was ook het beste E4M3-checkpoint.
De vergelijking met de eerdere directe en naïeve pilots staat in
`pilot_analysis/RESULTATEN.md`, met de oorspronkelijke rapporten en tabellen.
De volledige honderd-epochtraining is bij het maken van dit pakket nog
niet uitgevoerd.

## 1. Uitpakken en checken

Pak de map `mx_triple_scale64_reference100_v1` uit de ZIP uit onder
`C:\Users\renzo\master-thesis-mx\software_checks`. Daarna bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_scale64_reference100_v1\run_train.py`

Gebruik dezelfde trainingsomgeving als voor de pilot:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_reference100_v1\run_train.py" check
```

De check valideert op synthetische input de codecs, truncatieketen,
schaalcorrectie, gradients, broncache, BN, Adam en opslag/herladen. Hij start
geen SVHN-training en installeert geen pakketten. Bij FAIL: deel eerst de
`feedback_bundle.zip` uit de nieuwe checkmap.

## 2. Na PASS: de volledige proef starten

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_reference100_v1\run_train.py" run
```

De standaarddatamap is `data\svhn`. Als de data elders staan, voeg je
`--data-root "PAD"` toe. De bestaande train-split van 73.257 en test-split
van 26.032 beelden worden gebruikt, zonder extra-split. De eerdere optionele
`--download` blijft beschikbaar voor ontbrekende data.

`run` start vanaf de oorspronkelijke verse modelconstructie met seed
20261009. De pilot blijft een afzonderlijke proef. Het eigen resumecommando
kan een onderbroken run van dit honderd-epochpakket hervatten.

### Verwachte duur

De echte pilot duurde **4 minuten en 57 seconden** voor de volledige sessie;
de vijf epochs zelf samen **4 minuten en 50 seconden**. De bijgeleverde
projectie is ongeveer **1 uur en 36 minuten** voor een nieuwe honderd-epochrun.
Projecteren met de snelste en langzaamste gemeten niet-eerste epochs geeft
ongeveer **1 uur en 29 minuten tot 1 uur en 45 minuten**. Dit is een
looptijdprojectie, geen betrouwbaarheidsinterval of gemeten honderd-epochtijd.
Thermiek en achtergrondbelasting kunnen de werkelijke duur beïnvloeden.

## 3. Dezelfde numerieke route

| Modus | Herkomst | Schaal |
|---|---|---|
| E4M3 | Bestaande nearest-even encoding van de actuele G(W) | E4M3-bronschaal S |
| E2M3 | Hoogste twee exponentbits van dezelfde E4M3-code; teken en mantissa behouden | 64S, E8M0-broncode +6 |
| E2M1 | E2M3-code twee bits naar rechts schuiven | Dezelfde 64S als E2M3 |

De schaaloffset blijft vast op +6. De actuele bronschaal kan na een
masterupdate veranderen. Activaties blijven A32; updatemasters, gradients
en Adam-momenten blijven FP32. De bestaande identity STE over de volledige
route blijft behouden, zonder extra factor 64 in backward. De drie
BN-banken leren normaal mee tijdens training en zijn bevroren tijdens eval.

Elke minibatch heeft drie trainpasses met de bestaande recursieve distillatie
en één Adam-update op de gesommeerde gradients. Er zijn 573 minibatches per
epoch: een voltooide run bevat **57.300 updates en 171.900 trainingspasses**.

Het learning-rateschema blijft hetzelfde:

| Epochs | Learning rate |
|---|---:|
| 1–50 | 0,001 |
| 51–75 | 0,0001 |
| 76–90 | 0,00001 |
| 91–100 | 0,000001 |

## 4. Resultaten en notebook

Iedere start schrijft in een nieuwe map onder:

`mx_runs\mx_triple_scale64_reference100\<tijdstip>_joint_run\`

Open daar `MX_triple_scale64_reference100_resultaten.ipynb`. De resultaatmap
wordt automatisch ingevuld. De notebook wordt leeg meegeleverd; voer de
cellen zelf uit. De round-trip CSV-parsercorrectie is geïntegreerd.

De notebook toont de learning curves, de correct gelabelde train- en
testlosses, klasseverdelingen, bron-/schaalcontroles, tijden en GPU-geheugen.
De accuracyfiguur bevat beide gevraagde checkpointgroepen:

1. **Epoch 100.**
2. **Het checkpoint met de hoogste E4M3-testaccuracy.**

Elke groep toont alle drie precisies uit hetzelfde checkpoint. Bij een
onderbroken run toont de notebook de laatste beschikbare epoch en E4M3-best,
met een expliciete melding dat epoch 100 nog niet is gemeten.

Figuren verschijnen standaard in de notebook. Zet `SAVE_FIGURES = True`
voor PNG- en PDF-uitvoer in een nieuwe analysemap. De notebook leest de
opgeslagen rapporten en voert geen training of modelinferentie uit.

Na afloop: bewaar de volledige nieuwe runmap en deel haar
`feedback_bundle.zip`. Dataset en checkpointtensors blijven lokaal;
de bundel bevat de meetgegevens, controles en de lege notebook met helper.

## Stoppen en hervatten

Eén keer `Ctrl+C` laat de huidige epoch afwerken en volledig opslaan. Een
tweede keer onderbreekt onmiddellijk. Een bestand `STOP_AFTER_EPOCH` in de
actieve runmap vraagt eveneens om een stop op de volgende epochgrens.

Hervat vanaf een volledig checkpoint van deze honderd-epochproef:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Gebruik bijvoorbeeld `last_checkpoint.pt` uit een gecontroleerd gestopte
run of `checkpoints\epoch_00050.pt`, met zijn checksum. Hervatten schrijft
in een nieuwe map en bewaart de eerdere geschiedenis. Checkpoints uit de
pilot, de naïeve proef of de directe proef hebben een ander contract en
worden geweigerd. Het oorspronkelijke CUDA-backendbeleid geeft geen
bitexacte hervatgarantie.

## Documentatie en softwarevalidatie

`PROTOCOL.md` legt de bewerkingen en vergelijking vast.
`CODEWIJZIGINGEN.md` beschrijft de wijzigingen ten opzichte van de pilot;
de daadwerkelijke wijzigingen staan ook als patches onder `provenance`.
`local_validation/VALIDATION.md` beschrijft de begrensde synthetische
CPU-controles van dit nieuwe pakket. Die controles zijn onderscheiden van
de echte SVHN-pilot en van de nog uit te voeren honderd-epochtraining.

Een technische PASS vereist geldige, eindige en consistente uitvoering.
Accuracy, klasseverdeling en gewichtsveranderingen zijn te analyseren
uitkomsten. De instellingen worden tijdens de run niet op basis van
accuracy aangepast.
