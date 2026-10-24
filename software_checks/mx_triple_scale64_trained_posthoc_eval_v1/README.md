# A/B/C na training met exponenttruncatie en schaal ×64

Deze proef evalueert de voltooide **schaal ×64-training** op twee vaste
checkpoints: **epoch 100** en **E4M3-best, epoch 42**. De getrainde gewichten en
batchnormstatistieken blijven bevroren. Er zijn geen trainingepochs of updates.

De eerdere A/B/C-metingen van de directe en naïeve training zijn als
ongewijzigde historische tabellen opgenomen. De notebook zet de nieuwe
metingen ernaast. Die eerdere modellen worden niet opnieuw geëvalueerd.

## Wat wordt vergeleken?

Elke afleiding begint bij dezelfde **opgeslagen E4M3-codes** en de bijbehorende
gedeelde schaal `S`, uit het gekozen checkpoint.

| Route | E2M3-elementcode | Gedeelde schaal | E2M1 |
|---|---|---|---|
| A | Hoogste twee van vier exponentbits, teken en drie mantissabits behouden | `S` | Eigen E2M3-code logisch twee bits naar rechts |
| B | Exact dezelfde elementcode als A | `64*S` | Dezelfde mantissatruncatie, schaal blijft `64*S` |
| C | Lokale E4M3-waarde gedeeld door 64, daarna nearest-even naar E2M3 | `64*S` | Dezelfde mantissatruncatie uit C6 |

E4M3 blijft de opgeslagen bronmodus. De opgeslagen getrainde E2M3- en
E2M1-modi worden ook opnieuw gemeten. Daarmee zijn er negen routes per
checkpoint, dus **18 model-testsetcombinaties** en 24 gepaarde vergelijkingen.

**B is hier de eigen trainingsroute.** Het pakket eist dat de opnieuw afgeleide
B6/B4-codes, schalen en volledige bevroren modelstaat exact overeenkomen met
de opgeslagen lagere modi. Het vergelijkt vervolgens ook hun logits en
voorspellingen op dezelfde testbeelden. Dit is een reconstructiecontrole,
geen extra onafhankelijke trainingsherhaling.

Alle activaties blijven **A32**. De FP32-updatemaster blijft in de oorspronkelijke
trainingscheckpoints behouden; deze evaluatie leest die masters niet in.
Er zijn geen geleerde schalen, nieuwe checkpointselectie of BN-herkalibratie.

## 1. Uitpakken

Pak de map uit de ZIP uit onder `software_checks`, zodat dit bestand bestaat:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_scale64_trained_posthoc_eval_v1\run_eval.py
```

Gebruik dezelfde `.venv-train` als voor de training. Er is geen installatie
of wijziging van die omgeving nodig.

## 2. Eerst de korte controle

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_trained_posthoc_eval_v1\run_eval.py" check
```

`check` gebruikt synthetische codes, modellen en beelden. Het test de
bitbewerkingen, schaaloffset, afrondingscontrole, opslag/herladen en bevroren
reconstructie. Het leest geen SVHN-data of bestaande trainingsmodellen.
Bij `FAIL`: deel de nieuwe `feedback_bundle.zip` uit de checkmap.

## 3. Na CHECK PASS: de twee checkpoints evalueren

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_trained_posthoc_eval_v1\run_eval.py" evaluate `
  --scale64-run ".\mx_runs\mx_triple_scale64_reference100\20261024_151210_665535_joint_run"
```

Het argument wijst naar de **volledige oorspronkelijke runmap**, inclusief
de zes compacte MX-exports en hun `.sha256`-bestanden. Alleen een uitgepakte
feedbackbundel bevat daarvoor onvoldoende bestanden. Heb je de runmap
verplaatst, pas dan alleen het pad aan.

Het pakket controleert dat het om deze afgesproken run met checkpoints 42 en
100 gaat. Rapport, selectie, trainingsprotocol, codeversie, exporthashes,
modelherkomst, dataset en uitvoeringsomgeving moeten overeenkomen.

De bestaande dataset wordt verwacht onder `data\svhn\test_32x32.mat`. Staat
die elders, voeg dan `--data-root "PAD_NAAR_SVHN_MAP"` toe. Alleen de officiële
testset met 26.032 beelden wordt ingelezen; er is geen download.

Iedere start maakt een nieuwe map onder:

```text
mx_runs\mx_triple_scale64_trained_posthoc_eval\<tijdstip>_evaluate\
```

Alle oorspronkelijke runs, checkpoints en resultaten blijven behouden.
`Ctrl+C` bewaart een diagnostische feedbackbundel. Er is geen resumecommando;
opnieuw starten maakt een nieuwe resultaatmap.

## 4. Resultaten voorstellen

Open in de nieuwe resultaatmap:

```text
MX_triple_scale64_trained_posthoc_resultaten.ipynb
```

De runner vult `RUN_DIR` in. Voer alle cellen uit met je bestaande notebookkernel.
Er wordt één **lege, uitvoerbare notebook** meegeleverd. De notebook leest
uitsluitend resultaatbestanden en voert geen model of training uit.

De notebook bevat:

- accuracies van alle negen routes op **epoch 100** en **E4M3-best42**, naast elkaar;
- verschillen A→B en B→C, en vergelijking met de opgeslagen getrainde modus;
- gepaarde fout→correct- en correct→foutveranderingen binnen hetzelfde checkpoint;
- controle van B tegenover de opgeslagen trainingsroute, plus herhaling van de historische accuracies;
- gewichtsafwijkingen, RMS-grootte, nul-/codedistributies en voorspelde klasseverdelingen;
- twee overzichtsfiguren, voor E2M3 en E2M1, met A/B/C op **alle drie trainingsgeschiedenissen**, telkens voor epoch 100 en het eigen E4M3-best checkpoint;
- de oorspronkelijke checkpointselectie, herkomst en integriteitscontroles.

De eerdere directe en naïeve training gebruiken E4M3-best54; de nieuwe
schaal ×64-training gebruikt E4M3-best42. De figuren maken dit onderscheid
zichtbaar. Historische en nieuwe metingen blijven afzonderlijk herkenbaar.
Voorspellingen uit verschillende trainingen worden niet als één gepaarde
proef samengevoegd.

`SAVE_FIGURES = True` bewaart PNG/PDF in een nieuwe analysemap. De ruwe
resultaten blijven ongewijzigd. Bewaar de volledige resultaatmap en deel
vervolgens de nieuwe **`feedback_bundle.zip`** uit de `_evaluate`-map.
De bundel bevat metrics, beide geordende per-beeldbestanden, auditgegevens,
historische vergelijkingstabellen en de notebook; geen modellen of datasets.

## Hoe lezen we de uitkomst?

- **B versus opgeslagen E2M3/E2M1:** controle of we de eigen trainingsroute correct reconstrueren.
- **A versus B:** effect van het terugzetten van de gedeelde schaal van `64*S` naar `S`, met dezelfde elementcodes.
- **B versus C:** effect van een andere elementmapping bij dezelfde schaal `64*S` en dezelfde bevroren BN-bank.
- **Vergelijking over de drie trainingen:** beschrijvend overzicht van de wisselwerking tussen trainingsroute en evaluatieroute.

Een betere of slechtere C-score op een B-getraind checkpoint voorspelt niet
automatisch hoe goed een apart met C getraind netwerk zou worden. Deze proef
zoekt geen nieuwe schaal of checkpoint op basis van de testscores.

Lage accuracy is een geldige onderzoeksuitkomst. `PASS` betekent dat de
afgesproken metingen en technische controles voltooid zijn. Historische
accuracies hoeven onder de oorspronkelijke CUDA-backend niet bitexact te
worden herhaald; eventuele verschillen worden expliciet gerapporteerd.

## Validatie en documentatie

`PROTOCOL.md` beschrijft de exacte bitbewerking, schaal- en checkpointregels.
`CODEWIJZIGINGEN.md` legt hergebruik en wijzigingen vast.
`local_validation/VALIDATION.md` beschrijft de werkelijk uitgevoerde
softwarecontroles en hun grenzen. De echte checkpointbestanden staan op jouw
laptop; de voorbereiding hier start geen SVHN-training of echte evaluatie.
