# Vaste offsets na epoch 50 — vijf nieuwe epochs

Deze proef vertrekt van het bestaande checkpoint **na epoch 50** van de geleerde ×32/×64-training. Ze traint uitsluitend **epochs 51–55** met de toen gekozen offset per MX-blok vastgezet. De vergelijking gebruikt epochs 51–55 uit de al afgeronde training. Er wordt één nieuwe variant uitgevoerd.

## 1. Uitpakken

Pak de zip uit in:

```text
C:\Users\renzo\master-thesis-mx\software_checks
```

Het nieuwe startbestand moet daarna hier staan:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_freeze_offsets_e50_pilot5_v1\run_train.py
```

Behoud de volledige uitgepakte map, inclusief `reference_runtime`, `parent_reference` en de notebook. De runtime uit de vorige proef is meegeleverd en wordt op zijn oorspronkelijke checksums gecontroleerd. Het startbestand van deze nieuwe proef beperkt de voortzetting tot maximaal epoch 55.

## 2. Eerst de technische controle

Gebruik de bestaande trainingsomgeving. Er worden geen pakketten geïnstalleerd of bijgewerkt.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_freeze_offsets_e50_pilot5_v1\run_train.py" check
```

De controle verifieert de oorspronkelijke codecs en trainingsroute, de identiteit van het echte oudercheckpoint, het herstellen van alle trainingsstaat en de nieuwe freezecontroles. Hiervoor wordt geen SVHN-training uitgevoerd. De volledige databestanden worden bij `run` opnieuw gecontroleerd.

Het standaard oudercheckpoint is:

```text
mx_runs\mx_triple_learned_scale32_64_reference100\20261025_150724_515132_joint_run\checkpoints\epoch_00050.pt
```

Het bijhorende bestand `epoch_00050.pt.sha256` moet ernaast staan. De code controleert ook het model, protocol, de oorspronkelijke geschiedenis, de LR en de Adam-/schedulertellers. Een willekeurig ander checkpoint na vijftig epochs wordt geweigerd.

Als dezelfde bestanden verplaatst zijn, kun je hun nieuwe pad meegeven met `--parent-checkpoint "VOLLEDIG_PAD\epoch_00050.pt"`. De inhoud moet dezelfde blijven. Voer de opdrachten vanuit de projectmap uit zodat de bestaande data onder `data\svhn` gevonden worden.

## 3. De vijf nieuwe epochs uitvoeren

Na `EINDSTATUS: PASS` van de controle:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_freeze_offsets_e50_pilot5_v1\run_train.py" run
```

`run` herstelt eerst het checkpoint na epoch 50 en traint dan 51, 52, 53, 54 en 55. De eerdere vijftig epochs worden niet opnieuw uitgevoerd. Op basis van je eerdere run is ongeveer **zes minuten voor de vijf epochs**, plus controle, inladen en export, een bruikbare verwachting. Dit is een schatting; de werkelijke duur wordt opgeslagen.

## Wat precies vaststaat en verder leert

Per blok geldt:

```text
shared_exponent_E2M3(t) = shared_exponent_E4M3(t) + offset_na_epoch50
shared_exponent_E2M1(t) = shared_exponent_E2M3(t)
```

- Het blok behoudt de na epoch 50 geleerde offset 5 (×32) of 6 (×64).
- De FP32-mastergewichten en de actieve BatchNorm-banken blijven verder trainen.
- E4M3 wordt telkens opnieuw uit de actuele gewichten bepaald. De E4M3-bronschaal kan daardoor veranderen.
- E2M3 wordt met dezelfde exponentbitbewerking uit de E4M3-bron afgeleid. E2M1 volgt via dezelfde mantissatruncatie uit E2M3.
- Activaties blijven A32; masters, gradients en Adam-momenten blijven FP32.
- De gewichts-LR is in epochs 51–55 **0,0001**, volgens het oorspronkelijke 100-epochschema. Dat schema wordt niet herstart.
- De drie oorspronkelijke passes blijven gelijk: E4M3 hard CE, E2M3 met de losgekoppelde E4M3-teacher, E2M1 met de losgekoppelde E2M3-teacher; één Adamupdate per minibatch.

**Technische uitvoering van de freeze:** na volledig herstel zetten we alleen de LR van de theta-groep op nul en passen we de overeenkomstige scheduleradministratie aan. Theta zelf blijft exact op zijn oude FP32-waarde staan. De oorspronkelijke schaalgradient en Adam-momenten worden voor deze diagnose nog wel berekend. Deze proef onderzoekt dus het effect van vaste keuzes op accuracy; ze meet geen geoptimaliseerde besparing van schaaltraining.

Na iedere minibatch worden zowel alle theta-waarden als alle harde keuzes met het startcheckpoint vergeleken. Beide moeten exact gelijk blijven. Ook de volgorde van alle trainings- en testbeelden wordt per epoch met de opgeslagen historische run vergeleken. Bitexacte herhaalbaarheid van de CUDA-berekeningen is hiermee niet gegarandeerd.

## Resultaten en notebook

Iedere opdracht maakt een nieuwe resultatenmap onder:

```text
mx_runs\mx_triple_freeze_offsets_e50_pilot5
```

De nieuwe run bevat onder meer:

- `feedback_bundle.zip`: deel deze na de proef;
- `MX_triple_freeze_offsets_e50_pilot5_resultaten.ipynb`: lege, direct uitvoerbare notebook met het juiste runpad;
- `epoch_metrics.csv` en `comparison_with_historical.csv`;
- `freeze_summary.json`, `scale_epoch_metrics.csv` en `scale_block_state.csv`;
- `history.json`, `lineage.json` en `run_report.json`;
- checkpoints per nieuwe epoch, plus kopieën voor beste en laatste checkpoint;
- compacte E4M3-bronexports en daaruit afgeleide E2M3/E2M1-exports met de vaste blokkeuzes;
- `parent_reference`: de historische vergelijking, met herkomst en checksums.

De notebook bevat:

1. Accuracycurves voor alle drie precisies vanaf het gemeenschappelijke startpunt na epoch 50.
2. Een staafdiagram met alle accuracies op **epoch 55** voor beide varianten.
3. Een staafdiagram met alle drie accuracies op het per variant gekozen **E4M3-beste checkpoint binnen epochs 51–55**.
4. Accuracyverschillen ten opzichte van de bestaande dynamische voortzetting per epoch.
5. De verdeling ×32/×64 en het aantal wisselingen; bij de nieuwe variant moeten de wisselingen nul zijn.
6. Verliescurves en gemeten epochduren.
7. De controles op herkomst, datasetdekking, gewichtsupdates en vaste offsets.

Bij de selectie binnen epochs 51–55 komen alle drie precisies uit hetzelfde E4M3-geselecteerde checkpoint. Epoch 50 is het gedeelde startpunt en doet niet mee aan die selectie. Het globale beste checkpoint van de oude 100-epochrun wordt hiervoor niet gebruikt.

De notebook traint niets, laadt geen `.pt`-bestanden en wijzigt geen invoerresultaten. Met `SAVE_FIGURES = True` bewaart hij figuren in een nieuwe analysemap. Gebruik de notebook in de resultatenmap; de lege kopie in het pakket is het sjabloon.

## Stoppen en hervatten

Voor deze korte proef is één ononderbroken uitvoering het eenvoudigst. Indien nodig: één keer **Ctrl+C** vraagt een stop na de huidige volledige epoch. Een tweede Ctrl+C stopt onmiddellijk; alleen volledig opgeslagen checkpoints zijn dan bruikbaar. Een gecontroleerde gedeeltelijke run krijgt `STOPPED_AT_EPOCH_BOUNDARY`, pas na epoch 55 volgt `PASS` voor de volledige proef.

Voorbeeld om na epoch 52 te hervatten:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_freeze_offsets_e50_pilot5_v1\run_train.py" resume --checkpoint ".\mx_runs\mx_triple_freeze_offsets_e50_pilot5\JOUW_RESULTATENMAP\checkpoints\epoch_00052.pt"
```

Bewaar de volledige gedeeltelijke resultatenmap, inclusief `epoch_summaries` en alle checksums. Hervatten maakt een nieuwe resultatenmap en behoudt de eerdere vijf-epochgeschiedenis. De oorspronkelijke epoch-50-checkpoint blijft eveneens nodig voor de gecontroleerde herkomst.

## Interpretatie

Epoch 55 is het vooraf gekozen hoofdvergelijkingspunt. Een verbetering van de bevroren variant ondersteunt de gedachte dat vaste keuzes in deze fase kunnen helpen. Eén korte voortzetting met een historische controle bewijst nog geen algemene superioriteit, optimaal stopmoment of exacte oorzaak. De E4M3-selectie gebruikt zoals voorheen de officiële testset en is geen onafhankelijke validatieschatting.

De technische controles voor dit pakket staan in `local_validation`. Ze gebruiken kleine synthetische CPU-fixtures. De daadwerkelijke CUDA-/SVHN-proef gebeurt pas wanneer je bovenstaande `run`-opdracht uitvoert.
