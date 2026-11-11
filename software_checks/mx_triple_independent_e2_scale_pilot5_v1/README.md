# Onafhankelijke E2-schaal — eerste vijf epochs

Per groep van 32 gewichten blijft de bestaande dynamische E4M3-bronschaal behouden. E2M3 en E2M1 krijgen samen één eigen, absoluut geleerde E8M0-schaal. Hun elementcodes worden nog steeds uit dezelfde E4M3-broncodes getrunceerd. Activaties blijven A32, mastergewichten FP32.

De eerste schaalwaarden zijn exact gelijk aan de eerdere vaste ×64-route. Dat geldt uitsluitend bij initialisatie: daarna wordt de E2-schaal niet meer uit de actuele E4-schaal berekend. Er zijn geen tien relatieve keuzefactoren meer.

## Uitvoeren

Pak de ZIP uit zodat deze map bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_independent_e2_scale_pilot5_v1`

Gebruik de bestaande trainingsomgeving; installeer of upgrade niets voor deze proef.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_pilot5_v1\run_train.py" check
```

Na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_pilot5_v1\run_train.py" run
```

`check` controleert codec, schaalonafhankelijkheid, gradients, drie trainingsmodi en compacte exports op synthetische gegevens. Er wordt geen SVHN-training uitgevoerd. `run` traint vijf volledige epochs vanaf nieuwe initialisatie. Het pakket kan niet voorbij epoch 5 trainen. De leersnelheid en mijlpalen van het oorspronkelijke 100-epochrecept worden niet samengedrukt.

Bestaande SVHN-bestanden worden gecontroleerd en hergebruikt. Downloaden gebeurt alleen met `--download`. Ruwe datasets en eerdere resultaten worden niet overschreven. Iedere opdracht krijgt een nieuwe map onder `mx_runs\mx_triple_independent_e2_scale_pilot5`.

## Notebook en feedback

De resultatenmap bevat `MX_triple_independent_e2_scale_pilot5_resultaten.ipynb` en `notebook_support.py`. Open de notebook vanuit die map en voer alle cellen uit. Dezelfde onuitgevoerde notebook staat al in het pakket; daar kan `RUN_DIR` naar een resultatenmap worden gezet.

De notebook toont accuracycurves, de laatste gemeten epoch en het gemeenschappelijke E4-best-checkpoint, historische vergelijkingen, absolute E2-exponenten, geobserveerde schaalverhoudingen, schaalveranderingen, opslag en gemeten looptijd. De accuracyas zoomt in maar omvat alle getoonde meetpunten. Vergelijkingen gebruiken alleen gelijk genummerde, beschikbare epochs. De historische referenties zijn ongewijzigde onderzoeksgegevens, geen nieuwe meetresultaten.

De notebook leest CSV/JSON en start geen training. Zij opent geen SVHN-data of FP32-checkpoints. Deel na afloop `feedback_bundle.zip` uit de nieuwe resultatenmap. Checkpoints, compacte modelbestanden en dataset blijven lokaal; de bundel bevat rapporten, geschiedenis, diagnostiek, notebook en broncode.

## Onderbreken en hervatten

Eén keer Ctrl+C vraagt een stop na de lopende epoch. Een tweede Ctrl+C stopt direct; alleen een volledig opgeslagen epoch is hervatbaar. Hervatten herstelt gewichten, absolute schaalparameters, Adam, scheduler en RNG. De E2-schalen worden daarbij niet opnieuw op ×64 gezet.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00002.pt"
```

Alleen een checkpoint uit exact dit pakket en dezelfde vastgelegde omgeving is hervatbaar. Het oorspronkelijke cuDNN-beleid blijft behouden; bitexacte CUDA-hervatting is niet gegarandeerd.

## Betekenis van de proef

De onderzoeksvraag is of het netwerk zich kan aanpassen wanneer de lagere schaal zelfstandig leert en niet meer automatisch met de E4-amax-schaal meeschuift. Alleen de schalen zijn onafhankelijk: de lagere elementcodes blijven van de E4-bron afhangen.

De vijf epochs beoordelen werking, eerste ontwikkeling en looptijd. Lage accuracy of weinig harde schaalwissels zijn geldige waarnemingen en op zichzelf geen softwarefout. Eén proef met één seed bewijst geen algemene accuracywinst of optimale schaal. Een volledige training is een afzonderlijke vervolgstap.

Zie `PROTOCOL.md` voor exacte bewerkingen en kosten. `local_validation/` beschrijft de lokale softwarecontrole en haar beperkingen.
