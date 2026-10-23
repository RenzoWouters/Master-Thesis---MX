# Gezamenlijke directe MX-proef — E4M3, E2M3 en E2M1

Dit pakket breidt de eerdere directe tweeprecisieproef uit met E4M3.
Alle drie modi worden rechtstreeks uit dezelfde actuele FP32-mastergewichten
via het bestaande G(W) gekwantiseerd. Deze proef gebruikt geen mantissa- of
exponenttruncatie. Activaties blijven A32, de updatemasters blijven FP32.
De runner stopt na vijf volledige SVHN train-only epochs.

De nieuwe dedicated E4M3-referentie is geslaagd: beste checkpoint epoch 53
met 94,7987% testaccuracy; epoch 100 met 94,6835%. De sessie duurde 50m37s.
Die resultaten blijven afzonderlijk bewaard en worden niet als initialisatie
van deze nieuwe gezamenlijke proef geladen.

## 1. Uitpakken en de korte controle uitvoeren

Pak de map uit de ZIP uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_triple_direct_pilot5_v1\run_train.py`

Gebruik de bestaande `.venv-train`; geen nieuwe laptopafhankelijkheden of
upgrades zijn nodig.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_triple_direct_pilot5_v1\run_train.py" check
```

Deze controle gebruikt uitsluitend synthetische afbeeldingen op de GPU. Ze
controleert de drie codecs, directe conversie, de drie BN-banken, de recursieve
distillatie en de gezamenlijke optimizerupdate. De gecontroleerde
trainingslus wordt ook vergeleken met een onafhankelijke handmatige lus.
Er wordt geen SVHN-training gestart.

## 2. Bij CHECK PASS: de vijf epochs starten

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_triple_direct_pilot5_v1\run_train.py" run
```

Elke minibatch heeft drie gewichtsmodi: E4M3 leert met harde cross-entropy
van de labels; E2M3 leert met soft-target CE van de losgekoppelde E4M3-uitvoer;
E2M1 leert vervolgens van de losgekoppelde E2M3-uitvoer. De gradients worden
opgeteld, waarna precies één Adam-update volgt. De gewichten van iedere modus
blijven onafhankelijk rechtstreeks uit dezelfde FP32-bron bepaald.

De run begint met een nieuwe initialisatie, seed 20261009. Hij gebruikt de
volledige train-split en officiële testset, geen extra-split. Het bestaande
honderd-epoch-LR-schema blijft geconfigureerd, maar dit pakket voert maximaal
vijf epochs uit. Voor de nieuwe proef worden de looptijd en een projectie voor
een toekomstige run van honderd epochs gemeten; de dedicated looptijd is geen
gemeten tijd voor de gezamenlijke proef.

## 3. Resultaten en notebook

Nieuwe resultaten staan onder:

`mx_runs\mx_weights_triple_direct_pilot5\<tijdstip>_joint_run\`

Open in die map `MX_triple_direct_pilot5_resultaten.ipynb`, kies de bestaande
`.venv-train`-kernel en voer alle cellen uit. De runner vult de resultaatmap in.
De notebook heeft onder andere:

- train- en testaccuracy voor alle drie precisies;
- harde test-CE en afzonderlijk gelabelde trainingsobjectieven;
- een accuracyfiguur voor het laatste pilotcheckpoint en voor het beste
  checkpoint volgens de E4M3-accuracy, telkens met alle drie precisies;
- een expliciete melding dat epoch 100 nog niet is gemeten;
- gemeten tijden, een herkenbare honderd-epochprojectie, GPU-geheugen en
  technische controles.

De drie scores bij het beste checkpoint komen altijd uit dezelfde epoch.
De notebook combineert geen afzonderlijke maxima per precisie. Het pakket
bevat alleen de uitvoerbare, lege template, geen extra uitgevoerde notebook.
Figuren opslaan maakt een nieuwe analysemap; ruwe resultaten en checkpoints
blijven behouden. De notebook voert geen training uit.

Stuur na afloop de `feedback_bundle.zip` uit de nieuwe `_joint_run`-map.
Als de korte check faalt, stuur dan de bundel uit `_joint_check` en start nog
geen training. Bewaar de volledige runmap met checkpoints en directe exports.
De feedbackbundel bevat rapporten en analysecode, geen datasets of checkpoints.

## 4. Stoppen en hervatten

Eén `Ctrl+C` vraagt een stop na de lopende epoch, inclusief evaluatie en
checkpoint. Een tweede `Ctrl+C` onderbreekt direct. Een leeg bestand
`STOP_AFTER_EPOCH` in de actieve runmap vraagt eveneens een stop op de
epochgrens.

Een volledig opgeslagen checkpoint uit ditzelfde pakket is hervatbaar:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_triple_direct_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00002.pt"
```

Hervatten schrijft een nieuwe runmap en gaat maximaal tot epoch 5. Behoud de
oorspronkelijke runmap met eerdere checkpoints en epochrapporten. Checkpoints
van andere experimenten worden geweigerd. De bestaande cuDNN-instellingen
garanderen geen bitexacte CUDA-hervatting.

## Documentatie

`PROTOCOL.md` beschrijft de exacte conversie, trainingswijze en selectie.
`CODEWIJZIGINGEN.md` beschrijft wat is hergebruikt en gewijzigd.
`local_validation/VALIDATION.md` noemt de daadwerkelijk uitgevoerde controles.
