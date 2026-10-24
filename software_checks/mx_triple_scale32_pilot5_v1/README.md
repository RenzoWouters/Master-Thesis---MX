# Gezamenlijke MX-training met schaal ×32 — pilot van vijf epochs

Deze proef vergelijkt een vaste lagere schaal van **32 × de E4M3-bronschaal** met de bestaande ×64-referentie. Ze start opnieuw met dezelfde seed en modelinitialisatie. De drie gewichtsmodi zijn E4M3, E2M3 en E2M1; activaties zijn A32 en de updatemaster, gradients en Adam-momenten zijn FP32.

## Plaatsen en uitvoeren

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. Het script moet daarna hier staan:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_scale32_pilot5_v1\run_train.py
```

Gebruik de bestaande `.venv-train`. Het pakket installeert of vernieuwt geen dependencies en hergebruikt de aanwezige SVHN train- en testbestanden.

Voer eerst de technische controle uit:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale32_pilot5_v1\run_train.py" check
```

Bij `EINDSTATUS: PASS` start je de vijf epochs:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale32_pilot5_v1\run_train.py" run
```

Als de dataset op een andere plaats staat, voeg je `--data-root "PAD_NAAR_SVHN"` toe. Alleen wanneer de bestanden ontbreken, kan expliciet `--download` worden toegevoegd. Er wordt geen SVHN-extra gebruikt.

## Resultaten en notebook

Elke opdracht maakt een eigen nieuwe map onder `mx_runs\mx_triple_scale32_pilot5`. De training bewaart alle vijf epochcheckpoints, het beste gezamenlijke checkpoint op basis van E4M3, het laatste checkpoint, compacte MX-exports en een `feedback_bundle.zip`.

Open na de run `MX_triple_scale32_pilot5_resultaten.ipynb` **in de resultatenmap**. De runner vult het resultatenpad in. De notebook is onuitgevoerd en analyseert alleen opgeslagen resultaten; `SAVE_FIGURES` staat standaard op `False`.

De notebook toont onder meer de accuracycurves van de drie modi, de accuracies op epoch 5 en op het gezamenlijke E4M3-beste pilotcheckpoint, en de vergelijking met **alleen de eerste vijf epochs** van de bestaande ×64-training. De ongewijzigde referentiebestanden staan in `scale64_reference`; het globale beste ×64-checkpoint van epoch 42 wordt niet als pilotcheckpoint gebruikt. Er worden geen epoch-100-resultaten voor de nieuwe ×32-proef gesuggereerd.

De eerste vijf epochs van de bestaande ×64-run duurden samen ongeveer 5,1 minuten, inclusief de gemeten epochtraining, evaluatie en checkpointwrites. Dat is een indicatie voor dezelfde laptopomstandigheden, geen gemeten duur van ×32. Opstart, controles, dataload en export komen erbij.

Deel na afloop de `feedback_bundle.zip` uit de map van **de training**. De grote checkpoints blijven op de laptop en zitten niet in de feedbackbundel.

## Stoppen en hervatten

Eén keer `Ctrl+C` vraagt een stop na de lopende volledige epoch. Een tweede keer stopt direct; alleen het laatste volledig geschreven checkpoint is hervatbaar. Ook een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap vraagt een stop na de lopende epoch.

Hervat met het checkpoint van een volledig afgewerkte epoch, bijvoorbeeld:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale32_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Het checkpoint bewaart optimizer, scheduler en RNG-toestand. De oorspronkelijke CUDA-backend heeft geen bitexacte hervatgarantie. Deze pilot accepteert uitsluitend zijn eigen ×32-contract en stopt uiterlijk op epoch 5. Een ×64-checkpoint of een afgewerkte vijf-epochpilot wordt niet als startpunt ingeladen.

## Afbakening

De experimentele wijziging is uitsluitend de lagere exponentoffset **+6 → +5**. Er zijn geen geleerde schalen. De E4M3-bron wordt nog steeds eenmaal per MX-laag en minibatch gecodeerd; de lagere precisies worden daarvan met bittruncatie afgeleid. Het volledige 100-epoch-LR-schema blijft geconfigureerd met dalingen na 50/75/90; de pilot comprimeert dit schema niet naar vijf epochs.

Een lage accuracy is een onderzoeksuitkomst en veroorzaakt op zichzelf geen technische FAIL. De vijf epochs controleren werking, vroege training en looptijd; ze bewijzen niet welke factor na een volledige training optimaal is. Zie `PROTOCOL.md`, `CODEWIJZIGINGEN.md` en `local_validation/VALIDATION.md` voor de exacte keuzes en controles.
