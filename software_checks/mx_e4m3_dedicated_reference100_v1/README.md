# Dedicated E4M3/A32 — referentie van 100 epochs

Dit pakket volgt op de geslaagde E4M3-pilot van vijf epochs van 22 oktober 2026.
De werkende v2-codec, CNN-8, SVHN train-only, A32, FP32-updatemasters,
Adam en het learning-rate-schema zijn behouden. De referentie begint opnieuw
vanaf initialisatie met dezelfde seed; de pilot blijft afzonderlijk bewaard.

De pilot haalde 93,1392% testaccuracy op epoch 5 (ook het beste checkpoint).
De gemeten projectie voor een nieuwe honderd-epochrun is circa 41 minuten
53 seconden, met een op de warme epochtijden gebaseerd bereik van 41–44 minuten.
Dit is geen garantie of betrouwbaarheidsinterval. Er zijn nog geen echte
honderd-epochresultaten in dit pakket.

## 1. Uitpakken

Pak de pakketmap uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_e4m3_dedicated_reference100_v1\run_train.py`

De oude pilotmappen en resultaten blijven staan. Gebruik de bestaande
`.venv-train`; er zijn geen nieuwe laptopafhankelijkheden of upgrades nodig.

## 2. De dedicated referentie starten

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e4m3_dedicated_reference100_v1\run_train.py" run
```

Deze opdracht traint uitsluitend E4M3/A32 tot epoch 100. Ze laadt geen
pilotcheckpoint. De bestaande train/testbestanden onder `data\svhn` worden
gecontroleerd en hergebruikt. De `extra`-split wordt niet gebruikt.

De codec, adapter en modelcontrole zijn byte-identiek aan de werkende pilot-v2.
`check` blijft beschikbaar als afzonderlijke korte codec-/modelcontrole:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e4m3_dedicated_reference100_v1\run_train.py" check
```

`check` gebruikt synthetische invoer en start geen SVHN-training.
De productie-CLI accepteert uitsluitend eindepoch 100; stopverzoeken blijven
hervatbaar zonder het onderzoeksprotocol te veranderen.

## 3. Resultaten en notebook

De nieuwe runmap staat onder:

`mx_runs\mx_e4m3_dedicated_reference100\<tijdstip>_mx_run\`

Open daarin `MX_E4M3_dedicated_reference100_resultaten.ipynb` met de bestaande
`.venv-train`-kernel en voer alle cellen uit. De runner vult de resultaatmap in.
De notebook toont:

- train/testaccuracy en cross-entropy over de gemeten epochs;
- de accuracy van epoch 100 en van het beste checkpoint volgens E4M3-testaccuracy;
- het laatste beschikbare checkpoint, ook wanneer de run vroeg is gestopt;
- de werkelijk gemeten totale tijd zodra honderd epochs zijn voltooid;
- technische controles en de werkelijk gerapporteerde backendinstellingen.

Een ontbrekende epoch 100 wordt niet ingevuld of voorspeld. Figuren worden in
een nieuwe analysemap geschreven. Checkpoints, CSV's, JSON's en eerdere
figuren worden niet overschreven. De notebook start geen training.

Deel na afloop de `feedback_bundle.zip` uit de nieuwe `_mx_run`-map. Bewaar
ook de volledige runmap met checkpoints en E4M3-bronexports. De feedbackbundel
bevat rapporten en analysecode, geen datasets of trainingscheckpoints.

## 4. De uitgewerkte vijf-epochanalyse

`pilot5_analysis/` bevat een portable, ingevulde notebook en vier figuren op
basis van de werkelijk aangeleverde pilot. De originele bundelbestanden staan
byte-identiek in `pilot5_analysis/raw/`. Dit is een analyse van de vijf epochs;
er zijn geen verzonnen honderd-epochscores toegevoegd.

## 5. Stoppen en hervatten

Eén `Ctrl+C` vraagt een stop na de lopende epoch, inclusief evaluatie en
checkpoint. Een tweede `Ctrl+C` onderbreekt direct. Een leeg bestand
`STOP_AFTER_EPOCH` in de actieve runmap vraagt eveneens een stop na de epoch.
Alleen een volledig opgeslagen epochcheckpoint is hervatbaar.

Voorbeeld met een werkelijk opgeslagen checkpoint uit DIT honderd-epochpakket:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e4m3_dedicated_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00040.pt"
```

Hervatten schrijft een nieuwe runmap, herstelt het bestaande honderd-epochcontract
en gaat door tot epoch 100. Behoud ook de oorspronkelijke runmap met checkpoints
en epochrapporten. Een pilotcheckpoint hoort bij een ander contract en wordt
geweigerd. De bestaande cuDNN-instellingen garanderen geen bitexacte CUDA-hervatting.

## Documentatie

`PROTOCOL.md` beschrijft de onderzoeksinstellingen en selectie.
`CODEWIJZIGINGEN.md` beschrijft de beperkte wijziging ten opzichte van de pilot.
`local_validation/VALIDATION.md` noemt de werkelijk uitgevoerde controles.
