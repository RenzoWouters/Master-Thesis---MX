# Truncatiebewuste tweemodus-MX-pilot, vijf volledige epochs

Plaats de inhoud in:
C:\Users\renzo\master-thesis-mx\software_checks\mx_truncation_aware_pilot5_v1

Dit is een NIEUWE training vanaf de vastgelegde seed. Geen finetuning van het
vorige getrainde model. Verander niets in eerdere code- of resultatenmappen.
Bestaande .venv-train behouden; geen nieuwe pakketten of downloads vereist.

## PowerShell

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_truncation_aware_pilot5_v1\run_train.py" check
```

Alleen na EINDSTATUS: PASS:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_truncation_aware_pilot5_v1\run_train.py" run --stop-after-epoch 5
```

Check: synthetische gegevens, inclusief kleine optimizerupdates; geen SVHN.
Run: alle 73.257 trainbeelden, 26.032 testbeelden, geen extra. De .mat-bestanden
worden uit data\svhn hergebruikt. Alleen als train/test werkelijk ontbreken kan
--download toegevoegd worden. Bestaande of corrupte bestanden niet overschrijven.

Verwacht bij complete run: vijf epochs; 573 minibatches/epoch; 2.865 updates;
5.730 trainingsforwards; twintig epoch/phase/mode-metricrijen. Alle modi A32.
De runner stopt na vijf epochs, niet automatisch na honderd. LR blijft 0,001.

## Resultaten

mx_runs\mx_truncation_aware_pilot5\<tijdstip>_joint_run

Open MX_truncation_aware_pilot5_resultaten.ipynb UIT DE RUNMAP, kies .venv-train.
Run All. De notebook leest alleen bestanden. Pandas en matplotlib zijn dezelfde
reeds geinstalleerde notebookafhankelijkheden. BASELINE_DIR is optioneel; vul
het pad naar de gewone directe tweemodusrun in voor gelijk-epochvergelijking.
Beste checkpoint op E2M3-testaccuracy en laatste checkpoint blijven bewaard.
Iedere testloss is harde CE; lage trainingsloss is soft-target CE.

mx_source_E2M3_best.pt / ..._last.pt: enige nodige MX-gewichtsbron per rol,
met beide BN-banken en overige FP32-laagstaat. Geen zes FP32-mastergewichten.
mx_derived_E2M1_best.pt / ..._last.pt: expliciet afgeleide controlekopie,
geen noodzakelijke tweede bron. Laden met load_exported_model(..., bit=4)
kan ook rechtstreeks uit de hoge bron. De bevroren emulator wordt PER MODUS
opgebouwd; een andere modus vraagt reconstructie uit dezelfde hoge bron,
niet alleen een wissel van BN-selector. De GPU rekent op FP32-decodebuffers.

Trainingscheckpoints bevatten WEL masters, Adam, BN en RNG voor herstel;
een compact MX-inferentiebestand is geen trainingscheckpoint.

Stuur feedback_bundle.zip uit _joint_run. _joint_check bevat alleen de zelftest.
Nooit een mislukte controle omzeilen door tolerantie of afhankelijkheden aan te passen.

## Gecontroleerd onderbreken en noodherstel

Een Ctrl+C vraagt stoppen NA de lopende epoch; een tweede stopt direct.
Alleen volledig opgeslagen epochs zijn hervatbaar. Gebruik uitsluitend een
checkpoint uit DIT pakket, nooit een oude kwantisatiepilot of reference100.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_truncation_aware_pilot5_v1\run_train.py" resume --stop-after-epoch 5 --checkpoint "VOLLEDIG_PAD_NAAR_DEZE_RUN\checkpoints\epoch_00002.pt"
```

De hervatrun krijgt een nieuwe map. Oude contracten, corrupte hashes en reeds
bestaande checkpointnamen worden geweigerd. Geen mid-epoch-restart of garantie
van bitexacte CUDA-hervatting. Zie PROTOCOL.md en local_validation/VALIDATION.md.
