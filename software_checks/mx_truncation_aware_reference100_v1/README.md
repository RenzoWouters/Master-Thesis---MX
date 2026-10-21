# Honderd epochs truncatiebewuste MX-training

Nieuwe map (inhoud van ZIP rechtstreeks hierin plaatsen):
C:\Users\renzo\master-thesis-mx\software_checks\mx_truncation_aware_reference100_v1

E2M3 -> dezelfde broncodes >>2 -> E2M1; A32, FP32-updatemasters en Adamstaat.
Nieuwe initialisatie. Geen finetuning van pilot of directe kwantisatiereferentie.
Gebruik de bestaande .venv-train, geen nieuwe pakketten of downloads vereist.

## Controle en run in PowerShell
```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_truncation_aware_reference100_v1\run_train.py" check
```
Alleen na EINDSTATUS: PASS:
```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_truncation_aware_reference100_v1\run_train.py" run --stop-after-epoch 100
```
Check: kunstmatige getallen, modellen en enkele optimizerupdates, geen SVHN.
Run: alle100 epochs automatisch na elkaar. CNN-8, SVHN train73257/test26032,
geen extra. Batch128,573 trainingsminibatches/epoch. Adam lr0.001, weight decay0,
factor0.1 na50/75/90 (lagere LR vanaf51/76/91). E2M3 hard-label CE; E2M1
losgekoppelde teacher-soft-CE, som van gradients, een update per minibatch.
Alle overige keuzes zijn gelijk aan de geslaagde truncatiebewuste pilot.

De data worden hergebruikt uit data\svhn. Het extra-bestand wordt niet gelezen.
Alleen als train/test ontbreken, voeg --download toe. Bestaande/corrupte data
worden nooit overschreven. Laat de laptop aangesloten en voorkom slaapstand.
De pilotprojectie is circa65 minuten; geen garantie of echte honderd-epochmeting.

## Resultaten
mx_runs\mx_truncation_aware_reference100\<tijdstip>_joint_run

Verwacht bij volledige afronding in run_report.json:
status=PASS; completed_epoch=100; optimizer_steps=57300;
training_forward_passes=114600; full_training_completed=true.
400 metricrijen:100 epochs x train/test x2 modi.

Open MX_truncation_aware_reference100_resultaten.ipynb UIT DE RUNMAP en gebruik
.venv-train als kernel. Run All. Alleen bestanden lezen, geen model-/traincalls.
Bij BASELINE_DIR kan optioneel de oorspronkelijke directe100-runmap staan;
dan worden dezelfde epochs en initialisatie-/samplevolgordehashes vergeleken.
De lage directe baseline is niet dezelfde uitvoering als haar achteraf
getrunceerde variant. Geen claim dat herstel van de truncatiekloof gegarandeerd is.

Alle100 checkpoints plus best_checkpoint.pt en last_checkpoint.pt behouden.
Best = eerste strikte maximum van E2M3-testaccuracy; beide modi uit datzelfde
checkpoint. Last = epoch100. Testselectie is niet onafhankelijk. Geen selectie
van verschillende epochs per precisie. Geen BN-herkalibratie na training.

mx_source_E2M3_best.pt / ..._last.pt: een hoge MX-bron per checkpointrol,
met beide BN-banken en overige FP32-laagstaat; geen zes FP32-mastergewichten.
mx_derived_E2M1_best.pt / ..._last.pt: afgeleid uit de herladen hoge bron,
geen benodigde tweede opgeslagen master. load_exported_model(..., bit=4)
kan rechtstreeks uit de hoge bron. FP32-decodebuffers dienen de emulatie.
Exportgewichten worden gecontroleerd; geen extra volledige testsetpass na export.

Trainingscheckpoints bewaren WEL FP32-master/Adam/BN/RNG voor herstel.
De export is geen hervatbaar trainingscheckpoint. Sla alle oorspronkelijke
runmappen op; de feedbackbundel bevat geen gewichten of samplevolgorde-arrays.
Stuur feedback_bundle.zip uit _joint_run, niet uit _joint_check.

## Gecontroleerd stoppen en noodherstel
Een Ctrl+C vraagt stoppen NA de lopende epoch; tweede Ctrl+C stopt direct.
Alleen afgeronde, opgeslagen epochs zijn hervatbaar. Laat ook de oude runmap
staan voor oudere checkpoints en timing-sidecars. Resume krijgt een nieuwe map.
Gebruik alleen een checkpoint uit DIT pakket, niet de vijf-epochpilot.
```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_truncation_aware_reference100_v1\run_train.py" resume --stop-after-epoch 100 --checkpoint "VOLLEDIG_PAD_NAAR_DEZE_RUN\checkpoints\epoch_00050.pt"
```
Geen mid-epochherstel of bitexacte CUDA-garantie. Model-/Adam/RNG-staat wordt
hersteld op dezelfde machine/omgeving met identiek contract. Bron-/databytes
blijven behouden. Een zachte stop rapporteert niet valselijk honderd epochs.

## Afbakening en tests
Dit is eigen truncatiebewuste weight-only-MX, geen exacte paperreproductie,
geen native-kernelbenchmark, geen volledig MX-trainingsgeheugen, geen CL-claim.
PROTOCOL.md legt methodekeuzes vast; CODEWIJZIGINGEN.md de wrapperverschillen;
COMPATIBILITY.md de ongewijzigde omgeving. local_validation/VALIDATION.md
documenteert daadwerkelijk uitgevoerde tests en beperkingen. Geen zelftest
omzeilen door toleranties te verhogen of zonder overleg pakketten te upgraden.
