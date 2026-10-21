# Dedicated E2M1/A32 - honderd epochs

Een ontbrekende referentie, geen nieuwe truncatiemethode: een afzonderlijk CNN-8-model
leert uitsluitend directe E2M1-gewichten, A32 en FP32-masters. Geen E2M3-teacher,
distillatie, warme start, extra-data of truncatie. Zie PROTOCOL.md.

## 1. Plaatsen en check
Pak de inhoud uit in een nieuwe map, met run_train.py rechtstreeks daarbinnen:
C:\Users\renzo\master-thesis-mx\software_checks\mx_e2m1_dedicated_reference100_v1

PowerShell:

    Set-Location "C:\Users\renzo\master-thesis-mx"
    & ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e2m1_dedicated_reference100_v1\run_train.py" check

Check bevat alleen synthetische tests en enkele updates, geen SVHN-training.
Geen nieuwe afhankelijkheden nodig. Bij FAIL stoppen en feedback_bundle.zip uit de
nieuwe check-map sturen; geen pakketten of toleranties zelf aanpassen.

## 2. Volledige training
Na EINDSTATUS: PASS:

    & ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e2m1_dedicated_reference100_v1\run_train.py" run --stop-after-epoch 100

Default mode mx betekent in DIT pakket alleen MXFP4_E2M1. De interne upstream selector32
houdt activaties en de ene actieve BN-bank op A32/BN32; het is GEEN W32-gewichtspad.
Alle honderd epochs lopen achter elkaar vanaf nieuwe initialisatie. Geen resume gebruiken
voor een nieuwe referentie. Data/svhn/train_32x32.mat en test_32x32.mat worden hergebruikt;
extra_32x32.mat wordt niet geopend of gewijzigd. --download is alleen nodig bij ontbrekende
train/testbestanden. Deze runner zal bestaande corrupte databestanden niet overschrijven.

573 minibatches per epoch en 57300 optimizerupdates na honderd epochs.
LR: .001 in epochs1-50; .0001 in51-75; .00001 in76-90; .000001 in91-100.
De werkelijke E2M1-looptijd is nog niet bekend; de runner geeft vanaf epoch2 een raming.
Laat de laptop aangesloten, voorkom slaapstand en draai niet tegelijk een andere training.

## 3. Resultaten
Nieuwe map onder mx_runs/mx_e2m1_dedicated_reference100, eindigend op _mx_run.
Aan het einde verwacht: status PASS, completed_epoch100, optimizer_steps57300,
training_forward_passes57300, full_training_completed true.
Alle epochs + best_checkpoint.pt + last_checkpoint.pt met checksums blijven behouden.
Best = eerste strikte hoogste E2M1-testaccuracy; de testset stuurt die selectie.
Per rol een direct E2M1-bronbestand mx_source_best/last.pt. Geen hoge bron of truncatie.
Open MX_E2M1_dedicated_resultaten.ipynb UIT DE RUNMAP, kies .venv-train en Run All.
Deel feedback_bundle.zip uit die _mx_run-map. De bundel bevat rapporten, geen checkpoints.

## 4. Toevoegen aan het meetingoverzicht (pas NA voltooide training)
Het script gebruikt alleen JSON/CSV en Python-standaardbibliotheek. Het weigert incomplete,
synthetische of anders geconfigureerde resultaten. Het maakt altijd een NIEUWE overzichtsmap.
Vervang de drie voorbeeldpaden hieronder door de echte runmap en uitgepakte overzichtsmap:

    $Run = "C:\Users\renzo\master-thesis-mx\mx_runs\mx_e2m1_dedicated_reference100\<jouw_sessie>_mx_run"
    $Overzicht = "C:\pad\naar\MX_onderzoeksoverzicht_meeting_v1"
    $Nieuw = "C:\pad\naar\MX_onderzoeksoverzicht_met_E2M1"
    & ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e2m1_dedicated_reference100_v1\voeg_toe_aan_overzicht.py" --run-dir $Run --overview-dir $Overzicht --output-dir $Nieuw

Open daarna MX_onderzoeksoverzicht.ipynb UIT DE NIEUWE MAP en Run All. Er is geen actuele
HTML totdat die nieuwe notebook opnieuw is uitgevoerd/geexporteerd. De oorspronkelijke
notebook, HTML, figuren en brongegevens worden niet gewijzigd.

Toegevoegd: dedicated rij in hoofdresultaten, expliciete vergelijking van vier E2M1-uitvoeringen,
eindepoch versus eindepoch, best apart, curve in lage-modusfiguur, punt in accuracy/opslagfiguur,
looptijd. Sectie6 is verduidelijkt: het zijn opslagstrategieen, geen vier trainingen.
De omvang van de code/schaaltensors is een berekening; geen GPU-geheugen-/energiemeting.
Alle behouden BN-banken en FP32-uitzonderingen tellen mee, ook bij dedicated.
Er wordt geen vooraf bepaald voordeel/nadeel aangenomen; de resultaten beslissen.

## 5. Noodhervatten (alleen als deze nieuwe run is onderbroken)
Een Ctrl+C vraagt stoppen na de lopende epoch; tweede Ctrl+C stopt direct.
Volledig opgeslagen epochs zijn hervatbaar. Voorbeeld, pas het pad aan:

    & ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e2m1_dedicated_reference100_v1\run_train.py" resume --checkpoint "C:\...\checkpoints\epoch_00050.pt" --stop-after-epoch 100

Alleen checkpoints van ditzelfde pakket/contract/omgeving, geen oude E2M3- of dual-run.
Na stoppen staat STOPPED_AT_EPOCH_BOUNDARY, niet voltooide training. Een hervatting schrijft
in een nieuwe map en controleert dat haar invoercheckpoint ongewijzigd blijft. Bewaar de
oude runmap. Geen mid-epoch- of bitexacte CUDA-hervatgarantie.

## 6. Validatie
Zie local_validation/VALIDATION.md. Synthetische CPU-mini-epochs zijn softwaretests en
bevatten geen onderzoeksaccuracies of bruikbare GPU-looptijdschattingen.
