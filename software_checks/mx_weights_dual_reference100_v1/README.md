# Honderd epochs: een gezamenlijk MX-model, E2M3 en E2M1, beide A32

Een afzonderlijke volledige run vanaf initialisatie. Geen overname van de
pilot of vroegere referentie; train-only, geen extra-data, geen truncatie.

## Plaatsen en starten in PowerShell

Pak de inhoud uit met run_train.py rechtstreeks in:

    C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_dual_reference100_v1

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_dual_reference100_v1\run_train.py" check
```

Ga alleen verder na EINDSTATUS: PASS. De verbeterde v1.1-zelftest gebruikt
kunstmatige gegevens en een afzonderlijk check-proces; er wordt niet op SVHN
getraind. pip check moet slagen; geen automatische upgrades.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_dual_reference100_v1\run_train.py" run --stop-after-epoch 100
```

Alle honderd epochs lopen achter elkaar. Dit is EEN netwerk met twee passes
per batch, niet twee aparte runs. De oude proefbestanden blijven ongewijzigd.
De CLI weigert een ander eindpunt. De run initialiseert met dezelfde seed
20261009, zonder --checkpoint of resume. Train en test komen uit data\svhn;
geen --download nodig bij de bestaande data. Extra wordt niet geopend/gehasht.

## Leerregel

E2M3: hard-label cross-entropy. E2M1: oorspronkelijke soft-target CE op de
losgekoppelde E2M3-voorspelling (T=1). Som van gradients, niet delen door twee;
een Adam-update per minibatch. Twee BN-banken, steeds A32. Beide gewichtsmodi
worden rechtstreeks uit G(W) gevormd; E2M1 komt niet uit getrunceerde E2M3.
FP32-mastergewichten, gradients en Adam-momenten blijven behouden.

De LR daalt NA epochs 50, 75 en 90. Geen gewijzigd leerschema of AMP.
Oorspronkelijk cuDNN-benchmarkbeleid blijft behouden in run/resume, ook na
de tijdelijk deterministische controle in een apart check-proces.

## Tijd, rapporten en notebook

De eerder gemeten vijf-epochproef gaf circa 2 uur 03 minuten voor 100 epochs.
De rangeprojectie was circa 1 uur 33 tot 2 uur 14, geen betrouwbaarheidsinterval.
Geen garantie dat de volledige training onder twee uur blijft. Houd de laptop
op netvoeding, voorkom slaapstand en vermijd andere GPU-training tegelijk.

Resultaten komen in een NIEUWE sessiemap onder:

    C:\Users\renzo\master-thesis-mx\mx_runs\mx_weights_dual_reference100

De map eindigt op _joint_run; de voorafgaande checkmap op _joint_check.
Bij volledige afronding staan in run_report.json:

    status: PASS
    completed_epoch: 100
    optimizer_steps: 57300
    training_forward_passes: 114600
    full_training_completed: true

Per epoch worden alle rapporten en checkpoints met checksums bewaard.
Het beste checkpoint is het eerste strikte maximum van E2M3/A32-test-top1;
beide modi worden VAN DATZELFDE checkpoint getoond. Daarnaast beide op epoch100.
Testgestuurde selectie is geen onafhankelijke testmeting; geen per-modus cherry-pick.

Open MX_dual_reference100_resultaten.ipynb uit de runmap met .venv-train,
dan Run All. Het notebook leest alleen JSON/CSV, geen modelcalls of updates.
De aanwezige pandas/matplotlib-installatie kan worden hergebruikt. Volledige
looptijd wordt gemeten; niet verwarren met een extrapolatie bij onderbreking.

Deel feedback_bundle.zip UIT DE _joint_run-MAP. Die bundel bevat rapporten,
niet alle checkpoints/logits/datasets. Bewaar de gehele oorspronkelijke map
met alle honderd checkpoints, best/last en de vier aparte MX-exports.
Exports zijn directe quantisatie per modus, geen truncatieklare enkelvoudige bron.

## Onderbreking alleen voor nood

Een enkele Ctrl+C vraagt een stop na de huidige epoch. Een tweede Ctrl+C
stopt onmiddellijk. Het bestand STOP_AFTER_EPOCH in de huidige runmap doet
hetzelfde als de eerste Ctrl+C. Alleen volledig opgeslagen epochcheckpoints
zijn hervatbaar, met checksum en ongewijzigd contract/omgeving.

Voor hervatting van DEZE volledige run, niet van de vijf-epochproef:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_dual_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00050.pt" --stop-after-epoch 100
```

Gebruik het werkelijk laatste volledige checkpoint. Het originele resultaatpad
blijft nodig voor sidecars en voor best/last uit eerdere epochs. De hervatting
schrijft een nieuwe map. Geen overschrijven, mid-epoch-herstel of bitexacte
CUDA-garantie. Een onderbroken run wordt niet als voltooid gerapporteerd.

## Afbakening en lokale tests

local_validation/VALIDATION.md beschrijft de echte uitgevoerde softwaretests.
Synthetische mini-epochs zijn geen SVHN-resultaten of GPU-tijdsmetingen.
CPU-validatie vervangt niet de laptopcheck. Codeherkomst en de minimale
wijzigingen staan in CODEWIJZIGINGEN.md, FULLRUN_CHANGE_MANIFEST.json en
REUSE_MANIFEST.json; numerieke kernel/STE/test_dual blijven gelijk aan v1.1.
