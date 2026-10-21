# MX-bron naar inferentie, zonder training

Dit is de extra exportcontrole voor de VOLLEDIGE truncatiebewuste run. Het is
niet nogmaals de oude kwantisatie-truncatiekloofproef en geen nieuwe training.

## Gebruik op Windows / PowerShell

Pak de volledige inhoud uit in een nieuwe map:
C:\Users\renzo\master-thesis-mx\software_checks\mx_truncation_source_only_eval_v1

run_eval.py moet rechtstreeks in die map staan. Gebruik .venv-train; geen updates
of nieuwe installatie. Voer uit vanuit C:\Users\renzo\master-thesis-mx.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_truncation_source_only_eval_v1\run_eval.py" check
```

Alleen na EINDSTATUS: PASS:

```powershell
$ReferenceRun = "C:\Users\renzo\master-thesis-mx\mx_runs\mx_truncation_aware_reference100\20261021_153559_999222_joint_run"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_truncation_source_only_eval_v1\run_eval.py" evaluate --reference-run $ReferenceRun
```

Het pad is uit feedback_bundle(20261021-144630).zip gelezen. Alleen aanpassen
wanneer de gebruiker die map heeft verplaatst. De originele hoge exportbestanden
moeten nog in deze map staan, samen met hun .sha256-bestanden en JSON/CSV-rapporten.
Een feedbackbundel alleen bevat geen gewichten en is dus niet voldoende.

Nodige modelbestanden:
- mx_source_E2M3_best.pt (in deze run epoch 52), plus .pt.sha256
- mx_source_E2M3_last.pt (epoch 100), plus .pt.sha256

De best_checkpoint.pt, last_checkpoint.pt en mx_derived_E2M1_*.pt worden NIET
geopend, gehasht of vereist. Laat ze gewoon ongewijzigd staan. Alleen het
bestaande data/svhn/test_32x32.mat wordt gelezen, niet train of extra. Geen download.

## Wat gebeurt er?

Per checkpointrol worden een kopie van de hoge bron en noodzakelijke JSON-feiten
in een nieuwe map geplaatst. Een afzonderlijk Python-proces krijgt alleen die
bron plus de testsetlocatie. Het reconstrueert E2M3 en E2M1 uit dezelfde hoge bron
met de bestaande, ongewijzigde decoders en bitshift. E2M3 kiest BN-bank32, E2M1
BN-bank4; in beide gevallen blijven activaties A32. Schalen wijzigen niet.

Het model wordt bevroren en de volledige testset wordt doorlopen. Geen optimizer,
backward, directe lage encoder, gewichtsvoorbewerking of BN-kalibratie. De hoge
bron bevat al effectieve G(W)-gewichten: tanh/normalisatie niet opnieuw toepassen.

De oorspronkelijke .pt-bestanden en onderzoeksresultaten blijven ongewijzigd.
Een interrupt stopt de evaluatie; een volgende poging krijgt een nieuwe map.

## Resultaten

Nieuwe map: mx_runs/mx_truncation_source_only_eval/<timestamp>_evaluate

- run_report.json: technische status en APARTE reproduction_status.
- metrics.csv: nieuw gemeten testaccuracy, correcte aantallen en CE.
- historical_reproduction.csv: nieuwe scores tegenover de bestaande trainingsscores.
- worker_best/ en worker_last/: nieuw proces, broncontroles en per_image_outputs.npz.
- MX_bron_only_evaluatie.ipynb: leest alleen deze rapporten; geen training of GPU nodig.
- feedback_bundle.zip: rapporten en voorspellingen, geen grote dataset of modelbestanden.

Open de notebook UIT de _evaluate-map, kies .venv-train, Run All.
Stuur daarna feedback_bundle.zip uit de _evaluate-map, niet uit _check.

PASS betekent dat de technische uitvoering klopt. MATCH_COUNTS_CE_CLOSE betekent
ook dezelfde top1/top5-aantallen en CE binnen abs1e-6/rel1e-5. REVIEW_DIFFERENCE
is een expliciete waarschuwing, geen reden om zelf toleranties/gewichten/BN te
wijzigen. De oude training bevat geen individuele logits; gelijke accuracy is
niet hetzelfde als aantoonbaar gelijke voorspellingen of logits per oud testbeeld.

Deze vier evaluaties zijn geen nieuwe seeds of trainingsresultaten. FP32-buffers
voor de emulator en FP32-uitzonderingslagen/BN blijven nodig; geen native
MX-kernel, energiemeting of complete FP32-vrije trainingsstaat geclaimd.
