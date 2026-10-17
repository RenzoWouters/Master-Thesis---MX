# Eerste MX-bouwstap: alleen gewichten, E2M3, A32

Dit pakket introduceert geen nieuwe SVHN-training. Het controleert de codec en
past MXFP6 E2M3 toe op de EFFECTIEVE W32-gewichten van de bestaande referentie.
Activaties gebruiken expliciet het bestaande A32/ReLU-pad; geen MX-activaties.
Het is een eerste configuratie, niet een bewezen optimale formaatkeuze.

## Eerste uitvoering op de laptop

Pak de inhoud uit als:
`C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_step1_v1`.
Gebruik dezelfde `.venv-train`. Geen nieuwe pakketten of compiler nodig.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_step1_v1\run_mx.py" check
```

`check` test op CUDA met synthetische getallen en een willekeurig model.
Geen dataset, geen optimizerstappen, geen wijziging van opgeslagen gewichten.
Ga pas verder na `EINDSTATUS: PASS`. Bij een pip-conflict niet automatisch upgraden.

```powershell
$ReferenceRun = "C:\Users\renzo\master-thesis-mx\reference_runs\anyprecision_svhn_trainonly100\20261016_165710_766662_run"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_step1_v1\run_mx.py" evaluate --reference-run $ReferenceRun
```

Dit is het referentiepad uit de werkelijk aangeleverde feedbackbundel. Als de
volledige resultatenmap verplaatst is, pas alleen deze padvariabele aan. Geef
de map op met run_report.json, best_checkpoint.pt EN last_checkpoint.pt met
hun .sha256-bestanden. Dit laadt beide checkpoints READ-ONLY, geen resume.

Vier evaluatiepaden per checkpoint op dezelfde volledige testset:
1. oorspronkelijke W32/A32;
2. nieuwe wrapper met MX uit (controle van de wijziging zelf);
3. MXFP6 E2M3/A32 via de nieuwe quantizer;
4. hetzelfde MX-model na herladen van opgeslagen elementcodes en schalen.

Beste en laatste checkpoint blijven bepaald door de vorige referentie. Geen
nieuwe selectie op MX-accuracy. Geen BatchNorm-herkalibratie of finetuning.
Deze proef meet direct-cast-gevoeligheid, geen QAT- of truncatiebestendigheid.

## Resultaten

Nieuwe map onder `mx_runs\mx_weights_step1`, nooit onder de referentie:
- run_report.json, metrics.csv, layer_metrics.csv;
- mx_source_best.pt en mx_source_last.pt, met checksums;
- predictions_best.pt / predictions_last.pt voor latere gepaarde analyse;
- MX_weights_resultaten.ipynb (alleen opgeslagen gegevens lezen);
- feedback_bundle.zip (zonder checkpoints, gewichten of voorspellingstensors).

De export bevat alleen gepakte 6-bitcodes en E8M0-schalen voor de zes
kwantiseerbare convoluties. De eerste/laatste laag en alle oorspronkelijke
BatchNorm-banken blijven als aanvullende FP32-staat behouden; alleen BN32
wordt gebruikt. Er staan GEEN mastergewichten van de zes MX-lagen in de export.
De herladen emulator decodeert naar een FP32-rekenbuffer. Die RAM-behoefte mag
niet verward worden met de compact opgeslagen payload of met hardwarewinst.

`PASS` vereist dezelfde aantallen correcte voorspellingen voor de opnieuw
geevalueerde W32/A32-referentie als in het checkpoint en overeenstemming van de
controle-/exportpaden. Kleine GPU-verschillen worden afzonderlijk gelogd. De
vooraf vastgelegde logits-tolerantie voor controle/export is rtol=1e-5,
atol=1e-6, MET gelijke voorspellingen; exact-gelijkheid wordt ook gerapporteerd.
Bij `REVIEW_REQUIRED` blijven de bestanden bewaard, maar interpreteren we een
MX-accuracyverschil nog niet als onderzoeksresultaat voordat de controle klopt.

## Leesvolgorde om de code te begrijpen

1. CODEWIJZIGINGEN.md: mapping van referentiecode naar nieuwe code.
2. mx_adapter.py: de feitelijke ingreep op de zes lagen.
3. mx_format.py: elementcodering, blokschalen, afronden, opslag en STE.
4. run_mx.py: aparte evaluatie en logging; GEEN trainingslus.
5. test_mx.py: controle van codewaarden, randen, gradients en export.

De bestanden smoke_support.py, svhn_support.py, resume_core.py en de
meegeleverde auteurs-ZIP zijn hergebruikt zonder wijziging. De bestaande
compatibiliteitspatch blijft daarvan gescheiden. Nieuwe onderzoekswijzigingen
staan uitsluitend in mx_format.py, mx_adapter.py en run_mx.py.

De referentierunner blijft intact. Een latere QAT-runner kan dezelfde MX-module
gebruiken. Die vraagt eerst een expliciete keuze van modi, loss/distillatie,
initialisatie en een bijpassende no-MX-controle; dit pakket start die niet.
