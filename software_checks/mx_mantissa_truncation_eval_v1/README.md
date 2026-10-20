# E2M3 -> E2M1: directe kwantisatie tegenover echte mantissatruncatie

**Alleen evaluatie.** Geen nieuwe training, geen optimizerupdates, geen nieuwe
checkpointselectie en geen BatchNorm-herkalibratie. De bestaande honderd-epoch-
training en de ruwe data blijven ongewijzigd.

## Installatie en uitvoering

Pak de inhoud uit in een nieuwe map:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_mantissa_truncation_eval_v1
```

`run_eval.py` moet direct in die map staan. Gebruik dezelfde `.venv-train`.
Geen nieuwe dependencies of download; pandas/matplotlib zijn al voor de
notebooks toegevoegd. Check wijzigt of installeert niets in de omgeving.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_truncation_eval_v1\run_eval.py" check
```

Na `EINDSTATUS: PASS`:

```powershell
$ReferenceRun = "C:\Users\renzo\master-thesis-mx\mx_runs\mx_weights_dual_reference100\20261019_192538_709626_joint_run"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_truncation_eval_v1\run_eval.py" evaluate --reference-run $ReferenceRun
```

Dit pad komt uit de laatste feedbackbundel. Is de oorspronkelijke map verplaatst,
pas uitsluitend de padvariabele aan. Het pakket heeft bewust geen train/run/resume-
commando, geen downloadoptie en geen optie voor een gedeeltelijke testset.

## Welke bestanden zijn nodig?

De echte oorspronkelijke `_joint_run`-map bevat:
- run_report.json, selected_metrics.csv, epoch_metrics.csv, checkpoint_selection.json;
- mx_direct_MXFP6_E2M3_best.pt en mx_direct_MXFP6_E2M3_last.pt;
- mx_direct_MXFP4_E2M1_best.pt en mx_direct_MXFP4_E2M1_last.pt;
- de vier bijbehorende .pt.sha256-bestanden.

De FP32-trainingscheckpoints zelf worden NIET ingelezen. De vier directe exports
zijn bij het voltooien van de vorige training al gemaakt en gecontroleerd op
exacte reconstructie van hun gekwantiseerde gewichtstensors. De E2M3-export
bevat ook de getrainde lage BN-bank; die is voldoende om C zelfstandig af te leiden.
De eerder directe E2M1-export is alleen de vergelijkingsgroep A, geen input van C.

Alleen `data\svhn\test_32x32.mat` wordt gelezen en vergeleken met de oorspronkelijke
MD5/SHA256 en labels. De train- en extra-bestanden worden niet geopend.

## Vier paden per vast checkpoint

H: opgeslagen E2M3, hoge BN-bank (context).
A: opgeslagen, eerder rechtstreeks afgeronde E2M1, lage BN-bank.
B: opgeslagen E2M3-elementen afronden naar E2M1, zonder nieuwe schaal, lage BN-bank.
C: twee mantissabits verwijderen uit opgeslagen E2M3-codes, lage BN-bank.

A32, de niet-gekwantiseerde lagen, de auxiliaire modelstaat en de schalen blijven
hetzelfde. A/B/C gebruiken dezelfde lage BN. Het bronformaat is E2M3, groepsgrootte
32, E8M0; de bron wordt nooit met de lage precisie overschreven.

## Resultaten

Nieuwe map onder `mx_runs\mx_mantissa_truncation_eval`, eindigend op `_evaluate`.
Open daar `MX_mantissa_truncatie_resultaten.ipynb`, selecteer `.venv-train`, Run All.
De notebook leest alleen JSON/CSV/NPZ.

metrics.csv bevat acht rijen: vier paden voor beide bestaande checkpointrollen.
paired_metrics.csv bevat correcte/foute overgangen, wijzigingen in voorspellingen,
accuracyverschillen in procentpunten en logitverschillen.
weight_pair_metrics.csv en layer_code_metrics.csv bevatten de verschillen per laag.
historical_reproduction.csv vergelijkt A/H met de oude trainingsscores.
De nieuwe B/C-exportbestanden worden echt opgeslagen, herladen en op de testset
gemeten. De zes FP32-mastergewichten zitten daar niet in; FP32-uitzonderingen en
BN-staat wel. De emulator gebruikt gedecodeerde FP32-rekenbuffers.

**Stuur feedback_bundle.zip uit de nieuwe _evaluate-map.** De bundel bevat nu ook
NPZ-bestanden met labels, indices, logits en voorspellingen per beeld; daardoor
is hij groter dan eerdere bundels. Er zitten geen ruwe beelden, mastergewichten
of optimizercheckpoints in. De B/C-exportbestanden blijven lokaal buiten de bundel.

Een slechte accuracy van C is geen technische FAIL: dat is juist een mogelijke
onderzoeksuitkomst. Schaal-/code-/BN-inconsistenties geven wel een FAIL. Verschillen
met oude directe testscores worden expliciet gewaarschuwd, nooit weggecorrigeerd.
Bij een fout niets aanpassen of upgraden; deel de bijbehorende feedbackbundel.

Zie CODEWIJZIGINGEN.md, PROTOCOL.md en local_validation/VALIDATION.md voor afbakening.
