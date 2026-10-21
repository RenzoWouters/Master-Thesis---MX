# Extra evaluatie, geen nieuwe numerieke methode

Basis: mx_truncation_aware_reference100_v1.zip en de eerdere
mx_mantissa_truncation_eval_v1.zip. REUSE_MANIFEST.json legt de hergebruikhashes vast.

Ongewijzigd: trunc_adapter.py, dual_adapter.py, mx_format.py, mx_e2m1.py,
mx_adapter.py, svhn_support.py, smoke_support.py, resume_core.py, auteurs-ZIP.
De read-only functie prepare_test_only is letterlijk overgenomen uit de vorige
evaluator. Oude run_train.py wordt niet aangeroepen of gewijzigd.

Nieuwe aansturing:
- run_eval.py controleert de bestaande rapporten/selectie, kopieert alleen de hoge
  bronnen naar een nieuwe werkmap, start twee processen en vergelijkt scores.
- source_worker.py krijgt geen oorspronkelijk checkpointpad of lage export; hij
  herlaadt uitsluitend zijn hoge bron en voert beide modi uit op test.
- source_evaluation.py bewaart logits/labels/voorspellingen en controleert de
  onveranderde model-/BN-staat. build_models roept de bestaande loader aan:
  load_exported_model(models, high_payload, device, bit=6)
  load_exported_model(models, high_payload, device, bit=4)
  Voor bit4 gebruikt die bestaande loader truncate_codes: codes >> 2 en dezelfde
  scale_codes. Er is geen nieuwe encodeer- of schaalregel.

De modelarchitectuur wordt eerst met tijdelijke willekeurige parameters opgebouwd,
zoals voor load_state_dict gebruikelijk. Voor inferentie zijn de zes doelparameters
verwijderd en vervangen door gedecodeerde buffers; alle overige opgeslagen staat
wordt gecontroleerd hersteld. Er worden nooit getrainde masters uit een checkpoint
gelezen. De guards blokkeren torch.load naar andere bestanden, alle weight-encoders,
G(W), train(True), optimizerconstructie en backward tijdens de bronreconstructie
plus inferentie. Dit is een programmeerfoutencontrole, geen OS-securitysandbox.

De nieuwe evaluatie schrijft NOOIT in de oorspronkelijke resultaat- of datamap.
Geen nieuwe keuze van beste epoch, geen kalibratie, geen warmstart of training.
