# Verificatie versie2

`dataset_startup_v1_report.json` reproduceert de fout van v1. `dataset_startup_v2_report.json` bevestigt dat de echte contextconstructie en eerste databatches nu slagen. `test_dataset_startup.py` is de gerichte regressietest; hij gebruikt alleen synthetische CPU-fixtures, geen SVHN-training.

`independent_collateral_audit.json` beschrijft de controle op andere onbedoelde32x64-vervangingen. Er zijn geen andere aangetroffen in de productiecode. De bestaande codec-/training-/notebookcontroles staan expliciet als eerdere v1-controles in `inherited_v1`; de twintig bijbehorende productiemodules zijn byte-identiek.

De nieuwe verificatie omvat geen GPU-training. De gebruiker voert de technischecheck en vijfepochrun op de eigen Windows/CUDA-omgeving uit. Het vijfepochprotocol, groepsgrootte16, factor64, A32/FP32 en kostenanalyse zijn ongewijzigd.
