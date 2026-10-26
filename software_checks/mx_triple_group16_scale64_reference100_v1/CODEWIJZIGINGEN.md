# Codewijzigingen — B16 ×64 van pilot naar 100 epochs

De numerieke kern en het trainingsprotocol blijven gelijk aan de geslaagde vijfepochpilot v2. De volledige configuratie wordt met die pilot vergeleken. De bestandsnamen `train_32x32.mat` en `test_32x32.mat` blijven correct.

- `run_train.py`: eindpunt 100, eigen volledige-runcontract en resultatenmap, rapportage voor het E4M3-beste checkpoint en epoch 100, en een passende tijdmeting zonder pilotprojectie als eindresultaat. De bestaande kosten-/exportcontroles blijven actief.
- `notebook_support.py` en de nieuwe notebook: volledige geschiedenis en vergelijking met B32 ×64 op epoch 100 en per run het E4M3-beste checkpoint. Het kostenhoofdstuk voor groepen 16/32/64 blijft aanwezig.
- `scale64_reference`: dezelfde zes ruwe referentiebestanden; de referentiemanifest beschrijft nu de volledige 100-epochvergelijking. De oude pilotmanifest is bewaard als herkomstinformatie.
- `pilot_reference` en `pilot_analysis`: ongewijzigde gegevens en gecontroleerde analyse van de echte geslaagde B16-pilot. Deze zijn geen nieuwe honderd-epochresultaten.
- De synthetische controles omvatten de volledige uitvoeringsduur, LR-overgangen, stoppen/hervatten, export, selectie en de echte datasetopstartfunctie. Ze laden geen echte SVHN-data en worden niet als onderzoeksresultaten gerapporteerd.

De codecs, truncatieadapter, loss-/forwardvolgorde, groepsgrootte 16, vaste factor 64, A32/FP32, broncache, optimizer en LR-momenten veranderen niet. Alle bestaande pakketten, checkpoints en resultaten blijven behouden.
