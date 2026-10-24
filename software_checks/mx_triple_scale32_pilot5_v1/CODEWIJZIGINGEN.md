# Wijzigingen ten opzichte van vaste schaal ×64

## Numeriek verschil

De lagere E8M0-schaalcode is nu broncode **+5**, met factor **32**, in plaats van +6/factor64. De exponent- en mantissabitmapping, E4M3-broncodec, FP32-voorbewerking en identity-STE zijn behouden.

`trunc_adapter.py` heeft eigen schaal32-schema's en trainingsbeleid, een expliciete +5-overflowcontrole en aangepaste exportmetadata. De twee lagere modi blijven exact dezelfde schaal delen. Bij dezelfde E4M3-bron zijn hun elementcodes gelijk aan de oude ×64-route, en hun gedecodeerde gewichten op veilige representabele waarden exact de helft daarvan. Dit zegt niets over de uiteindelijke geleerde gewichten van twee verschillende trainingen.

`trunc_support.py` behoudt de oorspronkelijke driepasses/recursive-distillation/backward/updatevolgorde. Alleen de omschrijving van de schaalroute verandert. Er is geen vierde teacherpass en geen trainbare schaalparameter.

## Runner en afbakening

`run_train.py` is gebaseerd op de eerder gebruikte vijf-epochrunner, met nieuwe experimentidentiteit, nieuwe resultatenmap en +5/+32-controles. Het volledige LR-schema blijft 100 epochs met dalingen na 50/75/90; de uitvoerbare pilot stopt vast op 5. Een nieuw `run` begint vanaf dezelfde oorspronkelijke seed. Hervatten kan alleen vanuit een compatible, nog onvoltooid checkpoint van deze ×32-pilot.

De notebook, analysehulp en vastgelegde ×64-referentiebestanden worden na training in de resultatenmap gekopieerd en in de feedbackbundel opgenomen. Zo blijft de vergelijking uitvoerbaar zonder een hardcoded pad naar een oude run.

## Notebook

`notebook_support.py` en `MX_triple_scale32_pilot5_resultaten.ipynb` lezen uitsluitend resultaten. De notebook toont alle drie modi, epoch 5, het gezamenlijke E4M3-beste pilotcheckpoint en de vergelijking met de eerste vijf epochs van ×64. De CSV-parser gebruikt round-trip float parsing. De notebook bevat geen uitgevoerde resultaten en start geen training.

## Hergebruik en provenance

De elementcodecs, Any-Precision-bron-ZIP en overige ongewijzigde modules zijn byte-identiek aan het gevalideerde ×64-pakket. `REUSE_MANIFEST.json` legt vast welke bestanden ongewijzigd zijn. De oorspronkelijke ×64-runner, adapters, manifests en protocollen staan onder `provenance` en dienen als historische vergelijking; ze worden niet als nieuwe ×32-resultaten voorgesteld.

`TRAINING_CHANGE_MANIFEST.json` en `PROTOCOL.json` leggen het nieuwe contract vast. `PACKAGE_SHA256.json` bindt het afgeleverde pakket aan zijn bestanden. `local_validation/VALIDATION.md` beschrijft de werkelijk uitgevoerde tests en hun beperkingen.
