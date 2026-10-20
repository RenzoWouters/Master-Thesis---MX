# Van vijf naar honderd epochs: geen nieuwe methode

Basis is de op de laptop geslaagde mx_weights_dual_pilot5_v1_1.zip.

Ongewijzigd (byte-identiek): run_epochs in run_train.py; dual_support.py met
originele distillatie en drie selectorvervangingen; dual_adapter.py; beide
MX-codecs; identity-STE; svhn_support.py; smoke_support.py; resume_core.py;
test_dual.py met de verbeterde v1.1-CUDA-diagnostiek; test_mx.py; auteurs-ZIP.

Gewijzigd in run_train.py:
- nieuw protocolschema en target_final_epoch=100 in plaats van pilot_stop_epoch=5;
- check blijft synthetisch; run start 100 epochs vanaf dezelfde seed;
- resume accepteert alleen een identiek reference100-contract, geen pilot;
- andere uitvoermap, zonder wijzigen van oudere bestanden;
- eindexport valideert ook alle volgende LR-waarden, Adam/BN/evaluatie-invarianten;
- completion_summary.json en full_training_completed maken een zachte stop expliciet;
- vier MX-exportbestanden en beste/laatste checkpoints blijven dezelfde exportprocedure;
- notebook toont het gemeten totaal en LR-overgangen in plaats van alleen een raming.

Geen verandering in datasetsplit, initialisatie, batchgrootte, loss, lossweging,
volgorde van forwards/backwards, dropout, afronding, schaalkeuze, BN-mapping,
optimizer, LR-schema of backend. Nog geen truncatie, extra precisie of native kernel.

Het oude volledige wijzigingsdocument staat in provenance/pilot_v1_1_CODEWIJZIGINGEN.md.
fullrun.patch bevat de wrapperdiff; FULLRUN_CHANGE_MANIFEST.json bevat hashes
van de pilot en onveranderde numerieke kern. Compatibiliteit blijft apart in
COMPATIBILITY.md. Oude onderzoeksmappen worden niet als werkmap gebruikt.
