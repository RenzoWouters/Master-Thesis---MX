# Wijzigingen voor B64 ×64 pilot5

Basis: het gecorrigeerde en succesvol getrainde `mx_triple_group16_scale64_pilot5_v2`-pakket. De oorspronkelijke bestanden en eerdere resultaten blijven behouden.

- `group_config.py`: vaste groepsgrootte 64. Groepering blijft per outputkanaal met padding aan het einde van de rij.
- Runtime/adapters: eigen B64-payloadidentiteit en controle van groepsgrootte; dezelfde scalaire codecs, exponentbitbewerking, vaste +6-schaaloffset en identity-STE.
- `run_train.py`: eigen experimentidentiteit en resultatenmap; stop op 5, verse initialisatie, bestaand LR-schema. Kopieert zowel B16- als B32-referentiegegevens naar de resultatenmap en feedbackbundel.
- `group_costs.py`: dezelfde geometrieformules voor B16/B32/B64, met B64 als actieve variant. Laagvormen blijven gelijk.
- Testfixtures: correcte aantallen elementen voor 64-positiesblokken; tests verifiëren packing, payloads, afgeleide schalen, gradients, broncache, exports en hervatten. Datasetnamen blijven `train_32x32.mat` en `test_32x32.mat`.
- Notebook: B64-resultaten en beide historische referenties binnen hetzelfde pilotbereik; alle drie scores uit één gemeenschappelijk E4M3-geselecteerd checkpoint per run. Kostentabellen omvatten schaalbytes en padding.

`REUSE_MANIFEST.json` beschrijft de werkelijk ongewijzigde bestanden tegenover de B16-basis. `PACKAGE_SHA256.json` controleert het nieuwe pakket. Oude validatierapporten staan uitsluitend als historische provenance; de actuele B64-controles staan in `local_validation`.

Geen nieuwe activatiequantisatie, schaaloptimalisatie, extra teacher, andere loss, dataselectie of gewijzigde BN-banksleutel. Voorbereidingscontroles zijn synthetisch en starten geen SVHN-training.
