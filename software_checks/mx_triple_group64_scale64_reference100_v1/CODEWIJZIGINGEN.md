# Van B64-pilot naar B64-full100

Basis: `mx_triple_group64_scale64_pilot5_v1`, waarvan de echte vijf-epochrun technisch is geslaagd. Alle eerdere pakketten en resultaten blijven behouden.

- De numerieke modules, adapters, groepsconfiguratie, kostenformules en productiezelftests blijven byte-identiek.
- `run_train.py` krijgt de full100-levenscyclus: eigen contract en schema's, stop op 100, volledige voltooiingsrapportage en dezelfde oorspronkelijke LR-overgangen na 50/75/90.
- Nieuwe resultatenmap: `mx_runs/mx_triple_group64_scale64_reference100`.
- Verse oorspronkelijke initialisatie; hervatten alleen uit een eigen onvoltooid full100-checkpoint. Het pilotcheckpoint wordt als startpunt geweigerd.
- Notebook en helper analyseren volledige curves, epoch 100 en de globale E4M3-selectie. Beide bestaande B16/B32-full100-referenties blijven als ongewijzigde brondata opgenomen.
- De echte B64-pilot blijft apart in `pilot_reference`; analyse en bronbundel staan in `pilot_analysis`.

`REUSE_MANIFEST.json` legt de ongewijzigde modules en protocolverschillen vast. Historische pilotcontroles staan onder `provenance`; actuele full100-controles staan in `local_validation`. `PACKAGE_SHA256.json` dekt het complete nieuwe pakket.
