# Wijzigingen tegenover de tienkeuzepilot

Dit pakket bouwt voort op `mx_triple_learned_scale10_pilot5_v1` en de geslaagde run van 4 november 2026. De inhoudelijke quantisatie- en trainingsmethode blijft gelijk; de nieuwe proef begint opnieuw en loopt tot 100 epochs.

| Bestand | Wijziging en doel |
|---|---|
| `run_train.py` | Nieuw contract en resultatenpad; maximaal 100 epochs; juiste volledige/onderbroken status; voortzetting alleen binnen hetzelfde contract; bewaart iedere epoch en identificeert checkpoint 50. Leest de volledige opgeslagen historie terug voordat de export kan slagen. |
| `trunc_support.py` | Byte-identiek aan de geslaagde tienkeuzepilot; geen wijziging aan quantisatie-, schaalgradient- of updateberekeningen. |
| `PROTOCOL.json` | Eindepoch 100, schaalleren tot 100 en geen bevriezing; LR-milestones 50/75/90 blijven gelijk. |
| `notebook_support.py` en notebook | Volledige curves en epoch 100/beste-E4M3-overzichten; echte binaire 100- en vaste referenties; afzonderlijke vroege vergelijking met de tienkeuzepilot; correcte LR-controles na 50/75/90. |
| `pilot_reference` | Ongewijzigde resultaten van de werkelijk uitgevoerde tienkeuzepilot, met herkomst en hashes. |
| `fixed_references/learned_binary_reference100` | Ongewijzigde resultaten van de eerdere binaire training met leren tot 100; een nieuw analyseremanifest verwijst alleen naar de meegeleverde bestanden. Het oorspronkelijke herkomstmanifest is bewaard. |

Behoud: E4M3-encoder, exponent-MSB-truncatie, E2-mantissatruncatie, E8M0-schalen, tien offsets 0..9, B32, seed 20261009, oorspronkelijke modelinitialisatie, θ-init 6 zonder RNG-verbruik, één FP32-θ per groep, grenzen [0,9], dezelfde surrogate-gradient zonder normalisatie, Adam-lr 0,001, drie passes per minibatch, losses, BN-banken, A32 en FP32-masters.

**Gewichten en schaalkeuzes blijven alle 100 epochs leren.** De LR-daling na 50 is geen bevriezing. Een latere vriesproef kan uit het volledig bewaarde checkpoint 50 starten; die proef zit niet in dit pakket.

Oude pakketten en ruwe referentieresultaten zijn behouden. `REUSE_MANIFEST.json` beschrijft byte-identiek hergebruik en gecontroleerde wijzigingen. Nieuwe lokale validatie staat in `local_validation`; overgenomen pilotvalidatie is als herkomst onder `provenance` bewaard. De notebook wordt met lege outputs geleverd.

Bij de synthetische hervatcontrole werd één afgebroken `history.json` aangetroffen, terwijl het volledige checkpoint en de numerieke eindtoestand correct waren. De oorzaak kon niet worden vastgesteld. Twee nieuwe exports vanuit hetzelfde geverifieerde checkpoint waren volledig. De toegevoegde exportcontrole vergelijkt daarom de volledige teruggelezen historie met de verwachte inhoud en weigert een onvolledig bestand. Deze controle wijzigt geen trainingsberekening. De oorspronkelijke testuitvoer is behouden; de bevinding en gerichte regressiecontroles staan in `local_validation/RUNNER_VALIDATION.md`.
