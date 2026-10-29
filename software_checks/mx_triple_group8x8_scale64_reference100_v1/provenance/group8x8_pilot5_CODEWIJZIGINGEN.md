# Wijzigingen ten opzichte van het geteste 1×64-pilotpakket

- Expliciete 8×8-geometriehelpers voor vormen, groeperen, terugplaatsen en paddingvalidatie.
- De drie scalarcodecs behouden elementwaarden, afronding en schaalbeleid; tensorcodecs en diagnostiek gebruiken nu tweedimensionale blokken.
- Nieuwe layoutstrings en checkpoint-/exportschemas weigeren oude 1×64-bestanden ondanks dezelfde groepsomvang.
- De ×64-truncatieketen, drie passes, recursieve distillatie, identity-STE, A32, FP32-masterstaat, optimizer, scheduler en architectuur blijven gelijk.
- Kosten en notebook vergelijken 8×8 met 1×32 en 1×64. De echte volledige referentietrainingen worden beperkt tot hun eerste vijf epochs.
- Tests omvatten een onafhankelijke scalaroracle, beide paddingassen, transponeren, exportreconstructie en stoppen/hervatten.

Oude validatiebewijzen staan onder `provenance/inherited_group64_validation`; nieuw bewijs staat onder `local_validation`. Het protocoldelta staat in `TRAINING_CHANGE_MANIFEST.json`, bestandsherkomst in `REUSE_MANIFEST.json`. De originele B64-runner en het protocol zijn bewaard in `provenance/group64_pilot5_*`.
