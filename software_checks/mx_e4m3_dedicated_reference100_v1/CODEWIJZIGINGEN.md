# Wijzigingen ten opzichte van de geslaagde pilot-v2

De nieuwe runner gebruikt hetzelfde E4M3-model en dezelfde numerieke code.
De codec, adapter, G(W), identity-STE, train/evaluatieforward, optimizer,
datasetcode en stop/resume-ondersteuning zijn behouden. Het verschil is de
honderd-epochreferentie als afzonderlijk experiment.

## Runner

- Nieuw protocol-, pakket- en rapportlabel; eigen uitvoermap.
- Eindepoch100, nieuwe initialisatie met dezelfde seed20261009.
- De publieke CLI accepteert uitsluitend eindepoch100. Een nette stop of
  hervatting bewaart het volledige honderd-epochcontract.
- PASS voor een trainingsrun en full_100_epoch_training_completed gelden
  uitsluitend wanneer epoch100 daadwerkelijk is afgerond.
- Bestselectie blijft eerste strikte maximum van de actieve E4M3-testaccuracy.
- Gemeten honderd-epochtijd toegevoegd zodra die beschikbaar is.
- Nieuwe notebooknaam in runmap en feedbackbundel.

De vijf-epochtraining wordt niet verlengd vanuit haar checkpoint. De nieuwe
contractidentiteit voorkomt dat een pilotcheckpoint als reference100-resume
aangenomen wordt. De oude code en resultaten zijn niet gewijzigd.

## Notebook

Hoofdfiguur voor epoch100 en best; laatste beschikbare epoch blijft zichtbaar.
Onderbroken runs claimen geen epoch100. Bij volledige training wordt gemeten
totale tijd gebruikt in plaats van een pilotprojectie. De opgeslagen cuDNN-
backendinstelling wordt afzonderlijk van A32/FP32-tensordtypes weergegeven.
De notebook leest alleen rapporten en start geen training.

## Herkomst en controles

`provenance/pilot5_v2_run_train.py` is de oorspronkelijke runner;
`provenance/pilot5_v2_to_reference100.patch` toont de exacte wijziging.
`REUSE_MANIFEST.json` documenteert byte-identiek hergebruik.
`local_validation/VALIDATION.md` beschrijft de feitelijk uitgevoerde gerichte
controles. De uitgewerkte pilotanalyse is afzonderlijk opgenomen onder
`pilot5_analysis/`; zij bevat echte vijf-epochmetingen, geen honderd-epochresultaat.
