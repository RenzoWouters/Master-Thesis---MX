# Van vijf naar honderd epochs: geen nieuwe truncatiemethode

Basis: mx_truncation_aware_pilot5_v1.zip, gecontroleerd tegen de echte
feedback_bundle(20261021-124513).zip. Alle49 geregistreerde pakkethashes kwamen
bij de bouw overeen; de pilot is PASS met5 epochs en2865 updates.

Ongewijzigd (byte-identiek):
- run_epochs in run_train.py;
- trunc_adapter.py: een gedeelde hoge bron, lage codes >>2, identity-STE;
- trunc_support.py en de oorspronkelijke geselecteerde Any-Precision-forward;
- mx_format.py, mx_e2m1.py, mx_adapter.py, dual_adapter.py, dual_support.py;
- svhn_support.py, smoke_support.py, resume_core.py;
- test_trunc_training.py, trunc_invariants.py, test_dual.py, test_mx.py;
- auteurs-ZIP, compatibiliteitsdocument en compatibiliteitspatch.

Gewijzigd in run_train.py:
- een uniek reference100-contract, target_final_epoch=100;
- run start honderd epochs opnieuw; pilotcontracten worden niet geaccepteerd;
- nieuwe standaardresultatenmap, zonder mutatie van eerdere mappen;
- strengere eindexportcontroles op LR-volgende, BN/Adam en beide bron-audits;
- completion_summary.json, full_training_completed en trainingspassenteller;
- gemeten honderd-epocheindtijd onderscheiden van de schatting;
- juiste notebooknaam en opname van voltooiingsrapport in de feedbackbundel.

De echte epochlus is NIET gewijzigd; de bestaande melding over geen nieuwe
'pilot-epochs' bij een ongeldig resumecheckpoint is alleen historische tekst.
De nieuwe commandogrens en het volledige contract zijn honderd epochs.

Notebook: leercurves, beste en laatste checkpoint, daadwerkelijk gemeten tijd,
LR-overgangen, broncontroles en optionele vergelijking met directe kwantisatie.
Die lage baseline is niet de achteraf getrunceerde baseline. Daarvoor behouden
we het aparte evaluatierapport; die vergelijking is een volgende analyse.

De lokale testharness voert nu honderd synthetische mini-epochs uit en test
hervatten na50/75/90. Geen echte SVHN- of CUDA-resultaten uit de container.
De oorspronkelijke pilotuitleg staat in provenance/pilot_CODEWIJZIGINGEN.md.
fullrun.patch bevat alle wrapperwijzigingen; FULLRUN_CHANGE_MANIFEST.json de hashes.
