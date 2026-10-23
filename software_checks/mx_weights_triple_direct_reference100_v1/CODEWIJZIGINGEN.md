# Wijzigingen ten opzichte van de geslaagde gezamenlijke pilot

Dit pakket is een afzonderlijke nieuwe map. Het pilotpakket, de dedicated
E4M3-referentie en alle eerdere resultaten zijn behouden.

## Numerieke en trainingskern ongewijzigd

De volgende onderdelen zijn byte-identiek aan
`mx_weights_triple_direct_pilot5_v1`:

- alle drie codecs: `mx_e4m3.py`, `mx_format.py` (E2M3), `mx_e2m1.py`;
- de oorspronkelijke gewichtsvoorbewerking en adapter: `mx_adapter.py`;
- de drie gewichtsmodi, BN-bankmapping en directe exports: `joint_adapter.py`;
- de gezamenlijke trainingsstap, losses en controles: `joint_support.py`;
- de codec- en gezamenlijke controletests: `test_mx.py`, `test_e4m3.py`, `test_joint.py`;
- runtime/data/checkpointondersteuning: `svhn_support.py`, `smoke_support.py`, `resume_core.py`;
- `SOURCE_MANIFEST.json` en het gebundelde upstreamarchief.

De E4M3-codec blijft de geslaagde exact-binary-v2. E2M3/E2M1 behouden hun
bestaande conversie. Er is geen nieuwe quantizer, truncatieroute, schaalregel,
geleerde schaal, activatieprecisie of optimizerregel toegevoegd.

`REUSE_MANIFEST.json` legt deze 14 bestandsidentiteiten en hun hashes vast.
`PACKAGE_SHA256.json` beschermt het nieuwe pakket tijdens check/run/resume.

## Runner: van vijf naar honderd epochs

`run_train.py` heeft een nieuw referentiecontract, resultaatpad en notebooknaam.
Het vaste eindpunt is 100 in plaats van 5; het al aanwezige LR-schema
50/75/90 wordt nu volledig doorlopen. De modelinitialisatie, data, optimizer,
trainingstap en evaluatievolgorde zijn behouden.

De AST van `run_epochs` is gelijk aan de pilot op één foutmelding na:
"geen nieuwe pilot-epochs" is "geen nieuwe referentie-epochs" geworden.
De functies voor runtime/datacontext, afhankelijkheden, het herstellen van
epochtijden en de controle op ongewijzigde bronnen zijn inhoudelijk identiek.

Het nieuwe eindrapport:

- bevestigt voltooiing en PASS alleen wanneer alle 100 epochs zijn afgerond;
- bewaart een eerdere gecontroleerde stop als STOPPED_AT_EPOCH_BOUNDARY;
- verwacht bij volledige SVHN-data 57.300 Adam-updates en 171.900 trainingspasses;
- rapporteert bij voltooiing de som van alle 100 gemeten epochtijden, ook bij hervatten;
- onderscheidt gemeten tijd, huidige sessieduur en een eventuele looptijdprojectie;
- bewaart best/last en hun drie directe MX-exports onder hetzelfde beleid als de pilot.

Een nieuwe `run` begint vanaf seed 20261009. `resume` is uitsluitend voor
hetzelfde honderd-epochcontract, inclusief pakket-, bron-, data- en omgeving-
identiteit. Pilotcheckpoints worden geweigerd. De oorspronkelijke cuDNN-
instellingen blijven behouden; CPU-exactheid in de synthetische controle
is geen garantie op bitexacte CUDA-hervatting.

## Notebook: de afgesproken twee hoofdcheckpoints

`MX_triple_direct_reference100_resultaten.ipynb` is een lege uitvoerbare template.
`notebook_support.py` toont als hoofdvergelijking epoch 100 en het eerste
checkpoint met maximale E4M3-testaccuracy. Alle drie scores per groep komen
uit dezelfde epoch. Een ontbrekende of inconsistente epoch 100 krijgt geen
gevulde eindpuntgrafiek. De laatst beschikbare epoch is apart herkenbaar in
de tabel.

De accuracycurves, onderscheiden trainingsobjectieven/test-CE, epochtijden,
GPU-geheugen en audits blijven beschikbaar. Een voltooide referentie gebruikt
de werkelijk gemeten honderd-epochduur. De figuurfunctie schrijft uitsluitend
in nieuwe analysemappen. Er is geen extra uitgevoerde notebook meegeleverd.

## Herkomst en controle

`provenance/triple_pilot5_run_train.py` en
`provenance/triple_pilot5_notebook_support.py` zijn de ongewijzigde directe
voorgangers. De bijbehorende verschillen staan in de
`pilot5_to_reference100_*.patch`-bestanden. De oudere provenancebestanden
blijven historische onderbouwing van de stap van twee naar drie precisies.

`local_validation/pilot_feedback_audit.json` koppelt de werkelijk ontvangen
pilotbundel aan deze vervolgstap. De andere validatierapporten onderscheiden
synthetische CPU-controles uitdrukkelijk van echte SVHN- en GPU-resultaten.
Zie `local_validation/VALIDATION.md` voor wat daadwerkelijk is uitgevoerd.
