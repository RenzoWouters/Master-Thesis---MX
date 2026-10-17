# Van vijf-epochproef naar honderd epochs

Geen nieuwe kwantisatiemethode. De MX-run gebruikt dezelfde E2M3-gewichten, A32,
identity-STE en FP32-masters als in de succesvolle proef. De controle zet alleen
MX uit en behoudt G(W). Beide zijn afzonderlijke eenmodusmodellen vanaf initialisatie.

## Wat is letterlijk hergebruikt?

De functie run_epochs in run_train.py heeft dezelfde abstracte syntaxisboom als
in de pilot. De volgorde forward/backward/optimizer, test, scheduler, logging en
checkpoint blijft identiek. Dat is automatisch gecontroleerd bij de pakketbouw.

Deze modules zijn byte-identiek aan de pilot:
- single_support.py: modelbouw, interne selector [32], eigen recorder en controles;
- mx_adapter.py en mx_format.py: zes quantizers, codec, STE en export;
- svhn_support.py: data, auteurs-forward laden, registratie;
- resume_core.py en smoke_support.py: checkpoints, RNG en bronvoorbereiding;
- smoke_single.py, test_mx.py en het ingesloten auteursarchief.

## Wat verandert in run_train.py?

1. config: nieuw schema en reference_end_epoch=100 in plaats van pilot_stop_epoch=5.
   planned_epochs was al 100; optimizer en LR-mijlpalen veranderen niet.
2. CLI: standaard en toegelaten eindpunt 100. `run --mode ...` blijft verse initialisatie;
   --checkpoint alleen bij het expliciete noodcommando resume.
3. Nieuwe outputroot mx_runs/mx_weights_single_reference100 en nieuw rapportschema.
4. export_results: full_training_completed alleen bij exact 100 epochs;
   gemeten volledige epochtijd apart van projectie, en aanvullende LR/controlerapportchecks.
5. Nieuwe read-only notebook met expliciete vergelijking op epoch100 en apart beste/laatste.

Geen toevoeging van AMP, dataloader-optimalisatie, stochastic rounding, truncatie,
activatiekwantisatie, distillatie, gradientkwantisatie of wijziging van de masterdtype.
De eerste stap blijft W -> G(W) -> Q_MX(G(W)) (of alleen G in control).

De vorige code-toelichting en oorspronkelijke onderzoeksdiff zijn ongewijzigd
bewaard in provenance/pilot5. De NIEUWE diff staat in fullrun.patch en het
huidige auditmanifest in TRAINING_CHANGE_MANIFEST.json. Oudere verwijzingen naar
vijf epochs gelden alleen voor die historische bestanden.
