# Softwarevalidatie van het honderd-epochpakket

## Scope

De softwaretests gebruiken uitsluitend synthetische beelden op CPU. Er is geen
echte SVHN-training, dataset-inferentie of CUDA-training uitgevoerd bij de
voorbereiding van dit pakket. De productie-CLI blijft CUDA vereisen; de lokale
testharnassen vervangen runtime en data alleen binnen hun eigen controle.

De onderzoeksresultaten van de echte vijf-epochpilot staan afzonderlijk onder
`pilot_analysis`. De analyse daarvan is gebaseerd op de aangeleverde feedback;
grote checkpoints en MX-payloads zitten niet in die bundel.

## Hergebruik en onafhankelijke review

Achttien kern- en bronbestanden zijn byte-identiek aan de werkende pilot:
codecs, bitbewerking, adapters, broncache, STE, recursieve forward/backwardlus,
zelftests, dataset/runtime, upstream-bron en resume-infrastructuur.
`REUSE_MANIFEST.json` specificeert deze bestanden.

`config_difference.json` controleert dat de configuratie uitsluitend verandert
in experimentbeschrijving/schema, het eindpunt en de rapportage.
`independent_runner_review.json` legt de onafhankelijke codebeoordeling vast.
De trainingslus zelf verandert niet; de eindpuntweigering krijgt alleen een
bijpassende tekst. De eerder uitgevoerde kernvalidatie is als herkomstbewijs
bewaard in `provenance/pilot5_core_training_validation.json`.

## Werkelijke check-entry en runner

`check_entry_validation_report.json` betreft de echte `main/check`-route met
een CPU-runtimepatch in het harnas. Deze controle voert de codecs en de
truncatietrainingszelftests uit, inclusief alle eindige E4M3-codes, schaalbehoud,
identity-STE, onafhankelijke handmatige trainingsvergelijking en exports.

`test_training_integration.py` gebruikt de werkelijke runner met drie synthetische
beelden en batches van twee en één. Het protocol behoudt de echte
learning-ratemijlpalen 50/75/90. De controle omvat:

- honderd opeenvolgende kleine epochs en de juiste LR aan beide kanten van
  iedere mijlpaal;
- hervatten na epoch 99 tot 100, met vergelijking van model, Adam, scheduler,
  RNG en inhoudelijke epochrecords op CPU;
- stoppen op een epochgrens zonder onterechte voltooiingsstatus;
- weigeren van een ander experimentcontract en een voltooid checkpoint;
- correcte drie passes, twee detached teachers, broncycli en schaalcontroles
  per minibatch, plus de voorspellingstellingen in history en CSV;
- opgeslagen E4M3-bron en twee afleidingen, reconstructie vanuit uitsluitend
  die bron terwijl G(W) en alle encoders verboden zijn;
- alle drie modi op hetzelfde door E4M3 gekozen checkpoint en op epoch 100;
- lege notebookkopie, feedbackbundels, inputbehoud en CLI-grenzen.

De uitkomsten en exacte testscope staan in `training_integration_report.json`.
Dit zijn geen SVHN-accuracyresultaten of metingen van GPU-trainingsduur.

## Notebook

`notebook_validation_report.json` beschrijft vier synthetische rapportsituaties:
een complete honderd-epochrun, vroeg stoppen, een constante voorspelde klasse
en check-only. Twaalf negatieve controles bewaken onder meer protocol,
selectie, metrieken, voorspellingstellingen, exports en de gemeten looptijd.

Alle tien codecellen en zes figuren worden uitgevoerd. De volledige
honderd-epochfixture toont epoch 100 naast E4M3-best op epoch 54. Een gelijke
E4M3-score op epoch 55 behoudt epoch 54; eigen maxima van de lagere modi worden
niet als gezamenlijk checkpoint gebruikt. De zes figuren zijn visueel beoordeeld.

`runner_notebook_integration.json` controleert daarna de aansluiting op de echte
synthetische runneroutput. De figuren worden in geheugen gemaakt, zonder de
ruwe runbestanden te wijzigen of een uitgevoerde notebook op te slaan.
Het pakket bevat één lege notebooktemplate.

## Assemblage en praktische grens

`ASSEMBLY_AUDIT.json` controleert dat de definitieve runtime overeenkomt met
de geteste bestandhashes, de hergebruikte kern gelijk is aan de pilot, de
onderzoeksbronbestanden ongewijzigd zijn en de ZIP geen synthetische checkpoints
of datasets bevat. `PACKAGE_SHA256.json` bevat de uiteindelijke pakketbestanden.

Op de laptop controleert `check` de relevante zelftests opnieuw op CUDA.
De oorspronkelijke trainingsbackend blijft behouden; bitexacte CUDA-herhaling
of -hervatting is niet gegarandeerd. Softwarecorrectheid voorspelt geen
uiteindelijke accuracy van de volledige truncatietraining.
