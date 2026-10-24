# Wijzigingen ten opzichte van de schaal×64-pilot

De volledige proef behoudt exact de numerieke route en trainingmechanica
van `mx_triple_scale64_pilot5_v1`. De wijziging is de looptijd en de
bijbehorende rapportage van een volledige honderd-epochrun.

## Ongewijzigde bestanden

De achttien bestanden in `REUSE_MANIFEST.json` zijn byte voor byte gekopieerd
uit de pilot. Dit omvat alle codecs, de exponent- en mantissabewerkingen,
de scale64-adapter en support, gedeelde training- en resumeondersteuning,
de gebruikte codec-/trainingszelftests, bronmanifest en upstream-archive.

Daarmee blijven onder andere de broncode +6, identity STE zonder factor64,
G(W)-afgeleide, één E4M3-bron per laag/minibatch, A32/FP32, BN-banken en
recursieve distillatie dezelfde. De eerder uitgevoerde corevalidatie is
met herkomstbestanden opgenomen onder `provenance`.

## Runner

`run_train.py` gebruikt een eigen reference100-identiteit en resultatenroot.
De eindepoch en completioncontroles zijn 100. De trainconfig verschilt van
de pilot uitsluitend in schema, doelomschrijving, het vervangen van
`pilot_stop_epoch` door `reference_end_epoch` en de rapportagekeuze voor
epoch100 plus E4M3-best. De numerieke instellingen veranderen niet.

De complete-loopstructuur volgt het eerder gevalideerde naïeve
reference100-pakket. De functies voor context, afhankelijkheden,
timingherstel en eindcontroles zijn AST-identiek aan de scale64pilot.
De eigen resumechecks behouden de honderd-epochcontractgrens. Een pilot-
of andere-routecheckpoint kan niet als resumecheckpoint worden gebruikt.

## Notebook

`notebook_support.py` en de lege notebook verwerken de reference100-rapporten,
de honderd-epochcompletion en de twee gemeenschappelijke checkpointgroepen.
Schaalcode+6, mantissaschaalbehoud en correcte bronrelaties blijven gecontroleerd.
Bij een onderbroken run is epoch100 expliciet nog niet beschikbaar.

De round-trip CSV-parser en `SAVE_FIGURES=False` blijven behouden. De
notebook bevat uitsluitend analyse van opgeslagen meetgegevens en geen
modeluitvoering. Er wordt één lege notebook meegeleverd.

## Herkomst en validatie

Onder `provenance` staan de betreffende pilotbestanden, de oorspronkelijke
naïeve reference100-voorbeelden en de echte diffs van de aangepaste runner
en notebookhelper. `pilot_analysis` bevat uitsluitend de echte pilotresultaten
en hun vergelijking met de eerdere echte pilots. `local_validation` bevat
de afzonderlijke synthetische CPU-softwarecontroles van dit nieuwe pakket.

Alle eerdere pakketten en resultaten blijven behouden. Nieuwe runs worden
geschreven onder `mx_runs/mx_triple_scale64_reference100`.
