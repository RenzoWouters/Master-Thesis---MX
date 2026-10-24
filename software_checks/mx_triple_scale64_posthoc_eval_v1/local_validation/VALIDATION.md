# Uitgevoerde softwarevalidatie — 24 oktober 2026

De nieuwe kern, runner en analyse-notebook zijn gecontroleerd met begrensde
synthetische CPU-proeven. Er zijn hier geen echte SVHN-beelden of getrainde
checkpointtensors gebruikt. De vroegere feedbackbundels leveren uitsluitend
de te controleren JSON/CSV-contracten. Deze rapporten bevatten geen nieuwe
experimentele accuracies.

## Kern en numerieke bewerking: PASS

`core_validation_report.json` legt de omgeving en geteste bestandshashes vast.
De kern is gecontroleerd op alle 254 eindige E4M3-codes met onafhankelijke
bitstring- en exacte Fraction-orakels, inclusief 16 bereikbare afrondingsties.
Alle 255 geldige bronschalen voor A en alle 249 verhoogbare schalen voor B/C
zijn gecontroleerd. Broncode 249 t/m 254 wordt bij offset +6 afgewezen.

Daarnaast zijn signed zero, lokale subnormals, padding, packing, nulblokken,
meerdere tensorvormen, niet-eindige FP32-decodering en inputbehoud gecontroleerd.
Er zijn 41 negatieve controles en 15 modelvergelijkingen uitgevoerd. De
CPU-logits waren bij alle 15 modelvergelijkingen exact gelijk aan de
onafhankelijke verwachte reconstructie.

De modellen kwamen uit beide ondersteunde bronschema's. Tijdens afleiding,
reconstructie en inferentie waren G(W), FP32-tensorencoders, optimizer,
backward en training-mode forwards verboden. Alle activaties bleven A32,
de doel-BN stond bevroren en de zes FP32-MX-masterparameters ontbraken.

## Echte entrypoints op synthetische input: PASS

`check_entry_validation_report.json` en `integration_report.json` beschrijven
de uitvoering van de werkelijke `run_eval.main()` voor `check` en `evaluate`.
De integratieproef gebruikte twee willekeurige modelpopulaties, twee rollen
per populatie en drie kunstmatige beelden in batches van twee en één.

De uitvoer bevatte exact 36 metrics, 48 gepaarde regels, 432 gewichtsregels,
216 coderegels, 12 historische-herhalingsregels, vier NPZ-bestanden en
24 afgeleide modellen die vóór inferentie waren opgeslagen en herladen.
De 12 baselinelogits kwamen exact overeen met de bestaande exportloaders.
De A-routes van het truncatietrainingsschema kwamen exact overeen met de
opgeslagen getrainde lagere modi. Onafhankelijke code- en schaalorakels
controleerden alle 24 laag/checkpointgroepen.

Het harnas controleerde ook verkeerde schemas, onvoltooide bronnen, A16 in
het protocol, exporthashes, bronhashrelaties, gewijzigde lagere codes,
checkpointkeuze, ongeldige historie, verborgen masters, schaaloverflow,
dataset-/beginmodelverschillen en ongeldige CLI-argumenten. De bronbestanden,
feedbackbundels en productiebestanden bleven ongewijzigd.

De CPU-aanpassingen bestaan uitsluitend in `test_scale64_integration.py`:
een tijdelijke CPU-runtime, een substitutie voor upstream CUDA-initialisatie,
kunstmatige driebeelden-data en expliciet beschreven fixturemetadata.
De gewone CLI heeft geen mogelijkheid om bron-, provenance- of datacontroles
over te slaan. De werkelijke strikte referentie-inspector is afzonderlijk op
de volledige historische JSON/CSV-contracten van beide trainingen getest.
Alle synthetische resultaten blijven als synthetisch gemarkeerd.

## Notebook: PASS

`notebook_validation_report.json` bevat de hashes van de definitieve helper
en lege notebook. Alle tien codecellen zijn in het geheugen uitgevoerd;
er is geen uitgevoerde notebook opgeslagen of meegeleverd. Zeven figuren
zijn als PNG en PDF gerenderd en visueel nagekeken. Beide trainingen en
checkpointrollen blijven apart; de accuracyfiguren bevatten epoch 100 én
E4M3-best. De figuren zijn uitdrukkelijk met synthetische data getest.

De analyse controleert rapport/CSV/NPZ-consistentie, beeldvolgorde, percentages,
gepaarde veranderingen, schaalbeleid en modelcontroles. De 24 ongeldige
gevallen werden afgewezen. Ontbrekende, gedeeltelijke en mislukte runs en
check-only rapporten worden herkenbaar behandeld. Een constante voorspelde
klasse en afwijking van historische accuracy blijven geldige uitkomsten.
Model- en torchimports waren tijdens de analyse verboden; ruwe inputs bleven
ongewijzigd. De afzonderlijke notebookintegratie gebruikt vervolgens de
uitvoer van de echte synthetische evaluator, met behoud van dat label.

## Omgeving en grenzen

De modelproeven gebruikten een afzonderlijke tijdelijke omgeving met
Python 3.12.14, PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu. De omgeving
op de laptop is niet aangepast. De rapporten vermelden de overige versies
en alle fixture-aanpassingen. `independent_review.json` bevat de review
van API-aansluiting, protocollen, provenance en wetenschappelijke interpretatie.

De echte Windows/CUDA-uitvoering, SVHN-testset en getrainde tensorbestanden
zijn hier niet getest. Daarvoor dient eerst het meegeleverde `check`-commando
in de bestaande `.venv-train`, gevolgd door `evaluate` op de twee volledige
oorspronkelijke runmappen. Die meet de wetenschappelijke resultaten.

## Reproduceerbaarheid van de ontwikkelcontroles

De lokale harnassen worden uitsluitend voor softwareontwikkeling meegeleverd.
`test_core.py` gebruikt tijdelijk synthetische exports. `test_notebook.py`
verwacht één nieuwe uitvoermap. `test_scale64_integration.py` verwacht een
nieuwe uitvoermap en de twee eerdere trainingsfeedback-ZIPs als metadata-input.
Ze vereisen een geschikte CPU-ontwikkelomgeving en zijn niet de commando's
voor de laptopproef; gebruik daarvoor de twee commando's in `README.md`.

De eindverpakking controleert dat alle bevroren runtimehashes nog overeenkomen,
dat de 17 hergebruikte bestanden byte-identiek zijn en dat er precies één lege
notebook in de ZIP staat. Synthetische modelbestanden, datasets, previewfiguren
en uitgevoerde notebooks worden niet meegeleverd.
