# Uitgevoerde softwarevalidatie

Deze controles gebruiken synthetische modellen/beelden en echte opgeslagen
rapportmetadata. Ze bevatten **geen nieuwe SVHN-accuracies**. De echte getrainde
MX-exports en GPU staan op de laptop van de gebruiker. Daar worden vervolgens
`check` en `evaluate` uitgevoerd met de bestaande `.venv-train`.

## 1. Bitbewerkingen en bevroren modellen — PASS

`test_core.py` voert de werkelijke `test_scale64.py` uit met het oorspronkelijke
upstream CNN8-model op CPU. De gebruikte tijdelijke runtime bevat Python
3.12.14, PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu. Er worden geen wijzigingen
aan de gebruikersomgeving aangebracht.

Gecontroleerd:

- alle 254 eindige E4M3-codes met onafhankelijke bitstring- en `Fraction`-oracles;
- 16 bereikbare nearest-even midpointgevallen;
- alle 255 geldige A-schaalcodes en alle 249 exact verschuifbare B/C-schaalcodes;
- overflow, signed zero, nulblokken, packing, padding en brononveranderlijkheid;
- negen modelvergelijkingen: drie opgeslagen B-trainingsmodi en zes onafhankelijke A/B/C-oraclemodellen;
- echte afleiding, opslag/herladen, bevroren reconstructie en gepaarde evaluatie van negen routes op vijf synthetische beelden;
- 46 negatieve controles voor foutieve codes, schema's, bronhashes, schaalverwisseling, dubbele ×64, verborgen masters en ongeoorloofde herquantisatie;
- E4-relatieve metadata van alle negen routes, inclusief de opgeslagen lagere B-modi;
- exacte B-reconstructie van codes, schalen, modelstaat, CPU-logits, CE en voorspellingen;
- geen updates, geen backward, geen model- of BN-staatwijziging.

De omzettingsfuncties `transform_codes` en `shifted_scale_codes` zijn AST-identiek
aan de vorige A/B/C-evaluator. Zie `test_core_validation_report.json` en
`test_core_review.json` voor de bevindingen en exacte geteste bestandshashes.

## 2. Runner en broncontract — PASS

`test_runner_integration.py` controleert de productiepin tegen de echte
schaal ×64-feedbackmetadata en test de strikte inspectie daarnaast met een
afzonderlijke private fixturepin. De echte productiepin weigert die randomexports.
Er is geen bypass aan de productiecode toegevoegd.

De werkelijke `main()` doorloopt `check` en `evaluate` met twee synthetische
checkpointgroepen, drie beelden en batches van twee plus één beeld. De
nodige CPU-/datasetvervangingen bestaan uitsluitend in het testharnas en staan
expliciet in `runner_integration_report.json`.

Gecontroleerd:

- 16 referentie-/historische controles en 9 afwijzingen van ongeldige CLI-opties;
- 18 metricregels, 24 gepaarde regels, 216 gewichtsregels, 108 coderegels, 6 historische herhalingen en 2 NPZ-bestanden;
- eigen B-route exact gelijk aan de opgeslagen lagere modi;
- correcte blanco notebook en volledige feedbackZIP zonder dubbele bestandsnamen;
- vijf echte FAIL-paden voor verkeerde checkpointprovenance, verborgen master, onjuiste bronhash, gewijzigde B-codes en gewijzigde B-schalen;
- bronbestanden, datasetfixture en pakketbestanden blijven ongewijzigd;
- optimizer, backward, trainingmodus, `G(W)` en ongeoorloofde quantizers zijn tijdens evaluatie geblokkeerd.

De rapporten vermelden de hashes van de werkelijk geteste runtimebestanden.

## 3. Notebook — PASS

`test_notebook.py` voert alle 11 codecellen in het geheugen uit. Imports van
PyTorch, model-, checkpoint-, dataset- en trainingsmodules zijn geblokkeerd.
Het gebruikt expliciet gemarkeerde synthetische resultaatarrays; de opgenomen
historische vergelijkingstabellen blijven de echte, gehashte eerdere metingen.

De controles omvatten de volledige actuele tabellen en beide NPZ-bestanden,
B-reconstructie, historische herhaling, gelijke schaal-/codeafleidingen,
foutmeldingen bij onvolledige runs en 28 negatieve gevallen. Ze verifiëren
ook de historische tabellen en het onderscheid tussen eigen checkpoints.

Alle acht figuren zijn als PNG/PDF gerenderd en visueel gecontroleerd. De
belangrijkste accuracy- en overzichtsfiguren hebben daarnaast een tweede
visuele controle gekregen. Bronbestanden blijven ongewijzigd. Er wordt
**geen uitgevoerde notebook** meegeleverd of opgeslagen.

Zie `notebook_validation_report.json` en de aanvullende integratiecontrole
van de echte synthetische runneruitvoer.

## 4. Eerdere echte A/B/C-metingen

De eerdere feedbackbundel is onafhankelijk herlezen: 36 metrics, 48 gepaarde
regels, vier per-beeldbestanden en twaalf historische herhalingen zijn
onderling consistent. De 29 ruwe invoerbestanden bleven ongewijzigd.

Beginmodelstaat, dataset, preprocessing, modelbron, uitvoeringsomgeving en
backend komen overeen met de nieuwe B-training. De beste epoch is per run
verschillend: eerder 54, nieuw 42. Die oorspronkelijke keuze blijft behouden.

Een oude CSV-parsercontrole gaf bij een zeer grote CE één ULP verschil:
`16073888499.481255` werd standaard ingelezen als `16073888499.481256`.
JSON, CSV-tekst en NPZ-gemiddelde waren correct. Round-trip parsing lost dit
op zonder datawijziging; de nieuwe notebook gebruikt die leeswijze consequent.

## Grenzen

Deze lokale controles bewijzen de geteste code- en bestandslogica op CPU,
niet de accuracies van de nieuwe echte post-hoc-proef. De GPUcheck en de
26.032-beeldenevaluatie worden op de oorspronkelijke laptop uitgevoerd.
De oorspronkelijke CUDA-backend blijft nondeterministisch; bitexacte
historische accuracyreproductie wordt niet beloofd. Dat wordt in de echte
resultaten gecontroleerd en gerapporteerd.
