# Lokale validatie van de geleerde-schaalpilot

## Uitkomst en reikwijdte

De technische controles zijn geslaagd. Ze gebruiken synthetische tensors en beelden op CPU. Er is hier geen SVHN-dataset geladen of gedownload en er is geen onderzoeksrun van vijf of honderd epochs uitgevoerd. CUDA, de werkelijke accuracy en de looptijd op de RTX 5070 Laptop GPU worden pas met de meegeleverde `check`- en `run`-opdrachten vastgesteld.

De CPU-testomgeving gebruikte Python 3.12.14, PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu. De productiecode installeert of wijzigt geen pakketten op de trainingsmachine.

## Numerieke kern

`core_check_report.json` meldt PASS voor zeventien inhoudelijke controles en herstel van de RNG/backend. De controles omvatten alle 254 eindige E4M3-codes, beide offsets en gemengde blokkeuzes, signed zero, de geldige E8M0-schalen, expliciete overflowweigering, drempel/tiegedrag en gepakte keuzebits.

De schaalgradient wordt vergeleken met een onafhankelijke scalaire FP64-berekening over uitsluitend echte blokelementen. De bestaande mastergradient en G(W)-afgeleiden blijven gelijk, de lagere bijdragen tellen op, E4M3 geeft geen directe theta-gradient en tail-padding draagt niets bij. Een echte Adam-update kan de harde drempel passeren. Modelpasses, cachebewaking, projectie en reconstructie zonder encoder of master zijn afzonderlijk gecontroleerd.

## Werkelijke runner, hervatten en exports

`training_integration_report.json` beschrijft de echte `run_train.main`, trainingslus, losses, optimizer, scheduler, checkpoints en exports op drie synthetische CPU-beelden. Alleen data/context en CUDA-routering zijn voor deze lokale fixture aangepast.

- Een volledige synthetische vijf-epochrun bevatte tien gezamenlijke updates en dertig trainingspasses.
- Stoppen na twee epochs en hervatten tot vijf gaf exact dezelfde modelstaat inclusief theta, Adam, scheduler, RNG en beide schaal-CSV's als de doorlopende CPU-run.
- Stoppen na één epoch bleef correct als gedeeltelijk resultaat gemarkeerd.
- Vijf verkeerde of voltooide checkpointcontracten werden vóór updates geweigerd; twaalf ongeldige CLI-aanroepen werden eveneens geweigerd.
- De echte beginstaat met alle theta's op 6 gaf voor alle modi exact dezelfde gedecodeerde gewichten als een onafhankelijk opgebouwde vaste ×64-route. Toevoegen van theta gebruikte geen RNG en veranderde de oorspronkelijke modelstaat niet.
- De integratietraining gebruikte daarnaast bewust gemengde beginwaarden rond de drempel om beide keuzes, omschakelingen, projectie en hervatting daadwerkelijk te doorlopen. Dit staat expliciet als synthetische testfixture vermeld en verandert de productieinitialisatie niet.
- De gepakte keuzes zijn onafhankelijk uitgepakt. Alle drie modi konden met exact gelijke logits uit de opgeslagen bron worden herbouwd, terwijl de encoder en G(W) verboden waren. Er waren 510 keuzebytes en geen FP32-theta of masters van de zes MX-lagen in de compacte exports.

`check_entry_validation_report.json` bevestigt dat ook de daadwerkelijke `check`-entry, met CPU-routering, de codecs en de nieuwe schaaltests uitvoert. De gehele runnerintegratie omvatte 22 synthetische Adam-updates; dit aantal is geen trainingsresultaat op SVHN.

De integratiesnapshot werd gevolgd door strengere type-/eindigheidscontroles voor exportmetadata in `trunc_adapter.py`. De trainingsberekening bleef daarbij gelijk. `final_export_validation_report.json` controleert de finale adapter opnieuw tegen alle zes opgeslagen compacte exports en de bronreconstructie van de drie modi. `runner_validation_binding.json` legt de snapshot- en finale hashes en deze gerichte hercontrole vast. De rapporten van de snapshot zijn als historische testuitvoer behouden.

Exact CPU-hervatten is geen garantie voor bitexacte CUDA-hervatting met het behouden `cudnn.benchmark=True`-beleid.

## Notebook

De geleverde notebook bevat lege outputs. `notebook_validation_report.json` bevestigt uitvoering van alle vijftien codecellen en rendering van twaalf figuren voor een complete vijf-epochfixture, een gestopte twee-epochfixture en een fixture met constante voorspellingen en ongewijzigde schaal. Check-only wordt zonder verzonnen metrics verwerkt. Twaalf gerichte ongeldige resultaten worden afgewezen.

`runner_notebook_integration.json` bevestigt dezelfde analyse op de daadwerkelijke synthetische runneroutput: 24.480 opgeslagen blokstaten en dertig laag-epochrijen. Keuzes, theta-statistieken en netto updates worden onafhankelijk herberekend. Synthetische runs worden duidelijk aangeduid en niet met echte vaste referenties vergeleken.

De accuracy-, vergelijkings-, theta-, keuze- en gradientfiguren zijn visueel gecontroleerd. De vaste referenties zijn gepind op hun oorspronkelijke bestanden; de pilotvergelijking gebruikt uitsluitend de gezamenlijk afgewerkte eerste epochs. Iedere methode kiest haar E4M3-bestcheckpoint opnieuw binnen die beperkte periode. De lagere scores komen steeds uit datzelfde checkpoint.

## Pakket en bronbehoud

`package_assembly_validation.json` bindt het eindpakket aan de gevalideerde runtime- en notebookbestanden, controleert het protocol, de lege notebookoutputs, de referentiebestanden en de inhoud van de eerdere pakketmanifests. `PACKAGE_SHA256.json` in de pakketroot bevat de hashes van alle geleverde bestanden behalve zichzelf.

De oude ×32-pilot, het oude ×32-honderd-epochpakket en de aangeleverde feedbackbundel worden alleen gelezen. De nieuwe proef heeft een eigen map, contract en resultatenpad. Hun ongewijzigde hashes worden in het assemblagerapport vastgelegd.
