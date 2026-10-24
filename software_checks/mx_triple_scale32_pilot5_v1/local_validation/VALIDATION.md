# Uitgevoerde pakketvalidatie

Status: **PASS** voor de onderstaande controles. De tests gebruiken kleine, expliciet synthetische CPU-invoer. Er is tijdens pakketvoorbereiding geen SVHN-training uitgevoerd en er zijn geen nieuwe onderzoeksaccuracies of GPU-looptijden gemeten.

## Numerieke route en gradients

`test_core.py` voerde de werkelijke synthetische checks uit op PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu. Alle 254 eindige E4M3-codes kwamen overeen met een onafhankelijke bitstringmapping. Alle 250 toegelaten bronschaalcodes 0–249 en alle vijf niet-toegelaten eindige broncodes 250–254 zijn gecontroleerd, inclusief nulblokken. Niet-eindige FP32-forwardwaarden worden afgewezen, ook wanneer de E8M0-code zelf eindig is.

Bij dezelfde veilige E4M3-bron blijven de lagere elementcodes gelijk aan de gearchiveerde ×64-route en worden de gedecodeerde waarden exact gehalveerd. De schaal is exact ×32 tegenover de naïeve route; E2M1 behoudt de E2M3-schaal. De identity-STE zonder factor32 en de oorspronkelijke G(W)-afgeleiden zijn exact gecontroleerd.

Twee driepasses-updates kwamen exact overeen met een onafhankelijke handmatige CPU-lus, inclusief logits, gradients, parameters, Adamstaat en RNG. De drie BN-banken, detached teachers, broncache en afleiding zonder lagere encoder werden gecontroleerd. De test duurde circa 2,27 seconden. Zie `core_training_validation.json` en `core_static_review.json`.

## Echte runner, stop/hervat en exports

`test_training_integration.py` gebruikte de echte main-, trainings-, checkpoint-, export- en feedbackfuncties met drie synthetische beelden en batchgrootte twee. De testharness paste alleen datagrootte en CPU-routering aan; de productie-CLI kreeg geen synthetische of CPU-modus.

Een ononderbroken vijf-epochrun kwam exact overeen met een nieuwe run die na epoch 2 stopte en tot epoch 5 hervatte: model, Adam, scheduler, RNG, minibatchvolgorde en numerieke metrics. Het ongewijzigde schema 50/75/90 gaf gedurende de pilot geen LR-daling. Een vroege stop bleef als onvoltooide pilot herkenbaar.

De exports zijn onafhankelijk gecontroleerd op bitmapping, broncode +5, dezelfde lagere schaal, en afwezigheid van de zes MX-mastergewichten. Reconstructie vanuit alleen de opgeslagen E4M3-bron gaf dezelfde logits als de afzonderlijke exports, terwijl G(W) en alle encoders verboden waren.

Verkeerde ×64- en 100-epochcontracts en al afgewerkte checkpoints werden vóór updates geweigerd. Twaalf ongeldige CLI-invoeren werden zonder productierun geweigerd. In totaal gebruikte deze integratiecontrole 22 synthetische optimizerupdates en circa 20,27 seconden. Zie `training_integration_report.json`.

Ook de oorspronkelijke `check`-ingang met `create_context`, dependencycontrole, codeccontroles, trainingschecks en feedbackbundel is in de CPU-harness uitgevoerd: PASS. Zie `check_entry_validation_report.json`. De hashes van twintig actuele runtime-/bron-/testbestanden stemmen overeen met beide geteste snapshots; zie `independent_runner_review.json`.

## Notebook

Alle elf codecellen en acht figuurtypes zijn in het geheugen uitgevoerd op synthetische volledige en onderbroken rapporten en vervolgens op de werkelijk door de runner geschreven synthetische feedback. De notebook behoudt in het afgeleverde pakket lege outputs en lege execution counts.

De controle omvatte de gemeenschappelijke E4M3-checkpointselectie, ontbrekende of gemengde modi, verkeerde schalen/teacherlabels, onmogelijke voorspellingstellingen, CSV round-trip parsing, ongegronde epoch-100-claims en de read-only werking. Negentien gerichte foutvarianten werden correct herkend.

De echte ×64-referentie is via hashes gecontroleerd. Alleen epochs 1–5 komen in de vergelijking; de beste E4M3-epoch binnen dat bereik is 5 en niet het globale referentiecheckpoint 42. Een synthetische ×32-run wordt niet als onderzoeksvergelijking met die echte referentie getoond. Een afzonderlijke volledig synthetische presentatiecontrole controleerde de vergelijkingsgrafieken met verschillende beste epochs. De figuren zijn visueel gecontroleerd.

Zie `notebook_validation_report.json` en `runner_notebook_integration.json`.

## Behoud en grenzen

Alle 75 bestanden in het oorspronkelijke ×64-pakketmanifest, het oorspronkelijke pakketarchief en de zes gebruikte onbewerkte referentiebestanden bleven ongewijzigd; zie `prior_inputs_unchanged.json`. De nieuwe productiecode, notebook en analysehulp zijn aan de geteste hashes gebonden bij de eindverpakking.

Deze CPU-controles zijn geen bitexacte CUDA-herhaalgarantie. De gebruiker voert daarom eerst `run_train.py check` uit in zijn bestaande Windows/.venv-train/CUDA-omgeving en daarna de echte vijf-epochtraining. Er zijn geen dependencies op de gebruikerslaptop veranderd.
