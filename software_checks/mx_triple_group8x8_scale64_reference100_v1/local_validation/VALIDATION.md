# Validatie van de volledige 8×8 ×64-training

**Status: PASS.** Deze controles gebruiken kleine synthetische fixtures op CPU. Er is tijdens de voorbereiding geen echte SVHN-training uitgevoerd of dataset gedownload. De echte vijf-epochpilot van de gebruiker is afzonderlijk vastgelegd onder `pilot_analysis`.

## Hergebruik en protocol

Negentien Pythonbestanden in de pakketroot zijn byte-identiek aan de geslaagde 8×8-pilot. Daaronder vallen alle codecs, grouping/packing, de vaste ×64-truncatieroute, identity-STE, losses, datahelpers, kostenberekening en hervatlogica. Alleen `run_train.py` en `notebook_support.py` zijn gewijzigd.

De exacte vijf protocolwijzigingen zijn schema, purpose, verwijderen van `pilot_stop_epoch`, toevoegen van `reference_end_epoch=100` en reporting policy. Numerieke instellingen, seed, A32, FP32-masterstaat en LR-schema blijven gelijk. De runner start een verse training en accepteert alleen hervatten binnen het nieuwe contract.

## Training en opstart

- Werkelijke `main/check` geslaagd met een uitsluitend voor de test aangepaste CPU-runtime. De productie-CLI blijft CUDA vereisen.
- Werkelijke datasetopstart geslaagd met synthetische bestanden die de canonieke namen `train_32x32.mat` en `test_32x32.mat` dragen. Model, optimizer, scheduler, loaders en eerste batches zijn geconstrueerd; geen optimizerupdates in deze opstartcontrole.
- Werkelijke trainingslus uitgevoerd tot epoch 100 op drie synthetische beelden: 200 Adamupdates, 600 trainingspasses, alle LR-overgangen na 50/75/90 gecontroleerd.
- Verse uitvoering gestopt na epoch 74 en hervat tot 100. Model, Adam, scheduler, RNG, metriekgeschiedenis en samplevolgordes zijn exact gelijk aan de ononderbroken CPU-uitvoering.
- Eén E4M3-bron per MX-laag/minibatch, exponentbits, schaalcode +6, mantissatruncatie, A32, FP32-staat, drie actieve BN-banken en één Adamupdate per triplet gecontroleerd.
- Zes gekozen exports gecontroleerd, inclusief onafhankelijke reconstructie uit de E4M3-bron zonder masters of lagere encoders. Geometrie: 2.024 groepen en 64 paddingposities.
- Zes onjuiste train/test-geometrieaudits worden vóór exports geweigerd. Pilot-, B32-, 1D64- en reeds voltooide epoch-100-checkpoints worden vóór updates geweigerd.
- Vroege stop heeft de status `STOPPED_AT_EPOCH_BOUNDARY`; alleen een volledige run krijgt trainingsstatus PASS.

Alle integratiegevallen samen gebruikten **402 synthetische Adamupdates**. Dit zijn softwarecontroles, geen SVHN-accuracies, CUDA-hervatgarantie of GPU-tijdmetingen. De productierunner is na deze controles ongewijzigd.

Rapporten: `check_entry_validation_report.json`, `startup_regression_report.json` en `training_integration_report.json`. De rapporten bevatten de hashes van de tijdens hun uitvoering gecontroleerde momentopname.

## Notebook

**20/20 tests geslaagd.** Gecontroleerd zijn onder meer alle honderd meetpunten, epoch 100, het eerste strikte E4M3-maximum, dezelfde checkpointkeuze voor alle precisies, gedeeltelijke runs, ontbrekende modi, bronhashes en onjuiste geometrie/kosten.

De definitieve notebook is met echte IPython uitgevoerd op de synthetische runneroutput van epoch 100 en de onderbroken run van epoch 74: telkens **14 codecellen, 10 figuren en nul analysefouten**. Check-only en het lege sjabloon zijn eveneens uitgevoerd. Niet-bestaande resultaten worden niet ingevuld. Synthetische accuracies worden niet vergeleken met de echte B32/B64-resultaten.

De twaalf ruwe B32/B64-referentiebestanden zijn byte-identiek aan de eerdere kopieën. Alleen de vergelijkingsmetadata is aangepast naar epochs 1–100. De notebook wordt zonder outputs geleverd.

Na de trainingscontrole is uitsluitend een gedeeltelijke grafiektitel ingekort tot één regel en de bijbehorende testassert aangepast. De definitieve helper is daarna opnieuw getest en uitgevoerd op de volledige en gedeeltelijke runneroutput. Er veranderde geen training-, selectie- of numerieke logica. Documentatie en validatierapporten zijn vervolgens toegevoegd. `validation_summary.json` controleert dat alle definitieve trainingsbestanden overeenkomen met de gevalideerde hashes; `PACKAGE_SHA256.json` beschrijft het definitieve pakket.

## Behoud en grenzen

De volledige bestaande 8×8-pilot en 1×64-pilot zijn tegen hun oorspronkelijke pakketmanifesten gecontroleerd: geen wijzigingen. Oude validatierapporten blijven herkenbaar onder `provenance` staan.

De echte vijf-epochfeedback vermeldt geslaagde CUDA-exportcontroles. De betreffende `.pt`-bestanden zijn niet in de feedback aanwezig en zijn tijdens deze voorbereiding niet opnieuw gedecodeerd. De 100-epochresultaten moeten nog op de laptop worden gemeten. Daar controleert het commando `check` de daadwerkelijke Windows/CUDA-omgeving.
