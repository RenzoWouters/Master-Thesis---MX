# Lokale verificatie — B16 ×64 pilot

Alle onderstaande controles zijn geslaagd. De voorbereiding gebruikte PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu in een tijdelijke omgeving. Er zijn geen SVHN-data geladen en geen onderzoekstraining of CUDA-timing uitgevoerd. De gebruikersomgeving wordt door het pakket niet aangepast.

## Numeriek en codec

- Alle 254eindige E4M3-codes en64.770 code-schaalproducten vergeleken met een onafhankelijke decoder.
- Onafhankelijke controle van B16-groepering, padding, signedzero, bitpacking, payloadroundtrips en afwijzing van verkeerde groepsmetadata.
- De scalar-encode/decodefuncties en exacte machten-van-tweehelper zijn AST-identiek aan de bewezen oudercode.
- Werkelijke encoding van de zes lagen geeft8.096 groepen,64 paddingposities en137.632 / 105.248 / 72.864 bytes voor respectievelijk E4M3/E2M3/E2M1 inclusief schalen.
- Synthetische driemodusroute versus onafhankelijke handmatige referentie: forwards, gradients, optimizerstappen, RNG, cachebeheer en source-only exports.

Details: `codec_runtime_checks.json`, `static_delta_audit.json` en `test_group_size.py` in de pakketroot.

## Initialisatie en protocol

De B16-code en de bestaande B32-ouder geven onder dezelfde CPU-runtime exact dezelfde modelstate en RNG-toestand. Alle 132.954 parameters zijn FP32 en alle activaties A32. De modelarchitectuur, laagselectie en BN-banken blijven gelijk. Zie `paired_initialization.json`.

De configuratie is direct vergeleken met de bewaarde oorspronkelijke B32 ×64-pilot. Alleen `mx_block_size`32→16 en experiment-/vergelijkingsmetadata verschillen; schaalfactor, losses, precisies, seed, dataset, optimizer en LR-schema zijn identiek. Het werkelijke model levert dezelfde laagvormen als de onafhankelijke kostenberekening.

## Runner, stoppen en hervatten

De echte `run_train.main` is met drie synthetische CPU-beelden doorlopen. Uitsluitend runtime-/dataconstructie en CUDA-routing zijn in het afzonderlijke testharnas aangepast; productiebestanden en de werkelijke trainingslus zijn niet gemockt.

- Vijf mini-epochs, echte loss/backward/Adam, testfase, checkpoints, alle exports en feedbackbundel: PASS.
- Stop na epoch 2 en hervatten tot epoch 5: exact dezelfde CPU-model-, optimizer-, scheduler- en RNG-toestand als ononderbroken.
- Vroege stop en afwijzing van een B32-only-contractwijziging, andere schaalproef,100-epochcontract en reeds afgewerkte checkpoints: PASS vóór onjuiste updates.
- De echte `check`-entrypoint met CPU-runtimeaanpassing doorliep de codec-, groepsgrootte- en synthetische trainingschecks: PASS. `pip check` had geen conflicten in de controleomgeving.
- Onafhankelijke exportbitmapping en schaaloffset +6; geen verborgen FP32-masters; heropbouw zonder G(W) of encoders; B16-kosten per export: PASS.

Details: `training_integration_report.json`, `check_entry_validation_report.json`. De volledige scenarioharnas gebruikte 22 synthetische optimizerupdates, geen echte datasettraining. Deze controles geven geen CUDA-bitexactgarantie; voer op de laptop eerst de gewone `check` uit.

## Notebook en kosten

Alle 14 codecellen van de 24-cellige notebook zijn uitgevoerd tegen duidelijk gemarkeerde synthetische fixtures en tegen de daadwerkelijke uitvoer van de nieuwe runner. Er kwamen geen analysefouten voor; zes exportkostencontroles slaagden. Gemodificeerde kostentotalen en B32-uitvoer die als B16 wordt voorgesteld worden geweigerd. Synthetische accuracies worden niet met de echte historische referentie vergeleken alsof ze onderzoekresultaten zijn.

De grafieken voor het eindcheckpoint, het gezamenlijke E4M3-beste checkpoint, kosten en quantisatiediagnostiek zijn visueel gecontroleerd. De historische B32 ×64-referentie blijft byte-identiek en de vergelijking gebruikt uitsluitend epochs 1–5. Analyse met de standaardinstelling schrijft geen resultatenbestanden. De meegeleverde notebook bevat nul outputs en geen executioncounts.

Details: `notebook_validation_report.json`, `runner_notebook_integration.json`. Validatieafbeeldingen blijven interne testuitvoer; er wordt geen uitgevoerde notebook of synthetische onderzoeksgrafiek meegeleverd.
