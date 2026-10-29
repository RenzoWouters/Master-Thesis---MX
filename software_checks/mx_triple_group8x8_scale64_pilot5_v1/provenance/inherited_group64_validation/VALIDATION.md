# Lokale validatie van B64 ×64 pilot5

Alle controles hieronder horen bij de nieuwe B64-variant. Oude B16-rapporten staan alleen onder `provenance/inherited_group16_validation`. Er is geen echte SVHN-training uitgevoerd en er zijn geen CUDA-benchmarks uitgevoerd tijdens deze voorbereiding.

## Numerieke code en groepsgeometrie — PASS

De uitvoerbare AST van de drie codecs is gelijk aan die van de succesvolle B16-basis. De geometriewijziging komt uit de centrale `BLOCK_SIZE=64`. Acht runtimeonderdelen blijven byte-identiek. De exponentbitmapping, mantissabitshift, vaste schaaloffset +6, identity-STE, recursieve loss en BN-banksleutels zijn gelijk gebleven. De B64-adapter heeft eigen schema- en policy-identiteiten.

De codec-, truncatie- en groepscontroles lopen via de echte `main/check`-route, met een uitsluitend voor de test ingestelde CPU-runtime. Ze omvatten onafhankelijke bitstreamcontrole, alle finite scalarcodes, schaalcodes en overflow, packing/unpacking, de broncache, gradients, BN en brongebaseerde exports. Verouderde testfixtures met een vast aantal B16-elementen zijn expliciet aan de B64-groepsbreedte aangepast.

Voor de zes echte laagvormen zijn 2.160 groepen, 8.768 paddingposities en 140.400 / 105.840 / 71.280 bytes voor E4M3/E2M3/E2M1 vastgesteld. Zie `codec_report.json`, `group_size_check_report.json`, `trunc_training_check_report.json` en `check_entry_validation_report.json`.

## Echte opstartfunctie — PASS

`create_context(check_only=False)` is uitgevoerd met de echte `ensure_dataset` en `validate_dataset`. Alleen de externe data-inhoud/lader en de apparaatkeuze zijn vervangen door expliciete fixtures. De juiste bestandsnamen blijven `train_32x32.mat` en `test_32x32.mat`; het extra-bestand wordt niet gecontroleerd of geladen.

Virtuele datasets hebben de oorspronkelijke aantallen 73.257 en 26.032. De oorspronkelijke transformaties leveren eerste batches van 128 beelden met vorm 3×40×40. De werkelijke modelconstructie, Adam, scheduler, dataloaders en geometriecontrole zijn uitgevoerd; nul optimizerupdates. Dit dekt de opstarttak af waarin het eerdere B16-v1-bestandsnaamprobleem optrad. Zie `startup_regression_report.json`.

## Werkelijke runner op kleine synthetische data — PASS

De hoofdtrainingslus is met drie synthetische CPU-beelden uitgevoerd. De volledige test omvatte vijf epochs, hervatten van 2 naar 5, vroeg stoppen en afwijzen van B16/B32/incompatibele en al voltooide checkpoints vóór updates. In totaal waren er 22 kleine optimizerupdates over de testgevallen, met een gemeten testduur van 17,09 seconden. Dit zijn softwarefixtures, geen onderzoeksresultaten.

Hervatten reproduceert model, Adam, scheduler en RNG exact binnen deze CPU-test. Alle zes geselecteerde exports zijn onafhankelijk uit de opgeslagen E4M3-bron gereconstrueerd en gecontroleerd op codes, schaaloffset, padding en bytes. Zie `training_integration_report.json`.

De CPU-initialisatie is gelijk aan de oorspronkelijke builder binnen dezelfde testomgeving. De feitelijke Windows/CUDA-run heeft een andere historische hash; een CPU-test bevestigt niet die exacte CUDA-hash. De oorspronkelijke initialisatiecode, seed en beleidsinstellingen blijven behouden.

## Notebook — PASS

De downloadnotebook heeft 24 cellen, waarvan 14 codecellen, met lege outputs en `execution_count=None`. Ze maakt 10 figuren aan bij uitvoering. De cellen zijn lokaal gecontroleerd op volledige en gedeeltelijke synthetische runnerresultaten en een lege resultatenmap. Inputbestanden zijn tijdens analyse ongewijzigd gebleven.

Een afzonderlijke selectiefixture geeft andere beste epochs voor B64, B32 en B16 en nog andere maxima voor de lagere modi. De vergelijking gebruikt steeds de eerste strikte E4M3-maximale epoch binnen het gemeenschappelijke bereik; de drie modi blijven bij één checkpoint. Onjuiste kosten, gemengde checkpoints, een B16-contract en gewijzigde referentiegegevens worden geweigerd. Vergelijking tussen synthetische fixtures en echte historische accuracies is geblokkeerd.

De grafieken zijn visueel nagekeken. Zie `notebook_validation_report.json` en `notebook_runner_integration_report.json`. De reeds geleverde B16-full100-notebook is ook tegen de echte nieuwe bundel gecontroleerd en blijft bruikbaar; `notebook_actual_group16_reader.json` bevat die uitkomst. Er is geen uitgevoerde notebook meegeleverd.

## Eindassemblage en beperkingen

`runner_implementation_audit.json` legt de hashes vast van de werkelijk geteste numerieke productiecode. `independent_delta_audit.json` bevat de onafhankelijke statische vergelijking. De eindassemblage controleert dat de geleverde modules bij die controles horen, dat `PROTOCOL.json` overeenkomt met de echte config, dat referenties ongewijzigd zijn en dat de ZIP-manifest alle pakketbestanden dekt.

De hoofdlooptests gebruikten een privésnapshot terwijl documentatie en notebook werden afgewerkt. Hun opgenomen volledige snapshotmanifest kan daarom historische document- of notebooknamen tonen. De numerieke runnerhashes komen overeen met de definitieve code; de definitieve notebook is afzonderlijk gevalideerd. De geleverde `PACKAGE_SHA256.json` is de autoriteit voor het complete nieuwe pakket.

De oorspronkelijke CUDA-backend biedt geen bitexacte herhaal- of hervatgarantie. Voer de gedocumenteerde `check` uit op de Windows/CUDA-omgeving voordat de vijf echte trainingsepochs worden gestart.
