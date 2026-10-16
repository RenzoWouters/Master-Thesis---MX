# Uitgevoerde lokale validatie: train-only-pilot

## Reikwijdte

Uitgevoerd op Linux, Python 3.13.5, torch 2.10.0+cpu, torchvision 0.25.0+cpu,
NumPy 2.3.5 en SciPy 1.17.0. Alleen kleine synthetische CPU-softwaretests.
Geen echte SVHN-data gedownload of getraind. Geen CUDA-proef, GPU-tijdmeting
of onderzoeksaccuracy. De productie-CLI heeft geen CPU- of subset-bypass.

## Geslaagde controles

- Vijf synthetische mini-epochs via de originele forward tegenover de
  geinstrumenteerde runner: exacte gelijkheid van model, optimizer, scheduler,
  CPU/Python/NumPy-RNG en alle gerapporteerde epochmetrics.
- Vijf doorlopende mini-epochs tegenover twee plus hervatten tot vijf, inclusief
  dropout, BatchNorm, samplevolgorde, LR, optimizer en selectiegeschiedenis.
- Beste en laatste checkpointkopie gecontroleerd op modelhash, scores en epoch.
- Verkeerde contracten, corrupte checkpoints en overschrijven afgewezen.
- Een gewijzigd train_extra-contract wordt afgewezen bij hervatten.
- STOP_AFTER_EPOCH wordt verwerkt op een volledig opgeslagen epochgrens.
- De echte SVHNMIX-loader en datasets/data.py op kleine kunstmatige .mat-files:
  alleen train en test worden geopend. Extra is niet aanwezig en niet vereist.
- Label 10 wordt 0, uitvoerbeeldvorm 3x40x40; train/testvolgorde en labels correct.
- Een apart aanwezig ongeldig extra-bestand wordt genegeerd, niet vervangen.
- Mock-download vraagt alleen train/test; bestaande corrupte data blijven staan.
- Volledige productiegroottes worden vereist: een te kleine dataset wordt afgewezen.
- Strikte hoogste W32/A32-score, eerste bij gelijkheid; geen low-bit-piekselectie.
- Honderd-epoch-LR-schema afzonderlijk gecontroleerd bij 50/75/90.
- Productieconfig is vergeleken met de config uit de vorige feedbackbundel:
  alleen train_split en train_examples veranderen inhoudelijk; extra metadata
  beschrijft die keuze. Model/batch/LR/optimizer/precisies/backendbeleid blijven.
- Papervergelijking wordt altijd als andere datasplit en ander budget gelabeld.
- Notebook uitgevoerd op opgeslagen synthetische JSON/CSV, zonder training;
  een gegenereerde tijdgrafiek is ook visueel gecontroleerd.
- Bronbestanden en hergebruikte code blijven ongewijzigd volgens SHA-256.
- CLI weigert meer dan vijf epochs, CPU als productiedevice en een checkpoint
  bij de opdracht run. Alleen expliciete resume mag een checkpoint gebruiken.
- Het afgeleverde notebook bevat geen vooraf ingevulde onderzoeksresultaten.

Zie validation_report.json (17 softwaretests) en delivery_checks.json.
Een bestaande TERM-waarschuwing uit de oorspronkelijke terminalcode verscheen
bij de CPU-tests; de assertions en de notebookuitvoering slaagden. Er zijn geen
pakketten geinstalleerd of vervangen voor deze nieuwe train-only-versie.

## Reproduceren van de synthetische tests

    python local_validation/test_svhn.py <NIEUWE_LEGE_MAP>

Dat script vereist de ontwikkel/testafhankelijkheden torch, torchvision, scipy,
nbformat en nbclient. Het wijzigt alleen eigen tijdelijke synthetische bestanden.
Synthetische .mat-bestanden, testgewichten en uitgevoerde fixture-notebooks zitten
niet in de afgeleverde productie-resultatenmap en worden niet als onderzoek gedeeld.

## Nog op de laptop te controleren

De command check moet in jouw bestaande .venv-train met CUDA slagen. Daarna
controleert de productierun de echte train-/testbestanden, labels, alle beelden
per epoch, gradients, BatchNorm, checkpoints, bronintegriteit en tijdsverdeling.
De full-size train-only-GPU-proef en nauwkeurigheid zijn hier niet uitgevoerd.
De bestaande cuDNN-benchmarkinstelling blijft behouden; een exact CPU-herstel
bewijst geen algemene bitexacte CUDA-herhaalbaarheid.
