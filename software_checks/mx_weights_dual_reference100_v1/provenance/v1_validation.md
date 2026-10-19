# Uitgevoerde validatie en beperkingen

Datum: 2026-10-19. Python3.13.5, PyTorch2.10.0+cpu, torchvision0.25.0+cpu,
NumPy2.3.5. Alleen synthetische data en CPU, niet de Windows/CUDA-laptop.
Geen echte SVHN-training, accuracyresultaten of GPU-tijdmetingen uitgevoerd.

## Geslaagde softwaretests

- Alle bestaande E2M3-regressietests opnieuw uitgevoerd (64 codes, signed zero,
  4287 oraclewaarden, schalen, padding, STE en modelintegratie).
- Alle zestien E2M1-codes, exact herkenbare scalarwaarden en signed zero;
  4143 onafhankelijke oracle-inputs inclusief beide zijden van iedere tie.
- E2M1-schaalregels, nulblokken, padding, verzadiging, FP32-kleine/extreme
  exponenten, packed4-roundtrip, ongeldige codes en NaN/Inf afgewezen.
- Expliciet tegenvoorbeeld: directe E2M1-afronding is niet E2M3-code-truncatie.
- Identiteits-STE voor E2M1 inclusief de oorspronkelijke G(W)-gradients.
- Oorspronkelijke forward-AST exact gelijk buiten drie selectorparen.
- Twee gezamenlijke updates vergeleken met expliciet uitgeschreven manuele
  hoog-naar-laagdistillatie. Model, Adamstaat en RNG exact gelijk op CPU.
- Vijf synthetische mini-epochs zonder registratie versus volledige nieuwe
  runner: exacte gelijkheid van model, optimizer, scheduler, RNG en metrics.
- Twee epochs plus stoppen/herstarten plus drie epochs gelijk aan vijf
  ononderbroken epochs, inclusief dropout, twee actieve BN-banken en volgorde.
- Actieve BN32/BN4 eenmaal per minibatch; A32 blijft overal behouden;
  ongebruikte BN1/BN2/BN8 onveranderd en zonder gradients/Adamstaat.
- Gewijzigd contract, overschrijven en corrupte checksums worden afgewezen.
- Stopbestand stopt na de lopende epoch. Vijf-epochlimiet en geen stille
  CPU-fallback/warmstart gecontroleerd via de echte CLI (zie cli_guards.json).
- Selectie op eerste strikte E2M3-piek, niet op een E2M1-piek of latere tie.
- Oorspronkelijke scheduler 50/75/90 gecontroleerd in een parameter-only-
  honderd-stappenproef. Dit is geen honderd-epochtraining op echte data.
- Beide direct exports voor best/last gereconstrueerd in een nieuw proces
  zonder referentiecheckpoint en zonder zes FP32-masterparameters.
- De notebook uitgevoerd op opgeslagen synthetische runrapporten; grafieken
  visueel nagekeken. Notebook leest alleen onderzoeksbestanden.
- Negen hergebruikte infrastructuur-/bronbestanden byte-identiek; run_epochs-
  functie AST-identiek aan de geslaagde eenmodus-pilotrunner.

Details: validation_report.json, validation.log, cli_guards.json en
fresh_process_export_test.txt. Trainingscheckpoints uit deze tests worden niet
als onderzoeksresultaten meegeleverd. De productie-CLI kent geen synthetische
of CPU-bypass. Alleen het offline testscript patcht Tensor.cuda voor CPU-tests.

Herhalen (alleen offline ontwikkeling; een nieuwe doelmap vereist):

    python local_validation/test_training.py NIEUWE_LEGE_TESTMAP

## Werkelijke productiepreflight in deze container

De echte opdracht `run_train.py check` verifieerde pakket en auteursbron,
maar stopte bij pip check: bestaand MoviePy2.2.1 vraagt Pillow<12 terwijl hier
Pillow12.3.0 staat. Er is niets geupgraded of omzeild. Dit is niet de omgeving
van de gebruiker. Zie container_preflight_report.json en container_pip_check.txt.

Daarom is een geslaagde `check` op de laptop noodzakelijk voordat `run` wordt
gestart. CPU-gelijkheid bewijst geen bitexacte CUDA-uitvoering; het bestaande
cuDNN-benchmarkbeleid blijft behouden. Geen garantie van de uiteindelijke
accuracy, volledige trainingslooptijd of hardware-energiewinst.
