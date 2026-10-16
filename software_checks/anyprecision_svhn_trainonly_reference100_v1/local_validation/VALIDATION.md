# Uitgevoerde offline validatie: honderd-epochversie

## Afbakening

Linux, Python 3.13.5, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu,
NumPy 2.3.5, SciPy 1.17.0. Dit is niet de Windows/CUDA-laptop.
Geen echte SVHN-data gedownload; nul echte SVHN-epochs; geen GPU-test,
onderzoeksaccuracies of nieuwe train-only-GPU-tijdmeting in deze omgeving.

De hoofdtest gebruikt 6 kunstmatige trainbeelden en 4 testbeelden met de echte
CNN-8. Twee trainminibatches per mini-epoch. Alleen de tests vervangen tijdelijk
Tensor.cuda door de identiteit. De productie-CLI kent GEEN CPU-/subset-bypass.

## Geslaagde softwaretests

- 100 mini-epochs met de oorspronkelijke niet-geinstrumenteerde forward tegenover
  dezelfde forward plus bestaande logging en checkpointlus. Exacte gelijkheid
  van alle eindgewichten, optimizer, scheduler, RNG-toestanden en epochmetrics.
- 100 doorlopende mini-epochs tegenover 50 + hervatten tot 100, inclusief dropout,
  samplevolgorde, BatchNorm, LR en geschiedenis. Exacte CPU-gelijkheid.
- Aanvullend hervatten na epoch 75 naar 76 en na 90 naar 91: gelijk aan de
  betreffende opgeslagen toestand uit de ononderbroken test.
- Het volledige LR-verloop en de overgangen na 50/75/90 gecontroleerd.
- 100 checkpoints, beste en laatste checkpointkopie gecontroleerd op hash,
  correcte epoch, modelstaat en alle vijf metrics.
- Strikte W32/A32-selectie, eerste bij gelijke score; geen beste lagere-precisie-
  epoch kiezen. De synthetische test bevat een best checkpoint eerder dan 100.
- Veranderde contracten, corrupte checkpoints en overschrijven worden geweigerd.
  Zowel cross-split- als pilot-naar-fullrun-hervatten wordt geweigerd.
- Onjuiste LR of ontbrekende epochs worden door completion_summary geweigerd;
  een geschiedenis van vijf epochs wordt niet als voltooide training gerapporteerd.
- STOP_AFTER_EPOCH stopt na het afwerken en opslaan van de huidige mini-epoch.
- Oorspronkelijke SVHN-loader getest op synthetische .mat-bestanden: uitsluitend
  train/test gelezen, 10-naar-0-labelmapping en Resize(40) behouden. Geen extra nodig.
- Een onjuist extra-bestand wordt genegeerd en niet gewijzigd; een verkleinde
  productiedataset wordt geweigerd. Downloader alleen gemockt getest.
- Train-only-pilotconfig vergeleken met nieuwe config: alleen schema/doelomschrijving
  en expliciete eindpuntmetadata verschillen. Alle trainingshyperparameters gelijk.
- run_epochs, dependencies, restore_timing_sidecar en verify_unchanged AST-identiek
  aan het geslaagde pakket. svhn_support.py, smoke_support.py, resume_core.py en
  auteursarchief byte-identiek. Oorspronkelijke model-/optimizerbestanden intact.
- Notebook uitgevoerd op opgeslagen synthetische rapporten. Alle cellen geslaagd;
  learning-rate- en accuracygrafieken visueel gecontroleerd op leesbaarheid.
  De geleverde notebooktemplate bevat geen synthetische onderzoeksresultaten.
- CLI weigert stopgrens 5, CPU, een checkpoint bij run en resume zonder checkpoint.
- CLI/eindexport aanvullend getest met gemockte context op de opgeslagen synthetische
  honderd-epochfixture: standaard eindpunt 100, geen startcheckpoint, eindstatus,
  resultaatnotebook en feedbackbundel correct. Dit is geen GPU-preflight.

Details: validation_report.json, cli_validation.json en cli_export_report.json.

## Onderbreking van de testomgeving

Een eerste synthetische testaanroep bereikte de tijdslimiet van de tool tijdens
het aanvullende hervatgedeelte. Tot dan waren baseline en instrumented honderd
mini-epochs afgerond. Die onvoltooide aanroep is niet als volledige PASS geteld.
De hele suite is vervolgens opnieuw in een aparte testmap uitgevoerd en PASS.
Dit was geen onderbreking van een gebruikersrun en geen wijziging van ruwe data.

## Volledige productiepreflight in deze container

Een werkelijke aanroep van run_svhn.py check controleerde pakket- en bronhashes,
maar stopte bij pip check: bestaande MoviePy 2.2.1 vereist Pillow <12, terwijl
Pillow 12.3.0 aanwezig is. Ook pip meldde een niet-schrijfbare cache. Er zijn geen
pakketten aangepast en geen controles omzeild. Zie cli_preflight.log en
container_pip_check.txt. Die dependencycombinatie is niet de .venv-train van de
gebruiker. Echte CUDA-modelvormcontrole en productie-uitvoering zijn hier niet
uitgevoerd. De check-opdracht moet dus op de laptop slagen voor de training start.

## Herhalen

    python local_validation/test_svhn.py NIEUWE_TESTMAP
    python local_validation/test_cli_export.py DEZELFDE_TESTMAP NIEUWE_CLI_TESTMAP

Deze tests zijn niet nodig om de gewone training te starten. Ze zijn uitsluitend
voor codecontrole en mogen niet als thesisresultaten worden gepresenteerd.
