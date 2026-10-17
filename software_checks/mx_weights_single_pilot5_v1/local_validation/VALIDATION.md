# Uitgevoerde validatie

Alle numerieke tests zijn uitgevoerd met synthetische beelden op Linux/CPU,
Python 3.13.5, torch 2.10.0+cpu, torchvision 0.25.0+cpu, NumPy 2.3.5,
SciPy 1.17.0. Niet de Windows/CUDA-omgeving van de gebruiker.
Geen echte SVHN-training, echte onderzoeksaccuracies of GPU-tijdmeting uitgevoerd.

## Geslaagd

- Bestaande codec- en integratietests opnieuw uitgevoerd: alle 64 E2M3-codes,
  4287 scalar-oraclewaarden, grenzen, subnormals, padding, schalen, export en STE.
- Twee synthetische optimizerupdates per modus: FP32-masters, actieve gradients
  en Adam-momenten; alleen BN32 wordt bijgewerkt. Gewichten en MX-codes veranderen.
- Vijf synthetische mini-epochs, controle tegen ongewijzigde auteursforward met
  de argumentlijst [32]. Control gebruikt in de tegenproef de ORIGINELE W32-laag;
  de MX-tegenproef gebruikt dezelfde adapter zonder nieuwe instrumentatie.
- Exact gelijke modelparameters/buffers, Adamstaat, scheduler, RNG en alle
  epochmetrics tussen die tegenproeven en de nieuwe runner op CPU.
- Voor zowel control als MX: vijf doorlopende mini-epochs tegenover twee plus
  hervatten naar vijf; exacte CPU-overeenkomst inclusief dropout, data-volgorde,
  gradientdiagnostiek, codeveranderingen en checkpointselectie.
- Control en MX beginnen bij exact dezelfde initialisatie en samplevolgordes.
- Verkeerde contracten, corrupte checkpoints en overschrijven worden geweigerd.
- Stopbestand rondt de huidige epoch af voordat de run stopt.
- Eerste strikte maximum selecteert het checkpoint; gelijke score houdt eerdere epoch.
- Originele LR-mijlpalen 50/75/90 getest met honderd goedkope parameterupdates;
  geen verkort vijf-epoch-LR-schema ingevoerd.
- MX-exports na training gedecodeerd en gereconstrueerd zonder doel-mastergewichten.
- Resultatennotebook uitgevoerd, inclusief de optionele controlvergelijking, op
  opgeslagen synthetische rapporten. Figuurlayout visueel gecontroleerd.
- CLI wijst CPU-productie, honderd epochs, warm-start via run en resume zonder
  checkpoint af. Pakketmanifest en hergebruikhashes zijn gecontroleerd.

Machineleesbaar bewijs: `validation_report.json`, `cli_guards.json`.
Herhalen: `python local_validation/test_training.py <NIEUWE_TESTMAP>`.
De enige CPU-mocking zit in het testscript; productie kent geen CPU/subset-bypass.

## Niet geslaagd / niet uitgevoerd

De echte productiepreflight (`run_train.py check`) is hier aangeroepen maar werd
terecht geblokkeerd door de bestaande containerafhankelijkheden: MoviePy 2.2.1
vereist Pillow<12, terwijl hier Pillow 12.3.0 aanwezig is. Daarnaast meldt pip een
niet-beschrijfbare cache; die is uitgeschakeld. Geen pakketten gewijzigd om dit te
omzeilen. Zie `container_pip_check.txt` en `container_preflight_report.json`.
Dit conflict is niet vastgesteld op de laptop van de gebruiker.

De Windows/CUDA-check moet op de laptop slagen. CPU-exactheid bewijst geen
bitexacte CUDA-herhaling. De productiecheck voert eerst codec- en twee mini-update-
tests uit op kunstmatige tensors; pas de aparte run verwerkt SVHN-data.
Een technische PASS bewijst geen goede eindaccuracy, convergentie, MX-masterloze
updates, truncatierobuustheid of hardwareversnelling.
