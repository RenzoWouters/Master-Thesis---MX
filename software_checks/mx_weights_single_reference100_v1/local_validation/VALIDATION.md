# Uitgevoerde validatie van de honderd-epochversie

## Afbakening

Status van de synthetische regressie: PASS. Status van de volledige productiecheck
in deze container: FAIL wegens een reeds aanwezig afhankelijkheidsconflict,
zie afzonderlijke afbakening onderaan. Deze twee controles niet verwarren.

Omgeving synthetische regressie: Linux, Python 3.13.5, PyTorch 2.10.0+cpu,
torchvision 0.25.0+cpu, NumPy 2.3.5, SciPy 1.17.0. Geen NVIDIA-GPU hier.
Dit is niet de Windows/Python-patchversie/CUDA-omgeving op de laptop.
Alle leervoorbeelden waren kunstmatig: 6 trainvoorbeelden en 4 testvoorbeelden,
2 optimizerupdates per mini-epoch. Er is GEEN echte SVHN-training uitgevoerd.
Mini-epochaccuracies en tijden zijn GEEN onderzoeksresultaten of GPU-ramingen.

## Geslaagd, nieuw uitgevoerd

- Bevestigd: AST van run_epochs identiek aan de geslaagde pilot; alle gedeelde
  modules en het auteursarchief byte-identiek. Alleen wrapper en rapportage aangepast.
- Eerdere codec-/modeltest en twee mini-updates voor beide modi opnieuw uitgevoerd.
- Honderd mini-epochs voor control EN MX met echte CNN-8, gedeeltelijke laatste
  batch, dropout, BN32, Adam en het onverkorte LR-schema 50/75/90.
- Per modus vergeleken met honderd mini-epochs via de oorspronkelijke niet-
  geinstrumenteerde trainingsfunctie. Bij control is ook het originele model
  gebruikt, niet uitsluitend onze wrapper. Model, optimizer, scheduler, CPU-RNG
  en alle train-/testmetrics waren exact gelijk tussen beide uitvoeringen.
- Beide modi: honderd onveranderlijke checkpoints; beste/laatste checksums kloppen,
  laatste=100, juiste LR-overgangen en voltooiings-/gemeten-tijdsrapporten.
- Hervatten vanaf de echte opgeslagen mini-epochcheckpoints 50,75,90 en 99 naar
  de volgende epoch vergeleken met de doorlopende run: model, Adam, scheduler,
  RNG, metrics, samplevolgorde en updatecontroles exact gelijk op CPU.
- Hervatting vanaf 99 voltooit export op 100; invoercheckpoint blijft ongewijzigd.
- Init-hashes en alle 200 train/test-volgordehashes gelijk tussen control en MX.
- Verkeerde modus, pilotschema, corrupt checkpoint en overschrijven afgewezen.
- Stopmarker werkt na een volledige epoch; export noemt dit niet voltooid.
- Selectieregel behoudt de eerste epoch bij exact gelijke score.
- MX-best/last-export reconstrueert het model zonder zes FP32-masterparameters.
  Dit is een synthetische reconstructie, geen extra volledige SVHN-evaluatie.
- Notebook op honderd-epochrapporten uitgevoerd, inclusief gematchte vergelijking
  en afzonderlijke beste-/epoch100-tabellen. Gegenereerde figuren visueel nagekeken.
- CLI weigert 5,0,101 epochs, warme start via run, resume zonder checkpoint en CPU.
- CLI-dispatch apart gemockt: run --mode control en run --mode mx kiezen standaard
  ieder 100 epochs vanaf verse initialisatie. Deze mock doet geen echte training.

Bewijsbestanden: validation_report.json, test_stdout.txt, cli_guards.json.
Testscript: test_reference100.py. Het script wijzigt alleen tijdens zijn eigen
CPU-testproces Tensor.cuda. De productiecode heeft GEEN CPU/subset-bypass.

Herhalen (alleen ontwikkelomgeving met notebook-testafhankelijkheden):

    python local_validation/test_reference100.py <NIEUWE_map_voor_synthetische_test>

## Productiepreflight in de container

Een echte `run_train.py check` is geprobeerd, zonder checks te omzeilen en zonder
packages te wijzigen. De aanwezige MoviePy 2.2.1 vereist Pillow <12, terwijl
Pillow 12.3.0 geinstalleerd is. Daarom stopt de bestaande pip-check terecht.
CUDA is hier bovendien niet aanwezig. Zie container_preflight_report.json en
container_pip_check.txt. Dit is niet de gebruikersomgeving en geen resultaat
van training op de laptop. Op de laptop moet `check` eerst PASS geven.

Geen bewijsclaim voor bitexacte CUDA-hervatting. De backend blijft
cudnn.benchmark=True zoals de pilot; de werkelijke volledige accuracies en
looptijden moeten uit de nieuwe laptopruns komen.
