# Werkelijk uitgevoerde validatie: volledige tweemodusrunner

## Resultaat en afbakening

Synthetische CPU-regressies: **PASS**. Linux, Python 3.13.5,
PyTorch 2.10.0+cpu, NumPy 2.3.5. Geen echte SVHN-data, CUDA-training,
onderzoeksaccuracies of GPU-tijdsmeting. De gebruikte synthetische training heeft
zes beelden, vier testbeelden, batch vier en twee minibatches per mini-epoch.
De honderd mini-epochs zijn een softwaretest, niet de gevraagde echte training.

## Geslaagde controles

- Alle oorspronkelijke codec-/STE-/integratiecontroles voor E2M3 en E2M1.
- De v1.1-handmatige distillatielusvergelijking, ongewijzigde toleranties en
  herstel van de backendinstellingen; met CPU, niet met CUDA.
- Honderd mini-epochs met de kale oorspronkelijke dual-forward versus de
  geinstrumenteerde volledige runner: exacte gelijkheid van modelstaat,
  optimizer, scheduler, RNG en iedere epochmetric.
- Echte LR-overgangen na mini-epochs 50, 75 en 90 met dezelfde optimizer.
- Hervatten na 50, 75 en 90 naar epoch100: exacte gelijkheid met ononderbroken
  CPU-uitvoering, inclusief dropout, beide BN-banken, datavolgorde en selectie.
- Honderd afzonderlijke checkpointbestanden, 400 complete metricrijen en
  200 samplevolgorde-bestanden; beide modi op dezelfde best/last-checkpoints.
- Oude pilotcontracten, andere modi, corruptie en overschrijven worden geweigerd.
- Stopbestand voltooit de lopende epoch; gedeeltelijke resultaten claimen geen
  volledige run. De gedeeltelijke notebook is ook uitgevoerd.
- Verkeerde lr_next of ontbrekende modi worden door eindexport geweigerd.
- Beste-checkpointregel kiest eerste strikte maximum op E2M3, niet de lage piek.
- Vier geselecteerde direct-quantisatieexports herladen in een nieuw proces,
  zonder oorspronkelijke checkpoints; eindige modeluitvoer.
- Volledige en gedeeltelijke resultatennotebooks uitgevoerd en de gegenereerde
  grafieken visueel gecontroleerd. Geen modelcalls vanuit de notebooks.
- CLI weigert stop-after-epoch5, een warmstart bij run en resume zonder checkpoint.
- Dertien kernbestanden en de volledige run_epochs-functie zijn byte-identiek
  aan het op de laptop geslaagde pilot-v1.1-pakket.

## Niet geslaagde of niet uitgevoerde onderdelen

Een eerste testaanroep werd door een tooltijdlimiet onderbroken tijdens
synthetische epoch84. Dat was geen assertion failure. Dezelfde volledige test
is daarna opnieuw in een nieuwe map uitgevoerd en volledig geslaagd.

De echte productie-CLI-check in de container stopte bij pip check: de al
bestaande MoviePy 2.2.1 vereist Pillow<12, terwijl Pillow12.3.0 aanwezig is.
Niets is geinstalleerd, aangepast of omzeild. Zie container_preflight_report.json
plus container_pip_check.txt. CUDA is hier niet getest; de laptopcheck blijft nodig.
GPU-bitexactheid is niet aangetoond. De v1.1-diagnostiek in test_dual.py blijft
volledig behouden en vergelijkt op de laptop onder tijdelijk deterministische
instellingen naast de diagnostische vergelijking onder het productiebeleid.

## Zelf herhalen

    python local_validation/test_training.py <NIEUWE_MAP>
    python local_validation/test_check_fix.py

Alleen de offline test vervangt Tensor.cuda tijdelijk om de CUDA-gerichte
forward op CPU te testen. De echte CLI bevat geen CPU- of synthetische bypass.
Gedetailleerde controles: validation_report.json, validation.log,
validation_summary.json, cli_guard_results.json en check_fix_helper_tests.json.
