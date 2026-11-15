# Verificatie van de honderd-epochproef

## Afbakening

Deze controles toetsen de implementatie en de rapportage. De nieuwe volledige
SVHN-training op CUDA is hier niet uitgevoerd. De meegeleverde notebooktemplate
bevat geen verzonnen resultaten van die training. Synthetische testgegevens en
testfiguren staan buiten het pakket.

## Werkelijke vijf-epochpilot

De oorspronkelijke gebruikersbundel `feedback_bundle(20261115-133745).zip`
is ongewijzigd gebleven. De onafhankelijke controle staat in
`provenance/five_pilot_feedback_audit.json`. Alle 95 manifestbestanden zijn
op hash en grootte gecontroleerd. De 50 metricrijen zijn gecontroleerd tegen
de 3.885 minibatchrecords. Alle 650.800 opgeslagen top-1-voorspellingen zijn
opnieuw geteld. De laatste trainingsstaat heeft 2.865 Adam-updates; de vijf
exports en hun geregistreerde roundtrips zijn gecontroleerd.

De vijf testaccuracies op epoch 5 zijn uit de tellingen:

| Formaat | Correct / totaal | Testaccuracy |
|---|---:|---:|
| E4M3 | 24.286 / 26.032 | 93,292870% |
| E5M2 | 24.310 / 26.032 | 93,385065% |
| E2M3 | 24.292 / 26.032 | 93,315919% |
| E3M2 | 24.316 / 26.032 | 93,408113% |
| E2M1 | 24.278 / 26.032 | 93,262139% |

De twee geregistreerde waarschuwingstypen betreffen verouderde PyTorch-
deviceargumenten bij `pin_memory()` en `is_pinned()`. Er zijn geen gemelde
numerieke fouten. Deze pilot is geen meting van de honderd-epochuitkomst.

## Honderd-epochbesturing

`test_runner_100.py` en `reference100_validation_report.json` leggen de lokale
controles vast, uitgevoerd met Python 3.12.14 en PyTorch 2.10.0+cpu. De echte
CNN-8 doorliep honderd epochs met twee synthetische train- en testvoorbeelden.
Daarbij zijn de daadwerkelijke learningrates voor epochs 1–50, 51–75, 76–90
en 91–100 gecontroleerd.

Hervatting vanaf epoch 50 en vanaf epoch 99 reproduceerde op CPU exact de
model-, optimizer-, scheduler- en RNG-staat en metrics van de aaneengesloten
synthetische run. Een pilotcheckpoint, een reeds voltooid checkpoint en een
gewijzigd losscontract zijn terecht geweigerd. Bij een geïnjecteerde schrijffout
op epoch 100 bleef het bevestigde eindpunt op epoch 99 staan. De vijf exports
van epoch 100 zijn teruggelezen en gecontroleerd. Vier ongeldige CLI-aanroepen
eindigden met exitcode 2 en de globale FAIL-regel.

Dit bewijst geen bitidentieke CUDA-hervatting: het behouden CUDA-beleid staat
niet-deterministische uitvoering toe. De numerieke trainingsfuncties zijn met
de pilot vergeleken; uitsluitend de voortgangstekst in `phase` heeft een nieuw
instelbaar epochtotaal. De veertien byte-identiek hergebruikte bronbestanden
zijn vermeld in `REUSE_MANIFEST.json`.

## Historische referentie en notebook

De historische bron is de oorspronkelijke directe drie-formaattraining,
`feedback_bundle(20261023-150530).zip`, run `20261023_151331_732669_joint_run`.
Alle 600 metricrijen zijn gecontroleerd. De primaire bronbestanden stemmen
byte voor byte overeen met de later opgeslagen referentiekopieën. De directe
epoch-100-uitvoerpaden in die latere evaluatie reproduceren alle 78.096 directe
voorspellingen. Truncatieresultaten zijn niet als baseline gebruikt.

Zie `reference_data/triple_direct_reference_audit.json` voor de verificatie
en haar beperkingen. De ruwe voorspellingen en checkpoints voor alle oude
epochs zijn niet in de oorspronkelijke feedback aanwezig; de controles van
die epochs gebruiken de geaggregeerde meetbestanden en de kruisverwijzingen.

`reference100_test_notebook.py` en `reference100_notebook_report.json` toetsen
tien scenario's met synthetische nieuwe metingen en de echte historische
referentie. Voltooide en onvoltooide runs, ontbrekende gegevens, een
protocolverschil, dubbele rijen, onjuiste accuracies en een gewijzigde
referentiehash zijn gecontroleerd. Alleen een bevestigde nieuwe epoch 100 met
alle vijf formaatmetingen geeft de primaire eindpuntvergelijking. De
vergelijkingsfiguren en curvepanelen zijn visueel gecontroleerd.

De historische vergelijking beschrijft twee volledige trainingsopzetten met
verschillende distillatie, actieve BN-banken en aantallen passes. Zij is geen
causale ablatietest en omvat geen variatie over meerdere seeds.

## Bij de gebruiker

Voer eerst de normale CUDA-opdracht `check` uit. Begin na een geslaagde check
met `train` zonder `--resume`. Alleen de volledige SVHN-run tot epoch 100
levert de nieuwe onderzoeksresultaten. Beide opdrachten schrijven een
feedbackbundel en eindigen met één globale PASS- of FAIL-regel.
