# Lokale controle van de tienkeuze-pilot

De implementatie is gecontroleerd met PyTorch 2.10.0 en torchvision 0.25.0 op CPU,
Python 3.12.14. Er is hier geen echte SVHN-training uitgevoerd en geen CUDA-timing
gemeten. De productieopdrachten gebruiken de bestaande Python 3.13/CUDA-omgeving
van de gebruiker; `check` controleert die omgeving en de numerieke route lokaal.

## Numerieke kern

`core_check_report.json` rapporteert 21 geslaagde inhoudelijke controles en
herstel van RNG/backend-instellingen. Alle 254 eindige E4M3-codes zijn met de
tien offsets gecontroleerd. Ook de negen halve drempels en hun directe FP32-buren,
de exacte oude keuzes binnen [5,6], de gewichts- en schaalgradienten, caching,
projectie na Adam, geldige E8M0-schalen en vierbit-opslag zijn onderzocht.

De lage modi worden exact gereconstrueerd uit de bronexport, zonder de
FP32-mastergewichten, theta of een nieuwe encoder aan te roepen. De test omvat
ook uitsluitend factor ×1, oneven aantallen keuzes en ongeldige nibbles 10–15.

## Volledige runner op synthetische beelden

`training_integration_report.json` gebruikt de echte runner/model/loss/Adam/
scheduler/checkpoints/exports met slechts drie synthetische beelden per split.
De oorspronkelijke productieconfiguratie en gegevens worden niet veranderd.

Gecontroleerd zijn onder meer: vijf epochs, drie passes en één update per batch,
kleine eindbatches, alle tien factoren, 2.040 gepackte keuzebytes, alle drie
accuracies uit dezelfde E4M3-selectie, epoch-5-uitvoer en behouden oude referenties.
Een stop na epoch 2 gevolgd door hervatten tot 5 geeft dezelfde model-, Adam-,
scheduler- en RNG-staat en dezelfde schaal-CSV als de ononderbroken CPU-run.
Oude contracten en voltooide checkpoints worden vóór updates geweigerd.
De productie-CLI weigert meer dan vijf epochs en een CPU-productierun.

`check_entry_validation_report.json` controleert afzonderlijk de echte `check`-
entry met een uitsluitend voor deze test aangepaste CPU-runtime.

Deze synthetische uitkomsten zijn geen SVHN-accuracies of GPU-looptijden.
De oorspronkelijke CUDA-backend biedt geen bitexacte hervatgarantie.

## Protocol en presentatie

`CONFIG_DIFFERENCES.json` toont de wijzigingen tegenover de binaire pilot.
Alle overige protocolvelden zijn gelijk, inclusief B32, A32, FP32, initialisatie,
batchgrootte, optimizer, schaalgradient en het volledige LR-schema.
De originele pakketbestanden en meegeleverde referentiemetingen zijn behouden.

De notebookcontrole staat in `notebook_validation_report.json` en de aanvullende
notebookrapporten. De opgeleverde notebook bevat geen uitgevoerde cellen of
testfiguren. `PACKAGE_SHA256.json` bindt runtimecode, protocol en referentiedata;
het opslaan van Jupyter-uitvoer in de notebook wijzigt het trainingscontract niet.
