# Uitgevoerde controles

Alle onderstaande controles zijn geslaagd. Ze onderscheiden de audit van
de aangeleverde echte trainingsrapporten van synthetische softwarecontroles.
Er is bij het maken van dit pakket geen echte SVHN-evaluatie of training
uitgevoerd. De daadwerkelijke bronexports en dataset staan op de laptop.

## Audit van de volledige referentietraining

`provenance/reference100_feedback_audit.json` bevat 636 geslaagde controles op
`feedback_bundle(20261023-150530).zip`, met SHA256
`68b55b2432bdc6802094e8968ad5d97252b8166ebbca4d1651fc438c6c66e259`.

Gecontroleerd zijn onder meer de vijftig oorspronkelijke packagehashes,
brontransformaties, overeenstemming tussen de 600 metricrijen en history,
leertraject/updates, checkpointselectie en tijden. De echte gerapporteerde
training eindigde op epoch100; E4M3-best is epoch54. De bronrapporten bevatten
geen modeltensors of SVHN-arrays. Runtimecontroles op gradients, BN en exports
zijn daarom gecontroleerd als uitspraken van de gehashte oorspronkelijke code,
niet door die echte tensors hier opnieuw te laden.

## Bitconversies en bevroren modellen

`codec_and_frozen_model_validation.json` beschrijft controles met
Python3.12/Linux/torch2.10.0+cpu, zonder SVHN of optimizerupdates:

- alle 254 eindige E4M3-codes tegen onafhankelijke bit- en getalorakels;
- alle 64 E2M3-codes voor mantissatruncatie en afronding;
- alle 255 eindige E8M0-schaalcodes ongewijzigd in alle vijf afgeleide routes;
- signed zero, subnormals, padding, nulblokken, packing en afrondingsgrenzen;
- expliciete weigering van NaN-codes en veertien foutgevallen;
- traps die aantonen dat de drie bitroutes geen encoder of G(W) aanroepen;
- alle acht modellen op synthetische beelden met A32, juiste doel-BN,
  bevroren parameters en onveranderde volledige modelstaat;
- gelijkheid van live directe modellen en hun exportreconstructies op CPU;
- 48 gewichtsparen en 48 routediagnoses voor zes MX-lagen.

Dit verifieert de betekenis van de bitbewerking; het voorspelt geen accuracy.
De laptop voert dezelfde `test_truncation.run_tests` uit via `check`, op CUDA
en in de bestaande trainingsomgeving.

## Runner en evaluatie als geheel

`posthoc_integration_report.json` en `test_posthoc_integration.py` gebruiken
drie synthetische beelden, twee batches inclusief een laatste batch van één,
en willekeurige directe exports met de twee vaste checkpointrollen. Het
testharnas vervangt uitsluitend omgevings-/datainvoer en referentiemetadata
binnen de test. De productie-CLI heeft geen synthetische of CPU-omzeiling.

De werkelijke `check`- en `evaluate`-besturing is doorlopen. Alle tien
afgeleide modellen zijn opgeslagen en herladen. De test verifieert de zestien
metricrijen, 22 gepaarde vergelijkingen, 96 gewichtsparen, 96 routediagnoses,
zes historische vergelijkingen, NPZ-uitvoer en feedbackbundels. Het aanmaken
van een optimizer of aanroepen van backward is in de test verboden.

De echte referentiecontrole is afzonderlijk getest met zeven negatieve
gevallen. Drie incorrecte payloads zijn ook door de volledige evaluatieroute
geweigerd, ondanks geldige bestandchecksums. Foute CLI-keuzes en uitvoer
binnen de bronmap worden geweigerd. Alle oorspronkelijke inputs en de 22
geteste uitvoeringsbestanden blijven ongewijzigd.

## Onafhankelijke codereview

`independent_posthoc_review.json` beschrijft een afzonderlijke read-onlyreview
van de runner, evaluator en bronafleiding. Het daadwerkelijke referentieprotocol
en de packagehashes zijn vergeleken. Er zijn geen concrete blokkerende
bevindingen. De beperkingen van de feedback als bewijs zijn vastgelegd.

## Notebook

`notebook_validation_report.json` beschrijft uitvoering van alle negen
codecellen op synthetische complete uitvoer, inclusief vijf figuurtypen en
negatieve integriteitsgevallen. Figuren zijn visueel gecontroleerd. De
verificatie houdt rekening met geldige topk-voorspellingen bij gelijke logits.

`runner_notebook_integration.json` test daarna de door de echte synthetische
CLI-run gekopieerde notebook tegen die daadwerkelijke runneruitvoer. Beide
rollen, acht routes, NPZ-checksums, CE, voorspellingen en gepaarde tellingen
slagen zonder analysefouten. Vijf figuren zijn in geheugen gemaakt; ruwe
invoer blijft onveranderd. Er wordt uitsluitend een lege notebook geleverd.

## Assemblage en resterende praktische controle

`ASSEMBLY_AUDIT.json` legt vast dat de definitieve uitvoeringsbestanden
overeenkomen met de geteste hashes, de veertien hergebruikte bestanden
byte-identiek zijn, alle rapporten PASS hebben, de notebook leeg is en er
geen synthetische model-/resultaatbestanden in het pakket zijn opgenomen.
`PACKAGE_SHA256.json` dekt het gehele definitieve pakket behalve zichzelf.

De lokale controles zijn op CPU uitgevoerd. De eerste gebruikersstap is
`check` op de laptop met de oorspronkelijke CUDA-omgeving. De daaropvolgende
`evaluate` leest de echte exports en de volledige SVHN-testset. De bestaande
cuDNN-backend blijft behouden en garandeert geen bitexacte herhaling; de
proef meldt eventuele verschillen met historische directe accuracies.
