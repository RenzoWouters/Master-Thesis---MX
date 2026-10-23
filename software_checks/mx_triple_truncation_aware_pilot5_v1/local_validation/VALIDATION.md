# Softwarevalidatie van de vijf-epochpilot

De validaties gebruiken uitsluitend synthetische beelden en willekeurige
kleine fixtures. Er is geen echte SVHN-training, GPU-training of onderzoeks-
accuracy uitgevoerd bij het voorbereiden van dit pakket. De productie-CLI
heeft geen CPU- of synthetische omzeiling; lokale testharnassen vervangen
alleen binnen de controle de benodigde runtime/data-invoer.

## Kern: bitbewerking, broncache en training

`core_training_validation.json` en `test_core.py` beschrijven de uitvoering
van de echte `test_trunc_training.run_tests` op Python 3.12/Linux met
torch 2.10.0+cpu. De controle is geslaagd:

- Alle 254 eindige E4M3-codes tegen een onafhankelijke bitstring-orakel;
  signed zero, alle 255 eindige E8M0-codes, padding en NaN-weigering.
- Exacte forwardwaarden en identity-STE tegenover de oorspronkelijke
  G(W)-afgeleide voor iedere modus.
- De oorspronkelijke forward-AST blijft identiek afgezien van de drie
  bestaande selectorparen.
- Twee recursieve updates vergeleken met een onafhankelijk geschreven
  handmatige lus en eigen bitstringconversie: op CPU exact dezelfde outputs,
  teachers, gradients, BN, Adam, RNG en modelstaat.
- Zes daadwerkelijke E4M3-encoderingen per minibatch; lagere encoders zijn
  tijdens deze controle vervangen door functies die onmiddellijk falen.
- Verkeerde volgorde, onvolledige cyclus, gewijzigde master en gewijzigde
  broncodes worden geweigerd; caches worden ook na fouten opgeruimd.
- Opslag en herladen van E4-bron en beide afleidingen, gevolgd door model-
  reconstructie terwijl G(W) en alle encoders verboden zijn.
- Gelijke exportlogits op synthetische beelden, juiste doel-BN, afwezige
  zes MX-masters en ongewijzigde modelstaat tijdens evaluatie.
- Eindige nulveranderingen van MX-masters worden zonder FAIL gerapporteerd.
- RNG en backendinstellingen zijn na de controle hersteld.

De laptop voert deze zelftest opnieuw op CUDA uit via `check`. Bij CUDA blijft
de bestaande werkwijze behouden: productiebackend als diagnose en een
verplichte deterministische synthetische vergelijking met de bestaande
toleranties. De training zelf behoudt de oorspronkelijke backendinstellingen.

## Werkelijke runner en stop/resume

`test_training_integration.py` voert de echte runner uit met drie synthetische
beelden in batches van twee en één. Alleen in deze fixture liggen de LR-dalingen
na epochs 2/3/4 om binnen vijf mini-epochs de overgang bij resume te controleren.
Het productieprotocol blijft batch 128, volledige SVHN, mijlpalen 50/75/90.

De controle omvat een ononderbroken vijf-epochrun, hervatten na epoch 2 tot 5,
een gecontroleerde stop na epoch 1, weigering van een ander experimentcontract
en weigering van hervatten voorbij het pilot-eindpunt. De CPU-beginstaat wordt
vergeleken met de eerdere overeenkomstige directe CPU-fixture. Dat is
afzonderlijk van een historische beginhash uit een echte laptoprun.

Gecontroleerd worden onder meer optimizer-/forward-/teacher-/broncyclustellingen,
checkpoint- en sidecarintegriteit, LR, sampledekking, BN, FP32-masters en
Adam-momenten. De E4-bron en twee afgeleide exports per gekozen checkpoint
worden herladen; alle drie modi kunnen uitsluitend uit de E4-bron worden
gereconstrueerd. Het classificatieprobleem blijft tienklassen; iedere
testmodus houdt exact tien klasseaantallen bij.

De test controleert ook de nieuwe voorspellingstabel tegen de epochgeschiedenis,
de lege notebookkopie, feedbackbundels, onveranderde inputs en CLI-weigeringen.
Een afzonderlijke check-entrycontrole gebruikt de echte `create_context`-
checkroute met alleen een CPU-runtimepatch en zonder datasetgebruik.

## Notebook

`notebook_validation_report.json` beschrijft vier synthetische situaties:
complete vijf epochs, vroeg gestopt, blijvend één voorspelde klasse en check-only.
Twaalf negatieve controles bewaken protocol, checkpointkeuze, metrics,
voorspellingstellingen en exportbeleid. Alle tien codecellen en zes figuren
zijn uitgevoerd; de figuren zijn visueel gecontroleerd.

`runner_notebook_integration.json` bevestigt daarna de aansluiting op de
daadwerkelijke synthetische runneruitvoer: dertig train/testmetricrijen,
vijftien voorspellingstellingen, beide checkpointgroepen en de bron-/afgeleide
exportmetadata. De zes figuren zijn in geheugen gemaakt. De ruwe inputbestanden
blijven ongewijzigd. Alleen één lege notebooktemplate wordt geleverd.

## Onafhankelijke review en assemblage

`independent_runner_review.json` bevat de read-onlyreview van de nieuwe runner
en haar aansluiting op de core. Er zijn geen blokkerende bevindingen.
`ASSEMBLY_AUDIT.json` controleert de definitieve geteste bestandhashes, het
byte-identieke hergebruik van veertien basisbestanden plus de post-hocmodule,
de lege notebook en de afwezigheid van synthetische modellen/data in het ZIP.

De rapporten zijn softwarebewijs binnen de beschreven scope. Zij geven geen
garantie dat naïeve exponenttruncatie na training een gunstige accuracy zal
bereiken. Die empirische uitkomst komt uit de laptoprun.
