# Codewijzigingen

## Hergebruik

De bestaande codecs, MX-adapter, drieprecisie-adapter/support, bronselectie,
dataset-/runtimefuncties, resume-infrastructuur en upstream-ZIP blijven
byte-identiek aan `mx_weights_triple_direct_pilot5_v1`. De eerder geslaagde
E4M3 exact-binary-v2-codec blijft behouden. Zie `REUSE_MANIFEST.json`.

`posthoc_truncation.py` is byte-identiek aan de bitbewerkingmodule uit
`mx_triple_posthoc_truncation_eval_v1`. De nieuwe trainingsadapter gebruikt
haar exponent- en mantissafuncties rechtstreeks. De historische scripts en
protocollen staan ter referentie in `provenance`; ze worden niet als runner
uitgevoerd.

## Nieuwe trainingsmodules

`trunc_adapter.py` bouwt op de bestaande drieprecisie-adapter. De zes MX-lagen
krijgen een E4M3-broncache en twee exacte codeafleidingen. Iedere pass heeft
een eigen G(W)-graph; de identity-STE-backward blijft dezelfde benadering als
in de eerdere tweemodus-mantissatraining. De cache controleert identiteit en
versie van de master, codes en schalen en wordt voor Adam geleegd.

Dezelfde module schrijft een compact E4M3-bronformaat en leidt de twee lagere
exports alleen uit opgeslagen codes/schalen af. De normale BN-bankselectie,
architectuur en FP32-uitzonderingslagen blijven behouden.

`trunc_support.py` behoudt de oorspronkelijke forward-AST behalve de bestaande
selectorwijzigingen. Zij plaatst de broncyclus om die lus heen en controleert
per minibatch de drie passes, twee detached teachers en één Adam-update.
Training werkt BN normaal bij; test verandert de modelstaat niet. De
laagdiagnostiek codeert alle modi via dezelfde E4M3-bron.

Nieuwe testdiagnostiek telt voorspellingen per cijfer, zonder RNG of gradients
te veranderen. De eerdere eis dat ten minste één MX-masterwaarde per epoch
moest wijzigen is vervangen door rapportage: de optimizer-, gradient- en
eindigheidscontroles blijven verplicht, maar eindige nulupdates kunnen een
onderzoeksuitkomst zijn.

`test_trunc_training.py` controleert bitpatronen, schaalbehoud, STE,
broncachefouten, onafhankelijke handmatige recursieve training en export-
reconstructie op synthetische input. Een lage accuracy is geen zelftestcriterium.

## Runner en rapportage

`run_train.py` is afgeleid van de bestaande directe drieprecisiepilot.
Dataset, verse initialisatie, vijf-epochstop, honderd-epoch-LR-schema,
checkpointselectie, stop/resume en behoud van oude uitvoer blijven gelijk.
Nieuw zijn het truncatiecontract, broncontroles, klassevoorspellingstellingen
en de E4M3-bronexport met twee daarvan afgeleide exportbestanden.

Het broncontract en checkpointschema onderscheiden deze training van de
directe referentie. Resume van een ander experiment wordt geweigerd. De
nieuwe `TRAINING_CHANGE_MANIFEST.json` legt de bewuste veranderingen vast.
Een diff tegenover de directe runner staat in `provenance`.

## Notebook

De enige notebooktemplate en haar helper gebruiken de nieuwe protocolnaam en
routebeschrijvingen. De twee accuracygroepen zijn epoch 5 en het gezamenlijk
gekozen E4M3-bestcheckpoint. Naast de eerdere accuracy-, loss-, tijd- en
geheugenfiguren en de gewichtsdiagnostiek is er een figuur voor voorspelde
klassendiversiteit en dominantie.
Alleen de lege template wordt meegeleverd. De helper voert geen modellen,
training of checkpointwijzigingen uit.
