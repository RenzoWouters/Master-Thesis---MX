# Dedicated E2M1: wat verandert er?

Basis: mx_weights_single_reference100_v1.zip (niet de tweemodustrainingslus).

1. e2m1_adapter.py: fork van bestaande mx_adapter.py met import uit mx_e2m1 in plaats
   van mx_format, plus uniek exportschema en correcte labels. G(W), A32, BN32 en
   originele Conv2d blijven identiek. De zes FP32-parameters blijven optimizerparameters.
2. mx_e2m1.py: byte-identieke codec uit mx_weights_dual_reference100_v1.zip.
   RNE direct vanuit G(W), geen hoge tussenkwantisatie en geen bittruncatie.
3. single_support.py: alleen imports van adapter/codec en formaatlabel veranderen.
   Originele auteurs-forward blijft AST-identiek en gebruikt selector[32] voor A32/BN32.
   De distillatieloss is verboden; er is een hard-label loss en een update per minibatch.
4. run_train.py: unieke protocollen/paden/labels en E2M1-export; run_epochs AST-identiek.
   CHECK test nu E2M1. FULL RUN honderd epochs, aparte nieuwe initialisatie.
5. E2M1-tests: onafhankelijke 16-codewaardentabel, ties, schalen, padding, STE,
   geen E2M3-tussenstap, exportreconstructie. CPU-mini-epochs en resume testen volledige lus.
6. Notebook: dedicated labels en scores uit echte rapporten; geen nieuwe accuracy vooraf.
   voeg_toe_aan_overzicht.py maakt een nieuwe kopie van het bestaande overzicht en
   voegt gevalideerde E2M1-resultaten toe, nooit door bestaande bronnen te overschrijven.

De behouden bank32 is de enige BN-bank, zoals bij dedicated E2M3; de naam32 betekent
niet dat de gewichten W32 zijn. Geen pruning van ongebruikte banken; opslag blijft vergelijkbaar.
Nieuwe schemas weigeren per ongeluk hervatten vanuit E2M3-, gezamenlijke of pilotcheckpoints.
Zie DEDICATED_CHANGE_MANIFEST.json en dedicated.patch voor exacte wijzigingen.
