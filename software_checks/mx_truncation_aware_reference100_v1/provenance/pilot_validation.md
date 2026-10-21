# Uitgevoerde validatie

## Resultaat en afbakening
PASS voor de synthetische CPU-regressietests. Linux, Python3.13.5,
torch2.10.0+cpu, torchvision0.25.0+cpu, NumPy2.3.5, SciPy1.17.0.
Geen echte SVHN-training, geen GPU-training, geen gemeten onderzoeksaccuracy.
De CUDA-check moet nog op de eigen laptop slagen. Geen pakketten gewijzigd.

Gecontroleerd:
- Bestaande E2M3-codec en directe E2M1-codec regressies behouden.
- Alle64 E2M3-broncodes tegen onafhankelijke algebraische truncate/decode-oracle;
  teken/exponent/E8M0-bytebehoud, signed zero, subnormals, schaalgrenzen en padding.
- De 1,375-voorbeeldwaarde wordt getrunceerd naar1,0, NIET afgerond naar1,5.
- Identity-STE door het volledige lage codepad; de lage loss bereikt de
  FP32-master via de oorspronkelijke G(W)-gradient, zonder deling door4.
- Exact zes high.encode_tensor-aanroepen per gezamenlijke minibatch; beide
  passes gebruiken dezelfde bronobjecten. low.encode_tensor EN low.encode_elements
  zijn tijdens die test bewust geblokkeerd. Ook testvolgorde4->6 gecontroleerd.
- Mutatie van een master tussen de twee passes en onvolledige bronparen geven
  een fout; caches worden bij fouten opgeruimd. Geen source-cache in state_dict.
- Handmatige gezamenlijke training versus de bron-forward met drie selectors:
  logits, gradients, model en Adam exact gelijk op CPU; RNG gelijk. De bestaande
  CUDA-toleranties blijven staan; geen versoepeling na de eerdere GPU-fout.
- Vijf mini-epochs volledige runner versus ongeinstrumenteerde lus exact gelijk
  voor parameters, optimizer, scheduler, RNG en metrics. Beide BN-banken leren;
  ongebruikte banken blijven intact. Hoge/lage grads sommeren zonder /2.
- Twee plus drie hervatte mini-epochs gelijk aan vijf doorlopende; dropout,
  beide BN-banken, samplevolgorde en selectiegeschiedenis inbegrepen.
- Afwijkend checkpointcontract, corrupte checksum en overschrijven geweigerd.
- Stopmarker rondt lopende epoch af. Checkpointselectie: eerste strikte E2M3-piek,
  niet de losse lage piek; beste en laatste rollen afzonderlijk.
- LR-mijlpalen50/75/90 met oorspronkelijke scheduler op een testparameter getest;
  geen echte honderd-epochtraining.
- Hoge bron opslaan/herladen, lage export uitsluitend daaruit. In apart proces
  stonden alleen hoge bronbestanden en verwachte logits ter beschikking; geen
  checkpoint of laag exportbestand. Encoder- en checkpointtoegang waren
  geblokkeerd. Vier logitvergelijkingen exact: best/last x hoog/laag.
- Onbekende exportvelden, verborgen FP32-masters en corrupte bronchecksums geweigerd.
- Resultatennotebook uitgevoerd op opgeslagen synthetische JSON/CSV; geen
  training vanuit de notebook. Voorbeeldfiguur visueel gecontroleerd.
- Configverschillen met directe v1.1 zijn beperkt tot methode/schema/export-
  metadata. Dataset, initialisatie, loss, BN, Adam en LR onveranderd.

## Ontwikkelfouten en correcties
De eerste bouw had een dubbele komma in config; py_compile detecteerde dat.
De eerste notebooktest ontdekte dat de runmap-parametercel BASELINE_DIR niet
initialiseerde. Beide fouten zijn gecorrigeerd; volledige tests daarna opnieuw
uitgevoerd (laatste testrun validation3). Dit waren fouten in nieuwe lokale
code, niet in bestaande onderzoeksdata of eerdere experimenten.

## Volledige productie-CLI in deze container
De echte aanroep van run_train.py check stopt zoals bedoeld bij pip check:
MoviePy2.2.1 verlangt Pillow<12, terwijl Pillow12.3.0 is geinstalleerd.
Ook een lokale pip-cachewaarschuwing is geregistreerd. Geen dependency is
gewijzigd en geen productiecheck is omzeild. Zie container_pip_check.txt en
container_preflight_report.json. Dit betreft NIET de Windows-omgeving van de
gebruiker. De CPU-regressies hierboven werden onafhankelijk uitgevoerd met de
uitsluitend in local_validation gebruikte Tensor.cuda-testvervanging.

## Reproduceren van lokale softwaretests
Gebruik uitsluitend NIEUWE outputmappen:

    python local_validation/test_training.py <nieuwe_synthetische_map>
    python local_validation/test_source_only.py <die_map> <nieuwe_bron_only_map>

Deze commando's zijn voor offline softwarevalidatie; echte training gebeurt
alleen via de productie-CLI uit README.md. Voor een geplande laptoprun eerst
check uitvoeren. Falen is geen reden om toleranties te verhogen of data te wijzigen.
