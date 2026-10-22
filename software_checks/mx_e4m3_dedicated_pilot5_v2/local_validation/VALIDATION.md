# Uitgevoerde lokale controles — v2

Datum: 22 oktober 2026. Dit zijn technische controles op CPU en kleine
synthetische modelcontroles. Geen SVHN-training, nieuwe onderzoeksaccuracy,
GPU-tijdmeting of bevestigde CUDA-uitkomst.

## Omgeving

Afzonderlijke Linux/Python 3.12.14-omgeving, PyTorch 2.10.0+cpu,
torchvision 0.25.0+cpu, twee CPU-threads. De productie-CLI blijft de bestaande
Python 3.13/PyTorch 2.10.0/torchvision 0.25.0-CUDA-omgeving vereisen.

## Codec en foutdiagnostiek — opnieuw PASS

33 exacte controles, 21 verwachte afwijzingen. Alle 254 eindige E4M3-codes,
255 E8M0-factoren en 64.770 code/schaal-producten vergeleken met onafhankelijke
rationele/scalar-oracles. De oorspronkelijke 252 midpoint-ties, 504 aangrenzende
FP64-probes, signed zero, groepering, schaalgrenzen, payloadimmutabiliteit en
identity-STE-controles blijven slagen.

Op CPU geeft ook de oorspronkelijke v1-rekenprobe exacte waarden. De concrete
v1-afwijking op de GPU is nog onbekend en wordt door de nieuwe check vastgelegd.
De hoofdrapporten zijn `codec_report.json` en `legacy_cpu_diagnostics.json`.

`test_cuda_fix.py` injecteert uitsluitend binnen de test een fout van één ULP
bij code 56 en een tekenfout bij -0 (code 128). De nieuwe codeccontrole weigert
beide met afzonderlijke tellingen en exacte bitpatronen. Ook NaN/Inf-diagnostiek
blijft JSON-veilig. De strenge norm is dus behouden.

De echte runner-checkroute is zowel met PASS als met deze geïnjecteerde FAIL
uitgevoerd. Pakket-, bron- en dependencycontroles bleven actief; alleen de
runtime is testgebonden naar CPU geleid. De al afzonderlijk gecontroleerde
CNN-smoke is in deze rapportagetest expliciet gestubd, met nul optimizerupdates.
Beide routes bewaren de codec- en legacyrapporten in de feedbackbundel. Dataset,
trainingslus en export zijn in deze rapportagetest niet benaderd. Uitvoering met
warnings-as-errors. Bewijs: `cuda_fix_validation_report.json` en de testcode.

## Echte CNN-8-smoke — opnieuw PASS

Precies twee synthetische minibatchupdates met het echte meegeleverde CNN-8.
FP32-masters/gradients/Adam, updates in alle zes MX-lagen, uitsluitend BN32
actief en behoud van de G(W)-afgeleide gecontroleerd. Export en reconstructie
zijn exact; het maximale CPU-logitverschil is 0.0. Bewijs: `smoke_report.json`.

Dit is een echte modelcontrole, afzonderlijk van de genoemde rapportagestub.
Er zijn geen SVHN-beelden gebruikt.

## Runner en hervatten — opnieuw PASS

De bestaande `test_training.py` opnieuw uitgevoerd met v2: vier synthetische
trainbeelden en vier testbeelden, vijf mini-epochs, tien updates. Hervatten
vanaf epoch 2 geeft dezelfde CPU-model-, optimizer-, scheduler- en RNG-staat,
metrics en samplevolgorde. Stopmarker, bestselectie, checkpointbescherming en
CLI-beperking tot vijf epochs slagen. De productie-LR-mijlpalen blijven
50/75/90; de kortere mijlpalen bestaan uitsluitend in het synthetische testcontract.
Bewijs: `integration_report.json`.

## Notebook en ongewijzigde onderdelen

De notebook en `notebook_support.py` zijn byte-identiek aan de in v1 gecontroleerde
versie. Ook de adapter, trainingsondersteuning, dataset-/resumeondersteuning en
auteurs-ZIP zijn behouden. De eerdere notebookcontrole omvatte zes gerichte
tests, uitvoering op complete/onderbroken synthetische runneruitvoer en visuele
controle van de figuren. Er is geen nieuwe notebookuitvoer of fictieve accuracy
in v2 opgeslagen. Een ontbrekende epoch 100 blijft expliciet ontbrekend.

## Volgende werkelijke controle

`run_train.py check` op de laptop moet de nieuwe codec en echte modelcontrole
op CUDA bevestigen. Pas een geslaagde check geeft de basis om de vijf echte
SVHN-epochs te starten. V2 is hier niet op een GPU uitgevoerd. De CPU-resultaten
onderbouwen geen GPU-runtime, SVHN-accuracy of bitexacte CUDA-hervatgarantie.
