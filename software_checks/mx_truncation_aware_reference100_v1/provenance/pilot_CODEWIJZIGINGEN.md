# Van directe tweemodus-MX naar truncatiebewust trainen

## 1. Numerieke ingreep: trunc_adapter.py

Voorheen: Q_E2M3(G(W)) en Q_E2M1(G(W)) afzonderlijk.
Nu: H=encode_E2M3(G(W)), hoge uitvoering decode(H), lage uitvoering
`decode_E2M1(H.codes >> 2, H.scale_codes)`.

`truncate_codes` behoudt teken/exponent/schaal en ondersteunt alle64 broncodes.
`_CachedCodeSTE` geeft de exact gedecodeerde waarde terug en geeft in backward
identity naar het actuele G(W), geen schaal-/codegradient en geen factor1/4.
De FP32-master wordt door Adam bijgewerkt zoals tevoren.

`shared_source_pairs` maakt voor elke minibatch zes hoge broncaches, een per
MX-laag. De lage pass gebruikt diezelfde objecten; geen hoge herencoding en
geen lage encoder. De cache bevat detached codes, niet de hoge autograd-grafiek.
Daardoor is geen retain_graph=True nodig na de eerste backward. Alleen G(W)
wordt in elke pass opnieuw differentieerbaar berekend zoals in de baseline.
De update mag pas na beide passes plaatsvinden. Tensor._version is uitsluitend
een gecontroleerde mutatiewacht (getest met torch2.10), geen getalsbewerking.

## 2. Trainingslus: trunc_support.py

Gecontroleerde fork van dual_support.py; de originele drie selectorwijzigingen
blijven identiek. De echte auteurs-forwardfunctie, loss, volgorde van backwards,
detached softmax en enkele optimizerstep blijven AST-identiek.
Een context rond die forward beheert de broncache en registreert het aantal
hoge encoderingen/lage afleidingen. `layer_snapshot` meet nu werkelijke
getrunceerde codes in plaats van directe E2M1-codes. De masters worden niet
dubbel geteld; dezelfde masterupdate verschijnt in beide modusrijnen.

## 3. Runner en exports

run_train.py is een fork van de vijf-epoch-v1.1-runner met een uniek contract.
Data-, logging-, stop-/hervatkern en optimizer onveranderd. Oude checkpoints
worden door het afwijkende contract geweigerd. Testbroncontroles komen in
history.json. Nieuwe resultatenmap: mx_runs/mx_truncation_aware_pilot5.

Best en last: een hoge bron opslaan, checksum controleren, herladen, lage
controle-export uitsluitend daaruit afleiden. Beide decoders worden tegen
het actieve trainingsgewicht vergeleken; geen extra volledige testpass.
Hoge export is voldoende om beide vaste emulatoruitvoeringen te reconstrueren.
Er is een nieuwe export-schema-ID om oude directe exports niet stilzwijgend
als truncatiebewust te behandelen.

## 4. Tests

De verbeterde v1.1-testdiagnostiek is behouden in test_trunc_training.py:
productiereplay diagnostisch, deterministische replay verplicht, geen
verruimde tolerantie. Getest: logits, gradients, Adam, RNG en BN.
trunc_invariants.py test daarnaast dat de lage encoder niet wordt aangeroepen,
er precies zes hoge encoderingen per paar zijn en de lage gradient de master
bereikt. Mutaties tussen forwards en onvolledige bronparen moeten stoppen.

De README geeft check en run. PROTOCOL.md maakt alle onderzoekskeuzes expliciet;
COMPATIBILITY.md bevat uitsluitend uitvoerings-/omgevingseigenschappen.
