# Protocol — absolute E2-schaal bevriezen na epoch 50

## Vraag en onderzoeksstatus

Deze verkennende ablatietest is gekozen na inspectie van de volledige onafhankelijke-schaalrun. Zij onderzoekt het effect van stoppen met de expliciete optimalisatie van de absolute E2-schaal vanaf het voltooide epoch-50-checkpoint. Interventie en stop op epoch 55 zijn vóór uitvoering van deze vertakking vastgelegd. Accuracywinst is geen technische PASS-voorwaarde.

De parentrun is `20261111_172102_961497_joint_run`, met status PASS en 100 epochs. De feedbackbundel heeft SHA-256 `fb0c3b30896d4a5a9fea540338a92668ecc37cc27d4e68b3818fa1f70baf2837`. De bestanden in `independent_full100_reference/` zijn ongewijzigde kopieën. `PARENT_RUN_REVIEW.md` beschrijft de waarnemingen en hun beperkingen.

## Vaste afspraken

| Onderdeel | Afspraak |
|---|---|
| Dataset/model | SVHN train-only/test; CNN-8 `svhnq`, expand=2 |
| Modi | E4M3, E2M3/MXFP6 en E2M1/MXFP4 |
| Activaties/masters | A32; FP32-mastergewichten, gradients en Adam-momenten |
| Groepering | B32; oorspronkelijke indeling en staartvulling |
| Vertakpunt | Volledig opgeslagen epoch 50 |
| Nieuwe epochs | 51–55; geen nieuwe initialisatie of herhaling van 1–50 |
| Parentcheckpoint SHA-256 | `d6c6d7281f39f818ae7f7a61324362858c518ac5e9ebeb37def9b66ea0e1654a` |
| E2-schaal | Phi en harde E8M0-code van epoch 50 exact vast |
| E4-schaal | Bestaande dynamische berekening uit groepsmaximum |
| Gewichten/BN | Leren verder; gebruikte drie BN-banken behouden |
| Optimizer | Bestaande Adamstaat herstellen; geen reset |
| Gewicht-LR | 0,0001 in 51–55, volgens het oorspronkelijke schema |
| Schaal-LR | Nul na herstel; geen schaalgradient of schaal-Adamupdate |
| Loss | E4 hard CE; lagere modi recursieve detached soft CE, T=1 |
| Minibatch | Drie passes, gradients optellen, één gewichtupdate |
| Controle | Bestaande onafhankelijke-schaalrun met doorleren tot 100 |

De parent heeft 28.650 updates voltooid. De pilot voegt 2.865 updates en 8.595 passes toe. De trajectteller bereikt 31.515 updates en epoch 55; dit betekent niet dat 55 nieuwe epochs zijn uitgevoerd.

## Numerieke route

De E4-schaal wordt uit de effectieve gewichten G(W) bepaald. Voor een niet-nulgroep zonder schaalbegrenzing geldt `e4 = floor(log2(amax)) - 8`. E4-broncodes worden met de bestaande codec berekend.

De E2-schaal is `S2 = 2**round_half_up(phi)`. Bij de freeze wordt phi niet opnieuw geïnitialiseerd, afgerond of vervangen. Haar FP32-waarde en de afgeleide schaalbytes blijven exact behouden. E2M3 en E2M1 delen deze schaalbytes.

De elementcodes blijven uit dezelfde E4-bron ontstaan:

```text
E4M3: s e3 e2 e1 e0 m2 m1 m0
E2M3: s e3 e2       m2 m1 m0
E2M1: s e3 e2       m2
```

De bitmapping, identity-STE naar G(W), afgeleiden van G(W) en verliesberekening veranderen niet. De verhouding S2/S4 kan veranderen omdat S4 dynamisch blijft. Een constante absolute E2-schaal betekent niet dat de gequantiseerde elementcodes vaststaan.

## Herstel en freeze-invarianten

Het parentcheckpoint wordt vóór gebruik gecontroleerd op hash, epoch, contract, configuratie en omgeving. Eerst wordt de volledige model-, Adam-, scheduler- en RNG-staat hersteld. Daarna wordt de freeze toegepast, zonder nieuwe datavolgorde of reset van gewichtmomenten.

Phi krijgt `requires_grad=False` en `grad=None`. De schaalgroep blijft aanwezig met LR nul. Haar momenten en staptellers moeten gelijk blijven aan de herstelde staat. Nul-LR alleen is geen voldoende waarborg: Adam zou dan nog momenten kunnen aanpassen. Daarom worden ook afwezige gradients en onveranderde schaalstaat gecontroleerd.

Gedurende de nieuwe updates blijven phi, harde E2-exponenten en schaal-Adamstaat exact vast. De actieve gewichtgradients blijven eindig en de gewicht- en BN-updates gaan door. Schaalgradientdiagnostiek betekent na de freeze expliciet uitgeschakeld, niet een gemeten nulgradient.

Hervatten binnen de vertakking vereist haar eigen contract en herstelt de freezetoestand. Het oorspronkelijke `cudnn.benchmark=True`-beleid blijft behouden. Een geslaagde CPU-hervatproef bewijst geen bitexacte CUDA-hervatting.

## Vergelijking en rapportage

De bestaande controle wordt hergebruikt. De train/testsamplevolgordes van epochs 51–55 worden vergeleken. Vergelijkingen gebruiken hetzelfde beschikbare epochvenster; de historische controle tot 100 wordt niet als freeze-resultaat behandeld.

De notebook rapporteert het gedeelde startpunt op 50, het werkelijke eindpunt, de drie modi op het gemeenschappelijke beste-E4-checkpoint en de selectie binnen uitsluitend de nieuwe vertakking. Zo kan een geërfd optimum vóór 50 de nieuwe proef niet verbergen. Volledige curves markeren de overgenomen geschiedenis 1–50 en de freezegrens; een tweede weergave toont detail van de nieuwe epochs.

Accuracyvergelijkingen hebben varianten op de verticale as, accuracy op de horizontale as en afzonderlijke panelen voor de drie precisies. Bij ieder punt staat het percentage. De assen omvatten alle getoonde waarden. Absolute E2-schalen, geobserveerde verhoudingen, E4-bronveranderingen en freeze-invarianten blijven afzonderlijk zichtbaar.

De gemeten tijd van de nieuwe epochs wordt apart gehouden van de overgenomen looptijd. Een eventuele raming voor het resterende traject is expliciet een extrapolatie. De pilot heeft geen eigen epoch-100-accuracy.

Selectie op de officiële testset is geen onafhankelijke testschatting. Dit is één startpunt en één vertakking, geen vergelijking over onafhankelijke seeds. De eerdere schaalwissels en accuracyschommelingen zijn aanleiding, geen causaal bewijs. Een verbetering na freeze bewijst ook niet dat epoch 50 de optimale freezegrens is.

## Kosten en behoud

De E4-schaalbank plus één absolute E2-byte per groep kosten samen 8.160 bytes voor 4.080 groepen: 4.080 bytes extra tegenover vaste ×64 en 2.040 bytes extra tegenover de gepackte tienkeuzevariant. Elementcodes, randlagen, BN en headers komen apart. Freeze verandert deze compacte opslagkost niet.

Trainingscheckpoints behouden phi en schaal-Adamstaat voor herleidbaarheid en herstel. De numerieke adapter blijft gelijk; er wordt geen besparing in latency, energie of GPU-geheugen gegarandeerd. Die kosten moeten worden gemeten. Bestandsgrootte alleen bepaalt geen geheugentransfers.

Alle uitvoer staat in nieuwe resultatenmappen. Parentcheckpoints, ruwe data en historische referenties worden niet overschreven. De feedbackbundel bevat rapporten, histories en code, maar geen dataset of volledige checkpoints. Dit pakket eindigt uiterlijk op epoch 55; verdere training wordt na de pilot afzonderlijk beoordeeld.
