# Protocol — absolute E2-schaal vast na epoch 50, vervolg tot 100

## Onderzoeksvraag en status

Deze verkennende vervolgproef onderzoekt of het gunstige E2-accuracysignaal uit de vijfepochpilot behouden blijft tot epoch 100. De gebruiker kiest opnieuw starten vanuit het oorspronkelijke epoch-50-checkpoint. De ingreep en eindgrens worden vóór uitvoering van deze nieuwe run vastgelegd. De keuze voor het vervolg volgt na inspectie van de pilot en is geen vooraf geregistreerde onafhankelijke bevestigingsstudie.

De doorleercontrole is de onafhankelijke-E2-run `20261111_172102_961497_joint_run`, afgerond tot epoch 100. Haar referentiebestanden worden byte-identiek hergebruikt. De eerdere freeze-pilot dient als technische en verkennende onderbouwing; haar epochs 51–55 worden niet overgenomen in de nieuwe run.

## Vaste afspraken

| Onderdeel | Afspraak |
|---|---|
| Dataset/model | SVHN train/test; geen extra-split; CNN-8 svhnq, expand=2 |
| Modi | E4M3 / MXFP8, E2M3 / MXFP6, E2M1 / MXFP4 |
| Activaties/masters | A32; FP32-mastergewichten, gradients en Adam-momenten |
| Groepering | B32; ongewijzigde groepsindeling en staartvulling |
| Ouder | Oorspronkelijk volledig opgeslagen epoch 50 |
| Geërfde geschiedenis | Alleen epochs 1–50 van de referentierun |
| Nieuwe training | Epochs 51–100, in totaal 50 |
| E2-schaal | Absolute phi en harde E8M0-code van epoch 50 exact vast |
| E4-schaal | Bestaande dynamische berekening uit het groepsmaximum |
| Gewichten en BatchNorm | Blijven leren; dezelfde drie actieve BN-banken |
| Adam | Beide groepen herstellen; schaalstaat daarna onveranderd |
| Gewichts-LR | 0,0001 op 51–75; 0,00001 op 76–90; 0,000001 op 91–100 |
| Schaal-LR | Nul vanaf epoch 51 |
| Loss | E4 hard CE; lagere modi recursive detached soft CE, T=1 |
| Minibatch | Drie passes, gradients optellen, één Adam-stap |
| Controle | Bestaande onafhankelijke-E2-run met schalen leren tot 100 |

Parentcheckpoint SHA-256:
`d6c6d7281f39f818ae7f7a61324362858c518ac5e9ebeb37def9b66ea0e1654a`

De ouder heeft 28.650 optimizerstappen voltooid. Vijftig nieuwe epochs voegen 28.650 gewichtupdates en 85.950 trainingspasses toe. Het volledige traject heeft na epoch 100 in totaal 57.300 updates. De eerste vijftig epochs worden hier niet opnieuw getraind.

## Numerieke route en freeze

De E4-schaal wordt uit de actuele effectieve gewichten G(W) berekend. Voor een niet-nulgroep zonder schaalbegrenzing geldt `e4 = floor(log2(amax)) - 8`. De bestaande E4M3-codec en rounding blijven gelijk.

De absolute E2-schaal is `S2 = 2**round_half_up(phi)`. E2M3 en E2M1 delen deze schaal. De initialisatie uit E4+6 gebeurde vóór de oorspronkelijke training. Voor de vervolgtraining worden de schalen exact uit het epoch-50-checkpoint overgenomen; tijdelijke waarden die bij het opbouwen van het model ontstaan, worden volledig overschreven.

De truncatieroute blijft:
```text
E4M3: s e3 e2 e1 e0 m2 m1 m0
E2M3: s e3 e2       m2 m1 m0
E2M1: s e3 e2       m2
```

Na volledig herstel krijgt phi `requires_grad=False` en `grad=None`. De schaalgroep blijft bestaan met LR nul. Haar Adam-momenten, staptellers, phi-waarden en harde schaalcodes moeten bij iedere nieuwe update exact gelijk blijven aan epoch 50. De projectiehook wordt verwijderd. Alleen nul-LR instellen zou Adam-momenten niet noodzakelijk vast houden.

De gewichts-STE, afgeleiden van G(W), codec, bitmapping, loss en gedeelde E4-bron blijven ongewijzigd. E4 wordt dynamisch berekend en is geen nieuwe leerbare schaalparameter. De verhouding S2/S4 en de lagere elementcodes mogen veranderen.

## Herstel en behoud

De runner controleert de vaste parenthash, het oorspronkelijke contract, de modelhash, geschiedenis, softwareomgeving, datasets en samplevolgordes. Het volledige model, Adam, de scheduler en RNG worden hersteld voordat de nieuwe training begint. Er begint geen nieuwe vijftigepoch-LR-cyclus.

Het pakket heeft een eigen childcontract en uitvoermap. Een nieuwe `run` herstelt de oorspronkelijke epoch 50. `resume` accepteert alleen checkpoints uit het nieuwe pakket. Een nette onderbreking stopt na een volledig opgeslagen epoch; een gedeeltelijke run mag geen voltooiing tot 100 claimen. Het cuDNN-beleid biedt geen garantie op bitexacte CUDA-herhaling.

De volledige parentcheckpointmap blijft nodig voor export van een eventueel globaal beste checkpoint vóór de freeze. Alle oorspronkelijke checkpoints, ruwe data en historische referenties blijven ongewijzigd.

## Vergelijking en rapportage

De hoofdvergelijking gebruikt dezelfde beschikbare epoch voor freeze en doorleren, uiteindelijk 100 tegenover 100. Markeer de overgenomen geschiedenis 1–50 en nieuwe epochs 51–100 afzonderlijk.

Rapporteer de laatste werkelijk voltooide epoch, de drie modi op hetzelfde globaal gekozen beste-E4-checkpoint, en de drie modi op hetzelfde beste-E4-checkpoint binnen uitsluitend de nieuwe vertakking. Een globaal optimum vóór epoch 51 toont geen effect van verder trainen met bevroren schalen.

Accuracyvergelijkingen hebben varianten op de as, afzonderlijke precisiepanelen en percentages bij gelijke cirkels. De assen bevatten alle getoonde punten. Nieuwe epochtijd, historische tijd en hun som worden apart benoemd; er worden geen honderd nieuw uitgevoerde epochs geclaimd.

De officiële testset wordt ook voor checkpointselectie gebruikt. Een geselecteerde beste accuracy is daarom geen onafhankelijk eindtestresultaat. Er is één ouderrun en één vertakking; opeenvolgende epochs zijn geen onafhankelijke herhalingen.

## Kosten en grenzen

Per blok blijven één E4-schaalbyte en één absolute E2-schaalbyte nodig: samen 8.160 bytes voor 4.080 blokken. De freeze verandert deze compacte opslagkost niet. Trainingscheckpoints bewaren phi en de schaal-Adamstaat.

De numerieke adapter blijft gelijk. Tijdelijke schaalgradientberekeningen kunnen nog worden uitgevoerd terwijl phi geen gradient accumuleert. Deze proef bewijst dus geen volledige uitschakeling van alle schaaltrainingskosten en geen hardwarewinst.

De geschatte duur is 50 × 118,045 seconden = circa 98,4 minuten, afgeleid van de eerdere pilot. Alle nieuwe full100-accuracies en de werkelijk gemeten full100-branchtijd zijn bij levering nog onbekend.
