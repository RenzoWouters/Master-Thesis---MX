# Echte ×32-pilot: resultaat en besluit

Bron: `feedback_bundle(20261024-173425).zip`, SHA256 `2596b9222da600d814caf797b7b74ce9d3fbc74e14a8197fcbdee831e97d1d8b`. Run: `20261024_192400_420212_joint_run`.

## Technisch resultaat

**PASS.** De echte SVHN-run rondde vijf epochs af met 2.865 Adam-updates en 8.595 trainingspasses. A32, FP32-master/gradients/Adam, de vaste exponentoffset +5 en mantissatruncatie met behoud van de E2M3-schaal zijn bevestigd. De 66 pakketchecksums komen overeen met het afgeleverde pakket. Initialisatie, dataset, transforms, broncode, softwareomgeving en backendinstellingen stemmen overeen met ×64.

Alle vijftien testmetingen voorspellen alle tien cijferklassen. Er is geen klassecollapse. De enige waarschuwingen zijn de twee bekende PyTorch-deprecationmeldingen rond pin_memory/is_pinned.

## Vergelijking op epoch 5

Bij beide trainingen is het E4M3-beste checkpoint binnen de eerste vijf epochs epoch 5. De twee afgesproken pilotcheckpointvergelijkingen vallen dus samen.

| Modus | ×32 accuracy | ×64 accuracy | ×32 − ×64 |
|---|---:|---:|---:|
| E4M3 | 92,113553% | 93,008605% | −0,895052 procentpunt |
| E2M3 | 89,762600% | 91,299170% | −1,536570 procentpunt |
| E2M1 | 90,142901% | 90,891979% | −0,749078 procentpunt |

De harde test-CE is eveneens hoger voor ×32: 0,265791 / 0,335785 / 0,328520 tegenover 0,243082 / 0,298985 / 0,310513 voor E4M3 / E2M3 / E2M1.

E4M3 verbetert gedurende deze vijf epochs. De twee lagere ×32-modi halen hun eigen hoogste pilotaccuracy op epoch 3 en vlakken daarna af. Die modusspecifieke beste scores vervangen de afgesproken gezamenlijke selectie op E4M3 niet. E2M3 met ×32 ligt op epoch 3 tijdelijk boven ×64; de vroege rangschikking is dus niet overal gelijk.

## Interpretatie

×32 werkt technisch en leert, maar heeft op het afgesproken pilotmeetpunt geen voordeel boven ×64. Vijf epochs met één seed, nog vóór de LR-dalingen, bewijzen niet hoe de methoden na 100 epochs vergelijken. De volgende proef is daarom een volledige ×32-training met exact dezelfde numerieke route.

Dat ook E4M3 een andere accuracy heeft ondanks dezelfde broncodec is verenigbaar met het protocol: de drie modi leveren gradients aan dezelfde FP32-mastergewichten. De aangepaste lagere route verandert daarmee het gezamenlijke trainingstraject.

## Looptijd

De totale pilotsessie duurde 321,06 seconden (5 min 21 s). De som van de vijf epochtijden was 315,59 seconden. De projectie voor een nieuwe volledige run bedraagt circa 6.317 seconden, oftewel 1 uur 45 minuten. De waargenomen spreiding van vroege epochtijden geeft ongeveer 1 uur 44 tot 1 uur 52; dit is geen betrouwbaarheidsinterval of gegarandeerde looptijd.

## Bewijsgrens

De feedback bevat rapporten, CSV-bestanden en de notebook, maar niet de grote `.pt`-checkpoints of exporttensors. De audit controleerde de consistentie van de gelogde CUDA-exportchecks en laagtabellen; er is hier geen nieuwe CUDA-inferentie op de echte checkpoints uitgevoerd. De bestaande pilotnotebook is wel in het geheugen op deze echte feedback uitgevoerd; de bronbestanden bleven ongewijzigd. Er is geen nieuwe uitgevoerde notebook afgeleverd.
