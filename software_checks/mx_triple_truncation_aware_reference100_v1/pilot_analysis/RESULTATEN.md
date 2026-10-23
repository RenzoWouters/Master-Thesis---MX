# Resultaten van de vijf-epochpilot met exponent- en mantissatruncatie

Bron: `feedback_bundle(20261023-162654).zip`, run `20261023_182117_516609_joint_run`.

## Conclusie

De pilot is technisch geslaagd. Alle drie precisies leren verschillende cijfers herkennen met de ongewijzigde naïeve bitbewerking en gekopieerde E4M3-bronschaal. Na vijf epochs blijft de accuracy ongeveer elf procentpunt achter op de directe referentie. De volledige honderd-epochproef is nodig om vast te stellen hoeveel daarvan blijft bestaan.

## Vergelijking op epoch 5

| Modus | Directe pilot | Truncatietraining | Verschil [procentpunt] |
|---|---:|---:|---:|
| E4M3 | 93,11% | 82,08% | -11,03 |
| E2M3 | 93,07% | 81,50% | -11,57 |
| E2M1 | 92,80% | 81,86% | -10,94 |

De beginmodelhash, dataset, preprocessing, backendinstellingen en alle tien samplevolgordehashes van beide pilots komen overeen. De GPU-backend garandeert geen bitexacte herhaling. Dit is een vergelijking tussen twee trainingen met één seed.

## Verloop van de testaccuracy

| Epoch | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| 1 | 78,22% | 80,60% | 80,31% |
| 2 | 77,55% | 79,31% | 79,51% |
| 3 | 80,56% | 80,86% | 80,85% |
| 4 | 81,72% | 82,38% | 82,61% |
| 5 | 82,08% | 81,50% | 81,86% |

E4M3-best en last zijn beide epoch 5 en hebben dezelfde checkpoint-SHA. De twee hoofdgroepen van de pilotnotebook mogen dus dezelfde scores tonen. De eigen maxima van E2M3 en E2M1 liggen op epoch 4 (82,3794% en 82,6060%); de gezamenlijke selectieregel blijft die van E4M3.

## Wat de voorspellingen laten zien

Alle drie modi voorspellen alle tien cijfers in elk van de vijf epochs. Op epoch 5 is cijfer 1 het meest voorspeld: 19,68% bij E4M3, 18,33% bij E2M3 en 18,49% bij E2M1. Er is dus geen instorting naar één vaste voorspelde klasse.

In de eerdere post-hocproef voorspelden exponent86 en combined864 op zowel epoch 100 als het beste E4M3-checkpoint steeds cijfer 2: 4.149 correct op 26.032, of 15,9381%. Die proef gebruikte andere, reeds getrainde gewichten en vastgehouden BN-toestand. Het verschil met deze nieuwe pilot is daarom geen effectmeting op hetzelfde checkpoint. De nieuwe run past zowel de gedeelde masters als de BN-banken aan; hun afzonderlijke bijdragen zijn hier niet gemeten.

Ook E4M3 zelf blijft achter bij de directe pilot. Alle drie modi delen immers de FP32-master en dragen bij aan de optimizerupdate. De gemeten achterstand betreft de gezamenlijke trainingsmethode. Mogelijke verklaringen zoals aanpassing van BN of wisselwerking tussen de gradients vereisen afzonderlijke proeven voordat er causale uitspraken mogelijk zijn.

## Technische controle en tijd

2.314 onafhankelijke consistentiecontroles zijn geslaagd. De run bevat vijf volledige epochs, 2.865 Adamupdates, 8.595 trainingspasses en 5.730 controles van losgekoppelde teachers. Broncycli, schaalbehoud, FP32-beleid, BN-tellingen, datasetdekking en alle opgeslagen metriektabellen zijn consistent. Er zijn geen technische blokkers.

De gemeten som van epochtijden is 272,565 s (4 min 33 s); de sessie duurde 277,934 s (4 min 38 s). De projectie voor een verse honderd-epochrun is 89,87 minuten. De projecties op basis van het waargenomen tempo lopen van 85,98 tot 94,46 minuten; dit is geen betrouwbaarheidsinterval en voorspelt geen accuracy.

## Herkomst en grenzen van de audit

`raw/` bevat de ongewijzigde rapporten en meettabellen van de nieuwe pilot. `direct_reference_raw/` bevat de relevante ongewijzigde directe-pilotmetingen. `independent_audit.json` bevat controles en getelde vergelijkingen. De bronfeedback-SHA staat in die audit. Alle 52 geregistreerde pakketbestanden matchen het geleverde pilotpakket.

De feedbackbundel bevat geen grote trainingscheckpoints of compacte MX-payloads. Hun tensors zijn hier niet opnieuw uitgevoerd; de audit controleert de intern consistente runnerbewijzen voor exports, BN, gradients en Adam. De aparte CUDA-checkrapportage zit niet in deze runbundel. Tijdens de analyse zijn geen data gewijzigd en is geen echte training gestart.
