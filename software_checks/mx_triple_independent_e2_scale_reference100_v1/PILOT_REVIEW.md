# Beoordeling van de werkelijke onafhankelijke-schaalpilot

## Herkomst en integriteit

Bron: `feedback_bundle(20261111-160359).zip`, SHA256 `01dd87d6ef0cd882c02d51e20990466d8dc5328743de5b40f55e875726506d06`, run `20261111_165230_352318_joint_run`. De relevante ruwe bestanden staan ongewijzigd in `independent_pilot_reference/` en zijn met controlesommen vastgelegd.

De run heeft status PASS, vijf volledige epochs, 2.865 Adam-updates en 8.595 trainingspasses. De gebruikte adapter, trainingsondersteuning, runner en notebookhelper komen byte voor byte overeen met het geleverde pilotpakket. Geschiedenis, CSV's en geselecteerde metrics zijn onderling gecontroleerd. De basisgewichten en train/testsamplevolgorde komen overeen met de historische relatieve tienkeuzepilot. De enige twee waarschuwingen zijn bestaande PyTorch-deprecationmeldingen over `pin_memory` en `is_pinned`; er is geen trainingsfout gerapporteerd.

## Accuracy op epoch 5

| Variant | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| Vaste ×64, eerste vijf epochs van eerdere volledige training | 93,0086% | 91,2992% | 90,8920% |
| Tien relatieve keuzes, eerdere pilot met start ×64 | 92,6552% | 91,4259% | 91,6026% |
| Onafhankelijke E2-schaal, nieuwe pilot | 92,6898% | 91,6219% | 91,5335% |

Bronnen: `epoch_metrics.csv`, `history.json`, `selected_metrics.csv` en de twee historische referentiemappen in de feedbackbundel. Alle drie nieuwe accuracies nemen gedurende de vijf epochs toe. Epoch 5 is ook het gemeenschappelijke beste E4-checkpoint.

Tegenover de tienkeuzepilot zijn de verschillen +0,0346, +0,1959 en −0,0691 procentpunt, respectievelijk +9, +51 en −18 correcte testbeelden van 26.032. Tegenover vaste ×64 zijn de verschillen −0,3188, +0,3227 en +0,6415 procentpunt. Dit is een beschrijvende vergelijking van afzonderlijke runs met één seed; de pilot bewijst geen algemene of uiteindelijke accuracywinst.

## Wat leerden de schalen?

Bronnen: `scale_block_state.csv`, `scale_epoch_metrics.csv`, `scale_source_changes.csv` en `history.json`. Alle 24.480 blokregels zijn onafhankelijk herberekend.

- Alle 4.080 FP32-hulpparameters φ verschillen op epoch 5 van hun beginwaarde.
- Slechts 10 van de 4.080 groepen (0,245%) hebben op dat moment een andere afgeronde absolute E2-exponent dan bij initialisatie.
- Bij 2.037 groepen (49,93%) veranderde de E4-bronexponent; bij 2.041 veranderde de geobserveerde verhouding E2/E4.
- Er waren 124 harde E2-wisselgebeurtenissen, inclusief herhaald heen en terug schakelen. Dat zijn geen 124 verschillende groepen.
- De eerste drie epochs bevatten geen harde E2-wissels. Er waren geen grensprojecties; alle gerapporteerde schaalgradientnormen zijn eindig.

| Absolute E2-exponent | Bij initialisatie | Na epoch 5 |
|---|---:|---:|
| −9 | 66 | 72 |
| −8 | 3.710 | 3.703 |
| −7 | 304 | 304 |
| −6 | 0 | 1 |

| Geobserveerde verhouding E2/E4 na epoch 5 | Groepen | Aandeel |
|---|---:|---:|
| ×16 | 29 | 0,71% |
| ×32 | 1.678 | 41,13% |
| ×64 | 2.039 | 49,98% |
| ×128 | 293 | 7,18% |
| ×256 | 41 | 1,00% |

Deze verhoudingen zijn achteraf uit de twee schaalbanken berekend. De spreiding ontstaat hier vooral door verandering van de E4-bronschaal; zij toont geen brede verkenning van absolute E2-exponenten aan. Bijvoorbeeld: alle 288 groepen van `layers.13` eindigen op ×32, terwijl hun absolute E2-exponenten ongewijzigd bleven en hun E4-bronexponenten veranderden.

Het eindbereik van φ is −9,426300 tot −6,497450, ruim binnen [-127,127]. De gemiddelde absolute verandering sinds initialisatie is 0,134591; de grootste 0,556479. Een continue parameterupdate hoeft de afrondingsgrens niet meteen te overschrijden. De waarnemingen passen bij actief leren met nog weinig harde schaalveranderingen, zonder bewijs van convergentie of optimaliteit.

## Tijd en beslissing

Bronnen: `run_report.json`, `epoch_times.csv` en `timing_estimate.json`. De sessie duurde 344,65 seconden; de opgetelde epochtijd 335,95 seconden. De mediaan van epochs 2–5 was 67,48 seconden. De gerapporteerde projectie voor 100 epochs is 6.746,65 seconden, ongeveer 1 uur 52 minuten. Het geobserveerde epochbereik geeft een projectie van ongeveer 1 uur 51 tot 1 uur 54; dit is geen betrouwbaarheidsinterval.

**Vervolgkeuze:** dezelfde numerieke route 100 epochs vanaf nieuwe initialisatie trainen, met φ actief tot en met epoch 100. De langere looptijd onderzoekt of de beperkte vroege schaalveranderingen doorzetten. Er wordt in deze run geen freeze, extra schaalnormalisatie, andere initialisatie of aangepaste leersnelheid toegevoegd. Alle checkpoints blijven behouden, zodat een latere freeze vanaf epoch 50 afzonderlijk onderzocht kan worden.
