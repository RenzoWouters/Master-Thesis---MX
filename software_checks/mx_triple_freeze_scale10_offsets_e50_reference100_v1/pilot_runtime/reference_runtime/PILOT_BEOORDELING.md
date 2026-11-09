# Beoordeling van de tienkeuzepilot

**Eigen onderzoeksgegevens — 4 november 2026.** Bron: de aangeleverde
`feedback_bundle(20261104-180536).zip`, run
`20261104_185824_003629_joint_run`. De oorspronkelijke bestanden blijven
ongewijzigd. Een selectie voor vergelijking staat in `pilot_reference/`.

## Uitvoering

De run rapporteert **PASS**: vijf complete epochs, 2.865 Adamupdates en 8.595
trainingpasses. Iedere epoch gebruikt 73.257 trainingsbeelden en 26.032
testbeelden. De drie uitvoermodi zijn E4M3/A32, E2M3/A32 en E2M1/A32, met B32
en FP32-mastergewichten. De onafhankelijke controle bevestigt de aantallen,
de accuracies uit de aantallen correcte voorspellingen en de overeenkomst
tussen de groepsstaten en de schaalhistogrammen. Alle 48 runtimebestanden
komen overeen met de hashes in het runrapport.

Bronnen: `run_report.json`, `training_summary.json`, `epoch_metrics.csv`,
`history.json`, `scale_block_state.csv` en `scale_epoch_metrics.csv` uit de
feedbackbundel. De werkelijke checkpointtensors zijn niet meegestuurd;
de geslaagde exportcontroles zijn vastgesteld uit het runrapport.

## Accuracy op epoch 5

| Configuratie | E4M3 [%] | E2M3 [%] | E2M1 [%] |
|---|---:|---:|---:|
| Vast ×32 | 92,467 | 91,065 | 90,719 |
| Vast ×64 | 93,009 | 91,299 | 90,892 |
| Geleerd ×32/×64 — eerdere pilot | 92,694 | 91,541 | 91,311 |
| Geleerd tien opties — nieuwe pilot | 92,655 | 91,426 | 91,603 |

De nieuwe pilot selecteert **epoch 5** als beste gemeenschappelijke checkpoint
op basis van E4M3. Dat is dezelfde epoch als het laatste checkpoint.
Het verschil met de eerdere binaire pilot bedraagt respectievelijk −0,038,
−0,115 en +0,292 procentpunt. De vaste referenties gebruiken hier alleen hun
metingen op epoch 5; hun resultaten op epoch 100 zijn niet met de pilot gemengd.

Bronnen: de huidige `epoch_metrics.csv` en `checkpoint_selection.json`, plus
de `epoch_metrics.csv`-bestanden onder
`fixed_references/{scale32,scale64,learned_binary_pilot5}/`.

**Interpretatie:** de accuracies ontwikkelen normaal, maar deze ene pilot toont
geen algemene winst van tien opties aan. De initiële basisgewichten en seed
zijn gelijk aan die van de binaire pilot; de CUDA-uitvoering is niet bitexact
deterministisch. De vergelijking is verkennend, zonder herhaalde seeds of
statistische significantieclaim. Selectie op de officiële testset levert geen
onafhankelijke testschatting op.

## Schaalkeuzes

| Einde epoch | ×32 | ×64 | ×128 | Alle overige factoren samen |
|---:|---:|---:|---:|---:|
| 0 | 0 | 4.080 | 0 | 0 |
| 1 | 0 | 4.080 | 0 | 0 |
| 2 | 0 | 4.080 | 0 | 0 |
| 3 | 0 | 4.079 | 1 | 0 |
| 4 | 7 | 4.072 | 1 | 0 |
| 5 | 24 | 4.051 | 5 | 0 |

Alle 4.080 continue theta-parameters veranderen, ook wanneer de discrete keuze
hetzelfde blijft. Op epoch 5 ligt theta tussen 5,4459 en 6,5118. Alle zes MX-lagen
ontvangen niet-nul schaalgradiënten. Er zijn 319 keuzeveranderingen geregistreerd,
inclusief herhaalde heen-en-weerwissels, en geen projecties tegen de grenzen 0 of 9.

**Interpretatie:** ×128 is aantoonbaar bereikbaar. Initialisatie op ×64 en de
korte duur verklaren waarom 99,29% van de groepen nog ×64 kiest. Ongebruikte
opties zijn hiermee niet afgeschreven. Ook honderd epochs vormen geen
gegarandeerde globale zoektocht: dit blijft lokale optimalisatie met een
benaderde gradient.

Bronnen: `scale_block_state.csv`, `scale_epoch_metrics.csv` en
`scale_learning_summary.json`.

## Tijd en kosten

De vijf gemeten epochs inclusief tests, controles en checkpoints duren
338,816 seconden (5 minuten 39 seconden). De eerdere binaire pilot duurde
325,981 seconden. Het geobserveerde verschil is circa 13 seconden (+3,94%);
dit is geen geïsoleerde of herhaalde benchmark van alleen de schaalkeuze.
De projectie voor een verse run van honderd epochs is 6.823,746 seconden
(ongeveer 1 uur 54 minuten), onder vergelijkbare omstandigheden. Dit is geen
gemeten volledige trainingsduur en geen betrouwbaarheidsinterval.

Er blijven 4.080 FP32-hulpparameters en drie passes per minibatch. De analytische
extra opslag voor theta, gradient en twee Adam-momenten is 65.280 bytes, exclusief
framework en tijdelijke buffers. De compacte selectorinformatie is vier bits
per groep: 2.040 bytes, tegenover 510 bytes voor de eerdere twee keuzes. Dit
is 1.530 bytes extra voor de flexibele bronrepresentatie; een enkel afgeleid
precisieformaat gebruikt zijn resulterende E8M0-schalen.

Bronnen: `epoch_times.csv`, `timing_estimate.json`,
`scale_learning_summary.json` en de overeenkomstige eerdere pilotbestanden.

## Besluit voor de vervolgproef

De pilot geeft geen technische reden om het recept aan te passen. De volgende
stap is een **verse training vanaf nul tot epoch 100**, met dezelfde tien keuzes,
initialisatie op ×64, optimizer, learningrates en LR-overgangen na 50/75/90.
De offsets leren tot en met epoch 100; er is geen automatische freeze op epoch 50.
Iedere complete checkpoint wordt bewaard, inclusief de toestand na epoch 50
voor een mogelijke latere freeze-tak. De pilot wordt niet gebruikt als warmstart.

Er is bij het samenstellen van het nieuwe pakket geen SVHN- of GPU-training gestart.
