# SVHN CNN-8 — tien schaalkeuzes, start ×1, honderd epochs

Deze referentietraining onderzoekt de invloed van de initiële schaalfactor over het volledige trainingsschema. Alle 4.080 FP32-schaalparameters beginnen op `theta=0`, dus factor ×1. Daarna blijven alle tien factoren ×1, ×2, ×4, ×8, ×16, ×32, ×64, ×128, ×256 en ×512 leerbaar tot en met epoch 100. De gewichten blijven eveneens leren. Er is geen bevriezing op epoch 50.

`run` begint opnieuw vanaf dezelfde oorspronkelijke FP32-gewichten en BN-toestand. De geslaagde vijf-epochproef wordt niet als checkpoint ingeladen. Het numerieke pad en de instellingen zijn identiek aan die proef; alleen de geplande duur, experimentidentiteit en rapportage zijn aangepast.

## Bevindingen uit de afgeronde pilot

De init×1-pilot uit `feedback_bundle(20261110-164812).zip` is technisch geslaagd. Op epoch 5, tevens haar E4M3-beste checkpoint, zijn de testaccuracies:

| Precisie | Start ×1 | Eerdere vijf-epochpilot met start ×64 |
|---|---:|---:|
| E4M3/A32 | 80,962% | 92,655% |
| E2M3/A32 | 81,538% | 91,426% |
| E2M1/A32 | 81,719% | 91,603% |

De laatste factorenverdeling is ×1: 1.389 groepen, ×2: 2.679 groepen en ×4: 12 groepen; de overige zeven factoren zijn nog niet gekozen. Van de 4.080 continue schaalparameters zijn er 4.031 positief. De vroege achterstand is dus niet gelijk aan volledige stilstand van het schaalleren. Deze vijf epochs bepalen niet of de achterstand na honderd epochs verdwijnt.

Bronnen: de ongewijzigde `pilot_reference/history.json` en `init64_reference/history.json`; de controle en exacte cijfers staan in `pilot_analysis`. Er zijn twee bekende PyTorch-deprecationwaarschuwingen over het `device`-argument van pinned-memoryfuncties; geen runtimefout.

De tijdprognose uit deze pilot bedraagt ongeveer **1 uur en 40 minuten** voor een nieuwe volledige run, bij vergelijkbare laptop- en backendomstandigheden. Dit is een extrapolatie, geen gemeten duur van honderd epochs. De nieuwe notebook vergelijkt de volledige training met de afzonderlijke oorspronkelijke **init×64-training van honderd epochs** in `init64_full100_reference`; haar vroege accuracies kunnen door CUDA-herhaling iets verschillen van bovenstaande vijf-epochpilot.

## Uitvoeren in PowerShell

Plaats de map `mx_triple_learned_scale10_init1_reference100_v1` naast de bestaande pakketten onder `software_checks`.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_init1_reference100_v1\run_train.py" check

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_init1_reference100_v1\run_train.py" run
```

Voer `run` uit nadat `check` PASS meldt. `check` gebruikt synthetische data en traint niet op SVHN. `run` gebruikt de bestaande SVHN-train- en testbestanden. Alleen een expliciete `--download` staat toe dat ontbrekende bestanden worden opgehaald. Bestaande data en eerdere resultaten worden niet overschreven.

Elke opdracht schrijft een nieuwe map onder `mx_runs/mx_triple_learned_scale10_init1_reference100`.

## Gelijke trainingsinstellingen

- CNN-8 `svhnq`, expand=2; volledige SVHN train-split zonder extra; batch 128; seed 20261009.
- E4M3 → E2M3 via behoud van de hoogste exponentbits; E2M3 → E2M1 via mantissatruncatie.
- Groepen van 32, A32, FP32-mastergewichten, FP32-gradients en Adam-toestanden.
- De E4-bronschaal blijft uit de actuele gewichten bepaald. Beide E2-modi delen één leerbare verhouding tot die schaal.
- Identieke drie passes met recursieve distillatie en één Adam-update per minibatch.
- Identieke schaalgradient en projectie naar [0,9], zonder LSQ-normalisatie.
- Gewichts- en schaal-learningrate 0,001; LR-mijlpalen 50/75/90 met gamma 0,1. Beide parametergroepen blijven tot epoch 100 actief.

×1 is een relatieve factor: bij de start is `S_E2 = S_E4`. De absolute groepsschalen worden niet op 1 gezet. Dit is nog steeds gekoppelde schaalkeuze, geen proef met onafhankelijke schalen per exponentbreedte.

## Notebook en vergelijking

Elke resultatenmap bevat de onuitgevoerde notebook `MX_triple_learned_scale10_init1_reference100_resultaten.ipynb` en `notebook_support.py`. Voer de notebook vanuit die map uit. De oorspronkelijke init×64-training van 100 epochs is meegeleverd als gecontroleerde referentie. De recente init×1-pilot en de oorspronkelijke init×64-pilot zijn eveneens als ongewijzigde onderzoeksdata bewaard.

De rapportage bewaart de accuracy- en lossontwikkeling van de drie precisies, alle tien factoraandelen, de continue theta-ontwikkeling en grens-/gradientdiagnostiek. Zij rapporteert epoch 100 en het gemeenschappelijke checkpoint dat op de hoogste E4M3-accuracy gekozen werd. Een gedeeltelijke run toont zijn werkelijke laatste epoch en wordt niet als een voltooide 100-epochrun gepresenteerd.

`initialization_comparison.csv` vergelijkt uitsluitend gemeenschappelijke epochs. De runner controleert dezelfde oorspronkelijke basisgewichten/BN en dezelfde train-/testvolgorden. Ontbrekende historische epochs worden nooit geëxtrapoleerd. De bronbestanden en checksums staan in `init64_full100_reference`, `init64_reference` en `pilot_reference`.

## Onderbreken en hervatten

Elke volledig afgewerkte epoch krijgt een checkpoint. Een eerste Ctrl+C vraagt een stop na de lopende epoch; een tweede Ctrl+C stopt direct. Een bestand `STOP_AFTER_EPOCH` in de nieuwe resultatenmap vraagt eveneens een stop na de lopende epoch.

Voor een vooraf geplande onderbreking op epoch 50:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_init1_reference100_v1\run_train.py" run --stop-after-epoch 50
```

Dit is alleen een uitvoerstop: de schaalparameters worden niet bevroren. Hervat met het werkelijk opgeslagen checkpoint:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_init1_reference100_v1\run_train.py" resume --checkpoint "<volledig pad naar epoch_00050.pt>"
```

Alleen checkpoints van dit exacte pakket en experiment zijn geldig. Pilot-, init×64- en freeze-checkpoints worden niet geaccepteerd. `resume` bewaart de gewichten, theta's, Adam-status, scheduler, RNG en volledige epochgeschiedenis, en schrijft naar een nieuwe resultatenmap. Een eigen checkpoint dat al epoch 100 bevat, wordt niet nogmaals doorgetraind.

## Interpretatie en validatie

Een technische PASS vereist volledige dekking, eindige en consistente numeriek, juiste optimizerupdates en integere opslag. Een minimale accuracy of verplichte factorenverdeling is geen PASS-voorwaarde. Lage accuracy of blijvende keuze ×1 zijn geldige onderzoeksresultaten. De continue theta-waarden zijn nodig om beweging onder een harde keuzedrempel zichtbaar te houden.

De vergelijking gebruikt één seed en historische runs. De oorspronkelijke `cudnn.benchmark=True` blijft behouden; gelijke initialisatie en datavolgorde garanderen daarom geen bitexacte GPU-herhaling. Selectie van het beste checkpoint gebruikt de officiële testset en is geen onafhankelijke validatieschatting. Numerieke MX-emulatie op FP32-convoluties levert geen directe hardware- of energieclaim op.

De ontwikkelaarsvalidatie staat in `local_validation` wanneer het pakket is verzegeld. Ze gebruikt uitsluitend kleine synthetische CPU-fixtures. Hier is geen nieuwe SVHN- of CUDA-training uitgevoerd. Deel na de eigen run de nieuwe `feedback_bundle.zip`; checkpoints en dataset blijven lokaal.
