# Beoordeling van de echte vijf-epochpilot

Bron: `feedback_bundle(20261025-134757).zip`, SHA-256 `d911f57c76220211ab76cbac0144cc78f026ee532de45ec7d53eea0a6c83ae8d`. De geselecteerde ongewijzigde bronbestanden staan in `pilot_reference`; `AUDIT.json` bevat de onafhankelijke controle en exacte cijfers.

## Technische uitkomst

De echte RTX 5070 Laptop GPU-run voltooit vijf epochs, 2.865 Adam-updates en 8.595 trainingspasses. De protocol- en pakketgegevens, A32/FP32-beleid, broninitialisatie, bron-/schaalaudits, voorspellingstellingen en checkpointselectie zijn consistent. De opgeslagen 24.480 blokstaten en dertig laag-epochrijen zijn onafhankelijk herberekend. De FP32-basisinitialisatie is gelijk aan beide vaste referenties.

De feedbackbundel bevat rapportage over compacte exports en checkpoints, maar geen `.pt`-bestanden. De rapportage is hier gecontroleerd; de daadwerkelijke exports/checkpoints kunnen uit deze bundel niet opnieuw worden geladen.

## Accuracies op epoch 5

| Training | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| Vast ×32 | 92,4670% | 91,0648% | 90,7191% |
| Vast ×64 | 93,0086% | 91,2992% | 90,8920% |
| Geleerd ×32/×64 | 92,6936% | 91,5412% | 91,3107% |

De lagere modi liggen op epoch 5 iets hoger dan beide vaste referenties. E4M3 ligt tussen die referenties. Voor de geleerde run kiest de vaste E4M3-selectieregel echter epoch4: E4M3 = 92,7128%, E2M3 = 90,7959%, E2M1 = 90,8536%. Beide vaste runs kiezen binnen hun eerste vijf epochs epoch 5. De vergelijking op het geselecteerde checkpoint levert dus geen algemene overwinning van de geleerde methode op.

## Leren van de schaalkeuzes

| Einde epoch | Blokken op ×32 | Gemiddelde theta |
|---:|---:|---:|
| 0 | 0 | 6,0000 |
| 1 | 0 | 5,9275 |
| 2 | 0 | 5,9107 |
| 3 | 1 | 5,8939 |
| 4 | 17 | 5,8797 |
| 5 | 42 | 5,8658 |

Op epoch5 staan 42 van de 4.080 blokken op ×32 (1,0294%). Er zijn 3.924 hulpwaarden verschillend van de beginwaarde 6. Dit bevestigt dat de expliciete gradient hulpwaarden kan veranderen en echte harde keuzes kan omschakelen. De 242 geregistreerde omschakelingen tellen herhaalde wissels van hetzelfde blok mee; dit zijn geen 242 verschillende blokken.

De beginwaarde 6 bevoordeelt aanvankelijk ×64 en de meeste hulpwaarden hebben de drempel 5,5 nog niet bereikt. De kleine fractie ×32 toont daarom nog niet dat ×64 voor vrijwel alle blokken optimaal is. Ook een grotere fractie ×32 zou op zichzelf geen accuracysucces bewijzen.

Gedurende de eerste twee epochs waren er geen harde omschakelingen, terwijl de accuracies al verschilden van de afzonderlijke vaste ×64-run. De behouden CUDA-backend is niet bitexact. Kleine verschillen tussen afzonderlijke runs kunnen daarom niet alleen aan geleerde keuzes worden toegeschreven. Eén seed en vijf epochs onderbouwen geen significantie- of superioriteitsclaim.

## Kosten en vervolgstap

De vijf epochs kosten samen 325,98 seconden; de hele sessie 333,69 seconden. De mediane epochtijd van epochs 2–5 is 65,887 seconden, tegenover 62,142 voor vast ×32 en 61,337 voor vast ×64. Dat is ongeveer 6,0% en 7,4% extra in deze softwaremetingen. De tijden omvatten ook schaaldiagnostiek; ze meten geen native MX-hardwarekosten.

De gemeten piekallocatedGPU-geheugens van epochs 2–5 bedragen 61,948 MiB voor geleerd en 61,828 MiB voor beide vaste runs. Deze kleine gemeten verschil is geen exacte geïsoleerde meting van alleen de nieuwe parameters. Analytisch zijn theta, zijn gradient en twee FP32-Adam-momenten samen 65.280 bytes; compacte inferentie bewaart 510 bytes harde keuzebits.

De extrapolatie voorspelt 109,75 minuten voor honderd epochs onder vergelijkbare omstandigheden. De passende vervolgstap is een verse honderd-epochrun met hetzelfde protocol. Er is geen technische aanleiding om nu de beginwaarde, drempel, schaalgradient of leersnelheid te veranderen. Daarna kunnen de twee gekozen checkpointoverzichten, de schaalverdeling en het gedrag rond de LR-dalingen worden vergeleken.
