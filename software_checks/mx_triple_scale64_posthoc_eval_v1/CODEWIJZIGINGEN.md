# Codewijzigingen en herkomst

## Bestaande implementatie behouden

De drie MX-codecs, basisadapter, drieprecisieadapter, traininghulpmiddelen,
oude exponent-/mantissafuncties, bronmanifest en upstream ZIP zijn byte voor
byte gekopieerd uit `mx_triple_truncation_aware_reference100_v1`. De 17
hergebruikte bestanden staan met SHA-256 in `REUSE_MANIFEST.json`.

De oorspronkelijke trainingspakketten en resultaten zijn niet aangepast.
`provenance` bevat de twee trainingspakketmanifesten en kopieën van de vorige
post-hocevaluator en zijn protocol. De twee `*_REFERENCE_PROTOCOL.json`
bestanden leggen de werkelijke configuraties uit de eerder aangeleverde
honderd-epochrapporten vast. Ze worden bij de lokale evaluatie gecontroleerd.

## Nieuwe bestanden

- `scale64.py`: A/B/C-afleidingen uit opgeslagen E4-elementcodes; validatie van beide exportfamilies; vaste schaaloffset +6; bevroren reconstructie en gewichts-/codediagnostiek.
- `evaluation.py`: uitbreiding van de bestaande evaluator naar negen routes, twaalf outputparen en twee trainingspopulaties; opslag/herladen vóór inferentie; onveranderde modelstaat controleren.
- `run_eval.py`: `check` en `evaluate`, met twee verplichte bronrunmappen voor evaluate; identiteitscontroles vóór evaluatie, vier vaste checkpointgroepen en één nieuwe resultaatmap.
- `test_scale64.py`: onafhankelijke element- en schaalcontroles plus synthetische bron-/export-/modelcontroles voor het lokale checkcommando.
- `notebook_support.py` en `MX_triple_scale64_posthoc_resultaten.ipynb`: uitlezen/valideren van rapporten, CSV en NPZ en presentatie van alle vastgelegde vergelijkingen. Geen torch-modeluitvoering.
- `local_validation`: reproduceerbare synthetische softwarecontroles en hun rapporten uit de voorbereiding; deze vormen geen echte SVHN-resultaten.

## Bewuste implementatiekeuzes

A roept de oude `exponent_truncate` en `mantissa_truncate` rechtstreeks aan.
B kopieert diezelfde elementcodes en verandert uitsluitend de E8M0-code.
C roept uitsluitend de bestaande E2M3-elementencoder aan op het lokale
gedecodeerde E4-element /64. Elk E2M1-pad behoudt dezelfde mantissashift.

De runner importeert bestaande modules waar dat nodig is voor de bevroren
architectuur, maar roept geen trainingsforward, optimizer, backward of
FP32-tensorencoder aan. De oorspronkelijke checkpoints met optimizer- en
masterstaat worden niet geopend. De opgeslagen overige FP32-laaggewichten
en BN-toestand blijven nodig voor inferentie.

De synthetische integratieharness mag voor zijn fixture kleine CPU-modellen,
fixturedata en een fixtureomgeving aanbieden. Die aanpassingen staan uitsluitend
in de tests; de gewone CLI blijft CUDA en de werkelijke broncontracten eisen.
De productiecode bevat geen argument om provenance of datasetcontroles over
te slaan.
