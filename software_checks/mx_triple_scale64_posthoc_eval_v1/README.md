# Post-hoc schaal ×64 en afrondingscontrole op beide trainingen

Dit pakket voert de afgesproken volgende proef uit op de bestaande
**directe drieprecisietraining** en de **naïeve truncatietraining**, beide met
honderd voltooide epochs. Per training worden het opgeslagen **E4M3-best**
checkpoint en **epoch 100** gebruikt. De checkpointkeuze blijft vast.

De proef vergelijkt drie afleidingen uit dezelfde opgeslagen E4M3-bron:

| Familie | E2M3 | Schaal | E2M1 |
|---|---|---|---|
| A | Hoogste twee van de vier exponentbits behouden | Oorspronkelijke E4M3-schaal | Daarna dezelfde mantissatruncatie |
| B | Exact dezelfde elementcodes als A | E4M3-schaal ×64 | Daarna dezelfde mantissatruncatie |
| C | E2M3 nearest-even afronding van het gedecodeerde lokale E4M3-element /64 | E4M3-schaal ×64 | Daarna dezelfde mantissatruncatie |

De drie opgeslagen getrainde modi E4M3, E2M3 en E2M1 worden ook opnieuw
gemeten. Dat geeft negen routes per checkpoint, vier checkpointgroepen en
36 accuracyregels. Binnen elke groep krijgen alle routes dezelfde testbeelden.

Activaties blijven A32. Er zijn geen optimizerupdates, nieuwe training,
geleerde schalen of BN-herkalibratie. De bestaande FP32-updatemasters blijven
behouden; deze evaluatie gebruikt alleen de opgeslagen MX-exports en hun
overige modelstaat. Alle eerdere resultaten blijven bestaan.

## 1. Uitpakken en controleren

Pak de map uit de ZIP uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_scale64_posthoc_eval_v1\run_eval.py`

Gebruik dezelfde `.venv-train` als voor beide trainingen:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_posthoc_eval_v1\run_eval.py" check
```

`check` gebruikt synthetische codes, modellen en beelden. Het controleert de
bitbewerkingen, nearest-even afronding, signed zero, schaalgrenzen, packing,
opslag/herladen en bevroren modelreconstructie. Het leest geen SVHN-data en
start geen training. Bij FAIL: deel eerst de nieuwe `feedback_bundle.zip`
uit de checkmap.

## 2. Na CHECK PASS: beide trainingen evalueren

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_posthoc_eval_v1\run_eval.py" evaluate `
  --direct-run ".\mx_runs\mx_weights_triple_direct_reference100\20261023_151331_732669_joint_run" `
  --truncation-run ".\mx_runs\mx_triple_truncation_aware_reference100\20261023_184058_211397_joint_run"
```

Beide argumenten wijzen naar de volledige oorspronkelijke runmappen, inclusief
de zes opgeslagen MX-exports en hun `.sha256`-bestanden per training. Alleen
een uitgepakte feedbackbundel bevat daarvoor onvoldoende bestanden. Als je
een runmap hebt verplaatst, pas je het bijbehorende pad aan.

De runner controleert de oorspronkelijke protocollen, selectie, exporthashes,
checkpointprovenance, datasetidentiteit en uitvoeringsomgeving. Hij leest de
oude trainingscheckpoints met FP32-masters niet in. Voor de truncatietraining
controleert hij bovendien dat de opgeslagen E2M3/E2M1-codes exact uit de
bijbehorende opgeslagen E4M3-bron zijn afgeleid.

Standaard wordt `data\svhn\test_32x32.mat` gebruikt. Staat de bestaande dataset
elders, voeg dan `--data-root "PAD"` toe. Er wordt uitsluitend op de officiële
testset met 26.032 beelden geëvalueerd, met dezelfde preprocessing en volgorde
als bij de training. Er is geen download of nieuwe laptopinstallatie nodig.

Dit is één evaluatie met 36 model-testsetcombinaties; er zijn geen epochs of
parameterupdates. Een lage accuracy is een geldige onderzoeksuitkomst.
`PASS` betekent dat de afgesproken evaluatie en integriteitscontroles zijn
afgerond, niet dat schaal ×64 de accuracy noodzakelijk verbetert.

## 3. Resultaten en notebook

Iedere start maakt een nieuwe map onder:

`mx_runs\mx_triple_scale64_posthoc_eval\<tijdstip>_evaluate\`

Open in die map `MX_triple_scale64_posthoc_resultaten.ipynb` met je bestaande
notebookkernel en voer alle cellen uit. De resultaatmap wordt automatisch
ingevuld. Er wordt uitsluitend één lege, uitvoerbare notebook meegeleverd.

De notebook toont de twee trainingen afzonderlijk en bevat:

- twee accuracyfiguren, één per training, met epoch 100 en E4M3-best naast elkaar en alle negen routes;
- accuracyverschillen A→B en B→C, en vergelijking met de getrainde doelmodus;
- gepaarde fout→correct- en correct→foutveranderingen op dezelfde testbeelden;
- gewichtsafwijkingen, RMS-grootte en nul-/codedistributies per laag;
- voorspelde klasseverdeling, zodat eventuele instorting naar één klasse zichtbaar wordt;
- herhaling van de historische baselines en de hashes, selectie- en BN-controles.

De notebook leest de resultaten en NPZ-bestanden; hij voert geen model uit.
`SAVE_FIGURES = True` bewaart de figuren in een nieuwe analysemap. Ruwe
resultaten blijven behouden.

Bewaar de volledige resultaatmap en stuur daarna de `feedback_bundle.zip`
uit de nieuwe `_evaluate`-map. Die bevat de vier geordende per-beeldbestanden
met logits, voorspellingen en cross-entropy, plus metrics en auditrapporten.
De bundel is daardoor groter dan een trainingsfeedbackbundel. Modelbestanden
en datasets worden niet in die feedbackbundel opgenomen.

## Interpretatie

**A→B** verandert uitsluitend de gedeelde schaal; de elementcodes blijven
identiek. **B→C** vergelijkt de naïeve exponentmapping met een gewone
nearest-even omzetting bij dezelfde aangepaste schaal. C vertrekt nog steeds
van de opgeslagen E4M3-codes, niet van de oorspronkelijke FP32-gewichten.
Bij alle E2M1-routes blijft dezelfde mantissatruncatie aanwezig.

De bestaande BN-bank van de doelprecisie blijft in elke vergelijking bevroren.
Een schaalwijziging kan dus ook de aansluiting op die BN-toestand veranderen.
Het experiment meet het gezamenlijke netwerkeffect van de concrete ingreep;
het voorspelt nog niet het resultaat van later trainen met die ingreep.

De vaste factor 64 volgt uit het verschil van zes tussen de gebruikte maximale
lokale exponenten. Hij is niet geleerd of op deze testset gekozen, en herstelt
de weggegooide exponentbits niet. De routes zijn vaste controles; er wordt
geen beste schaal of nieuw beste checkpoint gezocht.

## Stoppen en documentatie

`Ctrl+C` onderbreekt de proef en bewaart een diagnostische feedbackbundel.
Er is geen resumecommando. Opnieuw starten maakt een nieuwe resultaatmap.

`PROTOCOL.md` legt de precieze bewerkingen en vergelijkingen vast.
`CODEWIJZIGINGEN.md` beschrijft de hergebruikte en nieuwe bestanden.
`local_validation/VALIDATION.md` beschrijft de daadwerkelijk uitgevoerde
softwarevalidatie. Bij de voorbereiding is geen echte SVHN-evaluatie of
training uitgevoerd: de benodigde getrainde MX-exports staan op jouw laptop.
