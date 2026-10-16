# Wijzigingslog checkpointselectie v1

## Nieuwe bestanden

- `select_checkpoints.py`: inventarisatie, trace-controle, W32-tests, selectie,
  vijf-precisie-evaluatie, rapportage, pauzeren en hergebruik van resultaten.
- `REFERENCE_SHA256.json`: vastgelegde hashes van vereiste GitHub200-hulpbestanden.
- `ADDON_SHA256.json`: hashes van de uitvoerbare evaluatie-aanvulling.
- README, PROTOCOL en lokale validatiescripts/verslagen.

## Wat NIET verandert

Geen enkel bestand in de reeds uitgevoerde trainingspakketten, trainingsdata of
checkpoints wordt aangepast. Geen retraining, geen andere quantizer, geen
BatchNorm-kalibratie, geen optimizerupdates. De oorspronkelijke .pt-bestanden en
hun checksum-bestanden blijven in hun oude sessiemappen staan.

## Wel een expliciete evaluatieprotocoluitbreiding

Naast het eerder vastgelegde eindcheckpoint wordt nu de beste epoch op de
W32/A32-testaccuracy geselecteerd, conform de aanvullende afspraak tijdens de
training. Dit is een test-gebaseerde selectie, niet een onafhankelijke
validatieset. De oude configuratiebestanden worden niet achteraf herschreven;
het aparte `evaluation_plan.json` registreert de aanvullende procedure.

De W32-only-sweep bespaart evaluaties van andere precisies die voor de selectieregel
niet meetellen. Alleen het geselecteerde checkpoint en epoch 200 worden op alle
vijf precisies getest. De uitkomst van W32 in beide lussen wordt exact vergeleken.

De uitvoer gebruikt een aparte hoofdmap en onveranderlijke resultaten per
checkpoint. Bij hervatten worden eerst opnieuw inventarisatie en checksums
gecontroleerd; oude JSON-metingen blijven behouden.
