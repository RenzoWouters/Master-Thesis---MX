# Uitgevoerde lokale validatie

## Geslaagde controle in drie echte processen

- Systeem: Linux, Python 3.13.5, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu,
  NumPy 2.3.5, Pillow 12.3.0. Geen CUDA-apparaat beschikbaar.
- Model: dezelfde oorspronkelijke resnet20q, expand=5, 6.786.330 parameters.
- Data: **kunstmatige afbeeldingen/labels in het CIFAR-batchbestandsformaat**.
  De oorspronkelijke loader en transformaties werden gebruikt. Alleen in de
  afzonderlijke fixture-wrapper werden checksummetadata in het geheugen aan
  de fixturebestanden aangepast; geen bronbestand werd daarvoor gewijzigd.
- Het normale gebruikerscommando kent deze wrapper niet. Op jouw laptop worden
  echte, reeds aanwezige CIFAR-batchbestanden met hun oorspronkelijke checksums
  gecontroleerd. De lokale test is GEEN CIFAR-10-accuracyresultaat.
- Drie daadwerkelijk verschillende subprocessen (PIDs 1292, 1346, 1383), allemaal
  normaal afgesloten met returncode 0.
- 3 mini-epochs ononderbroken versus 1 + 2 met procesexit, checkpoint laden en
  nieuwe optimizerstappen. Alle 16 controles slaagden, inclusief exacte inhoud
  van model/BatchNorm, optimizer, scheduler, RNGs, batches/augmentatiehashes,
  losses en validatielogits. Learning rates: 0.001, 0.0001, 0.00001.
- Geen opgevangen Python-waarschuwingen in de drie workers.
- Totale lokale diagnose-uitvoering: ongeveer 61,22 seconden. Dit is geen
  voorspelling van Windows/CUDA-duur of de volledige referentietraining.
- Checkpointopslag: 468,92 MiB voor zes checkpoints; niet in de download-ZIP.

Bewijs: `offline_cpu_report.json` en `offline_cpu.log`.

## Voorafgaande afgebroken uitvoerpoging

Een eerdere container-aanroep werd door de tooltimeout afgebroken tijdens het
tweede proces, na een geslaagd eerste proces. Er was geen volledig eindrapport
of geslaagde vergelijkingsstatus voor die poging. Het log is bewaard als
`first_attempt_tool_timeout.log`. De complete controle is vervolgens met dezelfde
productiecode opnieuw uitgevoerd en hierboven gerapporteerd. Dit was geen
opzettelijke resume-test van een crash midden in een epoch.

## Kleine unit- en negatieve tests

Acht CPU-unit-tests geslaagd (`guard_tests.log`):

- RNG herstel, ook met gecachete gaussische waarden;
- volledig herstel gevolgd door een nieuwe optimizer- en schedulerstap;
- afwijkend contract weigeren voordat modelstate wordt geladen;
- bestaande checkpoints niet overschrijven, weights_only roundtrip;
- corrupte checkpointbytes weigeren;
- checkpoint zonder voltooide checksummarker weigeren;
- ontbrekende DataLoader-RNG signaleren;
- negatieve controle: RNG niet herstellen verandert de diagnostische probe.

De data-guard is ook afzonderlijk getest (`dataset_guard.log`): de normale
productiefunctie weigert de kunstmatige files wegens een checksumverschil en
weigert een ontbrekende datamap zonder iets te downloaden.

## Niet lokaal vastgesteld

Windows/CUDA-exactheid moet op de laptop worden gecontroleerd. Evenmin getest:
mid-epoch-hervatting, fysieke stroomuitval, cross-device/cross-version-herstel,
of integratie met een nog te maken 400-epochrunner. De test op de laptop is
noodzakelijk; een lokale CPU-PASS vervangt die niet.
