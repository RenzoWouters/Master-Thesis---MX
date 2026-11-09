# Validatie van de honderd-epochversie

Er is bij het maken van dit pakket geen SVHN- of GPU-training uitgevoerd.
De echte vijf-epochresultaten van de gebruiker zijn afzonderlijk beoordeeld in
`PILOT_BEOORDELING.md` en `PILOT_AUDIT.json` in de pakketroot.

## Trainingslus en hervatting

De echte runner doorliep honderd epochs met drie synthetische CPU-beelden,
gevolgd door een hervatting vanaf checkpoint 50 en een gedeeltelijke run van
één epoch. De 302 synthetische updates controleren de software, niet de
SVHN-accuracy of GPU-snelheid. Alle LR-overgangen voor beide Adam-groepen zijn
getest. De doorlopende en hervatte run leverden exact gelijke modelgewichten,
schaalparameters, Adam-staat, scheduler en RNG-staat op. De schaal-CSV's zijn
byte-identiek. Checkpoint 50 bevat alle vereiste toestand en geen bevroren
schaaloptimizer. Ongeldige checkpointcontracten en CLI-grenzen worden geweigerd.

De numerieke kernbestanden zijn byte-identiek aan de geslaagde pilot. Het
definitieve script voegt na deze looptest alleen een exportcontrole toe. Een
AST-vergelijking bevestigt dat alle trainingsstatements gelijk zijn gebleven.
De definitieve exportcontrole is apart getest vanuit het bewaarde checkpoint,
zonder de training te herhalen.

## Waargenomen problemen en afhandeling

Een laatste administratieassertie in de eerste lokale test verwees nog naar
een pakketmanifest dat op dat moment werd samengesteld. De functionele
runnercontroles waren al geslaagd. Die testverwijzing is gecorrigeerd en de
bewaarde uitvoer is onafhankelijk opnieuw gecontroleerd.

Daarnaast was het oorspronkelijke synthetische hervatbestand `history.json`
afgebroken. De oorzaak is onbekend; het geverifieerde checkpoint en alle
numerieke eindtoestanden waren volledig. Twee exports vanuit hetzelfde
checkpoint en de onveranderde timing-sidecar leverden de complete historie op.
De definitieve runner leest die historie nu terug en vergelijkt haar volledig
voordat de export kan slagen. Ontbrekende, afgebroken en onvolledige geldige
JSON-bestanden worden aantoonbaar geweigerd. De oorspronkelijke afwijkende
testuitvoer is niet als geslaagde historie behandeld of overschreven.

Zie `runner_validation_binding.json`, `RUNNER_VALIDATION.md`,
`training_integration_report.json`, `history_export_investigation.json` en
`history_export_guard_validation.json` voor exacte herkomst, hashes en scope.

## Notebook

Alle 17 codecellen en 14 figuren zijn uitgevoerd op volledige synthetische
uitvoer, een gedeeltelijke run, een vroege FAIL en een leeg sjabloon. De
herexport van de hervatte run is vervolgens apart volledig ingelezen. De
oorspronkelijke corrupte historie wordt als analysefout herkend. De drie echte
honderd-epochreferenties en de aparte tienkeuzepilot worden op hun opgeslagen
hashes gecontroleerd. Een gedeeltelijke run krijgt geen toekomstinformatie
uit de referenties in haar vergelijking.

De volledige curves, de twee checkpointvergelijkingen en de tien
schaalverdelingen zijn visueel gecontroleerd. QA-afbeeldingen met synthetische
scores worden niet als onderzoeksresultaten meegeleverd. De geleverde notebook
heeft lege outputs. Zie `notebook_validation_report.json` en
`notebook_resumed_reader_report.json`.

## Praktische grens van deze controle

De lokale numerieke tests gebruiken CPU. De Windows/CUDA-controle gebeurt met
het meegeleverde `check`-commando. De oorspronkelijke CUDA-backend blijft
behouden; een bitexacte hervattingsgarantie op GPU wordt niet gegeven.
