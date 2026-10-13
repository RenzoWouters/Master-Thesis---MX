# Any-Precision: GitHub-coderecept, 200 epochs

Gebruik dit pakket in plaats van `anyprecision_reference_v1` (400 epochs).
Het oorspronkelijke archief en de model-/quantizerbestanden zijn ongewijzigd.
De training gebruikt het concrete CIFAR-10-recept uit de aangeleverde ZIP:
200 epochs; LR 0.001; verlaging na 100, 150, 180; Adam; weight decay 0;
batch 128; alle 50.000 trainingsbeelden; vijf gezamenlijke precisies.

De herkomst van de exacte README-resultaten uit dit recept is NIET bewezen.
Zie GITHUB_ALIGNMENT.md. Dit is geen garantie op dezelfde accuracy.

## Eerste sessie: vijf echte epochs

PowerShell, vanuit `C:\Users\renzo\master-thesis-mx`:

```powershell
& ".\.venv-train\Scripts\python.exe" -B `
    ".\software_checks\anyprecision_reference_github200_v1\run_reference.py" train `
    --device cuda `
    --data-root ".\data\cifar10" `
    --output-root ".\reference_runs\anyprecision_github200" `
    --session-epochs 5
```

De dataset is al lokaal aanwezig; geen downloads of pakketinstallatie.
Er zijn 391 optimizerstappen per epoch. Iedere stap doorloopt alle vijf precisies.
`PAUSED` na epoch 5 is een normale, geslaagde sessiestop, geen fout.
De 200-epochplanning blijft behouden. Train zonder --resume begint opnieuw;
gebruik dus --resume om dezelfde training later voort te zetten.

Bewaar de geprinte sessiemap. Deel eerst `reference_report.json`, `metrics.csv`
en `timings.csv` voordat je verdergaat. De tijdmetingen dienen om de resterende
rekentijd te schatten, niet als benchmark voor integer/MX-hardware.

## Eerder stoppen

Voer in een TWEEDE PowerShell-terminal uit:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
$Session = Read-Host "Plak de actieve sessiemap zonder aanhalingstekens"
& ".\.venv-train\Scripts\python.exe" -B `
    ".\software_checks\anyprecision_reference_github200_v1\run_reference.py" request-stop `
    --session-dir $Session
```

Wacht in de trainingsterminal op `STATUS: PAUSED` en het opslaan van de
checkpoint voordat je afsluit. Het lopende epoch wordt eerst afgemaakt.
De terminal sluiten, slaapstand of Ctrl+C is niet de nette stopprocedure.
Bij een noodonderbreking hervat je het laatste COMPLETE checkpoint;
het niet-afgeronde epoch moet opnieuw worden uitgevoerd.

## Hervatten in een nieuwe sessie

Gebruik uitsluitend een checkpoint van DIT pakket, niet van de pilot,
hervatcontrole of de eerdere 400-epochversie. Bewaar het .pt-bestand en de
bijbehorende .pt.sha256 samen. Code/configuratie/omgeving worden gecontroleerd.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
$Checkpoint = Read-Host "Plak het volledige pad naar de laatste complete epoch_XXXXX.pt, zonder aanhalingstekens"
& ".\.venv-train\Scripts\python.exe" -B `
    ".\software_checks\anyprecision_reference_github200_v1\run_reference.py" train `
    --device cuda `
    --data-root ".\data\cifar10" `
    --output-root ".\reference_runs\anyprecision_github200" `
    --resume $Checkpoint `
    --session-epochs 5
```

Vanuit epoch 5: sessie 2 traint epochs 6-10. Meer/minder epochs per sessie
verandert het protocol niet. Geen automatische CPU-terugval of batchverkleining.
Een resume maakt een nieuwe sessiemap; oude checkpoints worden niet gewijzigd.
Geen bestanden, bibliotheekversies of instellingen veranderen tijdens de run.
Bij onvoldoende GPU-geheugen: stop, bewaar de foutmelding en deel die eerst.

## Eindtest en vergelijking - PAS NA DE TRAINING

Het vooraf vastgelegde eindpunt is epoch 200. Het script selecteert geen
beste epoch op de testset. `TRAINING_COMPLETE_TEST_NOT_EVALUATED` betekent
voltooide training, maar nog geen testresultaat.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
$Checkpoint = Read-Host "Plak het volledige pad naar epoch_00200.pt, zonder aanhalingstekens"
& ".\.venv-train\Scripts\python.exe" -B `
    ".\software_checks\anyprecision_reference_github200_v1\run_reference.py" evaluate `
    --device cuda `
    --data-root ".\data\cifar10" `
    --output-root ".\reference_runs\anyprecision_github200" `
    --checkpoint $Checkpoint `
    --endpoint-epoch 200 `
    --endpoint-reason "Vooraf vastgelegd eindpunt van het 200-epoch GitHub-coderecept; geen selectie op testaccuracy"
```

Dit traint niet verder. Het evalueert hetzelfde checkpoint op alle 10.000
testbeelden, apart voor W1/A1, W2/A2, W4/A4, W8/A8 en W32/A32.
`test_metrics.csv` bevat de eigen scores. `github_comparison.csv` bevat de
eigen scores, de auteurswaarden uit de README en het verschil in procentpunten.
`github_comparison.json` bewaart ook de herkomst en beperkingen.

Tussencheckpoints zijn later ook evalueerbaar met een expliciete eindpuntreden,
maar horen niet als een volledige 200-epochtraining gepresenteerd te worden.
Kies niet alsnog het gunstigste checkpoint of de gunstigste seed op de testset.

## Opslag en Git

Iedere epoch krijgt een eigen volledig checkpoint plus checksum. Uitgaande van
ongeveer 78 MiB per checkpoint is voor 201 checkpoints ongeveer 15-16 GiB nodig;
de trace groeit en de werkelijke totale opslag kan hoger uitvallen. Logs komen erbij.
De bestaande `*.pt`-regel houdt checkpoints uit gewone Git-opslag. Rapporten,
metrics, configuraties en code blijven klein en kunnen wel in Git. Zorg zelf voor
een extra back-up van belangrijke checkpoints: deze staan niet automatisch op GitHub.

## Aanpassingen en controles

- `GITHUB_ALIGNMENT.md`: wat uit de auteurscode komt en wat onze uitvoerkeuze is.
- `PROTOCOL.md`: volledig vastgelegde uitvoering.
- `changes_from_reference_v1.patch`: runner/config-verschil met de 400-epochversie.
- `compatibility.patch`: uitsluitend de patches aan de oorspronkelijke bronkopie.
- `SOURCE_PROVENANCE.json`: oorspronkelijke archiefhash en hergebruikte modules.
- `local_validation/VALIDATION.md`: werkelijk uitgevoerde lokale tests en beperkingen.

De scripts onder local_validation gebruiken kunstmatige bestanden en een tiny
protocol voor integratietests. Deze zijn GEEN instructie voor een nieuwe laptop-pilot.
