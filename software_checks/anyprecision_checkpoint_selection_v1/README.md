# Any-Precision: checkpoints testen en een model selecteren

Dit pakket is een aanvulling op de reeds gebruikte
`software_checks/anyprecision_reference_github200_v1`.
Bewaar dat oorspronkelijke pakket en ALLE trainingssessiemappen.
Geen nieuwe pakketten installeren; geen checkpoints uploaden of opnieuw trainen.

## 1. Uitpakken

PowerShell in VS Code, niet Git Bash:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& {
    $ErrorActionPreference = "Stop"
    $Zip = ".\anyprecision_checkpoint_selection_v1.zip"
    $Destination = ".\software_checks\anyprecision_checkpoint_selection_v1"
    if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) {
        throw "Plaats het ZIP-bestand eerst in de projectmap."
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "De doelmap bestaat al; we overschrijven niets."
    }
    Expand-Archive -LiteralPath $Zip -DestinationPath $Destination
    if (-not (Test-Path "$Destination\select_checkpoints.py")) {
        throw "Het evaluatiescript werd niet gevonden."
    }
}
```

## 2. Sweep starten of hervatten

Deze opdracht gebruikt het reeds bekende eindcheckpoint. Verander geen
bestanden van de training en verwijder geen oude sessiemappen.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& {
    $ErrorActionPreference = "Stop"
    $FinalCheckpoint = "C:\Users\renzo\master-thesis-mx\reference_runs\anyprecision_github200\20261013_164217_232547_train\checkpoints\epoch_00200.pt"
    foreach ($Path in @(
        ".\.venv-train\Scripts\python.exe",
        ".\software_checks\anyprecision_checkpoint_selection_v1\select_checkpoints.py",
        ".\software_checks\anyprecision_reference_github200_v1\run_reference.py",
        $FinalCheckpoint,
        "$FinalCheckpoint.sha256"
    )) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Vereist bestand ontbreekt: $Path"
        }
    }
    & ".\.venv-train\Scripts\python.exe" -B `
        ".\software_checks\anyprecision_checkpoint_selection_v1\select_checkpoints.py" sweep `
        --reference-dir ".\software_checks\anyprecision_reference_github200_v1" `
        --run-root ".\reference_runs\anyprecision_github200" `
        --final-checkpoint $FinalCheckpoint `
        --data-root ".\data\cifar10" `
        --output-dir ".\evaluation_runs\anyprecision_github200_selection_v1" `
        --device cuda `
        --max-checkpoints 0
    if ($LASTEXITCODE -ne 0) {
        throw "De evaluatie meldde een fout. Bewaar de uitvoer en selection_report.json."
    }
}
```

`0` betekent alle nog ontbrekende checkpointtests, niet nul tests. Verander die
waarde naar bijvoorbeeld `20` om per sessie maximaal twintig nieuwe W32-tests
uit te voeren. Voer daarna PRECIES dezelfde opdracht opnieuw uit, met dezelfde
--output-dir. Reeds afgeronde resultaten worden hergebruikt.
De definitieve selectie gebeurt pas wanneer de volledige reeks van 200 is getest.
Er is voor evaluatie geen nieuw modelcheckpointpad per sessie nodig.

## 3. Eerder stoppen

In een tweede PowerShell-terminal:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
$Session = Read-Host "Plak de actieve Sessie-/rapportmap van de evaluatie, zonder aanhalingstekens"
& ".\.venv-train\Scripts\python.exe" -B `
    ".\software_checks\anyprecision_checkpoint_selection_v1\select_checkpoints.py" request-stop `
    --session-dir $Session
```

Wacht op STATUS: PAUSED en de terugkeer van de prompt in de eerste terminal.
Wanneer de afsluitende vijf-precisietests al bezig zijn, rondt het programma die af.
Na een onverwachte onderbreking dezelfde sweepopdracht opnieuw uitvoeren.
Een eventuele onvolledige checkpointtest wordt herhaald; geen training gaat verloren.

## Uitvoer

- Aan het begin: dataset en checkpointinventarisatie. Dat kan even duren zonder GPU-belasting.
- Vervolgens: W32-testscore, tijd en resterende tijdschatting per epochcheckpoint.
- `PAUSED`: complete tussentijdse resultaten, maar nog geen definitieve beste epoch.
- `PASS`: alle 200 gecontroleerd/getest; vijf precisies van beste en laatste checkpoint
  gerapporteerd. Geen minimumaccuracy als voorwaarde.
- `FAIL`: bijvoorbeeld ontbrekende checkpoints, checksumfout of afwijkende broncode.
  Niets verwijderen, niets hertrainen; eerst de melding onderzoeken.

De sessierapporten staan in
`evaluation_runs/anyprecision_github200_selection_v1/sessions/<tijdstip>/`.
Deel bij PASS: `selection_report.json`, `checkpoint_scores.csv`,
`selected_checkpoint_metrics.csv`, `final_checkpoint_metrics.csv`, `github_comparison.csv`.
Bij FAIL: `run.log`, `selection_report.json` en indien aanwezig `inventory.json`.
De .pt-bestanden zelf hoef je niet te uploaden.

De checkpoints blijven onaangeroerd en buiten Git. De kleine evaluatiebestanden,
protocol en code kunnen wel samen worden bewaard. Beoordeel trainaccuracy niet
alsof het een testscore is. Zie PROTOCOL.md voor de betekenis en beperkingen van
selectie op de officiele testset.
