# Any-Precision-resultaten in een Jupyter-notebook

Het notebook `AnyPrecision_resultaten.ipynb` bevat de bestaande grafiek,
brondata, berekeningen, een tabel en een korte methodologische toelichting.
De grafiekuitvoer is al opgeslagen in het notebook.

## Meteen bekijken in VS Code

1. Pak het volledige ZIP-bestand uit. Behoud de mapstructuur.
2. Installeer zo nodig de Microsoft-extensies **Python** en **Jupyter** in VS Code.
3. Open `AnyPrecision_resultaten.ipynb` in je eigen vertrouwde projectmap.
4. Bekijk de bestaande tabel en grafiek. Voor alleen deze opgeslagen uitvoer
   hoef je de code niet opnieuw uit te voeren.

Bij Restricted Mode kan VS Code rijke uitvoer verbergen. Controleer de inhoud
voordat je een map als vertrouwd markeert.

Voor een browserweergave is er ook `AnyPrecision_resultaten_leesversie.html`:
een statische leesversie van dit notebook, met ingesloten grafiek en zonder code.

## Uitpakken in de afgesproken projectmap

Sla het ZIP-bestand op in `C:\Users\renzo\master-thesis-mx` en voer uit in PowerShell:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& {
    $ErrorActionPreference = "Stop"
    $Destination = ".\notebooks\anyprecision_resultaten_v1"
    if (Test-Path -LiteralPath $Destination) {
        throw "De doelmap bestaat al. We overschrijven niets."
    }
    New-Item -ItemType Directory -Path ".\notebooks" -Force | Out-Null
    Expand-Archive `
        -LiteralPath ".\anyprecision_resultaten_notebook_v1.zip" `
        -DestinationPath $Destination
}
```

## De cellen zelf uitvoeren: aparte omgeving

Het notebook gebruikt geen PyTorch, GPU, CIFAR-10-afbeeldingen of checkpoints.
Laat de bestaande `.venv` en `.venv-train` ongewijzigd. Maak een aparte omgeving
voor het notebook; het onderstaande blok weigert een bestaande map te vervangen.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& {
    $ErrorActionPreference = "Stop"
    $Package = ".\notebooks\anyprecision_resultaten_v1"
    $BasePython = ".\.venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $BasePython -PathType Leaf)) {
        throw "De bestaande Python-interpreter werd niet gevonden."
    }
    if (-not (Test-Path "$Package\requirements-notebook.txt")) {
        throw "Pak eerst het notebookpakket uit."
    }
    if (Test-Path -LiteralPath ".\.venv-notebook") {
        throw ".venv-notebook bestaat al. Stop hier; er wordt niets vervangen."
    }
    & $BasePython -B -m venv ".venv-notebook"
    if ($LASTEXITCODE -ne 0) { throw "Aanmaken van de notebookomgeving mislukt." }

    # Deze nieuwe omgeving blijft buiten Git, zonder de project-.gitignore te wijzigen.
    Set-Content -LiteralPath ".\.venv-notebook\.gitignore" -Value "*" -Encoding ascii

    $Logs = "$Package\environment_checks\$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
    New-Item -ItemType Directory -Path $Logs | Out-Null
    & ".\.venv-notebook\Scripts\python.exe" -m pip install `
        -r "$Package\requirements-notebook.txt" `
        --report "$Logs\pip_install_report.json"
    if ($LASTEXITCODE -ne 0) { throw "Installatie mislukt; deel de foutmelding." }
    & ".\.venv-notebook\Scripts\python.exe" -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Pakketconflict; stop hier." }
    $Freeze = & ".\.venv-notebook\Scripts\python.exe" -m pip freeze --all
    if ($LASTEXITCODE -ne 0) { throw "Opslaan van pakketversies mislukt." }
    $Freeze | Set-Content -LiteralPath "$Logs\pip_freeze.txt" -Encoding utf8
    Write-Host "Notebookomgeving klaar. Selecteer .venv-notebook als notebookkernel."
}
```

Vervolgens in het notebook:

- **Select Kernel** (rechtsboven), eventueel **Select Another Kernel**, dan
  **Python Environments** en de interpreter van `.venv-notebook`.
- Het bedoelde pad is `C:\Users\renzo\master-thesis-mx\.venv-notebook\Scripts\python.exe`.
- Kies **Run All** om de cellen van boven naar beneden uit te voeren.
- Sla op met **Ctrl+S** om de nieuwe uitvoer in het notebook te bewaren.

De kernel is het Python-proces dat de notebookcellen uitvoert. De keuze is niet
noodzakelijk dezelfde als de geactiveerde omgeving in je terminal. Verschijnt de
omgeving niet, gebruik dan **Python: Select Interpreter** in het commandopalet
om het bovenstaande pad te kiezen en open daarna opnieuw de kernelkeuze.

De eisen in `requirements-notebook.txt` zijn vaste versies van de twee directe
pakketten, niet een volledige vergrendeling van alle indirecte afhankelijkheden.
De feitelijke installatie wordt daarom apart geregistreerd.

## Wat doet het notebook?

- Leest alleen de drie kleine invoerbestanden onder `input/`.
- Controleert hun SHA-256-waarden tegen `input_manifest.json`.
- Selecteert alleen de vijf regels met rol `github_style_test_selected`.
- Controleert de accuracies tegen het rapport en de correcte aantallen.
- Berekent eigen accuracy minus GitHub-accuracy in procentpunten.
- Maakt de bestaande grafiek opnieuw; geen nieuwe training of modelevaluatie.

`EXPORTEREN = False` schrijft standaard geen nieuwe bestanden. Met `True` komen
PNG en SVG in een nieuwe gedateerde map onder `exports/`. Oude exports en alle
invoerbestanden blijven onaangeroerd. De weergegeven tabel en grafiek kunnen
worden aangepast door de notebookcode te bewerken, zonder de trainingscode te wijzigen.

Dit is een vaste momentopname van de huidige resultaten, geen automatische zoeker
naar nieuwe trainingsruns. Gebruik voor latere analyses aparte notebookversies en
vervang de bewaarde brondata van deze referentie niet.

## Bronnen en beperking

De GitHub-waarden komen uit de bestaande `github_comparison.csv` en zijn gekoppeld
aan de rij `Resnet-20-Any (recursive)` in `selection_report.json`.
De eigen waarden zijn de aangeleverde testmetingen voor checkpoint 200.
Het notebook heeft ze niet opnieuw gemeten of live opgehaald.
De precieze historische run achter de auteurswaarden is niet vastgesteld.

Eerder werd op W32/A32-testaccuracy geselecteerd. Deze grafiek toont daarom geen
onafhankelijke nieuwe test en bevat geen statistische significantieclaim.
Er zijn geen MX-, truncatie-, continual-learning- of hardwaremetingen toegevoegd.

## Technische controle

Zie `VALIDATION.json`. Het notebook is hier op Linux/CPU van boven naar beneden
uitgevoerd met de echte aangeleverde resultaatbestanden, zonder notebookfouten.
De Windows-uitvoering moet nog lokaal worden bevestigd. Voor alleen tonen kun je
de reeds opgeslagen notebookuitvoer of de HTML-leesversie gebruiken.

## Git en bestandsindeling

Het notebook, de kleine invoerbestanden, manifesten en README kunnen in Git.
De meegeleverde `.gitignore` sluit tijdelijke Jupyter-checkpointmappen uit.
De nieuwe `.venv-notebook` krijgt een eigen `.gitignore` en gaat niet mee naar GitHub.
Geen van de grote modelcheckpoints of CIFAR-10-batches zit in dit pakket.

## Documentatie

- Microsoft: Jupyter Notebooks in VS Code, https://code.visualstudio.com/docs/datascience/jupyter-notebooks
- Microsoft: Manage Jupyter Kernels, https://code.visualstudio.com/docs/datascience/jupyter-kernel-management
