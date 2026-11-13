# Onafhankelijke E2-schaal bevriezen na epoch 50 — vijfepochproef

Deze proef vertakt vanuit het bestaande checkpoint van **epoch 50**. De gewichten en actieve BN-banken leren verder gedurende **epochs 51–55**. De absolute E2-schaalparameters en hun E8M0-schaalbytes blijven exact vast. De E4-bronschaal volgt de bestaande berekening uit de actuele gewichten.

De controle waarin de E2-schalen tot epoch 100 blijven leren is al uitgevoerd. Die controle wordt niet opnieuw getraind. Dit pakket bevat nog geen nieuwe freeze-resultaten.

## Uitvoeren

Pak deze map uit onder:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_independent_e2_scale_freeze50_pilot5_v1`

De standaardbron is het bestaande lokale checkpoint:

`mx_runs\mx_triple_independent_e2_scale_reference100\20261111_172102_961497_joint_run\checkpoints\epoch_00050.pt`

Behoud de oorspronkelijke resultatenmap en haar andere checkpoints. Een geselecteerd beste checkpoint kan nog vóór de freeze liggen; de notebook toont daarnaast de selectie binnen uitsluitend de nieuwe vertakking.

Eerst controleren:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_freeze50_pilot5_v1\run_train.py" check
```

Na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_freeze50_pilot5_v1\run_train.py" run
```

Is de parentmap verplaatst, geef dan het volledige pad van hetzelfde checkpoint met `--parent-checkpoint`. De inhoud moet de vastgelegde SHA-256 hebben; een ander checkpoint wordt afgewezen.

`check` gebruikt een begrensde synthetische proef en start geen SVHN-training. `run` voert alleen de vijf nieuwe epochs uit en stopt uiterlijk op epoch 55. Op basis van de vorige run is circa **5–6 minuten** een praktische raming, exclusief eventuele extra opstarttijd. Dit is nog geen gemeten freeze-looptijd.

## Wat wordt hersteld?

Gewichten, schaalparameters, Adam-momenten, scheduler, RNG en geschiedenis worden uit epoch 50 hersteld. Daarna krijgt de schaalgroep `requires_grad=False`, `grad=None` en LR nul. De groep blijft aanwezig, maar haar Adam-momenten en staptellers blijven onveranderd. De gewicht-LR is vanaf epoch 51 volgens het bestaande schema 0,0001.

De **absolute E2-schaal** wordt bevroren. De verhouding E2/E4 kan veranderen doordat E4 dynamisch blijft. Ook de getrunceerde elementcodes kunnen veranderen wanneer de gewichten en E4-broncodes veranderen.

A32, FP32-masters, B32, drie precisies, de oorspronkelijke loss, drie passes en één Adam-stap per minibatch blijven behouden. De dataset en oorspronkelijke run worden niet aangepast. Iedere opdracht schrijft naar een nieuwe map onder:

`mx_runs\mx_triple_independent_e2_scale_freeze50_pilot5`

## Stoppen en hervatten

Eén Ctrl+C vraagt stoppen na de lopende epoch. Alleen een volledig opgeslagen epoch is hervatbaar. Gebruik daarvoor een checkpoint uit deze freezeproef:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_independent_e2_scale_freeze50_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00052.pt"
```

De limiet blijft epoch 55. Het oorspronkelijke epoch-50-checkpoint hoort bij `run`; een checkpoint van deze vertakking hoort bij `resume`. Het bestaande cuDNN-beleid biedt geen garantie op bitexacte CUDA-hervatting.

## Notebook en feedback

Open `MX_triple_independent_e2_scale_freeze50_pilot5_resultaten.ipynb` vanuit de nieuwe resultatenmap en voer alle cellen uit. De notebook leest CSV/JSON en start geen training. De meegeleverde versie is onuitgevoerd.

Accuracyvergelijkingen gebruiken de gevraagde vorm: varianten op de verticale as, accuracy op de horizontale as, één paneel per precisie en een percentage bij ieder punt. De assen zoomen in zonder weergegeven waarden weg te laten. Ook volledige curves, detail vanaf epoch 50, checkpointselecties, schaaldiagnostiek en kosten blijven aanwezig.

De pilot vergelijkt **epoch 55 met epoch 55**. Het traject 1–50 wordt als overgenomen geschiedenis aangeduid. De controle heeft wel resultaten tot 100, maar die zijn geen nieuwe freeze-resultaten of gelijk eindpunt voor deze pilot.

Deel na afloop `feedback_bundle.zip` uit de nieuwe resultatenmap. Dataset en trainingscheckpoints blijven lokaal. Voortzetten van de vertakking tot epoch 100 volgt pas na beoordeling van deze pilot.
