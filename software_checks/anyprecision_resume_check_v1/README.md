# Any-Precision: technische controle van stoppen en hervatten

Dit pakket start **geen referentietraining**. Het vergelijkt dezelfde kleine
training ononderbroken en met een echt afgesloten/herstart Python-proces.
Gebruik de bestaande `.venv-train`; extra pakketten zijn niet nodig.

## Uitvoeren (PowerShell, vanuit de projectmap)

```powershell
& ".\.venv-train\Scripts\python.exe" -B `
    ".\software_checks\anyprecision_resume_check_v1\run_resume_check.py" `
    --device cuda `
    --data-root ".\data\cifar10" `
    --output-root ".\technical_checks\anyprecision_resume"
```

Het script kiest intern `cuda:0`. Er is geen automatische CPU-terugval.
Er wordt niets gedownload. De zes bestaande CIFAR-batchbestanden worden eerst
gecontroleerd; alleen beelden uit de officiele trainingsset worden ingelezen
voor modelberekeningen. Een ontbrekend of afwijkend bestand geeft een fout.

**Niet zelf Ctrl+C gebruiken, Python afsluiten of de laptop uitschakelen tijdens
deze test.** Het hoofdscript sluit het eerste trainingsdeel normaal af, wacht
op de procesexit en start het vervolg in een nieuw Python-proces. Zo wordt
hervatten getest zonder de laptop daadwerkelijk uit te hoeven zetten.

## Verwachte fasen

1. Controlepad A: mini-epochs 1, 2 en 3 in hetzelfde Python-proces.
2. Controlepad B: mini-epoch 1; checkpoint opslaan; Python-proces stopt.
3. Controlepad B: nieuw Python-proces; checkpoint laden; mini-epochs 2 en 3.
4. Exacte vergelijking van checkpoints en logboeken tussen A en B.

Iedere mini-epoch bevat drie minibatches (8, 8 en 4 beelden). Er zijn 20
trainingsbeelden en 20 andere controlebeelden. Alle vijf precisies worden bij
iedere minibatch doorlopen. Per pad zijn er negen optimizerstappen, dus achttien
in totaal. Deze omvang zegt niets over de snelheid van een volledige epoch.

## Uitvoer

Iedere aanroep maakt een nieuwe map onder `--output-root` met:

- `resume_check_report.json`: hoofdrapport, controles en eventuele fout;
- `run.log`: zichtbare voortgang en fouten;
- `test_config.json`, `source_manifest.json`, `compatibility.patch`;
- `executed_code/`: de werkelijk gebruikte runner- en checkpointcode;
- `uninterrupted/`, `before_stop/`, `after_resume/`: procesrapporten,
  volledige batchtrace, selecties en genummerde checkpoints.

De zes checkpoints kosten samen ongeveer een halve GiB. Bestaande onderzoeksdata
of checkpoints worden niet overschreven of verwijderd. Checkpoints eindigen op
`.pt` en vallen dus onder de eerder toegevoegde `*.pt`-regel in de root-.gitignore.
Dit pakket wijzigt je Git-instellingen niet. Upload geen checkpoints naar GitHub.

Deel `resume_check_report.json` en zo nodig `run.log`; geen `.pt`-bestanden nodig.
`STATUS: PASS` betekent exacte gelijkheid binnen deze geteste configuratie,
**geen** paperaccuracy, MX-resultaat of volledige validatie van een toekomstige
lange runner. Zie `PROTOCOL.md` voor de grens van de test.

## Code inspecteren

- `resume_core.py`: nieuw; opslaan, laden, RNG-toestanden, contractcontrole.
- `resume_worker.py`: nieuw; korte training per afzonderlijk Python-proces.
- `run_resume_check.py`: nieuw; drie processen starten en resultaten vergelijken.
- `run_pilot.py` en `smoke_support.py`: byte-identieke kopie uit het vorige
  pilotpakket, alleen als functiebibliotheek gebruikt; hun main() wordt niet gestart.
- `upstream/Any-Precision-DNNs-master.zip`: byte-identiek aan de aangeleverde bron.
- `PACKAGE_SHA256.json`: controlegetallen van uitvoerbestanden en bronarchief.

`local_validation/` bevat afzonderlijke ontwikkelaarstests met kunstmatige
CIFAR-bestanden. Deze worden NIET door het bovenstaande commando gebruikt.
