# Kleine Any-Precision CIFAR-10-pilot

Gebruik de bestaande Python 3.13 / PyTorch 2.10.0+cu128 / torchvision0.25.0+cu128
omgeving. Geen extra pakketten installeren. Lees eerst PROTOCOL.md en CHANGELOG.md.

Vanuit C:\Users\renzo\master-thesis-mx:

```powershell
& ".\.venv-train\Scripts\python.exe" -B `
    ".\software_checks\anyprecision_cifar10_pilot_v1\run_pilot.py" `
    --device cuda `
    --data-root ".\data\cifar10" `
    --output-root ".\pilot_runs\anyprecision_cifar10" `
    --download
```

`--download` geeft toestemming voor het officiele CIFAR-10-archief wanneer data
nog ontbreken. Bestaande correcte batches worden alleen gecontroleerd en gelezen.
Corrupte bestaande data worden niet vervangen: bewaar de foutmelding en stop.
Bij een netwerk-/certificaatfout geen TLS omzeilen; deel eerst de foutmelding.

Output in een nieuwe tijdgestempelde map:

- `pilot_report.json`: instellingen, checks, resultaten, fouten en waarschuwingen;
- `metrics.csv`: epoch0-validatie en train/validatie per epoch en precisie;
- `batch_metrics.jsonl`: exacte minibatchindices, invoerhashes en lossdiagnostiek;
- `split_indices.json`: volledige 45k/5k-pools en de kleinere pilotselecties;
- `train_order_epoch_*.json`: daadwerkelijk gebruikte samplevolgorde;
- `protocol.json`, PROTOCOL.md, CHANGELOG.md: vastgelegde keuzes;
- `source_workcopy`, `source_manifest.json`, `compatibility.patch`: gebruikte code;
- `runner_used.py`, `support_used.py`, `pip_freeze.txt`: uitvoeromgeving;
- `pilot_checkpoint.pt`: laatste model/optimizer/scheduler, geen compact lage-bitmodel;
- `run.log`: voortgang en foutmeldingen.

Deel na de run `pilot_report.json` en `metrics.csv`, bij fouten ook `run.log`.
Upload het checkpoint of de dataset niet.

De synthetische smoke test en experiment001-004 blijven apart. Elke nieuwe pilotrun
begint opnieuw vanaf een vaste modelinitialisatie; dit script hervat geen training.
Een tweede aanroep is een nieuwe herhaling in een nieuwe outputmap, geen overwrite.
De submap local_validation beschrijft ontwikkelcontroles en is geen researchdata.
