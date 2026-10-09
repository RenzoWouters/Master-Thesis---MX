# Wijzigingslog tegenover de aangeleverde Any-Precision-code en vorige pilot

## Originele bestanden: behouden

`upstream/Any-Precision-DNNs-master.zip` is byte-identiek aan
`Any-Precision-DNNs-master(1).zip` uit deze conversatie.
SHA-256: 7e36c5f039a78e3174421de7ff097066f95dde2837ea02f11913f3be0ba7f73c

Ook `run_pilot.py` en `smoke_support.py` zijn byte-identiek aan de bestanden in
`anyprecision_cifar10_pilot_v1.zip`. Ze worden geimporteerd, niet als zelfstandig
programma gestart. De vorige bronbestanden/resultaten op je laptop worden niet
gelezen als hervatcheckpoint en niet gewijzigd.

## Compatibiliteitsaanpassingen: uitsluitend een nieuwe uitgepakte werkkopie

Exact dezelfde patch als voor de geslaagde pilot:
1. `utils.py`: `correct[:k].view(-1)` wordt `correct[:k].reshape(-1)`.
2. `utils.py`: `import gpustat` uitsluitend binnen `setup_gpus()`.
3. `datasets/data.py`: `SVHNMIX` pas binnen de SVHN-tak importeren.

Geen aanvullende wijzigingen aan oorspronkelijke bronbestanden. De totale patch
staat na starten in `compatibility.patch`. Model, quantizer, STE, loss, optimizer,
CIFAR10-loader en beeldtransformaties worden niet gewijzigd.

## Nieuwe uitvoer- en checkpointcode

- `resume_core.py`: RNG-state opslaan/herstellen; checkpoint opslaan/laden met
  versie/omgeving/protocolcontrole; geen overschrijven van oude checkpoints.
- `resume_worker.py`: een klein trainingssegment. Hergebruikt pilot.train_batch,
  pilot.evaluate, pilot.summarize, de dataset/splitfuncties en bronmodellen.
- `run_resume_check.py`: drie losse processen achtereen uitvoeren en vergelijken.

Verschil met pilot: doorlopende toevalstoestanden en persistente DataLoader-
generators, in plaats van per-epoch opnieuw seeden. Dit is gedocumenteerde
uitvoerlogica; het is geen model- of quantisatiemethodewijziging.

## Technische testinstellingen, niet het onderzoeksprotocol

20+20 beelden, batch 8, drie mini-epochs en opzettelijk vroege LR-overgangen na
mini-epochs 1 en 2. Hierdoor controleren we herstel zonder honderden epochs.
Nieuwe directories voor beide paden. De toekomstige referentie begint opnieuw.

## Lokale validatie

Zie `local_validation/VALIDATION.md`. Kunstmatige data uit ontwikkelaartests mogen
niet als echte CIFAR-10-data/accuracy worden gerapporteerd. Het normale
`run_resume_check.py`-commando gebruikt uitsluitend geverifieerde bestaande data
op jouw laptop en bevat geen fixturekeuze of downloadoptie.

## Git/archief

Een `.gitattributes` binnen alleen deze pakketmap legt LF-regelafbrekingen vast
voor Python/JSON/Markdown/patchbestanden en behandelt ZIP als binair. De root-
`.gitignore` en Git-geschiedenis worden niet gewijzigd. Grote lokale
controlecheckpoints worden niet in het downloadpakket opgenomen.
