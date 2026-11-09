# Geleerde schaalkeuze ×32/×64 — gezamenlijke pilot van vijf epochs

Dit pakket traint dezelfde drie MX-gewichtsmodi met één gedeelde E4M3-bron. Iedere groep van 32 gewichten krijgt één FP32-hulpwaarde die kiest tussen bronschaal ×32 en ×64 voor E2M3 en E2M1. De exponent- en mantissabewerkingen blijven behouden. Activaties zijn A32; mastergewichten, gradients en Adam-momenten zijn FP32.

De pilot begint vanaf de oorspronkelijke modelinitialisatie met seed 20261009. Alle 4.080 schaalhulpwaarden beginnen op 6, dus aanvankelijk ×64. De echte forward gebruikt uitsluitend offset 5 of 6; de besproken logaritmische STE geeft een schaalgradient. Er worden geen extra netwerkpasses gedaan om keuzes afzonderlijk uit te proberen.

## Uitpakken en controleren

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. Het script moet dan hier staan:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_learned_scale32_64_pilot5_v1\run_train.py
```

Gebruik de bestaande `.venv-train`. Er worden geen pakketten automatisch geïnstalleerd of bijgewerkt.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale32_64_pilot5_v1\run_train.py" check
```

`check` voert codec-, gradient-, projectie- en exportcontroles uit op synthetische tensors en enkele synthetische modelpasses. Het downloadt geen dataset en start geen SVHN-training. De afzonderlijke resultaatmap bevat bij problemen een `feedback_bundle.zip`.

## De vijf-epochtraining starten

Start na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale32_64_pilot5_v1\run_train.py" run
```

Elke opdracht krijgt een nieuwe map onder:

```text
mx_runs\mx_triple_learned_scale32_64_pilot5
```

De bestaande SVHN-bestanden onder `data\svhn` worden hergebruikt. Indien nodig kan `--data-root "PAD_NAAR_SVHN"` worden opgegeven; alleen bij ontbrekende bestanden is `--download` nodig. De extra-split wordt niet gebruikt.

Het script stopt vast na vijf volledige epochs: 2.865 optimizerupdates en 8.595 trainingspasses. Het oorspronkelijke honderd-epoch-LR-schema blijft geconfigureerd, met dalingen na 50/75/90; die worden tijdens deze pilot nog niet bereikt. Zowel de gewichten als de schaalhulpwaarden hebben een eigen Adam-parametergroep met leersnelheid 0,001, zonder weight decay. Na iedere gezamenlijke Adam-update worden de hulpwaarden naar [5,6] geprojecteerd.

De eerdere vaste-schaalpilot duurde ongeveer vijf minuten. De nieuwe schaalgradient en diagnostiek voegen werk toe; de werkelijke nieuwe looptijd en de projectie voor honderd epochs worden door deze pilot gemeten. Een projectie is geen reeds uitgevoerde volledige training.

## Stoppen en hervatten

Eén `Ctrl+C` vraagt een stop na de huidige volledige epoch. Een tweede `Ctrl+C` stopt onmiddellijk; alleen een volledig opgeslagen checkpoint is hervatbaar. Een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultaatmap werkt eveneens.

Hervat een onvoltooide run van dit pakket met:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale32_64_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Modelgewichten, schaalhulpwaarden, beide Adam-groepen, scheduler, RNG en de volledige epochgeschiedenis worden hersteld. Hervatten maakt een nieuwe sessiemap. Checkpoints van eerdere vaste-schaalproeven worden niet als warmstart geaccepteerd. De oorspronkelijke CUDA-backend biedt geen bitexacte herhaal- of hervatgarantie.

## Notebook

Open in de resultaatmap `MX_triple_learned_scale32_64_pilot5_resultaten.ipynb`. Het resultatenpad is door de runner ingevuld. De notebook wordt met lege outputs geleverd en start geen training.

De notebook toont opnieuw accuracygrafieken voor alle drie modi op **epoch 5** en op het **gezamenlijke checkpoint gekozen door de hoogste E4M3-testaccuracy**. De twee lagere scores komen steeds uit datzelfde checkpoint. De vaste ×32- en ×64-referenties zijn meegeleverd; vergelijkingen gebruiken alleen de gezamenlijk afgewerkte eerste epochs. De beste epochs uit hun latere volledige training worden niet in de pilotvergelijking gebruikt.

Nieuwe figuren tonen de fractie ×32-keuzes per laag, de verdeling van FP32-hulpwaarden, omschakelingen tijdens iedere epoch en schaalgradienten. De tabellen maken onderscheid tussen een eindkeuze die is veranderd en een blok dat tijdens de epoch meerdere keren heen en weer schakelde. Een onderbroken run blijft herkenbaar onvolledig.

Alleen eindige, geldige berekeningen en correcte contracts zijn vereist voor een technische PASS. Lage accuracy, nulgradienten of nog geen harde omschakelingen zijn mogelijke onderzoeksresultaten. De hulpwaarden kunnen al veranderen voordat ze de grens 5,5 bereiken.

## Bestanden om te bewaren en te delen

Alle epochcheckpoints, best/last-checkpoints en compacte MX-exports blijven lokaal beschikbaar. De hervatbare checkpoints bevatten de FP32-schaalhulpwaarden en Adam-staat. De compacte bronexport bewaart de E4M3-codes en schalen, plus één gepakt keuzebit per blok: 510 bytes voor dit model. De lagere modi zijn volledig uit die bron en keuzebits afleidbaar.

Deel na afloop `feedback_bundle.zip` uit de resultaatmap van de training. Deze bevat metrics, schaaldiagnostiek, de notebook en vaste referenties; de grote modelcheckpoints en datasetbestanden blijven op de laptop.

Details staan in `PROTOCOL.md`, `CODEWIJZIGINGEN.md` en `local_validation/VALIDATION.md`. Eerdere experimenten en resultaten worden behouden.
