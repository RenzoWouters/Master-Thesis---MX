# Dedicated E4M3/A32 — proef van vijf epochs

Deze proef voegt uitsluitend de nieuwe **E4M3-gewichtsmodus** toe aan de
bestaande dedicated trainingsopzet. CNN-8, SVHN train-only, A32 en de
FP32-updatemasters blijven behouden. Zij levert een technische controle,
eerste leercurves en een gemeten tijdschatting voor een latere honderd-epochrun.
Er zijn nog geen nieuwe SVHN-resultaten in dit pakket.

De afgesproken mantissatruncatie blijft onderdeel van de volgende gezamenlijke
proef met E4M3/E2M3/E2M1. Deze dedicated proef voert één modus uit en heeft dus
nog geen lagere precisiepass, exponenttruncatie of distillatie.

## Correctie in v2

De GPU-controle van v1 stopte bij de exacte vergelijking van de 254 eindige
E4M3-codes. V2 vormt de elementwaarden met gehele bitbewerkingen en de machten
van twee met exacte FP64-exponentbits. Formaat, afronding, schaalkeuze en
trainingsprotocol zijn behouden; dit is geen schaalcompensatie.

De nieuwe `check` bewaart ook `legacy_codec_diagnostics.json` met de werkelijke
v1-afwijkingen op jouw GPU en `codec_report.json` met de strikte v2-controles.
Een gemelde v1-afwijking is diagnostiek; de eindstatus vereist dat de nieuwe
codec en de modelcontrole slagen. De v2-codectest vergelijkt aanvullend alle
255 geldige schaalfactoren en alle 64.770 element/schaal-combinaties exact.
Zie `CORRECTIE_V2.md` voor de oorzaakafbakening en de uitgevoerde controles.

## 1. Uitpakken

Pak deze map uit naast de bestaande pakketten, zodat het bestand hier staat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_e4m3_dedicated_pilot5_v2\run_train.py`

Overschrijf geen bestaande map. De werkende omgeving blijft `.venv-train`.
Het pakket bevat de oorspronkelijke Any-Precision-auteurscode uit de overdracht;
er is geen nieuwe GitHub-download of gewijzigde architectuur nodig.

Open PowerShell in de projectmap:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
```

## 2. Eerst de korte controle op je GPU

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e4m3_dedicated_pilot5_v2\run_train.py" check
```

Dit controleert de codec, het echte CNN-8-model, twee synthetische updates,
FP32-gradients/Adam, BatchNorm en de export/reconstructie. Het laadt of downloadt
geen SVHN-beelden. Ga verder wanneer de eindstatus `PASS` is. Bij een fout:
bewaar de nieuwe resultaatmap en deel de `feedback_bundle.zip` uit die map.

## 3. Vijf volledige epochs uitvoeren

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e4m3_dedicated_pilot5_v2\run_train.py" run
```

De bestaande bestanden `data\svhn\train_32x32.mat` en
`data\svhn\test_32x32.mat` worden gecontroleerd en hergebruikt. De `extra`-split
wordt niet geopend. Alleen wanneer de dataset ontbreekt, is hetzelfde commando
met `--download` nodig. Een bestaand bestand dat de checksumcontrole niet haalt,
blijft behouden; de proef stopt dan.

Het pakket stopt op epoch 5. `--stop-after-epoch 100` wordt geweigerd. Na deze
pilot beoordelen we eerst werking, leercurves en gemeten duur. Een latere
honderd-epochrun krijgt een apart pakket en begint opnieuw vanaf initialisatie.

## 4. Resultaten en notebook

Elke opdracht schrijft een nieuwe map onder:

`mx_runs\mx_e4m3_dedicated_pilot5\`

Gebruik voor de onderzoeksresultaten de map die eindigt op `_mx_run`, niet de
map van `check`. Open daarin `MX_E4M3_dedicated_pilot5_resultaten.ipynb` met de
`.venv-train`-kernel in VS Code en voer alle cellen uit. De runner vult `RUN_DIR`
automatisch in en kopieert `notebook_support.py` erbij. Bij verplaatsen van de
map wijzig je alleen `RUN_DIR` in het notebook, niet de ruwe rapporten.

Het notebook bevat:

- accuracy- en hard-label-lossleercurves voor train en test;
- accuracyfiguren en exacte cijfers voor het laatste checkpoint (hier epoch 5)
  en het beste checkpoint volgens E4M3-testaccuracy;
- een expliciete melding dat epoch 100 nog niet is uitgevoerd; geen fictieve
  of geëxtrapoleerde eindaccuracy;
- epochtijden en een tijdschatting voor honderd epochs, met een bereik op basis
  van gemeten warme epochs; dit bereik is geen betrouwbaarheidsinterval;
- status, waarschuwingen, configuratie en controle-informatie.

Figuren worden bij `SAVE_FIGURES=True` in een nieuwe `analysis_<tijdstip>`-map
geschreven. De notebook leest JSON/CSV en start geen training. Gebruik de
bestaande pandas/matplotlib/ipykernel-installatie; dit pakket upgrade niets.

Deel na de proef **`feedback_bundle.zip` uit de `_mx_run`-map**. Deze bevat de
rapporten en de analysecode, geen dataset of trainingscheckpoints. Bewaar de
hele oorspronkelijke runmap. De direct na training gemaakte feedbackbundel
bevat nog geen later in het notebook opgeslagen figuren; die blijven in de
aparte analysemap beschikbaar om te delen met je begeleider.

## 5. Stoppen en hervatten als dat nodig is

Eén `Ctrl+C` vraagt een stop na de lopende epoch, inclusief evaluatie en
checkpoint. Een tweede `Ctrl+C` stopt direct. Alleen een volledig opgeslagen
epochcheckpoint is hervatbaar. Ook een leeg bestand `STOP_AFTER_EPOCH` in de
actieve runmap vraagt een stop na de epoch.

Voorbeeld: hervat deze pilot vanaf een werkelijk opgeslagen epoch 2:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_e4m3_dedicated_pilot5_v2\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00002.pt"
```

Hervatten schrijft een nieuwe resultaatmap en gaat alleen door tot epoch 5.
Behoud ook de oorspronkelijke runmap met checkpoints en epochrapporten. De
check controleert hetzelfde E4M3-protocol, softwareomgeving en bronbestanden.
Een oud E2M3-checkpoint of een voltooide vijf-epochpilot wordt niet als nieuwe
trainingsstart gebruikt. De bestaande cuDNN-instellingen beloven geen bitexacte
CUDA-herhaalbaarheid.

## Verdere documentatie

- `PROTOCOL.md`: de numerieke en trainingskeuzes.
- `CODEWIJZIGINGEN.md`: onderzoekswijzigingen en behouden compatibiliteitslaag.
- `local_validation/VALIDATION.md`: werkelijk uitgevoerde lokale controles en
  de grens tussen CPU-validatie en de nog uit te voeren laptopproef.
- `REUSE_MANIFEST.json`: herkomst en hashes van hergebruikte bestanden.

