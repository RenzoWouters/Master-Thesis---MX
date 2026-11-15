# Gezamenlijke directe MX-training — vijf formaten, vijf epochs

Deze proef traint één Any-Precision-netwerk vanaf nul met **E4M3, E5M2, E2M3,
E3M2 en E2M1**. Ieder gewichtsformaat wordt rechtstreeks uit dezelfde actuele
FP32-mastergewichten via de bestaande voorbewerking `G(W)` gekwantiseerd.
Activaties blijven A32 en de MX-blokgrootte blijft 32.

Dit is de afgesproken **vijf-epochpilot**, met een nieuwe distillatiestructuur
voor de vijf formaten. Een latere volledige honderd-epochreferentie begint
opnieuw vanaf nul. Dit pakket start die volledige training niet.

## 1. Uitpakken

Pak de map uit de zip uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_five_direct_pilot5_v1\run_train.py`

De bestaande experimentmappen en ruwe onderzoeksdata blijven behouden. Gebruik
de bestaande `.venv-train` met Python 3.13, torch 2.10.0 en torchvision 0.25.0.
De normale proef vereist CUDA en schakelt niet stilzwijgend naar CPU over.
Het script installeert of upgradet geen afhankelijkheden.

## 2. Eerst de technische check

Open PowerShell in je projectmap:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_five_direct_pilot5_v1\run_train.py" check
```

`check` controleert de numerieke omzettingen, de selectie per formaat, de
BatchNorm-toestanden, de distillatiedoelen en de gezamenlijke gewichtsupdate
met synthetische invoer. Er wordt daarbij geen SVHN-accuracy gemeten en geen
dataset gedownload.

**De laatste terminalregel is:**

```text
GLOBAAL RESULTAAT: PASS
```

of, als een controle mislukt:

```text
GLOBAAL RESULTAAT: FAIL
```

Bij FAIL: stuur de nieuwe `feedback_bundle.zip` terug en start de training
nog niet. De voorafgaande foutmelding en `report.json` geven de oorzaak aan.

## 3. Daarna de vijf-epochtraining

Na een geslaagde check:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_five_direct_pilot5_v1\run_train.py" train
```

De dataset wordt standaard gezocht onder `data\svhn` in de projectmap.
Als jouw bestaande dataset ergens anders staat:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_five_direct_pilot5_v1\run_train.py" train --data-root "VOLLEDIG_PAD_NAAR_SVHN"
```

Alleen wanneer de dataset ontbreekt, kun je `--download` toevoegen. Bestaande
ruwe bestanden worden op hun checksum gecontroleerd en nooit overschreven.
De proef gebruikt de volledige SVHN train-split en officiële testset; de
extra-split wordt niet gebruikt.

Een voltooide training heeft **vijf epochs en 2.865 Adam-updates**. De runner
rapporteert ook hier een globale PASS of FAIL als laatste terminalregel.
Een op een epochgrens gestopte run met minder dan vijf epochs krijgt
`STOPPED_AT_EPOCH_BOUNDARY` in het rapport en een globale FAIL, omdat het afgesproken eindpunt
nog niet is bereikt; dat hoeft geen numerieke fout te betekenen.

## Wat wordt gezamenlijk geleerd?

| Formaat | Trainingsdoel | Lossgewicht |
|---|---|---:|
| E4M3 | Echte labels | 1 |
| E5M2 | Echte labels | 1 |
| E2M3 | Losgekoppelde E4M3-kansverdeling | 1 |
| E3M2 | Losgekoppelde E5M2-kansverdeling | 1 |
| E2M1 | Gemiddelde van de losgekoppelde E2M3- en E3M2-kansverdelingen | 1 |

Per minibatch gebeuren vijf forward/backward-passes, gevolgd door één
Adam-update op de som van de gradiënten. Temperatuur is 1. De twee 6-bitkansen
worden na softmax met gelijke gewichten gemiddeld; E2M1 krijgt daardoor één
lossbijdrage. De teachers zijn modi van hetzelfde meelerende netwerk.

Dit betreft distillatie van voorspellingen. Alle vijf gewichtsconversies
blijven rechtstreeks uit `G(W)` gebeuren: zonder mantissa- of exponenttruncatie,
zonder geleerde schaal en zonder een opgelegde ×64-schaalcorrectie.

De instellingen blijven: CNN-8 (`svhnq`, expand=2), batch 128, seed 20261009,
Adam met lr 0,001 en weight decay 0. De oorspronkelijke mijlpalen 50/75/90
blijven in het protocol vastgelegd; in deze vijf epochs vindt geen lr-daling
plaats. Iedere formaatmodus gebruikt zijn eigen BatchNorm-toestand.

## Resultaten en feedback

Iedere opdracht maakt een nieuwe map onder:

`mx_runs\mx_weights_five_direct_pilot5_v1\<tijdstip>_<opdracht>\`

Na de training vind je daar onder andere:

- `report.json`: status, instellingen, omgeving, controles en tijdmetingen;
- `metrics.csv`: metingen per epoch, fase en volledig gewichtsformaat;
- `MX_five_direct_pilot5_resultaten.ipynb`: uitvoerbare resultatennotebook;
- checkpoints om dezelfde pilot na een onderbreking te hervatten;
- `feedback_bundle.zip`: de bundel om terug te sturen.

Open de notebook in de resultaatmap, kies de bestaande `.venv-train`-kernel
en voer alle cellen uit. De runner vult `RUN_DIR` automatisch in. De
meegeleverde notebook is leeg en uitvoerbaar; er staan geen verzonnen
meetresultaten in.

De notebook geeft de vijf testaccuracies op **hetzelfde checkpoint van epoch
5**, curves in afzonderlijke panelen per formaat, percentages bij de punten,
harde cross-entropy en de daadwerkelijke trainingsobjectieven. Ontbreekt een
volledige epoch 5, dan wordt geen vroegere epoch als eindresultaat vervangen.
Er is geen selectie van een beste epoch op de testset. De primaire tabel
vereist ook `completed_epochs == 5` in het rapport, zodat eventuele metingen
vóór een mislukte checkpointopslag geen voltooid eindpunt voorstellen.

De looptijd voor een mogelijke honderd-epochtraining wordt uit deze pilot
**geraamd** uit de train- en testfasetijden. Datavoorbereiding, controles,
checkpointopslag en exports vallen buiten die raming. De werkelijke
honderd-epochduur en uiteindelijke accuracy zijn nog niet gemeten. De vijf-epochpilot dient om correctheid, leerverloop en
praktische looptijd te beoordelen.

Stuur na afloop de nieuwe **`feedback_bundle.zip`** terug. Bewaar daarnaast de
volledige runmap met checkpoints op je laptop. De notebook kan optioneel
figuren schrijven naar een nieuwe analysemap; ruwe meetbestanden worden niet
gewijzigd.

## Dezelfde pilot hervatten

Als een volledig opgeslagen checkpoint van epoch 1 tot en met 4 beschikbaar
is, kun je daarmee dezezelfde pilot tot en met epoch 5 afmaken:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_five_direct_pilot5_v1\run_train.py" train --resume "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Een checkpoint uit een ander experiment of een reeds voltooide epoch 5 wordt
niet gebruikt als nieuwe start. Hervatten schrijft naar een nieuwe runmap;
bewaar ook de oorspronkelijke runmap. Door het behouden CUDA-backendbeleid
is bitidentieke herhaling of hervatting niet gegarandeerd.

## Vervolg naar truncatie

Bij de latere truncatiefase behandelen we opnieuw drie ontwerpkeuzes:

1. Hoe worden beide verschillende 8-bitformaten vanuit de opgeslagen
   gewichtsrepresentatie beschikbaar? Een omzetting tussen E4M3 en E5M2 is
   geen eenvoudige verkorting door bits te verwijderen.
2. Welke schaalcorrectie is nodig voor iedere nieuwe route? De bestaande
   ×64-regel wordt niet automatisch overgenomen.
3. Langs welke ene vastgelegde route wordt de uiteindelijke E2M1-modus afgeleid?

Voor een eerlijke vergelijking blijven de vijf-formaatdistillatie en
lossgewichten dan gelijk aan deze referentie. Het volledige protocol en de
bronverwijzingen staan in `PROTOCOL.md`.
