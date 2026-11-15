# Gezamenlijke directe MX-training — vijf formaten, honderd epochs

Deze proef traint één Any-Precision-netwerk **vanaf nul** met E4M3, E5M2,
E2M3, E3M2 en E2M1. Iedere modus kwantiseert rechtstreeks uit dezelfde actuele
FP32-mastergewichten via de bestaande voorbewerking `G(W)`. Activaties blijven
A32, MX-blokgrootte 32. De numerieke aanpak en distillatie blijven die van de
geslaagde vijf-epochpilot. Het pilotcheckpoint wordt niet ingeladen.

## 1. Uitpakken

Pak de map uit de zip uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_five_direct_reference100_v1\run_train.py`

Gebruik de bestaande `.venv-train` met Python 3.13, torch 2.10.0 en torchvision
0.25.0. De normale proef vereist CUDA. Er is geen automatische CPU-fallback en
het script installeert of upgradet geen afhankelijkheden. Bestaande experimenten
en ruwe data blijven behouden.

## 2. Eerst de technische check

Open PowerShell in je projectmap:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_five_direct_reference100_v1\run_train.py" check
```

Deze check controleert met synthetische invoer de codecs, formaatgebaseerde
selectie, BatchNorm, distillatiedoelen, gezamenlijke updates en compacte exports.
Er wordt geen dataset gedownload of SVHN-accuracy gemeten.

De **laatste terminalregel** is `GLOBAAL RESULTAAT: PASS` of
`GLOBAAL RESULTAAT: FAIL`. Stuur bij FAIL de bijbehorende `feedback_bundle.zip`
terug voordat je de training start.

## 3. De honderd-epochtraining starten

Na een geslaagde check:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_five_direct_reference100_v1\run_train.py" train
```

Dit begint met een nieuwe initialisatie en eindigt op epoch 100. De dataset
wordt standaard gezocht in `data\svhn`. Als die elders staat, voeg dan toe:

```powershell
--data-root "VOLLEDIG_PAD_NAAR_SVHN"
```

Alleen wanneer de dataset ontbreekt, kun je `--download` toevoegen. Bestaande
ruwe bestanden worden op checksum gecontroleerd en nooit overschreven. De
proef gebruikt alle 73.257 SVHN-trainbeelden en 26.032 officiële testbeelden;
de extra-split wordt niet gebruikt.

## Trainingsopzet

| Formaat | Trainingsdoel | Lossgewicht |
|---|---|---:|
| E4M3 | Echte labels | 1 |
| E5M2 | Echte labels | 1 |
| E2M3 | Losgekoppelde E4M3-kansverdeling | 1 |
| E3M2 | Losgekoppelde E5M2-kansverdeling | 1 |
| E2M1 | Gemiddelde van de losgekoppelde E2M3- en E3M2-kansverdelingen | 1 |

Per minibatch: vijf forward/backward-passes, gevolgd door één Adam-update op
de gesommeerde gradiënten. Temperatuur 1. De twee 6-bitkansverdelingen worden
na softmax met gelijke gewichten gemiddeld. Alle gewichtsconversies blijven
direct uit `G(W)`: geen mantissa- of exponenttruncatie, geen geleerde schaal
en geen opgelegde ×64-correctie. Iedere modus heeft een eigen BatchNorm-bank.

CNN-8 (`svhnq`, expand=2), batch 128, seed 20261009, Adam en weight decay 0
blijven behouden. De learningrate is 0,001 voor epochs 1–50, 0,0001 voor 51–75,
0,00001 voor 76–90 en 0,000001 voor 91–100. Een voltooide run bevat 57.300
Adam-updates.

## Resultaten en notebook

Iedere opdracht schrijft naar een nieuwe map onder:

`mx_runs\mx_weights_five_direct_reference100_v1\<tijdstip>_<opdracht>\`

Daar staan onder andere:

- `report.json`: status, configuratie, omgeving, controles en tijdmetingen;
- `metrics.csv`: metingen per epoch, fase en volledig gewichtsformaat;
- `MX_five_direct_reference100_resultaten.ipynb`: uitvoerbare resultatennotebook;
- `reference_data/`: de geverifieerde historische directe drie-formaatreferentie;
- `checkpoints/`: afzonderlijke checkpoints van voltooide epochs;
- `exports/`: vijf compacte directe exports van epoch 100;
- `feedback_bundle.zip`: de bundel om terug te sturen.

Open de notebook uit de resultaatmap met de `.venv-train`-kernel en voer alle
cellen uit. `RUN_DIR` is automatisch ingevuld. De notebook bevat:

1. De vijf testaccuracies op hetzelfde checkpoint van **epoch 100**.
2. Accuracy- en lossverloop, in afzonderlijke panelen per formaat.
3. Een vergelijking met de eerdere **directe any-precisiontraining met drie
   MX-gewichtsformaten**, voor E4M3, E2M3 en E2M1 op epoch 100.
4. Curvenvergelijkingen op dezelfde epochs, met percentages bij de laatste
   gezamenlijke meting.

In de eindpuntvergelijking staan de methoden op een categorische as, met
dezelfde ronde stippen en procentlabels. E5M2 en E3M2 hebben geen oude
referentiescore. Een onvoltooide nieuwe run wordt niet met de oude epoch 100
als voltooid eindresultaat vergeleken. De oorspronkelijke beste E4M3-epoch
wordt niet als vervanging voor epoch 100 gebruikt.

De drie- en vijf-formaattraining hebben verschillende distillatiestructuren
en aantallen passes. De vergelijking is daarom beschrijvend: ze laat de
uitkomst van de gehele uitbreiding zien en is geen afzonderlijke causale
test van uitsluitend twee extra formaten. De oude truncatie- en ×64-runs
worden niet als directe referentie gebruikt.

De gemeten train- en testfasetijden worden afzonderlijk van de sessieduur
gerapporteerd. Datavoorbereiding, checks, checkpointopslag en exports vallen
buiten de som van die fasetijden. Een eventuele raming bij een onvoltooide
run wordt als raming aangegeven.

Stuur na afloop **`feedback_bundle.zip`** terug en bewaar de volledige runmap
op je laptop. Optionele figuurexport schrijft naar een nieuwe analysemap.

## Stoppen en dezelfde training hervatten

Een eerste Ctrl+C vraagt een nette stop na het huidige volledige epoch aan.
Een tweede Ctrl+C breekt onmiddellijk af. Een nette stop vóór epoch 100 heeft
status `STOPPED_AT_EPOCH_BOUNDARY` en een globale FAIL omdat de volledige proef
nog niet is afgewerkt; dit hoeft geen numerieke fout te betekenen.

Hervatten kan vanuit een volledig opgeslagen epoch 1 tot en met 99 van deze
zelfde honderd-epochproef:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_five_direct_reference100_v1\run_train.py" train --resume "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Een pilotcheckpoint of checkpoint uit een ander experiment wordt geweigerd.
Hervatten maakt een nieuwe resultaatmap, controleert de eerdere gegevens en
kopieert de benodigde ongewijzigde epochbestanden. Bewaar de oorspronkelijke
runmap. Een vaste seed garandeert met het behouden CUDA-beleid geen bitidentieke
herhaling of hervatting.

## Later: truncatie

Vóór de truncatievariant behandelen we opnieuw de opslag/omzetting voor beide
8-bitformaten, de numerieke schaalcorrectie per nieuwe exponentroute en één
canonieke E2M1-afleiding. De bestaande ×64-regel wordt niet automatisch
overgenomen. Voor de toekomstige directe/truncatievergelijking houden we de
vijf-formaatdistillatie en lossgewichten gelijk. Zie `PROTOCOL.md` voor alle
numerieke details en bronverwijzingen.
