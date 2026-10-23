# Gezamenlijke directe MX-referentie — 100 epochs

Eén Any-Precision-netwerk met E4M3, E2M3 en E2M1. Iedere gewichtsmodus wordt
onafhankelijk rechtstreeks uit dezelfde actuele FP32-master via het bestaande
G(W) bepaald. Activaties blijven A32; masters, gradients en Adam-momenten FP32.
Deze proef gebruikt geen mantissa- of exponenttruncatie en geen schaalcorrectie
of geleerde schaal.

Dit is het afzonderlijke honderd-epochvervolg op de geslaagde vijf-epochpilot.
De modelcode, drie codecs, gewichtsconversie, BN-mapping en gezamenlijke
trainingsstap zijn byte-identiek aan het pilotpakket. De training start vanaf
dezelfde vaste seed 20261009. Het vijf-epochcheckpoint wordt niet ingeladen;
een volledige referentie heeft hier 100 nieuwe epochs vanaf initialisatie.
Alle bestaande pakketten, checkpoints en resultaten blijven behouden.

## Uitpakken en starten

Pak de map uit deze ZIP uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_triple_direct_reference100_v1\run_train.py`

Gebruik dezelfde `.venv-train` als voor de geslaagde pilot. Er zijn geen nieuwe
laptopafhankelijkheden of upgrades nodig. De volledige run start met:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_triple_direct_reference100_v1\run_train.py" run
```

De korte controle is ook beschikbaar. Ze is optioneel na de geslaagde pilot,
omdat de numerieke kern en trainingsstap ongewijzigd zijn:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_triple_direct_reference100_v1\run_train.py" check
```

`check` gebruikt uitsluitend synthetische input op de GPU, geen SVHN-training.
Bij een FAIL: deel de nieuwe feedbackbundel en start de volledige run nog niet.
De dataset staat standaard in `data\svhn`; gebruik bij een afwijkende bestaande
locatie dezelfde `--data-root "PAD"` als eerder. Bestaande data worden alleen
gelezen en gecontroleerd. Er wordt zonder `--download` niets gedownload.

## Training en looptijd

Per minibatch: E4M3 hard-label CE, E2M3 soft-target CE van losgekoppelde E4M3,
E2M1 soft-target CE van losgekoppelde E2M3; daarna één Adam-update op de som van
de gradients. Temperatuur 1, geen deling door drie. De distillatie van outputs
verandert de onafhankelijke directe gewichtsconversie niet.

De volledige SVHN train-split bevat 73.257 beelden en de officiële testset
26.032; geen extra-split. Batch 128, seed 20261009, Adam, weight decay 0.

| Epochs | Learning rate |
|---|---:|
| 1–50 | 0,001 |
| 51–75 | 0,0001 |
| 76–90 | 0,00001 |
| 91–100 | 0,000001 |

De scheduler wordt na de train- en testfase van elke epoch bijgewerkt, zoals
in de pilot. Een voltooide run heeft 57.300 Adam-updates en 171.900
trainingspasses over een gewichtsmodus. Alle drie modi worden na iedere epoch
op de volledige testset geëvalueerd. Alleen een voltooide epoch 100 krijgt
bij training de status `PASS`; een nette eerdere stop krijgt
`STOPPED_AT_EPOCH_BOUNDARY`.

De echte pilot duurde 5m23s inclusief sessie-overhead; de vijf epochtijden
bedroegen samen 5m16s. De projectie voor een nieuwe honderd-epochrun is
1h46m21s onder vergelijkbare omstandigheden. Dat is een raming, geen gemeten
honderd-epochduur. Langdurige laptopbelasting kan de looptijd beïnvloeden.
De nieuwe run rapporteert de werkelijk gemeten tijd; bij hervatten zijn de
som van alle voltooide epochtijden en de duur van de huidige sessie gescheiden.

## Resultaten en notebook

Nieuwe resultaten staan onder:

`mx_runs\mx_weights_triple_direct_reference100\<tijdstip>_joint_run\`

Open in de resultaatmap `MX_triple_direct_reference100_resultaten.ipynb`, kies
de bestaande `.venv-train`-kernel en voer alle cellen uit. De runner vult
`RUN_DIR` automatisch in. Alleen de lege uitvoerbare notebook wordt geleverd;
er is geen tweede, al uitgevoerde versie.

De notebook bevat:

- de accuracyvergelijking van E4M3, E2M3 en E2M1 op **epoch 100**;
- dezelfde drie scores op **het checkpoint met de hoogste E4M3-testaccuracy**;
- train- en testaccuracycurves voor de drie precisies;
- apart gelabelde harde en zachte trainingsobjectieven en harde test-CE;
- gemeten tijden, GPU-geheugen en controles op datadekking, gradients,
  optimizer, BN en exports.

Alle drie scores bij het beste checkpoint komen uit dezelfde epoch, gekozen
op E4M3. Bij een gelijke hoogste E4M3-score blijft de eerste epoch behouden.
De notebook neemt geen afzonderlijke maxima per precisie. Als epoch 100 nog
niet bestaat, wordt dat expliciet getoond. Een andere laatste epoch wordt
niet als epoch 100 gepresenteerd. Selectie op de officiële testset betekent
dat de gekozen beste score geen onafhankelijke testschatting na selectie is.

Het opslaan van figuren maakt een nieuwe analysemap. Ruwe rapporten en
checkpoints blijven behouden; de notebook start geen training.

Stuur na afloop `feedback_bundle.zip` uit de nieuwe `_joint_run`-map. Bewaar
zelf de volledige runmap: de bundel bevat rapporten en analysecode, geen
datasets, trainingscheckpoints of MX-modelbestanden. Alle 100 epochcheckpoints
en de geselecteerde best/last-checkpoints blijven beschikbaar voor latere
post-hoctruncatieproeven. Per best/last worden bovendien drie afzonderlijke
directe MX-modellen geëxporteerd.

## Stoppen en hervatten

Eén `Ctrl+C` vraagt een stop na de lopende epoch, inclusief evaluatie,
checkpoint en rapportage. Een tweede `Ctrl+C` onderbreekt direct. Een leeg
bestand `STOP_AFTER_EPOCH` in de actieve runmap vraagt eveneens een stop op
de epochgrens. Hervat alleen een volledig opgeslagen checkpoint met de
bijbehorende `.sha256`, uit ditzelfde honderd-epochpakket:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_triple_direct_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD\checkpoints\epoch_00042.pt"
```

Gebruik uiteraard de daadwerkelijk bereikte epoch. Hervatten schrijft een
nieuwe resultaatmap en gaat door tot epoch 100; optimizer, scheduler en RNG
worden hersteld. Bewaar ook de oorspronkelijke runmap met alle eerdere
checkpoints en epochrapporten, omdat de gecombineerde geschiedenis daarnaar
kan verwijzen. Checkpoints uit de pilot of andere experimenten worden
geweigerd. Een al voltooide epoch 100 kan niet opnieuw worden hervat.

De oorspronkelijke cuDNN-instellingen blijven behouden; bitexacte CUDA-
herhaling of hervatting wordt niet gegarandeerd. Pas pakket of afhankelijkheden
tijdens een run niet aan: het hervatcontract controleert hun identiteit.

## Vastgelegde pilotuitkomst

Bron: `feedback_bundle(20261023-125617).zip`, run
`20261023_144246_066123_joint_run`, status PASS. Epoch 5 is zowel laatste als
beste checkpoint op E4M3.

| Gewichtsmodus | Epoch 5 testaccuracy | Correct / 26.032 |
|---|---:|---:|
| E4M3 | 93,1085% | 24.238 |
| E2M3 | 93,0662% | 24.227 |
| E2M1 | 92,7973% | 24.157 |

Dit toont een werkende gezamenlijke vijf-epochproef. Het voorspelt geen
honderd-epochaccuracy of algemene superioriteit uit één seed.

## Technische documentatie

`PROTOCOL.md` beschrijft conversie, training en checkpointselectie.
`CODEWIJZIGINGEN.md` en de manifests leggen het hergebruik vast.
`local_validation/VALIDATION.md` beschrijft de lokaal uitgevoerde controles.
In deze leveromgeving is geen volledige SVHN- of CUDA-training gestart.
