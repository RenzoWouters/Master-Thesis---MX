# Vijf epochs — E4M3-bron met exponent- en mantissatruncatie tijdens training

Deze proef traint de drie MX-gewichtsmodi gezamenlijk vanaf dezelfde verse
initialisatie als de directe referentie. E4M3 is de enige gekwantiseerde bron.
E2M3 ontstaat door de hoogste twee exponentbits te behouden; E2M1 ontstaat
vervolgens door de twee laagste mantissabits weg te laten. Alle activaties zijn
A32 en de gedeelde updatemasters, gradients en Adam-momenten zijn FP32.

| Modus | Gewichtsroute | Schaal |
|---|---|---|
| E4M3 | Directe RNE-kwantisatie van de actuele G(W) | Bestaande E4M3-schaalregel |
| E2M3 | Hoogste twee exponentbits van dezelfde E4M3-code | Exact dezelfde E4M3-schaalbytes |
| E2M1 | De afgeleide E2M3-code twee bits rechts schuiven | Exact dezelfde E4M3-schaalbytes |

De bronschaal wordt uit de actuele gewichten bepaald en kan na een optimizer-
update veranderen. Er is geen correctie van die schaal voor de lagere modi,
geen biascompensatie en geen aparte leerbare schaalparameter. BN-parameters
en running statistics leren tijdens de training normaal per gewichtsmodus.
Tijdens evaluatie zijn zij bevroren.

## Uitpakken

Pak de pakketmap uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_truncation_aware_pilot5_v1\run_train.py`

Gebruik de bestaande `.venv-train`; het pakket installeert of upgradet niets.
Het bevat een nieuwe experimentmap en verandert eerdere proeven niet.

## 1. Eerst de controle

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_truncation_aware_pilot5_v1\run_train.py" check
```

Dit controleert de bestaande codecs, de exacte bitroutes, schaalbehoud,
gedeelde bron per minibatch, STE, recursieve training, BN, Adam en source-only
exports op synthetische input. Er wordt geen SVHN-training gestart.
Bij FAIL: deel de `feedback_bundle.zip` uit de nieuwe checkmap.

## 2. Na CHECK PASS: vijf epochs trainen

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_truncation_aware_pilot5_v1\run_train.py" run
```

De dataset staat standaard onder `data\svhn`. Gebruik `--data-root "PAD"`
als die elders staat. Alleen de volledige train- en test-split worden gebruikt;
de extra-split wordt niet geladen. De bestaande bestanden worden gecontroleerd.
De optionele bestaande downloadmogelijkheid wordt alleen met `--download`
ingeschakeld; voor de reeds aanwezige data is die vlag niet nodig.

De runner begint vanaf seed 20261009, met dezelfde oorspronkelijke model-
initialisatie. Hij laadt geen checkpoint uit de directe training of post-hoc-
proef. Het LR-schema blijft dat van honderd epochs: lr 0,001, dalingen na
50/75/90. Deze pilot stopt vast na epoch 5; `--stop-after-epoch 100` is niet
toegestaan. Alle vijf pilotepochs gebruiken lr 0,001.

Per minibatch blijft de oorspronkelijke Any-Precision-volgorde behouden:
E4M3 met harde CE, E2M3 met losgekoppelde E4M3-softtargets, E2M1 met
losgekoppelde E2M3-softtargets; daarna één Adam-update op de gesommeerde
gradients. Temperatuur 1, zonder extra harde studentloss.

Een voltooide pilot heeft 2.865 updates en 8.595 trainingspasses. De uitvoer
bevat de gemeten duur en een projectie voor een nieuwe honderd-epochtraining.
De uiteindelijke accuracy van honderd epochs wordt hiermee niet voorspeld.

## Resultaten en notebook

De training maakt een nieuwe map onder:

`mx_runs\mx_triple_truncation_aware_pilot5\<tijdstip>_joint_run\`

Open daar de lege, uitvoerbare notebook
`MX_triple_truncation_aware_pilot5_resultaten.ipynb` met de bestaande kernel
en voer de cellen uit. De resultaatmap is automatisch ingevuld.

De notebook toont de drie training-/testcurves, de verschillende losses,
de gewichtsdiagnostiek, tijden en accuracyfiguren voor:

- **epoch 5**, het vaste eindpunt van de pilot;
- **E4M3-best**, het eerder in diezelfde pilot geselecteerde checkpoint.

Alle modi in een checkpointgroep horen bij hetzelfde checkpoint. Epoch 100 is
in deze pilot nog niet uitgevoerd en krijgt dus geen resultaat. De latere
volledige proef zal epoch 100 en E4M3-best tonen volgens dezelfde werkwijze.

Extra: per epoch wordt voor iedere modus geteld hoe vaak elk cijfer wordt
voorspeld. Een figuur toont hoeveel klassen worden voorspeld en welk aandeel
naar de dominante klasse gaat. Daarmee blijft een eventuele vaste voorspelling
zoals in de post-hocproef zichtbaar. Deze tellingen staan ook in
`test_prediction_counts.csv` en `history.json`.

Bewaar de volledige runmap en deel na afloop haar `feedback_bundle.zip`.
De bundel bevat rapporten, meettabellen, trainingsgeschiedenis en de lege
notebook met helper, maar geen trainingscheckpoints of datasetbestanden.

## Lage accuracy en technische status

Lage accuracy, één vaste voorspelde klasse of een eindige nulgradient is een
onderzoeksuitkomst. Er is geen vereiste accuracy of gegarandeerde verbetering.
Ook nul gewijzigde MX-masterwaarden wordt als diagnose gemeld. De controles
op correcte optimizerstappen, broncycli, BN, datadekking en eindige toestand
blijven verplicht; structurele fouten en NaN/Inf leiden wel tot FAIL.

De BN-banken leren nu mee met de getrunceerde gewichten. Daardoor kan deze
proef anders uitvallen dan de eerdere post-hocbewerking met vastgehouden BN.
De pilot alleen bepaalt nog niet wat de volledige honderd-epochtraining doet.

## Stoppen en hervatten

Eén keer `Ctrl+C` vraagt om de huidige epoch volledig af te werken en op te
slaan. Een tweede keer stopt onmiddellijk; alleen reeds voltooide checkpoints
zijn hervatbaar. Een bestand `STOP_AFTER_EPOCH` in de actieve resultaatmap
vraagt eveneens om een stop op de volgende epochgrens.

Hervatten gebeurt alleen binnen ditzelfde pilotprotocol, tot maximaal epoch 5:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_truncation_aware_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Gebruik bijvoorbeeld `last_checkpoint.pt` uit een gecontroleerd gestopte
pilotmap, of een volledig `checkpoints\epoch_00002.pt` met zijn checksum.
Alleen de volledige trainingscheckpoint bevat optimizer-, scheduler- en
RNG-staat. De compacte MX-exports zijn bestemd voor inferentie, niet resume.
Een checkpoint van een eerdere directe of tweemodusproef wordt geweigerd.

## Export en documentatie

Best en last krijgen elk één opgeslagen E4M3-bronexport met alle benodigde
BN- en FP32-uitzonderingsstaat. De E2M3- en E2M1-bestanden worden uitsluitend
uit die opgeslagen en herladen bron afgeleid, zonder de zes FP32-MX-masters.
De volledige FP32-checkpoints blijven daarnaast behouden voor resume.

`PROTOCOL.md` legt de exacte methode vast. `CODEWIJZIGINGEN.md` beschrijft
hergebruik en wijzigingen. `local_validation/VALIDATION.md` beschrijft de
synthetische controles. Het pakket bevat één notebooktemplate en geen
uitgevoerde kopie. Er is bij de voorbereiding geen echte SVHN-training gedaan.
