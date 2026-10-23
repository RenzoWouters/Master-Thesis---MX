# Honderd epochs — E4M3-bron met exponent- en mantissatruncatie

Dit pakket voert de volledige vervolgproef uit na de geslaagde vijf-epochpilot.
De methode blijft gelijk: E4M3 is de gekwantiseerde bron; E2M3 ontstaat door de
hoogste twee exponentbits te behouden; E2M1 ontstaat daarna door de twee laagste
mantissabits weg te laten. Beide lagere modi behouden exact de E4M3-bronschaal.

Activaties blijven A32. De gedeelde mastergewichten, gradients en Adam-momenten
blijven FP32. De drie BN-banken leren tijdens training normaal mee. Er is geen
expliciete schaalcorrectie, biascompensatie of aparte leerbare schaalparameter.

## Aanleiding en vergelijking

De pilot is technisch geslaagd. Na vijf epochs zijn de testaccuracies E4M3
82,0759%, E2M3 81,4997% en E2M1 81,8608%. Alle drie modi voorspellen alle tien
cijfers. De directe pilot haalde op hetzelfde eindpunt respectievelijk
93,1085%, 93,0662% en 92,7973%: de truncatietraining blijft dus ongeveer elf
procentpunt achter. Deze honderd-epochproef meet of dat verschil verder afneemt.

`pilot_analysis/RESULTATEN.md` bevat de volledige duiding, de vergelijking en
verwijzingen naar de bewaarde bronmetingen. Er wordt geen eindaccuracy voorspeld.

## Uitpakken

Pak de pakketmap uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_truncation_aware_reference100_v1\run_train.py`

Gebruik de bestaande `.venv-train`. Het pakket installeert of upgradet niets.
Alle uitvoer komt in nieuwe resultaatmappen; eerdere proeven blijven behouden.

## 1. Controle

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_truncation_aware_reference100_v1\run_train.py" check
```

De controle test codecs, bitbewerking, schaalbehoud, gedeelde bron, gradients,
recursieve training en exports op synthetische input. Er wordt geen SVHN-training
uitgevoerd. Bij FAIL: deel de feedbackbundel uit de nieuwe checkmap.

## 2. Na CHECK PASS: volledige training

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_truncation_aware_reference100_v1\run_train.py" run
```

Deze opdracht begint een nieuwe training vanaf de oorspronkelijke seed 20261009
met hetzelfde model en dezelfde initialisatieprocedure. Het pilotcheckpoint wordt
niet geladen. Zoals bij de eerdere volledige referenties loopt deze nieuwe proef
van epoch 1 tot en met 100.

Dataset: SVHN train-only, 73.257 trainbeelden en 26.032 officiële testbeelden.
De extra-split wordt niet gebruikt. De bestaande data staat standaard onder
`data\svhn`; alleen indien nodig gebruik je `--data-root "PAD"` voor een ander
bestaand pad. Download is uitsluitend met de optionele `--download` ingeschakeld.

Per minibatch: E4M3 met harde CE, E2M3 met detached E4M3-softtargets, E2M1 met
detached E2M3-softtargets. Eén Adam-update op de som van de drie gradients.
Batch 128; lr 0,001; LR-dalingen na epochs 50/75/90 met factor 0,1. Een volledige
run heeft 57.300 optimizerupdates en 171.900 trainingspasses.

De vijf gemeten pilotepochs duurden samen 4 min 33 s, inclusief test en checkpoints.
De projectie voor een nieuwe volledige run is ongeveer 1 uur 30 min. De projecties
uit het waargenomen tempo liggen tussen circa 86 en 94,5 minuten; dit is geen
betrouwbaarheidsinterval. Temperatuur en achtergrondbelasting kunnen de duur
veranderen. Data-inladen en eindrapportage komen daar nog bij.

## Resultaten en notebook

De nieuwe training schrijft onder:

`mx_runs\mx_triple_truncation_aware_reference100\<tijdstip>_joint_run\`

De runner kopieert één lege uitvoerbare notebook naar de resultaatmap:
`MX_triple_truncation_aware_reference100_resultaten.ipynb`.
Het pad naar de run is automatisch ingevuld. Voer de notebook zelf uit met de
bestaande kernel.

De accuracyfiguren tonen alle drie precisies op:

- epoch 100, het vaste eindpunt;
- het checkpoint met het eerste strikte maximum van E4M3-testaccuracy.

Voor de tweede groep worden alle precisies op hetzelfde door E4M3 gekozen
checkpoint getoond. Er worden geen afzonderlijke beste studentepochs gemengd.
Bij vroeg stoppen toont de notebook de laatste beschikbare epoch en vermeldt hij
dat epoch 100 nog niet is gemeten.

De notebook bevat tevens de curves, losses, gewichtsdiagnostiek, tijden,
geheugengebruik en de voorspelde klassendiversiteit met dominant klasseaandeel.
De ruwe tellingen staan in `test_prediction_counts.csv` en `history.json`.
Er wordt geen uitgevoerde notebookkopie meegeleverd.

Bewaar de volledige runmap en deel na afloop haar `feedback_bundle.zip`.
De feedbackbundel bevat rapporten en meettabellen, maar geen datasets of grote
trainingscheckpoints. Het gekozen beste checkpoint en epoch 100 blijven lokaal
bewaard voor latere proeven.

## Stoppen en hervatten

Eén keer `Ctrl+C` vraagt om de huidige epoch af te werken en op te slaan. Een
tweede keer stopt onmiddellijk; alleen volledig opgeslagen checkpoints zijn dan
hervatbaar. Een bestand `STOP_AFTER_EPOCH` in de actieve resultaatmap vraagt ook
om een stop op de volgende epochgrens.

Hervatten van een checkpoint uit ditzelfde honderd-epochpakket:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_truncation_aware_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_last_checkpoint.pt"
```

Gebruik het volledige FP32-trainingscheckpoint met zijn checksum, niet een compacte
MX-export. Een eerdere directe, tweemodus- of vijf-epochpilot heeft een ander
experimentcontract en wordt geweigerd. Hervatten na voltooiing van epoch 100 wordt
ook geweigerd. Resume maakt een nieuwe resultaatmap en bewaart de eerdere map.

## Technische status en onderzoeksuitkomst

Een lage accuracy is geen technische FAIL. Dat geldt ook voor een constante
voorspelling of eindige nulupdates: die worden zichtbaar gerapporteerd. Verkeerde
bitcodes, schalen, stappen, ontbrekende gradients, niet-eindige toestand en
onvolledige gegevens blijven fouten.

`PROTOCOL.md` beschrijft de methode; `CODEWIJZIGINGEN.md` de beperkte wijzigingen
tegenover de pilot. `local_validation/VALIDATION.md` beschrijft de softwaretests.
Er is bij de voorbereiding geen echte SVHN-training uitgevoerd.
