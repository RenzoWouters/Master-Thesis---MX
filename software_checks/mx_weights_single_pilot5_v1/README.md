# Eerste eenmodus-training: MXFP6 E2M3 / A32 / FP32-master

Dit pakket start een NIEUWE proef van vijf volledige SVHN-train-only-epochs.
Er wordt geen oud checkpoint ingeladen. Geen truncatie, activatiekwantisatie of
multi-precisiedistillatie. De oorspronkelijke auteurs-forward wordt uitgevoerd
met selector [32]; de aparte MX-gewichtsquantisator bepaalt E2M3.

## Plaatsen en starten (PowerShell)

Pak de inhoud uit naar:
`C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_single_pilot5_v1`
`run_train.py` moet rechtstreeks in die map staan.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_single_pilot5_v1\run_train.py" check

# Pas uitvoeren na EINDSTATUS: PASS van de controle.
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_single_pilot5_v1\run_train.py" run --mode mx
```

De controle gebruikt synthetische waarden en twee mini-updates per modus; geen
SVHN. De echte run gebruikt 73257 trainingsbeelden, 26032 testbeelden, batch 128:
573 trainingsminibatches per epoch en 2865 optimizerstappen na vijf epochs.
Bestaande train/test .mat-bestanden in `data\svhn` worden hergebruikt. `extra`
wordt niet geopend, gehasht of gewijzigd. Alleen wanneer train/test ontbreken
kan expliciet `--download` worden toegevoegd. Geen installaties of upgrades nodig
in de bestaande werkende .venv-train.

Resultaten komen in een NIEUWE sessiemap onder:
`mx_runs\mx_weights_single_pilot5\<tijdstip>_mx_run`.
Open daar `MX_single_training_resultaten.ipynb`, selecteer .venv-train en Run All.
Upload daarna `feedback_bundle.zip` uit diezelfde map, niet de `_check`-map.

## Wat verandert in de training?

- Nog steeds CNN-8 expand=2, Resize(40), dezelfde zes kwantiseerbare lagen,
  dropout, Adam lr=0.001 zonder weight decay en batch 128.
- Een modus in plaats van vijf: een forward/backward, hard-label cross-entropy,
  een Adam-update per minibatch. Geen teacher of distillatie.
- FP32-master W -> oorspronkelijke G(W) -> MX E2M3 -> FP32-convolutie-emulatie.
- De identiteit-STE geeft gradients door MX. Adam wijzigt de FP32-masterWAARDEN;
  alleen hun FORMAAT blijft FP32. Ook de Adam-momenten en gradients blijven FP32.
- A32/ReLU blijft vast. BN32 leert tijdens training en staat stil tijdens test.
  De andere BN-banken blijven ongebruikt, maar worden niet uit het model verwijderd.
- Het 100-epoch-LR-schema 50/75/90 blijft intact; de proef stopt na vijf epochs.
  De LR blijft hier dus 0.001. Dit pakket start nooit automatisch honderd epochs.

Zie `CODEWIJZIGINGEN.md` voor de werkelijke functies en `research_changes.patch`
voor de diff met de referentierunner. De MX-codec en STE zijn byte-identiek aan
het reeds geteste MX-evaluatiepakket. De bestaande raw onderzoeksdata blijven intact.

## Aanvullende gematchte controle (afzonderlijke run)

De oude vijf-precisietraining is geen gematchte controlegroep voor deze eenmodusrun.
Voor een vergelijking met dezelfde loss, initialisatie en batchvolgorde is er:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_single_pilot5_v1\run_train.py" run --mode control
```

Deze opdracht traint een AFZONDERLIJK W32/A32-model met dezelfde oorspronkelijke
G(W)-voorbewerking, maar zonder MX. Het is geen tweede precisiemodus binnen de
MX-training. De control-run start opnieuw met dezelfde seed; beide runs bewaren
initialisatie- en samplevolgordehashes. Geef zijn runpad als CONTROL_DIR in de
MX-notebook om de overeenkomst te controleren en dezelfde epochs te vergelijken.
Geen claim van bitexact deterministische CUDA-training of statistische zekerheid.

## Behouden bestanden

Alle vijf FP32-trainingscheckpoints, optimizer-/scheduler-/RNG-staat en checksums.
`best_checkpoint.pt`: eerste strikte maximum van de actieve modus op de testset.
`last_checkpoint.pt`: laatste complete epoch; normaal epoch 5.
`mx_source_best.pt` en `mx_source_last.pt`: alleen voor de MX-run; gepakte codes
plus schalen voor de zes lagen en de overige FP32-laag/BN-staat, zonder de zes
FP32-updatemasters. Dit zijn GEEN hervatbestanden; gebruik daarvoor de volledige
trainingscheckpoints. Exportcontrole vergelijkt gedecodeerde tensors, niet opnieuw
alle testvoorspellingen. Beide soorten bestanden hebben verschillende doelen.

## Onderbreken en hervatten

Een Ctrl+C vraagt een stop NA de lopende epoch en bewaart die. Een tweede Ctrl+C
onderbreekt direct. Alleen volledig afgeronde epochs zijn hervatbaar. Een bestand
`STOP_AFTER_EPOCH` in de huidige sessiemap heeft hetzelfde zachte-stop-effect.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_single_pilot5_v1\run_train.py" resume --mode mx --checkpoint "C:\...\checkpoints\epoch_00002.pt"
```

Gebruik het werkelijk bestaande checkpoint van DIT pakket, dezelfde modus,
broncode en omgeving. De nieuwe hervatsessie krijgt een aparte map. Oudere
Any-Precision-checkpoints of een checkpoint uit de andere modus worden geweigerd.
`cudnn.benchmark=True` blijft behouden; geen bitexacte CUDA-hervatgarantie.
Bij FAIL: geen pakketten automatisch upgraden; deel het rapport met foutmelding.
