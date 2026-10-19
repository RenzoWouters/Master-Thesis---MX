# Eerste gezamenlijke MX-training: E2M3 + E2M1, vaste A32

Dit pakket voert vijf volledige epochs uit op de bestaande SVHN-trainset,
zonder extra-data. Het is een nieuwe run vanaf initialisatie, geen hervatting
van een eerdere referentie. Geen truncatie, geen activatiequantisatie.

## Plaatsen en starten (Windows PowerShell)

Pak de inhoud uit zodat run_train.py rechtstreeks staat in:

    C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_dual_pilot5_v1

Voer uit vanuit de bestaande projectmap:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_dual_pilot5_v1\run_train.py" check
```

Ga alleen verder na `EINDSTATUS: PASS`. De check gebruikt synthetische getallen
op CUDA en enkele kleine testupdates; geen echte trainingsdata. Bestaande
pakketten niet upgraden wanneer een controle mislukt: bekijk eerst het rapport.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_dual_pilot5_v1\run_train.py" run
```

Dit traint een gezamenlijk model met twee passes per minibatch. Vijf epochs
lopen achter elkaar en stoppen daarna automatisch. De CLI weigert een andere
eindepoch; er start niet ongemerkt een honderd-epochtraining. Het bestaande
`data\svhn` bevat al train en test. Er wordt standaard niets gedownload.
Alleen bij ontbrekende train/testbestanden kan `--download` worden toegevoegd.
Het extra-bestand wordt nooit gelezen, gehasht of gewijzigd.

## Wat gebeurt er in iedere minibatch?

Eerst E2M3/A32 met hard-label cross-entropy, daarna E2M1/A32 met de originele
soft-target cross-entropy en losgekoppelde teacher uit de E2M3-pass. De twee
gradientbijdragen worden opgeteld, zonder deling door twee. Pas daarna volgt
een Adam-update op de gedeelde FP32-mastergewichten. Geen derde FP32-forward.

E2M1 wordt rechtstreeks uit de effectieve FP32-gewichten gevormd, niet via een
bitshift van E2M3. Beide hebben dezelfde blokindeling en dezelfde schaalregel.
Zie PROTOCOL.md voor de volledige configuratie en CODEWIJZIGINGEN.md voor de
concrete drie selectorwijzigingen in de oorspronkelijke trainingsfunctie.

## Resultaten

Nieuwe sessiemappen staan onder:

    C:\Users\renzo\master-thesis-mx\mx_runs\mx_weights_dual_pilot5

De training eindigt op `_joint_run`, de check op `_joint_check`. Na een geslaagde
volledige proef: status PASS, completed_epoch 5 en optimizer_steps 2865.
Er zijn 573 trainingsminibatches per epoch, met twee forward/backwardpasses
maar een optimizerupdate per minibatch (5730 trainingsforwards over vijf epochs).

Open `MX_dual_pilot5_resultaten.ipynb` UIT DE RUNMAP en kies `.venv-train`.
Het pad is al ingevuld. Het notebook leest alleen JSON/CSV. Het bevat geen
training, modelcalls of downloads. Pandas, matplotlib en ipykernel zijn alleen
nodig om het notebook uit te voeren; gebruik de eerder werkende installatie.

We bewaren alle vijf epochcheckpoints, best_checkpoint.pt en last_checkpoint.pt
met checksums. Best is het eerste strikte maximum van E2M3/A32-testaccuracy.
Beide modi worden van datzelfde checkpoint gerapporteerd, ook de laatste epoch.
De testscore is niet onafhankelijk van deze selectie. De resultaten na vijf
epochs zijn geen eindconvergentie-uitspraak of gelijkbudget-paperreproductie.

Exports: per best/last worden aparte direct afgeronde E2M3- en E2M1-bestanden
gemaakt. Zij bevatten niet de zes FP32-mastertensors, maar wel de overige
modelstaat. Dit is nog NIET een enkel opgeslagen MX-model waaruit beide modi
via truncatie afleidbaar zijn. Volledige FP32-checkpoints zijn voor hervatten.

Deel `feedback_bundle.zip` uit de `_joint_run`-map. Geen gewichten of ruwe beelden
nodig voor de eerste rapportanalyse. De bundel bevat geen individuele logits;
rapportcontrole is dus geen onafhankelijke checkpoint-herevaluatie.

## Onderbreking alleen als noodvoorziening

Een Ctrl+C vraagt een stop na de huidige epoch; nog een Ctrl+C stopt direct.
Ook een leeg bestand STOP_AFTER_EPOCH in de actieve runmap stopt na de epoch.
Alleen volledig opgeslagen epochcheckpoints zijn hervatbaar. Voor een
onderbroken proef met hetzelfde pakket/omgeving:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_dual_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR\checkpoints\epoch_00002.pt"
```

Vervang het voorbeeldpad door de bestaande volledige epoch uit DEZE proef.
Een nieuw resultaatpad wordt aangemaakt; oude bestanden blijven behouden.
Een checkpoint van een andere proef, modus of gewijzigd contract wordt geweigerd.
Na een directe onderbreking moet de meest recente volledig gecontroleerde
checkpoint/checksumcombinatie worden gebruikt. Geen mid-epoch- of bitexacte
CUDA-herstelgarantie. Laat de laptop aangesloten en voorkom slaapstand.

## Afbakening en tests

Alleen gewichtswaarden zijn MX; master, gradients en Adam-momenten zijn FP32.
De convoluties gebruiken gedecodeerde FP32-tensors, geen native MX-kernel.
De leesbare codec gebruikt FP64-intermediairen. Gemeten looptijd omvat ook de
registratie; geen voorspelde hardware- of energiewinst.

`local_validation/VALIDATION.md` beschrijft de werkelijk uitgevoerde synthetische
CPU-tests en de afbakening van de containerpreflight. Werkelijke SVHN-training
op de laptop is nog niet gedaan bij het aanmaken van dit pakket.
