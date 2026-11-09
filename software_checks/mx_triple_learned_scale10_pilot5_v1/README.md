# Tien leerbare schaalfactoren — SVHN/CNN-8, vijf epochs

Deze proef breidt de eerdere geleerde keuze ×32/×64 uit naar **×1, ×2, ×4, ×8, ×16, ×32, ×64, ×128, ×256 en ×512**, afzonderlijk per groep van 32 gewichten. E4M3 → E2M3 gebruikt dezelfde hoogste exponentbits; E2M3 → E2M1 behoudt dezelfde mantissatruncatie. Beide lagere modi delen de gekozen schaalcorrectie. Activaties blijven A32; masters, gradients en Adam-staat blijven FP32.

De 4.080 FP32-hulpwaarden beginnen opnieuw exact op 6, dus ×64. We behouden de oorspronkelijke modelinitialisatie, schaalgradient en optimizerinstellingen. Alleen de beschikbare offsets en parametergrenzen worden verruimd. Eén minibatch bevat nog steeds drie precisiepasses en één Adam-update.

## 1. Uitpakken en controleren

Pak de ZIP uit onder `C:\Users\renzo\master-thesis-mx\software_checks`. Het script staat daarna in `software_checks\mx_triple_learned_scale10_pilot5_v1`.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_pilot5_v1\run_train.py" check
```

Deze controle gebruikt synthetische tensors en korte modelpasses. Zij controleert alle tien factoren, drempels, gradients, projectie en exports. Er wordt geen SVHN-training gestart en geen dataset gedownload. De bestaande Python-omgeving wordt gebruikt; het pakket installeert of wijzigt geen dependencies.

## 2. De vijf-epochtraining uitvoeren

Start na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_pilot5_v1\run_train.py" run
```

De bestaande `data\svhn`-bestanden worden hergebruikt. Een andere locatie kan met `--data-root "PAD_NAAR_SVHN"`; ontbrekende data worden uitsluitend met `--download` opgehaald. De extra-split wordt niet gebruikt.

Elke opdracht maakt een nieuwe map onder `mx_runs\mx_triple_learned_scale10_pilot5`. De run begint opnieuw vanaf nul en stopt verplicht na **vijf volledige epochs**: 73.257 trainbeelden en 26.032 testbeelden per epoch; batch 128; 573 updates per epoch. Na vijf epochs zijn dat **2.865 updates en 8.595 trainingspasses**.

Adam start op 0,001 voor gewichten en schaalhulpwaarden. Het oorspronkelijke LR-schema voor honderd epochs blijft intact, met dalingen na 50/75/90. Die dalingen worden in deze pilot niet bereikt. De pilot meet de werkelijke looptijd; een projectie naar 100 is geen gemeten volledige training.

## 3. Notebook en feedback

**De notebook staat meteen in het uitgepakte pakket:**

`MX_triple_learned_scale10_pilot5_resultaten.ipynb`

Na een afgeronde training staat de notebook ook in de resultaatmap. In `feedback_bundle.zip` is hij met helper en referenties eveneens opgenomen, ook wanneer een controle of run vroeg mislukt. Houd die bestanden samen bij het uitpakken. De template heeft lege outputs; open hem in Jupyter/VS Code en voer **Run All** uit. Hij leest uitsluitend opgeslagen JSON/CSV, zonder training of modelcheckpoint te laden.

Met `RUN_DIR = None` zoekt de pakketnotebook de nieuwste training van dit experiment; een latere check-opdracht verdringt de training niet. Een nieuwe mislukte training wordt wel zichtbaar en wordt niet stilzwijgend vervangen door een oudere PASS. De runkopie bevat normaal het exacte resultaatpad; een uitgepakte feedbackmap blijft verplaatsbaar.

De notebook bevat:

- Accuracycurves van alle drie precisies, met de eerdere binaire pilot en de eerste vijf epochs van vaste ×32/×64.
- Accuracies op epoch 5 en op het gezamenlijke checkpoint gekozen door de hoogste E4M3-testaccuracy.
- Alle tien gekozen factoren per laag/epoch; θ-verdelingen, omschakelingen, sprongafstanden, gradients en grensprojecties.
- Gemeten tijd en geheugen, een voorlopige tijdprojectie naar 100 en aparte opslaggrafieken voor keuze-informatie en FP32-trainingstensors.

Een onvoltooide run blijft als onvoltooid aangeduid. Referenties worden alleen vergeleken binnen dezelfde reeds bereikte vroege epochs; hun latere beste checkpoints en epoch 100-scores worden niet gebruikt. Lage accuracy of nog geen keuze buiten ×32/×64 is op zichzelf geen technische FAIL.

Deel na afloop **`feedback_bundle.zip` uit de trainingsmap**. Checkpoints en datasetbestanden blijven lokaal.

## Stoppen en hervatten

Eén `Ctrl+C` vraagt een stop na de lopende volledige epoch. Een tweede stopt onmiddellijk; alleen eerder volledig opgeslagen checkpoints zijn hervatbaar. Een leeg `STOP_AFTER_EPOCH`-bestand in de actieve resultaatmap vraagt eveneens een stop op de epochgrens.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Alleen onvoltooide checkpoints van dit tienkeuzepakket worden geaccepteerd. Hervatten bewaart model, θ, Adam, scheduler, RNG en geschiedenis en maakt een nieuwe resultaatmap. Deze executable gaat nooit voorbij epoch 5. De oorspronkelijke CUDA-backend biedt geen bitexacte hervatgarantie.

## Kosten en afbakening

De compacte E4-bron bewaart nu vier keuzebits per groep: **2.040 bytes** in plaats van 510 voor twee keuzes, dus 1.530 bytes extra. De FP32-parameter/gradient/twee Adam-momenten blijven analytisch **65.280 bytes extra tegenover vaste schalen**, net als bij de binaire proef. Dit is geen totale VRAM- of `.pt`-bestandsgrootte.

E4-bronschalen worden nog niet geleerd. Onafhankelijke schalen per exponentbreedte, een volledige100-epochrun en eventueel bevriezen vanaf 50 zijn latere proeven. De technische afspraken staan in `PROTOCOL.md`; de lokale CPU-validatie staat in `local_validation`.
