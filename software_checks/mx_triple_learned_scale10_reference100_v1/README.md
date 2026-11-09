# Tien leerbare schaalfactoren — SVHN/CNN-8, honderd epochs

De vijf-epochpilot is geslaagd. Dit pakket voert dezelfde proef uit over **100 epochs vanaf een nieuwe initialisatie**. Gewichten én schaalhulpwaarden blijven tot epoch 100 leren. Er wordt in deze run niets bevroren op 50.

Per groep van 32 gewichten blijft de keuze **×1, ×2, ×4, ×8, ×16, ×32, ×64, ×128, ×256 of ×512**. De 4.080 FP32-hulpwaarden beginnen exact op 6, dus ×64. E4M3 → E2M3 behoudt de hoogste exponentbits; E2M3 → E2M1 gebruikt dezelfde mantissatruncatie. Beide lagere modi delen dezelfde gekozen schaal. Activaties blijven **A32**; masters, gradients en Adam-staat blijven **FP32**.

## 1. Uitpakken en controleren

Pak de ZIP uit onder `C:\Users\renzo\master-thesis-mx\software_checks`. Het script staat daarna in `software_checks\mx_triple_learned_scale10_reference100_v1`.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_reference100_v1\run_train.py" check
```

Deze controle gebruikt synthetische tensors en korte modelpasses. Zij controleert de tien factoren, drempels, gradients, projectie en exports. Er wordt geen SVHN-training gestart en geen dataset gedownload. Het pakket gebruikt de bestaande Python-omgeving en installeert of wijzigt geen dependencies.

## 2. Honderd epochs uitvoeren

Start na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_reference100_v1\run_train.py" run
```

De run begint opnieuw bij epoch 0 en gebruikt geen pilotcheckpoint. Elke opdracht maakt een nieuwe map onder `mx_runs\mx_triple_learned_scale10_reference100`. Bestaande resultaten en datasets worden niet overschreven.

De bestaande `data\svhn`-bestanden worden hergebruikt. Een andere locatie kan met `--data-root "PAD_NAAR_SVHN"`; ontbrekende data worden uitsluitend met `--download` opgehaald. De extra-split wordt niet gebruikt.

Per epoch worden 73.257 trainbeelden en 26.032 testbeelden verwerkt. Batch 128 geeft 573 updates per epoch: **57.300 updates en 171.900 trainingspasses** voor 100 epochs. Eén minibatch bevat drie precisiepasses en één Adam-update. Het ongewijzigde LR-schema voor gewichten en schaalhulpwaarden is:

| Epochs | Adam-leersnelheid gewichten en θ |
|---|---:|
| 1–50 | 0,001 |
| 51–75 | 0,0001 |
| 76–90 | 0,00001 |
| 91–100 | 0,000001 |

De echte tienkeuzepilot mat 338,816 seconden over vijf epochs. Zijn projectie voor een nieuwe volledige run is ongeveer **1 uur 54 minuten**, inclusief training, evaluatie, epochcontroles en checkpointschrijven. Dit is een schatting onder vergelijkbare omstandigheden, geen gemeten eindduur of garantie. Bron: `pilot_reference/epoch_times.csv` en `timing_estimate.json`; bespreking in `PILOT_BEOORDELING.md`.

## 3. Notebook en feedback

De niet-uitgevoerde notebook staat meteen in het pakket:

**`MX_triple_learned_scale10_reference100_resultaten.ipynb`**

Na de run staat hij ook in de resultaatmap. `feedback_bundle.zip` bevat de notebook, helper en referenties eveneens bij een vroege FAIL. Houd die bestanden samen bij het uitpakken. Open de notebook in Jupyter/VS Code en voer **Run All** uit. Hij leest uitsluitend JSON/CSV; hij start geen training en laadt geen modelcheckpoint.

De notebook bevat:

- Het volledige train- en testaccuracyverloop van alle drie precisies.
- Accuracies op **epoch 100** en op **het gezamenlijke checkpoint gekozen door de hoogste E4M3-testaccuracy**.
- Volledige vergelijking met vaste ×32, vaste×64 en de eerdere binaire schaalkeuze met leren tot 100. Iedere training gebruikt één gemeenschappelijk checkpoint voor haar drie precisies.
- Een afzonderlijke epoch 5-vergelijking met de uitgevoerde tienkeuzepilot; die pilot wordt niet als volledige training getoond.
- Alle tien gekozen factoren per laag en epoch, θ-verdelingen, omschakelingen, sprongafstanden, gradients en grensprojecties.
- Gemeten tijd en geheugen, plus aparte opslaggrafieken voor keuze-informatie en FP32-trainingstensors.

Een onvoltooide run blijft als onvoltooid aangeduid. Referenties worden dan alleen vergeleken binnen dezelfde afgeronde epochrange. Het ontbrekende epoch 100-eindpunt wordt niet aangevuld. Referentiebestanden worden met gepinde hashes gecontroleerd.

Met `RUN_DIR = None` zoekt de pakketnotebook de nieuwste training van dit experiment. Een latere check-opdracht verdringt de training niet; een nieuwe mislukte training blijft wel zichtbaar. Een uitgepakte feedbackmap is verplaatsbaar.

Deel na afloop **`feedback_bundle.zip` uit de trainingsmap**. Checkpoints en datasetbestanden blijven lokaal. Je hoeft geen uitgevoerde notebook mee te sturen.

## Stoppen, in delen uitvoeren en hervatten

Eén `Ctrl+C` vraagt een stop na de lopende volledige epoch. Een tweede stopt onmiddellijk; alleen eerder volledig opgeslagen checkpoints zijn hervatbaar. Een leeg `STOP_AFTER_EPOCH`-bestand in de actieve resultaatmap vraagt eveneens een stop op de epochgrens.

Voor een geplande onderbreking, bijvoorbeeld na 50:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_reference100_v1\run_train.py" run --stop-after-epoch 50
```

Vervolgens hervatten met het volledige checkpoint van die run:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_learned_scale10_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

`--stop-after-epoch` is een absoluut epochnummer van 1 t/m 100, standaard 100. Een stop onder 100 is geen volledige PASS-training; het rapport vermeldt de onderbreking. Alleen onvoltooide checkpoints van dit nieuwe honderd-epochcontract worden geaccepteerd. Pilotcheckpoints en checkpoints uit andere proeven zijn geen warmstartoptie. Hervatten bewaart model, θ, beide Adam-groepen, scheduler, RNG en geschiedenis en maakt een nieuwe resultaatmap. De CUDA-backend biedt geen bitexacte hervatgarantie.

Na iedere volledige epoch blijft het checkpoint bewaard. **Bewaar met name checkpoint 50** met volledige optimizer-, scheduler- en RNG-staat. `epoch50_checkpoint.json` verwijst ernaar zodra het beschikbaar is. Een latere proef met bevroren offsets kan vanuit die staat vertrekken; de huidige tak met verder leren hoeft daarvoor niet opnieuw te worden getraind.

## Kosten en afbakening

De compacte flexibele E4-bron bewaart vier keuzebits per groep: **2.040 bytes** tegenover 510 voor de binaire keuze, dus 1.530 bytes extra. De FP32-parameter, gradient en twee Adam-momenten blijven analytisch **65.280 extra bytes ten opzichte van vaste schalen**, evenveel als bij de binaire proef. Dit zijn geen totale VRAM- of `.pt`-bestandsgroottes.

Er blijven drie passes en één FP32-schaalparameter per groep. Het model evalueert niet alle tien factoren afzonderlijk. E4-bronschalen worden nog niet zelfstandig geleerd. Onafhankelijke schalen per exponentbreedte en bevriezen vanaf 50 zijn afzonderlijke latere proeven.

`PROTOCOL.md` bevat de numerieke afspraken, `CODEWIJZIGINGEN.md` de wijzigingen en `local_validation` de lokale softwarecontroles. CPU-fixtures zijn geen SVHN-onderzoeksresultaten.
