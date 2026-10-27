# B64 met vaste ×64 — gezamenlijke MX-training van 100 epochs

Dit pakket voert de volledige B64-proef uit na de geslaagde vijf-epochpilot. De gewichtsroute blijft **E4M3 → E2M3 → E2M1**, met behoud van de hoogste exponentbits, een vaste ×64-bronschaal voor de lagere modi en mantissatruncatie met dezelfde E2M3-schaal. Activaties blijven A32; mastergewichten, gradients en Adam-momenten blijven FP32.

De numerieke modules en productiezelftests zijn ongewijzigd tegenover de succesvolle B64-pilot. Deze proef begint opnieuw vanaf de oorspronkelijke initialisatie met seed 20261009 en traint tot **epoch 100**. De LR daalt na epochs 50, 75 en 90. De pilot wordt als historische referentie bewaard.

## Uitvoeren

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. De runner staat dan in `software_checks\mx_triple_group64_scale64_reference100_v1`.

Controleer eerst de code op de bestaande Windows/CUDA-trainingsomgeving:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group64_scale64_reference100_v1\run_train.py" check
```

Start bij `EINDSTATUS: PASS` de volledige training:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group64_scale64_reference100_v1\run_train.py" run
```

Op basis van de echte B64-pilot is de looptijdschatting **1 uur en 46 minuten**. De pilot duurde inclusief voorbereiding en export ongeveer 5 minuten en 24 seconden. De projectie veronderstelt vergelijkbare laptopomstandigheden; reserveer praktisch ongeveer 1 uur en 45 minuten tot 2 uur.

De bestaande `train_32x32.mat` en `test_32x32.mat` worden hergebruikt. De groepsgrootte verandert deze bestandsnamen niet. Er worden geen dependencies geïnstalleerd of vernieuwd. Gebruik indien nodig `--data-root "PAD_NAAR_SVHN"`; alleen expliciet `--download` staat downloaden van ontbrekende data toe. De extra-split wordt niet gebruikt.

## Resultaten en notebook

Iedere opdracht schrijft een nieuwe map onder `mx_runs\mx_triple_group64_scale64_reference100`. De training bewaart alle epochcheckpoints, een kopie van het E4M3-beste checkpoint, het laatste checkpoint, compacte exports en een feedbackbundel.

Open de **onuitgevoerde** `MX_triple_group64_scale64_reference100_resultaten.ipynb` in de resultatenmap. Het resultatenpad wordt door de runner ingevuld. De notebook bevat:

- Accuracycurves van de drie modi, met de bestaande volledige B16 ×64- en B32 ×64-runs als referenties.
- Accuracygrafieken voor **epoch 100** en het **globaal beste E4M3-checkpoint**. Alle drie precisies komen per run uit hetzelfde gekozen checkpoint.
- Vergelijkingen met de eigen E4M3-beste epochs van B16 en B32: respectievelijk 53 en 42. Er wordt geen afzonderlijk maximum voor iedere lagere precisie geselecteerd.
- Schaal-, padding- en opslagkosten, quantisatiediagnostiek, exports, gemeten looptijd en geregistreerd CUDA-geheugen.

Bij een onderbroken run vermeldt de notebook de werkelijk voltooide epochs en vergelijkt ze alleen binnen het beschikbare gemeenschappelijke bereik. Een gedeeltelijke run krijgt geen epoch-100-eindscore. `SAVE_FIGURES=False` is de standaard; zet dit op `True` om figuren in een nieuwe analysemap te bewaren.

De volledige B16/B32-bronrapporten zitten in `group16_reference` en `scale64_reference`. `pilot_reference` bevat de geslaagde echte B64-pilot. De analyse daarvan staat in `pilot_analysis/RESULTATEN.md`. De eerdere B16-analyse onder `previous_results` is historische context van vóór de B64-pilot.

Deel na afloop de `feedback_bundle.zip` uit de map van **de training**. De grote checkpoints blijven lokaal en zitten niet in de feedbackbundel.

## Wat de pilot liet zien

Op epoch 5 scoorde B64 92,8895% / 92,1481% / 91,9023% voor E4M3/E2M3/E2M1. Tegenover B32 is dat −0,1191 / +0,8490 / +1,0103 procentpunt. Alle varianten kiezen binnen de eerste vijf epochs ook epoch 5 als E4M3-beste checkpoint.

Het voordeel bij beide lagere modi verschijnt pas op epoch 5; in epochs 1–4 lagen ze onder B32. De pilot bevestigt de werking en geeft een vroege indicatie. De volledige training is nodig om dezelfde eindpunten als B16/B32 te vergelijken. De oorspronkelijke CUDA-backend garandeert geen bitexacte herhaling, zodat de eerste vijf epochs van deze nieuwe run kunnen afwijken van de pilot.

## Kosten van schalen en padding

Voor 129.472 echte gewichten in de zes MX-lagen worden groepen binnen iedere outputrij gevormd. Eén groep heeft één schaalbyte. Zonder padding is de schaaloverhead `8/B` bits per gewicht.

| Groepsgrootte | Groepen / schaalbytes | Paddingposities | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---:|---:|---:|---:|---:|---:|
|16|8.096|64|137.632 bytes|105.248 bytes|72.864 bytes|
|32|4.080|1.088|134.640 bytes|102.000 bytes|69.360 bytes|
|64|2.160|8.768|140.400 bytes|105.840 bytes|71.280 bytes|

B64 heeft minder schalen dan B32, maar meer padding. De grootste laag heeft 800 gewichten per outputkanaal; B64 vult iedere rij tot 832 aan. Netto kost B64 tegenover B32 **+4,28% voor de E4M3-bron**, **+3,76% voor een afzonderlijke E2M3-export** en **+2,77% voor een E2M1-export**.

De drie precisiekolommen zijn alternatieve representaties en worden niet bij elkaar opgeteld. De compacte E4M3-bron volstaat om de lagere codes af te leiden. De vaste +6-offset is globale configuratie en vraagt geen geleerde keuze per groep.

Deze aantallen omvatten codes, schalen en rijpadding. FP32-uitzonderingen, BatchNorm, optimizer, bestandscontainers en tijdelijke werkbuffers vallen erbuiten. Gemeten PyTorch-looptijden zijn geen meting van native-MX-hardware of energiegebruik. De runner controleert de geometrie van alle zes geselecteerde exports tegen de kostenberekening.

## Stoppen en hervatten

Eén keer `Ctrl+C`, of een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap, vraagt een stop na de lopende volledige epoch. Een tweede `Ctrl+C` stopt direct.

Hervat uitsluitend vanuit een volledig geschreven checkpoint van **dit full100-pakket**:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group64_scale64_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

De runner herstelt model, Adam, scheduler en RNG-toestand. Pilotcheckpoints, B16/B32-contracten en reeds voltooide epoch-100-checkpoints worden vóór updates geweigerd. De CUDA-backend biedt geen bitexacte hervatgarantie.

## Validatie

Zie `local_validation/VALIDATION.md` voor de synthetische CPU-controles van de 100-epochlus, LR-overgangen, hervatten, exports, startup en notebook. De voorbereiding start geen echte SVHN-training. De opdracht `check` controleert de ongewijzigde B64-zelftests op de eigen Windows/CUDA-omgeving.
