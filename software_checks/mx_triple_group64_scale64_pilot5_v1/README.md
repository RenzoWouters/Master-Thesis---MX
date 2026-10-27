# Groepsgrootte 64 met vaste ×64 — gezamenlijke MX-pilot van vijf epochs

Deze proef onderzoekt groepsgrootte **B64** met dezelfde vaste schaalfactor **×64** als de bestaande B16- en B32-runs. De numerieke experimentele wijziging is de groepsgrootte. E4M3 is de gedeelde bron; E2M3 ontstaat door de hoogste exponentbits te behouden en de bronschaal met 64 te vermenigvuldigen; E2M1 ontstaat door mantissatruncatie met behoud van de E2M3-schaal. Activaties blijven A32 en de mastergewichten, gradients en Adam-momenten blijven FP32.

De B16-training van 100 epochs is geslaagd. Op epoch 100 lag B16 bij alle drie modi onder B32; op de door E4M3 gekozen checkpoints waren de lagere modi van B16 beter. Die checkpoints liggen op verschillende epochs (53 en 42). De analyse staat in `previous_results/RESULTATEN_B16_100.md`. B64 vormt de volgende proef in dezelfde vergelijking; een hogere accuracy wordt niet verondersteld.

## Plaatsen en uitvoeren

Pak de ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. De runner staat dan hier:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_group64_scale64_pilot5_v1\run_train.py
```

Gebruik de bestaande trainingsomgeving. Voer eerst de technische controle uit:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group64_scale64_pilot5_v1\run_train.py" check
```

Start na `EINDSTATUS: PASS` de vijf epochs:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group64_scale64_pilot5_v1\run_train.py" run
```

De runner stopt uiterlijk na **vijf epochs**. Hij begint vanaf de oorspronkelijke initialisatie met seed 20261009. Het oorspronkelijke LR-schema voor 100 epochs blijft ingesteld, met dalingen na 50/75/90; het wordt niet ingekort. De LR blijft tijdens deze pilot dus 0,001.

De bestaande bestanden `train_32x32.mat` en `test_32x32.mat` worden hergebruikt. Hun namen hangen niet af van groepsgrootte of schaalfactor. Het pakket installeert of vernieuwt geen dependencies. Gebruik indien nodig `--data-root "PAD_NAAR_SVHN"`. Alleen met expliciet `--download` kunnen ontbrekende gegevens worden gedownload. SVHN-extra wordt niet gebruikt.

## Resultaten en notebook

Iedere opdracht schrijft een nieuwe map onder `mx_runs\mx_triple_group64_scale64_pilot5`. De training bewaart epochcheckpoints, het gezamenlijke E4M3-beste checkpoint, het laatste checkpoint, compacte MX-exports en een `feedback_bundle.zip`.

Open de **onuitgevoerde** `MX_triple_group64_scale64_pilot5_resultaten.ipynb` in die resultatenmap. De runner vult het juiste resultatenpad in. De notebook leest opgeslagen rapporten; `SAVE_FIGURES=False` is de standaard.

De notebook bevat:

- Accuracycurves van de drie modi en de eerste vijf epochs van de bestaande B16 ×64- en B32 ×64-trainingen.
- Accuracygrafieken voor epoch 5 en het checkpoint dat binnen de pilot het beste op E4M3 scoort. Alle drie modi komen steeds uit hetzelfde checkpoint per run.
- Vergelijkingen met dezelfde checkpointregels voor alle groepsgroottes. De globale referentiemaxima op epoch 53 en 42 worden niet als vijf-epochresultaat gebruikt.
- Trainingstijd, geregistreerd CUDA-geheugen, quantisatiediagnostiek en controle van de zes geselecteerde compacte exports.
- Kosten van schalen, padding en totale bitgepakte representaties, per laag en voor het geheel.

De B16-referentie komt uit de **nieuwe, vanaf initialisatie gestarte 100-epochrun**. De eerste vijf epochs daarvan kunnen afwijken van de afzonderlijke eerdere B16-pilot: de oorspronkelijke CUDA-backend biedt geen bitexacte herhaalgarantie. De referentiebron is vastgelegd.

Bij een latere volledige B64-proef komen opnieuw grafieken voor epoch 100 en het globale E4M3-beste checkpoint. Deze pilot levert geen nieuwe B64-accuracy op epoch 100. De bestaande B16-notebook blijft bruikbaar; er is geen opnieuw uitgevoerde versie toegevoegd.

Deel na afloop de `feedback_bundle.zip` uit de map van **de training**. De grote checkpoints blijven lokaal en zitten niet in de feedbackbundel.

## Kosten: schalen, padding en totale opslag

Zonder padding bedraagt de schaaloverhead `8/B` bits per gewicht: 0,5 bij B16, 0,25 bij B32 en 0,125 bij B64. Hier wordt iedere afgevlakte `Cin*Kh*Kw`-rij per outputkanaal afzonderlijk aangevuld.

| Groepsgrootte | Groepen / schaalbytes | Paddingposities | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---:|---:|---:|---:|---:|---:|
|16|8.096|64|137.632 bytes|105.248 bytes|72.864 bytes|
|32|4.080|1.088|134.640 bytes|102.000 bytes|69.360 bytes|
|64|2.160|8.768|140.400 bytes|105.840 bytes|71.280 bytes|

Er zijn **129.472 echte gewichten in zes MX-lagen**. B64 gebruikt 1.920 minder schaalbytes dan B32, maar 7.680 meer paddingposities. Het nettoverschil per precisie is:

- E4M3-bron: **+5.760 bytes, +4,28%** ten opzichte van B32.
- Afzonderlijke E2M3-export: **+3.840 bytes, +3,76%**.
- Afzonderlijke E2M1-export: **+1.920 bytes, +2,77%**.

Vooral de grootste laag telt mee: 800 gewichten per outputkanaal passen exact in groepen van 16 en 32, maar worden bij B64 aangevuld tot 832. Een andere groeperingsas of het samenvoegen van outputkanalen zou een andere proef zijn.

De drie precisiekolommen zijn **alternatieve exportrepresentaties**. Tel ze niet bij elkaar op alsof drie volledige modellen tegelijk nodig zijn. De E4M3-bron volstaat om de lagere codes af te leiden; de vaste +6-offset is globale configuratie en vereist geen geleerde keuze per groep.

Dit zijn berekende code- en schaalbytes inclusief padding. FP32-uitzonderingen, BatchNorm, optimizer, bestandheaders en tijdelijke emulatietensors vallen erbuiten. De getallen zijn geen totale checkpointgrootte of gemeten GPU-geheugen. Looptijd betreft PyTorch-emulatie en geeft geen gemeten native-MX-hardware- of energiewinst. De runner schrijft drie `group_size_costs`-rapporten en controleert de werkelijke exports tegen de berekende geometrie.

## Stoppen en hervatten

Eén keer `Ctrl+C` vraagt een stop na de lopende volledige epoch. Een tweede keer stopt direct. Een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap vraagt ook een stop na de lopende epoch.

Hervat alleen met een eigen checkpoint uit dit B64-pilotpakket:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group64_scale64_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Optimizer, scheduler en RNG-toestand worden hersteld. De oorspronkelijke CUDA-backend garandeert geen bitexacte hervatting. Het contract weigert B16-/B32-checkpoints, andere schaalvarianten en checkpoints die de vijf epochs al voltooid hebben.

## Interpretatie en controles

Lage accuracy is een meetresultaat en veroorzaakt op zichzelf geen technische FAIL. De pilot controleert werking, kosten, looptijd en vroege trends. De B16-resultaten laten zien waarom een pilot geen bewijs voor uiteindelijke accuracy is. De vergelijking gebruikt één seed en de oorspronkelijke CUDA-backend; checkpointselectie gebruikt de officiële testset.

De schaalfactor blijft voor alle groepsgroottes ×64 om de invloed van groepsgrootte bij die vaste route te onderzoeken. Factoren tussen ×32 en ×64 of geleerde schalen blijven een afzonderlijke vervolgvraag.

Zie `local_validation/VALIDATION.md` voor de lokale controles. Ook de echte opstartfunctie met de datasetnaamcontrole is apart afgedekt. De voorbereiding gebruikt kleine synthetische CPU-fixtures; er is geen SVHN-training gestart. `check` controleert de code daarna op jouw Windows/CUDA-omgeving.
