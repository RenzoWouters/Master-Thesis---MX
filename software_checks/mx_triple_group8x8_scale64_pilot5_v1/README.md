# 8×8-groepen met vaste ×64 — gezamenlijke MX-pilot van vijf epochs

Deze proef verandert de gedeelde MX-groepen naar **8 outputkanalen × 8 posities in de afgevlakte Cin×Kh×Kw-dimensie**. Eén groep bevat 64 gewichten en één E8M0-bronschaal. De bestaande route blijft **E4M3 → E2M3 → E2M1**, met exponenttruncatie, vaste ×64-schalen voor de lagere precisies en mantissatruncatie. Activaties blijven A32; mastergewichten, gradiënten en Adam-momenten blijven FP32.

## Plaatsen en uitvoeren

Pak de map `mx_triple_group8x8_scale64_pilot5_v1` uit onder `C:\Users\renzo\master-thesis-mx\software_checks`. Gebruik de bestaande trainingsomgeving:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group8x8_scale64_pilot5_v1\run_train.py" check
```

Na `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group8x8_scale64_pilot5_v1\run_train.py" run
```

De runner start vanaf de oorspronkelijke initialisatie met seed 20261009 en stopt uiterlijk na **vijf epochs**. Het 100-epoch-LR-schema met dalingen na 50/75/90 blijft ingesteld; de LR is in deze pilot 0,001. De bestaande `train_32x32.mat` en `test_32x32.mat` worden hergebruikt met de oorspronkelijke transformatie naar 40×40. Groepsvorm en schaalfactor veranderen deze bestandsnamen niet.

Geen automatische installatie of upgrade van dependencies. Gebruik voor een andere datamap `--data-root "PAD_NAAR_SVHN"`. Alleen expliciet `--download` staat het downloaden van ontbrekende gegevens toe. SVHN-extra wordt niet gebruikt.

## Resultaten en notebook

Iedere opdracht maakt een nieuwe map onder `mx_runs\mx_triple_group8x8_scale64_pilot5`. Bestaande code en resultaten blijven behouden.

Open **`MX_triple_group8x8_scale64_pilot5_resultaten.ipynb` in de resultatenmap** en kies Run All. De runner vult het resultatenpad in. De notebook wordt onuitgevoerd meegeleverd; `SAVE_FIGURES=False` is de standaard. Er is geen tweede uitgevoerde notebook nodig.

De notebook bevat:

- Accuracycurves voor alle drie precisies, naast de eerste vijf epochs van de bestaande volledige **1×32 ×64**- en **1×64 ×64**-trainingen.
- Accuracygrafieken op epoch 5 en op het gezamenlijke checkpoint met de beste E4M3-accuracy binnen het gemeenschappelijke bereik van maximaal vijf epochs. Alle drie scores komen per run uit hetzelfde checkpoint.
- Schaalbytes, paddingposities en bitgepakte code- en schaalopslag per laag en voor het gehele MX-gedeelte.
- Trainingstijd, geregistreerd CUDA-geheugen, quantisatiediagnostiek en controles van de zes geselecteerde compacte exports.
- Bronverantwoording en beperkingen; geen verzonnen 8×8-accuracy op epoch 100.

De referenties zijn echte afgeronde 100-epochtrainingen. Alleen hun eerste vijf epochs worden vergeleken. Dit zijn niet de afzonderlijke oude pilots; de oorspronkelijke CUDA-backend geeft geen bitexacte herhaalgarantie.

Deel na afloop **`feedback_bundle.zip` uit de resultatenmap van `run`**. De grote checkpoints blijven lokaal en zitten niet in de bundel.

## Exacte indeling en truncatie

De gewichtsmatrix heeft vorm `[Cout, Cin×Kh×Kw]`. Vul beide assen met positieve nullen aan tot veelvouden van acht. De codevorm is `[ceil(Cout/8), ceil(K/8), 64]`, met blokken in rijvolgorde en binnen ieder blok eerst de acht kolommen van een rij. De schaalvorm heeft de eerste twee dimensies van de codevorm.

De E4M3-bronschaal en scalaire afronding behouden het bestaande beleid. De exponentbewerking is `s e3 e2 e1 e0 m2 m1 m0 → s e3 e2 m2 m1 m0`. De E2M3-schaalcode is de E4M3-schaalcode +6, dus de schaal is exact ×64. E2M1 ontstaat met `E2M3_code >> 2` en behoudt de E2M3-schaal. Geen geleerde schaal en geen nieuwe afronding van afgeleide codes.

De unieke layoutnaam en het checkpointcontract onderscheiden 8×8 van 1×64, ondanks dezelfde groepsomvang. Oude checkpoints worden niet omgeïnterpreteerd. G(W), identity-STE, drie BN-banken, recursieve distillatie en één Adamupdate per minibatch blijven gelijk.

## Kosten: schalen, padding en opslag

De zes MX-lagen bevatten **129.472 echte gewichten**. De precisiekolommen zijn alternatieve exports; alle bedragen hieronder omvatten code- en schaalbytes inclusief padding.

| Indeling | Groepen / schaalbytes | Paddingposities | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---|---:|---:|---:|---:|---:|
| 1×32 | 4.080 | 1.088 | 134.640 B | 102.000 B | 69.360 B |
| 1×64 | 2.160 | 8.768 | 140.400 B | 105.840 B | 71.280 B |
| **8×8** | **2.024** | **64** | **131.560 B** | **99.176 B** | **66.792 B** |

De E4M3-bron volstaat om de lagere precisies af te leiden; tel de kolommen niet op als noodzakelijke opslag. De vaste +6-offset is globale configuratie. Zonder padding is de schaaloverhead voor 1×64 en 8×8 gelijk: 0,125 bit per gewicht. De concrete laagvormen verklaren het verschil hierboven.

FP32-uitzonderingen en BN-staat staan apart in de exports. Trainingsmasters, optimizer, bestandsheaders en tijdelijke emulatietensors vallen buiten deze payloadtabel. De bestaande exports behouden alle vijf BN-banken; de recente transferproef telde alleen operationele banken en gebruikte een andere opslagconventie.

Deze pilot meet accuracy en PyTorch-emulatietijd. Vierkante groepen behouden hun blokverzameling bij transponeren; de codecontroles toetsen dit. Er volgt geen claim over native hardwarelatency, energie of transfers uit deze training. Hogere accuracy is niet gegarandeerd, omdat de schaal nu tussen meerdere outputfilters wordt gedeeld.

## Stoppen en hervatten

Eén keer Ctrl+C vraagt een stop na de lopende volledige epoch; een tweede keer stopt direct. Een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap vraagt ook een stop op de epochgrens.

Hervat alleen een nog onvoltooide run uit ditzelfde 8×8-pilotpakket:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group8x8_scale64_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Model, optimizer, scheduler en RNG worden hersteld. De oorspronkelijke CUDA-backend garandeert geen bitexacte hervatting. Het contract weigert oude rijgroepen, andere schaalvarianten en checkpoints die al vijf epochs hebben voltooid.

## Interpretatie en controles

Vijf epochs controleren werking en vroege stabiliteit. De uiteindelijke vergelijking vraagt daarna een afzonderlijke 100-epochproef. Lage accuracy is op zichzelf een geldig meetresultaat en geen technische FAIL. Het betreft één seed; de officiële testset wordt ook voor checkpointselectie gebruikt.

Zie `PROTOCOL.md` en `local_validation/VALIDATION.md` voor het contract en werkelijk uitgevoerde controles. De voorbereiding gebruikt kleine synthetische CPU-fixtures en start geen echte SVHN-training. Het commando `check` controleert vervolgens jouw Windows/CUDA-omgeving.
