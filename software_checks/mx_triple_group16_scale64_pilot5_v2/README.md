# Groepsgrootte 16 met vaste ×64 — gezamenlijke MX-pilot van vijf epochs

## Herstel in versie 2

De verwachte SVHN-bestandsnamen zijn hersteld naar `train_32x32.mat` en `test_32x32.mat`. Versie 1 verwachtte door een tekstvervanging onterecht `32x64` en stopte vóór de eerste epoch. Gebruik dit nieuwe v2-pakket in een aparte map naast v1. Het onderzoeksprotocol, de MX-code, notebookanalyse en kostenberekeningen blijven gelijk. De echte opstartfunctie is nu apart getest; zie `HERSTEL.md`.

Deze proef verandert **alleen de groepsgrootte van 32 naar 16** ten opzichte van de bestaande vaste-×64-referentie. E4M3 is de gedeelde bron; E2M3 ontstaat door de hoogste exponentbits te behouden, gevolgd door vaste schaalcorrectie ×64; E2M1 ontstaat door mantissatruncatie met behoud van de E2M3-schaal. Activaties blijven A32; mastergewichten, gradients en Adam-momenten blijven FP32.

## Plaatsen en uitvoeren

Pak deze ZIP uit in `C:\Users\renzo\master-thesis-mx\software_checks`. Daarna staat de runner hier:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_group16_scale64_pilot5_v2\run_train.py
```

Gebruik dezelfde bestaande `.venv-train`. Eerst de technische controle:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group16_scale64_pilot5_v2\run_train.py" check
```

Bij `EINDSTATUS: PASS` start je de vijf epochs:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group16_scale64_pilot5_v2\run_train.py" run
```

Het pakket installeert of vernieuwt geen dependencies. Het hergebruikt de bestaande SVHN train- en testbestanden. Indien nodig kan `--data-root "PAD_NAAR_SVHN"` worden meegegeven; alleen met expliciet `--download` worden ontbrekende gegevens gedownload. SVHN-extra wordt niet gebruikt.

Het pakket stopt uiterlijk na vijf epochs. Het 100-epoch-LR-schema blijft ingesteld met dalingen na 50/75/90 en wordt niet samengedrukt. Deze proef begint met de oorspronkelijke initialisatie en seed, zonder een getraind B32-checkpoint als startpunt.

## Resultaten en notebook

Elke opdracht schrijft naar een nieuwe map onder `mx_runs\mx_triple_group16_scale64_pilot5`. De training bewaart de vijf epochcheckpoints, het gezamenlijke beste checkpoint geselecteerd op E4M3, het laatste checkpoint, compacte MX-exports en een `feedback_bundle.zip`.

Open `MX_triple_group16_scale64_pilot5_resultaten.ipynb` **in de resultatenmap**. De runner vult het resultatenpad in. De notebook is onuitgevoerd en leest de opgeslagen resultaten; `SAVE_FIGURES=False` is de standaard.

De analyse bevat:

- Accuracycurves van alle drie de precisies, met de eerste vijf epochs van de bestaande **B32 ×64**-run als referentie.
- Accuracygrafieken voor **epoch 5** en het **beste E4M3-checkpoint binnen de pilot**, steeds met alle drie de precisies uit hetzelfde checkpoint. Het globale beste referentiecheckpoint op epoch 42 wordt niet voor deze pilotvergelijking gebruikt.
- Trainingstijd en gemeten CUDA-geheugencijfers, los van de berekende bitgepakte representatiegrootte.
- Quantisatie- en exportdiagnostiek op de opgeslagen beste en laatste checkpoints.
- De expliciete kostenvergelijking hieronder, per laag en voor de zes MX-lagen samen.

Bij een latere volledige proef komen opnieuw grafieken voor epoch 100 en het globale E4M3-beste checkpoint. Dit pilotpakket claimt geen nieuwe epoch-100-resultaten.

Deel na afloop de `feedback_bundle.zip` uit de map van **de training**. De grote checkpoints blijven lokaal en zitten niet in de feedbackbundel.

## Kosten: schalen, padding en totale opslag

Zonder padding is de schaaloverhead `8/B` bits per gewicht. Maar onze groepering verloopt per outputkanaal over de afgevlakte `Cin*Kh*Kw`-rij, met padding aan het einde van iedere rij. Daarom worden alle kosten uit de werkelijke laagvormen berekend.

| Groepsgrootte B | Groepen / schalen | Paddingposities | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---:|---:|---:|---:|---:|---:|
| 16 | 8.096 | 64 | 137.632 bytes | 105.248 bytes | 72.864 bytes |
| 32 | 4.080 | 1.088 | 134.640 bytes | 102.000 bytes | 69.360 bytes |
| 64 | 2.160 | 8.768 | 140.400 bytes | 105.840 bytes | 71.280 bytes |

Dit betreft **129.472 werkelijke gewichten in zes MX-lagen**, met hun padding en één E8M0-schaal per groep. De E2M3/E2M1-kolommen zijn afzonderlijke lagere-precisie-exportrepresentaties; ze worden niet bij E4M3 opgeteld alsof drie volledige modellen tegelijk nodig zijn. De E4M3-bron volstaat voor de bitafleiding; de vaste +6-offset is globale configuratie en vraagt geen leerbare keuze per groep.

B64 heeft minder schalen, maar in deze indeling meer totale bronopslag dan B32 door extra padding. De grootste laag heeft 800 gewichten per outputkanaal: 16 en 32 delen dit exact, 64 vult elke rij tot 832 aan. B64 is in deze notebook een **berekende kostenvariant**, geen nieuwe gemeten accuracy.

Deze bedragen zijn geen volledige checkpointgrootte, geen gemeten GPU-geheugen en geen hardware-energie- of snelheidsmeting. FP32-uitzonderingen, BN-toestand, bestandheaders, containers en tijdelijke emulatietensors zitten er niet in. `group_size_costs.json`, `group_size_costs_summary.csv` en `group_size_costs_layers.csv` bevatten de reproduceerbare berekeningen. Iedere werkelijke export wordt tegen deze berekening gecontroleerd.

## Stoppen en hervatten

Eén keer `Ctrl+C` vraagt een stop na de lopende volledige epoch. Een tweede keer stopt direct. Ook een leeg bestand `STOP_AFTER_EPOCH` in de actieve resultatenmap vraagt een stop na de lopende epoch.

Hervat vanaf een volledig geschreven eigen B16-checkpoint, bijvoorbeeld:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_group16_scale64_pilot5_v2\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Het checkpoint bewaart optimizer, scheduler en RNG-toestand. De originele CUDA-backend heeft geen bitexacte hervatgarantie. B32-checkpoints en checkpoints van een andere schaalvariant worden door het contract geweigerd; deze pilot hervat uitsluitend tot uiterlijk epoch 5.

## Interpretatie

Een lage accuracy is een onderzoeksuitkomst en veroorzaakt op zichzelf geen technische FAIL. De pilot controleert werking, looptijd en vroege trends. Een conclusie over uiteindelijke accuracy vraagt de volledige training. B16 is een onderzoeksvariant van de groepsgrootte, niet de concrete standaard-MX-indeling met B32.

De codec, packing, exports, gradients, minibatchroute en stop/hervatwerking zijn met kleine synthetische CPU-controles gevalideerd. De lokale `check` controleert de code vervolgens op jouw Windows/CUDA-omgeving. Zie `local_validation/VALIDATION.md` voor de precieze testomvang. Er is bij de voorbereiding geen echte SVHN-training gestart.
