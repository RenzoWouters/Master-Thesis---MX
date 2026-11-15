# MSB met exponentcorrectie: vijf epochs vanaf nul met schaalverhouding ×64

Dit pakket voert **één nieuwe MSB-training vanaf een verse initialisatie** uit, gedurende vijf epochs. De vergelijking gebruikt de reeds gemeten eerste vijf epochs van de oorspronkelijke LSB ×64-run. Er worden geen nieuwe LSB- of RNE-runs gestart en er is geen getraind oudercheckpoint nodig.

De centrale schaalafspraak is:

`S_E2 = 64 × S_E4`

De **verhouding** staat vast. De absolute E4- en E2-schalen kunnen tijdens training veranderen, omdat de E4-schaal uit de actuele gewichten wordt berekend. Er zijn geen onafhankelijke `phi`-parameters en geen bevroren absolute E2-schalen.

## Uitpakken en controleren

Pak de pakketmap uit onder:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_mantissa_msb_scale64_scratch5_v1`

Gebruik de bestaande `.venv-train` en bestaande SVHN-train/testbestanden.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_scale64_scratch5_v1\run_train.py" check
```

`check` controleert bronbestanden, initialisatie en de numerieke implementatie. Het voert twee echte Adam-updates uit op een wegwerpmodel met drie synthetische beelden. Dat is geen SVHN-training of meting van nieuwe onderzoeksaccuracy. De oorspronkelijke dataset en historische onderzoeksresultaten blijven ongewijzigd.

## De vijf nieuwe epochs uitvoeren

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_scale64_scratch5_v1\run_train.py" run
```

Voor een andere datasetmap kan `--data-root "VOLLEDIG_PAD\svhn"` worden toegevoegd. Een `--parent-checkpoint` is niet nodig: het model wordt nieuw opgebouwd.

De runner gebruikt dezelfde beginseed `20261009` als de historische LSB ×64-run en controleert de volledige modelinitialisatie. De verwachte init-hash is:

`a8fcd20c40a204ba24cabfa36258f9501d55b9d4d4bdfb8ba742b8c827af127f`

Deze initmatch moet door `check` of `run` in de oorspronkelijke `.venv-train` en CUDA-omgeving worden bevestigd, vóór de eerste trainingupdate. Lokale CPU-softwaretests bevestigen deze productie-initialisatie niet.

Epochs **1–5** zijn de eerste vijf trainingsepochs vanaf die initialisatie. Er is geen epoch-0-baselinemeting en geen voortzetting vanaf epoch 50 of 100. De model-, data- en RNG-volgorde volgt de oorspronkelijke runner; extra controles mogen de trainings-RNG niet verbruiken.

## Trainingsafspraken

| Onderdeel | Instelling |
|---|---|
| Nieuwe methode | MSB-mantissaverwijdering met optimale beperkte exponentkeuze |
| Model | CNN-8 / `svhnq`, expand=2; 132.954 parameters |
| Data | SVHN train: 73.257; test: 26.032; geen extra-split |
| Budget | Vijf epochs vanaf nul |
| Optimizer | Nieuwe Adam, begin-LR 0,001, weight decay 0 |
| Scheduler | Oorspronkelijke milestones 50, 75 en 90; gamma 0,1 |
| LR binnen deze pilot | 0,001 in alle vijf epochs |
| Activaties en masters | A32; FP32-mastergewichten, gradients en Adam-momenten |
| MX-groepen | B32; oorspronkelijke groepsindeling |
| E4-schaal | Dynamisch berekend uit de actuele effectieve gewichten |
| E2-schaal | Steeds E4-schaal ×64; schaalcode +6, zonder clipping/wrap bij overloop |
| Geleerde schalen | Geen |
| Gewichten en BN | Trainbaar; drie actieve BN-banken 8/32/4 |
| Gezamenlijke training | E4M3 hard CE, E2M3/E2M1 recursieve detached soft CE; drie passes, één Adam-update per minibatch |
| Vergelijkingsmoment | Epoch 5 voor zowel nieuwe MSB als historische LSB |

Het oorspronkelijke 100-epochschema blijft inhoudelijk behouden. Binnen deze pilot wordt bij epoch 5 gestopt; de eerste LR-verlaging op epoch 50 wordt dus niet bereikt. De pilot loopt niet automatisch door tot 100 epochs.

De nieuwe mantissaregel behoudt uit de E2M3-bron de oorspronkelijke kleinste fractionele mantissabit `m0`. De exponent wordt gekozen om de lokale bronfout bij die beperking te minimaliseren. Voor deze codeboeken is dezelfde regel uit de numerieke proef equivalent aan `E' = min(3, E + int(m == 6))`. De tekenbit en gedeelde E2-schaal blijven behouden. “Optimaal” slaat uitsluitend op deze beperkte lokale omzettingsfout en is geen belofte over netwerkaccuracy.

De E4M3→E2M3-conversie en de gezamenlijke training blijven behouden. Omdat alle drie de precisies binnen de nieuwe run dezelfde trainbare masters gebruiken, kunnen ook de W6- en W8-resultaten gaan verschillen van de historische LSB-run.

## Tussentijds stoppen en hervatten

Voor een bewuste deelrun kan bijvoorbeeld `--stop-after-epoch 2` worden toegevoegd. De maximale waarde is 5; de volledige geplande pilot blijft vijf epochs.

Tijdens training vraagt één keer `Ctrl+C` een stop aan op de epochgrens: de lopende epoch wordt afgewerkt, geëvalueerd en opgeslagen. Een tweede `Ctrl+C` onderbreekt direct. Alleen volledig opgeslagen epochs zijn hervatbaar.

Hervat met een checkpoint **uit deze nieuwe MSB ×64-run**:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_scale64_scratch5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_RESULTATENMAP\checkpoints\epoch_00002.pt"
```

Bewaar de volledige resultatenmap en de bijbehorende `.pt.sha256`-bestanden. Geef zo nodig opnieuw `--data-root` op. Zonder verkorte stopoptie hervat de runner tot het geplande eindpunt van vijf epochs. Een checkpoint uit de historische LSB-run of een eerdere onafhankelijke-schaalproef is niet het hervatbestand voor deze nieuwe proef.

## Resultaten en feedback

Open in de nieuwe resultatenmap:

`MX_MSB_scale64_scratch5_resultaten.ipynb`

De notebook leest `run_report.json`, `accuracy.csv` en `reference_lsb.csv` via de meegeleverde `training_report.py`. Zij laadt geen model of dataset en start geen training. Als de resultatenmap is verplaatst, pas `RUN_DIR` in de eerste codecel aan.

De curves hebben afzonderlijke panelen voor W4, W6 en W8 en gebruiken duidelijke labels:

- **LSB ×64 (eerder gemeten):** de vastgelegde historische referentie.
- **MSB ×64 (nieuwe run):** uitsluitend de werkelijk gemeten nieuwe epochs.

De eindpuntgrafiek met methoden op de categorische as, gelijke cirkels en percentages verschijnt pas als alle vijf nieuwe epochs en de referentiecontroles compleet zijn. De tabel geeft het verschil in procentpunten tegenover de historische LSB-meting op epoch 5. Er worden geen ontbrekende scores ingevuld en geen beste checkpoints uit de testcurve geselecteerd.

Pending, check-, fout- en synthetische uitvoeringen tonen geen nieuwe onderzoeksaccuracies. Een eventueel zichtbare LSB-curve blijft expliciet als historische referentie gelabeld. Matplotlib is optioneel voor de figuren; de rapportage- en notebookgenerator gebruiken verder de Python-standaardbibliotheek.

Stuur na afloop de nieuwe `feedback_bundle.zip` terug en bewaar de volledige lokale resultatenmap voor eventueel hervatten. Nieuwe MSB-resultaten zijn bij levering nog onbekend.

## Wat de vergelijking wel en niet vaststelt

Deze proef vergelijkt vroege training onder hetzelfde vastgelegde budget en dezelfde gecontroleerde modelinitialisatie. De referentiebestanden bevestigen de historische configuratie, bron, data en modelinit-hash. Zij bevatten geen historische begin-RNG of samplevolgordebestanden waarmee die twee onderdelen rechtstreeks kunnen worden vergeleken. Gelijke seed is geen bewijs van identieke historische samplevolgorde of bitexacte CUDA-uitvoering.

Er is één nieuwe MSB-run en één eerder gemeten LSB-run, zonder herhalingen over meerdere seeds. Vijf epochs bewijzen geen convergentie of beste trainingsinstellingen. De officiële testset wordt herhaald gemeten; het vooraf gekozen primaire eindpunt is epoch 5. De historische beste epoch 42 en laatste epoch 100 worden niet als vergelijking voor deze vijf-epochpilot gebruikt.

De eerdere post-hocproef vertrok van een reeds getraind model met een andere schaaltoestand. Haar accuracies zijn geen baseline voor deze training vanaf nul. Een positief of negatief resultaat hier geldt voor deze nieuwe proef en wordt niet zonder meer veralgemeend. De FP32-emulatie onderbouwt evenmin hardwarelatency, energieverbruik of algemene hardwarewinst. De gemeten epochtijden bevatten extra controles van codes, decodeeruitkomsten en gradients; zij vormen geen gecontroleerde snelheidsvergelijking met de historische LSB-run.
