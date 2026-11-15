# MSB met exponentcorrectie: 100 epochs vanaf nul met schaalverhouding ×64

Dit pakket voert **één nieuwe MSB-training van 100 epochs vanaf een verse initialisatie** uit. De vergelijking gebruikt de reeds gemeten 100 epochs van de oorspronkelijke LSB ×64-run. De eerdere MSB-pilot van vijf epochs wordt niet hervat: haar checkpoint en metingen worden niet in deze nieuwe run opgenomen. Er worden geen nieuwe LSB- of RNE-runs gestart en er is geen getraind oudercheckpoint nodig.

De centrale schaalafspraak is:

`S_E2 = 64 × S_E4`

De **verhouding** staat vast. De absolute E4- en E2-schalen kunnen tijdens training veranderen, omdat de E4-schaal uit de actuele gewichten wordt berekend. Er zijn geen onafhankelijke `phi`-parameters en geen bevroren absolute E2-schalen.

## Uitpakken en controleren

Pak de pakketmap uit onder:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_mantissa_msb_scale64_reference100_v1`

Gebruik de bestaande `.venv-train` en bestaande SVHN-train/testbestanden.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_scale64_reference100_v1\run_train.py" check
```

`check` controleert bronbestanden, initialisatie en de numerieke implementatie. Het voert twee echte Adam-updates uit op een wegwerpmodel met drie synthetische beelden. Dat is geen SVHN-training of meting van nieuwe onderzoeksaccuracy. De oorspronkelijke dataset en historische onderzoeksresultaten blijven ongewijzigd.

Na afloop geeft de **laatste regel in de terminal** het globale resultaat van deze opdracht:

```text
GLOBAAL RESULTAAT: PASS
```

Bij een fout eindigt de opdracht met `GLOBAAL RESULTAAT: FAIL`. Dit resultaat verschijnt nadat de rapportage en feedbackbundel zijn afgehandeld. Een `PASS` bij `check` betekent dat de softwarecontrole is geslaagd; de 100-epochtraining start met de volgende opdracht. Waarschuwingen worden per soort met een aantal samengevat, zodat herhalingen het eindresultaat niet overspoelen.

## De 100 nieuwe epochs uitvoeren

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_scale64_reference100_v1\run_train.py" run
```

Voor een andere datasetmap kan `--data-root "VOLLEDIG_PAD\svhn"` worden toegevoegd. Een `--parent-checkpoint` is niet nodig: het model wordt nieuw opgebouwd.

De runner gebruikt dezelfde beginseed `20261009` als de historische LSB ×64-run en controleert de volledige modelinitialisatie. De verwachte init-hash is:

`a8fcd20c40a204ba24cabfa36258f9501d55b9d4d4bdfb8ba742b8c827af127f`

Deze initmatch moet door `check` of `run` in de oorspronkelijke `.venv-train` en CUDA-omgeving worden bevestigd, vóór de eerste trainingupdate. Lokale CPU-softwaretests bevestigen deze productie-initialisatie niet.

Epochs **1–100** vormen één nieuw traject vanaf die initialisatie. Het pakket zoekt of laadt geen pilotcheckpoint automatisch. Er is geen epoch-0-baselinemeting. De model-, data- en RNG-volgorde volgt de oorspronkelijke runner; extra controles mogen de trainings-RNG niet verbruiken.

## Trainingsafspraken

| Onderdeel | Instelling |
|---|---|
| Nieuwe methode | MSB-mantissaverwijdering met optimale beperkte exponentkeuze |
| Model | CNN-8 / `svhnq`, expand=2; 132.954 parameters |
| Data | SVHN train: 73.257; test: 26.032; geen extra-split |
| Budget | 100 epochs vanaf nul |
| Optimizer | Nieuwe Adam, begin-LR 0,001, weight decay 0 |
| Scheduler | Oorspronkelijke milestones 50, 75 en 90; gamma 0,1 |
| Activaties en masters | A32; FP32-mastergewichten, gradients en Adam-momenten |
| MX-groepen | B32; oorspronkelijke groepsindeling |
| E4-schaal | Dynamisch berekend uit de actuele effectieve gewichten |
| E2-schaal | Steeds E4-schaal ×64; schaalcode +6, zonder clipping/wrap bij overloop |
| Geleerde schalen | Geen |
| Gewichten en BN | Trainbaar; drie actieve BN-banken 8/32/4 |
| Gezamenlijke training | E4M3 hard CE, E2M3/E2M1 recursieve detached soft CE; drie passes, één Adam-update per minibatch |
| Primair vergelijkingsmoment | Epoch 100 voor zowel nieuwe MSB als historische LSB |

Het oorspronkelijke LR-schema wordt volledig behouden. De schedulerstap gebeurt na de evaluatie op de epochgrens; de lagere LR wordt vanaf de volgende epoch gebruikt:

| Trainingsepochs | Gebruikte LR |
|---|---:|
| 1–50 | 0,001 |
| 51–75 | 0,0001 |
| 76–90 | 0,00001 |
| 91–100 | 0,000001 |

De nieuwe mantissaregel behoudt uit de E2M3-bron de oorspronkelijke kleinste fractionele mantissabit `m0`. De exponent wordt gekozen om de lokale bronfout bij die beperking te minimaliseren. Voor deze codeboeken is dezelfde regel uit de numerieke proef equivalent aan `E' = min(3, E + int(m == 6))`. De tekenbit en gedeelde E2-schaal blijven behouden. “Optimaal” slaat uitsluitend op deze beperkte lokale omzettingsfout en is geen belofte over netwerkaccuracy.

De E4M3→E2M3-conversie en de gezamenlijke training blijven behouden. Omdat alle drie de precisies binnen de nieuwe run dezelfde trainbare masters gebruiken, kunnen ook de W6- en W8-resultaten gaan verschillen van de historische LSB-run.

## Tussentijds stoppen en hervatten

Voor een bewuste deelrun kan bijvoorbeeld `--stop-after-epoch 50` worden toegevoegd. Toegestane waarden zijn 1–100; het volledige geplande budget blijft 100 epochs. Een bewuste stop vóór epoch 100 krijgt in de terminal `GLOBAAL RESULTAAT: GESTOPT`. Een volledig voltooide training krijgt `PASS`; een fout krijgt `FAIL`.

Tijdens training vraagt één keer `Ctrl+C` een stop aan op de epochgrens: de lopende epoch wordt afgewerkt, geëvalueerd en opgeslagen. Een tweede `Ctrl+C` onderbreekt direct. Alleen volledig opgeslagen epochs zijn hervatbaar.

Hervat uitsluitend met een checkpoint **uit deze nieuwe volledige MSB ×64-run**:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_scale64_reference100_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_NIEUWE_FULL100_RESULTATENMAP\checkpoints\epoch_00050.pt"
```

Bewaar de volledige resultatenmap en de bijbehorende `.pt.sha256`-bestanden. Geef zo nodig opnieuw `--data-root` op. Zonder verkorte stopoptie hervat de runner tot epoch 100, met de opgeslagen model-, Adam-, scheduler- en RNG-toestand. Een checkpoint uit de vijf-epochpilot, de historische LSB-run of een onafhankelijke-schaalproef wordt niet als geldig hervatbestand voor dit pakket aanvaard.

## Resultaten en feedback

Open in de nieuwe resultatenmap:

`MX_MSB_scale64_reference100_resultaten.ipynb`

De notebook leest `run_report.json`, `accuracy.csv` en `reference_lsb.csv` via de meegeleverde `training_report.py`. Zij laadt geen model of dataset en start geen training. Als de resultatenmap is verplaatst, pas `RUN_DIR` in de eerste codecel aan.

De curves over epochs 1–100 hebben afzonderlijke panelen voor W4, W6 en W8 en gebruiken duidelijke labels:

- **LSB ×64 (eerder gemeten):** de vastgelegde historische referentie.
- **MSB ×64 (nieuwe run):** uitsluitend de werkelijk gemeten nieuwe epochs.

De eindpuntgrafiek met methoden op de categorische as, gelijke cirkels en percentages verschijnt pas als alle 100 nieuwe epochs en de referentiecontroles compleet zijn. De tabel geeft het verschil in procentpunten tegenover de historische LSB-meting op **epoch 100**. Alle 300 nieuwe en 300 historische testmetingen moeten geldig zijn; initialisatie, runtime, bron en dataset moeten als overeenkomstig zijn bevestigd. De exacte historische aantallen worden via één canonieke digest gecontroleerd. Er worden geen ontbrekende scores ingevuld en geen beste checkpoints uit de testcurve geselecteerd.

Een deelrun toont uitsluitend de daadwerkelijk opgeslagen nieuwe meetpunten en wordt als onvolledig aangeduid. De historische curve mag al alle 100 eerder gemeten epochs tonen; dat zijn geen nieuwe resultaten. Pending, check-, fout- en synthetische uitvoeringen tonen geen nieuwe onderzoeksaccuracies. In de actuele notebookstatus verschijnt iedere soort waarschuwing één keer, met het geregistreerde aantal. Matplotlib is optioneel voor de figuren; de rapportage- en notebookgenerator gebruiken verder de Python-standaardbibliotheek.

Stuur na afloop de nieuwe `feedback_bundle.zip` terug en bewaar de volledige lokale resultatenmap voor eventueel hervatten. Nieuwe MSB-resultaten zijn bij levering nog onbekend.

## Wat de vergelijking wel en niet vaststelt

Deze proef vergelijkt twee trajecten van 100 epochs onder hetzelfde vastgelegde trainingsschema en dezelfde gecontroleerde modelinitialisatie. De referentiebestanden bevestigen de historische configuratie, bron, data en modelinit-hash. Zij bevatten geen historische begin-RNG of samplevolgordebestanden waarmee die twee onderdelen rechtstreeks kunnen worden vergeleken. Gelijke seed is geen bewijs van identieke historische samplevolgorde of bitexacte CUDA-uitvoering.

Er is één nieuwe MSB-run en één eerder gemeten LSB-run, zonder herhalingen over meerdere seeds. Honderd epochs met één vaste configuratie bewijzen geen optimale trainingsinstellingen of algemene superioriteit. De officiële testset wordt herhaald gemeten; het vooraf gekozen primaire eindpunt is epoch 100. De historische beste epoch 42 wordt niet geselecteerd. De eerdere MSB-pilot is een afzonderlijke proef en maakt geen deel uit van deze nieuwe curves of eindpuntmeting.

De eerdere post-hocproef vertrok van een reeds getraind model met een andere schaaltoestand. Haar accuracies zijn geen baseline voor deze training vanaf nul. Een positief of negatief resultaat hier geldt voor deze nieuwe proef en wordt niet zonder meer veralgemeend. De FP32-emulatie onderbouwt evenmin hardwarelatency, energieverbruik of algemene hardwarewinst. De gemeten epochtijden bevatten extra controles van codes, decodeeruitkomsten en gradients; zij vormen geen gecontroleerde snelheidsvergelijking met de historische LSB-run.
