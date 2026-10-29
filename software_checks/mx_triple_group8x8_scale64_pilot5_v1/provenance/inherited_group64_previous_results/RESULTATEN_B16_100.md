# B16 ×64 tegenover B32 ×64 — resultaten van 100 epochs

Bron: de ongewijzigde gebruikersbundel `feedback_B16_full100_original.zip` in deze map, oorspronkelijk `feedback_bundle(20261026-195319).zip`. SHA256: `c8d3f0ff1d4f74eccda551a0c2c8cb3710cdcac8dddcc9c7fd22288fc4587271`. De ingebedde `scale64_reference` bevat de bestaande B32-run. De tabellen hieronder zijn afgeleid van `epoch_metrics.csv`, `selected_metrics.csv`, `checkpoint_selection.json`, `history.json`, `epoch_times.csv` en de kostenrapporten. CSV-tabellen en de audit staan naast dit document.

## Hoofdbevinding

De B16-training is technisch geslaagd. **B16 heeft op epoch 100 geen voordeel tegenover B32:** alle drie accuracies zijn lager en de bitgepakte representatie is groter. De checkpointselectie op E4M3 geeft wel een afweging: de gekozen B16-epoch heeft betere lagere modi, met iets minder E4M3-accuracy. Die twee conclusies horen bij verschillende checkpointregels en moeten afzonderlijk worden gerapporteerd.

## 1. Gemeenschappelijk eindpunt: epoch 100

| Precisie | B32 ×64 | B16 ×64 | B16 − B32 [procentpunt] | Verschil juiste voorspellingen |
|---|---:|---:|---:|---:|
| E4M3 |94,929318%|94,791026%|−0,138291|−36|
| E2M3 |94,898586%|94,545175%|−0,353411|−92|
| E2M1 |94,791026%|94,572065%|−0,218961|−57|

Iedere accuracy is gemeten op dezelfde 26.032 officiële testbeelden. Het gaat om verschillen in aantallen juiste voorspellingen; zonder voorspellingen per beeld zijn dit geen aantallen specifieke beelden waarop de runs van elkaar verschillen.

## 2. Gemeenschappelijk checkpoint binnen elke run, gekozen op E4M3

B32 selecteert **epoch 42**, B16 **epoch 53**. De drie scores in iedere kolom komen uit één checkpoint. Er worden geen afzonderlijke beste E2M3- of E2M1-epochs gecombineerd.

| Precisie | B32 ×64, epoch 42 | B16 ×64, epoch 53 | B16 − B32 [procentpunt] | Verschil juiste voorspellingen |
|---|---:|---:|---:|---:|
| E4M3 |95,086816%|94,940842%|−0,145974|−38|
| E2M3 |93,842194%|94,429932%|+0,587738|+153|
| E2M1 |93,922864%|94,399201%|+0,476337|+124|

Onder deze vooraf afgesproken selectieregel levert B16 dus betere lagere modi. Het is wel een vergelijking van verschillende trainingsfasen: B32 wordt vóór de LR-daling na epoch 50 gekozen, B16 erna.

Als aanvullende diagnose uit de reeds opgeslagen resultaten, zonder een nieuwe evaluatie of training, kunnen we beide runs op epoch 53 bekijken:

| Precisie | B32, epoch 53 | B16, epoch 53 | B16 − B32 [procentpunt] |
|---|---:|---:|---:|
| E4M3 |94,948525%|94,940842%|−0,007683|
| E2M3 |94,587431%|94,429932%|−0,157498|
| E2M1 |94,572065%|94,399201%|−0,172864|

De betere lagere-modusscores in de vorige tabel hangen dus samen met de gekozen epochs. Ze tonen geen algemeen voordeel van B16 doorheen de training. De aanvullende diagnose vervangt de afgesproken twee hoofdvergelijkingen niet.

## 3. Verloop en de eerdere pilot

In **alle 25 epochs van 76 tot en met 100** ligt B16 bij alle drie modi onder B32. De laatste epoch is daarmee geen geïsoleerde uitschieter binnen deze runs.

| Precisie | B32, gemiddelde epochs 91–100 | B16, gemiddelde epochs 91–100 | Verschil [procentpunt] |
|---|---:|---:|---:|
| E4M3 |94,919714%|94,841733%|−0,077981|
| E2M3 |94,847880%|94,581285%|−0,266595|
| E2M1 |94,741856%|94,541718%|−0,200138|

Deze gemiddelden beschrijven gecorreleerde epochs van dezelfde training. Het zijn geen tien onafhankelijke herhalingen en geen onzekerheidsinterval over seeds.

De full100-run begon opnieuw vanuit de oorspronkelijke initialisatie. Op epoch 5 bedroegen de scores 93,054702% / 91,452827% / 91,560387% voor E4M3/E2M3/E2M1. De afzonderlijke B16-pilot gaf op epoch 5 92,931776% / 92,059773% / 92,167333%. De initiële modelhash, seed en negentien numerieke/runtime/testmodules komen overeen; de oorspronkelijke CUDA-backend met `cudnn.benchmark=True` biedt geen bitexacte herhaalgarantie. De vijf-epochproef was een werkingscontrole en vroege indicatie, geen voorspelling van de eindrangschikking.

## 4. Kosten van groepsgrootte

Voor dezelfde 129.472 werkelijke gewichten worden groepen per outputkanaal gevormd en rijen afzonderlijk aangevuld. Eén gedeelde schaal kost acht bits. De ideale schaaloverhead zonder padding is `8/B` bits per gewicht.

| Groepsgrootte | Groepen / schaalbytes | Paddingposities | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---:|---:|---:|---:|---:|---:|
|16|8.096|64|137.632 bytes|105.248 bytes|72.864 bytes|
|32|4.080|1.088|134.640 bytes|102.000 bytes|69.360 bytes|
|64|2.160|8.768|140.400 bytes|105.840 bytes|71.280 bytes|

B16 kost ten opzichte van B32 2.992 extra bronbytes (+2,22%). Een afzonderlijke E2M3-export kost 3.248 extra bytes (+3,18%), een E2M1-export 3.504 extra bytes (+5,05%). De bron volstaat om de lagere codes af te leiden; tel de drie exportrepresentaties niet bij elkaar op.

B64 heeft minder schalen maar meer padding. De grootste laag heeft 800 gewichten per outputkanaal; B64 vult iedere rij aan tot 832. De B64-bron kost daardoor 5.760 bytes meer dan B32 (+4,28%). De B64-getallen zijn op dit moment analytische opslagkosten; er is nog geen echte B64-training uitgevoerd.

De kosten omvatten codes, padding en één schaalbyte per groep in de zes MX-lagen. FP32-uitzonderingen, BatchNorm, optimizer, bestandcontainers en tijdelijke werkbuffers zijn uitgesloten. Het zijn geen gemeten hardware-energiecijfers of totale GPU-geheugencijfers.

## 5. Gemeten duur

| Run | Som van de 100 epochs | Totale procesduur |
|---|---:|---:|
|B16 ×64|1 u 37 min 54 s|1 u 38 min 1 s|
|B32 ×64|1 u 40 min 33 s|1 u 40 min 42 s|

De som van de epochtijden is bij B16 158,850 seconden lager, of 2,633%. Dit is een beschrijvende vergelijking van afzonderlijke laptopruns onder PyTorch-emulatie. De meting bewijst geen native-hardwarewinst door een kleinere groepsgrootte.

## 6. Technische audit en grenzen

De run heeft alle 100 epochs, 57.300 Adam-updates en 171.900 trainingspasses voltooid. De audit controleert 1.712 consistentievoorwaarden: metrics en integeraantallen, volledige geregistreerde datadekking, LR-fasen, A32/FP32, BN, gedeelde bron, vaste schaaloffset +6, selectie, exportgeometrie en herkomst. De pakketmanifesten zijn gelijk aan het oorspronkelijk geleverde full100-pakket. De feedbacknotebook verschilt alleen door de bedoelde parametercel met het resultatenpad. De bestaande B16-notebook accepteert de run en referentie zonder analysefouten.

De feedback bevat geen grote `.pt`-checkpoints of compacte exportbestanden. De exacte CUDA-exportvergelijkingen zijn door de training gerapporteerd; hier zijn hun registratie en de exportgeometrie gecontroleerd. Er is geen nieuwe binaire modelevaluatie uitgevoerd.

Er is één seed per variant en geen bitexacte CUDA-garantie. De officiële testset wordt voor checkpointselectie gebruikt en vormt daarom geen onafhankelijke schatting na modelselectie. De conclusie blijft: **binnen deze gemeten runs is B32 gunstiger op het gemeenschappelijke eindpunt en qua opslag; de E4M3-selectieregel laat een afzonderlijke afweging zien.**

## Vervolg

Onderzoek B64 met dezelfde vaste factor ×64, dezelfde bronroute en dezelfde training, eerst vijf epochs. Vergelijk binnen die pilot uitsluitend hetzelfde bereik met de bestaande B16- en B32-runs. Houd de totale opslag inclusief padding zichtbaar. Een daaropvolgende volledige training kan vervolgens weer op epoch 100 en op het E4M3-geselecteerde checkpoint worden beoordeeld.
