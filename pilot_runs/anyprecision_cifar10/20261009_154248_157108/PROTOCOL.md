# Any-Precision CIFAR-10: verkennende trainingspilot v1

Dit is een vooraf vastgelegde technische/inhoudelijke pilot, GEEN volledige
referentietraining, paperreproductie, modelselectie of definitieve accuracyvergelijking.
De instellingen hieronder zijn voorgesteld door de assistent voor deze pilot.

## Wat blijft uit de aangeleverde code behouden?

- `resnet20q`, `expand=5`, 6.786.330 parameters inclusief BatchNorm-banken.
- De ongewijzigde quantizers, STE, model-, loss- en optimizerbestanden.
- Elke minibatch: W32/A32 met echte labels; daarna W8/A8, W4/A4, W2/A2,
  W1/A1 met telkens de softmax van de voorafgaande hogere precisie als doel.
- Gradientbijdragen worden opgeteld, NIET gemiddeld. Een optimizerstap per minibatch.
- Adam, lr=0.001, weight_decay=0.0. De oorspronkelijke scheduler met mijlpalen
  100, 150 en 180 blijft behouden; binnen drie epochs verandert de lr niet.
- De oorspronkelijke `get_transform`: horizontale flip, crop32 met padding4,
  ToTensor, Normalize; bij validatie alleen ToTensor en Normalize.
  Means=[125.3,123.0,113.9]/255; stds=[63.0,62.1,66.7]/255.
- Eerste/laatste laag, eventuele gedeelde skip-BatchNorm en andere niet-gequantiseerde
  onderdelen worden NIET omgebouwd. Wk/Ak beschrijft de instelbare quantized lagen,
  niet alle opgeslagen of berekende tensors. De W32-route is GEEN aparte gewone FP32-ResNet.
- Tijdens training/evaluatie worden quantized waarden opnieuw uit FP32-mastergewichten
  berekend. Er is geen export of truncatie van persistent opgeslagen lage-bitcodes.

Bronlocaties in het oorspronkelijke archief: `train_cifar10.sh`, `train.py:forward`,
`optimizer.py`, `models/quan_ops.py`, `models/resnet_quan.py`, `models/losses.py`,
`datasets/data.py:get_transform`, `datasets/cifar10.py:CIFAR10`.

## Data en splits

Alle indices verwijzen naar de concatenatie data_batch_1 t/m data_batch_5.
Controle: 50.000 trainingsbeelden, labels 0 t/m 9, 5.000 per klasse.
Gebruik van de officiele testset voor training of evaluatie: NUL beelden.
Het downloadarchief bevat wel die testset en de originele loader controleert haar
bestandshash; de testbeelden worden niet gedecodeerd of aan het model aangeboden.

Stratificatie: NumPy default_rng(20261009), per klasse een permutatie. De eerste
500 indices per klasse vormen de volledige vastgelegde validatiepool (5.000 totaal);
de rest vormt de trainingspool (45.000). De pilot gebruikt de eerste 500 indices
van elke trainingspool per klasse en de eerste 100 van elke validatiepool.
De subsets bevatten dus 5.000 trainingsbeelden en 1.000 validatiebeelden.
De indices worden gesorteerd voor opslag; de DataLoader shuffelt alleen training.
Alle indices en klassentellingen staan in `split_indices.json`.

Overige 40.000 beelden uit de trainingspool en 4.000 uit de validatiepool blijven
in deze pilot ongebruikt. Er worden geen voorbeelden op basis van loss verwijderd.
Ontbrekende/corrupte data veroorzaken een fout, geen automatische uitsluiting.

## Omvang en uitvoering

- 3 epochs op DEZELFDE 5.000 trainingsbeelden, met nieuwe gesede augmentaties per epoch.
- Batchgrootte 32, drop_last=False: 157 batches/epoch, inclusief laatste batch van 8.
- 471 optimizerstappen totaal, elk met 5 forward/backward-uitvoeringen.
- Een nieuwe modelinitialisatie met seed 20261009; geen pretrained/smoke-checkpoint.
- Volledige pilot wordt een keer uitgevoerd. Geen statistische multi-seedanalyse.
- Seeds voor Python, NumPy en PyTorch; aparte DataLoader-generator.
- Augmentatieseed per epoch: 20261009+1000+epoch; shuffle-generator: die seed+100000.
- Twee CPU-threads; num_workers=0; pin_memory=False; geen AMP, TF32 of torch.compile.
- CUDA expliciet cuda:0; geen automatische CPU-fallback.
- Deterministische algoritmen verplicht; fout in plaats van stilzwijgende afwijking.
- CPU/GPU/platformen/verschillende bibliotheekversies hoeven niet identiek te zijn.
- De moderne torchvision-transformaties implementeren het oude recept; historische
  bit-identieke beeldtransformaties met torchvision 0.2.1 zijn NIET bewezen.

## Metingen

Validatie bij epoch0 (voor training) en na elk epoch, op dezelfde ongeaugmenteerde
validatiesubset in model.eval(). Geen aanvullende BatchNorm-kalibratie.
Per precisie: top1/top5-accuracy en hard-label cross-entropy.
Trainingsmetrics zijn batchgroottegewogen gemiddelden uit de trainingmodus;
ze gebruiken geaugmenteerde beelden en een model dat onderweg verandert.

`objective_loss` is de daadwerkelijk geoptimaliseerde loss. Bij lagere precisies is
dat soft-target/distillatieloss; bij 32 bit en alle validaties hard-label CE.
`hard_ce` wordt extra gelogd tegen de echte labels, maar deze extra diagnostiek
wordt bij lagere precisies NIET teruggepropageerd. Vergelijk de trainingsobjectieven
van verschillende precisies dus niet alsof ze dezelfde targets gebruiken.

Geen vereiste accuracystijging voor PASS: PASS betekent uitvoering/consistentie,
niet voldoende wetenschappelijke modelkwaliteit. Een slechte leercurve blijft een
te analyseren pilotuitkomst, niet een reden om samples of resultaten te verbergen.
Tijd/GPU-geheugen worden uitsluitend diagnostisch gemeten, niet als hardwarewinst.

## Checkpoint en herhaalbaarheid

Alleen de laatste epoch wordt opgeslagen; geen selectie van beste validatie-epoch.
`pilot_checkpoint.pt` bevat FP32-mastergewichten, BN, optimizer, scheduler, RNGs,
configuratie, splitschecksum en validatielogits. Alleen dit zelfgemaakte checkpoint
wordt meteen met weights_only=True opnieuw geladen. Model, optimizer, scheduler,
validatielogits en metrics worden exact vergeleken.
Volledige trainingshervatting is NIET geimplementeerd of getest.

De volledige pilot wordt niet automatisch tweemaal uitgevoerd. Voor latere controle
worden modelhashes, selectie, samplevolgorde en hashes van geaugmenteerde minibatches
opgeslagen. Een nieuwe aanroep krijgt een nieuwe rapportmap en overschrijft niets.

## Lokale ontwikkeltest

`--profile local-check` gebruikt dezelfde code/model/quantizers, maar slechts 20
trainings- en 20 validatiebeelden en 1 epoch met batch2. Dit is GEEN pilotresultaat.
De meegeleverde offline validatietest gebruikt bovendien expliciet kunstmatige
CIFAR-vormige batchbestanden wanneer internet in de ontwikkelomgeving ontbreekt.
Bekijk `local_validation/VALIDATION.md` voor precies wat wel/niet uitgevoerd is.

## Primaire externe documentatie

- Officiele CIFAR-10-pagina: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch 2.10 reproducibility: https://docs.pytorch.org/docs/2.10/notes/randomness.html
- Python archive handling: https://docs.python.org/3.13/library/tarfile.html

De daadwerkelijke researchmethoden hierboven zijn ontleend aan de AANGELEVERDE
code, niet vervangen door een willekeurige nieuwere GitHub-versie.
