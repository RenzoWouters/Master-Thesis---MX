# Protocol: CNN-8 / SVHN train-only / 100 epochs / no MX

## 1. Bron en bewuste afwijking

Bron: Any-Precision Deep Neural Networks, aangeleverde PDF, p. 5,
Implementation Details; p. 6 (gedrukte p. 10768), tabel 1, CNN-8, Ours.
De paper beschrijft honderd SVHN-epochs, train+extra, Adam zonder weight decay,
lr 0.001, daling met factor 0.1 bij epochs 50/75/90, vijf precisies 1/2/4/8/32
met recursieve distillatie en niet-gekwantiseerde eerste/laatste laag.

Deze afgesproken ontwikkelreferentie gebruikt uitsluitend de volledige originele
train-split. De extra-split wordt niet gelezen, gehasht, gedownload of gewijzigd.
Dit is een afwijking van het data- en updatebudget in de paper, geen exacte
paperreproductie. De historische auteursseed, omgeving en volledige provenance
van iedere gepubliceerde score zijn niet vastgesteld.

## 2. Uitvoerbaar auteursrecept

Gebaseerd op de reeds aangeleverde Any-Precision-DNNs-master(3).zip; geen nieuwe
GitHub-versie ophalen. Archief-SHA256:
7e36c5f039a78e3174421de7ff097066f95dde2837ea02f11913f3be0ba7f73c

- model: svhnq, factory expand=2; 132954 parameters inclusief vijf BN-banken;
- dataset: train=73257, test=26032, extra=False, geen nieuwe validatiesplit;
- preprocessing: oorspronkelijke Resize(40), ToTensor, mean=std=(0.5,0.5,0.5);
- batch=128, drop_last=False, workers=0, train shuffle=True, test shuffle=False;
- pin_memory=True, cudnn.benchmark=True, geen AMP of torch.compile;
- Adam, lr=0.001, weight_decay=0, honderd volledige epochs;
- oorspronkelijke lr_scheduler, factor 0.1 na epochs 50, 75 en 90;
- precisies 1,2,4,8,32 voor gewichten en activaties;
- training 32 -> 8 -> 4 -> 2 -> 1; CE op de hoogste precisie, vervolgens
  recursieve soft-target CE met detached teacher, gradientaccumulatie en
  eenmaal optimizer.step per minibatch;
- test 1 -> 2 -> 4 -> 8 -> 32 na IEDERE epoch, zonder gradients of BN-updates;
- scheduler.step na train en test, zoals de aangeleverde train.py;
- eerste/laatste laag, dropout, per-precisie BN, tanh/normalisatie in de
  gewichtsvoorbewerking en oorspronkelijke activatiekwantisator blijven intact.

Het W32-pad is ook hier het oorspronkelijke auteurscodepad met zijn
gewichtsvoorbewerking, niet een nieuw ingevoegde gewone FP32-Conv2d.
Er is geen MX, geen mantissareductie en geen nieuwe truncatiekwantisator.

## 3. Epochnummering en LR

De opgeslagen epochnummers zijn 1 t/m 100; de UPSTREAM-log telt vanaf 0.
Gebruikte LR, niet de LR die pas voor de volgende epoch wordt klaargezet:

| Epochs | LR gebruikt tijdens training |
|---|---:|
| 1-50 | 0.001 |
| 51-75 | 0.0001 |
| 76-90 | 0.00001 |
| 91-100 | 0.000001 |

De runner verifieert de LR voor iedere epoch en de schedulerstand na iedere
schedulerstap. De eindexport controleert ook lr_used en lr_next voor alle epochs.
Er zijn 573 minibatches per epoch, dus 57300 optimizerstappen na honderd epochs.
Dit is niet gelijk aan de 472200 updates van het volledige train+extra-recept.

## 4. Initialisatie, logging en hervatten

Een nieuwe `run` initialiseert opnieuw met dezelfde lokale seed 20261009 als de
succesvolle proef. Geen proef-, ResNet- of train+extra-checkpoint wordt geladen.
De historische auteursseed is onbekend. De seed is geen belofte van bitexact
identieke GPU-resultaten; het bestaande cuDNN-backendbeleid blijft behouden.

De volledige epochlus is AST-identiek aan de geslaagde train-only-pilot.
svhn_support.py, resume_core.py en smoke_support.py zijn byte-identiek.
SOURCE_MANIFEST.json, FULLRUN_CHANGE_MANIFEST.json en fullrun.patch leggen de
herkomst en de beperkte wrapper-/rapportagewijzigingen vast. De oude
provenance/TRAINONLY_CHANGE_MANIFEST.json beschrijft de eerdere overgang naar
train-only en is een HISTORISCHE pilotaudit, geen huidige vijf-epochlimiet.

Alle epochs krijgen eigen checkpointnamen en checksums. Model, optimizer,
scheduler, BN-banken, Python/NumPy/Torch/CUDA-RNG en historie worden opgeslagen.
Hervatten is alleen een noodvoorziening voor een onderbroken volledige run van
DIT pakket, op dezelfde machine/omgeving en met ongewijzigde instellingen en
bronbestanden. Pilotcheckpoints worden door het contract geweigerd. Er is geen
mid-epoch-herstel, cross-machine-herstel of bitexacte CUDA-hervatgarantie.

Elke uitvoering krijgt een nieuwe resultatenmap; geen bestaande ruwe data of
checkpointbestanden worden overschreven. De uitgebreide logging heeft overhead,
zoals reeds in de vijf-epochrun. Er worden geen dataloader- of logoptimalisaties
in deze referentie ingevoerd.

## 5. Checkpointselectie en eindevaluatie

Selectie: eerste strikte maximum van oorspronkelijke W32/A32-test-top1.
Bij gelijke score blijft het vroegste checkpoint. Alle vijf precisies worden
van hetzelfde beste checkpoint gerapporteerd. Ook de laatste afgeronde epoch
wordt als afzonderlijke rol bewaard; na een complete run is dat epoch 100.
De testscore is niet onafhankelijk van deze checkpointselectie.

De eindexport laadt de twee gekozen checkpoints veilig en vergelijkt hun hashes,
modelstaat en opgeslagen metrics. Dit is geen extra nieuwe forward-evaluatie;
de scores zijn al direct na de bijbehorende epoch op de testset gemeten.
completion_summary.json vermeldt expliciet of alle honderd epochs voltooid zijn.
Een gecontroleerde onderbreking wordt niet als volledige run gerapporteerd.

## 6. Rapportage

Notebook: SVHN_trainonly_reference100_resultaten.ipynb in de runmap. Het leest
uitsluitend opgeslagen JSON/CSV; geen modelcalls, training, downloads of updates.
Werkelijk gemeten totale epochtijd wordt onderscheiden van de extrapolatie en
van sessieduur inclusief opstarten. Paperwaarden blijven literatuurcontext.
Eigen eindaccuracies, convergentie en daadwerkelijke duur zijn pas na de echte
run beschikbaar. Synthetische tests zijn geen onderzoeksresultaten.
