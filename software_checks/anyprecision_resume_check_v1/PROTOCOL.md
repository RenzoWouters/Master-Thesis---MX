# Vastgelegd protocol: hervatcontrole, versie 1

## Doel en reikwijdte

Technische toets: levert een volledige afsluiting van het trainingsproces na een
voltooide mini-epoch, gevolgd door herstel in een nieuw proces, exact dezelfde
vervolgstappen en eindtoestand op als ononderbroken uitvoeren?

Dit is geen nieuwe onderzoekspilot, geen trainingsduurbepaling en geen
referentietraining. De checkpoints uit deze test zijn niet het startpunt voor de
latere 400-epochtraining. De definitieve data-/evaluatiekeuzes daarvan worden
met deze test niet vastgelegd.

## Gelijk gehouden onderdelen uit de aangeleverde implementatie

- resnet20q, expand=5, 6.786.330 parameters inclusief BatchNorm-banken.
- W32/A32 -> W8/A8 -> W4/A4 -> W2/A2 -> W1/A1 per minibatch.
- Originele model-, quantizer-, loss- en optimizerbestanden, zonder wijzigingen.
- FP-mastergewichten; geen persistente lage-precisiecodes of MX.
- Recursieve distillatie en som van gradients voor een gezamenlijke Adam-stap;
  de eerder gebruikte `run_pilot.train_batch()` wordt ongewijzigd hergebruikt.
- Oorspronkelijke CIFAR10-loader en oorspronkelijke beeldtransformaties.

## Kleine testconfiguratie

- 20 trainingsbeelden (2 per klasse), 20 andere controlebeelden (2 per klasse).
- Selectie met de bestaande splitfunctie en seed 20261009; genest in de eerdere
  trainings-/validatiepools. Geen officiele testset-evaluatie.
- Batch 8, laatste batch 4; drie minibatches per mini-epoch.
- Drie mini-epochs; validatie op alle precisies na iedere mini-epoch.
- Adam, lr=0.001, weight_decay=0.
- ARTIFICIELE testmilestones [1,2], gamma 0.1, scheduler.step na validatie.
  Learning rates voor mini-epochs 1/2/3: 0.001 / 0.0001 / 0.00001.
  Het checkpoint wordt NA de schedulerstap opgeslagen. Daardoor testen we
  herstel van een al verlaagde learning rate EN een latere verlaging.
- workers=0, geen AMP, geen TF32, geen torch.compile, deterministische
  bewerkingen met fouten in plaats van stilzwijgende terugval.
- Alleen CPU of een expliciet geselecteerde CUDA-GPU, dezelfde omgeving per pad.

De milestones [1,2] zijn een testprikkel, GEEN wijziging van het later geplande
400-epochschema. Het model, de optimizer en de RNGs beginnen opnieuw voor pad A
en pad B; de tweede helft van B begint uitsluitend vanuit het testcheckpoint.

## Wat wordt vastgelegd en hersteld?

Modelstate inclusief alle BatchNorm-buffers; Adam-state inclusief momenten en
staptellers; MultiStepLR-state en huidige learning rate; voltooide epoch en
optimizerstappen; RNG-state voor Python, NumPy, PyTorch CPU en (bij CUDA)
CUDA; afzonderlijke RNG-state van trainings- en validatie-DataLoaders; selecties,
configuratie, versies en broncodecontrolegetallen. De meetgeschiedenis en de
validatielogits worden in deze kleine test eveneens opgeslagen.

Laden gebeurt op CPU met `weights_only=True`, gevolgd door laden in vooraf
geconstrueerd model/optimizer/scheduler. De scheduler is geconstrueerd VOOR het
laden van de optimizerstate. De toevalstoestanden worden als laatste hersteld.
Een afwijkend contract of een corrupte/onvolledige checkpointfile wordt geweigerd.

Voor iedere nieuwe epoch worden dezelfde bestaande loaders opnieuw geitereerd.
Er is GEEN halverwege afgebroken iterator. Alle beelden van een voltooide epoch
zijn gebruikt; het nieuwe proces start een nieuwe iterator voor de volgende epoch.
Er zijn geen workerprocessen of niet-opgeslagen prefetchqueues.

## RNG-keuze versus de vorige pilot

De pilot initialiseert globale RNGs en een trainingsgenerator opnieuw per epoch.
Deze nieuwe hervatcode gebruikt doorlopende RNG-streams en herstelt de streams
uit het checkpoint. Dat is een expliciete keuze voor de uitvoer-/hervatlogica,
niet een wijziging van quantizer of loss. Oude pilotresultaten worden daardoor
niet geclaimd als bit-identieke beginsituatie van deze nieuwe test.

Diagnostische RNG-probes trekken waarden uit iedere stream en zetten de volledige
state daarna direct terug. Zij veranderen de training niet. In het hervatte
proces worden alle RNGs bewust eerst anders ingesteld; correct herstel mag dus
niet toevallig afhangen van opnieuw dezelfde beginseed kiezen.

## Slaagcriterium

Exact gelijke tensorwaarden, vormen en dtypes, niet slechts ongeveer gelijke
accuracies. Zowel het checkpoint na mini-epoch 1 als de eindtoestand na mini-epoch
3 worden vergeleken. Controle omvat model/BatchNorm, optimizer, scheduler, RNGs,
volledige batchhistorie met indices en SHA-256 van getransformeerde invoer,
losses, metrics en validatielogits voor alle precisies. Ook de nieuwe
optimizerstappen na laden en beide learning-rateovergangen worden gecontroleerd.

De bytes van twee `.pt`-archieven hoeven niet identiek te zijn: metadata van
serialisatie kan verschillen. Hun geladen tensorwaarden en inhoud moeten dat wel.
Een ongelijke toestand geeft FAIL, met de paden naar afwijkende componenten.

## Opslag en beperkingen

Per voltooide mini-epoch een nieuw genummerd checkpoint: totaal zes bestanden.
Eerst een nieuw tijdelijk bestand, flush/fsync, daarna publiceren en checksum
schrijven. Bestaande checkpoints worden niet overschreven. Een afgebroken
schrijfactie wordt niet als een bruikbaar voltooid checkpoint geaccepteerd.
Dit is geen absolute garantie tegen schijffouten of stroomverlies.

Niet getest: mid-epoch-herstel, de fysieke stroomonderbreking van de laptop,
andere GPU/driver/PyTorch-versie, gedistribueerde training, meerdere DataLoader-
workers, volledige lange training, paperaccuracy, MX, of hardwarewinst.
Een latere referentierunner moet deze checkpointmodule hergebruiken en ook de
integratie ermee controleren; een PASS hier valideert geen nog niet geschreven code.

## Primaire implementatiedocumentatie

PyTorch 2.10, Reproducibility:
https://docs.pytorch.org/docs/2.10/notes/randomness.html

PyTorch, Saving and Loading Models (general checkpoint):
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

PyTorch, Optimizer.load_state_dict (scheduler/optimizer restoration order):
https://docs.pytorch.org/docs/main/generated/torch.optim.Optimizer.load_state_dict.html

Deze documentatie onderbouwt de technische API-keuzes; de concrete testomvang
is onze eigen technische controlekeuze, geen protocol uit een onderzoekspaper.
