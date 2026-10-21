# Protocol - twee afzonderlijke eenmodustrainingen van 100 epochs

## Afbakening

Vastgelegde vergelijking: W32_effective/A32 versus MXFP6 E2M3/A32 bij gelijke
initialisatie, data, preprocessing, loss, optimizer, LR-schema en updatebudget.
Beide starten opnieuw; geen pilot- of bestaand referentiecheckpoint inladen.
Dit is geen gezamenlijk tweemodusmodel en geen exacte Any-Precision-paperreproductie.
De dataset is uitsluitend SVHN train, niet train+extra. De eerdere vijf-precisietraining
is context, geen gematchte controle. Er is geen truncatie of nieuwe activatieprecisie.

## Instellingen (overgenomen uit de succesvolle vijf-epochpilot)

- CNN-8/svhnq, expand=2; oorspronkelijke eerste/laatste laag en dropout.
- Train 73257, test 26032; geen extra, geen nieuwe validatiesplit.
- Resize(40), ToTensor en Normalize(mean=std=0.5).
- Batch 128, drop_last=False, workers=0; train shuffle, test vaste volgorde.
- Seed 20261009; dezelfde initialisatiehash en train/test-volgordehashes worden bewaard.
- Adam via de auteurscode: lr=.001, weight_decay=0; dezelfde overige defaults.
- 100 volledige epochs; 573 updates per epoch, 57300 in totaal per model.
- Scheduler factor .1 NA epochs 50,75,90; lr gebruikt: .001/.0001/.00001/.000001.
- Een forward/backward en een optimizerstep per minibatch, hard-label cross-entropy.
- Geen teacher, distillatie, lossdeling door vijf of gezamenlijk gradientdoel.
- A32/ReLU en BN32; BN32 leert in training, niet tijdens evaluatie.
- Andere BN-banken blijven aanwezig maar ongebruikt, zonder gradient/Adamstaat.
- FP32-updatemaster W; bestaande effectieve gewichten G(W)=E*tanh(W)/max(abs(tanh(W))),
  E=mean(abs(W)).detach(). Geen nieuwe parametrisatie of verwijdering van G.
- Control: G(W) naar convolutie, MX-vlag uit. MX: Q_MX(G(W)) naar convolutie.
- MX: dezelfde zes lagen layers.3/6/10/13/16/20; E2M3, E8M0, blok32 per Cout
  langs afgevlakte Cin*Kh*Kw, dezelfde zero padding en nearest-even regels.
- Schalen en codes worden uit actuele G(W) opnieuw bepaald; geen geleerde schaalparameter.
- Dezelfde identity-STE; gewone gradients door G(W). Gradients en Adam-momenten FP32.
- Kern blijft FP32 F.conv2d; codec heeft FP64-intermediairen. Geen native MX-rekenclaim.
- cudnn.benchmark=True zoals de pilot, geen deterministische override, geen AMP/compile.

## Selectie en onderzoeksevaluatie

Per epoch complete train- en testevaluatie van de actieve modus. Beste checkpoint:
eerste strikte maximum van test-top1 in DIE run (W32/A32 of MX E2M3/A32).
Ook epoch 100 apart bewaren. De testscore is niet onafhankelijk van checkpointselectie.
Voor het gepaarde eindresultaat epoch 100 tegenover epoch 100; afzonderlijk geselecteerde
beste checkpoints zijn aanvullende resultaten en kunnen verschillende epochs hebben.
Eerst gelijke configuratie, init- en alle volgordehashes en softwareomgeving controleren.
Een run per modus/seed bewijst geen statistische gelijkwaardigheid. Niet bitexact CUDA.

## Wat wijzigt ten opzichte van de pilot?

Alleen het eindpunt (5 -> 100), protocol/schema, nieuwe outputmap en de eindrapportage.
De epochlus is AST-identiek. single_support.py, de codec/STE, resume_core.py,
svhn_support.py, smoke_support.py, smoke_single.py, test_mx.py en het auteursarchief
zijn byte-identiek. Nieuwe expliciete exportcontroles verifiieren ook het volgende LR.
Geen nieuwe compatibiliteitsaanpassingen; eerdere patch blijft behouden.
De exacte herkomst staat in TRAINING_CHANGE_MANIFEST.json en fullrun.patch.

## Opslag, integriteit en uitvoering

Alle 100 checkpoints met FP32-model, Adam, scheduler, RNG, trace en checksums bewaren.
MX-best/last-export is afzonderlijke inferentiestaat, zonder zes FP32-mastertensors,
maar met overige FP32-lagen/BN. Exacte decoded-weight-test; geen extra volledige testpass.
Logging controleert iedere gradientupdate, eindige waarden, Adamtellers en BNtellers.
Iedere uitvoering heeft een nieuwe map. Geen overschrijven van raw data of oude runs.
Hervatten alleen vanaf een complete epoch van dit pakket/dezelfde modus/omgeving.

Tijdsrapport: sum(total_epoch_seconds) over 100 is werkelijk gemeten; afzonderlijke
sessieduur omvat extra voorbereiding en eindexport. Extrapolatie apart houden.
Notebooks lezen opgeslagen JSON/CSV en starten nooit opnieuw training.
Controle en MX achter elkaar uitvoeren, ieder met een eigen opdracht.

## Bronnen

- Geslaagde mx_weights_single_pilot5_v1.zip; protocol en code opgeslagen in provenance/pilot5.
- Aangeleverde Any-Precision-DNNs-master(3).zip: train.py, models/quan_ops.py,
  models/svhn_quan.py en optimizer.py. Oorspronkelijke forward-AST wordt gebruikt.
- Any-Precision Deep Neural Networks, PDF p.5: trainingsrecept en algoritme;
  onze single-mode en train-only-keuzes zijn expliciete afwijkingen.
- Microscaling Data Formats for Deep Learning, PDF p.3 tabel1/Algorithm1 en p.4
  FP32-master. De volledige compute-flow uit die paper is hier niet gereproduceerd.
