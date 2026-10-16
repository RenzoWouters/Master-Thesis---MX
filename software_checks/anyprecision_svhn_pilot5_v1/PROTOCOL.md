# Protocol en bronafbakening

## 1. Wetenschappelijke bron

*Any-Precision Deep Neural Networks*, aangeleverde PDF, p. 5, Implementation
Details: CNN-8 op SVHN; train plus extra; 100 epochs; Adam zonder weight decay;
lr 0,001 en factoren 0,1 bij 50/75/90; precisies 1/2/4/8/32; recursieve KD;
eerste en laatste laag niet gekwantiseerd. P. 6 (gedrukte p. 10768), tabel 1,
CNN-8, Ours: 88,21 / 94,94 / 96,19 / 96,22 / 96,29 procent.
De kolommen Dedi. zijn NIET de hoofdvergelijking voor een any-precisionmodel.

De paper specificeert niet alle uitvoeringsdetails, de historische seed of de
exacte checkpointselectie voor ieder gepubliceerd getal. Dit pakket vult die
leemten niet met de bewering dat de volledige historische run is bewezen.

## 2. Aangeleverde auteurscode als uitvoerbaar recept

Bron: `Any-Precision-DNNs-master(3).zip`, SHA-256
`7e36c5f039a78e3174421de7ff097066f95dde2837ea02f11913f3be0ba7f73c`.
Er wordt geen nieuwe GitHub-versie opgehaald.

- `train_svhn.sh`: svhnq, svhn, train_extra, Adam, 100 epochs, lr 0.001,
  weight decay 0, lr_decay 50/75/90, bits 1/2/4/8/32.
- `train.py`: batch 128 en workers 0; train shuffle=True, test shuffle=False;
  pin_memory=True; cudnn.benchmark=True; per epoch train, test, scheduler,
  beste checkpoint bepalen op `val_prec1[-1]` met strikt `>`.
- `models/svhn_quan.py`: factory `svhnq()` gebruikt expand=2, niet de
  class-default expand=8. Dropout(0.5), BatchNorm en alle lagen blijven intact.
- `datasets/data.py`: Resize(40), ToTensor en Normalize([0.5]*3,[0.5]*3).
  `val` wordt bij SVHN naar de officiele `test` vertaald. Geen random crop/flip.
- `datasets/svhn.py`: originele loader met train gevolgd door extra, label 10
  omgezet naar 0. Geen `auto_quan`-split en geen eigen vervanging door een
  andere datasetklasse. Data worden lokaal op aantallen en hashes gecontroleerd.
- `models/quan_ops.py`, `models/losses.py`, `optimizer.py`: ongewijzigd.
  Ook de tanh/normalisatie van W32 blijft; W32 is hier geen nieuw ingevoegde
  gewone onbewerkte FP32-Conv2d.

De functie `forward()` wordt direct uit de syntaxisboom van de originele
`train.py` geladen, zonder wijziging van haar functietekst of syntaxisboom.
Dit vermijdt de buitenste CLI, GPU-autoselectie en tensorboardX-import; het is
GEEN herimplementatie van de loss, gradientaccumulatie of precisievolgorde.
De training doorloopt 32 -> 8 -> 4 -> 2 -> 1, gebruikt telkens de losgekoppelde
voorspelling van de vorige modus als teacher, en doet daarna een optimizerstap.
Evaluatie doorloopt 1 -> 2 -> 4 -> 8 -> 32 en gebruikt `torch.no_grad()`.

## 3. Expliciete proef- en infrastructuurkeuzes

De ENIGE inhoudelijke verkorting is de afgesproken stop na vijf volledige
epochs in plaats van honderd. LR-mijlpalen worden niet vervroegd. Er is geen
prestatiegestuurde vroegtijdige stop, nieuwe split, nieuw activatiebeleid, MX
of truncatie. Vijf-epochresultaten zijn geen gelijkbudget-paperreproductie.

Toegevoegd voor uitvoering en controle, niet afkomstig uit de paper:

- expliciete lokale seed 20261009 (de historische auteursseed is onbekend);
- expliciete selectie van de aanwezige CUDA-GPU in plaats van gpustat;
- bestaande Windows/Python/PyTorch-omgeving, geen reconstructie van een oude stack;
- JSON/CSV- en notebookrapportage in plaats van tensorboardX;
- read-only registratie van minibatchmetrics en samplevolgorde;
- datasetintegriteitscontroles, eenmalige download naar het gekozen data-root;
- afzonderlijke, onveranderlijke checkpointbestanden en checksums;
- volledige hervatstaat en controles op model/optimizer/scheduler/RNG;
- afzonderlijke rapportage van beste en laatste checkpoint.

Exacte herkomst en benodigde compatibiliteitswijzigingen staan in
COMPATIBILITY.md. De modelbestanden, datasetsemantiek, kwantisator en originele
trainingsfunctie worden niet aangepast. Instrumentatie heeft wel tijdsoverhead;
de tijdschatting geldt voor DEZE runner op dezelfde laptop, niet voor de
historische trainingstijd van de auteurs.

## 4. Backend versus vorige ResNet-runner

Deze versie zet pin_memory=True en cudnn.benchmark=True zoals `train.py`, en
forceert geen deterministische algoritmen. De vorige ResNet-runner gebruikte
strengere deterministische instellingen en pin_memory=False. Die instellingen
worden nu NIET stilzwijgend overgenomen. Er wordt ook geen AMP, torch.compile
of andere versnellingsmethode toegevoegd. PyTorch-backendprecisiedefaults
worden geregistreerd; geen ongedocumenteerde TF32-override.

Daardoor mag een correct herstelde CUDA-run numeriek afwijken van een
ononderbroken run. De toestand wordt wel volledig hersteld; de offline
CPU-hervattest is geen bewijs van bitexacte GPU-gelijkheid.

## 5. Checkpointselectie en interpretatie

Selectie: hoogste oorspronkelijke gewogen W32/A32-testaccuracy. Deze waarde
wordt door de ongewijzigde AverageMeter berekend. De gerapporteerde exacte
correct-aantallen en count_top1_percent zijn aanvullende controles, geen
andere selectiemaatstaf. Bij gelijkheid blijft de eerste epoch behouden.
Alle precisies worden gerapporteerd van datzelfde checkpoint. Last is een
aparte rij/rol van de laatste afgeronde epoch.

De testset wordt niet gebruikt voor gradients of BatchNorm-update, maar WEL
voor checkpointselectie. De aldus geselecteerde testscore is dus niet
onafhankelijk van die selectie. Dat is hier bewust de gekozen auteurscoderegel.
Latere ontwikkeling/afstelling van MX vormt een afzonderlijke onderzoeksfase.
