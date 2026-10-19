# Vastgelegd protocol: MX E2M3 + E2M1 / A32 / FP32-masters / vijf epochs

## Doel en bronafbakening

Onderzoeksvariant van de aangeleverde Any-Precision-auteurscode. De logica van
hoog-naar-laag distillatie, afzonderlijke BatchNorm per modus en geaccumuleerde
gradients blijft behouden. Gewichtsformaten, aantal modi en activatiebeleid
verschillen van de oorspronkelijke vijfmodusrun: geen zuivere alleen-MX-
vergelijking met die historische run. Geen keuze van nieuwe instellingen op
basis van deze pilotaccuracies; de proef toetst implementatie en looptijd.

## Model, data en basisrecept

CNN-8 / svhnq factory expand=2, 132954 parameters inclusief alle oorspronkelijke
vijf BN-banken. Eerste convolutie, eindlaag en BN FP32. De zes tussenliggende
quantiseerbare convoluties gebruiken de nieuwe adapter; overige architectuur,
dropout(0.5), preprocessing Resize(40)/ToTensor/Normalize(mean=std=0.5) behouden.

SVHN: alle 73257 train-beelden, 26032 test-beelden, geen extra-data en geen
nieuwe validatiesplit. Test na iedere epoch bij beide modi. Batch128,
drop_last=False, workers0, train shuffle=True, test shuffle=False,
pin_memory=True. Iedere afbeelding telt eenmaal per epoch in datadekking;
de twee precisiepasses verwerken dezelfde minibatch.

Adam via oorspronkelijke optimizer.py, lr0.001, weight_decay0. Honderd-epoch-
scheduler met factor0.1 na 50/75/90, nu expliciete stop na vijf epochs.
LR blijft dus gedurende de pilot 0.001. Seed20261009; nieuwe initialisatie;
geen oud checkpoint. Oorspronkelijk backendbeleid cudnn.benchmark=True,
geen deterministische override, AMP of torch.compile.

## Gezamenlijke loss en updates

Per minibatch: zero_grad; E2M3 forward; hard CE (mean); backward; detach hoge
logits en softmax(T=1); E2M1 forward; originele CrossEntropyLossSoft; backward;
een optimizer.step. Lossbijdragen worden opgeteld, NIET gemiddeld door twee.
De lage modus krijgt geen extra hard-label-loss en er is geen aparte
FP32-teachermodus. Dropout gebruikt opeenvolgende normale RNG-trekkingen,
zoals de auteurscode; maskers worden niet tussen modi gesynchroniseerd.

Twee gewichtsmodi delen dezelfde FP32-masterparameters. Legacy attributen
abit en wbit blijven overal 32. Een aparte weight_mode-selector kiest
E2M3 of E2M1 en de bijbehorende BatchNorm. Per BN-laag gebruiken we de
bestaande opslagbank '32' voor E2M3 en '4' voor E2M1. '4' is uitsluitend een
bankidentifier, NIET A4. Beide banken leren eenmaal per minibatch, ook hun
learnable affine parameters zijn afzonderlijk. Banken1/2/8 blijven ongebruikt,
onveranderd en zonder Adamstaat. Hun geheugen wordt niet verborgen.

## MX-regels

G(W) = mean(abs(W)).detach() * tanh(W)/max(abs(tanh(W))), met behoud van de
exacte bestaande bewerkingsvolgorde. Bestaande veilige nul-laagafhandeling
blijft; geen nieuwe epsilon op andere lagen. Beide modi quantiseren G(W)
rechtstreeks, de ene gebruikt de andere niet als input.

E2M3: sign1, exponent2 bias1, mantissa3; max7.5; subnormalstap0.125.
E2M1: sign1, exponent2 bias1, mantissa1; max6; subnormalwaarde0.5.
RNE ties-to-even, saturatie, signed zero en lokale subnormals. NaN/Inf in
bronnen wordt met fout geweigerd; scalar FP32-subnormals worden volgens het
reeds gekozen MX-paperconversievoorbeeld naar signed zero gebracht.

Blokken van 32 binnen ieder Cout, langs afgevlakte Cin*Kh*Kw; zero padding
zonder outputkanalen te vermengen. E8M0 shared scale, exponentbias127;
scale_exp=floor(log2(max_abs))-2, exact met frexp berekend; nulblok scale=1;
begrenzing van exponent tot [-127,127] expliciet gerapporteerd.
Omdat emax_elem=2 voor beide modi, zijn de schalen gelijk bij dezelfde W.
Bij toekomstige optimizerupdates mogen codes/schalen veranderen. Geen
getrunceerde of opnieuw afgeronde E2M3-bron als E2M1-input in deze proef.

Forward: exact encode/decode, backward: identity-STE door MX, met daarna de
bestaande G(W)-gradients. Ook bij saturatie identity. Discrete codes en
schalen hebben geen aparte leerbare parameters. Masters, gradients en
Adam-momenten blijven FP32; F.conv2d rekent op gedecodeerde FP32-buffers.

## Opslag en selectie

Alle epochs eigen onveranderlijke checkpointnamen en checksums. Hervatten
hergebruikt de gecontroleerde epochkern met model/optimizer/scheduler/
BN/Python/NumPy/CPU/CUDA-RNG. Contractcontrole voorkomt hervatten met andere
modi, software of data. Modelmodus wordt voor iedere pass expliciet gezet.
Geen bitexacte CUDA-replaybelofte en geen mid-epoch-herstel.

Hoofdselectie: eerste strikte maximum van E2M3/A32-test-top1. Beide modi van
DIT checkpoint, plus beide modi van de laatste epoch. Geen samengestelde rij
met de beste score per individuele precisie. De testscore is niet onafhankelijk
van die selectie. Accuracy en CE worden op test beide tegenover echte labels
berekend. Lage trainings-soft-CE is niet rechtstreeks hoge hard-CE.

MX-exports zijn onafhankelijke direct-quantizationmomentopnamen per modus,
geen truncatieklare enkele bronrepresentatie. De FP32-trainingscheckpoint en
iedere export zijn aparte bestanden. Exportgewichten worden gecontroleerd op
exacte reconstructie; geen extra volledige testpass uit export in deze runner.

## Herkomst

Aangeleverde Any-Precision-DNNs-master(3).zip, SHA256 in SOURCE_MANIFEST.json.
train.py:152-218, vooral 180-212; models/losses.py CrossEntropyLossSoft;
models/quan_ops.py:11-28 en 87-99; models/svhn_quan.py. Geen nieuwe GitHub-versie.

Any-Precision Deep Neural Networks, project-PDF p.5: gezamenlijke training,
recursieve KD en het SVHN-recept. Microscaling Data Formats for Deep Learning,
project-PDF p.3, Table1 en Algorithm1; p.4 FP32-masterkopie. Die paper quantiseert
ook andere operanden; dit pakket doet dat niet.

OCP Microscaling Formats v1.0 (Sep 2023), p.12-13, tabellen4/5: exacte
E2M3/E2M1-codering. Geraadpleegd op 2026-10-19:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Onze keuzes: uitsluitend gewichts-MX, vaste A32, laaguitzonderingen, behouden
G(W), identity-STE, bloklayout, train-only, 5-epochpilot en de concrete
bankmapping. Geen claim dat de bronnen deze combinatie als optimaal aanwijzen.

## Onderhoudsversie v1.1: alleen de voorafgaande zelftest

Het onderzoeksprotocol en de productiebackend blijven onveranderd. De manuele
vergelijking wordt verplicht onder tijdelijke deterministische instellingen
getest, met dezelfde CUDA-toleranties als v1. Een extra herhaling onder het
productiebeleid registreert numerieke verschillen als diagnostiek; structuur,
eindigheid, RNG, BatchNorm en Adam-invarianten moeten ook daar slagen.
De check configureert cuBLAS uitsluitend in zijn eigen proces, voor CUDA-init.
run/resume veranderen deze omgevingsvariabele niet. Zie CHECK_FIX.md.
