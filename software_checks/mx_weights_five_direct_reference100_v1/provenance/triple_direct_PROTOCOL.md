# Protocol — gezamenlijke directe E4M3/E2M3/E2M1-referentie, 100 epochs

## Doel en actuele keuze

Een Any-Precision-netwerk met drie MX-gewichtsformaten als directe referentie
voor latere truncatieproeven. Dit volgt de gebruikerskeuze van 23 oktober 2026:
alle drie precisies eerst rechtstreeks uit de gemeenschappelijke FP32-bron,
zonder de eerder geteste mantissatruncatie. De vijf-epochpilot is geslaagd.
Dit afzonderlijke pakket voert de volledige honderd-epochreferentie uit,
met dezelfde numerieke kern en trainingsstap. Het begint met een nieuwe
initialisatie met dezelfde seed; het pilotcheckpoint wordt niet ingeladen.

## Exacte gewichtsconversie

Voor iedere MX-convolutie wordt dezelfde eerdere voorbewerking gebruikt:

`G(W) = tanh(W) / max(abs(tanh(W))) * mean(abs(W)).detach()`.

Bij een volledig nulgewicht wordt de eerder ingevoerde nulguard behouden.
W is de gedeelde FP32-updatemaster. Voor elke modus gebeurt vervolgens een
onafhankelijke directe quantisatie: `Q_E4M3(G(W))`, `Q_E2M3(G(W))` of
`Q_E2M1(G(W))`. Geen van deze routes ontvangt de codes of gedecodeerde
gewichten van een andere precisie. Er is geen broncache met truncatie.

| Formaat | Elementbits | Lokale exponentbias | Grootste eindige magnitude | Schaalexponent bij niet-nulblok |
|---|---|---:|---:|---|
| E4M3 | `s eeee mmm` | 7 | 448 | `floor(log2(amax)) - 8` |
| E2M3 | `s ee mmm` | 1 | 7,5 | `floor(log2(amax)) - 2` |
| E2M1 | `s ee m` | 1 | 6 | `floor(log2(amax)) - 2` |

Blokgrootte 32, binnen ieder outputkanaal over afgevlakte Cin*Kh*Kw;
positieve nulpadding aan de staart. De schaal is E8M0 met exponentclipping
naar [-127,127]; nulblok krijgt schaal 1. De schalen worden per forward
uit de actuele G(W) bepaald. E2M3 en E2M1 krijgen bij dezelfde bron dezelfde
schalen; E4M3 gebruikt zijn eigen formaatregel. Dit is gewone directe
MX-schaalkeuze, geen schaalcompensatie of geleerde schaal.

Alle codecs gebruiken nearest rounding met ties-to-even, eindige saturatie,
lokale subnormals en getekende nul. Niet-eindige input wordt geweigerd;
FP32-invoersubnormals worden naar getekende nul geflusht. E4M3 gebruikt de
werkende exact-binary-v2-codec uit de geslaagde dedicated proef. De twee
bestaande E2-codecs blijven byte-identiek aan de eerdere directe proef.
Identity-STE door de quantizer; de oorspronkelijke afgeleide van G(W) blijft.

Deze proef voert geen mantissabitshift, exponentbitshift, schaaltruncatie,
schaalcorrectie of schaaltraining uit.

## Model, activaties en BatchNorm

Any-Precision CNN-8 (`svhnq`, expand=2), dezelfde 132.954 parameters inclusief
de vijf bewaarde BN-banken per BN-laag. MX vervangt alleen de gewichten van
`layers.3`, `.6`, `.10`, `.13`, `.16` en `.20`; eerste convolutie, laatste
lineaire laag en overige modelstaat behouden het eerdere beleid.

| Gewichtsmodus | Interne modus-ID | BN-opslagsleutel | Activatiebeleid |
|---|---:|---:|---|
| E4M3 | 8 | `8` | A32 |
| E2M3 | 6 | `32` | A32 |
| E2M1 | 4 | `4` | A32 |

De bestaande E2M3- en E2M1-bankmapping blijft behouden; E4M3 gebruikt een
voorheen ongebruikte bank. De sleutels zijn opslag-ID's en bepalen geen
activatieprecisie. Alle `abit`- en `wbit`-selectors blijven 32; een aparte
gewichtsmodus kiest de MX-codec en de bijbehorende BN-bank. Iedere actieve
bank krijgt één update per trainingsminibatch. Banken 1 en 2 blijven ongebruikt,
ongewijzigd en zonder Adamstaat. Evaluatie verandert geen BN-statistieken.

A32 en FP32 benoemen de tensor- en masterdtypes. Het bestaande backendbeleid
blijft behouden (`cudnn.benchmark=True`, geen deterministische override,
geen AMP/torch.compile, geen gewijzigde TF32-instelling). De convoluties zijn
FP32-tensoroperaties; dit bewijst niet dat elke interne GPU-vermenigvuldiging
een volledige FP32-mantissa gebruikt. Er zijn geen native MX-kernels en geen
gemeten hardwareversnelling of energiewinst.

## Oorspronkelijke recursieve Any-Precision-training

De oorspronkelijke forwardfunctie uit de auteurscode blijft inhoudelijk
behouden, met uitsluitend drie vervangen selectorparen. Modi worden aan die
functie meegegeven als `4,6,8`.

Per trainingsminibatch:

1. Gradients nulzetten.
2. E4M3 forward en harde cross-entropy met de labels; backward.
3. E2M3 forward en soft-target CE met `softmax(E4M3_logits.detach())`; backward.
4. E2M1 forward en soft-target CE met `softmax(E2M3_logits.detach())`; backward.
5. Eén Adam-update op de gesommeerde gradients.

Temperatuur 1; geen extra harde studentloss, geen deling door drie en geen
extra FP32-teacher. Dropout gebruikt opeenvolgende RNG-trekkingen zoals in de
oorspronkelijke lus. De twee teacheroutputs zijn losgekoppeld van hun graph.
Distillatie van outputs is onafhankelijk van de directe gewichtsconversie.

Bij evaluatie is de modusvolgorde E2M1, E2M3, E4M3 en is de CE voor iedere
modus hard-label CE. De notebook onderscheidt deze evaluatieloss van de
soft-target trainingsobjectieven.

## Data, optimizer en eindpunt

SVHN train-only: 73.257 trainbeelden en 26.032 officiële testbeelden;
geen extra-split of nieuwe validatiesplit. Oorspronkelijke Resize(40),
ToTensor en Normalize(mean=std=0,5). Batch 128, workers 0, pin_memory=True,
drop_last=False; train shuffelt en test niet. Seed 20261009, nieuwe
initialisatie; geen checkpoint van een voorgaande proef wordt ingelezen.

Adam, lr 0,001, weight_decay 0; FP32-masters, gradients en Adam-momenten.
Het bestaande schema is lr 0,001 tot en met epoch 50, 0,0001 voor 51–75,
0,00001 voor 76–90 en 0,000001 voor 91–100. De scheduler wordt na de train-
en testfase bijgewerkt. Dit pakket heeft een vast eindpunt op epoch 100.

Per epoch: 573 Adam-updates, 1.719 trainingspasses en 204 testminibatches
met telkens drie modi. Een voltooide referentie heeft 57.300 updates,
171.900 trainingspasses en 114.600 gecontroleerde softteacher-aanroepen.
De eerste looptijdraming komt uit de gemeten pilot; er is geen eindaccuracy-
voorspelling. Na voltooiing bevat measured_100_epoch_seconds de som van alle
100 gemeten epochtijden, ook bij hervatten. session_wall_seconds betreft
alleen de huidige sessie. Een eerder gestopte run heet
STOPPED_AT_EPOCH_BOUNDARY; alleen een voltooide training krijgt PASS.
Resume vereist exact hetzelfde volledige referentiecontract en gaat tot
epoch 100. Pilotcheckpoints en reeds voltooide epoch-100-checkpoints worden
geweigerd. Elke hervatting schrijft een nieuwe resultaatmap; behoud ook
de eerdere runmappen waarnaar de herstelde checkpointgeschiedenis verwijst.

## Checkpoints, exports en notebook

Best is het eerste strikte maximum van E4M3-testaccuracy. Alle drie modi
worden op dat ene checkpoint gerapporteerd. Last is de laatste volledig
afgeronde epoch, bij een voltooide referentie epoch 100. De notebook toont
als hoofdfiguur epoch 100 en E4M3-best, telkens met alle drie precisies.
Bij een eerder gestopte run blijft het epoch-100-paneel expliciet leeg; de
laatste beschikbare epoch staat dan alleen als zodanig in de tabel.
Best wordt geselecteerd op de officiële testset; dat is geen onafhankelijke
testschatting na selectie.

Per epoch blijft een volledig checkpoint met FP32-masters, optimizer,
scheduler, RNG en contract bewaard. De directe referentie exporteert voor
best en last drie afzonderlijke directgequantiseerde modellen. Dat is nog
geen truncatable bronmodel. Elke compacte export bevat elementcodes,
E8M0-schalen en overige benodigde modelstaat, zonder de zes MX-masters.
Encoding en de exacte gewichtsvergelijking gebeuren op het trainingsdevice.
De rapportage zegt expliciet dat geen volledige extra testsetpass vanuit de
export wordt uitgevoerd; de korte check test exportlogits op synthetische input.

Nieuwe uitvoermappen behouden oude resultaten. Het pakket bevat alleen een
uitvoerbare notebooktemplate; er wordt geen tweede uitgevoerde notebook
geleverd. De runner vult de nieuwe resultaatmap automatisch in.
