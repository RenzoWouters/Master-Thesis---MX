# Protocol - korte eenmodus-MX-trainingsproef

## Onderzoeksvraag en afbakening

Leert CNN-8 vanaf initialisatie wanneer de zes effectieve gewichtsblokken tijdens
iedere forward naar MXFP6 E2M3 worden gekwantiseerd, terwijl A32 en FP32-updatemasters
behouden blijven? Vijf volledige train-only-epochs controleren het leermechanisme,
logging, updates en runtime. Ze bewijzen geen eindconvergentie of truncatierobuustheid.

Dit is onze onderzoeksvariant, geen exacte Any-Precision- of MX-paperreproductie.
Any-Precision gebruikt meerdere gekoppelde W/A-modi; de MX-paper beschrijft ook
andere MX-operanden. De oude vijf-modereferentie is dus alleen context.

## Vastgelegd

- CNN-8 / svhnq, expand=2; bestaande architectuur inclusief alle vijf BN-banken.
- SVHN train=73257, test=26032; geen extra of nieuwe validatiesplit.
- Oorspronkelijke Resize(40), ToTensor, Normalize(mean=std=0.5).
- Batch128, drop_last=False, workers0, shuffle train, vaste testvolgorde.
- Adam uit de auteurscode, lr0.001, weight_decay0, alle overige optimizerdefaults
  uit dezelfde bestaande PyTorch 2.10.0-omgeving, niet opnieuw ingesteld.
- 100-epoch LR-planning (50,75,90, factor0.1) met expliciete pilotstop op epoch5.
- Seed20261009; nieuwe initialisatie, geen oude getrainde gewichten.
- Een forward/backward en een optimizerstep per minibatch; hard-label CE met
  gemiddelde reductie, geen lossdeling door vijf, geen distillatie/teacher.
- A32, oorspronkelijke ReLU, BN32 leert gedurende training. Andere banken blijven
  ongebruikt zonder gradient of Adamstaat. Evaluatie werkt geen BN-statistieken bij.
- Eerste Conv2d, laatste Linear en BN-parameters FP32, zoals de eerdere laagkeuze.
- De zes doellagen gebruiken Q_MX(G(W)); G(W) is de bestaande tanh/normalisatie.
- MX: teken1/exponent2/mantissa3, E8M0 schaal, blok32 langs Cin*Kh*Kw per Cout,
  zero padding, dezelfde nearest-even codec en dezelfde grensregels als stap1.
- Schalen/codes worden uit de actuele G(W) opnieuw berekend; niet geleerd als
  afzonderlijke parameters en niet vastgezet. Ze mogen veranderen na een update.
- STE: exact MX forward, identity-gradient door de MX-stap, oorspronkelijke
  gradients door G(W), detached bestaande factor E. Ook bij saturatie identity.
- Masterparameters, gradients en Adam-momenten FP32. Convolutie FP32-emulatie;
  de leesbare codec gebruikt FP64-intermediairen. Geen native MX-kernelclaim.
- Geen truncatie, activatie-/gradientkwantisatie, AMP, torch.compile of nieuw LR-schema.
- Origineel backendbeleid cudnn.benchmark=True, geen deterministische override.

## Optionele aparte controle

`--mode control` gebruikt dezelfde wrapper zonder MX. Zelfde initiatietoestand,
loss, optimizer, aantal updates, samplevolgorde en A32-beleid worden vastgelegd
voor vergelijking. Dit is een andere run, NIET gezamenlijke training in twee modi.
Het notebook vergelijkt hashes/configuratie alvorens beide runs samen te tonen.
Het eerdere test-geselecteerde checkpoint54 is niet de controle voor deze proef.

## Rapportage en integriteit

Per epoch: train/test CE en accuracy, gewogen upstream scores en correcte aantallen,
leerbare masterwijzigingen per laag, veranderde MX-codes/schalen, gradientchecks voor
ELKE update, FP32-Adamstaat, BN-tellers en eenmalige dekking van alle voorbeelden.
Niet-eindige waarden leiden tot een fout; geen stilzwijgend overslaan van batches.

Selectie: eerste strikte maximum van test-top1 van de actieve modus in deze run.
Dat is MXFP6/A32 voor de MX-run, W32/A32 voor control. Beste en laatste blijven
apart bewaard. Deze testscore is niet onafhankelijk van checkpointselectie.
Vijf-epochselectie is geen eindresultaat van honderd epochs.

Checkpoint = volledige FP32-trainingsstaat voor herstel. MX-export = codes/schalen
voor inferentie; geen hervatbare Adam/masterstaat. Alle eerste/laatste lagen en
BN-banken blijven apart aanwezig en tellen mee. Export is geen bewering dat de
hele trainingsstaat in MX staat. Geen herkalibratie na training.

Alle bestanden krijgen nieuwe mappen; ruwe datasets en eerdere onderzoeksresultaten
blijven ongewijzigd. Pakket- en bronhashes, omgeving en volledig hervatcontract
worden geregistreerd. De extra registratie heeft overhead en zit in de tijdmeting.
De tijdsraming van deze proef is geen garantie voor honderd epochs of hardwarewinst.

## Bronnen

- Aangeleverde Any-Precision-DNNs-master(3).zip: train.py:152-221,
  models/quan_ops.py:87-99,132-134; models/svhn_quan.py; optimizer.py.
- Any-Precision Deep Neural Networks, project-PDF p.5: algoritme en trainingsrecept.
- Microscaling Data Formats for Deep Learning, project-PDF p.3: tabel1/Algorithm1;
  p.4: FP32-masterkopie. Onze proef kwantiseert alleen gewichten.
- mx_weights_step1_v1: identiek hergebruik van mx_format.py en mx_adapter.py.
- PyTorch optimizer hooks: https://docs.pytorch.org/docs/main/optim.html
  Documentatie kan een nieuwere versie beschrijven; de hier gebruikte hooks zijn
  daadwerkelijk lokaal met PyTorch 2.10.0+cpu getest. Productie blijft 2.10.0+cu128.
