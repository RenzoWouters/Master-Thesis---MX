# Protocol: dedicated E2M1/A32, honderd epochs

Deze run vult de ontbrekende vaste vierbitsreferentie aan. Geen belofte dat dedicated
beter is; geen exacte paperreproductie en geen optimaal afgestelde dedicated upper bound.

- CNN-8/svhnq expand=2, zelfde zes quantiseerbare convoluties. Eerste/laatste laag FP32.
- Volledige SVHN train=73257; test=26032; geen extra, validatiesplit of nieuwe preprocessing.
- Bestaande Resize(40), ToTensor, normalisatie mean=std=0.5.
- Batch128, drop_last=False, workers0, pin_memory=True, cudnn.benchmark=True.
- Adam lr0.001, weight_decay0; overige defaults uit ongewijzigde auteursoptimizer.
- 100 epochs, factor0.1 na epochs50/75/90. 573 updates per epoch, totaal57300.
- Nieuwe initialisatie met seed20261009. Geen vroegere checkpoint/pilot/warmstart.
- Alleen directe MXFP4 E2M1; een tekenbit, twee exponentbits/bias1, een mantissabit.
- G(W)=mean(abs(W)).detach()*tanh(W)/max(abs(tanh(W))) ongewijzigd; nulguard behouden.
- E8M0-schalen, groepen32 langs Cin*Kh*Kw binnen Cout, nulpadding.
- Zelfde schaalregel en nearest-even-afronding als mx_weights_dual_reference100_v1.
- GEEN E2M3-tussenrepresentatie; GEEN truncatie of teacher/distillatie.
- Een forward/backward, hard-label cross-entropy, een Adam-update per minibatch.
- A32/ReLU; BN32 is de ene actieve bank en leert tijdens training. Label32 is geen W32-claim.
- Andere BN-banken blijven opgeslagen maar onveranderd; geen gradient/Adamstaat.
- Mastergewichten, gradients en Adam-momenten FP32; identity-STE door MX.
- FP32-convolutie-emulatie met FP64-codecintermediairen; geen native-MX-hardwareclaim.
- Test na iedere epoch zonder model/BN-updates; geen herkalibratie.
- Eerste strikte maximum van E2M1-testaccuracy als best; laatste epoch apart bewaren.
- Hoofdvergelijking: epoch100 versus epoch100. Best gebruikt de testset en is niet onafhankelijk.
- Checkpoints bevatten volledige trainingsstaat; exports alleen directe E2M1 plus bijgegevens.
- Bij beide exports worden de gewichtstensors opnieuw gedecodeerd en exact vergeleken;
  geen nieuwe volledige testpass vanuit export. Ongebruikte BN-banken tellen mee in opslag.
- Geen oude data/resultaten overschrijven. Resume alleen van ditzelfde contract.

Afzonderlijke en gezamenlijke training gebruiken een ander aantal passes en losses.
Hetzelfde epoch-/updatebudget is geen gelijk compute-budget. Werkelijke initialisatie,
data-/bronhashes en samplevolgorde worden geregistreerd, niet enkel uit de seed aangenomen.

Bronnen: Any-Precision auteurscode train.py, models/quan_ops.py, models/svhn_quan.py,
optimizer.py (exacte bronhashes in SOURCE_MANIFEST.json); onze bestaande single100-runner;
E2M1-codec uit de directe dual100-referentie. MX-paper p.3 tabel1/Algorithm1 en p.4
FP32-masterkopie. Keuze voor gewichts-MX met A32 is ons onderzoek, geen volledige MX-paperflow.
