# Protocol: honderd epochs truncatiebewuste MX-training

## Onderzoeksvraag
Wat is het resultaat na honderd epochs van dezelfde geslaagde vijf-epochproef?
De E2M1-uitvoering komt werkelijk uit de gedeelde E2M3-broncodes. Vergelijk
met directe gezamenlijke MX-kwantisatie op hetzelfde honderd-epochbudget;
geen voorspelling dat de eerder gemeten truncatiekloof hersteld wordt.
Dit is onze onderzoeksvariant, geen exacte paperreproductie.

## Behouden uit de truncatiebewuste vijf-epochpilot v1
CNN-8/svhnq expand=2, zes geselecteerde convolutionele gewichtsquantisatoren.
Eerste convolutie, laatste lineaire laag, BN/activaties en rest blijven zoals
voorheen. Alle vijf oorspronkelijke BN-banken behouden; key32 voor E2M3,
key4 voor E2M1, key1/2/8 ongebruikt. ALLE abit/wbit-selectors blijven 32.

SVHN train-only: 73.257 beelden, test 26.032; geen extra/held-out-validatiesplit.
Resize(40), ToTensor, Normalize(mean=std=0.5), batch128, drop_last=False,
workers0, train shuffle, test vaste volgorde, pin_memory=True. Zelfde seed
20261009; nieuwe initialisatie, geen pilot- of honderd-epochwarmstart.
Adam lr0.001, weight_decay0; oorspronkelijke optimizer.py en scheduler.
Honderd volledige epochs. Daling factor0.1 NA 50/75/90; eerste lagere LR in
epoch51/76/91. De eerdere vijf-epochstop vervalt. Geen extra vroegstopcriterium.

E2M3 hard CE met labels; E2M1 oorspronkelijke soft CE met detached hoge softmax
(T=1). EERST hoge backward, DAN lage forward/backward; gradients sommeren,
geen deling door twee, een Adam-update. Geen extra FP32-teacher of lage harde
loss in de optimalisatie. Dropout-maskers volgen de bestaande RNG-stromen,
geen nieuwe maskersynchronisatie. BN leert tijdens iedere moduspass; geen
post-training BN-herkalibratie, geen BN-updates tijdens test.

FP32-updatemasters, gradients en Adam-momenten. Dezelfde
G(W)=(tanh(W)/max(abs(tanh(W))))*mean(abs(W)).detach(), exacte bestaande
bewerkingsvolgorde en bestaande veilige nul-laagafhandeling. Geen extra epsilon.
A32. Geen exponent-, schaal-, activatie- of gradienttruncatie. Geen AMP,
torch.compile of gewijzigde cuDNN-productieinstellingen.

## Ongewijzigde numerieke methode
Hoge bron: identieke MXFP6 E2M3 round-nearest-even-codec. Groepen32 binnen
Cout langs afgevlakte Cin*Kh*Kw, +0-padding. E8M0 met frexp(power)-3-
schaalkeuze en bestaande nul-/subnormal-/verzadigingsregels.

Per laag en minibatch wordt die hoge bron eenmaal gevormd. De bron bestaat
uit integercodes en schalen; geen opgeslagen autograd-grafiek. De lage modus
krijgt C4=C6>>2 ([s ee mmm] -> [s ee m]) en een onveranderde kopie van X.
Geen low.encode_tensor/encode_elements, geen opnieuw afronden uit hoge
waarden en geen nieuwe schaalkeuze. De oorspronkelijke bron blijft intact.

Dezelfde bron wordt ook in de testlus met de oorspronkelijke volgorde4->6
gebruikt. Na beide passes wordt de cache geleegd, voor de volgende update.
Na Adam mag een nieuwe hoge bron met nieuwe schalen ontstaan. De cache is
geen masterparameter en wordt niet in hervatcheckpoints bewaard.

Gradientkeuze: voor BEIDE paden een identity-STE ten opzichte van G(W).
Backward door de samengestelde bron-plus-truncatiestap geeft grad_output
onveranderd aan G(W), inclusief verzadiging. Daarna de bestaande G(W)-gradient.
Geen factor1/4: de codeverschuiving deelt de numerieke waarde niet door vier.
Dit is een expliciete surrogate, niet de echte afgeleide van de discrete
bewerking of een afleiding van TruncQuant's integer-STE-factor.
De lage loss heeft een eigen G(W)-grafiek; de gecachte codes zijn detached,
maar de lage gradient wordt NIET afgesneden naar de masters.

## Opslag en evaluatie
Iedere epoch train/test op beide modi. Eerste strikte maximale E2M3-testscore
kiest best; last is de laatste voltooide epoch. Beide modi altijd van hetzelfde
checkpoint, geen samengestelde beste-per-modusrij. Testgestuurde selectie is
niet onafhankelijk. Last is bij volledige afronding epoch100. Honderd epochs
met een seed leveren nog geen multi-seedbewijs of gegarandeerde convergentie.

Per rol een hoge bronexport, vervolgens een E2M1-controlebestand dat UIT DE
GEVERIFIEERDE, HERLADEN hoge bron komt. De hoge bron bevat alle benodigde
uitzonderingslagen en BN-staat maar geen zes FP32-mastertensors. Het lage
bestand is niet vereist voor precisieomschakeling. Ruwe payloadgrootte,
serialisatiegrootte, FP32-herstelstaat en FP32-emulatiebuffers zijn verschillende
grootheden. Geen energie- of native versnellingclaim.
Exportgewichten worden vergeleken met de trainingsforward. Volledige testset
niet nogmaals na export doorlopen; synthetische uitvoergelijkheid wordt wel
gemeten. Alle eerdere bestanden en ruwe data blijven ongewijzigd.

## Vergelijkingsafbakening
De oude directe E2M1-testscore en getrunceerde E2M1-score zijn niet dezelfde
uitvoering. Vergelijk training op gelijke epochs, zelfde data/seed/modi/loss.
BN leert nu ook onder truncatie; dat is onderdeel van deze methode, geen
apart geisoleerde BN-kalibratieverbetering. De vroegere vijfprecisie W/A-run
is context, geen volledig gematchte controle. Geen continual-learningclaim.

## Bronnen en eigen keuzes
- Any-Precision-auteurs-ZIP uit het project: train.py forward, CrossEntropyLossSoft,
  models/quan_ops.py, svhn_quan.py, optimizer.py. ZIP/hash in SOURCE_MANIFEST.
- mx_weights_dual_pilot5_v1_1.zip: geslaagde directe tweemodus-referentiekern.
- mx_mantissa_truncation_eval_v1.zip, truncation.py: codebitshift van pad C.
- Microscaling Data Formats for Deep Learning, project-PDF p.3 Table1/Algorithm1,
  p.4 FP32-masterkopie. Wij gebruiken alleen gewichts-MX.
- TruncQuant, project-PDF p.2 sectieII: kwantisatie/truncatie en STE-raamwerk.
  Geen claim van letterlijke implementatie van de uniforme TruncQuant-formule.
Eigen keuzes: behoud van bestaande G(W), identity-STE, groepen32, BN-mapping,
train-only, twee modi en cachebeheer. Exacte bronherkomst en wijzigingen staan
in TRAINING_CHANGE_MANIFEST.json en de leesbare .patch-bestanden.

## Overgang naar deze volledige run
run_epochs, trunc_adapter.py, trunc_support.py, beide codecs, STE, BN-mapping,
source-cachebeheer en de data/optimizer/stop/herstelkern zijn byte-identiek aan
de op de laptop geslaagde pilot. Zie FULLRUN_CHANGE_MANIFEST.json en fullrun.patch.
TRAINING_CHANGE_MANIFEST.json beschrijft de HISTORISCHE numerieke wijziging van
directe kwantisatie naar truncatiebewust trainen; niet de nieuwe epochlimiet.

Elke run begint vanaf dezelfde seed en schrijft nieuwe mappen. Resume is alleen
voor een noodonderbreking van DIT reference100-contract; geen pilot-warmstart.
Bij een zachte onderbreking blijft status STOPPED_AT_EPOCH_BOUNDARY, met
full_training_completed=false. Pas na100 epochs: 57300 updates,114600 forwards,
400 metricrijen; completion_summary.json en gemeten totale epochtijd. Eindexport
controleert beide bronpaden, de LR voor deze en volgende epoch, BN/Adam/gradients.
De exports bevatten dezelfde inhoud/semantiek als in de pilot; GEEN aanvullende
volledige testforward na export. Geen verandering van productiebackend of
zelftesttoleranties. Wijzigingen in notebook en eindexport zijn rapportage.
