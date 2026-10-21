# Protocol: eerste truncatiebewuste MX-training

## Onderzoeksvraag
Kan gezamenlijke training met E2M3 en een werkelijk daaruit getrunceerde E2M1-
uitvoering leren? Deze vijf-epochproef controleert software en tijd. Vergelijk
later met directe gezamenlijke MX-kwantisatie op gelijke trainingsbudgetten;
geen voorspelling dat de eerder gemeten truncatiekloof hersteld wordt.
Dit is onze onderzoeksvariant, geen exacte paperreproductie.

## Behouden uit de directe tweemoduspilot v1.1
CNN-8/svhnq expand=2, zes geselecteerde convolutionele gewichtsquantisatoren.
Eerste convolutie, laatste lineaire laag, BN/activaties en rest blijven zoals
voorheen. Alle vijf oorspronkelijke BN-banken behouden; key32 voor E2M3,
key4 voor E2M1, key1/2/8 ongebruikt. ALLE abit/wbit-selectors blijven 32.

SVHN train-only: 73.257 beelden, test 26.032; geen extra/held-out-validatiesplit.
Resize(40), ToTensor, Normalize(mean=std=0.5), batch128, drop_last=False,
workers0, train shuffle, test vaste volgorde, pin_memory=True. Zelfde seed
20261009; nieuwe initialisatie, geen pilot- of honderd-epochwarmstart.
Adam lr0.001, weight_decay0; oorspronkelijke optimizer.py en scheduler.
Honderd-epoch LR-plan (daling factor0.1 NA 50/75/90), pilotstop op vijf.

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

## Enige nieuwe numerieke ingreep
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
niet onafhankelijk. Vijf epochs geven geen eindaccuracy of multi-seedbewijs.

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
