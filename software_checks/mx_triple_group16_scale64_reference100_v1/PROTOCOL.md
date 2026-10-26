# Protocol — B16 ×64, referentietraining van 100 epochs

`PROTOCOL.json` is exact de configuratie van `run_train.config()`. Tegenover de geslaagde B16-pilot verandert alleen de uitvoeringsduur en de daarbij horende rapportage-/experimentidentiteit. De volledige training start opnieuw vanaf dezelfde oorspronkelijke seed en initialisatie.

## Numerieke route

1. FP32-mastergewichten gebruiken de oorspronkelijke G(W)-preprocessing: losgekoppeld gemiddelde van de absolute gewichten, tanh en maxnormalisatie.
2. Groepeer per outputkanaal over de afgevlakte `Cin*Kh*Kw`-rij in 16 waarden. Alleen de staart van iedere rij krijgt nulpadding.
3. Codeer eenmaal per laag en minibatch een E4M3-bron met dezelfde exact geteste nearest-ties-to-even-codec. De gedeelde E8M0-bronschaal komt uit `floor(log2(amax))-8`, met dezelfde bestaande bereik- en nulblokregels.
4. Exponenttruncatie bewaart de hoogste twee exponentbits: `s e3 e2 e1 e0 m2 m1 m0 -> s e3 e2 m2 m1 m0`. Het doel wordt als E2M3 geïnterpreteerd. Geen extra afronding of elementbiascorrectie.
5. De E2M3-schaalcode is exact de broncode plus 6, dus factor 64. Integerpromotie voorkomt uint8-wrap; overflow boven 254 wordt geweigerd.
6. E2M1 ontstaat door de afgeleide E2M3-code `>>2`; de gedeelde schaal blijft exact gelijk aan die van E2M3.
7. Alle drie modi delen de losgekoppelde broncodes; de broncache wordt vóór Adam gewist. De hele route houdt dezelfde identity-STE naar G(W), zonder factor 64 in backward. Er zijn geen leerbare schalen of schaalgradients.

Scalarformaten, codecs, truncatie, packing, exportheropbouw en schaalbeleid zijn byte-identiek aan de geslaagde pilot v2. Payloads bevatten expliciet `block_size=16`; oude B32/B64-payloads worden niet als B16 geïnterpreteerd.

## Training

- CNN8 `svhnq`, expand 2, 132.954 parameters; dezelfde zes MX-lagen en FP32-uitzonderingen.
- SVHN train-only: 73.257 trainbeelden en 26.032 testbeelden; geen extra- of validatiesplit.
- Batch 128; seed 20261009; workers 0; shuffle op train; geen `drop_last`.
- A32 en FP32-mastergewichten, gradients en Adam-momenten.
- Drie passes: E4M3 harde CE, E2M3 soft CE van losgekoppelde E4M3, E2M1 soft CE van losgekoppelde E2M3; temperatuur 1. Gradients worden opgeteld, daarna één Adamupdate.
- Adam met lr 0,001 en weight_decay 0. Dalingen na 50/75/90: epochs 1–50 op 0,001; 51–75 op 0,0001; 76–90 op 0,00001; 91–100 op 0,000001.
- BN-bankopslag 8/32/4 voor E4M3/E2M3/E2M1. De sleutel 32 blijft een bank-ID en verandert niet met de groepsgrootte.
- Volledige evaluatie van alle drie modi na elke epoch; eerste strikte E4M3-maximum bepaalt het gezamenlijke beste checkpoint.
- Alle resultaten uit dat checkpoint en uit epoch 100 worden gerapporteerd. Bij een vroegtijdige stop wordt ook de laatste volledige epoch bewaard zonder te suggereren dat epoch 100 is bereikt.
- Oorspronkelijke CUDA-backend `cudnn.benchmark=True`; geen claim van bitexacte CUDA-herhaling/hervatting.

De productie-CLI start standaard een nieuwe honderd-epochrun. Hervatten accepteert uitsluitend een eigen volledig honderd-epochcontract en een onafgewerkt checkpoint. De pilot wordt niet als warmstart gepromoveerd. Alle eerdere resultaten blijven behouden.

## Referentie en kosten

De volledige bestaande B32 ×64-run wordt ongewijzigd als referentie meegenomen. De hoofdvergelijkingen zijn epoch 100 en per training het gezamenlijke E4M3-beste checkpoint. Tussentijdse vergelijkingen zijn beperkt tot het gemeenschappelijk afgewerkte bereik. De eerdere B16-pilot blijft apart beschikbaar als pilotbewijs.

Kosten worden onafhankelijk berekend voor B16/B32/B64 uit de zes werkelijke laagvormen, inclusief padding per rij. Actuele B16-exports moeten hiermee overeenkomen. B64 blijft een analytische kostenvariant; de huidige proef levert alleen B16-accuracies. Looptijd en GPU-geheugen zijn PyTorch-emulatiemetingen, geen hardware-energie- of native-MX-snelheidsclaims.

## Opslag en verificatie

Iedere opdracht krijgt een nieuwe resultatenmap. Checkpoints bewaren model, optimizer, scheduler, RNG, complete epochgeschiedenis en het volledige contract met pakket-/bronhashes. Compacte lagere exports worden uitsluitend uit de opnieuw ingelezen compacte E4M3-bron afgeleid, zonder de FP32-masters. De runner vergelijkt gedecodeerde exports exact met dezelfde route op het trainingsapparaat. De notebook voert geen modeltraining of nieuwe volledige testevaluatie uit.
