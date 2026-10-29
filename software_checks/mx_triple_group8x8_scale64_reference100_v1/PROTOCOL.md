# Protocol — 8×8, vaste ×64, 100 epochs

`PROTOCOL.json` is exact gelijk aan `run_train.config()`. De numerieke implementatie is ongewijzigd tegenover de geslaagde 8×8-pilot. Deze proef start opnieuw vanaf initialisatie en loopt tot epoch 100.

## Numerieke route

1. Bereken oorspronkelijk G(W) uit de FP32-mastergewichten: losgekoppeld gemiddelde van absolute waarden, tanh en maxnormalisatie.
2. Vorm `[Cout, Cin×Kh×Kw]`, vul beide assen met positieve nullen aan tot veelvouden van acht en vorm niet-overlappende 8×8-blokken. Binnen elk blok staan de 64 elementen in rijvolgorde.
3. Codeer E4M3 met dezelfde RNE-afronding, scalar-subnormal-flush, E8M0-bereikregels en nulblokregel. Bronschaalexponent: `floor(log2(amax))-8`.
4. Exponenttruncatie: `s e3 e2 e1 e0 m2 m1 m0 → s e3 e2 m2 m1 m0`, geïnterpreteerd als E2M3 zonder extra afronding of elementbiascorrectie.
5. Lagere schaalcode = broncode +6, dus bronschaal ×64. Gebruik een bredere integerdtype; overflow boven 254 wordt geweigerd, ook bij nulblokken. Geen wrap of clipping.
6. E2M1 ontstaat uit de E2M3-code met `>> 2`, met exact dezelfde lagere schaal.
7. E4M3 wordt eenmaal per MX-laag en minibatch gecodeerd. De drie modi delen die losgekoppelde bron; de cache wordt vóór Adam gewist.
8. De volledige route gebruikt de bestaande identity-STE naar G(W), inclusief G(W)-afgeleide. Geen extra factor 64 in backward en geen schaalgradient.

Codevorm: `[ceil(Cout/8), ceil(K/8), 64]`; schaalvorm: de eerste twee dimensies. Decoderen herstelt de oorspronkelijke vorm en verwijdert beide paddingranden. Alleen +0-codes zijn geldig op paddingposities. Exportlayoutstrings en nieuwe source/derived-schemas sluiten de oude 1×64-layout uit.

## Training

- SVHN CNN-8 `svhnq`, expand=2; dezelfde zes MX-convoluties, eerste en laatste laag FP32.
- Train-only: 73.257 train- en 26.032 testbeelden; geen extra of validatiesplit. Bestanden `train_32x32.mat` en `test_32x32.mat`, bestaande resize naar 40×40.
- Vanaf initialisatie, seed 20261009, batch 128, workers 0, shuffletrain, geen drop_last.
- A32; FP32-mastergewichten, gradiënten en Adam-momenten.
- E4M3 harde CE, E2M3 zachte CE van losgekoppelde E4M3, E2M1 zachte CE van losgekoppelde E2M3; T=1. Som van gradiënten, één Adamupdate per minibatch.
- Adam LR 0,001, weight_decay 0. Oorspronkelijke planning van 100 epochs met dalingen na 50/75/90; deze referentierun eindigt vast op epoch 100.
- Volledige run: 573 minibatches per epoch, 57.300 updates, 171.900 trainingspasses.
- BN-banken E4M3=8, E2M3=32, E2M1=4. Dit zijn opslag-ID's; alle activaties blijven A32.
- Alle drie modi worden na iedere epoch op de volledige testset geëvalueerd, met bevroren BN en zonder updates.
- Selectie op het eerste strikte E4M3-maximum; alle modi uit datzelfde checkpoint. Bewaar ook het laatste checkpoint.
- Oorspronkelijk `cudnn.benchmark=True`; geen bitexacte CUDA-herhaal- of hervatgarantie.

## Vergelijking en rapportage

De echte B32- en B64-runs met vaste ×64 staan als `scale64_reference` en `group64_reference` naast de notebook. Vergelijk alle 100 epochs bij een volledige run. Na onderbreking wordt het gemeenschappelijke bereik gebruikt; het beste E4M3-checkpoint wordt dan eveneens binnen dat bereik gekozen. Bij volledige runs telt het globale E4M3-maximum.

Grafieken: volledige curves per precisie, epoch 100, gezamenlijk E4M3-beste checkpoint, looptijd en kosten. De kostenanalyse telt per laag schalen, padding en codepayload voor 1×32, 1×64 en 8×8. De zes geselecteerde compacte exports worden tegen de werkelijke 8×8-geometrie gecontroleerd.

## Bewaring en hervatting

Iedere opdracht schrijft in een nieuwe resultatenmap. Checkpoints bewaren FP32-trainingstaat en het volledige groepsvorm-, schaal-, bron-, pakket- en omgevingscontract. Hervatting vereist hetzelfde volledige 100-epochcontract. Een pilotcheckpoint is geen toegestane warmstart of hervatbron. Lagere compacte exports worden uitsluitend afgeleid uit de opgeslagen en opnieuw geverifieerde E4M3-bron, zonder masters van de zes MX-lagen. Decodering wordt exact vergeleken met de trainingsroute op hetzelfde apparaat.

Deze proef verandert geen activatieprecisie, loss, optimizer of schaalfactor en introduceert geen geleerde schaal. Bestaande resultaten blijven behouden.
