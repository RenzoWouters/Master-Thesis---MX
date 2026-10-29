# Protocol — B64 ×64, pilot van vijf epochs

`PROTOCOL.json` is gelijk aan `run_train.config()`. De experimentele numerieke wijziging tegenover B16/B32 is groepsgrootte 64. Codecs, schaalfactor, groeperingsas, preprocessing en training behouden hun beleid.

## Numerieke route

1. Pas de oorspronkelijke G(W)-preprocessing toe op de FP32-mastergewichten: losgekoppeld gemiddelde van absolute gewichten, tanh en maxnormalisatie.
2. Groepeer per outputkanaal over de afgevlakte `Cin*Kh*Kw`-rij in 64 waarden. Vul alleen de staart van elke rij met nul aan.
3. Bepaal de E4M3-bronschaal met `floor(log2(amax))-8`, de bestaande E8M0-bereikregels en nulblokregel. De elementcodec behoudt nearest-ties-to-even.
4. Exponenttruncatie: `s e3 e2 e1 e0 m2 m1 m0 → s e3 e2 m2 m1 m0`. Interpreteer de doelcode als E2M3; geen elementbiascorrectie of extra afronding.
5. Lagere schaalcode = broncode **+6**, dus bronschaal **×64**. Promoveer voor de optelling naar een voldoende breed integerdtype. Overflow boven code 254 wordt geweigerd; geen wrap of clipping.
6. Mantissatruncatie: de afgeleide E2M3-code `>>2` geeft E2M1. E2M1 behoudt exact de E2M3-schaal.
7. Codeer E4M3 eenmaal per MX-laag en minibatch. De drie modi gebruiken dezelfde losgekoppelde broncodes; de cache wordt vóór de Adamupdate gewist.
8. Gebruik de identity-STE naar G(W) voor de volledige route, met behoud van de oorspronkelijke G(W)-afgeleide. Geen extra factor 64 in de backward en geen schaalgradient.

Scalarformaten behouden hun bestaande codewaarden, bias, subnormalen en afronding. Inclusief één schaalbyte bevat een B64-blok **65/49/33 bytes** voor respectievelijk 8/6/4 bit. Payloads slaan `block_size=64` expliciet op; een andere groepsgrootte wordt bij laden geweigerd.

## Training

- CNN8 `svhnq`, expand 2, 132.954 modelparameters, dezelfde zes MX-lagen.
- SVHN train-only 73.257 en test 26.032 beelden; geen extra/validatiesplit. Bestanden `train_32x32.mat` en `test_32x32.mat`; oorspronkelijke transformaties naar 40×40.
- Batch 128, seed 20261009, workers 0, shuffletrain en geen drop_last. Verse oorspronkelijke initialisatie.
- A32; FP32-mastergewichten, gradients en Adam-momenten.
- E4M3 harde CE; E2M3 soft CE van losgekoppelde E4M3; E2M1 soft CE van losgekoppelde E2M3; T=1.
- Som van gradients, één Adamupdate per minibatch, LR 0,001 en weight_decay 0.
- Oorspronkelijk LR-schema van 100 epochs met dalingen na 50/75/90; stop op 5. Bij volledige data: 573 minibatches per epoch, 2.865 updates en 8.595 trainingspasses in vijf epochs.
- BN-banken 8/32/4 voor E4M3/E2M3/E2M1. Bank 32 is een opslag-ID en verandert niet met groepsgrootte.
- Alle drie modi op de volledige officiële testset na elke epoch. Selecteer het eerste strikte E4M3-maximum; neem alle drie scores uit hetzelfde checkpoint. Bewaar daarnaast het laatste checkpoint.
- Originele `cudnn.benchmark=True`-backend; vaste seed geeft geen bitexacte CUDA-garantie.

## Referenties en notebook

`scale64_reference` bevat de bestaande B32 ×64-run van 100 epochs. `group16_reference` bevat de nieuwe B16 ×64-run van 100 epochs. Hun bronbestanden blijven ongewijzigd. De pilotvergelijking gebruikt alleen epochs 1–5, of het gemeenschappelijke kortere bereik bij een onderbroken run. Selecteer de beste E4M3-epoch opnieuw binnen dat bereik. De globale referentiemaxima op 42 en 53 zijn geen pilotcheckpointkeuzes.

De onuitgevoerde notebook toont curves, epoch 5, het gezamenlijke E4M3-beste pilotcheckpoint, beide referenties, quantisatie, looptijd en de kosten van schalen en padding. Epoch 100 is nog geen B64-resultaat. De kosten worden uit de zes echte laagvormen berekend en tegen alle zes geselecteerde exports gecontroleerd.

## Opslag en hervatten

Iedere opdracht schrijft naar een nieuwe tijdgestempelde map. Checkpoints bevatten het contract met groepsgrootte, schaalroute, bronhashes en pakketidentiteit. Een ander contract wordt vóór restore of updates geweigerd. Compacte lagere exports ontstaan uitsluitend uit de opgeslagen en opnieuw ingelezen compacte E4M3-bron, zonder FP32-masters. Gewichtsreconstructie wordt exact op het trainingsapparaat gecontroleerd.

Deze proef verandert geen precisies, activaties, schaalfactor, groeperingsas of loss. Tussenliggende schaalfactoren blijven een latere afzonderlijke onderzoeksvraag.
