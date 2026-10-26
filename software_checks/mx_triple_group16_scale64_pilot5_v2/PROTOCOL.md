# Protocol — B16 ×64, pilot 5 epochs

`PROTOCOL.json` is dezelfde configuratie die `run_train.config()` teruggeeft. De volledige configuratie is vergeleken met de bewaarde oorspronkelijke B32 ×64-pilotconfiguratie. Alleen de groepsgrootte, experimentidentiteit en beschrijvende velden over deze vergelijking verschillen.

## Numerieke route

1. De FP32-mastergewichten gebruiken dezelfde oorspronkelijke preprocessing G(W): losgekoppeld gemiddelde van absolute gewichten, tanh en maxnormalisatie.
2. Groepeer G(W) per outputkanaal over `Cin*Kh*Kw` in **16** waarden; vul alleen de staart van elke rij met nul aan.
3. Bepaal de E4M3-bronschaal als `floor(log2(amax))-8`, met bestaande E8M0-bereikregels en nulblokregel. Codeer E4M3 met dezelfde exact geteste nearest-ties-to-even-codec.
4. Exponenttruncatie: `s e3 e2 e1 e0 m2 m1 m0 -> s e3 e2 m2 m1 m0`. De twee laagste exponentbits vervallen; de doelcode wordt als E2M3 geïnterpreteerd. Er wordt geen elementexponent-biascorrectie of extra afronding toegevoegd.
5. De lagere gedeelde schaalcode is exact de broncode plus **6**: schaalfactor **64**. Eerst promoveren naar een integer van voldoende breedte; overflow boven code254 wordt geweigerd, nooit gewrapt of geclipt.
6. Mantissatruncatie: de afgeleide E2M3-code `>>2` geeft E2M1. De E2M1-schaal is exact gelijk aan de E2M3-schaal.
7. E4M3 wordt eenmaal per laag en minibatch gecodeerd; alle drie modi gebruiken dezelfde losgekoppelde broncodes. De cache wordt vóór de Adamupdate gewist.
8. De hele quantisatieroute gebruikt dezelfde identity-STE naar G(W). Geen extra factor64 in backward en geen schaalgradient/leerbare schaalparameter.

De drie scalarformaten behouden al hun bestaande codewaarden, bias, subnormalen en afrondingsregels. Alleen hun groepengeometrie en pakketgrootte veranderen. Inclusief één schaalbyte bedragen de blokken nu 17/13/9 bytes voor 8/6/4 bit. Payloads slaan `block_size=16` expliciet op; laden met afwijkende groepsgrootte wordt geweigerd.

## Trainingsprotocol

- CNN8 `svhnq`, expand 2;132.954 modelparameters; dezelfde zes MX-lagen en FP32-uitzonderingen.
- SVHN train-only: 73.257 beelden, test: 26.032 beelden, geen extra- of validatiesplit.
- Batch 128, seed 20261009, workers 0, shuffletrain, geen drop_last.
- A32, FP32-master/gradients/Adam-momenten.
- Trainingvolgorde E4M3→E2M3→E2M1: harde CE voor E4M3, recursieve soft CE van losgekoppelde teacher voor beide lagere modi, T=1.
- Som van gradients, één Adamupdate per minibatch; lr 0.001, weight_decay 0.
- Het oorspronkelijke 100-epoch-schema met dalingen na 50/75/90 blijft ingesteld; pilot stopt op 5.
- BN-banken 8/32/4 voor E4M3/E2M3/E2M1. Het getal 32 in de middelste BN-banksleutel is een opslag-ID en wordt **niet** naar 16 gewijzigd.
- Evaluatie van alle drie modi na iedere epoch; alle accuracies uit het gedeelde E4M3-beste checkpoint en daarnaast het laatste checkpoint.
- Originele backend `cudnn.benchmark=True`; seedcontrole is geen CUDA-bitexactgarantie.

## Referentie en kosten

De bestaande echte B32 ×64-resultaten in `scale64_reference` blijven byte-identiek. De pilot gebruikt alleen epochs 1–5, inclusief herselectie van het beste E4M3-checkpoint binnen dat bereik. De volledige 100-epochreferentieresultaten mogen niet als nieuwe B16-resultaten of pilotbestcheckpoint worden geïnterpreteerd.

Kosten worden onafhankelijk berekend voor B16/B32/B64 uit de zes laagvormen, inclusief padding per rij. De actuele B16-exportaantallen moeten hiermee overeenkomen. Performancemetingen betreffen PyTorch-emulatie en zijn geen native-MX-hardwareclaims. B64-accuracies worden in deze proef niet gemeten.

## Opslag, hervatten en verificatie

Iedere opdracht schrijft een nieuwe tijdgestempelde map. Checkpoints bevatten het volledige contract inclusief groepsgrootte, schaalroute, bronhashes en pakketidentiteit. Hervatten vanuit een ander contract wordt vóór modelrestore/updates geweigerd. Compacte lagere exports worden uitsluitend uit de opnieuw ingelezen compacte E4M3-bron afgeleid, zonder FP32-masters. Exportchecks vergelijken de gedecodeerde gewichten exact met dezelfde route op het trainingsapparaat; ze starten geen extra volledige testevaluatie.

Het netwerk wordt voor deze proef niet uitgebreid met nieuwe precisies, activatiequantisatie, continue schalen, leerbare offsets of een andere groeperingsas. De vervolgrichting met factoren tussen 32 en 64 blijft een afzonderlijke onderzoeksvraag.
