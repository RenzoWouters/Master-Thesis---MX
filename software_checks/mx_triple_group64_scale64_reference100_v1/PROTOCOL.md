# Protocol — B64 ×64, volledige training van 100 epochs

`PROTOCOL.json` is gelijk aan `run_train.config()`. De numerieke route is identiek aan de succesvolle B64-pilot; alleen trainingsduur, experimentidentiteit en rapportage veranderen.

## Numerieke route

1. De oorspronkelijke preprocessing G(W) op de FP32-mastergewichten blijft behouden: losgekoppeld gemiddelde van absolute gewichten, tanh en maxnormalisatie.
2. Groepeer per outputkanaal over `Cin*Kh*Kw` in 64 waarden; vul het einde van iedere rij met nul aan.
3. Codeer eenmaal per MX-laag en minibatch een E4M3-bron met dezelfde nearest-ties-to-even-codec. De bronexponent is `floor(log2(amax))-8`, met bestaande E8M0-bereikregels en nulblokregel.
4. Exponenttruncatie: `s e3 e2 e1 e0 m2 m1 m0 → s e3 e2 m2 m1 m0`. De lagere exponentbits vallen weg; doelbias en codeinterpretatie blijven die van E2M3.
5. Lagere gedeelde schaalcode = broncode **+6**, dus bronschaal **×64**. Integerpromotie vóór optelling; overflow boven code 254 wordt geweigerd.
6. Mantissatruncatie: E2M3-code `>>2` geeft E2M1, met exact dezelfde E2M3-schaal.
7. De drie modi delen de losgekoppelde broncodes. De cache wordt vóór de Adamupdate gewist.
8. Identity-STE naar G(W) over de hele route; geen extra factor 64 in de backward en geen schaalgradient.

Payloads leggen `block_size=64` vast. Een blok inclusief schaalbyte bevat 65/49/33 bytes voor 8/6/4-bitcodes. Scalarcodecs, packing, afgeleide schalen, adapters en productiezelftests blijven byte-identiek aan de geslaagde B64-pilot.

## Training

- CNN8 `svhnq`, expand 2, 132.954 modelparameters en dezelfde zes MX-lagen.
- SVHN train-only: 73.257 beelden; test: 26.032. Geen extra- of validatiesplit. Bronbestanden `train_32x32.mat` en `test_32x32.mat`; oorspronkelijke transformaties naar 40×40.
- Verse oorspronkelijke initialisatie met seed 20261009. Batch 128, workers 0, train-shuffle en geen drop_last.
- A32; FP32-mastergewichten, gradients en Adam-momenten.
- Trainingvolgorde E4M3 → E2M3 → E2M1. Harde CE voor E4M3; recursieve soft CE van de losgekoppelde vorige modus voor de lagere precisies, T=1.
- Som van de gradients, één Adamupdate per minibatch; geen extra FP32-teacher.
- 100 epochs. LR 0,001 voor epochs 1–50; 0,0001 voor 51–75; 0,00001 voor 76–90; 0,000001 voor 91–100.
- Bij volledige data: 573 minibatches per epoch, 57.300 updates en 171.900 trainingspasses.
- BN-banken 8/32/4 voor E4M3/E2M3/E2M1; bank 32 is een opslag-ID en blijft gelijk.
- Oorspronkelijke backend `cudnn.benchmark=True`; geen bitexacte CUDA-garantie.

## Evaluatie en selectie

Na iedere epoch worden alle drie modi op de volledige officiële testset geëvalueerd. Selectie gebruikt het **eerste strikte maximum van E4M3-top1**. Voor het gekozen checkpoint worden ook de twee lagere modi van diezelfde epoch getoond. Epoch 100 wordt afzonderlijk gerapporteerd.

De B16- en B32-referenties bevatten hun volledige bestaande trainingen. De twee hoofdvergelijkingen zijn alle varianten op epoch 100 en iedere variant op zijn eigen globale E4M3-beste checkpoint. De geselecteerde referentie-epochs zijn 53 voor B16 en 42 voor B32. Gedeeltelijke runs zijn expliciet gemarkeerd en worden alleen over het gemeenschappelijke beschikbare bereik vergeleken.

Checkpointselectie gebruikt de officiële testset; de geselecteerde score is daarom geen onafhankelijke testschatting na modelselectie. Er is één seed per variant.

## Opslag en hervatten

Iedere opdracht krijgt een nieuwe tijdgestempelde map. Alle epochcheckpoints bevatten model, Adam, scheduler, RNG en het exacte experimentcontract. Alleen een eigen onvoltooid full100-checkpoint kan worden hervat. De full100-proef begint niet vanuit het getrainde pilotcheckpoint.

De compacte lagere exports worden uitsluitend afgeleid uit de opgeslagen en opnieuw ingelezen E4M3-bron, zonder FP32-masters. Codes, schaaloffset en gedecodeerde gewichten worden op het trainingsapparaat gecontroleerd. Alle zes geselecteerde exports worden tegen de geometrie van de zes lagen gevalideerd.

## Kosten

B64 bevat 2.160 groepen en schaalbytes, 8.768 paddingposities en 138.240 opgeslagen gewichtsposities voor 129.472 echte gewichten. De representaties inclusief schalen bedragen 140.400 / 105.840 / 71.280 bytes voor E4M3/E2M3/E2M1. Dit zijn alternatieve exports; de bedragen worden niet opgeteld. FP32-uitzonderingen, BN, optimizer en werkbuffers vallen erbuiten.

De vaste schaalfactor blijft ×64. Groeperingsas, model, loss en schaaloptimalisatie worden niet tegelijk veranderd; tussenliggende of geleerde schaalfactoren blijven een afzonderlijke vervolgvraag.
