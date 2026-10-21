# MX-onderzoeksoverzicht voor de meeting - versie 2

Pak de volledige ZIP uit in een NIEUWE map. Behoud versie 1 en alle ruwe onderzoeksruns.
Open `MX_onderzoeksoverzicht.ipynb` in VS Code, kies `.venv-train` en voer Run All uit.
Open de notebook vanuit de uitgepakte pakketmap; bij een afwijkende working directory vul je BASE in de eerste cel in.
De HTML bevat dezelfde al uitgevoerde resultaten. Geen nieuwe installatie, datasetdownload of training nodig.
Dit overzicht importeert geen PyTorch en opent geen onderzoekscheckpoint.

Nieuw: de volledige dedicated E2M1-run is toegevoegd in alle betreffende tabellen,
de E2M1-figuur, de lage leercurve, testloss, opslag-afweging, looptijden en bijlagen.
Sectie 2b bevat de gevraagde nieuwe E2M3-vergelijkingsfiguur op epoch 100; in bijlage B
staan de drie vijf-epochpilots apart. Er worden geen identieke herevaluaties dubbel geteld.
Sectie 6 verklaart de vier opslagstrategieen; twee exports zijn niet twee dedicated trainingen.

De bronbestanden zitten byte-identiek in bronnen/. Het notebook controleert hun SHA-256,
CSV/JSON-consistentie, checkpointselectie en LR-schema. Nieuwe tabellen in afgeleid/ en
figuren/ worden bij Run All opnieuw berekend. Dat zijn afgeleide resultaten, geen ruwe data.
Voor de dedicated run staan ook het codepakket en alle 19 feedbackbestanden erbij.

Afgrenzing: een seed per methode; geen significantie- of gelijkwaardigheidsbewijs;
testgestuurde checkpointselectie; verschillende trainingsregimes; geen hardwarebenchmark.
Mastergewichten en Adam blijven FP32. Opslaggetallen zijn payloadberekeningen, geen VRAM.
Echte exports en checkpoints ontbreken; de nieuwe volledige exportinferentietest blijft open.
