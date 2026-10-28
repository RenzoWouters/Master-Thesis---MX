# MX-onderzoeksoverzicht — meeting 29 oktober 2026

Open `MX_onderzoeksoverzicht_meeting_20261029.ipynb` en kies **Run All / Alles uitvoeren**.
De notebook is zelfstandig: de gegevens en tekenfuncties zijn ingesloten. Hij kan dus ook
los uit deze map gebruikt worden. Nodig zijn Jupyter met NumPy, pandas en Matplotlib.
De bestaande notebookomgeving volstaat; er is geen PyTorch-, dataset- of GPU-toegang nodig.

## Inhoud

- De vijf GitHub Any-Precision-modi op SVHN/CNN-8, epoch 100.
- Dedicated E2M3/E2M1, gezamenlijk direct en gezamenlijk met mantissatruncatie.
- Dedicated E4M3 plus alle triple-direct, naïeve, ×64, ×32, learned, freeze en B16/B64-trajecten.
- Alle uitgevoerde posthoc-MX-, mantissa-, exponent-, schaal- en afrondingscontroles.
- Actieve modelopslag, flexibele bronopslag, groepspadding, FP32-trainingsstaat en gemeten tijd.
- De oorspronkelijke hoogste-precisie-checkpoints, volledige waarden en bronverantwoording.

15 volledige trainingstrajecten (waarvan één voortzettingsvertakking), 37 epoch-100-metingen,
3.700 testmetingen tijdens training, 90 posthocmetingen uit zes proeven. Alle MX-modi blijven
A32 met FP32-updatemaster. Geen nieuwe experimenten zijn uitgevoerd.

## Bestanden

- `data/results.json`: dezelfde data als in de notebook, leesbaar als JSON.
- `sources/`: 154 bytegetrouwe primaire rapporten, CSV's en broncodebestanden.
- `source_manifest.json`: bronhashes en geregistreerde herkomst.
- `tables/`: exact gecontroleerde, afgeleide vergelijkingstabellen.
- `figures/`: figuren van de gecontroleerde uitvoering, PNG en SVG.
- `validation/`: controle van data, bronkopieën en notebookuitvoering.
- `overview_plots.py`: dezelfde tekenfuncties als de notebook, als leesbare aanvullende code.
- `MX_onderzoeksoverzicht_meeting_20261029.html`: directe leesversie zonder code-uitvoer nodig.

De geleverde notebook heeft lege outputs, zodat je zelf de uitvoer kunt genereren. Er wordt
geen tweede uitgevoerde notebook meegeleverd. De HTML en figuren zijn een aanvullende leesversie.
`SAVE_FIGURES=True` maakt een nieuwe figurenmap; standaard blijft die optie uit.

## Interpretatie

Vergelijk eerst epoch 100. De gekozen checkpoints gebruiken altijd één hoogste-precisie-epoch
per gezamenlijk model. Dedicated categorieën combineren aparte runs. GitHub verandert W en A
samen; MX houdt A32. Posthoc gebruikt vaste BN-banken zonder herkalibratie. De freeze-variant
erft epochs 1–50 en blijft na 50 theta-gradients/Adam-momenten berekenen met learning rate 0.
Kosten voor een actieve modelkopie, een flexibele bron, een `.pt`-container en FP32-training
zijn verschillende scopes. De gedeelde directe multi-codecontainers zijn analytische layouts.
Eén seed per methode, testsetselectie en niet-bitexacte CUDA geven geen algemeen optimum of
statistische gelijkwaardigheid. Er is nog geen MX-activatie- of continual-learning-resultaat.
