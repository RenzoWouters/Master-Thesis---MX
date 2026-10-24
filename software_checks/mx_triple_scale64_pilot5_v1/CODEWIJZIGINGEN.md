# Wijzigingen ten opzichte van de naïeve vijf-epochpilot

## Hergebruik en behoud

Vijftien bestanden zijn byte-identiek gekopieerd uit
`mx_triple_truncation_aware_pilot5_v1`, waaronder alle drie de MX-codecs, de
basis- en drieprecisieadapters, de oude exponent-/mantissafuncties, dataset- en
resumehulpmiddelen, codeccontroles, bronmanifest en upstream ZIP.
`REUSE_MANIFEST.json` legt de bestandsnamen en hashes vast.

De oorspronkelijke pakketten en resultaten zijn niet gewijzigd. `provenance`
bevat kopieën van de relevante oude pilotbestanden en de gevalideerde
post-hoc schaal64-implementatie. De nieuwe schemas onderscheiden route B
expliciet van de oude A-export- en resumecontracten.

## Numerieke wijziging

`trunc_adapter.code_chain` voert dezelfde exponent-MSB-bewerking uit en verhoogt
dan de lagere E8M0-schaalcode met 6. De daaropvolgende mantissatruncatie behoudt
die lagere schaal. De E4-bron blijft ongewijzigd. De schaalgrenscontrole promoveert
de integer vóór optellen en weigert overflow. Dit is exact route B uit de
post-hocevaluatie; C maakt geen deel uit van dit trainingspad.

`_CachedCodeSTE` en de broncache behouden hun bestaande mechanica. De identity
in backward blijft over de volledige route gelden. Het bronbeleid, audits en
exportmetadata beschrijven nu de vaste offset 6 en factor 64. `trunc_support.py`
controleert die relaties tijdens train en test, met dezelfde loss-/updatevolgorde.

## Runner en presentatie

`run_train.py` gebruikt een nieuwe pakket-, rapport-, protocol- en resultaatnaam.
De kern van de epochloop en de oorspronkelijke hyperparameters blijven behouden.
Export- en samenvattingscontroles verwachten voor de lagere modi source+6;
voor de E4-bron blijft de offset 0. Run stopt vast op5; resume vereist exact het
nieuwe eigen contract. De notebookkopie krijgt automatisch RUN_DIR en standaard
SAVE_FIGURES=False.

`notebook_support.py` en de lege notebook zijn aangepast voor de nieuwe schemas,
lagere schaaloffset en auditvelden. CSV-floats worden met round_trip gelezen,
zodat de bekende parserafwijking bij grote CE-getallen niet terugkeert. Figuren
blijven de drie modi, epoch 5/E4-best, learning curves, losses, voorspelde klassen,
tijden en geheugen tonen. Epoch100 blijft herkenbaar ongemeten.

## Validatie

De nieuwe kern is onafhankelijk vergeleken met de oude bitmapping plus een
scalaire +6-schaaloracle en met de gearchiveerde post-hoc B6/B4-functies. De
gradients en echte upstream trainingslus zijn gecontroleerd tegen een handmatige
referentielus. De afzonderlijke runnerintegratie controleert run/check/resume,
stoppen op epochgrenzen, output en behoud van inputs op kleine CPU-fixtures.
De notebook wordt met synthetische resultaten getest en daarna op de echte
uitvoer van die synthetische runner. De rapporten benoemen die scope expliciet.
