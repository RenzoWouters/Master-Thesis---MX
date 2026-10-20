# Wat verandert in de code?

## Geen wijziging van de trainingscode

De bestaande training, checkpoints, schalen en BN-parameters blijven vast.
Beide oorspronkelijke codecs, dual_adapter.py, mx_adapter.py en de bron-/data-
hulpfuncties zijn byte-identiek overgenomen uit het werkende honderd-epochpakket.
Zie REUSE_MANIFEST.json. De originele auteurs-ZIP is eveneens ongewijzigd.

## Nieuwe module truncation.py

`truncate_codes(source)` bevat de eigenlijke ingreep:

```python
codes4 = (source.codes.to(torch.int64) >> 2).to(torch.uint8)
result = low.MXTensor(codes4, source.scale_codes.clone(), source.shape)
```

`source.codes` zijn reeds UITGEPAKTE zesbitcodewoorden, niet de packed bytes en
niet de FP32-gewichtswaarden. Bij [s ee mmm] schuift dit teken+exponent+hoogste
mantissabit door naar [s ee m]. Schaalbytes en oorspronkelijke tensorvorm blijven
gelijk; twee vierbitcodes worden daarna opnieuw in een byte gepakt.

De oude zesbitdata worden niet overschreven. Zo kan de hoge bron opnieuw worden
gebruikt voor een hogere precisie. C roept nergens een gewichts- of elementquantizer
aan. Zelfs als die quantizerfuncties in de test fouten opwerpen, moet C blijven werken.

`round_source_codes(source)` is alleen de aanvullende B-controle:

```python
codes4 = low.encode_elements(high.decode_elements(source.codes))
```

Deze rondt de reeds vertegenwoordigde, genormaliseerde elementen af; de opgeslagen
schaal wordt expliciet gekopieerd. Er wordt niet opnieuw over het blok geschaald.

Voorbeeld met schaal1: bronwaarde1.375 wordt als E2M3-code001011 bewaard.
- A direct E2M1:1.5 (code0011)
- B afronden vanuit E2M3:1.5
- C twee bits weg:001011 >> 2 = 0010, waarde1.0

Bij een oorspronkelijke waarde1.3 kan eerst naar E2M3 afronden1.25 geven. B rondt
dan naar1.0 (ties-to-even), terwijl A rechtstreeks1.5 geeft. Dit is waarom B apart
wordt gemeten en niet als synoniem voor A wordt behandeld.

`derive_payload` neemt uitsluitend de hoge export als input. Die hoge export bevat
ook de eerder getrainde lage BN-bank. Het afgeleide bestand krijgt een nieuw schema,
de methode, de hash van zijn bron en lage BN-bankkey4. Voor de zes MX-lagen staan er
geen FP32-masterparameters in. De uitzonderingslagen en de BN-staat blijven FP32.
`load_derived_model` reconstrueert de bestaande backbone met FrozenMXConv-buffers en
kiest bank4. De bestaande tanh/normalisatie wordt niet nogmaals toegepast.

## Nieuwe modules evaluation.py en run_eval.py

Er is alleen `check` en `evaluate`, geen trainingcommando. H/A worden uit de
bestaande exports geladen. B/C worden in nieuwe bestanden opgeslagen, herladen en
met dezelfde testbeelden gemeten. Model/BN-tensors worden voor/na gehasht.
Alle originele bestanden blijven read-only. Alleen de testdata worden geladen.

De aanvullende individuele logits/voorspellingen worden nu in NPZ bewaard en in
de feedbackbundel opgenomen. De notebook kan daardoor de tellingen zelf natellen.
Acht metricsrijen (vier paden, twee checkpoints), gepaarde voorspellingstellingen,
gewichts-/codeverschillen, waarschuwingen en bronhashes worden afzonderlijk gelogd.

## Compatibiliteit blijft gescheiden

Het eerdere gpustat-importuitstel en view->reshape in een nieuwe werkcopy van utils.py
blijven ongewijzigd. Geen nieuwe torch-, CUDA-, NumPy- of SciPy-versies. Geen AMP,
TF32-override, deterministic-algorithms-override of verandering van de datasetklasse.
Dit pakket importeert geen opgeslagen picklecode uit modelbestanden: uitsluitend
`torch.load(..., weights_only=True)` en bestaande checksumcontroles.
