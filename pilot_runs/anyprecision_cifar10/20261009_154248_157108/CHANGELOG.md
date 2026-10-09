# Wijzigingen ten opzichte van de aangeleverde Any-Precision-ZIP

## Origineel behouden

`upstream/Any-Precision-DNNs-master.zip` is byte-identiek aan de gebruikersbijlage.
Per run wordt een NIEUWE bronwerkkopie uitgepakt en gecontroleerd. Een bestaande
codebase, .venv, .venv-train of experiment001-004 wordt niet aangepast.
`SOURCE_MANIFEST.json` bevat de oorspronkelijke archief- en bestandshashes.

## Compatibiliteit / afhankelijkheden (niet de quantisatiemethode)

In de bronwerkkopie worden uitsluitend twee bestanden veranderd:

1. `utils.py`: `correct[:k].view(-1)` -> `correct[:k].reshape(-1)`.
2. `utils.py`: `import gpustat` van moduleniveau naar binnen `setup_gpus()`.
3. `datasets/data.py`: `from .svhn import SVHNMIX` van moduleniveau naar binnen
   de SVHN-tak van `get_dataset()`. Daardoor hoeft SciPy niet geinstalleerd te
   worden voor CIFAR-10. De CIFAR-10-loader en beeldtransformaties veranderen niet.

Punten 1-2 waren al aanwezig in de geslaagde smoke test; punt 3 is nieuw.
Elke run bevat de exacte volledige `compatibility.patch` en `source_manifest.json`.
De tussenstap van de hergebruikte modelvoorbereiding blijft apart bewaard in
`compatibility_model.patch` en `source_manifest_model_stage.json`.

## Nieuw uitvoerprogramma / uitvoeringsvoorzieningen

- `run_pilot.py` is nieuw: data, subsets, training/evaluatie, metrics en controles.
- `smoke_support.py` is een byte-identieke kopie van de geslaagde gecorrigeerde
  `run_smoke_cuda_fix.py`. Het wordt als helper geimporteerd; zijn eigen main loopt niet.
- Het oorspronkelijke `train.py` en de shell-launchers worden niet aangeroepen.
- Apparaatselectie, logging in CSV/JSON in plaats van TensorBoard, vaste RNGs en
  deterministische berekeningen staan in de nieuwe runner, niet in de quantizer.
- Adam.zero_grad(set_to_none=False) is expliciet zoals in de vorige smoke test.
  Alle precisies worden elke minibatch gebruikt; geen gradientaccumulatie over batches.
- De runner kiest het datasetpad via de bestaande data_paths-dictionary.
- De oude downloader wordt niet gebruikt: de nieuwe downloader behoudt TLS,
  controleert het bekende MD5 van het officiele archief en elk batchbestand,
  pakt alleen benoemde normale batchbestanden uit in een nieuwe stagingmap en
  weigert corrupte bestaande bestanden te overschrijven. SHA-256 wordt ook gelogd.
  Dit verandert de beelden/labels niet. Geen `batches.meta` nodig voor deze loader.

## Bewuste protocolafwijkingen (GEEN compatibiliteitsfixes)

- 3 in plaats van 200 epochs; batch32 in plaats van de oorspronkelijke default128.
- Vaste kleinere trainings- en validatiesubsets. Validatie uit de officiele trainset,
  NIET het oorspronkelijke `split='val'`-pad dat de officiele testset opent.
- Validatiemeting voor de eerste update toegevoegd (epoch0).
- Laatste checkpoint behouden in plaats van beste checkpoint selecteren.
- Geen automatic benchmarking; geen AMP/TF32; num_workers=0, pin_memory=False.
- Gewone labelscores naast distillatieloss toegevoegd, uitsluitend als diagnostiek.

Deze keuzes maken de pilot geen exacte reproductie van paperresultaten. De
architectuur, quantizers, STE en recursieve distillatiemethode worden behouden.
Lees PROTOCOL.md voor de volledige experimentele afbakening.
