# Compatibiliteit, hergebruik en open controles

## Overgenomen voorzieningen

`resume_core.py` en `smoke_support.py` zijn byte-identiek overgenomen uit het
aangeleverde `anyprecision_reference_github200_v1(1).zip`. De nieuwe runner
gebruikt onder meer Log, json_write, prepare_source, de toestandchecks en de
checkpoint/RNG-functies. Hij roept de ResNet-specifieke train_batch NIET aan.
Hij roept ook legacy.configure NIET aan, omdat dat de backendinstellingen
zou wijzigen ten opzichte van de auteurscode. Zie REUSE_MANIFEST.json.

## Patch uitsluitend op de nieuwe werkkopie

De reeds gecontroleerde `utils.py`-patch blijft:

1. `gpustat` wordt pas in setup_gpus geimporteerd; de runner kiest CUDA expliciet.
2. `correct[:k].view(-1)` wordt `correct[:k].reshape(-1)` zodat de top-k-telling
   met de huidige tensorstrides werkt. De definitie van accuracy verandert niet.

`datasets/data.py` krijgt hier GEEN lazy-SVHN-patch: SVHN is immers de bedoelde
workload. De oorspronkelijke `datasets/svhn.py` blijft volledig ongewijzigd.
Alle patches raken alleen `source_workcopy` in de nieuwe sessiemap. Iedere run
bewaart een volledige hashinventaris en een `compatibility.patch`.

## Afhankelijkheden en download

De oorspronkelijke SVHN-loader importeert scipy.io voor .mat-bestanden.
SciPy moet dus werkelijk aanwezig en compatibel zijn. Alleen indien nodig:
SciPy 1.17.0 installeren met --no-deps en --only-binary=:all:, daarna pip check.
De runner installeert NOOIT zelf pakketten en upgradet PyTorch/NumPy niet.
Pandas, Matplotlib en ipykernel zijn optioneel voor de runner, nodig voor het
notebook. Reeds werkende versies worden niet vervangen.

De datadownload is een IO-adapter: eerst de oorspronkelijke officiele URL,
zo nodig HTTPS op dezelfde host. Iedere mislukte poging wordt gelogd. Er zijn
geen alternatieve datasets, mirrors of uitgezette certificaatcontroles.
Publicatie gebeurt pas na controle tegen de MD5 uit de originele broncode.
Een bestaand fout databestand wordt NIET automatisch overschreven of verwijderd.

## Extern gecontroleerde compatibiliteitsinformatie

Deze bronnen zijn alleen gebruikt voor softwarecompatibiliteit, niet om het
onderzoekrecept uit de aangeleverde bestanden te vervangen:

- SciPy 1.17.0 release notes: Python 3.11-3.14 en NumPy vanaf 1.26.4.
  https://docs.scipy.org/doc/scipy/release/1.17.0-notes.html
- SciPy 1.17.0 distributies / Windows CPython 3.13:
  https://pypi.org/project/scipy/1.17.0/
- PyTorch 2.10 reproducibility: cuDNN benchmarking en niet-deterministische
  kernels verhinderen een algemene bitexacte herhaalbaarheidsgarantie.
  https://docs.pytorch.org/docs/2.10/notes/randomness.html

## Wat nog op de laptop gecontroleerd moet worden

Echte bestanden/downloadbereikbaarheid, werkelijke CUDA-uitvoering,
VRAM-gebruik, Windows-specifieke effecten en totale epochtijden. De meegeleverde
CPU-validatie is GEEN voltooide GPU-proef of netwerkresultaat.
