# Uitgevoerde validatie

## Geslaagd: uitsluitend synthetische CPU-tests

Omgeving: Linux, Python3.13.5, PyTorch2.10.0+cpu, torchvision0.25.0+cpu.
Zie validation_report.json voor alle pakketversies. Geen echte SVHN-beelden,
gebruikerscheckpoints of GPU-evaluaties uitgevoerd. Geen optimizerupdates.

- Alle 64 E2M3-codes vergeleken met onafhankelijke rationale-getallenoracles
  voor zowel C (truncatie) als B (nearest-even vanaf de bron).
- Alle 255 eindige E8M0-schaalcodes blijven identiek; decode-tests binnen
  het eindige FP32-bereik. Bronwaarden buiten dat bereik geven een fout bij decode.
- Teken, lokale exponent, signed zero, lokale subnormals, padding, nulblokken,
  packing en voorbeelden met verschillende directe/bronafronding gecontroleerd.
- De zesbitbron blijft ongewijzigd. C werkt ook wanneer alle full-tensor- en
  elementquantizers opzettelijk als foutfuncties zijn vervangen.
- Directe modelvoorwaartse berekening versus opgeslagen H/A-export: exact gelijke
  CPU-logits in de synthetische test. CUDA-check gebruikt de bestaande expliciete
  rtol1e-4/atol1e-5-inferentietolerantie en rapporteert ook absolute afwijking.
- Lage BN-bank bewust andere synthetische waarden gegeven: A/B/C kiezen bank4,
  H bank32. A32 blijft actief. Geen wijziging van model/BN-tensors bij inferentie.
- H/A/B/C op zeven synthetische beelden, met korte laatste minibatch: volledige
  evaluator, ruwe NPZ-uitvoer en gepaarde correcte/foute voorspellingentellingen.
- Afzonderlijk CPU-proces reconstrueert C met uitsluitend de hoge MX-export en
  kunstmatige beelden. Geen lage directe export of trainingscheckpoint meegegeven;
  alle quantizers EN G(W) staan als traps. Logits exact gelijk aan de oudertest.
- Negatieve tests weigeren NaN-schalen, ongeldige codewoorden, niet-nul padding,
  signed codeopslag, verschillende bronschalen, verkeerde BN-bank, verborgen
  mastergewichten, onbekende methode, foute checksum en overschrijven.
- CLI weigert een trainingscommando, CPU-productiefallback, ontbrekende referentie,
  downloadopties en uitvoer in de bestaande referentiemap.
- Notebook uitgevoerd op opgeslagen, expliciet synthetische gegevens. De twee
  grafieken zijn visueel gecontroleerd; geen foutcellen.

Herhaal zonder iets te overschrijven:

```text
python local_validation/test_all.py <NIEUWE_testmap>
```

Met de optionele notebooktest (vereist nbformat/nbclient in de testomgeving):

```text
python local_validation/test_all.py <NIEUWE_testmap> --notebook
```

Deze interne CPU-testoptie bestaat niet in de productie-CLI.

## Controle op beschikbare echte referentiemetadata

Het nieuwe referentiecontract, de model-/codec-hashes, alle vierhonderd
metricrijen en de geselecteerde beste/laatste rollen zijn succesvol gecontroleerd
op de laatst aangeleverde feedbackbundel. De daarna vereiste vier echte
MX-exportbestanden zitten niet in de feedbackbundel. Het volledige evaluatiepad
kan daardoor hier niet op de gebruikersgewichten worden uitgevoerd.

## Niet geslaagd: productiepreflight in deze container

Een echte `run_eval.py check`-aanroep is gestopt bij pip check: de al aanwezige
MoviePy2.2.1 vereist Pillow<12, maar Pillow12.3.0 is geinstalleerd. Er is niets
geinstalleerd, verwijderd of versoepeld om dit te omzeilen. Zie
container_preflight_report.json en container_pip_check.txt.

Dit is niet de laptopomgeving. Die moet eerst de eigen CUDA-check uitvoeren.
Er is geen claim dat deze containerpreflight PASS was of dat CUDA-resultaten al
zijn verkregen. De inhoudelijke synthetische tests hierboven zijn afzonderlijk
uitgevoerd met de bestaande werkende PyTorch-CPU-imports.
