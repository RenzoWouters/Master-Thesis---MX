# Uitgevoerde validatie

CPU/Linux, Python 3.13.5, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu,
NumPy 2.3.5, SciPy 1.17.0. Geen nieuwe dependencies geinstalleerd.

Numerieke en modeltests (test_mx.py): PASS.
Alle 64 E2M3-codes, signed zeros, 4287 onafhankelijke oraclewaarden,
elk afrondingsmidpunt met zijn buren, alle 255 eindige E8M0-schalen,
padding, scheiding tussen outputkanalen, extreme waarden, rejecties,
pack/unpack, identieke W32-voorbewerking, de bedoelde surrogate-gradient,
veilige nulgewichten, oude vijf-bitlus afwijzen en herladen zonder de
FP32-mastergewichten van de zes MX-lagen zijn gecontroleerd.

Integratietest (local_validation/test_evaluator.py): PASS.
De echte evaluate-functie is uitgevoerd met expliciet synthetische lokale
fixtures (8 beelden, willekeurig model, twee checkpointrollen). Alleen dit
testproces vervangt de datasetlevering en datasetgroottecontrole. De productie-
CLI heeft die bypass niet en vereist de echte 26032 testbeelden, een volledige
referentierun en CUDA. Dit zijn GEEN meetresultaten op SVHN.

De export is ook in een nieuw CPU-proces herladen dat alleen de broncodes,
auxiliaire staat en een invoerprobe kreeg, NIET het referentiecheckpoint.
De outputs waren exact gelijk. Het resultatennotebook is uitgevoerd op de
opgeslagen synthetische rapporten; de grafiek is visueel gecontroleerd.

Herhaalbaar met nieuwe uitvoermappen (nooit bestaande resultaten overschrijven):

```
python test_mx.py <nieuwe_numerieke_testmap>
python local_validation/test_evaluator.py <nieuwe_integratietestmap>
```

Geen echte gebruikerscheckpoint, echte SVHN-data of CUDA-evaluatie was hier
beschikbaar. Er is nog geen MX-accuracy of MX-trainingstijd gemeten. De
productiecheck moet op de laptop slagen voordat de echte evaluatie plaatsvindt.

De volledige CLI-preflight in deze container wordt afzonderlijk vastgelegd.
Een bestaand MoviePy/Pillow pip-conflict in deze container is geen oordeel over
de gebruikersomgeving. Niets wordt automatisch aangepast om die check te passeren.
