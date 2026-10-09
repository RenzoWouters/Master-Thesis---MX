# Lokale controle door de assistent

Dit zijn geen resultaten van de laptop van Renzo.

Het meegeleverde `run_smoke.py` is uitgevoerd op Linux, Python 3.13.5,
PyTorch 2.10.0+cpu en NumPy 2.3.5. CUDA was hier niet beschikbaar.
Met de standaard twee batches van twee, vijf precisies en twee verse
herhalingen gaf de test `PASS`, zonder opgevangen Python-waarschuwingen.
Model- en optimizertoestand, losses en evaluatie-uitvoer kwamen binnen de
herhalingen exact overeen; checkpoint-roundtrip slaagde eveneens.

Deze CPU-controle vervangt niet de volledige Windows/CUDA-modeltest.
De installatie op de laptop is anders, waaronder NumPy 2.5.2 en de CUDA-build.

De bron- en runnercontrolegetallen staan in het bijgevoegde rapport. Het grote
synthetische checkpoint is bewust niet meegeleverd. Er zijn geen datasets
gedownload en geen accuracy- of hardwarewinsten vastgesteld.
