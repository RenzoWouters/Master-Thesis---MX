# Compatibiliteit apart van onderzoekswijzigingen

De bron-ZIP is dezelfde gebruikersupload als voor de eerdere geslaagde runs.
Er wordt geen GitHub-versie opgehaald. SOURCE_MANIFEST.json verifieert de ZIP
en ieder oorspronkelijk bestand. prepare_source maakt een nieuwe werkkopie.

Alleen de bestaande utils.py-compatibiliteitspatch wordt opnieuw toegepast:
gpustat niet bij import vereisen en reshape gebruiken voor de accuracy-helper.
Zie compatibility.patch. Dit is dezelfde patch als in de vorige SVHN-runs.
Geen nieuwe wijziging in modelarchitectuur, datasetloader of optimizer.py.

De aanpassing van de drie mode-selectors, nieuwe E2M1-codec en BN-moduskeuze
zijn ONDERZOEKSwijzigingen, geen compatibiliteitspatches. Zie
CODEWIJZIGINGEN.md, forward_selectors.patch en TRAINING_CHANGE_MANIFEST.json.

Productie: bestaande Python3.13, torch2.10.0, torchvision0.25.0 en CUDA.
Geen nieuwe library vereist. SciPy voor de bestaande .mat-loader; pandas,
matplotlib/ipykernel voor het notebook waren eerder al in de werkende omgeving
geinstalleerd. Controle via pip check, geen automatische updates of installatie.

De container gebruikt CPU en andere Python/NumPy-patchversies. De offline
tests zijn geen garantie van GPU-bitexactheid. Het preflightresultaat van de
laptop is leidend voor de werkelijke run. Het bekende deprecationbericht over
pin_memory(device=...) wordt niet stil weggefilterd en verandert geen data.
