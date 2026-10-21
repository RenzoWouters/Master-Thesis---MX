# Compatibiliteit versus onderzoek

Bestaande Python3.13 / torch2.10 / torchvision0.25-omgeving ongewijzigd.
Geen nieuwe dependencies, native extensies of automatische upgrades.
Bestaande scipy/SVHN-loader, pandas/matplotlib/ipykernel voor notebook.

upstream/Any-Precision-DNNs-master.zip is byte-identiek. De bestaande beperkte
utils.py-compatibiliteitspatch wordt alleen in de nieuwe sessiewerkkopie
aangebracht; oorspronkelijke bestanden blijven ongewijzigd. SOURCE_MANIFEST
komt uit de geslaagde vorige runner. svhn_support.py, smoke_support.py en
resume_core.py zijn byte-identiek hergebruikt.

De tijdelijke deterministische controle en cuBLAS-workspacevariabele zijn
alleen actief tijdens `check`, in dat eigen proces. De trainingsopdracht
herstelt het vaste cuDNN-benchmarkbeleid. Geen bitexacte GPU-hervatgarantie.

De cache-mutatiecontrole gebruikt Tensor._version, getest in torch2.10.0; het
is geen nieuwe optimizer- of kwantisatiebewerking. De runner accepteert geen
stille CPU-productiefallback. De lokale tests gebruiken een expliciete CPU-
vervanging van Tensor.cuda, uitsluitend in local_validation, met synthetische
gegevens. Die tests zijn geen echte SVHN-training of nauwkeurigheidsresultaten.

Nieuwe code en contracten voor code-truncatie en gedeelde bron zijn
ONDERZOEKSWIJZIGINGEN, niet als louter compatibiliteit gemaskeerd.
