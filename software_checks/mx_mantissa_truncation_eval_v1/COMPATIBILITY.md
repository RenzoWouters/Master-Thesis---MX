# Compatibiliteit versus onderzoek

Hergebruikt: dezelfde twee compatibiliteitswijzigingen in een nieuwe bronwerkcopy
van utils.py: gpustat alleen in setup_gpus importeren, en correct[:k].view(-1)
vervangen door reshape(-1). De originele auteurs-ZIP blijft intact.

Het nieuwe experiment voegt uitsluitend een codeafleiding/evaluator toe. Dat is
onderzoekscode, geen compatibiliteitsfix. Codecs en bestaande modelklassen worden
niet gewijzigd om resultaten te verbeteren of zelftests te laten slagen.

Dezelfde gebruikersomgeving (.venv-train, Python3.13, torch2.10.0,
torchvision0.25.0) is vereist; CUDA is verplicht in de productie-CLI. Geen stille
CPU-fallback. De offline-validaties gebruiken expliciet CPU en kunstmatige waarden.
Pip check rapporteert bestaande conflicten en installeert/wijzigt geen pakketten.
