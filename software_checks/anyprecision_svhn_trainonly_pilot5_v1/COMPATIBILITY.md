# Compatibiliteit en hergebruik

De werkende Python/PyTorch/torchvision-omgeving wordt niet aangepast of opnieuw
geinstalleerd. pip freeze en pip check worden per aanroep gelogd. SciPy is nodig
voor de oorspronkelijke .mat-loader en was aanwezig in de vorige feedbackbundel.
Bij ontbrekende pakketten stopt de run; geen automatische installatie of upgrade.

resume_core.py en smoke_support.py zijn byte-identiek aan het vorige SVHN-pakket.
Hun oorspronkelijke herkomst uit de gecontroleerde ResNet-runner staat in
REUSE_MANIFEST.json. ResNet-specifieke train_batch en legacy.configure worden niet
gebruikt. Het vorige SVHN-backendbeleid wordt behouden.

De ongewijzigde upstream-ZIP wordt per run naar een eigen nieuwe source_workcopy
uitgepakt. Alleen de eerdere twee aanpassingen in utils.py blijven:
1. gpustat-import uitstellen tot setup_gpus; de runner kiest CUDA expliciet.
2. correct[:k].view(-1) -> correct[:k].reshape(-1) voor niet-aaneengesloten tensors.

Modelbestanden, datasets/data.py, datasets/svhn.py, optimizer.py, losses.py en de
originele forward() worden niet aangepast. De keuze 'train' is een bestaande
splitoptie van de originele loader, geen compatibiliteitswijziging. Ze staat
apart als onderzoekswijziging beschreven in PROTOCOL.md en trainonly.patch.

De veilige downloader is dezelfde als in het vorige pakket, maar de lijst vereiste
bestanden bevat alleen train/test. Bestaande juiste bestanden worden hergebruikt.
Onjuiste bestaande bestanden worden niet verwijderd of overschreven. Downloaden
vereist --download en gebeurt in staging met controles volgens de originele MD5.

Echte Windows/CUDA-uitvoering, ruwe SVHN-bestanden, nauwkeurigheid en looptijd moeten
op de laptop worden gecontroleerd. Synthetische CPU-tests zijn geen GPU- of
onderzoeksresultaten. Zie local_validation/VALIDATION.md voor de uitgevoerde tests.
