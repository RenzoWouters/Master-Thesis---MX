# Compatibiliteit versus onderzoekswijzigingen

Geen installatie of bibliotheekupgrade. De bestaande .venv-train blijft gebruikt.
Dezelfde reeds gecontroleerde utils.py-patch (optionele gpustat-import, reshape
in plaats van view) wordt alleen toegepast op een nieuwe auteurscodewerkcopy.
Het oorspronkelijke archief, de eerdere software_checks en data blijven intact.

De evaluatie gebruikt dezelfde PyTorch/CUDA-backendinstellingen als de bronrun.
Volledige bitexacte CUDA-herhaalbaarheid wordt niet verondersteld. Geen AMP,
TF32-override, torch.compile, andere testbatchgrootte of andere preprocessing.

Notebook-afhankelijkheden pandas, matplotlib en numpy zijn dezelfde als in het
werkende overzichtsnotebook. De evaluatie zelf installeert niets. pip check moet
slagen, anders wordt expliciet gestopt. Het bronbestand wordt geladen met
weights_only=True; geen willekeurige pickleklassen of onveilige deserialisatie.

Voor tests bestaat een expliciet synthetisch CPU-pad in de lokale testharness.
Dat is geen productieoptie of fallback. Geen SVHN-nauwkeurigheid wordt in de
container geclaimd. CODEWIJZIGINGEN.md beschrijft de nieuwe controleaansturing;
er is geen nieuwe kwantisatie-/trainingsmethode.
