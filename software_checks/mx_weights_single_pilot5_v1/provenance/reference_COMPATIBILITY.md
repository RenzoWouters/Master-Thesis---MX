# Compatibiliteit en hergebruik

Er worden geen nieuwe compatibiliteitsaanpassingen aan het auteursmodel of de
trainingsfunctie ingevoerd. De bestaande Windows/Python/PyTorch-omgeving blijft
ongewijzigd; er is geen automatische installatie of upgrade. pip freeze en
pip check worden per aanroep geregistreerd. Python 3.13, torch 2.10.0 en
vision 0.25.0 en CUDA worden gecontroleerd. De volledige gebruikte versies staan
in het runrapport; de patchversie van Python wordt geregistreerd, niet verzwegen.

Het auteursarchief blijft byte-identiek. Per run wordt het naar een eigen nieuwe
source_workcopy uitgepakt. Alleen de al gebruikte twee wijzigingen in utils.py:
1. gpustat-import uitstellen tot setup_gpus; de runner kiest CUDA expliciet;
2. correct[:k].view(-1) -> correct[:k].reshape(-1) voor niet-aaneengesloten tensors.

Die staan in compatibility.patch, gescheiden van de onderzoekswijziging
train_extra -> train. Modelbestanden, datasets/data.py, datasets/svhn.py,
optimizer.py, losses.py en de oorspronkelijke forward() blijven intact.
De forward() wordt uit de originele syntaxisboom geladen; de buitenste
train.py-CLI/GPU-selectie/tensorboardX worden niet uitgevoerd.

svhn_support.py, resume_core.py en smoke_support.py zijn byte-identiek aan het
geteste train-only-pakket. FULLRUN_CHANGE_MANIFEST.json geeft hun hashes.
De epochlus, restore_timing_sidecar, dependencies en verify_unchanged behouden
dezelfde AST. De oude resume_core/schema wordt hergebruikt, maar het volledige
configuratiecontract heeft een nieuwe schemawaarde: onbedoeld hervatten vanaf
een pilot- of train+extra-checkpoint wordt geweigerd.

Nieuwe wrapper- en rapportagefuncties: vast eindpunt 100, nieuwe outputmap,
nieuwe notebooknaam, expliciete voltooiingscontrole en LR-overzicht achteraf.
Geen wijziging van optimizerupdates, numerieke kwantisatie, seed of backend.
Geen nieuwe AMP, compile, TF32-override, grotere batch of parallelle datalading.

De bestaande pin_memory-waarschuwingen worden niet onderdrukt. Hun eventuele
verschijning wordt geregistreerd. De synthetische lokale tests draaien op Linux
CPU; GPU-compatibiliteit en echte SVHN-training moeten op de laptop slagen.
Er is geen claim van bitexacte CUDA-herhaling. Ter verificatie geraadpleegd:
PyTorch 2.10 Reproducibility, https://docs.pytorch.org/docs/2.10/notes/randomness.html
PyTorch optim, https://docs.pytorch.org/docs/main/optim.html#how-to-adjust-learning-rate
De werkelijk gebruikte scheduler is daarnaast met torch 2.10.0 lokaal getest.
