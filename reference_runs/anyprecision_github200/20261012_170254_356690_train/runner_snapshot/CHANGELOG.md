# Wijzigingslog GitHub200 v1

## Ten opzichte van het eerdere anyprecision_reference_v1 (400 epochs)

- Nieuw afzonderlijk pakket; oude map, resultaten en bronarchief onaangeroerd.
- reference_config.json: planned_epochs400->200; milestones150/250/350->100/150/180;
  naam en schema worden github200; expliciete onbekende herkomst README-resultaten.
- run_reference.py: bijpassende configcontrole, meldingen, uitvoermap en projectievelden.
- Opslagprojectie dynamisch planned_epochs+1 in plaats van vast401.
- Eindtestevaluatie schrijft een automatische vergelijking met de bevroren README-rij.
  Alleen rapportage; geen andere training, evaluatie of modelselectie.
- github_targets.json toegevoegd met bronhash en auteurswaarden.
- Documentatie aangepast; exact verschil in changes_from_reference_v1.patch.

## Ongewijzigd

Origineel Any-Precision-archief; modelarchitectuur, quantizers, lossfuncties,
optimizer- en data-modules. resume_core.py, resume_worker.py, run_pilot.py en
smoke_support.py zijn byte-identiek aan de eerdere referentieversie en de
hervatcontrole. SOURCE_MANIFEST en compatibility.patch zijn identiek.

## Bestaande wijzigingen ten opzichte van originele code

In een nieuwe bronwerkkopie: utils.py view->reshape; gpustat lazy import;
datasets/data.py SVHN lazy import. Geen nieuwe bronpatch.
Een apart uitvoerscript in plaats van origineel train.py, vaste seeds en determinisme,
volledige checkpoints met RNG, sessies, checksums, logging en timing.

## Protocolverschil dat blijft bestaan

Test alleen op een vooraf vastgelegd eindpunt, geen testset als val en geen
best-on-test-selectie. Er is dus geen claim dat het VOLLEDIGE historische
uitvoerings- en selectieprotocol van de auteurs exact wordt gereproduceerd.
Zie GITHUB_ALIGNMENT.md.

De nieuwe .sh-bronkopie krijgt expliciet LF via .gitattributes voor Git op Windows.
