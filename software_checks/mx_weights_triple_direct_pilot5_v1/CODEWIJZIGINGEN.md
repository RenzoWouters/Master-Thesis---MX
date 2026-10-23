# Wijzigingen ten opzichte van de bestaande directe tweeprecisieproef

Basis is `mx_weights_dual_reference100_v1` uit de aangeleverde overdracht.
Die adapter kwantiseert E2M3 en E2M1 onafhankelijk rechtstreeks uit G(W).
De latere truncation-aware adapter is niet als productiepad overgenomen.

## Behouden bestanden en gedrag

- `mx_format.py` (E2M3), `mx_e2m1.py`, `mx_adapter.py` (G(W) en algemene
  modelhulp), `test_mx.py`, `svhn_support.py`, `smoke_support.py`,
  `resume_core.py`, `SOURCE_MANIFEST.json` en de auteurs-ZIP blijven
  byte-identiek aan de directe tweeprecisiebasis.
- De werkende `mx_e4m3.py` en `test_e4m3.py` zijn byte-identiek overgenomen
  uit de dedicated E4M3-reference100-versie.
- Modelarchitectuur, dataset, voorbewerking, A32, FP32-masters/gradients/Adam,
  seed, optimizer en het volledige LR-schema blijven behouden.
- De auteurs-forward behoudt zijn loss-, backward-, dropout- en optimizer-
  sequencing; uitsluitend de drie selectorparen worden vervangen.

## Toegevoegd of uitgebreid

- `joint_adapter.py`: drie directe codecs uit dezelfde actuele G(W), drie
  actieve BN-banken, afzonderlijke directe exports per formaat.
- `joint_support.py`: negen meters, drie modi, twee recursieve
  softteachercontroles per minibatch, één gecontroleerde Adam-update,
  selectie op de hoogste precisie E4M3.
- `test_joint.py`: directe conversie tegenover truncatie onderscheiden,
  STE, onafhankelijke handmatige trainingsreplay, actieve/inactieve BN,
  gradients/Adam/RNG en drie exportmodellen.
- `run_train.py`: nieuw experiment, vast vijf-epocheindpunt, controle van
  alle drie modi en correcte pilotstatus. Exports worden op het trainingsdevice
  aangemaakt en vergeleken, zoals in het recente E4M3-pakket.
- `notebook_support.py` en de lege notebooktemplate: drie precisies,
  checkpointselectie via E4M3, correcte hard-/soft-losslabels en gemeten
  pilotduur tegenover een afzonderlijke honderd-epochprojectie.

Alle bestaande resultaten en bronpakketten blijven behouden. Historische
documentatie waarin mantissatruncatie voor deze jointproef nog aan stond,
beschrijft een ouder plan. Deze nieuwe proef volgt de expliciete keuze
van 23 oktober 2026: alle drie modi rechtstreeks, zonder truncatie.

`provenance/` bewaart de relevante oorspronkelijke bronbestanden en de
beperkte wijzigingen. `REUSE_MANIFEST.json` noemt de hergebruikte hashes.
`local_validation/VALIDATION.md` geeft de daadwerkelijk uitgevoerde controles
en hun grenzen; synthetische controle-uitkomsten zijn geen SVHN-resultaten.
