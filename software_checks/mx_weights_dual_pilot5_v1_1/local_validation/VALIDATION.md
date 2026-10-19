# Validatie van v1.1

Alleen synthetische CPU-tests; geen SVHN-data of echte training gebruikt.
Deze omgeving heeft PyTorch 2.10.0+cpu en geen CUDA-GPU. De oorspronkelijke
laptop-AssertionError is hier dus niet op dezelfde GPU gereproduceerd.

## Daadwerkelijk opnieuw uitgevoerd

- Bestaande E2M3- en nieuwe E2M1-codec-, STE- en modeltests.
- Manuele tweemoduslus tegenover auteurslus onder beide testbeleidsinstellingen:
  logits, geaccumuleerde gradients, volledige modelstaat, Adamstaat en
  RNG-toestanden exact gelijk op CPU.
- Vijf synthetische mini-epochs zonder registratie tegenover volledige runner:
  model/optimizer/scheduler/RNG/metrics exact gelijk.
- Hervatten 2+3 tegenover 5 mini-epochs, inclusief dropout en twee BN-banken.
- Controle van stopbestand, verkeerde contracten, overschrijven en corruptie.
- LR-mijlpalen en selectie op het eerste E2M3-maximum.
- Vier geselecteerde exports in nieuw proces herladen zonder masters.
- Ongewijzigde notebook uitgevoerd op synthetische rapporten, zonder training.
- Nieuwe negatieve tests: echte tensorafwijking, foutieve shape, NaN,
  onjuiste BN-teller en afwijkend RNG-verbruik worden niet geaccepteerd.
- Deterministische backendflags worden bij een opzettelijke exception hersteld.
- Alle Python-bestanden compileren; configuratie en run_epochs AST-identiek
  aan v1; productiecodecs/model/data/loss/optimizer-ondersteuning byte-identiek.

Alle bovenstaande tests zijn geslaagd. Details: validation_report.json en
check_fix_helper_tests.json. De eerdere v1-validatie staat alleen als
historische herkomst in provenance/v1_validation.md.

## Nog vereist op de laptop

Een volledige `check` met CUDA, inclusief de deterministische vergelijking,
blijft noodzakelijk. Als die faalt, bevat het nieuwe run_report.json nu de
betrokken tensors en afwijkingsgrootten. Er wordt geen GPU-PASS of bewezen
oplossing van iedere mogelijke fout geclaimd. De productie-training wordt
niet stilzwijgend deterministisch gemaakt.

## Volledige CLI-preflight in deze container

De echte v1.1-opdracht `run_train.py check` is geprobeerd. Pakket- en
bronverificatie slaagden; de check stopte bij het reeds aanwezige
MoviePy/Pillow-afhankelijkheidsconflict, voordat CUDA-tests konden draaien.
Geen dependencies zijn veranderd of controles omzeild. Zie
container_preflight_report.json en container_pip_check.txt. Dit is niet de
laptopomgeving; de gedeelde laptoplog meldt juist een geslaagde pip check.
