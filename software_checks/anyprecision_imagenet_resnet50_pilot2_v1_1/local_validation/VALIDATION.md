# Softwarevalidatie van v1_1

Deze versie corrigeert de controle die bij de gebruiker meer dan tien minuten in native256-opwarming bleef. De oorzaak op de laptop is nog niet gemeten. Standaard wordt nu de bedoelde serial_dp4-uitvoering gecontroleerd, met afzonderlijke deadline en voortgang.

Op Linux/Python 3.12 en PyTorch 2.10.0+cpu zijn geslaagd:

- Alle eerdere onafhankelijke replica-, BatchNorm-, gradient- en checkpointzelftests.
- Callback aan/uit: exact dezelfde parameters, gradients, buffers, SGD-momentum, teachers, metrics en RNG in vier kleine synthetische gevallen.
- Watchdog: normale/foute exit, stille en pratende timeouts, onvolledige regel, stdout gesloten zonder exit, onderbreking en kindproces-opruiming.
- Statusgates: TIMEOUT overschrijft een te late kind-PASS; cleanup-fout geeft FAIL; OOM blijft apart; nieuwste passende mislukte check blokkeert oudere PASS bij automatisch zoeken.
- Volledige openbare check-opdracht start standaard alleen serial_dp4. De CPU-only omgeving levert daarbij terecht FAIL wegens ontbrekende CUDA; volledig log, rapport, fasebestand en feedback-ZIP blijven beschikbaar.
- Notebook blijft onuitgevoerd. Mockscenario's voor SKIPPED, TIMEOUT en PASS leveren alleen echte meetstaven voor PASS.
- Trainingsconfiguratie, datasetcode, checkpointhelpers en oorspronkelijke bron-ZIP zijn exact behouden.

De JSON-bestanden in deze map horen bij v1_1. Eerdere, niet opnieuw uitgevoerde aanvullende v1-controles staan herkenbaar onder prior_v1; dat zijn historische softwaremetingen. Alle getallen over fixtures zijn testgegevens, geen ImageNet-resultaten.

Windows, de RTX 5070 Laptop GPU en echte ImageNet-training zijn hier niet uitgevoerd. De nieuwe check op de laptop blijft noodzakelijk. Er is geen bitexacte CUDA-garantie.
