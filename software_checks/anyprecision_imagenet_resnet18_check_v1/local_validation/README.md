# Lokale softwarevalidatie

Deze bestanden documenteren de controle van het pakket vóór aflevering.

- De model- en engineproeven zijn werkelijk op CPU uitgevoerd met PyTorch 2.10.0+cpu.
- De orchestrationproeven gebruiken gesimuleerde CUDA-antwoorden; de korte tests van procesafsluiting starten echte subprocessen.
- De notebook is met uitdrukkelijke mockrapporten gecontroleerd; de mockdata en figuren zijn niet in dit pakket opgenomen.
- De volledige publieke `check`-opdracht is op de CPU-host uitgevoerd. Alle CPU-zelftests slaagden; daarna volgde de verwachte FAIL omdat CUDA op deze host ontbreekt. Het foutverslag en de feedbackbundel zijn correct aangemaakt.

**Er is hier geen ResNet-18-GPU-meting verricht.** De meegeleverde validatie-PASS slaat op de softwarecontrole, niet op GPU-geheugen, snelheid of ImageNet-accuracy. De gebruiker voert daarvoor de nieuwe `check` op de laptop uit. De notebook blijft onuitgevoerd.
