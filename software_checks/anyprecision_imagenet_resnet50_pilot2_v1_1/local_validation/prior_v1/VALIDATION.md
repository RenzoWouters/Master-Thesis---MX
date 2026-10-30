# Vooraf uitgevoerde softwarevalidatie

Omgeving: Linux, Python 3.12.14, PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu. De tijdelijke testomgeving staat los van de trainingsomgeving op de laptop. Alle neural-network-inputs in deze controles waren kleine synthetische fixtures. Er is geen echte ImageNet-training uitgevoerd.

## Geslaagd

- **Onafhankelijke replica-oracle:** vier expliciete modelkopieën, samengevoegde outputs en globale losses versus de seriële implementatie. Gradients, parameters na SGD en alle buffers kwamen exact overeen in fixtures met 8, 13, 5 en 1 voorbeeld. De gevallen met 13 en 5 controleren oneven verdelingen. Ook native-modus, recursieve teachers, de ene optimizerupdate en bevroren evaluatie zijn gecontroleerd.
- **BatchNorm-onderscheid:** de tests tonen dat gewone viermaal-accumulatie en één grotere native batch andere running-statistieken opleveren. De gekozen uitvoering behoudt de bufferstaat van replica 0 per precisie.
- **Echte auteursarchitectuur:** ongewijzigde `resnet50q`, vier beelden van 64×64, alle vijf precisies, 20 fysieke forward/backwardpasses en één SGD-update met LR 0,5 / momentum 0,9 / weight decay 1e-5. Alle vereiste gradients, modelwaarden en momentumwaarden waren eindig en FP32. Alle 265 BN-banken verhoogden hun teller exact eenmaal. Er zijn 25.769.512 modelparameters. Evaluatie wijzigde geen BN-buffers.
- **Checkpointcontinuïteit:** twee opeenvolgende kleine updates kwamen exact overeen met één update, opslaan, veilig laden met `weights_only=True`, RNG-herstel en de tweede update. Python-, NumPy-, Torch- en loader-generatorstaat, parameters, BN, gradients en SGD-momentum kwamen overeen.
- **LR-schema:** 80 scalaire schedulerstappen (geen 80 trainingsepochs), inclusief de drie dalingen en herstel vóór een daling.
- **Foutdetectie:** NaN-loss, ongeldige accuracytellers en niet-eindige model-/optimizerstaat werden geweigerd.
- **Data:** read-only inventaris, juiste klassevolgorde, weigeren van subsets/onvolledige data en afwijkende klassen. Echte PIL- en torchvision-transforms op kleine afbeeldingen leverden RGB FP32 224×224 op. De batch-256-gate, laatste gedeeltelijke batch en generatorherstel zijn getest.
- **Spawn-worker:** dezelfde loaderfixture werkte met nul workers en met één afzonderlijk proces onder de startmethode `spawn`. Dit test de overdraagbaarheid van de loader/workerfunctie op Linux en is geen daadwerkelijke Windows-test.
- **CLI-gates:** zonder CUDA eindigt `check` terecht als FAIL en start geen CPU-training. Ontbrekende ImageNet-data worden geweigerd. `run` weigert een niet-geslaagde check. In alle gevallen bleven een leesbaar rapport, onuitgevoerde notebook en geldige feedbackbundel beschikbaar.
- **Notebook:** JSON/codecellen gecontroleerd; check-only/OOM en trainingfixtures geanalyseerd; grafieken visueel geïnspecteerd. Geen verzonnen nulmeting bij OOM. Laatste checkpoint en beste W32/A32-checkpoint houden de vijf precisies bij dezelfde epoch. Ontbrekende epoch-1-timing na hervatten wordt expliciet behandeld.
- **Provenance:** oorspronkelijke ZIP en bronbestanden ongewijzigd. Feedbackbundels sluiten modelcheckpoints en datasetbestanden uit. De notebookkopie krijgt de werkelijke resultatenmap, ook wanneer VS Code vanuit een andere werkmap start.

De bijbehorende kleine JSON-rapporten staan in deze map. Eventuele getallen over fixtures beschrijven softwarecontroles en zijn geen ImageNet-accuracy, trainingsduur of GPU-geheugenbewijs. Visuele mockdata en mockfiguren zijn niet in het testpakket opgenomen.

## Nog op de laptop uit te voeren

`run_train.py check` moet de volledige trainingsstappen bij 224×224 op de RTX 5070 Laptop GPU meten. Alleen die check kan een passende GPU-PASS opleveren. Daarna kan `run` met echte ImageNet-data twee epochs uitvoeren. CUDA-kernels, Windows en de volledige dataset zijn hier niet uitgevoerd; er is geen bitexacte CUDA-reproductieclaim.

## Zelftests opnieuw uitvoeren

Voor een CPU-only controle met dezelfde PyTorch/torchvision-versies:

```text
python run_train.py selftest
```

Deze opdracht maakt een afzonderlijke resultatenmap. Haar PASS wordt niet geaccepteerd als GPU-checkbewijs voor `run`.
