# Softwarevalidatie vóór aflevering

De publieke CPU-zelftest slaagde met PyTorch 2.10.0+cpu. De model- en replica-oracles zijn ongewijzigd overgenomen uit het geslaagde ResNet-18-controlepakket. Kleine beeldfixtures controleren selectie, transformaties, aantallen en datasetbehoud. Een echte kleine spawnworker is getest met een metadata-collator; Tensor-IPC is lokaal niet beproefd omdat de sandbox de daarvoor gebruikte Unix-socket blokkeert.

Een werkelijk klein CPU-model en SGD-state zijn opgeslagen en via `torch.load(weights_only=True)` teruggelezen. Gesimuleerde GPU-antwoorden controleren de begrenzing op 200 updates, 5.000 validatiebeelden, behoud van gedeeltelijke metingen en de eindstatus. De notebook is met expliciete mockrapporten uitgevoerd voor softwarecontrole; zulke mockdata en figuren zijn niet als experimentele resultaten opgenomen. De afgeleverde notebook blijft onuitgevoerd.

De procesboomproeven zijn op Linux echt uitgevoerd, met aanvullende Windows-routingmocks. De Windows-API zelf is hier niet uitgevoerd; de publieke CPU-zelftest gebruikt op Windows de eigen native procesboomproeven voordat de echte pilot kan starten.

De publieke `run`-opdracht is ook zonder ImageNet getest en geeft de verwachte heldere FAIL vóór GPU- of trainingswerk, met een feedbackbundel. Er is lokaal geen echte ImageNet-GPU-training uitgevoerd. Het bestand `approved_synthetic_check.json` in de pakketmap is daarentegen de werkelijk aangeleverde eerdere synthetische GPU-PASS op de laptop.
