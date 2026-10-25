# Lokale softwarevalidatie

Deze rapporten gaan over technische correctheid. Er is hier geen echte SVHN-training uitgevoerd en geen CUDA-accuracy of GPU-snelheid gemeten.

- `training_integration_report.json`: een echt getrainde synthetische CPU-fixture met drie beelden en twee minibatches per epoch leverde een volledig oudercheckpoint na epoch 50. De nieuwe runner is daarmee getest voor 51–55, stoppen na 52, exact hervatten naar 55, gecontroleerd stoppen, compacte exports, ongewijzigde invoer en foutgevallen.
- `check_entry_validation_report.json`: de nieuwe technische checkfunctie met de oorspronkelijke codec- en gradienttests, strikt ouderherstel en 14 nieuwe freezecontroles.
- `freeze_checks.json`: afzonderlijke controle dat gewichten verder veranderen, theta exact vast blijft, een kleine theta-afwijking wordt geweigerd en de E4M3-bronschaal dynamisch blijft.
- `notebook_validation_report.json`: 42 analysecontroles, alle codecellen en zeven figuren; ook de uitvoer van de echte kleine CPU-integratie is gelezen. De geleverde notebook blijft leeg.
- `code_review.json` en `CODE_REVIEW.md`: afzonderlijke controle van de herkomst en trainingslogica.

De productie-CLI biedt uitsluitend CUDA en de echte, vastgelegde epoch-50-referentie. Kleine synthetische data worden alleen door de tests geïnjecteerd; ze zijn geen trainingsoptie voor de gebruiker. Voor de daadwerkelijke laptoprun blijft `run_train.py check` de eerste stap.

De volledige oorspronkelijke runtime is onveranderd meegeleverd onder `reference_runtime`. De nieuwe proef gebruikt hetzelfde quantisatie-/trainingspad; alleen de wetenschappelijke vertakking, freeze, controles en rapportage zijn toegevoegd.
