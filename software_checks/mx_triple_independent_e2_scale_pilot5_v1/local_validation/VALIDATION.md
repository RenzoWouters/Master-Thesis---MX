# Lokale validatie van het nieuwe pakket

## Uitkomst

De numerieke featurechecks, integratie van de runner en notebookcontrole zijn geslaagd. Dit betreft **synthetische CPU-softwaretests**. Er is geen SVHN-dataset geladen, gedownload of getraind. De geteste modeltraining gebruikt drie kunstmatige beelden en levert geen onderzoeksaccuracy of GPU-tijd op.

- `feature_check_report.json`: 21 inhoudelijke controles van codec, absolute schaal, gradient, cache, initialisatie, projectie en exports, met herstel van RNG/backend.
- `runner_integration_report.json`: werkelijke `main/check`, `main/run` en `main/resume`; vijf kleine epochs, stop na twee en hervatten tot vijf. Model, Adam, scheduler, RNG en samplevolgorde komen exact overeen op CPU. Een oud checkpointcontract wordt vóór updates afgewezen.
- `check_entry_report.json`: precieze aanpassingen van de CPU-testomgeving en uitkomst van de werkelijke check-entry.
- `notebook_validation_report.json`: consistent lezen van volledige, gedeeltelijke en hervatte fixtures; afwijzen van tegenstrijdige metrics/schalen/selectie/referenties; alle figuren en notebookcellen uitgevoerd in tijdelijke werkruimte. De geleverde notebook blijft onuitgevoerd.
- `FINAL_VALIDATION.json`: identiteit van de gevalideerde runtimebestanden, behoud van referenties en afbakening van deze controle.

De accuracy-, schaalverdelings- en opslagfiguren zijn daarnaast visueel gecontroleerd. De synthetische figuren worden niet als resultaten meegeleverd. Accuracyassen omvatten alle getoonde waarden; de opslagfiguur scheidt schaalmetadata van de payload voor de zes MX-lagen.

## Expliciete beperkingen van de CPU-check

De echte CLI vereist de bestaande Windows/Python 3.13/CUDA-omgeving. De lokale test gebruikt Python 3.12 met PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu. Alleen de testharness past de runtimekeuze en `.cuda()`-routing aan.

De oorspronkelijke auteursmodelinitialisatie geeft in de lokale CPU-omgeving een andere hash dan de historische Windows-run. De test maakt daarom in een afzonderlijk proces een ongewijzigd auteursmodel als CPU-orakel. Alleen in het geheugen van de test wordt die verwachte hash gebruikt. De productiecontrole op de historische Windows-hash en de opgeslagen onderzoeksreferenties blijven intact. De specifieke oorzaak van het verschil tussen de omgevingen is hiermee niet vastgesteld.

De eerste poging stopte vóór de codeccontroles omdat de geïsoleerde CPU-venv geen `pip`-module bevat. De definitieve testharness registreert daarom de geïnstalleerde distributies via `importlib.metadata` en vermeldt expliciet dat `pip check` hier niet is uitgevoerd. De productiecode voert de bestaande dependencycontroles wel uit. Er zijn geen pakketten geïnstalleerd of geüpgraded.

Het opzettelijk ongeldige checkpoint geeft in de fixture `FAIL`; dat is het verwachte resultaat van de afwijzingstest. Een gewone stop op epoch 2 krijgt `STOPPED_AT_EPOCH_BOUNDARY`, nooit ten onrechte een voltooide pilotstatus.

Deze CPU-uitkomsten bewijzen geen CUDA-uitvoerbaarheid of bitexacte CUDA-hervatting. Voer daarom eerst de normale `check`-opdracht uit in de bestaande trainingsomgeving en daarna de vijfepochrun.

## Reproduceerbare tests

De productietests staan in `test_learned_scale.py` en worden door `run_train.py check` uitgevoerd. De aanvullende scripts hier zijn bedoeld voor softwarevalidatie in een geschikte CPU-omgeving:

```text
python local_validation/test_runner_integration.py run NIEUWE_LEGE_UITVOERMAP
python local_validation/test_notebook_support.py PAD_NAAR_PAKKET --fixture PAD_NAAR_SYNTHETISCHE_CONTINUOUS5_RUN --out NIEUWE_NOTEBOOK_TESTMAP
```

De integratietest stelt zijn kleine fixture expliciet samen; hij wijzigt het productieprotocol en de ruwe historische referenties niet. De notebooktests muteren uitsluitend wegwerpbare kopieën om foutdetectie te controleren. Er is geen wetenschappelijke dataset nodig.
