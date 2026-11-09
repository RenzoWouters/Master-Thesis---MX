# Technische validatie van de voortzetting tot epoch 100

Alle numerieke uitvoeringen hier gebruiken drie synthetische beelden op CPU, met twee batches per epoch. Er is geen SVHN gedownload of getraind en geen CUDA-accuracy of GPU-tijd gemeten.

- De ongewijzigde oorspronkelijke runtime voert 100 synthetische epochs uit en de ongewijzigde freeze-pilot levert de staat na epoch 55.
- De nieuwe productiecode voert uitsluitend epochs 56–100 uit, met behoud van 51–55, vaste theta na elke update, A32, FP32, drie passes en het oorspronkelijke LR-schema.
- Onderbreken na 74 en 89 en hervatten tot 100 levert exact hetzelfde CPU-model, Adam, scheduler en RNG op.
- De drie checkpointselecties, bronexports en afzonderlijke globale selectie vóór freeze zijn gecontroleerd. De lagere encoders zijn tijdens de vervolgtraining expliciet verboden in de test.
- De werkelijke check-entry, vroege stop, ongeldige input en beschermde invoermappen zijn gecontroleerd.
- De notebook heeft 25 lege cellen. De 13 codecellen en 11 figuren zijn uitgevoerd op de productie-fixture; 45 aanvullende grafiekcontroles zijn geslaagd. De grafieken zijn visueel bekeken. Synthetische illustraties staan niet als onderzoeksresultaat in de geleverde notebook.

De onderzoekscheckpointbestanden blijven op de laptop. De opdracht `check` controleert daarom ook het werkelijke checkpoint 55 en de oorspronkelijke lokale geschiedenis voordat de echte voortzetting begint. Bitexact hervatten is hier alleen voor CPU aangetoond; de oorspronkelijke CUDA-backendinstellingen blijven behouden.

Reproduceer de CPU-fixtures met de scripts `build_fixtures.py`, `fixture_evidence.py`, `test_training_integration.py` en `test_check_entry.py`, telkens met afzonderlijke nieuwe uitvoermappen. De scriptheaders tonen hun argumenten. `test_notebook_layout.py` controleert de figuren met expliciet synthetische plotgegevens. De JSON-verslagen leggen de geteste bronhashes vast.
