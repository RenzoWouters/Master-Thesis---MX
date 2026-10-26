# Lokale softwarevalidatie van de voortzetting na epoch 55

Dit zijn **synthetische CPU-softwareproeven**, uitgevoerd met PyTorch 2.10.0+cpu
en torchvision 0.25.0+cpu. Er is geen SVHN-data geladen of gedownload. De
accuracies en tijden uit deze fixtures zijn geen onderzoeksresultaten. CUDA,
de gebruikers-GPU en bitexact hervatten op CUDA zijn hiermee niet getest.

## Uitgevoerde controles

- De ongewijzigde oorspronkelijke runner traint een fixture met drie beelden,
  twee minibatches per epoch, werkelijk van 1 tot en met 100. De ongewijzigde
  freeze-pilot maakt daarna werkelijk epochs 51–55 vanuit het checkpoint na 50.
- De nieuwe runner herstelt checkpoint 55 inclusief model, Adam, scheduler en
  RNG. De toestand is exact gelijk aan de ouder voordat de eerste nieuwe batch
  wordt verwerkt. Daarna worden uitsluitend epochs 56–100 toegevoegd.
- Alle 4.080 FP32-offsetparameters en harde x32/x64-keuzes blijven exact gelijk
  aan epoch 50, terwijl de gewichten en BatchNorm doorgaan. De oorspronkelijke
  drie precisiepasses, losses, dataorders en absolute staptellers blijven behouden.
- De gewichts-LR daalt op de oorspronkelijke positie na epochs 75 en 90. De
  theta-LR blijft nul. Hervatten na 74 en na 89 levert op deze CPU-fixture exact
  hetzelfde eindmodel, Adam, scheduler en RNG als ononderbroken doorgaan.
- Er zijn drie onafhankelijke checkpointrollen: laatste epoch 100, beste E4M3
  in het bevroren venster (hier 51), en globale beste E4M3 over het gedeelde
  begin en nieuwe bevroren traject (hier 16). De globale export bewaart dus de
  werkelijke eigen offsets van checkpoint 16. De oude dynamische epochs 51–100
  worden uitsluitend als historische vergelijking gebruikt.
- Compacte exports van alle drie rollen zijn onafhankelijk gecontroleerd op
  exponentbits, mantissabits, schalen en 510 bytes blokkeuzes. Reconstructie uit
  uitsluitend de compacte bron geeft exacte logits; gewichtencoders zijn daarbij
  expliciet geblokkeerd. FP32-MX-masters en FP32-theta staan niet in de exports.
- Ongeldige oudertoestanden, gewijzigde referentiebestanden, verkeerde checksums,
  een gewijzigde hervatconfiguratie en schrijven binnen beschermde invoermappen
  worden geweigerd.
- De echte `main(check)` doorloopt in een afzonderlijk proces de oorspronkelijke
  codec-/gradienttests, strikte oudercontrole en geselecteerde eerdere
  checkpoints. De echte `main(run)` start met standaarddoel 100 en stopt correct
  op een volledige epochgrens via `STOP_AFTER_EPOCH`; rapport en feedbackzip
  blijven daarbij expliciet onvolledig. Ongeldige CLI-argumenten worden geweigerd.

De buitenste pakketidentiteit en referentiepins worden uitsluitend in de
testharness vervangen door herkenbaar synthetische waarden. De trainings-,
hervat-, quantisatie- en exportfuncties blijven ongewijzigd. De oorspronkelijke
interne runtimepakketten worden met hun echte manifests gecontroleerd.

## Bestanden

- `training_integration_report.json`: volledige voortzetting, beide hervatproeven,
  onafhankelijke exportcontrole, CLI- en negatieve controles; met SHA256-binding
  aan de drie geteste productiebestanden.
- `check_entry_validation_report.json`: afzonderlijke echte `check`-entrypointproef.
- `fixture_generation_report.json`: herkomst van de werkelijk uitgevoerde
  synthetische oorspronkelijke 100 epochs en bevroren pilot tot 55.
- De vier `.py`-bestanden bevatten de reproduceerbare fixture- en testcode.

De grote synthetische checkpoints en een uitgevoerde notebook zijn niet in het
leveringspakket opgenomen. De onuitgevoerde notebook hoort bij het hoofdprogramma.

## Reproduceren in een aparte CPU-testomgeving

Start vanuit de map boven het pakket. Gebruik nieuwe, nog niet bestaande
uitvoermappen. Deze ontwikkeltests zijn niet nodig om de GPU-training te starten.
Voorbeeld met een Python-interpreter met de bovengenoemde CPU-versies:

```bash
python mx_triple_freeze_offsets_e50_reference100_v1/local_validation/build_fixtures.py \
  mx_triple_freeze_offsets_e50_reference100_v1 validation_fixtures

python mx_triple_freeze_offsets_e50_reference100_v1/local_validation/fixture_evidence.py \
  mx_triple_freeze_offsets_e50_reference100_v1 validation_fixtures validation_references

python mx_triple_freeze_offsets_e50_reference100_v1/local_validation/test_training_integration.py \
  mx_triple_freeze_offsets_e50_reference100_v1 validation_training validation_fixtures validation_references

python mx_triple_freeze_offsets_e50_reference100_v1/local_validation/test_check_entry.py \
  mx_triple_freeze_offsets_e50_reference100_v1 validation_check validation_references
```

Het hoofdprogramma voor de gebruiker blijft uitsluitend de vastgelegde CUDA-run
toestaan. De synthetische fixture wordt niet via publieke trainingsopties ingeladen.
