# Codewijzigingen — groepsgrootte16, vaste×64

De wetenschappelijke vergelijking is **B32→B16 bij dezelfde vaste×64-route**. De bewezen pilotinfrastructuur is hergebruikt; de configuratie wordt ook rechtstreeks tegen de oorspronkelijke bewaarde ×64-pilotconfiguratie gecontroleerd.

- `group_config.py`: centrale groepsgrootte16 en generieke packed-byteberekening.
- `mx_e4m3.py`, `mx_format.py`, `mx_e2m1.py`: groepering, padding, packing, strikte payloadmetadata en diagnostiek volgen groepsgrootte16. De scalarcodecfuncties blijven ongewijzigd.
- `posthoc_truncation.py`: dezelfde bitmappings, met payloadgrootteberekeningen op basis van de groepengeometrie.
- `trunc_adapter.py`, `trunc_support.py`: dezelfde vaste+6/×64-route, broncache, identity-STE en recursieve trainingsvolgorde; nieuwe experimentidentiteit en B16-diagnostiek.
- `run_train.py`: eigen nieuwe resultatenmap en hervatcontract; vijfepochstop; automatische kostenrapporten en verificatie van de opgeslagen exports; kopieert de onuitgevoerde notebook en referentiegegevens naar de runmap.
- `group_costs.py`: onafhankelijke berekeningen van groepen, padding, schaalbytes, codebytes en totale representatiegrootte voor 16/32/64. Leest geen gewichten/checkpoints en traint niets.
- `notebook_support.py` en de notebook: vergelijking van B16 ×64 met de eerste vijf epochs van de bestaande B32 ×64-run; eigen kostenhoofdstuk met tabellen en figuren.

De oorspronkelijke upstreambron, architectuur, datasetbewerking, lossdefinities, optimizer, LR-schema, broncodecwaarden, BN-bankselectie en FP32/A32-beleid behouden hun bestaande werking. De synthetische testharnassen gebruiken uitsluitend kleine CPU-inputs en zitten buiten het productiepad.

De oude resultaten, oude pakketten en projectbronnen zijn niet gewijzigd. De nieuwe gebruikersrun maakt zelf nieuwe modelcheckpoints en resultaten; er zijn bij voorbereiding geen B16-SVHN-accuracies gemeten.
