# Softwarevalidatie van het 100-epochpakket

## Afbakening

De nieuwe 100-epochtraining op SVHN is hier niet uitgevoerd. De werkelijke onderzoeksgegevens in `independent_pilot_reference/` komen uitsluitend uit de aangeleverde vijfepochrun. De beoordeling daarvan staat in `PILOT_REVIEW.md`; de controles zijn ook vastgelegd in `local_validation/actual_pilot_*.json`.

De lokale softwareproef gebruikte Python 3.12 en PyTorch 2.10 op CPU, met drie synthetische beelden. De accuracies en tijden van deze fixture zijn uitsluitend testwaarden en geen SVHN-resultaten, GPU-metingen of prestatievoorspelling. Er worden geen synthetische checkpoints, grafieken of uitgevoerde notebooks meegeleverd.

## Gecontroleerde software

De zestien numerieke modules en bestaande ondersteuning zijn byte voor byte gelijk aan de code uit de geslaagde echte pilot. Alleen de runner, het rapportagevenster, de historische comparator en de notebook zijn aangepast voor 100 epochs. `REUSE_MANIFEST.json` en `TRAINING_CHANGE_MANIFEST.json` bevatten de hashes en wijzigingsscope. Alle 79 bestanden van de voorlopige seal bleven gedurende de tests ongewijzigd. Daarna zijn uitsluitend validatierapporten, deze toelichting en de definitieve manifesten toegevoegd.

## Runner: PASS

Bron: `local_validation/runner_integration_report.json` en `local_validation/check_entry_report.json`.

- De werkelijke `main/check`-ingang en codec-, schaal-, gradient-, cache- en exportcontroles zijn doorlopen.
- De werkelijke trainingslus is 100 epochs doorlopen: 200 updates op de kleine fixture.
- Een aaneengesloten run en stoppen op epoch 50 gevolgd door hervatten tot 100 leveren exact dezelfde model-, Adam-, scheduler- en RNG-staat, samplevolgorde en schaal-CSV's op CPU.
- Beide optimizer-LR's volgen de overgangen op 50/75/90. De nieuwe LR wordt telkens vanaf de volgende epoch gebruikt.
- De schaalparameters blijven trainbaar en bereiken alle 200 Adam-stappen. Er zijn werkelijke schaalparameterwijzigingen na epoch 50. Dit betekent niet dat iedere parameter bij iedere update numeriek verandert of een afrondingsgrens overschrijdt.
- Het checkpoint van epoch 50 en de verwijzing met controlesom zijn aanwezig. Een gedeeltelijke run krijgt `STOPPED_AT_EPOCH_BOUNDARY`, geen voltooide `PASS`.
- Een oud pilotcontract wordt afgewezen vóór herstel of updates. Ongeldige commandoregelopties worden afgewezen.
- Compacte exports bevatten 4.080 absolute E2-schaalbytes. Herladen uit de bronrepresentatie reconstrueert de logits exact; FP32-mastergewichten en schaalproxies ontbreken in de compacte MX-representatie.
- Een afzonderlijke synthetische vergelijking koppelt 100 epochs/300 rijen en weigert afwijkende samplevolgorde. De echte historische referenties zijn alleen gelezen en hun hashes zijn gecontroleerd.

De volledige lokale runnercontrole duurde 197,87 seconden. Deze duur is geen schatting voor de gebruikerstraining.

## Notebook: PASS

Bron: `local_validation/notebook_validation_report.json`, `local_validation/notebook_pending_validation_report.json` en `local_validation/partial_reader_validation.json`.

Alle 35 cellen zijn gecontroleerd; alle codecellen zijn uitgevoerd en alle tien figuren zijn gerenderd. De aangeleverde notebook is onuitgevoerd gebleven. De nieuwe resultaten worden vergeleken met de volledige vaste ×64- en tienkeuzereferenties; de onafhankelijke pilot blijft aparte evidence voor de eerste vijf epochs.

De lezer onderscheidt een nog niet uitgevoerde run, een gedeeltelijke run op epoch 50 en een voltooide run op epoch 100. Hij controleert het gemeenschappelijke checkpoint gekozen op E4-accuracy, de LR-overgangen, CSV/history-consistentie en de scheiding tussen gemeten tijd en extrapolatie. Gerichte fouten met een verouderde accuracy-CSV, verkeerde LR na 50, oud pilotcontract en verkeerde checkpointselectie zijn afgewezen.

De volledige accuracycurves en de extra zoom op de laatste tien epochs behouden alle getoonde meetpunten binnen de assen. De absolute E2-exponenten en de geobserveerde verhouding tot E4 zijn afzonderlijk weergegeven. Root heeft de volledige curves, laatste-epochzoom en schaalevolutie ook visueel gecontroleerd.

## Testspecifieke aanpassingen en beperkingen

De productieomgeving is Windows/CUDA; deze lokale validatie is CPU. Een apart proces bouwde het ongewijzigde auteursmodel als CPU-initialisatieorakel. Alleen de testharness verving tijdelijk de in-memory verwachte initialisatiehash door dat CPU-orakel. De historische Windows-hash, de ruwe referenties en de productiecontrole zijn niet gewijzigd.

De geïsoleerde CPU-venv bevat geen `pip`-module. Alleen de testharness gebruikte daarom een inventaris via `importlib.metadata`; `pip check` en de strikte Windows/CUDA-omgevingscontrole zijn lokaal niet getest. De productiecode behoudt die controles. De echte vijfepochpilot is wel al in de GPU-omgeving van de gebruiker geslaagd.

Bitexacte CPU-hervatting bewijst geen bitexacte CUDA-hervatting onder het bestaande cuDNN-beleid. De lokale tests valideren geen toekomstige accuracy of globaal optimum van de schalen. Voer vóór de volledige run het gewone `check`-commando uit zoals beschreven in `README.md`.

## Reproduceren van de CPU-softwarecontrole

Dit is alleen nodig om de meegeleverde tests zelf te herhalen. Gebruik voor normaal onderzoek de opdrachten in `README.md`. Voor de onderstaande controles is een geschikte CPU-omgeving met PyTorch, torchvision, NumPy, pandas, matplotlib en IPython nodig. Kies een nieuwe testmap en houd die gescheiden van de onderzoeksruns.

```text
python local_validation/test_runner_integration.py run PAD_NAAR_NIEUWE_TESTMAP
python local_validation/test_notebook_support.py PAD_NAAR_DIT_PAKKET --fixture PAD_NAAR_CONTINUOUS100_RESULTAATMAP --out PAD_NAAR_NIEUWE_NOTEBOOK_TESTMAP
```

De runnercontrole drukt zijn rapportpad af; dat rapport bevat de juiste resultaatmap voor de notebookfixture. De tests zijn expliciet als synthetisch gelabeld.
