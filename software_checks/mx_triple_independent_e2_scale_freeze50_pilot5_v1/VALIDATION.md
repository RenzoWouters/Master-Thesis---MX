# Validatie van de onafhankelijke E2-freezeproef

## Resultaat en scope

De lokale softwarevalidatie is **PASS**. Er is geen nieuwe SVHN- of CUDA-training uitgevoerd en het werkelijke Windows-checkpoint is niet naar deze omgeving overgebracht. De parentgegevens zijn de bestaande onderzoeksresultaten; de lokale checkpoints en testaccuracies zijn expliciet synthetisch.

De tests gebruikten Python 3.12, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu en drie synthetische beelden. Deze omgeving vervangt niet de controle op de laptop. De productiecode behoudt de vastgelegde checkpoint-, contract-, omgeving-, bron- en datacontroles.

## Freeze en hervatten

Bron: `local_validation/runner_integration_report.json` en `local_validation/check_report.json`.

De integratietest maakte een echte synthetische learning-control van 100 epochs. Vanuit het opgeslagen en gepinde epoch-50-checkpoint zijn zowel vijf aaneengesloten freeze-epochs als stoppen op 52 en hervatten tot 55 uitgevoerd. Deze test duurde 204,90 seconden; dat is geen GPU-tijdmeting of raming voor SVHN.

De beide routes leveren exact dezelfde model-, Adam-, scheduler- en RNG-staat op CPU op. De train/testsamplevolgordes komen overeen met de synthetische learning-control. Phi, harde E2-schaalbytes, schaal-Adam-momenten en hun staptellers blijven bij alle nieuwe updates onveranderd. De gewichtupdates en actieve BN-updates gaan door. In de kleine fixture blijft de schaal-Adamstapteller op 100, terwijl de gewichtstapteller 110 bereikt; in de echte proef zijn dat respectievelijk 28.650 en 31.515.

De controle toont ook dat E4 dynamisch kan veranderen bij een vaste absolute E2-schaal. Een ongeoorloofde wijziging van phi, schaalmoment of schaalgradient wordt afgewezen. Verkeerd contract, afwijkende parent-SHA en veranderde parenttrace worden geweigerd. De parentcheckpointbytes en echte referentiebestanden bleven behouden.

## Werkelijke checkingang

Bron: `local_validation/check_entry_report.json`.

De werkelijke `main → create_context → parentverify → restore → test_freeze`-ingang is doorlopen. Alleen de testharness gebruikte `unittest.mock` voor de CPU-runtime, de kleine configuratie en de bijbehorende synthetische parentreferentie. De normale dependencychecks, `pip check`, pakket-/bronhashes en parentguards bleven actief. Er zijn geen datasetfuncties vervangen: `check` laadt geen SVHN.

De directe integratie oefent de werkelijke trainingslus, exports en hervatting uit; de CLI-check oefent daarnaast de werkelijke checkingang uit. Er wordt niet beweerd dat een volledige Windows/CUDA-run of de echte datasetloader hier is uitgevoerd.

## Notebook en grafieken

Bron: `local_validation/notebook_validation_report.json` en `local_validation/notebook_pending_validation_report.json`.

Alle 31 cellen zijn gecontroleerd; alle codecellen zijn uitgevoerd en dertien figuren zijn weergegeven. De afgeleverde notebook bleef onuitgevoerd en ongewijzigd. De lezer valideerde zowel een voltooide pilot op epoch 55 als een gedeeltelijke uitvoering op 52. De eerste 50 epochs worden als overgenomen geschiedenis geteld en hun tijd wordt niet als nieuw werk gepresenteerd.

De checkpointgrafieken hebben afzonderlijke precisiepanelen, varianten op de categorie-as, identieke cirkels en percentages naast ieder punt. De ingezoomde assen behouden alle weergegeven scores. Root heeft de categorical layout en het diagram van afzonderlijke E4-bronveranderingen en E2-schaalwissels visueel gecontroleerd. Synthetische accuracies zijn niet met echte historische onderzoeksresultaten vermengd.

Gerichte fouten zijn afgewezen: een verouderde accuracy-CSV, niet-nul schaal-LR na de freeze, continue phi-drift zonder harde schaalwissel, ontbrekende bevestiging van onveranderde Adamstaat, een geërfde epoch als branchbest en een parent100-contract dat onjuist als freezeproef wordt aangeboden.

## Code en gegevens

De numerieke adapter, codecs, bitmapping, master-STE en oorspronkelijke loss zijn behouden. Vijftien oorspronkelijke modules zijn byte-identiek aan de parentcode. Wijzigingen zitten in de runner, de aanvullende freeze-audits in `trunc_support.py` en de notebookhelper; `freeze_support.py` en `test_freeze.py` zijn toegevoegd. `REUSE_MANIFEST.json` en `TRAINING_CHANGE_MANIFEST.json` leggen dit vast.

De 102 bestanden van de voorlopige productieseal bleven tijdens de validatie ongewijzigd. Daarna zijn uitsluitend validatierapporten, deze toelichting, definitieve metadata en testbestanden toegevoegd. De definitieve `PACKAGE_SHA256.json` omvat ook deze toevoegingen.

`local_validation/learning_control_runner.py` is de ongewijzigde parentrunner voor het maken van de synthetische controle. De overige tests staan in dezelfde map. De door de testharness gemaakte runrapport-wrapper rond de directe exports is expliciet synthetisch; zij is geen extra onderzoeksrun.

## Grenzen van het bewijs

Deze tests valideren de afgesproken freeze en het herstelmechanisme. Zij voorspellen geen accuracywinst en bewijzen geen bitexacte CUDA-hervatting onder het bestaande cuDNN-beleid. Omdat de numerieke adapter gelijk blijft, kan de custom backward intern nog een ongebruikte schaalafgeleide berekenen. Er wordt daarom geen gegarandeerde besparing in compute of energie geclaimd.

Voer vóór de echte vijfepochproef het gewone `check`-commando uit zoals beschreven in `README.md`. Alleen die controle gebruikt het werkelijke lokale epoch-50-checkpoint en de GPU-omgeving van de gebruiker.
