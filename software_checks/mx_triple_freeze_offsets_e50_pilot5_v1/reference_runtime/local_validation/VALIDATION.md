# Validatie van het honderd-epochpakket

## Wat werkelijk is gemeten

De gebruiker heeft de voorafgaande vijf-epochpilot op de RTX 5070 Laptop GPU uitgevoerd. De meegeleverde echte feedback is onafhankelijk gecontroleerd: vijf epochs, 2.865 updates, 8.595 trainingspasses, correcte checkpointselectie en consistente schaaldiagnostiek. De geselecteerde oorspronkelijke bestanden en het auditrapport staan onder `pilot_reference`.

De nieuwe honderd-epochtraining op SVHN is nog niet uitgevoerd. De nieuwe lokale tests gebruiken drie synthetische CPU-beelden. Hun accuracies en uitvoertijden zijn uitsluitend testgegevens en mogen niet als onderzoeksresultaten worden gelezen.

## Behoud van de numerieke methode

De adapter, optimizerondersteuning, codecs, oorspronkelijke modelonderdelen en numerieke tests zijn byte-identiek aan de gevalideerde pilot. De trainingslus verandert alleen in de tekst van een hervatmelding; de compacte-exportlus blijft bytegelijk. De verschillen in configuratie beperken zich tot experimentidentiteit, doel, eindepoch en rapportagebeschrijving.

`runner_change_review.json` en `CONFIG_DIFFERENCES.json` leggen dit vast. `core_check_report.json` is het behouden succesvolle numerieke pilotrapport; het is geen onnodig opnieuw uitgevoerde test. Het oorspronkelijke rapport en de overige pilottests zijn onder `provenance/learned_scale_pilot5_validation` bewaard.

## Nieuwe runnerintegratie: PASS

`training_integration_report.json` betreft de daadwerkelijke `run_train.main`, trainingslus, optimizer, scheduler, checkpointlogica en exports. Alleen data/context en CUDA-routering zijn lokaal vervangen door een expliciete CPU-testfixture.

De volledige synthetische run voert honderd epochs, tweehonderd gezamenlijke Adam-updates en zeshonderd trainingspasses uit. Hervatten vanaf epoch49 voert dezelfde resterende epochs uit en bereikt exact dezelfde modelstaat inclusief theta, beide Adam-groepen, scheduler en RNG. Ook beide schaal-CSV's zijn bytegelijk. De drie LR-grenzen na 50, 75 en 90 zijn voor beide parametergroepen vóór en na de overgang gecontroleerd.

Een gedeeltelijke CLI-stop en de stopmarker blijven correct onvoltooid gemarkeerd. Vijf ongeldige checkpointgevallen en tien ongeldige CLI-gevallen worden vóór updates geweigerd. Alle gevallen samen bevatten 306 synthetische Adam-updates.

De compacte bron en afgeleide exports zijn onafhankelijk uitgepakt. Beide harde schaalkeuzes zijn aanwezig; de keuzedata zijn precies 510 bytes. Er zitten geen FP32-theta's of mastergewichten van de zes MX-lagen in de compacte export. Bronreconstructie geeft voor alle drie modi exact gelijke logits zonder encoder of G(W).

`runner_validation_binding.json` bindt de finale runtime aan de geteste snapshot. Later aangepaste referentiewrappers en het toegevoegde pilot-auditrapport worden daar afzonderlijk vermeld; de numerieke code veranderde niet.

## Check-entry en CUDA

De check-entry gebruikt dezelfde contextfunctie, codeccontroles en numerieke tests als de gevalideerde pilot. Dit is met functie- en bestandshashes bevestigd in `check_entry_reuse_validation.json`; deze gelijk gebleven route is lokaal niet opnieuw uitgevoerd. De gebruiker voert de concrete nieuwe pakketcontrole uit met `run_train.py check` op de trainingsmachine.

De lokale testomgeving gebruikt Python 3.12.14, PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu. Het behouden `cudnn.benchmark=True`-beleid garandeert geen bitexacte CUDA-hervatting; de exacte CPU-vergelijking verandert die beperking niet.

## Notebook

De geleverde notebook heeft lege outputs en start geen training. De finale helper is gecontroleerd op volledige honderd-epochresultaten, gedeeltelijke resultaten en korte stationaire resultaten. De analyse bewaakt beide LR-groepen, de opgeslagen blokstaten, schaalstatistieken, gezamenlijke E4M3-checkpointselectie en vergelijkingen binnen de gezamenlijk gemeten epochrange.

`runner_notebook_integration.json` bevestigt uitvoering van alle vijftien codecellen en rendering van twaalf figuren op de daadwerkelijke nieuwe synthetische honderd-epochrunneroutput, met 412.080 blokstaatrijen en zeshonderd laag-epochrijen. `notebook_validation_report.json` bevat de aanvullende gerichte geldige en ongeldige voorbeelden en visuele controle. Synthetische resultaten worden niet met echte onderzoekscijfers vergeleken.

`actual_pilot_notebook_analysis.json` bevestigt bovendien dat de bestaande pilothelper de echte upload zonder analysefouten leest en beide vaste referenties als compatibel bevestigt. Daarbij bleven alle 43 bronbestanden gelijk; er is geen uitgevoerde notebook als extra deliverable gemaakt.

## Pakket en bronbehoud

`package_assembly_validation.json` koppelt het eindpakket aan de finale runtime- en notebookhashes, het protocol en de gevalideerde referenties. Alle ongewijzigd hergebruikte bestanden staan in `REUSE_MANIFEST.json`. `PACKAGE_SHA256.json` in de pakketroot bevat de hashes van ieder geleverd bestand behalve zichzelf.

De eerdere ×32-pilot, ×32-honderd-epochtraining, geleerde vijf-epochpilot en de bijbehorende feedbackbundels worden alleen gelezen. Hun eerder vastgelegde hashes en de ongewijzigde bronbestanden worden bij de assemblage opnieuw gecontroleerd. De nieuwe proef heeft een eigen naam, resultatenpad en checkpointcontract.
