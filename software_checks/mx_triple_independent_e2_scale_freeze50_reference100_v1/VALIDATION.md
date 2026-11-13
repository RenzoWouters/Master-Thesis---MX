# Pakketvalidatie — freeze na epoch 50, vervolg tot epoch 100

**Status: PASS voor de uitgevoerde softwarecontroles.** Er is hier geen nieuwe SVHN-training uitgevoerd. Het echte epoch-50-checkpoint staat op de laptop van de gebruiker en is niet in deze bouwomgeving ingeladen.

## Uitgevoerde controles

De bestaande trainings- en exportcode is op drie expliciet synthetische CPU-beelden uitgevoerd. Eerst is een doorleercontrole tot epoch 100 opgebouwd. Vanuit haar volledig opgeslagen epoch 50 zijn een aaneengesloten freeze-run tot 100 en een run met onderbreking op 75 en hervatting tot 100 vergeleken.

De twee routes eindigden met exact gelijke model-, Adam-, scheduler- en RNG-toestand. Alle 4.080 absolute E2-schaalparameters, schaalcodes en schaal-Adamtoestanden bleven vast. Gewichten en BatchNorm bleven leren. De learning-rateovergangen bij epochs 76 en 91 zijn gecontroleerd, evenals het startniveau bij 51.

De echte main/check-ingang is uitgevoerd met alleen de synthetische parent, configuratie en CPU-runtime als testvervanging. Package- en bronhashes, dependencycontroles, parentcontract, parenthash, geschiedenis, herstel en freezecontrole bleven actief. pip check slaagde. Verkeerde ouderhashes, childcontracten, gewijzigde oudergeschiedenis en ongeldige eindepochs werden afgewezen.

## Notebook

De onuitgevoerde notebook bevat 31 cellen. Gecontroleerd zijn: nog geen nieuwe resultaten; een duidelijk als compatibiliteitsfixture gemarkeerde kopie van de echte pilotgegevens; complete synthetische epoch-100-uitvoer; gedeeltelijke epoch-75-uitvoer; en foutpaden met inconsistente completionflags of ongeldige metrics.

De reader houdt de overgenomen epochs 1–50 en nieuwe epochs 51–100 uit elkaar. Hij gebruikt dezelfde beschikbare epoch voor vergelijkingen en behoudt aparte globale en branch-checkpointselectie. De figuren gebruiken afzonderlijke precisiepanelen, varianten op de as, gelijke cirkels en percentagelabels.

De compatibiliteitsfixture verandert geen gemeten accuracy-, gewicht- of schaalwaarden en is geen nieuwe onderzoeksrun. Deze fixtures en hun checkpoints zijn niet in het pakket opgenomen.

## Behoud en bewijsgrens

De numerieke adapter, codecs, verlies-/optimizerlogica en checkpointrestore zijn ongewijzigd ten opzichte van het gevalideerde pilotpakket. Alle 66 historische referentiebestanden zijn byte-identiek behouden. De oorspronkelijke ZIP-bestanden en ruwe onderzoeksdata zijn niet overschreven.

CPU-exactheid bewijst geen bitexacte CUDA-hervatting. Voer daarom op de laptop eerst het gedocumenteerde check-commando uit. De echte SVHN-training en haar accuracy-/tijdresultaten moeten nog worden uitgevoerd en gemeten.

De testomgeving gebruikt Python 3.12 en PyTorch 2.10.0+cpu / torchvision 0.25.0+cpu. Er zijn geen updates in de trainingsomgeving van de gebruiker aangebracht.

## Rapporten en herkomst

De actuele rapporten staan in local_validation en zijn met SHA-256 opgenomen in FINAL_VALIDATION.json. Daarin staan ook de hashes van de werkelijk geteste productie-Pythonbestanden.

Oude validatierapporten van het vijfepochpilotpakket staan expliciet als historische herkomst in provenance/pilot_package_validation. Zij worden niet als nieuwe full100-validatie gebruikt.
