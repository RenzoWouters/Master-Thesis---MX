# Bestaande onafhankelijke-schaaltraining als parent en controle

De bron is de werkelijke run `20261111_172102_961497_joint_run`, geleverd in `feedback_bundle(20261111-183238).zip`. Zij heeft 100 volledige epochs, 57.300 Adam-updates, 171.900 trainingspasses en status PASS. De totale sessie duurde 6.478,80 seconden, ongeveer 1 uur en 48 minuten. Er zijn twee bestaande PyTorch-deprecationmeldingen over `pin_memory`/`is_pinned`, geen trainingsfout.

De accuracies hieronder komen uit `epoch_metrics.csv`; alle modi binnen een rij horen bij hetzelfde checkpoint. Dit zijn bestaande controlemetingen, geen resultaten van de nieuwe freezeproef.

| Epoch | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| 34 | 94,707% | 93,108% | 93,312% |
| 50 | 94,457% | 93,108% | 92,932% |
| 55 | 94,607% | 94,634% | 94,687% |
| 100 | 94,595% | 94,941% | 94,822% |

Epoch 34 is het eerste maximum van E4M3 over de volledige parentrun. Epoch 50 wordt het gedeelde vertakpunt. Epoch 55 vormt straks het gelijke eindpunt voor de pilotvergelijking. Het bestaande epoch-100-resultaat kan pas als gelijk eindpunt worden gebruikt wanneer de freezevertakking ook tot 100 is uitgevoerd.

## Aanleiding en interpretatie

Op epoch 100 haalt de onafhankelijke route bij E2M3/E2M1 94,941%/94,822%, tegenover 94,899%/94,791% voor vaste ×64 en 93,934%/93,961% voor de relatieve tienkeuzeroute. E4M3 ligt met 94,595% lager dan vaste ×64 (94,929%). De kleine positieve verschillen tegenover vaste ×64 op precies epoch 100 bewijzen geen algemene winst.

In de laatste tien epochs loopt E2M3 van 94,269% tot 94,995% en E2M1 van 94,353% tot 95,041%. De vaste ×64-referentie varieert daar over ongeveer 0,11 procentpunt. Dit zijn beschrijvende trajectgegevens; epochs zijn geen onafhankelijke herhalingen.

Uit de schaalregistraties volgen 19.533 harde E2-wisselgebeurtenissen in epochs 91–100, inclusief herhaald heen-en-terugschakelen. Op de epochgrenzen zijn 336 verschillende betrokken groepen zichtbaar. Van epoch 50 naar 100 verschillen 384 groepen in hun uiteindelijke absolute E2-exponent. Over epochs 51–100 zijn 192.072 wisselgebeurtenissen geregistreerd. Deze tellingen hebben verschillende betekenissen en mogen niet worden verwisseld.

De freezeproef onderzoekt de hypothese dat stoppen met de expliciete schaaloptimalisatie een ander of stabieler traject kan opleveren. De bestaande gegevens bewijzen niet dat de wissels de accuracyvariatie veroorzaken, dat freeze zal verbeteren of dat epoch 50 optimaal is.

## Audit en beperkingen

`parent_analysis/` bevat de afgeleide data-audits. De 600 train/testrecords, juiste aantallen en history zijn consistent; alle 100 train/testsamplevolgordes en initiële basisgewichten komen overeen met de relatieve tienkeuzecontrole. Alle 412.080 schaalblokregels zijn gecontroleerd tegen de registratie. Er is geen voorspelde klassecollapse en er zijn geen schaalgrensprojecties gerapporteerd.

De feedback bevat geen volledige checkpointtensors. De werkelijke parentcheckpointcontrole gebeurt daarom op de laptop vóór de freezeproef, tegen de vastgelegde SHA-256. De nieuwe vertakking is nog niet uitgevoerd. Ruwe bronbestanden blijven ongewijzigd.
