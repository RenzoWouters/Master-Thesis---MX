# Resultaten van de echte 8×8 ×64-pilot

De pilot eindigde op **PASS** na vijf volledige epochs, 2.865 Adamupdates en 8.595 trainingspasses. De oorspronkelijke initialisatie komt overeen met de B32- en B64-referenties. A32, FP32-masterstaat en de vaste truncatieroute zijn bevestigd.

| Groepsindeling, steeds ×64 | E4M3 | E2M3 | E2M1 |
|---|---:|---:|---:|
| 1×32 | 93,008605% | 91,299170% | 90,891979% |
| 1×64 | 92,616779% | 90,972649% | 91,295329% |
| 8×8 | **93,139213%** | **91,675630%** | **91,502766%** |

Alle waarden zijn van epoch 5. Voor alle drie runs is dat tevens het unieke beste E4M3-checkpoint binnen epochs 1–5; de twee checkpointkeuzes vallen dus samen. Het zijn vroege resultaten van één seed, geen bewijs over de uiteindelijke accuracy.

De E4M3-accuracy en E2M1-accuracy stijgen iedere epoch. E2M3 kent een beperkte dip op epoch 4 en herstelt op epoch 5. Alle drie trainingsobjectieven dalen. Er zijn geen technische of numerieke fouten; de twee warnings betreffen bestaande PyTorch-deprecations rond pin_memory.

De totale sessie duurde 286,91 seconden (4 min 47 s). De mediaan van epochs 2–5 is 54,42 s. De projectie voor een nieuwe volledige run is circa 1 uur 31 minuten; dit is geen gemeten 100-epochduur of hardwareversnelling.

De zes MX-lagen vragen bij 8×8 2.024 schaalbytes en 64 paddingposities. Code+scale-payload: E4M3 131.560 B, E2M3 99.176 B, E2M1 66.792 B. De E4M3-bron is 2,29% kleiner dan 1×32 en 6,30% kleiner dan 1×64. FP32-uitzonderingen en BN-staat vallen buiten deze vergelijking.

De zes exports zijn door de CUDA-runner exact tegen de trainingsroute gecontroleerd; hun CSV-geometrie is onafhankelijk nagekeken. De echte .pt-bestanden zitten niet in deze feedbackbundel en zijn lokaal niet opnieuw gedecodeerd.

Bronnen: ongewijzigde pilotbestanden in `raw`, de oorspronkelijke `feedback_bundle.zip`, en de ongewijzigde eerste vijf epochs van `scale64_reference` / `group64_reference` elders in dit pakket. De volgende proef start vanaf initialisatie met dezelfde instellingen en loopt tot epoch 100.
