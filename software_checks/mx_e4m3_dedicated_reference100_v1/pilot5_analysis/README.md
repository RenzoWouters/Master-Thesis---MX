# Analyse van de echte E4M3/A32-vijfepochproef

**Datum:** 22 oktober 2026. **Runstatus:** PASS. **Pakket:** `mx_e4m3_dedicated_pilot5_v2`.

Open [de uitgevoerde notebook](MX_E4M3_dedicated_pilot5_analyse_uitgevoerd.ipynb) om de tabellen, vier figuren en beoordeling te bekijken. De resultaten staan al in de notebook; er hoeft niets getraind te worden. De oorspronkelijke feedbackbestanden, inclusief de oorspronkelijke notebook en analysehelper, zijn byte voor byte behouden in `raw/`.

## Resultaat

| Epoch | Trainaccuracy | Testaccuracy | Test-CE-loss | Totale epochtijd |
|---:|---:|---:|---:|---:|
| 1 | 78,5522% | 87,6998% | 0,399840 | 29,195 s |
| 2 | 89,4358% | 90,9765% | 0,310071 | 26,625 s |
| 3 | 91,0152% | 91,6564% | 0,285267 | 24,685 s |
| 4 | 91,8711% | 92,6129% | 0,252906 | 24,657 s |
| 5 | 92,2601% | **93,1392%** | **0,235969** | 25,494 s |

Bronnen: `raw/epoch_metrics.csv`, gecontroleerd tegen `raw/history.json`. Iedere epoch omvat 73.257 trainingsbeelden en 26.032 testbeelden; de SVHN-extra-split is niet gebruikt.

Het beste en laatste checkpoint zijn beide **epoch 5**, met 24.246 correcte voorspellingen op 26.032 testbeelden. De selectie volgt het eerste strikte maximum van de E4M3-testaccuracy. Dit zijn twee rapportagerollen van hetzelfde checkpoint, geen twee onafhankelijke trainingen. De selectie gebruikt de officiële testset en levert daarmee geen onafhankelijke testschatting na selectie. Bronnen: `raw/checkpoint_selection.json` en `raw/selected_metrics.csv`.

**Epoch100 is nog niet uitgevoerd.** De gemeten tijdsbasis geeft voor een nieuwe honderd-epochrun vanaf initialisatie ongeveer **41 minuten 53 seconden**. Het waargenomen tijdsbereik projecteert naar **41 minuten 10 seconden tot 44 minuten 25 seconden**; dit is geen betrouwbaarheidsinterval. Download, installatie, data-inladen en eindrapportage zijn uitgesloten. Langdurige warmteontwikkeling, achtergrondbelasting en LR-overgangen zijn nog niet gemeten. Bron: `raw/timing_estimate.json`, nagerekend vanuit `raw/epoch_times.csv`.

## Grafieken

- [Laatste en beste checkpoint; epoch100 nog niet beschikbaar](figures/checkpoint_accuracy.png)
- [Gemeten train/testaccuracy en CE-loss](figures/learning_curves.png)
- [Epochtijden en looptijdschatting](figures/timing.png)
- [Opgeslagen CUDA-geheugenpieken](figures/gpu_memory.png)

## Interpretatie

De pilot ondersteunt doorgaan naar de dedicated honderd-epochreferentie: training en technische controles zijn geslaagd, de eerste leerontwikkeling is consistent en de gemeten looptijd is bruikbaar. De vijf epochs bewijzen nog geen eindaccuracy, convergentie of voordeel tegenover andere formaten. Er is één seed gebruikt. Deze proef bevat geen truncatie, distillatie of MX-activaties.

Het contract vermeldt A32-activaties en FP32-masters, gradients en Adam-momenten. Tegelijk meldt `raw/run_report.json` het cuDNN-convolutiebeleid `tf32`, met AMP uit. **Tensordtypes en backendbeleid zijn verschillende zaken**: A32/FP32 is geen garantie dat ieder intern convolutieproduct met volledige FP32-mantissaprecisie is berekend. Er is geen kernelprofilering aangeleverd. Dit backendbeleid moet voor een vergelijkbare vervolgtraining behouden en gelogd worden.

De exportdiagnostiek omvat 129.472 MX-gewichten, 4.080 blokken en 1.088 paddingposities. Van de gewichten liggen 1.213 waarden vóór afronding buiten het lokale elementbereik: 0,9369%. De relatieve L2-kwantisatiefout per laag ligt tussen 2,7720% en 3,0473%; geen blok heeft schaalbegrenzing aan het E8M0-bereik. De runner rapporteert exact overeenkomstige gedecodeerde exportgewichten op het trainingsdevice. **Dit is geen volledige test-forward uit uitsluitend de export.** Checkpoints en exports zelf zitten niet in de feedbackbundel. Bron: `raw/mx_export_layers_best.csv` en `raw/run_report.json`.

De twee geregistreerde waarschuwingen betreffen het verouderde `device`-argument bij `Tensor.pin_memory()` en `Tensor.is_pinned()`. Zij blijven zichtbaar in de notebook; er is geen gerapporteerde numerieke trainingsfout.

## Opnieuw analyseren

Gebruik `pilot5_analysis` als werkmap en open de uitgevoerde notebook. De parameter `RUN_DIR = Path('raw')` maakt de analyse onafhankelijk van het oorspronkelijke Windows-pad. Gebruik de bestaande notebookomgeving met pandas, NumPy en matplotlib.

Standaard staat `SAVE_FIGURES = False`: de meegeleverde figuren staan al in `figures/`, en heranalyse overschrijft niets. Bij bewust inschakelen van de parameter maakt de oorspronkelijke helper een nieuwe `analysis_<tijdstip>`-map onder de ingelezen resultaatmap. Er wordt nooit een training of checkpoint-forward gestart.

`ANALYSIS_AUDIT.json` bevat SHA256-hashes van de 21 ongewijzigde bronbestanden, de vier figuren en de controle op de analyse-uitvoer. De notebookcellen zijn daadwerkelijk met Python uitgevoerd; de Markdown-, tabel- en PNG-weergaven zijn als notebookoutputs opgeslagen.
