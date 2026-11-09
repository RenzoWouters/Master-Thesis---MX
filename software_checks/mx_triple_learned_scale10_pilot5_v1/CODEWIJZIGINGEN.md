# Wijzigingen tegenover de binaire leerproef

| Bestand | Wijziging en doel |
|---|---|
| `trunc_adapter.py` | Tien harde offsets 0–9 met halfwaarden naar boven; parametergrenzen[0,9]; vierbit-keuze-export met afwijzing van ongeldige nibbles. Bestaande bitbewerkingen en schaalgradient behouden. |
| `trunc_support.py` | Diagnostiek voor alle tien factoren, uiterste keuzes tegenover exacte projectiegrenzen, wissels en sprongafstanden. Adam en projectie blijven één gezamenlijke update volgen. |
| `run_train.py` | Nieuw contract/resultatenpad, tienkeuzerapporten, nieuwe exportschema's, bewaakte hervatting en vaste vijf-epochlimiet. Notebook is ook bij een check of mislukte run beschikbaar. |
| `test_learned_scale.py` | Onafhankelijke controles voor alle tien offsets/drempels, oude5/6-route, gradients, codegrenzen, nibblepacking en reconstructie. |
| `notebook_support.py` en notebook | Accuracycurves en twee checkpointoverzichten; echte binaire pilot en vroege vaste referenties; alle tien factorhistogrammen, randgedrag, tijd en aparte opslagkosten. |

Behoud: B32, seed 20261009, oorspronkelijke modelinitialisatie, theta-init 6 zonder RNG-verbruik, één FP32-theta per groep, Adam-lr 0,001, oorspronkelijke100-epoch-LR-planning, drie passes per minibatch, losses, BN-banken, A32 en FP32-masters. Er worden geen extra passes uitgevoerd om alle factoren afzonderlijk uit te proberen.

De ruimere grenzen zijn inhoudelijk onderdeel van de wijziging: theta kan nu ook boven6 en onder5 bewegen. Het extra aantal keuzemogelijkheden verandert niet het aantal FP32-schaalparameters. De onafhankelijke schaal per exponentbreedte en bevriezen na 50 zijn toekomstige proeven.

Oude pakketten en resultaten zijn als bron behouden. De bijgevoegde referentiemetrics worden niet gewijzigd. `REUSE_MANIFEST.json` beschrijft byte-identiek hergebruik; nieuwe lokale CPU-controles staan in `local_validation`. Er wordt geen accuracywinst of GPU-snelheid geclaimd zonder de laptoprun.
