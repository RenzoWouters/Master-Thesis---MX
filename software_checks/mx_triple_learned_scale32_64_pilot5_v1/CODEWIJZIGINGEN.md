# Wijzigingen tegenover de vaste-schaalpilot

| Bestand | Wijziging en doel |
|---|---|
| `trunc_adapter.py` | Zes FP32 theta-tensors, exact binaire schaaloffsets, extra expliciete schaalgradient, bewaking van gedeelde keuzes en compacte éénbitexports. |
| `trunc_support.py` | Adam-parametergroep voor theta, projectie na elke update, controle van oorspronkelijke initialisatie zonder theta en schaaldiagnostiek gedurende de echte updates. |
| `run_train.py` | Nieuw experimentcontract en resultatenpad; modeltellingen, rapportage, schaal-CSV's, bewaakte hervatting en exports; vaste vijf-epochlimiet. |
| `test_learned_scale.py` | Gerichte onafhankelijke controles op bits, grensgevallen, surrogate-gradient, mastergradient, update/projectie, gepakte keuzes en reconstructie. |
| `notebook_support.py` en notebook | Twee checkpointoverzichten, vaste ×32/×64-vergelijkingen binnen dezelfde pilotduur en figuren voor hulpwaarden, keuzes, omschakelingen en gradients. |

De drie scalar-codecs, exponent- en mantissabitbewerkingen, oorspronkelijke netwerkcode, voorbewerking G(W), lossfuncties, transforms, data-loaderbeleid en schedulerimplementatie blijven behouden. De author-forward krijgt dezelfde drie modusselecties als in de vorige gezamenlijke proeven. Er worden geen extra netwerkpasses of optimizersstappen toegevoegd.

De oude vaste-schaaladapter, support en runner zijn als provenance bijgevoegd. `REUSE_MANIFEST.json` benoemt de byte-identiek overgenomen runtimebestanden. De concrete code en onafhankelijke validatierapporten maken de nieuwe backward-afspraak controleerbaar; er wordt geen gemeten voordeel van geleerde schalen verondersteld.
