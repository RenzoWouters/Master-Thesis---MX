# v1.1 - diagnose en herstel van de synthetische vergelijkingstest

## Aanleiding en onzekerheid

De laptoptrace van 2026-10-19 stopte in v1 `test_dual.py:147` op de
vergelijking tussen twee modeltoestanden na twee synthetische updates.
De tolerantie was rtol=1e-4, atol=1e-5. De test meldde niet welke tensors
verschilden of hoe groot het verschil was. De vijf SVHN-epochs waren nog
niet gestart. `pip check` en GPU-detectie waren in de meegedeelde log geslaagd.

Het productiebeleid laat niet-deterministische GPU-algoritmen toe. Een
herhaling van twee trainingsstappen is onder dat beleid geen betrouwbare
uitsluitende test van implementatiegelijkheid. Numerieke verschillen zijn
een mogelijke verklaring, geen hier bewezen oorzaak van de laptopfout.
CPU-tests zijn geen bewijs dat de CUDA-implementatie op deze laptop slaagt.

## Wijziging van de test, niet van het onderzoeksprotocol

1. Onder het oorspronkelijke backendbeleid worden de manuele en de
   Any-Precision-lus opnieuw vergeleken. Logits, gradients voor iedere update,
   uiteindelijke gewichten, BN-buffers, Adamstaat en RNG-verbruik worden gemeten.
   Alle tensorverschillen worden met naam en maximale absolute afwijking
   opgeslagen. Numerieke herhaalverschillen zijn in deze fase diagnostiek.
   Eindigheid, structuur, RNG-verbruik, BN-tellers, actieve gradients en
   Adam-invarianten zijn wel harde voorwaarden; die worden niet genegeerd.
2. Dezelfde manuele/Any-Precision-vergelijking moet vervolgens onder tijdelijk
   deterministische algoritmen slagen. Op CUDA gelden DEZELFDE toleranties
   als v1 (rtol=1e-4 en atol=1e-5), nu ook voor logits, gradients en Adamstaat.
   CPU vereist exacte gelijkheid. Er is geen hogere tolerantie gekozen.
3. De deterministische context herstelt backendflags ook bij een exception.
   Alleen de opdracht `check` zet in het eigen Python-proces vooraf
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` wanneer er niet al een ondersteunde waarde
   staat. Dit gebeurt voor CUDA-initialisatie. De parent PowerShell-omgeving
   wordt niet gewijzigd. `run` en `resume` bevatten die wijziging NIET.
4. Het runrapport houdt gedeeltelijke diagnostiek bij als de test faalt.
   Een mislukte deterministische vergelijking blijft een FAIL en geeft nu
   ten minste een tensornaam, foutgrootte en aantal afwijkende tensors.
5. Na herstel van de backend wordt ook de bestaande exportcontrole uitgevoerd.

De CUDA-toleranties zijn hier praktische softwaretestgrenzen, geen
statistisch bewijs van gelijke onderzoeksresultaten. Een PASS bewijst niet
bitexact hervatten onder de productiebackend. De productieherhaling blijft
in het rapport zichtbaar, ook als alleen de deterministische herhaling
binnen tolerantie valt.

## Niet veranderd

- Configuratie, `run_epochs`, model, MX-codecs, STE, beide BN-banken,
  distillatieloss en updatevolgorde.
- E2M3 en E2M1 worden beide DIRECT uit FP32-masters gevormd; geen truncatie.
- A32, FP32-masters/Adam, blok32, E8M0, train-only en vijf volledige epochs.
- Bestaande gewichtsvoorbewerking, data/transforms, seed, LR, batchgrootte.
- Checkpoint-, hervat- en exportcode en notebookberekeningen.
- Productiebeleid: cudnn.benchmark=True, geen deterministische override.

Alleen de nieuwe standaardresultatenmap bevat `_v1_1`, om versies herkenbaar
te scheiden. `CHECK_FIX_MANIFEST.json` en `check_fix.patch` documenteren
hashes en veranderingen. De volledige oude map en ruwe resultaten blijven
op de laptop ongewijzigd. Pakketmanifests zijn opnieuw berekend.

## Op de laptop

Pak de inhoud uit in een NIEUWE map:

    C:\Users\renzo\master-thesis-mx\software_checks\mx_weights_dual_pilot5_v1_1

Voer met de bestaande omgeving uit:

    & ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_dual_pilot5_v1_1\run_train.py" check

Pas na EINDSTATUS: PASS:

    & ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_weights_dual_pilot5_v1_1\run_train.py" run --stop-after-epoch 5

Bij opnieuw FAIL: geen training starten, niets upgraden, geen assert wissen.
Deel de nieuwe feedback_bundle.zip uit de `_joint_check`-map. De gedetailleerde
vergelijkingen staan in `run_report.json -> synthetic_training_checks`.
De oude FAIL-bundel mag behouden blijven; v1.1 overschrijft die niet.

## Bronnen

- Aangeleverde mx_weights_dual_pilot5_v1.zip, test_dual.py en run_train.py.
- PyTorch 2.10, Reproducibility:
  https://docs.pytorch.org/docs/2.10/notes/randomness.html
- PyTorch 2.10, torch.use_deterministic_algorithms:
  https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html

Dit is een correctie van de testopzet en rapportage. De precieze oorzaak van
de oorspronkelijke laptopafwijking en het slagen op CUDA zijn nog niet door
nieuwe laptopdiagnostiek bevestigd.
