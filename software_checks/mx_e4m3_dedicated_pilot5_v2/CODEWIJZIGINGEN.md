# Wijzigingen ten opzichte van de bestaande dedicated E2M3-pilot

## Onderzoekswijzigingen

1. `mx_e4m3.py` is de nieuwe E4M3-codec, gebaseerd op de bestaande E2M3-codec.
   Vier lokale exponentbits, bias7, maximaal448 en expliciete NaN-afwijzing.
   De bestaande schaalregel gebruikt nu emax8 in plaats van emax2. E4M3-codes
   passen per stuk in één byte, dus geen 6-bit-packing.
2. `mx_adapter.py` verwijst naar de nieuwe codec en een afzonderlijk E4M3-
   exportschema. De zes laagposities, G(W), A32/BN32, FP32-parameters en STE-
   aansluiting blijven hetzelfde.
3. `single_support.py` rapporteert E4M3 als daadwerkelijk gewichtsformaat.
   De oorspronkelijke auteursforward wordt nog steeds zonder AST-wijziging
   gebruikt, met interne selector32 en één actieve uitvoeringsmodus.

## Run- en rapportagewijzigingen

`run_train.py` is afgeleid van `provenance/single_run_train.py` uit het
overdrachtspakket. Nieuwe naam, uitvoermap, formaatspecificatie, tests,
notebookkopie en exports. Alleen `--mode mx` is publiek toegestaan, en de
stopgrens blijft exact5. Geen honderd-epochcommand of automatische voortzetting.

De eindexport herberekent G(W) en de E4M3-codes op hetzelfde device als de
training en vergelijkt de teruggeladen gewichten daar exact. De eerdere
dedicated runner verplaatste het model hiervoor eerst naar CPU. Die verplaatsing
is hier vermeden om een extra CPU/GPU-verschil in G(W) te voorkomen. Dit is een
exportcontrolewijziging, geen wijziging van de trainingslus. De volledige
testaccuracy wordt niet opnieuw uit de export gemeten; de oude historische
mantissa-exportafwijking blijft een afzonderlijk open controlepunt.

`test_e4m3.py` gebruikt een onafhankelijke scalar-oracle voor alle eindige
elementcodes, ties en grensgevallen. `smoke_single.py` controleert twee
synthetische updates in het echte CNN-8-model en export/reconstructie.
`local_validation/test_training.py` controleert kleine synthetische
mini-epochs, hervatten en de rapportage; het is geen onderzoeksdataset.

`MX_E4M3_dedicated_pilot5_resultaten.ipynb` en `notebook_support.py` analyseren
uitsluitend opgeslagen resultaten en schrijven eventuele figuren in een nieuwe
analysemap. Epoch100 blijft expliciet afwezig in een vijf-epochpilot.

## Behouden compatibiliteitscode

`svhn_support.py`, `smoke_support.py`, `resume_core.py`, `SOURCE_MANIFEST.json`
en de auteurs-ZIP zijn byte-identiek hergebruikt. De bestaande utils-patch in
een nieuwe auteurswerkkopie is nog steeds uitsluitend lazy gpustat-import en
`view`→`reshape` voor de accuracyberekening. Geen extra wijzigingen aan de
auteursarchitectuur, datasetcode, optimizer of LR-scheduler.

De lokale CPU-integratietest leidt alleen binnen de test de bestaande
`Tensor.cuda`-calls naar CPU om. De productiecode heeft geen CPU-fallback en
controleert de bestaande Python3.13/PyTorch2.10/torchvision0.25-CUDA-omgeving.

## Behoud en vervolg

Alle oude code, checkpoints, ruwe data en resultaten blijven ongewijzigd.
Dit is de dedicated E4M3-stap; de bestaande mantissatruncatie wordt behouden
bij de volgende gezamenlijke E4M3/E2M3/E2M1-proef. Exponenttruncatie en
aanpassing/leren van de shared scale zijn niet aan deze dedicated proef toegevoegd.


## Implementatiecorrectie v2 na de GPU-check

De eerste gebruikerscontrole van v1 faalde bij de exacte E4M3-decodering.
`mx_e4m3.py` gebruikt nu gehele E4M3-tellers maal 2^-9 en directe FP64-
exponentbits voor de schaalfactoren. `torch.frexp`, schaalkeuze, RNE, saturatie,
layout, G(W), STE en alle trainingsinstellingen blijven gelijk. Dit is geen
onderzoek naar aangepaste schalen en geen exponenttruncatie.

`test_e4m3.py` behoudt de bestaande exacte controles en voegt alle schaalfactoren
en alle element/schaal-producten toe. Afwijkingen bevatten nu ruwe FP64-bits,
hexadecimale waarden en afzonderlijke numerieke/tekenbit-tellingen. De runner
bewaart het gedeeltelijke codectestrapport ook bij FAIL en neemt de nieuwe
rapporten in de feedbackbundel op. Een afzonderlijke v1-rekenprobe wordt alleen
als diagnose gebruikt; de productiecodec gebruikt dit pad niet.

De verschillen met de daadwerkelijk geleverde v1 staan als `v1_to_v2_*.patch`
in `provenance/`. De notebook, adapter, auteurscode, datasets en eerdere
resultaten zijn ongewijzigd. Zie `CORRECTIE_V2.md` voor de bewijslimiet.
