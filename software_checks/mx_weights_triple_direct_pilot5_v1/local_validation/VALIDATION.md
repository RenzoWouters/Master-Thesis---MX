# Uitgevoerde controles — directe drieprecisiepilot

Alle onderstaande lokale controles zijn geslaagd. Ze gebruiken synthetische
invoer of gegenereerde rapportfixtures. Er is hier geen SVHN-training van de
nieuwe drieprecisieproef uitgevoerd en geen CUDA-device beschikbaar.
De meegeleverde `check`-opdracht voert de relevante korte controles op de
werkelijke laptop-GPU uit vóór de vijf-epochtraining.

## Hergebruik en bronbehoud

De twee bestaande E2-codecs, G(W)-adapter, E2M3-test, model-/datasethelpers,
hervatondersteuning en auteurs-ZIP zijn byte-identiek aan de directe
tweeprecisieproef. De E4M3-codec en bijbehorende tests zijn byte-identiek aan
de geslaagde dedicated referentie. Zie `REUSE_MANIFEST.json`.

De AST van de epochlus, de timing-sidecar voor hervatten, dependencycontrole
en bron-/dataintegriteitscontrole is gelijk aan de directe tweeprecisiebasis.
De auteurs-forward verschilt alleen in de drie selectorparen; de bestaande
loss-, backward-, dropout- en optimizersequencing blijft behouden.
Eerdere pakketten en de aangeleverde feedbackbundels zijn niet gewijzigd.

## Drie directe modi en onafhankelijke trainingsreplay

`joint_module_validation_report.json`: PASS op CPU.

- Alle drie quantizers geven rechtstreeks `Q_formaat(G(W))` en behouden de
  gedeclareerde identity-STE door de codec en oorspronkelijke G(W)-afgeleide.
- Een concreet voorbeeld onderscheidt directe E2M1-conversie van de oude
  E2M3-mantissabitshift. Een ander voorbeeld onderscheidt directe E2M3 van
  zowel herquantisatie uit gedecodeerde E4M3 als naïef exponentbitbehoud.
  Deze negatieve controles worden uitsluitend getest, niet tijdens training
  als gewichtsroute gebruikt.
- Twee gezamenlijke updates zijn vergeleken met een onafhankelijk
  uitgeschreven handmatige E4M3 → E2M3 → E2M1-supervisielus.
- Per replay zijn 346 tensorvergelijkingen exact gelijk op CPU; ook RNG,
  teacherinputs, gradients, model-/BN-staat en Adamstaat komen overeen.
- Negen meters leveren de juiste drie modi; de softteacher is telkens de
  losgekoppelde uitvoer van de direct voorafgaande hogere modus.
- Actieve BN-banken 8/32/4 krijgen ieder één update per minibatch; banken
  1/2 blijven ongebruikt. Alle activatieselectors blijven A32.
- Alle drie onafhankelijke exports reconstrueren de gewichten exact en
  geven dezelfde synthetische logits, zonder zes FP32-MX-masterparameters.

De bestaande aparte deterministische replaycontrole blijft behouden voor
de laptopcheck. CPU vereist exacte gelijkheid. Bij CUDA worden de bestaande
toleranties gebruikt en niet verruimd; de controle herstelt de backendflags.
Dit is geen garantie van bitexacte CUDA-training of CUDA-hervatting.

## Werkelijke check-entry

`check_entry_validation_report.json`: PASS. De echte `main()`- en
`create_context()`-checkroute zijn uitgevoerd, met uitsluitend in dit
testproces een CPU-runtime en omleiding van `Tensor.cuda`.
De ongewijzigde E2M3- en E4M3-codecchecks, nieuwe jointcheck, rapportbestanden
en feedbackbundel zijn daadwerkelijk door deze entry aangemaakt. Er is
geen dataset geladen of gedownload. De productie-CLI blijft CUDA-only.

## Runner, checkpointselectie en hervatten

`integration_report.json`: PASS in 19,41 seconden, met 18 synthetische
optimizerupdates op drie kunstmatige CPU-afbeeldingen, batch 2.

| Geval | Nieuwe updates | Uitkomst |
|---|---:|---|
| Vijf volledige mini-epochs | 10 | PASS; 30 trainingspasses en 30 metricrijen |
| Resume vanaf epoch 2 tot 5 | 6 | Exact dezelfde model-, Adam-, scheduler- en RNG-staat op CPU |
| Stop na één volledige epoch | 2 | STOPPED_AT_EPOCH_BOUNDARY; geen onterechte voltooiing |
| Verkeerd experimentcontract | 0 | Geweigerd vóór herstel en updates |
| Al voltooide pilot hervatten | 0 | Geweigerd vóór updates |

De fixture alleen gebruikt LR-mijlpalen 2/3/4 om alle overgangen binnen
vijf mini-epochs daadwerkelijk met resume te controleren. Het ongewijzigde
productieprotocol met 50/75/90 is apart gecontroleerd.

Verder gecontroleerd: korte laatste minibatch, sampledekking, drie forwards
en twee teacherchecks per update, A32/FP32, BN/Adam, eerste strikte E4M3-
maximum, alle drie geselecteerde scores uit hetzelfde checkpoint, zes directe
best-/last-exports, feedbackbundel en kopiëren van de onuitgevoerde notebook.
De productie-CLI weigert onder meer CPU, honderd epochs, een synthetische
bypass en een warmstart. De verwachte FAIL-statussen bij geweigerde
checkpoints in het rapport zijn geslaagde negatieve controles.

## Notebook zonder extra uitgevoerde kopie

`notebook_validation_report.json`: PASS op twee expliciet synthetische
rapportfixtures: complete5 en STOPPED3. Alle negen codecellen zijn uitgevoerd
zonder een uitgevoerde notebook op te slaan. De meegeleverde template blijft
leeg. Geen model, checkpoint of trainingsmodule is door de notebook geladen.

De complete fixture heeft verschillende optimale epochs per precisie en een
gelijke beste E4M3-score in twee epochs. De notebook kiest correct het eerste
E4M3-maximum en gebruikt diezelfde epoch voor alle drie precisies. Hij weigert
gemengde beste epochs, ontbrekende modi, verkeerde losslabels, een onterechte
epoch100-claim en een verkeerd aantal teachercontroles.

Vijf figuren zijn visueel gecontroleerd, inclusief de STOPPED-checkpointfiguur.
Alle fixturefiguren dragen zichtbaar SYNTHETISCHE CONTROLE en worden niet als
onderzoeksresultaat of gebruikersbijlage meegeleverd. De expliciete
SAVE_FIGURES-route maakt een nieuwe analysemap; invoerhashes blijven gelijk.

`runner_notebook_integration.json` bevestigt aanvullend dat de helper de
daadwerkelijk door de bovenstaande synthetische runner geschreven complete
en vroeg gestopte run zonder analysefouten leest. Hierbij is geen notebook
opnieuw uitgevoerd of opgeslagen.

## Omgeving en grenzen

Linux, Python 3.12.14, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu,
NumPy 2.3.5, SciPy 1.17.0. Dit is een geïsoleerde lokale testomgeving;
de laptopomgeving is niet aangepast. De productierunner verwacht zoals
voorheen Python 3.13, PyTorch 2.10.0, torchvision 0.25.0 en CUDA.

Synthetische accuracies en tijden zijn uitsluitend controlegegevens. Er zijn
nog geen gemeten SVHN-resultaten of GPU-looptijden voor de nieuwe drie-modus-
pilot. Het pakket bevat geen dataset, geen trainingscheckpoint en geen
uitgevoerde notebook.
