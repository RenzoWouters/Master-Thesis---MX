# Codewijzigingen en hergebruik

Dit is een nieuwe evaluatiemap naast de bestaande experimenten. De volledige
directe referentietraining en eerdere mantissaproeven worden niet gewijzigd.

## Ongewijzigde numerieke basis

Veertien bestanden zijn byte-identiek overgenomen uit
`mx_weights_triple_direct_reference100_v1`. `REUSE_MANIFEST.json` legt hun
SHA256-hashes vast. Het betreft de drie codecs en formaatbasis, de MX- en
joint-adapters, support- en bestaande testmodules, de upstreambron-ZIP en
`SOURCE_MANIFEST.json`. De FP32 G(W)-bewerking, directe MX-schaalregels,
packing, modelarchitectuur en BN-selectie zijn niet aangepast.

De volledige oorspronkelijke packagehashlijst staat daarnaast in
`provenance/reference100_PACKAGE_SHA256.json`. De nieuwe runner vereist dat
het aangewezen referentie-experiment overeenkomt met dat oorspronkelijke
pakket en met het bewaarde `REFERENCE_PROTOCOL.json`.

## Nieuwe onderdelen

| Bestand | Verantwoordelijkheid |
|---|---|
| `truncation.py` | Acht vaste routes, bronvalidatie, bitconversies, afrondingscontroles, afgeleide exports, bevroren modellen en gewichtsdiagnostiek |
| `test_truncation.py` | Onafhankelijke bit- en getalorakels, schaalbehoud, grensgevallen en synthetische modelcontroles |
| `evaluation.py` | Gelijke testbatches voor alle routes, opgeslagen logits/voorspellingen/CE, gepaarde tellingen en controle op onveranderde modelstaat |
| `run_eval.py` | Alleen `check` en `evaluate`; bron-, omgevings- en datacontroles, uitvoer, integriteitsrapport en feedbackbundel |
| `notebook_support.py` | Analyse van JSON/CSV/NPZ, vijf figuurtypen, tabellen en uitvoercontroles zonder torch of modellen te laden |
| `MX_triple_posthoc_truncatie_resultaten.ipynb` | Lege uitvoerbare presentatie van beide vaste checkpoints |

De bron voor de drie afgeleide bitroutes bestaat uitsluitend uit opgeslagen
elementcodes en schaalbytes. Er is geen nieuwe kwantisatie uit FP32-masters
in die routes. De twee afrondingscontroles gebruiken wel de lokale
elementencoder, met exact de bewaarde bronschaal. De opgeslagen directe
MX-exports van de training leveren de overige modelstaat en de getrainde
BN-banken. Trainingscheckpoints en optimizerstaat worden niet geladen.

## Aansluiting op de eerdere mantissaproef

Vier oorspronkelijke bestanden uit `mx_mantissa_truncation_eval_v1` staan
ongewijzigd in `provenance/mantissa_eval_*`. Ze documenteren de eerdere
werkwijze; deze historische scripts worden niet uitgevoerd of geïmporteerd.

De werkwijze blijft: directe doelprecisie opnieuw evalueren, lokaal afronden
vanaf dezelfde bron als controle, bittruncatie vanaf die bron, en meten met
de bestaande BN-bank van de doelprecisie. De oude checkpointresultaten blijven
bewaard. De mantissaproef wordt nu opnieuw uitgevoerd op de gezamenlijke
drieprecisiereferentie, naast exponenttruncatie en de combinatie.

## Extra meetuitvoer

Per checkpoint worden acht routeaccuracies, elf gepaarde vergelijkingen,
48 gewichtsvergelijkingen en 48 route/laagdiagnoses opgeslagen. De twee NPZ's
bevatten geordende labels, indices, logits, voorspellingen en harde CE. Daardoor
blijft zichtbaar welke beelden door een omzetting verbeteren of verslechteren.

De drie opnieuw gemeten directe accuracies worden apart vergeleken met de
historische rapportwaarden. Een verschil wordt zichtbaar gemeld. Lage
truncatieaccuracy leidt op zichzelf niet tot FAIL. Onjuiste bronbestanden,
niet-eindige uitvoer, onvolledige datadekking of veranderde modelstaat doen dat
wel. Alle resultaatmappen zijn nieuw en staan buiten de oorspronkelijke run.

## Afbakening

Geen training, BN-herkalibratie, schalecorrectie, geleerde schalen, andere
activatieprecisie of wijziging van de FP32-updatemaster. De reeds bewaarde
epoch100- en E4M3-bestcheckpointselectie blijft vast. Softwarecontroles staan
in `local_validation`; hun synthetische scores zijn geen SVHN-resultaten.
