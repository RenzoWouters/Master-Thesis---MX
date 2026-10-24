# Vast protocol: A/B/C op de schaal ×64-getrainde checkpoints

## Bron en checkpointkeuze

Deze uitbreiding meet uitsluitend de voltooide `mx_triple_scale64_reference100_v1`
run `20261024_151210_665535_joint_run`. Die training gebruikt E4M3 als bron,
exponent-MSB-truncatie met vaste schaal ×64 naar E2M3, en vervolgens dezelfde
mantissatruncatie naar E2M1. De drie modi delen één FP32-master tijdens training.

De vaste checkpointrollen zijn:

| Rol | Epoch | Keuze |
|---|---:|---|
| `last_epoch` | 100 | Laatste voltooide epoch |
| `best_E4M3_test` | 42 | Eerste strikte maximum van de oorspronkelijke E4M3-testaccuracy |

Alle precisies binnen een rol gebruiken hetzelfde checkpoint. Er wordt geen
checkpoint opnieuw geselecteerd. `EXPECTED_REFERENCE.json` legt het oorspronkelijke
rapport, de selectie en alle zes exporthashes vast. Verplaatsen van een runmap
is toegestaan; wijzigen van de inhoud leidt tot een fout.

## Ongewijzigde formaten

| Formaat | Elementvelden | Bias | Lokale maximale eindige grootte |
|---|---|---:|---:|
| E4M3 | `[s e3 e2 e1 e0 m2 m1 m0]` | 7 | 448 |
| E2M3 | `[s e1 e0 m2 m1 m0]` | 1 | 7,5 |
| E2M1 | `[s e1 e0 m0]` | 1 | 6 |

Dit zijn de bestaande codecs. E4M3-codes 127 en 255 zijn ongeldig; de overige
254 codes zijn eindig. Alle 64 E2M3- en 16 E2M1-codes zijn eindig. Signed zero,
lokale subnormals, packing en padding blijven volgens die codecs gelden.

Blokken bevatten 32 elementen binnen één outputkanaal. E8M0-code `k` geeft
`S = 2**(k-127)`, met `0 <= k <= 254`; code 255 wordt geweigerd. De zes MX-convs
en overige FP32-lagen blijven dezelfde. Alle activaties blijven A32.
BN-bankkeys zijn 8 voor E4M3, 32 voor E2M3 en 4 voor E2M1. Die keys zijn
opslagidentifiers, geen activatieprecisies.

## Exacte A/B/C-bewerking

Laat `c` een opgeslagen unsigned E4M3-elementcode zijn en `k` de bijbehorende
gedeelde schaalcode. De volgende code beschrijft de numerieke mapping:

```python
sign = c >> 7
exponent = (c >> 3) & 0b1111
mantissa = c & 0b111

A6 = (sign << 5) | ((exponent >> 2) << 3) | mantissa
A4 = A6 >> 2
k_A = k

B6 = A6
B4 = B6 >> 2
k_B = int(k) + 6

local_E4 = decode_E4M3(c)
C6 = encode_E2M3_nearest_even(local_E4 / 64)
C4 = C6 >> 2
k_C = int(k) + 6
```

A behoudt de hoogste twee exponentbits zonder biascorrectie. B gebruikt exact
dezelfde elementcodes en verandert alleen de gedeelde schaal naar `64*S`.
Voor B en C gebeurt de schaaloptelling in een breder integer-type. Ook een
nulblok krijgt +6. Broncode 248 kan naar 254; 249 t/m 254 geven een fout.
Geen clipping, wraparound of extra schaalvermenigvuldiging in de E2M1-stap.

C gebruikt uitsluitend de lokale gedecodeerde E4-waarde, gedeeld door 64,
met de bestaande nearest-ties-to-even E2M3-encoder. `encode_tensor` en `G(W)`
worden niet aangeroepen; er wordt geen blokmaximum of nieuwe schaal gezocht.
C4 blijft mantissatruncatie van C6, geen aparte nearest-even E2M1-conversie.

De E4-broncodes worden rechtstreeks uit de bestaande compacte exports gelezen.
Ze zijn tijdens export op het oorspronkelijke trainingsdevice bepaald. Ze
worden hier niet opnieuw uit FP32-masters gequantiseerd.

## Routes en reconstructiecontrole

| Route | Betekenis | Schaal t.o.v. opgeslagen E4-bron | BN-bank |
|---|---|---:|---:|
| `source8` | Opgeslagen E4M3 | 1 | 8 |
| `trained6` / `trained4` | Opgeslagen getrainde lagere modi | 64 | 32 / 4 |
| `A6` / `A4` | Exponent-MSB, eventueel mantissatruncatie | 1 | 32 / 4 |
| `B6` / `B4` | Zelfde codes als A met schaaloffset +6 | 64 | 32 / 4 |
| `C6` / `C4` | E4/64 nearest-even, eventueel mantissatruncatie | 64 | 32 / 4 |

De oorspronkelijke B-training gebruikt bronpayloadschema
`mx-triple-scale64-training-source-v1` en afgeleid schema
`mx-triple-scale64-training-derived-v1`. De nieuwe A/B/C-exports krijgen een
afzonderlijk schema `mx-triple-scale64-trained-posthoc-derived-v1`.

Voor alle opgeslagen bronmodi moeten checkpointprovenance en volledige
auxiliaire modelstaat exact gelijk zijn. De opgeslagen lagere modi moeten
naar de correcte opgeslagen E4-bestandshash verwijzen. De nieuw afgeleide
B6/B4 moeten exact dezelfde codes, schaalbytes, gedecodeerde gewichten en
bevroren modelstaat bevatten als `trained6`/`trained4`.

Die code-/staatgelijkheid is een harde integriteitsvoorwaarde. De herhaalde
inference legt daarnaast logit- en voorspellingverschillen vast. Logits moeten
numeriek overeenkomen binnen `rtol=1e-4, atol=1e-5`; bitexacte verschillen
worden zichtbaar gerapporteerd. De oorspronkelijke CUDA-backend blijft
behouden en geeft geen bitexacte herhaalgarantie.

## Uitvoering

Per checkpoint zijn er negen routes en twaalf gepaarde vergelijkingen:
`trained6/4` naar elk van A/B/C, plus A→B, B→C en A→C per lagere precisie.
Totaal: 18 metricregels en 24 gepaarde regels voor twee checkpoints.

Alle afgeleide exports worden eerst opgeslagen en opnieuw geladen. De
modellen hebben uitsluitend FP32-inferentiebuffers voor de zes MX-gewichten;
de bijbehorende FP32-masterparameters ontbreken. Overige parameters en alle
BN-bankstaten komen uit dezelfde bronexport. Modellen staan in eval-modus,
gradients staan uit. De volledige modelstaathash moet voor en na inference
exact gelijk blijven. Er zijn nul optimizerupdates.

Alle routes krijgen dezelfde 26.032 officiële SVHN-testbeelden, dezelfde
labels en volgorde, batches van 128, dezelfde Resize40/ToTensor/Normalize
preprocessing als bij training. Train- en extra-data worden niet ingelezen.
Datasetbestand, labels, broncode, pakketversie en runtime worden gecontroleerd.

Per beeld worden logits, voorspellingen en hard cross-entropy opgeslagen.
Accuracy, top-5, correct-aantallen en klasseconcentratie volgen de bestaande
conventies. Zeer grote eindige CE of lage accuracy is een onderzoeksuitkomst;
niet-eindige logits/CE, corrupte codes of gewijzigde bronbestanden zijn fouten.

De drie opnieuw gemeten opgeslagen modi worden vergeleken met hun historische
trainingsaccuracies. Verschillen worden gerapporteerd; de actuele herhaalde
meting is de referentie voor de gepaarde A/B/C-vergelijking.

## Historische vergelijking over drie trainingen

`prior_posthoc` bevat de ongewijzigde `metrics.csv`, `paired_metrics.csv` en
`run_report.json` uit de eerdere echte A/B/C-feedbackbundel. Een manifest
bewaart hun hashes en de oorspronkelijke bundelhash. Die eerdere proef bevat
36 metricregels voor directe en naïeve training, ieder op best54 en epoch100.

De nieuwe runner verifieert dat beginmodelstaat, dataset, preprocessing,
runtime en backend overeenkomen met de B-training. Hij kopieert de historische
tabellen naar de nieuwe resultaatmap zonder eerdere modellen te evalueren.
De notebook houdt historische en nieuwe metingen expliciet uit elkaar.
Gepaarde voorspellingverschillen worden uitsluitend binnen één training en
één checkpoint berekend. Een figuur over trainingen is een beschrijvende
vergelijking, geen onafhankelijke herhaling of cross-training causaliteitsproef.

## Bestanden

- `metrics.csv`: 18 actuele metingen.
- `paired_metrics.csv`: 24 actuele gepaarde vergelijkingen.
- `weight_pair_metrics.csv`: 216 regels, 18 vergelijkingen ×6 lagen ×2 checkpoints.
- `layer_code_metrics.csv`: 108 regels, 9 routes ×6 lagen ×2 checkpoints.
- `historical_reproduction.csv`: 6 herhalingen van de opgeslagen trainingsmodi.
- Twee `scale64/*_per_image_outputs.npz`-bestanden.
- Afgeleide modellen, checksums, herkomst, bevroren-staatcontroles en verslag.
- Historische vergelijkingstabellen en één lege, uitvoerbare notebook.

De feedbackbundel bevat geen modelbestanden of dataset. Het analyseprogramma
leest de CSV-floats met round-trip parsing, zodat ook zeer grote eindige
cross-entropies zonder kunstmatige decimale afwijking worden herladen.

## Grenzen van de conclusie

A→B verandert alleen de schaal bij dezelfde codes. B→C verandert de
elementmapping bij dezelfde schaal. Hun accuracy-effect omvat telkens het
gehele bevroren netwerk en de aansluiting op de al getrainde BN-statistieken.
De twee effecten hoeven niet onafhankelijk of optelbaar te zijn.

Route B is de eigen trainingsroute; A en C zijn wijzigingen achteraf.
Een C-uitkomst op dit B-getrainde model bewijst geen optimale C-training.
De factor 64 is vooraf vastgelegd, geen geleerde of op deze testset gezochte
schaal. Gewichtsafwijkingen worden gemeten tegenover opgeslagen MX-routes,
niet tegenover ongeopende FP32-masters.

Dit betreft één seed. De twee checkpoints van dezelfde training zijn geen
onafhankelijke herhalingen. E4M3-best is op de officiële testset geselecteerd;
de score is geen onafhankelijke validatieschatting. Deze proef selecteert
geen nieuwe route, schaal of checkpoint op basis van de nieuwe resultaten.
