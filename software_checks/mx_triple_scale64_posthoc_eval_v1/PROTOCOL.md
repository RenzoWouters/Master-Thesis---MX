# Vast protocol: twee trainingen, schaaloffset +6 en afrondingscontrole

## Doel en vaste vergelijkingseenheid

De proef onderzoekt de lage accuracy van naïeve exponenttruncatie met twee
afzonderlijke ingrepen: eerst de gedeelde schaal veranderen bij identieke
elementcodes, daarna de elementmapping veranderen bij diezelfde schaal.
Er wordt geëvalueerd op beide bestaande honderd-epochtrainingen:

1. `direct`: E4M3, E2M3 en E2M1 tijdens training elk rechtstreeks uit dezelfde FP32-bron bepaald.
2. `truncation`: E4M3 uit de FP32-bron, E2M3 via exponent-MSB-truncatie en E2M1 via daaropvolgende mantissatruncatie, telkens met behoud van de E4-bronschaal.

Per training blijven twee oorspronkelijke checkpoints vast: `last_epoch`
(epoch 100) en `best_E4M3_test` (het eerder op E4M3-testaccuracy geselecteerde
checkpoint). De runner verifieert die eerdere selectie tegen de volledige
opgeslagen geschiedenis. Hij selecteert geen checkpoint op nieuwe A/B/C-scores.
Elke vergelijking gebruikt één checkpoint en de BN-toestand uit diezelfde
training. De twee checkpointrollen zijn geen onafhankelijke trainingsherhalingen.

## Ongewijzigde formaten en modelbeleid

| Formaat | Elementbits | Bias | Lokale maximale eindige grootte |
|---|---|---:|---:|
| E4M3 | `[s e3 e2 e1 e0 m2 m1 m0]` | 7 | 448 |
| E2M3 | `[s e1 e0 m2 m1 m0]` | 1 | 7,5 |
| E2M1 | `[s e1 e0 m0]` | 1 | 6 |

Dit zijn precies de bestaande codecs `mx_e4m3.py`, `mx_format.py` en
`mx_e2m1.py`, ongewijzigd gekopieerd. E4M3-codes 127 en 255 zijn NaN-codes en
worden geweigerd; de overige 254 codes zijn eindig. Alle 64 E2M3-codes en
16 E2M1-codes zijn eindig. Signed zero, lokale subnormals, packing en padding
blijven volgens die codecs gelden.

Blokken bevatten 32 elementen binnen een outputkanaal. Een eindige E8M0-code
`k` representeert schaal `S = 2**(k - 127)`, met `0 <= k <= 254`. Code 255 is
ongeldig. De zes oorspronkelijke MX-convoluties blijven dezelfde. De overige
FP32-lagen en alle BN-staat komen ongewijzigd uit de opgeslagen export.
Activaties blijven A32. E4M3 gebruikt BN-sleutel `8`, E2M3 `32`, E2M1 `4`;
deze sleutels betekenen hier niet dat de activatieprecisie verandert.

## Exacte afleidingen

Noem een opgeslagen unsigned E4M3-elementcode `c`, de lokale gedecodeerde
elementwaarde `v4` en de bijbehorende schaalbyte `k`.

### A: oorspronkelijke naïeve truncatie

```python
sign = c >> 7
exponent = (c >> 3) & 0b1111
mantissa = c & 0b111
a6 = (sign << 5) | ((exponent >> 2) << 3) | mantissa
a4 = a6 >> 2
k_A = k
```

Dit bewaart de hoogste twee exponentveld-bits en alle drie mantissabits in
E2M3. Het doel decodeert het overgebleven veld met zijn eigen bias. Er is geen
biascorrectie. Daarna verwijdert dezelfde logische mantissashift twee bits om
E2M1 te maken. De oude functies worden rechtstreeks hergebruikt.

### B: exact dezelfde elementcodes, schaal ×64

```python
b6 = a6
b4 = b6 >> 2
k_B = int(k) + 6
```

De optelling gebeurt in een breder integer-type vóór terugconversie naar
`uint8`. Voor alle blokken, ook volledig nulblokken, geldt exact dezelfde
offset. Broncode 248 kan nog naar 254; broncodes 249 t/m 254 leiden expliciet
tot een fout. Er wordt nooit geklemd of modulo 256 teruggesprongen. Ook
oneindige gedecodeerde FP32-inferentiegewichten worden geweigerd.

Omdat `2**(k+6-127) = 64*S`, zijn B en A op exact-reële schaal identieke
elementcodes met factor 64 verschil. De uiteindelijke inferentiebuffers blijven
FP32 volgens de bestaande emulator. B corrigeert geen individueel exponentveld
en herstelt geen verloren exponentinformatie.

### C: nearest-even elementconversie op dezelfde aangepaste schaal

```python
v4 = mx_e4m3.decode_elements(c)
c6 = mx_format.encode_elements(v4 / 64)
c4 = c6 >> 2
k_C = int(k) + 6
```

`encode_elements` is de bestaande E2M3-encoder: nearest rounding, bij een
exact midden de even laagste codebit. De verhouding /64 is exact binair voor
de eindige lokale E4M3-waarden. Het globale E4-gewicht was `S*v4`; de nieuwe
representatie is `(64*S)*decode_E2M3(c6)`.

C leest geen FP32-master, voert `G(W)` niet opnieuw uit en roept geen
`encode_tensor` aan. Die laatste functie zou opnieuw blokschalen selecteren en
hoort niet bij deze controle. C4 blijft de vaste mantissatruncatie uit C6;
het is geen onafhankelijke nearest-even conversie rechtstreeks naar E2M1.

## Betekenis van factor 64

De bestaande schaalkeuze gebruikt maximale lokale exponent 8 voor E4M3 en
2 voor E2M3. Het verschil is 6, dus de overeenkomstige verschuiving is `2**6`.
Dit is een vooraf vastgelegde controle van de schaalgrootte. Het is niet de
verhouding 448/7,5 en evenmin een garantie op herstel van accuracy.

De maximale lokale E4-grootte /64 is 7, dus C6 kan deze eindige bronwaarden
zonder lokale bovengrenssaturatie naar E2M3 omzetten. Kleine bronwaarden
kunnen wel naar nul afronden. Bij A/B is bitverwijdering geen nearest-even
conversie: de mapping hoeft niet monotoon in numerieke grootte te zijn.
Een gedeelde schaal kan dat informatieverlies niet ongedaan maken.

## Negen routes en twaalf gepaarde vergelijkingen

| Route | Betekenis | Doel-BN |
|---|---|---:|
| source8 | Opgeslagen getrainde E4M3-modus | 8 |
| trained6 | Opgeslagen getrainde E2M3-modus | 32 |
| trained4 | Opgeslagen getrainde E2M1-modus | 4 |
| A6 / A4 | Naïeve exponenttruncatie, eventueel gevolgd door mantissatruncatie | 32 / 4 |
| B6 / B4 | Dezelfde elementcodes als A, bron-E8M0-code +6 | 32 / 4 |
| C6 / C4 | Nearest-even E2M3 uit E4/64, dezelfde schaal +6, eventueel mantissatruncatie | 32 / 4 |

Per checkpoint worden de getrainde E2M3/E2M1-modus elk vergeleken met A, B
en C: zes paren. Daar komen A6→B6, A4→B4, B6→C6, B4→C4, A6→C6 en A4→C4
bij: twaalf paren per groep, 48 in totaal.

De opgeslagen lagere modi van de naïeve truncatietraining moeten code- en
schaalexact gelijk zijn aan A6/A4 van hun opgeslagen E4-bron. Dit is een
integriteitscontrole; het is geen onafhankelijke training. De direct getrainde
lagere modi zijn afzonderlijk uit de oorspronkelijke bron bepaald en worden
uitsluitend als bewaarde referenties gebruikt.

## Meting en bestanden

Alle modellen staan in eval-modus met gradients uit; BN-tellers, buffers en
parameters moeten vóór en na de meting exact dezelfde hash hebben. De zes
afgeleide exports worden eerst opgeslagen en opnieuw geladen, en pas daarna
geëvalueerd. De directe FP32-MX-masters ontbreken uit de inferentiemodellen.

De officiële SVHN-testset bevat 26.032 beelden. Alle negen routes gebruiken
dezelfde batches van 128, dezelfde labels en dezelfde volgorde. Train- en
extra-data worden niet ingelezen. Per beeld worden logits, voorspellingen en
hard cross-entropy vastgelegd. De top-1/top-5-conventie blijft die van de
bestaande evaluator. De oorspronkelijke CUDA-backend blijft behouden en
geeft geen bitexacte herhaalgarantie.

De uitvoer bevat:

- `metrics.csv`: 36 regels met accuracy, aantallen, CE en klasseconcentratie;
- `paired_metrics.csv`: 48 regels met gepaarde voorspelling- en foutveranderingen;
- `weight_pair_metrics.csv`: 432 regels, 18 vergelijkingen ×6 lagen ×4 groepen;
- `layer_code_metrics.csv`: 216 regels, 9 routes ×6 lagen ×4 groepen;
- `historical_reproduction.csv`: 12 herhalingen van opgeslagen getrainde modi;
- vier `*_per_image_outputs.npz`-bestanden onder `direct` en `truncation`;
- afgeleide modellen, hashbestanden, rapporten en één lege analyse-notebook.

Gewichtsafwijkingen worden gemeten tegenover de opgeslagen E4-bron,
getrainde doelmodus of bijbehorende A/B/C-route. Er wordt geen fout tegenover
een ongeopende FP32-master gerapporteerd. Bereikdiagnostiek mag niet worden
gelezen als bewijs dat pure bittruncatie numerieke saturatie heeft uitgevoerd.

Een verschil met historische correcte aantallen wordt afzonderlijk gemeld;
de actuele herhaalde doelmodus vormt de basis voor de gepaarde vergelijking.
Lage accuracy of voorspellingen van één klasse zijn onderzoeksresultaten.
Ongeldige codes, gewijzigde inputs, andere protocollen of niet-eindige uitvoer
zijn technische fouten.

## Grenzen van de conclusie

A→B isoleert de voorgeschreven schaalwijziging in de implementatie. B→C
isoleert de verandering van elementmapping bij dezelfde schaal. Het
accuracyverschil omvat telkens de respons van het gehele bevroren netwerk,
inclusief de aansluiting op de al getrainde BN-banken. Er volgt nog geen
uitspraak over een hertraind model, een optimale schaal of een onafhankelijke
causale optelling van beide effecten. Deze proef gebruikt twee concrete
trainingsruns met dezelfde oorspronkelijke opzet, geen verzameling seeds.

De evaluatie kiest geen nieuwe route, schaal of checkpoint op de testset als
definitieve winnaar. De resultaten worden eerst samen beoordeeld voordat
eventueel een volgende vijf-epochtraining wordt voorbereid.
