# Protocol — exponent en mantissa op dezelfde directe MX-checkpoints

## Onderzoeksvraag en vergelijkingsbasis

Wat verandert er wanneer we de opgeslagen MX-codes van één vast, gezamenlijk
getraind netwerk achteraf trunceren? De proef herhaalt mantissatruncatie en
voegt naïeve exponenttruncatie toe op dezelfde twee checkpoints, zodat
verschillende trainingsgewichten geen extra verschil tussen de proeven vormen.

De bron is `mx_weights_triple_direct_reference100_v1`: CNN-8/svhnq expand=2,
SVHN train-only, E4M3/E2M3/E2M1 onafhankelijk direct uit FP32 G(W), A32,
FP32-master/Adam. De brontraining gebruikte recursieve distillatie en geen
truncatie. De opgeslagen checkpoints zijn niet op deze nieuwe routes getraind.

De twee vaste rollen zijn `last_epoch` (100) en `best_E4M3_test` (54 voor de
aangeleverde run). Best is eerder gekozen als het eerste strikte maximum van
E4M3-testaccuracy. Geen post-hocscore verandert deze selectie. Alle routes van
een rol gebruiken dezelfde checkpointprovenance en auxiliaire modelstaat.
Bestselectie op de officiële testset is geen onafhankelijke testschatting na
selectie; de twee checkpoints uit één seed zijn geen onafhankelijke herhalingen.

## Bronnen, blokken en schaalbeleid

De zes directe MX-modelbestanden uit de bestaande run worden gecontroleerd op
hun rapporthash, bestandsgrootte, checksumsidecar, schema, formaat en
checkpointprovenance. Het volledige trainingsprotocol, de broncodeversie,
omgeving, backend, 600 metricrijen en oorspronkelijke selectie worden eveneens
gecontroleerd. Voor iedere rol zijn de drie directe exports afkomstig uit
hetzelfde checkpoint, met identieke auxiliaire FP32-staat.

De zes MX-convoluties zijn `layers.3`, `.6`, `.10`, `.13`, `.16` en `.20`.
Elke gewichtstensor is per outputkanaal over Cin*Kh*Kw afgevlakt en in blokken
van 32 verdeeld, met positieve nulpadding. Schaalcodes zijn E8M0; code 255 is
ongeldig. De directe codecs en hun padding/packing blijven ongewijzigd.

Directe E2M3 en E2M1 gebruiken dezelfde blokschalen. Direct E4M3 gebruikt de
bestaande eigen schaalregel. Voor een gewoon niet-nulblok zonder schaalclipping:

- E4M3: schaalexponent floor(log2(amax)) − 8.
- E2M3/E2M1: schaalexponent floor(log2(amax)) − 2.

De E4-schaal is dan 2^−6 maal de E2-schaal. Nulblokken en exponentclipping
kunnen hiervan afwijken. Deze proef verlangt daarom geen gelijke E4- en
E2-schalen, maar rapporteert hun geobserveerde verschillen. Iedere afgeleide
route kopieert de schaalbytes van haar eigen bron exact, ook bij nulblokken.
Er vindt geen schaalherberekening, correctie, offset of learning plaats.

## Exacte bitbewerkingen

### Mantissa: E2M3 naar E2M1

De ongepakte uint8-container bevat een zesbitcode met indeling
`s e1 e0 m2 m1 m0`. De doelcode is `s e1 e0 m2`.

```python
u4 = u6 >> 2
```

Teken, beide lokale exponentbits en de hoogste mantissabit blijven behouden.
De twee laagste mantissabits verdwijnen. Dit is geen nearest rounding.
De E8M0-schaalbytes van de directe E2M3-bron blijven identiek.

### Exponent: E4M3 naar E2M3

De bronindeling is `s e3 e2 e1 e0 m2 m1 m0`.
De doelindeling is `s e3 e2 m2 m1 m0`.

```python
sign = u8 >> 7
exponent4 = (u8 >> 3) & 15
mantissa3 = u8 & 7
u6 = (sign << 5) | ((exponent4 >> 2) << 3) | mantissa3
```

Alle drie mantissabits en het teken blijven behouden; alleen de twee hoogste
bits van het vierbit-exponentveld blijven over. De doelcode wordt als E2M3
geïnterpreteerd met zijn gewone bias 1. De bronbias 7 wordt niet gecompenseerd.
Er is geen encode_tensor, nieuwe rounding of G(W)-bewerking in deze bitroute.
De schaalbytes blijven die van de E4M3-bron.

De twee E4M3-NaN-codes 127 en 255 worden geweigerd. Alle 254 eindige codes,
waaronder signed zero en lokale subnormals, worden exact volgens bovenstaande
mapping omgezet. Een bronwaarde kan hierdoor nul worden, groter worden of
kleiner worden. Het is dus geen uniforme vermenigvuldiging van alle gewichten.
Voor bron- en doelwaarden die beide normaal zijn (bronexponent e≥4), is de
verhouding van lokale doelwaarde tot bronwaarde:

`2 ** (floor(e / 4) - e + 6)`.

Bijvoorbeeld: e=8 geeft factor 1; e=12 factor 1/8; e=15 factor 1/64.
Dit zijn voorbeelden van de vastgelegde bitinterpretatie, geen correcties.

### Combinatie

`combined864` past eerst de exponentbewerking toe en verschuift vervolgens de
verkregen E2M3-code twee plaatsen rechts. De uiteindelijke indeling is
`s e3 e2 m2`, geïnterpreteerd als E2M1. De oorspronkelijke E4M3-schaal blijft
ook na beide stappen behouden. Deze route onderzoekt de gezamenlijke werking;
ze vervangt niet de afzonderlijke exponent- en mantissavergelijkingen.

## Acht routes en afrondingscontroles

| ID | Bron | Bewerking | Doel | Directe vergelijkingsbasis |
|---|---|---|---|---|
| direct8 | Opgeslagen directe export | Geen | E4M3 | direct8 |
| direct6 | Opgeslagen directe export | Geen | E2M3 | direct6 |
| direct4 | Opgeslagen directe export | Geen | E2M1 | direct4 |
| round64 | direct6 | Lokale RNE met bronschaal | E2M1 | direct4 |
| mantissa64 | direct6 | Mantissabitshift | E2M1 | direct4 |
| round86 | direct8 | Lokale RNE met bronschaal | E2M3 | direct6 |
| exponent86 | direct8 | Hoogste exponentbits | E2M3 | direct6 |
| combined864 | direct8 | Exponent, dan mantissa | E2M1 | direct4 |

RNE betekent nearest rounding met ties-to-even en de bestaande eindige
saturatie. De controles decoderen lokale elementen en voeren de elementencoder
van het doelformaat uit. Ze roepen nooit encode_tensor op gedecodeerde gewichten
aan, want dat zou de schaal opnieuw kiezen.

Bij mantissa vergelijkt direct4→round64 de extra bronafronding onder dezelfde
E2-schaalregel, zoals in de eerdere proef. round64→mantissa64 vergelijkt de
omzettingen vanaf dezelfde opgeslagen E2M3-bron.

Bij exponent omvat direct6→round86 óók het verschil tussen E4- en E2-bronschalen.
Dit mag daarom niet uitsluitend dubbelafronding worden genoemd. Bij de vaste
E4-schaal kunnen lokale E4-waarden boven 7,5 liggen, buiten het E2M3-bereik.
De RNE-controle verzadigt dan naar het grootste eindige doelelement. De
saturatietelling wordt opgeslagen. round86→exponent86 vergelijkt twee
omzettingen vanaf precies dezelfde E4-broncodes en dezelfde E4-bronschaal.

Alle veranderingen in classifieruitvoer zijn niet-lineair. De numerieke
verschillen tussen controlepaden vormen geen eenvoudige optelsom van causale
effecten op accuracy.

## BatchNorm en modeltoestand

Elke route gebruikt de bestaande getrainde BN-bank van haar doelprecisie:
E4M3 sleutel8, E2M3 sleutel32, E2M1 sleutel4. Het zijn opslag-ID's; activaties
blijven overal A32. De vergelijking met de directe doelmodus houdt dus BN,
alle FP32-uitzonderingslagen en overige modelstaat vast.

De vijf afgeleide modelbestanden per checkpoint worden werkelijk opgeslagen
en herladen vóór reconstructie. Ze bevatten geen FP32-masters van de zes
MX-lagen. Alle acht uitvoermodellen reconstrueren deze lagen uit elementcodes
en E8M0-schalen. Gedecodeerde FP32-buffers zijn uitsluitend emulatie-operanden;
dit is geen native MX-kernel of meting van hardwareversnelling.

Alle modules staan in eval(), parameters zijn bevroren en inferentie gebeurt
onder no_grad(). Voor en na iedere complete checkpoint-evaluatie wordt de
volledige model- en BN-staat gehasht. Er wordt geen optimizer aangemaakt,
backward uitgevoerd, BN herkalibreerd of getraind. Truncatie kan de
gewichtsverdeling veranderen ten opzichte van de verdeling waarvoor die BN
getraind is. Dat hoort bij deze ongewijzigde post-hocproef.

## Data, metingen en integriteit

Uitsluitend de volledige officiële SVHN-testset: 26.032 beelden, batch128,
shuffleFalse, workers0, pin_memoryTrue en drop_lastFalse. De standaard
preprocessing is exact die van de referentie. Ruwe bestandhash, auteurs-MD5,
labels, volgorde en volledige dekking worden gecontroleerd. Train en extra
worden niet gelezen. Er is geen downloadroute.

Binnen iedere minibatch worden alle acht routes uitgevoerd op dezelfde input.
Top-1 volgt dezelfde topk(5)-conventie als de oorspronkelijke training, ook bij
gelijke logits. Per beeld blijven indices, labels, logits, voorspellingen en
harde CE behouden. De aggregaties gebruiken integer correcte aantallen.

Per checkpoint worden elf gepaarde vergelijkingen opgeslagen, waaronder de
onderlinge directe precisies en alle truncatie-/afrondingsvergelijkingen.
Per vergelijking: beide correct, alleen referentie correct, alleen ander pad
correct, beide fout, gewijzigde voorspellingen en netto correcte verandering.
Er wordt geen significantieclaim uit alleen marginale accuracies afgeleid.

De drie directe modi worden opnieuw geëvalueerd. Afwijkingen tegenover de
historische correcte aantallen krijgen een expliciete waarschuwing en tabel;
er is geen stille correctie of automatische tuning. De nieuwe gepaarde scores
zijn de vergelijkingsbasis. De behouden CUDA-backend garandeert geen bitexacte
herhaling. Een verschil wordt daarom gemeld, zonder de oorzaak als bewezen
backendverschil te benoemen.

De rapporten onderscheiden technische PASS van gunstige onderzoeksprestaties.
Zeer lage exponentaccuracy blijft een meetresultaat als de numerieke en
integriteitscontroles slagen. Niet-eindige logits, verkeerde bron/provenance,
gewijzigde modelstaat of ontbrekende datadekking leiden wel tot FAIL.

## Bestanden en presentatie

De feedbackbundel bevat de CSV's, twee per-beeld-NPZ's, oorspronkelijke
referentiemetadata en een lege notebook met helper. De tien nieuwe afgeleide
modellen met checksums blijven lokaal in de resultaatmap, naast de rapporten.
De oorspronkelijke runmap en testdata worden achteraf opnieuw gehasht en
blijven ongewijzigd. Nieuwe uitvoer wordt buiten de referentiemap geschreven.

De hoofdfiguur toont alle acht routes op epoch100 en op het vastgelegde
E4M3-bestcheckpoint. De notebook bevat daarnaast deltas, gepaarde overgangen,
laaggewichts-MAE, RMS-verhoudingen en audits. Voorbeelden uit de lokale
softwarecontrole zijn duidelijk synthetisch en geen onderzoeksresultaten.
