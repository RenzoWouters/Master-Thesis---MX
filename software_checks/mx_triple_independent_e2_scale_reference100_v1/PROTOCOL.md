# Protocol — onafhankelijke absolute E2-schaal, 100 epochs

## Onderzoek en herkomst

Dit is een vooraf vastgelegde verkennende proef binnen het bestaande SVHN/CNN-8-onderzoek. De werkelijke vijfepochpilot is geslaagd; er zijn nog geen resultaten van deze volledige training. De pilotbeoordeling staat in `PILOT_REVIEW.md`. De aangeleverde auteursbron en bestaande codecs worden hergebruikt. `SOURCE_MANIFEST.json`, `REUSE_MANIFEST.json` en `PACKAGE_SHA256.json` leggen herkomst en bestanden vast. Ruwe gegevens en eerdere onderzoeksresultaten blijven ongewijzigd.

Dit is numerieke gewichtsquantisatie in FP32-convoluties, geen native MX-kernel of nieuwe reproductieclaim over de paper.

## Vaste onderdelen

| Onderdeel | Afspraak |
|---|---|
| Netwerk/data | `svhnq`, expand=2; volledige SVHN train/test; geen extra-split |
| Training | 100 volledige epochs, opnieuw vanaf nieuwe initialisatie; seed 20261009; geen warmstart uit de pilot |
| Modellen | E4M3, E2M3/MXFP6, E2M1/MXFP4; A32 in alle modi |
| Master/gradients/Adam-momenten | FP32 |
| Groepering | B32; flatten Cin×Kh×Kw per uitvoerkanaal; nulvulling staart |
| G(W) | Bestaande tanh/max-transformatie en detached mean(abs(W)); afgeleiden behouden |
| Optimizer | Bestaande Adam; gewicht- en schaal-LR 0,001; weight decay 0 |
| Scheduler | Oorspronkelijke mijlpalen 50/75/90 voor gewichten én schalen; schaalparameters blijven leerbaar tot en met 100 |
| Loss | E4 hard CE; E2M3 detached soft CE van E4; E2M1 detached soft CE van E2M3; T=1 |
| Minibatch | Drie forward/backward-passes; gradients optellen; één Adam-stap |
| BN | Dezelfde drie modusbanken leren mee; evaluatie verandert ze niet |
| Selectie | Eerste strikt hoogste E4-accuracy op officiële testset |

## Schalen en initialisatie

E4M3 wordt met de bestaande codec uit G(W) bepaald. De E4-schaal blijft de bestaande dynamische schaal uit het groepsmaximum. Zij wordt in deze proef niet als parameter geleerd.

Iedere groep krijgt één FP32-parameter φ: de absolute log2 van de E2-schaal. De harde forward gebruikt:

```python
e2 = floor(phi) + (phi - floor(phi) >= 0.5)
e2_scale_code = e2 + 127
S2 = 2**e2
```

Halve waarden worden naar de hogere exponent afgerond, ook bij negatieve exponenten. De actuele E4-schaal wordt bij deze berekening niet gebruikt. E2M3 en E2M1 delen exact dezelfde E2-schaalbytes. De verhouding S2/S4 kan achteraf worden berekend; zij is een diagnose, geen parameter met tien keuzes. Zij kan buiten het oude bereik ×1–×512 vallen.

Alleen bij constructie, vóór de optimizer, wordt `phi(0) = e4(0) + 6` genomen. Daardoor is de eerste kwantisatie exact gelijk aan de vaste ×64-route. Deze stap trekt geen toevalsgetallen en verandert geen mastergewichten of BN. Zij wordt niet herhaald tijdens training of na checkpoint-herstel.

Na Adam wordt φ geprojecteerd op [-127,127], het volledige bereik van eindige E8M0-codes 0–254. Code 255 is ongeldig. Er is geen grens die uit de E4-schaal wordt afgeleid. Een eindige schaal kan met bepaalde elementcodes toch FP32-overloop geven, vooral bij exponent 126/127. De decoder meldt dat als fout; er is geen stille clipping of andere schaalfallback.

## Bitbewerking en gradients

Van de E4-code `s e3 e2 e1 e0 m2 m1 m0` blijven voor E2M3 de bits `s e3 e2 m2 m1 m0` behouden. E2M1 ontstaat door twee mantissabits te verwijderen:

```python
c6 = ((u >> 7) << 5) | (((u >> 5) & 3) << 3) | (u & 7)
c4 = c6 >> 2
```

E2 gebruikt zijn eigen exponentbias. Er is geen aanvullende biascorrectie, elementafronding of directe E2-quantisatie uit FP32. De E4-bron wordt per laag/minibatch één keer bepaald en voor de drie modi gedeeld.

Naar G(W) blijft de identity-STE behouden. Daarvóór blijven de eigen afgeleiden van G(W) actief. Voor φ wordt dezelfde gekozen surrogate als voorheen gebruikt:

`dL/dphi_g ≈ ln(2) * sum_real_group((dL/dq_i) * q_i)`

De bijdragen uit beide lagere modi worden opgeteld. E4 heeft geen rechtstreekse φ-gradient. Nulvulling telt niet mee. Er is geen nieuwe LSQ-normalisatie, extra forward of uitputtende schaalzoekactie. Continue parameterupdates en afgeronde exponentwissels worden afzonderlijk gerapporteerd.

Deze surrogate is een expliciete ontwerpkeuze, geen exacte afgeleide van de discrete keuze. Lokale gradientoptimalisatie garandeert geen globale optimale schaal, net zoals gewichtstraining geen globaal optimum garandeert.

## Rapportage en vergelijking

Elke epoch bewaart accuracies, loss, timing, updates, gradientdiagnostiek, harde schaalwissels, absolute schaaltoestand en samplevolgorde. De notebook toont epoch 100 en het gemeenschappelijke checkpoint gekozen op E4. Bij een gedeeltelijke run toont zij de werkelijk bereikte laatste epoch; er wordt geen niet-gemeten epoch-100-resultaat ingevuld.

Historische referenties, ongewijzigd en met controlesommen:

- `relative_scale10_full100_reference/`: echte 100-epochtraining met tien relatieve keuzes, init ×64; schalen leren tot 100; run 20261104_193228_406260_joint_run.
- `relative_scale10_reference/`: oorspronkelijke vijfepochreferentie, bewaard als herkomst; run 20261104_185824_003629_joint_run.
- `independent_pilot_reference/`: nieuwe onafhankelijke-schaalpilot; run 20261111_165230_352318_joint_run. Deze vijf meetpunten zijn geen volledige comparator en worden nooit als 100-epochresultaten behandeld.
- `fixed_scale64_reference/`: echte volledige vaste ×64-training, 100 epochs; run 20261024_151210_665535_joint_run.

De nieuwe parameterisatie verandert hoe schalen op gewichtsupdates reageren. Het is dus geen zuivere vergelijking van alleen het aantal schaalwaarden. De officiële testset wordt ook voor selectie gebruikt; de geselecteerde score is geen onafhankelijke testschatting. Eén seed en historische runs volstaan niet om kleine verschillen als algemene verbetering te interpreteren.

## Kosten

Voor de zes MX-lagen zijn er 4.080 groepen. Dit zijn uitsluitend schaalmetadata voor één gedeelde bronrepresentatie; elementcodes, randlagen, BN, headers en overige overhead komen apart erbij.

| Variant | E4-schaalbank | Lagere schaalinformatie | Totaal |
|---|---:|---:|---:|
| Vaste ×64 | 4.080 B | 0 B; impliciete verhouding | 4.080 B |
| Tien relatieve keuzes | 4.080 B | 2.040 B; 4 bits/groep | 6.120 B |
| Onafhankelijke E2-schaal | 4.080 B | 4.080 B; 8 bits/groep | 8.160 B |

Nieuw: +4.080 B tegenover vaste ×64 en +2.040 B tegenover tien keuzes. Per volle B32-groep zijn twee schalen samen 0,5 bit/gewicht tegenover 0,25 bij vaste ×64. Staartvulling maakt de bijdrage per werkelijk modelgewicht iets anders.

Training houdt net als de vorige geleerde variant 4.080 FP32-proxies bij. Proxy, gradient en twee Adam-momenten hebben samen nominale tensorpayload 65.280 B, exclusief framework- en allocatoroverhead. De compacte export bewaart één E4-bron en absolute E2-schaalbytes, geen φ of FP32-MX-masters. Losse modusbestanden zijn controle-exports; hun gezamenlijke bestandsgrootte is geen noodzakelijke opslagkost van de gedeelde representatie.

Er blijven drie trainingspasses en één optimizerstap. Looptijd wordt gemeten. Opslag alleen bewijst geen verandering in DRAM-transfers, latency of energie; die hangen ook van caching, geladen precisies, packing en hardware af.

## Softwarecontrole en vervolg

Controles behandelen schaalonafhankelijkheid, ×64-initialisatie, bitmapping, gradient-orakels, E8M0-grenzen, cache, exports en hervatten. Lokale CPU-fixtures zijn geen SVHN-resultaten of CUDA-validatie. Voer daarom eerst `check` in de bestaande GPU-omgeving uit. De reeds werkelijke pilot vormt aanvullend bewijs dat de numerieke route daar uitvoerbaar is.

Deze volledige run stopt uiterlijk op epoch 100, zonder freeze. Alle voltooide checkpoints blijven behouden, inclusief epoch 50. De resultaten worden daarna beoordeeld voordat een eventuele freezeproef wordt opgesteld.
