# Mantissa-MSB verwijderen en corrigeren met de exponent

## Doel en afbakening

Dit is de eerste, verkennende proef voor de hypothese van de begeleider. We blijven bij de huidige standaardformaten **E2M3 naar E2M1**, met A32-activaties, B32-groepen en dezelfde onafhankelijke E2-shared scales. Er is geen nieuw getalformaat nodig.

De proef bestaat uit twee onderdelen:

1. Een exhaustieve, exacte numerieke controle van alle 64 E2M3-bitpatronen.
2. Een evaluatie van het reeds getrainde epoch-100-model op de volledige SVHN-testset, zonder training of BN-herkalibratie.

Het tweede onderdeel moet op je laptop worden uitgevoerd. De echte compacte model-export en SVHN-data zitten niet in dit pakket. Er staan vooraf geen nieuwe netwerkaccuracies in de resultaten.

## Precieze hypothese en eerlijke exponentkeuze

Schrijf een E2M3-element als tekenbit s, exponentveld E en opgeslagen mantissabits m2, m1, m0, van hoog naar laag. De uitvoer is standaard E2M1: één tekenbit, twee exponentbits en één mantissabit.

| Methode | Uitvoermantissa | Uitvoerexponent |
| --- | --- | --- |
| `lsb_trunc` | m2 | E ongewijzigd; bestaande methode |
| `rne` | afronding naar dichtstbijzijnde E2M1-waarde | eventuele afrondingscarry, ties-to-even en saturatie |
| `msb_bestexp` | m0, dus m2 en m1 worden verwijderd | beste van alle vier toegestane exponentvelden |

Voor `msb_bestexp` minimaliseren we voor iedere broncode de absolute fout tegenover de gedecodeerde E2M3-waarde. Bij gelijke fout kiezen we eerst de kleinste verandering van het exponentveld en daarna het kleinste exponentveld. Het teken en de shared scale blijven gelijk. Subnormals, signed zero en het maximale bereik zijn inbegrepen.

Deze volledige zoektocht levert exact dezelfde mapping als de eenvoudige bitregel:

```python
new_mantissa = old_mantissa & 1
increment = int(old_mantissa == 0b110)
new_exponent = min(3, old_exponent + increment)
```

De evaluator gebruikt deze bitregel en controleert haar exhaustief tegen de exacte zoektocht. Het gaat om **de kleinste lokale bronfout onder de beperking dat m0 behouden blijft**. Dit is geen keuze op basis van testaccuracy en geen bewijs dat deze mapping de hoogst mogelijke netwerkaccuracy geeft.

## Wat al numeriek is vastgesteld

Bij shared scale 1 en een uniforme weging van alle 64 broncodes:

| Methode | MAE | MSE | Grootste absolute fout |
| --- | ---: | ---: | ---: |
| Huidige LSB-truncatie | 0,375000 | 0,30078125 | 1,5 |
| Afronding naar E2M1 | 0,281250 | 0,19140625 | 1,5 |
| MSB verwijderen + beste exponent | 0,562500 | 0,72265625 | 3,0 |

De MSB-variant is op 8 codes beter, op 18 slechter en op 38 gelijk aan de bestaande truncatie. Ze heeft 12 dalingen in de oplopende reeks niet-negatieve bronwaarden en is dus niet monotoon. De rekenmethode gebruikt exacte rationale getallen; de uitkomsten zijn geen SVHN-resultaten of schatting van de werkelijke gewichtsverdeling.

`numerical_reference` bevat de werkelijk uitgevoerde codecontrole, de volledige mapping en een notebook. Een gewijzigd bitbudget, andere shared scale, andere behouden mantissabit of andere betekenis van de exponent valt buiten deze afgebakende proef.

## Installeren

Pak de ZIP uit in:

```text
C:\Users\renzo\master-thesis-mx\software_checks\
```

Daarna staat bijvoorbeeld dit bestand op:

```text
C:\Users\renzo\master-thesis-mx\software_checks\mx_mantissa_msb_exponent_probe_v1\run_probe.py
```

Open PowerShell in `C:\Users\renzo\master-thesis-mx`. Gebruik je bestaande `.venv-train`. Er zijn geen automatische installaties, upgrades of downloads.

## 1. Korte technische controle

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_exponent_probe_v1\run_probe.py" check
```

Dit controleert de numerieke regels, de echte bronexport en vijf modelvarianten met drie synthetische beelden. Er wordt geen accuracy op SVHN bepaald en er wordt niets getraind.

## 2. De daadwerkelijke accuracyproef

Als `check` slaagt:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_exponent_probe_v1\run_probe.py" evaluate
```

Deze opdracht verwerkt de volledige testset eenmaal per variant:

- Ongewijzigde E4M3-bron als controle.
- Ongewijzigde E2M3-bron als controle.
- E2M1 met de huidige LSB-truncatie.
- E2M1 met afronding.
- E2M1 met MSB-verwijdering en optimale beperkte exponentkeuze.

Er zijn vijf inferencepasses per minibatch, geen trainingsepochs. De looptijd wordt gemeten; er is geen vooraf gemeten laptopduur voor deze nieuwe evaluator.

De bron is de compacte export `mx_source_MXFP8_E4M3_last.pt` uit:

```text
C:\Users\renzo\master-thesis-mx\mx_runs\mx_triple_independent_e2_scale_freeze50_reference100\20261113_140915_385293_joint_run\
```

Bewaar ook het bijbehorende bestand `mx_source_MXFP8_E4M3_last.pt.sha256`. De evaluator controleert zowel de vastgelegde SHA-256 uit jouw feedback als deze sidecar, de epoch en de herkomst van het checkpoint. De globale beste-E4-export van epoch34 en de branch-best-export van epoch51 worden niet gebruikt.

Zijn je bestanden verplaatst, geef bij `check` en `evaluate` een expliciet pad:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_mantissa_msb_exponent_probe_v1\run_probe.py" evaluate --source-export "VOLLEDIG_PAD\mx_source_MXFP8_E4M3_last.pt" --data-root "VOLLEDIG_PAD\svhn"
```

`--data-root` moet de map met `test_32x32.mat` aanwijzen. Train en extra worden niet geladen. CUDA is standaard vereist; er is geen stille CPU-fallback. Een expliciete `--device cpu` is mogelijk voor technische diagnose, maar kan van de historische GPU-evaluatie verschillen.

## Uitvoer en feedback

Iedere opdracht maakt een nieuwe map onder:

```text
mx_runs\mx_mantissa_msb_exponent_probe_v1\
```

Gebruik voor feedback de **`feedback_bundle.zip` van de opdracht `evaluate`**. De runner print het volledige pad. Ook bij een fout wordt een bundel met de foutmelding en traceback gemaakt.

Belangrijkste bestanden:

| Bestand | Inhoud |
| --- | --- |
| `run_report.json` | Technische status, bronidentiteit, omgeving, waarschuwingen en bewijsgrenzen |
| `numeric_summary.json` | Exacte uniforme codeanalyse; verschillende populaties afzonderlijk |
| `codebook_mapping.csv` | Alle 64 broncodes maal drie omzettingen |
| `accuracy.csv` | Werkelijk gemeten SVHN-accuracy, correcte aantallen en testloss; alleen na volledige evaluatie |
| `weight_error.csv` | Fout tegenover de E2M3-bron per MX-laag en gezamenlijk; padding uitgesloten |
| `source_code_histogram.csv` | Werkelijke E2M3-codefrequenties per laag |
| `predictions.csv` | Voorspellingen van alle varianten voor dezelfde testbeelden |
| `paired_predictions.csv` | Welke beelden door de bestaande en nieuwe methode juist of fout worden geclassificeerd |
| `hardware_logic.json` | Gecontroleerde bitregel en beschrijving van benodigde informatie |
| Notebook en `figures` | Tabellen en figuren uit de beschikbare meetbestanden |

De oorspronkelijke testaantallen van epoch100 worden opnieuw gecontroleerd voor E4M3, E2M3 en de bestaande E2M1-methode. Verschillen worden expliciet gerapporteerd en moeten bij de interpretatie worden bekeken. Nieuwe resultaten worden nooit door de historische cijfers vervangen.

Figuren zijn optioneel: als matplotlib ontbreekt blijven CSV/JSON-resultaten beschikbaar, met een waarschuwing. Installeer of upgrade niet automatisch pakketten om die waarschuwing te verhelpen; de feedbackbundel volstaat om de figuren achteraf te maken.

## Wat deze proef kan aantonen

De numerieke controle bepaalt welke lokale bronfout onvermijdelijk is als het oorspronkelijke laagste mantissabit behouden moet blijven. De netwerkevaluatie meet wat deze omzetting doet op het bestaande getrainde model, met de echte gewichten en schalen.

De gewichten én de E2M1-BN-bank zijn eerder met LSB-truncatie getraind. Daarom sluit een slechtere posthocaccuracy niet uit dat aangepaste quantization-aware training of BN-aanpassing een deel van de daling opvangt. Dat zou een afzonderlijke vervolgproef zijn. Er wordt in dit pakket geen training gestart.

Een technische PASS betekent dat de proef correct is uitgevoerd, niet dat de nieuwe methode beter of slechter scoort. De hele richting wordt evenmin weerlegd door één afgebakende mapping op één checkpoint.

## Hardwarekosten

Er blijft een vierbits-E2M1-code over en de bestaande E8M0-schaal blijft gelijk. De afleiding vereist een controle op mantissapatroon110, een eventuele exponentincrement en saturatie. Een runtimezoekactie, logaritme of nieuw getalformaat is niet nodig.

De twee verwijderde bits zijn wel nodig tijdens die conversie. De invloed op geheugentoegang hangt af van het moment waarop de conversie wordt uitgevoerd en hoe de bitlanes worden opgeslagen. De FP32-evaluatietijd van deze software is geen meting van low-bit-hardwarelatency of energie. Er zijn geen area-, energie- of bandbreedtewinsten ingevuld.

## Herkomst en reproduceerbaarheid

- Bestaande codecs/adapters zijn ongewijzigd overgenomen uit het gevalideerde freeze50-full100-pakket.
- CNN-8 en de datasetcode zijn ongewijzigd uit de meegeleverde Any-Precision-bron-ZIP gehaald.
- `REFERENCE.json` legt de werkelijke bronexport, dataset en historische controletellingen vast.
- `PACKAGE_SHA256.json` controleert de uitvoercode, documentatie en vaste referentie. De opnieuw uitvoerbare resultaten in `numerical_reference` vallen buiten die controle.
- `tests/test_rules.py` test de numerieke mapping onafhankelijk met exacte getabelleerde decoderwaarden.
- `tests/test_probe.py` controleert de tensoromzetting en evaluatie met synthetische fixtures; die tests zijn geen onderzoeksresultaten.
- Voor de numerieke codecontrole zonder Torch: `python run_probe.py numeric`.

De ruwe feedbackbundel, oorspronkelijke exports, trainingscheckpoints en SVHN-bestanden worden uitsluitend gelezen. Alle nieuwe uitvoer komt in een nieuwe resultatenmap.
