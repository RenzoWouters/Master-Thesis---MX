# Geheugentransfers bij initiële loading en precisiewissels — v1

Deze proef vergelijkt de zes bestaande SVHN CNN-8-modellen: GitHub met vijf
W/A-modi, twee MX-modi direct en met mantissatruncatie, drie MX-modi direct
B32, en de truncatieketen met vaste ×64-schaal bij B32 en B64.

**Geen training, datasetdownload of nieuwe accuracy-evaluatie.** De runner
gebruikt alleen de Python-standaardbibliotheek. De notebook gebruikt de
gebruikelijke NumPy/pandas/Matplotlib-installatie. Alle bestaande checkpoints,
broncode en eerdere resultaten worden uitsluitend gelezen. Iedere opdracht
maakt een nieuwe resultatenmap.

## Installatie en uitvoeren (PowerShell)

Pak de map `mx_precision_switch_transfer_eval_v1` uit de ZIP uit onder
`C:\Users\renzo\master-thesis-mx\software_checks`.

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_precision_switch_transfer_eval_v1\run_check.py" check
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_precision_switch_transfer_eval_v1\run_check.py" run
```

Voer `run` uit na een geslaagde `check`. Beide opdrachten drukken de nieuwe
resultatenmap en de eindstatus af. Open na `run` de notebook
`MX_precisiewissels_geheugentransfers.ipynb` in die resultatenmap en voer
Run All uit. De notebook bevat grafieken en tabellen en start geen training.
Deel daarna de `feedback_bundle.zip` uit diezelfde resultatenmap.

De automatische broncontrole zoekt de bekende volledige trainingen onder
`mx_runs` en de GitHub-referentie onder `reference_runs`. Ze vergelijkt bestaande checkpoint-/exportbestanden met de eerder
gerapporteerde SHA-256-hashes. `.pt`-bestanden worden niet als pickle geladen.
Wanneer een bron niet wordt gevonden, meldt het rapport welke ontbreekt.
Voor verplaatste runs kan een JSON-object met `model_id: runmap` via
`--sources sources.json` worden opgegeven. Een override verandert de verwachte
identiteit of hash niet. Voor een afwijkende projectmap is `--project-root`
beschikbaar.

## Wat wordt onderzocht?

De geheugeninterface in dit model ligt tussen een externe parameteropslag en
een beperkte lokale parametercache. Eén request leest een aaneengesloten
code-, schaal- of parameterarray van een laag. Iedere laag wordt eenmaal per
inference bezocht. Binnen die laag wordt hergebruik voor alle MAC-operaties
verondersteld; extra tile-herlezingen en accumulator-spills worden niet
gesimuleerd. De toekomstige hardware moet zulke aannames nog toetsen.

De runner bewaart drie afzonderlijke grootheden:

1. **Nuttige payloadbytes:** werkelijk benodigde code- en parameterinformatie.
2. **Transfers:** gehele geheugenlijnen inclusief uitlijning en cachemisses.
3. **Opslag:** alle benodigde modi samen in de expliciet gekozen indeling.

Dit zijn berekende aantallen bytes/transacties, geen gemeten GPU-DRAM-verkeer,
cycli, energie, PyTorch-allocaties of `.pt`-bestandsgrootte.

### Initiële loading

De laadgrafiek toont een koude eerste toegang voor ieder precisieniveau en
een koude lezing van alle operationele modidata. Dit onderscheidt het laden
van alleen de startmodus van het beschikbaar maken van alle modi.

Voor de daaropvolgende traces bestaan twee beleidskeuzes:

- **Laden op aanvraag (`none`):** begin met een lege parametercache. Het
  gewichtverkeer van de eerste inference wordt als initiële laadfase getoond.
  De resterende 63 inferences vormen de vervolgfase.
- **Alles vooraf lezen (`all_regions`):** lees de hele operationele afbeelding
  eerst door dezelfde cache en voer daarna 64 inferences uit. Die voorlaadfase
  wordt apart geteld. Als de afbeelding niet in de cache past, worden oudere
  lijnen verdrongen; vooraf lezen garandeert dan geen volledige residentie.

In beide gevallen geldt exact: initiële laadbytes + vervolgbytes = totaal.
Het voorladen van ongebruikte modi wordt niet als gratis behandeld.

### Precisiereeksen

Iedere reeks gebruikt 64 verschillende invoerbeelden (batchgrootte 1), zonder
activatiehergebruik tussen beelden. Er worden geen beelden echt ingelezen;
hun tensorvormen bepalen het verkeer.

- Afwisselend hoogste/laagste precisie: 63 wissels.
- Acht inferences per hoogste/laagste precisie: 7 wissels.
- Alle ondersteunde modi cyclisch: aanvullend resultaat. Het aantal modi
  verschilt tussen de modellen, dus dit is geen identieke precisiemix.

Ook bij hoog/laag verschillen de absolute precisies: GitHub wisselt W32/W1,
dual-MX W6/W4 en triple-MX W8/W4. Alleen binnen het dual-paar en het triple-B32-
paar zijn de precisies gelijk. GitHub dient als contextreferentie; de
verschillende accuracies en activatieformaten staan afzonderlijk vermeld.

De hoofdgrafieken gebruiken 64-byte-transfers en buffercapaciteiten van
0, 16, 64, 128, 256 en 1024 KiB. Een gevoeligheidscontrole gebruikt daarnaast
16- en 32-byte-transfers bij 64 KiB en de afwisselende reeks. Een KiB is 1024
bytes. De cache is volledig associatief met LRU-vervanging en alleen-lezen
parameterlijnen; dit is een expliciet analytisch model, geen claim over een
bestaande cache of streamer. Elke nieuwe combinatie begint onafhankelijk.

### Eerlijke vergelijking van truncatie

De directe modellen bewaren afzonderlijke compacte codearrays per modus en
lezen alleen de gevraagde modus. Gemeenschappelijke eerste/laatste FP32-lagen
worden eenmaal opgeslagen. De actieve BN-bank blijft modusafhankelijk.
Direct E2M3 en E2M1 delen één schaalarray: dezelfde bron, groepen en
schaalselectie met emax=2 leveren dezelfde schaalcodes. Bij triple-direct is
de E4M3-schaalarray afzonderlijk. GitHub deelt één FP32-laagschaal E tussen de
vier lage W-modi; W32 bewaart de al geschaalde FP32-gewichten.

Elk truncatiemodel krijgt twee uitvoeringsvarianten, met exact hetzelfde
numerieke model:

- **Selectief:** compacte afzonderlijke codedelen per laag. Voor twee MX-modi
  een basis van vier bits plus twee extra mantissabits; voor drie MX-modi nog
  twee extra exponentbits. Lagere modi lezen alleen hun benodigde delen.
- **Volledige bron:** iedere modus vraagt de complete E2M3- of E4M3-bron op.
  Die kan wel in de parametercache worden hergebruikt.

De bitselectie wordt exhaustief gecontroleerd op alle 254 eindige E4M3-codes
en alle 64 E2M3-codes. Dense packing wordt afzonderlijk gecontroleerd.
De ×64-schaal wordt via de bestaande exponentoffset +6 afgeleid, zonder extra
opgeslagen schaalarray. Dit kost logica, waarvan latency/energie hier niet
worden geschat.

De layouts zijn **voorgestelde compacte inference-indelingen**. Dit pakket
converteert de bestaande checkpointtensors niet naar een nieuwe export en
pretendeert niet dat de huidige PyTorch-training deze fysieke layouts al
gebruikt. De checkpointcontrole verifieert bronidentiteit; de transfertelling
hangt af van tensorvormen en layouts, niet van individuele gewichtswaarden.

### Gewicht- en activatieverkeer apart

Gewichtverkeer is uitgesplitst in codes, schalen en overige inferenceparameters.
De laatste categorie bevat de gedeelde FP32-eerste/laatste laag en de gekozen
BN-bank. BN-tellers en niet-gebruikte BN-banken tellen niet als opgehaalde
inference-operanden. Ze blijven aanwezig in de oorspronkelijke checkpoints.
Dit verklaart kleine opslagverschillen met de eerdere notebook, die alle vijf
BN-banken en hun tellers in de exportkosten meenam.

De daadwerkelijke netwerkinput is **3×40×40**: SVHN-beelden van 32×32 worden
door de bestaande datatransformatie vergroot. Het activatiemodel gebruikt
de geauditeerde CNN-8-laagvormen. Per samengevoegde conv/BN/pool/activatiefase
wordt de ingang eenmaal gelezen en de uitgang eenmaal geschreven. Interne
tussenresultaten, im2col, spillverkeer en opnieuw lezen door tiling vallen
buiten dit model. Er is een aparte activatiebuffer; activaties verdringen
geen parameterlijnen in de gemodelleerde cache.

MX gebruikt in alle modi A32. GitHub gebruikt gekoppelde W/A-precisies, met
FP32-invoer, een laatste niet-gequantiseerde activatie en FP32-logits. Voor de
lage GitHub-activaties wordt compacte opslag als scenario aangenomen; de
bestaande PyTorch-tensors zijn ook daar FP32. De notebook houdt gewicht- en
activatiegrafieken daarom gescheiden. Het pakket simuleert inference; FP32
trainingsmasters, gradiënten en Adam-verkeer worden niet meegeteld.

## Instellingen en herhaalbaarheid

`experiment_config.json` bevat alle instellingen. `run_results.json`, CSV-
tabellen, de broncontrole en een kopie van die configuratie komen in de nieuwe
resultatenmap. Gegevens uit eerdere trainingen worden niet aangepast.
`data/accuracy_context.json` bevat uitsluitend eerder gemeten epoch-100-
accuracies voor context; deze proef meet geen nieuwe accuracies.

De B32-directe en B32-truncatiemodellen geven de vergelijking bij dezelfde
groepering. De B64-truncatievariant onderzoekt ook groepsgrootte en padding;
verschillen met B32 mogen niet uitsluitend aan truncatie worden toegeschreven.
Modelmetadata en geauditeerde broncode staan onder `data` en `provenance`.

Een ontwikkelcontrole zonder toegang tot Windows-checkpoints kan met
`check --self-test-only`. `reference` produceert alleen een expliciet als
REFERENCE_ONLY aangeduide berekening vanuit de bevroren metadata. Die status
mag niet worden verward met de brongeverifieerde `run` op de eigen projectmap.
