# B64 ×64 — analyse van de vijf-epochpilot

Bron: `feedback_B64_pilot5_original.zip` in deze map, de ongewijzigde gebruikersbundel `feedback_bundle(20261027-161453).zip`. SHA256: `2e47f6ea1c26d894eaa91b82b277c38f9297fe8c5a7773f48b4040643b4bbda8`. Run: `20261027_170656_785350_joint_run`. De CSV-tabellen en `audit_summary.json` naast dit document bevatten de reproduceerbare vergelijking met de twee ingebedde referenties.

## Uitkomst

De echte B64-pilot is technisch geslaagd: **5 epochs, 2.865 Adam-updates en 8.595 trainingspasses**. Alle 397 gecontroleerde consistentievoorwaarden slagen. Op epoch 5 zijn de twee lagere precisies beter dan B32 en B16; E4M3 scoort iets lager.

| Precisie | B16 ×64 | B32 ×64 | B64 ×64 | B64 − B32 [procentpunt] |
|---|---:|---:|---:|---:|
|E4M3|93,054702%|93,008605%|92,889521%|−0,119084|
|E2M3|91,452827%|91,299170%|92,148125%|+0,848955|
|E2M1|91,560387%|90,891979%|91,902274%|+1,010295|

Bij alle drie varianten is **epoch 5 ook het eerste strikte E4M3-maximum binnen epochs 1–5**. De vergelijking op het laatste checkpoint en op het E4M3-geselecteerde pilotcheckpoint valt dus samen. De drie scores per variant komen uit hetzelfde checkpoint.

De referenties zijn de eerste vijf epochs van de **volledige B16- en B32-trainingen**. De eerdere afzonderlijke B16-pilot wordt niet gebruikt. Globale full100-maxima op epochs 53 en 42 blijven buiten deze pilotvergelijking.

Tegenover B32 heeft B64 31 minder juiste voorspellingen bij E4M3, 221 meer bij E2M3 en 263 meer bij E2M1, op 26.032 testbeelden. Tegenover B16 zijn de verschillen −43 / +181 / +89 juiste voorspellingen. Dit zijn verschillen in totaalaantallen; zonder voorspellingen per beeld zeggen ze niet welke specifieke beelden verschillen.

## Vroege training: de winst verschijnt pas op epoch 5

| Epoch | B64 E4M3 | B64 E2M3 | B64 E2M1 | E2M3: B64 − B32 [pp] | E2M1: B64 − B32 [pp] |
|---:|---:|---:|---:|---:|---:|
|1|87,991703%|86,282268%|85,921174%|−0,499385|−1,025661|
|2|90,319607%|88,160725%|87,780424%|−1,409803|−1,790104|
|3|91,879226%|89,601260%|90,196681%|−0,322680|−0,487861|
|4|92,198064%|91,091733%|90,669176%|−0,034573|−0,222803|
|5|92,889521%|92,148125%|91,902274%|+0,848955|+1,010295|

Het eindpunt van de pilot is bemoedigend voor de lagere modi. Hun voordeel is nog geen consequent patroon doorheen meerdere vroege epochs. De eerdere B16-proef liet eveneens zien dat een gunstig vijf-epochresultaat geen voorspelling van het resultaat na 100 epochs is.

Een verse volledige B64-training met dezelfde instellingen is daarom de volgende stap. Die levert de afgesproken vergelijkingen op epoch 100 en het globale E4M3-beste checkpoint. Er is één seed per variant en de oorspronkelijke CUDA-backend heeft geen bitexacte herhaalgarantie; conclusies blijven beperkt tot deze gemeten runs.

## Gemeten duur en projectie

- Som van de vijf epochs: **315,953 seconden**, ongeveer **5 min 16 s**.
- Totale procesduur inclusief voorbereiding en export: **324,348 seconden**, ongeveer **5 min 24 s**.
- Mediaan van epochs 2–5: **63,417 seconden**.
- Projectie voor 100 epochs: eerste epoch + 99 × mediaan = **6.343,896 seconden**, ongeveer **1 uur en 46 minuten**.

De projectie is geen meting van een reeds uitgevoerde 100-epochrun. Ze veronderstelt vergelijkbare laptopomstandigheden. Praktisch is ongeveer 1 uur en 45 minuten tot 2 uur een bruikbare planning; het geobserveerde bereik van vier epochs is geen betrouwbaarheidsinterval.

De eerste vijf epochtijden van B16 en B32 bedragen respectievelijk 289,252 en 303,728 seconden. B64 is hier 4,03% langer dan B32 en 9,23% langer dan B16. Dit zijn afzonderlijke laptopruns onder PyTorch-emulatie, geen gecontroleerde native-hardwaremeting.

## Kosten: schalen, padding en totale representatie

De zes MX-lagen bevatten 129.472 echte gewichten. Groepering gebeurt binnen iedere outputrij met aparte staartpadding en één achtbitsschaal per groep.

| Groepsgrootte | Groepen / schaalbytes | Paddingposities | E4M3 + schalen | E2M3 + schalen | E2M1 + schalen |
|---:|---:|---:|---:|---:|---:|
|16|8.096|64|137.632 bytes|105.248 bytes|72.864 bytes|
|32|4.080|1.088|134.640 bytes|102.000 bytes|69.360 bytes|
|64|2.160|8.768|140.400 bytes|105.840 bytes|71.280 bytes|

De ideale schaaloverhead zonder padding daalt naar 0,125 bit per gewicht bij B64, tegenover 0,25 bij B32. De extra rijpadding weegt hier echter zwaarder dan de besparing op schalen. De grootste laag heeft 800 gewichten per outputkanaal: B64 vult die rij tot 832 aan. Netto kost B64 **4,28% meer E4M3-opslag**, **3,76% meer E2M3-opslag** en **2,77% meer E2M1-opslag** dan B32.

De drie kolommen zijn alternatieve representaties van dezelfde gewichten en worden niet opgeteld. De E4M3-bron volstaat om de lagere codes af te leiden. FP32-uitzonderingen, BN, optimizer, bestandscontainers en tijdelijke GPU-buffers zijn uitgesloten. Alle zes exporttabellen van de echte pilot stemmen overeen met de onafhankelijk berekende geometrie.

## Technische bevestiging en grenzen

Gecontroleerd zijn de 30 train/test-metricregels, vijf geschiedenisregels, 90 gewichtsdiagnostieken, 15 voorspellingstellingen, volledige geregistreerde datadekking, minibatchaantallen en LR 0,001. De rapporten bevestigen A32, FP32-masters/gradients/Adam, drie BN-banken, één gedeelde E4M3-bron, exponenttruncatie met schaaloffset +6, mantissatruncatie en één Adamupdate per minibatch.

Omgeving, backendbeleid, initiële modelhash, broncode, dataset-hashes en transformaties komen overeen met beide referenties. Het gerapporteerde pakketmanifest is exact gelijk aan het geleverde B64-pilotpakket. De bestaande pilotnotebook accepteert de run en beide referenties zonder analysefouten. Alleen de twee bekende PyTorch-deprecatiemeldingen zijn aanwezig.

De feedback bevat geen binaire checkpoints, compacte exports of sample-orderarrays. De exacte CUDA-reconstructie en datadekking zijn door de run geregistreerd; hier zijn hun rapportage, consistentie en exportgeometrie gecontroleerd. Er is geen nieuwe modelevaluatie uitgevoerd. De originele bundel en alle ingelezen resultaten blijven ongewijzigd.
