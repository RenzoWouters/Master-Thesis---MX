# Meetingoverzicht - Truncation-Ready Microscaling

Bestand: MX_onderzoeksoverzicht.ipynb
Klaar om te tonen: MX_onderzoeksoverzicht.html (uitgevoerd, code verborgen)

## Gebruik
1. Pak de volledige ZIP uit in een nieuwe map, bijvoorbeeld:
   C:\Users\renzo\master-thesis-mx\analysis\MX_onderzoeksoverzicht_meeting_v1
2. Open MX_onderzoeksoverzicht.ipynb uit die map in VS Code.
3. Kies de bestaande .venv-train-kernel en Run All.
4. Indien BASE niet naar de pakketmap wijst: zet BASE in de eerste codecel expliciet op die map.

Geen training, modelcalls, datasetdownloads of GPU vereist. De bestaande pandas,
matplotlib, numpy en IPython volstaan. Alle gebruikte CSV/JSON-bronnen zijn meegeleverd.
De invoercode is in de notebook inklapbaar; de HTML bevat alleen uitleg en resultaten.

Hoofdverhaal: secties 1-8. Bronreferenties, pilots en beste checkpoints: bijlagen.
PNG en SVG voor eigen presentaties staan in figuren/.
MEETING_NOTITIES.md bevat een korte spreeklijn en de getalsafbakening.

## Herkomst en beperkingen
- Brondata in bronnen/ zijn byte-identieke kopieen uit de ontvangen feedbackbundels.
- bronmanifest.json bevat oorspronkelijke ZIP-namen, runpaden en SHA-256.
- Grote NPZ-outputs en bestaande notebooks zijn niet nodig voor de samenvatting en zijn niet meegekopieerd.
- Alle 13 unieke SVHN-runs/evaluaties plus de CIFAR-10-selectierapportage zijn vertegenwoordigd.
- Dubbel geuploade pilots zijn niet als extra runs geteld.
- Ruwe bestanden van de vroege microproeven 001-004 ontbreken in dit pakket: geen uit geheugen ingevulde numerieke rij.
- Opslagberekeningen tellen zes gewichtstensors, blokschalen, padding en alle bestaande vijf BN-banken/uitzonderingslagen.
- Training gebruikt nog FP32-masters, gradients en Adam-momenten en A32-activaties.
- De data bewijzen geen energie-, latency- of VRAM-winst; compacte broninhoud is een andere grootheid.
- Geen dedicated E2M1-training of voltooide continual-learning/hardwareproef beschikbaar.

## Controle van dit nieuwe overzicht
Het notebook is volledig uitgevoerd in de analyseomgeving zonder training of nieuwe
onderzoeksresultaten te genereren. Tabellen worden opnieuw berekend uit opgeslagen
correcte aantallen, epochs, tijden en layoutinformatie. Zie afgeleid/controle_resultaat.json.
Het is dus niet afhankelijk van exact jouw oorspronkelijke Windows-resultatenpaden.

Code in bronnen/code is uitsluitend een bronverwijzing. meeting_data.py leest alleen
rapporten en metadata; de notebook importeert de modelcode niet.
