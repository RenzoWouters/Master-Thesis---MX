# Uitgevoerde lokale softwarecontroles

Datum:22 oktober2026. Dit zijn technische controles op CPU met synthetische
invoer. **Geen SVHN-training, geen nieuwe onderzoeksaccuracy en geen gemeten
looptijd op de laptop-GPU.** Er zijn geen checkpoints of datasets uit eerdere
experimenten gewijzigd.

## Testomgeving

Tijdelijke, afzonderlijke Linux/Python3.12.14-omgeving met PyTorch2.10.0+cpu,
torchvision0.25.0+cpu en twee CPU-threads. De productie-CLI blijft de bestaande
Python3.13/PyTorch2.10.0/torchvision0.25.0-CUDA-omgeving vereisen. Die wordt
op de laptop nog gecontroleerd door `run_train.py check`.

## Codec — PASS

`test_e4m3.run_tests('cpu')`:31 controles en21 verwachte afwijzingen.

- Alle254 eindige E4M3-bitpatronen vergeleken met een onafhankelijke rationele
  scalar-oracle; exacte decode- en encode-roundtrip.
- Beide NaN-codes expliciet geweigerd.
- Alle252 getekende midpoint-ties en504 aangrenzende FP64-waarden getest;
  ties-to-even, saturatie en lokale subnormals gecontroleerd.
- Groepering per outputkanaal, padding, niet-contigue invoer, E8M0-grenzen,
  signed zero en FP32-invoersubnormals gecontroleerd.
- Payload/serialisatie, onafhankelijke opslag, bronimmutabiliteit en
  identity-STE inclusief verzadiging gecontroleerd.
- De definitieve codeccontrole is ook met warnings-as-errors uitgevoerd.

Werkelijk rapport: `codec_report.json`.

## Echte CNN-8-integratie — PASS

`smoke_single.check`:exact twee synthetische minibatchupdates met het
meegeleverde CNN-8, expand2, invoer2x3x40x40. Twintig actieve FP32-gradienttensors,
FP32-Adam-momenten en updates in alle zes MX-doellagen. Alleen BN32 telt mee;
ongebruikte BN-banken blijven ongewijzigd en hebben geen Adamstaat.

Identity-STE bewaart de volledige G(W)-afgeleide exact. Actieve quantizer,
codecdecode en geëxporteerde gewichten komen exact overeen. Na serialisatie en
reconstructie is het maximale logitverschil in deze CPU-controle0,0.
CUDA gebruikt in de laptopcheck de vooraf vastgelegde logittolerantie
atol=rtol=1e-5, met nog steeds exacte vergelijking van gewichten.

Werkelijk rapport: `smoke_report.json`. Dit is geen volledige testsetevaluatie.

## Runner, stoppen en hervatten — PASS

`local_validation/test_training.py` gebruikt vier synthetische trainbeelden en
vier testbeelden, twee per batch. Vijf mini-epochs leveren tien updates en
daadwerkelijke checkpoints/CSV/JSON/exportbestanden op, uitsluitend in `/tmp`.

Hervatten vanaf epoch2 levert exact dezelfde CPU-modelstaat, optimizer,
scheduler, RNG, metrics, samplevolgorde en gewichtsupdates als doorlopen tot5.
De test verplaatst LR-mijlpalen uitsluitend in het synthetische testcontract
naar2/3/4 om herstel over LR-overgangen te controleren; het productieprotocol
blijft50/75/90. De auteursforward is ongewijzigd; alleen in de test worden
Tensor.cuda-calls naar CPU geleid.

Ook gecontroleerd: verkeerde formaatcontracten, corrupte checkpointbytes en
overschrijven worden geweigerd; een stopverzoek bewaart een complete epoch en
claimt geen voltooide pilot. De eerste hoogste accuracy wint bij gelijke scores.
CLI-aanvragen voor100 epochs, CPU-training, een control-modus of hervatten
zonder checkpoint worden geweigerd vóór een productierun wordt aangemaakt.

Werkelijk rapport: `integration_report.json`.

## Notebook — PASS

Zes gerichte offline controles voor leeg sjabloon, synthetische resultaten,
checkpointselectie, ontbrekende gegevens, onjuiste formaat-/runstatus en nieuwe
figuurmappen. De volledige notebook is ook op de daadwerkelijk door de
synthetische runner geschreven complete en onderbroken run uitgevoerd.
Beide hadden nul analyse-inconsistenties. De onderbroken run blijft duidelijk
als onderbroken gemarkeerd; alle voorbeelden zijn herkenbaar synthetisch.
De bestaande invoerbestanden bleven volgens hashcontrole onveranderd.

Checkpoint-, leer-, tijd- en geheugengrafieken zijn visueel gecontroleerd.
Het leverbare notebook bevat geen opgeslagen synthetische uitkomsten. Een
ontbrekende epoch100 wordt als ontbrekend getoond, niet als fictieve balk.

## Praktische grens

Deze controles onderbouwen de softwarewerking op CPU. Ze voorspellen geen
SVHN-accuracy, GPU-runtime of bitexacte CUDA-reproduceerbaarheid. De opdracht
`check` op de laptop is de volgende hardwarecontrole; `run` voert daarna pas
de vijf echte epochs uit. De tijdschatting ontstaat uit die gemeten run.

