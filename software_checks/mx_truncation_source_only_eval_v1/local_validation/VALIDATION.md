# Uitgevoerde validatie - bron-only evaluatie

De offline tests draaiden op Linux, Python 3.13.5, torch2.10.0+cpu,
torchvision0.25.0+cpu. Geen CUDA, geen echte SVHN-data, geen getrainde
gebruikersgewichten en geen onderzoeksaccuracies uit deze container.

Geslaagd:
- Alle64 E2M3-broncodes: correcte lage bitshift, teken/exponent/schaal en signed zero.
- Volledig synthetisch CNN-8-model met opzettelijk verschillende BN-banken.
- Hoge bron opslaan/herladen, beide modi reconstrueren zonder masters/lage export.
- Exacte CPU-logits tegenover het levende synthetische quantisatormodel.
- Nieuwe subprocess: alleen hoge .pt toegestaan; beide logitvergelijkingen exact.
- Onveranderde volledige model-/BN-staat gedurende inferentie.
- Guards blokkeren gewichtencoders, G(W), training(True), optimizer, andere modelbestanden.
- Afwijzing van onbekende exportschema-velden, verborgen masters, verkeerde BN-mapping,
  NaN-schalen, ongemerkte hulpstaat-dtypeconversie en corrupte bestanden.
- Historische scoreafwijking krijgt REVIEW_DIFFERENCE; niet verborgen of weggekalibreerd.
- Tweerol-aansturing, subprocessuitvoering, feedback-ZIP en notebookuitvoering.
- Metadata-inspectie tegen de werkelijk aangeleverde rapportstructuur; testfixture
  zonder checkpointbestanden of lage exports. Verkeerd protocol/schema/hash afgewezen.

De end-to-end aansturing gebruikt kleine synthetische jobs via testmocks; dit is
GEEN daadwerkelijke evaluate-run op de 26.032 SVHN-beelden. De template bevat geen
verzonnen wetenschappelijke resultaten. De preview is uitsluitend synthetisch.

Notebook visueel gecontroleerd; het leest JSON/CSV/NPZ en traint niets.

Voor de laptop-check gelden bij synthetische fresh-process logitvergelijkingen
rtol1e-4/atol1e-5, gelijk aan de eerder gebruikte vergelijkingstolerantie. De CPU
resultaten waren exact, maximale afwijking0. Codetensors/schalen/modelstaat worden
exact vergeleken. Dit bewijst geen bitexacte CUDA-herhaalbaarheid.

De productie-CLI check is hier echt aangeroepen en stopte bij pip check: de
bestaande MoviePy2.2.1 vereist Pillow<12, terwijl Pillow12.3.0 in de container
staat. Er zijn geen pakketten gewijzigd en geen controles omzeild. De aanwezige
hardware is bovendien CPU-only. Zie container_run_report.json en container_pip_check.txt.
De check op de laptop moet slagen voordat de echte evaluate wordt gestart.

Herhalen (nieuwe doelmap vereist):
python local_validation/test_offline.py <NIEUWE_SYNTHETISCHE_MAP>
