# Protocol: volledige export-naar-inferentiecontrole

Beide reeds geselecteerde rollen van de echte 100-epoch truncatiebewuste training:
beste op E2M3-testscore (in de aangeleverde run epoch52) en vast eindpunt epoch100.
Die rollen worden tegen de opgeslagen historie gecontroleerd, niet opnieuw gekozen.

Per rol: alleen hoge E2M3-codes + E8M0-schalen + bewaarde hulpstaat. De lage modus
ontstaat uit codes >>2, behoudt de schalen en gebruikt haar eigen bank4. Hoge modus
gebruikt bank32. A32, blok32, oorspronkelijke layout en FP32-emulator blijven gelijk.
De zes getrainde FP32-mastertensors en Adam-staat worden niet ingelezen. Overige
FP32-laagparameters en BN-staat zijn wel noodzakelijke onderdelen van de bron.

Data: volledige SVHN-testset26032, Resize40, batch128/drop_lastFalse, vaste
volgorde, workers0, pin_memoryTrue. Geen train/extra, subsampling of download.
Codecs, architectuur en backend worden tegen de trainingsprovenance gecontroleerd.
Geen automatische upgrades of afwijkende CPU-fallback op een echte evaluatie.

Nieuwe processen: per rol een aparte worker met eigen modelreconstructie. Alleen
het staged hoge .pt-bestand is toegestaan via torch.load(weights_only=True).
De hoge bron en alle gelezen oorspronkelijke bestanden worden voor/na gehasht.
De model-/BN-staat moet exact onveranderd blijven. Geen gradients of updates.

Historische vergelijking: correcte top1/top5-aantallen moeten gelijk zijn voor
MATCH_COUNTS_CE_CLOSE. CE-diagnose gebruikt atol1e-6/rtol1e-5 voor verschil in
aggregatie. Elke afwijking wordt kwantitatief getoond en REVIEW_DIFFERENCE genoemd;
geen automatische reparatie of extra kalibratie. Modelstate-, code-, en schaal-
controles hebben geen verslapte tolerantie. Een succesvol evaluatieproces kan dus
PASS zijn maar een expliciete historische reproductiewaarschuwing geven.

Nieuwe logits en voorspellingen worden opgeslagen. Oude individuele logits zijn
niet beschikbaar: geen claim dat de inferentie exact dezelfde oude logits heeft.
Beide modelrollen zijn uit dezelfde seed, geen onafhankelijke herhalingen. De
beste score is niet onafhankelijk van testgestuurde selectie. De sessieduur is
geen benchmark van een native low-bit accelerator.

Bronnen: de aangeleverde trainingscode en feedback_bundle(20261021-144630).zip.
API-controle (niet de numerieke methode): PyTorch torch.load en eval/no_grad:
https://docs.pytorch.org/docs/stable/generated/torch.load.html
https://docs.pytorch.org/docs/stable/notes/autograd.html
De actuele webdocumentatie kan nieuwer zijn; lokale tests gebruiken werkelijk
PyTorch2.10.0+cpu; de gebruikersomgeving is2.10.0+cu128 en blijft ongewijzigd.
