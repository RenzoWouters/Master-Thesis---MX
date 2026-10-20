# Protocol: vaste-checkpointdiagnose van mantissatruncatie

## Onderzoeksvraag

Wat verandert in het reeds gezamenlijk getrainde CNN-8-model wanneer de lage
gewichtsmodus niet meer rechtstreeks wordt afgerond, maar door codebittruncatie
uit de behouden hoge MX-bron wordt gevormd?

De referentie is de volledige train-only E2M3/E2M1/A32-training uit
mx_weights_dual_reference100_v1. De huidige feedback vermeldt best epoch75 en last
epoch100; de runner controleert deze rollen opnieuw uit de volledige geschiedenis,
maar selecteert GEEN nieuw checkpoint op de nieuwe scores. De bestaande
best-rol is eerder op de testset geselecteerd en is geen onafhankelijke schatting.

## Vooraf vastgelegde vergelijking

H = hoge E2M3-export met haar hoge BN-bank, slechts context.
A = bestaande directe E2M1-export uit dezelfde trainingscheckpoint, lage BN-bank.
B = nearest-even E2M1 van genormaliseerde E2M3-elementen, met de schaalbytes van H.
C = logische right-shift van ieder uitgepakt E2M3-codewoord met twee bits, met de
schaalbytes van H. Beide afleidingen gebruiken uitsluitend H, niet A of een master.

C-A is het primaire gemeten verschil. A-B maakt de invloed van eerst de bron
kwantiseren zichtbaar; B-C vergelijkt afronden en trunceren vanuit dezelfde bron.
De accuracyverschillen voldoen rekenkundig aan C-A=(C-B)+(B-A). Dit maakt het geen
ontbinding in onafhankelijke causale effecten: het netwerk is niet lineair en
verschillende beelden kunnen tegengestelde veranderingen ondergaan.

Beide oorspronkelijke schalen moeten byte-identiek zijn. Dit past bij de reeds
vastgelegde gelijke power-3-schaalregel en emax=2. Een verschil leidt tot FAIL,
in plaats van stilzwijgend een andere schaal te kiezen. Er wordt geen lokale
exponent, groepsgrootte, activatiebeleid of laagselectie veranderd.

## Blijvende instellingen

- Model: svhnq/CNN-8, expand2. Dezelfde zes MX-convoluties.
- Eerste/laatste laag, biasbeleid en alle auxiliaire staat zoals opgeslagen.
- A32 in alle paden. A/B/C: dezelfde lage BN-bankkey4, H: hoge bankkey32.
- model.eval(), requires_grad=False, torch.no_grad(); dropout uit.
- Geen train(), backward, optimizer, BN-herkalibratie, fine-tuning of nieuwe labelsplits.
- Groepsindeling: per Cout, afgevlakte Cin*Kh*Kw, blok32 en dezelfde opvulnullen.
- C: [s ee mmm] -> [s ee m]. Logische shift op ongetekende codes; geen delen van een numerieke waarde.
- Hoge bron en twee laagste bits blijven ongewijzigd in het originele bestand.
- B: low.encode_elements(high.decode_elements(codes)), NIET low.encode_tensor(high.decode()).
  Die laatste vorm zou onbedoeld opnieuw schalen kunnen kiezen.
- De bestaande G(W)-voorbewerking zit al in de export; ze wordt nooit opnieuw op opgeslagen MX-waarden toegepast.
- Hele testset26032, batch128, drop_lastFalse, originele Resize40/normalisatie.
- Geen train- of extra-bestanden gelezen; geen download of optimizercheckpoint nodig.
- Zelfde bestaande software-/precisiebackend. Afwijking in relevante omgeving of configuratie: stop.

## Metingen en controle

Acht accuracies en hard-label CE-scores. Alle vier paden krijgen binnen een batch
dezelfde beelden. Top1 gebruikt dezelfde topk(5)-conventie als de auteurscode,
ook om ongedocumenteerde verschillen bij ties te vermijden. Per-beeld-CE wordt
opgeslagen; de FP64-som kan minimaal van de historische FP32-batchmean afwijken.

Controleer voor en na iedere evaluatie de volledige model/BN-staathash, afwezigheid
van gradients en de lage bankkeuze. Bewaar labels, indices, logits, voorspellingen
en CE per beeld met checksum. Rapporteer fout->correct, correct->fout en
fout->andere-fout apart. Geen nieuwe significance- of equivalentieclaim.

Per laag: codeverschillen, numerieke MAE/RMSE/maxverschil tussen A/B/C, lokale
subnormals, nulvorming en niet-nul weggegooide mantissasuffixen. Padding telt niet
mee als echt gewicht. Er is geen foutmaat tegenover de afwezige FP32-masters.

De originele hoge/lage exporthashes worden tegen hun sidecars en het runrapport
getoetst; bronprovenance en alle auxiliaire tensors moeten gelijk zijn. Na de proef
moeten alle gelezen oorspronkelijke bestanden en het testbestand ongewijzigd zijn.
De FP32-checkpointbestanden blijven onaangeraakt en worden niet geopend.

B en C krijgen eigen eerlijk gelabelde schemas, geen vermomming als directe export.
Opslaan/herladen moet de codes en tensors behouden. Afgeleide modellen worden
uitsluitend uit die herladen bestanden opgebouwd en ge-evalueerd.

Historische A/H-testscores kunnen minimaal afwijken door CPU-export tegenover
GPU-forwardberekening of backend-herhaalbaarheid. Dat wordt geconstateerd en
gewaarschuwd, niet zonder bewijs aan een oorzaak toegeschreven. De nieuw gemeten
A-C-vergelijking blijft leidend. Een lage C-accuracy is geen reden tot afwijzen
van een technisch correct uitgevoerde proef.

## Interpretatie en grenzen

Dit is gewone, niet-truncatiebewuste training met gewijzigde inferentie. Een verlies
bij C bewijst niet dat truncatiebewuste training niet kan werken. Een klein verlies
bewijst evenmin robuustheid voor andere workloads, seeds of continual learning.
BN-herkalibratie is bewust uitgesloten; de waargenomen impact omvat dus ook de
interactie van getrunceerde gewichten met de vastgehouden lage BN-statistieken.

Dit is numerieke emulatie met FP32-convoluties. Omzetten/repacken van bestanden
hier is geen hardware-runtimebenchmark. Opgeslagen zesbitbron, lagere auditbestanden,
FP32-rekenbuffers en auxiliaire modelstaat worden niet als dezelfde geheugensoort
aangeduid. Geen bewezen energie- of doorvoerwinst.

## Bronnen

- Gebruikerspakket mx_weights_dual_reference100_v1: codecs, twee BN-banken,
  exportstructuur; exacte hergebruikhashes in REUSE_MANIFEST.json.
- Projectpaper Microscaling Data Formats for Deep Learning, p.3 tabel1 en
  Algorithm1: E2M3/E2M1, E8M0 en groepsgrootte32. Ons schaalbeleid blijft identiek.
- TruncQuant (projectpaper), p.2 vergelijking3: bittruncatie van integercodewoorden.
  Onze toepassing op de mantissastructuur van MX is een eigen implementatie;
  geen claim dat het TruncQuant-paper deze MX-proef al beschrijft.
