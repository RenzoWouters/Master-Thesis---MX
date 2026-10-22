# Protocol — dedicated E4M3/A32, vijf epochs

## Onderzoeksvraag en status

Werkt de nieuwe E4M3-gewichtsmodus in dezelfde dedicated CNN-8-trainingsopzet,
blijven de trainingsmechanismen geldig, en wat is de gemeten duur van vijf
volledige epochs? De pilot is geen eindvergelijking van geconvergeerde accuracy.
Nog geen honderd-epochtraining of exponenttruncatie in deze proef.

## Behouden trainingsopzet

- CNN-8 (`svhnq`, `expand=2`), 132.954 parameters inclusief bewaarde BN-banken.
- MX alleen in `layers.3`, `layers.6`, `layers.10`, `layers.13`, `layers.16`,
  `layers.20`. Eerste convolutie, laatste lineaire laag en de overige modelstaat
  behouden hun oorspronkelijke beleid.
- Activaties A32; interne auteursselectors blijven `[32]`. Dat is de selector
  voor activaties/BN en gewichtsvoorbewerking, geen uitspraak dat de zes
  uitvoeringsgewichten ongekwantiseerd blijven.
- Alleen BN-bank32 actief. De oorspronkelijke banken1/2/4/8 blijven aanwezig,
  ongebruikt en zonder optimizerstaat. De BN-rekenprecisie is FP32.
- FP32-updatemasters, FP32-gradients en FP32-Adam-momenten. Eén forward,
  hard-label cross-entropy, één backward en één Adam-update per minibatch.
- Bestaande voorbewerking, exact in deze volgorde:
  `G(W)=(tanh(W)/max(abs(tanh(W))))*mean(abs(W)).detach()`.
  De bestaande all-zero-laagafhandeling blijft behouden, zonder extra epsilon.
- MX-forward als numerieke emulator met FP64-codecintermediairen en FP32
  convoluties. Identity-STE ten opzichte van G(W), ook bij verzadiging; daarna
  de bestaande G(W)-afgeleide. Geen native MX-kernel of snelheids-/energieclaim.
- Adam lr0,001, weight_decay0, batch128, drop_last=False, workers0,
  pin_memory=True. Seed20261009; nieuwe initialisatie, geen oud checkpoint.
- Honderd-epoch-LR-schema behouden: factor0,1 na epochs50/75/90. De pilot voert
  alleen epochs1–5 uit en bereikt dus geen LR-overgang.
- Geen AMP/torch.compile. `cudnn.benchmark=True`, oorspronkelijke overige
  precisie-instellingen; geen belofte van bitexacte CUDA-hervatting.

## Data

SVHN volledige `train`-split:73.257 beelden, officiële `test`:26.032 beelden.
Geen `extra` en geen nieuwe validatiesplit. Oorspronkelijke Resize(40), ToTensor,
Normalize(mean=std=0,5). Training shuffelt; testvolgorde ligt vast. Verwacht573
trainingsminibatches en204 testminibatches per epoch; na vijf epochs2.865
optimizerupdates. Elk beeld wordt precies eenmaal per epoch gebruikt, ook de
laatste onvolledige batch. Ruwe bestanden blijven ongewijzigd.

## Nieuwe codec: E4M3

Bitindeling `[s eeee mmm]`, bias7. Normals:
`(-1)^s * 2^(E-7) * (1+M/8)`. Bij E=0:
`(-1)^s * 2^(-6) * M/8`, inclusief getekende nul en lokale subnormals.
Kleinste positieve subnormal2^-9, kleinste normal2^-6, maximum448.
Elementcodes127 en255 (`s 1111 111`) zijn NaN en worden expliciet geweigerd.
Invoer-NaN/Inf wordt ook geweigerd. Finite overflow bij directe elementconversie
verzadigt op +/-448. Afronding: nearest, ties-to-even.

Groepen32 binnen elk outputkanaal langs afgevlakte Cin*Kh*Kw, met +0-padding
van de staart. E8M0-schalen: code0…254 betekent2^(code-127);255 is ongeldig.
Voor een niet-nulblok:

`scale_exp = clip(floor(log2(max(abs(G(W_block)))))-8, -127, 127)`.

De implementatie gebruikt `frexp_power-9`. Een nulblok krijgt schaal1. Zoals in
de bestaande codec worden scalar FP32-invoersubnormals vooraf naar getekende
nul geflusht; dit staat los van ondersteuning van lokale E4M3-subnormals.
Schaalbegrenzing wordt gerapporteerd. Een geladen code/schaal-combinatie die
buiten eindig FP32-bereik decodeert, geeft een fout.

De schaal wordt per forward uit de actuele gewichten bepaald. Er zijn geen
geleerde schaalparameters, geen exponenttruncatie en geen compensatie.

## Checkpoints, export en rapportage

Evaluatie na elke epoch op de volledige officiële testset. Geen updates of
BN-herkalibratie tijdens evaluatie. Best = eerste strikte maximum van de enige
actieve E4M3-testaccuracy. Last = laatste volledig afgeronde epoch (bij succes5).
Selectie op de testset is geen onafhankelijke testschatting. Vijf epochs/één seed
zijn geen bewijs van convergentie of statistische superioriteit.

Per epoch een onveranderlijk FP32-checkpoint met masters, optimizer, scheduler,
RNG, protocol en historie. Best/last worden gecontroleerde kopieën. Hervatten
is alleen op epochgrenzen van ditzelfde contract. Oude E2M3-runs blijven intact.

Best/last krijgen ook een E4M3-bronexport: één byte per element en één E8M0-byte
per blok, plus dezelfde noodzakelijke niet-MX-/BN-modelstaat. De zes FP32-masters
zitten niet in die bronexport. De FP32-trainingsstaat blijft afzonderlijk bewaard.
Exportgewichten worden met de codecforward vergeleken. Geen aanvullende
volledige testsetpass na export; historische numerieke reproductie van de oude
mantissa-export wordt met deze proef niet opnieuw geclaimd of opgelost.

Het notebook leest de bewaarde meetbestanden. Bij deze pilot: figuren voor
epoch5/last en best, plus expliciete afwezigheid van epoch100. Geen afgeleide
accuracyvoorspelling. Tijdprojectie100 = eerste gemeten epoch +99 maal de
mediaan van niet-eerste-procesepochs. Het geobserveerde min/max-bereik is geen
statistisch interval en omvat geen download, data-inladen of definitieve export.

## Bronnen en eigen keuzes

- Overdracht22-10-2026, §§2–5: bestaande dedicated opzet, A32/FP32, CNN-8/SVHN,
  stop/herstel, best/last en notebookwerkwijze.
- Bijgevoegd `mx_weights_dual_reference100_v1.zip`,
  `provenance/single_run_train.py` en `single_single_support.py`: eerdere
  dedicated vijf-epochrunner. De meegeleverde auteurs-ZIP en utils-compatibiliteit
  zijn onveranderd hergebruikt.
- Projectpaper *Microscaling Data Formats for Deep Learning*, PDFp.3,
  tabel1 en algoritme1: formaatfamilie en schaalkeuze; p.4: FP32-masterkopie.
- *OCP Microscaling Formats (MX) Specification v1.0*, pp.11–14:
  E4M3-codering, special cases en E8M0.
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/

Eigen experimentele keuzes: alleen zes gewichtsquantisatoren, behoud van G(W),
identity-STE, train-only, dedicated vijf-epochpilot, afwijzen niet-eindige invoer.
Geen claim dat deze gehele trainingsopzet letterlijk uit de MX-paper komt.

