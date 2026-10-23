# Protocol — dedicated E4M3/A32, honderd epochs

## Doel en uitgangspunt

Een afzonderlijke E4M3-referentie trainen voordat de gezamenlijke
E4M3/E2M3/E2M1-proef wordt toegevoegd. De werkende dedicated pilot-v2 is de
implementatiebasis. Honderd epochs beginnen opnieuw vanaf initialisatie,
seed 20261009. Het pilotcheckpoint wordt niet als beginpunt gebruikt.

De bestaande mantissatruncatie blijft onderdeel van de volgende gezamenlijke
proef. Deze dedicated referentie heeft één actieve E4M3-modus. Er is hier nog
geen exponenttruncatie, distillatie, schaalcompensatie of geleerde schaal.

## Behouden model en numeriek beleid

- Any-Precision CNN-8 (`svhnq`, expand=2), 132.954 parameters inclusief bewaarde BN-banken.
- MX-gewichten uitsluitend in layers.3, .6, .10, .13, .16 en .20.
- Eerste convolutie, laatste lineaire laag en overige modelstaat behouden het eerdere beleid.
- A32 en actieve BN-bank32; overige BN-banken ongebruikt en zonder Adamstaat.
- FP32-updatemasters, gradients en Adam-momenten, geen AMP of torch.compile.
- G(W)=(tanh(W)/max(abs(tanh(W))))*mean(abs(W)).detach(), met dezelfde nul-laagafhandeling.
- Identity-STE door MX; de bestaande G(W)-afgeleide blijft behouden.
- Eén hard-label cross-entropy-forward/backward en één Adam-update per minibatch.
- Bestaand backendbeleid behouden: cudnn.benchmark=True, geen deterministische override.

A32 en FP32 benoemen de tensordtypes en het trainingsbeleid. Het pilotrapport
meldt cuDNN-convolutieprecisie `tf32`; dat backendbeleid wordt hier niet gewijzigd.
Het is dus geen claim dat iedere interne CUDA-vermenigvuldiging IEEE-FP32 is.
Er wordt geen hardwareversnelling door native MX-kernels geclaimd.

## Codec — byte-identiek aan pilot-v2

E4M3: teken, vier exponentbits, drie mantissabits, bias7, maximale eindige
magnitude448. Lokale subnormals en getekende nul ondersteund. Codes127/255 zijn
NaN en worden afgewezen. Nearest rounding met ties-to-even en eindige saturatie.

De exacte elementmagnitude is de gehele teller M (bij E=0) of
(8+M) << (E-1) (bij E>0), maal 2^-9. E8M0-machten van twee worden exact gevormd
uit FP64-exponentbits; de oude torch.ldexp/pow-route is niet het productiepad.
De codec houdt FP64-tussenwaarden en levert FP32-uitvoeringsgewichten.

Blokgrootte32, per outputkanaal over afgevlakte Cin*Kh*Kw, met positieve
nulpadding. Voor een niet-nulblok: schaalexponent=floor(log2(maxabs))-8,
begrensd op[-127,127]. Nulblok: schaal1. FP32-invoersubnormals worden vooraf
naar getekende nul geflusht. De schaal wordt per forward uit G(W) bepaald.
Dit zijn dezelfde regels als in de geslaagde pilot.

## Data en optimalisatie

SVHN volledige train-split73.257 en officiële testset26.032, geen extra en geen
nieuwe validatiesplit. De oorspronkelijke Resize(40), ToTensor en
Normalize(mean=std=0,5) blijven behouden. Batch128, workers0, pin_memory=True,
drop_last=False. Train shuffelt, test niet. Per epoch573 trainingsminibatches,
204 testminibatches; bij een volledige run57.300 optimizerupdates.

Adam, initiële lr0,001, weight_decay0. De scheduler stapt na iedere epoch:

| Epochs | Learning rate |
|---|---:|
| 1–50 | 0,001 |
| 51–75 | 0,0001 |
| 76–90 | 0,00001 |
| 91–100 | 0,000001 |

Dit is hetzelfde honderd-epoch-LR-schema dat al tijdens de pilot geconfigureerd
was. Er verandert uitsluitend hoe lang de volledige referentie doorloopt.

## Checkpoints en analyse

Na iedere epoch: officiële testset, geen updates of BN-herkalibratie. Best is
het eerste strikte maximum van E4M3-testaccuracy. Last is de laatste volledig
afgeronde epoch; bij een voltooide referentie epoch100. Beide worden afzonderlijk
bewaard. De hoofdfiguur vergelijkt epoch100 met best; de laatste beschikbare
epoch blijft zichtbaar bij onderbreking. Best is geselecteerd op de officiële
testset en vormt geen onafhankelijke testschatting na selectie.

Per epoch een onveranderlijk volledig checkpoint met FP32-masters, optimizer,
scheduler, RNG en contract. Nieuwe runs hebben een eigen uitvoermap. Resume
vereist hetzelfde honderd-epochcontract; een pilotcheckpoint wordt geweigerd.
Een nette vroege stop blijft STOPPED_AT_EPOCH_BOUNDARY, geen voltooide referentie.

Best en last krijgen een E4M3-bronexport: elementcodes, E8M0-schalen en de
noodzakelijke overige modelstaat. De zes FP32-masters staan niet in die export.
De runner herberekent G(W)/codes en vergelijkt geladen gewichten op het
trainingsdevice. Er is geen extra volledige testsetforward uit de export;
het oudere mantissa-exportreproductiepunt wordt hier niet opnieuw bewezen.

## Looptijd en bewijslimiet

De echte pilot-v2 van22-10-2026 duurde137,36 seconden inclusief sessieoverhead;
de vijf epochs zelf130,66 seconden. De mediane warme epoch was25,09 seconden.
De projectie voor een nieuwe honderd-epochrun is2513,06 seconden (41m53s).
Het waargenomen min/max-scenario is41m10s–44m25s, geen betrouwbaarheidsinterval.
Langdurige temperatuur- of achtergrondbelasting kan de werkelijke tijd wijzigen.
Na honderd epochs rapporteert de notebook de gemeten som van de epochtijden.

Een pilot van vijf epochs voorspelt geen eindaccuracy. Ook deze volledige
referentie gebruikt één seed en bewijst geen statistische superioriteit. De
gezamenlijke drieprecisieproef krijgt later een eigen vijf-epochcontrole en
looptijdmeting. Er is nog geen echte honderd-epochtraining uitgevoerd met dit pakket.

## Herkomst

- Werkende mx_e4m3_dedicated_pilot5_v2, exact bevestigd tegen de43 pakketbestanden
  waarvan hashes in de aangeleverde run staan.
- Werkelijke run20261022_184112_866638_mx_run, bewaard in pilot5_analysis/raw/.
- Bestaande Any-Precision-auteurscode en compatibiliteitspatch, ongewijzigd.
- Oorspronkelijke dedicated honderd-epochwerkwijze uit de overdracht:
  vaste eindepoch100, dezelfde LR-mijlpalen, best/last, nieuwe initialisatie.
