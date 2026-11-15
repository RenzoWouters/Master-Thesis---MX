# Protocol — directe MX-gewichten met vijf formaten, honderd epochs

## Doel en vooraf vastgelegde keuze

Deze proef volgt op de geslaagde vijf-epochpilot met E4M3, E5M2, E2M3,
E3M2 en E2M1. Eén nieuw model ondersteunt de vijf gewichtsformaten. De volledige
referentietraining start vanaf nul met seed 20261009 en stopt na honderd
volledige epochs. Het pilotcheckpoint wordt niet als initialisatie gebruikt.
Hervatten is alleen toegestaan binnen deze honderd-epochproef, vanaf een volledig
opgeslagen epoch 1 tot en met 99 met hetzelfde protocol.

Het primaire accuracy-eindpunt is epoch 100 voor alle vijf formaten op hetzelfde
checkpoint. Er wordt geen beste epoch op de officiële testset geselecteerd.
De vergelijking met de eerdere directe drie-formaattraining is beschrijvend en
betreft de drie gedeelde formaten op exact dezelfde epoch. Ze is geen geïsoleerde
causale meting van het effect van twee extra formaten, omdat ook de gezamenlijke
distillatiestructuur en het aantal actieve trainingmodi veranderen.

## Bronnen en hergebruik

Directe implementatiebasis: het ongewijzigde numerieke en gezamenlijke
trainingsgedeelte van `mx_weights_five_direct_pilot5_v1.zip` (SHA-256
`0f7b0834fa67c6ef73764516582500e9fa9d08c1f8f35915adf0c21b7740ba21`).
De gebruikersfeedback van de vijf-epochpilot is `feedback_bundle(20261115-133745).zip`
(SHA-256 `77162698b82752cc1be87c853c18f8b4e5e56691e836b539832676aa6fcbe61e`).

De eerdere projectbron is `mx_weights_triple_direct_reference100_v1(1).zip`, met SHA-256
`d3a73669ea40d46f2e8cd78b99c7c4972a4cf67f735a586096fff56675356729`.
De vorige `PROTOCOL.md`, adapter, gezamenlijke trainingsondersteuning, runner
en pakketmanifest zijn onder `provenance/` opgenomen. De ingesloten
Any-Precision-auteurscode blijft de bron voor het model, de datasettransforms
en optimizer. De bestaande drie MX-codecs en gewichtsvoorbewerking blijven
behouden. De bron- en pakketmanifesten leggen het gebruikte materiaal vast.

De vijf-formaattrainingslus is een expliciete uitbreiding van de eerdere
recursieve distillatie, geen ongewijzigde overname van de drie-formaatlus.

Externe bronnen:

- OCP, *Microscaling Formats (MX) Specification*, versie 1.0, september 2023,
  tabel 1 op p. 10 en elementdefinities op pp. 11–13:
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Hinton, Vinyals en Dean, *Distilling the Knowledge in a Neural Network*,
  arXiv:1503.02531, 2015, p. 2: middelen van voorspellingsverdelingen als
  soft target. De specifieke keuze van onze twee 6-bitteachers is een
  eigen ontwerpkeuze:
  https://arxiv.org/abs/1503.02531

## Exacte gewichtsconversie

Iedere MX-convolutie behoudt:

`G(W) = tanh(W) / max(abs(tanh(W))) * mean(abs(W)).detach()`.

De bestaande nulguard blijft behouden. W is de gedeelde FP32-master. Voor
elk formaat f gebeurt onafhankelijk `Q_f(G(W))`. Er worden geen codes of
al gekwantiseerde gewichten van een andere modus als bron gebruikt.

| Formaat | Indeling | Bias | Grootste eindige magnitude | Kleinste positieve lokale subnormal | Schaalexponent niet-nulblok |
|---|---|---:|---:|---:|---|
| E4M3 | `s eeee mmm` | 7 | 448 | 2^-9 | `floor(log2(amax)) - 8` |
| E5M2 | `s eeeee mm` | 15 | 57344 | 2^-16 | `floor(log2(amax)) - 15` |
| E2M3 | `s ee mmm` | 1 | 7,5 | 2^-3 | `floor(log2(amax)) - 2` |
| E3M2 | `s eee mm` | 3 | 28 | 2^-4 | `floor(log2(amax)) - 4` |
| E2M1 | `s ee m` | 1 | 6 | 2^-1 | `floor(log2(amax)) - 2` |

De elementdefinities komen uit de MX-specificatie. De concrete
schaalberekening is het behouden projectbeleid, uitgebreid met de maximale
lokale exponent van E5M2 en E3M2. De MX-specificatie schrijft deze specifieke
berekeningsregel niet verplicht voor; gedetailleerde algoritmen voor de
blokschaal vallen volgens haar scope op p. 7 buiten de specificatie.

Blokken bevatten 32 waarden binnen elk outputkanaal, over afgevlakte
`Cin*Kh*Kw`, met positieve nulpadding aan de staart. De gedeelde schaal is
E8M0; schaalexponenten worden begrensd tot [-127,127], een nulblok krijgt
schaal 1. De schaal wordt iedere forward opnieuw uit de actuele `G(W)`
berekend. E2M3 en E2M1 krijgen bij dezelfde bron dezelfde schalen. Er is geen
schaaltraining en geen opgelegde koppeling tussen bron- en doelformaten.

Nearest rounding met ties-to-even, eindige saturatie, lokale subnormals en
getekende nul blijven behouden. Niet-eindige FP32-invoer wordt geweigerd.
FP32-invoersubnormals worden naar getekende nul geflusht. E5M2 heeft in de
standaard Inf/NaN-codes; deze eindige-gewichtenemulator produceert ze niet en
weigert ze bij decoderen. E4M3-NaN-codes worden eveneens geweigerd. Alle E3M2-
elementcodes zijn eindig. De quantizers gebruiken identity-STE; de bestaande
afgeleide van `G(W)` blijft onderdeel van de backward pass.

## Model, formaatselectie en BatchNorm

Any-Precision CNN-8 (`svhnq`, expand=2), met het bestaande beleid voor eerste
convolutie en laatste lineaire laag. MX vervangt uitsluitend gewichten in
`layers.3`, `.6`, `.10`, `.13`, `.16` en `.20`. Activaties blijven A32; alle
mastergewichten, gradiënten en Adam-momenten blijven FP32.

Het volledige formaat is de modus-ID. Een bitbreedte alleen is onvoldoende:
E4M3/E5M2 hebben beide acht elementbits en E2M3/E3M2 beide zes. Iedere modus
krijgt een eigen BatchNorm-bank met formaatnaam. De vijf BN-banken die in de
oude modelstructuur al werden bewaard, worden nu alle vijf actief gebruikt.
De oorspronkelijke tensors worden onder formaatsleutels hernoemd: E4M3 uit
bank `8`, E5M2 uit `1`, E2M3 uit `32`, E3M2 uit `2` en E2M1 uit `4`.
Het totale parameteraantal blijft daardoor 132.954.
Iedere actieve bank krijgt per trainingsminibatch één update. Evaluatie
verandert geen BatchNorm-statistieken.

Het oorspronkelijke backendbeleid wordt behouden: `cudnn.benchmark=True`,
geen deterministische override, geen AMP/torch.compile en geen aangepaste
TF32-instelling. Een vaste seed garandeert daardoor geen bitidentieke
CUDA-herhaling. De uitvoering blijft een numerieke emulator met FP32-
convoluties en precisieconversie; deze proef meet geen native MX-versnelling,
energie of hardware-oppervlakte.

## Gezamenlijke training en distillatie

Per minibatch worden gradiënten eenmaal nulgezet. Dezelfde beelden worden in
deze vaste volgorde verwerkt; er gebeurt geen optimizerupdate tussen de
passes:

1. E4M3: forward, harde CE met de labels, backward.
2. E5M2: forward, harde CE met de labels, backward.
3. E2M3: forward, zachte CE met `softmax(E4M3_logits.detach())`, backward.
4. E3M2: forward, zachte CE met `softmax(E5M2_logits.detach())`, backward.
5. E2M1: forward, zachte CE met het rekenkundig gemiddelde van de twee
   losgekoppelde 6-bitkansverdelingen, backward.
6. Eén Adam-update op de som van alle vijf gradiëntbijdragen.

Temperatuur T=1. Alle vijf moduslosses hebben coëfficiënt 1 en worden
opgeteld, zonder deling door vijf. Iedere individuele CE is gemiddeld over
de beelden in de minibatch. E2M1 krijgt één lossbijdrage; het gemiddelde van
beide teachers betekent intern 1/2 plus 1/2. Er zijn geen aanvullende harde
studentlosses en geen extra FP32-teacherforward.

De teacheroutputs zijn losgekoppeld van hun berekeningsgraph. Teachers en
studenten blijven dezelfde mastergewichten delen. Hun voorspellingen kunnen
allemaal veranderen door de gezamenlijke update. Dropout gebruikt
achtereenvolgende RNG-trekkingen per modus, zoals in het bestaande beleid.
Gelijke lossgewichten garanderen geen gelijke gradiëntnormen.

De lussen en logs onderscheiden de optimalisatieloss van harde cross-entropy
met de labels. De harde CE wordt per modus als aanvullende diagnose gemeten;
alleen de in de bovenstaande lijst genoemde objective krijgt een backward.
Bij evaluatie worden alle vijf modi op de volledige testset beoordeeld en
is de gerapporteerde loss harde CE. Distillatietakken zijn geen gewichts-
conversieroutes: in deze referentie worden alle gewichten direct gekwantiseerd.

## Data, optimizer en eindpunt

SVHN train-split: 73.257 beelden; officiële testset: 26.032 beelden. Geen
extra-split en geen nieuwe validatiesplit. Behouden transforms: Resize(40),
ToTensor en Normalize(mean=std=0,5). Batch 128, workers 0, pin_memory=True,
drop_last=False; train shuffelt, test niet. Bestaande databestanden worden
alleen gecontroleerd en gelezen. Downloads vereisen `--download` en mogen
bestaande ruwe data niet vervangen.

Seed 20261009, nieuwe initialisatie. Adam, start-lr 0,001, weight_decay 0.
De scheduler wordt na de train- en testfase van iedere epoch bijgewerkt.
Het daadwerkelijke learningrateschema is:

| Epochs | Gebruikte learningrate |
|---|---:|
| 1–50 | 0,001 |
| 51–75 | 0,0001 |
| 76–90 | 0,00001 |
| 91–100 | 0,000001 |

De runtime controleert dat de gebruikte learningrate overeenkomt met dit schema.

Elke epoch heeft 573 Adam-updates, 2.865 trainingspasses en 204 testbatches
met elk vijf formaatpasses. Een voltooide referentie bevat 57.300 Adam-updates,
286.500 trainingspasses en 102.000 testpasses. Per epoch worden datadekking,
modussequentie, gradiënten, BN-tellers en optimizerstaat gecontroleerd.

Alleen een voltooide honderd-epochtraining krijgt PASS als trainingsresultaat.
Een nette stop op een eerdere epochgrens krijgt STOPPED_AT_EPOCH_BOUNDARY; de globale terminal-
uitkomst is dan FAIL omdat de afgesproken proef niet volledig is afgewerkt.
Een numerieke of andere fout krijgt FAIL met foutinformatie in de feedback.
De check heeft uitsluitend een technische PASS/FAIL en meet geen SVHN-accuracy.

## Checkpoints, rapportage en reproduceerbaarheid

Iedere opdracht schrijft naar een nieuwe resultaatmap. Checkpoints leggen de
trainingsstaat vast, inclusief mastergewichten, optimizer, scheduler en RNG.
Hervatten vereist een geldig checkpoint van dezelfde honderd-epochproef en
hetzelfde protocol; het loopt tot epoch 100. Pilotcheckpoints en checkpoints
uit de oude drie-formaatproef worden geweigerd. De oorspronkelijke en nieuwe runmappen
blijven behouden. Het package-manifest en bronmanifest maken veranderingen
zichtbaar; pas bestanden of afhankelijkheden niet aan midden in een run.

`metrics.csv` bevat per epoch/fase/formaat de exacte correcte aantallen,
voorbeeldtelling, top-1/top-5-percentages, optimalisatieloss, harde CE,
teacherbeschrijving en learningrate. De notebook rekent accuracy opnieuw uit
de tellingen en weigert ongeldige of dubbele rijen. Elk formaat krijgt een
eigen curvepaneel met procentlabels. De primaire tabel toont alleen een
volledige epoch 100 van alle vijf modi wanneer het rapport tevens
`completed_epochs == 100` bevestigt. Beschikbare metingen vóór een mislukte
checkpointopslag mogen wel in de curves staan; ze vormen geen bevestigd
primair checkpointresultaat. Er wordt geen beste checkpoint op de
testset gekozen.

De historische directe drie-formaatreferentie staat in
`reference_data/triple_direct_metrics.csv` en `triple_direct_reference.json`.
De metadata bevat de bronidentiteit, hash van de tabel en het historische
protocol. De notebook controleert het directe karakter, A32, voorbeeldtellingen,
accuracyberekening en gedeelde model-/dataset-/optimizerinstellingen. De primaire
vergelijking is epoch 100 tegen epoch 100, met een eigen paneel voor E4M3, E2M3
en E2M1. De methoden staan op een categorische as met dezelfde ronde stippen en
procentlabels. E5M2 en E3M2 hebben geen oude score en worden niet gefingeerd.
Curvenvergelijkingen gebruiken uitsluitend epochs waarvoor beide runs een
volledige testmeting bevatten. De oude E4M3-bestselectie wordt niet overgenomen.

De historische opzet heeft drie passes, één harde-labelteacher en de keten
E4M3→E2M3→E2M1. De nieuwe opzet heeft vijf passes, twee harde-labelteachers en
een gemiddeld teacherdoel voor E2M1. Daardoor verschillen ook actieve
BatchNorm-banken en RNG/dropoutverbruik. De vergelijking beschrijft deze
volledige protocoluitbreiding; ze bewijst niet dat een verschil alleen door
extra gewichtformaten ontstaat. Beide zijn afzonderlijke runs met één seed.

Bij voltooiing worden de honderd train- en testfasetijden opgeteld als gemeten
duur. Tijdens een onvoltooide run kan daarnaast een expliciete raming worden
getoond. Datavoorbereiding, controles, checkpointopslag en exports vallen buiten
die fasetiming; de sessieduur bevat ook bijkomende werkzaamheden. Waarschuwingen
en fouten worden in de feedback opgenomen. De notebooktemplate bevat geen
nieuwe meetresultaten; alleen de afzonderlijk geïdentificeerde historische
referentiemetingen worden als vergelijkingsdata meegeleverd.

## Vast aandachtspunt voor de latere truncatiefase

Vóór een truncatievariant worden de volgende keuzes opnieuw besproken:

1. Beide 8-bitformaten kunnen niet door alleen bits verwijderen uit elkaar
   worden afgeleid. De opslagrepresentatie en omzetting worden expliciet.
2. Iedere nieuwe exponentovergang krijgt een numeriek gecontroleerde
   schaalregel. De bestaande ×64-correctie wordt niet automatisch overgenomen.
3. E2M1 via twee verschillende routes kan verschillende codes of schalen
   krijgen. Daarom wordt één canonieke afleiding gekozen, of worden de routes
   expliciet als verschillende uitvoeringsvarianten onderzocht.

Voor de directe/truncatievergelijking blijven daarna dezelfde vijf formaten,
distillatiestructuur, lossgewichten, initialisatieprocedure en overige
trainingsinstellingen gelden. De afgeronde mantissa-MSB-richting vormt geen
onderdeel van deze nieuwe referentie.
