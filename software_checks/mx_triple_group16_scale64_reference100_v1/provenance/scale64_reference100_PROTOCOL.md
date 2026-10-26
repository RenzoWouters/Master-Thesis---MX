# Protocol — route B met vaste schaal ×64, honderd epochs

## Doel en vergelijking

Deze proef onderzoekt het volledige leergedrag van de route die in de
vijf-epochpilot succesvol leerde. De numerieke route, optimizer,
initialisatie en data blijven die van de pilot. De training loopt nu door
tot epoch 100, inclusief de oorspronkelijke LR-dalingen na 50/75/90.

De proef begint vanaf de oorspronkelijke verse initialisatie met seed
20261009. Een eerder getrainde pilot of andere route dient niet als
warmstart. Dit volgt de eerdere werkwijze met een afzonderlijke pilot en
volledige referentierun. De oude bestanden en resultaten blijven behouden.

## Exacte code- en schaalbewerking

Per MX-laag en minibatch wordt de actuele FP32-master W met de oorspronkelijke
G(W) voorbewerkt. De bestaande E4M3-codec bepaalt één gedeelde bronencoding.
Voor E4M3-code c met indeling `[s e3 e2 e1 e0 m2 m1 m0]` geldt:

```python
sign = c >> 7
exponent = (c >> 3) & 15
mantissa = c & 7
b6 = (sign << 5) | ((exponent >> 2) << 3) | mantissa
b4 = b6 >> 2
```

De E4M3-bias blijft 7; de bestaande E2-bias blijft 1. De E2M3-bewerking
behoudt de hoogste twee exponentbits en alle drie mantissabits. E2M1 wordt
vervolgens door dezelfde bestaande mantissashift uit E2M3 afgeleid.

Bij E8M0-broncode k is `S = 2**(k-127)`. E4M3 behoudt k; beide lagere
modi gebruiken k+6, dus `64*S`. De optelling gebeurt na promotie naar int32.
Codes boven 254 worden geweigerd, zonder clipping of wraparound. Dezelfde
regel geldt voor nulblokken en padding. Ongeldige broncodes en niet-eindige
gedecodeerde FP32-gewichten worden geweigerd. E2M1 behoudt de E2M3-schaal.

De verhouding 64 is vooraf vastgelegd. De actuele E4M3-bronschaal kan door
masterupdates veranderen. Er zijn geen geleerde schaalparameters, aanvullende
elementafronding of alternatieve exponentmapping in deze route.

## Backward en broncache

De bestaande identity STE wordt toegepast op de volledige forwardroute
tegenover G(W). De upstreamgradient wordt onveranderd doorgegeven;
backward bevat geen extra factor 64. De oorspronkelijke afgeleide van
G(W) naar W blijft behouden. Integercodes en schaalcodes krijgen geen gradient.

Elke precisiepass bouwt zijn eigen G(W)-autogradgraph. De gedeelde cache
bevat detached codes en schalen. Er is één E4M3-encoding per laag per
minibatch en er wordt geen lagere encoder aangeroepen. Na het triplet is
de broncache leeg, vóór de Adam-update. De corebestanden zijn exact dezelfde
als in de geslaagde pilot; hun hashes staan in `REUSE_MANIFEST.json`.

## Trainingsinstellingen

| Onderdeel | Instelling |
|---|---|
| Netwerk | oorspronkelijke svhnq/CNN-8, expand 2, 132.954 parameters |
| MX-lagen | dezelfde zes convoluties |
| Gewichten | E4M3, E2M3, E2M1; één E4M3-bron |
| Blokindeling | 32 waarden, binnen outputkanaal; oorspronkelijke padding |
| Activaties | A32 in alle drie modi |
| Master / gradients / Adam-momenten | FP32 |
| Data | SVHN train 73.257 / test 26.032; geen extra of validatiesplit |
| Initialisatie | verse modelconstructie, seed 20261009 |
| Minibatch | 128; trainshuffle; volledige dekking; geen drop_last |
| Optimizer | Adam, lr 0,001, weight decay 0 |
| Scheduler | oorspronkelijke dalingen na 50/75/90, factor 0,1 |
| Duur | 100 volledige epochs |
| Trainvolgorde | E4M3 → E2M3 → E2M1 |
| Trainloss | hard CE voor E4; detached E4-softtargets voor E2M3; detached E2M3-softtargets voor E2M1 |
| Update | gesommeerde gradients, geen deling door 3; één Adam-stap |
| Evaluatie | E2M1 → E2M3 → E4M3; harde CE voor alle modi |
| BN-banken | 8/32/4 voor E4M3/E2M3/E2M1; normaal leren in train, bevroren in eval |
| Backend | oorspronkelijk cuDNNbenchmark=True; geen nieuwe deterministische override |

Elke epoch omvat 573 trainminibatches en 204 testminibatches. Honderd epochs
omvatten 57.300 optimizerupdates en 171.900 trainingspasses. De scheduler
wordt op de oorspronkelijke epochgrens bijgewerkt. De leerwaarden per bereik
zijn 0,001 voor 1–50, 0,0001 voor 51–75, 0,00001 voor 76–90 en 0,000001
voor 91–100.

## Checkpoints en presentatie

Iedere volledig afgeronde epoch wordt opgeslagen met model, Adam, scheduler,
RNG, voortgang en contract. Een stop op de epochgrens is hervatbaar binnen
hetzelfde nieuwe reference100-contract. Een eerdere stop wordt als
`STOPPED_AT_EPOCH_BOUNDARY` gerapporteerd; volledige training is uitsluitend
voltooid bij epoch 100.

De twee presentatie-eindpunten zijn epoch 100 en het eerste strikte maximum
van E4M3-testaccuracy over de volledige run. Alle drie modi worden voor elk
eindpunt uit hetzelfde checkpoint gebruikt. Bij een onderbreking wordt de
laatste beschikbare epoch getoond en blijft epoch 100 expliciet ongemeten.

De officiële testset wordt volgens de bestaande werkwijze ook voor de
checkpointselectie gebruikt. De geselecteerde score is daarmee geen
onafhankelijke held-out modelschatting.

Voor ieder geselecteerd checkpoint wordt de E4-bronexport opgeslagen en
gecontroleerd herladen. E2M3 en E2M1 worden uitsluitend uit die opgeslagen
broncodes, bron-schaalcodes +6 en overige modelstaat afgeleid. De compacte
exports bevatten de BN-/hulpstaat, maar niet de zes FP32-MX-masters.
Afzonderlijke volledige trainingscheckpoints blijven beschikbaar.

De opgeslagen exports worden op het trainingdevice exact vergeleken met
de live gedecodeerde gewichten. Hiervoor is geen aanvullende volledige
testsetevaluatie nodig. Bronhashes en schaalrelaties worden gerapporteerd.

## Interpretatie

Het rapport bewaakt brondeling, schaalcode+6, mantissaschaalbehoud, lege
caches, teachers, optimizerstappen, FP32-toestand, BN-tellers en dataomvang.
Accuracy en klasseverdeling zijn meetuitkomsten. Er is geen minimumaccuracy
voor een technische PASS en er vindt geen automatische factorzoektocht plaats.

De huidige pilot ondersteunt dat gezamenlijke training met deze route kan
leren en geeft aanleiding tot deze langere proef. Eén seed en vijf epochs
bewijzen geen eindconvergentie of algemene superioriteit. Het verschil met
directe quantisatie omvat de volledige conversieketen en leerdynamiek; deze
proef is geen geïsoleerde meting van uitsluitend de resterende exponentbits,
BN-aanpassing of mantissatruncatie.

De convoluties gebruiken FP32 met numeriek geëmuleerde MX-gewichten.
De gemeten looptijd is geen native MX-throughput- of energiemeting.
De notebook voert geen training of modelinferentie uit en bevat geen
opgeslagen resultaten bij levering.
