# Checkpointselectie voor de Any-Precision GitHub200-referentie

## Doel en vastgelegd selectiecriterium

Dit is een NIEUWE, afzonderlijke evaluatiestap, geen wijziging aan de voltooide
training. De gebruiker heeft tijdens de training afgesproken de selectieregel
van de aangeleverde GitHub-code achteraf op de bewaarde checkpoints toe te passen.
Het training-contract met `explicit_endpoint_evaluation_only` blijft ongewijzigd
in de oorspronkelijke bestanden; deze aanvullende analyse legt haar eigen beleid
vast in `evaluation_plan.json`.

Bron: `train.py` uit `Any-Precision-DNNs-master(1).zip`, SHA-256
`318ef0c73b80634e9a4de0cf7dbf7caff2c0761abfa8d48c78b08a9a8d266b5b`.
Regels 107-134 evalueren na iedere epoch en gebruiken `val_prec1[-1] > best_prec1`.
Regels 153-175 sorteren de precisies en voeren evaluatie zonder gradients uit.
In de gekozen configuratie is de laatste precisie 32. Bij een exact gelijke
score blijft het eerder gekozen checkpoint behouden.

## Procedure

1. Gebruik het volledige checkpoint van epoch 200 als referentie voor het
   gevolgde trainingstraject. Gebruik alleen eigen vertrouwde checkpoints.
2. Inventariseer `*/checkpoints/epoch_*.pt` onder de opgegeven trainingshoofdmap.
   Epoch 0 is initialisatie en wordt NIET als kandidaat getest.
3. Verifieer SHA-256, checkpointformaat, configuratie/omgeving/broncode,
   epochteller en optimizerstappen. Vergelijk de volledige opgeslagen
   trainingshistorie van iedere kandidaat met het overeenkomstige beginstuk
   van de historie in epoch 200. Controleer daarnaast de modelinhoud met de
   modelhash uit die eindhistorie. Zo worden andere trainingstrajecten niet
   stilzwijgend vermengd.
4. Alle epochs 1-200 moeten geldig aanwezig zijn. Ontbrekende epochs blokkeren
   de sweep. Andere/ongeldige kandidaten en identieke duplicaten worden
   afzonderlijk in `inventory.json` vermeld; ze worden niet gewist.
5. Test de 10.000 officiele CIFAR-10-testbeelden per checkpoint bij W32/A32.
   Gebruik batchgrootte 128, `shuffle=False`, de oorspronkelijke `val`
   transformatiefunctie (ToTensor en dezelfde normalisatie; geen crop/flip),
   `model.eval()` en `torch.no_grad()`. Er is geen BatchNorm-herkalibratie.
6. Behoud de oorspronkelijke accuracyhelper en gewogen AverageMeter-berekening.
   Bewaar daarnaast gehele aantallen correcte voorspellingen als controle.
   Pas strikt `>` toe in oplopende epochvolgorde. Bewaar bij gelijkstand de eerste.
7. Test alle vijf precisies van DATZELFDE gekozen checkpoint. Rapporteer GEEN
   afzonderlijke maxima per precisie als waren die een enkel model.
8. Rapporteer daarnaast alle vijf precisies van het vooraf vastgelegde
   eindcheckpoint (epoch 200). Wanneer dit ook de geselecteerde epoch is,
   hoeft de volledige precisievector maar eenmaal te worden berekend.
9. Controleer dat de W32-uitvoer, geteld over alle testbeelden en gehasht, identiek
   is in de W32-only-sweep en de volledige vijf-precisie-evaluatie.
10. Controleer dat model/BatchNorm en alle gekozen invoercheckpoints ongewijzigd
    blijven. Er wordt geen optimizerstap, backwardpass of training uitgevoerd.

## Hergebruik van broncode

De aanvulling importeert het BESTAANDE `anyprecision_reference_github200_v1`
via `--reference-dir`. Er wordt geen tweede modelimplementatie gebruikt.
De relevante referentiebestanden worden geverifieerd via REFERENCE_SHA256.json
EN het oorspronkelijke pakketmanifest. De oorspronkelijke drie compatibiliteits-
aanpassingen blijven hetzelfde (accuracy reshape, lazy gpustat en SVHN-import).
De nieuwe evaluatielus gebruikt de oorspronkelijke modelmodules, precisieselectie,
loss en accuracyhelper. Zij is geen aanroep van de oorspronkelijke train.py-launcher.
De eval-besturing zelf is nieuw en lokaal vergeleken met de originele forward-functie.

## Opslaan, pauzeren en hervatten

Elke volledig afgeronde checkpointtest krijgt een afzonderlijk verzegeld JSON-
resultaat met checkpoint-, protocol- en uitvoerhash. Dezelfde opdracht met dezelfde
uitvoermap hergebruikt die resultaten. Een wijziging aan code, dataset, checkpoint-
reeks of protocol wordt niet stilzwijgend als dezelfde evaluatie behandeld.

`--max-checkpoints 0` verwerkt alle nog ontbrekende W32-tests. Een positief getal
begrenst het aantal NIEUWE W32-tests in die sessie. Deze limiet verandert niet de
kandidatenreeks: pas als alle 200 resultaten aanwezig zijn, volgt de eindselectie.

`request-stop --session-dir <actieve sessie>` vraagt een stop na het huidige
checkpoint. Tijdens de twee afsluitende volledige precisie-evaluaties wordt niet
halverwege gestopt. Bij onverwachte onderbreking worden reeds afgeronde JSON-
resultaten behouden; een onvolledige checkpointtest wordt opnieuw uitgevoerd.
Een systeemlock voorkomt twee gelijktijdige schrijvers in dezelfde evaluatiemap.

Oude sessierapporten worden niet overschreven: elke uitvoering krijgt een nieuwe
map onder `sessions/`. De modelcheckpoints blijven op hun oorspronkelijke plaats;
`best_checkpoint.json` verwijst naar het geselecteerde bestand en zijn hash.

## Rapportage en beperkingen

- `checkpoint_scores.csv`: 200 W32/A32-testmetingen als de sweep compleet is.
- `selected_checkpoint_metrics.csv`: vijf precisies van het ene gekozen model.
- `final_checkpoint_metrics.csv`: vijf precisies van epoch 200.
- `github_comparison.csv`: beide scorevectoren naast de bevroren README-rij
  `Resnet-20-Any (recursive)`: 91.89, 93.90, 93.86, 93.75, 94.11 procent.
- `selection_report.json`: protocol, herkomst, controles, waarschuwingen en timing.

Het op de testset geselecteerde resultaat is GEEN onafhankelijke testschatting
na modelselectie. Het dient hier expliciet om het selectiecriterium uit de
GitHub-code na te bootsen op ONZE checkpointreeks. Het bewijst niet dat de
historische README-cijfers met hetzelfde checkpoint, seed of software zijn gemaakt.
Het uitstellen van evaluatie reconstrueert ook niet alsnog het exacte historische
trainingstraject. De training en haar bekende implementatie-/runtimeverschillen
blijven gedocumenteerd zoals voorheen.

PASS betekent volledige evaluatie en geslaagde consistentiecontroles, NIET dat een
accuracydrempel is behaald. Er worden geen onderzoekscijfers voorspeld of ingevuld.
Geen MX, truncatie vanuit opgeslagen broncodes, continual learning of hardwarewinst
wordt met deze stap gemeten. Voor het later afstemmen van MX blijft een aparte
ontwikkel-/validatieprocedure nodig, niet herhaald optimaliseren op deze testset.
