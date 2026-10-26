# Onafhankelijke codecontrole — freeze na epoch50

Status: PASS voor de gelezen code en herkomstcontrole. Dit is geen verslag van een uitgevoerde SVHN-training.

De 93 geërfde runtimebestanden en hun manifest zijn ongewijzigd. Het oudercontract, de exacte epoch50-staat, de historische vergelijkingsrecords51–55 en de gereconstrueerde checkpointgeschiedenis zijn onafhankelijk vergeleken met de ontvangen feedbackbundel. De branch herstelt het volledige oudercheckpoint vóór het vastzetten van de theta-LR, bewaart model/BN/Adam/RNG en controleert na elke update dat theta en harde offsetkeuzes exact gelijk blijven. De E4M3-bronschaal blijft uit de actuele gewichten bepaald. Absolute epoch-, Adam- en BN-tellers, het oorspronkelijke LR-schema en de historische beeldvolgorde blijven gecontroleerd. De beste checkpointselectie is beperkt tot de nieuwe epochs51–55.

Twee concrete bevindingen zijn opgelost en opnieuw gelezen:

1. Het oorspronkelijke wijzigingsmanifest wordt meteen na het aanmaken bewaard als `parent_training_change_manifest.json`, zodat de nieuwe branch zijn eigen manifest eenmaal kan schrijven.
2. De bescherming van invoermappen onderscheidt checkpoints in `checkpoints/` van kopieën direct in een resultatenmap. Daardoor blijven nieuwe resultaatmappen naast de bestaande run mogelijk bij beide hervatpaden.

De implementatie bevriest theta via LR=0; schaalgradients en Adam-momenten worden nog berekend. Er wordt geen geoptimaliseerde trainingskost geclaimd. De precieze gelezen codehashes staan in `code_review.json`. Uitvoerings- en hervattests worden afzonderlijk gerapporteerd.
