# Lokale verificatie — B16 ×64, volledige 100-epochproef

## Numerieke hergebruikcontrole

De negentien numerieke/runtime/testmodules die de MX-codecs, groepsgrootte, schaalroute, truncatie, G(W), BN, broncache, databewerking en optimizerlus bepalen zijn byte-identiek aan de geslaagde B16-pilot v2. Het upstreamarchief is hetzelfde. De volledige configuratie verschilt van de pilot uitsluitend in schema, doelomschrijving, eindpuntveld en rapportagebeleid. De werkelijke gebruikerspilot heeft 196 consistentiecontroles doorstaan; zie `pilot_analysis/audit_summary.json`.

## Werkelijke opstartfunctie

`dataset_startup_regression.json` beschrijft een geslaagde controle van de echte `create_context(check_only=False)`, met alleen CPU-keuze en datasetinhoud/-lader vervangen door expliciete synthetische fixtures. De bestandscontrole en datasetvalidatie zelf bleven actief. Canonieke `train_32x32.mat`/`test_32x32.mat`, aantallen 73.257/26.032, model/Adam/scheduler, laagvormen en eerste 128 beelden per loader werden gecontroleerd. Geen datasetdownload, echte SVHN-beelden of optimizerupdates in deze test.

## Volledige levenscyclus op kleine synthetische CPU-inputs

`training_integration_report.json` beschrijft een PASS voor de echte runnerloop op drie synthetische beelden. De fixture vervangt alleen contextconstructie/dataselectie en CUDA-routing; de werkelijke loop, loss, backward, Adam, scheduler, checkpointrestore, exports en feedbackopbouw worden uitgevoerd.

Gecontroleerd zijn honderd mini-epochs, alle vier LR-fasen, exact hervatten 99→100 en exact hervatten over de grenzen 49→50/74→75/89→90. Model, optimizer, scheduler en RNG stemmen overeen met de ononderbroken CPU-route. Ook vroege stop, afwijzing van het pilotcontract, een uitsluitend op B32 afwijkend contract en een al afgewerkt epoch 100-checkpoint slagen. De zes compacte exports worden via onafhankelijke bits/schalen, heropbouw uit alleen de bron en kosten gecontroleerd.

De volledige fixture duurde 41,87 seconden en gebruikte 210 synthetische optimizerupdates over alle scenario's. Dit is geen SVHN-training, accuracy- of GPU-timingmeting. De oorspronkelijke CUDA-backend geeft geen bitexacte herhaal- of hervatgarantie.

## Notebook

`notebook_validation_report.json` bevestigt uitvoering van alle 14 codecellen van de 24-cellige notebook op de daadwerkelijke volledige en gedeeltelijke synthetische runneruitvoer. De volledige fixture eindigt op 100; de gedeeltelijke op 1 en krijgt geen gefingeerd epoch 100-resultaat. Beide analyses zijn read-only, maken tien figuren en valideren zes best/last-exporttabellen.

De E4M3-selectieregel is apart gecontroleerd met een maximum op 54 en een tie op 55: 54 blijft geselecteerd, terwijl de eigen maxima van lagere precisies op 77/83 niet worden bijgemengd. De B32-historie is hashbevestigd; synthetische resultaten worden niet als echte trainingsvergelijking gepresenteerd. Bij gedeeltelijke resultaten gebruiken vergelijkingscurves het gedeelde afgewerkte bereik.

De primaire figuren zijn visueel gecontroleerd; een overlap tussen legenda en accuracylabel is opgelost en de cellencontrole is daarna opnieuw geslaagd. De gebruikersnotebook blijft onuitgevoerd met lege outputs. Er wordt geen uitgevoerde notebook of synthetische onderzoeksgrafiek meegeleverd. De bestaande pilotnotebook werkt correct op de aangeleverde echte feedback.

## Bereik

Alle oude resultaten, bronbestanden en feedbackbestanden blijven behouden. De gebruiker voert de gewone technische `check` en de nieuwe honderd-epochrun zelf op de bestaande Windows/CUDA-omgeving uit. Er is bij pakketvoorbereiding geen nieuwe echte SVHN-training gestart.
