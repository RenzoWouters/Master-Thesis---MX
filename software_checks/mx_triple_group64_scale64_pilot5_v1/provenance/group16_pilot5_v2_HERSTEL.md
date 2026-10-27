# Herstel van datasetcontrole — versie 2

## Oorzaak

Bij de aanpassing van schaalfactor ×32 naar ×64 is de tekst `x32` te breed vervangen. Daardoor veranderden in de runner twee verwachte SVHN-bestandsnamen van `train_32x32.mat` en `test_32x32.mat` naar `train_32x64.mat` en `test_32x64.mat`. De datasetcontrole zelf vond en verifieerde juist de correcte 32×32-bestanden. De daaropvolgende expliciete vergelijking stopte daarom vóór modelconstructie en training.

## Wijziging

De twee verwachte namen zijn hersteld. In `run_train.py` verschilt verder uitsluitend de gerapporteerde pakketversie v1→v2. Alle 20 andere productiemodules en `PROTOCOL.json` zijn byte-identiek aan v1. De nieuwe pakketchecksums zijn opnieuw berekend. De notebook blijft onuitgevoerd; alleen haar versieaanduiding is bijgewerkt.

## Gerichte regressiecontrole

De oude fout is gereproduceerd via de echte `create_context(check_only=False)`. Vervolgens doorliep v2 diezelfde functie succesvol, inclusief bestandcontrole, datasetvalidatie, modelconstructie, optimizer, scheduler, kostencontrole en het laden van de eerste train-/testbatch.

Alleen de CPU-keuze en de externe datasetinhoud/-lader werden voor deze test vervangen door expliciete synthetische fixtures. De werkelijke bestandscontrole en `validate_dataset` bleven actief. De virtuele gegevens hadden de oorspronkelijke aantallen 73.257/26.032 en vorm `(N,3,32,32)`; de echte transformaties gaven batches `(128,3,40,40)`. De normale loaderaantallen 573/204 werden gecontroleerd. Er werden nul trainingsepochs en nul optimizerupdates uitgevoerd. Het aanwezige extra-fixturebestand werd niet gehasht of geladen; alle fixturebestanden bleven ongewijzigd.

De eerdere numerieke tests worden in `local_validation/inherited_v1` bewaard als historische verificatie van de ongewijzigde modules. Ze waren geen dekking van deze opstartroute: `check` eindigt vóór datasetladen en de vroegere synthetische trainingstest verving de hele contextconstructie. Dat ontbrekende deel heeft nu een eigen regressietest.

## Uitvoeren

Pak de ZIP uit in `software_checks`; de nieuwe map heet `mx_triple_group16_scale64_pilot5_v2`. Gebruik de v2-commando's in de README: eerst `check`, daarna `run`. Gebruik de bestaande SVHN-data. De mislukte v1-run bevatte nog geen trainingsepoch om te hervatten; v2 begint met dezelfde oorspronkelijke initialisatie. Iedere opdracht schrijft een nieuwe resultatenmap, zodat eerdere resultaten behouden blijven.
