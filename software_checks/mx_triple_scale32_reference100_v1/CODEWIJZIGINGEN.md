# Wijzigingen: ×32-pilot5 naar ×32-reference100

## Ongewijzigde numerieke kern

De achttien bestanden in `REUSE_MANIFEST.json` zijn byte-identiek aan de geslaagde ×32-pilot. Dit omvat E4M3/E2M3/E2M1-codecs, de +5-schaalcorrectie, bitmapping, identity-STE, broncache, gezamenlijke trainingsondersteuning, checkpointmechanisme, oorspronkelijke bron-ZIP en kerncontroles.

De vaste factor blijft 32. Er worden geen extra precisies, activatiequantisatie, trainbare schalen of nieuwe verliesfuncties toegevoegd.

## Runner

`run_train.py` gebruikt de beproefde structuur van de eerdere volledige ×64-run, met de numerieke ×32-route uit de pilot. In het trainingscontract veranderen uitsluitend:

- schema en omschrijving van het experiment;
- `pilot_stop_epoch=5` naar `reference_end_epoch=100`;
- rapportage van het gezamenlijke E4M3-bestcheckpoint en epoch100.

De seed, initialisatie, dataset, batchgrootte, Adam, LR=0.001, mijlpalen 50/75/90, driepassesvolgorde en shared-source-regel blijven behouden. De nieuwe volledige run begint vanaf de oorspronkelijke initialisatie, zoals de eerdere volledige referenties. Hervatten is voor onderbroken sessies binnen dit nieuwe 100-epochcontract.

De runner bewaart alle honderd epochcheckpoints en schrijft pas een voltooide 100-epochstatus wanneer epoch 100 werkelijk is gemeten. Een vroege stop blijft `STOPPED_AT_EPOCH_BOUNDARY`. Looptijdprojecties en gemeten honderd-epochtijden krijgen afzonderlijke velden.

## Notebook en vergelijking

`MX_triple_scale32_reference100_resultaten.ipynb` en `notebook_support.py` verwerken de volledige history. De hoofdfiguren zijn epoch100 en het gezamenlijke E4M3-bestcheckpoint. De ×64-referentie wordt nu over de volledige honderd epochs gebruikt. Bij onvolledige runs zijn vergelijkingen beperkt tot de gedeelde reeds gemeten range.

De zes originele ×64-referentiebestanden blijven ongewijzigd. Alleen het nieuwe `scale64_reference/REFERENCE.json` beschrijft dat dit pakket de volledige vergelijking uitvoert. Het oude pilotpakket en zijn vijf-epochvergelijking blijven behouden.

## Bewijs en reproduceerbaarheid

`pilot_analysis` bevat de echte vijf-epochmetingen en de onafhankelijke audit, zonder een nieuw uitgevoerde notebook. `provenance` bevat de oorspronkelijke pilotrunner, relevante manifests en validaties. Het concrete runnerdiff staat eveneens daar.

De tests in `local_validation` controleren het eindpunt 100, LR-overgangen, hervatting en exports met kleine synthetische gegevens. De bestaande numerieke kernvalidatie wordt aan de ongewijzigde hashes gebonden. De werkelijke SVHN-reference100-training moet op de gebruikerslaptop worden uitgevoerd.
