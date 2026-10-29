# Wijzigingen ten opzichte van de geslaagde 8×8-pilot

De numerieke trainingsimplementatie is byte-identiek behouden: geometrie, drie scalar/tensorcodecs, packing, G(W), ×64-truncatie, identity-STE, recursieve losses, model, optimizer, datahelpers en RNG-hervatting.

De runner configureert nu een vaste eindepoch 100, met een nieuwe protocol-/rapportschema-identiteit en nieuwe resultatenmap. Het bestaande LR-schema 50/75/90 wordt nu volledig doorlopen. Voltooide- en deeltijdrapportage onderscheiden de gemeten 100-epochduur van een projectie. Hervatten blijft beperkt tot hetzelfde contract en weigert de pilot.

De notebook bevat volledige curves en vergelijkt epoch 100 plus het gezamenlijke E4M3-beste checkpoint met de bestaande B32- en B64-runs. De referentiebestanden blijven byte-identiek; alleen hun vergelijkingsmetadata gaat naar het volledige bereik. Kostenberekening en exportgeometrie blijven ongewijzigd.

Alleen identiteit, omschrijving, eindepoch en rapportagebeleid verschillen in PROTOCOL.json. De exacte vijf gewijzigde/verwijderde/toegevoegde velden staan in TRAINING_CHANGE_MANIFEST.json. De geslaagde echte pilot is vastgelegd in pilot_analysis; oudere validatie is als zodanig onder provenance bewaard.
