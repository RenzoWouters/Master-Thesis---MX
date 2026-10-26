# Onafhankelijke code- en herkomstcontrole

Status: **PASS**. Deze controle is een beoordeling van de broncode en de koppeling met de echte feedbackbundels. De afzonderlijke uitvoeringstests worden apart gerapporteerd.

Het pakket herstelt het echte, bevroren checkpoint na epoch55 strikt onder het oorspronkelijke pilotcontract. Model, BatchNorm, Adam, scheduler en RNG worden volledig teruggezet. De extra offset blijft op de waarden na epoch50; alleen de horizon van het experiment wordt uitgebreid. Er wordt getraind vanaf56, met de oorspronkelijke LR-dalingen na75 en90.

De 127 geërfde pilotbestanden zijn bytegelijk aan het eerder geleverde pakket en het pakketmanifest van de werkelijke run. De actualpilot55contracthash, modelhash, checkpoint-hash en tracehash zijn onafhankelijk uit de feedback gereconstrueerd. De oorspronkelijke dynamische100geschiedenis is nog steeds verplicht als lokaal invoerbestand en wordt op de eerder vastgelegde SHA256 gecontroleerd voordat zij als vergelijkingsbron wordt gebruikt.

De selectie/export onderscheidt drie situaties: de gedeelde fase1–50 met haar oorspronkelijke geleerde theta, de geërfde bevroren pilot51–55, en de nieuwe voortzetting56–100. Elk geselecteerd checkpoint wordt onder zijn eigen contract gecontroleerd. Het globale beste checkpoint wordt expliciet als vóór freeze gelabeld wanneer dat van toepassing is. Het beste checkpoint na freeze wordt uitsluitend binnen51–100 gekozen, telkens met alle drie precisies uit hetzelfde checkpoint. Epoch100 blijft het hoofdvergelijkingspunt.

De samenvatting onderscheidt alle100epochs, de50bevrorenepochs en de45nieuweepochs. Oude resultaten en de oorspronkelijke invoermappen worden beschermd. De oorspronkelijke manifestnaam wordt gearchiveerd voordat het nieuwe manifest wordt geschreven.

Er zijn bij deze controle geen materiële codeproblemen gevonden. Er zijn hier geen daadwerkelijke CUDA-/SVHN-checkpoints uitgevoerd; de echte .pt-bestanden staan op de laptop en worden door de runner tegen de vastgelegde herkomst gecontroleerd.
