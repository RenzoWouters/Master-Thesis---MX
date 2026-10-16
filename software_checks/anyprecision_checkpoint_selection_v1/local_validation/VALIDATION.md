# Lokale validatie van checkpointselectie v1

## Werkelijk uitgevoerd

Python 3.13.5, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu op Linux.
De oorspronkelijke verbrede ResNet-20 (expand=5) en quantizers zijn gebruikt.
De datasets bestonden uit KUNSTMATIGE pixels/labels opgeslagen in CIFAR-formaat;
er zijn drie mini-epochs getraind op twintig beelden om checkpointfixtures te
maken. Dit zijn geen CIFAR-10-accuracyresultaten van de gebruiker.

Alle 24 controles in `validation_result.json` zijn geslaagd. Onder andere:

- Loss en top-1/top-5 exact gelijk aan de echte `forward`-functie uit het
  aangeleverde `train.py`. Voor deze CPU-test werd uitsluitend `.cuda()` tijdelijk
  een identiteitsbewerking; deze aanpassing zit NIET in het evaluatiescript.
- W32-only gelijk aan W32 in de vijf-precisielus, inclusief hash van alle logits.
- Modelgewichten en BatchNorm niet veranderd; geen gradients aangemaakt.
- Gelijke maximale scores kiezen de vroegste epoch; strikt hogere score vervangt die.
- Afwijkende trainingshistorie, verkeerde checksum, ontbrekende epoch en gewijzigd
  resultaat worden afgewezen. Identieke duplicaten worden expliciet gemeld.
- Volledige sweep, gepauzeerde/hervatte sweep in nieuwe processen en hergebruik van
  alle resultaten leveren dezelfde selectie en metrics; score-CSV identiek.
- Bestaande onderzoekscheckpoints en oude resultaten worden niet overschreven.
- Een tweede proces kan niet tegelijk schrijven in dezelfde evaluatiemap.
- Stopverzoekbestand gemaakt; stopverzoek aan een afgesloten sessie wordt geweigerd.
- Geen geregistreerde Python-waarschuwingen in de geslaagde evaluatiesessies.

Twee eerste lokale uitvoeringen van de testorchestratie werden door de externe
commandotijdslimiet afgebroken. De afzonderlijke evaluatiefasen die al klaar waren,
waren geslaagd. De volledige testorchestratie is daarna in een nieuwe werkmap
uitgevoerd en afgerond; het bijgevoegde resultaat hoort bij die complete uitvoering.
Er zijn geen trainingsbestanden van de gebruiker gebruikt of gewijzigd.

## Niet getest in deze omgeving

- Geen Windows/CUDA-uitvoering; de Windows-locktak is codegecontroleerd maar niet
  op Windows uitgevoerd. De Linux-locktak is met twee processen getest.
- Geen volledige sweep met de 200 echte checkpoints: die staan op de laptop.
- Geen echte CIFAR-10-testaccuracy; de echte dataset is niet in deze omgeving aanwezig.
- Geen garantie op gelijke resultaten tussen bibliotheekversies of GPU-types.
- Stopverzoeken worden tussen W32-checkpointtests gehonoreerd; de twee laatste
  vijf-precisietests worden als afsluiting uitgevoerd, niet halverwege gepauzeerd.

## Reproduceerbaarheid van de lokale technische controle

Het reeds meegeleverde referentiepakket bevat
`local_validation/test_integration.py` en `fixture_reference.py`. De eerste maakt
in een nieuwe werkmap kunstmatige bestanden plus drie mini-epochcheckpoints.
`local_validation/test_selection.py --help` beschrijft de paden die nodig zijn
om die kunstmatige fixtures voor de selectiecontrole te gebruiken.
Deze scripts zijn geen instructies om extra onderzoekstraining op de laptop te doen.

De scripts draaien de ongewijzigde referentiemodules en de nieuwe evaluatie-aanvulling.
Geen fixture-optie is beschikbaar in de gewone productie-CLI. De lokale shim
markeert alle uitvoer als `valid_research_result: false`.
