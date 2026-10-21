# Korte spreeklijn voor de meeting

1. Startpunt: oorspronkelijke Any-Precision-code werkend, ResNet20/CIFAR10 en daarna CNN8/SVHN. Train-only gekozen om herhaalbare ontwikkelruns haalbaar te maken; geen exacte paperreproductie met train+extra.
2. Gewichts-MX werkt tijdens training: FP32-updatemasters, A32-activaties, zes gekwantiseerde convoluties. Geen algemene claim dat dedicated nauwkeuriger is; de dedicated E2M1-controle ontbreekt.
3. Gewone gezamenlijke MX-training haalt 95,0254% bij directe E2M1-uitvoering op epoch100. Achteraf trunceren van hetzelfde model haalt 93,5810%.
4. Dezelfde truncatie tijdens het leren gebruiken geeft 94,9408%. Dat is +1,3599 procentpunt tegenover achteraf trunceren en -0,0845 tegenover directe kwantisatie. Een seed; geen statistische gelijkwaardigheid.
5. Een bewaarde E2M3-bron kan ook de E2M1-modus leveren. Ruwe modelinhoud: 121.248 bytes inclusief schalen, padding en alle bestaande uitzonderingen/BN, versus 537.136 bytes vergelijkbare FP32-modelstaat. Factor4,43 betreft modelinhoud, niet trainingsgeheugen.
6. Binnen MX is de bronwinst tegenover twee directe codesets met gedeelde metadata circa35% (186.528 -> 121.248 bytes). Dit is een expliciet layoutscenario, geen energiemeting.
7. De volgende exportcontrole laadt de hoge bron zonder mastercheckpoint en evalueert beide modi opnieuw. Geen nieuwe training nodig. Daarna prioriteiten bespreken: dedicated lage baseline, meerdere seeds, onafhankelijke validatie, train+extra-bevestiging en hardware/continual learning.

## Formuleringen om niet te gebruiken
- 'Any-precision is altijd minder nauwkeurig dan dedicated.' Niet aangetoond.
- 'Ons volledige trainingsgeheugen is 4,43x kleiner.' Masters en Adam blijven FP32.
- 'Vierbitmodus betekent dat de opgeslagen bron nog maar vier bit bevat.' De hoge bron blijft behouden voor terugschakelen.
- 'De GPU of accelerator verbruikt nu 32% minder energie.' Niet gemeten.
- 'Onze implementatie gebruikt minder opslag dan iedere int4-oplossing.' MX voegt groepsschalen toe; vergelijking hangt af van de functie en metadata.
- 'Het uiteindelijke exportbestand is 121.248 bytes.' Dat is ruwe inhoud; de runner rapporteert 164.005 bytes voor de hoge .pt-export, inclusief serialisatie/metadata.
