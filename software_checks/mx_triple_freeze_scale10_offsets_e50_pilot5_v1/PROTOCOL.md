# Tien schaalkeuzes vast na epoch 50 — proef 51–55

## Onderzoeksvraag en vergelijking

Helpt het vastzetten van de tot epoch 50 geleerde offset per groep, terwijl de gewichten en BN verder trainen, in deze korte fase ten opzichte van het blijven leren van die offsets?

Bron: de voltooide tienkeuzerun `mx_triple_learned_scale10_reference100/20261104_193228_406260_joint_run`. Alleen de bevroren voortzetting wordt nieuw uitgevoerd. De doorlerende controle komt uit de reeds gemeten epochs 51–55. De primaire vergelijking is de accuracy op epoch 55 voor E4M3, E2M3 en E2M1. Een tweede vergelijking gebruikt per variant het eerste strikte E4M3-maximum binnen dezelfde nieuwe fase 51–55; alle drie precisies worden op datzelfde checkpoint gerapporteerd.

Dit is een vervolgproef op één bestaande seed en een historische controle. De gegevens zijn geschikt om de volgende stap te kiezen, maar bewijzen geen algemene superioriteit of optimaal freeze-moment. De oorspronkelijke CUDA-backend is niet bitexact reproduceerbaar. Selectie op de officiële testset blijft herkenbaar als zodanig.

## Beginstaat en ingreep

Checkpoint `epoch_00050.pt` en zijn checksum blijven lokaal bij de gebruiker. SHA256:

`0276dc622aaa3b763ef3929de8713e673b6734947d53f1f3f4c7316e0d228b61`

Vóór de eerste update worden bestand, protocol, model, theta, Adam, scheduler, RNG en oorspronkelijke geschiedenis gecontroleerd en hersteld. Alleen daarna wordt de leersnelheid van de `scale_proxy`-groep nul, inclusief bijhorende `initial_lr`, `scheduler.base_lrs` en `_last_lr`. Modelwaarden en Adam-momenten veranderen niet tijdens die overgang. De bestaande gewichts-LR vanaf epoch 51 blijft 0,0001; het globale schema 50/75/90 wordt niet herstart.

Theta behoudt `requires_grad=True`. De bestaande schaalgradient en Adam-momentupdates blijven actief, maar leiden bij LR=0 niet tot theta-updates. Dit volgt de eerdere binary freezeproef en is geen geoptimaliseerde uitvoering voor kostenmetingen. Een guard controleert iedere optimizer-update op exact ongewijzigde FP32-theta én harde keuze; ook drift binnen een keuzebin leidt tot FAIL.

## Formaten en bitbewerking

De numerieke runtime van de tienkeuze-referentie is byte-identiek meegeleverd. Groepsgrootte B32; E8M0-schalen; A32; FP32-masters, gradients en Adam-momenten. De eerste en laatste laag blijven FP32. De zes MX-lagen bevatten samen 4.080 groepen.

De hoogste precisie wordt steeds opnieuw uit de actuele voorbewerkte gewichten bepaald:

`E4M3: s e3 e2 e1 e0 m2 m1 m0 → E2M3: s e3 e2 m2 m1 m0 → E2M1: s e3 e2 m2`.

De E4M3-bron behoudt de bestaande afronding en schaalbepaling. Lagere codes ontstaan uitsluitend door de bestaande bitbewerkingen. Voor een groep en trainingsstap t geldt:

`scale_E2M3(t) = scale_E4M3(t) × 2**offset_epoch50`

`scale_E2M1(t) = scale_E2M3(t)`.

De vaste offset ligt in 0..9, overeenkomstig ×1, ×2, ×4, ×8, ×16, ×32, ×64, ×128, ×256 en ×512. **De offset bevriest, niet de E4M3-bronschaal.** E4M3-codes en bronschalen blijven de veranderende gewichten volgen. Exponenttruncatie, mantissatruncatie, schaalgradient en identity-STE volgen verder exact dezelfde runtime als de parent.

## Data en updates

SVHN train-only: 73.257 trainingsbeelden en 26.032 officiële testbeelden; geen extra-split en geen nieuwe downloads. Batch 128; 573 optimizer-updates per epoch. Drie trainingspasses per minibatch: E4M3 hard CE, E2M3 soft CE met losgekoppelde E4M3-teacher, E2M1 soft CE met losgekoppelde E2M3-teacher; temperatuur 1, gradients gesommeerd, één Adam-update. Alle drie BN-banken leren verder. Volledige evaluatie van de drie precisies volgt iedere epoch zonder updates of BN-herkalibratie.

De nieuwe trace bevat uitsluitend epochs 51–55. De parentgeschiedenis staat als aparte, gecontroleerde referentie in `parent_reference`. De samplevolgorde per epoch moet overeenkomen met de bestaande doorlerende run. Een gedeeltelijke proef kan alleen via een nieuw checkpoint 51–54 worden hervat; checkpoint 55 bewaart de volledige trainingsstaat voor een later vervolg tot 100.

## Notebook en rapportage

Accuracycurves laten beide voortzettingen vanaf hetzelfde checkpoint 50 zien, met een extra overzicht van de geërfde fase 1–50. Accuracygrafieken tonen het laatste checkpoint en het gemeenschappelijke beste E4M3-checkpoint binnen de nieuwe fase. Een globaal beste checkpoint van vóór de freeze wordt niet als resultaat van de freeze gepresenteerd.

Bij **volgende proeven met meerdere schaalfactoren** is een staafdiagram verplicht: alle factoren op de as, inclusief nulwaarden, met aantal groepen én percentage bij iedere balk. Deze notebook bevat die figuur voor het laatste en voor het beste E4M3-checkpoint. De bevroren variant hoort op beide dezelfde verdeling als checkpoint 50 te tonen. Bestaande notebooks worden daarvoor niet achteraf gewijzigd.

Verder bevat de notebook verlies- en tijdcurves, verschillen tegenover de historische controle, freezecontroles en opslagkosten. Vier bits per groep geven 2.040 bytes selectorinformatie voor een flexibele E4M3-bronexport. Een afgewerkte lagere precisie verwerkt de offset in de eigen E8M0-schaal. Deze getallen zijn een analytische payloadtelling, geen gemeten GPU-geheugenverkeer.

## Behoud en controleerbaarheid

Iedere opdracht schrijft naar een nieuwe tijdgestempelde map. Originele checkpoints, datasets, feedbackbundels en notebooks blijven ongewijzigd. Compacte exports moeten exact dezelfde gedequantiseerde gewichten opleveren als de live trainingsroute, zonder opnieuw lagere precisies uit de masters te bepalen. De geschreven `history.json` wordt volledig teruggelezen en exact met de branchtrace vergeleken voordat de proef PASS kan krijgen.

Een technische `check` voert de codec-, gradient-, restore- en freezecontroles uit zonder SVHN-training. De meegeleverde lokale validatie gebruikt uitsluitend kleine synthetische CPU-tensors. Die testwaarden zijn geen onderzoeksaccuracies en voorspellen geen CUDA-herhaalbaarheid.
