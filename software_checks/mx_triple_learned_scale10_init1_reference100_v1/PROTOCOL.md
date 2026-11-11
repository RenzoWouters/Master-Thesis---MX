# Protocol: honderd epochs met tien leerbare schaalkeuzes, initialisatie ×1

## Onderzoeksvraag en vooraf vastgelegde vergelijking

Onderzoek of de accuracy en schaalkeuzes over het volledige schema veranderen wanneer de tienkeuzeproef bij ×1 in plaats van ×64 begint. De nieuwe proef omvat honderd volledige epochs, opnieuw vanaf dezelfde oorspronkelijke FP32-/BN-toestand. Vergelijk elk gemeten epoch met hetzelfde epoch van de eerdere init×64-training van honderd epochs. Train- en testvolgorden worden per gemeenschappelijk epoch via hashes gecontroleerd. De exacte historische bronidentiteit staat in `init64_full100_reference/REFERENCE.json`. De vijf-epochpilot wordt niet als warmstart gebruikt.

De machineleesbare configuratie staat in `PROTOCOL.json`. `TRAINING_CHANGE_MANIFEST.json`, `REUSE_MANIFEST.json` en `provenance/*.patch` onderscheiden de nieuwe planning, ongewijzigde numerieke runtimebestanden en rapportagewijzigingen. De adapter is byte-identiek aan de geslaagde init×1-pilot.

## Schaal en bitbewerking

Per MX-groep bestaat één FP32-hulpparameter theta. De eerste waarde is nu nul, met bereik [0,9]. De harde forwardkeuze is de dichtstbijzijnde gehele offset; halve waarden kiezen de hogere offset:

`k = floor(theta) + (theta - floor(theta) >= 0.5)`

`S_E2M3 = S_E2M1 = S_E4M3 * 2**k`

De tien mogelijke factoren zijn 1,2,4,8,16,32,64,128,256,512. De E4M3-broncode en -schaal worden zoals voorheen uit de actuele gewichten bepaald. Beide E2-modi gebruiken één geleerde verhouding tot die bronschaal.

De exponentmapping blijft:

`s e3 e2 e1 e0 m2 m1 m0 -> s e3 e2 m2 m1 m0`

E2M1 ontstaat daarna uit de afgeleide E2M3-code door de twee laagste mantissabits te verwijderen (`code >> 2`), met exact dezelfde geleerde schaal. Er is geen extra afronding, biascompensatie of direct uit de master afgeleide E2-route.

## Leren

De gewichts-STE, preprocessing G(W) en gradient naar FP32-masters blijven behouden. Voor theta blijft de bestaande benadering gelden:

`dL/dtheta = ln(2) * sum_real_group(dL/dq * q)`

De bijdragen uit E2M3 en E2M1 worden opgeteld; E4M3 levert geen directe theta-gradient. De schaalgradient wordt niet genormaliseerd en niet door een grensmasker op nul gezet. Na iedere Adam-stap wordt theta geprojecteerd op [0,9]. De optimizer en de schaal-learningrate worden niet aan de nieuwe beginwaarde aangepast.

Alle drie precisies leren gedurende alle honderd epochs. E4M3 gebruikt hard CE, E2M3 soft CE van het losgekoppelde E4M3-resultaat en E2M1 soft CE van het losgekoppelde E2M3-resultaat. De drie gradients worden opgeteld vóór één Adam-stap. A32 en FP32-masters blijven gehandhaafd. De trainingscode gebruikt FP32-convoluties met numerieke MX-emulatie.

## Constante instellingen

| Instelling | Waarde |
|---|---|
| Dataset | SVHN train 73.257, test 26.032; geen extra |
| Model | CNN-8 `svhnq`, expand=2 |
| Groepsgrootte | 32 |
| Seed | 20261009 |
| Batchgrootte | 128, laatste onvolledige batch behouden |
| Optimizer | Adam, gewichts- en schaal-LR 0,001, weight decay 0 |
| LR-mijlpalen | 50/75/90, gamma 0,1; oorspronkelijke 100-epochplanning |
| Trainingpasses | E4M3 → E2M3 → E2M1; één optimizerstap |
| Evaluatie | Alle drie modi op de volledige officiële testset na elk epoch |
| Trainingseinde | Epoch 100; schalen leren alle epochs, geen freeze |
| Checkpointselectie | Eerste strikt hoogste E4M3-testaccuracy binnen de gemeten training |

## Controles en opslag

Het pakket behoudt de onafhankelijke codec- en gradientcontroles van de bestaande tienkeuzeruntime. Eveneens behouden zijn de theta0-initialisatiecontrole en een gerichte grenscontrole: Adam moet theta vanaf nul omhoog kunnen bewegen en een neerwaartse update moet naar nul worden geprojecteerd. In `check` wordt de RNG-toestand vóór de codeccontroles hersteld voordat de beginweights/BN met de historische referentie worden vergeleken.

De initiële basis-modelhash, zonder theta-parameters, is:

`a8fcd20c40a204ba24cabfa36258f9501d55b9d4d4bdfb8ba742b8c827af127f`

Per epoch blijven volledige checkpoints en een geschiedenis met gehele aantallen correcte voorspellingen bewaard. Export vereist dat de geschreven geschiedenis exact overeenkomt met die in het geheugen. Broncode, datasetbestanden en vooraf bestaande onderzoeksdata worden niet overschreven.

Compacte exportkosten blijven gelijk aan de eerdere tienkeuzevariant: vier bits keuze per groep, voor 4.080 groepen samen 2.040 bytes. De FP32-theta's en Adam-toestanden zijn trainingsinformatie en staan niet in de compacte bronexport. De beginfactor wijzigt de exportindeling niet.

## Vooraf bepaalde rapportage en grenzen aan conclusies

Rapporteer de volledige accuracy- en lossontwikkeling, de resultaten op epoch100 en op het binnen de training gekozen beste E4M3-checkpoint. Toon aantallen en aandelen voor alle tien factoren en de ontwikkeling van de continue theta's. Een theta kan bewegen zonder al een harde keuzedrempel te passeren.

De accuracies worden vergeleken met de oorspronkelijke init×64-training op dezelfde epochs; de E4M3-best checkpoints van twee runs mogen verschillende epochs zijn en worden daarom apart gelabeld. Een gedeeltelijke run toont alleen zijn werkelijk gemeten epochs; ontbrekende historische resultaten worden niet geëxtrapoleerd. De oorspronkelijke init×64-pilot en de actuele init×1-pilot blijven apart bewaard voor traceerbaarheid.

Een technische PASS vereist geldige numeriek/codec, volledige bedoelde dekking en consistente opslag; er geldt geen vereiste minimumaccuracy en geen vereiste factorswitch. De proef vergelijkt twee initialisaties bij één seed en één trainingsschema. Eventuele verschillen tonen geen statistisch bewezen algemene superioriteit of gegarandeerde convergentie. CUDA-randomness en checkpointselectie op de testset blijven methodologische beperkingen. Deze proef is bovendien nog steeds gekoppelde schaalkeuze, niet leren van twee onafhankelijke shared scales.
