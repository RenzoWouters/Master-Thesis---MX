# Post-hoc exponent- én mantissatruncatie op dezelfde checkpoints

Deze proef evalueert de bestaande gezamenlijke directe E4M3/E2M3/E2M1-referentie
op twee vaste checkpoints: **epoch 100** en **epoch 54**, het al geselecteerde
beste checkpoint volgens E4M3-testaccuracy. De keuze wordt niet opnieuw gemaakt
op basis van de post-hocresultaten.

De runner gebruikt de zes al opgeslagen directe MX-exports van die twee
checkpoints. Hij controleert hun identiteit en checkpointprovenance. Afgeleide
modellen komen uitsluitend uit hun elementcodes, ongewijzigde E8M0-schaalbytes
en overige modelstaat. De zes MX-mastergewichten worden niet opnieuw ingelezen.
Activaties blijven A32. De bestaande training en FP32-updatemasters blijven
behouden; deze proef doet uitsluitend evaluatie.

## Uitpakken en controleren

Pak de map uit deze ZIP uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_posthoc_truncation_eval_v1\run_eval.py`

Gebruik dezelfde `.venv-train` als voor de volledige training:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_posthoc_truncation_eval_v1\run_eval.py" check
```

`check` test de bitbewerkingen, signed zero, vaste schaalbytes, packing,
opslag/herladen en bevroren modellen op synthetische input. Het gebruikt geen
SVHN-data en voert geen training of optimizerupdate uit. Bij een FAIL: deel de
nieuwe feedbackbundel uit de checkmap voordat je de evaluatie start.

## Na CHECK PASS: beide checkpoints evalueren

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_posthoc_truncation_eval_v1\run_eval.py" evaluate --reference-run ".\mx_runs\mx_weights_triple_direct_reference100\20261023_151331_732669_joint_run"
```

Het argument wijst naar de volledige oorspronkelijke runmap, met rapporten,
zes directe exports en hun `.sha256`-bestanden. Alleen een uitgepakte
feedbackbundel is onvoldoende: daarin ontbreken de modelbestanden. Bij een
verplaatste runmap pas je alleen `--reference-run` aan.

Standaard gebruikt de proef `data\svhn\test_32x32.mat`. Als de dataset elders
staat, voeg je `--data-root "PAD"` toe. Alleen de officiële testset met 26.032
beelden wordt gelezen, in dezelfde volgorde en met dezelfde preprocessing als
bij de training. Er zijn geen downloads of nieuwe laptopafhankelijkheden nodig.
De bestaande omgeving en backendinstellingen worden gecontroleerd.

## Acht paden, telkens op hetzelfde checkpoint

| Route-ID | Gewichten | Doel-BN | Betekenis |
|---|---|---:|---|
| direct8 | Direct E4M3 | 8 | Opgeslagen directe referentie opnieuw evalueren |
| direct6 | Direct E2M3 | 32 | Opgeslagen directe referentie opnieuw evalueren |
| direct4 | Direct E2M1 | 4 | Opgeslagen directe referentie opnieuw evalueren |
| round64 | E2M3 naar E2M1 | 4 | Afrondingscontrole, oorspronkelijke E2M3-schaal |
| mantissa64 | E2M3 naar E2M1 | 4 | Mantissacode twee plaatsen rechts schuiven |
| round86 | E4M3 naar E2M3 | 32 | Afrondingscontrole, oorspronkelijke E4M3-schaal |
| exponent86 | E4M3 naar E2M3 | 32 | Alleen de hoogste twee exponentbits behouden |
| combined864 | E4M3 naar E2M3 naar E2M1 | 4 | Eerst exponenttruncatie, daarna mantissatruncatie |

De BN-sleutels zijn opslag-ID's; alle activaties blijven A32. Voor iedere
vergelijking gebruikt de afgeleide route exact dezelfde bestaande BN-bank
als de directe doelprecisie. BN-statistieken en parameters blijven bevroren;
er is geen herkalibratie of training.

De twee afrondingscontroles gebruiken nearest rounding met ties-to-even op
het lokale, uit de MX-bron gedecodeerde element. Ze kiezen geen nieuwe schaal.
De oorspronkelijke E4M3-schaal is doorgaans kleiner dan de directe E2-schaal.
Daarom kan `round86` sterk satureren: het is expliciet een controle met vaste
E4M3-bronschaal, geen gecompenseerde conversie. Zie `PROTOCOL.md` voor de
exacte bitindeling en de interpretatie.

Voor deze proef worden de directe modi ook opnieuw gemeten. Het verschil
tegenover de historische accuracies wordt apart gerapporteerd. Truncatie wordt
vergeleken met de in deze proef gemeten directe doelmodus op dezelfde beelden.
Een lage truncatieaccuracy is een onderzoeksuitkomst; `PASS` betekent dat de
vastgelegde evaluatie en integriteitscontroles technisch zijn afgerond.

## Resultaten en notebook

Nieuwe resultaten staan onder:

`mx_runs\mx_triple_posthoc_truncation_eval\<tijdstip>_evaluate\`

Open daar `MX_triple_posthoc_truncatie_resultaten.ipynb` met de bestaande
`.venv-train`-kernel en voer alle cellen uit. De resultaatmap is automatisch
ingevuld. Het pakket bevat uitsluitend de lege, uitvoerbare notebook.

De notebook bevat:

- accuracyfiguren voor alle acht routes op epoch 100 en op E4M3-best;
- accuracyverschillen tegenover de directe doelprecisie;
- gepaarde fout→correct- en correct→fouttellingen per testbeeld;
- gewichtsafwijkingen en RMS-grootteverhoudingen per laag;
- schaal-, nul-, saturatie- en integriteitsdiagnostiek, plus tijden.

Figuren opslaan maakt een nieuwe analysemap. De notebook verandert geen ruwe
resultaten of modellen en start geen training. De helper leest geen torchmodel.

Bewaar de volledige nieuwe resultaatmap. Stuur daarna de `feedback_bundle.zip`
uit de `_evaluate`-map. Die bevat ook geordende logits, voorspellingen en
per-beeld-CE in twee NPZ-bestanden, zodat de vergelijkingen controleerbaar zijn.
De bundel is daardoor groter dan een trainingsfeedbackbundel. Modelbestanden
en datasets blijven buiten de bundel.

## Stoppen

Dit is een eenmalige evaluatie, geen training met epochs. `Ctrl+C` onderbreekt
de proef en bewaart een diagnostisch feedbackrapport. Het pakket heeft geen
resumecommando. Opnieuw `evaluate` starten maakt een nieuwe resultaatmap en
voert de test opnieuw uit; alle eerdere resultaten blijven bestaan.

## Uitgangspunt: volledige directe referentie

De gecontroleerde training heeft status PASS: honderd epochs, 57.300 updates
en 171.900 trainingspasses. De sessie duurde 1u42m19s.

| Gewichten | Epoch 54 — E4M3-best | Epoch 100 |
|---|---:|---:|
| E4M3/A32 | 95,0369% | 94,8602% |
| E2M3/A32 | 95,1137% | 95,0061% |
| E2M1/A32 | 95,1368% | 95,0715% |

Bron: `feedback_bundle(20261023-150530).zip`; de onafhankelijke audit staat in
`provenance/reference100_feedback_audit.json`. De bewaarde lagere precisies
scoren hier iets hoger op test, terwijl E4M3 de hoogste gerapporteerde
trainaccuracy heeft. Dit is verenigbaar met verschillende generalisatie door
gezamenlijke optimalisatie, soft targets en precisiespecifieke BN; het stelt
de oorzaak niet vast. E4M3 en E2M3 hebben beide drie mantissabits. De twee
extra bits zijn exponentbits, met een andere directe MX-schaalregel.

## Documentatie

`PROTOCOL.md` beschrijft de vaste proefopzet en exacte bitbewerkingen.
`CODEWIJZIGINGEN.md` beschrijft het hergebruik en de nieuwe modules.
`local_validation/VALIDATION.md` beschrijft de uitgevoerde softwarecontroles.
Er is bij het voorbereiden van dit pakket geen echte SVHN-post-hocevaluatie
of nieuwe training uitgevoerd; de echte exports staan op jouw laptop.
