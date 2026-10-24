# Vijf epochs — gezamenlijke route-B-training met vaste schaal ×64

Deze pilot traint de drie MX-gewichtsmodi gezamenlijk vanaf dezelfde verse
initialisatie als de eerdere proeven. De lagere modi gebruiken route B uit de
post-hocevaluatie: naïeve exponenttruncatie, met een vaste factor 64 op de
E4M3-bronschaal, gevolgd door de bestaande mantissatruncatie.

| Modus | Gewichtsroute | Schaal |
|---|---|---|
| E4M3 | Bestaande nearest-even kwantisatie van de actuele G(W) | Bestaande E4M3-bronschaal S |
| E2M3 | Hoogste twee exponentbits van dezelfde E4M3-code behouden | 64S: E8M0-broncode +6 |
| E2M1 | De afgeleide E2M3-code twee bits rechts schuiven | Dezelfde 64S als E2M3 |

Activaties blijven A32. De gedeelde updatemasters, gradients en Adam-momenten
blijven FP32. De BN-banken leren tijdens de gewone training per gewichtsmodus
mee en zijn tijdens evaluatie bevroren. De gradientbehandeling blijft dezelfde
identity STE door de complete route; er komt geen extra factor 64 in backward.

De bron wordt na masterupdates opnieuw bepaald. De verhouding64 tussen de
lagere schalen en de actuele E4-bronschaal blijft vast. Er is geen geleerde
schaal, alternatieve exponentmapping of afrondingsroute C in deze training.

## 1. Uitpakken en controleren

Pak de map uit de ZIP uit onder `software_checks`, zodat dit bestand bestaat:

`C:\Users\renzo\master-thesis-mx\software_checks\mx_triple_scale64_pilot5_v1\run_train.py`

Gebruik dezelfde `.venv-train` als voor de eerdere trainingen:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_pilot5_v1\run_train.py" check
```

De check controleert codecs, bitbewerkingen, schaaloffset, STE-gradients,
gedeelde broncache, recursieve loss, BN, Adam en opslag/herladen op synthetische
input. Er wordt geen SVHN-training gestart. Bij FAIL: deel eerst de nieuwe
`feedback_bundle.zip` uit de checkmap.

## 2. Na CHECK PASS: de vijf-epochtraining

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_pilot5_v1\run_train.py" run
```

De bestaande dataset staat standaard onder `data\svhn`. Als die elders staat,
voeg je `--data-root "PAD"` toe. De volledige train-split met 73.257 beelden en
test-split met 26.032 beelden worden gebruikt; de extra-split wordt niet geladen.
De optionele oude downloadmogelijkheid blijft beschikbaar via `--download`,
maar is voor de reeds aanwezige data niet nodig. Er worden geen pakketten
geïnstalleerd of geüpgraded.

De runner begint met seed 20261009 en dezelfde oorspronkelijke modelconstructie.
Er wordt geen eerder getraind checkpoint ingeladen voor de nieuwe proef.
Alleen het expliciete resumecommando kan een eigen, onderbroken route-B-pilot
hervatten, met controle van het volledige nieuwe contract.

Het LR-schema blijft dat van 100 epochs: Adam lr 0,001, geen weight decay,
dalingen na 50/75/90. De pilot stopt vast na epoch 5, dus alle vijf epochs
gebruiken lr 0,001. `--stop-after-epoch 100` wordt geweigerd.

Per minibatch volgt de oorspronkelijke volgorde: E4M3 met harde CE,
E2M3 met detached E4M3-softtargets, E2M1 met detached E2M3-softtargets;
daarna één Adam-update op de gesommeerde gradients. De drie passes delen
dezelfde actuele E4M3-elementcodes per laag. Er zijn geen afzonderlijke lagere
encoders of extra teacherpasses.

Een voltooide pilot bevat 2.865 updates en 8.595 trainingspasses. De vorige
vergelijkbare naïeve truncatiepilot duurde ongeveer 4 minuten en 38 seconden op deze laptop;
voor deze nieuwe route wordt de werkelijke tijd opnieuw gemeten. Een
projectie voor 100 epochs is alleen een tijdschatting, geen accuracyschatting.
De herkomst van die oude timing staat in `provenance/naive_pilot5_reference.json`.

## 3. Resultaten en notebook

Iedere start maakt een nieuwe map onder:

`mx_runs\mx_triple_scale64_pilot5\<tijdstip>_joint_run\`

Open daar `MX_triple_scale64_pilot5_resultaten.ipynb` en voer de cellen uit.
De resultaatmap is automatisch ingevuld. Er wordt één lege, uitvoerbare
notebook meegeleverd, met de eerdere CSV-parsercorrectie al geïntegreerd.

De notebook toont:

- de train- en testaccuracy per epoch voor E4M3, E2M3 en E2M1;
- de losses, met onderscheid tussen harde CE en softtargets tijdens training;
- de accuracies op **epoch 5** en op **E4M3-best binnen deze vijf epochs**;
- voorspelde klasseverdelingen en eventuele concentratie op één klasse;
- master-, code- en schaalwijzigingen en de bron-/BN-/exportcontroles;
- gemeten tijden, GPU-geheugen en een voorlopige looptijdprojectie.

De notebook meldt expliciet dat epoch 100 nog niet is gemeten. Bij een latere
volledige proef komen epoch 100 en E4M3-best weer als beide eindpunten terug.
Voor deze pilot worden de drie modi steeds uit hetzelfde checkpoint getoond.

Figuren verschijnen standaard in de notebook. Zet `SAVE_FIGURES = True` als
je PNG- en PDF-bestanden in een nieuwe analysemap wilt bewaren. Ruwe resultaten
blijven behouden. De notebook voert geen training of modelinferentie uit.

Bewaar de volledige nieuwe runmap en stuur na afloop haar `feedback_bundle.zip`.
Die bevat rapporten, tabellen, trainingsgeschiedenis, klassentellingen en de
lege notebook met helper. Trainingscheckpoints en datasets blijven lokaal.

## Stoppen en hervatten

Eén keer `Ctrl+C` vraagt om de huidige epoch volledig af te werken en op te
slaan. Een tweede keer stopt onmiddellijk; alleen volledige checkpoints zijn
hervatbaar. Een bestand `STOP_AFTER_EPOCH` in de actieve resultaatmap vraagt
ook om een stop op de volgende epochgrens.

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\mx_triple_scale64_pilot5_v1\run_train.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_CHECKPOINT.pt"
```

Gebruik een volledig trainingscheckpoint uit deze route-B-pilot, bijvoorbeeld
`last_checkpoint.pt` uit een gecontroleerd gestopte run of
`checkpoints\epoch_00002.pt`, met zijn checksum. De compacte MX-exports zijn
voor inferentie. Een checkpoint uit de vorige naïeve of directe training
heeft een ander contract en wordt geweigerd. Hervatten schrijft in een
nieuwe map en gaat uiterlijk door tot epoch 5. De oorspronkelijke CUDA-backend
geeft geen bitexacte herhaalgarantie.

## Beoordeling van deze pilot

De proef onderzoekt of training zich aan route B kan aanpassen. De lage
post-hocaccuracy geeft daarop nog geen antwoord. Lage accuracy, een constante
klasse of eindige nulgradients worden als meetresultaten bewaard. NaN/Inf,
verkeerde optimizerstappen, gewijzigde bronrelaties of een ongeldige schaal
geven een technische FAIL; er worden geen instellingen stilzwijgend aangepast.

De factor 64 is vooraf vastgelegd. Het pakket zoekt geen betere factor en leert
geen schaal. We bekijken de vijf-epochresultaten voordat we een volledige
honderd-epochtraining beoordelen.

## Documentatie

`PROTOCOL.md` legt de exacte bewerking, gradients en vergelijkingsvoorwaarden
vast. `CODEWIJZIGINGEN.md` beschrijft het hergebruik.
`local_validation/VALIDATION.md` beschrijft de uitgevoerde softwarecontroles.
Bij de voorbereiding is uitsluitend op kleine synthetische CPU-fixtures getest;
de echte SVHN/CUDA-training wordt met bovenstaande commando's lokaal uitgevoerd.
