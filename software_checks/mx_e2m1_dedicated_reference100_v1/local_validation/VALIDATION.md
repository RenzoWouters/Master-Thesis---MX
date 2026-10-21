# Validatie dedicated E2M1/A32

Werkelijk uitgevoerd in Linux/Python3.13.5, PyTorch2.10.0+cpu, torchvision0.25.0+cpu.
Geen SVHN-download, echte SVHN-training, GPU-training of onderzoeksaccuracy uitgevoerd.

## Geslaagde numerieke en trainingscontroles
- De E2M1-codec is byte-identiek aan de al aangeleverde directe tweemoduscode.
- Alle16 codes inclusief signed zero; onafhankelijke waarde-oracle met 4143 testwaarden;
  alle midpoint/tie-buren en finite schaalcodes, padding, nulblokken, saturatie,
  ongeldige NaN/Inf-ingangen en packing/unpacking. Exact aantal in validation_report.json.
- Directe E2M1-conversie van1.375 bij schaal1 geeft1.5, geen verborgen E2M3-truncatie1.0.
- De wrapper zonder MX stemt overeen met de oorspronkelijke W32-voorbewerking.
- Identity-STE en gradient door G(W) getoetst; FP32-masters/momenten; alleen BN32 leert.
- Honderd mini-epochs dedicated E2M1: zes kunstmatige trainbeelden, vier testbeelden,
  twee optimizerupdates per epoch. Echte CNN-8 en oorspronkelijke trainingsfunctie gebruikt.
- Ongeregistreerde originele functie versus geregistreerde runner: exact gelijke
  modelstaat, optimizer, scheduler, RNG en alle metrics na honderd CPU-mini-epochs.
- run_epochs is AST-identiek aan single_reference100; data/hervatkern byte-identiek.
- Honderd checkpoints, best/last, LR50/75/90, volledige versus onderbroken status correct.
- Resume na50/75/90/99: volgende CPU-epoch exact gelijk met ononderbroken uitvoering.
- Afwijkende checkpointcontracten, corrupte bestanden, overschrijven geweigerd.
- Dedicated E2M1-bron zonder de zes FP32-masters herladen met juiste gewichten.
- Dedicated notebook volledig uitgevoerd op synthetische rapporten, geen training daarin.
- Meetingintegratie volledig uitgevoerd met expliciet synthetisch datapunt; originele
  overzichtsbronnen ongewijzigd. Productie weigert synthetische of onvoltooide resultaten.
  De nieuwe notebook bevat tabellen, leercurves, opslag/accuracypunt en verduidelijking sectie6.

## Productiepreflight
Een echte aanroep van run_train.py check stopte correct bij het bestaande containerconflict:
MoviePy2.2.1 vereist Pillow<12; hier is Pillow12.3.0 aanwezig. Er is niets geinstalleerd
of aangepast om die bewaking te omzeilen. Zie container_pip_check.txt en
container_run_report.json. Dit is NIET de Windows-omgeving van de gebruiker.
De CUDA-check moet op de laptop slagen voordat de honderd echte epochs beginnen.

## Ontwikkelafbakening
Een eerste brede test met control+MX werd door de tooltijdslimiet onderbroken na de
controlchecks; de volledige dedicated E2M1-test is vervolgens afzonderlijk voltooid.
De eerste integratietest miste in de synthetische fixture een productieprovenancebestand;
de fixture-opbouw is aangevuld. Bij de notebooktest is een ontbrekend cell-id-waarschuwingspunt
opgelost. Geen van deze aanpassingen verandert de numerieke trainingslus of onderzoeksdata.

Herhalen CPU (alleen testscript heeft een Tensor.cuda-mock; productie heeft geen CPU/subsetmodus):

    python local_validation/test_reference100.py <NIEUWE_map>

Meetingintegratie testen met die synthetische run (niet de echte onderzoeksdata):

    python local_validation/test_integration.py <NIEUWE_testmap> <synthetische_runmap/mx_run> <uitgepakt_meeting_v1>

Twee synthetische updates per controlemodus testen alleen correctheid, niet eindaccuracy.
Vaste seed/correcte hervatstaat beloven geen bitexacte CUDA-uitvoering.
