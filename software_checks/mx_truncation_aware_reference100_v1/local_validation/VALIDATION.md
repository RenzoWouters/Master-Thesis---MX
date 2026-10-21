# Uitgevoerde validatie: truncatiebewuste honderd-epochwrapper

## Omgeving en afbakening
Linux-container, Python3.13.5, torch2.10.0+cpu, torchvision0.25.0+cpu,
NumPy2.3.5. Geen CUDA beschikbaar. De laptop gebruikt een andere OS/backend.
Alle modeltests hieronder zijn SYNTHETISCH:6 trainvoorbeelden en4
controlevoorbeelden van40x40, batch4. Geen echte SVHN-data, geen
onderzoeksaccuracies en geen echte honderd-epoch-GPU-training.
In de testharness alleen wordt Tensor.cuda tijdelijk identiteit; de productie-
CLI biedt geen CPU/synthetische bypass. Productietoleranties zijn ongewijzigd.

## Geslaagd in deze bouw
- Werkelijke pilot-ZIP geverifieerd tegen alle49 pakkethashes uit de geslaagde
  gebruikersfeedback. De numerieke modules en run_epochs zijn byte-identiek.
- Honderd mini-epochs met echte CNN-8 en hoge-brontruncatie: ongeregistreerde
  oorspronkelijke trainingsfunctie versus volledige runner. Exact dezelfde
  modelparameters, Adam/scheduler, CPU/Python/NumPy-RNG en alle epochmetrics.
- Hervatten na50,75,90 naar100: exact dezelfde eindstaat, losses, histories,
  samplevolgordehashes, BN- en bron/gradientcontroles op CPU.
- Werkelijke LR-overgangen50/51,75/76,90/91 getest tijdens de modeltraining.
- Alle100 checkpoints,400 metricrijen,200 samplevolgordebestanden en correcte
  completion_summary.json; best op hoge modus, first-strict bij gelijkheid.
- Foutieve volgende LR, ontbrekende modus, foute bron/schaalaudit worden afgewezen.
- Gewijzigde modi, pilotcontract, corrupte checksum en overschrijven geweigerd.
- Stopbestand werkt pas na afronden van de lopende epoch; een onvolledige run
  krijgt full_training_completed=false, niet een fictief volledig resultaat.
- Bestaande E2M3-,E2M1- en truncatie/STE-oracles en verbeterde zelftest behouden.
- Hoge export opnieuw in vers proces geladen: beide modi reconstrueren zonder
  zes masters, zonder lage encoder en zonder toegang tot checkpoint/lage export.
  Vier synthetische logitvergelijkingen met de getrainde modellen exact gelijk.
- Notebook uitgevoerd op opgeslagen100-epoch- en gedeeltelijke testgegevens;
  geen training vanuit notebook. Accuracy- en LR-figuren visueel nagekeken.
- CLI wijst --stop-after-epoch5, run-met-checkpoint en resume-zonder-checkpoint af.

De eerste containerpoging stopte door de tijdslimiet van de uitvoertool nadat
83 van de100 runner-mini-epochs waren opgeslagen. Dat was geen methodefout
of afgeronde validatie. Een volledig nieuwe tweede uitvoering slaagde voor
alle bovenstaande tests. De onvoltooide poging is niet als PASS geteld.

## Productiepreflight in deze container
De echte check-opdracht werd uitgevoerd en stopte bij pip check:
MoviePy2.2.1 vereist Pillow<12; de container heeft Pillow12.3.0.
Ook verscheen de pip-waarschuwing over een niet-schrijfbare cachemap.
Geen pakketten werden geupgraded/verwijderd en geen controle werd omzeild.
Zie container_preflight_report.json en container_pip_check.txt. Dit zijn GEEN
fouten gemeten in de gebruikersomgeving. De check op de laptop moet slagen
voordat de echte run gestart wordt; er is hier geen CUDA-PASS geclaimd.

## Herhalen van alleen de offline regressietest
Vanuit de pakketmap, met benodigde CPU-testomgeving:
    python local_validation/test_training.py NIEUWE_VALIDATIEMAP
    python local_validation/test_source_only.py NIEUWE_VALIDATIEMAP NIEUWE_BRONTESTMAP
Geen bestaande onderzoeksmap als testuitvoer gebruiken.

## Beperkingen
Een CPU-hervattest bewijst geen bitexacte CUDA-herhaling. Echte run/resume
behouden cudnn.benchmark=True. De hoge/afgeleide exports van de echte run
worden op gewichtsniveau gecontroleerd, maar niet automatisch nogmaals op
alle testbeelden uitgevoerd. Geen native snelheid/energie- of CL-claim.
