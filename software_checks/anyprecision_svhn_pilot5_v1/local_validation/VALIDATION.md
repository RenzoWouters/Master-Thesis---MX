# Uitgevoerde offline validatie

Omgeving: Python 3.13.5, torch 2.10.0+cpu, torchvision 0.25.0+cpu,
NumPy 2.3.5, SciPy 1.17.0. Andere OS/backend/Python-patchversie dan de laptop.
Dit zijn UITSLUITEND kleine synthetische softwaretests; geen echte SVHN-data,
geen GPU-training, geen onderzoeksaccuracies en geen bruikbare GPU-tijdschatting.

Geslaagd:

- Vijf mini-epochs met de echte CNN-8 en de originele forward, vergeleken met
  dezelfde functie plus de nieuwe registratie. Exacte gelijkheid van alle
  modelparameters, optimizer, scheduler, CPU/Python/NumPy-RNG en epochmetrics.
- Vijf doorlopende mini-epochs versus twee plus hervatten naar vijf, inclusief
  dropout, samplevolgorde, per-precisie BatchNorm en selectiegeschiedenis.
- Beste en laatste checkpointkopie: checksum, epoch, metrics en modelhash.
- Afwijzen van gewijzigde contracten, corrupte checkpoints en overschrijven.
- Stopbestand werkt pas na het afronden en bewaren van de lopende epoch.
- Originele SVHNMIX-loader op kunstmatige .mat-bestanden: train-extra-volgorde,
  10-naar-0-labelmapping en 40x40-preprocessing.
- Productiedataguard wijst een subset af.
- Gemockte downloader: staging/MD5/bestaande corrupte data niet overschrijven.
- Checkpointselectie: strikt hoogste W32/A32, eerste bij gelijke score, niet de
  piek van een afzonderlijke lagere precisie.
- Oorspronkelijke scheduler: 50/75/90 blijven behouden, niet naar pilot geschaald.
- Notebook uitgevoerd op opgeslagen synthetische rapporten, zonder training.
- Bron- en hergebruikhashes ongewijzigd.

De productie-CLI staat geen synthetische of CPU-bypass toe. Alleen de
validatiescript vervangt tijdelijk Tensor.cuda door een identiteit voor CPU.
`test_svhn.py` schrijft uitsluitend eigen nieuwe validatiemappen. Herhaal:

    python local_validation/test_svhn.py <NIEUWE_map_voor_synthetische_test>

Bij echte uitvoering blijft run_report.json leidend. Een PASS van deze tests is
geen belofte dat datasetservers bereikbaar zijn of dat CUDA-kernels op een
ander apparaat bitexact dezelfde resultaten geven.

## Volledige CLI-preflight in deze container

Een echte aanroep van `run_svhn.py check` werd geblokkeerd door
`pip check`: de reeds aanwezige MoviePy 2.2.1 vraagt Pillow <12,
terwijl deze container Pillow 12.3.0 bevat. Er zijn geen pakketten
gewijzigd om die controle te omzeilen. De foutmelding en het rapport
staan naast dit document. Dit is NIET de omgeving van de gebruiker.
De CPU-functietests hierboven konden afzonderlijk wel uitgevoerd
worden. De productiepreflight moet dus op de laptop slagen.
