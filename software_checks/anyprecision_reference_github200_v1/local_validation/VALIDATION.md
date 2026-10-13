# Lokale validatie van GitHub200 v1

## Uitgevoerd in deze chatomgeving

Linux/CPU, Python 3.13.5, torch 2.10.0+cpu, torchvision 0.25.0+cpu,
NumPy 2.3.5, Pillow 12.3.0. Geen CUDA-apparaat beschikbaar.

- Configuratie en trainingsepochs vergeleken met het aangeleverde train_cifar10.sh:
  200 epochs, LR100/150/180, Adam, LR0.001, weight decay0 en batch128-default.
- De ingesloten oorspronkelijke ZIP en alle hergebruikte Python-modules zijn
  byte-identiek aan de eerder gebruikte bron/runnerpakketten.
- Nieuwe runner-integratie: ononderbroken mini-training tegenover twee afzonderlijke
  processen met opslaan, stoppen en hervatten. Exacte gelijke gehele checkpointinhoud,
  model/optimizer/scheduler/RNG/geschiedenis en CSV-metrics.
- Hervatten vanaf initialisatiecheckpoint epoch0 geeft dezelfde uitkomst.
- Stopverzoek tijdens de eerste mini-epoch stopt na het opslaan van epoch1 en
  verandert de checkpointinhoud niet.
- Checkpoints weigeren overschrijven, corrupte inhoud en afwijkende configuratie.
- De volledige 200-epoch learning-ratereeks is apart gecontroleerd met een
  enkel scalair parameterobject en 200 optimizer-/schedulerstappen; geen netwerktraining.
- Eindtestevaluatie getest met twintig synthetische controlebeelden. Dit zijn geen
  echte CIFAR-resultaten; de vergelijking tegen de README is alleen een functietest.
- Rapportagereferentie gecontroleerd tegen de README; verschil in procentpunten
  gecontroleerd met verzonnen unit-testwaarden. Onvolledige precisies, trainingsmetrics
  en gemengde checkpoint-epochs worden afgewezen.
- De productie-CLI weigert de kunstmatige CIFAR-bestanden op checksumcontrole.
  De FAIL in production_data_guard.log is dus de verwachte negatieve test.

Alle controles in integration_result.json, stop_guard_result.json en
alignment_reporting_result.json zijn PASS. Er zijn geen Python-waarschuwingen
in de geslaagde mini-trainingen. De shell-LF-regel en toelichting zijn na deze
numerieke tests toegevoegd; de geteste runner/core/config/targets zijn onveranderd.

## Belangrijke grenzen

De mini-integratietest gebruikt een afzonderlijke testshim, synthetische bestanden
in CIFAR-formaat, twintig trainingsbeelden, drie mini-epochs, batch8 en kunstmatige
LR-milestones1/2. Deze instellingen komen NIET in de normale productie-CLI terecht.
De feitelijke configuratie met 50.000 echte beelden, batch128, 200 epochs is niet
hier uitgevoerd, evenmin een Windows/CUDA-run van deze nieuwe runner.

De eerdere hervatcontrole is door de gebruiker op zijn Windows/NVIDIA-laptop
uitgevoerd; de bijbehorende kernmodules worden hier ongewijzigd hergebruikt.
De eerste vijf echte epochs op die laptop zijn nog uit te voeren. Deze lokale
validatie bewijst geen reproduceerbare GitHub-accuracy, volledige datasetperformance,
betrouwbare trainingstijd op een GPU of stroomuitvalduurzaamheid.

## Reproduceerbare lokale checks (niet nogmaals nodig als laptop-pilot)

Gebruik drie NIEUWE testbestemmingen, geen onderzoeksmap met bestaande data:

```bash
python -B local_validation/test_integration.py --work-root /tmp/github200_validation_new
python -B local_validation/test_stop_and_guards.py --work-root /tmp/github200_validation_new
python -B local_validation/test_alignment_and_reporting.py --output /tmp/github200_alignment_new.json
```

De eerste test maakt eigen kunstmatige data en controleert die op ongewijzigde
inhoud. De publieke train-opdracht downloadt en installeert niets.
