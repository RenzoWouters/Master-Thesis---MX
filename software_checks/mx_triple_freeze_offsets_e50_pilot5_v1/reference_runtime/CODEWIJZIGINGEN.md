# Wijzigingen tegenover de gevalideerde geleerde-schaalpilot

De numerieke route en trainingswiskunde blijven byte-identiek aan het vijf-epochpakket. Dit geldt voor `trunc_adapter.py`, `trunc_support.py`, de codecs, oorspronkelijke netwerkonderdelen, lossfuncties en schedulerimplementatie. De expliciete schaalgradient, drempel, beginwaarde, projectie, beide Adam-groepen en leersnelheden veranderen niet.

| Bestand | Wijziging en doel |
|---|---|
| `run_train.py` | Nieuw experimentcontract en resultatenpad; standaard honderd epochs; gedeeltelijke stops 1–100; volledige completion alleen bij100; timing en verwachte updatetellingen voor100; bewaren van de eerdere pilotreferentie. |
| `notebook_support.py` en notebook | Analyse van honderd epochs, accuracygrafieken voor epoch100 en het gezamenlijke E4M3-bestcheckpoint, vergelijkingen met beide vaste volledige referenties, volledige schaalhistorie en juiste gedeeltelijke status. |
| Documentatie en manifests | Protocol, uitvoeropdrachten, herkomst van de pilot en expliciete vergelijking van alle configuratievelden. |

`CONFIG_DIFFERENCES.json` benoemt uitsluitend de wijzigingen in schema, doel, eindepoch en rapportagetekst. Alle andere configuratievelden zijn gelijk. `REUSE_MANIFEST.json` legt de byte-identieke runtimebestanden vast. De oude pilotrunner, documentatie en validatierapporten zijn als provenance opgenomen.

De nieuwe lokale integratiecontrole richt zich op wat deze verlenging nieuw bereikbaar maakt: de drie LR-overgangen voor beide optimizerparametergroepen, hervatten over die grenzen en volledige honderd-epochrapportage. Er wordt geen nieuwe SVHN-accuracy door de lokale synthetische tests geclaimd.
