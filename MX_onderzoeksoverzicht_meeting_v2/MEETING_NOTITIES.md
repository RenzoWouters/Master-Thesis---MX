# Spreeklijn voor de meeting - overzicht v2

## Hoofdverhaal (secties 1-8)
Het doel is een compacte bewaarde MX-bron voor meerdere uitvoeringsprecisies, niet per se
hogere accuracy dan alle afzonderlijke modellen. Alle hoofdresultaten zijn CNN-8, SVHN
train-only en het vaste eindpunt epoch 100. Beste checkpoints staan apart in bijlage D.

1. Dedicated referenties bestaan nu op W32/A32, E2M3/A32 en E2M1/A32. De nieuwe
   dedicated E2M1-run is volledig: 100 epochs, 57.300 updates, 94,8218% op epoch 100.
   Beste E2M1-checkpoint: epoch 97, 94,9063%. Sessieduur ongeveer 44 min 13 s.
2. Gezamenlijke directe training bereikt 95,0254% op E2M1. Dat model achteraf trunceren
   geeft 93,5810%. Truncatiebewust trainen bereikt 94,9408% met echte truncatie.
3. De dedicated E2M1-referentie is in dit trainingspaar 0,2036 procentpunt lager dan
   gezamenlijk direct en 0,1191 procentpunt lager dan gezamenlijk truncatiebewust.
   Niet zeggen dat dedicated of any-precision altijd beter is: een seed, andere loss en passes.
4. E2M3 op epoch 100: dedicated 94,8256%; gezamenlijk direct 95,0330%; gezamenlijk
   truncatiebewust 94,9639%. De oudere Any-Precision-bron alleen naar E2M3 converteren
   geeft 94,7296%; dat is geen E2M3-bewuste training. Nieuwe figuur in sectie 2b.
5. Opslag: vaste E2M1-kopie inclusief alle behouden bijgegevens 88.608 bytes;
   hoge E2M3-bron voor twee modi 121.248 bytes; twee directe codesets met gemeenschappelijke
   gegevens 186.528 bytes; FP32-modelinhoud 537.136 bytes. Eenheden: ruwe payload,
   geen fysieke checkpointgrootte, totaal trainingsgeheugen, VRAM of energie.
6. Het specifieke truncatievoordeel is een lage uitvoering uit de hoge codes, zonder
   aparte lage codeset of FP32-herkwantisatie. De hoge bron blijft behouden voor terugschakelen.

## Open punten eerlijk benoemen
Geen volledige nieuwe export-inferentietest van de laatste truncatiebewuste exports,
geen herhalingen over seeds, geen geisoleerde BN-ablaties, geen volledige train+extra-
bevestiging van de MX-methode en geen continual-learning- of hardwarebenchmark.
De bekende deprecationwaarschuwingen zijn gedocumenteerd; de rapportcontroles slagen.
De echte gewichtsexports/checkpoints zijn niet in de feedbackbundels aangeleverd.

## Waar tonen?
Sectie 2: volledige referentietabel; 2b: E2M3-figuur; 2c en 3: nieuwe vierbitsvergelijking;
4: curves/CE; 5: bits; 6: vier opslagstrategieen; 7: accuracy versus actieve payload;
bijlage D: beste/laatste; bijlage F: herkomst en controles. Pilots staan apart.
