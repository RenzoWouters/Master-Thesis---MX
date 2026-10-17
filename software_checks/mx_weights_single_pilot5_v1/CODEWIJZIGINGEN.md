# Wat verandert er in de code?

## 1. De originele training heeft al een eenmoduspad

`train.py::forward` maakt een bitlijst uit `args.bit_width_list`. Eerst traint hij
het hoogste element met hard-label CE, daarna pas de lagere elementen met distillatie.
Met `bit_width_list='32'` blijft alleen dat eerste pad over. De lower-bit-loop is
leeg. De oorspronkelijke softmax voor ongebruikte soft targets blijft als kleine
ongebruikte bewerking aanwezig; er is geen tweede forward, loss of gradient.

`single_support.load_single_forward` hergebruikt de originele functie via
`svhn_support.load_forward` en verandert uitsluitend het argument:

```python
forward, digest = s.load_forward(source, utils, log, meter_class)
forward.__globals__['args'] = SimpleNamespace(bit_width_list='32', print_freq=20)
```

De functietekst en AST blijven identiek. **32 is hier de interne selector voor
A32, BN32 en de bestaande W32-voorbewerking; het is NIET het MX-gewichtsformaat.**
Dat laatste wordt bepaald door de reeds geteste MX-adapter. Niet `bits=[6]` invullen:
dat zou opnieuw het oude gekoppelde W/A-precisiepad aanspreken.

## 2. Bestaande MX-code wordt hergebruikt, niet herschreven

`single_support.build_model` maakt dezelfde CNN-8 en vervangt in een kopie de zes
quantize_fn-modules via `mx_adapter.adapt_copy(enabled=True)`.
`mx_format.py` en `mx_adapter.py` zijn byte-identiek aan stap1.

Conceptueel gebeurt per minibatch:

```python
optimizer.zero_grad()
logits = model(images)   # iedere doellaag: FP32 W -> G(W) -> MX -> FP32 conv
loss = CrossEntropyLoss(logits, labels)
loss.backward()         # identity-STE door MX; overige derivatives blijven
optimizer.step()        # Adam past de waarden van de FP32-master aan
```

Het formaat van W blijft FP32; W is niet bevroren. De volgende forward encodeert
de nieuwe G(W). De schaal kan daarbij ook veranderen. Er is geen cache van
verouderde gedecodeerde gewichten tijdens training. FrozenMXConv wordt NIET als
trainbare laag ingezet: die is uitsluitend voor export/reconstructie bedoeld.

## 3. Wat is nieuw aan de runner?

`run_train.py` is afgeleid van de honderd-epoch-referentierunner; de volledige diff
staat in `research_changes.patch`. Nieuwe configuratie, vijf-epochstop en rapportage.
De datafuncties, sample-tracker, originele forward-loader en checkpoint-/RNG-kern
zijn identiek hergebruikt. Kernwijzigingen:

- `t.phase` kiest een modus; `SingleMeterRecorder` registreert drie AverageMeters
  (loss/top1/top5) in plaats van vijftien, met dezelfde upstream berekening.
- `ForbiddenDistillation` werpt een fout als toch een distillatieloss aangeroepen wordt.
- Voor elke Adam-step controleert een niet-mutating pre-hook de actieve gradients.
  De hook retourneert niets en verandert geen gradients, loss of optimizerarguments.
- Inactieve BN-banken mogen nu geen gradients of Adamstaat krijgen. BN32 verwacht
  precies een update per minibatch, andere banken nul. Tijdens training is BN32
  dus niet bevroren; tijdens test wel.
- Per epoch worden master- en codewijzigingen gelogd, zonder grad/RNG-mutatie.
- Beste checkpoint wordt geselecteerd op de actieve modus (MX of aparte controle),
  niet op een extra verborgen W32-evaluatie van de MX-run.
- Control-mode is dezelfde eenmoduslus met enabled=False, niet de oude vijfmodetraining.

## 4. Hergebruik en compatibiliteit

`REUSE_MANIFEST.json` legt hashes vast. `resume_core.py`, `smoke_support.py`,
`svhn_support.py`, de auteursbron-ZIP, MX-codec en MX-adapter zijn ongewijzigd.
`svhn_support` bevat nog de oude vijfmodusfuncties, maar `run_train.py` gebruikt
voor fase/metrics/BN/selection expliciet de nieuwe `single_support`-functies.
Geen globale wijziging van `s.BITS`: die lijst blijft alleen de modelbankstructuur.

Alleen de al bestaande utils.py-compatibiliteitspatch wordt op een nieuwe werkkopie
aangebracht (gpustat-import uitstellen en reshape voor accuracy). Er zijn geen
nieuwe bron-, package- of backendupgrades. De huidige compatibiliteit staat in
`provenance/reference_COMPATIBILITY.md`; historische details zijn geen nieuwe ingreep.

## 5. Grenzen

Dit test learning met MX-uitvoeringsgewichten en FP32-updatemasters. Geen masterloze
MX-training, geen meerdere MX-modi, geen truncatie, geen activatieaanpassing.
Een STE is een gradientbenadering. De codec is leesbaar maar niet voor doorvoer
geoptimaliseerd; train-tijd wordt opnieuw gemeten in plaats van als 1/5 voorspeld.
