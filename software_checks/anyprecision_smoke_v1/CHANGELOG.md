# Wijzigingslog - Any-Precision technische uitvoertest v1

## A. Oorspronkelijke bronnen

Het meegeleverde ZIP-archief is ongewijzigd. Een run pakt het opnieuw uit in een
nieuwe uitvoermap, verifieert alle SHA-256-controlegetallen en past alleen die
nieuwe werkkopie aan. De originele experimenten 001-004 worden niet aangeraakt.

## B. Compatibiliteit en module-integratie

### K01 - Flatten in accuracyfunctie

Bestand: `utils.py`, oorspronkelijke regel 41.

```python
# Voor:
correct_k = correct[:k].view(-1).float().sum(0)
# Na:
correct_k = correct[:k].reshape(-1).float().sum(0)
```

Reden: deze top-k-tensor kan niet-contigue strides hebben. De flatten-operatie met
`view` faalde in de eerdere lokale PyTorch 2.10-controle. `reshape` behoudt hier de
bedoelde telling, zo nodig door een kopie. Een apart fixture met bekende top-1 en
top-5-uitkomsten controleert de telling. Geen verandering aan quantisatie of loss.

### K02 - Optionele GPU-autoselectie niet op moduleniveau importeren

Bestand: `utils.py`, oorspronkelijke regels 5 en 61-70.

De import `import gpustat` verhuist naar binnen `setup_gpus()`, direct voor de
bestaande aanroep van gpustat. De functie zelf blijft aanwezig en vereist nog
steeds gpustat wanneer zij wordt aangeroepen. De runner roept haar niet aan en
kiest expliciet `--device cuda` of `--device cpu`.

Dit is een integratieaanpassing om de accuracyhelper zonder een niet-gebruikte
GPU-queryafhankelijkheid te kunnen importeren. Het is geen vervanging van
`setup_gpus()` voor toekomstige originele launcher-runs en geen quantizerwijziging.

De runner schrijft de concrete unified diff naar `compatibility.patch`.

## C. Nieuwe technische test - geen volledige trainingslauncher

`run_smoke.py` is een nieuw stuurprogramma naast de broncode. Het importeert de
oorspronkelijke model-, loss- en optimizermodules. De kleine trainingslus volgt
`train.py:180-212`, met dezelfde precisievolgorde, lossfuncties, recursieve
teacherkeuze en een optimizerstap na alle precisies. Dit is geen bewijs dat de
hele oorspronkelijke `train.py`-launcher werkt.

Afzonderlijk vastgelegde uitvoerinstellingen: deterministische algoritmen,
vaste seeds, twee CPU-threads, TF32 uit via de PyTorch 2.10-API, AMP uit, geen
compile, expliciet device. `optimizer.zero_grad(set_to_none=False)` legt de
zero-grad-keuze expliciet vast; de nieuwe optimizer begint zonder gradients.

Afzonderlijke omvangsinstellingen: twee synthetische minibatches van twee,
geen scheduler, twee identieke verse herhalingen en checkpoint-roundtrip.
Dit zijn technische testinstellingen; ze mogen niet als ongewijzigd
200-epoch-CIFAR-10-trainingsprotocol worden beschreven.

## D. Onderzoeksmethode

Nog geen MX, truncation-ready quantizer, wijziging aan de STE, activatiepolicy,
BatchNorm-architectuur, distillatie of netwerkbreedte. Geen getraind checkpoint
van internet geladen. Geen accuracyclaim en geen keuze op grond van testsetprestaties.

## E. Lokale controle

Zie `local_validation/VALIDATION.md`: deze runner is lokaal op Linux/CPU uitgevoerd.
Windows/CUDA-validatie van het volledige model volgt pas op de laptop van de gebruiker.
