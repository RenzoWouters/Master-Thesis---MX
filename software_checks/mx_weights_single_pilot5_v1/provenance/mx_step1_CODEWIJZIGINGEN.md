# Van referentiecode naar MX, zonder de referentie te overschrijven

## Afbakening

Beslist door gebruiker: eerst MX uitsluitend voor gewichten.
Expliciet eerste implementatievoorstel: MXFP6 E2M3, blok 32, E8M0-schaal,
vaste A32/ReLU en BN32. Geen activatie-, gradient-, exponent- of schaaltruncatie.
Nog niet besloten: latere QAT-modi, hun losses en precieze vergelijkingsgroepen.
Dit pakket heeft bewust alleen `check` en `evaluate`, geen honderd-epochstart.

## 1. De werkelijke aangrijpingsplaats

`models/quan_ops.py:132-134` in de aangeleverde auteurscode:

```python
def forward(self, input, order=None):
    weight_q = self.quantize_fn(self.weight)
    return F.conv2d(input, weight_q, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)
```

De oorspronkelijke convolutie hoeft niet te veranderen. In een kopie van het
model vervangen we `quantize_fn` voor `layers.3`, `layers.6`, `layers.10`,
`layers.13`, `layers.16`, `layers.20`. Dit zijn precies de zes bestaande
kwantiseerbare Conv2d-lagen; niet de eerste convolutie of fc-eindlaag.
`mx_adapter.adapt_copy` controleert die lijst en weigert een afwijkend model.
Er worden geen netwerkgewichten met `.data = ...` overschreven.

## 2. Wat zijn eigenlijk de gewichten in de W32-referentie?

`models/quan_ops.py:87-99` doet ook bij W32 voorbewerking:

```
E = mean(abs(W)).detach()
G(W) = (tanh(W) / max(abs(tanh(W)))) * E
```

`W` is de trainbare FP32-master. `G(W)` is wat de bestaande W32-convolutie
werkelijk ziet. De nieuwe forward wordt dus `Q_MX(G(W))`, NIET onverwacht
`Q_MX(W)`. E is een bestaande laagfactor, niet de nieuwe MX-blokschaal X.
E is verwerkt in de effectieve waarden voor MX-encoding; de export heeft
geen extra losse E nodig om die gewichten te reconstrueren.

De volgorde tanh/delen/vermenigvuldigen is gelijk aan de auteurscode.
Alleen voor een volledig nulgewicht is de oorspronkelijke deling 0/0 niet
gedefinieerd. De adapter kiest daar expliciet een veilige nuluitkomst; hij
voegt geen epsilon toe aan andere gewichten. Dit is apart getest en vermeld.

## 3. Van gewichtstensor naar MX-blokken

Een Conv2d-gewicht heeft vorm [Cout, Cin, Kh, Kw]. We flatten alleen de laatste
3 dimensies tot [Cout, Cin*Kh*Kw], in de bestaande C-order. Elke rij wordt in
blokken van 32 gesplitst. Blokken overschrijden nooit een outputkanaalgrens.
Onvolledige blokken worden met +0 aangevuld. De originele vorm staat in de
export; padding wordt na decoding verwijderd. Dit is een concrete software-
layoutkeuze; niet de 8x8-gewichtstegeling uit de robotics-hardwarepaper.

Per blok volgt de schaal het voorbeeld in de MX-paper, Algorithm 1:

```
e = floor(log2(max(abs(block)))) - 2
X = 2**e
P = quantize_E2M3(block / X)
W_mx = X * P
```

2 is de grootste normale lokale E2M3-exponent, NIET het aantal bits van E8M0.
De schaal wordt als exponentcode e+127 opgeslagen; toegestane codes 0..254.
Code 255 betekent NaN en wordt in deze finite-only module afgewezen.
Nulblokken krijgen e=0, dus X=1. Voor extreme kleine bronnen wordt e naar
[-127,127] begrensd en het aantal begrensde blokken gerapporteerd.

`torch.frexp` bepaalt het exponentdeel exact voor machten van twee, zonder
onnodige log2-afrondingskwesties. De numerieke codec gebruikt FP64-intermediairen
om de regels helder te implementeren; input en output van de gewichtsquantizer
zijn FP32. Dit is niet op doorvoer geoptimaliseerd.

## 4. E2M3-codes en afronding

Elke code heeft [s ee mmm]. Voor een normale waarde (ee>0):
`(-1)**s * 2**(ee-1) * (1+mmm/8)`.
Voor ee=0: `(-1)**s * (mmm/8)` (lokale subnormal of signed zero).

De codec zoekt het dichtstbijzijnde representabele E2M3-getal; bij een tie
kiest hij de code met een even minst-significant bit. Buiten het bereik wordt
naar +/-7.5 verzadigd. Dit is gewone MX-kwantisatie, nog GEEN bittruncatie.
Lokale E2M3-subnormals worden ondersteund. Subnormale FP32-BRONWAARDEN worden
volgens het voorbeeld in de MX-paper naar signed zero gebracht en geteld.
NaN/Inf-ingangen worden met een fout geweigerd, niet stilzwijgend gewist.
Dit is geen claim van een volledige IEEE/OCP-implementatie voor alle speciale
waarden of van een OCP-conforme dot-productkern.

Klein kunstmatig voorbeeld, niet afkomstig uit het netwerk:
blokmaximum 7.0 -> e=0, X=1; 1.30 -> 1.25, -3.33 -> -3.25.

## 5. Activaties, BatchNorm en de vijf-precisielus

De bestaande `train.py:166-168,183-198` zet wbit EN abit samen.
`models/svhn_quan.py:24-30` gebruikt ReLU bij A32, anders clamp[0,1].
`models/quan_ops.py:26-28` kiest de BN-bank via abit.
Daarom zetten we expliciet ALLE abit=32 en behouden BN32. Gewichtscode 6
betekent hier niet abit=6 en ook niet de oude uniforme 6-bitquantizer.

`MXWeightQuantizer` heeft een aparte enabled-vlag. Hij weigert de oude
gekoppelde vijf-bitlus: de eerste diagnose moet altijd A32 behouden.
Een wijziging van gewichten mag natuurlijk de resulterende activatiewaarden
veranderen; alleen het activatieFORMaat/beleid blijft gelijk.

Voor deze diagnose: model.eval(), torch.no_grad(), geen optimizer, geen
loss.backward(), geen BN-herkalibratie. De vier paden worden vergeleken bij
zowel het vooraf geselecteerde beste als het laatste referentiecheckpoint.
Het huidige referentiemodel is via joint W/A-training geleerd, NIET via een
onafhankelijke single-mode FP32-training. Het gepaarde direct-cast-verschil is
interpreteerbaar; een toekomstige andere trainingsmethode vraagt een eigen
bijpassende controle. Niet ieder verschil met de oude vijf-bitrun is MX alleen.

## 6. Voorbereid voor later trainen: een expliciete STE

`fake_quantize_mx` is een custom autograd.Function:
forward = exact encode/decode; backward = identity voor de MX-stap.

Dit is een straight-through estimator, een gekozen benadering van de gradient;
NIET de echte afgeleide van afronden. Via G(W) blijven de oorspronkelijke tanh-
en normalisatiegradients lopen en blijft E detached. Schaalkeuze en discrete
codes krijgen geen eigen gradient. Ook bij saturatie gebruiken we in dit eerste
voorstel identity (geen verborgen gradientmasker). Er worden geen gradient-
precisies verlaagd; de master en optimizerstaat blijven later FP32.

Een gewone numerieke gradcheck van de afrondfunctie zou geen identity opleveren
(en is dus geen juist bewijs voor de STE). De test controleert de expliciet
bedoelde surrogate-gradient rechtstreeks. Geen hogere-orde-gradientclaim.
De forward gebruikt een custom functie om de gedecodeerde waarde exact terug
te geven, niet een algebraische detach-truc die extra FP-afronding kan geven.

Deze STE is geimplementeerd en getest, maar het pakket start nog geen QAT.
Een simpele latere single-mode trainingsstap zou conceptueel zijn:

```python
model.train()
optimizer.zero_grad()
logits = model(images)  # intern G(W) -> MX -> FP32-emulatie
loss = criterion(logits, labels)
loss.backward()         # gradientbenadering door MX
optimizer.step()       # past FP32-master W aan
```

Dit is NIET ongemerkt hetzelfde als de oude vijf-mode-distillatielus. Modi,
loss en vergelijking worden voor die volgende trainingsproef apart vastgelegd.

## 7. Opslag versus rekenbuffer

`export_payload` encodeert eenmalig uit G(W). Vier 6-bitcodes worden little-
endian in drie bytes gepakt. Elk blok gebruikt 24 bytes plus een schaalbyte.
Padding en aanvullende FP32-laag/BN-staat worden apart geteld; torch.save
bestandsheaders maken de fysieke bestandsgrootte groter dan de payload.

`load_exported_model` construeert een model uit die codes, schalen en aanvullende
staat. Voor de zes doellagen bestaat GEEN FP32-masterparameter in dit model.
Het decodeert wel naar FP32-buffers voor F.conv2d. Dat is numerieke emulatie;
geen native 6-bit-GPU-berekening, geen bewezen energiewinst. De overige vier
ongebruikte BN-banken blijven in de export uit transparante compatibiliteit en
tellen dus ook mee. Later kunnen we die apart verwijderen, niet stilzwijgend.

Dit legt de basis voor een bronbestand waaruit later lagere mantissaprecisies
worden afgeleid. Die latere afleiding mag geen FP-master raadplegen en geen
schalen opnieuw bepalen. In DIT pakket bestaat nog maar een MX-precisie.

## Bronnen

- Aangeleverde Any-Precision-DNNs-master(3).zip: SHA256 in SOURCE_MANIFEST.json.
  models/quan_ops.py, models/svhn_quan.py, train.py; bronregels hierboven.
- Projectpaper Microscaling Data Formats for Deep Learning, PDF p. 2-4,
  tabel 1, Algorithm 1 en compute-flow. Die paper voert ook andere operanden in
  MX uit; onze weight-only diagnose is dus geen reproductie van die compute-flow.
- Extern geraadpleegd voor exacte bitcodering: OCP Microscaling Formats v1.0
  (Sep 2023), p. 11-14, E2M3 en E8M0:
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- PyTorch documentatie Extending torch.autograd voor Function.forward/backward:
  https://docs.pytorch.org/docs/stable/notes/extending.html
  Implementatie hier daadwerkelijk getest met torch 2.10.0+cpu, niet met de
  mogelijk nieuwere stable-documentatieversie. De gebruikersomgeving blijft 2.10.0.

Keuzes zoals A32, de zes lagen, checkpointdiagnose en identity-STE zijn onze
expliciete eerste onderzoeksconfiguratie; de bronnen bepalen niet dat dit de
beste configuratie voor deze CNN is.
