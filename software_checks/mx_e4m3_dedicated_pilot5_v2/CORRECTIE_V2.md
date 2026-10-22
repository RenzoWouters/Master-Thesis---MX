# Correctie v2 — exacte E4M3-berekening op CPU en CUDA

## Aanleiding en wat vaststaat

De laptopcontrole op 22 oktober 2026 stopte in v1 bij
`all_254_finite_codes_match_independent_decode`, vóór de modelcheck en vóór
SVHN-training. De omgevingcontrole en GPU-herkenning waren geslaagd.
De foutmelding van v1 bevatte nog geen afwijkende waarden: de concrete codes,
ULP-afwijkingen of eventuele verschillen in het teken van nul zijn daardoor
nog niet bekend.

De oorspronkelijke elementdecoder gebruikte `torch.ldexp`. In de officiële
PyTorch 2.10.0-bron berekent deze functie een macht van twee via `pow` en
vermenigvuldigt die met de invoer. Dat vormt een mogelijke route voor een
verschil tussen CPU en CUDA. Het is geen bewezen diagnose van een specifieke
hardware- of driverfout. De v2-check meet het oude pad apart op de laptop en
bewaart de exacte afwijkingen in `legacy_codec_diagnostics.json`.

## Uitgevoerde correctie

Voor de E4M3-elementmagnitude gebruikt v2 uitsluitend een gehele teller:

- E=0: teller=M.
- E>0: teller=(8+M) << (E-1).
- Waarde=teller * 2^-9, daarna het oorspronkelijke teken, inclusief -0.

Dit is algebraïsch dezelfde E4M3-definitie. De grootste teller is 229376 en is
exact representabel in FP64. NaN-codes 127 en 255 blijven afgewezen.

Voor de E8M0-factor 2^k, met k van -127 tot en met 127, vormt v2 de normale
FP64-bitrepresentatie rechtstreeks: teken 0, exponent k+1023, mantissa 0.
Normalisatie en reconstructie gebruiken FP64-vermenigvuldiging met deze
factor. Dit verandert geen geselecteerde schaal, voegt geen compensatie toe
en introduceert geen geleerde parameter. De schaalregel blijft
floor(log2(block_amax))-8 met dezelfde begrenzing en nulblokregel.

G(W), RNE, saturatie, FP32-invoersubnormalbeleid, identity-STE, A32,
FP32-masters/Adam en de gehele vijf-epochtrainingsopzet blijven behouden.
De notebook en de adapter zijn byte-identiek aan v1.

## Strenge controle en betere diagnose

De oorspronkelijke 31 codeccontroles blijven behouden. Twee exacte controles
zijn toegevoegd: alle 255 E8M0-factoren en alle 64770 combinaties van eindige
E4M3-elementen en geldige E8M0-schalen als FP64-tussenresultaat. Een combinatie
die bij omzetting buiten het eindige FP32-bereik valt, blijft in de bestaande
MXTensor-controle afgewezen.

De runner bewaart het codectestrapport ook bij FAIL. Afwijkingen noemen
numerieke en tekenbitmismatches afzonderlijk, met de eerste 16 exacte
hexadecimale waarden en ruwe FP64-bitpatronen. Beide nieuwe JSON-rapporten
zitten in `feedback_bundle.zip`.

De v1-probe is uitsluitend diagnose en wordt niet voor v2-gewichten gebruikt.
Een mismatch in die probe wordt zichtbaar gemeld. De v2-codec moet zelf aan
alle exacte tests voldoen om PASS te krijgen; geen tolerantie is verruimd.

## Validatie en praktische grens

De lokale rapporten en gebruikte testcode staan in `local_validation/`.
Het zijn uitsluitend CPU-controles en kleine synthetische modelcontroles.
Deze omgeving heeft geen CUDA-GPU. De werkelijke oplossing op de RTX 5070
Laptop GPU moet daarom nog blijken uit `run_train.py check` van v2.

Er zijn nog geen nieuwe SVHN-accuracies. De trainingsopdracht stopt op epoch 5.
De oorspronkelijke v1-code en eerdere resultaten zijn bewaard. Een oude v1-run
wordt niet stilzwijgend naar een nieuw hervatcontract omgezet.

## Primaire implementatiebronnen

- PyTorch v2.10.0, BinaryOps.cpp, regels 1561–1577: _pow2 en ldexp.
  https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/BinaryOps.cpp#L1561-L1577
- PyTorch v2.10.0, CUDA PowKernel.cu, regels 125–138: pow_-kernel.
  https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/PowKernel.cu#L125-L138
- PyTorch v2.10.0, CUDA Pow.cuh, regels 17–32: Windows-wrapper voor pow.
  https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Pow.cuh#L17-L32

De bronroute is gecontroleerd op de vaste tag v2.10.0. Zij onderbouwt de
implementatiekeuze; de concrete laptopafwijkingen volgen uit de nieuwe probe.
