# Lokale validatie — 8×8, vaste ×64, vijf-epochpakket

**Status: PASS.** Deze controles zijn uitgevoerd op CPU met PyTorch 2.10.0+cpu en torchvision 0.25.0+cpu. Er zijn geen echte SVHN-beelden geladen, geen vijf-epoch-SVHN-run uitgevoerd en geen GPU-timings of nieuwe onderzoeksaccuracies gemeten.

## Uitgevoerde controles

- Onafhankelijke E4M3-scalaroracle: alle 254 eindige codes, 255 schaalkrachten en 64.770 code/schaalproducten, afronding, signed zero en foutgevallen.
- 8×8-coördinaten, beide paddingassen, niet-aaneengesloten tensors, roundtrips en transpositie-equivalentie voor de codecs en truncatieketen.
- AST-gelijkheid van 14 scalaire codec-, packing- en STE-definities ten opzichte van het gevalideerde 1×64-pakket.
- Oorspronkelijke recursieve driepassentraining versus onafhankelijke referentielus; identity-STE, losgekoppelde teachers, FP32-masters en Adam, drie BN-banken, exact herstel van RNG/backend.
- Werkelijk check-entrypoint met uitsluitend testmatige CPU-adaptatie; canonieke datasetnamen en echte create_context-opstart via begrensde virtuele datafixtures, zonder updates/downloads.
- Werkelijke runner/main met drie synthetische beelden: vijf epochs, stoppen na twee, hervatten tot vijf en vroege stop. Model, Adam, scheduler en RNG zijn op CPU exact gelijk tussen doorlopend en hervat.
- Source-only compacte exports herstellen dezelfde logits zonder masters of quantizers. Oude 1×64-layouts, verkeerde contracten, voltooide pilotcheckpoints en ongeldige CLI-opties worden geweigerd.
- Zes gerichte negatieve tests weigeren verkeerde train/test-brongeometrie vóór exportwrites. Kostenvalidatie weigert verkeerde groepsvorm en assen, ook als de aantallen kloppen.
- Twaalf notebook-/kostentests; echte B32- en B64-referenties via hashes bevestigd en tot epochs 1–5 beperkt.
- Alle 14 notebookcodecellen via echte IPython uitgevoerd; tien figuren gerenderd. De laatste synthetische runneroutput heeft nul analysefouten en zes geldige exportaudits. De afgeleverde notebook bevat geen uitgevoerde outputs.
- Het oorspronkelijke B64-pakket bleef volgens alle 115 oorspronkelijke manifestvermeldingen ongewijzigd.

## Grenzen

De productie-CLI houdt de bestaande Python 3.13 / Windows / CUDA-eisen en biedt geen stille CPU-fallback. CPU-adaptaties staan alleen in de expliciete local_validation-tests. CUDA-bitexactheid wordt niet geclaimd. Het commando `check` moet op de laptop PASS geven voordat de echte vijf-epochtraining wordt gestart.

Bewijs en geteste runtimehashes staan in `validation_summary.json` en de genoemde rapporten. Rapporten uit het oudere 1×64-pakket staan uitsluitend onder `provenance/inherited_group64_validation`.
