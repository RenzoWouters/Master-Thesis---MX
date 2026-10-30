# Softwarevalidatie

De lokale controle gebruikt Linux/Python 3.12 en PyTorch 2.10.0+cpu. Er is geen echte GPU-meting of ImageNet-training uitgevoerd.

Geslaagd: bestaande onafhankelijke engine-, checkpoint- en watchdogzelftests; backend-workermocks voor succes/fout/OOM en behoud van gedeeltelijke metingen; referentiehashes en weigeren van gewijzigde bron; parser zonder trainingscommando; afwijzen van het diagnoserapport als pilot-checkbewijs; timeout/exitcoderegels; notebooksyntax, lege outputs en vijf visueel gecontroleerde representatieve statusgevallen.

De volledige publieke check-opdracht eindigt hier terecht met FAIL wegens ontbrekende CUDA. Het volledige log, rapport, onuitgevoerde notebook en feedbackbundel blijven behouden. De oorspronkelijke v1_1-bestanden zijn byte voor byte gecontroleerd en niet gewijzigd.

De bestaande GPU-getallen in baseline_results.json zijn de twee echte door de gebruiker aangeleverde controles. Nieuwe GPU-resultaten zijn niet ingevuld. Mockwaarden uit softwaretests zijn geen prestatie- of accuracyresultaten en de mockfiguren zijn niet in dit pakket opgenomen.
