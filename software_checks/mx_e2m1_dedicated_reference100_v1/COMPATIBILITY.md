# Compatibiliteit afzonderlijk van onderzoek

Geen nieuwe pakketten, nieuwe PyTorch-versie, AMP of compile. De vaste Python3.13 / torch2.10.0 /
torchvision0.25.0 Windows/CUDA-omgeving blijft behouden. Bestaande bronpatch voor utils.py
wordt met bronmanifest gecontroleerd. Data-, log-, checkpoint- en hervatkern byte-identiek.
Geen stille CPU-productiefallback. Alleen lokale synthetische tests mogen Tensor.cuda mocken.
Alle concrete backendinstellingen en waarschuwingen worden geregistreerd. Vaste seed is
geen garantie voor bitexacte CUDA-herhaling. Pakketten worden nooit automatisch geupgrade.
De notebook vereist dezelfde bestaande pandas/matplotlib/ipykernel-installatie.
