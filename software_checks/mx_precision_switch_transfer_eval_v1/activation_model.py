"""Modelled inference activation traffic at explicitly fused stage boundaries.

SVHN images are resized from32 to40 before this CNN. GitHub's compact low-bit
storage is a candidate hardware representation; its current PyTorch tensors
remainFP32. MX experiments keepA32. No input/activation is reused across images.
"""
from __future__ import annotations

BOUNDARIES = (
    ('input', (3,40,40), False),
    ('after_layers_2', (12,18,18), True),
    ('after_layers_5', (16,18,18), True),
    ('after_layers_9', (16,9,9), True),
    ('after_layers_12', (32,7,7), True),
    ('after_layers_15', (32,7,7), True),
    ('after_layers_18', (32,5,5), True),
    ('after_layers_21', (128,1,1), False),
    ('logits', (10,), False),
)

STAGE_NAMES = ('layers.0', 'layers.3', 'layers.6', 'layers.10',
               'layers.13', 'layers.16', 'layers.20', 'fc')

ASSUMPTIONS = (
    'Inference eval(): frozen BN and disabled dropout.',
    'Actual network input3x40x40 after Resize(40); source SVHN images are32x32.',
    'Conv/BN/pool/activation fused within each stage; only stage-boundary activations are materialized.',
    'Each stage input read once and output written once; no im2col, tiled rereads, accumulator spills or workspace traffic.',
    'GitHub low-bit activations are a proposed packed representation; current PyTorch tensors remainFP32.',
    'All MX weight modes keepA32. Input, final unquantized activation and logits remainFP32.',
    'Last Activate(quantize=False) clamps in low GitHub modes but does not quantize.',
    'Each boundary region begins at a line-aligned address; transfers round up independently.',
    'Writes have no read-for-ownership. Activation traffic does not enter the separate weight cache.',
    'Every inference has a new input; no activation reuse across precision switches.',
    'Modelled materialization-boundary traffic; not measured GPU, RTL or DRAM traffic.',
)


def activation_traffic(activation_bits: int, line_bytes: int = 64,
                       batch_size: int = 1) -> dict:
    """Return logical bytes and whole-line transactions for one inference batch."""
    if type(activation_bits) is not int or activation_bits not in (1,2,4,8,32):
        raise ValueError('activation_bits must be one of1,2,4,8,32')
    if type(line_bytes) is not int or line_bytes <= 0:
        raise ValueError('line_bytes must be a positive integer')
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    boundaries = []
    address = 0
    for name, shape, quantized in BOUNDARIES:
        elements = batch_size
        for dim in shape: elements *= dim
        bits = activation_bits if quantized else 32
        useful = (elements*bits+7)//8
        transactions = (useful+line_bytes-1)//line_bytes
        allocated = transactions*line_bytes
        boundaries.append(dict(id=name, shape=[batch_size,*shape], numel=elements,
                               activation_bits=bits, mode_quantized=quantized,
                               address=address, useful_bytes=useful,
                               transfer_bytes=allocated, transactions=transactions))
        address += allocated
    stages = []
    for index,name in enumerate(STAGE_NAMES):
        inp,out = boundaries[index:index+2]
        stages.append(dict(stage=name, read_boundary=inp['id'], write_boundary=out['id'],
                           useful_read_bytes=inp['useful_bytes'],
                           useful_write_bytes=out['useful_bytes'],
                           transfer_read_bytes=inp['transfer_bytes'],
                           transfer_write_bytes=out['transfer_bytes'],
                           read_transactions=inp['transactions'],
                           write_transactions=out['transactions'],
                           transactions=inp['transactions']+out['transactions'],
                           useful_bytes=inp['useful_bytes']+out['useful_bytes'],
                           transfer_bytes=inp['transfer_bytes']+out['transfer_bytes']))
    keys = ('useful_read_bytes','useful_write_bytes','transfer_read_bytes',
            'transfer_write_bytes','read_transactions','write_transactions',
            'transactions','useful_bytes','transfer_bytes')
    result = dict(activation_bits=activation_bits,line_bytes=line_bytes,
                  batch_size=batch_size,boundaries=boundaries,stages=stages,
                  assumptions=list(ASSUMPTIONS))
    result.update({key:sum(stage[key] for stage in stages) for key in keys})
    result['total_bytes']=result['transfer_bytes']
    result['total']=dict(useful_read_bytes=result['useful_read_bytes'],
                         useful_write_bytes=result['useful_write_bytes'],
                         transfer_read_bytes=result['transfer_read_bytes'],
                         transfer_write_bytes=result['transfer_write_bytes'],
                         transactions=result['transactions'],
                         useful_bytes=result['useful_bytes'],
                         transfer_bytes=result['transfer_bytes'])
    result['totals']=dict(result['total'])
    return result
