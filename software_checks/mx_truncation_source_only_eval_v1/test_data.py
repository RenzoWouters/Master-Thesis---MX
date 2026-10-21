"""Unchanged test-only dataset preparation from the previous evaluator."""
from pathlib import Path
import svhn_support as s

def prepare_test_only(args, modules, reference: dict, out: Path, log):
    import torch
    data = args.data_root.resolve()
    path = data / 'test_32x32.mat'
    s.require(path.is_file(), 'test_32x32.mat ontbreekt; gebruik de bestaande data/svhn-map. Geen automatische download.')
    info = s.hashes(path)
    expected = reference['dataset_files']['files']['test_32x32.mat']
    s.require(all(info[k] == expected[k] for k in info), 'Testdata wijken af van de trainingsreferentie.')
    s.require(info['md5'] == modules.data.SVHNMIX.split_list['test'][2], 'Testdata falen auteurs-MD5.')
    modules.data.data_paths['svhn'] = str(data)
    transform = modules.data.get_transform('svhn', 'val')
    s.require(repr(transform) == reference['transforms']['test'], 'Andere testpreprocessing.')
    dataset = modules.data.get_dataset('svhn', 'val', transform, download=False)
    s.require(dataset.split == 'test', 'Verkeerde split.')
    selection = s.validate_dataset(dataset, 26032, 'SVHN test')
    s.require(selection == reference['dataset_selection']['test'], 'Testlabels of volgorde verschillen.')
    loader = torch.utils.data.DataLoader(s.IndexedDataset(dataset), batch_size=128,
                  shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    log('DATA PASS: alleen test=26032; batch=128; 204 batches; train en extra niet gelezen.')
    s.write_json(out / 'dataset_manifest.json', {'test_file': {'path': str(path), **info},
                 'selection': selection, 'transform': repr(transform), 'train_read': False, 'extra_read': False})
    return loader, path, info
