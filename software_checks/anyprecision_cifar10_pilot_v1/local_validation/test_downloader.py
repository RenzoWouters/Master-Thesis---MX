"""Offline unit test of download/extraction guards; HTTP bytes are mocked."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_pilot as p

names = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
contents = {n: ('TEST FIXTURE: '+n).encode() for n in names}
stream = io.BytesIO()
with tarfile.open(fileobj=stream, mode='w:gz') as tar:
    for n, blob in contents.items():
        member = tarfile.TarInfo('cifar-10-batches-py/'+n)
        member.size = len(blob)
        tar.addfile(member, io.BytesIO(blob))
archive = stream.getvalue()
entries = [[n, hashlib.md5(contents[n]).hexdigest()] for n in names]
cls = SimpleNamespace(train_list=entries[:5],test_list=entries[5:],base_folder='cifar-10-batches-py',
    filename='cifar-10-python.tar.gz', url='https://example.invalid/offline-fixture.tar.gz',
    tgz_md5=hashlib.md5(archive).hexdigest())
class Response(io.BytesIO):
    def geturl(self):
        return cls.url
with tempfile.TemporaryDirectory() as temp:
    cache = Path(temp) / 'raw'
    with patch('urllib.request.urlopen', return_value=Response(archive)) as http:
        result = p.ensure_dataset(cache, cls, True, lambda _: None)
        assert http.call_count == 1 and result['new_download']
    with patch('urllib.request.urlopen', side_effect=AssertionError('No network allowed')):
        reused = p.ensure_dataset(cache, cls, False, lambda _: None)
        assert reused['new_download'] is False
    changed = cache / cls.base_folder / names[0]
    changed.write_bytes(b'INTENTIONALLY CORRUPTED UNIT-TEST FIXTURE')
    before = changed.read_bytes()
    try:
        p.ensure_dataset(cache, cls, True, lambda _: None)
        raise AssertionError('Corrupt fixture incorrectly accepted')
    except RuntimeError as exc:
        assert 'Checksumfout' in str(exc)
    assert changed.read_bytes() == before
print(json.dumps(dict(status='PASS',network_used=False,mocked_archive_extraction=True,
    verified_reuse=True,corrupt_existing_file_rejected_without_overwrite=True),indent=2))
