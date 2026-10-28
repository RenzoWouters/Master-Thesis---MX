"""Read-only verification regression checks using temporary synthetic bytes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import source_catalog as sources


class SourceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = copy.deepcopy(sources.load_catalog())
        for spec in self.catalog["models"]:
            run = self.root / spec["run_relative_path"]
            run.mkdir(parents=True)
            original = sources.PACKAGE_ROOT / spec["reference_report_file"]
            (run / "run_report.json").write_bytes(original.read_bytes())
            for artifact in spec["artifacts"]:
                blob = ("SYNTHETIC TEST BYTES; NOT A TRAINED MODEL: "
                        + spec["model_id"] + "/" + artifact["file"]).encode()
                artifact["sha256"] = hashlib.sha256(blob).hexdigest()
                artifact["bytes"] = len(blob)
                (run / artifact["file"]).write_bytes(blob)
        self.catalog_path = self.root / "test_catalog.json"
        self.catalog_path.write_text(json.dumps(self.catalog))
        self.mock = patch.object(sources, "CATALOG_FILE", self.catalog_path)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def report_path(self, index=0):
        return self.root / self.catalog["models"][index]["run_relative_path"] / "run_report.json"

    def modify_report(self, function, index=0):
        path = self.report_path(index)
        report = json.loads(path.read_text())
        function(report)
        path.write_text(json.dumps(report))

    def test_valid_exact_sources(self):
        result = sources.verify_sources(self.root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["verified_file_count"], 19)
        self.assertFalse(result["pickle_deserialized"])

    def test_changed_export_is_rejected(self):
        spec = self.catalog["models"][1]
        path = self.root / spec["run_relative_path"] / spec["artifacts"][-1]["file"]
        path.write_bytes(path.read_bytes() + b"changed")
        result = sources.verify_sources(self.root)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("SHA256" in message for message in result["errors"]))

    def test_missing_export_is_rejected(self):
        spec = self.catalog["models"][1]
        (self.root / spec["run_relative_path"] / spec["artifacts"][-1]["file"]).unlink()
        self.assertEqual(sources.verify_sources(self.root)["status"], "FAIL")

    def test_epoch100_original_checkpoint_fallback(self):
        spec = self.catalog["models"][0]
        run = self.root / spec["run_relative_path"]
        source = run / "last_checkpoint.pt"
        target = run / "checkpoints/epoch_00100.pt"
        target.parent.mkdir()
        source.rename(target)
        result = sources.verify_sources(self.root)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["models"]["github5"]["artifacts"][0]["path"].endswith("epoch_00100.pt"))

    def test_synthetic_report_is_rejected(self):
        self.modify_report(lambda report: report.update(synthetic_fixture_only=True))
        self.assertEqual(sources.verify_sources(self.root)["status"], "FAIL")

    def test_wrong_group_size_is_rejected(self):
        self.modify_report(lambda report: report["config"].update(mx_block_size=64), 1)
        self.assertEqual(sources.verify_sources(self.root)["status"], "FAIL")

    def test_pilot_is_rejected(self):
        self.modify_report(lambda report: report.update(completed_epoch=5))
        self.assertEqual(sources.verify_sources(self.root)["status"], "FAIL")

    def test_relocated_run_override(self):
        run = self.report_path(0).parent
        destination = self.root / "verplaatst model"
        run.rename(destination)
        result = sources.verify_sources(self.root, {"github5": "verplaatst model/run_report.json"})
        self.assertEqual(result["status"], "PASS")

    def test_metadata_only_is_labelled(self):
        spec = self.catalog["models"][1]
        (self.root / spec["run_relative_path"] / spec["artifacts"][-1]["file"]).unlink()
        result = sources.verify_sources(self.root, hash_files=False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["verification_scope"], "report_metadata_only")
        self.assertEqual(result["verified_file_count"], 0)

    def test_unknown_override_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Onbekende modelnamen"):
            sources.verify_sources(self.root, {"typo": "anything"})


if __name__ == "__main__":
    unittest.main()
