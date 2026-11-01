"""Проверка сборки датасета без сети: python -m unittest tests.test_build_dataset -v."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import zstandard

from scripts import build_dataset
from tests.test_parse_pgn import ANALYZED


class BuildDatasetTests(unittest.TestCase):
    def test_month_range_crosses_year_and_rejects_reversed(self):
        self.assertEqual(build_dataset.month_range("2013-11", "2014-02"),
                         ["2013-11", "2013-12", "2014-01", "2014-02"])
        self.assertEqual(len(build_dataset.month_range("2013-01", "2014-12")), 24)
        with self.assertRaises(ValueError):
            build_dataset.month_range("2014-01", "2013-12")

    def test_build_month_downloads_parses_and_skips_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, out = root / "raw", root / "full"
            raw.mkdir(), out.mkdir()
            payload = zstandard.ZstdCompressor().compress((ANALYZED * 3).encode())

            def fake_download(url, path):
                self.assertTrue(url.endswith("lichess_db_standard_rated_2013-02.pgn.zst"))
                path.write_bytes(payload)

            with mock.patch.object(build_dataset.download_data, "download", side_effect=fake_download) as dl:
                entry = build_dataset.build_month("2013-02", raw, out, "http://x", delete_raw=True)
                self.assertEqual(dl.call_count, 1)
                self.assertEqual(entry["games_written"], 3)
                self.assertTrue(entry["raw_deleted"])
                self.assertFalse((raw / "lichess_2013-02.pgn.zst").exists())
                with (out / "games_2013-02.csv").open(encoding="utf-8") as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 3)
                self.assertEqual(list(out.glob("*.part")), [])
                self.assertIsNone(build_dataset.build_month("2013-02", raw, out, "http://x", False))
                self.assertEqual(dl.call_count, 1)

            manifest = out / "manifest.json"
            data = build_dataset.load_manifest(manifest)
            data["months"]["2013-02"] = entry
            build_dataset.save_manifest(manifest, data)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["months"]["2013-02"]["games_written"], 3)


if __name__ == "__main__":
    unittest.main()
