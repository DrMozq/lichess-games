"""Проверка сборки и загрузки датасета без сети: python -m unittest tests.test_build_dataset -v
(или python tests/test_build_dataset.py)."""

import csv
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pandas as pd
import zstandard

# Корень проекта в sys.path: тесты запускаются и файлом, и из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_dataset, download_data
from tests.test_parse_pgn import ANALYZED

PAYLOAD = os.urandom(300_000)


class FakeServer(ThreadingHTTPServer):
    """Локальный сервер с настройками поведения для RangeHandler."""

    def __init__(self, ranges: bool, cut_first: bool):
        super().__init__(("127.0.0.1", 0), RangeHandler)
        self.ranges = ranges
        self.cut_first = cut_first
        self.requests: list[str | None] = []


class RangeHandler(BaseHTTPRequestHandler):
    """Отдаёт PAYLOAD; поддержку Range и обрыв первого ответа задают атрибуты сервера."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        server = self.server
        assert isinstance(server, FakeServer)
        server.requests.append(self.headers.get("Range"))
        start = 0
        match = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range") or "")
        if match and server.ranges:
            start = int(match.group(1))
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if server.cut_first:
            server.cut_first = False
            self.wfile.write(body[:len(body) // 3])
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(body)


class DownloadTests(unittest.TestCase):
    def serve(self, ranges=True, cut_first=False):
        server = FakeServer(ranges, cut_first)
        # Локальный сервер — мимо системного прокси, если он задан в окружении.
        proxy_env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
        proxy_env.start()
        self.addCleanup(proxy_env.stop)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return server, f"http://127.0.0.1:{server.server_address[1]}/f.zst", Path(tmp.name) / "f.zst"

    def test_resumes_existing_part(self):
        server, url, out = self.serve()
        out.with_name("f.zst.part").write_bytes(PAYLOAD[:100_000])
        download_data.download(url, out, chunk_size=4096)
        self.assertEqual(out.read_bytes(), PAYLOAD)
        self.assertEqual(server.requests, ["bytes=100000-"])
        self.assertFalse(out.with_name("f.zst.part").exists())

    def test_complete_part_is_finalized(self):
        _, url, out = self.serve()
        out.with_name("f.zst.part").write_bytes(PAYLOAD)
        download_data.download(url, out)
        self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_restarts_when_server_ignores_range(self):
        _, url, out = self.serve(ranges=False)
        out.with_name("f.zst.part").write_bytes(b"x" * 1000)
        download_data.download(url, out, chunk_size=4096)
        self.assertEqual(out.read_bytes(), PAYLOAD)

    def test_retries_after_dropped_connection(self):
        server, url, out = self.serve(cut_first=True)
        with mock.patch.object(download_data.time, "sleep") as sleep:
            download_data.download(url, out, chunk_size=4096)
        self.assertEqual(out.read_bytes(), PAYLOAD)
        self.assertEqual(sleep.call_count, 1)
        self.assertIsNone(server.requests[0])
        self.assertRegex(server.requests[1] or "", r"bytes=\d+-")


class BuildDatasetTests(unittest.TestCase):
    def test_sample_is_exact_reproducible_and_lossless(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "games_2015-01.csv"
            frame = pd.DataFrame({"game_id": [f"g{i}" for i in range(1000)],
                                  "white_acpl": ["" if i % 3 else str(i) for i in range(1000)],
                                  "evals": ["1,2,-3" if i % 2 else "" for i in range(1000)]})
            frame.to_csv(path, index=False)
            first = pd.concat(build_dataset.sample_month(path, 1000, 120, 42, "2015-01"))
            again = pd.concat(build_dataset.sample_month(path, 1000, 120, 42, "2015-01"))
            other = pd.concat(build_dataset.sample_month(path, 1000, 120, 42, "2016-01"))
            self.assertEqual(len(first), 120)
            self.assertEqual(first["game_id"].nunique(), 120)
            self.assertEqual(first["game_id"].tolist(), again["game_id"].tolist())
            self.assertNotEqual(first["game_id"].tolist(), other["game_id"].tolist())
            original = frame.astype(str).replace("nan", "").set_index("game_id")
            pd.testing.assert_frame_equal(first.set_index("game_id"), original.loc[first["game_id"]])
            self.assertEqual(len(pd.concat(build_dataset.sample_month(path, 1000, 5000, 42, "2015-01"))), 1000)

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
            raw.mkdir()
            out.mkdir()
            payload = zstandard.ZstdCompressor().compress((ANALYZED * 3).encode())

            def fake_download(url, path):
                self.assertTrue(url.endswith("lichess_db_standard_rated_2013-02.pgn.zst"))
                path.write_bytes(payload)

            with mock.patch.object(build_dataset.download_data, "download", side_effect=fake_download) as dl:
                entry = build_dataset.build_month("2013-02", raw, out, "http://x", delete_raw=True)
                assert entry is not None
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
