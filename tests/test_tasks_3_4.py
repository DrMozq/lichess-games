"""Проверки задач 3–4: python -m unittest tests.test_tasks_3_4 -v (или python tests/test_tasks_3_4.py).

Фикстуры синтетические, задают проверяемые вручную знаменатели и потери.
Отдельная проверка использует штатный парсер PGN проекта.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Корень проекта в sys.path: тесты запускаются и файлом, и из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import analysis_common as common
from scripts import analyze_blunders as blunders
from scripts import analyze_speed as speed
from scripts import parse_pgn
from tests.test_parse_pgn import ANALYZED, rows

ROOT = Path(__file__).resolve().parents[1]


def scalar(value: Any) -> float:
    """Одно значение из таблицы pandas как float (для проверок и анализатора типов)."""
    return float(value)


def row(game_id, **overrides):
    record: dict[str, object] = dict.fromkeys(parse_pgn.COLUMNS, "")
    record.update(game_id=game_id, speed="blitz", result="1-0", utc_date="2013-01-01",
                  base_sec=300, inc_sec=0, white_elo=1200, black_elo=1800,
                  n_plies=40, has_eval=0, termination="Normal")
    record.update(overrides)
    return record


def options(out, chunksize=2, **kwargs):
    args = argparse.Namespace(out=out, chunksize=chunksize, speed_source="time_control",
                              rating_bin=200, min_player_games=1, min_evaluated=1,
                              speeds=list(common.ALL_SPEEDS))
    vars(args).update(kwargs)
    return args


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def csv(self, name, records):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=parse_pgn.COLUMNS)
            writer.writeheader()
            writer.writerows(records)
        return path

    def test_time_control_boundaries_and_increment(self):
        base = pd.Series([29, 30, 179, 180, 479, 480, 1499, 1500, 300, -1, "", 0])
        inc = pd.Series([0] * 8 + [5, 0, 0, 0])
        actual = common.classify_speed(base, inc).tolist()
        self.assertEqual(actual, ["ultrabullet", "bullet", "bullet", "blitz", "blitz", "rapid",
                                  "rapid", "classical", "rapid", "unknown", "unknown", "unknown"])

    def test_speed_counts_deduplication_and_missing_ratings(self):
        p1 = self.csv("a.csv", [row("a", n_plies=21), row("b", result="1/2-1/2", n_plies=40, white_elo="?"),
                                row("unfinished", result="*"), row("", result="0-1")])
        p2 = self.csv("b.csv", [row("a", result="0-1"), row("c", result="0-1", n_plies=62),
                                row("r", speed="classical", base_sec=600), row("u", base_sec=60)])
        args = options(self.root, speeds=list(common.SPEEDS), chunksize=1)
        tables, quality = speed.analyze([p1, p2], args)
        blitz = tables["speed_summary"].set_index("speed").loc["blitz"]
        self.assertEqual(blitz.games, 3)
        self.assertAlmostEqual(scalar(blitz.draw_share), 1 / 3)
        self.assertEqual(blitz.median_plies, 40)
        self.assertEqual(blitz.q25_plies, 30.5)
        self.assertEqual(blitz.white_rating_games, 2)
        self.assertEqual(quality["duplicate_games"], 1)
        self.assertEqual(quality["missing_id"], 1)
        self.assertEqual(quality["unfinished_or_invalid_result"], 1)
        self.assertEqual(quality["speed_mismatches"], 2)
        self.assertEqual(quality["outside_selected_speeds"], 1)
        args.chunksize = 100
        other, other_quality = speed.analyze([p1, p2], args)
        for name in tables:
            pd.testing.assert_frame_equal(tables[name], other[name])
        self.assertEqual(quality, other_quality)

    def test_weighted_acpl_uses_moves_not_average_of_game_means(self):
        path = self.csv("weighted.csv", [
            row("a", has_eval=1, white_acpl=200, white_evaluated=1, white_blunders=1),
            row("b", has_eval=1, white_acpl=20, white_evaluated=9, white_blunders=0),
            row("c", has_eval=0, white_acpl=0, white_evaluated=10, white_blunders=0),
        ])
        tables, quality = blunders.analyze([path], options(self.root, chunksize=1))
        group = tables["blunders_by_rating"].iloc[0]
        self.assertEqual(group.player_games, 2)
        self.assertEqual(group.evaluated_moves, 10)
        self.assertEqual(group.blunders, 1)
        self.assertEqual(group.blunders_per_100_moves, 10)
        self.assertEqual(group.weighted_acpl, 38)
        self.assertEqual(group.mean_player_game_acpl, 110)
        self.assertEqual(group.mean_blunders_per_player_game, .5)
        cov = tables["analysis_coverage"].set_index("rating_lower").loc[1200]
        self.assertEqual(cov.all_player_games, 3)
        self.assertAlmostEqual(scalar(cov.analyzed_share), 2 / 3)
        self.assertEqual(quality["valid_player_games"], 2)

    def test_parser_integration_and_separate_sides(self):
        path = self.csv("parsed.csv", rows(ANALYZED))
        tables, quality = blunders.analyze([path], options(self.root))
        result = tables["blunders_by_rating"].set_index("rating_lower")
        self.assertEqual(quality["valid_player_games"], 2)
        self.assertEqual(result.loc[1800, "weighted_acpl"], round(1270 / 3, 1))
        self.assertEqual(result.loc[1600, "weighted_acpl"], 335)
        self.assertEqual(result.loc[1600, "blunders"], 1)
        self.assertEqual(result.loc[1800, "evaluated_moves"], 3)

    def test_invalid_metrics_and_missing_rating_are_excluded(self):
        template = {"has_eval": 1, "white_acpl": 50, "white_evaluated": 10, "white_blunders": 1}
        records = []
        for i, change in enumerate(({"white_evaluated": 0}, {"white_blunders": 11},
                                    {"white_acpl": "inf"}, {"white_evaluated": 1.5},
                                    {"white_blunders": -1}, {"white_acpl": -2},
                                    {"white_evaluated": 20}, {"white_acpl": ""},
                                    {"white_elo": "?"})):
            records.append(row(str(i), **(template | change)))
        tables, quality = blunders.analyze([self.csv("invalid.csv", records)], options(self.root))
        self.assertTrue(tables["blunders_by_rating"].empty)
        self.assertEqual(quality["unrated_eval_player_games"], 1)
        self.assertEqual(quality["invalid_eval_player_games"], 17)  # 8 белых + 9 чёрных без метрик

    def test_months_and_chunk_size_do_not_change_aggregate(self):
        analyzed = rows(ANALYZED)[0]
        a = self.csv("january.csv", [analyzed])
        second = analyzed | {"game_id": "newid", "utc_date": "2024-02-01"}
        b = self.csv("february.csv", [second, analyzed])
        small, q1 = blunders.analyze([a, b], options(self.root, chunksize=1))
        large, q2 = blunders.analyze([a, b], options(self.root, chunksize=100))
        for name in small:
            pd.testing.assert_frame_equal(small[name], large[name])
        self.assertEqual(q1, q2)
        self.assertEqual(q1["duplicate_games"], 1)
        self.assertEqual(set(small["coverage_by_month"].month), {"2024-01", "2024-02"})

    def test_no_evaluations_produces_empty_metrics_not_zero_errors(self):
        path = self.csv("plain.csv", [row("plain")])
        tables, quality = blunders.analyze([path], options(self.root))
        self.assertTrue(tables["blunders_by_rating"].empty)
        self.assertEqual(tables["analysis_coverage"].analyzed_player_games.sum(), 0)
        self.assertEqual(quality["valid_player_games"], 0)

    def test_missing_columns_and_unmatched_inputs_fail_clearly(self):
        path = self.root / "broken.csv"
        path.write_text("game_id,result\nx,1-0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "нет колонок"):
            speed.analyze([path], options(self.root))
        with self.assertRaisesRegex(ValueError, "Не найден"):
            common.resolve_inputs([str(self.root / "missing*.csv")])

    def test_duplicates_inside_one_chunk_keep_first(self):
        path = self.csv("dup.csv", [row("a", result="1-0"), row("a", result="0-1"), row("b"),
                                    row("a", result="1/2-1/2")])
        tables, quality = speed.analyze([path], options(self.root, speeds=list(common.SPEEDS), chunksize=10))
        blitz = tables["speed_summary"].set_index("speed").loc["blitz"]
        self.assertEqual(quality["duplicate_games"], 2)
        self.assertEqual((blitz.games, blitz.white_wins, blitz.draws), (2, 2, 0))

    def test_relabel_table_counts_pgn_label_against_time_control(self):
        path = self.csv("relabel.csv", [
            row("a", speed="classical", base_sec=600), row("b", speed="classical", base_sec=1800),
            row("c", speed="classical", base_sec=900, inc_sec=10), row("d", speed="blitz"),
            row("e", speed="correspondence", base_sec="", inc_sec="")])
        tables, quality = speed.analyze([path], options(self.root, speeds=list(common.SPEEDS)))
        table = tables["speed_relabel"].set_index(["declared_speed", "computed_speed"])
        self.assertEqual(table.loc[("classical", "rapid"), "games"], 2)
        self.assertAlmostEqual(scalar(table.loc[("classical", "rapid"), "share_of_declared"]), 2 / 3)
        self.assertEqual(table.loc[("correspondence", "correspondence"), "games"], 1)
        self.assertEqual(int(table["games"][table["relabeled"]].sum()), quality["speed_mismatches"])
        speed.write_report(tables, [path], options(self.root, speeds=list(common.SPEEDS)), quality)
        report = (self.root / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Перекодировка контролей", report)
        self.assertIn("Рапид: 2 из 2 партий", report)

    def test_wilson_interval_and_draw_conclusion(self):
        low, high = common.wilson_interval([0, 5, 50], [10, 100, 100])
        np.testing.assert_allclose(low, [0, 0.021543, 0.403832], atol=1e-5)
        np.testing.assert_allclose(high, [0.277528, 0.111754, 0.596168], atol=1e-5)
        def table(draws, games):
            frame = pd.DataFrame({"speed": ["blitz", "rapid"], "draws": draws, "games": games})
            frame["draw_share"] = frame["draws"] / frame["games"]
            frame["draw_share_ci_low"], frame["draw_share_ci_high"] = common.wilson_interval(frame["draws"], frame["games"])
            return frame
        self.assertIn("не выделяется", speed.draw_conclusion(table([345, 360], [10000, 10000])))
        self.assertIn("не пересекаются", speed.draw_conclusion(table([300, 600], [10000, 10000])))

    def test_histogram_quantile_matches_numpy(self):
        from collections import Counter
        values = [0, 1, 1, 3, 20, 55, 55, 100]
        for q in (0, .25, .5, .75, 1):
            self.assertEqual(speed.histogram_quantile(Counter(values), q), np.quantile(values, q))

    def test_cli_empty_and_unanalyzed_inputs_write_reports(self):
        for empty in (True, False):
            path = self.csv(f"empty-{empty}.csv", [] if empty else [row("plain")])
            for script in ("analyze_speed", "analyze_blunders"):
                out = self.root / f"{script}-{empty}"
                result = subprocess.run([sys.executable, "-m", f"scripts.{script}", str(path),
                                         "--out", str(out)],
                                        text=True, capture_output=True, check=False, cwd=ROOT)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((out / "summary.md").is_file())
                self.assertTrue((out / "run_metadata.json").is_file())
                self.assertGreater(len(list(out.glob("*.png"))), 0)
                if script == "analyze_blunders":
                    self.assertIn("Пригодных оценённых наблюдений нет", (out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
