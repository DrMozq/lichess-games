"""Проверки задач 1–2: python -m unittest tests.test_tasks_1_2 -v (или python tests/test_tasks_1_2.py).

Фикстуры синтетические: доли, остатки и эффекты дебютов проверяются по
значениям, посчитанным вручную.
"""

import argparse
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Корень проекта в sys.path: тесты запускаются и файлом, и из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import analysis_common as common
from scripts import analyze_openings as openings
from scripts import analyze_results as results
from scripts import parse_pgn
from tests.test_tasks_3_4 import row, scalar


def options(out, **kwargs):
    args = argparse.Namespace(out=out, chunksize=3, speed_source="time_control",
                              speeds=list(common.ALL_SPEEDS), top=15, min_period_games=2, min_period_days=1,
                              diff_step=50, diff_limit=600, min_games=1)
    vars(args).update(kwargs)
    return args


class TasksOneTwoTests(unittest.TestCase):
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

    def test_opening_family_and_eco_group(self):
        names = pd.Series(["Sicilian Defense: Najdorf Variation", "Philidor Defense #3",
                           "King's Gambit Accepted, MacLeod Defense", "?", "", None])
        self.assertEqual(common.opening_family(names).tolist(),
                         ["Sicilian Defense", "Philidor Defense", "King's Gambit Accepted",
                          "Unknown", "Unknown", "Unknown"])
        self.assertEqual(common.eco_group(pd.Series(["B20", "e00", "", "Z9"])).tolist(), ["B", "E", "?", "?"])

    def test_shares_are_within_period_and_periods_weigh_equally(self):
        records = [row(f"a{i}", utc_date="2013-01-05", opening="French Defense", eco="C00") for i in range(4)]
        records += [row(f"b{i}", utc_date="2014-01-05", opening="Sicilian Defense: Open", eco="B40")
                    for i in range(8)]
        records += [row(f"c{i}", utc_date="2014-01-06", opening="French Defense #2", eco="C01") for i in range(4)]
        records += [row("edge", utc_date="2012-12-31", opening="Sicilian Defense", eco="B20")]
        path = self.csv("games.csv", records)
        tables, quality = openings.analyze([path], options(self.root))
        summary = tables["opening_summary"].set_index("family")
        # В сумме 8 vs 8, но по периодам: French 100% и 33%, Sicilian 0% и 67%.
        self.assertEqual(summary.index[0], "French Defense")
        self.assertAlmostEqual(scalar(summary.loc["French Defense", "mean_share"]), (1 + 1 / 3) / 2)
        self.assertAlmostEqual(scalar(summary.loc["Sicilian Defense", "share_2013-01"]), 0.0)
        self.assertAlmostEqual(scalar(summary.loc["French Defense", "change_pp"]), -100 * 2 / 3)
        self.assertEqual(quality["games_outside_periods"], 1)
        self.assertEqual(quality["periods"], 2)
        eco = tables["eco_share_by_period"].set_index(["month", "eco_group"])
        self.assertAlmostEqual(scalar(eco.loc[("2014-01", "B"), "share"]), 2 / 3)
        openings.plots(tables, options(self.root))
        openings.write_report(tables, [path], options(self.root), quality)
        report = (self.root / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Сильнее всего выросла доля Sicilian Defense", report)
        for name in ("openings_trend.png", "openings_top.png", "eco_by_period.png", "openings_by_rating.png"):
            self.assertTrue((self.root / name).stat().st_size > 0)

    def test_partial_month_is_not_a_period(self):
        records = [row(f"j{i}", utc_date=f"2016-01-{1 + i % 31:02d}", opening="Italian Game") for i in range(62)]
        records += [row(f"d{i}", utc_date="2015-12-31", opening="French Defense") for i in range(40)]
        path = self.csv("games.csv", records)
        tables, quality = openings.analyze([path], options(self.root, min_period_days=20))
        self.assertEqual(openings.periods_of(tables), ["2016-01"])
        self.assertEqual(quality["games_outside_periods"], 40)
        self.assertNotIn("French Defense", set(tables["opening_summary"]["family"]))

    def test_elo_expected(self):
        np.testing.assert_allclose(results.elo_expected([0, 400, -400]), [0.5, 10 / 11, 1 / 11])

    def test_opening_effect_removes_rating_gap_and_white_baseline(self):
        same = {"white_elo": 1500, "black_elo": 1500}
        records = [
            row("a1", opening="Italian Game", result="1-0", **same),
            row("a2", opening="Italian Game", result="1-0", **same),
            row("b1", opening="French Defense", result="1-0", **same),
            row("b2", opening="French Defense", result="0-1", **same),
            row("c1", opening="Owen Defense", result="1/2-1/2", **same),
            row("c2", opening="Owen Defense", result="1/2-1/2", **same),
            # Сильные белые: высокий счёт объясняется рейтингом, эффект дебюта 0.
            row("d1", opening="Ruy Lopez", result="1-0", white_elo=1900, black_elo=1500),
            row("u1", opening="Ruy Lopez", result="0-1", white_elo="", black_elo=1500),
        ]
        path = self.csv("games.csv", records)
        tables, quality = results.analyze([path], options(self.root))
        expected_d = float(results.elo_expected(400))
        rated_scores = [1, 1, 1, 0, 0.5, 0.5, 1]
        rated_expected = [0.5] * 6 + [expected_d]
        baseline = (sum(rated_scores) - sum(rated_expected)) / 7
        self.assertAlmostEqual(results.baseline_of(tables), baseline)
        self.assertEqual(quality["rated_games"], 7)
        opening = tables["score_by_opening"].set_index("family")
        self.assertAlmostEqual(scalar(opening.loc["Italian Game", "effect"]), 0.5 - baseline)
        self.assertAlmostEqual(scalar(opening.loc["Owen Defense", "effect"]), 0.0 - baseline)
        ruy = opening.loc["Ruy Lopez"]
        self.assertEqual(scalar(ruy["games"]), 2)
        self.assertEqual(scalar(ruy["rated_games"]), 1)
        self.assertAlmostEqual(scalar(ruy["residual"]), 1 - expected_d)
        self.assertAlmostEqual(scalar(ruy["white_score"]), 0.5)
        diff = tables["score_by_rating_diff_all"]
        games_by_bin = dict(zip(diff["diff_bin"], diff["games"], strict=True))
        self.assertEqual(games_by_bin, {0.0: 6, 400.0: 1})
        close = tables["white_advantage_by_rating"].set_index(["speed", "close_band"])
        self.assertAlmostEqual(scalar(close.loc[("blitz", 1400.0), "white_score"]), 4 / 6)
        results.plots(tables, options(self.root))
        results.write_report(tables, [path], options(self.root), quality)
        report = (self.root / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Лучший для белых с поправкой на рейтинги: Italian Game", report)

    def test_rating_difference_limit_and_bins(self):
        records = [row("a", white_elo=1620, black_elo=1500), row("b", white_elo=1500, black_elo=1549),
                   row("c", white_elo=2300, black_elo=1500)]
        path = self.csv("games.csv", records)
        tables, quality = results.analyze([path], options(self.root, diff_limit=600))
        bins = sorted(tables["score_by_rating_diff_all"]["diff_bin"].tolist())
        self.assertEqual(bins, [-50.0, 100.0])
        self.assertEqual(quality["outside_diff_limit"], 1)

    def test_duplicates_across_files_counted_once(self):
        first = self.csv("a.csv", [row("same", opening="Italian Game"), row("x", opening="Ruy Lopez")])
        second = self.csv("b.csv", [row("same", opening="French Defense")])
        tables, quality = openings.analyze([first, second], options(self.root, min_period_games=1))
        summary = tables["opening_summary"].set_index("family")
        self.assertEqual(quality["duplicate_games"], 1)
        self.assertNotIn("French Defense", summary.index)


if __name__ == "__main__":
    unittest.main()
