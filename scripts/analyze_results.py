"""Задача 2: результат партии в зависимости от дебюта и разницы рейтингов.

    python scripts/analyze_results.py "data/processed/full/*.csv"   # -> reports/results/

Счёт белых: победа 1, ничья 0.5, поражение 0. Чтобы отделить влияние дебюта
от разницы рейтингов, для каждой партии считается ожидаемый по формуле Эло
счёт белых E = 1 / (1 + 10^(−Δ/400)), Δ = рейтинг белых − рейтинг чёрных,
и для дебюта сравнивается средний фактический счёт со средним ожидаемым.
"""

from collections import Counter

import numpy as np
import pandas as pd

if __package__:
    from . import analysis_common as common
else:
    import analysis_common as common

plt = common.plt
EXTRA_COLUMNS = {"opening", "eco"}
SUMS = ["games", "white_wins", "draws", "black_wins", "score_sum"]
RATED_SUMS = ["rated_games", "rated_score_sum", "expected_sum", "residual_sq_sum"]


def parser():
    ap = common.argument_parser(__doc__, "reports/results")
    ap.add_argument("--speeds", nargs="+", choices=common.ALL_SPEEDS, default=list(common.ALL_SPEEDS),
                    help="контроли времени (по умолчанию все)")
    ap.add_argument("--diff-step", type=common.positive_int, default=50, help="ширина корзины разницы рейтингов")
    ap.add_argument("--diff-limit", type=common.positive_int, default=600,
                    help="анализировать |Δ| < limit; дальше редкие партии")
    ap.add_argument("--min-games", type=common.positive_int, default=1000,
                    help="минимум партий для дебюта и корзины в таблицах выводов")
    ap.add_argument("--top", type=common.positive_int, default=25, help="сколько популярных дебютов сравнивать")
    return ap


def elo_expected(diff):
    return 1.0 / (1.0 + np.power(10.0, -np.asarray(diff, dtype=float) / 400.0))


def prepare(frame, args):
    frame["games"] = 1
    for result, column in zip(common.RESULTS, ("white_wins", "draws", "black_wins"), strict=True):
        frame[column] = frame["result"].eq(result).astype(int)
    frame["score_sum"] = frame["result"].map(common.RESULT_SCORE).astype(float)
    diff = frame["white_elo"] - frame["black_elo"]
    rated = diff.notna()
    expected = pd.Series(elo_expected(diff.fillna(0)), index=frame.index)
    frame["rated_games"] = rated.astype(int)
    frame["rated_score_sum"] = frame["score_sum"].where(rated, 0.0)
    frame["expected_sum"] = expected.where(rated, 0.0)
    frame["residual_sq_sum"] = ((frame["score_sum"] - expected) ** 2).where(rated, 0.0)
    frame["family"] = common.opening_family(frame["opening"])
    frame["eco_group"] = common.eco_group(frame["eco"])
    in_range = rated & diff.abs().lt(args.diff_limit)
    frame["diff_bin"] = (np.floor(diff / args.diff_step) * args.diff_step).where(in_range)
    frame["close_band"] = common.rating_band((frame["white_elo"] + frame["black_elo"]) / 2).where(
        rated & diff.abs().lt(100))
    return frame, int((rated & ~in_range).sum())


def analyze(paths, args):
    quality = Counter()
    acc = common.Accumulator()
    values = SUMS + RATED_SUMS
    for frame in common.iter_chunks(paths, EXTRA_COLUMNS, args, quality):
        frame, outside = prepare(frame, args)
        quality["rated_games"] += int(frame["rated_games"].sum())
        quality["outside_diff_limit"] += outside
        acc.add("diff", frame, ["speed", "diff_bin"], values)
        acc.add("diff_all", frame, ["diff_bin"], values)
        acc.add("family", frame, ["family"], values)
        acc.add("eco", frame, ["eco_group"], values)
        acc.add("close", frame, ["speed", "close_band"], values)
        acc.add("period", frame, ["month"], values)
    tables = {
        "score_by_rating_diff": derive(acc.result("diff", ["speed", "diff_bin"], values)),
        "score_by_rating_diff_all": derive(acc.result("diff_all", ["diff_bin"], values)),
        "score_by_opening": derive(acc.result("family", ["family"], values)),
        "score_by_eco": derive(acc.result("eco", ["eco_group"], values)),
        "white_advantage_by_rating": derive(acc.result("close", ["speed", "close_band"], values)),
        "results_by_period": derive(acc.result("period", ["month"], values)),
    }
    for name in ("score_by_rating_diff", "score_by_rating_diff_all"):
        tables[name]["diff_center"] = tables[name]["diff_bin"] + args.diff_step / 2
    # Базовый остаток по всем партиям с рейтингами: в нём преимущество первого хода,
    # которого нет в формуле Эло. Эффект дебюта считается относительно него.
    family = tables["score_by_opening"]
    rated = family["rated_games"].sum()
    baseline = float((family["rated_score_sum"].sum() - family["expected_sum"].sum()) / rated) if rated else 0.0
    tables["baseline"] = pd.DataFrame([{
        "rated_games": rated,
        "white_score": family["rated_score_sum"].sum() / rated if rated else np.nan,
        "expected_score": family["expected_sum"].sum() / rated if rated else np.nan,
        "residual": baseline}])
    for name in ("score_by_opening", "score_by_eco"):
        frame = tables[name]
        frame["effect"] = frame["residual"] - baseline
        frame["effect_ci_low"] = frame["residual_ci_low"] - baseline
        frame["effect_ci_high"] = frame["residual_ci_high"] - baseline
    opening = tables["score_by_opening"].sort_values("games", ascending=False).reset_index(drop=True)
    opening["popularity_rank"] = np.arange(1, len(opening) + 1)
    tables["score_by_opening"] = opening
    return tables, quality


def baseline_of(tables):
    return float(tables["baseline"]["residual"].iloc[0])


def derive(frame):
    frame = frame.copy()
    for count, name in (("white_wins", "white_win_share"), ("draws", "draw_share"),
                        ("black_wins", "black_win_share")):
        frame[name] = common.ratio(frame[count], frame["games"])
    frame["white_score"] = common.ratio(frame["score_sum"], frame["games"])
    frame["expected_score"] = common.ratio(frame["expected_sum"], frame["rated_games"])
    frame["rated_white_score"] = common.ratio(frame["rated_score_sum"], frame["rated_games"])
    frame["residual"] = frame["rated_white_score"] - frame["expected_score"]
    # SE среднего остатка: дисперсия (s − E) по партиям / n.
    mean_sq = common.ratio(frame["residual_sq_sum"], frame["rated_games"])
    variance = (mean_sq - frame["residual"] ** 2).clip(lower=0)
    se = np.sqrt(variance / frame["rated_games"].replace(0, np.nan))
    frame["residual_ci_low"] = frame["residual"] - 1.96 * se
    frame["residual_ci_high"] = frame["residual"] + 1.96 * se
    return frame


def plots(tables, args):
    common.configure_plots()
    out = args.out
    diff = tables["score_by_rating_diff"]
    fig, ax = plt.subplots(figsize=(11, 6))
    shown = False
    for speed in [s for s in common.ALL_SPEEDS if s in set(diff["speed"])]:
        part = diff[diff["speed"].eq(speed) & diff["games"].ge(args.min_games)].sort_values("diff_center")
        if len(part) >= 3:
            ax.plot(part["diff_center"], part["white_score"], marker="o", ms=3, lw=1.8,
                    color=common.COLORS[speed], label=common.LABELS[speed])
            shown = True
    if shown:
        x = np.linspace(-args.diff_limit, args.diff_limit, 200)
        ax.plot(x, elo_expected(x), color="black", ls="--", lw=1.4, label="Ожидание по Эло")
        ax.axhline(0.5, color="#999", lw=0.8)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_xlabel("Δ = рейтинг белых − рейтинг чёрных")
        ax.set_ylabel("Средний счёт белых")
        ax.legend(frameon=False, ncols=2)
        ax.grid(alpha=0.3)
    else:
        common.empty_plot(ax)
    ax.set_title("Результат белых и разница рейтингов")
    fig.text(0.5, 0.01, f"Корзины по {args.diff_step}; показаны корзины с ≥{args.min_games} партиями",
             ha="center", fontsize=9, color="#555")
    common.savefig(fig, out / "score_vs_rating_diff.png")

    opening = top_openings(tables, args)
    fig, ax = plt.subplots(figsize=(10, 0.36 * max(len(opening), 4) + 1.8))
    if not opening.empty:
        view = opening.sort_values("effect")
        y = np.arange(len(view))
        colors = np.where(view["effect"] >= 0, "#4c72b0", "#c44e52")
        err = np.vstack([100 * (view["effect"] - view["effect_ci_low"]),
                         100 * (view["effect_ci_high"] - view["effect"])])
        ax.barh(y, 100 * view["effect"], xerr=err, color=colors, ecolor="#333", capsize=2)
        ax.set_yticks(y, view["family"])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Счёт белых относительно ожидания и средней партии, п.п.")
        ax.grid(axis="x", alpha=0.3)
    else:
        common.empty_plot(ax)
    ax.set_title("Дебют и результат с поправкой на рейтинги")
    common.savefig(fig, out / "opening_effect.png")

    close = tables["white_advantage_by_rating"]
    fig, ax = plt.subplots(figsize=(11, 6))
    shown = False
    for speed in [s for s in common.ALL_SPEEDS if s in set(close["speed"])]:
        part = close[close["speed"].eq(speed) & close["games"].ge(args.min_games)].sort_values("close_band")
        if len(part) >= 2:
            ax.plot(part["close_band"] + 100, 100 * part["white_score"], marker="o", lw=1.8,
                    color=common.COLORS[speed], label=common.LABELS[speed])
            shown = True
    if shown:
        ax.axhline(50, color="#999", lw=0.8)
        ax.set_xlabel("Средний рейтинг пары: центры групп шириной 200")
        ax.set_ylabel("Счёт белых, %")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
    else:
        common.empty_plot(ax)
    ax.set_title("Преимущество белых при равных соперниках (|Δ| < 100)")
    common.savefig(fig, out / "white_advantage_by_rating.png")


def top_openings(tables, args):
    opening = tables["score_by_opening"]
    opening = opening[opening["family"].ne("Unknown") & opening["rated_games"].ge(args.min_games)]
    return opening.head(args.top)


def pp(value):
    return f"{100 * value:+.2f} п.п."


def write_report(tables, paths, args, quality):
    lines = common.report_preamble("Задача 2 — результат, дебют и разница рейтингов", paths, args, quality)
    lines += ["## Определения", "",
              "- Счёт белых: победа 1, ничья 0.5, поражение 0; Δ = рейтинг белых − рейтинг чёрных.",
              ("- Ожидаемый счёт по формуле Эло: E = 1 / (1 + 10^(−Δ/400)). Lichess считает рейтинги по "
              "Glicko-2, поэтому формула Эло здесь — ориентир для сравнения, а не модель сайта."),
              ("- Остаток = средний фактический счёт белых − средний ожидаемый по рейтингам. У средней партии "
              f"он равен {pp(baseline_of(tables))}: это преимущество первого хода, которого нет в формуле Эло."),
              ("- Эффект дебюта = остаток дебюта − остаток средней партии (в процентных пунктах). Так дебют, "
              "который чаще играют против слабых соперников, не получает «преимущество» из-за рейтингов, а "
              "преимущество белых не приписывается всем дебютам сразу. 95% интервал считает партии независимыми."),
              (f"- Разница рейтингов анализируется при |Δ| < {args.diff_limit} "
              f"(дальше — {quality['outside_diff_limit']} партий), корзины по {args.diff_step}."), ""]

    overall = tables["score_by_rating_diff_all"]
    center = overall[overall["diff_bin"].isin([-args.diff_step, 0])]
    lines += ["## Разница рейтингов", ""]
    reliable = overall[overall["games"].ge(args.min_games)].sort_values("diff_bin")
    if not reliable.empty:
        view = pd.DataFrame({
            "Δ от": reliable["diff_bin"].astype(int), "Δ до": (reliable["diff_bin"] + args.diff_step).astype(int),
            "Партий": reliable["games"].astype(int), "Счёт белых, %": 100 * reliable["white_score"],
            "Ожидание Эло, %": 100 * reliable["expected_score"], "Ничьих, %": 100 * reliable["draw_share"]})
        step = max(1, len(view) // 12)
        note = f"Показана каждая {step}-я корзина. " if step > 1 else ""
        lines += [common.md_table(view.iloc[::step]), "",
                  note + f"Корзины от {args.min_games} партий; полная таблица — `score_by_rating_diff_all.csv`, "
                  "по контролям — `score_by_rating_diff.csv`.", ""]
    lines += ["## Дебют", ""]
    opening = top_openings(tables, args)
    if not opening.empty:
        view = pd.DataFrame({
            "#": opening["popularity_rank"], "Дебют": opening["family"], "Партий": opening["games"].astype(int),
            "Счёт белых, %": 100 * opening["white_score"], "Ожидание Эло, %": 100 * opening["expected_score"],
            "Эффект, п.п.": 100 * opening["effect"],
            "95% ДИ эффекта": [f"{100 * a:+.2f}…{100 * b:+.2f}" for a, b in
                               zip(opening["effect_ci_low"], opening["effect_ci_high"], strict=True)],
            "Ничьих, %": 100 * opening["draw_share"]})
        lines += [f"{len(opening)} самых популярных семейств (от {args.min_games} партий с рейтингами обоих игроков).",
                  "", common.md_table(view), ""]
    eco = tables["score_by_eco"]
    eco = eco[eco["rated_games"].ge(args.min_games)]
    if not eco.empty:
        eco = eco.set_index("eco_group").reindex([g for g in "ABCDE?" if g in set(eco["eco_group"])]).reset_index()
        view = pd.DataFrame({"Группа ECO": [common.ECO_LABELS[g] for g in eco["eco_group"]],
                             "Партий": eco["games"].astype(int), "Счёт белых, %": 100 * eco["white_score"],
                             "Эффект, п.п.": 100 * eco["effect"], "Ничьих, %": 100 * eco["draw_share"]})
        lines += ["### Группы ECO", "", common.md_table(view), ""]

    lines += ["## Выводы по этой выборке", ""]
    if not center.empty and center["games"].sum() > 0:
        score = center["score_sum"].sum() / center["games"].sum()
        lines.append(f"- При почти равных рейтингах (|Δ| < {args.diff_step}) белые набирают {100 * score:.1f}% очков: "
                     f"преимущество первого хода {pp(score - 0.5)}")
    if len(reliable) >= 2:
        hi = reliable[reliable["diff_bin"] >= 200]
        lo = reliable[reliable["diff_bin"] < -200]
        if not hi.empty and not lo.empty:
            lines.append(f"- При Δ ≥ +200 белые набирают {100 * hi['score_sum'].sum() / hi['games'].sum():.1f}%, "
                         f"при Δ < −200 — {100 * lo['score_sum'].sum() / lo['games'].sum():.1f}%.")
        gap = (reliable["white_score"] - reliable["expected_score"])
        steep = reliable.loc[reliable["diff_bin"].abs() >= 200]
        if not steep.empty:
            weaker = ((steep["white_score"] - 0.5).abs() < (steep["expected_score"] - 0.5).abs()).mean()
            lines.append(f"- Фактический счёт в среднем отличается от ожидания Эло на {100 * gap.abs().mean():.1f} п.п.; "
                         f"при |Δ| ≥ 200 он ближе к 50%, чем ожидание, в {weaker:.0%} корзин "
                         "(зависимость от рейтинга слабее формулы Эло).")
    if not opening.empty:
        sure = opening[(opening["effect_ci_low"] > 0) | (opening["effect_ci_high"] < 0)]
        best, worst = opening.loc[opening["effect"].idxmax()], opening.loc[opening["effect"].idxmin()]
        lines.append(f"- Лучший для белых с поправкой на рейтинги: {best['family']} ({pp(best['effect'])}); "
                     f"худший: {worst['family']} ({pp(worst['effect'])})")
        lines.append(f"- У {len(sure)} из {len(opening)} дебютов 95% интервал эффекта не включает 0. "
                     f"Разброс эффектов: {pp(opening['effect'].min())} … {pp(opening['effect'].max())}; "
                     "для сравнения, разница рейтингов в 100 пунктов по Эло — это "
                     f"{pp(float(elo_expected(100)) - 0.5)}")
    lines += ["", "## Ограничения", "",
              ("- Дебют выбирают игроки, а не случай: названия вроде «Owen Defense» или «King's Pawn Game» "
              "часто означают неудачные первые ходы, а не систему. Эффект — описательный, не причинный."),
              "- Семейство включает ходы обеих сторон: для «защит» эффект отражает и выбор чёрных.",
              ("- Рейтинги разных контролей — разные пулы; сравнение Δ корректно внутри партии, а кривые по "
              "контролям показаны отдельно."),
              ("- Ожидание по Эло — ориентир: Lichess использует Glicko-2 с отклонением рейтинга, и у новых "
              "игроков рейтинг неточен."), "",
              "## Графики", "", "![Разница рейтингов](score_vs_rating_diff.png)", "",
              "![Эффект дебюта](opening_effect.png)", "", "![Преимущество белых](white_advantage_by_rating.png)", "",
              "## Таблицы", "",
              "- `score_by_rating_diff_all.csv`, `score_by_rating_diff.csv`: исходы по корзинам Δ (всего и по контролям).",
              "- `score_by_opening.csv`: все семейства — исходы, ожидание Эло, остаток (`residual`), эффект (`effect`) и 95% интервалы.",
              "- `score_by_eco.csv`: то же по буквам ECO; `baseline.csv` — остаток средней партии.",
              "- `white_advantage_by_rating.csv`: счёт белых при |Δ| < 100 по уровню и контролю.",
              "- `results_by_period.csv`: исходы по месяцам — проверка стабильности.",
              "- `run_metadata.json`: входы, параметры, версии библиотек и качество данных."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = parser()
    args = ap.parse_args()
    try:
        paths = common.resolve_inputs(args.csv)
        tables, quality = analyze(paths, args)
    except (OSError, ValueError) as error:
        ap.error(str(error))
    common.save_tables(tables, args.out)
    plots(tables, args)
    write_report(tables, paths, args, quality)
    common.write_metadata(args.out, paths, args, quality, "ok" if quality["selected_games"] else "no_data")
    print(f"Готово: {args.out}/summary.md; партий: {quality['selected_games']:,}")


if __name__ == "__main__":
    main()
