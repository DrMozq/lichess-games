"""Задача 1: популярность дебютов и её динамика по периодам (месяцам).

    python scripts/analyze_openings.py "data/processed/full/*.csv"   # -> reports/openings/

Дебют — семейство из тега Opening (до двоеточия/запятой, без «#N»), плюс буква
кода ECO как устойчивая к переименованиям проверка. Доли считаются внутри
периода, поэтому рост числа партий от года к году на них не влияет.
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


def parser():
    ap = common.argument_parser(__doc__, "reports/openings")
    ap.add_argument("--speeds", nargs="+", choices=common.ALL_SPEEDS, default=list(common.ALL_SPEEDS),
                    help="контроли времени (по умолчанию все)")
    ap.add_argument("--top", type=common.positive_int, default=15, help="сколько дебютов в топе")
    ap.add_argument("--min-period-games", type=common.positive_int, default=1000,
                    help="периоды с меньшим числом партий не анализируются")
    ap.add_argument("--min-period-days", type=common.positive_int, default=20,
                    help="периоды, покрытые меньшим числом дней, не анализируются: это пограничные "
                         "партии соседнего месяца (31 декабря в январском дампе)")
    return ap


def analyze(paths, args):
    quality = Counter()
    acc = common.Accumulator()
    days: dict[str, set[str]] = {}
    for frame in common.iter_chunks(paths, EXTRA_COLUMNS, args, quality):
        for month, dates in frame.groupby("month")["utc_date"].unique().items():
            days.setdefault(str(month), set()).update(dates)
        frame["family"] = common.opening_family(frame["opening"])
        frame["eco_group"] = common.eco_group(frame["eco"])
        frame["rating_band"] = common.rating_band((frame["white_elo"] + frame["black_elo"]) / 2)
        frame["games"] = 1
        acc.add("period_family", frame, ["month", "family"], ["games"])
        acc.add("period_eco", frame, ["month", "eco_group"], ["games"])
        acc.add("speed_family", frame, ["speed", "family"], ["games"])
        acc.add("band_family", frame, ["rating_band", "family"], ["games"])
    raw = {name: acc.result(name, keys, ["games"]) for name, keys in (
        ("period_family", ["month", "family"]), ("period_eco", ["month", "eco_group"]),
        ("speed_family", ["speed", "family"]), ("band_family", ["rating_band", "family"]))}
    day_counts = {month: len(dates) for month, dates in days.items()}
    return build_tables(raw, args, quality, day_counts), quality


def with_share(frame, group):
    frame = frame.copy()
    frame["share"] = common.ratio(frame["games"], frame.groupby(group)["games"].transform("sum"))
    return frame.sort_values([group, "games"], ascending=[True, False]).reset_index(drop=True)


def build_tables(raw, args, quality, day_counts):
    totals = raw["period_family"].groupby("month")["games"].sum()
    periods = sorted(m for m, n in totals.items() if m != "unknown" and n >= args.min_period_games
                     and day_counts.get(str(m), 0) >= args.min_period_days)
    quality["games_outside_periods"] = int(totals.drop(periods).sum())
    quality["periods"] = len(periods)

    by_period = with_share(raw["period_family"].query("month in @periods"), "month")
    wide = by_period.pivot_table(index="family", columns="month", values="share", fill_value=0.0)
    wide = wide.reindex(columns=periods, fill_value=0.0)
    games = by_period.groupby("family")["games"].sum()
    summary = pd.DataFrame({
        "family": wide.index,
        "games": games.reindex(wide.index).to_numpy(),
        "mean_share": wide.mean(axis=1).to_numpy(),  # каждый период с равным весом
    })
    summary["rank"] = summary["mean_share"].rank(ascending=False, method="first").astype(int)
    if periods:
        summary["first_share"] = wide[periods[0]].to_numpy()
        summary["last_share"] = wide[periods[-1]].to_numpy()
        summary["change_pp"] = 100 * (summary["last_share"] - summary["first_share"])
    for period in periods:
        summary[f"share_{period}"] = wide[period].to_numpy()
    summary = summary.sort_values("rank").reset_index(drop=True)

    eco = with_share(raw["period_eco"].query("month in @periods"), "month")
    speed = with_share(raw["speed_family"], "speed")
    band = with_share(raw["band_family"], "rating_band")
    return {"opening_summary": summary, "opening_share_by_period": by_period,
            "eco_share_by_period": eco, "opening_share_by_speed": speed,
            "opening_share_by_rating": band}


def top_families(tables, n):
    return tables["opening_summary"].query("family != 'Unknown'").head(n)["family"].tolist()


def periods_of(tables):
    return [c.removeprefix("share_") for c in tables["opening_summary"].columns if c.startswith("share_")]


def plots(tables, args):
    common.configure_plots()
    out = args.out
    periods = periods_of(tables)
    summary = tables["opening_summary"]
    top = top_families(tables, 8)

    fig, ax = plt.subplots(figsize=(11, 6))
    if len(periods) >= 2 and top:
        cmap = plt.get_cmap("tab10")
        for i, family in enumerate(top):
            row = summary.set_index("family").loc[family]
            ax.plot(periods, [100 * row[f"share_{p}"] for p in periods], marker="o", lw=2,
                    color=cmap(i), label=family)
        ax.set_ylabel("Доля партий в периоде, %")
        ax.set_xlabel("Период")
        ax.legend(ncols=2, frameon=False, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    else:
        common.empty_plot(ax, "Для динамики нужно хотя бы два периода")
    ax.set_title("Динамика популярности дебютов")
    common.savefig(fig, out / "openings_trend.png")

    head = summary.query("family != 'Unknown'").head(args.top)
    fig, ax = plt.subplots(figsize=(10, 0.42 * max(len(head), 4) + 1.5))
    if not head.empty:
        y = np.arange(len(head))[::-1]
        ax.barh(y, 100 * head["mean_share"], color="#4c72b0", label="средняя доля по периодам")
        if len(periods) >= 2:
            ax.scatter(100 * head["first_share"], y, color="#c44e52", zorder=3, s=28, label=periods[0])
            ax.scatter(100 * head["last_share"], y, color="#55a868", zorder=3, s=28, label=periods[-1])
        ax.set_yticks(y, head["family"])
        ax.set_xlabel("Доля партий, %")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
    else:
        common.empty_plot(ax)
    ax.set_title(f"Топ-{len(head)} дебютов")
    common.savefig(fig, out / "openings_top.png")

    eco = tables["eco_share_by_period"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if not eco.empty:
        wide = eco.pivot_table(index="month", columns="eco_group", values="share", fill_value=0.0)
        bottom = np.zeros(len(wide))
        colors = {"A": "#8172b3", "B": "#4c72b0", "C": "#dd8452", "D": "#55a868", "E": "#c44e52", "?": "#999999"}
        for group in [g for g in "ABCDE?" if g in wide.columns]:
            values = 100 * wide[group].to_numpy()
            ax.bar(wide.index, values, bottom=bottom, color=colors[group], label=common.ECO_LABELS[group])
            bottom += values
        ax.set_ylabel("Доля партий, %")
        ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        common.empty_plot(ax)
    ax.set_title("Группы ECO по периодам")
    common.savefig(fig, out / "eco_by_period.png")

    band = tables["opening_share_by_rating"]
    fig, ax = plt.subplots(figsize=(11, 6))
    totals = band.groupby("rating_band")["games"].sum()
    bands = totals[totals >= 1000].index
    if len(bands) >= 2 and top:
        cmap = plt.get_cmap("tab10")
        for i, family in enumerate(top[:6]):
            part = band[band["family"].eq(family) & band["rating_band"].isin(bands)].set_index("rating_band")
            part = part.reindex(bands, fill_value=0.0)
            ax.plot(bands + 100, 100 * part["share"], marker="o", lw=2, color=cmap(i), label=family)
        ax.set_xlabel("Средний рейтинг пары: центры групп шириной 200")
        ax.set_ylabel("Доля партий в группе, %")
        ax.legend(frameon=False, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    else:
        common.empty_plot(ax)
    ax.set_title("Популярность дебютов по уровню игроков")
    fig.text(0.5, 0.01, "Группы с ≥1000 партий", ha="center", fontsize=9, color="#555")
    common.savefig(fig, out / "openings_by_rating.png")


def pct(value):
    return f"{100 * value:.2f}%"


def write_report(tables, paths, args, quality):
    periods = periods_of(tables)
    summary = tables["opening_summary"]
    lines = common.report_preamble("Задача 1 — популярность дебютов и её динамика", paths, args, quality)
    lines += ["## Определения", "",
              ("- Дебют — семейство из тега `Opening`: часть до «:» и «,», без номера «#N» "
              "(`Sicilian Defense: Najdorf Variation` → `Sicilian Defense`)."),
              ("- Доля — доля партий периода. Период — месяц по `utc_date`. Месяцы, где партии есть "
              f"меньше чем за {args.min_period_days} дней или их меньше {args.min_period_games}, исключены: "
              "это пограничные партии соседнего месяца, например 31 декабря в январском дампе "
              f"({quality['games_outside_periods']} партий)."),
              ("- Топ ранжируется по средней доле по периодам: каждый период весит одинаково, "
              "поэтому самый большой месяц не определяет рейтинг дебютов."), ""]
    head = summary.query("family != 'Unknown'").head(args.top)
    view = pd.DataFrame({"#": head["rank"], "Дебют": head["family"],
                         "Партий": head["games"].astype(int),
                         "Средняя доля, %": 100 * head["mean_share"]})
    for period in periods:
        view[f"{period}, %"] = 100 * head[f"share_{period}"]
    if len(periods) >= 2:
        view["Изменение, п.п."] = head["change_pp"]
    lines += [f"## Топ-{len(head)} дебютов", "", common.md_table(view), ""]

    eco = tables["eco_share_by_period"]
    if not eco.empty:
        wide = eco.pivot_table(index="eco_group", columns="month", values="share", fill_value=0.0)
        wide = wide.reindex([g for g in "ABCDE?" if g in wide.index])
        eco_view = pd.DataFrame({"Группа ECO": [common.ECO_LABELS[g] for g in wide.index]})
        for period in wide.columns:
            eco_view[f"{period}, %"] = 100 * wide[period].to_numpy()
        lines += ["## Группы ECO по периодам", "",
                  ("Код ECO от названий не зависит, поэтому это проверка, что динамика семейств "
                  "не объясняется переименованиями в разных дампах."), "", common.md_table(eco_view), ""]

    lines += ["## Выводы по этой выборке", ""]
    if not head.empty:
        lead = head.iloc[0]
        lines.append(f"- Самый популярный дебют: {lead['family']} — в среднем {pct(lead['mean_share'])} партий периода.")
        top5 = head.head(5)["mean_share"].sum()
        lines.append(f"- Пять самых популярных семейств вместе дают в среднем {pct(top5)} партий.")
    if len(periods) >= 2:
        movers = summary.query("family != 'Unknown' and mean_share >= 0.005").sort_values("change_pp")
        if not movers.empty:
            up, down = movers.iloc[-1], movers.iloc[0]
            lines.append(f"- Сильнее всего выросла доля {up['family']}: {pct(up['first_share'])} → "
                         f"{pct(up['last_share'])} ({up['change_pp']:+.2f} п.п., {periods[0]} → {periods[-1]}).")
            lines.append(f"- Сильнее всего упала доля {down['family']}: {pct(down['first_share'])} → "
                         f"{pct(down['last_share'])} ({down['change_pp']:+.2f} п.п.).")
            lines.append("  Учитываются семейства со средней долей не меньше 0,5%.")
        changed = (summary.query("family != 'Unknown'").head(args.top)["first_share"].rank(ascending=False)
                   != summary.query("family != 'Unknown'").head(args.top)["last_share"].rank(ascending=False)).sum()
        lines.append(f"- В топ-{len(head)} у {int(changed)} дебютов место в первом и последнем периодах различается.")
    else:
        lines.append("- Период один: динамику по этой выборке оценить нельзя.")
    speed = tables["opening_share_by_speed"]
    if not speed.empty and not head.empty:
        lines += ["", "### Лидер по контролям времени", ""]
        rows = []
        for name in [s for s in common.ALL_SPEEDS if s in set(speed["speed"])]:
            part = speed[speed["speed"].eq(name) & speed["family"].ne("Unknown")]
            if part["games"].sum() < 1000:
                continue
            best = part.iloc[0]
            rows.append((common.LABELS[name], int(part["games"].sum()), best["family"], 100 * best["share"]))
        lines += [common.md_table(pd.DataFrame(rows, columns=["Контроль", "Партий", "Самый частый дебют", "Доля, %"])), ""]

    lines += ["## Ограничения", "",
              ("- Доли описывают выбор игроков Lichess в эти периоды, а не «силу» дебюта; "
              "состав игроков от года к году меняется вместе с ростом сайта."),
              ("- Названия дебютов берутся из PGN каждого дампа. Если Lichess переименовывал "
              "варианты, семейство может «перетекать»; таблица ECO выше служит проверкой."),
              ("- Различия долей на таком объёме почти всегда статистически заметны; важен их размер "
              "в процентных пунктах, а не сам факт отличия."), "",
              "## Графики", "", "![Динамика](openings_trend.png)", "", "![Топ](openings_top.png)", "",
              "![ECO](eco_by_period.png)", "", "![По рейтингу](openings_by_rating.png)", "",
              "## Таблицы", "",
              "- `opening_summary.csv`: все семейства — средняя доля, доля в каждом периоде, изменение.",
              "- `opening_share_by_period.csv`: партии и доли семейств по периодам (длинный формат).",
              "- `eco_share_by_period.csv`: то же для букв ECO.",
              ("- `opening_share_by_speed.csv`, `opening_share_by_rating.csv`: доли внутри контроля "
              "времени и внутри группы среднего рейтинга пары."),
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
    print(f"Готово: {args.out}/summary.md; партий: {quality['selected_games']:,}; периодов: {quality['periods']}")


if __name__ == "__main__":
    main()
