"""
Сборка датасета проекта: скачивание, разбор и выборка для ML — одной командой.

Датасет проекта — январь каждого года 2013–2017 (DATASET_MONTHS): срез по годам
для динамики дебютов, контролей и ошибок. Месяцы берутся целиком: дамп
упорядочен по времени, поэтому «первые N партий» дали бы только первые дни
января (праздники) и сделали бы год несопоставимым с остальными.

Шаги (каждый пропускается, если уже сделан, — сборку можно прерывать и
запускать заново):
  1. скачать все недостающие дампы в data/raw/ (с докачкой .part);
  2. разобрать каждый в data/processed/full/games_ГГГГ-ММ.csv
     (итоги — в data/processed/full/manifest.json);
  3. собрать выборку для задачи 5: data/processed/sample/games_sample.csv —
     по --sample-per-month случайных партий из каждого месяца (воспроизводимо).

Примеры:
    python scripts/build_dataset.py                        # весь датасет проекта
    python scripts/build_dataset.py --download-only        # только докачать дампы
    python scripts/build_dataset.py 2013-01 2014-01        # свои месяцы
    python scripts/build_dataset.py --range 2013-01 2013-06
    python scripts/build_dataset.py --delete-raw           # удалять .zst после разбора
    python scripts/build_dataset.py --sample-per-month 0   # без выборки

parse_pgn.py используется без изменений.
"""

import argparse
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import download_data, parse_pgn
else:
    import download_data
    import parse_pgn

DATASET_MONTHS = ["2013-01", "2014-01", "2015-01", "2016-01", "2017-01"]
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/full")
SAMPLE_PATH = Path("data/processed/sample/games_sample.csv")
SAMPLE_PER_MONTH = 200_000
SAMPLE_SEED = 42
MONTH_RE = re.compile(r"(\d{4})-(0[1-9]|1[0-2])")

log = logging.getLogger("build_dataset")


def month(value: str) -> str:
    if not MONTH_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"месяц должен иметь вид ГГГГ-ММ: {value}")
    return value


def month_range(first: str, last: str) -> list[str]:
    """'2013-11', '2014-02' -> ['2013-11', '2013-12', '2014-01', '2014-02']."""
    y, m = map(int, first.split("-"))
    end = tuple(map(int, last.split("-")))
    if (y, m) > end:
        raise ValueError(f"начало диапазона {first} позже конца {last}")
    months = []
    while (y, m) <= end:
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def raw_path_for(raw_dir: Path, ym: str) -> Path:
    return raw_dir / f"lichess_{ym}.pgn.zst"


def csv_path_for(out_dir: Path, ym: str) -> Path:
    return out_dir / f"games_{ym}.csv"


def load_manifest(path: Path, source: str = download_data.BASE_URL) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"source": source, "months": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def download_missing(months, raw_dir: Path, out_dir: Path, base_url: str) -> None:
    """Шаг 1: докачивает дампы месяцев, для которых ещё нет ни дампа, ни CSV."""
    todo = [ym for ym in months
            if not raw_path_for(raw_dir, ym).exists() and not csv_path_for(out_dir, ym).exists()]
    if not todo:
        log.info("все дампы уже скачаны")
    for i, ym in enumerate(todo, 1):
        log.info("скачивание [%d/%d] %s", i, len(todo), ym)
        download_data.download(f"{base_url}/lichess_db_standard_rated_{ym}.pgn.zst",
                               raw_path_for(raw_dir, ym))


def build_month(ym: str, raw_dir: Path, out_dir: Path, base_url: str, delete_raw: bool):
    """Шаг 2 для одного месяца. Возвращает запись манифеста или None, если CSV уже есть."""
    csv_path = csv_path_for(out_dir, ym)
    if csv_path.exists():
        log.info("%s: %s уже есть, пропускаю", ym, csv_path)
        return None
    raw_path = raw_path_for(raw_dir, ym)
    if not raw_path.exists():
        download_data.download(f"{base_url}/lichess_db_standard_rated_{ym}.pgn.zst", raw_path)
    # Разбор во временный файл: оборванный запуск не оставит «готовый» неполный CSV.
    part = csv_path.with_name(csv_path.name + ".part")
    total, written, rejected = parse_pgn.parse_file(raw_path, part)
    part.replace(csv_path)
    entry = {
        "csv": csv_path.name,
        "raw": raw_path.name,
        "raw_bytes": raw_path.stat().st_size,
        "games_read": total,
        "games_written": written,
        "games_rejected": rejected,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if delete_raw:
        raw_path.unlink()
        entry["raw_deleted"] = True
    return entry


def count_rows(path: Path) -> int:
    with pd.read_csv(path, usecols=["game_id"], dtype=str, chunksize=500_000) as reader:
        return sum(len(chunk) for chunk in reader)


def sample_month(path: Path, n_rows: int, size: int, seed: int, ym: str):
    """Ровно min(size, n_rows) случайных строк CSV, потоково и воспроизводимо."""
    year, mon = map(int, ym.split("-"))
    rng = np.random.default_rng([seed, year, mon])
    take = np.sort(rng.choice(n_rows, size=min(size, n_rows), replace=False))
    start = 0
    with pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=200_000) as reader:
        for chunk in reader:
            stop = start + len(chunk)
            lo, hi = np.searchsorted(take, [start, stop])
            if hi > lo:
                yield chunk.iloc[take[lo:hi] - start]
            start = stop
    if start != n_rows:
        raise ValueError(f"{path}: ожидалось {n_rows} строк, прочитано {start}")


def build_sample(months, out_dir: Path, manifest: dict, sample_path: Path, size: int, seed: int) -> dict:
    """Шаг 3: выборка для задачи 5. Пересобирается, только если изменились параметры."""
    spec = {"months": list(months), "per_month": size, "seed": seed}
    current = manifest.get("sample", {})
    if sample_path.exists() and {k: current.get(k) for k in spec} == spec:
        log.info("выборка %s уже собрана с теми же параметрами, пропускаю", sample_path)
        return current
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    part = sample_path.with_name(sample_path.name + ".part")
    per_month, header = {}, True
    for ym in months:
        csv_path = csv_path_for(out_dir, ym)
        entry = manifest["months"].get(ym) or {}
        n_rows = entry.get("games_written") or count_rows(csv_path)
        taken = 0
        for rows in sample_month(csv_path, n_rows, size, seed, ym):
            rows.to_csv(part, mode="w" if header else "a", header=header, index=False)
            header = False
            taken += len(rows)
        per_month[ym] = taken
        log.info("выборка %s: %d из %d партий", ym, taken, n_rows)
    part.replace(sample_path)
    return {**spec, "path": str(sample_path), "rows_by_month": per_month,
            "rows": sum(per_month.values()),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("months", nargs="*", type=month,
                    help="месяцы ГГГГ-ММ (по умолчанию датасет проекта: " + ", ".join(DATASET_MONTHS) + ")")
    ap.add_argument("--range", nargs=2, type=month, metavar=("FIRST", "LAST"),
                    help="непрерывный диапазон месяцев включительно вместо списка")
    ap.add_argument("--download-only", action="store_true", help="только скачать недостающие дампы")
    ap.add_argument("--delete-raw", action="store_true", help="удалять .pgn.zst после разбора месяца")
    ap.add_argument("--sample-per-month", type=int, default=SAMPLE_PER_MONTH,
                    help=f"партий из каждого месяца в выборку для ML "
                         f"(по умолчанию {SAMPLE_PER_MONTH}; 0 — не собирать)")
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--sample-path", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--base-url", default=download_data.BASE_URL, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.range and args.months:
        ap.error("укажите либо список месяцев, либо --range")
    try:
        months = month_range(*args.range) if args.range else (args.months or DATASET_MONTHS)
    except ValueError as error:
        ap.error(str(error))
    months = list(dict.fromkeys(months))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("месяцы: %s", ", ".join(months))

    download_missing(months, args.raw_dir, args.out_dir, args.base_url)
    if args.download_only:
        log.info("дампы скачаны; разбор: python scripts/build_dataset.py %s", " ".join(args.months))
        return

    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path, args.base_url)
    for i, ym in enumerate(months, 1):
        log.info("разбор [%d/%d] %s", i, len(months), ym)
        entry = build_month(ym, args.raw_dir, args.out_dir, args.base_url, args.delete_raw)
        if entry:
            manifest["months"][ym] = entry
            save_manifest(manifest_path, manifest)
    manifest["dataset_months"] = months
    save_manifest(manifest_path, manifest)
    built = {ym: manifest["months"].get(ym) for ym in months}
    games = sum(e["games_written"] for e in built.values() if e)
    log.info("датасет: %d месяцев, %d партий -> %s", len(months), games, args.out_dir)
    missing = [ym for ym, e in built.items() if e is None]
    if missing:
        log.warning("нет записей манифеста для: %s (CSV собраны вручную?)", ", ".join(missing))

    if args.sample_per_month > 0:
        manifest["sample"] = build_sample(months, args.out_dir, manifest, args.sample_path,
                                          args.sample_per_month, args.seed)
        save_manifest(manifest_path, manifest)
        log.info("выборка для ML: %d партий -> %s", manifest["sample"]["rows"], args.sample_path)


if __name__ == "__main__":
    main()
