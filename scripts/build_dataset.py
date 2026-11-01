"""
Сборка полного датасета проекта: скачивание и разбор диапазона месяцев.

Полный датасет проекта — 2013-01 … 2014-12 (24 месяца). Для каждого месяца:
дамп скачивается в data/raw/ (download_data.py), разбирается в
data/processed/full/games_ГГГГ-ММ.csv (parse_pgn.py) и попадает в
data/processed/full/manifest.json. Готовые месяцы пропускаются, поэтому
прерванную сборку можно просто запустить заново.

Примеры:
    python scripts/build_dataset.py                       # весь датасет проекта
    python scripts/build_dataset.py 2013-01 2013-06       # свой диапазон
    python scripts/build_dataset.py --delete-raw          # удалять .zst после разбора

download_data.py и parse_pgn.py используются без изменений.
"""

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from . import download_data, parse_pgn
else:
    import download_data
    import parse_pgn

DEFAULT_FIRST, DEFAULT_LAST = "2013-01", "2014-12"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/full")
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


def load_manifest(path: Path, source: str = download_data.BASE_URL) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"source": source, "months": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_month(ym: str, raw_dir: Path, out_dir: Path, base_url: str, delete_raw: bool):
    """Возвращает запись манифеста или None, если месяц уже собран."""
    csv_path = out_dir / f"games_{ym}.csv"
    if csv_path.exists():
        log.info("%s: %s уже есть, пропускаю", ym, csv_path)
        return None
    raw_path = raw_dir / f"lichess_{ym}.pgn.zst"
    if raw_path.exists():
        log.info("%s: использую скачанный %s", ym, raw_path)
    else:
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
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if delete_raw:
        raw_path.unlink()
        entry["raw_deleted"] = True
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first", nargs="?", type=month, default=DEFAULT_FIRST, help=f"первый месяц ({DEFAULT_FIRST})")
    ap.add_argument("last", nargs="?", type=month, default=None,
                    help=f"последний месяц включительно (по умолчанию {DEFAULT_LAST}, "
                         "а если задан только первый — он же)")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--delete-raw", action="store_true", help="удалять .pgn.zst после разбора месяца")
    ap.add_argument("--base-url", default=download_data.BASE_URL, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.last is None:
        args.last = DEFAULT_LAST if args.first == DEFAULT_FIRST else args.first
    try:
        months = month_range(args.first, args.last)
    except ValueError as error:
        ap.error(str(error))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path, args.base_url)
    for i, ym in enumerate(months, 1):
        log.info("[%d/%d] %s", i, len(months), ym)
        entry = build_month(ym, args.raw_dir, args.out_dir, args.base_url, args.delete_raw)
        if entry:
            manifest["months"][ym] = entry
            save_manifest(manifest_path, manifest)
    built = {ym: manifest["months"].get(ym) for ym in months}
    games = sum(e["games_written"] for e in built.values() if e)
    log.info("готово: %d месяцев, %d партий в манифесте -> %s", len(months), games, args.out_dir)
    missing = [ym for ym, e in built.items() if e is None]
    if missing:
        log.warning("нет записей манифеста для: %s (CSV были собраны раньше вручную?)", ", ".join(missing))


if __name__ == "__main__":
    main()
