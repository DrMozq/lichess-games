"""
Скачивает месячные дампы партий Lichess (сырые данные) для курсового проекта.
Источник: https://database.lichess.org/

Примеры:
    python scripts/download_data.py                 # 2013-01 (маленький, ~18 МБ)
    python scripts/download_data.py 2013-01 2014-01 # несколько месяцев

Файл сначала пишется во временный .part и переименовывается только после
полной загрузки, поэтому оборванная загрузка не оставит «готовый» битый файл.
Если .part уже есть (Ctrl+C, обрыв сети), загрузка продолжается с того места,
где остановилась. Обрывы соединения во время загрузки повторяются
автоматически. Уже скачанные месяцы пропускаются.
"""

import argparse
import re
import time
from pathlib import Path

import requests

BASE_URL = "https://database.lichess.org/standard"
OUT_DIR = Path("data/raw")
RETRIES = 5


def _total_size(response: requests.Response, offset: int) -> int:
    """Полный размер файла из ответа 200 или 206 (0, если сервер его не сообщил)."""
    match = re.search(r"/(\d+)$", response.headers.get("content-range", ""))
    if match:
        return int(match.group(1))
    length = int(response.headers.get("content-length", 0))
    return length + offset if length and response.status_code == 206 else length


def _download_once(url: str, part: Path, chunk_size: int) -> int:
    """Одна попытка: дописывает .part с текущего размера. Возвращает полный размер."""
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code == 416:  # запрошено с конца файла: .part уже полный
            return offset
        r.raise_for_status()
        if offset and r.status_code != 206:
            print(f"Сервер не поддерживает докачку, начинаю {part.name} заново")
            offset = 0
        total = _total_size(r, offset)
        if offset:
            print(f"Продолжаю {part.name} с {offset / 1e6:.1f} МБ")
        done = offset
        with open(part, "ab" if offset else "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{part.name}: {done / total:.1%}", end="", flush=True)
        print()
    return total


def download(url: str, out_path: Path, chunk_size: int = 1024 * 1024, retries: int = RETRIES) -> None:
    part = out_path.with_name(out_path.name + ".part")
    total = 0
    for attempt in range(1, retries + 1):
        try:
            total = _download_once(url, part, chunk_size)
            break
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as error:
            if attempt == retries:
                raise
            wait = min(60, 5 * attempt)
            print(f"\nОбрыв загрузки ({error.__class__.__name__}), "
                  f"повтор {attempt}/{retries - 1} через {wait} с")
            time.sleep(wait)
    size = part.stat().st_size
    if total and size != total:
        raise OSError(f"Загрузка неполная: получено {size} из {total} байт ({part}); "
                      "запустите скрипт ещё раз, чтобы докачать")
    part.replace(out_path)
    print(f"Сохранено: {out_path} ({size / 1e6:.1f} МБ)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "months",
        nargs="*",
        default=["2013-01"],
        help="месяцы в формате ГГГГ-ММ (по умолчанию 2013-01)",
    )
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for month in args.months:
        out_path = OUT_DIR / f"lichess_{month}.pgn.zst"
        if out_path.exists():
            print(f"Пропускаю {month}: {out_path} уже есть")
            continue
        url = f"{args.base_url}/lichess_db_standard_rated_{month}.pgn.zst"
        download(url, out_path)


if __name__ == "__main__":
    main()
