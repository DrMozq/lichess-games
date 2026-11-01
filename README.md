# Анализ шахматных партий с Lichess

Курсовой проект по курсу «Большие данные» (НоДиАБоИ).

**Участники:** Поляков Вячеслав, Юдин Пётр, Инюшин Михаил, гр. 5040102/60201

**Источник данных:** https://database.lichess.org/ — официальные помесячные PGN-дампы партий (лицензия CC0, открытый доступ без авторизации).

## Задачи анализа
1. Анализ популярности дебютов и её динамики по годам/месяцам
2. Зависимость результата партии от дебюта и разницы рейтингов соперников
3. Сравнение статистики по контролям времени (блиц/рапид/классика)
4. Анализ частоты грубых ошибок в зависимости от рейтинга игрока
5. Предсказание исхода партии по дебюту и рейтингам игроков

## Структура
```
scripts/download_data.py   — скачивание сырого дампа из источника
scripts/parse_pgn.py       — потоковый разбор PGN (.zst) в CSV, по строке на партию
scripts/analyze_basic.py   — первичная аналитика: сводка и графики
scripts/analyze_speed.py   — задача 3: сравнение блица, рапида и классики
scripts/analyze_blunders.py — задача 4: ошибки и ACPL в зависимости от рейтинга
scripts/analysis_common.py — общие функции задач 3–4
scripts/build_dataset.py   — сборка полного датасета: скачивание и разбор диапазона месяцев
tests/                     — тесты парсера, сборки датасета и задач 3–4
docs/                      — описание задач 3–4 и заметки о проверке
data/raw/                  — сырые .pgn.zst дампы (не хранятся в git)
data/processed/            — разобранные CSV (не хранятся в git)
data/processed/full/       — полный датасет и manifest.json
reports/                   — результаты аналитики: summary.md и графики
reports/speed/             — задача 3: таблицы, графики и выводы
reports/blunders/          — задача 4: таблицы, графики и выводы
```

## Запуск
```bash
pip install -r requirements.txt
python scripts/download_data.py 2013-01                          # скачать дамп(ы) в data/raw/
python scripts/parse_pgn.py data/raw/lichess_2013-01.pgn.zst     # -> data/processed/games_2013-01.csv
python scripts/analyze_basic.py data/processed/games_2013-01.csv # -> reports/
```
Месяцы передаются аргументами (`python scripts/download_data.py 2013-01 2014-01`), уже скачанные пропускаются.
Загрузка идёт во временный `.part`-файл, поэтому оборванная загрузка не оставляет битый файл.
Для больших дампов в `parse_pgn.py` есть `--max-games N`: файл читается потоком и обработка останавливается после N партий.

## Поля CSV
`game_id, speed, event, result, utc_date, utc_time, white_elo, black_elo, white_diff, black_diff, eco, opening, time_control, base_sec, inc_sec, termination, n_plies, has_eval, has_clock, moves_6`

Для задачи 4 (заполнены только у партий с оценками Stockfish, иначе пусто):
`white_acpl, black_acpl` – средняя потеря оценки за ход (сотые пешки), `*_evaluated` – сколько ходов учтено, `*_inaccuracies / *_mistakes / *_blunders` – число ходов с потерей [50, 100) / [100, 200) / ≥200 соответственно (категории не пересекаются), `evals` – оценка после каждого полухода через запятую (с точки зрения белых, сотые пешки, мат = ±1000).

Имена игроков не сохраняются: для задач проекта они не нужны.

## Полный датасет

Полный датасет проекта — помесячные дампы **2013-01 … 2014-12** (24 месяца):
два года дают динамику по месяцам и годам для задачи 1 и больше партий
с оценками Stockfish для задачи 4, чем один месяц. Сборка одной командой:

```bash
python scripts/build_dataset.py                    # 2013-01 … 2014-12
python scripts/build_dataset.py 2013-01 2013-06    # другой диапазон
python scripts/build_dataset.py --delete-raw       # удалять .zst после разбора
```

Каждый месяц скачивается в `data/raw/` и разбирается в
`data/processed/full/games_ГГГГ-ММ.csv`; итоги (сколько партий прочитано,
записано и отброшено) пишутся в `data/processed/full/manifest.json`.
Готовые месяцы пропускаются, поэтому оборванную сборку можно запустить заново.
`download_data.py` и `parse_pgn.py` при этом не меняются. Нужно несколько ГБ
на диске: сырые дампы плюс CSV (с `--delete-raw` — только CSV).

## Задачи 3–4

Контроли времени (задача 3) и грубые ошибки (задача 4) — Юдин Пётр.
Подробности, параметры и ограничения — в [docs/tasks_3_4.md](docs/tasks_3_4.md).

```bash
python scripts/analyze_speed.py "data/processed/full/*.csv"      # -> reports/speed/
python scripts/analyze_blunders.py "data/processed/full/*.csv"   # -> reports/blunders/
```

## Тесты

```bash
python -m tests.test_parse_pgn
python -m unittest tests.test_tasks_3_4 tests.test_build_dataset -v
```

## Статус
- [x] Репозиторий создан, первичные данные собраны
- [x] Парсер PGN → CSV, скрипт первичной аналитики
- [x] Скрипт сборки полного датасета (2013-01 … 2014-12)
- [x] Задачи 3–4: скрипты и отчёты (отчёты в репозитории пока по 2013-01)
- [ ] Полный датасет собран, отчёты перестроены на нём
- [ ] ELT-пайплайн в PostgreSQL
- [ ] Анализ по задачам 1–5
