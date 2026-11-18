"""成績表データの出走馬について、JRA 競走馬情報から馬ID・血統を取得する.

出力 (``data/jra-pedigree``):

* ``race_horses/{年}.csv``  1行 = 1出走馬。race_id・馬番 → 馬ID の対応
* ``horses.csv``            1行 = 1頭。父・母・母の父・母の母など

どちらも追記型で、取得済みのレース・馬は再取得しない（``--refresh-horses`` で馬を再取得）。
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .jradb import JraDb

RACE_HORSE_COLUMNS = ["race_id", "umaban", "horse", "horse_id", "jradb_cname"]
HORSE_COLUMNS = [
    "horse_id", "name", "name_en", "country", "sex", "color", "birth_date",
    "sire", "sire_id", "dam", "dam_id", "damsire", "damdam",
    "breeder", "region", "market", "retired_date", "fetched_at",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def load_race_horses(root: Path, year: int) -> list[dict]:
    return _read_csv(root / "race_horses" / f"{year}.csv")


def save_race_horses(root: Path, year: int, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["race_id"], int(r["umaban"])))
    _write_csv(root / "race_horses" / f"{year}.csv", RACE_HORSE_COLUMNS, rows)


def load_horses(root: Path) -> dict[str, dict]:
    return {r["horse_id"]: r for r in _read_csv(root / "horses.csv")}


def save_horses(root: Path, horses: dict[str, dict]) -> None:
    _write_csv(root / "horses.csv", HORSE_COLUMNS, [horses[k] for k in sorted(horses)])


def _seiseki_races(seiseki_root: Path, year: int) -> dict[str, dict]:
    """race_id -> {venue_code, kai, day, date, horses: {umaban: name}} from the converted JSON."""
    races = {}
    for p in sorted((seiseki_root / "json" / str(year)).glob("*.json")):
        day = json.loads(p.read_text(encoding="utf-8"))
        for r in day["races"]:
            if not r.get("race_id") or not r.get("date"):
                continue
            races[r["race_id"]] = {
                "venue_code": day["venue_code"], "kai": day["kai"], "day": day["day"],
                "date": r["date"],
                "horses": {h["umaban"]: h["horse"] for h in r["results"] if h.get("umaban")},
            }
    return races


class Deadline:
    def __init__(self, minutes: float):
        self.at = time.monotonic() + minutes * 60 if minutes else None

    def passed(self) -> bool:
        return self.at is not None and time.monotonic() >= self.at


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fetch_race_horses(db: JraDb, root: Path, seiseki_root: Path, year: int, deadline: Deadline,
                      workers: int, limit: int) -> tuple[int, list[str]]:
    """Map every converted race of ``year`` to JRADB horse IDs. Returns (#races done, warnings)."""
    races = _seiseki_races(seiseki_root, year)
    rows = load_race_horses(root, year)
    done = {r["race_id"] for r in rows}
    pending = {rid: r for rid, r in races.items() if rid not in done}
    if limit:
        pending = dict(sorted(pending.items())[:limit])
    warnings: list[str] = []
    _log(f"{year}: {len(races)} races, {len(pending)} without horse IDs")
    if not pending:
        return 0, warnings

    by_month: dict[tuple[int, int], set[tuple[str, int, int]]] = defaultdict(set)
    for r in pending.values():
        by_month[(year, int(r["date"][5:7]))].add((r["venue_code"], r["kai"], r["day"]))

    # 1) month -> day pages -> race cnames (cheap: ~1 request per race day)
    race_cnames: dict[str, str] = {}
    for (y, m), days in sorted(by_month.items()):
        if deadline.passed():
            break
        try:
            listed = db.month_days(y, m)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{y}-{m:02d}: {e}")
            continue
        for key in sorted(days):
            if deadline.passed():
                break
            if key not in listed:
                warnings.append(f"{y}-{m:02d}: race day {key} not listed on JRADB")
                continue
            try:
                race_cnames.update(db.day_races(listed[key]))
            except Exception as e:  # noqa: BLE001
                warnings.append(f"{key}: {e}")

    targets = [rid for rid in sorted(pending) if rid in race_cnames]
    if not deadline.passed():
        missing = [rid for rid in sorted(pending) if rid not in race_cnames]
        if missing:
            warnings.append(f"{year}: {len(missing)} races not found on JRADB (e.g. {missing[0]})")

    # 2) race pages -> (umaban, horse_id)
    count = 0
    mismatches = 0

    def work(rid: str):
        return rid, db.race_horses(race_cnames[rid])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for rid in targets:
            futures.append(ex.submit(lambda r=rid: None if deadline.passed() else work(r)))
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                warnings.append(str(e))
                continue
            if res is None:
                continue
            rid, horses = res
            expected = pending[rid]["horses"]
            if not horses:
                warnings.append(f"{rid}: no horse links on the JRADB result page")
                continue
            for h in horses:
                name = expected.get(h["umaban"])
                if name is not None and name.replace(" ", "") != h["horse"].replace(" ", ""):
                    mismatches += 1
                rows.append({"race_id": rid, "umaban": h["umaban"], "horse": h["horse"],
                             "horse_id": h["horse_id"], "jradb_cname": h["cname"]})
            count += 1
            if count % 100 == 0:
                save_race_horses(root, year, rows)
                _log(f"  races {count}/{len(targets)} ({db.requests} requests)")
    save_race_horses(root, year, rows)
    if mismatches:
        warnings.append(f"{year}: {mismatches} horse names differ between PDF and JRADB (gaiji etc.)")
    _log(f"{year}: mapped {count} races")
    return count, warnings


def fetch_horses(db: JraDb, root: Path, years: list[int], deadline: Deadline, workers: int,
                 limit: int, refresh: bool) -> tuple[int, list[str]]:
    horses = load_horses(root)
    cnames: dict[str, str] = {}
    for y in years:
        for r in load_race_horses(root, y):
            cnames.setdefault(r["horse_id"], r["jradb_cname"])
    targets = [hid for hid in sorted(cnames, reverse=True) if refresh or hid not in horses]
    if limit:
        targets = targets[:limit]
    _log(f"horses: {len(cnames)} referenced, {len(targets)} to fetch")
    warnings: list[str] = []
    count = 0
    stamp = time.strftime("%Y-%m-%d")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(lambda h=hid: None if deadline.passed() else db.horse(h, cnames[h]))
                   for hid in targets]
        for fut in as_completed(futures):
            try:
                info = fut.result()
            except Exception as e:  # noqa: BLE001
                warnings.append(str(e))
                continue
            if info is None:
                continue
            info["fetched_at"] = stamp
            horses[info["horse_id"]] = {k: ("" if v is None else v) for k, v in info.items()}
            count += 1
            if count % 200 == 0:
                save_horses(root, horses)
                _log(f"  horses {count}/{len(targets)} ({db.requests} requests)")
    save_horses(root, horses)
    _log(f"horses: fetched {count}")
    return count, warnings


def run(root: Path, seiseki_root: Path, years: list[int], *, wait: float, workers: int,
        max_minutes: float, limit: int, refresh_horses: bool, skip_horses: bool) -> list[str]:
    db = JraDb(wait=wait)
    deadline = Deadline(max_minutes)
    warnings: list[str] = []
    for y in years:
        if deadline.passed():
            break
        _, w = fetch_race_horses(db, root, seiseki_root, y, deadline, workers, limit)
        warnings += w
    if not skip_horses and not deadline.passed():
        all_years = sorted(int(p.stem) for p in (root / "race_horses").glob("*.csv"))
        _, w = fetch_horses(db, root, all_years, deadline, workers, limit, refresh_horses)
        warnings += w
    if deadline.passed():
        _log("time limit reached; progress has been saved and the next run continues from here")
    _log(f"{db.requests} JRADB requests, {len(warnings)} warnings")
    for w in warnings[:50]:
        _log("  " + w)
    return warnings
