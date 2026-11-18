"""Write parsed race days to the repository (per-day JSON + per-year CSV)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

RACE_COLUMNS = [
    "race_id", "date", "year", "venue_code", "venue", "kai", "day", "race_no", "race_name",
    "conditions", "weight_rule", "marks", "surface", "course", "distance", "weather", "going",
    "start_time", "starters", "runners", "prize_1st", "laps", "passing_times", "last_4f", "last_3f",
    "corners", "notes", "source_file",
]
RESULT_COLUMNS = [
    "race_id", "date", "venue", "race_no", "surface", "distance", "going",
    "finish", "status", "waku", "umaban", "horse", "horse_mark", "sex", "age", "color",
    "weight", "weight_before_allowance", "jockey", "jockey_mark", "trainer", "owner",
    "region", "breeder", "equipment", "body_weight", "body_weight_diff", "time", "time_sec",
    "margin", "odds", "popularity", "corner_positions",
]
PAYOUT_COLUMNS = ["race_id", "date", "type", "combo", "payout", "popularity", "votes", "note"]


def _join(values, sep="-") -> str:
    if not values:
        return ""
    return sep.join("" if v is None else str(v) for v in values)


def json_path(root: Path, year: int, name: str) -> Path:
    return root / "json" / str(year) / f"{name}.json"


def write_day(root: Path, day: dict) -> Path:
    path = json_path(root, day["year"], Path(day["file"]).stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(day, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def iter_days(root: Path, year: int):
    for p in sorted((root / "json" / str(year)).glob("*.json")):
        yield json.loads(p.read_text(encoding="utf-8"))


def rebuild_csv(root: Path, year: int) -> None:
    out = root / "csv" / str(year)
    out.mkdir(parents=True, exist_ok=True)
    races, results, payouts = [], [], []
    for day in iter_days(root, year):
        for r in day["races"]:
            rid = r.get("race_id")
            races.append({
                "race_id": rid, "date": r.get("date"), "year": day.get("year"),
                "venue_code": day.get("venue_code"), "venue": day.get("venue_name"),
                "kai": day.get("kai"), "day": day.get("day"), "race_no": r.get("race_no"),
                "race_name": r.get("race_name"), "conditions": r.get("conditions"),
                "weight_rule": r.get("weight_rule"), "marks": "".join(r.get("marks") or []),
                "surface": r.get("surface"), "course": r.get("course"), "distance": r.get("distance"),
                "weather": r.get("weather"), "going": r.get("going"), "start_time": r.get("start_time"),
                "starters": r.get("starters"), "runners": r.get("runners"),
                "prize_1st": (r.get("prize") or [None])[0],
                "laps": _join(r.get("laps")), "passing_times": _join(r.get("passing_times")),
                "last_4f": r.get("last_4f"), "last_3f": r.get("last_3f"),
                "corners": " ".join(f"{c['corner']}:{c['order']}" for c in r.get("corners", [])),
                "notes": " / ".join(r.get("notes") or []), "source_file": day.get("file"),
            })
            for h in r["results"]:
                results.append({
                    **{k: h.get(k) for k in RESULT_COLUMNS},
                    "race_id": rid, "date": r.get("date"), "venue": day.get("venue_name"),
                    "race_no": r.get("race_no"), "surface": r.get("surface"),
                    "distance": r.get("distance"), "going": r.get("going"),
                    "corner_positions": _join(h.get("corner_positions")),
                })
            for p in r["payouts"]:
                payouts.append({
                    "race_id": rid, "date": r.get("date"), "type": p["type"],
                    "combo": _join(p.get("combo")), "payout": p.get("payout"),
                    "popularity": p.get("popularity"), "votes": p.get("votes"), "note": p.get("note"),
                })
    races.sort(key=lambda r: r["race_id"] or "")
    results.sort(key=lambda r: (r["race_id"] or "", r["finish"] is None, r["finish"] or 0, r["umaban"] or 0))
    payouts.sort(key=lambda r: (r["race_id"] or ""))  # stable: keeps PDF order inside a race
    for name, cols, rows in (
        ("races.csv", RACE_COLUMNS, races),
        ("results.csv", RESULT_COLUMNS, results),
        ("payouts.csv", PAYOUT_COLUMNS, payouts),
    ):
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def load_manifest(root: Path) -> dict:
    p = root / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_manifest(root: Path, manifest: dict) -> None:
    p = root / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(sorted(manifest.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
