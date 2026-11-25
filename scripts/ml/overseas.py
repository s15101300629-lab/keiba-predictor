"""海外の走の見積もり（2026-11-24 採用。overseas_eval.py の「見積もり」。外国馬 33走・遠征帰りの日本馬 831走で logloss が一番良かった）.

    python scripts/ml/overseas.py build   # → data/overseas/impute.json（式）・data/overseas/runs.json（racedb で分かった馬ごとの値）

海外の走（netkeiba の戦績のうち JRA・地方の競馬場でないもの）を過去走に足すときに、
- 1着賞金：主要国（MAJOR_VENUES）の格のあるレース（G1〜OP）は、JRA の同じ格の値（G1 17000万円など。2026-11-26、依頼者の指示で欧州・香港・ドバイの G1 を日本の G1 と同格に）。
  それ以外は racedb.com のレース結果の1着賞金を円に直した値（JRA の一番高い 5億円で頭打ち）。racedb になければ格で決めた値
  （check_jra_day.OVERSEAS_PRIZE）で、主要国（MAJOR_VENUES）以外の競馬場は格を1つ下げる（依頼者の指示「区別する」）
- 勝ち馬とのタイム差：racedb の着差（馬身）× 1馬身の秒（racedb の JRA の重賞と成績表から決めた値）。なければ着順ごとの中央値
- 能力指数・タイム指数：成績表の JRA の走で、芝ダごとに log 1着賞金・タイム差（3秒で頭打ち）・log 頭数 から当てた式
アプリ（app/api/races/[raceId]/route.ts の overseasRuns）も同じ impute.json・runs.json を読む
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMPUTE = ROOT / "data" / "overseas" / "impute.json"
RUNS = ROOT / "data" / "overseas" / "runs.json"
LOWER = {17000: 5700, 5700: 4100, 4100: 2400}
_cache: dict = {}


def load() -> tuple[dict, dict]:
    if "imp" not in _cache:
        _cache["imp"] = json.loads(IMPUTE.read_text(encoding="utf-8")) if IMPUTE.exists() else None
        _cache["runs"] = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else {}
    return _cache["imp"], _cache["runs"]


def improve(run: dict, venue: str, race_name: str, horse_id: str | None) -> dict:
    """check_jra_day.overseas_runs の1走（History.add_race の形）を見積もりの形にする."""
    imp, runs = load()
    if not imp:
        return run
    x = (runs.get(horse_id or "") or {}).get(run["date"])
    prize = run["prize"]
    major = any(m in venue for m in imp["major_venues"])
    graded = next((p for pat, p in imp["grade_prize"][:5] if re.search(pat, race_name)), None)  # G1〜OP
    if major and graded is not None:
        prize = graded  # 主要国（欧州・香港・ドバイなど）の格のあるレースは、JRA の同じ格と同じ値（2026-11-26、依頼者の指示）
    elif x and x.get("prize") is not None:
        prize = x["prize"]
    elif not major and graded is not None:
        prize = {int(k): v for k, v in imp["lower"].items()}.get(graded, graded)  # 主要国以外（impute.json の lower）
    td = run["time_diff"]
    if x and x.get("time_diff") is not None and run["finish"]:
        td = x["time_diff"]
    out = {**run, "prize": prize, "time_diff": td}
    if run["finish"] and td is not None:
        c = imp["coef"][str(run["surface"])]
        z = [1.0, math.log(max(prize, 100)), min(td, 3.0), math.log(run["field"])]
        out["ab"] = round(sum(a * b for a, b in zip(c["ab"], z)), 4)
        out["tidx"] = round(sum(a * b for a, b in zip(c["tidx"], z)), 4)
    return out


def build() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_jra_day import OVERSEAS_PRIZE, is_overseas_venue, overseas_runs  # noqa: PLC0415
    from overseas_eval import MAJOR_VENUES, load_sources, racedb_for_run  # noqa: PLC0415
    ev = json.loads((ROOT / "data/ml/overseas_eval.json").read_text(encoding="utf-8"))
    spl = ev["sec_per_length"]
    IMPUTE.write_text(json.dumps({"note": "海外の走の見積もり（scripts/ml/overseas.py）", "sec_per_length": spl, "coef": ev["coef"],
                                  "major_venues": MAJOR_VENUES, "grade_prize": OVERSEAS_PRIZE, "lower": {str(k): v for k, v in LOWER.items()},
                                  "prize_cap": 50000}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    hist, rdb, races = load_sources()
    out: dict[str, dict] = {}
    for hid, v in hist.items():
        for run in overseas_runs(v["rows"]):
            x = racedb_for_run(hid, v.get("title", ""), run["date"], rdb, races)
            if not x:
                continue
            e = {}
            if x.get("prize"):
                e["prize"] = round(min(x["prize"] / 10000, 50000), 1)
            if x.get("lengths") is not None and run["finish"]:
                e["time_diff"] = round(min(x["lengths"] * spl, 9.9), 1)
            if x.get("odds") is not None:
                e["odds"] = x["odds"]
            if e:
                out.setdefault(hid, {})[run["date"]] = e
    RUNS.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"impute.json・runs.json：{len(out)} 頭・{sum(len(v) for v in out.values())} 走", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
