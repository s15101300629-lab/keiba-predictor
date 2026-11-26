"""海外の走の見積もり（2026-11-24 採用。overseas_eval.py の「見積もり」。外国馬 33走・遠征帰りの日本馬 831走で logloss が一番良かった）.

    python scripts/ml/overseas.py build   # → data/overseas/impute.json（式）・data/overseas/runs.json（racedb で分かった馬ごとの値）

海外の走（netkeiba の戦績のうち JRA・地方の競馬場でないもの）を過去走に足すときに、
- 1着賞金：欧州・香港・ドバイ（impute.json の equal_venues）の格のあるレース（G1〜OP）は、JRA の同じ格の値（G1 17000万円など。2026-11-26、依頼者の指示で欧州・香港・ドバイの G1 を日本の G1 と同格に）。
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


IFHA_PATH = ROOT / "data" / "overseas" / "ifha.json"
GRADES = [(r"\((?:GI|G1)\)", 1), (r"\((?:GII|G2)\)", 2), (r"\((?:GIII|G3)\)", 3), (r"\(L\)", 4), (r"\(OP\)", 5)]
JRA_GRADE_PRIZE = {2: 5700, 3: 4100, 4: 2400, 5: 2200}


def age_ok(rule: str, age: int | None) -> bool:
    if age is None:
        return True
    m = re.match(r"(\d)(yo)?(\+)?", rule or "")
    if not m:
        return True
    a = int(m.group(1))
    return age >= a if m.group(3) else age == a


def ifha_prize(venue: str, race_name: str, date: str, surface: int, distance: int, horse_id: str | None, default: float, sex: str | None = None) -> float:
    """IFHA の格で海外の走の1着賞金（万円）を決める（2026-11-26、依頼者の指示。scripts/ml/ifha.py）。app/api/races/[raceId]/route.ts の ifhaPrize と同じ.
    Part I の G1：トップ100のレースレーティングから（同じ年・国・競馬場・芝ダ・距離±60m・年齢の条件・牡とセン馬は牝馬限定戦を除く。年は表の範囲に収める。複数なら平均）、
    なければトップ100に入らない JRA の G1 の中央値。Part I の G2〜OP：JRA の同じ格。Part II の重賞・L：リステッド相当。Part III・ほか：格なし（default）"""
    if "ifha" not in _cache:
        _cache["ifha"] = json.loads(IFHA_PATH.read_text(encoding="utf-8")) if IFHA_PATH.exists() else None
    d = _cache["ifha"]
    if not d:
        return default
    country, tracks = next(((c, t) for pat, c, t in d["venues"] if pat in venue), (None, []))
    part = d["parts"].get(country, 3) if country else 3
    grade = next((g for pat, g in GRADES if re.search(pat, race_name)), None)
    if grade is None or part == 3:
        return default
    if part == 2:
        return 2400.0
    if grade != 1:
        return float(JRA_GRADE_PRIZE[grade])
    years = sorted({r["year"] for r in d["races"]})
    y = min(max(date[:4], years[0]), years[-1])
    age = int(date[:4]) - int(horse_id[:4]) if horse_id and horse_id[:4].isdigit() else None
    cands = [r["prize"] for r in d["races"] if r["year"] == y and r["country"] == country and (not tracks or r["track"] in tracks)
             and r["surface"] == surface and abs(r["distance"] - distance) <= 60 and age_ok(r["age"], age)
             and not (sex in ("牡", "セ") and "F" in (r.get("sex") or ""))]  # 牡・セン馬は牝馬限定戦を外す
    return sum(cands) / len(cands) if cands else float(d["unrated_g1"])


def hk_fields(run: dict, venue: str) -> dict | None:
    import hkjc  # noqa: PLC0415
    return hkjc.hk_run(venue, run["date"], run.get("ov_r", ""), run["distance"], run.get("ov_umaban", ""), str(run.get("ov_finish") or ""))


def improve(run: dict, venue: str, race_name: str, horse_id: str | None, sex: str | None = None) -> dict:
    """check_jra_day.overseas_runs の1走（History.add_race の形）を見積もりの形にする."""
    imp, runs = load()
    if not imp:
        return run
    x = (runs.get(horse_id or "") or {}).get(run["date"])
    # 1着賞金は IFHA の格で決める（2026-11-26。前は racedb の1着賞金と、主要国かどうか）
    prize = round(ifha_prize(venue, race_name, run["date"], run["surface"], run["distance"], horse_id, run["prize"], sex), 1)
    td = run["time_diff"]
    if x and x.get("time_diff") is not None and run["finish"]:
        td = x["time_diff"]
    out = {**run, "prize": prize, "time_diff": td}
    # 香港の走：公式の区間タイムから上がり・位置取り・タイム差（hkjc.py。2026-11-26、依頼者の指示）
    hk = hk_fields(run, venue)
    if hk:
        out.update({k: hk[k] for k in ("ag_dev", "ag_rank", "early_gap", "fade", "pos_first", "pos_last")})
        if run["finish"]:
            out["time_diff"] = td = hk["time_diff"]
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
                                  "prize_cap": 50000,
                                  "equal_venues": (json.loads(IMPUTE.read_text(encoding="utf-8")).get("equal_venues", []) if IMPUTE.exists() else [])}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
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
