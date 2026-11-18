"""成績表データ (data/jra-seiseki/json) から LightGBM 用の特徴量を作る.

アプリ側 (app/lib/ml/features.ts) も同じ定義で特徴量を計算するので、
ここを変えたら TypeScript 側も合わせて変え、parity テストを通すこと。

各特徴量は「そのレースの日付より前」のデータだけで計算する（当日の他レースも使わない）。
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

FEATURES = [
    # レース条件
    "surface", "distance", "going", "field_size", "venue", "prize_1st",
    # 馬・枠
    "waku", "umaban", "umaban_ratio", "weight", "age", "sex", "body_weight", "body_weight_diff",
    # 経歴・調子
    "n_prior", "days_since_last", "form_raw", "form_dev", "last1_finish_c", "last1_time_c",
    "form_finish_mean", "form_time_mean",
    # 適性
    "surface_n", "surface_top3", "dist_n", "dist_top3", "venue_n", "venue_top3",
    "heavy_n", "heavy_top3",
    # 騎手（今年）
    "jockey_n", "jockey_top3", "jockey_venue_n", "jockey_venue_top3",
    "jockey_course_n", "jockey_course_top3",
]

GOING = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
SURFACE = {"芝": 0, "ダート": 1}
SEX = {"牡": 0, "牝": 1, "セ": 2}
FORM_RUNS = 5
TIME_DIFF_SCALE = 2.0  # 勝ち馬と2秒差で タイム差成分 = 0
NAN = float("nan")


def norm_going(value: str | None) -> int | None:
    if not value:
        return None
    for key in ("不良", "稍重", "重", "良"):  # 長い表記から照合（「不良」を「良」と誤認しない）
        if value.endswith(key):
            return GOING[key]
    return None


def norm_jockey(name: str | None) -> str:
    """表記ゆれ吸収: 全角→半角、所属・国名の括弧、空白や '.' を除く."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name)
    s = s.split("(")[0]
    return "".join(ch for ch in s if ch not in " .・　")


def run_components(finish: int | None, field: int, time_diff: float | None) -> tuple[float, float]:
    """1走分の 着順成分・タイム差成分 (0〜1, 大きいほど良い). 競走中止は 0."""
    if finish is None or field <= 1:
        return 0.0, 0.0
    finish_c = 1.0 - (finish - 1) / (field - 1)
    if time_diff is None:
        return finish_c, 0.0
    return finish_c, min(1.0, max(0.0, 1.0 - time_diff / TIME_DIFF_SCALE))


def rate(top3: int, n: int) -> float:
    return top3 / n if n else NAN


def load_races(data_root: Path) -> list[dict]:
    """平地レースを日付・race_id 順に。各出走馬に time_diff（勝ち馬との差, 秒）を付ける."""
    races = []
    for path in sorted((data_root / "json").glob("*/*.json")):
        day = json.loads(path.read_text(encoding="utf-8"))
        for r in day["races"]:
            if r.get("surface") not in SURFACE or not r.get("race_id") or not r.get("date"):
                continue
            starters = [h for h in r["results"] if h.get("status") not in ("取消", "除外")]
            if len(starters) < 2:
                continue
            win_time = min((h["time_sec"] for h in starters if h.get("finish") and h.get("time_sec")), default=None)
            runners = []
            for h in starters:
                finish = h.get("finish") if h.get("status") != "中止" else None
                td = None
                if finish is not None and h.get("time_sec") and win_time is not None:
                    td = max(0.0, round(h["time_sec"] - win_time, 1))
                runners.append({**h, "finish": finish, "time_diff": td})
            races.append({
                "race_id": r["race_id"], "date": r["date"], "venue": int(day["venue_code"]),
                "surface": SURFACE[r["surface"]], "distance": r.get("distance"),
                "going": norm_going(r.get("going")), "prize_1st": (r.get("prize") or [None])[0],
                "field_size": len(starters), "runners": runners,
            })
    races.sort(key=lambda x: (x["date"], x["race_id"]))
    return races


class History:
    """馬ごとの過去走と、騎手ごとの今年の騎乗を、日付順に積み上げる."""

    def __init__(self):
        self.horses: dict[str, list[dict]] = defaultdict(list)
        self.jockeys: dict[str, list[dict]] = defaultdict(list)

    def add_race(self, race: dict) -> None:
        for h in race["runners"]:
            run = {
                "date": race["date"], "venue": race["venue"], "surface": race["surface"],
                "distance": race["distance"], "going": race["going"], "field": race["field_size"],
                "finish": h["finish"], "time_diff": h["time_diff"],
            }
            self.horses[h["horse"]].append(run)
            jk = norm_jockey(h.get("jockey"))
            if jk:
                self.jockeys[jk].append({
                    "date": race["date"], "venue": race["venue"], "surface": race["surface"],
                    "distance": race["distance"], "top3": 1 if h["finish"] is not None and h["finish"] <= 3 else 0,
                })


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def horse_features(runs: list[dict], race: dict) -> dict:
    """runs は対象レースより前の走（古い順）."""
    f: dict[str, float] = {"n_prior": len(runs)}
    if runs:
        f["days_since_last"] = days_between(runs[-1]["date"], race["date"])
        comps = [run_components(r["finish"], r["field"], r["time_diff"]) for r in runs[-FORM_RUNS:]]
        f["form_finish_mean"] = sum(c[0] for c in comps) / len(comps)
        f["form_time_mean"] = sum(c[1] for c in comps) / len(comps)
        f["form_raw"] = (f["form_finish_mean"] + f["form_time_mean"]) / 2
        f["last1_finish_c"], f["last1_time_c"] = comps[-1]
    else:
        for k in ("days_since_last", "form_finish_mean", "form_time_mean", "form_raw", "last1_finish_c", "last1_time_c"):
            f[k] = NAN

    def agg(prefix: str, cond) -> None:
        sel = [r for r in runs if cond(r)]
        top3 = sum(1 for r in sel if r["finish"] is not None and r["finish"] <= 3)
        f[prefix + "_n"] = len(sel)
        f[prefix + "_top3"] = rate(top3, len(sel))

    s, d = race["surface"], race["distance"]
    agg("surface", lambda r: r["surface"] == s)
    agg("dist", lambda r: r["surface"] == s and d is not None and r["distance"] is not None and abs(r["distance"] - d) <= 200)
    agg("venue", lambda r: r["surface"] == s and r["venue"] == race["venue"])
    agg("heavy", lambda r: r["surface"] == s and r["going"] is not None and r["going"] >= 2)
    return f


def jockey_features(rides: list[dict], race: dict) -> dict:
    """rides は対象レースより前の騎乗。今年（レースと同じ暦年）だけを集計する."""
    year = race["date"][:4]
    this_year = [r for r in rides if r["date"][:4] == year]
    f: dict[str, float] = {}

    def agg(prefix: str, cond) -> None:
        sel = [r for r in this_year if cond(r)]
        f[prefix + "_n"] = len(sel)
        f[prefix + "_top3"] = rate(sum(r["top3"] for r in sel), len(sel))

    agg("jockey", lambda r: True)
    agg("jockey_venue", lambda r: r["venue"] == race["venue"])
    agg("jockey_course", lambda r: r["venue"] == race["venue"] and r["surface"] == race["surface"] and r["distance"] == race["distance"])
    return f


def runner_features(race: dict, runner: dict, horse_runs: list[dict], jockey_rides: list[dict]) -> dict:
    f = {
        "surface": race["surface"], "distance": race["distance"],
        "going": NAN if race["going"] is None else race["going"],
        "field_size": race["field_size"], "venue": race["venue"],
        "prize_1st": NAN if race["prize_1st"] is None else race["prize_1st"] / 10000,
        "waku": runner.get("waku") or NAN, "umaban": runner.get("umaban") or NAN,
        "weight": runner.get("weight") or NAN, "age": runner.get("age") or NAN,
        "sex": SEX.get(runner.get("sex") or "", NAN),
        "body_weight": runner.get("body_weight") or NAN,
        "body_weight_diff": NAN if runner.get("body_weight_diff") is None else runner["body_weight_diff"],
    }
    f["umaban_ratio"] = f["umaban"] / race["field_size"] if not math.isnan(f["umaban"]) else NAN
    f.update(horse_features(horse_runs, race))
    f.update(jockey_features(jockey_rides, race))
    return f


def add_form_deviation(rows: list[dict]) -> None:
    """同じレースの出走馬の中で form_raw を偏差値化 (平均50, 標準偏差10)."""
    vals = [r["form_raw"] for r in rows if not math.isnan(r["form_raw"])]
    if len(vals) < 2:
        for r in rows:
            r["form_dev"] = NAN if math.isnan(r["form_raw"]) else 50.0
        return
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    for r in rows:
        v = r["form_raw"]
        r["form_dev"] = NAN if math.isnan(v) else (50.0 if sd == 0 else 50 + 10 * (v - mean) / sd)


def build_dataset(data_root: Path) -> tuple[list[dict], History]:
    """全平地レースの出走馬1頭=1行。label=3着以内. 戻り値の History は全期間分."""
    races = load_races(data_root)
    hist = History()
    rows: list[dict] = []
    i = 0
    while i < len(races):
        date = races[i]["date"]
        same_day = []
        while i < len(races) and races[i]["date"] == date:
            same_day.append(races[i])
            i += 1
        for race in same_day:
            race_rows = []
            for h in race["runners"]:
                f = runner_features(race, h, hist.horses.get(h["horse"], []), hist.jockeys.get(norm_jockey(h.get("jockey")), []))
                f.update({
                    "race_id": race["race_id"], "date": race["date"], "horse": h["horse"],
                    "finish": h["finish"], "popularity": h.get("popularity"), "jockey": h.get("jockey"), "label": 1 if h["finish"] is not None and h["finish"] <= 3 else 0,
                })
                race_rows.append(f)
            add_form_deviation(race_rows)
            rows.extend(race_rows)
        for race in same_day:  # 当日のレースは翌日以降の履歴としてだけ使う
            hist.add_race(race)
    return rows, hist
