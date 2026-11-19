"""成績表データ (data/jra-seiseki/json) から LightGBM 用の特徴量を作る.

アプリ側 (app/lib/ml/features.ts) も同じ定義で特徴量を計算するので、
ここを変えたら TypeScript 側も合わせて変え、parity テストを通すこと。

各特徴量は「そのレースの日付より前」のデータだけで計算する（当日の他レースも使わない）。
"""

from __future__ import annotations

import json
import re
import math
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

# 追加候補の5グループ
EXTRA_GROUPS = {
    "class": ["prize_ratio_last", "prize_ratio_mean5"],
    "trainer": ["trainer_n", "trainer_top3", "trainer_venue_n", "trainer_venue_top3"],
    "change": ["dist_change", "surface_change", "weight_change", "jockey_change"],
    "time_index": ["tidx_mean5", "tidx_best5", "tidx_last1"],
    "waku_bias": ["waku_course_n", "waku_course_diff"],
}
EXTRA_FEATURES = [f for g in EXTRA_GROUPS.values() for f in g]

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
    "wet_n", "wet_top3",  # 稍重以上（重・不良のみより年別比較で良かった。heavy_n は道悪補正の判定用に計算だけ残す）
    # 騎手（今年）
    "jockey_n", "jockey_top3", "jockey_venue_n", "jockey_venue_top3",
    "jockey_course_n", "jockey_course_top3",
    # 展開（脚質・位置取り・ペース適性・出走馬全体の先行力）
    "pos_first_mean", "pos_last_mean", "nige_rate", "senko_rate",
    "hp_n", "hp_top3", "sp_n", "sp_top3",
    "field_nige_cands", "field_front", "field_front_ratio", "rel_pos_rank",
    # 追加分（compare_features.py extra で比較し、効いた4グループを採用。枠の有利不利は4年とも悪化したので不採用）
    *[f for g in ("class", "trainer", "change", "time_index") for f in EXTRA_GROUPS[g]],
    # 不利（成績表の〔制裁〕の「被害馬」。compare_furi.py で4年とも logloss 改善）
    "last1_victim", "victim_n5",
]

NIGE_CAND_RATE = 0.4  # 近5走で最初のコーナー1番手が4割以上 → 逃げ候補
FRONT_RATIO = 1 / 3  # 最初のコーナーで前3分の1 → 先行

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


def lap_list(laps) -> list[float]:
    if not laps:
        return []
    if isinstance(laps, str):
        laps = laps.split("-")
    out = []
    for x in laps:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return []
    return out


def pace_index(laps, distance) -> float | None:
    """後半3F の平均ラップ − それより前の平均ラップ（200m あたり秒）。大きいほど前半が速い = ハイペース.
    距離が200で割り切れない場合の最初の端数ラップは除く."""
    xs = lap_list(laps)
    if distance and distance % 200 != 0 and xs:
        xs = xs[1:]
    if len(xs) < 5:
        return None
    front, back = xs[:-3], xs[-3:]
    return sum(back) / 3 - sum(front) / len(front)


def pace_group(surface: int, distance) -> str:
    d = distance or 0
    band = 0 if d <= 1400 else 1 if d <= 1800 else 2 if d <= 2200 else 3
    return f"{surface}-{band}"


def rate(top3: int, n: int) -> float:
    return top3 / n if n else NAN


def victim_umabans(notes: list[str]) -> set[int]:
    """成績表の〔制裁〕にある「(被害馬:5番・4番)」から、不利を受けた馬の馬番を取り出す."""
    out: set[int] = set()
    for note in notes:
        if "〔制裁〕" not in note:
            continue
        for grp in re.findall(r"被害馬[:：]([^)）]+)", note):
            out.update(int(u) for u in re.findall(r"(\d+)番", grp))
    return out


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
            victims = victim_umabans(r.get("notes") or [])
            runners = []
            for h in starters:
                finish = h.get("finish") if h.get("status") != "中止" else None
                td = None
                if finish is not None and h.get("time_sec") and win_time is not None:
                    td = max(0.0, round(h["time_sec"] - win_time, 1))
                runners.append({**h, "finish": finish, "time_diff": td, "victim": 1 if h.get("umaban") in victims else 0})
            races.append({
                "pace_index": pace_index(r.get("laps"), r.get("distance")),
                "race_id": r["race_id"], "date": r["date"], "venue": int(day["venue_code"]),
                "surface": SURFACE[r["surface"]], "distance": r.get("distance"),
                "going": norm_going(r.get("going")), "prize_1st": (r.get("prize") or [None])[0],
                "field_size": len(starters), "runners": runners,
            })
    races.sort(key=lambda x: (x["date"], x["race_id"]))
    # ペース判定: 芝ダ×距離帯ごとに pace_index を3等分（0=スロー 1=ミドル 2=ハイ）。結果(着順)は使わない
    groups: dict[str, list[float]] = defaultdict(list)
    for r in races:
        if r["pace_index"] is not None:
            groups[pace_group(r["surface"], r["distance"])].append(r["pace_index"])
    cuts = {}
    for g, vals in groups.items():
        vals.sort()
        cuts[g] = (vals[len(vals) // 3], vals[2 * len(vals) // 3])
    # タイム指数: 同じ 競馬場・芝ダ・距離・馬場 の勝ちタイムの中央値（10レース未満なら馬場を問わない）との差を
    # 1000m あたりの秒に直す。大きいほど速い。基準は全期間の勝ちタイムから作る（着順ラベルは使わない）
    win_times: dict[tuple, list[float]] = defaultdict(list)
    for r in races:
        wt = min((h["time_sec"] for h in r["runners"] if h["finish"] == 1 and h.get("time_sec")), default=None)
        if wt is not None and r["distance"]:
            win_times[(r["venue"], r["surface"], r["distance"], r["going"])].append(wt)
            win_times[(r["venue"], r["surface"], r["distance"], None)].append(wt)
    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    standard = {k: median(v) for k, v in win_times.items() if len(v) >= 10}
    for r in races:
        std = standard.get((r["venue"], r["surface"], r["distance"], r["going"])) or standard.get((r["venue"], r["surface"], r["distance"], None))
        for h in r["runners"]:
            t = h.get("time_sec") if h["finish"] is not None else None
            h["tidx"] = None if std is None or not t or not r["distance"] else round((std - t) / r["distance"] * 1000, 4)
    for r in races:
        c = cuts.get(pace_group(r["surface"], r["distance"]))
        r["pace"] = None if r["pace_index"] is None or c is None else (0 if r["pace_index"] < c[0] else 2 if r["pace_index"] > c[1] else 1)
    return races


class History:
    """馬ごとの過去走と、騎手ごとの今年の騎乗を、日付順に積み上げる."""

    def __init__(self):
        self.horses: dict[str, list[dict]] = defaultdict(list)
        self.jockeys: dict[str, list[dict]] = defaultdict(list)
        self.trainers: dict[str, list[dict]] = defaultdict(list)
        # (競馬場, 芝ダ, 距離) -> 枠番 -> [出走数, 3着以内数]
        self.course_waku: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.course_waku_events: dict[tuple, list[tuple]] = defaultdict(list)  # アプリ用 store の元

    def add_race(self, race: dict) -> None:
        for h in race["runners"]:
            run = {
                "date": race["date"], "venue": race["venue"], "surface": race["surface"],
                "distance": race["distance"], "going": race["going"], "field": race["field_size"],
                "finish": h["finish"], "time_diff": h["time_diff"], "pace": race["pace"],
                "pos_first": next((x for x in (h.get("corner_positions") or []) if x), None),
                "pos_last": next((x for x in reversed(h.get("corner_positions") or []) if x), None),
                "prize": None if race["prize_1st"] is None else race["prize_1st"] / 10000,
                "weight": h.get("weight"), "jockey": norm_jockey(h.get("jockey")), "tidx": h.get("tidx"),
                "trainer": norm_jockey(h.get("trainer")), "victim": h.get("victim", 0),
            }
            top3 = 1 if h["finish"] is not None and h["finish"] <= 3 else 0
            tr = norm_jockey(h.get("trainer"))  # 表記ゆれ吸収は騎手と同じ
            if tr:
                self.trainers[tr].append({"date": race["date"], "venue": race["venue"], "top3": top3})
            if h.get("waku"):
                key = (race["venue"], race["surface"], race["distance"])
                cw = self.course_waku[key][h["waku"]]
                cw[0] += 1
                cw[1] += top3
                self.course_waku_events[key].append((race["date"], h["waku"], top3))
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
    agg("wet", lambda r: r["surface"] == s and r["going"] is not None and r["going"] >= 1)  # 稍重以上
    agg("hp", lambda r: r["pace"] == 2)
    agg("sp", lambda r: r["pace"] == 0)

    # クラスの上下（今回の1着賞金 ÷ 前走 / 近5走平均）
    today_prize = None if race["prize_1st"] is None else race["prize_1st"] / 10000
    prizes = [r["prize"] for r in runs[-FORM_RUNS:] if r["prize"]]
    last_prize = runs[-1]["prize"] if runs else None
    f["prize_ratio_last"] = today_prize / last_prize if today_prize and last_prize else NAN
    f["prize_ratio_mean5"] = today_prize / (sum(prizes) / len(prizes)) if today_prize and prizes else NAN
    # 前走からの変化（騎手の乗り替わりは runner_features で計算）
    if runs:
        last = runs[-1]
        f["dist_change"] = d - last["distance"] if d is not None and last["distance"] is not None else NAN
        f["surface_change"] = 1.0 if last["surface"] != s else 0.0
    else:
        f["dist_change"] = f["surface_change"] = NAN
    # タイム指数（近5走）
    # 不利（前走で被害馬になったか・近5走で被害馬になった回数）
    f["last1_victim"] = runs[-1]["victim"] if runs else NAN
    f["victim_n5"] = sum(r["victim"] for r in runs[-FORM_RUNS:]) if runs else NAN
    tx = [r["tidx"] for r in runs[-FORM_RUNS:] if r["tidx"] is not None]
    f["tidx_mean5"] = sum(tx) / len(tx) if tx else NAN
    f["tidx_best5"] = max(tx) if tx else NAN
    f["tidx_last1"] = runs[-1]["tidx"] if runs and runs[-1]["tidx"] is not None else NAN

    # 脚質・位置取り（近5走のうち通過順位がある走）
    pos = [r for r in runs[-FORM_RUNS:] if r["pos_first"] and r["pos_last"] and r["field"]]
    if pos:
        f["pos_first_mean"] = sum(r["pos_first"] / r["field"] for r in pos) / len(pos)
        f["pos_last_mean"] = sum(r["pos_last"] / r["field"] for r in pos) / len(pos)
        f["nige_rate"] = sum(1 for r in pos if r["pos_first"] == 1) / len(pos)
        f["senko_rate"] = sum(1 for r in pos if r["pos_first"] / r["field"] <= FRONT_RATIO) / len(pos)
    else:
        for k in ("pos_first_mean", "pos_last_mean", "nige_rate", "senko_rate"):
            f[k] = NAN
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


def trainer_features(entries: list[dict], race: dict) -> dict:
    """entries は対象レースより前の出走。今年（同じ暦年）だけを集計する."""
    year = race["date"][:4]
    this_year = [r for r in entries if r["date"][:4] == year]
    venue = [r for r in this_year if r["venue"] == race["venue"]]
    return {
        "trainer_n": len(this_year), "trainer_top3": rate(sum(r["top3"] for r in this_year), len(this_year)),
        "trainer_venue_n": len(venue), "trainer_venue_top3": rate(sum(r["top3"] for r in venue), len(venue)),
    }


def waku_features(course: dict, waku) -> dict:
    """course: 枠番 -> [出走数, 3着以内数]（対象レースより前）。今回の枠の3着内率 − コース全体の3着内率."""
    if not waku or not course:
        return {"waku_course_n": 0 if not course else course.get(waku, [0, 0])[0], "waku_course_diff": NAN}
    n, t = course.get(waku, [0, 0])
    total_n = sum(v[0] for v in course.values())
    total_t = sum(v[1] for v in course.values())
    return {"waku_course_n": n, "waku_course_diff": t / n - total_t / total_n if n and total_n else NAN}


def runner_features(race: dict, runner: dict, horse_runs: list[dict], jockey_rides: list[dict],
                    trainer_entries: list[dict] = (), course: dict | None = None) -> dict:
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
    f.update(trainer_features(list(trainer_entries), race))
    f.update(waku_features(course or {}, runner.get("waku")))
    w, jk = runner.get("weight"), norm_jockey(runner.get("jockey"))
    last = horse_runs[-1] if horse_runs else None
    f["weight_change"] = w - last["weight"] if last and w and last["weight"] else NAN
    f["jockey_change"] = (0.0 if jk == last["jockey"] else 1.0) if last and jk and last["jockey"] else NAN
    return f


def add_field_features(rows: list[dict]) -> None:
    """出走馬全体の先行力（逃げ候補・先行馬の頭数）と、その中での位置取りの順位."""
    nige = sum(1 for r in rows if not math.isnan(r["nige_rate"]) and r["nige_rate"] >= NIGE_CAND_RATE)
    front = sum(1 for r in rows if not math.isnan(r["pos_first_mean"]) and r["pos_first_mean"] <= FRONT_RATIO)
    known = sorted(r["pos_first_mean"] for r in rows if not math.isnan(r["pos_first_mean"]))
    for r in rows:
        r["field_nige_cands"] = nige
        r["field_front"] = front
        r["field_front_ratio"] = front / len(rows)
        v = r["pos_first_mean"]
        r["rel_pos_rank"] = NAN if math.isnan(v) else (1 + sum(1 for x in known if x < v)) / len(rows)


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
                course = hist.course_waku.get((race["venue"], race["surface"], race["distance"]))
                f = runner_features(race, h, hist.horses.get(h["horse"], []), hist.jockeys.get(norm_jockey(h.get("jockey")), []),
                                    hist.trainers.get(norm_jockey(h.get("trainer")), []), {k: list(v) for k, v in course.items()} if course else None)
                f.update({
                    "race_id": race["race_id"], "date": race["date"], "horse": h["horse"],
                    "finish": h["finish"], "popularity": h.get("popularity"), "jockey": h.get("jockey"), "trainer": h.get("trainer"), "label": 1 if h["finish"] is not None and h["finish"] <= 3 else 0,
                })
                race_rows.append(f)
            add_form_deviation(race_rows)
            add_field_features(race_rows)
            rows.extend(race_rows)
        for race in same_day:  # 当日のレースは翌日以降の履歴としてだけ使う
            hist.add_race(race)
    return rows, hist
