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
    # 今回の出走馬の中での比較（compare_level.py。タイム指数・クラスの、出走馬平均との差と順位÷頭数、出走馬全体の調子）
    "rel_tidx_mean5", "rank_tidx_mean5", "rel_tidx_best5", "rank_tidx_best5",
    "rel_prize_ratio_mean5", "rank_prize_ratio_mean5", "field_form_mean",
    # 調教評価（netkeiba の調教ページの A〜D。eval_training_years.py で 2025・2026年とも改善。2024年以前は欠損）
    "train_rank", "train_rank_rel", "train_rank_pos",
    # 血統（父・母父・母母父の産駒成績、レース前年まで、50:25:12.5 の平均。compare_pedigree_model.py で4年とも改善）
    # 値は scripts/ml/pedigree_all_features.ts（アプリと同じ app/lib/pedigree.ts の pedigreeModelValues）が作る
    "ped_heavy", "ped_wet", "ped_top3", "ped_dist",
]
# 能力指数（タイム指数 − その日の馬場差 ＋ 斤量の補正）と推定上がり3F（compare_ability.py。3年とも logloss 改善）
ABILITY_FEATURES = [
    *[f"ab_{s}" for s in ("wmean", "best", "last", "trend", "std", "n")],
    *[f"ab_{k}_{s}" for k in ("wmean", "best") for s in ("rank", "z", "dmean", "dmax")], "ab_wmean_fstd",
    *[f"{k}_{s}" for k in ("ag_dev", "ag_rank", "early_gap", "fade") for s in ("wmean", "best")],
    *[f"ag_dev_wmean_{s}" for s in ("rank", "z", "dmean", "dmax", "fstd")],
]
# 能力指数をもとにした4つの指数（compare_ability2.py。どれも3年とも logloss 改善。成長は2年）
INDEX_GROWTH = ["g_last2", "g_m3_m10", "g_last_best", "g_up_streak", "g_layoff"]
INDEX_COND_KEYS = ["dist", "venue", "shape", "going", "turn", "change"]
INDEX_COND = [f"fit_{k}{s}" for k in INDEX_COND_KEYS for s in ("", "_n")] + ["fit_surface", "cond_ab"]
INDEX_PACE = ["pace_hi", "pace_hi_n", "pace_sl", "pace_sl_n", "pos_std"]
INDEX_REL = [f"{k}_{s}" for k in ("ab_wmean", "cond_ab") for s in ("d3", "dtop3", "top3_gap", "close")] + ["cond_ab_rank", "cond_ab_dmax"]
ABILITY_FEATURES += INDEX_GROWTH + INDEX_COND + INDEX_PACE + INDEX_REL
FEATURES_BEFORE_ABILITY = list(FEATURES)  # 能力指数を入れる前の77項目（compare_*.py の比較の基準）
FEATURES = FEATURES + ABILITY_FEATURES
PEDIGREE_FEATURES = ["ped_heavy", "ped_wet", "ped_top3", "ped_dist"]
TRAIN_RANK = {"A": 4, "B": 3, "C": 2, "D": 1}

INDEX_RUNS = 20  # 4つの指数の「ふだんの能力」・条件ごとの走は、同じ芝ダの指数がある走の新しい順に最大20走
INDEX_SHRINK = 2.0  # 条件適性 = その条件での（指数 − ふだんの能力）の合計 ÷ (走数 + 2)
LAYOFF_DAYS = 90
# コースの直線距離（m）。(競馬場, 芝0/ダ1, 内回り0・外回り1・直線2) → JRA のコース紹介の A コース（compare_similar_course.py と同じ）
COURSE_STRAIGHT = {
    (1, 0, 0): 266.1, (2, 0, 0): 262.1, (3, 0, 0): 292.0, (4, 0, 0): 358.7, (4, 0, 1): 658.7, (4, 0, 2): 1000.0,
    (5, 0, 0): 525.9, (6, 0, 0): 310.0, (6, 0, 1): 310.0, (7, 0, 0): 412.5, (8, 0, 0): 328.4, (8, 0, 1): 403.7,
    (9, 0, 0): 356.5, (9, 0, 1): 473.6, (10, 0, 0): 293.0,
    (1, 1, 0): 264.3, (2, 1, 0): 260.3, (3, 1, 0): 295.7, (4, 1, 0): 353.9, (5, 1, 0): 501.6, (6, 1, 0): 308.0,
    (7, 1, 0): 410.7, (8, 1, 0): 329.1, (9, 1, 0): 352.7, (10, 1, 0): 291.3,
}
AB_RUNS = 5  # 能力指数・推定上がりは、今回と同じ芝ダの過去走（指数がある走）の新しい順に最大5走
AB_DECAY = 0.8  # 直近加重平均の重み 0.8^k
AGARI_KEYS = ("ag_dev", "ag_rank", "early_gap", "fade")
AGARI_LOWER_BETTER = ("ag_rank", "early_gap", "fade")
FIELD_COMPARE_KEYS = ("tidx_mean5", "tidx_best5", "prize_ratio_mean5")
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


def load_training_ranks(data_root: Path) -> dict[tuple[str, int], int]:
    """data/training/*.jsonl（collect_training.py）の調教評価 → (race_id, 馬番): A=4〜D=1."""
    out = {}
    for path in sorted((data_root.parent / "training").glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for h in d.get("horses", []):
                v = TRAIN_RANK.get(h.get("rank") or "")
                if h.get("umaban") and v:
                    out[(d["race_id"], int(h["umaban"]))] = v
    return out


def load_pedigree_features(data_root: Path) -> dict[tuple[str, int], dict]:
    """data/pedigree/all/runner_features.csv → (race_id, 馬番): 血統の特徴量. ファイルがなければ空（全部欠損になる）."""
    import csv
    path = data_root.parent / "pedigree" / "all" / "runner_features.csv"
    out = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[(r["race_id"], int(r["umaban"]))] = {k: float(r[k]) if r[k] != "" else None for k in PEDIGREE_FEATURES}
    return out


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return NAN if n == 0 else xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def class_band(prize) -> int:
    """1着賞金（円）の区分."""
    if not prize:
        return -1
    p = prize / 10000
    for i, cut in enumerate((600, 800, 1100, 1600, 2500, 5000)):
        if p < cut:
            return i
    return 6


def course_turn_band(venue: int, surface: int, text: str | None) -> tuple[int | None, int | None]:
    """成績表のコース欄（「芝・右・外」など）→ (回り 0=右 1=左 2=直線, 直線の区分 0=〜300m 1=〜400m 2=400m〜)."""
    text = text or ""
    turn = 0 if "右" in text else 1 if "左" in text else 2 if "直線" in text else None
    outer = 1 if "外" in text.split("・")[-1] and "外内" not in text and "内外" not in text else 0
    layout = 0 if surface == 1 else 2 if "直線" in text else outer
    st = COURSE_STRAIGHT.get((venue, surface, layout))
    return turn, None if st is None else 0 if st < 300 else 1 if st < 400 else 2


def gap_counts(order: str) -> dict[int, list[int]]:
    """通過順の文字列 → 馬番: 先頭からの区切りの数 [",", "-", "＝", 記号なし]. （ ）の中は横並びで差 0."""
    out: dict[int, list[int]] = {}
    cnt = [0, 0, 0, 0]
    depth = 0
    num = ""
    prev = None
    for ch in order.replace(" ", "") + "#":
        if ch.isdigit():
            num += ch
            continue
        if num:
            if prev == "horse" and depth == 0:
                cnt[3] += 1
            out[int(num)] = list(cnt)
            num = ""
            prev = "horse"
        if ch == "(":
            if prev == "horse" and depth == 0:
                cnt[3] += 1
            depth += 1
            prev = "open"
        elif ch == ")":
            depth -= 1
            prev = "horse"
        elif ch in (",", "-", "＝"):
            if depth == 0:
                cnt[(",", "-", "＝").index(ch)] += 1
            prev = "sep"
    return out


def add_ability(races: list[dict], data_root: Path) -> None:
    """各出走馬に 能力指数 ab と推定上がりの指標を付ける（compare_ability.py の ab2・run_agari と同じ定義）.

    能力指数 = タイム指数 − その日の馬場差 − 斤量の係数 ×（斤量 − 55）
    - その日の馬場差：同じ日・同じ競馬場・芝ダのレースで「勝ち馬のタイム指数 − そのクラス（芝ダ×1着賞金の区分）の勝ち馬の中央値」の中央値 × n/(n+2)
    - 斤量の係数：同じ馬・同じ芝ダで続けて走ったレースの差分の回帰（原点を通る）
    推定上がり（その馬の上がり3F）= 走破タイム − 先頭がゴール前600m を通った時刻（勝ちタイム − レースの上がり3F）− そのときの先頭との差
    - 先頭との差：最後から2つ目のコーナーの隊列の記号の数 × 秒（data/ml/agari_estimate.json。本物の上がりに合わせて決めた係数）
    - ag_dev：レース内の推定上がりの平均 − 自分（大きいほど速い）、ag_rank：推定上がりの順位÷完走頭数、early_gap：600m 地点での先頭との差、
      fade：自分の上がり − レースの上がり
    どれも、そのレースが終わった後に分かる値。過去走としてだけ使う（そのレース自身の特徴量には使わない）."""
    winner = {}
    for r in races:
        w = [h["tidx"] for h in r["runners"] if h["finish"] == 1 and h.get("tidx") is not None]
        if w:
            winner[r["race_id"]] = w[0]
    by_class: dict[tuple, list[float]] = defaultdict(list)
    for r in races:
        if r["race_id"] in winner:
            by_class[(r["surface"], class_band(r["prize_1st"]))].append(winner[r["race_id"]])
    class_med = {k: median(v) for k, v in by_class.items()}
    day: dict[tuple, list[float]] = defaultdict(list)
    for r in races:
        if r["race_id"] in winner:
            day[(r["date"], r["venue"], r["surface"])].append(winner[r["race_id"]] - class_med[(r["surface"], class_band(r["prize_1st"]))])
    variant = {k: median(v) * len(v) / (len(v) + 2) for k, v in day.items()}
    seq: dict[tuple, list[tuple]] = defaultdict(list)
    for r in races:
        v = variant.get((r["date"], r["venue"], r["surface"]), 0.0)
        for h in r["runners"]:
            h["ab"] = None if h.get("tidx") is None else h["tidx"] - v
            if h["ab"] is not None:
                seq[(h["horse"], r["surface"])].append((r["date"], h.get("weight"), h["ab"]))
    sxy = sxx = 0.0
    for xs in seq.values():
        xs.sort(key=lambda x: x[0])
        for a, b in zip(xs, xs[1:]):
            if a[1] and b[1] and b[1] != a[1] and abs(b[2] - a[2]) < 3:
                sxy += (b[1] - a[1]) * (b[2] - a[2])
                sxx += (b[1] - a[1]) ** 2
    k = sxy / sxx if sxx else 0.0
    coef = json.loads((data_root.parent / "ml" / "agari_estimate.json").read_text(encoding="utf-8"))["corner_last-1"]["sec_per_symbol"]
    w = [coef[","], coef["-"], coef["＝"], coef["join"]]
    for r in races:
        for h in r["runners"]:
            if h["ab"] is not None:
                h["ab"] = round(h["ab"] - k * ((h.get("weight") or 55.0) - 55.0), 4)
            for key in AGARI_KEYS:
                h[key] = None
        if not r["agari_order"] or not r["last_3f"]:
            continue
        gaps = gap_counts(r["agari_order"])
        fin = [h for h in r["runners"] if h["finish"] is not None and h.get("time_sec") and h.get("umaban") in gaps]
        if len(fin) < 3:
            continue
        lead600 = min(h["time_sec"] for h in fin) - r["last_3f"]
        est = {}
        for h in fin:
            g = sum(a * b for a, b in zip(w, gaps[h["umaban"]])) + coef["intercept"]
            est[id(h)] = (h["time_sec"] - lead600 - g, g)
        m = sum(a for a, _ in est.values()) / len(est)
        srt = sorted(a for a, _ in est.values())
        for h in fin:
            a, g = est[id(h)]
            h["ag_dev"] = round(m - a, 4)
            h["ag_rank"] = round((1 + srt.index(a)) / len(est), 4)
            h["early_gap"] = round(g, 4)
            h["fade"] = round(a - r["last_3f"], 4)


def summarize_ability(vals: list[float]) -> dict:
    """新しい順の値の列 → 直近加重平均・最高・前走・傾き（古い→新しい、3走以上）・ばらつき（標準偏差、3走以上）・走数."""
    n = len(vals)
    if n == 0:
        return {"wmean": NAN, "best": NAN, "last": NAN, "trend": NAN, "std": NAN, "n": 0}
    w = [AB_DECAY ** i for i in range(n)]
    out = {"wmean": sum(a * b for a, b in zip(w, vals)) / sum(w), "best": max(vals), "last": vals[0], "n": n,
           "trend": NAN, "std": NAN}
    if n >= 3:
        xs = list(range(n))[::-1]
        mx, my = sum(xs) / n, sum(vals) / n
        out["trend"] = sum((x - mx) * (y - my) for x, y in zip(xs, vals)) / sum((x - mx) ** 2 for x in xs)
        out["std"] = math.sqrt(sum((y - my) ** 2 for y in vals) / (n - 1))
    return out


def add_relative(rows: list[dict], key: str, with_fstd: bool) -> None:
    """出走馬の中での 順位÷頭数・偏差（平均・母標準偏差で標準化）・平均との差・最高との差（・出走馬のばらつき）."""
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    m = sum(vals) / len(vals) if vals else NAN
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else NAN
    top = max(vals) if vals else NAN
    for r in rows:
        v = r[key]
        ok = not math.isnan(v)
        r[f"{key}_rank"] = (1 + sum(1 for x in vals if x > v)) / len(rows) if ok else NAN
        r[f"{key}_z"] = (v - m) / sd if ok and sd > 0 else NAN
        r[f"{key}_dmean"] = v - m if ok else NAN
        r[f"{key}_dmax"] = v - top if ok else NAN
        if with_fstd:
            r[f"{key}_fstd"] = sd


def load_races(data_root: Path) -> list[dict]:
    """平地レースを日付・race_id 順に。各出走馬に time_diff（勝ち馬との差, 秒）と調教評価を付ける."""
    ranks = load_training_ranks(data_root)
    peds = load_pedigree_features(data_root)
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
                runners.append({**h, "finish": finish, "time_diff": td, "victim": 1 if h.get("umaban") in victims else 0,
                                "train_rank": ranks.get((r["race_id"], h.get("umaban"))),
                                "pedigree": peds.get((r["race_id"], h.get("umaban")))})
            races.append({
                "pace_index": pace_index(r.get("laps"), r.get("distance")),
                "race_id": r["race_id"], "date": r["date"], "venue": int(day["venue_code"]),
                "surface": SURFACE[r["surface"]], "distance": r.get("distance"),
                "going": norm_going(r.get("going")), "prize_1st": (r.get("prize") or [None])[0],
                "field_size": len(starters), "runners": runners,
                "agari_order": (r["corners"][-2 if len(r["corners"]) >= 2 else -1]["order"] if r.get("corners") else None),
                "last_3f": r.get("last_3f"),
                **dict(zip(("turn", "sband"), course_turn_band(int(day["venue_code"]), SURFACE[r["surface"]], r.get("course")))),
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
    add_ability(races, data_root)
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
                "ab": h.get("ab"), **{k: h.get(k) for k in AGARI_KEYS}, "turn": race["turn"], "sband": race["sband"],
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

    # 能力指数・推定上がり（今回と同じ芝ダで指数がある走、新しい順に最大5走）
    prev = [r for r in runs if r["surface"] == s and r["ab"] is not None][-AB_RUNS:][::-1]
    for k, v in summarize_ability([r["ab"] for r in prev]).items():
        f[f"ab_{k}"] = v
    ag = [r for r in prev if r["ag_dev"] is not None]
    for key in AGARI_KEYS:
        vals = [r[key] for r in ag]
        f[f"{key}_wmean"] = summarize_ability(vals)["wmean"]
        f[f"{key}_best"] = (min if key in AGARI_LOWER_BETTER else max)(vals) if vals else NAN

    f.update(index_features(runs, race))

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


def index_features(runs: list[dict], race: dict) -> dict:
    """能力指数をもとにした4つの指数（成長・条件適性・展開適性。レース内の比較は add_field_features）.
    使う走は指数がある走だけ（runs は古い順）。距離の延長・短縮と間隔は、その1つ前の「指数がある走」と比べる."""
    past = [r for r in runs if r["ab"] is not None]
    info = []  # 各走の (延長1・短縮−1・同じ0, 前の走からの日数)
    for i, r in enumerate(past):
        p = past[i - 1] if i else None
        ch = None if p is None or not p["distance"] or not r["distance"] else (
            1 if r["distance"] > p["distance"] else -1 if r["distance"] < p["distance"] else 0)
        info.append((ch, None if p is None else days_between(p["date"], r["date"])))
    same_idx = [i for i, r in enumerate(past) if r["surface"] == race["surface"]][::-1]  # 新しい順
    same = [past[i] for i in same_idx]
    ab = [r["ab"] for r in same]
    f: dict[str, float] = {}
    mean = lambda xs: sum(xs) / len(xs) if xs else NAN  # noqa: E731
    usual = mean(ab[:INDEX_RUNS])
    # 成長
    f["g_last2"] = ab[0] - ab[1] if len(ab) >= 2 else NAN
    f["g_m3_m10"] = mean(ab[:3]) - mean(ab[:10]) if len(ab) >= 4 else NAN
    f["g_last_best"] = ab[0] - max(ab[:10]) if ab else NAN
    k = 0
    while k + 1 < len(ab) and ab[k] > ab[k + 1]:
        k += 1
    f["g_up_streak"] = k if ab else NAN
    f["g_layoff"] = mean([ab[j] - ab[j + 1] for j in range(len(ab) - 1)
                          if info[same_idx[j]][1] is not None and info[same_idx[j]][1] >= LAYOFF_DAYS])
    # 条件適性
    s20 = list(zip(same[:INDEX_RUNS], same_idx[:INDEX_RUNS]))
    d = race["distance"]
    wet = race["going"] is not None and race["going"] >= 1
    last = past[-1] if past else None
    change = None if last is None or not d or not last["distance"] else (1 if d > last["distance"] else -1 if d < last["distance"] else 0)
    groups = {
        "dist": [r["ab"] for r, _ in s20 if d and r["distance"] and abs(r["distance"] - d) <= 200],
        "venue": [r["ab"] for r, _ in s20 if r["venue"] == race["venue"]],
        "shape": [r["ab"] for r, _ in s20 if race["sband"] is not None and r["sband"] == race["sband"]],
        "going": [r["ab"] for r, _ in s20 if r["going"] is not None and (r["going"] >= 1) == wet],
        "turn": [r["ab"] for r, _ in s20 if race["turn"] is not None and r["turn"] == race["turn"]],
        "change": [r["ab"] for r, i in s20 if change is not None and info[i][0] == change],
    }

    def fit(sel):
        return (NAN if not sel or math.isnan(usual) else sum(x - usual for x in sel) / (len(sel) + INDEX_SHRINK)), len(sel)

    for key, sel in groups.items():
        f[f"fit_{key}"], f[f"fit_{key}_n"] = fit(sel)
    allab = [r["ab"] for r in past[-INDEX_RUNS:]]
    f["fit_surface"] = usual - mean(allab) if ab and allab else NAN
    adj = [f[f"fit_{k}"] for k in ("dist", "venue", "going") if not math.isnan(f[f"fit_{k}"])]
    f["cond_ab"] = usual + sum(adj) if not math.isnan(usual) else NAN
    # 展開適性
    f["pace_hi"], f["pace_hi_n"] = fit([r["ab"] for r, _ in s20 if r["pace"] == 2])
    f["pace_sl"], f["pace_sl_n"] = fit([r["ab"] for r, _ in s20 if r["pace"] == 0])
    pos = [r["pos_first"] / r["field"] for r in past[-5:] if r["pos_first"] and r["field"]]
    if len(pos) >= 3:
        m = sum(pos) / len(pos)
        f["pos_std"] = math.sqrt(sum((x - m) ** 2 for x in pos) / len(pos))
    else:
        f["pos_std"] = NAN
    return f


def add_top_gap(rows: list[dict], key: str, with_rank: bool) -> None:
    """3番目に高い馬との差・上位3頭の平均との差・上位3頭の平均 − 全体の平均・平均から 0.1 以内の馬の割合（・順位・最高との差）."""
    vals = sorted((r[key] for r in rows if not math.isnan(r[key])), reverse=True)
    m = sum(vals) / len(vals) if vals else NAN
    top3 = sum(vals[:3]) / len(vals[:3]) if vals else NAN
    third = vals[2] if len(vals) >= 3 else (vals[-1] if vals else NAN)
    close = sum(1 for x in vals if abs(x - m) <= 0.1) / len(rows) if vals else NAN
    for r in rows:
        v = r[key]
        ok = not math.isnan(v)
        r[f"{key}_d3"] = v - third if ok else NAN
        r[f"{key}_dtop3"] = v - top3 if ok else NAN
        r[f"{key}_top3_gap"] = top3 - m if vals else NAN
        r[f"{key}_close"] = close
        if with_rank:
            r[f"{key}_rank"] = (1 + sum(1 for x in vals if x > v)) / len(rows) if ok else NAN
            r[f"{key}_dmax"] = v - vals[0] if ok else NAN


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
        "train_rank": runner.get("train_rank") or NAN,
    }
    ped = runner.get("pedigree") or {}
    for k in PEDIGREE_FEATURES:
        f[k] = NAN if ped.get(k) is None else ped[k]
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
    # 今回の出走馬の中での比較（compare_level.py の「今回の比較」で4年とも logloss 改善）
    for key in FIELD_COMPARE_KEYS:
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        m = sum(vals) / len(vals) if vals else NAN
        for r in rows:
            v = r[key]
            r["rel_" + key] = NAN if math.isnan(v) else v - m
            r["rank_" + key] = NAN if math.isnan(v) else (1 + sum(1 for x in vals if x > v)) / len(rows)
    forms = [r["form_raw"] for r in rows if not math.isnan(r["form_raw"])]
    for r in rows:
        r["field_form_mean"] = sum(forms) / len(forms) if forms else NAN
    # 能力指数・推定上がりの、出走馬の中での比較
    add_relative(rows, "ab_wmean", True)
    add_relative(rows, "ab_best", False)
    add_relative(rows, "ag_dev_wmean", True)
    add_top_gap(rows, "ab_wmean", False)
    add_top_gap(rows, "cond_ab", True)
    # 調教評価の、出走馬の平均との差と、自分より評価が高い馬の数÷頭数
    ranks = [r["train_rank"] for r in rows if not math.isnan(r["train_rank"])]
    mr = sum(ranks) / len(ranks) if ranks else NAN
    for r in rows:
        v = r["train_rank"]
        r["train_rank_rel"] = NAN if math.isnan(v) else v - mr
        r["train_rank_pos"] = NAN if math.isnan(v) else sum(1 for x in ranks if x > v) / len(rows)


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
                    "race_id": race["race_id"], "date": race["date"], "horse": h["horse"], "turn": race["turn"], "sband": race["sband"],
                    "finish": h["finish"], "popularity": h.get("popularity"), "jockey": h.get("jockey"), "trainer": h.get("trainer"), "label": 1 if h["finish"] is not None and h["finish"] <= 3 else 0,
                })
                race_rows.append(f)
            add_form_deviation(race_rows)
            add_field_features(race_rows)
            rows.extend(race_rows)
        for race in same_day:  # 当日のレースは翌日以降の履歴としてだけ使う
            hist.add_race(race)
    return rows, hist
