"""能力指数をもとにした指数（成長・条件適性・展開適性・レース内の相対）の検証.

    cd scripts/ml && python compare_ability2.py build   # → data/ml/cache/ability2.npz（compare_ability.py build の後）
    cd scripts/ml && python compare_ability2.py eval    # → data/ml/feature_comparison_ability2.json

1走ごとの能力指数は compare_ability.py の ab2（今の指数 − その日の馬場差 ＋ 斤量の補正。1000m あたりの秒）
「ふだんの能力」＝その馬の同じ芝ダの過去走（最大20走）の ab2 の平均

成長（同じ芝ダの過去走、新しい順）
- g_last2：前走 − 2走前、g_m3_m10：近3走の平均 − 近10走の平均、g_last_best：前走 − 近10走の最高
- g_up_streak：前走から遡って、続けて指数が上がった回数
- g_layoff：過去の休み明け（間隔 90日以上）の走で「休み明けの指数 − その前の走」の平均（今回が休み明けかは days_since_last が持つ）

条件適性（その条件での ab2 − ふだんの能力。縮めるため 合計 ÷ (走数 + 2)）。それぞれ走数も入れる
- 距離（今回 ±200m）、競馬場、コースの形（直線の区分。compare_similar_course の表）、馬場（今回が稍重以上なら稍重以上、良なら良）、
  回り（右・左・直線）、距離の延長・短縮（今回が前走より長いなら、過去に延長したときの走。短いなら短縮。同じなら同距離）
- 芝ダ：この芝ダの平均 − 全部の走の平均
- cond_ab：ふだんの能力 ＋ 距離・競馬場・馬場の適性（今回の条件での能力）

展開適性
- pace_hi / pace_sl：ハイペース・スローペースの走での ab2 − ふだんの能力（縮める）。走数も
- pos_std：近5走の最初のコーナーの位置÷頭数 のばらつき（位置取りの安定性）
- ag_early：近5走の推定「600m 地点での先頭との差」と推定上がり偏差の積の平均（先行して上がりも使えるか）は推定上がりで既にあるので入れない

レース内の相対（ab2 の直近加重平均と cond_ab について）
- 3番目に高い馬との差、上位3頭の平均との差、上位3頭の平均 − 全体の平均（上位の抜け具合）、平均から 0.1 以内の馬の割合

評価：前年までで学習して 2024〜2026年を予測（tune_params.evaluate）。基準は「今の特徴量＋能力指数（ab2）＋推定上がり」
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ability import CACHE as AB_CACHE  # noqa: E402
from compare_ability import agari_columns, columns, run_abilities  # noqa: E402
from compare_course import course_of  # noqa: E402
from compare_similar_course import COURSE, straight_band  # noqa: E402
from features import FEATURES, load_races  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402

CACHE = OUT / "cache" / "ability2.npz"
NAN = float("nan")
SHRINK = 2.0
GROWTH = ["g_last2", "g_m3_m10", "g_last_best", "g_up_streak", "g_layoff"]
COND_KEYS = ["dist", "venue", "shape", "going", "turn", "change"]
COND = [f"fit_{k}{s}" for k in COND_KEYS for s in ("", "_n")] + ["fit_surface", "cond_ab"]
PACE = ["pace_hi", "pace_hi_n", "pace_sl", "pace_sl_n", "pos_std"]
REL = [f"{k}_{s}" for k in ("ab2_wmean", "cond_ab") for s in ("d3", "dtop3", "top3_gap", "close")] + ["cond_ab_rank", "cond_ab_dmax"]


def mean(xs):
    return sum(xs) / len(xs) if xs else NAN


def fit(sel, base) -> tuple[float, int]:
    if not sel or math.isnan(base):
        return NAN, len(sel)
    return sum(x - base for x in sel) / (len(sel) + SHRINK), len(sel)


def race_info():
    """race_id → (回り, 直線の区分). 成績表のコース欄から."""
    info = {}
    for path in sorted((DATA / "json").glob("*/*.json")):
        day = json.loads(path.read_text(encoding="utf-8"))
        for r in day["races"]:
            if isinstance(r, dict) and r.get("race_id"):
                text = r.get("course") or ""
                direction, outer = course_of(text)
                surf = 0 if text.startswith("芝") else 1
                layout = 2 if "直線" in text else outer
                c = COURSE.get((int(day["venue_code"]), surf, 0 if surf == 1 else layout))
                info[r["race_id"]] = (direction, straight_band(c[0]) if c else None)
    return info


def build() -> None:
    races = load_races(DATA)
    runs, _ = run_abilities(races)
    rinfo = race_info()
    rmeta = {r["race_id"]: r for r in races}
    hist = defaultdict(list)  # 馬 → 過去走（日付順）
    keys, rows = [], []
    for race in races:  # build_dataset と同じ並び（日付・race_id 順、出走馬の順）
        direction, sband = rinfo.get(race["race_id"], (None, None))
        wet = race["going"] is not None and race["going"] >= 1
        today = []
        for h in race["runners"]:
            past = [x for x in hist[h["horse"]] if x["date"] < race["date"]]
            same = [x for x in past if x["surface"] == race["surface"]][::-1]  # 新しい順
            ab = [x["ab"] for x in same]
            f = {}
            usual = mean(ab[:20])
            # 成長
            f["g_last2"] = ab[0] - ab[1] if len(ab) >= 2 else NAN
            f["g_m3_m10"] = mean(ab[:3]) - mean(ab[:10]) if len(ab) >= 4 else NAN
            f["g_last_best"] = ab[0] - max(ab[:10]) if ab else NAN
            k = 0
            while k + 1 < len(ab) and ab[k] > ab[k + 1]:
                k += 1
            f["g_up_streak"] = k if ab else NAN
            lay = [b["ab"] - a["ab"] for a, b in zip(same[1:], same[:-1]) if b["gap"] is not None and b["gap"] >= 90]
            f["g_layoff"] = mean(lay)
            # 条件適性
            s20 = same[:20]
            last = past[-1] if past else None
            change = None if last is None or not race["distance"] or not last["distance"] else (
                1 if race["distance"] > last["distance"] else -1 if race["distance"] < last["distance"] else 0)
            groups = {
                "dist": [x["ab"] for x in s20 if race["distance"] and abs(x["distance"] - race["distance"]) <= 200],
                "venue": [x["ab"] for x in s20 if x["venue"] == race["venue"]],
                "shape": [x["ab"] for x in s20 if sband is not None and x["sband"] == sband],
                "going": [x["ab"] for x in s20 if x["going"] is not None and (x["going"] >= 1) == wet],
                "turn": [x["ab"] for x in s20 if direction is not None and x["turn"] == direction],
                "change": [x["ab"] for x in s20 if change is not None and x["change"] == change],
            }
            for key, sel in groups.items():
                f[f"fit_{key}"], f[f"fit_{key}_n"] = fit(sel, usual)
            allab = [x["ab"] for x in past[-20:]]
            f["fit_surface"] = usual - mean(allab) if ab and allab else NAN
            adj = [f[f"fit_{k}"] for k in ("dist", "venue", "going") if not math.isnan(f[f"fit_{k}"])]
            f["cond_ab"] = usual + sum(adj) if not math.isnan(usual) else NAN
            # 展開
            f["pace_hi"], f["pace_hi_n"] = fit([x["ab"] for x in s20 if x["pace"] == 2], usual)
            f["pace_sl"], f["pace_sl_n"] = fit([x["ab"] for x in s20 if x["pace"] == 0], usual)
            pos = [x["pos1"] for x in past[-5:] if x["pos1"] is not None]
            f["pos_std"] = float(np.std(pos)) if len(pos) >= 3 else NAN
            today.append((h, f))
        if race["date"] >= TRAIN_FROM:
            for h, f in today:
                keys.append((race["race_id"], h["horse"]))
                rows.append(f)
        for h in race["runners"]:  # レース後に過去走として積む（同じ日の他レースは date < で除かれる）
            x = runs.get((race["race_id"], h["horse"]))
            prev = hist[h["horse"]][-1] if hist[h["horse"]] else None
            cp = h.get("corner_positions") or []
            p1 = next((c for c in cp if c), None)
            if x is not None:
                hist[h["horse"]].append({
                    "date": race["date"], "surface": race["surface"], "ab": x["ab2"], "distance": race["distance"] or 0,
                    "venue": race["venue"], "going": race["going"], "pace": race["pace"], "sband": rinfo.get(race["race_id"], (None, None))[1],
                    "turn": rinfo.get(race["race_id"], (None, None))[0],
                    "change": None if prev is None or not prev["distance"] or not race["distance"] else (
                        1 if race["distance"] > prev["distance"] else -1 if race["distance"] < prev["distance"] else 0),
                    "gap": None if prev is None else (np.datetime64(race["date"]) - np.datetime64(prev["date"])).astype(int),
                    "pos1": None if p1 is None else p1 / race["field_size"],
                })
    a = np.load(AB_CACHE)
    assert len(keys) == len(a["race_id"]) and all(k[0] == r for k, r in zip(keys, a["race_id"])), "行の並びが ability.npz と違う"
    acol = list(a["features"])
    wm = a["X"][:, acol.index("ab2_wmean")]
    by_race = defaultdict(list)
    for i, (rid, _) in enumerate(keys):
        rows[i]["ab2_wmean"] = wm[i]
        by_race[rid].append(rows[i])
    for rs in by_race.values():
        for key in ("ab2_wmean", "cond_ab"):
            vals = sorted((r[key] for r in rs if not math.isnan(r[key])), reverse=True)
            m = mean(vals)
            top3 = mean(vals[:3])
            third = vals[2] if len(vals) >= 3 else (vals[-1] if vals else NAN)
            for r in rs:
                v = r[key]
                ok = not math.isnan(v)
                r[f"{key}_d3"] = v - third if ok else NAN
                r[f"{key}_dtop3"] = v - top3 if ok else NAN
                r[f"{key}_top3_gap"] = top3 - m if vals else NAN
                r[f"{key}_close"] = sum(1 for x in vals if abs(x - m) <= 0.1) / len(rs) if vals else NAN
                if key == "cond_ab":
                    r["cond_ab_rank"] = (1 + sum(1 for x in vals if x > v)) / len(rs) if ok else NAN
                    r["cond_ab_dmax"] = v - vals[0] if ok else NAN
    cols = GROWTH + COND + PACE + REL
    np.savez_compressed(CACHE, X=np.array([[r[c] for c in cols] for r in rows], dtype=np.float64), features=np.array(cols),
                        race_id=a["race_id"])
    print(f"wrote {CACHE} rows {len(rows)}", file=sys.stderr)


def evaluate_sets() -> None:
    from tune_params import evaluate

    a = np.load(AB_CACHE)
    b = np.load(CACHE)
    acol, bcol = list(a["features"]), list(b["features"])
    base = a["X"][:, [acol.index(f) for f in FEATURES + columns("ab2") + agari_columns()]]
    pick = lambda fs: b["X"][:, [bcol.index(f) for f in fs]]  # noqa: E731
    y, race_id, finish = a["label"].astype(float), a["race_id"], a["finish"]
    year = np.array([s[:4] for s in a["date"]])
    tuned = OUT / "param_tuning.json"
    params = dict(PARAMS, **json.loads(tuned.read_text())["tuning"]["best_params"]) if tuned.exists() else PARAMS
    sets = {"基準(今＋能力指数＋推定上がり)": base, "+成長": np.hstack([base, pick(GROWTH)]),
            "+条件適性": np.hstack([base, pick(COND)]), "+展開適性": np.hstack([base, pick(PACE)]),
            "+相対(追加)": np.hstack([base, pick(REL[:4])]), "+条件適性+相対(条件込み)": np.hstack([base, pick(COND + REL)]),
            "+全部": np.hstack([base, pick(GROWTH + COND + PACE + REL)])}
    out = {"sets": {}}
    path = OUT / "feature_comparison_ability2.json"
    first = None
    for name, X in sets.items():
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X, y, year, race_id, finish, params)
        if first is None:
            first = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss"] < first[t]["logloss"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else evaluate_sets()
