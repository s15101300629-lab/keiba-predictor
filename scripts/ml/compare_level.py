"""前走レースレベル・相手関係の比較（1年ずつ外す）.

    cd scripts/ml && python compare_level.py   # → data/ml/feature_comparison_level.json

勝ち負けだけでなく、相手のそのときの調子・強さや着差も合わせて評価する。どれも予測するレースより前の情報だけを使う。

前走レースレベル
- lv_field_form / lv_field_tidx：前走の他の出走馬の、そのレース前の調子（form_raw）とタイム指数（近5走平均）の平均
- lv_after_n / lv_after_top3：前走の他の出走馬が、前走の後〜今回の前に走ったレースの数と3着内率（後続成績）
- lv_beaten_form：前走で先着した相手の、そのときの調子の平均（強い相手に先着したか）
- lv_lost_form：前走で先着された相手の、そのときの調子の平均（弱い相手に負けたか）

相手関係
- 今回の出走馬の中での比較：タイム指数（近5走平均・最高）・クラス（1着賞金の比）の、出走馬平均との差と順位÷頭数、出走馬全体の調子の平均
- 直接対決（今回の相手と過去に同じレースを走ったとき）
  - h2h_n：対戦の回数、h2h_ahead：先着した割合
  - h2h_adj：先着したか（1/0）−「そのときの両馬の調子から見た先着の見込み」の平均。調子の悪い相手に勝っても加点は小さく、調子の良い相手に勝つと大きい
  - h2h_margin：相手とのタイム差（秒、±2秒で打ち切り）の平均
"""

from __future__ import annotations

import bisect
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset, load_races  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

NAN = float("nan")
FORM_SCALE = 6.0  # 調子（0〜1）の差 → 先着の見込み（ロジスティック）の傾き


def isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def mean(xs):
    xs = [x for x in xs if not isnan(x)]
    return sum(xs) / len(xs) if xs else NAN


def main() -> None:
    # 各レースの勝ち馬とのタイム差（結果）。特徴量の行には入っていないので、成績表から取る
    time_diff = {(r["race_id"], h["horse"]): h["time_diff"] for r in load_races(DATA) for h in r["runners"]}
    rows, _ = build_dataset(DATA)
    for r in rows:
        r["time_diff"] = time_diff.get((r["race_id"], r["horse"]))
    races: dict[str, list[dict]] = {}
    by_horse: dict[str, list[dict]] = {}
    for r in rows:
        races.setdefault(r["race_id"], []).append(r)
        by_horse.setdefault(r["horse"], []).append(r)
    for runs in by_horse.values():
        runs.sort(key=lambda r: r["date"])
    dates_of = {h: [r["date"] for r in runs] for h, runs in by_horse.items()}
    labels_of = {h: [r["label"] for r in runs] for h, runs in by_horse.items()}
    # 着順の比較用（中止・不明は最後）
    order = lambda r: r["finish"] if r["finish"] is not None else 99  # noqa: E731

    for h, runs in by_horse.items():
        for i, r in enumerate(runs):
            # --- 前走レースレベル ---
            if i == 0:
                for k in ("lv_field_form", "lv_field_tidx", "lv_after_n", "lv_after_top3", "lv_beaten_form", "lv_lost_form"):
                    r[k] = NAN
            else:
                prev = runs[i - 1]
                others = [o for o in races[prev["race_id"]] if o["horse"] != h]
                r["lv_field_form"] = mean([o["form_raw"] for o in others])
                r["lv_field_tidx"] = mean([o["tidx_mean5"] for o in others])
                n = top3 = 0
                for o in others:
                    ds = dates_of[o["horse"]]
                    a = bisect.bisect_right(ds, prev["date"])
                    b = bisect.bisect_left(ds, r["date"])
                    n += b - a
                    top3 += sum(labels_of[o["horse"]][a:b])
                r["lv_after_n"] = n
                r["lv_after_top3"] = top3 / n if n else NAN
                r["lv_beaten_form"] = mean([o["form_raw"] for o in others if order(o) > order(prev)])
                r["lv_lost_form"] = mean([o["form_raw"] for o in others if order(o) < order(prev)])

    for race_id, rs in races.items():
        # --- 今回の出走馬の中での比較 ---
        for key in ("tidx_mean5", "tidx_best5", "prize_ratio_mean5"):
            vals = [r[key] for r in rs if not isnan(r[key])]
            m = sum(vals) / len(vals) if vals else NAN
            srt = sorted(vals, reverse=True)
            for r in rs:
                v = r[key]
                r[f"rel_{key}"] = v - m if not isnan(v) else NAN
                r[f"rank_{key}"] = (srt.index(v) + 1) / len(rs) if not isnan(v) else NAN
        fm = mean([r["form_raw"] for r in rs])
        for r in rs:
            r["field_form_mean"] = fm
        # --- 直接対決 ---
        names = {r["horse"] for r in rs}
        for r in rs:
            runs = by_horse[r["horse"]]
            i = bisect.bisect_left(dates_of[r["horse"]], r["date"])
            n = ahead = 0
            adj = margin = 0.0
            for past in runs[:i]:
                for o in races[past["race_id"]]:
                    if o["horse"] == r["horse"] or o["horse"] not in names:
                        continue
                    if past["finish"] is None and o["finish"] is None:
                        continue
                    n += 1
                    won = 1 if order(past) < order(o) else 0
                    ahead += won
                    fh, fo = past["form_raw"], o["form_raw"]
                    expected = 0.5 if isnan(fh) or isnan(fo) else 1 / (1 + math.exp(-FORM_SCALE * (fh - fo)))
                    adj += won - expected
                    if past["time_diff"] is not None and o["time_diff"] is not None:
                        margin += max(-2.0, min(2.0, o["time_diff"] - past["time_diff"]))
            r["h2h_n"] = n
            r["h2h_ahead"] = ahead / n if n else NAN
            r["h2h_adj"] = adj / n if n else NAN
            r["h2h_margin"] = margin / n if n else NAN

    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    level = ["lv_field_form", "lv_field_tidx", "lv_after_n", "lv_after_top3", "lv_beaten_form", "lv_lost_form"]
    rel = [f"{p}_{k}" for k in ("tidx_mean5", "tidx_best5", "prize_ratio_mean5") for p in ("rel", "rank")] + ["field_form_mean"]
    h2h = ["h2h_n", "h2h_ahead", "h2h_adj", "h2h_margin"]
    sets = {"base": FEATURES, "+前走レベル": FEATURES + level, "+今回の比較": FEATURES + rel, "+直接対決": FEATURES + h2h,
            "+相手関係(比較+直接対決)": FEATURES + rel + h2h, "+全部": FEATURES + level + rel + h2h}
    coverage = {f: round(sum(1 for r in rows if not isnan(r[f])) / len(rows), 3) for f in level + rel + h2h}
    print("coverage", coverage, file=sys.stderr, flush=True)
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = sorted({r["date"][:4] for r in rows})
    result = {"coverage": coverage, "sets": {}}
    for name, feats in sets.items():
        per = {}
        for year in years:
            tr = [r for r in rows if r["date"][:4] != year]
            te = [r for r in rows if r["date"][:4] == year]
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            b = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = b.predict(m(te))
            y = [r["label"] for r in te]
            per[year] = {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "top3_hit": race_top3_hit(te, p)}
            print(name, year, per[year], file=sys.stderr, flush=True)
        result["sets"][name] = {"per_year": per, "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in ("logloss", "auc", "top3_hit")}}
    (OUT / "feature_comparison_level.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result["sets"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
