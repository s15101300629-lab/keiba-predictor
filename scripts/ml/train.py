"""成績表データで LightGBM（3着以内確率）を学習し、アプリ用ファイルを書き出す.

    cd scripts/ml && python train.py

出力 (data/ml/):
  model.json          木構造（アプリの TypeScript 推論器が読む）
  store.json          直近の馬の過去走・今年の騎手騎乗（アプリが特徴量計算に使う）
  metrics.json        検証結果
  parity.json         Python と TypeScript の特徴量・予測一致テスト用サンプル
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "jra-seiseki"
OUT = REPO / "data" / "ml"
TRAIN_FROM = "2020-01-01"  # 2019年は過去走の蓄積期間として使い、学習行には入れない（range_comparison.json の D）
STORE_DAYS = 550  # この日数以内に走った馬だけ store に入れる

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "num_threads": 4,
}


def matrix(rows):
    return np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float64)


def race_top3_hit(rows, pred) -> float:
    """各レースで予測上位3頭のうち実際に3着以内だった割合."""
    by_race: dict[str, list[tuple[float, int]]] = {}
    for r, p in zip(rows, pred):
        by_race.setdefault(r["race_id"], []).append((p, r["label"]))
    hits = total = 0
    for items in by_race.values():
        items.sort(key=lambda x: -x[0])
        hits += sum(lbl for _, lbl in items[:3])
        total += min(3, len(items))
    return hits / total


def export_model(booster: lgb.Booster) -> dict:
    """LightGBM の木を、TypeScript で辿りやすい配列形式に変換する."""
    dump = booster.dump_model()
    trees = []
    for t in dump["tree_info"]:
        nodes: list = []

        def walk(node) -> int:
            idx = len(nodes)
            nodes.append(None)
            if "leaf_value" in node:
                nodes[idx] = {"v": node["leaf_value"]}
                return idx
            if node["decision_type"] != "<=":
                raise ValueError("categorical splits are not supported")
            left = walk(node["left_child"])
            right = walk(node["right_child"])
            nodes[idx] = {
                "f": node["split_feature"], "t": node["threshold"],
                "d": 1 if node["default_left"] else 0,
                "m": {"None": 0, "Zero": 1, "NaN": 2}[node["missing_type"]],
                "l": left, "r": right,
            }
            return idx

        walk(t["tree_structure"])
        trees.append(nodes)
    return {"features": FEATURES, "objective": "binary", "trees": trees}


def build_store(hist, last_date: str) -> dict:
    cutoff = (date.fromisoformat(last_date) - timedelta(days=STORE_DAYS)).isoformat()
    horses = {}
    names: list[str] = []  # 騎手・調教師名の一覧（各走には番号だけ入れる）
    name_index: dict[str, int] = {}

    def idx(name: str) -> int:
        if name not in name_index:
            name_index[name] = len(names)
            names.append(name)
        return name_index[name]

    for name, runs in hist.horses.items():
        if runs and runs[-1]["date"] >= cutoff:
            horses[name] = [[r["date"], r["venue"], r["surface"], r["distance"],
                             -1 if r["going"] is None else r["going"], r["field"],
                             -1 if r["finish"] is None else r["finish"],
                             -1 if r["time_diff"] is None else r["time_diff"],
                             -1 if r["pace"] is None else r["pace"],
                             r["pos_first"] or -1, r["pos_last"] or -1,
                             r["prize"], r["weight"], idx(r["jockey"]), idx(r["trainer"]), r["tidx"], r["victim"],
                             r["ab"], r["ag_dev"], r["ag_rank"], r["early_gap"], r["fade"]] for r in runs]
    year_start = last_date[:4] + "-01-01"
    prev_year_start = str(int(last_date[:4]) - 1) + "-01-01"
    jockeys = {}
    for name, rides in hist.jockeys.items():
        # 年明け直後でも今年の集計ができるよう、前年分も残す
        sel = [[r["date"], r["venue"], r["surface"], r["distance"], r["top3"]] for r in rides if r["date"] >= prev_year_start]
        if sel and sel[-1][0] >= prev_year_start:
            jockeys[name] = sel
    trainers = {name: [[r["date"], r["venue"], r["top3"]] for r in entries if r["date"] >= prev_year_start]
                for name, entries in hist.trainers.items()}
    trainers = {k: v for k, v in trainers.items() if v}
    # コース(競馬場-芝ダ-距離) -> [[日付, 枠番, 出走数, 3着以内数], ...]（日付・枠ごとに集計）
    course_waku = {}
    for (venue, surface, distance), events in hist.course_waku_events.items():
        agg: dict[tuple, list[int]] = {}
        for d, waku, top3 in events:
            a = agg.setdefault((d, waku), [0, 0])
            a[0] += 1
            a[1] += top3
        course_waku[f"{venue}-{surface}-{distance}"] = [[d, w, n, t] for (d, w), (n, t) in sorted(agg.items())]
    # 枠の有利不利（courseWaku）は比較で不採用だったので store には入れない（サイズ削減）
    del course_waku
    return {"lastDate": last_date, "yearStart": year_start, "horses": horses, "jockeys": jockeys,
            "trainers": trainers, "names": names,
            "runFields": ["date", "venue", "surface", "distance", "going", "field", "finish", "timeDiff", "pace", "posFirst", "posLast",
                          "prize", "weight", "jockey", "trainer", "tidx", "victim",
                          "ab", "agDev", "agRank", "earlyGap", "fade"],
            "rideFields": ["date", "venue", "surface", "distance", "top3"]}


def split_rows(rows: list[dict], test_year: str):
    """評価する年（test_year）は、学習にも木の本数の決定にも使わない。
    木の本数は、その前年（val）を検証にして、それより前の年（fit）で学習して決める."""
    val_year = str(int(test_year) - 1)
    fit = [r for r in rows if r["date"][:4] < val_year]
    val = [r for r in rows if r["date"][:4] == val_year]
    train = [r for r in rows if r["date"][:4] < test_year]
    test = [r for r in rows if r["date"][:4] == test_year]
    return fit, val, train, test


def best_rounds(fit, val, y_fit, y_val) -> int:
    dfit = lgb.Dataset(matrix(fit), label=y_fit, feature_name=FEATURES)
    dval = lgb.Dataset(matrix(val), label=y_val, reference=dfit)
    b = lgb.train(PARAMS, dfit, num_boost_round=3000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
    return max(b.best_iteration, 50)


# 3着以内のほかに、勝率（1着）・連対率（2着以内）も同じ特徴量で予測する
EXTRA_TARGETS = {"win": 1, "top2": 2}


def train_extra_targets(rows: list[dict], test_year: str) -> dict:
    """1着・2着以内のモデルを学習して model_<name>.json に書き出し、検証の結果を返す."""
    fit, val, train, test = split_rows(rows, test_year)
    out = {}
    for name, k in EXTRA_TARGETS.items():
        label = lambda rs: [1 if r["finish"] is not None and r["finish"] <= k else 0 for r in rs]  # noqa: E731
        best = best_rounds(fit, val, label(fit), label(val))
        booster = lgb.train(PARAMS, lgb.Dataset(matrix(train), label=label(train), feature_name=FEATURES), num_boost_round=best)
        pred = booster.predict(matrix(test))
        y = label(test)
        out[name] = {"best_iteration": best, "auc": roc_auc_score(y, pred), "logloss": log_loss(y, pred), "base_rate": sum(y) / len(y)}
        print(name, out[name], file=sys.stderr)
        final = lgb.train(PARAMS, lgb.Dataset(matrix(rows), label=label(rows), feature_name=FEATURES), num_boost_round=max(best, 50))
        (OUT / f"model_{name}.json").write_text(json.dumps(export_model(final), separators=(",", ":")), encoding="utf-8")
    return out


def main() -> None:
    rows, hist = build_dataset(DATA)
    if sys.argv[1:] == ["extra"]:  # 勝率・連対率のモデルだけ学習し直す
        rows = [r for r in rows if r["date"] >= TRAIN_FROM]
        metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
        metrics["extra_targets"] = train_extra_targets(rows, metrics["test_year"])
        (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    years = sorted({r["date"][:4] for r in rows})
    test_year = years[-1]
    fit, val, train, test = split_rows(rows, test_year)
    print(f"rows: fit {len(fit)} / val {len(val)} / train {len(train)} / test {len(test)} (test year {test_year})", file=sys.stderr)
    # 評価する年は使わずに木の本数を決め、その年より前で学習したモデルで評価する（評価の数字を甘くしない）
    best = best_rounds(fit, val, [r["label"] for r in fit], [r["label"] for r in val])
    booster = lgb.train(PARAMS, lgb.Dataset(matrix(train), label=[r["label"] for r in train], feature_name=FEATURES), num_boost_round=best)
    pred = booster.predict(matrix(test))
    y = [r["label"] for r in test]
    metrics = {
        "test_year": test_year, "train_rows": len(train), "test_rows": len(test),
        "best_iteration": best,
        "auc": roc_auc_score(y, pred), "logloss": log_loss(y, pred),
        "base_rate": sum(y) / len(y),
        "race_top3_hit": race_top3_hit(test, pred),
        # 参考: 単勝人気順で上位3頭を選んだ場合（人気は特徴量には使っていない）
        "race_top3_hit_popularity": race_top3_hit(test, [-(r["popularity"] or 99) for r in test]),
    }
    gain = booster.feature_importance("gain")
    metrics["importance"] = dict(sorted(zip(FEATURES, (round(float(g), 1) for g in gain)), key=lambda x: -x[1]))
    print(json.dumps({k: v for k, v in metrics.items() if k != "importance"}, ensure_ascii=False, indent=1), file=sys.stderr)

    # 本番モデル: 全期間で、検証で決めた木の本数で学習し直す
    dall = lgb.Dataset(matrix(rows), label=[r["label"] for r in rows], feature_name=FEATURES)
    final = lgb.train(PARAMS, dall, num_boost_round=max(best, 50))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model.json").write_text(json.dumps(export_model(final), separators=(",", ":")), encoding="utf-8")
    store = build_store(hist, max(r["date"] for r in rows))
    (OUT / "store.json").write_text(json.dumps(store, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    metrics["extra_targets"] = train_extra_targets(rows, test_year)
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # parity: 最新日のレースから数レース分、特徴量と予測値を保存
    last_races = sorted({r["race_id"] for r in rows if r["date"] == store["lastDate"]})[:3]
    sample = [r for r in rows if r["race_id"] in last_races]
    sample_pred = final.predict(matrix(sample))
    parity = [{"race_id": r["race_id"], "date": r["date"], "horse": r["horse"], "jockey": r["jockey"], "trainer": r["trainer"],
               "features": {f: (None if isinstance(r[f], float) and math.isnan(r[f]) else r[f]) for f in FEATURES},
               "pred": float(p)} for r, p in zip(sample, sample_pred)]
    (OUT / "parity.json").write_text(json.dumps(parity, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
