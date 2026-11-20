"""似たコース（直線の長さ・高低差）での成績の比較（1年ずつ外す）.

    cd scripts/ml && python compare_similar_course.py   # → data/ml/feature_comparison_similar_course.json

コースの数値は JRA のコース紹介（https://www.jra.go.jp/facilities/race/<競馬場>/course/）の A コースの値（2026-11-20 に取得）。
- 直線距離（m）と高低差（m）。芝は内回り・外回りで分ける（成績表のコース欄「芝・右・外」などで判定）。新潟の直線1000m は直線 1000m・高低差不明
- 本人の成績（同じ芝ダの過去走）
  - 直線が似た：直線距離が ±50m 以内のコースでの出走数・3着内率
  - 直線の区分：短い（〜300m）・中くらい（〜400m）・長い（400m〜）で同じ区分のコース
  - 高低差の区分：平坦（〜2.0m）・中くらい（〜3.5m）・坂が大きい（3.5m〜）で同じ区分のコース
  - 両方：直線の区分と高低差の区分が両方同じコース
- 比べるもの：今の特徴量（base）、+コースの数値（今回の直線・高低差・一周）、+直線、+高低差、+両方、+全部
- 評価：2023〜2026年を1年ずつ外す（他の比較と同じ）。学習は 2020年〜
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_course import course_of  # noqa: E402
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

NAN = float("nan")
# (競馬場, 芝0/ダ1, 外回り1/0/直線2) → (直線距離, 高低差, 一周距離)
COURSE = {
    (1, 0, 0): (266.1, 0.7, 1640.9), (2, 0, 0): (262.1, 3.5, 1626.6), (3, 0, 0): (292.0, 1.9, 1600.0),
    (4, 0, 0): (358.7, 0.8, 1623.0), (4, 0, 1): (658.7, 2.2, 2223.0), (4, 0, 2): (1000.0, NAN, NAN),
    (5, 0, 0): (525.9, 2.7, 2083.1),
    (6, 0, 0): (310.0, 5.3, 1667.1), (6, 0, 1): (310.0, 5.3, 1839.7),
    (7, 0, 0): (412.5, 3.5, 1705.9),
    (8, 0, 0): (328.4, 3.1, 1782.8), (8, 0, 1): (403.7, 4.3, 1894.3),
    (9, 0, 0): (356.5, 1.9, 1689.0), (9, 0, 1): (473.6, 2.4, 2089.0),
    (10, 0, 0): (293.0, 3.0, 1615.1),
    (1, 1, 0): (264.3, 0.9, 1487.0), (2, 1, 0): (260.3, 3.5, 1475.8), (3, 1, 0): (295.7, 2.1, 1444.6),
    (4, 1, 0): (353.9, 0.6, 1472.5), (5, 1, 0): (501.6, 2.5, 1899.0), (6, 1, 0): (308.0, 4.5, 1493.0),
    (7, 1, 0): (410.7, 3.4, 1530.0), (8, 1, 0): (329.1, 3.0, 1607.6), (9, 1, 0): (352.7, 1.6, 1517.6),
    (10, 1, 0): (291.3, 2.9, 1445.4),
}
SIMILAR_STRAIGHT = 50.0


def straight_band(s: float) -> int:
    return 0 if s < 300 else 1 if s < 400 else 2


def elev_band(e: float) -> int | None:
    return None if e != e else 0 if e < 2.0 else 1 if e < 3.5 else 2


def add_course_features(rows: list[dict]) -> None:
    """コースの数値と似たコースでの成績の列を rows に足す（rows は build_dataset の全行）."""
    layout = {}
    for path in sorted((DATA / "json").glob("*/*.json")):
        day = json.loads(path.read_text(encoding="utf-8"))
        for r in day["races"]:
            if r.get("race_id"):
                text = r.get("course") or ""
                _, outer = course_of(text)
                layout[r["race_id"]] = 2 if "直線" in text else outer
    for r in rows:
        c = COURSE.get((int(r["venue"]), int(r["surface"]), 0 if r["surface"] == 1 else layout.get(r["race_id"], 0)))
        r["c_straight"], r["c_elev"], r["c_lap"] = c if c else (NAN, NAN, NAN)

    by_horse: dict[str, list[dict]] = {}
    for r in rows:
        by_horse.setdefault(r["horse"], []).append(r)
    for runs in by_horse.values():
        runs.sort(key=lambda r: r["date"])
        for i, r in enumerate(runs):
            prev = [p for p in runs[:i] if p["date"] < r["date"] and p["surface"] == r["surface"] and p["c_straight"] == p["c_straight"]]
            s, e = r["c_straight"], r["c_elev"]
            sb, eb = (straight_band(s), elev_band(e)) if s == s else (None, None)
            groups = {
                "sim_straight": [p for p in prev if s == s and abs(p["c_straight"] - s) <= SIMILAR_STRAIGHT],
                "band_straight": [p for p in prev if sb is not None and straight_band(p["c_straight"]) == sb],
                "band_elev": [p for p in prev if eb is not None and elev_band(p["c_elev"]) == eb],
                "band_both": [p for p in prev if sb is not None and eb is not None
                              and straight_band(p["c_straight"]) == sb and elev_band(p["c_elev"]) == eb],
            }
            for key, sel in groups.items():
                r[f"{key}_n"] = len(sel)
                r[f"{key}_top3"] = sum(p["label"] for p in sel) / len(sel) if sel else NAN


def main() -> None:
    rows, _ = build_dataset(DATA)
    add_course_features(rows)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]

    shape = ["c_straight", "c_elev", "c_lap"]
    straight = ["sim_straight_n", "sim_straight_top3", "band_straight_n", "band_straight_top3"]
    elev = ["band_elev_n", "band_elev_top3"]
    both = ["band_both_n", "band_both_top3"]
    sets = {"base": FEATURES, "+コースの数値": FEATURES + shape, "+直線": FEATURES + straight, "+高低差": FEATURES + elev,
            "+両方": FEATURES + both, "+全部": FEATURES + shape + straight + elev + both}
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = ["2023", "2024", "2025", "2026"]
    coverage = {f: round(sum(1 for r in rows if r[f] == r[f]) / len(rows), 3) for f in shape + ["sim_straight_top3", "band_both_top3"]}
    result: dict = {"coverage": coverage, "sets": {}}
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
        if name != "base":
            base = result["sets"]["base"]["per_year"]
            result["sets"][name]["years_logloss_better"] = sum(1 for y in years if per[y]["logloss"] < base[y]["logloss"])
    (OUT / "feature_comparison_similar_course.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result["sets"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
