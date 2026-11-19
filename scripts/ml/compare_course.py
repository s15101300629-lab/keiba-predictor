"""競馬場・コースの特徴の比較（1年ずつ外す）.

    cd scripts/ml && python compare_course.py   # → data/ml/feature_comparison_course.json

- コースの形：回り（右・左・直線）、外回りか（成績表のコース欄「芝・右・外」など）
- その馬の適性：同じ芝ダ・同じ回りでの出走数と3着内率、同じコース（競馬場・芝ダ・距離・内外）での出走数と3着内率
- 競馬場の扱い：今は番号（1〜10）のまま。競馬場ごとの 0/1 の列（one-hot）にした場合と比べる
  （TypeScript の推論は区分（カテゴリ）の分岐に対応していないので、one-hot で比べる）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

NAN = float("nan")


def course_of(text: str | None) -> tuple[int | None, int]:
    """「芝・右・外」→ (回り 0=右 1=左 2=直線, 外回り 1/0)."""
    text = text or ""
    direction = 0 if "右" in text else 1 if "左" in text else 2 if "直線" in text else None
    outer = 1 if "外" in text.split("・")[-1] and "外内" not in text and "内外" not in text else 0
    return direction, outer


def main() -> None:
    course = {}
    for path in sorted((DATA / "json").glob("*/*.json")):
        day = json.loads(path.read_text(encoding="utf-8"))
        for r in day["races"]:
            if r.get("race_id"):
                course[r["race_id"]] = course_of(r.get("course"))
    rows, _ = build_dataset(DATA)
    for r in rows:
        r["course_dir"], r["course_outer"] = course.get(r["race_id"], (None, 0))
        if r["course_dir"] is None:
            r["course_dir"] = NAN
    by_horse: dict[str, list[dict]] = {}
    for r in rows:
        by_horse.setdefault(r["horse"], []).append(r)
    for runs in by_horse.values():
        runs.sort(key=lambda r: r["date"])
        for i, r in enumerate(runs):
            prev = runs[:i]
            same_dir = [p for p in prev if p["surface"] == r["surface"] and p["course_dir"] == r["course_dir"]]
            same_course = [p for p in prev if p["surface"] == r["surface"] and p["venue"] == r["venue"]
                           and p["distance"] == r["distance"] and p["course_outer"] == r["course_outer"]]
            for key, sel in (("dir", same_dir), ("course", same_course)):
                r[f"{key}_n"] = len(sel)
                r[f"{key}_top3"] = sum(p["label"] for p in sel) / len(sel) if sel else NAN
    venues = sorted({r["venue"] for r in rows})
    for r in rows:
        for v in venues:
            r[f"venue_{v}"] = 1 if r["venue"] == v else 0
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]

    shape = ["course_dir", "course_outer"]
    apt = ["dir_n", "dir_top3", "course_n", "course_top3"]
    onehot = [f"venue_{v}" for v in venues]
    sets = {"base": FEATURES, "+形": FEATURES + shape, "+適性": FEATURES + apt, "+形+適性": FEATURES + shape + apt,
            "+競馬場one-hot": FEATURES + onehot, "+全部": FEATURES + shape + apt + onehot}
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = sorted({r["date"][:4] for r in rows})
    result = {}
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
        result[name] = {"per_year": per, "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in ("logloss", "auc", "top3_hit")}}
    (OUT / "feature_comparison_course.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
