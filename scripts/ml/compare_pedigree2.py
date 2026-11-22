"""血統の追加候補の比較（前年までで学習して 2024〜2026年を予測。基準は今の本番 196項目）.

    npx tsx scripts/ml/pedigree_all_features.ts && npx tsx scripts/ml/pedigree_extra_features.ts
    cd scripts/ml && python compare_pedigree2.py   # → data/ml/feature_comparison_pedigree2.json

今の血統の項目：今回の芝ダでの3着内率・距離区分の得意さ（割合）・道悪倍率・稍重〜不良の倍率（父・母父・母母父を 50:25:12.5）
候補（pedigree_extra_features.ts）：
K1 芝ダ×距離区分の3着内率そのもの（得意さの割合ではなく、その条件での産駒の質こみ）
K2 芝ダ×今回の馬場（良・稍重・重・不良）の3着内率
K3 今回の芝ダでの産駒の勝率
K4 直近3年の産駒の3着内率（今の産駒の質）
K5 父だけの芝ダ×距離区分の3着内率
競馬場別・年齢別の産駒成績は、取得済みの種牡馬のページにないので比べていない（取るには約2,900頭の追加取得が要る）
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

NAN = float("nan")
KS = {"K1 芝ダ×距離の3着内率": ["pb_band_top3"], "K2 芝ダ×馬場の3着内率": ["pb_going_top3"], "K3 産駒の勝率": ["pb_win"],
      "K4 直近3年の3着内率": ["pb_recent_top3"], "K5 父だけの芝ダ×距離": ["pb_f_band_top3"]}


def main() -> None:
    extra = {}
    with (DATA.parent / "pedigree" / "all" / "runner_features_extra.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            extra[(r["race_id"], int(r["umaban"]))] = {k: float(v) if v != "" else NAN for k, v in r.items() if k.startswith("pb_")}
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    allk = [k for ks in KS.values() for k in ks]
    for r in rows:
        e = extra.get((r["race_id"], int(r["umaban"])) if r["umaban"] == r["umaban"] else None, {})
        for k in allk:
            r[k] = e.get(k, NAN)
    cols = FEATURES + allk
    X = np.array([[r[f] for f in cols] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    sets = {"基準(196)": FEATURES, **{"+" + n: FEATURES + ks for n, ks in KS.items()}, "+全部": cols}
    out = {"coverage": {k: round(float(np.mean(~np.isnan(X[:, cols.index(k)]))), 3) for k in allk}, "sets": {}}
    first = None
    for name, fs in sets.items():
        idx = [cols.index(f) for f in fs]
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X[:, idx], y, year, race_id, finish, PARAMS)
        if first is None:
            first = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss_sum3"] < first[t]["logloss_sum3"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_pedigree2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
