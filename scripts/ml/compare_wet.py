"""道悪補正で稍重をどう扱うかの比較（経験の判定 × 血統の道悪倍率 の4通り）.

    npx tsx scripts/ml/pedigree_features.ts && (cd scripts/ml && python compare_wet.py)

経験の判定:  heavy = 重・不良を走ったことがない馬だけ補正 / wet = 稍重以上を走ったことがない馬だけ補正
血統の倍率:  heavy_ratio = (重+不良)÷良 / wet_ratio = (稍重+重+不良)÷良
注意: 経験の判定を変えると対象の馬が変わるので、比べるのは「補正なしからの改善量」。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import OUT  # noqa: E402
from train_stack import PED, heavy_adjustment, oof_base  # noqa: E402


def main() -> None:
    data = list(csv.DictReader(PED.open(encoding="utf-8")))
    keys = {(r["race_id"], int(r["umaban"])) for r in data}
    base, context = oof_base(keys, sorted({r["date"][:4] for r in data}))
    data = [r for r in data if (r["race_id"], int(r["umaban"])) in base]
    result = {}
    for experience in ("heavy", "wet"):
        for ratio_col in ("heavy_ratio", "wet_ratio"):
            ratios = {(r["race_id"], int(r["umaban"])): float(r[ratio_col]) for r in data if r.get(ratio_col)}
            res = heavy_adjustment(data, base, context, ratios, experience=experience)
            cv = res["cv_logloss"]
            best = min(cv["beta1"], cv["estimated"])
            name = f"exp_{experience}__{ratio_col}"
            result[name] = {
                "runners": res["runners"], "beta": res["beta"], "chosen": res["chosen"],
                "cv_none": cv["none"], "cv_beta1": cv["beta1"], "cv_estimated": cv["estimated"],
                "improvement": cv["none"] - best, "improvement_per_runner": (cv["none"] - best) / res["runners"],
                "years_improved": res["years_improved"],
            }
            print(name, json.dumps(result[name], ensure_ascii=False), file=sys.stderr)
    (OUT / "wet_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
