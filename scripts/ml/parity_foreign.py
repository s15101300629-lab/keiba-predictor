"""外国馬の扱い（2026-11-26）の一致テストの Python 側。外国馬が出た今年のレースを、答え合わせと同じ扱い（海外の走を足す・
FOREIGN_FIX を入れる）で計算し、data/ml/parity_foreign.json に書く。TS 側は parity_foreign.ts.

    cd scripts/ml && python parity_foreign.py && node --experimental-strip-types parity_foreign.ts

parity_day.ts は「store の最終日より後の日」しか比べないので、外国馬の出る日（次は 11/29）まで外国馬の扱いを確かめられないため。
海外の走は data/overseas/histories.json（netkeiba の戦績）から作る（アプリ・答え合わせと同じ overseas.improve）
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import overseas  # noqa: E402
from check_jra_day import overseas_runs, predict, store_run  # noqa: E402
from train import DATA, OUT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    store = json.loads((OUT / "store.json").read_text(encoding="utf-8"))
    since = store["yearStart"]  # store の騎手・馬主などは前年から。今年のレースなら Python と同じ値になる
    hist = json.loads((ROOT / "data/overseas/histories.json").read_text(encoding="utf-8"))
    extra = {}
    for hid, v in hist.items():
        runs = [overseas.improve(r, r["ov_venue"], r["ov_name"], hid) for r in overseas_runs(v["rows"])]
        if runs:
            for nm in v["names"]:
                extra[features.clean_horse_name(nm)] = runs
    features.EXTRA_RUNS = extra
    features.FOREIGN_FIX = {"rating": True, "people": True}
    ri = ROOT / "data/overseas/rating_init.json"
    features.RATING_INIT = json.loads(ri.read_text(encoding="utf-8"))["coef"] if ri.exists() else None
    rows, _ = features.build_dataset(DATA)
    target = {r["race_id"] for r in rows if since <= r["date"] <= store["lastDate"] and features.is_foreign_trainer(r.get("trainer"))}
    rows = [r for r in rows if r["race_id"] in target]
    model = json.loads((OUT / "model.json").read_text(encoding="utf-8"))
    X = np.array([[r.get(f, float("nan")) for f in model["features"]] for r in rows], dtype=np.float64)
    pred = predict(model, X)
    out = []
    for r, p in zip(rows, pred):
        out.append({"race_id": r["race_id"], "date": r["date"], "horse": r["horse"], "turn": r["turn"], "sband": r["sband"],
                    **{k: r.get(k) for k in ("jockey", "trainer", "owner", "breeder")},
                    "foreign": features.is_foreign_trainer(r.get("trainer")),
                    "extra": [store_run(x) for x in extra.get(r["horse"], []) if x["date"] < r["date"]] or None,
                    "estimated": r.get("_estimated"),
                    "features": {f: None if (r.get(f) is None or (isinstance(r[f], float) and math.isnan(r[f]))) else r[f] for f in model["features"]},
                    "pred": float(p)})
    (OUT / "parity_foreign.json").write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"外国馬のいるレース {len(target)}・{len(out)} 行（外国馬 {sum(o['foreign'] for o in out)}）", file=sys.stderr)


if __name__ == "__main__":
    main()
