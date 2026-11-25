"""外国馬の扱い（2026-11-26）でアプリが読む表を作る（再学習なしで作れる。ml-train.yml で再学習のあとにも回す）.

    cd scripts/ml && python foreign_jockeys.py   # → data/overseas/jockey_career.json

- jockeys：短期免許の外国人騎手（成績表で「R.ムーア(英)」の形。features.is_short_term_jockey）の、JRA での全部の騎乗
  [日付, 競馬場, 芝ダ, 距離, 3着以内]（features.load_races と同じ平地レース・同じ年から）。外国馬に乗るとき、今年の成績のかわりに通算を使う
- foreign_training_runs：学習データ（train.TRAIN_FROM〜）の外国馬（海外の厩舎の馬）の走の数（画面の注意「学習の例が少ない」に出す）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "jra-seiseki"
OUT = ROOT / "data" / "overseas" / "jockey_career.json"
TRAIN_FROM = "2018-01-01"  # train.TRAIN_FROM（train.py を読み込むと lightgbm が要るので、値だけ写す）


def main() -> None:
    rides: dict[str, list] = {}
    foreign_runs, foreign_horses, last = 0, set(), ""
    for path in sorted((DATA / "json").glob("*/*.json")):
        if path.parent.name < features.HISTORY_FROM:
            continue
        day = json.loads(path.read_text(encoding="utf-8"))
        for r in day["races"]:
            if r.get("surface") not in features.SURFACE or not r.get("race_id") or not r.get("date"):
                continue
            starters = [h for h in r["results"] if h.get("status") not in ("取消", "除外")]
            if len(starters) < 2:
                continue
            last = max(last, r["date"])
            for h in starters:
                finish = h.get("finish") if h.get("status") != "中止" else None
                top3 = 1 if finish is not None and finish <= 3 else 0
                if features.is_short_term_jockey(h.get("jockey")):
                    rides.setdefault(features.norm_jockey(h.get("jockey")), [])
                if features.is_foreign_trainer(h.get("trainer")) and r["date"] >= TRAIN_FROM:
                    foreign_runs += 1
                    foreign_horses.add(features.clean_horse_name(h.get("horse")))
                jk = features.norm_jockey(h.get("jockey"))
                if jk:
                    rides.setdefault("\0" + jk, []).append([r["date"], int(day["venue_code"]), features.SURFACE[r["surface"]], r.get("distance"), top3])
    career = {k: sorted(rides.get("\0" + k, [])) for k in rides if not k.startswith("\0")}
    OUT.write_text(json.dumps({"note": "短期免許の外国人騎手の JRA での全部の騎乗 [日付, 競馬場, 芝ダ, 距離, 3着以内]（scripts/ml/foreign_jockeys.py）",
                               "lastDate": last, "foreign_training_runs": foreign_runs, "foreign_training_horses": len(foreign_horses),
                               "jockeys": career}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"騎手 {len(career)} 人・{sum(len(v) for v in career.values())} 騎乗、学習データの外国馬 {foreign_runs} 走（{len(foreign_horses)} 頭）", file=sys.stderr)


if __name__ == "__main__":
    main()
