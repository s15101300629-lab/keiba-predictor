"""馬主・生産牧場の成績と、騎手・調教師の前年の成績を足したときの比較（前年までで学習して 2024〜2026年を予測）.

    cd scripts/ml && python compare_owner.py   # → data/ml/feature_comparison_owner.json

足すもの（どれもレースの日より前の結果だけ。同じ日の他レースは使わない）
- 馬主・生産牧場（成績表の owner・breeder）：今年と前年の出走数・3着内率
- 騎手・調教師：前年の騎乗（出走）数・3着内率（今は今年の分だけ入っている）
表記ゆれは空白と「氏」「㈱」などをそろえるだけ（名寄せはしない）
基準は今の本番（160項目、train.py の PARAMS）。評価は tune_params.evaluate
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset, load_races, norm_jockey  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

NAN = float("nan")
GROUPS = {
    "owner": ["owner_n_y", "owner_top3_y", "owner_n_py", "owner_top3_py"],
    "breeder": ["breeder_n_y", "breeder_top3_y", "breeder_n_py", "breeder_top3_py"],
    "prev_year": ["jockey_n_py", "jockey_top3_py", "trainer_n_py", "trainer_top3_py"],
}


def norm_name(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for x in (" ", "　", "氏", "(株)", "(有)", "株式会社", "有限会社"):
        s = s.replace(x, "")
    return s


def owner_features() -> dict:
    """(race_id, 馬名) → 馬主・生産牧場の今年・前年、騎手・調教師の前年の出走数と3着内率."""
    races = load_races(DATA)
    counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])  # (種類, 名前, 年) -> [出走数, 3着以内]
    feats: dict[tuple, dict] = {}
    i = 0
    while i < len(races):
        day = races[i]["date"]
        same = []
        while i < len(races) and races[i]["date"] == day:
            same.append(races[i]); i += 1
        y, py = day[:4], str(int(day[:4]) - 1)
        for race in same:
            for h in race["runners"]:
                keys = {"owner": norm_name(h.get("owner")), "breeder": norm_name(h.get("breeder")),
                        "jockey": norm_jockey(h.get("jockey")), "trainer": norm_jockey(h.get("trainer"))}
                f = {}
                for kind in ("owner", "breeder"):
                    for suf, yr in (("y", y), ("py", py)):
                        n, t = counts[(kind, keys[kind], yr)] if keys[kind] else (0, 0)
                        f[f"{kind}_n_{suf}"] = n
                        f[f"{kind}_top3_{suf}"] = t / n if n else NAN
                for kind in ("jockey", "trainer"):
                    n, t = counts[(kind, keys[kind], py)] if keys[kind] else (0, 0)
                    f[f"{kind}_n_py"] = n
                    f[f"{kind}_top3_py"] = t / n if n else NAN
                feats[(race["race_id"], h["horse"])] = f
        for race in same:  # 当日のレースは翌日以降の集計にだけ入れる
            for h in race["runners"]:
                top3 = 1 if h["finish"] is not None and h["finish"] <= 3 else 0
                for kind, name in (("owner", norm_name(h.get("owner"))), ("breeder", norm_name(h.get("breeder"))),
                                   ("jockey", norm_jockey(h.get("jockey"))), ("trainer", norm_jockey(h.get("trainer")))):
                    if name:
                        c = counts[(kind, name, y)]
                        c[0] += 1; c[1] += top3

    return feats


def main() -> None:
    feats = owner_features()
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    extra = [f for g in GROUPS.values() for f in g]
    X = np.array([[r[f] for f in FEATURES] + [feats[(r["race_id"], r["horse"])][f] for f in extra] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    cols = FEATURES + extra
    coverage = {f: round(float(np.mean(~np.isnan(X[:, cols.index(f)]))), 3) for f in extra if "top3" in f}
    print("coverage", coverage, file=sys.stderr, flush=True)
    sets = {"base": FEATURES, "+馬主": FEATURES + GROUPS["owner"], "+生産牧場": FEATURES + GROUPS["breeder"],
            "+騎手・調教師の前年": FEATURES + GROUPS["prev_year"], "+全部": cols}
    out = {"coverage": coverage, "sets": {}}
    for name, fs in sets.items():
        idx = [cols.index(f) for f in fs]
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X[:, idx], y, year, race_id, finish, PARAMS)
        if name != "base":
            base = out["sets"]["base"]
            res["years_logloss_better"] = sum(res[t]["logloss"] < base[t]["logloss"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_owner.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
