"""netkeiba から取った新しい情報（調教評価・備考・上がり3F）の効果を、的中率と回収率で確かめる.

    cd scripts/ml && python eval_new_info.py   # → data/ml/new_info_eval.json

- 新しい情報は今は 2026年分だけなので、2026年を前半（〜6月）と後半（7月〜）に分ける
  - 学習：2023年〜2026年6月（新しい情報は 2026年分だけ値が入り、それ以前は欠損）
  - 評価：2026年7月〜
- 比べるもの
  - base：今のモデルの特徴量
  - +調教：今回のレースの調教評価（A=4〜D=1）と、出走馬の中での差
  - +備考：前走・近5走の備考（何か記載あり／出遅れ）
  - +上がり：前走・近5走の上がり3Fの、そのレース内での順位÷頭数
  - +全部
- 指標：logloss・AUC・上位3頭の3着内率、1位の馬の単勝・複勝の回収率
"""

from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "2026-07-01"
RANK = {"A": 4, "B": 3, "C": 2, "D": 1}
NAN = float("nan")


def load_jsonl(pattern: str) -> dict[str, list[dict]]:
    out = {}
    for path in glob.glob(str(ROOT / pattern)):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            out[d["race_id"]] = d["horses"]
    return out


def umaban_of(r) -> int | None:
    u = r["umaban"]
    return None if u is None or (isinstance(u, float) and math.isnan(u)) else int(u)


def main() -> None:
    training = load_jsonl("data/training/*.jsonl")
    remarks = load_jsonl("data/remarks/*.jsonl")
    print(f"training races {len(training)}, remark races {len(remarks)}", file=sys.stderr)

    # レース・馬番ごとの情報
    train_rank = {(rid, h["umaban"]): RANK.get(h["rank"] or "", NAN) for rid, hs in training.items() for h in hs if h["umaban"]}
    remark_info = {}
    for rid, hs in remarks.items():
        agaris = [(h["umaban"], float(h["agari"])) for h in hs if h.get("agari") and h["agari"].replace(".", "").isdigit()]
        order = {u: i for i, (u, _) in enumerate(sorted(agaris, key=lambda x: x[1]))}
        for h in hs:
            text = h.get("remark") or ""
            remark_info[(rid, h["umaban"])] = {
                "any": 1 if text else 0, "slow": 1 if "出遅" in text else 0,
                "agari_rank": (order[h["umaban"]] / max(1, len(agaris) - 1)) if h["umaban"] in order else NAN,
            }

    rows, _ = build_dataset(DATA)
    # 馬ごとの過去走（古い順）から、前走・近5走の備考と上がり
    by_horse: dict[str, list[dict]] = {}
    for r in rows:
        by_horse.setdefault(r["horse"], []).append(r)
    for runs in by_horse.values():
        runs.sort(key=lambda r: r["date"])
        hist = []
        for r in runs:
            u = umaban_of(r)
            prev = [h for h in hist[-5:] if h is not None]
            last = hist[-1] if hist else None
            r["rm_last1_any"] = last["any"] if last else NAN
            r["rm_last1_slow"] = last["slow"] if last else NAN
            r["rm_n5_any"] = sum(h["any"] for h in prev) if prev else NAN
            r["rm_n5_slow"] = sum(h["slow"] for h in prev) if prev else NAN
            r["agari_last1"] = last["agari_rank"] if last else NAN
            ag = [h["agari_rank"] for h in prev if not math.isnan(h["agari_rank"])]
            r["agari_mean5"] = sum(ag) / len(ag) if ag else NAN
            r["train_rank"] = train_rank.get((r["race_id"], u), NAN)
            hist.append(remark_info.get((r["race_id"], u)))
    # 調教評価の、出走馬の中での差
    races: dict[str, list[dict]] = {}
    for r in rows:
        races.setdefault(r["race_id"], []).append(r)
    for rs in races.values():
        vals = [r["train_rank"] for r in rs if not math.isnan(r["train_rank"])]
        mean = sum(vals) / len(vals) if vals else NAN
        for r in rs:
            r["train_rank_rel"] = r["train_rank"] - mean if vals and not math.isnan(r["train_rank"]) else NAN

    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    tr = [r for r in rows if r["date"] < SPLIT]
    te = [r for r in rows if r["date"] >= SPLIT]

    # 配当（回収率用）
    place_pay, win_pay = {}, {}
    for f in glob.glob(str(ROOT / "data/jra-seiseki/csv/*/payouts.csv")):
        for p in csv.DictReader(open(f, encoding="utf-8")):
            if p["type"] in ("複勝", "単勝") and p["combo"] and p["payout"]:
                try:
                    (place_pay if p["type"] == "複勝" else win_pay)[(p["race_id"], int(p["combo"]))] = int(p["payout"].replace(",", ""))
                except ValueError:
                    pass

    def roi(rs, pred):
        by = {}
        for r, x in zip(rs, pred):
            by.setdefault(r["race_id"], []).append((x, r))
        n = w = pl = 0
        for items in by.values():
            items.sort(key=lambda t: -t[0])
            top = items[0][1]
            u = umaban_of(top)
            n += 1
            w += win_pay.get((top["race_id"], u), 0) if top["finish"] == 1 else 0
            pl += place_pay.get((top["race_id"], u), 0) if top["label"] else 0
        return {"races": n, "top1_win_roi": round(w / n / 100, 4), "top1_place_roi": round(pl / n / 100, 4)}

    groups = {
        "調教": ["train_rank", "train_rank_rel"],
        "備考": ["rm_last1_any", "rm_last1_slow", "rm_n5_any", "rm_n5_slow"],
        "上がり": ["agari_last1", "agari_mean5"],
    }
    sets = {"base": FEATURES, **{f"+{k}": FEATURES + v for k, v in groups.items()},
            "+全部": FEATURES + [f for v in groups.values() for f in v]}
    coverage = {f: round(sum(1 for r in te if not math.isnan(r[f])) / len(te), 3) for v in groups.values() for f in v}
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    result = {"split": SPLIT, "test_rows": len(te), "coverage_in_test": coverage, "sets": {}}
    for name, feats in sets.items():
        m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
        b = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
        p = b.predict(m(te))
        y = [r["label"] for r in te]
        result["sets"][name] = {"logloss": round(log_loss(y, p), 5), "auc": round(roc_auc_score(y, p), 5),
                                "top3_hit": round(race_top3_hit(te, p), 4), **roi(te, p)}
        print(name, result["sets"][name], file=sys.stderr)
    (OUT / "new_info_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
