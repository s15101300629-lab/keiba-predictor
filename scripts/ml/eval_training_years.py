"""netkeiba の調教評価（A〜D・短評）の効果を、2025年・2026年を1年ずつ外して確かめる.

    cd scripts/ml && python eval_training_years.py   # → data/ml/training_eval_years.json

- 調教評価は 2025・2026年だけある（それより前は欠損として学習に入る）
- loyo：評価する年を除いた全部（2020年〜）で学習 → その年を予測（他の比較スクリプトと同じ「1年ずつ外す」）
- forward：前年までで学習 → 2026年を予測（2025年の調教評価だけで学んだ、本番に近い形）
- base（今の特徴量）と +調教（今回の評価 A=4〜D=1、出走馬の平均との差、出走馬の中での順位÷頭数）を比べる
- 指標：logloss・AUC・上位3頭の3着内率、1位の馬の単勝・複勝の回収率。初出走の馬（過去走0）だけの logloss も出す
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
from eval_new_info import RANK, load_jsonl, umaban_of  # noqa: E402
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NAN = float("nan")
TRAIN_FEATS = ["train_rank", "train_rank_rel", "train_rank_pos"]


def main() -> None:
    training = load_jsonl("data/training/*.jsonl")
    rank = {(rid, h["umaban"]): RANK.get(h.get("rank") or "", NAN) for rid, hs in training.items() for h in hs if h.get("umaban")}
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    races: dict[str, list[dict]] = {}
    for r in rows:
        r["train_rank"] = rank.get((r["race_id"], umaban_of(r)), NAN)
        races.setdefault(r["race_id"], []).append(r)
    for rs in races.values():
        vals = [r["train_rank"] for r in rs if not math.isnan(r["train_rank"])]
        mean = sum(vals) / len(vals) if vals else NAN
        for r in rs:
            v = r["train_rank"]
            ok = vals and not math.isnan(v)
            r["train_rank_rel"] = v - mean if ok else NAN
            # 自分より評価が高い馬の数 ÷ 頭数（0 なら一番上の評価）
            r["train_rank_pos"] = sum(1 for x in vals if x > v) / len(rs) if ok else NAN

    place_pay, win_pay = {}, {}
    for f in glob.glob(str(ROOT / "data/jra-seiseki/csv/*/payouts.csv")):
        for p in csv.DictReader(open(f, encoding="utf-8")):
            if p["type"] in ("複勝", "単勝") and p["combo"] and p["payout"]:
                try:
                    (place_pay if p["type"] == "複勝" else win_pay)[(p["race_id"], int(p["combo"]))] = int(p["payout"].replace(",", ""))
                except ValueError:
                    pass

    def roi(rs, pred):
        by: dict[str, list] = {}
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
        return {"top1_win_roi": round(w / n / 100, 4), "top1_place_roi": round(pl / n / 100, 4)}

    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    sets = {"base": FEATURES, "+調教": FEATURES + TRAIN_FEATS}

    def evaluate(tr, te):
        out = {"test_races": len({r["race_id"] for r in te}),
               "coverage": round(sum(1 for r in te if not math.isnan(r["train_rank"])) / len(te), 3)}
        y = np.array([r["label"] for r in te])
        debut = np.array([r["n_prior"] == 0 for r in te])
        for name, feats in sets.items():
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            b = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = b.predict(m(te))
            out[name] = {"logloss": round(log_loss(y, p), 5), "auc": round(roc_auc_score(y, p), 5),
                         "top3_hit": round(race_top3_hit(te, p), 4), **roi(te, p),
                         "debut_logloss": round(log_loss(y[debut], p[debut]), 5) if debut.sum() else None}
            print(name, out[name], file=sys.stderr, flush=True)
        return out

    result = {"rounds": rounds, "features": TRAIN_FEATS, "loyo": {}, "forward": {}}
    for year in ["2025", "2026"]:
        tr = [r for r in rows if r["date"][:4] != year]
        te = [r for r in rows if r["date"][:4] == year]
        result["loyo"][year] = evaluate(tr, te)
    tr = [r for r in rows if r["date"][:4] < "2026"]
    te = [r for r in rows if r["date"][:4] == "2026"]
    result["forward"]["2026"] = evaluate(tr, te)
    (OUT / "training_eval_years.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
