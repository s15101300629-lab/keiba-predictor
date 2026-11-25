"""案 B（2026-11-25、依頼者の指示）：勝率と組み合わせは着順モデル（Plackett-Luce）、3着以内は今の二値モデル、を画面と同じ形で比べる.

    cd scripts/ml && python cache_dataset.py && python compare_pl_b.py   # → data/ml/pl_b.json

画面と同じ形＝オッズと合わせた確率で AI印を付け、印の馬から確率の高い順に決まった点数（ワイド3・馬連3・馬単3・3連複5・3連単6）を買う。
2024・2025・2026年を、年ごとに前の年で決めた合わせ方で計算する（2026-11-21 以降は使わない＝着順モデルの表が HOLDOUT で切ってある）。確定オッズ・確定の払戻
- 今：◎・▲は二値の勝率モデル（AI勝率 → 市場と条件付きロジットで合わせた勝率）。組み合わせは、3着以内の合わせた確率から逆算した強さ（ハービル）
- B：◎・▲は着順モデルの勝率（強さの割合）を同じ形で市場と合わせたもの。組み合わせは、その合わせた勝率をそのまま強さにする（ハービル）
- B'：◎・▲は B と同じ、組み合わせは今と同じ（組み合わせの変更の効き目を分けて見るため）
- ☆は人気薄専用の2段目の学習に時間がかかるので、どの形でも付けない（同じ条件で比べる）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_value as bv  # noqa: E402
import compare_pl as cp  # noqa: E402
from backtest_bets2 import invert_strength, ticket_probs  # noqa: E402
from backtest_mark_bets import FIXED_N, TYPES, USE, key, load_payouts  # noqa: E402
from backtest_marks import assign  # noqa: E402
from train import OUT  # noqa: E402
from train_longshot import popularity  # noqa: E402
from tune_params import normalize_sum  # noqa: E402

YEARS = ["2024", "2025", "2026"]
PRED = ["2023"] + YEARS  # 2023 は 2024年の合わせ方を決めるため


def pl_scores() -> dict[tuple[str, int], float]:
    """年ごとに前の年までで学習した着順モデルの強さ（log）. 鍵は (race_id, 馬番)."""
    d = cp.load()
    X, fin, year, rid, um = d["X"], d["finish"], d["year"], d["race_id"], d["umaban"]
    out = {}
    for ty in PRED:
        vy = str(int(ty) - 1)
        fit, val, tr, te = year < vy, year == vy, year < ty, year == ty
        m, r, _ = cp.train_pl(X[fit], cp.Groups(rid[fit], fin[fit]), X[val], cp.Groups(rid[val], fin[val]), X[tr], cp.Groups(rid[tr], fin[tr]))
        f = m.predict(X[te], raw_score=True)
        for a, b, v in zip(rid[te], um[te], f):
            out[(str(a), int(b))] = float(v)
        print(ty, "PL rounds", r, file=sys.stderr, flush=True)
    return out


def main() -> None:
    bv.PRED_YEARS = PRED
    bv.OUT = OUT / "cache" / "pl_b"  # prepare が書く calibration.json・予測のためおきを本番のファイルと分ける（git に入れない場所）
    bv.OUT.mkdir(parents=True, exist_ok=True)
    d = bv.prepare()
    year, gs, odds = d["year"], d["gs"], d["odds"]
    logit = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))  # noqa: E731
    sc = pl_scores()
    # 着順モデルの勝率（強さの割合）。着順モデルの表にないレース（2026-11-21 以降など）は除く
    pl_win = np.full(d["n"], np.nan)
    keep = []
    for g in gs:
        f = np.array([sc.get((str(d["race_id"][i]), int(d["umaban"][i])), np.nan) for i in g])
        if np.isnan(f).any() or not (d["umaban"][g] > 0).all():
            continue
        w = np.exp(f - f.max())
        pl_win[g] = w / w.sum()
        keep.append(g)
    gs = keep
    la_pl = np.log(np.clip(np.nan_to_num(pl_win, nan=1e-6), 1e-6, 1))
    pop = popularity(d)
    pay = load_payouts(PRED)
    out = {"note": __doc__.split("\n\n")[2].strip(), "races": len(gs), "years": {}}
    for y in YEARS:
        prev = str(int(y) - 1)
        gp = [g for g in gs if year[g[0]] == prev]
        gt = [g for g in gs if year[g[0]] == y]
        res = {}
        # 合わせた勝率（今・B）
        a, b = bv.fit_benter(gp, d["la"], d["lm"], d["is_win"])
        a2, b2 = bv.fit_benter(gp, la_pl, d["lm"], d["is_win"])
        cw = bv.benter(gs, d["la"], d["lm"], a, b, d["n"])
        cw_pl = bv.benter(gs, la_pl, d["lm"], a2, b2, d["n"])
        it = np.concatenate(gt)
        yw = d["is_win"][it].astype(float)
        res["勝率"] = {
            "AIだけ 今": {"logloss": round(float(log_loss(yw, np.clip(d["ai_win"][it], 1e-9, 1))), 5), "auc": round(float(roc_auc_score(yw, d["ai_win"][it])), 5)},
            "AIだけ 着順モデル": {"logloss": round(float(log_loss(yw, np.clip(pl_win[it], 1e-9, 1))), 5), "auc": round(float(roc_auc_score(yw, pl_win[it])), 5)},
            "市場と合わせた 今": {"logloss": round(float(log_loss(yw, np.clip(cw[it], 1e-9, 1))), 5), "auc": round(float(roc_auc_score(yw, cw[it])), 5),
                            "ai_weight": round(float(a), 4), "market_weight": round(float(b), 4)},
            "市場と合わせた 着順モデル": {"logloss": round(float(log_loss(yw, np.clip(cw_pl[it], 1e-9, 1))), 5), "auc": round(float(roc_auc_score(yw, cw_pl[it])), 5),
                                 "ai_weight": round(float(a2), 4), "market_weight": round(float(b2), 4)},
        }
        # 3着以内の合わせた確率（今のアプリ。value_combine の combined の形）と、割高の比（消）
        mp = year == prev
        Xc = np.column_stack([logit(d["ai_top3"]), logit(d["mkt_top3"])])
        comb3 = LogisticRegression(C=1e6, max_iter=1000).fit(Xc[mp], d["is_top3"][mp]).predict_proba(Xc)[:, 1]
        lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(d["mkt_top3"][mp])[:, None], d["is_top3"][mp])
        ratio = comb3 / lr_m.predict_proba(logit(d["mkt_top3"])[:, None])[:, 1]
        star = np.zeros(d["n"])
        variants = {"今": (d["ai_win"], cw, "top3"), "B": (pl_win, cw_pl, "win"), "B'": (pl_win, cw_pl, "top3")}
        for name, (aiw, cwin, combo) in variants.items():
            ll, n_ord = 0.0, 0
            honmei = {"n": 0, "win": 0, "top3": 0, "win_ret": 0.0, "place_ret": 0.0}
            bets = {t: {"races": 0, "points": 0, "hit_races": 0, "ret": 0.0} for t in TYPES}
            for g in gt:
                f = d["finish"][g]
                s = cwin[g] / cwin[g].sum() if combo == "win" else invert_strength(normalize_sum(np.zeros(len(g)), comb3[g], 3.0), 1.0, 1.0)
                pos = [np.flatnonzero(f == k) for k in (1, 2, 3)]
                if all(len(x) == 1 for x in pos):
                    i, j, k = pos[0][0], pos[1][0], pos[2][0]
                    ll += np.log(max(s[i] * s[j] / (1 - s[i]) * s[k] / (1 - s[i] - s[j]), 1e-15)); n_ord += 1
                mark = assign(g, aiw, d["ai_top3"], cwin * odds, ratio, pop, "B", star)
                h = mark.index("◎") if "◎" in mark else None
                if h is not None:
                    gi = g[h]
                    honmei["n"] += 1; honmei["win"] += int(f[h] == 1); honmei["top3"] += int(f[h] <= 3)
                    honmei["win_ret"] += d["win_ret"][gi]; honmei["place_ret"] += d["place_ret"][gi]
                rid = str(d["race_id"][g[0]])
                marked = [i for i, m in enumerate(mark) if m in USE]
                if rid not in pay or len(marked) < 3:
                    continue
                nums = [int(x) for x in d["umaban"][g]]
                for t in TYPES:
                    probs = ticket_probs(s, 1.0, 1.0, marked, t)
                    pick = sorted(probs, key=lambda c: -probs[c])[:FIXED_N[t]]
                    got = sum(pay[rid].get(t, {}).get(key(t, [nums[i] for i in c]), 0) for c in pick)
                    r = bets[t]
                    r["races"] += 1; r["points"] += len(pick); r["ret"] += got; r["hit_races"] += got > 0
            pts = sum(r["points"] for r in bets.values())
            ret = sum(r["ret"] for r in bets.values())
            res[name] = {
                "order_loglik_per_race": round(ll / max(n_ord, 1), 4),
                "◎": {"n": honmei["n"], "win_rate": round(honmei["win"] / max(1, honmei["n"]), 4), "top3_rate": round(honmei["top3"] / max(1, honmei["n"]), 4),
                      "win_roi": round(honmei["win_ret"] / max(1, 100 * honmei["n"]), 4), "place_roi": round(honmei["place_ret"] / max(1, 100 * honmei["n"]), 4)},
                "bets": {t: {"races": r["races"], "hit_rate": round(r["hit_races"] / max(1, r["races"]), 4), "roi": round(r["ret"] / max(1, 100 * r["points"]), 4)} for t, r in bets.items()}
                | {"全券種": {"points": pts, "roi": round(ret / max(1, 100 * pts), 4)}},
            }
            print(y, name, json.dumps({k: res[name][k] for k in ("order_loglik_per_race", "◎")}, ensure_ascii=False), res[name]["bets"]["全券種"], file=sys.stderr, flush=True)
        print(y, json.dumps(res["勝率"], ensure_ascii=False), file=sys.stderr, flush=True)
        out["years"][y] = res
        (OUT / "pl_b.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
