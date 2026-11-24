"""依頼者の案 1（2026-11-24）：着順を1つのモデルでまとめて学ぶ（Plackett-Luce）と、今の3つの二値モデルの比較.

    cd scripts/ml && python cache_dataset.py
    cd scripts/ml && python compare_pl.py base   # 今：3着以内・連対・勝率の二値モデル3つ → data/ml/pl_base.json
    cd scripts/ml && python compare_pl.py pl     # Plackett-Luce（1つのモデル）            → data/ml/pl_model.json

Plackett-Luce：各馬の強さ f（木の合計）、重み w = exp(f)。1着は全馬の w に比例して選ばれ、2着は残りの馬の w に比例、3着も同じ。
学習では、実際の1〜3着の起こりやすさ L = Π_k w_(k) / Σ_{残り} w の −log を小さくする（LightGBM の自作の目的関数。レースごとに勾配と
ヘッセ行列の対角を計算）。勝率は w の割合、連対率・3着以内・組み合わせはハービル（同じ強さのくじ引き）で出す。

共通：前年までで学習して 2024・2025・2026年を予測（木の本数は前年を検証にして決める）。2026-11-21 以降は使わない。
比べるもの：3着以内・連対・勝率の logloss（今のモデルは合計 3・2・1 にそろえて大小を直した後）と的中率、実際の 1-2-3着の対数尤度、
買い目の回収率（確定の払戻・100円均一）。買い目は「AI の3着以内確率の上位6頭の組から、確率の高い順に ワイド3・馬連3・馬単3・3連複5・3連単6点」。
画面の印の買い目は市場（オッズ）と合わせた確率と印で選ぶので、ここは AI だけの確率で比べる簡略版
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_bets2 import invert_strength, ticket_probs, top3_from_strength  # noqa: E402
from backtest_mark_bets import FIXED_N, TYPES, key, load_payouts  # noqa: E402
from train import OUT, PARAMS  # noqa: E402
from tune_params import normalize_sum, race_top3_hit  # noqa: E402

CACHE = OUT / "cache"
HOLDOUT = "2026-11-21"
YEARS = ["2024", "2025", "2026"]


def load():
    d = np.load(CACHE / "dataset.npz")
    keep = d["date"].astype(str) < HOLDOUT
    rid = d["race_id"][keep].astype(str)
    order = np.argsort(rid, kind="stable")  # レースごとに並べる
    feats = [str(f) for f in d["features"]]
    X = d["X"][keep][order]
    return {"X": X, "finish": d["finish"][keep][order], "race_id": rid[order], "year": np.array([s[:4] for s in d["date"][keep][order].astype(str)]),
            "umaban": X[:, feats.index("umaban")]}


class Groups:
    """行（レースごとに並んだもの）を、レース×最大頭数の行列にそろえる."""

    def __init__(self, race_id: np.ndarray, finish: np.ndarray):
        start = np.r_[0, np.flatnonzero(race_id[1:] != race_id[:-1]) + 1]
        size = np.diff(np.r_[start, len(race_id)])
        self.R, self.M = len(start), int(size.max())
        self.idx = np.full((self.R, self.M), -1)
        for r, (s, n) in enumerate(zip(start, size)):
            self.idx[r, :n] = np.arange(s, s + n)
        self.mask = self.idx >= 0
        # 1〜3着の位置（行列の列）。着順の小さい順（同着は並んだ順）。いなければ -1
        fin = np.where(self.mask, finish[np.clip(self.idx, 0, None)], 99)
        self.top = np.full((self.R, 3), -1)
        for r in range(self.R):
            cand = [c for c in np.argsort(fin[r], kind="stable") if fin[r, c] <= 3][:3]
            self.top[r, :len(cand)] = cand

    def stages(self, f: np.ndarray):
        """−log L と、f についての勾配・ヘッセ行列の対角."""
        F = np.where(self.mask, f[np.clip(self.idx, 0, None)], -np.inf)
        F = F - F.max(1, keepdims=True)
        W = np.where(self.mask, np.exp(F), 0.0)
        remain = self.mask.copy()
        g = np.zeros((self.R, self.M))
        h = np.zeros((self.R, self.M))
        nll = 0.0
        rows = np.arange(self.R)
        for k in range(3):
            c = self.top[:, k]
            ok = c >= 0
            S = (W * remain).sum(1)
            P = np.where(remain, W / np.where(S > 0, S, 1)[:, None], 0.0)
            g[ok] += P[ok]
            h[ok] += (P * (1 - P))[ok]
            g[rows[ok], c[ok]] -= 1
            nll -= np.log(np.clip(P[rows[ok], c[ok]], 1e-15, None)).sum()
            remain[rows[ok], c[ok]] = False
        grad = np.zeros(len(f))
        hess = np.zeros(len(f))
        grad[self.idx[self.mask]] = g[self.mask]
        hess[self.idx[self.mask]] = np.maximum(h[self.mask], 1e-6)
        return nll, grad, hess


def pl_params():
    p = {k: v for k, v in PARAMS.items() if k not in ("objective",)}
    p["metric"] = "None"
    return p


def train_pl(X, gfit: Groups, Xval, gval: Groups, Xtr, gtr: Groups):
    def fobj(preds, _):
        _, gr, he = gfit.stages(preds)
        return gr, he

    def feval(preds, _):
        nll, _, _ = gval.stages(preds)
        return "pl_nll", nll / gval.R, False

    p = pl_params()
    p["objective"] = fobj
    dfit = lgb.Dataset(X, np.zeros(len(X)), free_raw_data=False)
    dval = lgb.Dataset(Xval, np.zeros(len(Xval)), reference=dfit)
    b = lgb.train(p, dfit, 6000, valid_sets=[dval], feval=feval, callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = max(b.best_iteration, 50)

    def fobj2(preds, _):
        _, gr, he = gtr.stages(preds)
        return gr, he
    p["objective"] = fobj2
    return lgb.train(p, lgb.Dataset(Xtr, np.zeros(len(Xtr))), rounds), rounds


def race_slices(race_id):
    start = np.r_[0, np.flatnonzero(race_id[1:] != race_id[:-1]) + 1]
    return [slice(s, e) for s, e in zip(start, np.r_[start[1:], len(race_id)])]


def evaluate(win, top2, top3, strengths, d, te, pays) -> dict:
    """win・top2・top3：各馬の確率（合計 1・2・3）、strengths：レースごとの強さ（組み合わせ用）."""
    fin, rid, um = d["finish"][te], d["race_id"][te], d["umaban"][te]
    y3, y2, y1 = (fin <= 3).astype(float), (fin <= 2).astype(float), (fin == 1).astype(float)
    out = {"top3": {"logloss": log_loss(y3, top3), "auc": roc_auc_score(y3, top3), "top3_hit": race_top3_hit(rid, y3, top3)},
           "top2": {"logloss": log_loss(y2, top2), "auc": roc_auc_score(y2, top2)},
           "win": {"logloss": log_loss(y1, win), "auc": roc_auc_score(y1, win)}}
    ll_order, n_order, top1_win = 0.0, 0, 0
    bets = {t: {"points": 0, "ret": 0.0, "hit_races": 0, "races": 0} for t in TYPES}
    sl = race_slices(rid)
    for k, s in enumerate(sl):
        f = fin[s]
        top1_win += f[np.argmax(win[s])] == 1
        st = strengths[k]
        # 実際の 1-2-3着の対数尤度（ハービル）
        pos = [np.flatnonzero(f == j) for j in (1, 2, 3)]
        if all(len(x) == 1 for x in pos):
            a, b, c = pos[0][0], pos[1][0], pos[2][0]
            p = st[a] * st[b] / (1 - st[a]) * st[c] / (1 - st[a] - st[b])
            ll_order += np.log(max(p, 1e-15)); n_order += 1
        race = str(rid[s.start])
        pay = pays.get(race)
        if not pay or len(f) < 5:
            continue
        nums = um[s].astype(int)
        top6 = list(np.argsort(-top3[s], kind="stable")[:6])
        for t in TYPES:
            probs = ticket_probs(st, 1.0, 1.0, top6, t)
            pick = sorted(probs, key=lambda c: -probs[c])[:FIXED_N[t]]
            got = sum(pay.get(t, {}).get(key(t, [int(nums[i]) for i in c]), 0) for c in pick)
            r = bets[t]
            r["races"] += 1; r["points"] += len(pick); r["ret"] += got; r["hit_races"] += got > 0
    out["top1_win"] = top1_win / len(sl)
    out["order_loglik_per_race"] = ll_order / max(n_order, 1)
    for t, r in bets.items():
        r["roi"] = r["ret"] / (100 * r["points"]) if r["points"] else None
        r["hit_rate"] = r["hit_races"] / r["races"] if r["races"] else None
    tot_p, tot_r = sum(r["points"] for r in bets.values()), sum(r["ret"] for r in bets.values())
    bets["全券種"] = {"points": tot_p, "roi": tot_r / (100 * tot_p) if tot_p else None}
    out["bets"] = bets
    return json.loads(json.dumps(out, default=float))


def run(mode: str) -> None:
    d = load()
    X, fin, year, rid = d["X"], d["finish"], d["year"], d["race_id"]
    pays = load_payouts(YEARS)
    out = {"mode": mode, "note": __doc__.split("\n\n")[2].strip(), "years": {}}
    for ty in YEARS:
        vy = str(int(ty) - 1)
        fit, val, tr, te = year < vy, year == vy, year < ty, year == ty
        if mode == "base":
            preds, rounds = {}, {}
            for name, k in (("top3", 3), ("top2", 2), ("win", 1)):
                y = (fin <= k).astype(float)
                dfit = lgb.Dataset(X[fit], y[fit])
                b = lgb.train(PARAMS, dfit, 6000, valid_sets=[lgb.Dataset(X[val], y[val], reference=dfit)], callbacks=[lgb.early_stopping(100, verbose=False)])
                rounds[name] = max(b.best_iteration, 50)
                m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), rounds[name])
                preds[name] = normalize_sum(rid[te], np.clip(m.predict(X[te]), 1e-9, 1 - 1e-9), float(k))
            top3 = preds["top3"]
            top2 = np.minimum(preds["top2"], top3)
            win = np.minimum(preds["win"], top2)
            # 組み合わせ：3着以内確率から強さを逆算（今のアプリ。案 1）
            strengths = [invert_strength(top3[s], 1.0, 1.0) for s in race_slices(rid[te])]
        else:
            gfit, gval, gtr = Groups(rid[fit], fin[fit]), Groups(rid[val], fin[val]), Groups(rid[tr], fin[tr])
            m, r = train_pl(X[fit], gfit, X[val], gval, X[tr], gtr)
            rounds = {"pl": r}
            f = m.predict(X[te], raw_score=True)
            win, top2, top3, strengths = np.zeros(te.sum()), np.zeros(te.sum()), np.zeros(te.sum()), []
            for s in race_slices(rid[te]):
                w = np.exp(f[s] - f[s].max())
                st = w / w.sum()
                strengths.append(st)
                t3, p12, _ = top3_from_strength(st, 1.0, 1.0)
                win[s], top2[s], top3[s] = st, st + p12.sum(0), t3
            win, top2, top3 = (np.clip(x, 1e-9, 1 - 1e-9) for x in (win, top2, top3))
        res = evaluate(win, top2, top3, strengths, d, te, pays)
        res["rounds"] = rounds
        out["years"][ty] = res
        print(ty, json.dumps({k: res[k] for k in ("top3", "top2", "win", "top1_win", "order_loglik_per_race")}, ensure_ascii=False),
              json.dumps(res["bets"]["全券種"]), file=sys.stderr, flush=True)
    (OUT / f"pl_{'base' if mode == 'base' else 'model'}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(sys.argv[1])
