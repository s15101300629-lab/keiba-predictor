"""依頼者の7項目（2026-11-24）のうち、2・3・6・7 の比較.

    cd scripts/ml && python cache_dataset.py                 # 特徴量の表（data/ml/cache/dataset.npz）
    cd scripts/ml && python compare_round4.py venue          # 3：競馬場・性別を種類として → data/ml/round4_venue.json
    cd scripts/ml && python compare_round4.py missing        # 2：欠けた入力も学習 → data/ml/round4_missing.json
    cd scripts/ml && python compare_round4.py rounds         # 6：本番の木の本数（検証で決めた本数のまま／学習データの量の比で増やす）
    cd scripts/ml && python compare_round4.py tuned          # 6：調整し直した設定（tune_params.py tune の結果）との比較
    cd scripts/ml && python compare_round4.py seeds          # 7：seed を変えた5本のばらつき → data/ml/round4_seeds.json

共通：前年までで学習して 2024・2025・2026年を予測（木の本数は前年を検証にして決める）。3着以内の logloss は合計3にそろえた後。
2026-11-21 以降のレースは使わない（最後の確かめ用に残す）
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import OUT, PARAMS  # noqa: E402
from tune_params import normalize_sum, race_top3_hit  # noqa: E402

CACHE = OUT / "cache"
HOLDOUT = "2026-11-21"
YEARS = ["2024", "2025", "2026"]
SEX_MEDIAN = {0: 476.0, 1: 448.0, 2: 468.0}  # 初出走馬の性別ごとの馬体重の中央値（アプリと同じ）


def load():
    d = np.load(CACHE / "dataset.npz")
    keep = d["date"] < HOLDOUT
    feats = [str(f) for f in d["features"]]
    return {"X": d["X"][keep], "y": d["label"][keep].astype(float), "date": d["date"][keep].astype(str),
            "year": np.array([s[:4] for s in d["date"][keep]]), "race_id": d["race_id"][keep].astype(str), "finish": d["finish"][keep],
            "pop": d["popularity"][keep], "features": feats}


def metrics(y, p, race_id, finish) -> dict:
    pn = normalize_sum(race_id, np.clip(p, 1e-9, 1 - 1e-9))
    return {"logloss_sum3": round(float(log_loss(y, pn)), 5), "auc": round(float(roc_auc_score(y, p)), 5),
            "top3_hit": round(race_top3_hit(race_id, y, p), 5)}


def fit_year(X, y, year, ty, params, cat=None, rounds_scale=False, augment=None, seed=None):
    """ty を予測するモデル（ty より前で学習。木の本数は ty−1 を検証にして ty−2 までで決める）."""
    vy = str(int(ty) - 1)
    fit, val, tr = year < vy, year == vy, year < ty
    p = dict(params)
    if seed is not None:
        p.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    Xf, Xt = X[fit], X[tr]
    if augment:
        Xf, Xt = augment(Xf, 1), augment(Xt, 2)
    kw = {"categorical_feature": cat} if cat else {}
    dfit = lgb.Dataset(Xf, y[fit], **kw)
    b = lgb.train(p, dfit, 6000, valid_sets=[lgb.Dataset(X[val], y[val], reference=dfit, **kw)], callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = max(b.best_iteration, 50)
    if rounds_scale:
        rounds = int(round(rounds * tr.sum() / fit.sum()))
    return lgb.train(p, lgb.Dataset(Xt, y[tr], **kw), rounds), rounds


def run_variants(d, variants: dict) -> dict:
    out = {}
    for name, (X, params, cat, extra) in variants.items():
        res = {}
        for ty in YEARS:
            m, rounds = fit_year(X, d["y"], d["year"], ty, params, cat, **extra)
            te = d["year"] == ty
            res[ty] = {"rounds": rounds, **metrics(d["y"][te], m.predict(X[te]), d["race_id"][te], d["finish"][te])}
            print(name, ty, res[ty], file=sys.stderr, flush=True)
        res["mean"] = {k: round(float(np.mean([res[t][k] for t in YEARS])), 5) for k in ("logloss_sum3", "auc", "top3_hit")}
        out[name] = res
    base = next(iter(out))
    for name in out:
        out[name]["improved_years"] = sum(out[name][t]["logloss_sum3"] < out[base][t]["logloss_sum3"] for t in YEARS)
    return out


def write(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ---------- 3：競馬場・性別を種類として ----------
def venue(d) -> None:
    F = d["features"]
    X = d["X"]
    vi, si = F.index("venue"), F.index("sex")
    onehot = [X]
    for code in range(1, 11):
        onehot.append((X[:, vi] == code).astype(float)[:, None])
    for code in range(3):
        onehot.append(np.where(np.isnan(X[:, si]), np.nan, (X[:, si] == code).astype(float))[:, None])
    Xo = np.delete(np.hstack(onehot), [vi, si], axis=1)
    pc = dict(PARAMS, max_cat_to_onehot=4, cat_smooth=10)
    out = run_variants(d, {
        "今（番号を数値のまま）": (X, PARAMS, None, {}),
        "0/1 の列（競馬場10・性別3）": (Xo, PARAMS, None, {}),
        "カテゴリ機能（競馬場・性別）": (X, pc, [vi, si], {}),
    })
    # ほかに番号を数値として入れている項目（目で確かめた結果）
    out["note"] = {"数値のままの種類の項目": ["venue（JRA の番号 1〜10）", "sex（牡0・牝1・セ2）"],
                   "順序に意味があるので数値のまま": ["going（良0〜不良3）", "surface（芝0・ダート1）", "course_letter（A〜D。内から外へ）", "fit_turn など（回り・直線の区分は比較用の値）"]}
    write("round4_venue.json", out)


# ---------- 2：欠けた入力も学習 ----------
GROUPS = {"馬体重": ["body_weight", "body_weight_diff"], "枠順": ["waku", "umaban", "umaban_ratio"]}


def missing(d) -> None:
    F = d["features"]
    X, y, year, rid, fin = d["X"], d["y"], d["year"], d["race_id"], d["finish"]
    gi = {g: [F.index(f) for f in fs] for g, fs in GROUPS.items()}
    bw, bwd, sx = F.index("body_weight"), F.index("body_weight_diff"), F.index("sex")

    def aug(q):
        def f(A, salt):
            A = A.copy()
            rng = np.random.default_rng(100 + salt)
            for idx in gi.values():
                sel = rng.random(len(A)) < q
                A[np.ix_(sel, idx)] = np.nan
            return A
        return f

    def fill_bw(A):  # 今の手当て：前走の馬体重（＝今回−増減）・増減0。前走がなければ性別の中央値
        A = A.copy()
        prev = A[:, bw] - A[:, bwd]
        med = np.array([SEX_MEDIAN.get(int(s), 468.0) if not np.isnan(s) else 468.0 for s in A[:, sx]])
        A[:, bw] = np.where(np.isnan(prev), med, prev)
        A[:, bwd] = 0.0
        return A

    def nan_group(A, g):
        A = A.copy()
        A[:, gi[g]] = np.nan
        return A

    out = {}
    for q in (0.0, 0.05, 0.10, 0.20):
        name = f"欠けにする割合 {int(q * 100)}%"
        res = {}
        for ty in YEARS:
            m, rounds = fit_year(X, y, year, ty, PARAMS, augment=aug(q) if q else None)
            te = year == ty
            Xt = X[te]
            r = {"rounds": rounds, "欠けなし": metrics(y[te], m.predict(Xt), rid[te], fin[te]),
                 "馬体重の発表前（欠け）": metrics(y[te], m.predict(nan_group(Xt, "馬体重")), rid[te], fin[te]),
                 "馬体重の発表前（前走で埋める）": metrics(y[te], m.predict(fill_bw(Xt)), rid[te], fin[te]),
                 "枠順の確定前（欠け）": metrics(y[te], m.predict(nan_group(Xt, "枠順")), rid[te], fin[te]),
                 "両方の前（馬体重は前走・枠は欠け）": metrics(y[te], m.predict(nan_group(fill_bw(Xt), "枠順")), rid[te], fin[te]),
                 "両方の前（両方欠け）": metrics(y[te], m.predict(nan_group(nan_group(Xt, "枠順"), "馬体重")), rid[te], fin[te])}
            res[ty] = r
            print(name, ty, json.dumps(r, ensure_ascii=False), file=sys.stderr, flush=True)
        out[name] = res
    out["note"] = {"欠けにする項目": GROUPS, "学習データで欠けがない項目（model.json で missing_type None）": "約60項目。本番で欠けうるのは馬体重・枠順の2組",
                   "今の手当て": "馬体重は前走の馬体重・増減0（なければ性別の中央値）。枠順の確定前は欠けのまま（木は0として比べる）"}
    write("round4_missing.json", out)


# ---------- 6：木の本数・調整し直し ----------
def rounds(d) -> None:
    out = run_variants(d, {"検証で決めた本数のまま": (d["X"], PARAMS, None, {}),
                           "学習データの量の比で増やす": (d["X"], PARAMS, None, {"rounds_scale": True})})
    write("round4_rounds.json", out)


def tuned(d) -> None:
    import optuna
    from tune_params import study_name
    study = optuna.load_study(study_name=study_name(d["X"].shape[1]), storage=f"sqlite:///{CACHE / 'tune.db'}")
    best = study.best_params
    out = run_variants(d, {"今の設定": (d["X"], PARAMS, None, {}), "調整し直した設定": (d["X"], dict(PARAMS, **best), None, {}),
                           "調整し直した設定・本数を量の比で": (d["X"], dict(PARAMS, **best), None, {"rounds_scale": True})})
    out["tuning"] = {"trials": len(study.trials), "baseline_val_logloss": study.trials[0].value, "best_val_logloss": study.best_value,
                     "best_params": best, "best_val_losses": study.best_trial.user_attrs.get("losses")}
    write("round4_tuned.json", out)


# ---------- 7：予測の確かさ（seed を変えた5本） ----------
SEEDS = [1, 2, 3, 4, 5]


def seeds(d) -> None:
    F = d["features"]
    X, y, year, rid, fin = d["X"], d["y"], d["year"], d["race_id"], d["finish"]
    logit = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))  # noqa: E731
    out = {"seeds": SEEDS, "years": {}}
    preds = {}
    for ty in YEARS:
        te = year == ty
        ps = []
        for s in SEEDS:
            m, _ = fit_year(X, y, year, ty, PARAMS, seed=s)
            ps.append(normalize_sum(rid[te], np.clip(m.predict(X[te]), 1e-9, 1 - 1e-9)))
        P = np.array(ps)
        sd = logit(P).std(0)
        mean = P.mean(0)
        preds[ty] = (P, sd)
        col = lambda f: X[te][:, F.index(f)]  # noqa: E731
        groups = {"全体": np.ones(te.sum(), bool), "初出走（新馬など）": col("n_prior") == 0, "出走1〜2回": (col("n_prior") >= 1) & (col("n_prior") <= 2),
                  "休み明け半年以上": col("days_since_last") >= 180, "2歳": col("age") == 2, "重賞・OP（1着賞金4000万円以上）": col("prize_1st") >= 4000,
                  "16頭以上": col("field_size") >= 16, "出走10回以上": col("n_prior") >= 10}
        yr = {"logloss_mean_of_5": round(float(log_loss(y[te], mean)), 5),
              "logloss_single": round(float(np.mean([log_loss(y[te], P[k]) for k in range(len(SEEDS))])), 5),
              "top3_hit_mean_of_5": round(race_top3_hit(rid[te], y[te], mean), 5),
              "sd_logit": {g: {"n": int(m.sum()), "mean": round(float(sd[m].mean()), 4), "p90": round(float(np.percentile(sd[m], 90)), 4)} for g, m in groups.items()}}
        # ばらつき上位10%の馬の内訳
        top = sd >= np.percentile(sd, 90)
        yr["ばらつき上位10%の内訳"] = {g: round(float((m & top).sum() / top.sum()), 3) for g, m in groups.items() if g != "全体"}
        yr["全体での割合"] = {g: round(float(m.mean()), 3) for g, m in groups.items() if g != "全体"}
        # ばらつき別の確率のずれ（予測の平均と実際）
        q = np.quantile(sd, [0.25, 0.5, 0.75])
        band = np.searchsorted(q, sd)
        yr["ばらつきの帯ごと"] = [{"band": int(b), "n": int((band == b).sum()), "pred": round(float(mean[band == b].mean()), 4),
                             "actual": round(float(y[te][band == b].mean()), 4)} for b in range(4)]
        out["years"][ty] = yr
        print(ty, json.dumps(yr, ensure_ascii=False)[:600], file=sys.stderr, flush=True)
    out["ev_discount"] = ev_discount(d, preds)
    write("round4_seeds.json", out)


def ev_discount(d, preds) -> dict:
    """期待値の順の買い目：期待値＝確率×確定オッズで選ぶ／（確率−ばらつき）×オッズで選ぶ. 全部の組み合わせの確定オッズがある日（data/jra-odds）だけ.
    確率：5本それぞれの3着以内確率（合計3）から強さを逆算（ハービル）して組の確率を出し、5本の平均と標準偏差.
    候補：AI の3着以内確率の上位6頭の組（画面は印の馬。印には市場との合わせ方が要るので、ここは簡略化した目安）."""
    from backtest_bets2 import invert_strength, ticket_probs
    root = Path(__file__).resolve().parents[2] / "data"
    F = d["features"]
    te = d["year"] == "2026"
    P, _ = preds["2026"]
    rid, um = d["race_id"][te], d["X"][te][:, F.index("umaban")]
    n_points = {"ワイド": 3, "馬連": 3, "馬単": 3, "3連複": 5, "3連単": 6}
    res = {k: {t: {"points": 0, "hit_races": 0, "races": 0, "ret": 0} for t in n_points} for k in ("期待値（今）", "期待値−ばらつき")}
    days = sorted(p.name[:10] for p in (root / "jra-odds").glob("*.json.gz") if p.name[:10] < HOLDOUT)
    for day in days:
        odds = json.loads(gzip.open(root / "jra-odds" / f"{day}.json.gz", "rt", encoding="utf-8").read())["races"]
        rawp = root / "jra-results" / f"{day}.json"
        if not rawp.exists():
            continue
        pays = {r["race_id"]: r.get("payouts") or {} for dd in json.loads(rawp.read_text(encoding="utf-8"))["days"] for r in dd["races"]}
        for race_id, o in odds.items():
            idx = np.flatnonzero(rid == race_id)
            if len(idx) < 5 or not pays.get(race_id):
                continue
            nums = um[idx].astype(int)
            top6 = list(np.argsort(-P[:, idx].mean(0))[:6])
            per = []
            for k in range(P.shape[0]):
                s = invert_strength(P[k, idx], 1.0, 1.0)
                per.append({t: ticket_probs(s, 1.0, 1.0, top6, t) for t in n_points})
            for t, n in n_points.items():
                table = o.get(t, {})
                cands = []
                for c in per[0][t]:
                    key = "-".join(str(nums[i]) for i in (c if t in ("馬単", "3連単") else sorted(c, key=lambda i: nums[i])))
                    ov = table.get(key)
                    ov = ov[0] if isinstance(ov, list) else ov
                    if not ov:
                        continue
                    ps = np.array([per[k][t][c] for k in range(len(per))])
                    cands.append((key, ps.mean(), ps.std(), ov))
                if not cands:
                    continue
                floor = sorted((x[1] for x in cands), reverse=True)[min(n, len(cands)) - 1] / 3
                cands = [x for x in cands if x[1] >= floor]
                for name, score in (("期待値（今）", lambda x: x[1] * x[3]), ("期待値−ばらつき", lambda x: max(x[1] - x[2], 0) * x[3])):
                    pick = sorted(cands, key=lambda x: -score(x))[:n]
                    pay = pays[race_id].get(t, {})
                    r = res[name][t]
                    r["races"] += 1; r["points"] += len(pick)
                    got = sum(pay.get(x[0], 0) for x in pick)
                    r["ret"] += got; r["hit_races"] += got > 0
    for name in res:
        for t, r in res[name].items():
            r["roi"] = round(r["ret"] / (100 * r["points"]), 4) if r["points"] else None
            r["hit_rate"] = round(r["hit_races"] / r["races"], 4) if r["races"] else None
    return {"days": days, "note": "2026年の予測（2025年までで学習した5本）。候補は AI の上位6頭の組（簡略化）", "result": res}


def main() -> None:
    d = load()
    {"venue": venue, "missing": missing, "rounds": rounds, "tuned": tuned, "seeds": seeds}[sys.argv[1]](d)


if __name__ == "__main__":
    main()
