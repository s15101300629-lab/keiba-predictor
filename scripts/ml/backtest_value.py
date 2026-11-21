"""AI と市場の確率を合わせ（Benter 方式）、過小評価された馬を買ったときの回収率（前年までで学習して、その年を予測する）.

    cd scripts/ml && python backtest_value.py   # → data/ml/value_backtest.json

考え方
- オッズは特徴量に入れない。AI（今の特徴量の LightGBM）の確率と、市場の確率（確定単勝オッズの逆数を合計1に）を、後から合わせる
- 単勝：条件付きロジット p_i ∝ exp(α log AI勝率_i + β log 市場勝率_i)。α・β は「前の年」の結果で最尤推定する
- 複勝：logit(p) = a·logit(AI 3着以内) + b·logit(市場 3着以内) + c（市場 3着以内はハービルの式）。a・b・c は前の年で推定
- 買い方
  - 単勝：期待値 = 合わせた勝率 × 単勝オッズ が閾値以上の馬を全部 100円ずつ
  - 複勝：過去の複勝オッズは当たった馬の払戻しかないので、期待値は出せない。「合わせた 3着以内 ÷ 市場の 3着以内（市場だけを同じ形で補正したもの）」が
    閾値以上の馬を買う。補正前の市場（ハービルの式）と比べると、式の癖（人気薄の扱い）を AI の上乗せと取り違えるため
  - 閾値は前の年の回収率で決め（購入 300点以上の中で最大）、その年で試す。評価する年は閾値の決定に使わない
  - 比べるために、AI 単独の確率での期待値・固定の閾値の結果も出す
- 回収率の幅：その年のレースを選び直すブートストラップ（2,000回）で 95% の範囲。購入点数・的中数も出す
- 比べる基準：roi_market_matched（選んだ馬と同じ「市場の確率の帯」の配分で全馬を買った回収率）。選んだ馬は人気側に寄りやすく、
  人気薄ほど回収率が低い傾向があるので、「全馬を買う」と比べると AI の上乗せが大きく見える（第三者レビューの指摘）
- 学習：年 Y の予測は Y 年より前で学習。木の本数は Y−1 年を検証にして決める（tune_params.evaluate と同じ）。パラメータは train.py の PARAMS
- 確定オッズを使うので、レース前に見られるオッズとは少し違う（検証は甘めに出る）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, matrix  # noqa: E402

PRED_YEARS = ["2023", "2024", "2025", "2026"]  # 2023 は 2024年の閾値・合わせ方を決めるためだけに使う
TEST_YEARS = ["2024", "2025", "2026"]
WIN_THRESHOLDS = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0]
PLACE_THRESHOLDS = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
# 画面の目安（AI予想タブ）：合わせた3着以内 ÷ 補正した市場 が 1.1 以上＝妙味あり、0.9 以下＝割高、その間＝妥当
LABEL_BANDS = {"妙味あり": (1.1, 1e9), "妥当": (0.9, 1.1), "割高": (0.0, 0.9)}
MIN_BETS = 300
BOOT = 2000


def race_info() -> dict:
    """race_id → {馬名: (馬番, 単勝オッズ)}, 単勝・複勝の払戻（100円あたり）. backtest_gap.race_info と同じで 2023年から."""
    out = {}
    for path in sorted((DATA / "json").glob("*/*.json")):
        if path.parent.name < PRED_YEARS[0]:
            continue
        day = json.loads(path.read_text(encoding="utf-8"))
        for race in day.get("races", []):
            if not isinstance(race, dict) or not race.get("race_id"):
                continue
            win, place = {}, {}
            for p in race.get("payouts") or []:
                if p.get("type") == "単勝" and len(p.get("combo") or []) == 1:
                    win[p["combo"][0]] = p["payout"]
                if p.get("type") == "複勝" and len(p.get("combo") or []) == 1:
                    place[p["combo"][0]] = p["payout"]
            horses = {r["horse"]: (r.get("umaban"), r.get("odds")) for r in race.get("results") or []}
            out[race["race_id"]] = {"horses": horses, "win": win, "place": place}
    return out


def harville_top3(p: np.ndarray) -> np.ndarray:
    """勝率 p（合計1）→ 3着以内の確率（ハービルの式。backtest_gap.harville と同じ値をまとめて計算）."""
    p = np.clip(p, 1e-9, 1 - 1e-6)
    q = p / (1 - p)  # j が1着のとき、i が2着になる倍率
    second = p * (q.sum() - q)
    den = 1 - p[:, None] - p[None, :]
    A = np.where(den > 0, p[:, None] * p[None, :] / (1 - p[:, None]) / np.where(den > 0, den, 1), 0.0)
    np.fill_diagonal(A, 0.0)
    third = p * (A.sum() - A.sum(0) - A.sum(1))
    return p + second + third


def predict_years(X, year, labels: dict[str, np.ndarray], params) -> dict[str, np.ndarray]:
    out = {k: np.full(len(year), np.nan) for k in labels}
    for y in PRED_YEARS:
        vy = str(int(y) - 1)
        fit, val, tr, te = year < vy, year == vy, year < y, year == y
        for k, lab in labels.items():
            dfit = lgb.Dataset(X[fit], lab[fit])
            b = lgb.train(params, dfit, 6000, valid_sets=[lgb.Dataset(X[val], lab[val], reference=dfit)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            m = lgb.train(params, lgb.Dataset(X[tr], lab[tr]), max(b.best_iteration, 50))
            out[k][te] = m.predict(X[te])
            print(y, k, "rounds", b.best_iteration, file=sys.stderr, flush=True)
    return out


def groups(race_id):
    order = np.argsort(race_id, kind="stable")
    return np.split(order, np.flatnonzero(race_id[order][1:] != race_id[order][:-1]) + 1)


def fit_benter(gs, la, lm, win) -> tuple[float, float]:
    def nll(w):
        s = 0.0
        for g in gs:
            z = w[0] * la[g] + w[1] * lm[g]
            z = z - z.max()
            s -= (z[win[g]].sum() - np.log(np.exp(z).sum()))
        return s
    return tuple(minimize(nll, [0.5, 0.8], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3}).x)


def benter(gs, la, lm, a, b, n):
    p = np.zeros(n)
    for g in gs:
        z = a * la[g] + b * lm[g]
        z = np.exp(z - z.max())
        p[g] = z / z.sum()
    return p


def result(sel, ret, race_idx, rng, pool=None, mkt=None) -> dict:
    """sel：買う馬（bool）、ret：100円あたりの払戻（外れは0）、pool：その年の行（ブートストラップの母集団）、
    mkt：市場の確率（あれば、選んだ馬と同じ「市場の確率の帯」の配分で全馬を買ったときの回収率 roi_market_matched も出す）."""
    n = int(sel.sum())
    if n == 0:
        return {"bets": 0}
    out = {"bets": n, "hits": int((ret[sel] > 0).sum()), "roi": float(ret[sel].sum() / (100 * n)),
           "races": int(len(np.unique(race_idx[sel])))}
    if pool is None:
        pool = np.ones(len(sel), bool)
    if mkt is not None:
        # 人気の偏りの分を除く：市場の確率で20等分した帯ごとに、全馬を買った回収率を、選んだ馬の帯の配分で平均する
        edges = np.quantile(mkt[pool], np.linspace(0, 1, 21)[1:-1])
        band = np.searchsorted(edges, mkt)
        all_roi = np.array([ret[pool & (band == b)].sum() / (100 * max(1, (pool & (band == b)).sum())) for b in range(20)])
        cnt = np.bincount(band[sel], minlength=20)
        out["roi_market_matched"] = float((cnt * all_roi).sum() / cnt.sum())
    # レースを選び直すブートストラップ（その年のレースだけ）
    races = np.unique(race_idx[pool])
    stake = np.bincount(race_idx, weights=sel * 100.0, minlength=race_idx.max() + 1)
    pay = np.bincount(race_idx, weights=np.where(sel, ret, 0.0), minlength=race_idx.max() + 1)
    st, py = stake[races], pay[races]
    rois = []
    for _ in range(BOOT):
        i = rng.integers(0, len(races), len(races))
        s = st[i].sum()
        rois.append(py[i].sum() / s if s else np.nan)
    out["roi_95"] = [float(np.nanpercentile(rois, 2.5)), float(np.nanpercentile(rois, 97.5))]
    out["p_roi_over_1"] = float(np.nanmean(np.array(rois) > 1))
    return out


def prepare() -> dict:
    """年ごとの予測（AI の勝率・3着以内）とオッズ・払戻を、レースごとにそろえた配列で返す（backtest_marks.py も使う）."""
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    info = race_info()
    year = np.array([r["date"][:4] for r in rows])
    X = matrix(rows)
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    params = PARAMS  # train.py の設定（調整済み）
    cache = OUT / "cache" / "value_pred.npz"
    if cache.exists() and list(np.load(cache)["features"]) == FEATURES and len(np.load(cache)["win"]) == len(rows):
        pred = {k: np.load(cache)[k] for k in ("win", "top3")}
    else:
        pred = predict_years(X, year, {"win": (finish == 1).astype(float), "top3": (finish <= 3).astype(float)}, params)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, features=np.array(FEATURES), **pred)

    # 評価する年（2023〜）の行だけにする
    keep = np.isin(year, PRED_YEARS)
    rows = [r for r, k in zip(rows, keep) if k]
    year, finish, pwin, ptop3 = year[keep], finish[keep], pred["win"][keep], pred["top3"][keep]
    race_id = np.array([r["race_id"] for r in rows])
    odds = np.full(len(rows), np.nan)
    win_ret = np.zeros(len(rows))
    place_ret = np.zeros(len(rows))
    for i, r in enumerate(rows):
        ri = info.get(r["race_id"])
        if not ri or r["horse"] not in ri["horses"]:
            continue
        umaban, o = ri["horses"][r["horse"]]
        odds[i] = o if o else np.nan
        win_ret[i] = ri["win"].get(umaban, 0) if finish[i] == 1 else 0
        place_ret[i] = ri["place"].get(umaban, 0)
    # オッズがないレース（不成立など）は除く
    gs_all = groups(race_id)
    ok = np.ones(len(rows), bool)
    for g in gs_all:
        if np.isnan(odds[g]).any():
            ok[g] = False
    sel_rows = np.flatnonzero(ok)
    race_id, year, finish, pwin, ptop3, odds, win_ret, place_ret = (a[sel_rows] for a in (race_id, year, finish, pwin, ptop3, odds, win_ret, place_ret))
    gs = groups(race_id)
    _, race_idx = np.unique(race_id, return_inverse=True)
    n = len(race_id)
    # レースごとに合計をそろえる
    ai_win, mkt_win, ai_top3, mkt_top3 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for g in gs:
        ai_win[g] = pwin[g] / pwin[g].sum()
        m = 1 / odds[g]
        mkt_win[g] = m / m.sum()
        t = ptop3[g]
        ai_top3[g] = np.clip(t * min(3, len(g)) / t.sum(), 1e-4, 1 - 1e-4)
        mkt_top3[g] = np.clip(harville_top3(mkt_win[g]), 1e-4, 1 - 1e-4)
    la, lm = np.log(np.clip(ai_win, 1e-6, 1)), np.log(np.clip(mkt_win, 1e-6, 1))
    is_win = finish == 1
    is_top3 = finish <= 3
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    return dict(race_id=race_id, year=year, finish=finish, odds=odds, win_ret=win_ret, place_ret=place_ret, gs=gs, race_idx=race_idx, n=n,
                ai_win=ai_win, mkt_win=mkt_win, ai_top3=ai_top3, mkt_top3=mkt_top3, la=la, lm=lm, is_win=is_win, is_top3=is_top3)


def main() -> None:
    d = prepare()
    race_id, year, finish, odds, win_ret, place_ret, gs, race_idx, n = (d[k] for k in ("race_id", "year", "finish", "odds", "win_ret", "place_ret", "gs", "race_idx", "n"))
    ai_win, mkt_win, ai_top3, mkt_top3, la, lm, is_win, is_top3 = (d[k] for k in ("ai_win", "mkt_win", "ai_top3", "mkt_top3", "la", "lm", "is_win", "is_top3"))
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    rng = np.random.default_rng(0)

    out = {"min_bets_for_threshold": MIN_BETS, "years": {}}
    for y in TEST_YEARS:
        prev = str(int(y) - 1)
        mp, mt = year == prev, year == y
        gp = [g for g in gs if year[g[0]] == prev]
        gt = [g for g in gs if year[g[0]] == y]
        a, b = fit_benter(gp, la, lm, is_win)
        comb_win = benter(gs, la, lm, a, b, n)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack([logit(ai_top3[mp]), logit(mkt_top3[mp])]), is_top3[mp])
        comb_top3 = lr.predict_proba(np.column_stack([logit(ai_top3), logit(mkt_top3)]))[:, 1]
        # 市場だけを同じ形で補正したもの（ハービルの式の癖・人気薄の過大評価を直す）。AI の上乗せはこれとの比で見る
        lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(mkt_top3[mp])[:, None], is_top3[mp])
        mkt_top3_cal = lr_m.predict_proba(logit(mkt_top3)[:, None])[:, 1]
        # 単勝も同じ：市場だけの条件付きロジット（β だけ）
        _, b_m = fit_benter(gp, 0 * la, lm, is_win)
        mkt_win_cal = benter(gs, 0 * la, lm, 0.0, b_m, n)
        yr = {"races": len(gt), "runners": int(mt.sum()),
              "benter_win": {"alpha_ai": a, "beta_market": b},
              "place_coef": {"ai": float(lr.coef_[0][0]), "market": float(lr.coef_[0][1]), "intercept": float(lr.intercept_[0])},
              "logloss_win": {k: float(log_loss(is_win[mt], np.clip(p[mt], 1e-6, 1 - 1e-6))) for k, p in
                              (("ai", ai_win), ("market", mkt_win), ("combined", comb_win))},
              "logloss_top3": {k: float(log_loss(is_top3[mt], p[mt])) for k, p in
                               (("ai", ai_top3), ("market", mkt_top3), ("market_calibrated", mkt_top3_cal), ("combined", comb_top3))}}
        yr["logloss_win"]["market_calibrated"] = float(log_loss(is_win[mt], np.clip(mkt_win_cal[mt], 1e-6, 1 - 1e-6)))
        yr["benter_win"]["beta_market_only"] = b_m
        ev_comb, ev_ai = comb_win * odds, ai_win * odds
        ratio_place = comb_top3 / mkt_top3_cal

        def choose(score, ret, ths):
            best, best_roi = None, -1
            for t in ths:
                s = mp & (score >= t)
                if s.sum() >= MIN_BETS and ret[s].sum() / (100 * s.sum()) > best_roi:
                    best, best_roi = t, ret[s].sum() / (100 * s.sum())
            return best

        for name, score, ret, ths, mkt in (("単勝_合わせた期待値", ev_comb, win_ret, WIN_THRESHOLDS, mkt_win),
                                           ("単勝_AI単独の期待値", ev_ai, win_ret, WIN_THRESHOLDS, mkt_win),
                                           ("複勝_合わせた確率÷市場(補正後)", ratio_place, place_ret, PLACE_THRESHOLDS, mkt_top3_cal)):
            t = choose(score, ret, ths)
            yr[name] = {"threshold_from_prev_year": t,
                        "result": result(mt & (score >= t), ret, race_idx, rng, mt, mkt) if t is not None else {"bets": 0},
                        "fixed": {str(th): result(mt & (score >= th), ret, race_idx, rng, mt, mkt) for th in ths}}
        yr["目安ごと(複勝)"] = {k: result(mt & (ratio_place >= lo) & (ratio_place < hi), place_ret, race_idx, rng, mt, mkt_top3_cal)
                                for k, (lo, hi) in LABEL_BANDS.items()}
        yr["全馬を買う"] = {"単勝": result(mt, win_ret, race_idx, rng, mt), "複勝": result(mt, place_ret, race_idx, rng, mt)}
        out["years"][y] = yr
        print(y, json.dumps({k: v for k, v in yr.items() if k.startswith("logloss") or k.startswith("benter")}, ensure_ascii=False),
              file=sys.stderr, flush=True)
    # 3年を通した結果（その年の閾値で買った分の合計）
    tot = {}
    for name in ("単勝_合わせた期待値", "単勝_AI単独の期待値", "複勝_合わせた確率÷市場(補正後)"):
        bets = sum(out["years"][y][name]["result"].get("bets", 0) for y in TEST_YEARS)
        pay = sum(out["years"][y][name]["result"].get("roi", 0) * 100 * out["years"][y][name]["result"].get("bets", 0) for y in TEST_YEARS)
        hits = sum(out["years"][y][name]["result"].get("hits", 0) for y in TEST_YEARS)
        tot[name] = {"bets": bets, "hits": hits, "roi": pay / (100 * bets) if bets else None}
    out["total"] = tot
    # アプリ用の係数（直近の年 TEST_YEARS[-1] で当てはめ）。入力は AI の3着以内確率（モデルの出力を出走馬の合計 min(3, 頭数) に比例でそろえたもの）と、
    # 単勝オッズの逆数を合計1にした勝率からハービルの式で出した市場の3着以内確率
    last = year == TEST_YEARS[-1]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack([logit(ai_top3[last]), logit(mkt_top3[last])]), is_top3[last])
    lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(mkt_top3[last])[:, None], is_top3[last])
    combine = {"fitted_year": TEST_YEARS[-1], "n_features": len(FEATURES),
               "combined": {"ai": float(lr.coef_[0][0]), "market": float(lr.coef_[0][1]), "intercept": float(lr.intercept_[0])},
               "market_only": {"market": float(lr_m.coef_[0][0]), "intercept": float(lr_m.intercept_[0])},
               "bands": {k: list(v) for k, v in LABEL_BANDS.items()},
               "evidence": {y: {k: {kk: out["years"][y]["目安ごと(複勝)"][k].get(kk) for kk in ("bets", "hits", "roi", "roi_market_matched")}
                                for k in LABEL_BANDS} for y in TEST_YEARS}}
    (OUT / "value_combine.json").write_text(json.dumps(combine, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (OUT / "value_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(tot, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
