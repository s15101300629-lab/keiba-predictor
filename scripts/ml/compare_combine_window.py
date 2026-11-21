"""AI と市場を合わせる係数を、何年分・どう作った予測で決めるのがよいかの比較.

    cd scripts/ml && python compare_combine_window.py   # → data/ml/combine_window_comparison.json

試す年 Y（2025・2026）の予測は、どれも Y 年より前で学習したモデル（今と同じ）。係数を決める期間 W 年（Y の直前 1〜3年）の予測の作り方を2通り比べる
- 前向き（今の方式）：その年 y を、y 年より前で学習したモデルで予測
- 交差：その年 y を、「Y 年より前で、y 年を除いた年」で学習したモデルで予測（y の答えも Y 年以降も見ていない。学習の量が多い）
見るもの（試す年で）：3着以内（AI と市場を合わせたロジスティック回帰）と勝率（条件付きロジット）の logloss、係数、
☆の条件（合わせた3着以内 ÷ 市場だけを補正した3着以内 ≥ 1.1）に当たる馬の頭数・3着内率・複勝回収率
木の本数は metrics.json の値に固定（どの予測も同じ条件にするため）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_value import benter, fit_benter, groups, harville_top3  # noqa: E402
from features import build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, matrix  # noqa: E402

TEST_YEARS = ["2025", "2026"]
WINDOWS = [1, 2, 3]


def main() -> None:
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    year = np.array([r["date"][:4] for r in rows])
    X = matrix(rows)
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    labels = {"win": (finish == 1).astype(float), "top3": (finish <= 3).astype(float)}
    rounds = {"top3": json.loads((OUT / "metrics.json").read_text())["best_iteration"],
              "win": json.loads((OUT / "metrics.json").read_text())["extra_targets"]["win"]["best_iteration"]}
    cache: dict[tuple, np.ndarray] = {}

    def predict(train_years: tuple[str, ...], target_year: str, k: str) -> np.ndarray:
        key = (train_years, target_year, k)
        if key not in cache:
            tr = np.isin(year, train_years)
            m = lgb.train(PARAMS, lgb.Dataset(X[tr], labels[k][tr]), rounds[k])
            cache[key] = m.predict(X[year == target_year])
            print("trained", k, train_years, "->", target_year, file=sys.stderr, flush=True)
        return cache[key]

    all_years = sorted(set(year))
    # オッズ・払戻（2022年から。race_info は PRED_YEARS[0] 以降を読むので、ここで読み直す）
    import backtest_value as bv
    bv.PRED_YEARS = ["2022"]
    info = bv.race_info()
    odds = np.full(len(rows), np.nan)
    place_ret = np.zeros(len(rows))
    for i, r in enumerate(rows):
        ri = info.get(r["race_id"])
        if ri and r["horse"] in ri["horses"]:
            umaban, o = ri["horses"][r["horse"]]
            odds[i] = o if o else np.nan
            place_ret[i] = ri["place"].get(umaban, 0)
    race_id = np.array([r["race_id"] for r in rows])
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731

    def race_arrays(mask: np.ndarray, pwin: np.ndarray, ptop3: np.ndarray):
        """mask の行（オッズのあるレースだけ）について、合計をそろえた AI・市場の確率."""
        idx = np.flatnonzero(mask)
        gs = [g for g in groups(race_id[idx]) if not np.isnan(odds[idx][g]).any()]
        keep = np.concatenate(gs)
        idx, pw, pt = idx[keep], pwin[keep], ptop3[keep]
        gs = groups(race_id[idx])
        n = len(idx)
        ai_win, mkt_win, ai_top3, mkt_top3 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for g in gs:
            ai_win[g] = pw[g] / pw[g].sum()
            m = 1 / odds[idx][g]
            mkt_win[g] = m / m.sum()
            ai_top3[g] = np.clip(pt[g] * min(3, len(g)) / pt[g].sum(), 1e-4, 1 - 1e-4)
            mkt_top3[g] = np.clip(harville_top3(mkt_win[g]), 1e-4, 1 - 1e-4)
        return dict(idx=idx, gs=gs, ai_win=ai_win, mkt_win=mkt_win, ai_top3=ai_top3, mkt_top3=mkt_top3,
                    is_win=finish[idx] == 1, is_top3=finish[idx] <= 3, place_ret=place_ret[idx])

    out = {"rounds": rounds, "tests": {}}
    for Y in TEST_YEARS:
        past = [y for y in all_years if y < Y]
        test = race_arrays(year == Y, predict(tuple(past), Y, "win"), predict(tuple(past), Y, "top3"))
        res = {}
        for mode in ("forward", "cross"):
            for W in WINDOWS:
                fit_years = past[-W:]
                parts = []
                for y in fit_years:
                    tr = tuple(p for p in past if p < y) if mode == "forward" else tuple(p for p in past if p != y)
                    parts.append(race_arrays(year == y, predict(tr, y, "win"), predict(tr, y, "top3")))
                cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
                Xf = np.column_stack([logit(cat("ai_top3")), logit(cat("mkt_top3"))])
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(Xf, cat("is_top3"))
                lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(cat("mkt_top3"))[:, None], cat("is_top3"))
                # 勝率：レースの添字を通しにして条件付きロジット
                gs_all, off = [], 0
                for p in parts:
                    gs_all += [g + off for g in p["gs"]]
                    off += len(p["idx"])
                la_f, lm_f = np.log(np.clip(cat("ai_win"), 1e-6, 1)), np.log(np.clip(cat("mkt_win"), 1e-6, 1))
                a, b = fit_benter(gs_all, la_f, lm_f, cat("is_win"))
                Xt = np.column_stack([logit(test["ai_top3"]), logit(test["mkt_top3"])])
                comb = lr.predict_proba(Xt)[:, 1]
                mcal = lr_m.predict_proba(logit(test["mkt_top3"])[:, None])[:, 1]
                la_t, lm_t = np.log(np.clip(test["ai_win"], 1e-6, 1)), np.log(np.clip(test["mkt_win"], 1e-6, 1))
                cw = benter(test["gs"], la_t, lm_t, a, b, len(test["idx"]))
                star = comb / mcal >= 1.1
                res[f"{mode}_{W}y"] = {
                    "fit_years": fit_years, "fit_rows": int(len(cat("is_top3"))),
                    "top3_coef": {"ai": float(lr.coef_[0][0]), "market": float(lr.coef_[0][1])},
                    "win_coef": {"ai": float(a), "market": float(b)},
                    "logloss_top3": float(log_loss(test["is_top3"], comb)), "logloss_top3_market_cal": float(log_loss(test["is_top3"], mcal)),
                    "logloss_win": float(log_loss(test["is_win"], np.clip(cw, 1e-6, 1 - 1e-6))),
                    "star": {"n": int(star.sum()), "top3_rate": float(test["is_top3"][star].mean()) if star.any() else None,
                             "place_roi": float(test["place_ret"][star].sum() / (100 * star.sum())) if star.any() else None},
                }
                print(Y, mode, W, {k: v for k, v in res[f"{mode}_{W}y"].items() if k.startswith(("logloss", "top3_coef", "star"))},
                      file=sys.stderr, flush=True)
        out["tests"][Y] = res
        (OUT / "combine_window_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
