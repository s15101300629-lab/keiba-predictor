"""LightGBM のハイパーパラメータ調整（Optuna）と、その結果の正直な検証.

    cd scripts/ml && python cache_dataset.py          # 先に特徴量の表を作る（data/ml/cache/dataset.npz）
    cd scripts/ml && python tune_params.py tune 80    # 調整（続きから再開できる。data/ml/cache/tune.db）
    cd scripts/ml && python tune_params.py eval       # 今の PARAMS と調整後を比べる → data/ml/param_tuning.json

調整（tune）
- 2023・2024・2025年をそれぞれ検証の年にして、その前の年までで学習（早期終了）。検証の logloss の平均を小さくする
- 学習率は 0.05 に固定（速くするため）。2026年は調整に使わない

検証（eval）
- 前年までで学習して 2024〜2026年を予測（backtest_years と同じ）。木の本数は、その前年を検証にして決める
- 学習率 0.03。2026年だけは調整にまったく使っていない年なので、ここが一番正直な数字
- あわせて「レース内で確率の合計を 3 にそろえる」補正（各レースで logit に同じ値を足す）の効果も見る
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import OUT, PARAMS  # noqa: E402

CACHE = OUT / "cache"
TUNE_LR = 0.05
VAL_YEARS = ["2023", "2024", "2025"]


HOLDOUT = "2026-11-21"  # これ以降のレースは使わない（最後の確かめ用に残す。2026-11-24 から）


def load():
    d = np.load(CACHE / "dataset.npz")
    keep = d["date"].astype(str) < HOLDOUT
    year = np.array([s[:4] for s in d["date"][keep]])
    return d["X"][keep], d["label"][keep].astype(float), year, d["race_id"][keep], d["finish"][keep]


def race_top3_hit(race_id, y, p) -> float:
    order = np.lexsort((-p, race_id))
    rid, yy = race_id[order], y[order]
    start = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1]
    rank = np.arange(len(rid)) - np.repeat(start, np.diff(np.r_[start, len(rid)]))
    sel = rank < 3
    return float(yy[sel].sum() / sel.sum())


def normalize_sum(race_id, p, target=3.0):
    """各レースで logit に同じ値 c を足し、確率の合計を min(target, 頭数) にそろえる（二分法）."""
    out = np.empty_like(p)
    logit = np.log(p / (1 - p))
    _, inv = np.unique(race_id, return_inverse=True)
    for g in range(inv.max() + 1):
        idx = np.flatnonzero(inv == g)
        z = logit[idx]
        t = min(target, len(idx) - 0.5)
        lo, hi = -10.0, 10.0
        for _ in range(50):
            c = (lo + hi) / 2
            if (1 / (1 + np.exp(-(z + c)))).sum() > t:
                hi = c
            else:
                lo = c
        out[idx] = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    return out


def space(trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 30, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 30, log=True),
    }


def study_name(n_features: int) -> str:
    return "lgb" if n_features == 77 else f"lgb_{n_features}"  # 77項目のときの調整は "lgb"（data/ml/param_tuning.json）


def tune(n_trials: int) -> None:
    import optuna

    X, y, year, _, _ = load()
    sets = []
    for vy in VAL_YEARS:
        fit, val = year < vy, year == vy
        dfit = lgb.Dataset(X[fit], y[fit], free_raw_data=False, params={"feature_pre_filter": False})
        dval = lgb.Dataset(X[val], y[val], reference=dfit)
        sets.append((dfit, dval))

    def objective(trial):
        p = dict(PARAMS, learning_rate=TUNE_LR, feature_pre_filter=False, **space(trial))
        losses, iters = [], []
        for i, (dfit, dval) in enumerate(sets):
            b = lgb.train(p, dfit, 5000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
            losses.append(b.best_score["valid_0"]["binary_logloss"])
            iters.append(b.best_iteration)
            trial.report(float(np.mean(losses)), i)
        trial.set_user_attr("losses", losses)
        trial.set_user_attr("iters", iters)
        return float(np.mean(losses))

    study = optuna.create_study(study_name=study_name(X.shape[1]), storage=f"sqlite:///{CACHE / 'tune.db'}", load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    if not study.trials:  # 今の設定（train.py の PARAMS）を最初に入れて基準にする
        study.enqueue_trial({k: PARAMS.get(k, 0.0) for k in ("num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction",
                                                              "lambda_l1", "lambda_l2", "min_gain_to_split", "min_sum_hessian_in_leaf")})
    done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    import os  # TUNE_TIMEOUT（秒）：Actions の時間の上限に収めるため
    study.optimize(objective, n_trials=max(0, n_trials - done), timeout=float(os.environ["TUNE_TIMEOUT"]) if os.environ.get("TUNE_TIMEOUT") else None)
    print("best", study.best_value, study.best_params, file=sys.stderr)


def evaluate(X, y, year, race_id, finish, params: dict) -> dict:
    res = {}
    for ty in ["2024", "2025", "2026"]:
        vy = str(int(ty) - 1)
        fit, val, tr, te = year < vy, year == vy, year < ty, year == ty
        dfit = lgb.Dataset(X[fit], y[fit])
        b = lgb.train(params, dfit, 6000, valid_sets=[lgb.Dataset(X[val], y[val], reference=dfit)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = max(b.best_iteration, 50)
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), rounds)
        p = m.predict(X[te])
        pn = normalize_sum(race_id[te], p)
        res[ty] = {"rounds": rounds, "auc": roc_auc_score(y[te], p), "logloss": log_loss(y[te], p),
                   "top3_hit": race_top3_hit(race_id[te], y[te], p),
                   "logloss_sum3": log_loss(y[te], pn), "auc_sum3": roc_auc_score(y[te], pn),
                   "top1_win": float(np.mean([finish[te][race_id[te] == r][np.argmax(p[race_id[te] == r])] == 1
                                              for r in np.unique(race_id[te])]))}
        print(ty, res[ty], file=sys.stderr, flush=True)
    res["mean"] = {k: float(np.mean([res[t][k] for t in ["2024", "2025", "2026"]])) for k in res["2024"]}
    return res


def main() -> None:
    if sys.argv[1] == "tune":
        tune(int(sys.argv[2]) if len(sys.argv) > 2 else 80)
        return
    import optuna

    X, y, year, race_id, finish = load()
    study = optuna.load_study(study_name=study_name(X.shape[1]), storage=f"sqlite:///{CACHE / 'tune.db'}")
    best = study.best_params
    out = {"tuning": {"trials": len(study.trials), "val_years": VAL_YEARS, "learning_rate": TUNE_LR,
                      "baseline_val_logloss": study.trials[0].value, "best_val_logloss": study.best_value,
                      "best_val_losses": study.best_trial.user_attrs["losses"], "best_params": best},
           "current": evaluate(X, y, year, race_id, finish, PARAMS),
           "tuned": evaluate(X, y, year, race_id, finish, dict(PARAMS, **best))}
    (OUT / ("param_tuning.json" if X.shape[1] == 77 else "param_tuning_final.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
