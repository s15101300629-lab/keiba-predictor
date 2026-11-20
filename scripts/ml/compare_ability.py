"""能力指数（補正した走破タイム）とその集計・レース内の比較の検証.

    cd scripts/ml && python compare_ability.py build   # → data/ml/cache/ability.npz
    cd scripts/ml && python compare_ability.py eval    # → data/ml/feature_comparison_ability.json

1走ごとの能力指数（1000m あたりの秒。大きいほど速い）
- tidx：今のタイム指数。同じ競馬場・芝ダ・距離・馬場の勝ちタイムの中央値との差（features.py と同じ）
- 馬場差（その日の速さ）：同じ日・同じ競馬場・芝ダのレースで「勝ち馬の tidx − そのクラス（1着賞金の区分）の勝ち馬の tidx の中央値」の中央値。
  その日のレース数 n で n/(n+2) を掛けて縮める。指数からこれを引く
- 斤量：同じ馬・同じ芝ダの続けて走ったレースどうしで「指数の差」を「斤量の差」で回帰した傾き（データから決める）で、55kg にそろえる
- ab0 = tidx、ab1 = tidx − 馬場差、ab2 = ab1 + 斤量補正

予測するレースの馬ごと（今回と同じ芝ダの過去走、新しい順に最大5走）
- 直近加重平均（重み 0.8^k）、最高、前走、傾き（古い→新しい、3走以上）、ばらつき（標準偏差、3走以上）、走数
レース内の比較（直近加重平均・最高について）
- 順位÷頭数、偏差（出走馬の平均・標準偏差で標準化）、平均との差、最高との差。あわせて出走馬全体のばらつき

評価：前年までで学習して 2024〜2026年を予測（tune_params.evaluate）。パラメータは param_tuning.json の調整後があればそれ
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset, load_races  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402

CACHE = OUT / "cache" / "ability.npz"
NAN = float("nan")
RUNS = 5
DECAY = 0.8


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return NAN if n == 0 else xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def class_band(prize) -> int:
    """1着賞金（円）の区分."""
    if not prize:
        return -1
    p = prize / 10000
    for i, cut in enumerate((600, 800, 1100, 1600, 2500, 5000)):
        if p < cut:
            return i
    return 6


def run_abilities(races: list[dict]) -> tuple[dict, dict]:
    """(race_id, 馬名) → {ab0, ab1, ab2, surface, date, weight}. 斤量の係数も返す."""
    winner = {}
    for r in races:
        w = [h["tidx"] for h in r["runners"] if h["finish"] == 1 and h.get("tidx") is not None]
        if w:
            winner[r["race_id"]] = w[0]
    by_class = defaultdict(list)
    for r in races:
        if r["race_id"] in winner:
            by_class[(r["surface"], class_band(r["prize_1st"]))].append(winner[r["race_id"]])
    class_med = {k: median(v) for k, v in by_class.items()}
    day = defaultdict(list)
    for r in races:
        if r["race_id"] in winner:
            day[(r["date"], r["venue"], r["surface"])].append(winner[r["race_id"]] - class_med[(r["surface"], class_band(r["prize_1st"]))])
    variant = {k: median(v) * len(v) / (len(v) + 2) for k, v in day.items()}

    runs = {}
    for r in races:
        v = variant.get((r["date"], r["venue"], r["surface"]), 0.0)
        for h in r["runners"]:
            t = h.get("tidx")
            if t is None:
                continue
            runs[(r["race_id"], h["horse"])] = {"race_id": r["race_id"], "ab0": t, "ab1": t - v, "surface": r["surface"], "date": r["date"],
                                                "weight": h.get("weight"), "horse": h["horse"]}
    # 斤量の係数：同じ馬・同じ芝ダで続けて走ったレースの差分で回帰（原点を通る）
    seq = defaultdict(list)
    for x in runs.values():
        seq[(x["horse"], x["surface"])].append(x)
    sxy = sxx = 0.0
    for xs in seq.values():
        xs.sort(key=lambda x: x["date"])
        for a, b in zip(xs, xs[1:]):
            if a["weight"] and b["weight"] and b["weight"] != a["weight"]:
                dw = b["weight"] - a["weight"]
                da = b["ab1"] - a["ab1"]
                if abs(da) < 3:
                    sxy += dw * da
                    sxx += dw * dw
    k = sxy / sxx if sxx else 0.0  # 1kg 重いと指数がどれだけ変わるか（ふつうは負）
    for x in runs.values():
        x["ab2"] = x["ab1"] - k * ((x["weight"] or 55.0) - 55.0)
    return runs, {"weight_coef_per_kg": k, "variant_days": len(variant),
                  "variant_abs_median": median([abs(v) for v in variant.values()])}


def run_agari(races: list[dict]) -> dict:
    """(race_id, 馬名) → 推定上がりの指標（estimate_agari.py の係数。最後から2つ目のコーナー、なければ最後）.
    ag_dev：レース内の推定上がりの平均 − 自分（秒。大きいほど速い）、ag_rank：推定上がりの順位÷頭数（小さいほど速い）
    early_gap：ゴール前600m 地点での先頭との差（秒）、fade：自分の上がり − レースの上がり（先頭のラップ）"""
    from estimate_agari import gap_counts

    coef = json.loads((OUT / "agari_estimate.json").read_text())["corner_last-1"]["sec_per_symbol"]
    w = [coef[","], coef["-"], coef["＝"], coef["join"]]
    raw = {}
    for path in sorted((DATA / "json").glob("*/*.json")):
        for r in json.loads(path.read_text(encoding="utf-8"))["races"]:
            if isinstance(r, dict) and r.get("race_id") and r.get("last_3f") and r.get("corners"):
                raw[r["race_id"]] = (r["corners"][-2 if len(r["corners"]) >= 2 else -1]["order"], r["last_3f"])
    out = {}
    for r in races:
        if r["race_id"] not in raw:
            continue
        order, last3f = raw[r["race_id"]]
        gaps = gap_counts(order)
        fin = [h for h in r["runners"] if h["finish"] is not None and h.get("time_sec") and h.get("umaban") in gaps]
        if len(fin) < 3:
            continue
        lead600 = min(h["time_sec"] for h in fin) - last3f
        est = {}
        for h in fin:
            g = sum(a * b for a, b in zip(w, gaps[h["umaban"]])) + coef["intercept"]
            est[h["horse"]] = (h["time_sec"] - lead600 - g, g)
        m = sum(a for a, _ in est.values()) / len(est)
        srt = sorted(a for a, _ in est.values())
        for name, (a, g) in est.items():
            out[(r["race_id"], name)] = {"ag_dev": m - a, "ag_rank": (1 + srt.index(a)) / len(est), "early_gap": g, "fade": a - last3f}
    return out


def summarize(vals: list[float]) -> dict:
    """新しい順の指数の列 → 集計."""
    n = len(vals)
    if n == 0:
        return {"wmean": NAN, "best": NAN, "last": NAN, "trend": NAN, "std": NAN, "n": 0}
    w = [DECAY ** i for i in range(n)]
    out = {"wmean": sum(a * b for a, b in zip(w, vals)) / sum(w), "best": max(vals), "last": vals[0], "n": n,
           "trend": NAN, "std": NAN}
    if n >= 3:
        xs = list(range(n))[::-1]  # 古い走ほど小さい x
        mx, my = sum(xs) / n, sum(vals) / n
        out["trend"] = sum((x - mx) * (y - my) for x, y in zip(xs, vals)) / sum((x - mx) ** 2 for x in xs)
        out["std"] = math.sqrt(sum((y - my) ** 2 for y in vals) / (n - 1))
    return out


def add_relative(race_rows: list[dict], key: str) -> None:
    vals = [r[key] for r in race_rows if not math.isnan(r[key])]
    n = len(race_rows)
    if not vals:
        for r in race_rows:
            for s in ("rank", "z", "dmean", "dmax"):
                r[f"{key}_{s}"] = NAN
            r[f"{key}_fstd"] = NAN
        return
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else NAN
    top = max(vals)
    for r in race_rows:
        v = r[key]
        ok = not math.isnan(v)
        r[f"{key}_rank"] = (1 + sum(1 for x in vals if x > v)) / n if ok else NAN
        r[f"{key}_z"] = (v - m) / sd if ok and sd and sd > 0 else NAN
        r[f"{key}_dmean"] = v - m if ok else NAN
        r[f"{key}_dmax"] = v - top if ok else NAN
        r[f"{key}_fstd"] = sd


def columns(ver: str) -> list[str]:
    own = [f"{ver}_{s}" for s in ("wmean", "best", "last", "trend", "std", "n")]
    rel = [f"{ver}_{k}_{s}" for k in ("wmean", "best") for s in ("rank", "z", "dmean", "dmax")] + [f"{ver}_wmean_fstd"]
    return own + rel


AGARI_KEYS = ("ag_dev", "ag_rank", "early_gap", "fade")


def agari_columns() -> list[str]:
    own = [f"{k}_{s}" for k in AGARI_KEYS for s in ("wmean", "best")]
    return own + [f"ag_dev_wmean_{s}" for s in ("rank", "z", "dmean", "dmax", "fstd")]


def build() -> None:
    races = load_races(DATA)
    runs, info = run_abilities(races)
    agari = run_agari(races)
    info["agari_runs"] = len(agari)
    print("info", info, file=sys.stderr, flush=True)
    rows, _ = build_dataset(DATA)
    past = defaultdict(list)  # 馬 → [(日付, 芝ダ, ab0, ab1, ab2)]（日付順）
    for x in sorted(runs.values(), key=lambda x: x["date"]):
        past[x["horse"]].append(x)
    surface_of = {r["race_id"]: r["surface"] for r in races}
    by_race = defaultdict(list)
    for r in rows:
        prev = [x for x in past.get(r["horse"], []) if x["date"] < r["date"] and x["surface"] == surface_of[r["race_id"]]][-RUNS:][::-1]
        for ver in ("ab0", "ab1", "ab2"):
            for s, v in summarize([x[ver] for x in prev]).items():
                r[f"{ver}_{s}"] = v
        ag = [agari[(x["race_id"], r["horse"])] for x in prev if (x["race_id"], r["horse"]) in agari]
        for key in AGARI_KEYS:
            sm = summarize([a[key] for a in ag])
            r[f"{key}_wmean"] = sm["wmean"]
            r[f"{key}_best"] = (min if key in ("ag_rank", "early_gap", "fade") else max)([a[key] for a in ag]) if ag else NAN
        by_race[r["race_id"]].append(r)
    for rs in by_race.values():
        for ver in ("ab0", "ab1", "ab2"):
            add_relative(rs, f"{ver}_wmean")
            add_relative(rs, f"{ver}_best")
        add_relative(rs, "ag_dev_wmean")
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    cols = FEATURES + columns("ab0") + columns("ab1") + columns("ab2") + agari_columns()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, X=np.array([[r[f] for f in cols] for r in rows], dtype=np.float64), features=np.array(cols),
        label=np.array([r["label"] for r in rows]),
        finish=np.array([r["finish"] if r["finish"] is not None else 99 for r in rows]),
        date=np.array([r["date"] for r in rows]), race_id=np.array([r["race_id"] for r in rows]),
        info=np.array(json.dumps(info)))
    print(f"wrote {CACHE} rows {len(rows)}", file=sys.stderr)


def evaluate_sets() -> None:
    from tune_params import evaluate

    d = np.load(CACHE)
    cols = list(d["features"])
    X, y, race_id, finish = d["X"], d["label"].astype(float), d["race_id"], d["finish"]
    year = np.array([s[:4] for s in d["date"]])
    tuned = OUT / "param_tuning.json"
    params = dict(PARAMS, **json.loads(tuned.read_text())["tuning"]["best_params"]) if tuned.exists() else PARAMS
    own = lambda v: columns(v)[:6]  # noqa: E731
    rel = lambda v: columns(v)[6:]  # noqa: E731
    sets = {
        "base": FEATURES,
        "+集計(今の指数)": FEATURES + own("ab0"),
        "+集計(馬場差補正)": FEATURES + own("ab1"),
        "+集計(馬場差+斤量)": FEATURES + own("ab2"),
        "+集計+比較(馬場差+斤量)": FEATURES + own("ab2") + rel("ab2"),
        "+集計+比較(馬場差)": FEATURES + own("ab1") + rel("ab1"),
        "+推定上がり": FEATURES + agari_columns(),
        "+能力(馬場差+斤量)+推定上がり": FEATURES + own("ab2") + rel("ab2") + agari_columns(),
    }
    out = {"info": json.loads(str(d["info"])), "sets": {}}
    for name, feats in sets.items():
        idx = [cols.index(f) for f in feats]
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X[:, idx], y, year, race_id, finish, params)
        if name != "base":
            base = out["sets"]["base"]
            res["years_logloss_better"] = sum(res[t]["logloss"] < base[t]["logloss"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_ability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else evaluate_sets()
