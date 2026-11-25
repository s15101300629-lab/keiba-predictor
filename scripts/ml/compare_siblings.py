"""新馬・実績の少ない馬に、兄弟（母系の近い馬）の成績を足す比較（2026-11-25、依頼者の案）.

    cd scripts/ml && python cache_dataset.py && python compare_siblings.py   # → data/ml/siblings.json

母の ID は取っていない（依頼者の判断：取得している範囲でやる）。そこで、手元の全馬の血統（data/pedigree/all/horses.json の
父・母父・母母父）と成績表の生産者から、次の2つを「兄弟」とみなす
- 母系（fam）：母父・母母父・生産者が同じ馬。同じ母の兄弟（父ちがいを含む）と、母の姉妹の子などごく近い母系がまざる
- 全兄弟に近い（full）：それに加えて父も同じ馬
その馬自身を除き、今回のレースの日より前の JRA の走（成績表 2017年〜、馬の ID が分かる走）で、頭数・出走数・勝率・3着内率・
1頭でも勝った馬の割合・勝ったレースの1着賞金の最高（log）・走ったレースの1着賞金の平均（log）を作る。
前年までで学習して 2024〜2026年を予測（compare_round4.fit_year と同じ。2026-11-21 以降は使わない）。全体・初出走・新馬戦（全馬が初出走）で比べる
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_round4 import HOLDOUT, YEARS, fit_year, load, metrics, write  # noqa: E402
from train import PARAMS  # noqa: E402
from tune_params import normalize_sum  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEISEKI = ROOT / "data/jra-seiseki/json"
PED = ROOT / "data/pedigree/all"
STATS = ["horses", "runs", "win_rate", "top3_rate", "winners_rate", "max_lprize_won", "mean_lprize"]


def runs_table():
    """成績表の走（馬の ID が分かるもの）：日付・馬・勝ち・3着内・1着賞金（log 万円）. 鍵は (race_id, 馬番)."""
    race_horses = json.loads((PED / "race_horses.json").read_text())
    runs, by_key = [], {}
    for yd in sorted(SEISEKI.iterdir()):
        if yd.name < "2017":
            continue
        for f in sorted(yd.glob("*.json")):
            for r in json.loads(f.read_text(encoding="utf-8"))["races"]:
                if r.get("surface") not in ("芝", "ダート") or not r.get("race_id"):
                    continue
                ids = race_horses.get(r["race_id"], {})
                prize = (r.get("prize") or [0])[0] or 0
                lp = math.log(prize / 10000 + 1)
                for h in r.get("results", []):
                    hid = ids.get(str(h.get("umaban")))
                    if not hid:
                        continue
                    by_key[(r["race_id"], int(h["umaban"]))] = (hid, h.get("breeder"))
                    fin = h.get("finish")
                    if fin is None:
                        continue
                    runs.append((r["date"], hid, int(fin == 1), int(fin <= 3), lp))
    return runs, by_key


def family_keys(hid, breeder, peds):
    p = peds.get(hid)
    if not p or not breeder:
        return None, None
    nid = lambda a: (a[0] or a[1]) if a else None  # noqa: E731 ID がない祖先は名前
    mf, mmf, f = nid(p.get("MF")), nid(p.get("MMF")), nid(p.get("F"))
    if not mf or not mmf:
        return None, None
    fam = (mf, mmf, breeder)
    return fam, (f, *fam) if f else None


def sibling_features(d) -> np.ndarray:
    peds = json.loads((PED / "horses.json").read_text())
    runs, by_key = runs_table()
    print("runs", len(runs), "keys", len(by_key), file=sys.stderr, flush=True)
    horse_keys = {}
    for hid, br in set(by_key.values()):
        horse_keys[hid] = family_keys(hid, br, peds)
    groups = [defaultdict(list), defaultdict(list)]  # fam, full → 走の行
    for run in runs:
        ks = horse_keys.get(run[1])
        if not ks:
            continue
        for g, k in zip(groups, ks):
            if k:
                g[k].append(run)
    arrs = []
    for g in groups:
        a = {}
        for k, rs in g.items():
            rs.sort()
            a[k] = (np.array([r[0] for r in rs]), np.array([r[1] for r in rs]), np.array([r[2] for r in rs]),
                    np.array([r[3] for r in rs]), np.array([r[4] for r in rs]))
        arrs.append(a)
    feats = d["features"]
    um = d["X"][:, feats.index("umaban")]
    out = np.full((len(um), 2 * len(STATS)), np.nan)
    for i, (rid, u, date) in enumerate(zip(d["race_id"], um, d["date"])):
        if np.isnan(u):
            continue
        hb = by_key.get((rid, int(u)))
        if not hb:
            continue
        ks = horse_keys.get(hb[0])
        if not ks:
            continue
        for j, (a, k) in enumerate(zip(arrs, ks)):
            if not k or k not in a:
                continue
            dt, hs, w, t3, lp = a[k]
            n = np.searchsorted(dt, date)  # その日より前の走だけ
            m = hs[:n] != hb[0]
            if not m.any():
                continue
            hh, ww = hs[:n][m], w[:n][m]
            uniq = np.unique(hh)
            winners = len(np.unique(hh[ww == 1]))
            won = lp[:n][m][ww == 1]
            out[i, j * len(STATS):(j + 1) * len(STATS)] = [len(uniq), m.sum(), ww.mean(), t3[:n][m].mean(), winners / len(uniq),
                                                           won.max() if len(won) else 0.0, lp[:n][m].mean()]
        if i % 50000 == 0:
            print("rows", i, file=sys.stderr, flush=True)
    return out


def subset_metrics(y, p, rid, sel):
    pn = normalize_sum(rid, np.clip(p, 1e-9, 1 - 1e-9))
    return {"n": int(sel.sum()), "logloss_sum3": round(float(log_loss(y[sel], pn[sel])), 5), "auc": round(float(roc_auc_score(y[sel], p[sel])), 5)}


def main() -> None:
    d = load()
    assert d["date"].max() < HOLDOUT
    S = sibling_features(d)
    names = [f"sib_{g}_{s}" for g in ("fam", "full") for s in STATS]
    cover = {n: round(float((~np.isnan(S[:, k])).mean()), 4) for k, n in enumerate(names) if n.endswith("_runs")}
    first = d["X"][:, d["features"].index("n_prior")] == 0
    cover_first = {n: round(float((~np.isnan(S[first, k])).mean()), 4) for k, n in enumerate(names) if n.endswith("_runs")}
    print("cover", cover, "first", cover_first, file=sys.stderr, flush=True)
    rid, year, y = d["race_id"], d["year"], d["y"]
    # 新馬戦＝出走馬が全部初出走のレース
    max_prior = defaultdict(float)
    for r, v in zip(rid, d["X"][:, d["features"].index("n_prior")]):
        max_prior[r] = max(max_prior[r], 0 if np.isnan(v) else v)
    debut_race = np.array([max_prior[r] == 0 for r in rid])
    variants = {"今": d["X"], "+母系": np.hstack([d["X"], S[:, :len(STATS)]]), "+全兄弟に近い": np.hstack([d["X"], S[:, len(STATS):]]),
                "+両方": np.hstack([d["X"], S])}
    out = {"note": __doc__.split("\n\n")[1].strip(), "features": names, "coverage_all": cover, "coverage_first_start": cover_first, "years": {}}
    for ty in YEARS:
        te = year == ty
        res = {}
        for name, X in variants.items():
            m, rounds = fit_year(X, y, year, ty, PARAMS)
            p = m.predict(X[te])
            res[name] = {"rounds": rounds, "全体": metrics(y[te], p, rid[te], d["finish"][te]),
                         "初出走": subset_metrics(y[te], p, rid[te], first[te]),
                         "新馬戦": metrics(y[te][debut_race[te]], p[debut_race[te]], rid[te][debut_race[te]], d["finish"][te][debut_race[te]])}
            print(ty, name, json.dumps(res[name], ensure_ascii=False), file=sys.stderr, flush=True)
        out["years"][ty] = res
        write("siblings.json", out)
    base = "今"
    out["improved_years"] = {n: {s: sum(out["years"][t][n][s]["logloss_sum3"] < out["years"][t][base][s]["logloss_sum3"] for t in YEARS)
                                 for s in ("全体", "初出走", "新馬戦")} for n in variants if n != base}
    write("siblings.json", out)


if __name__ == "__main__":
    main()
