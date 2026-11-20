"""成績表だけで各馬の上がり3Fを推定し、本物（2026年の備考データ・netkeiba）と比べる.

    cd scripts/ml && python estimate_agari.py   # → data/ml/agari_estimate.json

考え方
- 先頭がゴール前600m を通った時刻 = 勝ちタイム − レースの上がり3F（成績表のラップは先頭のもの）
- その馬の上がり3F = その馬の走破タイム − 先頭が600m 地点を通った時刻 − その時点での先頭との差（秒）
- 「先頭との差」は、最後（または最後から2つ目）のコーナーの通過順の記号から推定する
  - 先頭からその馬までの「,」「-」「＝」と、記号なしでつながる区切り（「)3」など）の数。（ ）の中は横並びで差 0
  - 記号1つあたりの秒は、本物の上がり3Fに合わせて最小二乗で決める（推測で決めない）
- 2026年の前半（〜6月）で係数を決め、後半（7月〜）で合い方を見る
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEISEKI = REPO / "data" / "jra-seiseki" / "json"
OUT = REPO / "data" / "ml"
KINDS = (",", "-", "＝", "join")


def gap_counts(order: str) -> dict[int, list[int]]:
    """通過順の文字列 → 馬番: 先頭からの区切りの数 [",", "-", "＝", "join"]."""
    out: dict[int, list[int]] = {}
    cnt = [0, 0, 0, 0]
    depth = 0
    num = ""
    prev = None  # 直前の区切りの種類（None=最初、"horse"=直前が馬番か")"）
    i = 0
    s = order.replace(" ", "")
    while i <= len(s):
        ch = s[i] if i < len(s) else "#"
        if ch.isdigit():
            num += ch
        else:
            if num:
                if prev == "horse" and depth == 0:  # 記号なしで続く（")3" など）
                    cnt[3] += 1
                out[int(num)] = list(cnt)
                num = ""
                prev = "horse"
            if ch == "(":
                if prev == "horse" and depth == 0:
                    cnt[3] += 1
                depth += 1
                prev = "open"
            elif ch == ")":
                depth -= 1
                prev = "horse"
            elif ch in (",", "-", "＝"):
                if depth == 0:
                    cnt[KINDS.index(ch)] += 1
                prev = "sep"
            elif ch == "*":
                pass
        i += 1
        if ch == "(" or ch in (",", "-", "＝"):
            pass
    return out


def load_actual() -> dict[tuple[str, int], float]:
    act = {}
    for line in (REPO / "data" / "remarks" / "2026.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for h in r["horses"]:
            try:
                act[(r["race_id"], int(h["umaban"]))] = float(h["agari"])
            except (TypeError, ValueError, KeyError):
                pass
    return act


def samples(corner_from_last: int):
    act = load_actual()
    rows = []
    for path in sorted((SEISEKI / "2026").glob("*.json")):
        for r in json.loads(path.read_text(encoding="utf-8"))["races"]:
            if not isinstance(r, dict) or r.get("surface") not in ("芝", "ダート") or not r.get("last_3f"):
                continue
            corners = r.get("corners") or []
            if len(corners) < corner_from_last:
                continue
            gaps = gap_counts(corners[-corner_from_last]["order"])
            fin = [h for h in r["results"] if h.get("time_sec") and h.get("finish")]
            if not fin:
                continue
            win = min(h["time_sec"] for h in fin)
            lead600 = win - r["last_3f"]
            for h in fin:
                a = act.get((r["race_id"], h["umaban"]))
                g = gaps.get(h["umaban"])
                if a is None or g is None:
                    continue
                rows.append({"race_id": r["race_id"], "month": r["month"], "gap_sec": h["time_sec"] - lead600 - a, "counts": g,
                             "actual": a, "naive": h["time_sec"] - lead600, "time_diff": h["time_sec"] - win,
                             "last3f": r["last_3f"]})
    return rows


def spearman_in_race(rows, pred) -> float:
    by = {}
    for r, p in zip(rows, pred):
        by.setdefault(r["race_id"], []).append((p, r["actual"]))
    cs = []
    for items in by.values():
        if len(items) < 5:
            continue
        p = np.argsort(np.argsort([x[0] for x in items]))
        a = np.argsort(np.argsort([x[1] for x in items]))
        cs.append(np.corrcoef(p, a)[0, 1])
    return float(np.nanmean(cs))


def main() -> None:
    result = {}
    for k in (1, 2):
        rows = samples(k)
        fit = [r for r in rows if r["month"] <= 6]
        te = [r for r in rows if r["month"] >= 7]
        A = np.array([r["counts"] + [1] for r in fit], dtype=float)
        coef, *_ = np.linalg.lstsq(A, np.array([r["gap_sec"] for r in fit]), rcond=None)
        pred = [r["naive"] - float(np.dot(r["counts"] + [1], coef)) for r in te]
        naive = [r["last3f"] + r["time_diff"] for r in te]  # 位置を見ない（上がり＝レースの上がり＋着差）
        act = np.array([r["actual"] for r in te])
        result[f"corner_last{'' if k == 1 else '-1'}"] = {
            "fit_rows": len(fit), "test_rows": len(te),
            "sec_per_symbol": dict(zip(list(KINDS) + ["intercept"], map(float, coef))),
            "mae": float(np.mean(np.abs(np.array(pred) - act))), "mae_naive": float(np.mean(np.abs(np.array(naive) - act))),
            "corr": float(np.corrcoef(pred, act)[0, 1]), "corr_naive": float(np.corrcoef(naive, act)[0, 1]),
            "rank_corr_in_race": spearman_in_race(te, pred), "rank_corr_in_race_naive": spearman_in_race(te, naive),
        }
        print(k, result[f"corner_last{'' if k == 1 else '-1'}"], file=sys.stderr, flush=True)
    (OUT / "agari_estimate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
