"""印から作る買い目の計算（強さの逆算・組み合わせの確率）が Python（backtest_bets2.py）と TS（app/lib/markBets.ts）で同じか確かめるための例を作る.

    cd scripts/ml && python parity_bets.py   # → data/ml/parity_bets.json（npx tsx scripts/ml/parity_bets.ts で比べる）
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_bets2 import invert_strength, top3_from_strength  # noqa: E402
from train import OUT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ts = (ROOT / "app/lib/markBets.ts").read_text(encoding="utf-8")
    l2 = float(re.search(r"BENTER_L2 = ([\d.]+)", ts).group(1))
    l3 = float(re.search(r"BENTER_L3 = ([\d.]+)", ts).group(1))
    rng = np.random.default_rng(0)
    races = []
    for n in (6, 10, 14, 18):
        x = rng.gamma(1.0, 1.0, n)
        t = x / x.sum() * 3
        t = np.clip(t, 0.01, 0.95)
        t = t / t.sum() * 3
        s = invert_strength(t, l2, l3)
        _, p12, p123 = top3_from_strength(s, l2, l3)
        tri = sum(p123.transpose(q) for q in itertools.permutations(range(3)))
        races.append({"t3": t.tolist(), "s": s.tolist(),
                      "umaren": {f"{i}-{j}": float(p12[i, j] + p12[j, i]) for i, j in itertools.combinations(range(n), 2)},
                      "trio": {f"{i}-{j}-{k}": float(tri[i, j, k]) for i, j, k in itertools.combinations(range(n), 3)},
                      "trifecta": {f"{i}-{j}-{k}": float(p123[i, j, k]) for i, j, k in itertools.permutations(range(min(n, 8)), 3)}})
    (OUT / "parity_bets.json").write_text(json.dumps({"lambda": [l2, l3], "races": races}) + "\n", encoding="utf-8")
    print("parity_bets.json", len(races), file=sys.stderr)


if __name__ == "__main__":
    main()
