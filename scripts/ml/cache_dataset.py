"""特徴量の表を1回だけ作って npz に保存する（調整・比較で何度も使うため）.

    cd scripts/ml && python cache_dataset.py [出力先]   # 既定は data/ml/cache/dataset.npz（git には入れない）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, TRAIN_FROM, matrix  # noqa: E402

DEFAULT = OUT / "cache" / "dataset.npz"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    out.parent.mkdir(parents=True, exist_ok=True)
    fin = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    np.savez_compressed(
        out, X=matrix(rows), features=np.array(FEATURES),
        label=np.array([r["label"] for r in rows]), finish=fin,
        date=np.array([r["date"] for r in rows]), race_id=np.array([r["race_id"] for r in rows]),
        popularity=np.array([r["popularity"] or 99 for r in rows]),
    )
    print(f"wrote {out} rows {len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
