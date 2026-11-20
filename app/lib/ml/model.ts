// scripts/ml/train.py が書き出した LightGBM の木（data/ml/model.json）で推論する。

type Split = { f: number; t: number; d: 0 | 1; m: 0 | 1 | 2; l: number; r: number };
type Leaf = { v: number };
export type Model = { features: string[]; objective: "binary"; trees: (Split | Leaf)[][] };

function goesLeft(node: Split, value: number): boolean {
  // LightGBM の missing_type: 0=None(NaN は 0 として比較) 1=Zero(0/NaN は既定方向) 2=NaN(NaN は既定方向)
  if (Number.isNaN(value)) {
    if (node.m === 0) value = 0;
    else return node.d === 1;
  } else if (node.m === 1 && value === 0) {
    return node.d === 1;
  }
  return value <= node.t;
}

export function predictRaw(model: Model, row: number[]): number {
  let sum = 0;
  for (const tree of model.trees) {
    let node = tree[0];
    while (!("v" in node)) node = tree[goesLeft(node, row[node.f]) ? node.l : node.r];
    sum += node.v;
  }
  return sum;
}

export function predictProbability(model: Model, features: Record<string, number>): number {
  const row = model.features.map((name) => features[name] ?? NaN);
  return 1 / (1 + Math.exp(-predictRaw(model, row)));
}

/**
 * レースの中で確率の合計を target にそろえる（各馬の logit に同じ値を足す。二分法）。
 * 3着以内は target = min(3, 頭数 − 0.5)。scripts/ml/tune_params.py の normalize_sum と同じ（2024〜2026年の検証で3年とも logloss 改善）.
 */
export function normalizeSum(probs: number[], target: number): number[] {
  if (!probs.length) return probs;
  const t = Math.min(target, probs.length - 0.5);
  const logit = probs.map((p) => Math.log(Math.min(Math.max(p, 1e-12), 1 - 1e-12) / (1 - Math.min(Math.max(p, 1e-12), 1 - 1e-12))));
  let lo = -10;
  let hi = 10;
  for (let k = 0; k < 50; k++) {
    const c = (lo + hi) / 2;
    let s = 0;
    for (const z of logit) s += 1 / (1 + Math.exp(-(z + c)));
    if (s > t) hi = c; else lo = c;
  }
  const c = (lo + hi) / 2;
  return logit.map((z) => 1 / (1 + Math.exp(-(z + c))));
}
