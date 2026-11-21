// TreeSHAP（Lundberg ほか。LightGBM の pred_contrib と同じ計算）。各項目が、その馬の予測（log-odds）をどれだけ押し上げ・押し下げたか。
// 使う木は model.json（scripts/ml/train.py の export_model）。各ノードの c（学習データの頭数）が要る。
// 戻り値：model.features と同じ並びの寄与と、最後に全体の平均（bias）。寄与の合計＋bias＝予測の log-odds。

import type { Model } from "./model";

type Node = { f?: number; t?: number; d?: 0 | 1; m?: 0 | 1 | 2; l?: number; r?: number; v?: number; c?: number };
type PathEl = { feature: number; zero: number; one: number; weight: number };

function goesLeft(node: Node, value: number): boolean {
  // model.ts の goesLeft と同じ（LightGBM の missing_type）
  if (Number.isNaN(value)) {
    if (node.m === 0) value = 0;
    else return node.d === 1;
  } else if (node.m === 1 && value === 0) {
    return node.d === 1;
  }
  return value <= node.t!;
}

function extendPath(path: PathEl[], depth: number, zero: number, one: number, feature: number) {
  path[depth] = { feature, zero, one, weight: depth === 0 ? 1 : 0 };
  for (let i = depth - 1; i >= 0; i--) {
    path[i + 1].weight += (one * path[i].weight * (i + 1)) / (depth + 1);
    path[i].weight = (zero * path[i].weight * (depth - i)) / (depth + 1);
  }
}

function unwindPath(path: PathEl[], depth: number, index: number) {
  const one = path[index].one;
  const zero = path[index].zero;
  let next = path[depth].weight;
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = path[i].weight;
      path[i].weight = (next * (depth + 1)) / ((i + 1) * one);
      next = tmp - (path[i].weight * zero * (depth - i)) / (depth + 1);
    } else {
      path[i].weight = (path[i].weight * (depth + 1)) / (zero * (depth - i));
    }
  }
  for (let i = index; i < depth; i++) {
    path[i].feature = path[i + 1].feature;
    path[i].zero = path[i + 1].zero;
    path[i].one = path[i + 1].one;
  }
}

function unwoundPathSum(path: PathEl[], depth: number, index: number): number {
  const one = path[index].one;
  const zero = path[index].zero;
  let next = path[depth].weight;
  let total = 0;
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = (next * (depth + 1)) / ((i + 1) * one);
      total += tmp;
      next = path[i].weight - tmp * zero * ((depth - i) / (depth + 1));
    } else {
      total += path[i].weight / zero / ((depth - i) / (depth + 1));
    }
  }
  return total;
}

function recurse(tree: Node[], x: number[], phi: number[], nodeIdx: number, parentPath: PathEl[], depth: number,
  zero: number, one: number, feature: number) {
  const path = parentPath.slice(0, depth).map((p) => ({ ...p }));
  extendPath(path, depth, zero, one, feature);
  const node = tree[nodeIdx];
  if (node.v !== undefined) {
    for (let i = 1; i <= depth; i++) {
      const w = unwoundPathSum(path, depth, i);
      const el = path[i];
      phi[el.feature] += w * (el.one - el.zero) * node.v;
    }
    return;
  }
  const hot = goesLeft(node, x[node.f!]) ? node.l! : node.r!;
  const cold = hot === node.l ? node.r! : node.l!;
  const cover = node.c || 1;
  const hotZero = (tree[hot].c ?? 0) / cover;
  const coldZero = (tree[cold].c ?? 0) / cover;
  let inZero = 1;
  let inOne = 1;
  let d = depth;
  let k = 0;
  while (k <= d && path[k].feature !== node.f) k++;
  if (k !== d + 1) {
    inZero = path[k].zero;
    inOne = path[k].one;
    unwindPath(path, d, k);
    d -= 1;
  }
  recurse(tree, x, phi, hot, path, d + 1, hotZero * inZero, inOne, node.f!);
  recurse(tree, x, phi, cold, path, d + 1, coldZero * inZero, 0, node.f!);
}

/** 木の葉の平均（学習データの頭数で重み付け）＝その木の全体の平均. */
function expectedValue(tree: Node[]): number {
  let sum = 0;
  let n = 0;
  for (const node of tree) if (node.v !== undefined) { sum += node.v * (node.c ?? 0); n += node.c ?? 0; }
  return n ? sum / n : 0;
}

const biasCache = new WeakMap<Model, number>();

export function treeShap(model: Model, features: Record<string, number>): number[] {
  const x = model.features.map((name) => features[name] ?? NaN);
  const phi = new Array(model.features.length + 1).fill(0);
  for (const t of model.trees as Node[][]) {
    if (t[0].v !== undefined) continue; // 葉だけの木は寄与なし（平均に入る）
    recurse(t, x, phi, 0, [], 0, 1, 1, -1);
  }
  let bias = biasCache.get(model);
  if (bias === undefined) {
    bias = 0;
    for (const t of model.trees as Node[][]) bias += expectedValue(t);
    biasCache.set(model, bias);
  }
  phi[model.features.length] = bias;
  return phi;
}
