// AI印から作る買い目（2026-11-23）。単勝・複勝はなし。決め方は scripts/ml/backtest_mark_bets.py の「合わせた勝率｜確率上位 N点」
// - 勝率：AI と市場の勝率を条件付きロジットで合わせたもの（winCoef。オッズがないときは AI の勝率）
// - 組み合わせの確率：1着は勝率に比例、2着・3着は残りの馬の勝率を少し平らにした値（Benter の補正。下の BENTER_L2・L3）に比例
// - 印の付いた馬（◎○▲☆△。消を除く）の組み合わせから、確率の高い順に券種ごとの点数
// 過去の成績（2024〜2026年、確定の払戻）は data/ml/mark_bets_backtest.json

import type { AiMark } from "./marks";

export type MarkBetType = "ワイド" | "馬連" | "馬単" | "3連複" | "3連単";
export const MARK_BET_TYPES: MarkBetType[] = ["ワイド", "馬連", "馬単", "3連複", "3連単"];
export const MARK_BET_POINTS: Record<MarkBetType, number> = { ワイド: 3, 馬連: 3, 馬単: 3, "3連複": 5, "3連単": 6 };
const ORDERED: Record<MarkBetType, boolean> = { ワイド: false, 馬連: false, 馬単: true, "3連複": false, "3連単": true };
const USE: AiMark[] = ["◎", "○", "▲", "☆", "△"];

export type MarkBetHorse = { id: string; num: number; mark: AiMark | null; aiWin: number; odds: number | null };
export type MarkBetTicket = { nums: number[]; marks: (AiMark | null)[]; prob: number; key: string; idx: number[] };
export type MarkBetPlan = { type: MarkBetType; tickets: MarkBetTicket[]; hitProb: number };

/** 合わせた勝率（合計1）。オッズが全頭そろっていて係数があれば合わせ、なければ AI の勝率. */
export function combinedWin(hs: MarkBetHorse[], winCoef: { ai: number; market: number } | null): { p: number[]; usedOdds: boolean } {
  const aiSum = hs.reduce((s, h) => s + h.aiWin, 0);
  const useOdds = !!winCoef && hs.every((h) => h.odds != null && h.odds > 0);
  if (!useOdds) return { p: hs.map((h) => h.aiWin / aiSum), usedOdds: false };
  const invSum = hs.reduce((s, h) => s + 1 / h.odds!, 0);
  const z = hs.map((h) => winCoef!.ai * Math.log(Math.max(h.aiWin / aiSum, 1e-6)) + winCoef!.market * Math.log(Math.max(1 / h.odds! / invSum, 1e-6)));
  const m = Math.max(...z);
  const e = z.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return { p: e.map((x) => x / s), usedOdds: true };
}

// Benter の補正（2026-11-24）：2着は残りの馬の p^λ2、3着は p^λ3 に比例させる（ハービルは λ=1）。ハービルは人気馬の2・3着を多めに、
// 人気薄を少なめに見積もるため。λ は scripts/ml/benter_exotic.py が前の年の着順で決めた値（2024〜2026年で 0.80〜0.82・0.66〜0.70、
// 3年とも実際の2着・3着・並び・組の対数尤度がハービルより良い）。ここは直近（2026年に使った 2025年の当てはめ）の値
export const BENTER_L2 = 0.82;
export const BENTER_L3 = 0.70;
const cache = new WeakMap<number[], { w2: number[]; w3: number[]; s2: number; s3: number }>();
function weights(p: number[]) {
  let c = cache.get(p);
  if (!c) {
    const w2 = p.map((x) => Math.pow(x, BENTER_L2));
    const w3 = p.map((x) => Math.pow(x, BENTER_L3));
    c = { w2, w3, s2: w2.reduce((a, b) => a + b, 0), s3: w3.reduce((a, b) => a + b, 0) };
    cache.set(p, c);
  }
  return c;
}
/** i が1着・j が2着の確率. */
const exacta = (p: number[], i: number, j: number) => {
  const w = weights(p);
  return p[i] * w.w2[j] / (w.s2 - w.w2[i]);
};
/** i→j→k の並びの確率. */
const tri = (p: number[], i: number, j: number, k: number) => {
  const w = weights(p);
  return exacta(p, i, j) * w.w3[k] / (w.s3 - w.w3[i] - w.w3[j]);
};
const perms3 = (a: number, b: number, c: number) => [[a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];
const trioProb = (p: number[], a: number, b: number, c: number) => perms3(a, b, c).reduce((s, q) => s + tri(p, q[0], q[1], q[2]), 0);

function candidates(p: number[], t: MarkBetType, use: number[], n: number): { idx: number[]; prob: number }[] {
  const out: { idx: number[]; prob: number }[] = [];
  for (const i of use) for (const j of use) {
    if (j === i || (!ORDERED[t] && j < i)) continue;
    if (t === "馬単") out.push({ idx: [i, j], prob: exacta(p, i, j) });
    else if (t === "馬連") out.push({ idx: [i, j], prob: exacta(p, i, j) + exacta(p, j, i) });
    else if (t === "ワイド") {
      let s = 0;
      for (let k = 0; k < n; k++) if (k !== i && k !== j) s += trioProb(p, i, j, k);
      out.push({ idx: [i, j], prob: s });
    } else {
      for (const k of use) {
        if (k === i || k === j || (!ORDERED[t] && k < j)) continue;
        out.push({ idx: [i, j, k], prob: t === "3連単" ? tri(p, i, j, k) : trioProb(p, i, j, k) });
      }
    }
  }
  return out;
}

/** ワイドの「どれか当たる確率」：3着以内の3頭の組を全部数え、選んだ組のどれかを含む確率. */
function wideHitProb(p: number[], picks: number[][]): number {
  const n = p.length;
  let s = 0;
  for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) for (let c = b + 1; c < n; c++) {
    if (picks.some(([i, j]) => [a, b, c].includes(i) && [a, b, c].includes(j))) s += trioProb(p, a, b, c);
  }
  return s;
}

/** 券種ごとに、印の付いた馬（消を除く）の組み合わせを全部、確率の高い順に. */
export function markBetCandidates(hs: MarkBetHorse[], p: number[]): Record<MarkBetType, MarkBetTicket[]> | null {
  const use = hs.map((h, i) => (h.mark && USE.includes(h.mark) ? i : -1)).filter((i) => i >= 0);
  if (use.length < 3 || hs.length < 3) return null;
  return Object.fromEntries(MARK_BET_TYPES.map((t) => [t, candidates(p, t, use, hs.length).sort((a, b) => b.prob - a.prob).map((c) => {
    const nums = c.idx.map((i) => hs[i].num);
    const keyNums = ORDERED[t] ? nums : [...nums].sort((x, y) => x - y);
    const idx = ORDERED[t] ? c.idx : [...c.idx].sort((x, y) => hs[x].num - hs[y].num);
    return { nums: keyNums, marks: idx.map((i) => hs[i].mark), prob: c.prob, key: keyNums.join("-"), idx };
  })])) as Record<MarkBetType, MarkBetTicket[]>;
}

/** 選んだ買い目のどれかが当たる確率。馬連・馬単・3連複・3連単は当たる組が1つなので確率の合計、ワイドは3着以内の3頭の組を数える. */
export function hitProbOf(t: MarkBetType, p: number[], tickets: MarkBetTicket[]): number {
  return t === "ワイド" ? wideHitProb(p, tickets.map((c) => c.idx)) : tickets.reduce((s, c) => s + c.prob, 0);
}

export function markBetPlans(hs: MarkBetHorse[], p: number[]): MarkBetPlan[] {
  const all = markBetCandidates(hs, p);
  if (!all) return [];
  return MARK_BET_TYPES.map((t) => {
    const tickets = all[t].slice(0, MARK_BET_POINTS[t]);
    return { type: t, tickets, hitProb: hitProbOf(t, p, tickets) };
  });
}

/**
 * 期待値の順（2026-11-23、依頼者の案）：期待値＝当たる確率 × オッズ（ワイドは下限のオッズ）。オッズのある買い目だけ。
 * 確率がとても低い組（見積もりの誤差が大きく、オッズも大きいので期待値が上がりやすい）を避けるため、
 * 確率順で選んだ最後の買い目の確率の 1/3 以上の組から選ぶ。点数は確率順と同じ。過去の成績は確かめられない（全部の組の過去のオッズがない）
 */
export function evPlan(t: MarkBetType, p: number[], all: MarkBetTicket[], oddsOf: (key: string) => number | null):
  { tickets: (MarkBetTicket & { odds: number; ev: number })[]; hitProb: number } {
  const n = MARK_BET_POINTS[t];
  const floor = (all[Math.min(n, all.length) - 1]?.prob ?? 0) / 3;
  const tickets = all.filter((c) => c.prob >= floor)
    .map((c) => ({ ...c, odds: oddsOf(c.key) ?? NaN }))
    .filter((c) => Number.isFinite(c.odds))
    .map((c) => ({ ...c, ev: c.prob * c.odds }))
    .sort((a, b) => b.ev - a.ev)
    .slice(0, n);
  return { tickets, hitProb: hitProbOf(t, p, tickets) };
}
