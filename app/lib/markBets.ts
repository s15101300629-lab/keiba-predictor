// AI印から作る買い目（2026-11-23）。単勝・複勝はなし。決め方は scripts/ml/backtest_mark_bets.py の「合わせた勝率｜確率上位 N点」
// - 勝率：AI と市場の勝率を条件付きロジットで合わせたもの（winCoef。オッズがないときは AI の勝率）
// - 組み合わせの確率：ハービルの式（勝率に比例して1着が決まり、残りの馬で同じように2着・3着が決まる）
// - 印の付いた馬（◎○▲☆△。消を除く）の組み合わせから、確率の高い順に券種ごとの点数
// 過去の成績（2024〜2026年、確定の払戻）は data/ml/mark_bets_backtest.json

import type { AiMark } from "./marks";

export type MarkBetType = "ワイド" | "馬連" | "馬単" | "3連複" | "3連単";
export const MARK_BET_TYPES: MarkBetType[] = ["ワイド", "馬連", "馬単", "3連複", "3連単"];
export const MARK_BET_POINTS: Record<MarkBetType, number> = { ワイド: 3, 馬連: 3, 馬単: 3, "3連複": 5, "3連単": 6 };
const ORDERED: Record<MarkBetType, boolean> = { ワイド: false, 馬連: false, 馬単: true, "3連複": false, "3連単": true };
const USE: AiMark[] = ["◎", "○", "▲", "☆", "△"];

export type MarkBetHorse = { id: string; num: number; mark: AiMark | null; aiWin: number; odds: number | null };
export type MarkBetTicket = { nums: number[]; marks: (AiMark | null)[]; prob: number; key: string };
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

const tri = (p: number[], i: number, j: number, k: number) => p[i] * (p[j] / (1 - p[i])) * (p[k] / (1 - p[i] - p[j]));
const perms3 = (a: number, b: number, c: number) => [[a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]];
const trioProb = (p: number[], a: number, b: number, c: number) => perms3(a, b, c).reduce((s, q) => s + tri(p, q[0], q[1], q[2]), 0);

function candidates(p: number[], t: MarkBetType, use: number[], n: number): { idx: number[]; prob: number }[] {
  const out: { idx: number[]; prob: number }[] = [];
  for (const i of use) for (const j of use) {
    if (j === i || (!ORDERED[t] && j < i)) continue;
    if (t === "馬単") out.push({ idx: [i, j], prob: p[i] * p[j] / (1 - p[i]) });
    else if (t === "馬連") out.push({ idx: [i, j], prob: p[i] * p[j] / (1 - p[i]) + p[j] * p[i] / (1 - p[j]) });
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

export function markBetPlans(hs: MarkBetHorse[], p: number[]): MarkBetPlan[] {
  const use = hs.map((h, i) => (h.mark && USE.includes(h.mark) ? i : -1)).filter((i) => i >= 0);
  if (use.length < 3 || hs.length < 3) return [];
  return MARK_BET_TYPES.map((t) => {
    const picks = candidates(p, t, use, hs.length).sort((a, b) => b.prob - a.prob).slice(0, MARK_BET_POINTS[t]);
    const tickets = picks.map((c) => {
      const nums = c.idx.map((i) => hs[i].num);
      const keyNums = ORDERED[t] ? nums : [...nums].sort((x, y) => x - y);
      return { nums: ORDERED[t] ? nums : keyNums, marks: (ORDERED[t] ? c.idx : [...c.idx].sort((x, y) => hs[x].num - hs[y].num)).map((i) => hs[i].mark),
        prob: c.prob, key: keyNums.join("-") };
    });
    // 馬連・馬単・3連複・3連単は1レースで当たる組が1つなので、確率の合計がどれか当たる確率
    const hitProb = t === "ワイド" ? wideHitProb(p, picks.map((c) => c.idx)) : picks.reduce((s, c) => s + c.prob, 0);
    return { type: t, tickets, hitProb };
  });
}
