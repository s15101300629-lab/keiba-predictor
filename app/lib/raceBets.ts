// AI印と、印から作る買い目を1レース分まとめて決める（2026-11-24）。
// レース詳細の画面（app/race/[raceId]/page.tsx の MarksPanel）と、毎日の買い方の成績（scripts/ml/day_bets.ts）の両方がこれを使う。
// 同じ入力なら画面と同じ印・買い目になる。

import { assignAiMarks, type AiMark, type MarkResult } from "./marks";
import { evPlan, hitProbOf, MARK_BET_POINTS, MARK_BET_TYPES, markBetCandidates, poolWithComboOdds, strengthsFromTop3,
  type MarkBetTicket, type MarkBetType } from "./markBets";
import type { OddsTable, ValueResult } from "./market";
import { normalizeSum } from "./ml/model";

export type PlanHorse = {
  id: string;
  num: number;
  top3: number; // 3着以内確率（合計3にそろえた後。AIスコアの元）
  baseWin: number | null; // AI の勝率（T を掛けた後、合計をそろえる前）
};
export type PlanOdds = { win: OddsTable; place: OddsTable; umaren: OddsTable; wide: OddsTable; trio: OddsTable; trifecta: OddsTable };
export type HorseOddsInfo = { win: number | null; place: { min: number; max: number | null } | null; popularity: number | null };

/** 単勝オッズの低い順の人気（同じオッズは馬番の順ではなく、並べた順）. 画面の horseOdds と同じ. */
export function horseOddsOf(hs: { id: string; num: number | null }[], odds: PlanOdds | null): Map<string, HorseOddsInfo> {
  const out = new Map<string, HorseOddsInfo>();
  if (!odds) return out;
  const nums = hs.map((h) => h.num).filter((n): n is number => n != null && !!odds.win[String(n)]?.min);
  const pop = new Map([...nums].sort((a, b) => odds.win[String(a)].min - odds.win[String(b)].min).map((n, i) => [n, i + 1]));
  for (const h of hs) {
    if (h.num == null) continue;
    out.set(h.id, { win: odds.win[String(h.num)]?.min ?? null, place: odds.place[String(h.num)] ?? null, popularity: pop.get(h.num) ?? null });
  }
  return out;
}

/** AI印. values：目安（valueLabels）、longshot：馬番 → ☆の2段目の倍率. */
export function planMarks(hs: PlanHorse[], odds: PlanOdds | null, values: Map<number, ValueResult> | null, longshot: Map<number, number> | null,
  horseOdds: Map<string, HorseOddsInfo>, winCoef: { ai: number; market: number } | null): Map<string, MarkResult> {
  const rows = hs.filter((h) => h.baseWin != null);
  return assignAiMarks(rows.map((h) => {
    const v = values?.get(h.num) ?? null;
    const o = horseOdds.get(h.id);
    return { id: h.id, win: h.baseWin!, top3: h.top3, ratio: v?.ratio ?? null, odds: o?.win ?? null, popularity: o?.popularity ?? null,
      longshot: longshot?.get(h.num) ?? null };
  }), odds && values ? winCoef : null);
}

export type BetPlans = { all: Record<MarkBetType, MarkBetTicket[]> | null; p: number[]; usedCombined: boolean; pooled: MarkBetType[] };

/** 印から作る買い目の候補（確率の高い順）. 案1：合わせた3着以内確率から強さを逆算。案2（未検証）：馬連・3連複のオッズとも合わせる. */
export function planBets(hs: PlanHorse[], marks: Map<string, MarkResult>, horseOdds: Map<string, HorseOddsInfo>, odds: PlanOdds | null,
  values: Map<number, ValueResult> | null, valueCoef: { ai: number; market: number } | null): BetPlans {
  if (hs.length < 3) return { all: null, p: [], usedCombined: false, pooled: [] };
  const bh = hs.map((h) => ({ id: h.id, num: h.num, mark: marks.get(h.id)?.mark ?? null, aiWin: h.baseWin ?? h.top3, odds: horseOdds.get(h.id)?.win ?? null }));
  const aiT3 = hs.map((h) => h.top3);
  const comb = values && hs.every((h) => values.get(h.num)?.combined != null)
    ? normalizeSum(hs.map((h) => values.get(h.num)!.combined), 3) : null;
  const s = strengthsFromTop3(comb ?? aiT3);
  const all = markBetCandidates(bh, s);
  const pooled: MarkBetType[] = [];
  if (all && odds && valueCoef) {
    const sAi = comb ? strengthsFromTop3(aiT3) : s;
    for (const t of ["馬連", "3連複"] as const) {
      const table = odds[t === "馬連" ? "umaren" : "trio"] ?? {};
      if (!Object.keys(table).length) continue;
      const pm = poolWithComboOdds(t, sAi, bh.map((h) => h.num), table, valueCoef);
      if (!pm) continue;
      all[t] = all[t].map((c) => ({ ...c, prob: pm.get(c.key) ?? c.prob })).sort((a, b) => b.prob - a.prob);
      pooled.push(t);
    }
  }
  return { all, p: s, usedCombined: !!comb, pooled };
}

/** 券種のオッズ（ワイドは下限）. 馬単は取っていないので null. */
export function comboOdds(odds: PlanOdds | null, t: MarkBetType, key: string): { min: number; max: number | null } | undefined {
  const field = ({ ワイド: "wide", 馬連: "umaren", 馬単: null, "3連複": "trio", "3連単": "trifecta" } as const)[t];
  return field && odds ? odds[field][key] : undefined;
}

export type BetList = { type: MarkBetType; tickets: (MarkBetTicket & { ev?: number })[]; hitProb: number };

/** 券種ごとの買い目. prob：確率の高い順に決まった点数、ev：期待値（確率×オッズ）の高い順. */
export function betLists(plans: BetPlans, odds: PlanOdds | null, mode: "prob" | "ev"): BetList[] {
  const all = plans.all;
  if (!all) return [];
  return MARK_BET_TYPES.map((t) => {
    if (mode === "prob") {
      const picked = all[t].slice(0, MARK_BET_POINTS[t]);
      const tickets = picked.map((c) => {
        const o = comboOdds(odds, t, c.key);
        return { ...c, ev: o ? c.prob * o.min : undefined };
      });
      return { type: t, tickets, hitProb: plans.pooled.includes(t) ? tickets.reduce((a, c) => a + c.prob, 0) : hitProbOf(t, plans.p, picked) };
    }
    const r = evPlan(t, plans.p, all[t], (key) => comboOdds(odds, t, key)?.min ?? null);
    return { type: t, tickets: r.tickets, hitProb: r.hitProb };
  });
}

export type StarTicket = { label: string; kind: "複勝☆" | "ワイド◎-☆" | "ワイド○-☆"; type: "複勝" | "ワイド"; key: string; odds: string | null };

/** 案5：☆の馬の複勝と、ワイド ◎-☆・○-☆. */
export function starTickets(hs: PlanHorse[], marks: Map<string, MarkResult>, horseOdds: Map<string, HorseOddsInfo>, odds: PlanOdds | null): StarTicket[] {
  const numOf = (m: AiMark) => hs.find((h) => marks.get(h.id)?.mark === m)?.num ?? null;
  const fmt = (o: { min: number; max: number | null } | null | undefined) => (o ? o.min.toFixed(1) + (o.max ? "〜" + o.max.toFixed(1) : "") + "倍" : null);
  return hs.filter((h) => marks.get(h.id)?.mark === "☆").flatMap((h) => {
    const n = h.num;
    const out: StarTicket[] = [{ label: `複勝 ☆${n}`, kind: "複勝☆", type: "複勝", key: String(n), odds: fmt(horseOdds.get(h.id)?.place) }];
    for (const top of ["◎", "○"] as const) {
      const m = numOf(top);
      if (m == null) continue;
      const k = [m, n].sort((a, b) => a - b).join("-");
      out.push({ label: `ワイド ${top}${m}-☆${n}`, kind: `ワイド${top}-☆`, type: "ワイド", key: k, odds: fmt(comboOdds(odds, "ワイド", k)) });
    }
    return out;
  });
}
