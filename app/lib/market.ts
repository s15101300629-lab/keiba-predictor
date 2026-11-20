// 市場（オッズ）から見た勝率・連対率・3着以内の確率と、AI の確率とのズレ。
//   - 勝率：単勝オッズの逆数を合計1にそろえる（控除の分を除く）
//   - 連対率：馬連オッズの逆数を合計1にそろえ、その馬を含む組を足す（合計2）。馬連がなければ単勝からハービルの式
//   - 3着以内：3連複オッズで同じように（合計3）。なければワイド（その馬を含む組の和÷2、合計3）、それもなければ単勝からハービルの式
// AI の確率も、出走馬の合計を 1・2・3 にそろえてから比べる（市場と同じ土俵にする）。

export type OddsEntry = { min: number; max: number | null };
export type OddsTable = Record<string, OddsEntry>;
export type OddsSet = { win: OddsTable; place: OddsTable; umaren: OddsTable; wide: OddsTable; trio: OddsTable };
export type MarketSource = "馬連" | "3連複" | "ワイド" | "単勝から推定";
export type MarketProbs = {
  win: Map<number, number>; top2: Map<number, number>; top3: Map<number, number>;
  top2Source: MarketSource; top3Source: MarketSource;
  popularity: Map<number, number>; // 単勝オッズの低い順
};

const keyNums = (key: string) => key.split("-").map(Number);

/** オッズの逆数を合計 total（馬連・3連複は1、ワイドは3＝3着以内の組の数）にそろえ、各馬が含まれる組の和を返す. */
function fromCombos(table: OddsTable, nums: Set<number>, size: number, total: number): Map<number, number> | null {
  const combos = Object.entries(table).map(([k, v]) => ({ ns: keyNums(k), inv: 1 / v.min }))
    .filter((c) => c.ns.length === size && c.ns.every((n) => nums.has(n)) && Number.isFinite(c.inv));
  if (combos.length < 3) return null;
  const sum = combos.reduce((s, c) => s + c.inv, 0);
  const out = new Map<number, number>([...nums].map((n) => [n, 0]));
  for (const c of combos) for (const n of c.ns) out.set(n, out.get(n)! + (c.inv / sum) * total);
  return out;
}

/** 勝率（合計1）から2着以内・3着以内の確率（ハービルの式）. */
export function harville(win: Map<number, number>): { top2: Map<number, number>; top3: Map<number, number> } {
  const ns = [...win.keys()];
  const p = ns.map((n) => win.get(n)!);
  const top2 = new Map<number, number>();
  const top3 = new Map<number, number>();
  ns.forEach((n, i) => {
    let t2 = p[i];
    let t3 = p[i];
    ns.forEach((_, j) => {
      if (j === i || p[j] >= 1) return;
      const second = (p[j] * p[i]) / (1 - p[j]);
      t2 += second; t3 += second;
      ns.forEach((__, k) => {
        if (k === i || k === j || 1 - p[j] - p[k] <= 0) return;
        t3 += ((p[j] * p[k]) / (1 - p[j])) * (p[i] / (1 - p[j] - p[k]));
      });
    });
    top2.set(n, Math.min(t2, 1)); top3.set(n, Math.min(t3, 1));
  });
  return { top2, top3 };
}

export function marketProbs(odds: OddsSet, horseNumbers: number[]): MarketProbs | null {
  const nums = new Set(horseNumbers.filter((n) => odds.win[String(n)]?.min));
  if (nums.size < 3) return null;
  const winInv = new Map([...nums].map((n) => [n, 1 / odds.win[String(n)].min]));
  const winSum = [...winInv.values()].reduce((s, x) => s + x, 0);
  const win = new Map([...winInv].map(([n, x]) => [n, x / winSum]));
  const h = harville(win);
  const umaren = fromCombos(odds.umaren ?? {}, nums, 2, 1);
  const trio = fromCombos(odds.trio ?? {}, nums, 3, 1);
  const wide = trio ? null : fromCombos(odds.wide ?? {}, nums, 2, 3);
  const top3Wide = wide ? new Map([...wide].map(([n, x]) => [n, x / 2])) : null;
  const popularity = new Map([...nums].sort((a, b) => odds.win[String(a)].min - odds.win[String(b)].min).map((n, i) => [n, i + 1]));
  return {
    win, popularity,
    top2: umaren ?? h.top2, top2Source: umaren ? "馬連" : "単勝から推定",
    top3: trio ?? top3Wide ?? h.top3, top3Source: trio ? "3連複" : top3Wide ? "ワイド" : "単勝から推定",
  };
}

/** AI の確率を出走馬の合計 total にそろえる（1頭 1 を超えないように）. */
export function normalizeTo(values: Map<number, number>, total: number): Map<number, number> {
  const sum = [...values.values()].reduce((s, x) => s + x, 0);
  if (!sum) return values;
  return new Map([...values].map(([n, x]) => [n, Math.min((x / sum) * total, 1)]));
}
