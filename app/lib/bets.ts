// 買い目の自動作成（予算 1000円）。AIの勝率から着順を何万回も試し（ハービルの式：1着の馬を除いた残りで次の着順が決まる）、
// 買い目の組み合わせごとに「どれか当たる確率」「払戻が予算以上になる確率」「期待回収率」を数える。
//   - 的中率重視：オッズがあれば「払戻が予算以上になる確率（当たって損しない確率）」、なければ「どれか当たる確率」が一番高くなるように選ぶ
//   - 回収率重視：オッズが必要。期待値（的中確率 × 配当）の高い買い目を選ぶ
// 予算は 100円単位。オッズがあるときは、どれが当たっても払戻がそろうように配当の逆数に比例させる。

export const BET_BUDGET = 1000;
const MAX_TICKETS = 5;
const SAMPLES = 20000;
const TOP_HORSES = 6; // 候補にする馬（勝率・3着以内確率の上位）

export type BetType = "単勝" | "複勝" | "ワイド" | "馬連" | "3連複";
export type BetHorse = { id: string; num: number | null; name: string; win: number; place: number };
export type Candidate = { type: BetType; horses: BetHorse[]; key: string | null; prob: number; odds: number | null; hits: Uint8Array };
export type Plan = {
  tickets: { c: Candidate; stake: number }[];
  hitProb: number; // どれか当たる確率
  profitProb: number | null; // 払戻が予算以上になる確率（オッズがあるとき）
  expectedReturn: number | null; // 期待回収率（払戻の期待値 ÷ 予算）
  payoutRange: [number, number] | null; // 当たったときの払戻の幅（1点だけ当たったとき）
  usedOdds: boolean;
};

function rng(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 上位3着の並びを SAMPLES 回つくる（各回 [1着, 2着, 3着] の馬の番号＝hs の添字）. */
function sampleOrders(hs: BetHorse[]): Int16Array {
  const total = hs.reduce((s, h) => s + h.win, 0);
  const w = hs.map((h) => h.win / total);
  const out = new Int16Array(SAMPLES * 3);
  const rand = rng(12345);
  for (let s = 0; s < SAMPLES; s++) {
    let left = 1;
    const used: number[] = [];
    for (let k = 0; k < 3; k++) {
      let r = rand() * left;
      let pick = -1;
      for (let i = 0; i < w.length; i++) {
        if (used.includes(i)) continue;
        r -= w[i];
        pick = i;
        if (r <= 0) break;
      }
      used.push(pick);
      left -= w[pick];
      out[s * 3 + k] = pick;
    }
  }
  return out;
}

const pairs = <T,>(xs: T[]) => xs.flatMap((a, i) => xs.slice(i + 1).map((b) => [a, b]));
const triples = <T,>(xs: T[]) => xs.flatMap((a, i) => xs.slice(i + 1).flatMap((b, j) => xs.slice(i + j + 2).map((c) => [a, b, c])));

export function buildCandidates(hs: BetHorse[], oddsOf: (type: BetType, nums: (number | null)[]) => number | null): Candidate[] {
  if (hs.length < 3) return [];
  const orders = sampleOrders(hs);
  const idx = new Map(hs.map((h, i) => [h.id, i]));
  const placeK = hs.length <= 7 ? 2 : 3; // 複勝は7頭以下だと2着まで
  const topWin = [...hs].sort((a, b) => b.win - a.win).slice(0, TOP_HORSES);
  const topPlace = [...hs].sort((a, b) => b.place - a.place).slice(0, TOP_HORSES);
  const defs: { type: BetType; horses: BetHorse[]; hit: (o: number[]) => boolean }[] = [
    ...topWin.map((h) => ({ type: "単勝" as const, horses: [h], hit: (o: number[]) => o[0] === idx.get(h.id) })),
    ...topPlace.map((h) => ({ type: "複勝" as const, horses: [h], hit: (o: number[]) => o.slice(0, placeK).includes(idx.get(h.id)!) })),
    ...pairs(topPlace).map((p) => ({ type: "ワイド" as const, horses: p, hit: (o: number[]) => p.every((h) => o.includes(idx.get(h.id)!)) })),
    ...pairs(topWin).map((p) => ({ type: "馬連" as const, horses: p, hit: (o: number[]) => p.every((h) => o.slice(0, 2).includes(idx.get(h.id)!)) })),
    ...triples(topPlace).map((t) => ({ type: "3連複" as const, horses: t, hit: (o: number[]) => t.every((h) => o.includes(idx.get(h.id)!)) })),
  ];
  return defs.map((d) => {
    const hits = new Uint8Array(SAMPLES);
    let n = 0;
    for (let s = 0; s < SAMPLES; s++) {
      if (d.hit([orders[s * 3], orders[s * 3 + 1], orders[s * 3 + 2]])) { hits[s] = 1; n++; }
    }
    const nums = d.horses.map((h) => h.num);
    const sorted = nums.some((x) => x == null) ? null : [...nums].sort((a, b) => a! - b!);
    return { type: d.type, horses: d.horses, key: sorted ? sorted.join("-") : null, prob: n / SAMPLES, odds: sorted ? oddsOf(d.type, sorted) : null, hits };
  });
}

/** 予算を100円単位で割り振る。重みに比例させ、端数は小数部分の大きい順に100円ずつ足す. */
export function allocate(budget: number, weights: number[]): number[] {
  const units = Math.floor(budget / 100);
  const total = weights.reduce((s, w) => s + w, 0);
  if (!units || !total) return weights.map(() => 0);
  const raw = weights.map((w) => (units * w) / total);
  const base = raw.map(Math.floor);
  let left = units - base.reduce((s, x) => s + x, 0);
  for (const { i } of raw.map((x, i) => ({ f: x - Math.floor(x), i })).sort((p, q) => q.f - p.f)) {
    if (left <= 0) break;
    base[i] += 1; left -= 1;
  }
  return base.map((u) => u * 100);
}

function evaluate(cs: Candidate[], useOdds: boolean): Plan {
  const stakes = allocate(BET_BUDGET, cs.map((c) => (useOdds ? 1 / c.odds! : c.prob)));
  const tickets = cs.map((c, i) => ({ c, stake: stakes[i] })).filter((t) => t.stake > 0);
  let hit = 0;
  let profit = 0;
  let payoutSum = 0;
  for (let s = 0; s < SAMPLES; s++) {
    let pay = 0;
    let any = false;
    for (const t of tickets) {
      if (t.c.hits[s]) { any = true; if (useOdds) pay += t.stake * t.c.odds!; }
    }
    if (any) hit++;
    if (useOdds) { payoutSum += pay; if (pay >= BET_BUDGET) profit++; }
  }
  const pays = useOdds ? tickets.map((t) => t.stake * t.c.odds!) : [];
  return {
    tickets, hitProb: hit / SAMPLES, usedOdds: useOdds,
    profitProb: useOdds ? profit / SAMPLES : null,
    expectedReturn: useOdds ? payoutSum / SAMPLES / BET_BUDGET : null,
    payoutRange: pays.length ? [Math.min(...pays), Math.max(...pays)] : null,
  };
}

/** 的中率重視：複勝・ワイドのオッズがあれば「払戻が予算以上になる確率」、なければ「どれか当たる確率」を、1点ずつ足しながら最大にする（最大5点）. */
export function hitFocusedPlan(all: Candidate[]): Plan | null {
  // 複勝とワイドのオッズが出ていれば、オッズのある買い目だけで「払戻が予算以上になる確率」を最大にする
  const withOdds = all.filter((c) => c.odds);
  const useOdds = withOdds.some((c) => c.type === "複勝") && withOdds.some((c) => c.type === "ワイド");
  const cands = useOdds ? withOdds : all;
  const score = (p: Plan) => (useOdds ? p.profitProb! : p.hitProb);
  let chosen: Candidate[] = [];
  let best: Plan | null = null;
  while (chosen.length < MAX_TICKETS) {
    let next: { c: Candidate; plan: Plan } | null = null;
    for (const c of cands) {
      if (chosen.includes(c)) continue;
      const plan = evaluate([...chosen, c], useOdds);
      if (!next || score(plan) > score(next.plan)) next = { c, plan };
    }
    if (!next || (best && score(next.plan) <= score(best) + 1e-9)) break;
    chosen = [...chosen, next.c];
    best = next.plan;
  }
  return best;
}

/** 回収率重視：オッズが必要（オッズのある買い目だけを使う）。期待値（的中確率 × 配当）の高い順に、期待回収率が一番高くなる点数（最大5点）。払戻がそろうように割り振る. */
export function returnFocusedPlan(all: Candidate[]): Plan | null {
  // オッズが出ている券種の買い目だけで組む（発売前は単勝の予想オッズしかないことがある）
  const cands = all.filter((c) => c.odds);
  if (!cands.length) return null;
  const ranked = [...cands].sort((a, b) => b.prob * b.odds! - a.prob * a.odds!);
  // 期待回収率が一番高くなる点数（1〜5点）を選ぶ
  let best: Plan | null = null;
  for (let n = 1; n <= MAX_TICKETS; n++) {
    const plan = evaluate(ranked.slice(0, n), true);
    if (!best || plan.expectedReturn! > best.expectedReturn!) best = plan;
  }
  return best;
}
