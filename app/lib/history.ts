// netkeiba の全戦績から作る特徴量（上がり3F・位置取り・全キャリア）。
// 成績表にない情報なので、1段目の LightGBM ではなく2段目（scripts/ml/train_history_stack.py）で使う。
import { cleanText, extractTables } from "./netkeiba";

export const HISTORY_KEYS = ["日付", "開催", "頭数", "着順", "距離", "馬場", "着差", "通過", "ペース", "上り"] as const;
export type HistoryValues = Partial<Record<(typeof HISTORY_KEYS)[number], string>>;

/** netkeiba の戦績ページ（db.netkeiba.com/horse/result/{id}/）から、必要な列だけ取り出す. */
export function parseHistoryPage(html: string): HistoryValues[] | null {
  const table = extractTables(html).find((t) => t[0]?.includes("レース名") && t[0]?.includes("着順"));
  if (!table) return null;
  const header = table[0].map((h) => cleanText(h).replace(/\s+/g, ""));
  return table.slice(1).map((row) => {
    const values: HistoryValues = {};
    for (const key of HISTORY_KEYS) {
      const i = header.indexOf(key);
      if (i >= 0) values[key] = row[i] ?? "";
    }
    return values;
  });
}

/** route.ts の performances（列名がキーの値）を HistoryValues に揃える. */
export function toHistoryValues(rows: { values: Record<string, string> }[]): HistoryValues[] {
  return rows.map((row) => {
    const values: HistoryValues = {};
    for (const [k, v] of Object.entries(row.values)) {
      const key = k.replace(/\s+/g, "") as (typeof HISTORY_KEYS)[number];
      if ((HISTORY_KEYS as readonly string[]).includes(key)) values[key] = v;
    }
    return values;
  });
}

export const HISTORY_FEATURES = [
  "h_agari_mean", "h_agari_last1", "h_pos_first_mean", "h_gain_mean", "h_career_n", "h_career_top3",
] as const;
const RUNS = 5;
const AGARI_SCALE = 2;
const num = (s: string | undefined) => (s && /^-?\d+(\.\d+)?$/.test(s.trim()) ? Number(s.trim()) : null);

/**
 * raceDate より前の戦績から:
 *  h_agari_mean / h_agari_last1  上がり成分（レースの上がり3Fと同じ=0.5, 1秒速い=1, 1秒遅い=0）の近5走平均 / 前走
 *  h_pos_first_mean             最初のコーナーの位置 ÷ 頭数 の近5走平均（小さいほど先行）
 *  h_gain_mean                  (最後のコーナーの位置 − 着順) ÷ 頭数 の近5走平均（大きいほど追い上げ）
 *  h_career_n / h_career_top3   地方・海外・2021年以前も含む全出走数 / 3着内率
 * 値がないものは NaN.
 */
export function historyFeatures(history: HistoryValues[], raceDate: string): Record<string, number> {
  const runs = history
    .map((v) => ({ ...v, date: (v["日付"] ?? "").replace(/\//g, "-") }))
    .filter((v) => /^\d{4}-\d{2}-\d{2}$/.test(v.date) && v.date < raceDate)
    .filter((v) => !/取|除/.test(v["着順"] ?? "") && (v["着順"] ?? "") !== "")
    .sort((a, b) => (a.date < b.date ? 1 : -1)); // 新しい順
  const finish = (v: HistoryValues) => num(v["着順"]); // 中止などは null
  const recent = runs.slice(0, RUNS);
  const agari = recent.map((v) => {
    if (finish(v) === null) return 0;
    const a = num(v["上り"]);
    const pace = (v["ペース"] ?? "").match(/-(\d+\.\d)$/);
    return a !== null && pace ? Math.min(1, Math.max(0, 0.5 - (a - Number(pace[1])) / AGARI_SCALE)) : null;
  });
  const agariVals = agari.filter((x): x is number => x !== null);
  const posFirst: number[] = [];
  const gains: number[] = [];
  for (const v of recent) {
    const field = num(v["頭数"]);
    const passes = (v["通過"] ?? "").match(/\d+/g)?.map(Number) ?? [];
    if (!field || !passes.length) continue;
    posFirst.push(passes[0] / field);
    const f = finish(v);
    if (f !== null) gains.push((passes[passes.length - 1] - f) / field);
  }
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  const top3 = runs.filter((v) => { const f = finish(v); return f !== null && f <= 3; }).length;
  return {
    h_agari_mean: mean(agariVals),
    h_agari_last1: agari.length && agari[0] !== null ? agari[0] : NaN,
    h_pos_first_mean: mean(posFirst),
    h_gain_mean: mean(gains),
    h_career_n: runs.length,
    h_career_top3: runs.length ? top3 / runs.length : NaN,
  };
}
