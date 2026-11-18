// 父・母父・母母父の産駒成績（netkeiba 種牡馬ページ）から、道悪・距離の適性を出す。

export type Counts = [number, number, number, number]; // 1着, 2着, 3着, 着外
export type StallionRole = "sire" | "damsire";
export type StallionRecord = {
  stallionId: string;
  name: string | null;
  role: StallionRole;
  going: Record<string, Partial<Record<"良" | "稍重" | "重" | "不良", Counts>>>; // surface -> going -> counts
  distance: Record<string, { label: string; upper: number | null; counts: Counts }[]>; // surface -> bands
};

// 血統表の位置ごとの重み（血の割合）。父系は父の産駒成績に含まれるので、母系だけ1段深く見る
export const PEDIGREE_SOURCES = [
  { position: "F", role: "sire" as const, weight: 0.5, label: "父" },
  { position: "MF", role: "damsire" as const, weight: 0.25, label: "母父" },
  { position: "MMF", role: "damsire" as const, weight: 0.125, label: "母母父" },
];
const SHRINK_STARTS = 30;
// netkeiba の距離区分 → 距離タイプ
const BAND_TYPE: Record<string, string> = { "-1400": "短距離", "-1800": "マイル", "-2200": "中距離", "-2600": "長距離", "2600-": "長距離" };

const starts = (c: Counts) => c[0] + c[1] + c[2] + c[3];
const top3 = (c: Counts) => c[0] + c[1] + c[2];
const add = (a: Counts, b: Counts | undefined): Counts => (b ? [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]] : a);

/** 種牡馬ページの表（extractTables 済み）から、役割(種牡馬 / BMS)ごとの累計を読む. */
export function parseStallionSection(
  record: StallionRecord, tables: string[][][], kind: "going" | "distance"
): void {
  for (const table of tables) {
    const header = table[0] ?? [];
    const total = table.find((row) => row[0] === "累計");
    if (!total || header[0] !== "年度") continue;
    header.slice(1).forEach((label, index) => {
      const counts = total.slice(1 + index * 4, 5 + index * 4).map((v) => Number(v.replace(/,/g, "")) || 0) as Counts;
      if (counts.length < 4) return;
      if (kind === "going") {
        const m = label.match(/^(芝|ダート)・(良|稍重|重|不良)$/);
        if (m) (record.going[m[1]] ??= {})[m[2] as "良"] = counts;
      } else {
        const m = label.match(/^(-?\d+-?)\((芝|ダート)\)$/);
        if (m) {
          const upper = m[1].startsWith("-") ? Number(m[1].slice(1)) : null;
          (record.distance[m[2]] ??= []).push({ label: m[1], upper, counts });
        }
      }
    });
  }
}

export type StallionAptitude = {
  heavyRatio: number | null; // 重・不良の3着内率 ÷ 良の3着内率
  heavyStarts: number;
  bandRates: Record<string, number>; // 距離区分ごとの3着内率（その馬全体の率に寄せた補正値）
  band: string | null; bandStarts: number; bandTop3: number; // 今回の距離帯
};

export function stallionAptitude(record: StallionRecord, surface: string, distance: number | null): StallionAptitude {
  const g = record.going[surface] ?? {};
  const zero: Counts = [0, 0, 0, 0];
  const good = g["良"] ?? zero;
  const heavy = add(g["重"] ?? zero, g["不良"]);
  const goodRate = starts(good) ? top3(good) / starts(good) : null;
  const heavyRate = goodRate !== null && goodRate > 0
    ? (top3(heavy) + SHRINK_STARTS * goodRate) / (starts(heavy) + SHRINK_STARTS)
    : null;
  const bands = record.distance[surface] ?? [];
  const totalStarts = bands.reduce((sum, b) => sum + starts(b.counts), 0);
  const overall = totalStarts ? bands.reduce((sum, b) => sum + top3(b.counts), 0) / totalStarts : 0;
  const bandRates: Record<string, number> = {};
  for (const b of bands) {
    if (BAND_TYPE[b.label] && starts(b.counts)) {
      bandRates[b.label] = (top3(b.counts) + SHRINK_STARTS * overall) / (starts(b.counts) + SHRINK_STARTS);
    }
  }
  let current = distance === null ? undefined : bands.find((b) => b.upper !== null && distance <= b.upper);
  if (!current && distance !== null) current = bands.find((b) => b.upper === null);
  return {
    heavyRatio: heavyRate !== null && goodRate ? heavyRate / goodRate : null,
    heavyStarts: starts(heavy),
    bandRates,
    band: current?.label ?? null,
    bandStarts: current ? starts(current.counts) : 0,
    bandTop3: current ? top3(current.counts) : 0,
  };
}

export const heavyLabel = (r: number | null) => (r === null ? null : r >= 1.1 ? "得意" : r <= 0.9 ? "苦手" : "普通");

export type HorsePedigreeAptitude = {
  sources: { label: string; name: string | null; weight: number; aptitude: StallionAptitude }[];
  heavyRatio: number | null; heavyLabel: string | null;
  bandRates: Record<string, number>; distanceType: string | null; // 補正後の3着内率が最も高い距離区分
  conditionRate: number | null; // 今回条件の3着内率（縮小推定・道悪補正込み）
  deviation: number | null; // 出走馬内の偏差値
};

/** 出走馬全体の血統適性。records[horseIndex] は PEDIGREE_SOURCES 順の StallionRecord|null. */
export function pedigreeAptitudes(
  records: (StallionRecord | null)[][], surface: string, distance: number | null, going: string | null
): HorsePedigreeAptitude[] {
  const all = records.map((list) => list.map((r) => (r ? stallionAptitude(r, surface, distance) : null)));
  // 今回の距離帯の平均3着内率（縮小推定の事前値）
  let poolStarts = 0;
  let poolTop3 = 0;
  for (const list of all) for (const a of list) if (a) { poolStarts += a.bandStarts; poolTop3 += a.bandTop3; }
  const prior = poolStarts ? poolTop3 / poolStarts : 0;
  const isHeavy = going === "重" || going === "不良";

  const result = all.map((list, i) => {
    const sources = PEDIGREE_SOURCES.map((s, j) => ({ ...s, record: records[i][j], aptitude: list[j] }))
      .filter((s): s is typeof s & { aptitude: StallionAptitude } => s.aptitude !== null);
    const avg = (get: (a: StallionAptitude) => number | null) => {
      let sum = 0;
      let w = 0;
      for (const s of sources) {
        const v = get(s.aptitude);
        if (v !== null) { sum += s.weight * v; w += s.weight; }
      }
      return w ? sum / w : null;
    };
    const heavyRatio = avg((a) => a.heavyRatio);
    const bandRates: Record<string, number> = {};
    for (const label of Object.keys(BAND_TYPE)) {
      const v = avg((a) => a.bandRates[label] ?? null);
      if (v !== null) bandRates[label] = v;
    }
    const best = Object.entries(bandRates).sort((x, y) => y[1] - x[1])[0];
    const bandRate = avg((a) => (a.band ? (a.bandTop3 + SHRINK_STARTS * prior) / (a.bandStarts + SHRINK_STARTS) : null));
    const conditionRate = bandRate === null ? null : bandRate * (isHeavy && heavyRatio !== null ? heavyRatio : 1);
    return {
      sources: sources.map((s) => ({ label: s.label, name: s.record?.name ?? null, weight: s.weight, aptitude: s.aptitude })),
      heavyRatio, heavyLabel: heavyLabel(heavyRatio),
      bandRates, distanceType: best ? BAND_TYPE[best[0]] : null,
      conditionRate, deviation: null as number | null,
    };
  });
  const vals = result.map((r) => r.conditionRate).filter((v): v is number => v !== null);
  if (vals.length >= 2) {
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
    for (const r of result) if (r.conditionRate !== null) r.deviation = sd === 0 ? 50 : 50 + 10 * (r.conditionRate - mean) / sd;
  }
  return result;
}
