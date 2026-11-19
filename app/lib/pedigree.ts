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

export type PedigreeSource = { position: string; role: StallionRole; weight: number; label: string };

/** 血統表の位置（F=父, M=母 の並び）から、使う成績と重み（血の割合）を決める.
 *  母方を真っすぐたどる位置（母父・母母父…）は BMS 成績、それ以外は産駒成績を使う. */
export function pedigreeSource(position: string): PedigreeSource {
  return {
    position,
    role: /^M+F$/.test(position) ? "damsire" : "sire",
    weight: 0.5 ** position.length,
    label: [...position].map((c) => (c === "F" ? "父" : "母")).join(""),
  };
}

/** maxGeneration 代までの牡馬の祖先すべて. */
export function allMaleSources(maxGeneration: number): PedigreeSource[] {
  const out: PedigreeSource[] = [];
  const walk = (prefix: string) => {
    if (prefix.length >= maxGeneration) return;
    for (const c of ["F", "M"]) {
      const pos = prefix + c;
      if (c === "F") out.push(pedigreeSource(pos));
      walk(pos);
    }
  };
  walk("");
  return out.sort((a, b) => a.position.length - b.position.length || a.position.localeCompare(b.position));
}

// 現在の採用: 父・母父・母母父（父系は父の産駒成績に含まれるので、母系だけ1段深く見る）
export const PEDIGREE_SOURCES: PedigreeSource[] = ["F", "MF", "MMF"].map(pedigreeSource);
const SHRINK_STARTS = 30;
// netkeiba の距離区分 → 距離タイプ
const BAND_TYPE: Record<string, string> = { "-1400": "短距離", "-1800": "マイル", "-2200": "中距離", "-2600": "長距離", "2600-": "長距離" };

const starts = (c: Counts) => c[0] + c[1] + c[2] + c[3];
const top3 = (c: Counts) => c[0] + c[1] + c[2];
const add = (a: Counts, b: Counts | undefined): Counts => (b ? [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]] : a);

import { extractTables } from "./netkeiba";

type StallionKind = "going" | "distance";
type RawTable = { labels: string[]; rows: Record<string, number[]> }; // rows: 年度 or "累計" -> 4値ずつ並んだ数
// 種牡馬ページ（馬場別・距離別）の年度別データ。data/pedigree/stallions/{id}.json の形
export type StallionRaw = {
  stallionId: string; name: string | null; fetchedAt: string;
  sections: Partial<Record<StallionRole, Partial<Record<StallionKind, RawTable>>>>;
};

function parseSection(html: string): RawTable {
  const labels: string[] = [];
  const rows: Record<string, number[]> = {};
  for (const table of extractTables(html)) {
    const header = table[0] ?? [];
    if (header[0] !== "年度") continue;
    const width = (header.length - 1) * 4;
    const offset = labels.length * 4;
    labels.push(...header.slice(1));
    for (const row of table) {
      const key = row[0];
      if (key !== "累計" && !/^\d{4}$/.test(key)) continue;
      const values = row.slice(1, 1 + width).map((v) => Number(v.replace(/,/g, "")) || 0);
      const target = (rows[key] ??= []);
      while (target.length < offset) target.push(0);
      target.push(...values, ...Array(Math.max(0, width - values.length)).fill(0));
    }
  }
  for (const key of Object.keys(rows)) while (rows[key].length < labels.length * 4) rows[key].push(0);
  return { labels, rows };
}

/** 種牡馬ページ（type=1 馬場別, type=2 距離別）の HTML から、種牡馬・BMS 両方の年度別データを読む. */
export function parseStallionPages(
  stallionId: string, name: string | null, pages: Partial<Record<StallionKind, string | null>>, fetchedAt: string
): StallionRaw {
  const raw: StallionRaw = { stallionId, name, fetchedAt, sections: {} };
  for (const kind of ["going", "distance"] as const) {
    const html = pages[kind];
    if (!html) continue;
    const sireStart = html.indexOf("block=horse_sire_results");
    const bmsStart = html.indexOf("block=horse_bms_results");
    const parts: Record<StallionRole, string> = {
      sire: html.slice(Math.max(0, sireStart), bmsStart > sireStart ? bmsStart : undefined),
      damsire: bmsStart >= 0 ? html.slice(bmsStart) : "",
    };
    for (const role of ["sire", "damsire"] as const) {
      const table = parseSection(parts[role]);
      if (table.labels.length) (raw.sections[role] ??= {})[kind] = table;
    }
  }
  return raw;
}

/** throughYear 年までの成績（累計 − それより後の年度）。null なら累計そのまま. */
export function stallionRecord(raw: StallionRaw, role: StallionRole, throughYear: number | null): StallionRecord | null {
  const section = raw.sections[role];
  if (!section) return null;
  const record: StallionRecord = { stallionId: raw.stallionId, name: raw.name, role, going: {}, distance: {} };
  for (const kind of ["going", "distance"] as const) {
    const table = section[kind];
    const total = table?.rows["累計"];
    if (!table || !total) continue;
    const values = [...total];
    if (throughYear !== null) {
      for (const [year, row] of Object.entries(table.rows)) {
        if (/^\d{4}$/.test(year) && Number(year) > throughYear) row.forEach((v, i) => { values[i] -= v; });
      }
    }
    table.labels.forEach((label, index) => {
      const counts = values.slice(index * 4, index * 4 + 4).map((v) => Math.max(0, v)) as Counts;
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
  return Object.keys(record.going).length || Object.keys(record.distance).length ? record : null;
}

export type StallionAptitude = {
  heavyRatio: number | null; // 重・不良の3着内率 ÷ 良の3着内率
  heavyStarts: number;
  wetRatio: number | null; // 稍重・重・不良の3着内率 ÷ 良の3着内率（比較用）
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
  const wet = add(heavy, g["稍重"]);
  const wetRate = goodRate !== null && goodRate > 0
    ? (top3(wet) + SHRINK_STARTS * goodRate) / (starts(wet) + SHRINK_STARTS)
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
    wetRatio: wetRate !== null && goodRate ? wetRate / goodRate : null,
    bandRates,
    band: current?.label ?? null,
    bandStarts: current ? starts(current.counts) : 0,
    bandTop3: current ? top3(current.counts) : 0,
  };
}

export const heavyLabel = (r: number | null) => (r === null ? null : r >= 1.1 ? "得意" : r <= 0.9 ? "苦手" : "普通");

export type HorsePedigreeAptitude = {
  sources: { label: string; name: string | null; weight: number; aptitude: StallionAptitude }[];
  heavyRatio: number | null; heavyLabel: string | null; wetRatio: number | null;
  bandRates: Record<string, number>; distanceType: string | null; // 補正後の3着内率が最も高い距離区分
  conditionRate: number | null; // 今回条件の3着内率（縮小推定・道悪補正込み）
  distanceFit: number | null; // 今回の距離区分の3着内率 ÷ 全距離区分の平均（>1 なら今回の距離が得意）
  deviation: number | null; // 出走馬内の偏差値
};

/** 出走馬全体の血統適性。records[horseIndex] は PEDIGREE_SOURCES 順の StallionRecord|null. */
export function pedigreeAptitudes(
  records: (StallionRecord | null)[][], surface: string, distance: number | null, going: string | null,
  sources: PedigreeSource[] = PEDIGREE_SOURCES,
): HorsePedigreeAptitude[] {
  const all = records.map((list) => list.map((r) => (r ? stallionAptitude(r, surface, distance) : null)));
  // 今回の距離帯の平均3着内率（縮小推定の事前値）
  let poolStarts = 0;
  let poolTop3 = 0;
  for (const list of all) for (const a of list) if (a) { poolStarts += a.bandStarts; poolTop3 += a.bandTop3; }
  const prior = poolStarts ? poolTop3 / poolStarts : 0;
  const isHeavy = going === "重" || going === "不良";

  const result = all.map((list, i) => {
    const usable = sources.map((s, j) => ({ ...s, record: records[i][j], aptitude: list[j] }))
      .filter((s): s is typeof s & { aptitude: StallionAptitude } => s.aptitude !== null);
    const avg = (get: (a: StallionAptitude) => number | null) => {
      let sum = 0;
      let w = 0;
      for (const s of usable) {
        const v = get(s.aptitude);
        if (v !== null) { sum += s.weight * v; w += s.weight; }
      }
      return w ? sum / w : null;
    };
    const heavyRatio = avg((a) => a.heavyRatio);
    const wetRatio = avg((a) => a.wetRatio);
    const bandRates: Record<string, number> = {};
    for (const label of Object.keys(BAND_TYPE)) {
      const v = avg((a) => a.bandRates[label] ?? null);
      if (v !== null) bandRates[label] = v;
    }
    const best = Object.entries(bandRates).sort((x, y) => y[1] - x[1])[0];
    const todayBand = usable.find((s) => s.aptitude.band)?.aptitude.band ?? null;
    const bandValues = Object.values(bandRates);
    const bandMean = bandValues.length ? bandValues.reduce((a, b) => a + b, 0) / bandValues.length : 0;
    const distanceFit = todayBand && bandRates[todayBand] !== undefined && bandMean > 0 ? bandRates[todayBand] / bandMean : null;
    const bandRate = avg((a) => (a.band ? (a.bandTop3 + SHRINK_STARTS * prior) / (a.bandStarts + SHRINK_STARTS) : null));
    const conditionRate = bandRate === null ? null : bandRate * (isHeavy && heavyRatio !== null ? heavyRatio : 1);
    return {
      sources: usable.map((s) => ({ label: s.label, name: s.record?.name ?? null, weight: s.weight, aptitude: s.aptitude })),
      heavyRatio, heavyLabel: heavyLabel(heavyRatio), wetRatio,
      bandRates, distanceType: best ? BAND_TYPE[best[0]] : null,
      conditionRate, distanceFit, deviation: null as number | null,
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

// --- 2段目モデル（AI確率 + 血統）。scripts/ml/train_stack.py が data/ml/stack.json を書き出す ---
export const STACK_FEATURES = ["base_logit", "ped_dev", "dist_fit", "heavy_fit"] as const;
export type StackModel = { features: string[]; mean: number[]; scale: number[]; coef: number[]; intercept: number };

export function stackFeatures(baseProbability: number, apt: HorsePedigreeAptitude | null, isHeavy: boolean): Record<string, number> {
  const p = Math.min(1 - 1e-6, Math.max(1e-6, baseProbability));
  return {
    base_logit: Math.log(p / (1 - p)),
    ped_dev: apt?.deviation ?? 50,
    dist_fit: apt?.distanceFit ?? 1,
    heavy_fit: isHeavy ? apt?.heavyRatio ?? 1 : 1,
  };
}

export function stackProbability(model: StackModel, features: Record<string, number>): number {
  let z = model.intercept;
  model.features.forEach((name, i) => { z += model.coef[i] * ((features[name] - model.mean[i]) / model.scale[i]); });
  return 1 / (1 + Math.exp(-z));
}
