"use client";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { type BetType } from "@/app/lib/bets";
import { AI_MARK_ORDER, assignAiMarks, type AiMark } from "@/app/lib/marks";
import { useParams } from "next/navigation";
import { valueLabels, type ValueModel, type ValueResult } from "@/app/lib/market";
import { featureLabel, SHAP_GROUPS } from "@/app/lib/ml/shapGroups";

type Conditions = {
  raceName: string | null; date: string | null; venue: string | null;
  raceNumber: number | null; startTime: string | null; surface: string | null;
  distanceMeters: number | null; direction: string | null; courseLayout: string | null;
  courseLetter: string | null; weather: string | null; weatherForecast?: boolean; conditionSource?: string; going: string | null;
};
type Entry = {
  frameNumber: number | null; horseNumber: number | null; sexAge: string | null;
  carriedWeight: number | null; jockey: string | null; trainer: string | null;
  bodyWeight: number | null; popularity: number | null;
};
type PerformanceStats = {
  parsedStarts: number; wins: number; top3: number; averageFinish: number | null;
  gradedStarts: number; gradedWins: number;
  sameVenueStarts: number; sameVenueTop3: number;
  sameVenueSurfaceStarts: number; sameVenueSurfaceTop3: number;
  similarDistanceStarts: number; similarDistanceTop3: number;
  sameGoingStarts: number; sameGoingTop3: number;
  positionStarts: number; positionGainTotal: number;
  positionForwardStarts: number; positionForwardTop3: number;
  goingUsed?: string | null;
  top2?: number; gradedTop2?: number; gradedTop3?: number;
  sameVenueSurfaceWins?: number; sameVenueSurfaceTop2?: number;
  similarDistanceWins?: number; similarDistanceTop2?: number; sameGoingWins?: number; sameGoingTop2?: number;
};
type CourseFrame = { starts: number; top3: number; courseStarts: number; courseTop3: number; years: string[] } | null;
type Pedigree = {
  sire: string | null; sireId: string | null; sireLine: string | null;
  dam: string | null; damId: string | null;
  damsire: string | null; damsireId: string | null; damdam: string | null;
  ancestors: { position: string; generation: number; name: string; horseId: string | null; country: string | null }[];
  crosses: { name: string; percentage: number | null; generations: string }[];
};
type StallionAptitude = {
  heavyRatio: number | null; heavyStarts: number; bandRates: Record<string, number>;
  band: string | null; bandStarts: number; bandTop3: number;
};
type PedigreeAptitude = {
  sources: { label: string; name: string | null; weight: number; aptitude: StallionAptitude }[];
  heavyRatio: number | null; heavyLabel: string | null;
  bandRates: Record<string, number>; distanceType: string | null;
  conditionRate: number | null; deviation: number | null;
};
type FormRun = {
  date: string; race: string; finish: number | null; field: number | null;
  timeDiff: number | null; agari: number | null; raceLast3f: number | null; score: number;
};
type JockeyYear = { n: number; top3: number; venueN: number; venueTop3: number; courseN: number; courseTop3: number };
type PaceInfo = {
  style: string | null; posFirstMean: number | null; nigeRate: number | null;
  highPace: { n: number; top3: number | null }; slowPace: { n: number; top3: number | null }; relPosRank: number | null;
};
type Extras = {
  prizeRatioLast: number | null; distChange: number | null; surfaceChange: number | null;
  weightChange: number | null; jockeyChange: number | null;
  tidxMean5: number | null; tidxBest5: number | null; wakuCourseN: number; wakuCourseDiff: number | null;
};
type Prediction = {
  ability?: { ab: number | null; agari: number | null; cond: number | null; early?: number | null };
  pace?: PaceInfo;
  extras?: Extras;
  trainerName?: string | null;
  trainer?: { n: number; top3: number; venueN: number; venueTop3: number };
  probability: number; baseProbability: number; pedigreeIncluded: boolean;
  winProbability?: number | null; top2Probability?: number | null; baseWinProbability?: number | null;
  inputRow?: (number | null)[]; // 人気薄専用の2段目（☆）の入力
  shapGroups?: { group: string; phi: number; top: { name: string; phi: number; value: number | null }[] }[]; // 各観点（SHAP）
  rank: number; modelFormDeviation: number | null; priorRuns: number; overseasRuns?: number; training?: { rank: string | null; critic: string | null } | null;
  heavyNote: {
    applied: boolean; show: boolean; label: string | null; ratio: number;
    before: number; after: number; goingKnown: boolean; beta: number;
    going: string | null; goingAssumedByUser: boolean;
  } | null;
  jockeyName: string | null; jockey: JockeyYear;
};
type MlSummary = {
  pace?: { nigeCandidates: number; frontRunners: number; frontRatio: number; expected: string } | null;
  status: "available" | "unavailable"; dataUntil?: string; goingAssumed?: boolean; pedigreeIncluded?: boolean; bodyWeightAssumed?: boolean;
  raceInTraining?: boolean;
  goingSource?: "announced" | "user" | "default";
  metrics?: { auc: number; raceTop3Hit: number; testYear: string };
  valueModel?: ValueModel | null;
  longshotReady?: boolean;
  marksEvidence?: Record<string, { marks: Record<string, MarkStats>; popLe3: MarkStats }> | null;
};
type MarkStats = { n: number; win_rate?: number; top3_rate?: number; win_roi?: number; place_roi?: number; place_roi_market_matched?: number };
type Horse = {
  horseId: string; name: string; status: string; raceEntry: Entry | null;
  pedigree?: Pedigree | null;
  prediction?: Prediction | null;
  form?: { runs: FormRun[]; raw: number | null; deviation: number | null } | null;
  pedigreeAptitude?: PedigreeAptitude | null;
  profile: { label: string; value: string }[] | null;
  performances: { values: Record<string, string> }[]; historyReturned: number;
  performanceStats?: PerformanceStats;
  courseFrame?: CourseFrame;
};
type ResponseData = {
  ok: boolean; raceId: string; raceConditions: Conditions; horseCount: number;
  completeCount: number; partialCount: number; failedCount: number;
  results: Horse[]; error?: string; ml?: MlSummary;
};
type CachedRace = { savedAt: number; data: ResponseData };
type OddsMap = Record<string, { min: number; max: number | null }>;
type OddsData = { ok: boolean; fetchedAt: string; officialTime: string | null; errors: string[]; status?: string | null;
  win: OddsMap; place: OddsMap; umaren: OddsMap; wide: OddsMap; trio: OddsMap; trifecta: OddsMap };
type HorseOdds = { win: number | null; place: { min: number; max: number | null } | null; popularity: number | null };
type FactorScore = { label: string; score: number; starts: number; rate: number; rank: number; rankedCount: number };
type CategoryRank = { label: string; rank: number | null; total: number; detail: string; score?: number; type?: string | null };

/**
 * ラップタイプ（近走の推定上がりの速さと、ゴール前600m 地点での先頭との差。どちらも直近加重平均）。
 * 区切りは store の全馬の3等分（2026-11-22）：600m 地点の差 0.65秒以下＝前、1.09秒以上＝後ろ。上がり +0.16以上＝速い、−0.54以下＝遅い
 */
function lapType(p: Prediction): string | null {
  const ag = p.ability?.agari, eg = p.ability?.early;
  if (ag == null || eg == null) return null;
  if (eg <= 0.65) return ag <= -0.54 ? "先行失速型" : "持続型";
  if (eg >= 1.09 && ag >= 0.16) return "終い伸び型";
  return "平均型";
}
const typeOf = (group: string, p: Prediction) =>
  group === "脚質" ? p.pace?.style ?? null : group === "上がり・ラップタイプ" ? lapType(p) : null;

const CLIENT_CACHE_TTL_MS = 5 * 60 * 1000;
// モデルが変わったら（再学習）保存済みの AI 予想を使わない。キーにモデルの版を入れる（next.config.ts）
const cacheKey = (id: string) => "keiba-race-detail-v7:" + (process.env.NEXT_PUBLIC_MODEL_VERSION ?? "") + ":" + id;
const marksCacheKey = (id: string) => "keiba-race-marks-v1:" + id;
const oddsCacheKey = (id: string) => "keiba-race-odds-v1:" + id;
const MARK_OPTIONS = ["", "◎", "○", "▲", "△", "☆", "注", "消"];

function readMarks(id: string): Record<string, string> {
  try {
    const raw = window.localStorage.getItem(marksCacheKey(id));
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const validEntries: [string, string][] = Object.entries(parsed).filter(
      (entry): entry is [string, string] =>
        typeof entry[1] === "string" && MARK_OPTIONS.includes(entry[1])
    );
    return Object.fromEntries(validEntries);
  } catch { return {}; }
}
function saveMarks(id: string, marks: Record<string, string>) {
  try { window.localStorage.setItem(marksCacheKey(id), JSON.stringify(marks)); }
  catch { /* 保存できない場合も画面操作は継続する */ }
}
// 取得したオッズは端末に保存し、タブやページを移っても消えないようにする（「オッズを取得」で毎回取り直す）
function readCachedOdds(id: string): OddsData | null {
  try {
    const raw = window.localStorage.getItem(oddsCacheKey(id));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as OddsData;
    return parsed && parsed.ok && parsed.win && typeof parsed.fetchedAt === "string" ? parsed : null;
  } catch { return null; }
}
function saveCachedOdds(id: string, odds: OddsData) {
  try { window.localStorage.setItem(oddsCacheKey(id), JSON.stringify(odds)); }
  catch { /* 保存できない場合も画面表示は継続する */ }
}
function readCachedRace(id: string): CachedRace | null {
  try {
    const raw = window.localStorage.getItem(cacheKey(id));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachedRace;
    if (!parsed || typeof parsed.savedAt !== "number" || !parsed.data ||
      parsed.data.raceId !== id || !parsed.data.raceConditions ||
      !Array.isArray(parsed.data.results)) return null;
    return parsed;
  } catch { return null; }
}
function saveCachedRace(id: string, data: ResponseData, savedAt: number) {
  try { window.localStorage.setItem(cacheKey(id), JSON.stringify({ savedAt, data })); }
  catch { /* キャッシュできない場合も画面表示は継続する */ }
}
function display(value: string | number | null | undefined) {
  return value === null || value === undefined || value === "" ? "" : String(value);
}
function formatRaceDate(date: string | null) {
  const m = date?.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return display(date);
  const weekday = ["日", "月", "火", "水", "木", "金", "土"][new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])).getUTCDay()];
  return `${+m[2]}/${+m[3]}（${weekday}）`;
}
function formatSavedAt(timestamp: number | null) {
  if (!timestamp) return "保存データなし";
  return "保存データ：" + new Date(timestamp).toLocaleString("ja-JP", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

// 今年の騎手成績: 同じ競馬場の3着内率を、騎乗数が少ないほど今年全体の3着内率に寄せる
const JOCKEY_PRIOR_RIDES = 20;
function jockeyYearRate(j: JockeyYear | undefined) {
  if (!j || j.n === 0) return { starts: 0, rate: 0 };
  return { starts: j.n, rate: (j.venueTop3 + JOCKEY_PRIOR_RIDES * (j.top3 / j.n)) / (j.venueN + JOCKEY_PRIOR_RIDES) };
}
const pct = (rate: number) => Math.round(rate * 100) + "%";
// 「N走・勝率 a%・連対率 b%・3着以内率 c%」（勝ち・連対の数がないときは3着以内率だけ）
function recordText(starts: number, top3: number, wins?: number, top2?: number) {
  if (!starts) return "0走";
  const parts = [`${starts}走`];
  if (wins != null) parts.push(`勝率 ${pct(wins / starts)}`);
  if (top2 != null) parts.push(`連対率 ${pct(top2 / starts)}`);
  parts.push(`3着以内率 ${pct(top3 / starts)}`);
  return parts.join("・");
}

function buildFactorScores(horses: Horse[]): Map<string, FactorScore[]> {
  const factors = [
    { label: "全体成績", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.parsedStarts ?? 0, rate: s?.parsedStarts ? s.top3 / s.parsedStarts : 0 }; } },
    { label: "重賞実績", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.gradedStarts ?? 0, rate: s?.gradedStarts ? (s.gradedTop3 ?? s.gradedWins) / s.gradedStarts : 0 }; } },
    { label: "競馬場・馬場適性", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.sameVenueSurfaceStarts ?? 0, rate: s?.sameVenueSurfaceStarts ? s.sameVenueSurfaceTop3 / s.sameVenueSurfaceStarts : 0 }; } },
    { label: "距離適性", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.similarDistanceStarts ?? 0, rate: s?.similarDistanceStarts ? s.similarDistanceTop3 / s.similarDistanceStarts : 0 }; } },
    { label: "馬場適性", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.sameGoingStarts ?? 0, rate: s?.sameGoingStarts ? s.sameGoingTop3 / s.sameGoingStarts : 0 }; } },
    { label: "枠順実績", get: (horse: Horse) => { const c = horse.courseFrame; return { starts: c?.starts ?? 0, rate: c?.starts ? c.top3 / c.starts : 0 }; } },
    { label: "調子", get: (horse: Horse) => ({ starts: horse.form?.runs.length ?? 0, rate: horse.form?.raw ?? 0 }) },
    { label: "種牡馬", get: (horse: Horse) => { const a = horse.pedigreeAptitude?.sources.find((x) => x.label === "父")?.aptitude; return { starts: a?.bandStarts ?? 0, rate: a?.bandStarts ? a.bandTop3 / a.bandStarts : 0 }; } },
    { label: "血統適性", get: (horse: Horse) => { const a = horse.pedigreeAptitude; return { starts: a?.conditionRate != null ? 1 : 0, rate: a?.conditionRate ?? 0 }; } },
    { label: "騎手(今年)", get: (horse: Horse) => jockeyYearRate(horse.prediction?.jockey) },
  ];
  const result = new Map<string, FactorScore[]>();
  for (const factor of factors) {
    const values = horses.map((horse) => ({
      horseId: horse.horseId,
      ...factor.get(horse),
    })).filter((item) => item.starts > 0);
    if (!values.length) continue;
    const mean = values.reduce((sum, item) => sum + item.rate, 0) / values.length;
    const variance = values.reduce((sum, item) => sum + Math.pow(item.rate - mean, 2), 0) / values.length;
    const sd = Math.sqrt(variance);
    const scored = values.map((item) => ({
      ...item,
      score: sd === 0 ? 50 : Math.max(20, Math.min(80, 50 + 10 * (item.rate - mean) / sd)),
    })).sort((a, b) => b.score - a.score);
    let currentRank = 0;
    let previousScore: number | null = null;
    scored.forEach((item, index) => {
      if (previousScore === null || Math.abs(previousScore - item.score) >= 0.05) currentRank = index + 1;
      previousScore = item.score;
      const list = result.get(item.horseId) ?? [];
      list.push({
        label: factor.label,
        score: Math.round(item.score * 10) / 10,
        starts: item.starts,
        rate: item.rate,
        rank: currentRank,
        rankedCount: scored.length,
      });
      result.set(item.horseId, list);
    });
  }
  return result;
}

const NO_DATA = "評価に必要なデータが未取得";
function buildCategoryRanks(
  horses: Horse[],
  factorScores: Map<string, FactorScore[]>,
  distanceMeters: number | null,
  venue: string | null
): Map<string, CategoryRank[]> {
  const result = new Map<string, CategoryRank[]>();
  const categories = ["全体成績", "重賞実績", "調子", "騎手", "調教", "血統", "距離適性", "コース適性", "馬場適性", "枠順", "種牡馬", "ポジション"];
  for (const horse of horses) {
    result.set(horse.horseId, categories.map((label) => ({
      label, rank: null, total: 0, detail: NO_DATA,
    })));
  }
  const distanceValues = horses.map((horse) => {
    const value = (factorScores.get(horse.horseId) ?? []).find((item) => item.label === "距離適性");
    return value ? { horseId: horse.horseId, score: value.score, starts: value.starts, rate: value.rate } : null;
  }).filter((value): value is { horseId: string; score: number; starts: number; rate: number } => value !== null)
    .sort((a, b) => b.score - a.score);
  let distanceRank = 0;
  let previousDistanceScore: number | null = null;
  distanceValues.forEach((value, index) => {
    if (previousDistanceScore === null || Math.abs(previousDistanceScore - value.score) >= 0.05) distanceRank = index + 1;
    previousDistanceScore = value.score;
    const row = result.get(value.horseId)?.find((item) => item.label === "距離適性");
    if (row) {
      row.rank = distanceRank;
      row.total = distanceValues.length;
      const st = horses.find((horse) => horse.horseId === value.horseId)?.performanceStats;
      row.detail = `今回の距離±200mの全過去走 ${recordText(value.starts, st?.similarDistanceTop3 ?? Math.round(value.rate * value.starts), st?.similarDistanceWins, st?.similarDistanceTop2)}`;
    }
  });
  const assignFactorRank = (factorLabel: string, categoryLabel: string, detailBuilder: (value: FactorScore, horseId: string) => string) => {
    const values = horses.map((horse) => {
      const factor = (factorScores.get(horse.horseId) ?? []).find((item) => item.label === factorLabel);
      return factor ? { horseId: horse.horseId, factor } : null;
    }).filter((value): value is { horseId: string; factor: FactorScore } => value !== null)
      .sort((a, b) => b.factor.score - a.factor.score);
    let rank = 0;
    let previousScore: number | null = null;
    values.forEach((value, index) => {
      if (previousScore === null || Math.abs(previousScore - value.factor.score) >= 0.05) rank = index + 1;
      previousScore = value.factor.score;
      const row = result.get(value.horseId)?.find((item) => item.label === categoryLabel);
      if (row) {
        row.rank = rank;
        row.total = values.length;
        row.detail = detailBuilder(value.factor, value.horseId);
      }
    });
  };
  const statsOf = (horseId: string) => horses.find((horse) => horse.horseId === horseId)?.performanceStats;
  assignFactorRank("競馬場・馬場適性", "コース適性", (value, horseId) => {
    const st = statsOf(horseId);
    return `同競馬場・同芝ダート ${recordText(value.starts, st?.sameVenueSurfaceTop3 ?? 0, st?.sameVenueSurfaceWins, st?.sameVenueSurfaceTop2)}`;
  });
  const goingUsed = horses.find((horse) => horse.performanceStats?.goingUsed)?.performanceStats?.goingUsed ?? null;
  assignFactorRank("馬場適性", "馬場適性", (value, horseId) => {
    const st = statsOf(horseId);
    return `${goingUsed ?? "同じ馬場"}での過去走 ${recordText(value.starts, st?.sameGoingTop3 ?? 0, st?.sameGoingWins, st?.sameGoingTop2)}`;
  });
  assignFactorRank("全体成績", "全体成績", (value, horseId) => {
    const st = statsOf(horseId);
    return `全過去走 ${recordText(value.starts, st?.top3 ?? 0, st?.wins, st?.top2)}`;
  });
  assignFactorRank("種牡馬", "種牡馬", () => "");
  assignFactorRank("重賞実績", "重賞実績", (value, horseId) => {
    const st = statsOf(horseId);
    return `重賞 ${recordText(value.starts, st?.gradedTop3 ?? 0, st?.gradedWins, st?.gradedTop2)}`;
  });
  assignFactorRank("騎手(今年)", "騎手", () => "");
  assignFactorRank("調子", "調子", () => "");
  assignFactorRank("血統適性", "血統", () => "");
  assignFactorRank("枠順実績", "枠順", (_value, horseId) => {
    const c = horses.find((horse) => horse.horseId === horseId)?.courseFrame;
    return c ? `${venue ?? ""}${distanceMeters ? distanceMeters + "m" : ""}のこの枠（${c.years[0]}〜${c.years[c.years.length - 1]}年）${c.starts}頭・3着以内率 ${pct(c.top3 / c.starts)}（コース全体 ${pct(c.courseTop3 / c.courseStarts)}）` : "";
  });


  const positionValues = horses.map((horse) => {
    const stats = horse.performanceStats;
    if (!stats || stats.positionStarts === 0) return null;
    return {
      horseId: horse.horseId,
      starts: stats.positionStarts,
      averageGain: stats.positionGainTotal / stats.positionStarts,
    };
  }).filter((value): value is { horseId: string; starts: number; averageGain: number } => value !== null)
    .sort((a, b) => b.averageGain - a.averageGain);
  let positionRank = 0;
  let previousPositionValue: number | null = null;
  positionValues.forEach((value, index) => {
    if (previousPositionValue === null || Math.abs(previousPositionValue - value.averageGain) >= 0.05) {
      positionRank = index + 1;
    }
    previousPositionValue = value.averageGain;
    const row = result.get(value.horseId)?.find((item) => item.label === "ポジション");
    if (row) {
      row.rank = positionRank;
      row.total = positionValues.length;
      const direction = value.averageGain >= 0 ? "平均順位上昇" : "平均順位後退";
      row.detail = `4角からゴールまで ${direction} ${Math.abs(value.averageGain).toFixed(1)}位（${value.starts}走）`;
    }
  });

  for (const horse of horses) {
    const rows = result.get(horse.horseId) ?? [];
    const row = (label: string) => rows.find((item) => item.label === label);
    const apt = horse.pedigreeAptitude;
    const bloodlineRow = row("血統");
    if (bloodlineRow && apt) {
      const names = apt.sources.map((s) => `${s.label}${s.name ?? ""}`).join("・");
      bloodlineRow.detail = [
        apt.deviation != null ? `今回条件の血統適性 偏差値${apt.deviation.toFixed(1)}` : null,
        apt.distanceType ? `距離タイプ：${apt.distanceType}（${bandText(apt.bandRates)}）` : null,
        apt.heavyLabel ? `道悪：${apt.heavyLabel}（重・不良の3着内率が良の${apt.heavyRatio!.toFixed(2)}倍）` : null,
        `元データ：${names}の産駒成績（父50%・母父25%・母母父12.5%で加重）`,
      ].filter(Boolean).join(" / ");
    }
    const sireRow = row("種牡馬");
    const sire = apt?.sources.find((s) => s.label === "父");
    if (sireRow) {
      if (sire) {
        const a = sire.aptitude;
        sireRow.detail = [
          `父：${sire.name ?? horse.pedigree?.sire ?? ""}${horse.pedigree?.sireLine ? `（${horse.pedigree.sireLine}）` : ""}`,
          a.band && a.bandStarts ? `産駒の今回距離帯 ${a.bandStarts}走・3着内率 ${pct(a.bandTop3 / a.bandStarts)}` : null,
          Object.keys(a.bandRates).length ? `距離別3着内率：${bandText(a.bandRates)}` : null,
          a.heavyRatio != null ? `道悪：${a.heavyRatio >= 1.1 ? "得意" : a.heavyRatio <= 0.9 ? "苦手" : "普通"}（良の${a.heavyRatio.toFixed(2)}倍）` : null,
        ].filter(Boolean).join(" / ");
      } else if (horse.pedigree?.sire) {
        sireRow.detail = `父：${horse.pedigree.sire}（産駒成績を取得できず）`;
      }
    }
    const jockeyRow = row("騎手");
    const j = horse.prediction?.jockey;
    if (jockeyRow && j) {
      jockeyRow.detail = j.n === 0
        ? `今年の騎乗データなし（${horse.raceEntry?.jockey ?? ""}）`
        : `今年：全体 ${j.n}騎乗・3着内率 ${pct(j.top3 / j.n)}` +
          ` / ${venue ?? "同競馬場"} ${j.venueN}騎乗${j.venueN ? "・" + pct(j.venueTop3 / j.venueN) : ""}` +
          ` / 同コース ${j.courseN}騎乗${j.courseN ? "・" + pct(j.courseTop3 / j.courseN) : ""}`;
    }
    const formRow = row("調子");
    const form = horse.form;
    if (formRow && form) {
      formRow.detail = form.runs.length === 0 ? "近走データなし" :
        `調子偏差値 ${form.deviation != null ? form.deviation.toFixed(1) : "-"}（近${form.runs.length}走: ` +
        form.runs.map((r) => `${r.finish ?? "中止"}着${r.field ? "/" + r.field + "頭" : ""}${r.timeDiff != null ? " " + r.timeDiff.toFixed(1) + "秒差" : ""}${r.agari != null ? " 上がり" + r.agari.toFixed(1) : ""}`).join("、") + "）";
    }
  }

  return result;
}

// AI予測の補足情報を「項目名・内容」の行にする
const FRAME_COLORS: Record<number, [string, string]> = {
  1: ["#f4f4f4", "#111"], 2: ["#222", "#fff"], 3: ["#d8323a", "#fff"], 4: ["#2a6ad8", "#fff"],
  5: ["#f2cf2a", "#111"], 6: ["#2e9e4f", "#fff"], 7: ["#f08a24", "#111"], 8: ["#f39ac0", "#111"],
};
function HorseNumberBadge({ entry }: { entry: Entry | null }) {
  if (entry?.horseNumber == null) return null;
  const colors = entry.frameNumber != null ? FRAME_COLORS[entry.frameNumber] : undefined;
  return <span className="horse-number-badge" title={entry.frameNumber != null ? entry.frameNumber + "枠" : undefined}
    style={colors ? { background: colors[0], color: colors[1] } : undefined}>{entry.horseNumber}</span>;
}

const BAND_NAMES: Record<string, string> = { "-1400": "〜1400m", "-1800": "〜1800m", "-2200": "〜2200m", "-2600": "〜2600m", "2600-": "2600m〜" };
const bandText = (rates: Record<string, number>) =>
  Object.entries(rates).map(([band, rate]) => `${BAND_NAMES[band] ?? band} ${Math.round(rate * 100)}%`).join("・");

function HorseDetailCard({ horse, mark, onMarkChange, odds }: {
  horse: Horse; mark: string; onMarkChange: (horseId: string, mark: string) => void; odds: HorseOdds | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const entry = horse.raceEntry;
  const profile = horse.profile ?? [];
  const profileValue = (label: string) => profile.find((item) => item.label.includes(label))?.value;
  const histories = horse.performances ?? [];
  const pedigree = horse.pedigree;
  const sire = pedigree?.sire ?? profileValue("父") ?? "";
  const damsire = pedigree?.damsire ?? "";
  const facts: [string, string][] = ([
    ["馬体重", entry?.bodyWeight == null ? "" : String(entry.bodyWeight) + "kg"],
    ["人気", odds?.popularity != null ? String(odds.popularity) + "番人気" : entry?.popularity == null ? "" : String(entry.popularity) + "番人気"],
    ["単勝", odds?.win ? odds.win.toFixed(1) + "倍" : ""],
    ["複勝", odds?.place ? odds.place.min.toFixed(1) + (odds.place.max ? "〜" + odds.place.max.toFixed(1) : "") + "倍" : ""],
    ["調教師", entry?.trainer ?? ""],
    ["獲得賞金", profileValue("中央獲得賞金") ?? ""],
    ["主な勝ち鞍", profileValue("主な勝ち鞍") ?? ""],
    ["父系統", pedigree?.sireLine ?? ""],
    ["母", pedigree?.dam ?? ""],
    ["クロス", (pedigree?.crosses ?? []).map((cross) =>
      `${cross.name} ${cross.generations}${cross.percentage === null ? "" : `（${cross.percentage}%）`}`).join("、")],
  ] as [string, string][]).filter(([, value]) => value !== "" && value !== "-");
  const pedigreeRows = (pedigree?.ancestors ?? []).filter((ancestor) => ancestor.generation <= 3);
  const recent = histories.slice(0, 5);
  const v = (run: { values: Record<string, string> }, key: string) => run.values[key] ?? "";
  return <details className="horse-detail" onToggle={(event) => setExpanded(event.currentTarget.open)}>
    <summary className="horse-summary entry-row">
      <HorseNumberBadge entry={entry} />
      <span className="horse-main">
        <b>{horse.name}</b>
        <small>{[entry?.sexAge, entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) : null].filter(Boolean).join("・")}</small>
      </span>
      <span className="ai-odds">{odds?.win ? <><b className={odds.popularity != null && odds.popularity <= 3 ? "fav" : ""}>{odds.win.toFixed(1)}</b><small>{odds.popularity}番人気</small></> : <small>-</small>}</span>
      {/* 印の操作で開閉しないように、クリックの既定動作（summary の開閉）を止める */}
      <span className="horse-mark-slot" onClick={(event) => event.preventDefault()}>
        <select className={"horse-mark-select" + (mark ? " marked" : "")} aria-label={horse.name + "の印"} value={mark}
          onChange={(event) => onMarkChange(horse.horseId, event.target.value)}>
          {MARK_OPTIONS.map((option) => <option key={option || "none"} value={option}>{option || "印"}</option>)}
        </select>
      </span>
    </summary>
    {expanded && <div className="horse-body entry-body">
      {recent.length > 0 && <div className="entry-runs">
        <h4>近5走</h4>
        {recent.map((run, index) => {
          const finish = Number(v(run, "着順"));
          return <div className="entry-run" key={index}>
            <span className="entry-run-date">{v(run, "日付").slice(2)}</span>
            <span className="entry-run-race">{v(run, "レース名")}<small>{[v(run, "距離"), v(run, "馬場"), v(run, "騎手")].filter(Boolean).join(" ")}</small></span>
            <span className={"entry-run-finish" + (finish >= 1 && finish <= 3 ? " place-" + finish : "")}><b>{v(run, "着順")}</b><small>/{v(run, "頭数")}着</small>
              {v(run, "着差") && finish !== 1 && <small>{v(run, "着差")}秒差</small>}</span>
          </div>;
        })}
      </div>}
      <p className="entry-blood">父 <b>{sire || "-"}</b>　母 {pedigree?.dam ?? "-"}　母父 {damsire || "-"}</p>
      <p className="entry-sub">{[entry?.bodyWeight != null ? "馬体重 " + entry.bodyWeight + "kg" : null, entry?.trainer ? "調教師 " + entry.trainer : null,
        odds?.place ? "複勝 " + odds.place.min.toFixed(1) + (odds.place.max ? "〜" + odds.place.max.toFixed(1) : "") + "倍" : null].filter(Boolean).join("　")}</p>
      <button type="button" className="more-button" onClick={() => setShowMore((x) => !x)}>{showMore ? "閉じる" : "もっと見る（全戦績・血統表など）"}</button>
      {showMore && <>
        {facts.length > 0 && <dl className="fact-list">
          {facts.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
        </dl>}
        {histories.length > 0 && <div className="detail-block">
          <h4>戦績（{histories.length}走）</h4>
          <div className="history-table-wrap">
            <table className="history-table run-table">
              <thead><tr><th>日付</th><th>レース</th><th>着順</th><th>人気</th><th>距離</th><th>馬場</th><th>タイム</th><th>着差</th><th>通過</th><th>上り</th><th>騎手</th><th>斤量</th><th>馬体重</th></tr></thead>
              <tbody>{histories.map((run, index) => {
                const finish = Number(v(run, "着順"));
                return <tr key={index}>
                  <td>{v(run, "日付").slice(2)}</td>
                  <td className="run-race"><span>{v(run, "開催")}</span>{v(run, "レース名")}</td>
                  <td className={"run-finish" + (finish >= 1 && finish <= 3 ? " place-" + finish : "")}>{v(run, "着順")}<small>/{v(run, "頭数")}</small></td>
                  <td>{v(run, "人気")}</td><td>{v(run, "距離")}</td><td>{v(run, "馬場")}</td><td>{v(run, "タイム")}</td>
                  <td>{v(run, "着差")}</td><td>{v(run, "通過")}</td><td>{v(run, "上り")}</td><td>{v(run, "騎手")}</td>
                  <td>{v(run, "斤量")}</td><td>{v(run, "馬体重")}</td>
                </tr>;
              })}</tbody>
            </table>
          </div>
        </div>}
        {pedigreeRows.length > 0 && <div className="detail-block">
          <h4>血統（3代）</h4>
          <div className="history-table-wrap">
            <table className="pedigree-table">
              <tbody>{pedigreeRows.filter((ancestor) => ancestor.generation === 3).map((ancestor) => {
                const parent = pedigreeRows.find((item) => item.position === ancestor.position.slice(0, 2));
                const grand = pedigreeRows.find((item) => item.position === ancestor.position.slice(0, 1));
                const first = ancestor.position.endsWith("F");
                const side = ancestor.position.startsWith("F") ? "sire-side" : "dam-side";
                return <tr key={ancestor.position} className={side}>
                  {ancestor.position.slice(1) === "FF" && grand && <td rowSpan={4} className="ped-1"><b>{grand.name}</b></td>}
                  {first && parent && <td rowSpan={2} className="ped-2">{parent.name}</td>}
                  <td className="ped-3">{ancestor.name}{ancestor.country ? `（${ancestor.country}）` : ""}</td>
                </tr>;
              })}</tbody>
            </table>
          </div>
        </div>}
      </>}
    </div>}
  </details>;
}

const FACTOR_BY_CATEGORY: Record<string, string> = {
  "コース適性": "競馬場・馬場適性",
  "騎手": "騎手(今年)",
  "枠順": "枠順実績",
  "血統": "血統適性",
};

// 出走馬の中で上位3位以内 → 強み、下位3位以内 → 弱み
function strengthsOf(categories: CategoryRank[]) {
  const ranked = categories.filter((c) => c.rank !== null && c.total >= 4);
  return {
    strong: ranked.filter((c) => c.rank! <= 3).map((c) => c.label),
    weak: ranked.filter((c) => c.rank! > 3 && c.rank! > c.total - 3).map((c) => c.label),
  };
}

/** AIスコア＝3着以内確率の、出走馬の中での偏差値（平均50・標準偏差10）。全馬同じ確率なら50 */
function scoreDeviations(horses: Horse[]): Map<string, number> {
  const rows = horses.filter((h) => h.prediction);
  const ps = rows.map((h) => h.prediction!.probability);
  const mean = ps.reduce((s, x) => s + x, 0) / (ps.length || 1);
  const sd = Math.sqrt(ps.reduce((s, x) => s + (x - mean) * (x - mean), 0) / (ps.length || 1));
  return new Map(rows.map((h) => [h.horseId, sd > 0 ? 50 + 10 * (h.prediction!.probability - mean) / sd : 50]));
}

/** 偏差値をバーの長さ（%）に：20→0%、80→100%（平均50は真ん中） */
const scoreBar = (score: number) => Math.max(0, Math.min(100, (score - 20) / 60 * 100));

function probabilityTier(rank: number | undefined) {
  if (rank === undefined) return "";
  return rank <= 3 ? " tier-top" : rank <= 6 ? " tier-mid" : "";
}

const VALUE_CLASS: Record<string, string> = { 妙味あり: "good", 妥当: "even", 割高: "bad" };

function OverallHorseCard({ horse, categories, factors, mark, odds, value, score }: {
  horse: Horse; total: number; categories: CategoryRank[]; factors: FactorScore[]; mark: string;
  odds: HorseOdds | null; value: ValueResult | null; score: number | null;
}) {
  const prediction = horse.prediction;
  const probability = prediction ? prediction.probability * 100 : null;
  const probabilityText = probability === null ? "算出不可" : probability.toFixed(1) + "%";
  const entry = horse.raceEntry;
  const factorFor = (label: string) => {
    const factorLabel = FACTOR_BY_CATEGORY[label] ?? label;
    return factors.find((factor) => factor.label === factorLabel);
  };
  const { strong, weak } = strengthsOf(categories);
  const missing = categories.filter((c) => c.rank === null && !factorFor(c.label) && c.detail === NO_DATA).map((c) => c.label);
  const heavy = prediction?.heavyNote && (prediction.heavyNote.applied || prediction.heavyNote.show) ? prediction.heavyNote : null;

  return <details className={"prediction-horse-details" + probabilityTier(prediction?.rank)}>
    <summary className="prediction-horse-summary ai-row">
      <HorseNumberBadge entry={entry} />
      <span className="horse-main">
        <b>{mark && <span className="horse-mark">{mark}</span>}{horse.name}</b>
        <small>{entry?.jockey ?? ""}</small>
      </span>
      <span className="ai-odds">
        {odds?.win != null ? <><b className={odds.popularity != null && odds.popularity <= 3 ? "fav" : ""}>{odds.win.toFixed(1)}</b><small>{odds.popularity}番人気</small></> : <small>-</small>}
      </span>
      <span className="ai-score" role="img" aria-label={horse.name + " AIスコア（出走馬の中の偏差値） " + (score === null ? "算出不可" : score.toFixed(1)) + "、3着以内確率 " + probabilityText}>
        <strong>{score === null ? "-" : Math.round(score)}</strong>
        <span className="ai-score-track">
          <span className="ai-score-fill" style={{ width: (score === null ? 0 : scoreBar(score)) + "%" }} />
          <span className="ai-score-avg" style={{ left: scoreBar(50) + "%" }} />
        </span>
      </span>
      <span className={"ai-value " + (value ? VALUE_CLASS[value.label] : "none")}>{value?.label ?? "-"}</span>
    </summary>

    <div className="horse-result-content ai-detail">
      <div className="ai-detail-head">
        <div>
          <b>{horse.name}</b>
          <small>{[odds?.win != null ? odds.win.toFixed(1) + "倍 " + (odds.popularity ?? "-") + "番人気" : null, entry?.sexAge, entry?.jockey,
            entry?.carriedWeight != null ? entry.carriedWeight.toFixed(1) + "kg" : null].filter(Boolean).join(" / ")}</small>
          {prediction && prediction.winProbability != null && prediction.top2Probability != null &&
            <small>勝率 {(prediction.winProbability * 100).toFixed(1)}% ・ 連対率 {(prediction.top2Probability * 100).toFixed(1)}% ・ 3着以内 {probabilityText}</small>}
        </div>
        <span className="ai-detail-score"><small>AIスコア</small><strong>{score === null ? "-" : Math.round(score)}</strong><small>3着内 {probability === null ? "-" : probability.toFixed(1) + "%"}</small></span>
      </div>
      {strong.length > 0 && <section className="ai-points plus">
        <h4>プラス評価ポイント</h4><p>出走馬の中で上位3頭に入る観点</p>
        <div>{strong.map((label) => <span key={label}>{label}</span>)}</div>
      </section>}
      {weak.length > 0 && <section className="ai-points minus">
        <h4>マイナス評価ポイント</h4><p>出走馬の中で下位3頭に入る観点</p>
        <div>{weak.map((label) => <span key={label}>{label}</span>)}</div>
      </section>}
      {heavy && <p className="heavy-note" role="note">
        ⚠ 道悪（重・不良）未経験・血統は道悪{heavy.label}（血統の道悪倍率 {heavy.ratio.toFixed(2)}倍）。血統の道悪適性は、AIスコアにすでに含めて計算している。
      </p>}
      <section className="ai-ranks">
        <h4>各観点の偏差値</h4>
        {categories.filter((c) => !missing.includes(c.label)).map((c) => {
          const top = c.rank !== null && c.rank <= 3 && c.total >= 4;
          return <details key={c.label} className={top ? "top" : ""}>
            <summary><span className="rank-label">{c.label}<small>{[c.type, c.rank !== null ? `${c.rank}位/${c.total}頭` : ""].filter(Boolean).join("・")}</small></span>
              {c.score != null ? <span className="rank-bar" role="img" aria-label={`${c.label} 偏差値 ${Math.round(c.score)}`}>
                <b>{Math.round(c.score)}</b>
                <span className="rank-track">
                  <span className="rank-fill" style={{ width: scoreBar(c.score) + "%" }} />
                  {[30, 40, 60, 70].map((v) => <span key={v} className="rank-grid" style={{ left: scoreBar(v) + "%" }} />)}
                  <span className="rank-avg" style={{ left: scoreBar(50) + "%" }} />
                </span>
              </span> : <span>{c.rank !== null ? <><b>{c.rank}</b>位<small>/{c.total}頭</small></> : <small>-</small>}</span>}</summary>
            {c.detail && <p>{c.detail}</p>}
          </details>;
        })}
        {missing.length > 0 && <p className="missing-note">データなし：{missing.join("・")}</p>}
        <p className="ai-ranks-note">観点の数字＝AIの182項目を観点ごとにまとめ、その観点がAIスコアをどれだけ押し上げたか（SHAP）を、出走馬の中の偏差値（平均50）にしたもの。順位も同じ順。押すと、確率への影響（pt）と主な項目が出る。</p>
      </section>
    </div>
  </details>;
}

/** 観点別：観点を選ぶと、全馬をその観点の偏差値のバーで並べる（偏差値の高い順）。脚質・ラップタイプは型の名前も出す. */
function FactorView({ horses, categoryRanks, factor, onFactor }: {
  horses: Horse[]; categoryRanks: Map<string, CategoryRank[]>; factor: string; onFactor: (f: string) => void;
}) {
  const groups = SHAP_GROUPS.filter((g) => horses.some((h) => categoryRanks.get(h.horseId)?.some((c) => c.label === g)));
  const rows = horses.map((h) => ({ h, c: categoryRanks.get(h.horseId)?.find((c) => c.label === factor) }))
    .filter((x): x is { h: Horse; c: CategoryRank } => !!x.c && x.c.score != null)
    .sort((a, b) => b.c.score! - a.c.score!);
  return <section className="factor-view">
    <div className="factor-chips" role="group" aria-label="観点">
      {groups.map((g) => <button key={g} type="button" className={g === factor ? "active" : ""} onClick={() => onFactor(g)}>{g}</button>)}
    </div>
    <ul className="factor-list ai-ranks">
      {rows.map(({ h, c }) => <li key={h.horseId}>
        <details className={c.rank !== null && c.rank <= 3 && c.total >= 4 ? "top" : ""}>
          <summary>
            <span className="factor-horse"><HorseNumberBadge entry={h.raceEntry} /><span className="rank-label">{h.name}
              <small>{[c.type, `${c.rank}位/${c.total}頭`].filter(Boolean).join("・")}</small></span></span>
            <span className="rank-bar" role="img" aria-label={`${h.name} ${factor} 偏差値 ${Math.round(c.score!)}`}>
              <b>{Math.round(c.score!)}</b>
              <span className="rank-track">
                <span className="rank-fill" style={{ width: scoreBar(c.score!) + "%" }} />
                {[30, 40, 60, 70].map((v) => <span key={v} className="rank-grid" style={{ left: scoreBar(v) + "%" }} />)}
                <span className="rank-avg" style={{ left: scoreBar(50) + "%" }} />
              </span>
            </span>
          </summary>
          {c.detail && <p>{c.detail}</p>}
        </details>
      </li>)}
    </ul>
    <p className="ai-ranks-note">観点の数字＝その観点がAIスコアをどれだけ押し上げたか（SHAP）の、出走馬の中の偏差値（平均50）。押すと中身が出る。
      ラップタイプは近走の推定上がりの速さと、ゴール前600m 地点での先頭との差から：持続型（前で粘る）・終い伸び型（後ろから伸びる）・先行失速型・平均型。</p>
  </section>;
}

// ===== 買い目（AIの確率から自動で組む。予算 1000円。計算は app/lib/bets.ts） =====
const ODDS_FIELD: Record<BetType, keyof OddsData> = { 単勝: "win", 複勝: "place", ワイド: "wide", 馬連: "umaren", "3連複": "trio" };
const BET_TYPES: BetType[] = ["単勝", "複勝", "ワイド", "馬連", "3連複"];

function OddsBar({ odds, state, error, onLoad, compact = false, weightNote = "" }: { odds: OddsData | null; state: "idle" | "loading" | "error"; error: string; onLoad: () => void; compact?: boolean; weightNote?: string }) {
  const missing = odds ? BET_TYPES.filter((t) => !Object.keys((odds[ODDS_FIELD[t]] as OddsMap) ?? {}).length) : [];
  return <div className="odds-bar">
    <div className="bet-controls">
      <span className="odds-bar-title">{odds ? (odds.status === "yoso" ? "予想オッズ" : odds.status === "result" ? "確定オッズ" : "オッズ") : "オッズ未取得"}</span>
      <button type="button" className="refresh-button" onClick={onLoad} disabled={state === "loading"}>
        {state === "loading" ? <span className="loading-spinner" aria-hidden="true" /> : "↻"} オッズを取得
      </button>
    </div>
    <p className="bet-note">
      {odds ? <>netkeiba {odds.officialTime ? odds.officialTime.replace(/^\d{4}-/, "").replace("-", "/") + " 時点" : ""}（取得 {new Date(odds.fetchedAt).toLocaleString("ja-JP", { timeZone: "Asia/Tokyo", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}）
        {odds.status === "yoso" && <><br /><b className="bet-warn">発売前。netkeiba の予想で、実際のオッズではない</b></>}</>
        : compact ? "取得すると、市場と比べた目安を出す" : "押すと netkeiba から今のオッズを取る（押すたびに取り直す。取ったオッズはこの端末に保存する）"}
      {!compact && missing.length > 0 && <><br />まだ出ていない券種：{missing.join("・")}</>}
      {state === "error" && <><br /><b className="bet-warn">オッズを取得できなかった：{error}</b></>}
      {weightNote && <><br /><b>{weightNote}</b></>}
    </p>
  </div>;
}

const MARK_MEANING: Record<AiMark, string> = {
  "◎": "本命", "○": "対抗", "▲": "単穴", "☆": "穴", "△": "連下", "消": "消し",
};

function MarksPanel({ horses, inTraining, odds, oddsBar, values, longshot, horseOdds, winCoef, evidence }: {
  horses: Horse[]; inTraining: boolean; odds: OddsData | null; oddsBar: ReactNode; values: Map<number, ValueResult> | null;
  longshot: Map<number, number> | null;
  horseOdds: Map<string, HorseOdds>; winCoef: { ai: number; market: number } | null;
  evidence: Record<string, { marks: Record<string, MarkStats>; popLe3: MarkStats }> | null;
}) {
  const marks = useMemo(() => {
    const rows = horses.filter((h) => h.prediction && h.prediction.baseWinProbability != null && h.raceEntry?.horseNumber != null);
    return assignAiMarks(rows.map((h) => {
      const v = values?.get(h.raceEntry!.horseNumber!) ?? null;
      const o = horseOdds.get(h.horseId);
      return { id: h.horseId, win: h.prediction!.baseWinProbability!, top3: h.prediction!.probability, ratio: v?.ratio ?? null,
        odds: o?.win ?? null, popularity: o?.popularity ?? null, longshot: longshot?.get(h.raceEntry!.horseNumber!) ?? null };
    }), odds && values ? winCoef : null);
  }, [horses, values, longshot, horseOdds, odds, winCoef]);
  const years = evidence ? Object.keys(evidence).sort() : [];
  const avg = (m: string, key: keyof MarkStats) => {
    const xs = years.map((y) => evidence![y].marks[m]?.[key]).filter((x): x is number => typeof x === "number");
    return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
  };
  const pct = (x: number | null) => (x == null ? "-" : Math.round(x * 100) + "%");
  const marked = AI_MARK_ORDER.flatMap((m) => horses.filter((h) => marks.get(h.horseId)?.mark === m));
  const scores = useMemo(() => scoreDeviations(horses), [horses]);

  return <section className="bets-panel">
    {inTraining && <p className="heavy-note" role="note">⚠ このレースの結果はAIの学習データに入っている。印は参考として見ること。</p>}
    {oddsBar}
    {!marks.size ? <p className="state-panel">AIの勝率が出ていないため、印を付けられない。</p> : <>
      {!odds && <p className="bet-note">オッズを取得すると、☆（穴）・消と、▲の妙味の判定が加わる。</p>}
      <ul className="ai-mark-list">
        {marked.map((h) => {
          const r = marks.get(h.horseId)!;
          const o = horseOdds.get(h.horseId);
          return <li key={h.horseId} className={"mark-" + MARK_MEANING[r.mark]}>
            <span className="ai-mark">{r.mark}</span>
            <HorseNumberBadge entry={h.raceEntry} />
            <span className="ai-mark-name"><b>{h.name}</b><small>{r.reason}</small></span>
            <span className="ai-odds">{o?.win != null ? <><b>{o.win.toFixed(1)}</b><small>{o.popularity}番人気</small></> : <small>-</small>}</span>
            <span className="ai-mark-score"><b>{Math.round(scores.get(h.horseId) ?? 50)}</b><small>AIスコア</small></span>
          </li>;
        })}
      </ul>
      {evidence && <details className="model-note" open>
        <summary>印の意味と過去の成績</summary>
        <div className="mark-table-wrap">
          <table className="mark-table">
            <thead><tr><th>印</th><th>勝率</th><th>3着内率</th><th>複勝回収率</th></tr></thead>
            <tbody>{AI_MARK_ORDER.map((m) => <tr key={m}>
              <td><b>{m}</b> {MARK_MEANING[m]}</td><td>{pct(avg(m, "win_rate"))}</td><td>{pct(avg(m, "top3_rate"))}</td>
              <td>{pct(avg(m, "place_roi"))}<small>（基準 {pct(avg(m, "place_roi_market_matched"))}）</small></td>
            </tr>)}</tbody>
          </table>
        </div>
        <p>{years.join("・")}年の全レース（前年までで学習したAIで印を付け、確定オッズで計算）の平均。基準は、同じくらいの人気の馬を全部買ったときの複勝回収率。</p>
        <p>◎ 勝率1位（割高なら次点）／○ 残りでAIスコア1位／▲ 勝率上位4頭のうち単勝の妙味が一番／☆ 6番人気以下で、人気薄専用のAI（182項目＋オッズ）の3着以内確率がオッズから見た確率の1.2倍以上／△ 残りでAIスコア6位以内／消 3番人気以内で割高。▲☆△消は複数頭に付くことがある。</p>
        <p className="bet-warn">☆は人気の割に来やすく、基準より複勝回収率が高かった（2024〜2026年、確定オッズ）が、100%を安定して超えてはいない。消は3番人気以内の馬全体（3着内率 約53%）よりはっきり走っていない。</p>
      </details>}
    </>}
  </section>;
}

export default function RaceDetailPage() {
  const params = useParams<{ raceId: string }>();
  const raceId = params.raceId;
  const [data, setData] = useState<ResponseData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [cachedAt, setCachedAt] = useState<number | null>(null);
  const [cacheChecked, setCacheChecked] = useState(false);
  const [marks, setMarks] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState<"detail" | "prediction" | "marks">("detail");
  const [predSort, setPredSort] = useState<"number" | "prob">("number");
  const [predView, setPredView] = useState<"horse" | "factor">("horse");
  const [factor, setFactor] = useState<string>("能力指数");
  const [assumedGoing, setAssumedGoing] = useState(""); // 馬場未発表時に想定する馬場（空 = 想定なし）
  const [odds, setOdds] = useState<OddsData | null>(null);
  const [oddsState, setOddsState] = useState<"idle" | "loading" | "error">("idle");
  const [oddsError, setOddsError] = useState("");
  const [weightNote, setWeightNote] = useState(""); // オッズ取得のあとの馬体重の反映の結果
  // 「オッズを取得」は押すたびに netkeiba から取り直す（ブラウザ・サーバーのキャッシュを使わない）
  const loadOdds = useCallback(async () => {
    setOddsState("loading"); setOddsError("");
    try {
      const res = await fetch("/api/odds/" + raceId + "?t=" + Date.now(), { cache: "no-store" });
      const json = await res.json() as OddsData & { error?: string };
      if (!res.ok || !json.ok) throw new Error(json.errors?.join(" / ") || json.error || "HTTP " + res.status);
      setOdds(json); saveCachedOdds(raceId, json); setOddsState("idle");
      // 馬体重がまだ入っていない（前走の馬体重で計算中の）ときは、出馬表を取り直して予想を計算し直す
      if (dataRef.current?.ml?.bodyWeightAssumed) {
        setWeightNote("馬体重を確認中…");
        const fresh = await loadRef.current?.(false, true, assumedGoingRef.current, true);
        setWeightNote(!fresh ? "" : fresh.ml?.bodyWeightAssumed ? "馬体重はまだ発表されていない（前走の馬体重で計算中）" : "発表された馬体重で予想を計算し直した");
      }
    } catch (err) {
      setOddsState("error"); setOddsError(err instanceof Error ? err.message : "取得に失敗した");
    }
  }, [raceId]);

  const load = useCallback(async (forceRefresh: boolean, background: boolean, going = "", weights = false): Promise<ResponseData | null> => {
    try {
      if (forceRefresh || background) setRefreshing(true);
      else setLoading(true);
      setError("");
      const params = new URLSearchParams();
      if (forceRefresh) params.set("refresh", "1");
      if (weights) params.set("weights", "1");
      if (going) params.set("going", going);
      const query = params.toString();
      const response = await fetch("/api/races/" + raceId + (query ? "?" + query : ""), { cache: "no-store" });
      const result = (await response.json()) as ResponseData;
      if (!response.ok || !result.raceConditions || !Array.isArray(result.results)) {
        throw new Error(result.error || "レース情報の取得に失敗しました（HTTP " + response.status + "）");
      }
      const savedAt = Date.now();
      setData(result); setCachedAt(savedAt);
      if (!going) saveCachedRace(raceId, result, savedAt); // 想定馬場での結果は保存しない
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : "レース情報の取得に失敗しました。";
      if (background || forceRefresh) setError("最新情報への更新に失敗しました。表示中の保存データを利用しています。 " + message);
      else setError(message);
      return null;
    } finally { setLoading(false); setRefreshing(false); }
  }, [raceId]);

  // loadOdds から最新の値を使うための参照
  const dataRef = useRef<ResponseData | null>(null);
  const loadRef = useRef<typeof load | null>(null);
  const assumedGoingRef = useRef("");
  useEffect(() => { dataRef.current = data; loadRef.current = load; assumedGoingRef.current = assumedGoing; }, [data, load, assumedGoing]);

  useEffect(() => {
    if (!/^\d{12}$/.test(raceId)) {
      setError("レースIDが正しくありません。"); setLoading(false); setCacheChecked(true); return;
    }
    setMarks(readMarks(raceId));
    setOdds(readCachedOdds(raceId));
    const cached = readCachedRace(raceId);
    setCacheChecked(true);
    if (cached) {
      setData(cached.data); setCachedAt(cached.savedAt); setLoading(false);
      if (Date.now() - cached.savedAt >= CLIENT_CACHE_TTL_MS) void load(false, true);
    } else void load(false, false);
  }, [raceId, load]);

  const conditions = data?.raceConditions;
  const horses = data?.results ?? [];
  const factorScores = useMemo(() => buildFactorScores(horses), [horses]);
  const categoryRanks = useMemo(() => {
    // 各観点：SHAP の寄与を観点ごとに足し、出走馬の中で AIスコアを押し上げた量の順位（app/lib/ml/shapGroups.ts）
    const base = buildCategoryRanks(horses, factorScores, conditions?.distanceMeters ?? null, conditions?.venue ?? null);
    const oldDetail = (horseId: string, label: string) => base.get(horseId)?.find((c) => c.label === label && c.detail !== NO_DATA)?.detail ?? null;
    const abilityText: Record<string, (p: Prediction) => string | null> = {
      能力指数: (p) => p.ability?.ab != null ? "近走（同じ芝ダ）の能力指数の直近加重平均 " + (p.ability.ab >= 0 ? "+" : "") + p.ability.ab.toFixed(2) + "（その日の馬場差・斤量を補正したタイム指数。1000mあたりの秒）" : null,
      "上がり・ラップタイプ": (p) => p.ability?.agari != null ? "近走の推定上がり3F が、そのレースの平均より " + Math.abs(p.ability.agari).toFixed(2) + "秒" + (p.ability.agari >= 0 ? "速い" : "遅い") + "（直近加重平均。成績表の隊列から推定）" : null,
      条件適性: (p) => p.ability?.cond != null ? "能力指数に、今回の距離・競馬場・馬場での得意不得意を足した値 " + (p.ability.cond >= 0 ? "+" : "") + p.ability.cond.toFixed(2) : null,
    };
    const OLD: Record<string, string> = { 騎手: "騎手", 調子: "調子", 血統: "血統" };
    const logit = (p: number) => Math.log(p / (1 - p));
    const pt = (p: number, phi: number) => (p - 1 / (1 + Math.exp(-(logit(p) - phi)))) * 100; // その寄与がなかったときとの確率の差（pt）
    const signed = (x: number) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(1) + "pt";
    const rows = horses.filter((h) => h.prediction?.shapGroups);
    const out = new Map<string, CategoryRank[]>();
    for (const g of SHAP_GROUPS) {
      const vals = rows.map((h) => ({ h, c: h.prediction!.shapGroups!.find((x) => x.group === g) })).filter((x) => x.c);
      for (const { h, c } of vals) {
        const p = h.prediction!.probability;
        const items = c!.top.map((t) => `${featureLabel(t.name)}${t.value != null ? " " + (Math.abs(t.value) >= 100 ? Math.round(t.value) : Math.round(t.value * 100) / 100) : "（なし）"}（${signed(pt(p, t.phi))}）`);
        const detail = [`3着以内確率への影響 ${signed(pt(p, c!.phi))}`, items.length ? "主な項目：" + items.join("、") : null,
          abilityText[g]?.(h.prediction!) ?? null, OLD[g] ? oldDetail(h.horseId, OLD[g]) : null].filter(Boolean).join(" / ");
        const list = out.get(h.horseId) ?? [];
        // 観点の偏差値：その観点の寄与（SHAP）の、出走馬の中での偏差値（平均50・標準偏差10）
        const mean = vals.reduce((a, x) => a + x.c!.phi, 0) / vals.length;
        const sd = Math.sqrt(vals.reduce((a, x) => a + (x.c!.phi - mean) * (x.c!.phi - mean), 0) / vals.length);
        const score = sd > 0 ? 50 + 10 * (c!.phi - mean) / sd : 50;
        list.push({ label: g, rank: 1 + vals.filter((x) => x.c!.phi > c!.phi).length, total: vals.length, detail, score, type: typeOf(g, h.prediction!) });
        out.set(h.horseId, list);
      }
    }
    return out;
  }, [horses, factorScores, conditions?.distanceMeters, conditions?.venue]);
  const sortedHorses = useMemo(() => [...horses].sort((a, b) => predSort === "prob"
    ? (a.prediction?.rank ?? 999) - (b.prediction?.rank ?? 999)
    : (a.raceEntry?.horseNumber ?? 999) - (b.raceEntry?.horseNumber ?? 999)), [horses, predSort]);
  const horseOdds = useMemo(() => {
    const out = new Map<string, HorseOdds>();
    if (!odds) return out;
    const nums = horses.map((h) => h.raceEntry?.horseNumber).filter((n): n is number => n != null && !!odds.win[String(n)]?.min);
    const pop = new Map([...nums].sort((a, b) => odds.win[String(a)].min - odds.win[String(b)].min).map((n, i) => [n, i + 1]));
    for (const h of horses) {
      const n = h.raceEntry?.horseNumber;
      if (n == null) continue;
      out.set(h.horseId, { win: odds.win[String(n)]?.min ?? null, place: odds.place[String(n)] ?? null, popularity: pop.get(n) ?? null });
    }
    return out;
  }, [odds, horses]);
  const oddsBar = <OddsBar odds={odds} state={oddsState} error={oddsError} onLoad={() => void loadOdds()} weightNote={weightNote} />;
  // AI予想タブの目安（妙味あり・妥当・割高）。AI の確率（合計をそろえる前）と単勝オッズから（app/lib/market.ts の valueLabels）
  const values = useMemo(() => {
    const model = data?.ml?.valueModel;
    if (!odds || !model) return null;
    const base = new Map(horses.filter((h) => h.prediction && h.raceEntry?.horseNumber != null)
      .map((h) => [h.raceEntry!.horseNumber!, h.prediction!.baseProbability]));
    return valueLabels(base, odds.win, model);
  }, [odds, horses, data?.ml?.valueModel]);
  // AI印の☆：人気薄専用の2段目（/api/longshot）。オッズを取ったら、各馬の入力と確率を送って倍率をもらう
  const [longshot, setLongshot] = useState<Map<number, number> | null>(null);
  useEffect(() => {
    setLongshot(null);
    const rows = horses.filter((h) => h.prediction?.inputRow && h.prediction.baseWinProbability != null && h.raceEntry?.horseNumber != null)
      .map((h) => ({ n: h.raceEntry!.horseNumber!, x: h.prediction!.inputRow!, top3: h.prediction!.baseProbability, win: h.prediction!.baseWinProbability! }));
    if (!odds || rows.length < 3 || !data?.ml?.longshotReady) return;
    const winOdds = Object.fromEntries(rows.filter((r) => odds.win[String(r.n)]?.min).map((r) => [r.n, odds.win[String(r.n)].min]));
    let cancelled = false;
    fetch("/api/longshot", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ rows, odds: winOdds }) })
      .then((res) => (res.ok ? res.json() : null))
      .then((json: { ok: boolean; horses: Record<string, { ratio: number }> } | null) => {
        if (!cancelled && json?.ok) setLongshot(new Map(Object.entries(json.horses).map(([n, v]) => [Number(n), v.ratio])));
      })
      .catch(() => { /* ☆なしで続ける */ });
    return () => { cancelled = true; };
  }, [odds, horses, data?.ml?.longshotReady]);
  const predictedCount = horses.filter((h) => h.prediction).length;
  const scores = useMemo(() => scoreDeviations(horses), [horses]);
  const evidence = data?.ml?.valueModel?.evidence;
  const evidenceYear = evidence ? Object.keys(evidence).sort().at(-1) : undefined;
  const handleMarkChange = (horseId: string, mark: string) => {
    setMarks((current) => {
      const next = { ...current };
      if (mark) next[horseId] = mark; else delete next[horseId];
      saveMarks(raceId, next); return next;
    });
  };

  return <main className="race-detail-shell">
    {loading && cacheChecked && !data && <div className="state-panel loading-panel">
      <span className="loading-spinner" aria-hidden="true" /><span>初回データを取得しています…</span>
    </div>}
    {!data && !loading && error && <div className="state-panel error-panel"><b>レース情報を取得できませんでした</b><p>{error}</p></div>}
    {data && conditions && <>
      <header className="detail-hero">
        <p className="detail-kicker">{formatRaceDate(conditions.date)} {display(conditions.venue)}{display(conditions.raceNumber)}R{conditions.startTime ? ` ・ ${conditions.startTime}発走` : ""}</p>
        <h1>{conditions.raceName || "レース詳細"}</h1>
        <div className="condition-chips">
          {[
            [conditions.surface, conditions.distanceMeters ? conditions.distanceMeters + "m" : null].filter(Boolean).join(""),
            [conditions.direction, conditions.courseLayout, conditions.courseLetter ? conditions.courseLetter + "コース" : null].filter(Boolean).join(" "),
            data.horseCount ? data.horseCount + "頭" : "",
            "天候 " + (conditions.weather ? conditions.weather + (conditions.weatherForecast ? "（予報）" : "") : "未発表"),
            "馬場 " + (data.ml?.goingSource === "user" ? "未発表" : conditions.going ?? "未発表"),
          ].filter((value) => value !== "").map((value) => <span className="condition-chip" key={value}>{value}</span>)}
        </div>
        {conditions.conditionSource && <p className="detail-meta">天候・馬場：{conditions.conditionSource}</p>}
        <div className="refresh-row">
          <span>{formatSavedAt(cachedAt)}{refreshing && " ・ 更新を確認中"}</span>
          <button type="button" className="refresh-button" onClick={() => void load(true, false, assumedGoing)} disabled={refreshing || loading}>
            {refreshing ? <span className="loading-spinner" aria-hidden="true" /> : "↻"} 更新
          </button>
        </div>
        {error && <p className="detail-meta" role="status">{error}</p>}
      </header>
      <div className="race-tabs" role="tablist" aria-label="レース情報">
        <button type="button" role="tab" aria-selected={activeTab === "detail"} className={"race-tab" + (activeTab === "detail" ? " active" : "")}
          onClick={() => setActiveTab("detail")}>出馬表</button>
        <button type="button" role="tab" aria-selected={activeTab === "prediction"} className={"race-tab" + (activeTab === "prediction" ? " active" : "")}
          onClick={() => setActiveTab("prediction")}>AI予想</button>
        <button type="button" role="tab" aria-selected={activeTab === "marks"} className={"race-tab" + (activeTab === "marks" ? " active" : "")}
          onClick={() => setActiveTab("marks")}>AI印</button>
      </div>
      {activeTab === "detail" && <div role="tabpanel">
        {oddsBar}
        <section className="detail-section">
          {horses.map((horse) => <HorseDetailCard key={horse.horseId} horse={horse} odds={horseOdds.get(horse.horseId) ?? null}
            mark={marks[horse.horseId] ?? ""} onMarkChange={handleMarkChange} />)}
        </section>
      </div>}
      {activeTab === "marks" && <div role="tabpanel"><MarksPanel horses={horses} inTraining={!!data.ml?.raceInTraining} odds={odds} oddsBar={oddsBar}
        values={values} longshot={longshot} horseOdds={horseOdds} winCoef={data.ml?.valueModel?.win ?? null} evidence={data.ml?.marksEvidence ?? null} /></div>}
      {activeTab === "prediction" && <section className="prediction-panel" role="tabpanel">
        {data.ml?.status === "available" ? <>
          <div className="ai-conditions">
            <span>馬場{data.ml.goingSource === "announced" ? " " : "（想定）"}
              {data.ml.goingSource !== "announced"
                ? <select aria-label="想定する馬場" value={assumedGoing} disabled={refreshing || loading} onChange={(event) => {
                  setAssumedGoing(event.target.value); void load(false, true, event.target.value);
                }}>
                  <option value="">良</option>
                  {["良", "稍重", "重", "不良"].map((going) => <option key={going} value={going}>{going}</option>)}
                </select>
                : <b>{conditions.going ?? "-"}</b>}
            </span>
            {data.ml.pace && <span>想定ペース <b>{data.ml.pace.expected}</b></span>}
          </div>
          {data.ml.raceInTraining && <p className="heavy-note" role="note">
            ⚠ このレースの結果は、すでにAIの学習データに入っている（〜{data.ml.dataUntil}）。答えを見て学習した状態なので、参考として見ること。
          </p>}
          <OddsBar odds={odds} state={oddsState} error={oddsError} onLoad={() => void loadOdds()} weightNote={weightNote} compact />
          <section className="horse-results-section">
            <div className="results-toolbar">
              <div className="sort-toggle" role="group" aria-label="見方">
                <button type="button" className={predView === "horse" ? "active" : ""} onClick={() => setPredView("horse")}>馬ごと</button>
                <button type="button" className={predView === "factor" ? "active" : ""} onClick={() => setPredView("factor")}>観点別</button>
              </div>
              {predView === "horse" && <div className="sort-toggle" role="group" aria-label="並び順">
                <button type="button" className={predSort === "number" ? "active" : ""} onClick={() => setPredSort("number")}>馬番順</button>
                <button type="button" className={predSort === "prob" ? "active" : ""} onClick={() => setPredSort("prob")}>スコア順</button>
              </div>}
            </div>
            {predView === "factor" && <FactorView horses={horses} categoryRanks={categoryRanks} factor={factor} onFactor={setFactor} />}
            {predView === "horse" && <><div className="ai-row-head" aria-hidden="true">
              <span>馬</span><span>オッズ</span><span>AIスコア<i />平均</span><span>目安</span>
            </div>
            {sortedHorses.map((horse) => <OverallHorseCard key={horse.horseId} horse={horse}
              total={predictedCount}
              mark={marks[horse.horseId] ?? ""}
              odds={horseOdds.get(horse.horseId) ?? null}
              value={horse.raceEntry?.horseNumber != null ? values?.get(horse.raceEntry.horseNumber) ?? null : null}
              score={scores.get(horse.horseId) ?? null}
              categories={categoryRanks.get(horse.horseId) ?? []}
              factors={factorScores.get(horse.horseId) ?? []} />)}</>}
            <div className="ai-legend">
              <p><b>AIスコア</b>＝AIが出した3着以内に入る確率を、出走馬の中での偏差値にしたもの（平均50。差が大きいほど抜けている）。線は平均の50。確率（%）は行を押すと出る。</p>
              <p><b>目安</b>＝AIと市場（単勝オッズ）を合わせた3着以内の確率が、オッズから見た確率より高いか。
                <span className="ai-value good">妙味あり</span>1.1倍以上・<span className="ai-value even">妥当</span>・<span className="ai-value bad">割高</span>0.9倍以下。
                {!odds && " オッズを取得すると出る。"}</p>
              {evidence && evidenceYear && <p className="bet-warn">過去の検証（{evidenceYear}年、確定オッズ）：「妙味あり」の馬の複勝回収率は {Math.round((evidence[evidenceYear]["妙味あり"]?.roi ?? 0) * 100)}%（{evidence[evidenceYear]["妙味あり"]?.bets ?? 0}点）、
                「割高」は {Math.round((evidence[evidenceYear]["割高"]?.roi ?? 0) * 100)}%。妙味ありでも長く買い続けて勝てるとは言えない。</p>}
            </div>
            <details className="model-note">
              <summary>このAI予想について</summary>
              <p>機械学習（LightGBM）で、3着以内に入る確率を出している。人気・オッズはAIスコアには使っていない（目安だけがオッズを使う）。
                学習データ：JRA成績表 〜{data.ml?.dataUntil ?? "-"}。
                {data.ml?.goingSource === "user" ? `想定馬場「${conditions.going}」で計算。` : data.ml?.goingAssumed ? "馬場状態が未発表のため「良」として計算。" : ""}
                {data.ml?.bodyWeightAssumed ? "馬体重が未発表の馬は、前走の馬体重・増減0として計算。" : ""}
                {data.ml?.metrics ? `${data.ml.metrics.testYear}年の検証：AIスコア上位3頭の3着内率 ${(Math.round(data.ml.metrics.raceTop3Hit * 1000) / 10)}%。` : ""}</p>
            </details>
          </section>
        </> : <p className="state-panel">AI予想は利用できない（モデル未学習、または対象外のレース）。</p>}
      </section>}
    </>}
  </main>;
}
