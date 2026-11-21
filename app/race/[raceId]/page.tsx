"use client";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { BET_BUDGET, buildCandidates, hitFocusedPlan, returnFocusedPlan, type BetHorse, type BetType, type Candidate, type Plan } from "@/app/lib/bets";
import { useParams } from "next/navigation";
import { valueLabels, type ValueModel, type ValueResult } from "@/app/lib/market";

type Conditions = {
  raceName: string | null; date: string | null; venue: string | null;
  raceNumber: number | null; startTime: string | null; surface: string | null;
  distanceMeters: number | null; direction: string | null; courseLayout: string | null;
  courseLetter: string | null; weather: string | null; going: string | null;
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
  pace?: PaceInfo;
  extras?: Extras;
  trainerName?: string | null;
  trainer?: { n: number; top3: number; venueN: number; venueTop3: number };
  probability: number; baseProbability: number; pedigreeIncluded: boolean;
  winProbability?: number | null; top2Probability?: number | null;
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
};
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
type CategoryRank = { label: string; rank: number | null; total: number; detail: string };

const CLIENT_CACHE_TTL_MS = 5 * 60 * 1000;
const cacheKey = (id: string) => "keiba-race-detail-v7:" + id;
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
function predictionFacts(p: Prediction): [string, string][] {
  const rows: [string, string][] = [];
  const e = p.extras;
  if (e) {
    if (e.prizeRatioLast != null) {
      const r = e.prizeRatioLast;
      rows.push(["クラス", `${r > 1.15 ? "昇級・格上挑戦" : r < 0.87 ? "降級・格下" : "前走と同程度"}（1着賞金 前走の${r.toFixed(2)}倍）`]);
    }
    const change: string[] = [];
    if (e.distChange) change.push(`距離${e.distChange > 0 ? "+" : ""}${e.distChange}m`);
    if (e.surfaceChange === 1) change.push("芝ダ替わり");
    if (e.weightChange) change.push(`斤量${e.weightChange > 0 ? "+" : ""}${e.weightChange}kg`);
    if (e.jockeyChange === 1) change.push("乗り替わり");
    if (e.distChange != null) rows.push(["前走から", change.length ? change.join("・") : "変化なし"]);
    if (e.tidxMean5 != null) rows.push(["タイム指数", `近5走平均 ${e.tidxMean5.toFixed(2)}・最高 ${e.tidxBest5!.toFixed(2)}（1000mあたり基準より何秒速いか）`]);
  }
  if (p.trainer && p.trainer.n) rows.push(["調教師", `${p.trainerName ?? "-"}：今年 ${p.trainer.n}頭・3着内 ${Math.round(p.trainer.top3 / p.trainer.n * 100)}%`]);
  if (p.pace) {
    const pc = p.pace;
    rows.push(["脚質", (pc.style ?? "不明") + (pc.posFirstMean != null
      ? `（最初のコーナー 平均${Math.round(pc.posFirstMean * 100)}%地点${pc.nigeRate ? `・逃げ${Math.round(pc.nigeRate * 100)}%` : ""}）` : "")]);
    const paceText = (label: string, v: { n: number; top3: number | null }) =>
      `${label} ${v.n}走${v.top3 != null ? `・3着内${Math.round(v.top3 * 100)}%` : ""}`;
    rows.push(["ペース別", paceText("ハイ", pc.highPace) + " ／ " + paceText("スロー", pc.slowPace)]);
  }
  rows.push(["調子偏差値", `${p.modelFormDeviation == null ? "-" : p.modelFormDeviation.toFixed(1)}（成績表の近5走：着順＋タイム差）・過去走 ${p.priorRuns}走` +
    (p.jockeyName ? "" : "・騎手を成績表で特定できず")]);
  rows.push(["調教評価", p.training?.rank ? `${p.training.rank}${p.training.critic ? "（" + p.training.critic + "）" : ""}` : "未発表（評価なしとして計算）"]);
  if (p.overseasRuns) rows.push(["海外の走", `海外${p.overseasRuns}走を過去走に含めて計算（着順・頭数・格から。海外のレースは学習データにないので参考）`]);
  if (e?.wakuCourseN && e.wakuCourseDiff != null) rows.push(["枠（参考）", `同コースでこの枠の3着内率はコース平均より${e.wakuCourseDiff >= 0 ? "+" : ""}${Math.round(e.wakuCourseDiff * 100)}pt（${e.wakuCourseN}頭）`]);
  return rows;
}

// JRA の枠の色
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
  const [showAllRuns, setShowAllRuns] = useState(false);
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
  const runs = showAllRuns ? histories : histories.slice(0, 5);
  const v = (run: { values: Record<string, string> }, key: string) => run.values[key] ?? "";
  return <details className="horse-detail" onToggle={(event) => setExpanded(event.currentTarget.open)}>
    <summary className="horse-summary">
      <HorseNumberBadge entry={entry} />
      <span className="horse-main">
        <b>{horse.name}</b>
        <small>{[entry?.sexAge, entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).join(" ・ ")}</small>
        {sire && <small className="horse-blood">{sire}{damsire && <> <i>×</i> 母父{damsire}</>}</small>}
      </span>
      {odds?.win && <span className="horse-odds"><b>{odds.win.toFixed(1)}</b><small>{odds.popularity}人気</small></span>}
      {/* 印の操作で開閉しないように、クリックの既定動作（summary の開閉）を止める */}
      <span className="horse-mark-slot" onClick={(event) => event.preventDefault()}>
        <select className={"horse-mark-select" + (mark ? " marked" : "")} aria-label={horse.name + "の印"} value={mark}
          onChange={(event) => onMarkChange(horse.horseId, event.target.value)}>
          {MARK_OPTIONS.map((option) => <option key={option || "none"} value={option}>{option || "印"}</option>)}
        </select>
      </span>
      <span className="horse-expand">{expanded ? "－" : "＋"}</span>
    </summary>
    {expanded && <div className="horse-body">
      {facts.length > 0 && <dl className="fact-list">
        {facts.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      </dl>}
      {histories.length > 0 && <div className="detail-block">
        <h4>戦績（{histories.length}走）</h4>
        <div className="history-table-wrap">
          <table className="history-table run-table">
            <thead><tr><th>日付</th><th>レース</th><th>着順</th><th>人気</th><th>距離</th><th>馬場</th><th>タイム</th><th>着差</th><th>通過</th><th>上り</th><th>騎手</th><th>斤量</th><th>馬体重</th></tr></thead>
            <tbody>{runs.map((run, index) => {
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
        {histories.length > 5 && <button type="button" className="more-button" onClick={() => setShowAllRuns((s) => !s)}>
          {showAllRuns ? "近5走だけ表示" : `全${histories.length}走を表示`}
        </button>}
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
    </div>}
  </details>;
}

const CATEGORY_GROUPS = [
  { title: "成績・距離", labels: ["全体成績", "重賞実績", "距離適性"] },
  { title: "コース・馬場適性", labels: ["コース適性", "馬場適性", "枠順"] },
  { title: "調子・騎手・位置取り", labels: ["調子", "騎手", "ポジション"] },
  { title: "血統・調教", labels: ["血統", "種牡馬", "調教"] },
];

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

function probabilityTier(rank: number | undefined) {
  if (rank === undefined) return "";
  return rank <= 3 ? " tier-top" : rank <= 6 ? " tier-mid" : "";
}

const VALUE_CLASS: Record<string, string> = { 妙味あり: "good", 妥当: "even", 割高: "bad" };

function OverallHorseCard({ horse, total, categories, factors, mark, odds, value, average }: {
  horse: Horse; total: number; categories: CategoryRank[]; factors: FactorScore[]; mark: string;
  odds: HorseOdds | null; value: ValueResult | null; average: number;
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
      <span className="ai-score" role="img" aria-label={horse.name + " AIスコア（3着以内確率） " + probabilityText}>
        <strong>{probability === null ? "-" : Math.round(probability)}</strong>
        <span className="ai-score-track">
          <span className="ai-score-fill" style={{ width: (probability === null ? 0 : Math.max(0, Math.min(100, probability))) + "%" }} />
          <span className="ai-score-avg" style={{ left: Math.min(100, average) + "%" }} />
        </span>
      </span>
      <span className={"ai-value " + (value ? VALUE_CLASS[value.label] : "none")}>{value?.label ?? "-"}</span>
    </summary>

    <div className="horse-result-content">
      <section className="horse-result-overall">
        <div className="horse-result-score-line">
          <div><small>3着以内に入る確率</small><strong>{probabilityText}</strong></div>
          <span>{prediction ? prediction.rank + "位 / " + total + "頭" : "順位対象外"}</span>
        </div>
        {prediction && prediction.winProbability != null && prediction.top2Probability != null && <div className="sub-probs">
          <span>勝率 <b>{(prediction.winProbability * 100).toFixed(1)}%</b></span>
          <span>連対率 <b>{(prediction.top2Probability * 100).toFixed(1)}%</b></span>
        </div>}
        {(strong.length > 0 || weak.length > 0) && <div className="point-chips">
          {strong.map((label) => <span className="point-chip strong" key={label}>◎ {label}</span>)}
          {weak.map((label) => <span className="point-chip weak" key={label}>△ {label}</span>)}
        </div>}
        {heavy && <p className="heavy-note" role="note">
          ⚠ 道悪（重・不良）未経験・血統は道悪{heavy.label}（血統の道悪倍率 {heavy.ratio.toFixed(2)}倍）。
          {" 血統の道悪適性は、AIの確率にすでに含めて計算している。"}
        </p>}
        {prediction && <dl className="fact-list">
          {predictionFacts(prediction).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
        </dl>}
      </section>

      <section className="horse-result-section">
        <div className="result-section-heading">
          <h4>項目別の評価</h4>
          <p>スコアは出走馬の中での偏差値（50が平均）。順位は出走馬の中での順位。</p>
        </div>
        {CATEGORY_GROUPS.map((group) => {
          const items = group.labels.map((label) => ({
            item: categories.find((category) => category.label === label),
            factor: factorFor(label),
          })).filter((value): value is { item: CategoryRank; factor: FactorScore | undefined } =>
            value.item !== undefined && !missing.includes(value.item.label));
          if (!items.length) return null;
          return <section className="result-category-group" key={group.title}>
            <h5>{group.title}</h5>
            <div className="result-category-grid">
              {items.map(({ item, factor }) => <details className="result-category-card" key={item.label}>
                <summary className="result-category-summary">
                  <span className="result-category-heading">
                    <b>{item.label}</b>
                    <span className="result-category-value">
                      {item.rank !== null && <em className={item.rank <= 3 && item.total >= 4 ? "good" : ""}>{item.rank}位/{item.total}</em>}
                      {factor && <strong>{factor.score.toFixed(1)}</strong>}
                    </span>
                  </span>
                  {factor && <span className="category-score-track" role="img" aria-label={item.label + "スコア " + factor.score.toFixed(1)}>
                    <span className="category-score-fill" style={{ width: Math.max(0, Math.min(100, (factor.score - 20) / 60 * 100)) + "%" }} />
                    <span className="category-score-mid" />
                  </span>}
                </summary>
                {item.detail && <p>{item.detail}</p>}
              </details>)}
            </div>
          </section>;
        })}
        {missing.length > 0 && <p className="missing-note">データなし：{missing.join("・")}</p>}
      </section>
    </div>
  </details>;
}

// ===== 買い目（AIの確率から自動で組む。予算 1000円。計算は app/lib/bets.ts） =====
const ODDS_FIELD: Record<BetType, keyof OddsData> = { 単勝: "win", 複勝: "place", ワイド: "wide", 馬連: "umaren", "3連複": "trio" };
const BET_TYPES: BetType[] = ["単勝", "複勝", "ワイド", "馬連", "3連複"];

function OddsBar({ odds, state, error, onLoad, compact = false }: { odds: OddsData | null; state: "idle" | "loading" | "error"; error: string; onLoad: () => void; compact?: boolean }) {
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
    </p>
  </div>;
}

function PlanCard({ title, lead, plan, horses, empty }: { title: string; lead: string; plan: Plan | null; horses: Horse[]; empty: string }) {
  return <section className="bet-group">
    <div className="bet-group-head"><h3>{title}</h3>
      {plan && <span>{plan.tickets.length}点・{BET_BUDGET.toLocaleString()}円</span>}</div>
    <p className="bet-note">{lead}</p>
    {!plan ? <p className="bet-note">{empty}</p> : <>
      <div className="bet-summary">
        <div><small>どれか当たる確率</small><b>{(plan.hitProb * 100).toFixed(1)}%</b></div>
        {plan.profitProb !== null && <div><small>払戻が予算以上</small><b>{(plan.profitProb * 100).toFixed(1)}%</b></div>}
        {plan.expectedReturn !== null && <div><small>期待回収率</small><b>{Math.round(plan.expectedReturn * 100)}%</b></div>}
      </div>
      <ul className="bet-list">
        {plan.tickets.map(({ c, stake }) => <li key={c.type + c.horses.map((h) => h.id).join("-")}>
          <span className="bet-combo"><em className="bet-type">{c.type}</em>{c.horses.map((h, i) => <span key={h.id}>
            {i > 0 && <i>-</i>}
            <HorseNumberBadge entry={horses.find((x) => x.horseId === h.id)?.raceEntry ?? null} />
          </span>)}</span>
          <span className="bet-names">{c.horses.map((h) => h.name).join("・")}
            <small>的中確率 {(c.prob * 100).toFixed(1)}%{c.odds ? `・配当 ${c.odds.toFixed(1)}倍・期待値 ${(c.prob * c.odds).toFixed(2)}` : ""}
              {c.odds && stake ? `・払戻 ${Math.round(stake * c.odds).toLocaleString()}円` : ""}</small>
          </span>
          <strong>{stake.toLocaleString()}円</strong>
        </li>)}
      </ul>
      {plan.payoutRange && plan.payoutRange[0] < BET_BUDGET && <p className="bet-note"><b className="bet-warn">1点だけの的中だと払戻が予算を下回る買い目がある（当たっても損）</b></p>}
      {!plan.usedOdds && <p className="bet-note"><b className="bet-warn">オッズがないので払戻は分からない。複勝を細かく分けると、当たっても予算を下回ることが多い</b></p>}
    </>}
  </section>;
}

// 券種ごとの期待値（AIの的中確率 × 配当）の上位。上の2パターンが単勝・複勝に寄るときも、馬連・3連複などの候補を見られるように
function TypeBest({ cands, horses, hasOdds }: { cands: Candidate[]; horses: Horse[]; hasOdds: boolean }) {
  return <section className="bet-group">
    <div className="bet-group-head"><h3>券種ごとの候補</h3><span>{hasOdds ? "期待値の高い順・各3点" : "的中確率の高い順・各3点"}</span></div>
    <p className="bet-note">上の2パターンは確率の計算上、単勝・複勝に寄りやすい（的中率重視は当たりやすい複勝、回収率重視は期待値の高い1〜2点になりやすい）。ここでは券種ごとに候補を出す。期待値1.0は「AIの確率が正しければ回収率100%」の意味。</p>
    {BET_TYPES.map((type) => {
      const list = cands.filter((c) => c.type === type && (!hasOdds || c.odds))
        .sort((a, b) => hasOdds ? b.prob * b.odds! - a.prob * a.odds! : b.prob - a.prob).slice(0, 3);
      if (!list.length) return <p className="bet-note" key={type}>{type}：オッズがまだ出ていない</p>;
      return <ul className="bet-list type-best" key={type}>
        {list.map((c) => <li key={c.type + c.horses.map((h) => h.id).join("-")}>
          <span className="bet-combo"><em className="bet-type">{c.type}</em>{c.horses.map((h, i) => <span key={h.id}>
            {i > 0 && <i>-</i>}
            <HorseNumberBadge entry={horses.find((x) => x.horseId === h.id)?.raceEntry ?? null} />
          </span>)}</span>
          <span className="bet-names">{c.horses.map((h) => h.name).join("・")}
            <small>的中確率 {(c.prob * 100).toFixed(1)}%{c.odds ? `・配当 ${c.odds.toFixed(1)}倍` : ""}</small></span>
          <strong>{c.odds ? (c.prob * c.odds).toFixed(2) : (c.prob * 100).toFixed(1) + "%"}</strong>
        </li>)}
      </ul>;
    })}
  </section>;
}

function BetsPanel({ horses, inTraining, odds, oddsBar }: { horses: Horse[]; inTraining: boolean; odds: OddsData | null; oddsBar: ReactNode }) {
  const hs: BetHorse[] = horses.filter((h) => h.prediction && h.prediction.winProbability != null).map((h) => ({
    id: h.horseId, num: h.raceEntry?.horseNumber ?? null, name: h.name,
    win: h.prediction!.winProbability!, place: h.prediction!.probability,
  }));
  const plans = useMemo(() => {
    const cands = buildCandidates(hs, (type, nums) => {
      const table = odds ? odds[ODDS_FIELD[type]] as OddsMap : null;
      return table?.[nums.join("-")]?.min ?? null;
    });
    return { hit: hitFocusedPlan(cands), ret: returnFocusedPlan(cands), count: cands.length, cands };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [horses, odds]);
  if (hs.length < 3 || !plans.count) return <p className="state-panel">AIの勝率が出ていないため、買い目を組めない。</p>;

  return <section className="bets-panel">
    {inTraining && <p className="heavy-note" role="note">⚠ このレースの結果はAIの学習データに入っている。買い目は参考として見ること。</p>}
    {oddsBar}
    <p className="bet-note">予算 {BET_BUDGET.toLocaleString()}円{!odds && "。オッズを取得すると、払戻を見て買い目を組み直す（回収率重視はオッズが必要）"}</p>
    <PlanCard title="的中率重視" horses={horses} plan={plans.hit}
      lead={plans.hit?.usedOdds ? "払戻が予算（1000円）以上になる確率が一番高くなるように、単勝・複勝・ワイド・馬連・3連複から最大5点を選んだ" : "どれか当たる確率が一番高くなるように、最大5点を選んだ（オッズを取得すると、当たって損しない組み方に変わる）"}
      empty="買い目を組めなかった" />
    <PlanCard title="回収率重視" horses={horses} plan={plans.ret}
      lead="期待値（AIの的中確率 × 配当）が高い買い目から、期待回収率が一番高くなる点数（最大5点）を選んだ"
      empty="回収率重視はオッズが必要。上の「オッズを取得」を押す" />
    <TypeBest cands={plans.cands} horses={horses} hasOdds={!!odds} />
    <details className="model-note">
      <summary>買い目の決め方と注意</summary>
      <p>AIの勝率から、着順を2万回試して（勝率を合計1にそろえ、1着の馬を除いた残りで2着・3着が決まると考える）、買い目ごとの的中確率と、組み合わせたときの「どれか当たる確率」「払戻が予算以上になる確率」「期待回収率」を数えている。候補は勝率・3着以内確率の上位6頭。</p>
      <p>買い目の的中確率は勝率から計算しているので、「AI予測結果」の3着以内確率とは少し違う。</p>
      <p>予算は100円単位。オッズがあるときは、どれが当たっても払戻がほぼ同じになるように配当の逆数に比例させ、ないときは的中確率に比例させる。複勝・ワイドの配当は幅の下限で計算する。</p>
      <p>期待回収率はAIの確率が正しい前提の値。過去の検証では、AIだけが高く見ている（期待値が高く見える）馬ほど実際には当たりにくく、AIの1位の馬を買い続けた回収率は単勝 約76%・複勝 約84%で100%を下回る。回収率重視の買い目も、100%を超えるとは限らない。</p>
    </details>
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
  const [activeTab, setActiveTab] = useState<"detail" | "prediction" | "bets">("detail");
  const [predSort, setPredSort] = useState<"number" | "prob">("number");
  const [assumedGoing, setAssumedGoing] = useState(""); // 馬場未発表時に想定する馬場（空 = 想定なし）
  const [odds, setOdds] = useState<OddsData | null>(null);
  const [oddsState, setOddsState] = useState<"idle" | "loading" | "error">("idle");
  const [oddsError, setOddsError] = useState("");
  // 「オッズを取得」は押すたびに netkeiba から取り直す（ブラウザ・サーバーのキャッシュを使わない）
  const loadOdds = useCallback(async () => {
    setOddsState("loading"); setOddsError("");
    try {
      const res = await fetch("/api/odds/" + raceId + "?t=" + Date.now(), { cache: "no-store" });
      const json = await res.json() as OddsData & { error?: string };
      if (!res.ok || !json.ok) throw new Error(json.errors?.join(" / ") || json.error || "HTTP " + res.status);
      setOdds(json); saveCachedOdds(raceId, json); setOddsState("idle");
    } catch (err) {
      setOddsState("error"); setOddsError(err instanceof Error ? err.message : "取得に失敗した");
    }
  }, [raceId]);

  const load = useCallback(async (forceRefresh: boolean, background: boolean, going = "") => {
    try {
      if (forceRefresh || background) setRefreshing(true);
      else setLoading(true);
      setError("");
      const params = new URLSearchParams();
      if (forceRefresh) params.set("refresh", "1");
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
    } catch (err) {
      const message = err instanceof Error ? err.message : "レース情報の取得に失敗しました。";
      if (background || forceRefresh) setError("最新情報への更新に失敗しました。表示中の保存データを利用しています。 " + message);
      else setError(message);
    } finally { setLoading(false); setRefreshing(false); }
  }, [raceId]);

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
  const categoryRanks = useMemo(() => buildCategoryRanks(horses, factorScores, conditions?.distanceMeters ?? null, conditions?.venue ?? null), [horses, factorScores, conditions?.distanceMeters, conditions?.venue]);
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
  const oddsBar = <OddsBar odds={odds} state={oddsState} error={oddsError} onLoad={() => void loadOdds()} />;
  // AI予想タブの目安（妙味あり・妥当・割高）。AI の確率（合計をそろえる前）と単勝オッズから（app/lib/market.ts の valueLabels）
  const values = useMemo(() => {
    const model = data?.ml?.valueModel;
    if (!odds || !model) return null;
    const base = new Map(horses.filter((h) => h.prediction && h.raceEntry?.horseNumber != null)
      .map((h) => [h.raceEntry!.horseNumber!, h.prediction!.baseProbability]));
    return valueLabels(base, odds.win, model);
  }, [odds, horses, data?.ml?.valueModel]);
  const predictedCount = horses.filter((h) => h.prediction).length;
  const averageScore = predictedCount ? Math.min(3, predictedCount) / predictedCount * 100 : 0;
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
            "天候 " + (conditions.weather ?? "未発表"),
            "馬場 " + (data.ml?.goingSource === "user" ? "未発表" : conditions.going ?? "未発表"),
          ].filter((value) => value !== "").map((value) => <span className="condition-chip" key={value}>{value}</span>)}
        </div>
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
        <button type="button" role="tab" aria-selected={activeTab === "bets"} className={"race-tab" + (activeTab === "bets" ? " active" : "")}
          onClick={() => setActiveTab("bets")}>買い目</button>
      </div>
      {activeTab === "detail" && <div role="tabpanel">
        {oddsBar}
        <section className="detail-section">
          {horses.map((horse) => <HorseDetailCard key={horse.horseId} horse={horse} odds={horseOdds.get(horse.horseId) ?? null}
            mark={marks[horse.horseId] ?? ""} onMarkChange={handleMarkChange} />)}
        </section>
      </div>}
      {activeTab === "bets" && <div role="tabpanel"><BetsPanel horses={horses} inTraining={!!data.ml?.raceInTraining} odds={odds} oddsBar={oddsBar} /></div>}
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
          <OddsBar odds={odds} state={oddsState} error={oddsError} onLoad={() => void loadOdds()} compact />
          <section className="horse-results-section">
            <div className="results-toolbar">
              <div className="sort-toggle" role="group" aria-label="並び順">
                <button type="button" className={predSort === "number" ? "active" : ""} onClick={() => setPredSort("number")}>馬番順</button>
                <button type="button" className={predSort === "prob" ? "active" : ""} onClick={() => setPredSort("prob")}>スコア順</button>
              </div>
            </div>
            <div className="ai-row-head" aria-hidden="true">
              <span>馬</span><span>オッズ</span><span>AIスコア<i />平均</span><span>目安</span>
            </div>
            {sortedHorses.map((horse) => <OverallHorseCard key={horse.horseId} horse={horse}
              total={predictedCount}
              mark={marks[horse.horseId] ?? ""}
              odds={horseOdds.get(horse.horseId) ?? null}
              value={horse.raceEntry?.horseNumber != null ? values?.get(horse.raceEntry.horseNumber) ?? null : null}
              average={averageScore}
              categories={categoryRanks.get(horse.horseId) ?? []}
              factors={factorScores.get(horse.horseId) ?? []} />)}
            <div className="ai-legend">
              <p><b>AIスコア</b>＝AIが出した3着以内に入る確率（%）。線は出走馬の平均。行を押すと理由が出る。</p>
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
