"use client";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

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
};
type RaceHistoryMetric = { starts: number; wins: number; top3: number; winRate: number; placeRate: number };
type RaceHistoryForHorse = {
  frame: RaceHistoryMetric | null;
  jockey: RaceHistoryMetric | null;
  frameYears: number[];
  jockeyYears: number[];
} | null;
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
type Prediction = {
  probability: number; rank: number; modelFormDeviation: number | null; priorRuns: number;
  jockeyName: string | null; jockey: JockeyYear;
};
type MlSummary = {
  status: "available" | "unavailable"; dataUntil?: string; goingAssumed?: boolean;
  metrics?: { auc: number; raceTop3Hit: number; testYear: string };
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
  raceHistory?: RaceHistoryForHorse;
};
type ResponseData = {
  ok: boolean; raceId: string; raceConditions: Conditions; horseCount: number;
  completeCount: number; partialCount: number; failedCount: number;
  results: Horse[]; error?: string; ml?: MlSummary;
};
type CachedRace = { savedAt: number; data: ResponseData };
type FactorScore = { label: string; score: number; starts: number; rate: number; rank: number; rankedCount: number };
type CategoryRank = { label: string; rank: number | null; total: number; detail: string };

const CLIENT_CACHE_TTL_MS = 5 * 60 * 1000;
const cacheKey = (id: string) => "keiba-race-detail-v6:" + id;
const marksCacheKey = (id: string) => "keiba-race-marks-v1:" + id;
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

function buildFactorScores(horses: Horse[]): Map<string, FactorScore[]> {
  const factors = [
    { label: "全体成績", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.parsedStarts ?? 0, rate: s?.parsedStarts ? s.top3 / s.parsedStarts : 0 }; } },
    { label: "重賞実績", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.gradedStarts ?? 0, rate: s?.gradedStarts ? s.gradedWins / s.gradedStarts : 0 }; } },
    { label: "競馬場・馬場適性", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.sameVenueSurfaceStarts ?? 0, rate: s?.sameVenueSurfaceStarts ? s.sameVenueSurfaceTop3 / s.sameVenueSurfaceStarts : 0 }; } },
    { label: "距離適性", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.similarDistanceStarts ?? 0, rate: s?.similarDistanceStarts ? s.similarDistanceTop3 / s.similarDistanceStarts : 0 }; } },
    { label: "馬場適性", get: (horse: Horse) => { const s = horse.performanceStats; return { starts: s?.sameGoingStarts ?? 0, rate: s?.sameGoingStarts ? s.sameGoingTop3 / s.sameGoingStarts : 0 }; } },
    { label: "騎手と馬の相性", get: (horse: Horse) => { const m = horse.raceHistory?.jockey; return { starts: m?.starts ?? 0, rate: m?.placeRate ?? 0 }; } },
    { label: "枠順実績", get: (horse: Horse) => { const m = horse.raceHistory?.frame; return { starts: m?.starts ?? 0, rate: m?.placeRate ?? 0 }; } },
    { label: "調子", get: (horse: Horse) => ({ starts: horse.form?.runs.length ?? 0, rate: horse.form?.raw ?? 0 }) },
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

function buildCategoryRanks(
  horses: Horse[],
  factorScores: Map<string, FactorScore[]>,
  distanceMeters: number | null,
  venue: string | null
): Map<string, CategoryRank[]> {
  const result = new Map<string, CategoryRank[]>();
  const categories = ["調子", "騎手", "騎手と馬の相性", "調教", "血統", "距離適性", "コース適性", "馬場適性", "枠順", "種牡馬", "ポジション"];
  for (const horse of horses) {
    result.set(horse.horseId, categories.map((label) => ({
      label, rank: null, total: 0, detail: "評価に必要なデータが未取得",
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
      row.detail = `今回の距離±200mの全過去走 ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`;
    }
  });
  const assignFactorRank = (factorLabel: string, categoryLabel: string, detailBuilder: (value: FactorScore) => string) => {
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
        row.detail = detailBuilder(value.factor);
        if (categoryLabel === "枠順") {
          const source = horses.find((horse) => horse.horseId === value.horseId)?.raceHistory;
          const metric = source?.frame;
          if (metric) {
            const percent = (rate: number) => Math.round(rate * 100) + "%";
            const yearLabel = (source?.frameYears ?? []).length + "年分";
            row.detail = `エリザベス女王杯の${venue ?? "同競馬場"}開催・過去${yearLabel}・今回の枠番 ${metric.starts}回／勝率 ${percent(metric.winRate)}／複勝率 ${percent(metric.placeRate)}`;
          }
        }
      }
    });
  };
  assignFactorRank("競馬場・馬場適性", "コース適性", (value) =>
    `同競馬場・同芝ダート ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`);
  assignFactorRank("馬場適性", "馬場適性", (value) =>
    `同馬場状態 ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`);
  assignFactorRank("騎手と馬の相性", "騎手と馬の相性", (value) =>
    `今回の騎手との過去騎乗 ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`);
  assignFactorRank("騎手(今年)", "騎手", () => "");
  assignFactorRank("調子", "調子", () => "");
  assignFactorRank("血統適性", "血統", () => "");
  assignFactorRank("枠順実績", "枠順", (value) =>
    `今回と同じ枠番 ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`);


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

const BAND_NAMES: Record<string, string> = { "-1400": "〜1400m", "-1800": "〜1800m", "-2200": "〜2200m", "-2600": "〜2600m", "2600-": "2600m〜" };
const bandText = (rates: Record<string, number>) =>
  Object.entries(rates).map(([band, rate]) => `${BAND_NAMES[band] ?? band} ${Math.round(rate * 100)}%`).join("・");

function HorseDetailCard({ horse, mark, onMarkChange }: {
  horse: Horse; mark: string; onMarkChange: (horseId: string, mark: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const entry = horse.raceEntry;
  const profile = horse.profile ?? [];
  const profileValue = (label: string) => profile.find((item) => item.label.includes(label))?.value;
  const histories = horse.performances ?? [];
  const pedigree = horse.pedigree;
  const facts: [string, string][] = ([
    ["枠番", entry?.frameNumber == null ? "" : String(entry.frameNumber)],
    ["馬体重", entry?.bodyWeight == null ? "" : String(entry.bodyWeight) + "kg"],
    ["人気", entry?.popularity == null ? "" : String(entry.popularity)],
    ["調教師", entry?.trainer ?? ""],
    ["父", pedigree?.sire ?? profileValue("父") ?? ""],
    ["父系統", pedigree?.sireLine ?? ""],
    ["母", pedigree?.dam ?? profileValue("母") ?? ""],
    ["母父", pedigree?.damsire ?? profileValue("母父") ?? ""],
    ["母母", pedigree?.damdam ?? ""],
    ["クロス", (pedigree?.crosses ?? []).map((cross) =>
      `${cross.name} ${cross.generations}${cross.percentage === null ? "" : `（${cross.percentage}%）`}`).join("、")],
  ] as [string, string][]).filter(([, value]) => value !== "");
  const pedigreeRows = (pedigree?.ancestors ?? []).filter((ancestor) => ancestor.generation <= 3);
  return <details className="horse-detail" onToggle={(event) => setExpanded(event.currentTarget.open)}>
    <summary className="horse-summary">
      {entry?.horseNumber != null && <span className="horse-number-badge">{entry.horseNumber}番</span>}
      <span className="horse-main">
        <b>{horse.name}</b>
        {[entry?.sexAge, entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).length > 0 &&
          <small>{[entry?.sexAge, entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).join(" ・ ")}</small>}
      </span>
      <span className="horse-expand">{expanded ? "－" : "＋"}</span>
    </summary>
    <select className="horse-mark-select" aria-label={horse.name + "の印"} value={mark}
      onChange={(event) => onMarkChange(horse.horseId, event.target.value)}>
      {MARK_OPTIONS.map((option) => <option key={option || "none"} value={option}>{option || "印なし"}</option>)}
    </select>
    {expanded && <div className="horse-body">
      {facts.length > 0 && <div className="horse-facts">
        {facts.map(([label, value]) => <div className="horse-fact" key={label}><span>{label}：</span>{value}</div>)}
      </div>}
      {pedigreeRows.length > 0 && <div className="history-table-wrap">
        <b style={{ fontSize: 12 }}>血統（3代）</b>
        <table className="history-table pedigree-table">
          <tbody>{pedigreeRows.filter((ancestor) => ancestor.generation === 3).map((ancestor) => {
            const parent = pedigreeRows.find((item) => item.position === ancestor.position.slice(0, 2));
            const grand = pedigreeRows.find((item) => item.position === ancestor.position.slice(0, 1));
            const first = ancestor.position.endsWith("F");
            return <tr key={ancestor.position}>
              {ancestor.position.slice(1) === "FF" && grand && <td rowSpan={4}><b>{grand.name}</b></td>}
              {first && parent && <td rowSpan={2}>{parent.name}</td>}
              <td>{ancestor.name}{ancestor.country ? `（${ancestor.country}）` : ""}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>}
      {histories.length > 0 && <div className="history-table-wrap">
        <b style={{ fontSize: 12 }}>過去の戦績（{histories.length}走）</b>
        <table className="history-table">
          <thead><tr>{Object.keys(histories[0].values).map((key) => <th key={key}>{key}</th>)}</tr></thead>
          <tbody>{histories.map((history, index) => <tr key={index}>
            {Object.keys(histories[0].values).map((key) => <td key={key}>{history.values[key] ?? ""}</td>)}
          </tr>)}</tbody>
        </table>
      </div>}
    </div>}
  </details>;
}

const CATEGORY_GROUPS = [
  { title: "成績・距離", labels: ["全体成績", "重賞実績", "距離適性"] },
  { title: "コース・馬場適性", labels: ["コース適性", "馬場適性", "枠順"] },
  { title: "調子・騎手・位置取り", labels: ["調子", "騎手", "騎手と馬の相性", "ポジション"] },
  { title: "血統・調教", labels: ["血統", "種牡馬", "調教"] },
];

function OverallHorseCard({ horse, total, categories, factors }: {
  horse: Horse; total: number; categories: CategoryRank[]; factors: FactorScore[];
}) {
  const prediction = horse.prediction;
  const probability = prediction ? prediction.probability * 100 : null;
  const probabilityText = probability === null ? "算出不可" : probability.toFixed(1) + "%";
  const entry = horse.raceEntry;
  const factorByCategory: Record<string, string> = {
    "コース適性": "競馬場・馬場適性",
    "騎手": "騎手(今年)",
    "枠順": "枠順実績",
    "血統": "血統適性",
  };
  const factorFor = (label: string) => {
    const factorLabel = factorByCategory[label] ?? label;
    return factors.find((factor) => factor.label === factorLabel);
  };

  return <details className="prediction-horse-details">
    <summary className="prediction-horse-summary">
      {entry?.horseNumber != null && <span className="horse-number-badge">{entry.horseNumber}番</span>}
      <span className="horse-main">
        <b>{horse.name}</b>
        <small>{[entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).join(" ・ ")}</small>
      </span>
      <span className="horse-inline-score">
        <strong>{probabilityText}</strong>
        <span className="horse-inline-score-track" role="img" aria-label={horse.name + " 3着以内確率 " + probabilityText}>
          <span className="horse-inline-score-fill" style={{ width: (probability === null ? 0 : Math.max(0, Math.min(100, probability))) + "%" }} />
        </span>
      </span>
      <span className="horse-expand">＋</span>
    </summary>

    <div className="horse-result-content">
      <section className="horse-result-overall">
        <div className="result-section-heading">
          <h4>AI予測：3着以内に入る確率</h4>
          <p>LightGBM（成績表データで学習）。血統適性は下の項目に別表示</p>
        </div>
        <div className="horse-result-score-line">
          <strong>{probabilityText}</strong>
          <span>{prediction ? prediction.rank + "位 / " + total + "頭" : "順位対象外"}</span>
        </div>
        {probability !== null && <div className="overall-score-track" role="img" aria-label={"3着以内確率 " + probabilityText}>
          <div className="overall-score-fill" style={{ width: Math.max(0, Math.min(100, probability)) + "%" }} />
        </div>}
        {prediction && <p className="result-category-rank">
          モデル内の調子偏差値 {prediction.modelFormDeviation == null ? "-" : prediction.modelFormDeviation.toFixed(1)}
          （成績表の近5走: 着順＋タイム差）・成績表の過去走 {prediction.priorRuns}走
          {prediction.jockeyName ? "" : "・騎手を成績表で特定できず"}
        </p>}
      </section>

      <section className="horse-result-section">
        <div className="result-section-heading">
          <h4>評価項目別の結果</h4>
          <p>各項目のスコア・順位・集計内容です。</p>
        </div>
        {CATEGORY_GROUPS.map((group) => {
          const items = group.labels.map((label) => ({
            item: categories.find((category) => category.label === label),
            factor: factorFor(label),
          })).filter((value): value is { item: CategoryRank; factor: FactorScore | undefined } => value.item !== undefined);
          return <section className="result-category-group" key={group.title}>
            <h5>{group.title}</h5>
            <div className="result-category-grid">
              {items.map(({ item, factor }) => <article className="result-category-card" key={item.label}>
                <div className="result-category-heading">
                  <b>{item.label}</b>
                  <strong>{factor ? "スコア " + factor.score.toFixed(1) : item.rank === null ? "算出不可" : item.rank + "位 / " + item.total + "頭"}</strong>
                </div>
                {factor && <div className="category-score-track" role="img" aria-label={item.label + "スコア " + factor.score.toFixed(1)}>
                  <div className="category-score-fill" style={{ width: Math.max(0, Math.min(100, factor.score)) + "%" }} />
                </div>}
                {factor && item.rank !== null && <div className="result-category-rank">{item.rank}位 / {item.total}頭</div>}
                <p>{item.detail}</p>
              </article>)}
            </div>
          </section>;
        })}
      </section>
    </div>
  </details>;
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
  const [activeTab, setActiveTab] = useState<"detail" | "prediction">("detail");

  const load = useCallback(async (forceRefresh: boolean, background: boolean) => {
    try {
      if (forceRefresh || background) setRefreshing(true);
      else setLoading(true);
      setError("");
      const response = await fetch("/api/races/" + raceId + (forceRefresh ? "?refresh=1" : ""), { cache: "no-store" });
      const result = (await response.json()) as ResponseData;
      if (!response.ok || !result.raceConditions || !Array.isArray(result.results)) {
        throw new Error(result.error || "レース情報の取得に失敗しました（HTTP " + response.status + "）");
      }
      const savedAt = Date.now();
      setData(result); setCachedAt(savedAt); saveCachedRace(raceId, result, savedAt);
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
  const handleMarkChange = (horseId: string, mark: string) => {
    setMarks((current) => {
      const next = { ...current };
      if (mark) next[horseId] = mark; else delete next[horseId];
      saveMarks(raceId, next); return next;
    });
  };

  return <main className="race-detail-shell">
    <Link href="/" className="back-link">← 開催レース一覧へ戻る</Link>
    {loading && cacheChecked && !data && <div className="state-panel loading-panel">
      <span className="loading-spinner" aria-hidden="true" /><span>初回データを取得しています…</span>
    </div>}
    {!data && !loading && error && <div className="state-panel error-panel"><b>レース情報を取得できませんでした</b><p>{error}</p></div>}
    {data && conditions && <>
      <header className="detail-hero">
        <p className="eyebrow">RACE DETAIL</p><h1>{conditions.raceName || "レース詳細"}</h1>
        <p className="detail-meta">{display(conditions.date)} ・ {display(conditions.venue)}競馬場 ・ {display(conditions.raceNumber)}R
          <br />{display(conditions.startTime)} 発走</p>
      </header>
      <button type="button" className="refresh-button" onClick={() => void load(true, false)} disabled={refreshing || loading}>
        {refreshing && <span className="loading-spinner" aria-hidden="true" />}
        {refreshing ? "最新情報を取得中…" : "最新情報に更新"}
      </button>
      <p className="detail-meta">{formatSavedAt(cachedAt)}{refreshing && " ・ バックグラウンドで最新情報を確認中"}</p>
      {error && <p className="detail-meta" role="status">{error}</p>}
      <div className="race-tabs" role="tablist" aria-label="レース情報">
        <button type="button" role="tab" aria-selected={activeTab === "detail"} className={"race-tab" + (activeTab === "detail" ? " active" : "")}
          onClick={() => setActiveTab("detail")}>レース詳細</button>
        <button type="button" role="tab" aria-selected={activeTab === "prediction"} className={"race-tab" + (activeTab === "prediction" ? " active" : "")}
          onClick={() => setActiveTab("prediction")}>AI予測結果</button>
      </div>
      {activeTab === "detail" && <div role="tabpanel">
        <section className="detail-section">
          <h2>レース条件</h2><div className="condition-grid">
            {([
              ["コース", [conditions.surface, conditions.distanceMeters ? String(conditions.distanceMeters) + "m" : null].filter(Boolean).join(" ")],
              ["回り・コース", [conditions.direction, conditions.courseLayout, conditions.courseLetter].filter(Boolean).join(" ")],
              ["天候", conditions.weather ?? ""], ["馬場状態", conditions.going ?? ""],
            ] as [string, string][]).filter(([, value]) => value !== "").map(([label, value]) =>
              <div className="condition-item" key={label}><small>{label}</small><b>{value}</b></div>)}
          </div>
        </section>
        <section className="detail-section">
          <h2>出走馬 <span style={{ color: "var(--muted)", fontSize: 12 }}>（{data.horseCount}頭）</span></h2>
          {horses.map((horse) => <HorseDetailCard key={horse.horseId} horse={horse}
            mark={marks[horse.horseId] ?? ""} onMarkChange={handleMarkChange} />)}
        </section>
      </div>}
      {activeTab === "prediction" && <section className="prediction-panel" role="tabpanel">
        <div className="prediction-intro">
          <h2>AI予測結果</h2>
          <p>各馬の右側は、機械学習（LightGBM）で予測した3着以内に入る確率。人気・オッズは使っていない。馬名をタップすると、調子偏差値・騎手の今年成績・血統適性などの項目別結果を確認できる。</p>
          {data.ml?.status === "available" ? <p>
            学習データ：JRA成績表 〜{data.ml.dataUntil}（{data.ml.metrics?.testYear}年の検証で、予測上位3頭の3着以内的中率 {data.ml.metrics ? Math.round(data.ml.metrics.raceTop3Hit * 1000) / 10 : "-"}%）。
            {data.ml.goingAssumed ? "馬場状態が未発表のため「良」として計算。" : ""}
          </p> : <p>AI予測は利用できない（モデル未学習、または対象外のレース）。</p>}
        </div>
        <section className="horse-results-section">
          <div className="result-section-heading">
            <h3>各馬の評価結果</h3>
            <p>出馬表の枠番・馬番順で表示しています。馬名をタップすると詳細が開きます。</p>
          </div>
          {horses.map((horse) => <OverallHorseCard key={horse.horseId} horse={horse}
            total={horses.filter((item) => item.prediction).length}
            categories={categoryRanks.get(horse.horseId) ?? []}
            factors={factorScores.get(horse.horseId) ?? []} />)}
        </section>
      </section>}
    </>}
  </main>;
}
